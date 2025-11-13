"""Analyse occupancy against world-specific masks in fixed episode chunks."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np

from helper_functions_ import load_structure
from plotting_functions import get_occupancy_plot
from structure_functions_ import get_hole_locations, get_parameter_values, sort_paths

try:  # pragma: no cover - exercised in environments without optional deps
    from analyse_wall_occupancy import (
        _OccupancyModelAdapter,
        _convert_for_helper,
        _derive_wall_mask,
        _expand_mask,
        _iter_coordinates,
    )
except (ModuleNotFoundError, ImportError):  # pragma: no cover - fallback for minimal test envs
    class _OccupancyModelAdapter:  # type: ignore[dead-code]
        def __init__(self, *args, **kwargs):  # type: ignore[unused-argument]
            raise RuntimeError(
                "analyse_wall_occupancy is unavailable; install NumPy to enable occupancy helpers"
            )

    def _convert_for_helper(*args, **kwargs):  # type: ignore[unused-argument]
        raise RuntimeError(
            "analyse_wall_occupancy is unavailable; install NumPy to enable occupancy helpers"
        )

    def _derive_wall_mask(*args, **kwargs):  # type: ignore[unused-argument]
        raise RuntimeError(
            "analyse_wall_occupancy is unavailable; install NumPy to enable occupancy helpers"
        )

    def _expand_mask(*args, **kwargs):  # type: ignore[unused-argument]
        raise RuntimeError(
            "analyse_wall_occupancy is unavailable; install NumPy to enable occupancy helpers"
        )

    def _iter_coordinates(path):  # type: ignore[unused-argument]
        raise RuntimeError(
            "analyse_wall_occupancy is unavailable; install NumPy to enable occupancy helpers"
        )


_LAG_TARGET_PREFIX = "t_minus_"
_LAG_TARGET_SUFFIX = "_hole_locations"
def _lag_target_name(lag_offset: int) -> str:
    if lag_offset <= 1:
        return "previous_hole_locations"
    return f"{_LAG_TARGET_PREFIX}{int(lag_offset)}{_LAG_TARGET_SUFFIX}"


def _lag_offset_for_target(target: str) -> Optional[int]:
    if target == "previous_hole_locations":
        return 1
    if target.startswith(_LAG_TARGET_PREFIX) and target.endswith(_LAG_TARGET_SUFFIX):
        body = target[len(_LAG_TARGET_PREFIX) : -len(_LAG_TARGET_SUFFIX)]
        try:
            offset = int(body)
        except ValueError:
            return None
        if offset >= 1:
            return offset
    return None


def _max_world_count(structure: Mapping[str, object]) -> int:
    """Return the maximum number of worlds present across ``structure``."""

    max_worlds = 0
    for seed_block in structure.values():
        worlds: Optional[np.ndarray] = None
        if isinstance(seed_block, Mapping):
            candidate = seed_block.get("worlds")
            if candidate is not None:
                worlds = np.asarray(candidate)
            else:
                for variant_block in seed_block.values():
                    if isinstance(variant_block, Mapping) and "worlds" in variant_block:
                        worlds = np.asarray(variant_block["worlds"])
                        break
        if worlds is None or getattr(worlds, "size", 0) == 0:
            continue
        if worlds.ndim >= 1:
            max_worlds = max(max_worlds, int(worlds.shape[0]))
        else:
            max_worlds = max(max_worlds, 1)
    return max_worlds


def derive_occupancy_targets(structure: Mapping[str, object]) -> Tuple[str, ...]:
    """Return the ordered occupancy targets supported by ``structure``."""

    world_count = _max_world_count(structure)
    max_lag = max(1, world_count - 1)
    lag_targets = tuple(_lag_target_name(offset) for offset in range(1, max_lag + 1))
    return ("current_hole_locations", *lag_targets, "walls")



@dataclass(slots=True)
class _SeedPrefixJob:
    seed: str
    prefix: str
    env_switch_every: int
    chunk_size: int
    occupancy_target: str
    hole_masks: List[np.ndarray]
    wall_masks: List[np.ndarray]
    reward_locations: List[List[Tuple[int, int]]]
    episode_indices: Sequence
    trajectories: Sequence
    sample_config: Optional["_SampleConfig"]


@dataclass(frozen=True)
class _SampleConfig:
    """Configuration controlling per-path sample extraction."""

    episode_window: Optional[Tuple[Optional[int], Optional[int]]]
    agent_prefixes: Optional[frozenset[str]]
    min_start_distance: Optional[float]
    occupancy_thresholds: Optional[Tuple[Optional[float], Optional[float]]]
    max_samples: Optional[int]

    def allows_agent(self, prefix: str) -> bool:
        if not self.agent_prefixes:
            return True
        return prefix in self.agent_prefixes

    def episode_within_window(self, episode: int) -> bool:
        if not self.episode_window:
            return True
        start, end = self.episode_window
        if start is not None and episode < start:
            return False
        if end is not None and episode > end:
            return False
        return True

    def occupancy_within_thresholds(self, fraction: float) -> bool:
        if not self.occupancy_thresholds:
            return True
        lower, upper = self.occupancy_thresholds
        if lower is not None and fraction < lower:
            return False
        if upper is not None and fraction > upper:
            return False
        return True

    def to_dict(self) -> Dict[str, object]:
        if self.episode_window:
            start, end = self.episode_window
            window_payload: Optional[Dict[str, Optional[int]]] = {
                "start": start,
                "end": end,
            }
        else:
            window_payload = None

        return {
            "episode_window": window_payload,
            "agent_prefixes": sorted(self.agent_prefixes) if self.agent_prefixes else None,
            "min_start_distance": self.min_start_distance,
            "occupancy_thresholds": self.occupancy_thresholds,
            "max_samples": self.max_samples,
        }


# def _resolve_run_path(job_id: int, results_dir: Path) -> Tuple[Path, Path]:
#     """Return the run directory referenced by ``run_path.txt`` for ``job_id``."""

#     job_id_str = str(job_id)
#     job_plus_one = results_dir / str(job_id + 1) / "run_path.txt"
#     job_exact = results_dir / job_id_str / "run_path.txt"

#     for candidate in (job_plus_one, job_exact):
#         if candidate.exists():
#             run_path = Path(candidate.read_text().strip()).expanduser().resolve()
#             return run_path, candidate.parent

#     raise FileNotFoundError(
#         f"Could not locate run_path.txt for job {job_id} in {results_dir}."
#     )


def _compute_path_occupancy(
    path: Sequence,
    grid_size: int,
    model_cache: Dict[int, _OccupancyModelAdapter],
) -> np.ndarray:
    """Return an occupancy heatmap for a single saved trajectory."""

    if grid_size not in model_cache:
        model_cache[grid_size] = _OccupancyModelAdapter(grid_size)
    converted = _convert_for_helper(path, grid_size)
    if not converted:
        return np.zeros((grid_size, grid_size), dtype=float)
    return get_occupancy_plot(model_cache[grid_size], converted)


def _first_coordinate(path: Sequence) -> Optional[Tuple[int, int]]:
    """Return the first valid (row, col) coordinate from ``path`` if present."""

    for row, col in _iter_coordinates(path):
        return int(row), int(col)
    return None


def _nearest_reward(
    start: Tuple[int, int], reward_locations: Sequence[Tuple[int, int]]
) -> Tuple[Optional[Tuple[int, int]], Optional[float]]:
    """Return the nearest reward coordinate (Manhattan distance) and distance."""

    if not reward_locations:
        return None, None

    row, col = start
    best_coord: Optional[Tuple[int, int]] = None
    best_distance: Optional[float] = None
    for reward_row, reward_col in reward_locations:
        distance = abs(row - reward_row) + abs(col - reward_col)
        if best_distance is None or distance < best_distance:
            best_distance = float(distance)
            best_coord = (int(reward_row), int(reward_col))
    return best_coord, best_distance


def _build_hole_mask(world: np.ndarray) -> np.ndarray:
    """Return a boolean mask for hole locations in ``world``."""

    world = np.asarray(world)
    if world.ndim != 2:
        world = np.squeeze(world)
    if world.ndim != 2:
        raise ValueError("Expected a 2D array for world layout")

    mask = np.zeros_like(world, dtype=float)
    for row, col in get_hole_locations(world):
        if 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]:
            mask[row, col] = 1.0
    return mask


class _ChunkAccumulator:
    """Helper for aggregating occupancy statistics within an episode chunk."""

    def __init__(self, chunk_index: int, chunk_size: int) -> None:
        self.chunk_index = int(chunk_index)
        self.chunk_size = int(chunk_size)
        start = self.chunk_index * self.chunk_size
        end = start + self.chunk_size - 1 if self.chunk_size > 0 else start
        self.episode_window = {"start": start, "end": end}
        self.world_indices: set[int] = set()
        self.total_steps: float = 0.0
        self.mask_steps: float = 0.0
        self.path_count: int = 0
        self.episode_count: int = 0
        self._episode_fraction_sum: float = 0.0
        self._episode_fraction_count: int = 0
        self.world_breakdown: Dict[int, Dict[str, object]] = {}

    def update(
        self,
        world_index: int,
        episode_fraction: float,
        *,
        episode_total_steps: float,
        episode_mask_steps: float,
        path_count: int,
    ) -> None:
        self.world_indices.add(world_index)
        self.total_steps += float(episode_total_steps)
        self.mask_steps += float(episode_mask_steps)
        world_block = self.world_breakdown.setdefault(
            world_index,
            {
                "total_steps": 0.0,
                "mask_steps": 0.0,
                "path_count": 0,
                "episode_count": 0,
                "episode_fraction_sum": 0.0,
                "episode_fraction_count": 0,
                "lag_breakdown": {},
            },
        )
        world_block["total_steps"] += float(episode_total_steps)
        world_block["mask_steps"] += float(episode_mask_steps)
        world_block["path_count"] += int(path_count)
        world_block["episode_count"] += 1
        self.path_count += int(path_count)

        if math.isfinite(episode_fraction):
            fraction_value = float(episode_fraction)
            self._episode_fraction_sum += fraction_value
            self._episode_fraction_count += 1
            world_block["episode_fraction_sum"] += fraction_value
            world_block["episode_fraction_count"] += 1

    def update_lag_breakdown(
        self,
        *,
        world_index: int,
        world_mod_index: int,
        world_count: int,
        episode_total_steps: float,
        path_count: int,
        lag_fractions: Mapping[int, Tuple[float, float]],
    ) -> None:
        if not lag_fractions:
            return

        if world_index not in self.world_breakdown:
            self.world_breakdown[world_index] = {
                "total_steps": 0.0,
                "mask_steps": 0.0,
                "path_count": 0,
                "episode_count": 0,
                "episode_fraction_sum": 0.0,
                "episode_fraction_count": 0,
                "lag_breakdown": {},
            }

        world_block = self.world_breakdown[world_index]
        world_block.setdefault("world_mod_index", int(world_mod_index))
        if world_count > 0:
            world_block.setdefault("world_count", int(world_count))
        lag_block = cast(Dict[int, Dict[str, float | int]], world_block.setdefault("lag_breakdown", {}))

        for previous_world, (mask_steps, fraction) in lag_fractions.items():
            entry = lag_block.setdefault(
                int(previous_world),
                {
                    "total_steps": 0.0,
                    "mask_steps": 0.0,
                    "path_count": 0,
                    "episode_count": 0,
                    "episode_fraction_sum": 0.0,
                    "episode_fraction_count": 0,
                    "lag_offset": None,
                },
            )
            entry["total_steps"] += float(episode_total_steps)
            entry["mask_steps"] += float(mask_steps)
            entry["path_count"] += int(path_count)
            entry["episode_count"] += 1
            if math.isfinite(fraction):
                entry["episode_fraction_sum"] += float(fraction)
                entry["episode_fraction_count"] += 1
            if world_count > 0:
                lag_offset = int(
                    (int(world_mod_index) - int(previous_world)) % world_count
                )
                if lag_offset > 0:
                    entry["lag_offset"] = lag_offset

    def serialise(self) -> Dict[str, object]:
        fraction = (
            self._episode_fraction_sum / self._episode_fraction_count
            if self._episode_fraction_count > 0
            else float("nan")
        )
        return {
            "chunk_index": self.chunk_index,
            "episode_window": self.episode_window,
            "world_indices": sorted(self.world_indices),
            "path_count": self.path_count,
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "mask_steps": self.mask_steps,
            "episode_fraction_sum": self._episode_fraction_sum,
            "episode_fraction_count": self._episode_fraction_count,
            "occupancy_fraction": fraction,
            "world_breakdown": [
                {
                    "world_index": world_index,
                    "world_mod_index": (
                        int(block["world_mod_index"])
                        if isinstance(block.get("world_mod_index"), (int, float))
                        else None
                    ),
                    "world_count": (
                        int(block["world_count"])
                        if isinstance(block.get("world_count"), (int, float))
                        else None
                    ),
                    "path_count": block["path_count"],
                    "total_steps": block["total_steps"],
                    "mask_steps": block["mask_steps"],
                    "episode_count": block["episode_count"],
                    "episode_fraction_sum": block["episode_fraction_sum"],
                    "episode_fraction_count": block["episode_fraction_count"],
                    "mean_episode_fraction": (
                        block["episode_fraction_sum"]
                        / block["episode_fraction_count"]
                        if block["episode_fraction_count"] > 0
                        else float("nan")
                    ),
                    "lag_breakdown": [
                        {
                            "previous_world_index": previous_world,
                            "path_count": lag_block["path_count"],
                            "total_steps": lag_block["total_steps"],
                            "mask_steps": lag_block["mask_steps"],
                            "episode_count": lag_block["episode_count"],
                            "episode_fraction_sum": lag_block["episode_fraction_sum"],
                            "episode_fraction_count": lag_block["episode_fraction_count"],
                            "lag_offset": (
                                int(lag_block["lag_offset"])
                                if isinstance(lag_block.get("lag_offset"), (int, float))
                                else None
                            ),
                            "mean_episode_fraction": (
                                lag_block["episode_fraction_sum"]
                                / lag_block["episode_fraction_count"]
                                if lag_block["episode_fraction_count"] > 0
                                else float("nan")
                            ),
                        }
                        for previous_world, lag_block in sorted(
                            cast(Dict[int, Dict[str, float | int]], block.get("lag_breakdown", {})).items(),
                            key=lambda item: item[0],
                        )
                    ],
                }
                for world_index, block in sorted(
                    self.world_breakdown.items(), key=lambda item: item[0]
                )
            ],
        }

def _prepare_world_masks(
    worlds: Sequence[np.ndarray],
    hole_thickness: int,
    wall_thickness: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Return pre-computed masks for hole and wall targets."""

    hole_masks: List[np.ndarray] = []
    wall_masks: List[np.ndarray] = []

    for index, world in enumerate(worlds):
        world_array = np.array(world)
        if world_array.ndim != 2:
            world_array = np.squeeze(world_array)
        if world_array.ndim != 2:
            raise ValueError(
                f"World index {index} is not 2D: shape={world_array.shape}"
            )

        hole_masks.append(_expand_mask(_build_hole_mask(world_array), hole_thickness))
        wall_masks.append(_derive_wall_mask(world_array, wall_thickness))

    return hole_masks, wall_masks


def _mask_for_world(
    world_index: int,
    target: str,
    hole_masks: Sequence[np.ndarray],
    wall_masks: Sequence[np.ndarray],
) -> np.ndarray:
    """Return the mask to use for ``world_index`` based on ``target``."""

    if not hole_masks:
        raise ValueError("No masks available for occupancy analysis")

    world_count = len(hole_masks)
    world_mod = world_index % world_count if world_count else 0
    base_mask = hole_masks[world_mod]
    zero_mask = np.zeros_like(base_mask, dtype=float)

    if target == "current_hole_locations":
        return base_mask

    lag_offset = _lag_offset_for_target(target)
    if lag_offset is not None:
        if world_count <= 0:
            return zero_mask

        if world_index < lag_offset or world_count <= lag_offset:
            return zero_mask
        prev_index = (world_index - lag_offset) % world_count
        prev_mask = hole_masks[prev_index]
        if prev_mask.shape != base_mask.shape:
            return zero_mask
        return prev_mask

    if target == "walls":
        return wall_masks[world_mod]

    raise ValueError(f"Unknown occupancy target: {target}")


def _process_seed_prefix(job: _SeedPrefixJob) -> Tuple[
    str,
    str,
    Dict[int, Dict[str, object]],
    Dict[int, List[float]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    """Return per-seed metrics for a single (seed, prefix) combination."""

    chunk_metrics: Dict[int, _ChunkAccumulator] = {}
    model_cache: Dict[int, _OccupancyModelAdapter] = {}
    sample_records: List[Dict[str, object]] = []
    episode_records: List[Dict[str, object]] = []
    sample_config = job.sample_config

    for episode_idx_raw, trajectories in zip(job.episode_indices, job.trajectories):
        if trajectories is None:
            continue
        try:
            episode_idx = int(episode_idx_raw)
        except (TypeError, ValueError):
            continue

        world_index = 0 if job.env_switch_every <= 0 else episode_idx // job.env_switch_every
        chunk_index = episode_idx // job.chunk_size

        world_count = len(job.hole_masks) if job.hole_masks else 0
        world_mod = world_index % world_count if world_count else 0
        grid_size = job.hole_masks[world_mod].shape[0]

        mask = _mask_for_world(
            world_index,
            job.occupancy_target,
            job.hole_masks,
            job.wall_masks,
        )

        lag_masks: Dict[int, np.ndarray] = {}
        lag_offset = _lag_offset_for_target(job.occupancy_target)
        if job.hole_masks and world_count > 0 and lag_offset is not None:
            if job.occupancy_target == "previous_hole_locations":
                max_lag = min(world_index, world_count - 1)
                lag_range = range(1, max_lag + 1)
            elif world_index >= lag_offset and world_count > lag_offset:
                lag_range = range(lag_offset, lag_offset + 1)
            else:
                lag_range = range(0)

            for lag in lag_range:
                if lag <= 0:
                    continue

                previous_world = (world_index - lag) % world_count
                if previous_world < 0 or previous_world >= world_count:
                    continue
                lag_masks.setdefault(previous_world, job.hole_masks[previous_world])

        reward_locations: Sequence[Tuple[int, int]] = (
            job.reward_locations[world_mod] if job.reward_locations else []
        )
        episode_allowed = (
            not sample_config or sample_config.episode_within_window(episode_idx)
        )

        accumulator = chunk_metrics.setdefault(
            chunk_index, _ChunkAccumulator(chunk_index, job.chunk_size)
        )

        accumulator.episode_count += 1

        episode_total_steps = 0.0
        episode_mask_steps = 0.0
        episode_path_count = 0
        lag_episode_mask_steps: Dict[int, float] = {key: 0.0 for key in lag_masks}

        for path in trajectories:
            occupancy = _compute_path_occupancy(path, grid_size, model_cache)
            if occupancy.size == 0:
                continue
            if mask.shape != occupancy.shape:
                mask_aligned = np.zeros_like(occupancy)
            else:
                mask_aligned = mask
            total_steps = float(occupancy.sum())
            mask_steps = float(np.sum(occupancy * mask_aligned))
            episode_total_steps += total_steps
            episode_mask_steps += mask_steps
            episode_path_count += 1

            for previous_world, lag_mask in lag_masks.items():
                if lag_mask.shape != occupancy.shape:
                    lag_aligned = np.zeros_like(occupancy)
                else:
                    lag_aligned = lag_mask
                lag_steps = float(np.sum(occupancy * lag_aligned))
                lag_episode_mask_steps[previous_world] = (
                    lag_episode_mask_steps.get(previous_world, 0.0) + lag_steps
                )

            path_fraction = (
                mask_steps / total_steps
                if total_steps > 0 and math.isfinite(total_steps)
                else float("nan")
            )

            if (
                sample_config
                and episode_allowed
                and (sample_config.max_samples is None
                     or len(sample_records) < sample_config.max_samples)
            ):
                start_coord = _first_coordinate(path)
                if start_coord is None:
                    continue

                nearest_reward_coord, start_distance = _nearest_reward(
                    start_coord, reward_locations
                )
                if start_distance is None:
                    continue
                if (
                    sample_config.min_start_distance is not None
                    and start_distance < sample_config.min_start_distance
                ):
                    continue
                if not math.isfinite(path_fraction):
                    continue
                if not sample_config.occupancy_within_thresholds(path_fraction):
                    continue

                delay = episode_idx - (world_index * job.env_switch_every)
                sample_records.append(
                    {
                        "seed": job.seed,
                        "agent": job.prefix,
                        "episode": int(episode_idx),
                        "world_index": int(world_index),
                        "chunk_index": int(chunk_index),
                        "delay": int(delay),
                        "occupancy_fraction": float(path_fraction),
                        "total_steps": float(total_steps),
                        "mask_steps": float(mask_steps),
                        "start_row": int(start_coord[0]),
                        "start_col": int(start_coord[1]),
                        "start_distance": float(start_distance),
                        "reward_row": (nearest_reward_coord[0]
                                        if nearest_reward_coord else None),
                        "reward_col": (nearest_reward_coord[1]
                                        if nearest_reward_coord else None),
                    }
                )

        if (
            episode_total_steps > 0
            and math.isfinite(episode_total_steps)
            and math.isfinite(episode_mask_steps)
        ):
            episode_fraction = episode_mask_steps / episode_total_steps
        else:
            episode_fraction = float("nan")

        lag_fractions: Dict[int, Tuple[float, float]] = {}
        for previous_world, mask_steps in lag_episode_mask_steps.items():
            if episode_total_steps > 0 and math.isfinite(episode_total_steps):
                fraction = mask_steps / episode_total_steps
            else:
                fraction = float("nan")
            lag_fractions[previous_world] = (mask_steps, fraction)

        # Record per-episode metrics for this target
        episode_row: Dict[str, object] = {
            "seed": job.seed,
            "agent": job.prefix,
            "episode": int(episode_idx),
            "world_index": int(world_index),
            "chunk_index": int(chunk_index),
            "occupancy_fraction": float(episode_fraction),
            "total_steps": float(episode_total_steps),
            "path_count": int(episode_path_count),
        }
        if job.occupancy_target == "previous_hole_locations" and job.hole_masks:
            world_count = len(job.hole_masks)
            if world_count > 1:
                world_mod = world_index % world_count
                # Fill lag_k columns by converting previous_world to lag offset
                for k in range(1, world_count):
                    episode_row[f"lag_{k}"] = float("nan")
                for previous_world, (mask_steps, _) in lag_fractions.items():
                    if episode_total_steps <= 0 or not math.isfinite(episode_total_steps):
                        continue
                    offset = (int(world_mod) - int(previous_world)) % world_count
                    if offset <= 0:
                        continue
                    value = float(mask_steps) / float(episode_total_steps)
                    if math.isfinite(value):
                        episode_row[f"lag_{offset}"] = value
        episode_records.append(episode_row)

        accumulator.update(
            world_index=world_index,
            episode_fraction=episode_fraction,
            episode_total_steps=episode_total_steps,
            episode_mask_steps=episode_mask_steps,
            path_count=episode_path_count,
        )
        if lag_fractions:
            accumulator.update_lag_breakdown(
                world_index=world_index,
                world_mod_index=world_mod,
                world_count=world_count,
                episode_total_steps=episode_total_steps,
                path_count=episode_path_count,
                lag_fractions=lag_fractions,
            )

    if not chunk_metrics:
        return job.seed, job.prefix, {}, {}, [], sample_records

    serialised_chunks: Dict[int, Dict[str, object]] = {}
    aggregate_contrib: Dict[int, List[float]] = {}
    per_seed_records: List[Dict[str, object]] = []

    for chunk_index, accumulator in sorted(chunk_metrics.items()):
        record = accumulator.serialise()
        serialised_chunks[chunk_index] = record

        fraction = record["occupancy_fraction"]
        if isinstance(fraction, (int, float)) and math.isfinite(fraction):
            aggregate_contrib.setdefault(chunk_index, []).append(float(fraction))

        per_seed_records.append(
            {
                "seed": job.seed,
                "agent": job.prefix,
                "chunk_index": chunk_index,
                "episode_start": record["episode_window"]["start"],
                "episode_end": record["episode_window"]["end"],
                "world_indices": " ".join(
                    str(idx) for idx in record["world_indices"]
                ),
                "path_count": record["path_count"],
                "episode_count": record["episode_count"],
                "total_steps": record["total_steps"],
                "mask_steps": record["mask_steps"],
                "occupancy_fraction": fraction,
            }
        )

    return (
        job.seed,
        job.prefix,
        serialised_chunks,
        aggregate_contrib,
        per_seed_records,
        sample_records,
        episode_records,
    )


def _analyse_target(
    structure: Dict[str, object],
    metadata: object | None,
    run_path: Path,
    job_id: Optional[int],
    env_switch_every: int,
    chunk_size: int,
    occupancy_target: str,
    barrier_thickness: int,
    wall_thickness: int,
    output_dir: Path,
    *,
    max_workers: Optional[int],
    enable_parallel: bool,
    sample_config: Optional[_SampleConfig] = None,
) -> None:
    """Generate occupancy metrics for ``occupancy_target`` and persist outputs."""

    seed_summaries: Dict[str, Dict[str, object]] = {}
    aggregate_tracker: Dict[str, Dict[int, List[float]]] = {}
    aggregate_records: List[Dict[str, object]] = []
    per_seed_records: List[Dict[str, object]] = []
    selected_samples: List[Dict[str, object]] = []
    jobs: List[_SeedPrefixJob] = []
    seeds_ready_for_paths: set[str] = set()

    for seed_key, seed_block in structure.items():
        seed_name = str(seed_key)
        seed_summary: Dict[str, object] = {
            "world_count": 0,
            "chunk_metrics": {},
            "warnings": [],
        }
        seed_summaries[seed_name] = seed_summary
        warnings_list = cast(List[str], seed_summary["warnings"])

        worlds = None
        if isinstance(seed_block, dict) and "worlds" in seed_block:
            worlds = np.array(seed_block["worlds"])
        elif isinstance(seed_block, dict):
            for variant_block in seed_block.values():
                if isinstance(variant_block, dict) and "worlds" in variant_block:
                    worlds = np.array(variant_block["worlds"])
                    break

        if worlds is None or getattr(worlds, "size", 0) == 0:
            warnings_list.append("missing_world_data")
            continue

        flat_worlds: List[np.ndarray] = []
        for idx in range(worlds.shape[0]):
            world = np.array(worlds[idx])
            if world.ndim != 2:
                world = np.squeeze(world)
            if world.ndim != 2:
                raise ValueError(
                    f"Seed {seed_name} world index {idx} is not 2D: shape={world.shape}"
                )
            flat_worlds.append(world)

        if not flat_worlds:
            warnings_list.append("missing_world_data")
            continue

        seed_summary["world_count"] = len(flat_worlds)

        reward_locations: List[List[Tuple[int, int]]] = []
        for world in flat_worlds:
            reward_coords = np.argwhere(np.asarray(world) == -1)
            if reward_coords.size == 0:
                reward_locations.append([])
            else:
                reward_locations.append(
                    [(int(row), int(col)) for row, col in reward_coords.tolist()]
                )

        hole_masks, wall_masks = _prepare_world_masks(
            flat_worlds,
            barrier_thickness,
            wall_thickness,
        )
        if not hole_masks:
            warnings_list.append("missing_world_masks")
            continue
        seeds_ready_for_paths.add(seed_name)

        for prefix in ("unlesioned", "lesionLEC"):
            try:
                paths_y, paths_x = get_parameter_values(
                    "paths", {seed_key: seed_block}, prefix=prefix
                )
            except Exception as exc:
                warnings_list.append(f"paths_error_{prefix}:{exc}")
                continue

            if not paths_y or not paths_x:
                continue

            paths_y_sorted, paths_x_sorted = sort_paths(paths_y, paths_x)
            if not paths_y_sorted or not paths_x_sorted:
                continue

            jobs.append(
                _SeedPrefixJob(
                    seed=seed_name,
                    prefix=prefix,
                    env_switch_every=env_switch_every,
                    chunk_size=chunk_size,
                    occupancy_target=occupancy_target,
                    hole_masks=hole_masks,
                    wall_masks=wall_masks,
                    reward_locations=reward_locations,
                    episode_indices=paths_x_sorted,
                    trajectories=paths_y_sorted,
                    sample_config=(
                        sample_config
                        if sample_config and sample_config.allows_agent(prefix)
                        else None
                    ),
                )
            )

    if enable_parallel:
        pool_worker_count = max_workers if max_workers and max_workers > 0 else None
    else:
        pool_worker_count = 1
    job_results: List[
        Tuple[
            str,
            str,
            Dict[int, Dict[str, object]],
            Dict[int, List[float]],
            List[Dict[str, object]],
            List[Dict[str, object]],
        ]
    ] = []

    if not jobs:
        job_results = []
    elif not enable_parallel or pool_worker_count == 1 or len(jobs) == 1:
        job_results = [_process_seed_prefix(job) for job in jobs]
    else:
        ordered_results: List[Optional[
            Tuple[
                str,
                str,
                Dict[int, Dict[str, object]],
                Dict[int, List[float]],
                List[Dict[str, object]],
                List[Dict[str, object]],
            ]
        ]] = [None] * len(jobs)
        with ProcessPoolExecutor(max_workers=pool_worker_count) as executor:
            future_to_index = {
                executor.submit(_process_seed_prefix, job): index
                for index, job in enumerate(jobs)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                ordered_results[index] = future.result()

        job_results = [result for result in ordered_results if result is not None]

    for (
        seed_name,
        prefix,
        serialised_chunks,
        aggregate_contrib,
        prefix_records,
        sample_records,
        episode_records,
    ) in job_results:
        chunk_metrics_map = cast(
            Dict[str, Dict[int, Dict[str, object]]],
            seed_summaries[seed_name].setdefault("chunk_metrics", {}),
        )
        if serialised_chunks:
            chunk_metrics_map[prefix] = serialised_chunks

        if aggregate_contrib:
            aggregate_prefix_map = aggregate_tracker.setdefault(prefix, {})
            for chunk_index, fractions in aggregate_contrib.items():
                aggregate_prefix_map.setdefault(chunk_index, []).extend(fractions)

        per_seed_records.extend(prefix_records)
        selected_samples.extend(sample_records)
        if episode_records:
            seed_summaries[seed_name].setdefault("episode_records", {})[prefix] = episode_records

    for seed_name in seeds_ready_for_paths:
        chunk_metrics_map = cast(
            Dict[str, Dict[int, Dict[str, object]]],
            seed_summaries[seed_name].get("chunk_metrics", {}),
        )
        if not chunk_metrics_map:
            warnings_list = cast(List[str], seed_summaries[seed_name]["warnings"])
            if "missing_paths" not in warnings_list:
                warnings_list.append("missing_paths")

    for prefix, chunk_map in aggregate_tracker.items():
        for chunk_index, fractions in chunk_map.items():
            values = np.asarray(fractions, dtype=float)
            if values.size == 0:
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            aggregate_records.append(
                {
                    "agent": prefix,
                    "chunk_index": chunk_index,
                    "seed_count": int(values.size),
                    "mean_fraction": mean,
                    "std_fraction": std,
                }
            )

    aggregate_summary: Dict[str, Dict[str, object]] = {}
    for prefix, chunk_map in aggregate_tracker.items():
        aggregate_summary[prefix] = {}
        for chunk_index, fractions in sorted(chunk_map.items()):
            values = np.asarray(fractions, dtype=float)
            if values.size == 0:
                continue
            aggregate_summary[prefix][str(chunk_index)] = {
                "seed_count": int(values.size),
                "mean_fraction": float(values.mean()),
                "std_fraction": float(values.std(ddof=0)),
            }

    summary = {
        "job_id": job_id,
        "run_path": str(run_path),
        "env_switch_every": env_switch_every,
        "chunk_size": chunk_size,
        "occupancy_target": occupancy_target,
        "barrier_thickness": barrier_thickness,
        "wall_thickness": wall_thickness,
        "metadata": metadata,
        "seeds": seed_summaries,
        "aggregate": aggregate_summary,
        "sample_paths": {
            "config": sample_config.to_dict() if sample_config else None,
            "records": selected_samples,
        },
        "episodes": seed_summaries and {
            seed: {
                prefix: records
                for prefix, records in cast(Dict[str, List[Dict[str, object]]], seed_block.get("episode_records", {})).items()
            }
            for seed, seed_block in seed_summaries.items()
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_json = output_dir / "chunked_occupancy_metrics.json"
    metrics_json.write_text(json.dumps(summary, indent=2, sort_keys=True))

    metrics_csv = output_dir / "chunked_occupancy_metrics.csv"
    with metrics_csv.open("w", encoding="utf-8") as handle:
        header = [
            "seed",
            "agent",
            "chunk_index",
            "episode_start",
            "episode_end",
            "world_indices",
            "path_count",
            "episode_count",
            "total_steps",
            "mask_steps",
            "occupancy_fraction",
        ]
        handle.write(",".join(header) + "\n")
        for record in per_seed_records:
            row = [
                record["seed"],
                record["agent"],
                str(record["chunk_index"]),
                str(record["episode_start"]),
                str(record["episode_end"]),
                record["world_indices"],
                str(record["path_count"]),
                str(record["episode_count"]),
                f"{record['total_steps']:.6g}",
                f"{record['mask_steps']:.6g}",
                (
                    f"{record['occupancy_fraction']:.6g}"
                    if isinstance(record["occupancy_fraction"], (int, float))
                    and math.isfinite(record["occupancy_fraction"])
                    else "nan"
                ),
            ]
            handle.write(",".join(row) + "\n")

    if selected_samples:
        sample_csv = output_dir / "chunked_occupancy_sample_paths.csv"
        with sample_csv.open("w", encoding="utf-8") as handle:
            header = [
                "seed",
                "agent",
                "episode",
                "world_index",
                "chunk_index",
                "delay",
                "occupancy_fraction",
                "total_steps",
                "mask_steps",
                "start_row",
                "start_col",
                "start_distance",
                "reward_row",
                "reward_col",
            ]
            handle.write(",".join(header) + "\n")
            for record in sorted(
                selected_samples,
                key=lambda item: (
                    item["agent"],
                    item["seed"],
                    item["episode"],
                    item["delay"],
                ),
            ):
                row = [
                    str(record["seed"]),
                    str(record["agent"]),
                    str(record["episode"]),
                    str(record["world_index"]),
                    str(record["chunk_index"]),
                    str(record["delay"]),
                    (
                        f"{record['occupancy_fraction']:.6g}"
                        if isinstance(record["occupancy_fraction"], (int, float))
                        and math.isfinite(record["occupancy_fraction"])
                        else "nan"
                    ),
                    f"{record['total_steps']:.6g}",
                    f"{record['mask_steps']:.6g}",
                    str(record["start_row"]),
                    str(record["start_col"]),
                    f"{record['start_distance']:.6g}",
                    (str(record["reward_row"]) if record["reward_row"] is not None else ""),
                    (str(record["reward_col"]) if record["reward_col"] is not None else ""),
                ]
                handle.write(",".join(row) + "\n")

    aggregate_csv = output_dir / "chunked_occupancy_fraction_summary.csv"
    with aggregate_csv.open("w", encoding="utf-8") as handle:
        header = [
            "agent",
            "chunk_index",
            "seed_count",
            "mean_fraction",
            "std_fraction",
        ]
        handle.write(",".join(header) + "\n")
        for record in sorted(
            aggregate_records, key=lambda item: (item["agent"], item["chunk_index"])
        ):
            row = [
                record["agent"],
                str(record["chunk_index"]),
                str(record["seed_count"]),
                f"{record['mean_fraction']:.6g}",
                f"{record['std_fraction']:.6g}",
            ]
            handle.write(",".join(row) + "\n")

    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)

    color_map = {"unlesioned": "blue", "lesionLEC": "red"}
    prefixes_with_data = [
        prefix for prefix, chunk_map in aggregate_summary.items() if chunk_map
    ]

    if prefixes_with_data:
        all_chunk_indices = sorted(
            {
                int(index)
                for prefix in prefixes_with_data
                for index in aggregate_summary[prefix].keys()
            }
        )

        chunk_episode_pairs = [(idx, idx * chunk_size) for idx in all_chunk_indices]
        if env_switch_every and env_switch_every > 0:
            switch_aligned_pairs = [
                (idx, episode)
                for idx, episode in chunk_episode_pairs
                if episode % env_switch_every == 0
            ]
        else:
            switch_aligned_pairs = []

        x_values = [episode for _, episode in chunk_episode_pairs]
        chunk_indices_for_plotting = [idx for idx, _ in chunk_episode_pairs]
        switch_xticks = [episode for _, episode in switch_aligned_pairs] or x_values


        fig, ax = plt.subplots(figsize=(8, 4))
        legend_label_map = {
            "unlesioned": "allocentric + egocentric",
            "lesionLEC": "allocentric",
        }
        all_lower_bounds: List[float] = []
        all_upper_bounds: List[float] = []

        for prefix in prefixes_with_data:
            chunk_map = aggregate_summary[prefix]
            means: List[float] = []
            stds: List[float] = []
            for idx in chunk_indices_for_plotting:
                record = chunk_map.get(str(idx))
                if record is None:
                    means.append(np.nan)
                    stds.append(np.nan)
                else:
                    means.append(record["mean_fraction"])
                    stds.append(record["std_fraction"])

            color = color_map.get(prefix)
            label = legend_label_map.get(prefix, prefix)
            mean_array = np.asarray(means, dtype=float)
            std_array = np.asarray(stds, dtype=float)
            lower = mean_array - std_array
            upper = mean_array + std_array

            finite_lower = lower[np.isfinite(lower)]
            finite_upper = upper[np.isfinite(upper)]
            if finite_lower.size:
                all_lower_bounds.extend(finite_lower.tolist())
            if finite_upper.size:
                all_upper_bounds.extend(finite_upper.tolist())

            ax.plot(
                x_values,
                mean_array,
                color=color,
                linewidth=1.5,
                label=label,
            )
            ax.fill_between(
                x_values,
                lower,
                upper,
                color=color,
                alpha=0.2,
                linewidth=0,
            )

        ax.set_xlabel("Episode (chunk start)")
        ax.set_ylabel("Proportion of time spent within barriers")
        ax.set_title(
            f"Chunked occupancy vs {occupancy_target.replace('_', ' ')}"
        )
        if all_lower_bounds and all_upper_bounds:
            y_min = min(all_lower_bounds)
            y_max = max(all_upper_bounds)
            if math.isfinite(y_min) and math.isfinite(y_max):
                y_range = y_max - y_min
                if y_range <= 0:
                    y_range = max(abs(y_max), 1.0) * 0.1
                margin = y_range * 0.1
                y_min = max(0.0, y_min - margin)
                y_max = min(1.0, y_max + margin)
                if y_max <= y_min:
                    center = (y_max + y_min) / 2.0
                    half_range = max(y_range * 0.5, 0.05)
                    y_min = max(0.0, center - half_range)
                    y_max = min(1.0, center + half_range)
                ax.set_ylim(y_min, y_max)
            else:
                ax.set_ylim(0, 1)
        else:
            ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xticks(switch_xticks, [str(value) for value in switch_xticks])
        if len(x_values) > 1:
            ax.set_xlim(x_values[0], x_values[-1])

        world_switch_handle = None
        if switch_aligned_pairs:
            for _, episode in switch_aligned_pairs:
                if episode == 0:
                    continue
                line = ax.axvline(
                    episode,
                    color="grey",
                    linestyle=":",
                    linewidth=1,
                    label="world switch" if world_switch_handle is None else "_nolegend_",
                )
                if world_switch_handle is None:
                    world_switch_handle = line

        if world_switch_handle is not None:
            handles, labels = ax.get_legend_handles_labels()
            if world_switch_handle not in handles:
                handles.append(world_switch_handle)
                labels.append("world switch")
            ax.legend(handles, labels)
        else:
            ax.legend()
        fig.tight_layout()

        figure_stem = (
            plot_dir
            / f"lesion_vs_unlesioned_{occupancy_target}_chunked_occupancy"
        )
        figure_png = figure_stem.with_suffix(".png")
        figure_svg = figure_stem.with_suffix(".svg")
        fig.savefig(figure_png, dpi=200)
        fig.savefig(figure_svg)
        plt.close(fig)
        print(f"Saved chunked occupancy figure to {figure_png}")

    print(
        "Saved chunked occupancy metrics to"
        f" {metrics_json} (target={occupancy_target})"
    )


def _analyse_target_entrypoint(payload: Dict[str, object]) -> None:
    """ProcessPool entrypoint forwarding arguments to :func:`_analyse_target`."""

    _analyse_target(**payload)