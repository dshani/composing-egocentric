"""Analyse how often agents occupy cells near the outer walls of the grid."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np



class _SquareEnvAdapter:
    """Lightweight adapter that mimics the attributes used by plotting helpers."""

    def __init__(self, size: int) -> None:
        self.size = int(size)

    def get_2d_pos(self, state: int) -> Tuple[int, int]:
        state = int(state)
        return divmod(state, self.size)


class _OccupancyModelAdapter:
    """Wrapper that provides a minimal ``env`` namespace."""

    def __init__(self, size: int) -> None:
        self.env = _SquareEnvAdapter(size)


# def _resolve_run_path(job_id: int, results_dir: Path) -> Tuple[Path, Path]:
#     """Return the run directory referenced by ``run_path.txt`` for ``job_id``.

#     The helper searches ``job_id + 1`` (new-style) and ``job_id`` (legacy) slots.
#     The returned tuple contains the resolved run path and the directory that
#     housed the metadata files (``Results/job_ids/<slot>``).
#     """

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


def _iter_coordinates(path: Sequence) -> Iterable[Tuple[int, int]]:
    """Yield (row, col) pairs from a stored trajectory.

    The saved ``paths`` entries store sequences of egocentric positions. Each
    position may be represented as a list, tuple, or NumPy array with at least
    two elements. This helper flattens the nested structure and returns the
    first two entries as integer coordinates.
    """

    for step in path:
        if step is None:
            continue
        if isinstance(step, (list, tuple)):
            if not step:
                continue
            if isinstance(step[0], (list, tuple, np.ndarray)):
                # Some save files may have nested coordinate arrays. Recurse.
                for coord in _iter_coordinates(step):
                    yield coord
                continue
            values = step
        elif isinstance(step, np.ndarray):
            if step.size == 0:
                continue
            if step.ndim > 1:
                for coord in _iter_coordinates(step):
                    yield coord
                continue
            values = step.tolist()
        else:
            continue

        if len(values) >= 2:
            try:
                row = int(values[0])
                col = int(values[1])
            except (TypeError, ValueError):
                continue
            yield row, col


def _convert_for_helper(path: Sequence, grid_size: int) -> List[Tuple[int, Optional[int]]]:
    """Transform raw coordinates into the format expected by ``get_occupancy_plot``."""

    converted: List[Tuple[int, Optional[int]]] = []
    for row, col in _iter_coordinates(path):
        if row < 0 or col < 0:
            continue
        if row >= grid_size or col >= grid_size:
            continue
        state_index = row * grid_size + col
        converted.append((state_index, None))
    return converted


def _expand_mask(mask: np.ndarray, thickness: int) -> np.ndarray:
    """Dilate ``mask`` so neighbours within ``thickness`` steps are included."""

    thickness = int(thickness)
    if thickness <= 0:
        return mask.astype(float, copy=True)

    base = mask.astype(bool, copy=False)
    rows, cols = base.shape
    if rows == 0 or cols == 0:
        return mask.astype(float, copy=True)

    padded = np.pad(base, thickness, mode="constant", constant_values=False)
    expanded = base.copy()

    offsets = [
        (dr, dc)
        for dr in range(-thickness, thickness + 1)
        for dc in range(-thickness, thickness + 1)
        if abs(dr) + abs(dc) <= thickness
    ]

    for dr, dc in offsets:
        if dr == 0 and dc == 0:
            continue
        row_slice = slice(thickness + dr, thickness + dr + rows)
        col_slice = slice(thickness + dc, thickness + dc + cols)
        expanded |= padded[row_slice, col_slice]

    return expanded.astype(float, copy=False)


def _derive_wall_mask(world: np.ndarray, thickness: int) -> np.ndarray:
    """Return a binary mask that highlights cells close to the outer walls."""

    world = np.asarray(world)
    if world.ndim != 2:
        world = np.squeeze(world)
    if world.ndim != 2:
        raise ValueError("Expected a 2D array for world layout")

    rows, cols = world.shape
    if rows == 0 or cols == 0:
        return np.zeros_like(world, dtype=float)

    row_indices = np.arange(rows)[:, None]
    col_indices = np.arange(cols)[None, :]
    on_boundary = (
        (row_indices == 0)
        | (row_indices == rows - 1)
        | (col_indices == 0)
        | (col_indices == cols - 1)
    )

    boundary_mask = on_boundary.astype(float)
    expanded = _expand_mask(boundary_mask, thickness)

    # Exclude the impassable outer wall itself from the reported area while
    # retaining the neighbouring interior cells.
    interior_mask = expanded.astype(bool)
    interior_mask &= ~on_boundary

    return interior_mask.astype(float)



# def _extract_worlds(seed_block: object) -> Optional[np.ndarray]:
#     """Return the stored worlds array for ``seed_block`` if present."""

#     if not isinstance(seed_block, dict):
#         return None

#     if "worlds" in seed_block:
#         return np.array(seed_block["worlds"])

#     for variant_block in seed_block.values():
#         if isinstance(variant_block, dict) and "worlds" in variant_block:
#             return np.array(variant_block["worlds"])

#     return None



# def _aggregate_paths_by_world(
#     paths_y: Sequence[Sequence],
#     paths_x: Sequence[Sequence],
#     env_switch_every: int,
# ) -> Dict[int, List[Sequence]]:
#     """Group saved trajectories by the world index they belong to."""

#     paths_y_sorted, paths_x_sorted = sort_paths(paths_y, paths_x)

#     world_paths: Dict[int, List[Sequence]] = {}
#     for episode_idx, trajectories in zip(paths_x_sorted, paths_y_sorted):
#         if trajectories is None:
#             continue
#         try:
#             episode = int(episode_idx)
#         except (TypeError, ValueError):
#             continue
#         if env_switch_every <= 0:
#             world_index = 0
#         else:
#             world_index = episode // env_switch_every
#         world_paths.setdefault(world_index, []).extend(list(trajectories))
#     return world_paths


# def _compute_occupancy(
#     paths: Iterable[Sequence],
#     grid_size: int,
#     model_cache: Dict[int, _OccupancyModelAdapter],
# ) -> np.ndarray:
#     """Return an occupancy heatmap aggregated across ``paths``."""

#     if grid_size not in model_cache:
#         model_cache[grid_size] = _OccupancyModelAdapter(grid_size)
#     model = model_cache[grid_size]

#     occupancy = np.zeros((grid_size, grid_size), dtype=float)
#     for path in paths:
#         converted = _convert_for_helper(path, grid_size)
#         if not converted:
#             continue
#         occupancy += get_occupancy_plot(model, converted)
#     return occupancy


# def _normalise(matrix: np.ndarray) -> np.ndarray:
#     total = float(matrix.sum())
#     if total <= 0 or not math.isfinite(total):
#         return np.zeros_like(matrix)
#     return matrix / total


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Analyse how often agents occupy cells near the outer walls"
#     )
#     parser.add_argument("--job-id", type=int, required=False,
#                         help="Training job identifier used to resolve run_path.txt")
#     parser.add_argument("--results-dir", type=Path, default=Path("Results/job_ids"),
#                         help="Directory that contains job metadata (default: Results/job_ids)")
#     parser.add_argument("--run-path", type=Path, default=None,
#                         help="Override the run directory instead of reading run_path.txt")
#     parser.add_argument("--env-switch-every", type=int, default=1000,
#                         help="Number of episodes between world switches")
#     parser.add_argument("--max-workers", type=int, default=8,
#                         help="Thread pool size for loading trajectories")
#     parser.add_argument("--world-metadata", type=str, default=None,
#                         help="Optional JSON string describing the worlds that were used")
#     parser.add_argument("--output", type=Path, default=None,
#                         help="Optional directory for the output metrics (defaults to job folder)")
#     parser.add_argument(
#         "--barrier-thickness",
#         type=int,
#         default=1,
#         help=(
#             "Include cells within this many Manhattan steps of the boundary when"
#             " scoring near-wall occupancy (default: 1)"
#         ),
#     )
#     return parser.parse_args()


# def main() -> None:
#     args = parse_args()

#     if args.run_path is None and args.job_id is None:
#         raise SystemExit("Specify either --run-path or --job-id.")

#     results_dir = args.results_dir.resolve()
#     results_dir.mkdir(parents=True, exist_ok=True)

#     job_id = args.job_id
#     job_dir: Optional[Path] = None
#     run_path: Path
#     if args.run_path is not None:
#         run_path = args.run_path.expanduser().resolve()
#         if job_id is not None:
#             job_dir = results_dir / str(job_id)
#     else:
#         assert job_id is not None  # for type checkers
#         run_path, job_dir = _resolve_run_path(job_id, results_dir)

#     if job_dir is None and job_id is not None:
#         job_dir = results_dir / str(job_id)
#     if job_dir is None:
#         raise SystemExit("Unable to determine output directory for metrics.")
#     job_dir.mkdir(parents=True, exist_ok=True)

#     if not run_path.exists():
#         raise FileNotFoundError(f"Resolved run directory does not exist: {run_path}")

#     # ``load_structure`` expects the parent directories that contain date-stamped
#     # folders. ``run_path`` points at ``.../<date>/<run_name>``.
#     if len(run_path.parents) < 2:
#         raise FileNotFoundError(
#             "Run directory is missing parent hierarchy required for load_structure"
#         )
#     save_root = run_path.parents[1]

#     structure = load_structure(
#         run=None,
#         date=None,
#         seed=None,
#         save_dirs=[save_root],
#         compare="lesion",
#         dict_params=["paths"],
#         seeds_path=run_path,
#         max_workers=args.max_workers,
#     )

#     if not structure:
#         raise RuntimeError("No data was loaded from the run directory")

#     seed_summaries: Dict[str, Dict[str, object]] = {}
#     overall_metrics: List[Dict[str, object]] = []
#     fraction_accumulator: Dict[int, Dict[str, List[float]]] = {}
#     model_cache: Dict[int, _OccupancyModelAdapter] = {}

#     for seed_key, seed_block in structure.items():
#         seed_name = str(seed_key)
#         seed_summary: Dict[str, object] = {
#             "world_count": 0,
#             "metrics": [],
#             "warnings": [],
#         }
#         seed_summaries[seed_name] = seed_summary
#         warnings_list = cast(List[str], seed_summary["warnings"])

#         worlds = _extract_worlds(seed_block)
#         if worlds is None or getattr(worlds, "size", 0) == 0:
#             print(f"WARNING: seed {seed_name} does not contain world layouts; skipping")
#             warnings_list.append("missing_world_data")
#             continue

#         # Normalise worlds into a list of 2D arrays.
#         flat_worlds: List[np.ndarray] = []
#         for idx in range(worlds.shape[0]):
#             world = np.array(worlds[idx])
#             if world.ndim != 2:
#                 world = np.squeeze(world)
#             if world.ndim != 2:
#                 raise ValueError(
#                     f"Seed {seed_name} world index {idx} is not 2D: shape={world.shape}"
#                 )
#             flat_worlds.append(world)

#         seed_summary["world_count"] = len(flat_worlds)

#         if not flat_worlds:
#             print(f"WARNING: seed {seed_name} has no world layouts after normalisation; skipping")
#             warnings_list.append("missing_world_data")
#             continue

#         per_seed_structure = {seed_key: seed_block}
#         grouped_paths: Dict[str, Dict[int, List[Sequence]]] = {}
#         for prefix in ("unlesioned", "lesionLEC"):
#             try:
#                 paths_y, paths_x = get_parameter_values("paths", per_seed_structure, prefix=prefix)
#                 if paths_y and paths_x:
#                     grouped_paths[prefix] = _aggregate_paths_by_world(
#                         paths_y, paths_x, args.env_switch_every
#                     )
#             except Exception as exc:  # pragma: no cover - defensive catch
#                 print(f"WARNING: failed to read paths for seed {seed_name} {prefix}: {exc}")

#         if not grouped_paths:
#             print(f"WARNING: seed {seed_name} has no saved paths; skipping")
#             warnings_list.append("missing_paths")
#             continue

#         seed_metrics: List[Dict[str, object]] = []
#         for world_index, world in enumerate(flat_worlds):
#             if world.shape[0] != world.shape[1]:
#                 raise ValueError(
#                     f"Seed {seed_name} world index {world_index} is not square: shape={world.shape}"
#                 )

#             grid_size = world.shape[0]
#             mask = _derive_wall_mask(world, args.barrier_thickness)

#             if mask.shape != world.shape:
#                 raise ValueError("Wall mask shape mismatch")
#             mask_sum = float(mask.sum())

#             agents_block: Dict[str, Dict[str, float]] = {}
#             for prefix, grouped in grouped_paths.items():
#                 world_paths = grouped.get(world_index, [])
#                 occupancy = _compute_occupancy(world_paths, grid_size, model_cache)
#                 total_steps = float(occupancy.sum())
#                 wall_steps = float(np.sum(occupancy * mask))
#                 occupancy_norm = _normalise(occupancy)
#                 wall_fraction = float(np.sum(occupancy_norm * mask))

#                 agents_block[prefix] = {
#                     "path_count": len(world_paths),
#                     "total_steps": total_steps,
#                     "near_wall_steps": wall_steps,
#                     "near_wall_fraction": wall_fraction,
#                 }

#                 if math.isfinite(wall_fraction):
#                     fraction_accumulator.setdefault(world_index, {}).setdefault(prefix, []).append(
#                         wall_fraction
#                     )

#             block = {
#                 "seed": seed_name,
#                 "world_index": world_index,
#                 "near_wall_cells": mask_sum,
#                 "episode_window": {
#                     "start": world_index * args.env_switch_every,
#                     "end": (world_index + 1) * args.env_switch_every - 1,
#                 },
#                 "agents": agents_block,
#             }
#             seed_metrics.append(block)
#             overall_metrics.append(block)

#         seed_summary["metrics"] = seed_metrics

#     if not overall_metrics:
#         raise RuntimeError("No occupancy metrics were produced for any seed")

#     metadata = None
#     if args.world_metadata:
#         try:
#             metadata = json.loads(args.world_metadata)
#         except json.JSONDecodeError:
#             metadata = args.world_metadata

#     aggregate_fractions: Dict[str, Dict[str, Dict[str, object]]] = {}
#     aggregate_records: List[Dict[str, object]] = []
#     for world_index in sorted(fraction_accumulator.keys()):
#         per_agent = fraction_accumulator[world_index]
#         aggregate_fractions[str(world_index)] = {}
#         for agent, fractions in per_agent.items():
#             if not fractions:
#                 continue
#             values = np.asarray(fractions, dtype=float)
#             mean = float(values.mean())
#             std = float(values.std(ddof=0))
#             aggregate_fractions[str(world_index)][agent] = {
#                 "seed_count": int(values.size),
#                 "mean_fraction": mean,
#                 "std_fraction": std,
#             }
#             aggregate_records.append(
#                 {
#                     "world_index": world_index,
#                     "agent": agent,
#                     "seed_count": int(values.size),
#                     "mean_near_wall_fraction": mean,
#                     "std_near_wall_fraction": std,
#                 }
#             )

#     summary = {
#         "job_id": job_id,
#         "run_path": str(run_path),
#         "env_switch_every": args.env_switch_every,
#         "world_count": {seed: summary["world_count"] for seed, summary in seed_summaries.items()},
#         "metadata": metadata,
#         "metrics": overall_metrics,
#         "seeds": seed_summaries,
#         "aggregate_near_wall_fraction": aggregate_fractions,
#     }

#     output_dir = args.output or job_dir
#     output_dir.mkdir(parents=True, exist_ok=True)

#     metrics_json = output_dir / "near_wall_occupancy_metrics.json"
#     metrics_json.write_text(json.dumps(summary, indent=2, sort_keys=True))

#     # Provide a tabular summary for quick inspection.
#     metrics_csv = output_dir / "near_wall_occupancy_metrics.csv"
#     with metrics_csv.open("w", encoding="utf-8") as handle:
#         header = [
#             "seed",
#             "world_index",
#             "agent",
#             "path_count",
#             "total_steps",
#             "near_wall_steps",
#             "near_wall_fraction",
#         ]
#         handle.write(",".join(header) + "\n")
#         for block in overall_metrics:
#             world_index = block["world_index"]
#             seed_name = block["seed"]
#             for agent, stats in block["agents"].items():
#                 row = [
#                     seed_name,
#                     str(world_index),
#                     agent,
#                     str(stats["path_count"]),
#                     f"{stats['total_steps']:.6g}",
#                     f"{stats['near_wall_steps']:.6g}",
#                     f"{stats['near_wall_fraction']:.6g}",
#                 ]
#                 handle.write(",".join(row) + "\n")

#     aggregate_csv = output_dir / "near_wall_fraction_summary.csv"
#     with aggregate_csv.open("w", encoding="utf-8") as handle:
#         header = [
#             "world_index",
#             "agent",
#             "seed_count",
#             "mean_near_wall_fraction",
#             "std_near_wall_fraction",
#         ]
#         handle.write(",".join(header) + "\n")
#         for record in sorted(
#             aggregate_records,
#             key=lambda item: (int(item["world_index"]), item["agent"]),
#         ):
#             row = [
#                 str(record["world_index"]),
#                 record["agent"],
#                 str(record["seed_count"]),
#                 f"{record['mean_near_wall_fraction']:.6g}",
#                 f"{record['std_near_wall_fraction']:.6g}",
#             ]
#             handle.write(",".join(row) + "\n")

#     print(f"Saved near-wall occupancy metrics to {metrics_json}")


# if __name__ == "__main__":
#     main()
