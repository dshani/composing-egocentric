#!/usr/bin/env python3
"""Analyse world dynamics by fitting a GLM to average episode length."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance

from helper_functions_ import load_structure
from structure_functions_ import get_parameter_values, sort_paths


# def _resolve_run_path(job_id: int, results_dir: Path) -> Tuple[Path, Path]:
#     """Locate the run directory for a job id and return the associated paths."""
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
    """Yield coordinate pairs from arbitrarily nested path representations."""
    for step in path:
        if step is None:
            continue
        if isinstance(step, (list, tuple)):
            if not step:
                continue
            if isinstance(step[0], (list, tuple, np.ndarray)):
                yield from _iter_coordinates(step)
                continue
            values = step
        elif isinstance(step, np.ndarray):
            if step.size == 0:
                continue
            if step.ndim > 1:
                yield from _iter_coordinates(step)
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


def _aggregate_paths_by_world(
    paths_y: Sequence[Sequence],
    paths_x: Sequence[Sequence],
    env_switch_every: int,
) -> Dict[int, List[Sequence]]:
    """Group paths by inferred world index from the trajectory metadata."""
    paths_y_sorted, paths_x_sorted = sort_paths(paths_y, paths_x)

    world_paths: Dict[int, List[Sequence]] = {}
    for episode_idx, trajectories in zip(paths_x_sorted, paths_y_sorted):
        if trajectories is None:
            continue
        try:
            episode = int(episode_idx)
        except (TypeError, ValueError):
            continue
        if env_switch_every <= 0:
            world_index = 0
        else:
            world_index = episode // env_switch_every
        world_paths.setdefault(world_index, []).extend(list(trajectories))
    return world_paths


def _extract_worlds(seed_block: object) -> Optional[np.ndarray]:
    """Pull the array of world templates from a seed entry."""
    if not isinstance(seed_block, dict):
        return None

    if "worlds" in seed_block:
        return np.array(seed_block["worlds"])

    for variant_block in seed_block.values():
        if isinstance(variant_block, dict) and "worlds" in variant_block:
            return np.array(variant_block["worlds"])
    return None


def _normalise_worlds(worlds: Optional[np.ndarray]) -> List[np.ndarray]:
    """Ensure worlds are 2D arrays and strip extraneous dimensions."""
    if worlds is None:
        return []
    flat_worlds: List[np.ndarray] = []
    for idx in range(worlds.shape[0]):
        world = np.array(worlds[idx])
        if world.ndim != 2:
            world = np.squeeze(world)
        if world.ndim != 2:
            continue
        flat_worlds.append(world)
    return flat_worlds


def _hole_feature_order(name: str) -> int:
    """Return a sort key for hole feature columns based on their lag."""
    if name == "holes_world_t":
        return 0
    prefix = "holes_world_t_minus_"
    if name.startswith(prefix):
        try:
            return int(name[len(prefix) :])
        except ValueError:
            return 10**6
    return 10**6


def _count_barriers(world: np.ndarray) -> int:
    """Count interior tiles with positive integer identifiers as barriers."""
    world = np.asarray(world)
    if world.ndim != 2:
        world = np.squeeze(world)
    if world.ndim != 2:
        raise ValueError("Expected a 2D array for world layout")

    positive_integer = (world > 0) & np.isclose(world, np.round(world))
    unity_like = np.isclose(world, 1)

    rows, cols = world.shape
    row_indices = np.arange(rows)[:, None]
    col_indices = np.arange(cols)[None, :]
    on_boundary = (
        (row_indices == 0)
        | (row_indices == rows - 1)
        | (col_indices == 0)
        | (col_indices == cols - 1)
    )

    mask = (~on_boundary) & (positive_integer | unity_like)
    return int(mask.sum())


def _average_steps(paths: Iterable[Sequence]) -> Optional[float]:
    """Compute the mean length of a collection of trajectories."""
    lengths: List[int] = []
    for path in paths:
        steps = sum(1 for _ in _iter_coordinates(path))
        if steps > 0:
            lengths.append(steps)
    if not lengths:
        return None
    return float(np.mean(lengths))


def _build_dataset(
    structure: Dict,
    prefix: str,
    env_switch_every: int,
) -> List[Dict[str, object]]:
    """Create a tabular dataset of world metrics across seeds."""
    rows: List[Dict[str, object]] = []

    for seed_key, seed_block in structure.items():
        worlds = _normalise_worlds(_extract_worlds(seed_block))
        if not worlds:
            continue

        single_structure = {seed_key: seed_block}
        try:
            paths_y, paths_x = get_parameter_values(
                "paths", structure=single_structure, prefix=prefix
            )
        except Exception:
            continue
        if not paths_y or not paths_x:
            continue

        world_paths = _aggregate_paths_by_world(paths_y, paths_x, env_switch_every)
        if not world_paths:
            continue

        barrier_counts = [
            _count_barriers(np.array(world_template)) for world_template in worlds
        ]

        if not barrier_counts:
            continue

        num_templates = len(barrier_counts)


        for world_index, trajectories in sorted(world_paths.items()):
            mean_steps = _average_steps(trajectories)
            if mean_steps is None:
                continue

            row: Dict[str, object] = {
                "seed": str(seed_key),
                "world_index": int(world_index),
                "average_steps": mean_steps,
            }

            template_index = int(world_index % num_templates)
            row["barrier_count"] = barrier_counts[template_index]

            rows.append(row)

    return rows


# def _dump_dataset(dataset: List[Dict[str, object]], destination: Path) -> None:
#     """Persist the assembled dataset as either JSON or CSV."""
#     destination = destination.expanduser().resolve()
#     destination.parent.mkdir(parents=True, exist_ok=True)

#     hole_keys = sorted(
#         {
#             key
#             for row in dataset
#             for key in row.keys()
#             if key.startswith("holes_world_t")
#         },
#         key=_hole_feature_order,
#     )

#     if destination.suffix.lower() == ".json":
#         destination.write_text(json.dumps(dataset, indent=2))
#     elif destination.suffix.lower() == ".csv":
#         fieldnames = ["seed", "world_index", "barrier_count", *hole_keys, "average_steps"]
#         with destination.open("w", newline="") as fh:
#             writer = csv.DictWriter(fh, fieldnames=fieldnames)
#             writer.writeheader()
#             writer.writerows(dataset)
#     else:
#         raise ValueError("Unsupported output format; use .json or .csv")


# def parse_args() -> argparse.Namespace:
#     """Parse command line arguments for the analysis script."""
#     parser = argparse.ArgumentParser(
#         description=(
#             "Fit a Poisson GLM predicting average episode steps from world index "
#             "and barrier count."
#         )
#     )
#     parser.add_argument(
#         "--job-id",
#         type=int,
#         required=False,
#         help="Training job identifier used to resolve run_path.txt",
#     )
#     parser.add_argument(
#         "--run-path",
#         type=Path,
#         default=None,
#         help="Override the run directory instead of reading run_path.txt",
#     )
#     parser.add_argument(
#         "--results-dir",
#         type=Path,
#         default=Path("Results/job_ids"),
#         help="Directory that contains job metadata (default: Results/job_ids)",
#     )
#     parser.add_argument(
#         "--prefix",
#         type=str,
#         default="unlesioned",
#         help="Agent prefix (e.g. unlesioned or lesionLEC)",
#     )
#     parser.add_argument(
#         "--env-switch-every",
#         type=int,
#         default=1000,
#         help="Number of episodes between world switches",
#     )
#     parser.add_argument(
#         "--alpha",
#         type=float,
#         default=0.0,
#         help="L2 regularisation strength for the Poisson GLM",
#     )
#     parser.add_argument(
#         "--max-iter",
#         type=int,
#         default=1000,
#         help="Maximum solver iterations for the Poisson GLM",
#     )
#     parser.add_argument(
#         "--dump-data",
#         type=Path,
#         default=None,
#         help="Optional path to save the aggregated dataset (.json or .csv)",
#     )
#     parser.add_argument(
#         "--max-workers",
#         type=int,
#         default=8,
#         help="Thread pool size for loading trajectories",
#     )
#     return parser.parse_args()


# def main() -> None:
#     args = parse_args()

#     if args.run_path is None and args.job_id is None:
#         raise SystemExit("Specify either --run-path or --job-id.")

#     results_dir = args.results_dir.resolve()
#     results_dir.mkdir(parents=True, exist_ok=True)

#     job_dir: Optional[Path] = None
#     if args.run_path is not None:
#         run_path = args.run_path.expanduser().resolve()
#         if args.job_id is not None:
#             job_dir = results_dir / str(args.job_id)
#     else:
#         assert args.job_id is not None
#         run_path, job_dir = _resolve_run_path(args.job_id, results_dir)

#     if job_dir is None and args.job_id is not None:
#         job_dir = results_dir / str(args.job_id)
#     if job_dir is None:
#         raise SystemExit("Unable to determine output directory for metrics.")
#     job_dir.mkdir(parents=True, exist_ok=True)

#     if not run_path.exists():
#         raise FileNotFoundError(f"Resolved run directory does not exist: {run_path}")

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

#     dataset = _build_dataset(structure, args.prefix, args.env_switch_every)
#     if not dataset:
#         raise RuntimeError("Unable to build dataset of average steps")

#     seeds = sorted({row["seed"] for row in dataset})
#     feature_names = ["world_index", "barrier_count"]
#     X = np.array(
#         [
#             [float(row["world_index"]), float(row["barrier_count"])]

#             for row in dataset
#         ],
#         dtype=float,
#     )
#     y = np.array([row["average_steps"] for row in dataset], dtype=float)

#     model = PoissonRegressor(
#         alpha=args.alpha,
#         fit_intercept=True,
#         max_iter=args.max_iter,
#     )
#     model.fit(X, y)
#     predictions = model.predict(X)
#     deviance = mean_poisson_deviance(y, predictions)

#     print(f"Analysed {len(dataset)} world segments across {len(seeds)} seeds.")
#     print(f"Prefixes analysed: {args.prefix}")
#     print("Coefficients (log link):")
#     for name, coef in zip(feature_names, model.coef_):
#         print(f"  {name:>13}: {coef: .6f}")
#     print(f"Intercept        : {model.intercept_: .6f}")
#     print(f"Mean Poisson deviance on training data: {deviance:.6f}")

#     if args.dump_data is not None:
#         _dump_dataset(dataset, args.dump_data)
#         print(f"Saved aggregated dataset to {args.dump_data}")


# if __name__ == "__main__":
#     main()
