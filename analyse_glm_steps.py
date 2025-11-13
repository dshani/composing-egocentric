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

from structure_functions_ import get_parameter_values, sort_paths

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