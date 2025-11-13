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
