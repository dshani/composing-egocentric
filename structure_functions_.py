from matplotlib.patches import FancyArrow
from matplotlib.path import Path
from copy import deepcopy
from collections.abc import Sequence
from typing import Dict, List, Set, Tuple
import numpy as np
import matplotlib.pyplot as plt
import argparse
import ast
import json
import os
import pathlib
import pickle
import sys

from fractions import Fraction

import matplotlib.cm as mcm
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter1d, generate_binary_structure, label
from scipy.spatial import ConvexHull, QhullError




def get_hole_locations(world):
    """Return grid coordinates that lie inside barrier holes.

    The detector now combines the legacy seed-based approach with a convex hull
    analysis of connected wall components.  The legacy pass guarantees
    backwards compatibility with the historical "three neighbouring walls"
    heuristic, while the convex hull pass recovers the full extent of wide
    barrier corridors enclosed by the same walls.
    """

    world_array = np.array(world, copy=True)
    if world_array.ndim != 2:
        world_array = np.squeeze(world_array)
    if world_array.ndim != 2:
        raise ValueError("Expected a 2D array describing the world layout")

    world_array = np.maximum(np.minimum(world_array, 1), 0)
    world_array[0, :] = 0
    world_array[-1, :] = 0
    world_array[:, 0] = 0
    world_array[:, -1] = 0

    def _legacy_holes() -> Set[Tuple[int, int]]:
        hole_set: Set[Tuple[int, int]] = set()
        wall_cache: Dict[Tuple[int, int], Set[str]] = {}
        seeds: List[Tuple[int, int]] = []

        directions = {
            "N": (-1, 0),
            "S": (1, 0),
            "W": (0, -1),
            "E": (0, 1),
        }
        opposite_pairs = [("N", "S"), ("E", "W")]

        def cache_walls(i: int, j: int) -> Set[str]:
            key = (i, j)
            if key not in wall_cache:
                walls = {
                    name
                    for name, (di, dj) in directions.items()
                    if world_array[i + di, j + dj] == 1
                }
                wall_cache[key] = walls
            return wall_cache[key]

        for i in range(1, world_array.shape[0] - 1):
            for j in range(1, world_array.shape[1] - 1):
                if world_array[i, j] != 0:
                    continue
                walls = cache_walls(i, j)
                if len(walls) >= 3:
                    seeds.append((i, j))
                    hole_set.add((i, j))

        for seed in seeds:
            seed_walls = cache_walls(*seed)
            for pair in opposite_pairs:
                if not all(direction in seed_walls for direction in pair):
                    continue
                stack = [seed]
                visited: Set[Tuple[int, int]] = set()
                while stack:
                    cell = stack.pop()
                    if cell in visited:
                        continue
                    visited.add(cell)
                    if world_array[cell] != 0:
                        continue
                    cell_walls = cache_walls(*cell)
                    if not all(direction in cell_walls for direction in pair):
                        continue
                    hole_set.add(cell)
                    for di, dj in directions.values():
                        neighbour = (cell[0] + di, cell[1] + dj)
                        if world_array[neighbour] == 0 and neighbour not in visited:
                            stack.append(neighbour)

        return hole_set

    def _component_polygon(component_cells: np.ndarray) -> np.ndarray:
        points = component_cells[:, ::-1].astype(float) + 0.5
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                return points[hull.vertices]
            except QhullError:
                pass

        rows = component_cells[:, 0]
        cols = component_cells[:, 1]
        min_row = rows.min()
        max_row = rows.max() + 1
        min_col = cols.min()
        max_col = cols.max() + 1
        return np.array(
            [
                (min_col, min_row),
                (max_col, min_row),
                (max_col, max_row),
                (min_col, max_row),
            ],
            dtype=float,
        )

    def _hull_holes() -> Set[Tuple[int, int]]:
        hull_hole_set: Set[Tuple[int, int]] = set()
        structure = generate_binary_structure(2, 1)
        labels, component_count = label(world_array == 1, structure=structure)

        for component_index in range(1, component_count + 1):
            component_cells = np.argwhere(labels == component_index)
            if component_cells.size == 0:
                continue

            polygon = _component_polygon(component_cells)
            path = Path(polygon, closed=True)

            min_col = max(int(np.floor(polygon[:, 0].min())), 0)
            max_col = min(int(np.ceil(polygon[:, 0].max())), world_array.shape[1])
            min_row = max(int(np.floor(polygon[:, 1].min())), 0)
            max_row = min(int(np.ceil(polygon[:, 1].max())), world_array.shape[0])

            if min_row >= max_row or min_col >= max_col:
                continue

            rows = np.arange(min_row, max_row)
            cols = np.arange(min_col, max_col)
            grid_rows, grid_cols = np.meshgrid(rows, cols, indexing="ij")
            centres = np.column_stack(
                (grid_cols.ravel() + 0.5, grid_rows.ravel() + 0.5)
            )
            inside = path.contains_points(centres, radius=-1e-9)

            for (row, col), is_inside in zip(
                zip(grid_rows.ravel(), grid_cols.ravel()), inside
            ):
                if not is_inside:
                    continue
                if world_array[row, col] != 0:
                    continue
                hull_hole_set.add((int(row), int(col)))

        return hull_hole_set

    legacy_holes = _legacy_holes()
    hull_holes = _hull_holes()

    combined = sorted(legacy_holes | hull_holes)
    return combined


def get_vector_field(verts, world):
    # get vector field
    X, Y = verts[:, 0], verts[:, 1]
    dY = Y[1:] - Y[:-1]
    dX = X[1:] - X[:-1]
    im = plt.quiver(X[:-1], Y[:-1], dX, dY, scale=1,
                    scale_units='xy', angles='xy')
    im = plt.imshow(world, cmap='Greys')
    return im


def add_SR_grid(fig, model, ego_state, directions, **kwargs):
    grids = fig.add_gridspec(nrows=2, ncols=1, **kwargs)
    add_allocentric_SR_grid(grids[0, 0], fig, model, ego_state)
    add_egocentric_SR_grid(grids[1, 0], fig, model, ego_state)


# grids = fig.add_gridspec(nrows=2, ncols=1, **kwargs)
# add_allocentric_SR_grid(fig, grids[0, 0], model, ego_state, directions, height_ratios=[0.01, 1])
# add_egocentric_SR_grid(fig, grids[0, 1], model, ego_state)

def add_allocentric_SR_sas_grid(grid, fig, model, state, ego=False, colorbar=True, title=None):
    directions = ["\u2191", "\u2192", "\u2193", "\u2190"]

    inner_grid = grid.subgridspec(4, 2, width_ratios=(0.0000001, 1))
    for action in range(4):

        ax0 = fig.add_subplot(inner_grid[action, 0])
        ax0.axis("off")
        ax0.text(0.5, 0.5, directions[action], fontsize=12,
                 ha='center', va='center', color='b')

        ax = fig.add_subplot(inner_grid[action, 1])
        array = np.zeros((model.env.size, model.env.size))
        for x in range(model.env.size):
            for y in range(model.env.size):
                if model.env.world[x, y] <= 0:
                    allo_state = model.env.get_1d_pos([x, y])
                    array[x, y] += model.allo_SR.SR_sas[state][action][allo_state]

        arrmin = np.min(array)
        arrmax = np.max(array)

        im = ax.imshow(
            array[:, :], vmin=arrmin, vmax=arrmax, cmap='plasma')
        if title:
            ax.title.set_text(title)
        ax.set_xticks([])
        ax.set_yticks([])
        if colorbar:
            plt.colorbar(im, ax=ax)


def add_allocentric_SR_grid(grid, fig, model, state, ego=True, colorbar=True, title=None):
    directions = ["\u2191", "\u2192", "\u2193", "\u2190"]
    if ego:
        inner_grid = grid.subgridspec(1, 5, width_ratios=(
            1, 1, 1, 1, 0.03), wspace=0.01, hspace=0.01)

        array = np.zeros((model.env.size, model.env.size, 4))
        for x in range(model.env.size):
            for y in range(model.env.size):
                if model.env.world[x, y] <= 0:
                    for d in range(4):
                        egocentric_state = model.env.allo_to_ego[-1][(x, y, d)]
                        array[x, y, d] += model.ego_SR.SR_ss[state][egocentric_state]

        arrmin = np.min(array)
        arrmax = np.max(array)
        for d in range(4):
            ax = fig.add_subplot(inner_grid[0, d])
            im = ax.imshow(
                array[:, :, d],
                vmin=arrmin, vmax=arrmax, cmap='plasma')
            ax.title.set_text(f"{directions[d]}")
            ax.set_xticks([])
            ax.set_yticks([])

        if colorbar:
            cbar_ax = fig.add_subplot(inner_grid[0, 4])
            plt.colorbar(im, cax=cbar_ax)
    else:
        ax = fig.add_subplot(grid)
        array = np.zeros((model.env.size, model.env.size))
        for x in range(model.env.size):
            for y in range(model.env.size):
                if model.env.world[x, y] <= 0:
                    allo_state = model.env.get_1d_pos([x, y])
                    array[x, y] += model.allo_SR.SR_ss[state][allo_state]

        arrmin = np.min(array)
        arrmax = np.max(array)

        im = ax.imshow(
            array[:, :], vmin=arrmin, vmax=arrmax, cmap='plasma')
        if title:
            ax.title.set_text(title)
        ax.set_xticks([])
        ax.set_yticks([])
        if colorbar:
            plt.colorbar(im, ax=ax)


def add_egocentric_SR_sas_grid(grid, fig, model, ego_state, coords=[-1, 0], **kwargs):
    from plotting_functions import show_ego_state
    directions = ["\u2191", "\u2192", "\u2193", "\u2190"]
    actions = ["\u2191", "\u21b1", "\u21b7", "\u21b0"]
    inner_grid = grid.subgridspec(
        4, 3, wspace=0.5, hspace=0.5, width_ratios=(0.2, 0.2, 1))

    hists = []
    bin_edgess = []

    for action in range(4):
        hist, bin_edges = np.histogram(
            model.ego_SR.SR_sas[ego_state][action], bins=5)
        hists.append(hist)
        bin_edgess.append(bin_edges)

    shape0 = np.max([hist.shape[0] for hist in hists])
    shape1 = np.max([np.max(hist[(hist.shape[0] // 2):]) for hist in hists])

    ax0 = fig.add_subplot(inner_grid[:, 0])
    show_ego_state(ego_state, model, local=True, ax=ax0, cmap='Greens')
    ax0.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
        labelcolor='g')

    for action in range(4):
        ax0 = fig.add_subplot(inner_grid[action, 1])
        ax0.axis("off")
        ax0.text(0.5, 0.5, actions[action], fontsize=20,
                 ha='center', va='center', color='b')

        hist, bin_edges = np.histogram(
            model.ego_SR.SR_sas[ego_state][action], bins=5)
        inner_inner_grid = inner_grid[action, 2].subgridspec(
            shape0 - shape0 // 2, shape1, **kwargs)

    # fig, axs = plt.subplots(
    #     hist.shape[0] - hist.shape[0] // 2,
    #     np.max(hist[hist.shape[0] // 2:]), figsize=(5, 5))
    # if len(axs.shape) == 2:
    #     for k in range(axs.shape[0]):
    #         for j in range(axs.shape[1]):
    #             if j != 0:
    #                 axs[k, j].axis("off")

        k = 0
        for i in range(hist.shape[0] - 1, hist.shape[0] // 2 - 1, -1):

            if hist[i]:

                ego_states = np.where(np.logical_and(
                    model.ego_SR.SR_sas[ego_state][action] > bin_edges[i], model.ego_SR.SR_sas[ego_state][action] <= bin_edges[i + 1]))[0]
                if len(ego_states) > 1:

                    for j in range(len(ego_states)):
                        state = ego_states[j]
                        ax = fig.add_subplot(inner_inner_grid[k, j])
                        #                 ax.axis("off")
                        show_ego_state(state, model, local=True, ax=ax)
                        ax.tick_params(
                            axis='both',  # changes apply to the x-axis
                            which='both',  # both major and minor ticks are affected
                            bottom=False,  # ticks along the bottom edge are off
                            top=False,  # ticks along the top edge are off
                            left=False,
                            right=False,
                            labelbottom=False,
                            labelleft=False)
                        # axs[k, j].imshow(
                        #     np.reshape(
                        #         model.env.ego_bins[state],
                        #         (model.env.pars.horizon + 1,
                        #          2 * model.env.pars.horizon + 1)), vmax=vmax, vmin=vmin)
                        if j == 0:
                            ax.set_ylabel(
                                str(round(bin_edges[i], 1)) + " - " + str(
                                    round(
                                        bin_edges[
                                            i +
                                            1],
                                        1)),
                                fontsize='xx-small', rotation=0)
                            ax.yaxis.set_label_coords(coords[0], coords[1])
                else:
                    state = ego_states[0]
                    ax = fig.add_subplot(inner_inner_grid[k, 0])
                    #             ax.axis("off")
                    show_ego_state(state, model, local=True, ax=ax)
                    ax.set_ylabel(
                        str(round(bin_edges[i], 1)) + " - " + str(
                            round(
                                bin_edges[
                                    i +
                                    1],
                                1)),
                        fontsize='xx-small', rotation=0)
                    ax.yaxis.set_label_coords(coords[0], coords[1])

                ax.tick_params(
                    axis='both',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    left=False,
                    right=False,
                    labelbottom=False,
                    labelleft=False)

            k += 1



def show_histogram_ego(hist, bin_edges, ego_state, model, fig, grid, coords=[-1, 0], ego=True, direction=None, proportion = Fraction(1, 2), **kwargs):
    
    
    from plotting_functions import show_ego_state

    inner_grid = grid.subgridspec(
        hist.shape[0] - hist.shape[0]*proportion.numerator // proportion.denominator + 1, np.max(hist[(hist.shape[0]*proportion.numerator // proportion.denominator):]), **kwargs)

    ax = fig.add_subplot(inner_grid[0, 0])
    if ego:
        show_ego_state(ego_state, model, local=True, ax=ax, cmap='Greens')
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
            labelcolor='g')
    else:
        x, y = model.env.get_2d_pos(ego_state)
        show_ego_state(model.env.get_ego_obs(
            [x, y], direction), model, local=True, ax=ax, cmap='Greens')
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
            labelcolor='g')
    k = 0
    for i in range(hist.shape[0] - 1, hist.shape[0]*proportion.numerator // proportion.denominator - 1, -1):

        if hist[i]:
            if ego:
                ego_states = np.where(np.logical_and(
                                      model.ego_SR.SR_ss[ego_state] > bin_edges[i],
                                      model.ego_SR.SR_ss[ego_state] <= bin_edges[i + 1]))[0]
            else:
                allo_states = np.where(np.logical_and(
                    model.allo_SR.SR_ss[ego_state] > bin_edges[i], model.allo_SR.SR_ss[ego_state] <= bin_edges[i + 1]))[0]
                d = direction
                ego_states = []
                for state in allo_states:
                    x, y = model.env.get_2d_pos(state)
                    ego_states.append(model.env.get_ego_obs([x, y], d))


            if len(ego_states) > 1:

                for j in range(len(ego_states)):
                    state = ego_states[j]
                    ax = fig.add_subplot(inner_grid[k + 1, j])
                    #                 ax.axis("off")
                    show_ego_state(state, model, local=True, ax=ax)
                    ax.tick_params(
                        axis='both',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        bottom=False,  # ticks along the bottom edge are off
                        top=False,  # ticks along the top edge are off
                        left=False,
                        right=False,
                        labelbottom=False,
                        labelleft=False)
                    # axs[k, j].imshow(
                    #     np.reshape(
                    #         model.env.ego_bins[state],
                    #         (model.env.pars.horizon + 1,
                    #          2 * model.env.pars.horizon + 1)), vmax=vmax, vmin=vmin)
                    if j == 0:
                        ax.set_ylabel(
                            str(round(bin_edges[i], 1)) + " - " + str(
                                round(
                                    bin_edges[
                                        i +
                                        1],
                                    1)),
                            fontsize='xx-small', rotation=0)
                        ax.yaxis.set_label_coords(coords[0], coords[1])
            else:
                state = ego_states[0]
                ax = fig.add_subplot(inner_grid[k + 1, 0])
                #             ax.axis("off")
                show_ego_state(state, model, local=True, ax=ax)
                ax.set_ylabel(
                    str(round(bin_edges[i], 1)) + " - " + str(
                        round(
                            bin_edges[
                                i +
                                1],
                            1)),
                    fontsize='xx-small', rotation=0)
                ax.yaxis.set_label_coords(coords[0], coords[1])

            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False)

        k += 1


def add_egocentric_SR_grid(grid, fig, model, state, coords=[-1, 0], ego=True, proportion=Fraction(1,2), **kwargs):

    if ego:
        ego_state = state
        directions = ["\u2191", "\u2192", "\u2193", "\u2190"]
        # panel H: egocentric SR in ego coords
        hist, bin_edges = np.histogram(model.ego_SR.SR_ss[ego_state], bins=5)
        show_histogram_ego(hist, bin_edges, ego_state, model,
                           fig, grid, coords=coords, ego=True, proportion=proportion, **kwargs)

    else:
        allo_state = state
        outer_grid = grid.subgridspec(4, 1, **kwargs)

        directions = ["\u2191", "\u2192", "\u2193", "\u2190"]
        # panel H: egocentric SR in ego coords
        hist, bin_edges = np.histogram(model.allo_SR.SR_ss[allo_state], bins=5)

        from plotting_functions import show_ego_state

        for d in range(4):
            show_histogram_ego(hist, bin_edges, state, model, fig,
                               outer_grid[d, 0], coords=coords, ego=False, direction=d, **kwargs)

def add_sr_grid_from_sr(fig, ego_sr, allo_sr, model, ego_state, allo_state):
    model1 = deepcopy(model)
    model1.ego_SR.SR_ss = ego_sr
    model1.allo_SR.SR_ss = allo_sr

    grids = fig.add_gridspec(nrows=1, ncols=2, wspace=0.1)
    add_allocentric_SR_grid(grids[0, 0], fig, model1, allo_state, ego=False)
    add_egocentric_SR_grid(grids[0, 1], fig, model1, ego_state)

def plot_track(verts, ax, world=None, label=None, heatmap=True, lims=None, fontsize=12, **kw_args):
    if world is not None:
        ax.imshow(world, cmap='Greys')
    '''Plot followed track: verts is 2D array: x, y'''
    # rotate x,y coordinates
    transitions = np.zeros(
        (world.shape[0], world.shape[1], world.shape[0], world.shape[1]))
    for xy0, xy1 in zip(verts[:-1], verts[1:]):
        if np.linalg.norm(xy1 - xy0) == 1:
            transitions[xy0[1], xy0[0], xy1[1], xy1[0]] += 1

    # normalise transitions over last two dimensions, if non-zero
    transitions = transitions / \
        np.maximum(np.sum(transitions, axis=(2, 3))[
                   :, :, np.newaxis, np.newaxis], 1)

    for i in range(transitions.shape[0]):
        for j in range(transitions.shape[1]):
            shift_x, shift_y = 0, 0
            for k in range(max(i - 1, 0), min(i + 2, transitions.shape[0])):
                for l in range(max(j - 1, 0), min(j + 2, transitions.shape[1])):
                    if transitions[i, j, k, l] > 0:
                        patch = FancyArrow(j + shift_x, i + shift_y, l - j, k - i, head_width=0.5*transitions[i, j, k, l],
                                           head_length=0.1 * transitions[i, j, k, l], **kw_args)
                        ax.add_patch(patch)

    if heatmap:
        heatmap_data = get_heatmap_data(verts, world)
        if lims:
            vmin, vmax = lims
        else:
            vmin, vmax = np.min(heatmap_data), np.max(heatmap_data)
        heatmap_im = ax.imshow(heatmap_data, cmap='hot',
                               alpha=0.5, vmin=vmin, vmax=vmax)

    if label:
        ax.set_title(label, fontsize=fontsize)

    ax.set_xticks([])
    ax.set_yticks([])

    return ax


def get_value_functions(model, env=None, split=True, action=None):
    if env is None:
        env = model.env
    size = env.size

    if action == "all":
        state_values = np.zeros((size, size, 4, 4))
        allo_state_values = np.zeros((size, size, 4, 4))
        ego_state_values = np.zeros((size, size, 4, 4))
    else:
        state_values = np.zeros((size, size, 4))
        allo_state_values = np.zeros((size, size, 4))
        ego_state_values = np.zeros((size, size, 4))

    egomin, egomax, amin, amax = None, None, None, None
    w = model.weight
    if split:
        w_allo = w.copy()
        w_ego = w.copy()
        w_allo[model.allo_dim+1:] = np.zeros_like(w_allo[model.allo_dim+1:])
        w_ego[1:model.allo_dim+1] = np.zeros_like(w_ego[1:model.allo_dim+1])
        

    for d in range(4):
        for i_ in range(size):
            for j_ in range(size):
                if env.world[i_, j_] <= 0:
                    state = env.get_1d_pos([i_, j_])
                    ego = env.get_ego_obs([i_, j_], d)

                    state_values[i_, j_, d] = model.q_w(
                        w, state, ego, d, action=action)
                    if split:
                        allo_state_values[i_, j_, d] = model.q_w(
                            w_allo, state, ego, d, action=action)
                        ego_state_values[i_, j_, d] = model.q_w(
                            w_ego, state, ego, d, action=action)

    state_values[state_values == 0] = np.mean(state_values[state_values != 0])
    # TODO: change from using mean to using Nan and then filling in the mean later

    vmin = np.min(state_values)
    vmax = np.max(state_values)

    if split:
        allo_state_values[allo_state_values == 0] = np.mean(
            allo_state_values[allo_state_values != 0])
        ego_state_values[ego_state_values == 0] = np.mean(
            ego_state_values[ego_state_values != 0])

        egomin = np.min(ego_state_values)
        egomax = np.max(ego_state_values)

        amin = np.min(allo_state_values)
        amax = np.max(allo_state_values)

    if split:
        return state_values, ego_state_values, allo_state_values, vmin, vmax, egomin, egomax, amin, amax

    else:
        return state_values, vmin, vmax

def get_locations(paths_y, paths_start):
    """
    This function takes two lists: paths_y and paths_start.
    paths_y: A list of paths, where each path is a list of coordinates.
    paths_start: A list of starting locations.

    Returns:
      A flat list of all locations from paths that started from a location in paths_start.
    """
    locations = []
    # Iterate through each path in paths_y
    for group in paths_y:
        # Check if the starting location of the path is in paths_start
        for path in group:
            if len(path) != 0:
                if np.any(tuple(path[0]) in [tuple(start) for start in paths_start]):
                    # Add all locations in the path to the all_locations list
                    locations.extend(path)
    # Return the flat list of all locations
    return locations

def get_certain_SR_model(ego_srs_y=None, ego_srs_x=None, allo_srs_y=None, allo_srs_x=None, model=None, time0=0):
    if model:
        import bisect
        if ego_srs_y and ego_srs_x:
            first_index = bisect.bisect_left(ego_srs_x[0], time0)
        elif allo_srs_y and allo_srs_x:
            first_index = bisect.bisect_left(allo_srs_x[0], time0)
        else:
            raise NotImplementedError
        model1 = deepcopy(model)
        if ego_srs_y:
            first_ego_sr = ego_srs_y[0][first_index]
            model1.ego_SR.SR_ss = first_ego_sr
        if allo_srs_y:
            first_allo_sr = allo_srs_y[0][first_index]
            model1.allo_SR.SR_ss = first_allo_sr
        return model1
    else:
        raise NotImplementedError

def get_hole_times_(paths_y, paths_x, worlds, switch_every):
    hole_times = []
    world_index = 0
    world = worlds[world_index]
    hole_locations = set(get_hole_locations(world))
    
    x_ = []
    for p, x in zip(paths_y, paths_x):
        world_index_ = x // switch_every
        if world_index_ != world_index:
            world_index = world_index_
            world = worlds[world_index % len(worlds)]
            hole_locations = set(get_hole_locations(world))
        average_hole_time = 0
        if len(p[0]) == 2:
            p = [p]
        for path in p:
            hole_time = 0
            for location in path:
                if tuple(location) in hole_locations:
                    hole_time += 1
            average_hole_time += hole_time / len(path)
        average_hole_time /= len(p)
        hole_times.append(average_hole_time)
        x_.append(x)
    return hole_times, x_



def get_heatmap_data(verts, world):
    heatmap_data = np.zeros_like(world)

    for x, y in verts:
        heatmap_data[y, x] += 1
    return heatmap_data


# def sort_paths(paths_y_, paths_x_):
#     path_dict = {}
#     for seed in range(len(paths_x_)):
#         for savepoint in range(len(paths_x_[seed])):
#             # non-empty paths
#             non_empty_paths = [path for path in paths_y_[
#                 seed][savepoint] if len(path) > 0]
#             # Keep the original episode index as the x-key so x remains
#             # monotone and aligned with env switches.
#             for path in non_empty_paths:
#                 key = paths_x_[seed][savepoint]
#                 if path_dict.get(key):
#                     path_dict[key].append(path)
#                 else:
#                     path_dict[key] = [path]

#             # path_num = len(paths_y_[seed][savepoint])
#             # for backdelay in range(path_num):
#             #     if path_dict.get(paths_x_[seed][savepoint]-path_num+backdelay+1):
#             #         path_dict[paths_x_[seed][savepoint]-path_num+backdelay+1].append(paths_y_[seed][savepoint][backdelay])
#             #     else:
#             #         path_dict[paths_x_[seed][savepoint]-path_num+backdelay+1] = [paths_y_[seed][savepoint][backdelay]]

#     paths_x = list(path_dict.keys())
#     paths_y = [path_dict[x] for x in paths_x]

#     paths_y = [y for _, y in sorted(
#         zip(paths_x, paths_y), key=lambda pair: pair[0])]
#     paths_x = sorted(paths_x)
#     return paths_y, paths_x

def sort_paths(paths_y_, paths_x_, return_seed_indices=False):
    path_dict = {}
    seed_dict = {} if return_seed_indices else None
    for seed in range(len(paths_x_)):
        for savepoint in range(len(paths_x_[seed])):
            non_empty_paths = [path for path in paths_y_[seed][savepoint] if len(path) > 0]

            # Assign each trajectory to its proper episode: most recent → savepoint,
            # earlier ones → savepoint-1, savepoint-2, ...
            for back, path in enumerate(non_empty_paths[::-1]):
                key = max(0, paths_x_[seed][savepoint] - back)
                if path_dict.get(key):
                    path_dict[key].append(path)
                    if return_seed_indices:
                        seed_dict[key].append(seed)
                else:
                    path_dict[key] = [path]
                    if return_seed_indices:
                        seed_dict[key] = [seed]

    paths_x = sorted(path_dict.keys())
    paths_y = [path_dict[x] for x in paths_x]
    if return_seed_indices:
        seed_indices = [seed_dict[x] for x in paths_x]
        return paths_y, paths_x, seed_indices
    return paths_y, paths_x


def remove_empty_dicts(data):
    """Removes empty dictionaries from a nested dictionary structure.

    Args:
        data: A dictionary potentially containing nested dictionaries.

    Returns:
        A new dictionary with empty dictionaries removed.
    """

    if not isinstance(data, dict):
        return data  # Non-dictionary data is simply returned

    new_data = {}
    for key, value in data.items():
        if value:  # Check if the value is not empty
            # Recursively remove empty dicts from nested structures
            new_value = remove_empty_dicts(value)
            if new_value:
                # Add only non-empty values to the new dictionary
                new_data[key] = new_value
    return new_data


def get_parameter_values_(param, structure=None, prefix=None, timesteps=None):
    
    structure = remove_empty_dicts(structure)
    param_list_y = [[] for _ in range(len(structure))]
    param_list_x = [[] for _ in range(len(structure))]
    for index, i in enumerate(structure.keys()):
        if structure[i].get(prefix) is not None:
            if structure[i][prefix].get(param) is not None:
                for j in range(len(structure[i][prefix][param])):
                    for k in range(len(structure[i][prefix][param][j])):
                        value = structure[i][prefix][param][j][k]
                        # Some compact accuracy files store raw scalars instead of
                        # ``(timestep, value)`` pairs. Handle both formats here.
                        if (
                            isinstance(value, Sequence)
                            and not isinstance(value, (str, bytes))
                            and len(value) >= 2
                        ):
                            x_val, y_val = value[0], value[1]
                        else:
                            x_val, y_val = k, value

                        if timesteps:
                            if x_val in timesteps:
                                param_list_y[index].append(y_val)
                                param_list_x[index].append(x_val)
                        else:
                            param_list_y[index].append(y_val)
                            param_list_x[index].append(x_val)
            else:
                param_list_y[index] = None
                param_list_x[index] = None
        else:
            param_list_y[index] = None
            param_list_x[index] = None
            
    return param_list_y, param_list_x
                    
            
def get_parameter_values(param, structure=None, prefix=None, timesteps=None, ignore=False):
    """Take param and structure and extract list of saved params for each seed and the prefix key"""
    if prefix:
        
        if structure:
            param_list_y__, param_list_x__ = get_parameter_values_(param, structure, prefix=prefix, timesteps=timesteps)
            param_list_y_ = [x for x in param_list_y__ if x is not None]
            param_list_x_ = [x for x in param_list_x__ if x is not None]
            
            max_len = np.max([len(x) for x in param_list_y_])
            param_list_y = [x for x in param_list_y_ if len(x) == max_len]
            param_list_x = [x for x in param_list_x_ if len(x) == max_len]
            
            ignore_indices_y = [i for i, x in enumerate(param_list_y__) if x is None or len(x) != max_len]
            
            
            
            
            
        if ignore:
            return param_list_y, param_list_x, ignore_indices_y
        
        else:  
            return param_list_y, param_list_x
    else:
        if param == 'ratio':
            y_un, x_un = get_parameter_values_('accuracies', structure, prefix='unlesioned', timesteps=timesteps)
            y_les, x_les = get_parameter_values_('accuracies', structure, prefix='lesionLEC', timesteps=timesteps)
            
            kept_y_un = []
            kept_x_un = []
            kept_y_les = []
            kept_x_les = []
            
            max_len = np.max([np.max([len(x) for x in x_un if x is not None]), np.max([len(x) for x in x_les if x is not None])])
            
            for i in range(len(x_un)):
                if x_un[i] is not None and x_les[i] is not None and len(x_un[i]) == len(x_les[i]) and len(x_un[i]) == max_len:
                    #sort y_un by x_un
                    y_un_sorted = [y for _, y in sorted(zip(x_un[i], y_un[i]), key=lambda pair: pair[0])]
                    x_un_sorted = sorted(x_un[i])
                    
                    x_les_sorted = sorted(x_les[i])
                    y_les_sorted = [y for _, y in sorted(zip(x_les[i], y_les[i]), key=lambda pair: pair[0])]

                    assert x_un_sorted == x_les_sorted
                    kept_y_un.append(y_un_sorted)
                    kept_x_un.append(x_un_sorted)
                    kept_y_les.append(y_les_sorted)
                    kept_x_les.append(x_les_sorted)
                    
                    
                    
            
            ratio_y = []
            ratio_x = []
            
            for i in range(len(kept_y_un)):
                ratio_y.append(np.array(kept_y_un[i]) / np.array(kept_y_les[i]))
                ratio_x.append(kept_x_un[i])
            return ratio_y, ratio_x
        else:
            y_un, x_un = get_parameter_values_(param, structure, prefix='unlesioned', timesteps=timesteps)
            y_les, x_les = get_parameter_values_(param, structure, prefix='lesionLEC', timesteps=timesteps)
        
        # #intersect x_un and x_les
        # x_un_ = []
        # y_un_ = []
        
        # x_les_ = []
        # y_les_ = []
        
        # for i in range(len(x_un)):
        #     if x_un[i] == x_les[i]:
        #         x_un_.append(x_un[i])
        #         y_un_.append(y_un[i])
        #         x_les_.append(x_les[i])
        #         y_les_.append(y_les[i])
    
        
                    
        
        return y_les, x_les, y_un, x_un
    
    


def clean_structure(structure, param='accuracies', prefix='lesionLEC'):
    lengths = []
    for i in structure.keys():
        # if exists
        if structure[i][prefix].get(param) is not None:
            lengths.append(len(structure[i][prefix][param]))
    max_len = np.max(lengths)

    indices = []
    for i in range(len(structure)):
        if structure[i][prefix].get(param) is not None:
            if len(structure[i][prefix].get(param)) == max_len:
                indices.append(i)

    structure_fin = {i: structure[i] for i in indices}
    return structure_fin


import numpy as np
from scipy.ndimage import gaussian_filter1d

def get_lesion_values_(structure, sigma, lesionMEC=False):
    # Retrieve unlesioned and lesioned parameter values
    unlesioned_y, unlesioned_x = get_parameter_values('accuracies', structure, prefix='unlesioned')
    lesioned_y, lesioned_x = get_parameter_values('accuracies', structure, prefix='lesionLEC')
    
    # Convert to NumPy arrays for easier manipulation
    unlesioned_y = np.array(unlesioned_y)
    unlesioned_x = np.array(unlesioned_x)
    lesioned_y = np.array(lesioned_y)
    lesioned_x = np.array(lesioned_x)
    
    # Sort the data based on x-values before computing means
    # For lesioned data
    lesioned_sort_indices = np.argsort(lesioned_x, axis=1)
    lesioned_sorted_x = np.take_along_axis(lesioned_x, lesioned_sort_indices, axis=1)
    lesioned_sorted_y = np.take_along_axis(lesioned_y, lesioned_sort_indices, axis=1)
    
    # For unlesioned data
    unlesioned_sort_indices = np.argsort(unlesioned_x, axis=1)
    unlesioned_sorted_x = np.take_along_axis(unlesioned_x, unlesioned_sort_indices, axis=1)
    unlesioned_sorted_y = np.take_along_axis(unlesioned_y, unlesioned_sort_indices, axis=1)
    
    # Calculate mean and SEM for lesioned data
    lesioned_x_mean = np.mean(lesioned_sorted_x, axis=0)
    lesioned_y_mean = np.mean(lesioned_sorted_y, axis=0)
    lesioned_y_sem = np.std(lesioned_sorted_y, axis=0, ddof=1) / np.sqrt(lesioned_sorted_y.shape[0])
    
    # Calculate mean and SEM for unlesioned data
    unlesioned_x_mean = np.mean(unlesioned_sorted_x, axis=0)
    unlesioned_y_mean = np.mean(unlesioned_sorted_y, axis=0)
    unlesioned_y_sem = np.std(unlesioned_sorted_y, axis=0, ddof=1) / np.sqrt(unlesioned_sorted_y.shape[0])
    
    # Apply Gaussian smoothing
    y_les_mean_smoothed = gaussian_filter1d(lesioned_y_mean, sigma=sigma)
    y_les_sem_smoothed = gaussian_filter1d(lesioned_y_sem, sigma=sigma)
    y_un_mean_smoothed = gaussian_filter1d(unlesioned_y_mean, sigma=sigma)
    y_un_sem_smoothed = gaussian_filter1d(unlesioned_y_sem, sigma=sigma)
    
    if not lesionMEC:
        return (
            (y_les_mean_smoothed, y_les_sem_smoothed, lesioned_x_mean),
            (y_un_mean_smoothed, y_un_sem_smoothed, unlesioned_x_mean)
        )
    else:
        # Process lesionMEC data similarly
        mec_y, mec_x = get_parameter_values('accuracies', structure, prefix='lesionMEC')
        mec_y = np.array(mec_y)
        mec_x = np.array(mec_x)
        
        # Sort the data
        mec_sort_indices = np.argsort(mec_x, axis=1)
        mec_sorted_x = np.take_along_axis(mec_x, mec_sort_indices, axis=1)
        mec_sorted_y = np.take_along_axis(mec_y, mec_sort_indices, axis=1)
        
        # Calculate mean and SEM
        mec_x_mean = np.mean(mec_sorted_x, axis=0)
        mec_y_mean = np.mean(mec_sorted_y, axis=0)
        mec_y_sem = np.std(mec_sorted_y, axis=0, ddof=1) / np.sqrt(mec_sorted_y.shape[0])
        
        # Apply Gaussian smoothing
        y_mec_mean_smoothed = gaussian_filter1d(mec_y_mean, sigma=sigma)
        y_mec_sem_smoothed = gaussian_filter1d(mec_y_sem, sigma=sigma)
        
        return (
            (y_les_mean_smoothed, y_les_sem_smoothed, lesioned_x_mean),
            (y_un_mean_smoothed, y_un_sem_smoothed, unlesioned_x_mean),
            (y_mec_mean_smoothed, y_mec_sem_smoothed, mec_x_mean)
        )


# def get_lesion_values_(structure, sigma, lesionMEC=False):
#     # Retrieve unlesioned and lesioned parameter values
#     unlesioned_y, unlesioned_x = get_parameter_values('accuracies', structure, prefix='unlesioned')
#     lesioned_y, lesioned_x = get_parameter_values('accuracies', structure, prefix='lesionLEC')
    

    
#     # Convert to NumPy arrays for easier manipulation
#     unlesioned_y = np.array(y_un)
#     unlesioned_x = np.array(x_un)
#     lesioned_y = np.array(y_les)
#     lesioned_x = np.array(x_les)

#     # Calculate mean and SEM for lesioned data
#     lesioned_x_mean = np.mean(lesioned_x, axis=0)
#     lesioned_y_mean = np.mean(lesioned_y, axis=0)
#     lesioned_y_sem = np.std(lesioned_y, axis=0, ddof=1) / np.sqrt(lesioned_y.shape[0])

#     # Calculate mean and SEM for unlesioned data
#     unlesioned_x_mean = np.mean(unlesioned_x, axis=0)
#     unlesioned_y_mean = np.mean(unlesioned_y, axis=0)
#     unlesioned_y_sem = np.std(unlesioned_y, axis=0, ddof=1) / np.sqrt(unlesioned_y.shape[0])

#     # Sort the data based on x-values
#     lesioned_sorted = sorted(zip(lesioned_x_mean, lesioned_y_mean, lesioned_y_sem))
#     unlesioned_sorted = sorted(zip(unlesioned_x_mean, unlesioned_y_mean, unlesioned_y_sem))

#     # Unzip the sorted data
#     x_les, y_les_mean, y_les_sem = map(np.array, zip(*lesioned_sorted))
#     x_un, y_un_mean, y_un_sem = map(np.array, zip(*unlesioned_sorted))

#     # Apply Gaussian smoothing
#     y_les_mean = gaussian_filter1d(y_les_mean, sigma=sigma)
#     y_les_sem = gaussian_filter1d(y_les_sem, sigma=sigma)
#     y_un_mean = gaussian_filter1d(y_un_mean, sigma=sigma)
#     y_un_sem = gaussian_filter1d(y_un_sem, sigma=sigma)
    
#     if not lesionMEC:

#         return (y_les_mean, y_les_sem, x_les), (y_un_mean, y_un_sem, x_un)
    
#     else:
#         mec_y, mec_x = get_parameter_values('accuracies', structure, prefix='lesionMEC')
#         mec_y = np.array(mec_y)
#         mec_x = np.array(mec_x)
#         mec_x_mean = np.mean(mec_x, axis=0)
#         mec_y_mean = np.mean(mec_y, axis=0)
#         mec_y_sem = np.std(mec_y, axis=0, ddof=1) / np.sqrt(mec_y.shape[0])
#         mec_sorted = sorted(zip(mec_x_mean, mec_y_mean, mec_y_sem))
#         x_mec, y_mec_mean, y_mec_sem = map(np.array, zip(*mec_sorted))
#         y_mec_mean = gaussian_filter1d(y_mec_mean, sigma=sigma)
#         y_mec_sem = gaussian_filter1d(y_mec_sem, sigma=sigma)
#         return (y_les_mean, y_les_sem, x_les), (y_un_mean, y_un_sem, x_un), (y_mec_mean, y_mec_sem, x_mec)


def get_lesion_values(structure, sigma):
    unlesioned_y, unlesioned_x = get_parameter_values(
        'accuracies', structure, prefix='unlesioned')
    lesioned_y, lesioned_x = get_parameter_values(
        'accuracies', structure, prefix='lesionLEC')
    
    

    lesioned_x = np.mean(lesioned_x, 0)
    lesioned_y = np.mean(lesioned_y, 0)

    unlesioned_x = np.mean(unlesioned_x, 0)
    unlesioned_y = np.mean(unlesioned_y, 0)

    unlesioned_y_ = [x for _, x in sorted(zip(unlesioned_x, unlesioned_y))]
    x_un = sorted(unlesioned_x)

    lesioned_y_ = [x for _, x in sorted(zip(lesioned_x, lesioned_y))]
    x_les = sorted(lesioned_x)

    y_les = gaussian_filter1d(lesioned_y_, sigma=sigma)
    y_un = gaussian_filter1d(unlesioned_y_, sigma=sigma)

    return y_les, y_un, x_les, x_un


def get_single_environment_learning(structure=None, sigma=0.1, env_switch_every=1000, path=None):
    if path:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
    else:
        path = pathlib.Path('.')

    if path.joinpath(f'single_learning_data_{sigma}_{env_switch_every}.pkl').exists() and not structure:
        with open(path.joinpath(f'single_learning_data_{sigma}_{env_switch_every}.pkl'), 'rb') as f:
            data = pickle.load(f)
            x, y, first_path, last_path, first_ep, last_ep, world = data['x'], data['y'], data['first_path'], data[
                'last_path'], data['first_ep'], data['last_ep'], data['world']
    else:
        if structure:
            structure = remove_empty_dicts(structure)
            world = structure[list(structure.keys())[0]
                              ]['unlesioned']['worlds'][0]
            y, x = get_parameter_values(
                'accuracies', structure, prefix='unlesioned')
            first_path, first_ep = get_path(
                structure, prefix='unlesioned', episode=0)
            last_path, last_ep = get_path(structure, prefix='unlesioned',
                                          episode=env_switch_every - 1)
            data = {'x': x, 'y': y, 'first_path': first_path, 'last_path': last_path, 'first_ep': first_ep,
                    'last_ep': last_ep, 'world': world}
            with open(path.joinpath(f'single_learning_data_{sigma}_{env_switch_every}.pkl'), 'wb') as f:
                pickle.dump(data, f)
        else:
            raise NotImplementedError


def compare_paths_after_switch(structure=None, env_switch_every=1000, path=None, path_start=None, heatmap_path_num=100):
    if path:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
    else:
        path = pathlib.Path('.')

    if path.joinpath(
            f'compare_paths_after_switch_data_{env_switch_every}_{heatmap_path_num}.pkl').exists() and not structure:
        with open(path.joinpath(
                f'compare_paths_after_switch_data_{env_switch_every}_{heatmap_path_num}.pkl'),
                'rb') as f:
            data = pickle.load(f)
            path_unlesioned, un_ep, path_lesioned, les_ep, heatmap_data_unlesioned, heatmap_data_lesioned, larger_verts_unles, larger_verts_les, world = \
                data['path_unlesioned'], data['un_ep'], data['path_lesioned'], data['les_ep'], data[
                    'heatmap_data_unlesioned'], data['heatmap_data_lesioned'], data['larger_verts_unles'], \
                data['larger_verts_unles'], data['world']
    else:
        if structure:
            structure = remove_empty_dicts(structure)
            world = structure[list(structure.keys())[0]
                              ]['unlesioned']['worlds'][1]
            path_unlesioned, un_ep = get_path(structure, prefix='unlesioned',
                                              episode=env_switch_every + 1, path_start=path_start)
            path_lesioned, les_ep = get_path(structure, prefix='lesionLEC',
                                             episode=env_switch_every + 1, path_start=path_start)
            heatmap_data_unlesioned = get_heatmap_data(
                path_unlesioned, world=world)
            heatmap_data_lesioned = get_heatmap_data(
                path_lesioned, world=world)
            larger_verts_unles = get_path_verts_multiple(structure, prefix='unlesioned',
                                                         ep_start=env_switch_every + 1,
                                                         ep_end=env_switch_every + heatmap_path_num,
                                                         world=world,
                                                         path_start=path_start)

            larger_verts_les = get_path_verts_multiple(structure, prefix='lesionLEC',
                                                       ep_start=env_switch_every + 1,
                                                       ep_end=env_switch_every + heatmap_path_num, world=world,
                                                       path_start=path_start)

            data = {'path_unlesioned': path_unlesioned, 'un_ep': un_ep, 'path_lesioned': path_lesioned,
                    'les_ep': les_ep,
                    'heatmap_data_unlesioned': heatmap_data_unlesioned, 'heatmap_data_lesioned': heatmap_data_lesioned,
                    'larger_verts_unles': larger_verts_unles,
                    'larger_verts_les': larger_verts_les, 'world': world}
            with open(path.joinpath(
                    f'compare_paths_after_switch_data_{env_switch_every}_{heatmap_path_num}.pkl'),
                    'wb') as f:
                pickle.dump(data, f)
        else:
            raise NotImplementedError

    larger_heatmap_les_data = get_heatmap_data(larger_verts_les, world)
    larger_heatmap_unlesioned_data = get_heatmap_data(
        larger_verts_unles, world)

    vmin = np.min([np.min(heatmap_data_unlesioned),
                  np.min(heatmap_data_lesioned)])
    vmax = np.max([np.max(heatmap_data_unlesioned),
                  np.max(heatmap_data_lesioned)])
    lims = [vmin, vmax]
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 2, 1)
    plot_track(path_unlesioned, ax, world=world, color='r',
               label=f"Allo+Ego, Episode: {un_ep} ", lims=lims)
    ax = fig.add_subplot(1, 2, 2)
    plot_track(path_lesioned, ax, world=world, color='r',
               label=f'Allo, Episode: {les_ep}', lims=lims)
    plt.savefig(path / 'compare_paths_after_switch.png')

    vmin = np.min([np.min(larger_heatmap_unlesioned_data),
                  np.min(larger_heatmap_les_data)])
    vmax = np.max([np.max(larger_heatmap_unlesioned_data),
                  np.max(larger_heatmap_les_data)])
    lims = [vmin, vmax]
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 2, 1)
    plot_track(larger_verts_unles, ax, world=world, color='r',
               label=f"Allo+Ego, Episode: {un_ep} ", lims=lims)
    ax.set_title('Allo+Ego')
    ax = fig.add_subplot(1, 2, 2)
    plot_track(larger_verts_les, ax, world=world, color='r',
               label=f'Allo, Episode: {les_ep}', lims=lims)
    ax.set_title('Allo')
    plt.savefig(path / 'compare_heatmaps_after_switch.png')
    

    
def get_sr_change(structure=None, ego_state=0, allo_state=0, model=None, time0=0, time1=1000, path=None,
                  prefix='unlesioned',
                  figtitle=None):
    if path:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
    else:
        path = pathlib.Path('.')

    if path.joinpath(f'SR_change_data_{prefix}_{time0}_{time1}.pkl').exists() and not structure and not model:

        with open(path.joinpath(f'SR_change_data_{prefix}_{time0}_{time1}.pkl'), 'rb') as f:
            data = pickle.load(f)
        model1, model2 = data['model1'], data['model2']
    else:
        if structure and model:
            ego_srs_y, ego_srs_x = get_parameter_values(
                'ego_SR.SR_ss', structure, prefix=prefix)
            allo_srs_y, allo_srs_x = get_parameter_values(
                'allo_SR.SR_ss', structure, prefix=prefix)
            model1 = get_certain_SR_model(
                ego_srs_y, ego_srs_x, allo_srs_y, allo_srs_x, model, time0)
            model2 = get_certain_SR_model(
                ego_srs_y, ego_srs_x, allo_srs_y, allo_srs_x, model, time1 - 1)
            data = {'model1': model1, 'model2': model2}
            with open(path.joinpath(f'SR_change_data_{prefix}_{time0}_{time1}.pkl'), 'wb') as f:
                pickle.dump(data, f)
        else:
            raise NotImplementedError

    fig = plt.figure(figsize=(10, 10))

    grids = fig.add_gridspec(nrows=2, ncols=2, wspace=0.1)
    add_allocentric_SR_grid(grids[0, 0], fig, model1, allo_state, ego=False)
    add_egocentric_SR_grid(grids[0, 1], fig, model1, ego_state)
    add_allocentric_SR_grid(grids[1, 0], fig, model2, allo_state, ego=False)
    add_egocentric_SR_grid(grids[1, 1], fig, model2, ego_state)
    if figtitle:
        fig.suptitle(figtitle)
        plt.savefig(path / f'{figtitle}_SR_change_{prefix}.png')
    else:
        plt.savefig(path / f'SR_change_{prefix}.png')
        

def get_all_paths(structure, prefix='unlesioned', episode=0, path_start=None):
    raise NotImplementedError


def get_path(structure, prefix='unlesioned', episode=0, path_start=None, struct_all_seeds=False, seednum=None,
             return_seed_id=False):
    structure_clean = remove_empty_dicts(structure)
    structure_keys = list(structure_clean.keys())

    if return_seed_id or seednum is not None:
        paths_y_, paths_x_, ignore_indices = get_parameter_values(
            'paths', structure, prefix=prefix, ignore=True)
        ignore_indices = ignore_indices or []
        kept_keys = [structure_keys[idx] for idx in range(len(structure_keys)) if idx not in ignore_indices]
    else:
        paths_y_, paths_x_ = get_parameter_values(
            'paths', structure, prefix=prefix)
        kept_keys = structure_keys

    if return_seed_id or seednum is not None:
        paths_y, paths_x, seed_indices = sort_paths(paths_y_, paths_x_, return_seed_indices=True)
    else:
        paths_y, paths_x = sort_paths(paths_y_, paths_x_)
        seed_indices = None

    import bisect
    index = bisect.bisect_left(paths_x, episode)
    # if path_start:
    #     later_paths = paths_y[index:]
    #     for num, path in enumerate(later_paths):
    #         for start in path_start:
    #             for k in range(len(path)):
    #                 if path[k][0] == start:
    #                     index = index + num
    #                     episode = paths_x[index]
    #                     path_y = np.stack(paths_y[index][0])
    #                     return np.flip(path_y, 1), episode

    paths_at_index = []
    for seed_num_ in range(len(paths_y[index])):
        paths_at_index.append(paths_y[index][seed_num_])

    if seednum is not None:
        def _match_seed(target):
            for idx, key in enumerate(kept_keys):
                if key == target or str(key) == str(target):
                    return idx, key
            raise KeyError(f"Seed {target} not found in structure")

        target_idx, matched_key = _match_seed(seednum)
        seed_indices_at_index = seed_indices[index]
        try:
            path_position = seed_indices_at_index.index(target_idx)
        except ValueError as err:
            raise ValueError(
                f"No path available for seed {seednum} at episode {episode}") from err
        selected_path = paths_at_index[path_position]
        selected_seed_key = matched_key
    else:
        longest_idx = max(
            range(len(paths_at_index)),
            key=lambda idx: np.linalg.norm(paths_at_index[idx][-1] - paths_at_index[idx][0])
            if len(paths_at_index[idx]) > 1 else 0)
        selected_path = paths_at_index[longest_idx]
        if seed_indices is not None:
            selected_seed_key = kept_keys[seed_indices[index][longest_idx]]
        else:
            selected_seed_key = None

    path_y = np.stack(selected_path)

    if return_seed_id:
        return np.flip(path_y, 1), episode, selected_seed_key
    return np.flip(path_y, 1), episode


def get_hole_times(structure, prefix='unlesioned', worlds=None, all_worlds=None, switch_every=1000):
    if not all_worlds:
        paths_y_, paths_x_ = get_parameter_values(
            'paths', structure, prefix=prefix)
        paths_y, paths_x = sort_paths(paths_y_, paths_x_)
        if worlds is None:
            worlds = structure[list(structure.keys())[0]]['unlesioned']['worlds']
        hole_times, x_ = get_hole_times_(paths_y, paths_x, worlds, switch_every=switch_every)
        hole_times_mean = hole_times
        hole_times_sem = np.zeros_like(hole_times)
    else:
        paths_y_, paths_x_, ignore = get_parameter_values(
            'paths', structure, prefix=prefix, ignore=True)
        paths_y, paths_x = sort_paths(paths_y_, paths_x_)
        hole_times = []
        x_ = []
        index_ = 0
        for index, worlds_ in enumerate(all_worlds):
            
            if index not in ignore:
                paths_y_i = []
                paths_x_i = []
                for path in paths_y:
                    paths_y_i.append(path[index_])
                        
                paths_x_i = paths_x
                hole_times_, x = get_hole_times_(paths_y_i, paths_x_i, worlds_, switch_every=switch_every)
                x = np.array(x)
                hole_times.append(hole_times_)
                x_.append(x)
                index_ += 1
            
        
        x_ = np.array(x_)
        hole_times = np.array(hole_times)
        hole_times_mean = np.mean(hole_times, axis=0)
        hole_times_sem = np.std(hole_times, axis=0, ddof=1) / np.sqrt(hole_times.shape[0])
            
        
    return hole_times_mean, hole_times_sem, x_

def get_path_verts_multiple(structure, prefix, ep_start, ep_end, world=None, path=None, path_start=None):
    if path:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
    paths_y, paths_x = get_parameter_values('paths', structure, prefix=prefix)
    paths_y = paths_y[0]
    paths_x = paths_x[0]
    paths_y = [y for _, y in sorted(
        zip(paths_x, paths_y), key=lambda pair: pair[0])]
    paths_x = sorted(paths_x)
    import bisect
    index_start = bisect.bisect_left(paths_x, ep_start)
    index_end = bisect.bisect_left(paths_x, ep_end)
    if path_start:
        flat_list = get_locations(paths_y[index_start:index_end], path_start)
    else:
        flat_list = [x for xss in paths_y[index_start:index_end]
                     for xs in xss for x in xs]
    flat_list = np.stack(flat_list)
    verts = np.flip(flat_list, 1)

    return verts

        
    
                
                
    