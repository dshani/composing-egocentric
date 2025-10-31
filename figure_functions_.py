"""
@author: Daniel Shani
"""
from matplotlib.patches import FancyArrow
from copy import deepcopy
import bisect
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pickle

import matplotlib.cm as mcm
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter1d
from matplotlib.lines import Line2D  # Import Line2D for custom legend entries
from matplotlib.transforms import Bbox, TransformedBbox
from collections.abc import Sequence



from structure_functions_ import *

fsize_label = 20
fsize_leg = 10
cm = 1.0/3.0


def generate_schematic(ax, env, pos=None, alias=True):
    world = np.maximum(env.world, 0)
    masked_array = np.ma.masked_where(env.world == -1, world)
    cmap = mcm.Greys
    cmap.set_bad(color='green')

    ax.imshow(
        masked_array,
        cmap=cmap,
        vmin=np.min(env.world),
        vmax=np.max(env.world))

    # add agent at position (x,y) which consists of a small yellow
    #  CHARACTER with a white border in direction d

    if pos:
        x, y, d = pos

        ax.add_patch(
            patches.Circle(
                (x, y), 0.5, edgecolor='w', facecolor='y', fill=True))
        ax.arrow(
            x, y, 0.5 * np.sin(np.pi * d / 2),
            -0.5 * np.cos(np.pi * d / 2), facecolor='y', edgecolor='w',
            linewidth=1,
            head_width=0.5, head_length=0.5)

        # plt.gca().add_patch(
        #     patches.RegularPolygon(
        #         (x, y), 3, 0.5, orientation=np.pi + np.pi * d / 2,
        #         edgecolor='w', facecolor='y', fill=True))

        # add red rectangle of width 5 and height 3 around location (x,
        # y) rotated in direction d around point (x, y)
        ax.add_patch(
            patches.Rectangle(
                (x - env.pars.horizon - .5, y - env.pars.horizon - .5),
                2 * env.pars.horizon + 1,
                env.pars.horizon + 1,
                edgecolor='r',
                facecolor='none',
                fill=False, angle=90 * d, rotation_point=(x, y)))

        ego = env.get_egocentric_view(env.world, (y, x), d, display=False)
        if alias:
            aliases = env.ego_to_allo[-1].get(ego[0])
            for y_, x_, d_ in aliases:
                ax.add_patch(
                    patches.Rectangle(
                        (x_ - env.pars.horizon - .5, y_ - env.pars.horizon - .5),
                        2 * env.pars.horizon + 1,
                        env.pars.horizon + 1,
                        edgecolor='lightblue',
                        linestyle='dashed',
                        facecolor='none',
                        fill=False, angle=90 * d_, rotation_point=(x_, y_)))

    ax.set_xticks([])
    ax.set_yticks([])


def generate_allocentric_plot(ax, env, x, y, relative_sizes = [1, 1, 1, 1]):
    square = np.zeros_like(env.world)
    # add cross at position (x,y)

    square[y, x] = 1

    # use axes coordinates to place the center of the cross at (x,y)

    ax.add_patch(
        patches.Circle(
            (x, y), 0.5, edgecolor='k', facecolor='k', fill=True))

    ax.imshow(square, cmap='Greys')
    ax.set_xticks([])
    ax.set_yticks([])
    # add dotted arrows in directions North, East, South, West from
    # position (x,y) or length 2
    
    add_allo_arrows(x, y, ax, relative_sizes)

    ax.set_title('$s^A$')
    ax.set_ylabel('$a^A$', color='b')
    
def add_allo_arrows(x, y, ax, relative_sizes = [1, 1, 1, 1]):
    
    ax.arrow(
        x, y + 0.5, 0, 1., color='b', linewidth=1*relative_sizes[2],
        head_width=0.5*relative_sizes[2], head_length=0.5*relative_sizes[2])
    # label arrow
    ax.text(x, y + 2.9, 'S', color='b', fontsize=4*relative_sizes[2], ha='center')
    ax.arrow(
        x + 0.5, y, 1., 0, color='b', linewidth=1*relative_sizes[1],
        head_width=0.5*relative_sizes[1], head_length=0.5*relative_sizes[1])
    ax.text(x + 2.7, y + 0.2, 'E', color='b', fontsize=4*relative_sizes[1], ha='center')
    ax.arrow(
        x, y - 0.5, 0, -1., color='b', linewidth=1*relative_sizes[0],
        head_width=0.5*relative_sizes[0], head_length=0.5*relative_sizes[0])
    ax.text(x, y - 2.5, 'N', color='b', fontsize=4*relative_sizes[0], ha='center')
    ax.arrow(
        x - 0.5, y, -1., 0, color='b', linewidth=1*relative_sizes[3],
        head_width=0.5*relative_sizes[3], head_length=0.5*relative_sizes[3])
    ax.text(x - 2.7, y + 0.2, 'W', color='b', fontsize=4*relative_sizes[3], ha='center')
    


def generate_egocentric_plot(ax, env, x, y, d, relative_sizes = [1, 1, 1, 1]):
    
    env.get_egocentric_view(env.world, (y, x), d, display=True, ax=ax)
    ax.set_ylabel('$a^E$', color='b')
    # add up arrow and curly clockwise and anticlockwise arrows
    
    y_shift = -0.25
    x_shift = 0
    
    c_y = env.pars.horizon + y_shift
    c_x = env.pars.horizon + x_shift
    
    
    endpoints = [(c_x, c_y-1),
                 (c_x+0.5, c_y),
                 (c_x, c_y+0.4),
                 (c_x-0.5, c_y),
                 ]
    
    connection_styles = ['arc3,rad=0', 'arc3,rad=-1', 'arc3,rad=0', 'arc3,rad=1']
    
    text_positions = [(c_x, c_y - 1),
                      (c_x + 0.65, c_y),
                      (c_x, c_y + 0.55),
                      (c_x - 0.65, c_y),
                      ]
    
    text_titles = ['F', 'R', 'B', 'L']
    
    for i, (x_, y_) in enumerate(endpoints):
        ax.add_patch(
            patches.FancyArrowPatch(
                (c_x, c_y), (x_, y_),
                arrowstyle='Fancy, '
                           f'tail_width={0.1*relative_sizes[i]}, '
                           f'head_width={5*relative_sizes[i]}, head_length={8*relative_sizes[i]}',
                color='b',
                connectionstyle=connection_styles[i],
                lw=1*relative_sizes[i]))
        
        ax.text(text_positions[i][0], text_positions[i][1], text_titles[i], color='b', fontsize=10*relative_sizes[i], ha='center')

    ax.set_title('$s^E$')


def generate_value_plot(ax, state_values, vmin, vmax, direction=None, colorbar=False, **kwargs):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    if direction is not None:
        ax.title.set_text(f"{direction}")
    im = ax.imshow(state_values[:, :], vmin=vmin, vmax=vmax, cmap='plasma')
    ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False)
    fig = ax.get_figure()
    cax = None
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        # colorbar
        fig.colorbar(im, cax=cax, orientation='vertical')
        # change fontsize of colorbar
        cax.tick_params(**kwargs)
    return ax, cax
        
def generate_lesion_plot_(ax, inputs, labels, env_switch_every=1000):

    lines = []
    if len(inputs) == 2:
        colors = ['r', 'b']
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, len(inputs)))
    
    all_x_values = []

    for i, input_ in enumerate(inputs):
        y, y_sem, x = input_
        
        # Check for length mismatch
        if len(x) != len(y) or len(y) != len(y_sem):
            raise ValueError(f"Lengths of x ({len(x)}), y ({len(y)}), and y_sem ({len(y_sem)}) do not match.")
        
        # Sort the data
        x = np.array(x)
        y = np.array(y)
        sorted_indices = np.argsort(x)
        x = x[sorted_indices]
        y = y[sorted_indices]
        y_sem = y_sem[sorted_indices]
        
        # Collect all x-values
        all_x_values.extend(x)
        
        
        # Plot mean line
        line, = ax.plot(x, y, color=colors[i], label=labels[i])
        
        # Fill between mean ± SEM
        ax.fill_between(x, y - y_sem, y + y_sem,
                        color=colors[i], alpha=0.3)
        
        lines.append(line)
    
    switch_color = '0.7'
    
    # Determine the number of switches based on max x-value
    max_x = max(all_x_values)
    num_switches = int(max_x // env_switch_every) + 1
    for i in range(1, num_switches):
        ax.axvline(x=i * env_switch_every, color=switch_color, linestyle='--')
        
    # Custom legend entry for vertical lines
    vline = Line2D([0], [0], color=switch_color, linestyle='--')
    
    lines.append(vline)
    labels.append('world switch')
    
    ax.legend(lines, labels)
    ax.set_ylabel('Steps', fontsize='x-small')
    ax.tick_params(labelsize='xx-small')
    ax.set_yscale("log")
    ax.set_xlim(0, max_x)


def generate_lesion_plot(ax, lesion_y, unlesion_y, lesion_x, unlesion_x, env_switch_every=1000):
    ax.plot(lesion_x, lesion_y, color='r')
    ax.plot(unlesion_x, unlesion_y, color='b')
    switch_color = '0.7'

    for i in range(1, 6):
        ax.axvline(x=i * env_switch_every, color=switch_color, linestyle='--')

    ax.legend(['allocentric', 'egocentric + allocentric', 'world switch'])
    ax.set_ylabel('steps', fontsize='x-small')
    ax.tick_params(
        labelsize='xx-small')
    ax.set_yscale("log")


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def _find_chunked_metrics_path(base_path):
    """Return the chunked-occupancy metrics JSON within ``base_path`` or its parents."""

    if base_path is None:
        return None

    base_path = pathlib.Path(base_path)
    if base_path.is_file():
        search_roots = [base_path.parent]
    else:
        search_roots = [base_path]
    search_roots.extend(base_path.parents)

    seen = set()
    for root in search_roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        candidate = root / 'chunked_occupancy' / 'current_hole_locations_barrier0' / 'chunked_occupancy_metrics.json'
        if candidate.exists():
            return candidate
        legacy_candidate = root / 'chunked_occupancy' / 'current_hole_locations' / 'chunked_occupancy_metrics.json'
        if legacy_candidate.exists():
            return legacy_candidate
    return None


def _load_chunked_occupancy_summary(base_path, expected_chunk_size=25):
    """Load per-chunk occupancy summaries if available."""

    metrics_path = _find_chunked_metrics_path(base_path)
    if metrics_path is None:
        return None

    try:
        payload = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    barrier_value = payload.get('barrier_thickness')
    barrier_int = _safe_int(barrier_value)
    if barrier_int is None:
        barrier_float = _safe_float(barrier_value)
        if math.isfinite(barrier_float):
            barrier_int = int(barrier_float)
    if barrier_int not in (None, 0):
        return None

    chunk_size_value = _safe_int(payload.get('chunk_size'))
    if not chunk_size_value or chunk_size_value <= 0:
        chunk_size_value = expected_chunk_size

    aggregate = payload.get('aggregate')
    if not isinstance(aggregate, dict):
        return None

    result = {}
    for prefix in ('lesionLEC', 'unlesioned'):
        prefix_block = aggregate.get(prefix)
        if not isinstance(prefix_block, dict):
            continue

        indices = []
        means = []
        stds = []
        counts = []
        for key, stats in prefix_block.items():
            chunk_index = _safe_int(key)
            if chunk_index is None:
                continue
            if not isinstance(stats, dict):
                continue
            mean_val = _safe_float(stats.get('mean_fraction'))
            if not math.isfinite(mean_val):
                continue
            std_val = _safe_float(stats.get('std_fraction'))
            seed_count = _safe_int(stats.get('seed_count'))
            indices.append(chunk_index)
            means.append(mean_val)
            stds.append(std_val)
            counts.append(seed_count if seed_count is not None else 0)

        if not indices:
            continue

        order = np.argsort(indices)
        indices_arr = np.asarray(indices, dtype=float)[order]
        means_arr = np.asarray(means, dtype=float)[order]
        stds_arr = np.asarray(stds, dtype=float)[order]
        counts_arr = np.asarray(counts, dtype=float)[order]

        sem_arr = np.full_like(means_arr, np.nan, dtype=float)
        valid_mask = (counts_arr > 0) & np.isfinite(stds_arr)
        sem_arr[valid_mask] = stds_arr[valid_mask] / np.sqrt(counts_arr[valid_mask])

        result[prefix] = (indices_arr, means_arr, sem_arr)

    if not result:
        return None

    return result, int(chunk_size_value)


def _plot_chunked_occupancy(ax, chunk_data, chunk_size, env_switch_every):
    """Render chunked occupancy traces on ``ax``."""

    color_map = {'lesionLEC': 'r', 'unlesioned': 'b'}
    legend_labels = {'lesionLEC': 'allocentric', 'unlesioned': 'allocentric + egocentric'}

    handles = []
    labels = []
    lower_bounds = []
    upper_bounds = []
    max_x = 0.0

    for prefix in ('lesionLEC', 'unlesioned'):
        if prefix not in chunk_data:
            continue

        indices, means, sems = chunk_data[prefix]
        indices = np.asarray(indices, dtype=float)
        means = np.asarray(means, dtype=float)
        sems = np.asarray(sems, dtype=float)

        if chunk_size and chunk_size > 0:
            x_values = indices * float(chunk_size)
        else:
            x_values = indices

        finite_mask = np.isfinite(x_values) & np.isfinite(means)
        if not finite_mask.any():
            continue

        x_values = x_values[finite_mask]
        means = means[finite_mask]
        sems = sems[finite_mask] if sems.size else np.full_like(means, np.nan)

        order = np.argsort(x_values)
        x_values = x_values[order]
        means = means[order]
        if sems.size == means.size:
            sems = sems[order]
        else:
            sems = np.full_like(means, np.nan)

        color = color_map[prefix]
        label = legend_labels[prefix]

        line, = ax.plot(x_values, means, color=color, label=label)
        handles.append(line)
        labels.append(label)

        if sems.size == means.size:
            finite_sem = np.isfinite(sems)
            if finite_sem.any():
                lower = means[finite_sem] - sems[finite_sem]
                upper = means[finite_sem] + sems[finite_sem]
                lower_bounds.extend(lower.tolist())
                upper_bounds.extend(upper.tolist())
                ax.fill_between(x_values[finite_sem], lower, upper, color=color, alpha=0.3)

        if x_values.size:
            max_x = max(max_x, float(x_values[-1]))

    if not handles:
        return False

    switch_color = '0.7'
    if env_switch_every and env_switch_every > 0 and max_x > 0:
        num_switches = int(max_x // env_switch_every) + 1
        for i in range(1, num_switches):
            ax.axvline(x=i * env_switch_every, color=switch_color, linestyle='--')
        handles.append(Line2D([0], [0], color=switch_color, linestyle='--'))
        labels.append('world switch')

    ax.legend(handles, labels)

    size_label = chunk_size if chunk_size and chunk_size > 0 else '?'
    ax.set_ylabel(f'Proportion of time in current hole (chunk size {size_label})', fontsize='xx-small')
    if chunk_size and chunk_size > 0:
        ax.set_xlabel('Episode (chunk start)', fontsize='xx-small')
    else:
        ax.set_xlabel('Chunk index', fontsize='xx-small')
    ax.tick_params(labelsize='xx-small')

    if lower_bounds and upper_bounds:
        y_min = min(lower_bounds)
        y_max = max(upper_bounds)
        if math.isfinite(y_min) and math.isfinite(y_max):
            y_range = y_max - y_min
            margin = 0.05 * y_range if y_range > 0 else 0.05
            lower = max(0.0, y_min - margin)
            upper = min(1.0, y_max + margin)
            if upper <= lower:
                upper = min(1.0, lower + 0.1)
            ax.set_ylim(lower, upper)
        else:
            ax.set_ylim(0, 1)
    else:
        ax.set_ylim(0, 1)

    if max_x > 0:
        ax.set_xlim(0, max_x)
    elif chunk_size and chunk_size > 0:
        ax.set_xlim(0, chunk_size)
    else:
        ax.set_xlim(0, 1)

    return True


def generate_allo_SR_plot(ax, array, arrmin, arrmax, direction):
    im = ax.imshow(
        array[:, :],
        vmin=arrmin, vmax=arrmax, cmap='plasma')
    ax.title.set_text(f"{direction}")
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def generate_action_value_fig(model=None, struct_single_seed=None, struct_all_seeds=None, sigma=0.1, path=None, fsize_label=12,
                              env_switch_every=1000, delay=1):

    directions = ["\u2191", "\u2192", "\u2193", "\u2190"]
    if path:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
    else:
        path = pathlib.Path('.')

    fig = plt.figure(figsize=(40, 20))
    grids = fig.add_gridspec(nrows=4, ncols=4, wspace=0.5,
                             hspace=0.5)
    if struct_single_seed is not None:
        struct_single_seed = remove_empty_dicts(struct_single_seed)
    struct_all_seeds = remove_empty_dicts(struct_all_seeds)
    worlds = struct_all_seeds[list(struct_all_seeds.keys())[
        0]]['unlesioned']['worlds']

    inner_grid = grids[0, :-1].subgridspec(
        5, 6, width_ratios=(1, 1, 1, 1, 1, 0.03), wspace=0.01, hspace=0.01)

    inner_grid1 = grids[1, :-1].subgridspec(
        5, 6, width_ratios=(1, 1, 1, 1, 1, 0.03), wspace=0.01, hspace=0.01)
    inner_grid2 = grids[2, :-1].subgridspec(
        5, 6, width_ratios=(1, 1, 1, 1, 1, 0.03), wspace=0.01, hspace=0.01)

    model1 = deepcopy(model)

    world = struct_single_seed[list(struct_single_seed.keys())[
        0]]['unlesioned']['worlds'][0]
    model1.switch_world(world)
    model_unlesioned = deepcopy(model1)

    model_lesioned = deepcopy(model1)

    lesioned_allo_srs_y, lesioned_allo_srs_x = get_parameter_values('allo_SR.SR_ss', struct_single_seed,
                                                                    prefix='lesionLEC')
    unlesioned_allo_srs_y, unlesioned_allo_srs_x = get_parameter_values('allo_SR.SR_ss', struct_single_seed,
                                                                        prefix='unlesioned')

    lesioned_ego_srs_y, lesioned_ego_srs_x = get_parameter_values(
        'ego_SR.SR_ss', struct_single_seed, prefix='lesionLEC')

    unlesioned_ego_srs_y, unlesioned_ego_srs_x = get_parameter_values('ego_SR.SR_ss', struct_single_seed,
                                                                      prefix='unlesioned')

    lesioned_weights_y, lesioned_weights_x = get_parameter_values(
        'weight', struct_single_seed, prefix='lesionLEC')
    unlesioned_weights_y, unlesioned_weights_x = get_parameter_values(
        'weight', struct_single_seed, prefix='unlesioned')

    import bisect
    index = bisect.bisect_left(unlesioned_allo_srs_x[0], env_switch_every - 1)

    model_unlesioned.ego_SR.SR_ss = unlesioned_ego_srs_y[0][index]
    model_unlesioned.allo_SR.SR_ss = unlesioned_allo_srs_y[0][index]
    model_unlesioned.weight = unlesioned_weights_y[0][index]

    model_lesioned.ego_SR.SR_ss = lesioned_ego_srs_y[0][index]
    model_lesioned.allo_SR.SR_ss = lesioned_allo_srs_y[0][index]
    model_lesioned.weight = lesioned_weights_y[0][index]
    unlesioned_state_values, ego_unlesioned_state_values, allo_unlesioned_state_values, vmin, vmax, egomin, egomax, amin, amax = get_value_functions(
        model_unlesioned,
        split=True, action="all")
    
    plot_xyda = [[np.zeros((model.env.size, model.env.size)) for _ in range(4)] for _ in range(4)]
    
    for d in range(4):
        mu_a_unles_d = np.mean(unlesioned_state_values, axis=3)
        for a in range(4):
            plot_xyda[d][a] = unlesioned_state_values[:, :, d, a] - mu_a_unles_d[:, :, d]
            
    grids = fig.add_gridspec(nrows=4, ncols=4, wspace=0.5,
                             hspace=0.5)
    
    for d in range(4):
        for a in range(4):
            ax = fig.add_subplot(grids[d, a])
            vmin, vmax = np.min(plot_xyda[d][:]), np.max(plot_xyda[d][:])
            im = ax.imshow(
                plot_xyda[d][a],
                vmin=vmin, vmax=vmax, cmap='bwr')
            ax.title.set_text(f"{directions[d]} {directions[a]}")
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax)
            
            
    plt.savefig(path.joinpath(f"action_value_functions.svg"))
    plt.close(fig)
    
    fig = plt.figure(figsize=(40, 20))
    grids = fig.add_gridspec(nrows=4, ncols=4, wspace=0.5,
                             hspace=0.5)
    
    plot_xyda = [[np.zeros((model.env.size, model.env.size)) for _ in range(4)] for _ in range(4)]
    
    for d in range(4):
        mu_a_unles_d = np.mean(ego_unlesioned_state_values, axis=3)
        for a in range(4):
            plot_xyda[d][a] = ego_unlesioned_state_values[:, :, d, a] - mu_a_unles_d[:, :, d]
            

    
    for d in range(4):
        for a in range(4):
            ax = fig.add_subplot(grids[d, a])
            vmin, vmax = np.min(plot_xyda[d][:]), np.max(plot_xyda[d][:])
            im = ax.imshow(
                plot_xyda[d][a],
                vmin=vmin, vmax=vmax, cmap='bwr')
            ax.title.set_text(f"{directions[d]} {directions[a]}")
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax)
            
            
    plt.savefig(path.joinpath(f"action_value_functions_ego.svg"))
    plt.close(fig)
    
    fig = plt.figure(figsize=(40, 20))
    grids = fig.add_gridspec(nrows=4, ncols=4, wspace=0.5,
                             hspace=0.5)
    
    plot_xyda = [[np.zeros((model.env.size, model.env.size)) for _ in range(4)] for _ in range(4)]
    
    for d in range(4):
        mu_a_unles_d = np.mean(allo_unlesioned_state_values, axis=3)
        for a in range(4):
            plot_xyda[d][a] = allo_unlesioned_state_values[:, :, d, a] - mu_a_unles_d[:, :, d]
            
    
    for d in range(4):
        for a in range(4):
            ax = fig.add_subplot(grids[d, a])
            vmin, vmax = np.min(plot_xyda[d][:]), np.max(plot_xyda[d][:])
            im = ax.imshow(
                plot_xyda[d][a],
                vmin=vmin, vmax=vmax, cmap='bwr')
            ax.title.set_text(f"{directions[d]} {directions[a]}")
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax)
            
            
    plt.savefig(path.joinpath(f"action_value_functions_allo.svg"))
    plt.close(fig)
    
    fig = plt.figure(figsize=(40, 20))
    grids = fig.add_gridspec(nrows=4, ncols=4, wspace=0.5,
                             hspace=0.5)
    
    
    model2 = deepcopy(model)

    world = struct_single_seed[list(struct_single_seed.keys())[
        0]]['unlesioned']['worlds'][1]
    model2.switch_world(world)
    model_unlesioned = deepcopy(model2)
    model_lesioned = deepcopy(model2)

    lesioned_allo_srs_y, lesioned_allo_srs_x = get_parameter_values('allo_SR.SR_ss', struct_single_seed,
                                                                    prefix='lesionLEC')
    unlesioned_allo_srs_y, unlesioned_allo_srs_x = get_parameter_values('allo_SR.SR_ss', struct_single_seed,
                                                                        prefix='unlesioned')
    lesioned_ego_srs_y, lesioned_ego_srs_x = get_parameter_values(
        'ego_SR.SR_ss', struct_single_seed, prefix='lesionLEC')
    unlesioned_ego_srs_y, unlesioned_ego_srs_x = get_parameter_values('ego_SR.SR_ss', struct_single_seed,
                                                                      prefix='unlesioned')

    lesioned_weights_y, lesioned_weights_x = get_parameter_values(
        'weight', struct_single_seed, prefix='lesionLEC')
    unlesioned_weights_y, unlesioned_weights_x = get_parameter_values(
        'weight', struct_single_seed, prefix='unlesioned')

    import bisect
    index = bisect.bisect_left(
        unlesioned_allo_srs_x[0], env_switch_every + delay)

    model_unlesioned.ego_SR.SR_ss = unlesioned_ego_srs_y[0][index]
    model_unlesioned.allo_SR.SR_ss = unlesioned_allo_srs_y[0][index]
    model_unlesioned.weight = unlesioned_weights_y[0][index]

    model_lesioned.ego_SR.SR_ss = lesioned_ego_srs_y[0][index]
    model_lesioned.allo_SR.SR_ss = lesioned_allo_srs_y[0][index]
    model_lesioned.weight = lesioned_weights_y[0][index]
    unlesioned_state_values, vmin, vmax = get_value_functions(model_unlesioned,
                                                              split=False, action="all")
    lesioned_state_values, lmin, lmax = get_value_functions(
        model_lesioned, split=False, action="all")
    
    
    plot_xyda = [[np.zeros((model.env.size, model.env.size)) for _ in range(4)] for _ in range(4)]
    
    for d in range(4):
        mu_a_unles_d = np.mean(unlesioned_state_values, axis=3)
        for a in range(4):
            plot_xyda[d][a] = unlesioned_state_values[:, :, d, a] - mu_a_unles_d[:, :, d]
            
    grids = fig.add_gridspec(nrows=4, ncols=4, wspace=0.5,
                             hspace=0.5)
    
    for d in range(4):
        for a in range(4):
            ax = fig.add_subplot(grids[d, a])
            vmin, vmax = np.min(plot_xyda[d][:]), np.max(plot_xyda[d][:])
            im = ax.imshow(
                plot_xyda[d][a],
                vmin=vmin, vmax=vmax, cmap='bwr')
            ax.title.set_text(f"{directions[d]} {directions[a]}")
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax)
            
            
    plt.savefig(path.joinpath(f"action_value_functions_switch_{delay}.svg"))
    plt.close(fig)
    
    fig = plt.figure(figsize=(40, 20))
    grids = fig.add_gridspec(nrows=4, ncols=4, wspace=0.5,
                             hspace=0.5)
    
    plot_xyda = [[np.zeros((model.env.size, model.env.size)) for _ in range(4)] for _ in range(4)]
    
    for d in range(4):
        mu_a_unles_d = np.mean(lesioned_state_values, axis=3)
        for a in range(4):
            plot_xyda[d][a] = lesioned_state_values[:, :, d, a] - mu_a_unles_d[:, :, d]
                
    for d in range(4):
        for a in range(4):
            ax = fig.add_subplot(grids[d, a])
            vmin, vmax = np.min(plot_xyda[d][:]), np.max(plot_xyda[d][:])
            im = ax.imshow(
                plot_xyda[d][a],
                vmin=vmin, vmax=vmax, cmap='bwr')
            ax.title.set_text(f"{directions[d]} {directions[a]}")
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax)
            
            
    plt.savefig(path.joinpath(f"lesion_switch_{delay}.svg"))
    plt.close(fig)
    
def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.tick_params(labelsize='xx-small')
    plt.sca(last_axes)
    return cbar
    



def _extract_parameter_series_for_seed(prefix_data, param):
    """Extract ordered time-series snapshots for a stored parameter."""

    if prefix_data is None or param not in prefix_data:
        return None

    xs = []
    ys = []
    for block in prefix_data[param]:
        if block is None:
            continue
        try:
            iterable = list(block)
        except TypeError:
            continue
        for value in iterable:
            if value is None:
                continue
            if (
                isinstance(value, Sequence)
                and not isinstance(value, (str, bytes))
                and len(value) >= 2
            ):
                x_val, y_val = value[0], value[1]
            else:
                x_val, y_val = len(xs), value

            try:
                scalar_x = float(np.asarray(x_val).item())
            except Exception:
                try:
                    scalar_x = float(x_val)
                except Exception:
                    continue

            xs.append(scalar_x)
            ys.append(np.array(y_val, copy=True))

    if not xs:
        return None

    ordered_pairs = sorted(zip(xs, ys), key=lambda pair: pair[0])
    combined = {}
    for x_val, y_val in ordered_pairs:
        combined[x_val] = y_val

    x_sorted = list(combined.keys())
    y_sorted = [combined[x_val] for x_val in x_sorted]
    return x_sorted, y_sorted


def _select_snapshot(series, target_time):
    """Return the saved snapshot closest to ``target_time``."""

    xs, ys = series
    if not xs:
        raise ValueError("No recorded timesteps for parameter")

    index = bisect.bisect_left(xs, target_time)

    if index == 0:
        chosen = ys[0]
    elif index >= len(xs):
        chosen = ys[-1]
    else:
        prev_idx = index - 1
        prev_x = xs[prev_idx]
        next_x = xs[index]

        if abs(next_x - target_time) < abs(target_time - prev_x):
            chosen = ys[index]
        else:
            chosen = ys[prev_idx]

    return np.array(chosen, copy=True)


def _initialise_model_for_world(model_template, worlds, target_index):
    model_copy = deepcopy(model_template)
    worlds_list = list(worlds)
    if not worlds_list:
        raise ValueError("No worlds available for seed")

    resolved_index = max(0, min(target_index, len(worlds_list) - 1))
    for world in worlds_list:
        model_copy.switch_world(world)
    model_copy.switch_world(worlds_list[resolved_index])
    return model_copy, resolved_index, worlds_list


def _apply_snapshot_to_model(model, series_map, target_time):
    ego_snapshot = _select_snapshot(series_map['ego_SR.SR_ss'], target_time)
    allo_snapshot = _select_snapshot(series_map['allo_SR.SR_ss'], target_time)
    weight_snapshot = _select_snapshot(series_map['weight'], target_time)

    model.ego_SR.SR_ss = ego_snapshot
    model.ego_SR.SR_sas = model.ego_SR.SR_ss[:, np.newaxis, :].repeat(4, axis=1)
    model.allo_SR.SR_ss = allo_snapshot
    model.allo_SR.SR_sas = model.allo_SR.SR_ss[:, np.newaxis, :].repeat(4, axis=1)
    model.weight = weight_snapshot


def _directional_mean_value_function(model):
    """Collapse directional Q-values into a single grid of state values.

    The correlation analysis compares spatial value predictions drawn from
    different time-points or lesion conditions on a per-location basis.
    ``get_value_functions`` returns a tensor with a final dimension indexing
    the four egocentric directions.  Averaging over that axis yields a grid
    that matches the spatial layout of the world and is therefore suitable for
    computing spatial correlations.
    """

    state_values, *_ = get_value_functions(model, split=True)
    return np.mean(state_values, axis=2)


def _safe_correlation(values_a, values_b):
    if values_a.size == 0 or values_b.size == 0:
        return np.nan
    mask = np.isfinite(values_a) & np.isfinite(values_b)
    if not np.any(mask):
        return np.nan
    filtered_a = values_a[mask]
    filtered_b = values_b[mask]
    if filtered_a.size == 0 or filtered_b.size == 0:
        return np.nan
    if np.allclose(filtered_a, filtered_a[0]) or np.allclose(filtered_b, filtered_b[0]):
        # When one array is (near) constant, the Pearson correlation is undefined.
        # For robustness in downstream summaries, treat this as zero correlation
        # rather than dropping the seed entirely.
        return 0.0
    corr_matrix = np.corrcoef(filtered_a, filtered_b)
    return float(corr_matrix[0, 1])


def _compute_seed_value_function_correlations(seed_entry, model_template, pre_time, post_time):
    """Return the within-agent pre/post correlations for the provided seed."""

    def _series_for_agent(agent_struct):
        if agent_struct is None:
            return None
        series = {}
        value_series = _extract_parameter_series_for_seed(
            agent_struct, 'value_snapshots'
        )
        if value_series is not None:
            series['value_snapshots'] = value_series
            return series
        for param in ('ego_SR.SR_ss', 'allo_SR.SR_ss', 'weight'):
            series[param] = _extract_parameter_series_for_seed(agent_struct, param)
            if series[param] is None:
                return None
        return series

    def _correlate_snapshots(agent_struct, series):
        if agent_struct is None or series is None:
            return np.nan

        try:
            worlds = list(agent_struct['worlds'])
        except (KeyError, TypeError):
            worlds = []

        if 'value_snapshots' in series:
            if not worlds:
                return np.nan
            pre_index = 0
            post_index = min(1, len(worlds) - 1)
            try:
                pre_world = np.asarray(worlds[pre_index])
                post_world = np.asarray(worlds[post_index])
                pre_grid = _select_snapshot(series['value_snapshots'], pre_time)
                post_grid = _select_snapshot(series['value_snapshots'], post_time)
            except (ValueError, KeyError, IndexError, TypeError):
                return np.nan

            joint_mask = (pre_world <= 0) | (post_world <= 0)
            return _safe_correlation(pre_grid[joint_mask], post_grid[joint_mask])

        if model_template is None:
            return np.nan

        try:
            pre_model, pre_index, pre_worlds = _initialise_model_for_world(
                model_template, agent_struct['worlds'], target_index=0)
            post_model, post_index, post_worlds = _initialise_model_for_world(
                model_template, agent_struct['worlds'], target_index=1)
            _apply_snapshot_to_model(pre_model, series, pre_time)
            _apply_snapshot_to_model(post_model, series, post_time)

            pre_world = np.asarray(pre_worlds[pre_index])
            post_world = np.asarray(post_worlds[post_index])
            # Use the union of free-space masks to retain locations that are
            # navigable in either world, increasing the number of comparable
            # positions across environment switches.
            joint_mask = (pre_world <= 0) | (post_world <= 0)

            return _safe_correlation(
                _directional_mean_value_function(pre_model)[joint_mask],
                _directional_mean_value_function(post_model)[joint_mask],
            )
        except (ValueError, KeyError, IndexError, TypeError):
            return np.nan

    unlesioned = seed_entry.get('unlesioned')
    lesioned = seed_entry.get('lesionLEC')

    unlesioned_series = _series_for_agent(unlesioned)
    lesioned_series = _series_for_agent(lesioned)

    unlesioned_corr = (
        _correlate_snapshots(unlesioned, unlesioned_series)
        if unlesioned is not None and unlesioned_series is not None
        else np.nan
    )
    lesioned_corr = (
        _correlate_snapshots(lesioned, lesioned_series)
        if lesioned is not None and lesioned_series is not None
        else np.nan
    )

    return unlesioned_corr, lesioned_corr


def _summarise_value_function_correlations(struct_all_seeds, model_template, env_switch_every, delay):
    if not struct_all_seeds or model_template is None:
        return None

    struct_all_seeds = remove_empty_dicts(struct_all_seeds)
    if not struct_all_seeds:
        return None

    switch_every = env_switch_every or 0
    pre_time = max(switch_every - 10, 0)
    post_time = switch_every + delay

    unlesioned_values = []
    lesioned_values = []
    unlesioned_seeds = []
    lesioned_seeds = []

    for seed_key, seed_entry in struct_all_seeds.items():
        un_corr, les_corr = _compute_seed_value_function_correlations(
            seed_entry, model_template, pre_time, post_time)
        if np.isfinite(un_corr):
            unlesioned_values.append(float(un_corr))
            unlesioned_seeds.append(str(seed_key))
        if np.isfinite(les_corr):
            lesioned_values.append(float(les_corr))
            lesioned_seeds.append(str(seed_key))

    if not unlesioned_values and not lesioned_values:
        return None

    return {
        'unlesioned': np.array(unlesioned_values, dtype=float),
        'lesioned': np.array(lesioned_values, dtype=float),
        'unlesioned_seeds': unlesioned_seeds,
        'lesioned_seeds': lesioned_seeds,
    }


def _render_value_correlation_figure(summary, path, stem, save_combined):
    unlesioned_values = summary['unlesioned']
    lesioned_values = summary['lesioned']

    fig, ax = plt.subplots(figsize=(5, 4))

    means = [
        np.mean(unlesioned_values) if len(unlesioned_values) else np.nan,
        np.mean(lesioned_values) if len(lesioned_values) else np.nan,
    ]

    x_positions = np.array([0, 1])
    colours = ['#4C72B0', '#C44E52']
    labels = [
        f"Unlesioned\n(n={len(unlesioned_values)})",
        f"Lesioned\n(n={len(lesioned_values)})",
    ]

    ax.scatter(x_positions, means, color=colours, s=120, zorder=3)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Correlation (r)')
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color='0.4', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_title('Pre/post value correlation by agent type')

    fig.tight_layout()
    axes_map = {'value_function_correlation': (ax,)}

    if save_combined:
        import matplotlib as mpl

        with mpl.rc_context({"svg.fonttype": "path"}):
            fig.savefig(path.joinpath(f"{stem}.svg"))
            fig.savefig(path.joinpath(f"{stem}.pdf"), dpi=2000)
        plt.close(fig)
        return []

    return [(stem, fig, axes_map)]




def generate_value_fig(model=None, struct_single_seed=None, struct_all_seeds=None, sigma=0.1, path=None, fsize_label=12,
                       env_switch_every=1000, delay=1, models=None, seednum=None, save_combined=True):
    """Generate the value-function summary figure.

    Parameters
    ----------
    save_combined : bool, optional
        When ``True`` (default) the figure is written to disk as before. When
        ``False`` the caller receives ``(fig, axes_map)`` to enable per-panel
        exports.
    """

    if struct_single_seed is None:
        return

    if models:
        model_unles, model_les = models

    directions = ["\u2191", "\u2192", "\u2193", "\u2190"]
    if path:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
    else:
        path = pathlib.Path('.')

    if struct_single_seed is not None:
        struct_single_seed = remove_empty_dicts(struct_single_seed)
    struct_all_seeds = remove_empty_dicts(struct_all_seeds)

    correlation_figures = []
    if struct_all_seeds is not None and model is not None:
        correlation_summary = _summarise_value_function_correlations(
            struct_all_seeds, model, env_switch_every, delay
        )
        if correlation_summary is not None:
            stem_correlation = f"value_functions_{delay}_correlation"
            correlation_figures = _render_value_correlation_figure(
                correlation_summary, path, stem_correlation, save_combined
            )

    def _has_sr_snapshots(struct):
        try:
            seed_key = next(iter(struct))
        except (StopIteration, TypeError):
            return False
        per_seed = struct.get(seed_key)
        if not isinstance(per_seed, dict):
            return False
        required_params = ("ego_SR.SR_ss", "allo_SR.SR_ss", "weight")
        for prefix in ("unlesioned", "lesionLEC"):
            prefix_data = per_seed.get(prefix)
            if not isinstance(prefix_data, dict):
                return False
            for param in required_params:
                values = prefix_data.get(param)
                if not values:
                    return False
        return True

    if not struct_single_seed or not _has_sr_snapshots(struct_single_seed):
        return None, {}, correlation_figures

    fig = plt.figure(figsize=(10, 5))
    grids = fig.add_gridspec(nrows=4, ncols=5, wspace=0.1,
                             hspace=0.1)
    panel_axes = {}

    def register_panel(name, ax):
        panel_axes.setdefault(name, []).append(ax)
    
    try:
        primary_key   = list(struct_all_seeds.keys())[0]
        worlds_unles  = struct_single_seed[primary_key]['unlesioned']['worlds']
        worlds_les    = struct_single_seed[primary_key]['lesionLEC']['worlds']
    except (KeyError, IndexError):
        # Fallback to whatever key is available in struct_single_seed
        primary_key   = next(iter(struct_single_seed))
        worlds_unles  = struct_single_seed[primary_key]['unlesioned']['worlds']
        worlds_les    = struct_single_seed[primary_key]['lesionLEC']['worlds']
    
    
    expectation_grid0 = grids[0, :].subgridspec(
        1, 4, width_ratios=(1, 1, 1, 1), wspace=0.001, hspace=0.01)
    diff_grid0 = grids[1, :].subgridspec(
        1, 4, width_ratios=(1, 1, 1, 1), wspace=0.001, hspace=0.01)
    expectation_grid1 = grids[2, :].subgridspec(
        1, 4, width_ratios=(1, 1, 1, 1), wspace=0.001, hspace=0.01)
    diff_grid1 = grids[3, :].subgridspec(
        1, 4, width_ratios=(1, 1, 1, 1), wspace=0.001, hspace=0.01)
        
    if not models:
        model1 = deepcopy(model)

        # world = struct_single_seed[list(struct_single_seed.keys())[
        #     0]]['unlesioned']['worlds'][0]
        # model1.switch_world(world)
        model_unlesioned = deepcopy(model1)

        model_lesioned = deepcopy(model1)
    else:
        model_unlesioned, model_lesioned = models
    
    for world in worlds_unles:
        model_unlesioned.switch_world(world)
    model_unlesioned.switch_world(worlds_unles[0])
    for world in worlds_les:
        model_lesioned.switch_world(world)
    model_lesioned.switch_world(worlds_les[0])

    lesioned_allo_srs_y, lesioned_allo_srs_x = get_parameter_values('allo_SR.SR_ss', struct_single_seed,
                                                                    prefix='lesionLEC')
    unlesioned_allo_srs_y, unlesioned_allo_srs_x = get_parameter_values('allo_SR.SR_ss', struct_single_seed,
                                                                        prefix='unlesioned')

    lesioned_ego_srs_y, lesioned_ego_srs_x = get_parameter_values(
        'ego_SR.SR_ss', struct_single_seed, prefix='lesionLEC')

    unlesioned_ego_srs_y, unlesioned_ego_srs_x = get_parameter_values('ego_SR.SR_ss', struct_single_seed,
                                                                      prefix='unlesioned')

    lesioned_weights_y, lesioned_weights_x = get_parameter_values(
        'weight', struct_single_seed, prefix='lesionLEC')
    
    unlesioned_weights_y, unlesioned_weights_x = get_parameter_values(
        'weight', struct_single_seed, prefix='unlesioned')
    
    assert unlesioned_allo_srs_x == unlesioned_ego_srs_x == unlesioned_weights_x
    assert lesioned_allo_srs_x == lesioned_ego_srs_x == lesioned_weights_x
    

    index = bisect.bisect_left(sorted(unlesioned_allo_srs_x[0]), env_switch_every - 10)

    model_unlesioned.ego_SR.SR_ss = sorted(zip(unlesioned_ego_srs_x[0], unlesioned_ego_srs_y[0]))[index][1]
    
    model_unlesioned.ego_SR.SR_sas = model_unlesioned.ego_SR.SR_ss[:,np.newaxis,:].repeat(4, axis=1)
    
    model_unlesioned.allo_SR.SR_ss = sorted(zip(unlesioned_allo_srs_x[0], unlesioned_allo_srs_y[0]))[index][1]
    model_unlesioned.allo_SR.SR_sas = model_unlesioned.allo_SR.SR_ss[:,np.newaxis,:].repeat(4, axis=1)
    
    model_unlesioned.weight = sorted(zip(unlesioned_weights_x[0], unlesioned_weights_y[0]))[index][1]

    model_lesioned.ego_SR.SR_ss = sorted(zip(lesioned_ego_srs_x[0], lesioned_ego_srs_y[0]))[index][1]
    model_lesioned.ego_SR.SR_sas = model_lesioned.ego_SR.SR_ss[:,np.newaxis,:].repeat(4, axis=1)
    
    model_lesioned.allo_SR.SR_ss = sorted(zip(lesioned_allo_srs_x[0], lesioned_allo_srs_y[0]))[index][1]
    model_lesioned.allo_SR.SR_sas = model_lesioned.allo_SR.SR_ss[:,np.newaxis,:].repeat(4, axis=1)
    
    model_lesioned.weight = sorted(zip(lesioned_weights_x[0], lesioned_weights_y[0]))[index][1]
    
    unlesioned_state_values, ego_unlesioned_state_values, allo_unlesioned_state_values, vmin, vmax, egomin, egomax, amin, amax = get_value_functions(
        model_unlesioned,
        split=True)
    lesioned_state_values, _, _, lmin, lmax, _, _, _, _ = get_value_functions(
        model_lesioned, split=True)

    mu_ego_unlesioned = np.mean(ego_unlesioned_state_values, axis=2)
    mu_unlesioned = np.mean(unlesioned_state_values, axis=2)

    unlesioned_state_values -= mu_unlesioned[:, :, np.newaxis]
    ego_unlesioned_state_values -= mu_ego_unlesioned[:, :, np.newaxis]
    
    
    # for x in range(model.env.size):
    #     for y in range(model.env.size):
    #         sum_unlesioned = np.sum(unlesioned_state_values[x, y, :])
    #         sum_ego_unlesioned = np.sum(ego_unlesioned_state_values[x, y, :])
    #         if sum_unlesioned != 0:
    #             unlesioned_state_values[x, y, :] /= sum_unlesioned
    #             ego_unlesioned_state_values[x, y, :] /= sum_ego_unlesioned
    #         else:
    #             unlesioned_state_values[x, y, :] = 0
    #             ego_unlesioned_state_values[x, y, :] = 0
                

    egomin, egomax = np.min(ego_unlesioned_state_values), np.max(
        ego_unlesioned_state_values)
    vmin, vmax = np.min(unlesioned_state_values), np.max(
        unlesioned_state_values)
    mu_ego_min, mu_ego_max = np.min(
        mu_ego_unlesioned), np.max(mu_ego_unlesioned)
    mu_min, mu_max = np.min(mu_unlesioned), np.max(mu_unlesioned)

    # visualise_Q_values(path, model_unlesioned, ego_state, allo_state,
    #                    directions=directions, actions=["\u2191", "\u27f3", "\u2193", "\u27f2"])

    # unlesioned_state_values, vmin, vmax = get_value_functions(model_unlesioned,
    #                                                           split=False)
    # #TODO: fix this -- breaks here wrong phi and weight dimensions
    # lesioned_state_values, lmin, lmax = get_value_functions(
    #     model_lesioned, split=False)
    # for d in range(4):
    #     ax = fig.add_subplot(inner_grid[0, d])
    #     generate_value_plot(ax, unlesioned_state_values[:, :, d], vmin, vmax, direction=directions[d],
    #                         colorbar=True, labelsize='xx-small')
    
    mu_unlesioned[world == -1] = np.mean(mu_unlesioned)
    masked_array = np.ma.array(mu_unlesioned, mask=world == -1)
    cmap = plt.cm.plasma
    cmap.set_bad(color='green')
    
    mu_max, mu_min = np.max(mu_unlesioned), np.min(mu_unlesioned)
    
    ax = fig.add_subplot(expectation_grid0[0])
    im = ax.imshow(
        masked_array,
        vmin=mu_min, vmax=mu_max, cmap=cmap)

    # ax.title.set_text("E_d V(s)")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = colorbar(im)
    register_panel('pre_mean_value', ax)
    if cbar is not None:
        register_panel('pre_mean_value', cbar.ax)
    
    # normalised_difference_values = unlesioned_state_values / (np.sum(unlesioned_state_values, axis=2)[:, :, np.newaxis] + 1e-10)
    
    normalised_difference_values = unlesioned_state_values
    normalised_difference_values[world == -1] = np.mean(normalised_difference_values)
    
    # normalised_difference_values = unlesioned_state_values / (np.max(unlesioned_state_values, axis=2)[:, :, np.newaxis] - np.min(unlesioned_state_values, axis=2)[:, :, np.newaxis] + 1e-10)


    vmax = np.max([np.max(normalised_difference_values), np.abs(np.min(normalised_difference_values))])
    vmin = -vmax
    

    for d in range(4):
        
        ax = fig.add_subplot(diff_grid0[d])
        masked_array = np.ma.array(normalised_difference_values[:, :, d], mask=world == -1)
        cmap = plt.cm.bwr
        cmap.set_bad(color='green')
        
        im = ax.imshow(
            masked_array,
            vmin=vmin, vmax=vmax, cmap=cmap)
        # ax.title.set_text(f"V(s|d={directions[d]}) - E_d V(s)")
        ax.set_xticks([])
        ax.set_yticks([])
        register_panel(f'pre_value_direction_{directions[d]}', ax)
    cbar = colorbar(im)
    if cbar is not None:
        register_panel('pre_value_direction_colorbar', cbar.ax)
    
    mu_ego_unlesioned[world == -1] = np.mean(mu_ego_unlesioned)
    masked_array = np.ma.array(mu_ego_unlesioned, mask=world == -1)
    cmap = plt.cm.plasma
    cmap.set_bad(color='green')
    
    mu_ego_max = np.max(mu_ego_unlesioned)
    mu_ego_min = np.min(mu_ego_unlesioned)

    ax = fig.add_subplot(expectation_grid0[1])
    im = ax.imshow(
        masked_array,
        vmin=mu_ego_min, vmax=mu_ego_max, cmap=cmap)

    # ax.title.set_text("E_d V_ego(s)")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = colorbar(im)
    register_panel('pre_mean_value_ego', ax)
    if cbar is not None:
        register_panel('pre_mean_value_ego', cbar.ax)
    
    allo_unlesioned_state_values[:,:,0][world == -1] = np.mean(allo_unlesioned_state_values[:,:,0])
    masked_array = np.ma.array(allo_unlesioned_state_values[:, :, 0], mask=world == -1)
    cmap = plt.cm.plasma
    cmap.set_bad(color='green')
    
    amin, amax = np.min(allo_unlesioned_state_values[:,:,0]), np.max(
        allo_unlesioned_state_values[:,:,0])

    ax = fig.add_subplot(expectation_grid0[2])
    im = ax.imshow(
        masked_array,
        vmin=amin, vmax=amax, cmap=cmap)

    # ax.title.set_text("V_allo(s)")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = colorbar(im)
    register_panel('pre_allo_component', ax)
    if cbar is not None:
        register_panel('pre_allo_component', cbar.ax)
    
    lesioned_state_values[:, :, 0][world == -1] = np.mean(lesioned_state_values[:, :, 0])
    masked_array = np.ma.array(lesioned_state_values[:, :, 0], mask=world == -1)
    cmap = plt.cm.plasma
    cmap.set_bad(color='green')
    
    lmax, lmin = np.max(lesioned_state_values[:, :, 0]), np.min(
        lesioned_state_values[:, :, 0])
    
    
    ax = fig.add_subplot(expectation_grid0[3])
    im = ax.imshow(
        masked_array,
        vmin=lmin, vmax=lmax, cmap=cmap)

    # ax.title.set_text("V_les(s)")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = colorbar(im)
    register_panel('pre_lesioned_value', ax)
    if cbar is not None:
        register_panel('pre_lesioned_value', cbar.ax)

    model2 = deepcopy(model)

    world = struct_single_seed[list(struct_single_seed.keys())[
        0]]['unlesioned']['worlds'][1]
    model2.switch_world(world)
    model_unlesioned = deepcopy(model2)
    model_lesioned = deepcopy(model2)

    lesioned_allo_srs_y, lesioned_allo_srs_x = get_parameter_values('allo_SR.SR_ss', struct_single_seed,
                                                                    prefix='lesionLEC')
    unlesioned_allo_srs_y, unlesioned_allo_srs_x = get_parameter_values('allo_SR.SR_ss', struct_single_seed,
                                                                        prefix='unlesioned')
    lesioned_ego_srs_y, lesioned_ego_srs_x = get_parameter_values(
        'ego_SR.SR_ss', struct_single_seed, prefix='lesionLEC')
    unlesioned_ego_srs_y, unlesioned_ego_srs_x = get_parameter_values('ego_SR.SR_ss', struct_single_seed,
                                                                      prefix='unlesioned')

    lesioned_weights_y, lesioned_weights_x = get_parameter_values(
        'weight', struct_single_seed, prefix='lesionLEC')
    unlesioned_weights_y, unlesioned_weights_x = get_parameter_values(
        'weight', struct_single_seed, prefix='unlesioned')

    index = bisect.bisect_left(sorted(unlesioned_allo_srs_x[0]), env_switch_every + delay)

    model_unlesioned.ego_SR.SR_ss = sorted(zip(unlesioned_ego_srs_x[0], unlesioned_ego_srs_y[0]))[index][1]
    
    model_unlesioned.ego_SR.SR_sas = model_unlesioned.ego_SR.SR_ss[:,np.newaxis,:].repeat(4, axis=1)
    
    model_unlesioned.allo_SR.SR_ss = sorted(zip(unlesioned_allo_srs_x[0], unlesioned_allo_srs_y[0]))[index][1]
    model_unlesioned.allo_SR.SR_sas = model_unlesioned.allo_SR.SR_ss[:,np.newaxis,:].repeat(4, axis=1)
    
    model_unlesioned.weight = sorted(zip(unlesioned_weights_x[0], unlesioned_weights_y[0]))[index][1]

    model_lesioned.ego_SR.SR_ss = sorted(zip(lesioned_ego_srs_x[0], lesioned_ego_srs_y[0]))[index][1]
    model_lesioned.ego_SR.SR_sas = model_lesioned.ego_SR.SR_ss[:,np.newaxis,:].repeat(4, axis=1)
    
    model_lesioned.allo_SR.SR_ss = sorted(zip(lesioned_allo_srs_x[0], lesioned_allo_srs_y[0]))[index][1]
    model_lesioned.allo_SR.SR_sas = model_lesioned.allo_SR.SR_ss[:,np.newaxis,:].repeat(4, axis=1)
    
    model_lesioned.weight = sorted(zip(lesioned_weights_x[0], lesioned_weights_y[0]))[index][1]
       
    unlesioned_state_values, ego_unlesioned_state_values, allo_unlesioned_state_values, vmin, vmax, egomin, egomax, amin, amax = get_value_functions(
        model_unlesioned,
        split=True)

    mu_ego_unlesioned = np.mean(ego_unlesioned_state_values, axis=2)
    mu_unlesioned = np.mean(unlesioned_state_values, axis=2)
    
    lesioned_state_values, lmin, lmax = get_value_functions(
        model_lesioned, split=False)

    mu_ego_unlesioned = np.mean(ego_unlesioned_state_values, axis=2)
    mu_unlesioned = np.mean(unlesioned_state_values, axis=2)

    unlesioned_state_values -= mu_unlesioned[:, :, np.newaxis]
    ego_unlesioned_state_values -= mu_ego_unlesioned[:, :, np.newaxis]

    egomin, egomax = np.min(ego_unlesioned_state_values), np.max(
        ego_unlesioned_state_values)
    vmin, vmax = np.min(unlesioned_state_values), np.max(
        unlesioned_state_values)
    mu_ego_min, mu_ego_max = np.min(
        mu_ego_unlesioned), np.max(mu_ego_unlesioned)
    mu_min, mu_max = np.min(mu_unlesioned), np.max(mu_unlesioned)

    mu_unlesioned[world == -1] = np.mean(mu_unlesioned)
    masked_array = np.ma.array(mu_unlesioned, mask=world == -1)
    
    mu_max, mu_min = np.max(mu_unlesioned), np.min(mu_unlesioned)
    
    cmap = plt.cm.plasma
    cmap.set_bad(color='green')

    ax = fig.add_subplot(expectation_grid1[0])
    im = ax.imshow(
        masked_array,
        vmin=mu_min, vmax=mu_max, cmap=cmap)

    # ax.title.set_text("E_d V(s)")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = colorbar(im)
    register_panel('post_mean_value', ax)
    if cbar is not None:
        register_panel('post_mean_value', cbar.ax)
    
    # normalised_difference_values = unlesioned_state_values/(np.sum(unlesioned_state_values, axis=2)[:, :, np.newaxis]+1e-10)
    normalised_difference_values = unlesioned_state_values
    normalised_difference_values[world == -1] = np.mean(normalised_difference_values)
    # normalised_difference_values = unlesioned_state_values / (np.max(unlesioned_state_values, axis=2)[:, :, np.newaxis] - np.min(unlesioned_state_values, axis=2)[:, :, np.newaxis] + 1e-10)
    
    vmax = np.max([np.max(normalised_difference_values), np.abs(np.min(normalised_difference_values))])
    vmin = -vmax

    for d in range(4):
        masked_array = np.ma.array(normalised_difference_values[:, :, d], mask=world == -1)
        cmap = plt.cm.bwr
        cmap.set_bad(color='green')

        ax = fig.add_subplot(diff_grid1[d])
        im = ax.imshow(
            masked_array,
            vmin=vmin, vmax=vmax, cmap=cmap)
        # ax.title.set_text(f"V(s|d={directions[d]}) - E_d V(s)")
        ax.set_xticks([])
        ax.set_yticks([])
        register_panel(f'post_value_direction_{directions[d]}', ax)
    cbar = colorbar(im)
    if cbar is not None:
        register_panel('post_value_direction_colorbar', cbar.ax)
    

    ax = fig.add_subplot(expectation_grid1[3])

    lesioned_state_values[:, :, 0][world == -1] = np.mean(lesioned_state_values)
    masked_array = np.ma.array(lesioned_state_values[:, :, 0], mask=world == -1)
    cmap = plt.cm.plasma
    cmap.set_bad(color='green')

    lmax, lmin = np.max(lesioned_state_values[:, :, 0]), np.min(
        lesioned_state_values[:, :, 0])

    im = ax.imshow(
        masked_array,
        vmin=lmin, vmax=lmax, cmap=cmap)

    # ax.title.set_text("V_les(s)")
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = colorbar(im)
    register_panel('post_lesioned_value', ax)
    if cbar is not None:
        register_panel('post_lesioned_value', cbar.ax)
        
    amin, amax = np.min(allo_unlesioned_state_values), np.max(
        allo_unlesioned_state_values)

    ax = fig.add_subplot(expectation_grid1[2])

    allo_unlesioned_state_values[:,:,0][world == -1] = np.mean(allo_unlesioned_state_values)
    masked_array = np.ma.array(allo_unlesioned_state_values[:, :, 0], mask=world == -1)
    cmap = plt.cm.plasma
    cmap.set_bad(color='green')

    amax, amin = np.max(allo_unlesioned_state_values[:,:,0]), np.min(
        allo_unlesioned_state_values[:,:,0])


    im = ax.imshow(
        masked_array,
        vmin=amin, vmax=amax, cmap=cmap)

    # ax.title.set_text("V_allo(s)")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = colorbar(im)
    register_panel('post_allo_component', ax)
    if cbar is not None:
        register_panel('post_allo_component', cbar.ax)
    
    ax = fig.add_subplot(expectation_grid1[1])

    mu_ego_unlesioned[world == -1] = np.mean(mu_ego_unlesioned)
    masked_array = np.ma.array(mu_ego_unlesioned, mask=world == -1)
    cmap = plt.cm.plasma
    cmap.set_bad(color='green')

    mu_ego_max, mu_ego_min = np.max(mu_ego_unlesioned), np.min(mu_ego_unlesioned)


    im = ax.imshow(
        masked_array,
        vmin=mu_ego_min, vmax=mu_ego_max, cmap=cmap)

    # ax.title.set_text("E_d V_ego(s)")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = colorbar(im)
    register_panel('post_mean_value_ego', ax)
    if cbar is not None:
        register_panel('post_mean_value_ego', cbar.ax)
        
    # for d in range(4):
    #     ax = fig.add_subplot(diff_grid1[d])
    #     im = ax.imshow(
    #         unlesioned_state_values[:, :, d],
    #         vmin=vmin, vmax=vmax, cmap='bwr')
    #     ax.title.set_text(f"V(s|d={directions[d]} - E_d V(s)")
    #     ax.set_xticks([])
    #     ax.set_yticks([])

    #     # add dotted square outline pixel where reward is
    #     reward = np.where(world == -1)
    #     ax.add_patch(patches.Rectangle(
    #         (reward[1][0] - 1.5, reward[0][0] - 1.5), 3, 3, fill=False, edgecolor='green', lw=1))

    # colorbar(im)
    
    if save_combined:
        if seednum is None:
            stem = f"value_functions_{delay}"
        else:
            stem = f"value_functions_{delay}_{seednum}"
        fig.savefig(path.joinpath(f"{stem}.svg"))
        fig.savefig(path.joinpath(f"{stem}.pdf"), dpi=2000)
        plt.close(fig)
        return None

    return fig, {name: tuple(axes) for name, axes in panel_axes.items()}, correlation_figures

def generate_generalisation_plot(model=None, struct_single_seed=None, struct_all_seeds=None, sigma=1, path=None, fsize_label=12,
                                 env_switch_every=1000, delay=1, seednum=None, save_combined=True):
    """Generate the generalisation figure.

    Parameters
    ----------
    save_combined : bool, optional
        When ``True`` (default) the combined figure is saved to disk and the
        matplotlib figure is closed as in the original implementation. When
        ``False`` the function skips the save/close calls and instead returns
        a tuple ``(fig, axes_map)`` where ``axes_map`` maps descriptive panel
        names to the axes (or axes groups) that make up each subplot so they
        can be saved individually by the caller.
    """
    directions = ["\u2191", "\u2192", "\u2193", "\u2190"]
    if path:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
    else:
        path = pathlib.Path('.')

    fig = plt.figure(figsize=(20, 10))
    grids = fig.add_gridspec(nrows=3, ncols=6, wspace=0.5, hspace=0.5, height_ratios=[0.5, 1, 1],
                             width_ratios=[1, 1, 1, 1, 1, 1])

    if struct_single_seed is not None:
        struct_single_seed = remove_empty_dicts(struct_single_seed)
    struct_all_seeds = remove_empty_dicts(struct_all_seeds)

    seed_keys = list(struct_all_seeds.keys())

    def _resolve_seed_key(target_seed):
        for key in seed_keys:
            if key == target_seed or str(key) == str(target_seed):
                return key
        raise KeyError(f"Seed {target_seed} not found in structure")

    if seednum is not None:
        world_seed_key = _resolve_seed_key(seednum)
    else:
        world_seed_key = seed_keys[0]

    worlds = struct_all_seeds[world_seed_key]['unlesioned']['worlds']

    all_worlds = [struct_all_seeds[key]['unlesioned']['worlds'] for key in seed_keys]
    

    world = worlds[0]

    # A: Task schematic
    panel_axes = {}

    def register_panel(name, ax):
        panel_axes.setdefault(name, []).append(ax)

    task_axes = generate_task_plot(worlds, grids[0, :5], fig, return_axes=True)
    for idx, task_ax in enumerate(task_axes):
        register_panel(f'task_world_{idx}', task_ax)
    # B: Lesion Plot
    ax = fig.add_subplot(grids[1, :5])
    (y_les, y_les_sem, x_les), (y_un, y_un_sem, x_un) = get_lesion_values_(struct_all_seeds, sigma=sigma)
    
    N_les = y_les.shape[0]
    N_un = y_un.shape[0]
    
    
    
    generate_lesion_plot_(ax, inputs=[(y_les, y_les_sem, x_les), (y_un, y_un_sem, x_un)], labels=['allocentric','allocentric + egocentric'], env_switch_every=env_switch_every)
    register_panel('lesion_learning', ax)

    ax.set_ylabel(f'N unlesioned = {N_un}, N lesioned = {N_les}', fontsize='xx-small')

    ax2 = fig.add_subplot(grids[2, :5])
    chunk_size_target = 25
    chunk_summary = _load_chunked_occupancy_summary(path, expected_chunk_size=chunk_size_target)
    plotted_chunked = False
    if chunk_summary is not None:
        chunk_data, chunk_size_value = chunk_summary
        plotted_chunked = _plot_chunked_occupancy(ax2, chunk_data, chunk_size_value, env_switch_every)

    if not plotted_chunked:
        ax2.cla()
        hole_times_unles_mean, hole_times_unles_sem, x_hole_unles = get_hole_times(
            struct_all_seeds, prefix='unlesioned', worlds=worlds, all_worlds=all_worlds, switch_every=env_switch_every)
        hole_times_les_mean, hole_times_les_sem, x_hole_les = get_hole_times(
            struct_all_seeds, prefix='lesionLEC', worlds=worlds, all_worlds=all_worlds, switch_every=env_switch_every)

        hole_times_unles_mean = gaussian_filter1d(hole_times_unles_mean, sigma=10)
        hole_times_les_mean = gaussian_filter1d(hole_times_les_mean, sigma=10)

        hole_times_unles_sem = gaussian_filter1d(hole_times_unles_sem, sigma=10)
        hole_times_les_sem = gaussian_filter1d(hole_times_les_sem, sigma=10)

        generate_lesion_plot_(
            ax2,
            inputs=[
                (hole_times_les_mean, hole_times_les_sem, x_hole_les[0]),
                (hole_times_unles_mean, hole_times_unles_sem, x_hole_unles[0]),
            ],
            labels=['allocentric', 'allocentric + egocentric'],
            env_switch_every=env_switch_every,
        )
        ax2.set_yscale("linear")
        ax2.set_ylabel('Proportion of time spent inside barrier', fontsize='xx-small')

    register_panel('barrier_time', ax2)

    # E path comparison after switch:

    inner_grid = grids[:, 5].subgridspec(2, 1, height_ratios=[1, 1])

    path_unlesioned, un_ep, seed_key_unles = get_path(
        struct_all_seeds,
        prefix='unlesioned',
        episode=env_switch_every + delay,
        path_start=None,
        seednum=seednum,
        return_seed_id=True)
    path_lesioned, les_ep, seed_key_les = get_path(
        struct_all_seeds,
        prefix='lesionLEC',
        episode=env_switch_every + delay,
        path_start=None,
        seednum=seednum,
        return_seed_id=True)

    def _get_world_for_episode_for_seed(seed_key, episode):
        seed_key = seed_key or world_seed_key
        worlds_for_seed = struct_all_seeds[seed_key]['unlesioned']['worlds']

        if env_switch_every:
            world_idx = episode // env_switch_every
        else:
            world_idx = 0

        if world_idx >= len(worlds_for_seed):
            world_idx = world_idx % len(worlds_for_seed)

        return worlds_for_seed[world_idx]

    world_for_paths_unles = _get_world_for_episode_for_seed(seed_key_unles, un_ep)
    world_for_paths_les = _get_world_for_episode_for_seed(seed_key_les, les_ep)

    heatmap_data_unlesioned = get_heatmap_data(path_unlesioned, world=world_for_paths_unles)
    heatmap_data_lesioned = get_heatmap_data(path_lesioned, world=world_for_paths_les)

    vmin = np.min([np.min(heatmap_data_unlesioned),
                  np.min(heatmap_data_lesioned)])
    vmax = np.max([np.max(heatmap_data_unlesioned),
                  np.max(heatmap_data_lesioned)])
    lims = [vmin, vmax]

    ax = fig.add_subplot(inner_grid[0, 0])

    plot_track(path_unlesioned, ax, world=world_for_paths_unles, color='r', label=f"Allo+Ego, Episode: {un_ep} ", lims=lims,
               fontsize='xx-small')
    register_panel('path_unlesioned', ax)

    ax = fig.add_subplot(inner_grid[1, 0])
    plot_track(path_lesioned, ax, world=world_for_paths_les, color='r', label=f'Allo, Episode: {les_ep}', lims=lims,
               fontsize='xx-small')
    register_panel('path_lesioned', ax)
    
    if save_combined:
        if seednum is None:
            stem = f'generalisation_{sigma}_{env_switch_every}_{delay}'
        else:
            stem = f'generalisation_{sigma}_{env_switch_every}_{delay}_{seednum}'
        fig.savefig(path.joinpath(f'{stem}.png'), dpi=1200)
        fig.savefig(path.joinpath(f'{stem}.svg'), format='svg')
        plt.close(fig)
        return None

    return fig, {name: tuple(axes) for name, axes in panel_axes.items()}


def generate_fig3__(model=None, struct_single_seed=None, struct_all_seeds=None, sigma=1, path=None, fsize_label=12,
                    env_switch_every=1000, delay=1, save_combined=True):
    """Generate the combined lesion/value figure used for Figure 3."""
    directions = ["\u2191", "\u2192", "\u2193", "\u2190"]
    if path:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
    else:
        path = pathlib.Path('.')

    fig = plt.figure(figsize=(10, 5))
    grids = fig.add_gridspec(nrows=3, ncols=6, wspace=0.5, hspace=0.5, height_ratios=[0.7, 1, 1],
                             width_ratios=[1, 1, 1, 1, 1, 1])
    panel_axes = {}

    def register_panel(name, ax):
        panel_axes.setdefault(name, []).append(ax)

    struct_single_seed = remove_empty_dicts(struct_single_seed)
    struct_all_seeds = remove_empty_dicts(struct_all_seeds)

    worlds = struct_all_seeds[list(struct_all_seeds.keys())[
        0]]['unlesioned']['worlds']

    # A: Task schematic
    task_axes = generate_task_plot(worlds, grids[0, :5], fig, return_axes=True)
    for idx, task_ax in enumerate(task_axes):
        register_panel(f'task_world_{idx}', task_ax)
    # B: Lesion Plot
    ax = fig.add_subplot(grids[1, :5])
    generate_lesion_plot_(ax, inputs = get_lesion_values(struct_all_seeds, sigma=sigma), labels=['allocentric', 'allocentric + egocentric'],env_switch_every=env_switch_every)
    register_panel('lesion_learning', ax)

    model2 = deepcopy(model)

    world = struct_single_seed[list(struct_single_seed.keys())[
        0]]['unlesioned']['worlds'][1]
    model2.switch_world(world)
    model_unlesioned = deepcopy(model2)
    model_lesioned = deepcopy(model2)

    lesioned_allo_srs_y, lesioned_allo_srs_x = get_parameter_values('allo_SR.SR_ss', struct_single_seed,
                                                                    prefix='lesionLEC')
    unlesioned_allo_srs_y, unlesioned_allo_srs_x = get_parameter_values('allo_SR.SR_ss', struct_single_seed,
                                                                        prefix='unlesioned')
    lesioned_ego_srs_y, lesioned_ego_srs_x = get_parameter_values(
        'ego_SR.SR_ss', struct_single_seed, prefix='lesionLEC')
    unlesioned_ego_srs_y, unlesioned_ego_srs_x = get_parameter_values('ego_SR.SR_ss', struct_single_seed,
                                                                      prefix='unlesioned')

    lesioned_weights_y, lesioned_weights_x = get_parameter_values(
        'weight', struct_single_seed, prefix='lesionLEC')
    unlesioned_weights_y, unlesioned_weights_x = get_parameter_values(
        'weight', struct_single_seed, prefix='unlesioned')

    import bisect
    index = bisect.bisect_left(
        unlesioned_allo_srs_x[0], env_switch_every + delay)

    model_unlesioned.ego_SR.SR_ss = unlesioned_ego_srs_y[0][index]
    model_unlesioned.allo_SR.SR_ss = unlesioned_allo_srs_y[0][index]
    model_unlesioned.weight = unlesioned_weights_y[0][index]

    model_lesioned.ego_SR.SR_ss = lesioned_ego_srs_y[0][index]
    model_lesioned.allo_SR.SR_ss = lesioned_allo_srs_y[0][index]
    model_lesioned.weight = lesioned_weights_y[0][index]
    unlesioned_state_values, vmin, vmax = get_value_functions(model_unlesioned,
                                                              split=False)
    lesioned_state_values, lmin, lmax = get_value_functions(
        model_lesioned, split=False)
    for d in range(4):
        ax = fig.add_subplot(grids[2, d])
        _, cax = generate_value_plot(ax, unlesioned_state_values[:, :, d], vmin, vmax, direction=directions[d],
                                     colorbar=True, labelsize='xx-small')
        register_panel(f'unlesioned_value_{directions[d]}', ax)
        if cax is not None:
            register_panel(f'unlesioned_value_{directions[d]}', cax)

    ax = fig.add_subplot(grids[2, 4])
    _, cax = generate_value_plot(ax, lesioned_state_values[:, :, 0], lmin, lmax, direction='all', colorbar=True,
                                 labelsize='xx-small')
    register_panel('lesioned_value_all', ax)
    if cax is not None:
        register_panel('lesioned_value_all', cax)

    # E path comparison after switch:

    inner_grid = grids[:, 5].subgridspec(2, 1, height_ratios=[1, 1])

    path_unlesioned, un_ep = get_path(struct_all_seeds, prefix='unlesioned',
                                      episode=env_switch_every + delay, path_start=path_start)
    path_lesioned, les_ep = get_path(struct_all_seeds, prefix='lesionLEC',
                                     episode=env_switch_every + delay, path_start=path_start)
    heatmap_data_unlesioned = get_heatmap_data(path_unlesioned, world=world)
    heatmap_data_lesioned = get_heatmap_data(path_lesioned, world=world)
    # larger_verts_unles = get_path_verts_multiple(struct_single_seed, prefix='unlesioned',
    #                                              ep_start=env_switch_every + delay,
    #                                              ep_end=env_switch_every + delay + heatmap_path_num,
    #                                              world=world,
    #                                              path_start=path_start)
    #
    # larger_verts_les = get_path_verts_multiple(struct_single_seed, prefix='lesionLEC',
    #                                            ep_start=env_switch_every + delay,
    #                                            ep_end=env_switch_every + delay + heatmap_path_num, world=world,
    #                                            path_start=path_start)
    #
    # larger_heatmap_les_data = get_heatmap_data(larger_verts_les, world)
    # larger_heatmap_unlesioned_data = get_heatmap_data(larger_verts_unles, world)

    vmin = np.min([np.min(heatmap_data_unlesioned),
                  np.min(heatmap_data_lesioned)])
    vmax = np.max([np.max(heatmap_data_unlesioned),
                  np.max(heatmap_data_lesioned)])
    lims = [vmin, vmax]

    ax = fig.add_subplot(inner_grid[0, 0])

    plot_track(path_unlesioned, ax, world=world, color='r', label=f"Allo+Ego, Episode: {un_ep} ", lims=lims,
               fontsize='xx-small')
    register_panel('path_unlesioned', ax)

    ax = fig.add_subplot(inner_grid[1, 0])
    plot_track(path_lesioned, ax, world=world, color='r', label=f'Allo, Episode: {les_ep}', lims=lims,
               fontsize='xx-small')
    register_panel('path_lesioned', ax)

    plt.text(0.05, 0.9, "A", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.05, 0.7, "B", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.1, 0.32, "C", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.65, 0.32, "D", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.79, 0.81, "E", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.79, 0.4, "F", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)

    if save_combined:
        print("Saving fig 3")
        fig.savefig(path.joinpath(
            f'fig3_{sigma}_{env_switch_every}_{delay}.png'), dpi=1200)
        fig.savefig(path.joinpath(
            f'fig3_{sigma}_{env_switch_every}_{delay}.svg'), format='svg')
        plt.close(fig)
        return None

    return fig, {name: tuple(axes) for name, axes in panel_axes.items()}


def generate_fig2(model=None, struct_all_seeds=None, struct_single_seed=None, sigma=0.01, env_switch_every=1000, path=None,
                  ego_state=50, allo_state=200, fsize_label='small', save_separate=False):
    if path:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
    else:
        path = pathlib.Path('.')

    if path.joinpath(f'single_learning_data_{sigma}_{env_switch_every}.pkl').exists() and not struct_single_seed:
        with open(path.joinpath(f'single_learning_data_{sigma}_{env_switch_every}.pkl'), 'rb') as f:
            data = pickle.load(f)
            x, y, first_path, last_path, first_ep, last_ep, world = data['x'], data['y'], data['first_path'], data[
                'last_path'], data['first_ep'], data['last_ep'], data['world']
    else:
        if struct_single_seed:
            struct_single_seed = remove_empty_dicts(struct_single_seed)
            struct_all_seeds = remove_empty_dicts(struct_all_seeds)
            world = struct_single_seed[list(struct_single_seed.keys())[
                0]]['unlesioned']['worlds'][0]
            worlds = struct_all_seeds[list(struct_all_seeds.keys())[0]
                                    ]['unlesioned']['worlds']
            

            
            # TODO: sort worlds bit

            if struct_all_seeds:
                y, x = get_parameter_values(
                    'accuracies', struct_all_seeds, prefix='unlesioned')
                first_path, first_ep = get_path(
                    struct_all_seeds, prefix='unlesioned', episode=0, struct_all_seeds=True)
                last_path, last_ep = get_path(struct_all_seeds, prefix='unlesioned',
                                              episode=env_switch_every - 1, struct_all_seeds=True)

            else:
                y, x = get_parameter_values(
                    'accuracies', struct_single_seed, prefix='unlesioned')
                first_path, first_ep = get_path(
                    struct_single_seed, prefix='unlesioned', episode=0)
                last_path, last_ep = get_path(struct_single_seed, prefix='unlesioned',
                                              episode=env_switch_every)
                hole_times = get_hole_times(
                    struct_single_seed, prefix='unlesioned', worlds=worlds, switch_every=env_switch_every)
            data = {'x': x, 'y': y, 'first_path': first_path, 'last_path': last_path, 'first_ep': first_ep,
                    'last_ep': last_ep, 'world': world}
            with open(path.joinpath(f'single_learning_data_{sigma}_{env_switch_every}.pkl'), 'wb') as f:
                pickle.dump(data, f)
        else:
            raise NotImplementedError

    directions = ["\u2191", "\u2192", "\u2193", "\u2190"]

    # A
    fig = plt.figure(figsize=(10, 5))
    grids = fig.add_gridspec(nrows=5, ncols=3, hspace=0.75, wspace=0.75, height_ratios=[
                             2, 3, 3, 3, 3], width_ratios=[1, 1, 1.5])
    ax = fig.add_subplot(grids[0, :])
    y = np.array(y)
    x = np.array(x)
    mu_y = np.mean(y, 0)
    sigma_y = np.std(y, 0)
    x = np.mean(x, 0)
    mu_y = mu_y[x < env_switch_every]
    sigma_y = sigma_y[x < env_switch_every]
    x = x[x < env_switch_every]
    sorted_y = [y_ for _, y_ in sorted(zip(x, mu_y))]
    sorted_sigma_y = [y_ for _, y_ in sorted(zip(x, sigma_y))]
    x = sorted(x)
    y = gaussian_filter1d(sorted_y, sigma=sigma)
    ax.plot(x, y)
    ax.fill_between(x, y - sorted_sigma_y, y + sorted_sigma_y, alpha=0.2)
    ax.set_ylabel('Steps', fontsize='xx-small')
    ax.set_xlabel('Episode', fontsize='xx-small', labelpad=0, loc='right')
    ax.set_yscale("log")
    # fontsize
    ax.tick_params(
        labelsize='xx-small')
    ax.spines[['top', 'right']].set_visible(False)

    # B
    ax = fig.add_subplot(grids[1, 0])
    # y = gaussian_filter1d(y, sigma=sigma)
    plot_track(first_path, ax, world=world, color='r')

    # C
    ax = fig.add_subplot(grids[1, 1])
    plot_track(last_path, ax, world=world, color='r')

    ego_srs_y, ego_srs_x = get_parameter_values(
        'ego_SR.SR_ss', struct_single_seed, prefix='unlesioned')
    allo_srs_y, allo_srs_x = get_parameter_values(
        'allo_SR.SR_ss', struct_single_seed, prefix='unlesioned')
    model0 = get_certain_SR_model(
        ego_srs_y, ego_srs_x, allo_srs_y, allo_srs_x, model, 0)
    model1 = get_certain_SR_model(
        ego_srs_y, ego_srs_x, allo_srs_y, allo_srs_x, model, env_switch_every - 1)

    # D
    add_allocentric_SR_grid(grids[1, 2], fig, model0, allo_state, ego=False,
                            colorbar=True)
    add_allocentric_SR_grid(grids[2, :-1], fig, model0, ego_state, ego=True)

    add_egocentric_SR_grid(grids[2:-1, -1], fig, model0, ego_state)

    if save_separate:
        fig2, ax2 = plt.subplots(
            1, 1, figsize=(2, 5))

        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.axis("off")

        grids2 = fig2.add_gridspec(nrows=1, ncols=1)
        add_allocentric_SR_sas_grid(
            grids2[0, 0], fig2, model0, allo_state, ego=False)
        fig2.savefig(path.joinpath(f'allocentric_sas.svg'))
        plt.close(fig2)

        fig3, ax3 = plt.subplots(
            1, 1, figsize=(10, 10))

        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.axis("off")

        grids3 = fig3.add_gridspec(nrows=1, ncols=1)
        add_egocentric_SR_sas_grid(grids3[0, 0], fig3, model0, ego_state)
        fig3.savefig(path.joinpath(f'egocentric_sas.svg'))
        plt.close(fig3)

    # # E: Value plots

    # inner_grid = grids[3, :-1].subgridspec(
    #     1, 5, width_ratios=(1, 1, 1, 1, 0.03), wspace=0.01, hspace=0.01)
    # inner_grid1 = grids[4, :-1].subgridspec(
    #     1, 5, width_ratios=(1, 1, 1, 1, 0.03), wspace=0.01, hspace=0.01)

    # model1 = deepcopy(model)

    # world = struct_single_seed[list(struct_single_seed.keys())[
    #     0]]['unlesioned']['worlds'][0]
    # model1.switch_world(world)
    # model_unlesioned = deepcopy(model1)

    # model_lesioned = deepcopy(model1)

    # lesioned_allo_srs_y, lesioned_allo_srs_x = get_parameter_values('allo_SR.SR_ss', struct_single_seed,
    #                                                                 prefix='lesionLEC')
    # unlesioned_allo_srs_y, unlesioned_allo_srs_x = get_parameter_values('allo_SR.SR_ss', struct_single_seed,
    #                                                                     prefix='unlesioned')
    # lesioned_ego_srs_y, lesioned_ego_srs_x = get_parameter_values(
    #     'ego_SR.SR_ss', struct_single_seed, prefix='lesionLEC')
    # unlesioned_ego_srs_y, unlesioned_ego_srs_x = get_parameter_values('ego_SR.SR_ss', struct_single_seed,
    #                                                                   prefix='unlesioned')

    # lesioned_weights_y, lesioned_weights_x = get_parameter_values(
    #     'weight', struct_single_seed, prefix='lesionLEC')
    # unlesioned_weights_y, unlesioned_weights_x = get_parameter_values(
    #     'weight', struct_single_seed, prefix='unlesioned')

    # import bisect
    # index = bisect.bisect_left(unlesioned_allo_srs_x[0], env_switch_every - 1)

    # model_unlesioned.ego_SR.SR_ss = unlesioned_ego_srs_y[0][index]
    # model_unlesioned.allo_SR.SR_ss = unlesioned_allo_srs_y[0][index]
    # model_unlesioned.weight = unlesioned_weights_y[0][index]

    # model_lesioned.ego_SR.SR_ss = lesioned_ego_srs_y[0][index]
    # model_lesioned.allo_SR.SR_ss = lesioned_allo_srs_y[0][index]
    # model_lesioned.weight = lesioned_weights_y[0][index]
    # unlesioned_state_values, ego_unlesioned_state_values, allo_unlesioned_state_values, vmin, vmax, egomin, egomax, amin, amax = get_value_functions(
    #     model_unlesioned,
    #     split=True)

    # visualise_Q_values_new(path, model_unlesioned, ego_state, allo_state,
    #                    directions=directions, actions=["\u2191", "\u21b1", "\u21b6", "\u21b0"])

    # # unlesioned_state_values, vmin, vmax = get_value_functions(model_unlesioned,
    # #                                                           split=False)
    # lesioned_state_values, lmin, lmax = get_value_functions(
    #     model_lesioned, split=False)
    # # for d in range(4):
    # #     ax = fig.add_subplot(inner_grid[0, d])
    # #     generate_value_plot(ax, unlesioned_state_values[:, :, d], vmin, vmax, direction=directions[d],
    # #                         colorbar=True, labelsize='xx-small')

    # for d in range(4):
    #     ax = fig.add_subplot(inner_grid[0, d])
    #     im = ax.imshow(
    #         unlesioned_state_values[:, :, d],
    #         vmin=vmin, vmax=vmax, cmap='plasma')
    #     ax.title.set_text(f"{directions[d]}")
    #     ax.set_xticks([])
    #     ax.set_yticks([])

    # cbar_ax = fig.add_subplot(inner_grid[0, 4])
    # plt.colorbar(im, cax=cbar_ax)

    # for d in range(4):
    #     ax = fig.add_subplot(inner_grid1[0, d])
    #     im = ax.imshow(
    #         ego_unlesioned_state_values[:, :, d],
    #         vmin=egomin, vmax=egomax, cmap='plasma')
    #     ax.title.set_text(f"{directions[d]}")
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    # cbar_ax = fig.add_subplot(inner_grid1[0, 4])
    # plt.colorbar(im, cax=cbar_ax)

    # ax = fig.add_subplot(grids[4, -1])
    # im = ax.imshow(
    #     allo_unlesioned_state_values[:, :, 0],
    #     vmin=amin, vmax=amax, cmap='plasma')
    # ax.title.set_text("All")
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.colorbar(im, ax=ax)

    # E
    # add_allocentric_SR_grid(grids[2, 1], fig, model1, allo_state, ego=False,
    #                         colorbar=True)
    # add_egocentric_SR_grid(grids[3, 1], fig, model1, ego_state, coords=[-0.7, 0.3])

    # Add panels
    plt.text(0.05, 0.9, "A", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.15, 0.75, "B", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.41, 0.75, "C", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.75, 0.75, "D", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.1, 0.58, "E", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.65, 0.58, "F", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.1, 0.4, "G", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.1, 0.22, "H", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.75, 0.22, "I", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)

    # plt.text(0.65, 0.5, "G", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    # plt.text(0.65, 0.32, "H", ha="left", va="top", transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)

    print('Saving fig2')
    plt.savefig(path / 'fig2.png', dpi=1200)
    plt.savefig(path / 'fig2.svg', format='svg')


def generate_task_plot(worlds, grid, fig, return_axes=False, **kwargs):
    # for list of worlds in structure, create a plot of the worlds next to eachother with arrows in between
    # to indicate the transition between worlds and arrow from end to beginning

    # **kwargs: left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    grids = GridSpecFromSubplotSpec(1, 5, subplot_spec=grid, **kwargs)

    # create a line of 5 subplots

    sizes = [world.shape[0] for world in worlds]
    size = np.max(sizes)

    # Use a fixed normalization across all panels to match the schematic
    vmin = min(np.min(w) for w in worlds)
    vmax = max(np.max(w) for w in worlds)

    axes = []
    for i in range(5):
        ax = fig.add_subplot(grids[0, i])
        world = worlds[i % len(worlds)]
        world = np.pad(world, ((0, size - world.shape[0]), (0, size - world.shape[1])), mode='constant',
                       constant_values=0)
        masked_array = np.ma.masked_where(world == -1, world)
        cmap = mcm.Greys
        cmap.set_bad(color='green')

        import matplotlib.font_manager as fm

        fontprops = fm.FontProperties(size=8)

        ax.imshow(masked_array, cmap=cmap, vmin=vmin, vmax=vmax)

        # scalebar = AnchoredSizeBar(ax.transData,
        #                            10, '10', 'center left',
        #                            pad=0.1,
        #                            color='y',
        #                            frameon=False,
        #                            size_vertical=0,
        #                            fontproperties=fontprops)
        #
        # ax.add_artist(scalebar)
        plt.xticks([])
        plt.yticks([])
        # make border dotted
        ax.spines['top'].set_linestyle('dotted')
        ax.spines['right'].set_linestyle('dotted')
        ax.spines['bottom'].set_linestyle('dotted')
        ax.spines['left'].set_linestyle('dotted')

        axes.append(ax)

        if i != 4:
            plt.annotate('', xy=(1.2, 0.5), xycoords='axes fraction', xytext=(1., 0.5), textcoords='axes fraction',
                         arrowprops=dict(arrowstyle="->, head_width=0.3, head_length=0.3", linewidth=1, color='r',
                                         connectionstyle="arc3"))

    if return_axes:
        return axes


def save_panel_crops(fig, axes_map, output_dir, stem, formats=("png", "svg"), pad_fraction=0.02):
    """Save individual panel crops from a multi-axis figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure that contains the panels.
    axes_map : Mapping[str, Sequence[matplotlib.axes.Axes]]
        Mapping of panel names to the axes (or axes groups) that should be
        cropped together.
    output_dir : Union[str, pathlib.Path]
        Directory to which the cropped images will be written.
    stem : str
        Base filename stem used for each panel (``{stem}_{panel}.ext``).
    formats : Iterable[str], optional
        File formats to emit for each panel.
    pad_fraction : float, optional
        Fractional padding applied to the bounding box on each side.
    """

    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    canvas = fig.canvas
    canvas.draw()
    renderer = canvas.get_renderer()

    for panel, axes in axes_map.items():
        if axes is None:
            continue
        if not isinstance(axes, (list, tuple)):
            axes_seq = [axes]
        else:
            axes_seq = [ax for ax in axes if ax is not None]
        if not axes_seq:
            continue

        bboxes = []
        for ax in axes_seq:
            bbox = ax.get_tightbbox(renderer)
            # ``get_tightbbox`` can occasionally return NaNs or degenerate
            # boxes (for example when a colour bar artist has not been fully
            # initialised yet). In those cases fall back to the less strict
            # window extent so that we still capture the drawn content.
            if bbox.width <= 0 or bbox.height <= 0 or not np.isfinite(bbox.bounds).all():
                bbox = ax.get_window_extent(renderer)
            if bbox.width <= 0 or bbox.height <= 0 or not np.isfinite(bbox.bounds).all():
                continue
            bboxes.append(bbox)
        if not bboxes:
            continue

        bbox = Bbox.union(bboxes)
        if pad_fraction:
            bbox = bbox.expanded(1 + pad_fraction, 1 + pad_fraction)
        bbox_inches = TransformedBbox(bbox, fig.dpi_scale_trans.inverted())

        for fmt in formats:
            fig.savefig(output_path / f"{stem}_{panel}.{fmt}", bbox_inches=bbox_inches)


def generate_cosyne_fig(structure, model, path=None, ego_state=76):
    print("Generating Cosyne Figure")

    if path:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

    directions = ["\u2191", "\u2192", "\u2193", "\u2190"]

    self = model.env

    fig = plt.figure(figsize=(24 * cm, 10 * cm))  # create figure

    x = 3
    y = 4
    d = 2
    grids = fig.add_gridspec(nrows=2, ncols=1, left=0,
                             right=0.15, bottom=0, top=1.0, hspace=-0.05)

    # panel A: maze schematic

    ax = fig.add_subplot(grids[0, 0])

    generate_schematic(ax, self, pos=(x, y, d))

    # panel B: allocentric state

    grids = fig.add_gridspec(nrows=2, ncols=1, left=0.2,
                             right=0.3, bottom=0.5, top=0.9, hspace=0.20)
    ax = fig.add_subplot(grids[0, 0])

    generate_allocentric_plot(ax, self, x, y)

    # panel C: egocentric state
    ax = fig.add_subplot(grids[1, 0])
    generate_egocentric_plot(ax, self, x, y, d)

    # ## panel D: full value function
    grids = fig.add_gridspec(nrows=3, ncols=4, left=0.35,
                             right=0.7, bottom=0.5, top=1.0, hspace=0.45)

    state_values, ego_state_values, allo_state_values, vmin, vmax, egomin, egomax, amin, amax = get_value_functions(
        model, split=True)

    for d in range(4):
        ax = fig.add_subplot(grids[0, d])
        generate_value_plot(
            ax, state_values[:, :, d], vmin, vmax, directions[d])

    # panel E: egocentric component

    for d in range(4):
        ax = fig.add_subplot(grids[1, d])
        generate_value_plot(ax, ego_state_values[:, :, d], egomin, egomax)

    # panel F: allocentric component

    ax = fig.add_subplot(grids[2, 0])
    generate_value_plot(ax, allo_state_values[:, :, 0], amin, amax)

    # panel G: lesion results

    ax = fig.add_subplot(grids[2, 1:4])

    (y_les, y_les_sem, x_les), (y_un, y_un_sem, x_un) = get_lesion_values_(structure, sigma=1)

    generate_lesion_plot_(ax, y_les, y_les_sem, y_un, y_un_sem, x_les, x_un,
                     env_switch_every=env_switch_every)

    add_SR_grid(fig, model, ego_state, directions, left=0.72, right=1.0,
                bottom=0.5, top=1.0, hspace=0.1)

    # add some panel labels
    plt.text(-0.04, 0.99, "A", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.18, 0.99, "B", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.18, 0.7, "C", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.34, 0.99, "D", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.34, 0.82, "E", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.34, 0.65, "F", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.427, 0.65, "G", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.705, 0.98, "H", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.705, 0.73, "I", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)

    # plt.text(0.17, 0.7, "C", ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label)
    # plt.text(0.17, 0.7, "C", ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label)

    # save figure
    plt.tight_layout
    if path:
        plt.savefig(path.joinpath("cosyne_fig.pdf"), bbox_inches="tight")
    else:
        plt.savefig("cosyne_fig.pdf", bbox_inches="tight")
    plt.close()


def generate_fig1(structure, model, path=None, save_separate=False):
    if path:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
    else:
        path = pathlib.Path('.')

    if path.joinpath("fig1_data.pkl").exists() and structure is None and model is None:
        with open(path.joinpath("fig1_data.pkl"), 'rb') as f:
            data = pickle.load(f)
            self, worlds = data['self'], data['worlds']
    else:
        self = model.env
        structure = remove_empty_dicts(structure)
        worlds = structure[list(structure.keys())[0]]['unlesioned']['worlds']
        with open(path.joinpath("fig1_data.pkl"), 'wb') as f:
            pickle.dump({'self': self, 'worlds': worlds}, f)

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(layout="constrained")

    gs = GridSpec(3, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 2])

    # ax4 = fig.add_subplot(gs[2, 0:3])
    grid4 = gs[2, 0:3]
    x = 3
    y = 4
    d = 2

    # panel A: maze schematic

    generate_schematic(ax1, self, pos=(x, y, d), alias=True)

    if save_separate:
        fig2, ax2 = plt.subplots()
        generate_schematic(ax2, self, pos=(x, y, d), alias=False)
        fig2.savefig(path.joinpath("maze_schematic.svg"), bbox_inches="tight")
        plt.close(fig2)

    # panel B: allocentric state

    generate_allocentric_plot(ax2, self, x, y)

    if save_separate:
        fig2, ax2 = plt.subplots()
        generate_allocentric_plot(ax2, self, x, y)
        fig2.savefig(path.joinpath("allocentric_state.svg"),
                     bbox_inches="tight")
        plt.close(fig2)

    # panel C: egocentric state

    generate_egocentric_plot(ax3, self, x, y, d)

    if save_separate:
        fig2, ax2 = plt.subplots()
        generate_egocentric_plot(ax2, self, x, y, d)
        fig2.savefig(path.joinpath("egocentric_state.svg"),
                     bbox_inches="tight")
        plt.close(fig2)

    #
    generate_task_plot(worlds, grid4, fig)

    if save_separate:
        fig2, ax2 = plt.subplots()
        generate_task_plot(worlds, grid4, fig2)
        fig2.savefig(path.joinpath("task_plot.svg"), bbox_inches="tight")
        plt.close(fig2)

    # add some panel labels
    plt.text(0.14, 0.95, "A", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.6, 0.95, "B", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.6, .6, "C", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.1, 0.4, "D", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)

    # plt.text(0.17, 0.7, "C", ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label)
    # plt.text(0.17, 0.7, "C", ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label)

    # save figure
    print("Saving Figure 1")
    plt.tight_layout
    if path:
        plt.savefig(path.joinpath("fig1.pdf"), dpi=1200, bbox_inches="tight")
        plt.savefig(path.joinpath("fig1.svg"), format='svg')
    else:
        plt.savefig("fig1.pdf", bbox_inches="tight", dpi=1200)
        plt.savefig("fig1.svg", format='svg')
    plt.close()


def generate_value_figure(model=None, path=None):
    if path:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
    else:
        path = pathlib.Path('.')

    if path.joinpath("value.pkl").exists() and not model:
        with open(path.joinpath("value.pkl"), 'rb') as f:
            data = pickle.load(f)
            model, state_values, ego_state_values, allo_state_values, vmin, vmax, egomin, egomax, amin, amax = data[
                'model'], \
                data['state_values'], data['ego_state_values'], data['allo_state_values'], data['vmin'], data['vmax'], \
                data['egomin'], data['egomax'], data['amin'], data['amax']
    else:
        if model:
            state_values, ego_state_values, allo_state_values, vmin, vmax, egomin, egomax, amin, amax = get_value_functions(
                model, split=True)
            data = {'model': model, 'state_values': state_values, 'ego_state_values': ego_state_values,
                    'allo_state_values': allo_state_values,
                    'vmin': vmin, 'vmax': vmax, 'egomin': egomin, 'egomax': egomax, 'amin': amin, 'amax': amax}
            with open(path.joinpath("value.pkl"), 'wb') as f:
                pickle.dump(data, f)
        else:
            raise NotImplementedError

    directions = ["\u2191", "\u2192", "\u2193", "\u2190"]

    fig = plt.figure(figsize=(24 * cm, 10 * cm))  # create figure

    # ## panel A: full value function
    grids = fig.add_gridspec(nrows=3, ncols=4, left=0.35,
                             right=0.7, bottom=0.5, top=1.0, hspace=0.45)

    for d in range(4):
        ax = fig.add_subplot(grids[0, d])
        generate_value_plot(
            ax, state_values[:, :, d], vmin, vmax, directions[d], colorbar=True if d == 3 else False)

    # panel B: egocentric component

    for d in range(4):
        ax = fig.add_subplot(grids[1, d])
        generate_value_plot(
            ax, ego_state_values[:, :, d], egomin, egomax, colorbar=True if d == 3 else False)

    # panel B: allocentric component

    ax = fig.add_subplot(grids[2, 0])
    generate_value_plot(
        ax, allo_state_values[:, :, 0], amin, amax, colorbar=True)

    # save figure
    print("Saving Value Figure")
    plt.tight_layout
    if path:
        plt.savefig(path.joinpath("value.pdf"), bbox_inches="tight")
    else:
        plt.savefig("value.pdf", bbox_inches="tight")
    plt.close()


def generate_fig2_(structure, model, path=None):
    if path:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
    else:
        path = pathlib.Path('.')

    if path.joinpath("fig2_data.pkl").exists() and structure is None and model is None:
        with (open(path.joinpath("fig2_data.pkl"), 'rb') as f):
            data = pickle.load(f)
            state_values, ego_state_values, allo_state_values, vmin, vmax, egomin, egomax, amin, amax, y_les, y_un, x_les, x_un = \
                data['state_values'], data['ego_state_values'], data['allo_state_values'], data['vmin'], data['vmax'], \
                data['egomin'], data['egomax'], data['amin'], data['amax'], data['y_les'], data['y_un'], data[
                    'x_les'], data['x_un']
    else:
        state_values, ego_state_values, allo_state_values, vmin, vmax, egomin, egomax, amin, amax = get_value_functions(
            model, split=True)
        (y_les, y_les_sem, x_les), (y_un, y_un_sem, x_un) = get_lesion_values_(structure, sigma=1)
        data = {'state_values': state_values, 'ego_state_values': ego_state_values,
                'allo_state_values': allo_state_values,
                'vmin': vmin, 'vmax': vmax, 'egomin': egomin, 'egomax': egomax, 'amin': amin, 'amax': amax,
                'y_les': y_les, 'y_un': y_un, 'x_les': x_les, 'x_un': x_un}
        with open(path.joinpath("fig2_data.pkl"), 'wb') as f:
            pickle.dump(data, f)

    directions = ["\u2191", "\u2192", "\u2193", "\u2190"]

    fig = plt.figure(figsize=(24 * cm, 10 * cm))  # create figure

    # ## panel A: full value function
    grids = fig.add_gridspec(nrows=3, ncols=4, left=0.35,
                             right=0.7, bottom=0.5, top=1.0, hspace=0.45)

    for d in range(4):
        ax = fig.add_subplot(grids[0, d])
        generate_value_plot(
            ax, state_values[:, :, d], vmin, vmax, directions[d])

    # panel B: egocentric component

    for d in range(4):
        ax = fig.add_subplot(grids[1, d])
        generate_value_plot(ax, ego_state_values[:, :, d], egomin, egomax)

    # panel B: allocentric component

    ax = fig.add_subplot(grids[2, 0])
    generate_value_plot(ax, allo_state_values[:, :, 0], amin, amax)

    # panel D: lesion results

    ax = fig.add_subplot(grids[2, 1:4])

    generate_lesion_plot_(ax, inputs = [(y_les, y_les_sem, x_les), (y_un, y_un_sem,  x_un)], labels=['Lesioned', 'Unlesioned'],env_switch_every=env_switch_every)

    # add some panel labels
    plt.text(0.34, 0.99, "A", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.34, 0.82, "B", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.34, 0.65, "C", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.427, 0.65, "D", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)

    # save figure
    print("Saving Figure 2")
    plt.tight_layout
    if path:
        plt.savefig(path.joinpath("fig2.pdf"), bbox_inches="tight")
    else:
        plt.savefig("fig2.pdf", bbox_inches="tight")
    plt.close()


def generate_fig3_(model=None, ego_state=0, path=None):
    if path:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
    else:
        path = pathlib.Path('.')
    if path.joinpath(f"fig3_data_{ego_state}.pkl").exists() and model is None:
        data = pickle.load(
            open(path.joinpath(f"fig3_data_{ego_state}.pkl"), 'rb'))
        model = data['model']

    else:
        if model:
            data = {'model': model}
            with open(path.joinpath(f"fig3_data_{ego_state}.pkl"), 'wb') as f:
                pickle.dump(data, f)
        else:
            print("No model provided")
            return
        
    directions = ["\u2191", "\u2192", "\u2193", "\u2190"]

    fig = plt.figure(figsize=(24 * cm, 10 * cm))  # create figure
    add_SR_grid(fig, model, ego_state, directions, left=0.35, right=0.63,
                bottom=0.5, top=1.0, hspace=0.1)

    # add some panel labels
    plt.text(0.33, 0.98, "A", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)
    plt.text(0.33, 0.73, "B", ha="left", va="top",
             transform=fig.transFigure, fontweight="bold", fontsize=fsize_label)

    # save figure
    print("Saving Figure 3")
    plt.tight_layout
    if path:
        plt.savefig(path.joinpath("fig3.pdf"), bbox_inches="tight")
    else:
        plt.savefig("fig3.pdf", bbox_inches="tight")
    plt.close()


def generate_ego_sr_fig(model=None, struct_single_seed=None, struct_all_seeds=None, path=None, states=None,
                        save_combined=True):
    """Generate egocentric SR figures for the requested states."""

    if path:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
    else:
        path = pathlib.Path('.')

    if struct_single_seed:
        struct_single_seed = remove_empty_dicts(struct_single_seed)
        struct_all_seeds = remove_empty_dicts(struct_all_seeds)
        world = struct_single_seed[list(struct_single_seed.keys())[
            0]]['unlesioned']['worlds'][0]
        worlds = struct_all_seeds[list(struct_all_seeds.keys())[0]
                                ]['unlesioned']['worlds']

    if states is None:
        states = range(model.ego_dim)

    results = {}

    for ego_state in states:
        figsr = plt.figure(figsize=(24, 10))
        grids = figsr.add_gridspec(nrows=2, ncols=1)

        axes_map = {}
        prev_axes = list(figsr.axes)
        add_allocentric_SR_grid(
            grids[0, 0], figsr, model, state=ego_state, ego=True)
        allocentric_axes = [ax for ax in figsr.axes if ax not in prev_axes]
        if allocentric_axes:
            axes_map['allocentric_sr'] = tuple(allocentric_axes)

        prev_axes = list(figsr.axes)
        add_egocentric_SR_grid(
            grids[1, 0], figsr, model, state=ego_state, ego=True)
        egocentric_axes = [ax for ax in figsr.axes if ax not in prev_axes]
        if egocentric_axes:
            axes_map['egocentric_sr'] = tuple(egocentric_axes)

        if save_combined:
            figsr.savefig(path.joinpath(f"ego_SR_{ego_state}.svg"))
            plt.close(figsr)
        else:
            results[ego_state] = (figsr, axes_map)

    if save_combined:
        return None

    return results


def generate_allo_sr_fig(model=None, struct_single_seed=None, struct_all_seeds=None, path=None, states=None,
                         save_combined=True):
    """Generate allocentric SR figures for the requested states."""

    if path:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
    else:
        path = pathlib.Path('.')

    if struct_single_seed:
        struct_single_seed = remove_empty_dicts(struct_single_seed)
        struct_all_seeds = remove_empty_dicts(struct_all_seeds)
        world = struct_single_seed[list(struct_single_seed.keys())[
            0]]['unlesioned']['worlds'][0]
        worlds = struct_all_seeds[list(struct_all_seeds.keys())[0]
                                ]['unlesioned']['worlds']

    if states is None:
        states = range(model.allo_dim)

    results = {}

    for allo_state in states:
        figsr = plt.figure(figsize=(24, 10))
        grids = figsr.add_gridspec(nrows=2, ncols=1)

        axes_map = {}
        prev_axes = list(figsr.axes)
        add_allocentric_SR_grid(
            grids[0, 0], figsr, model, state=allo_state, ego=False)
        allocentric_axes = [ax for ax in figsr.axes if ax not in prev_axes]
        if allocentric_axes:
            axes_map['allocentric_sr'] = tuple(allocentric_axes)

        prev_axes = list(figsr.axes)
        add_egocentric_SR_grid(
            grids[1, 0], figsr, model, state=allo_state, ego=False)
        egocentric_axes = [ax for ax in figsr.axes if ax not in prev_axes]
        if egocentric_axes:
            axes_map['egocentric_sr'] = tuple(egocentric_axes)

        if save_combined:
            figsr.savefig(path.joinpath(f"allo_SR_{allo_state}.svg"))
            plt.close(figsr)
        else:
            results[allo_state] = (figsr, axes_map)

    if save_combined:
        return None

    return results


def generate_aliasing_plot(env, ego, path=None, world=None, save_combined=True):
    """Visualise aliasing for a given egocentric observation."""
    if world:
        env.switch_world(world)

    fig = plt.figure(figsize=(24, 10))  # create figure
    grid = fig.add_gridspec(nrows=2, ncols=2)

    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(
        grid[0, 1]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]

    direction_worlds = [deepcopy(env.world).astype(float) for _ in range(4)]
    

    # add agent at position (x,y) which consists of a small yellow
    #  CHARACTER with a white border in direction d

    aliases = env.ego_to_allo[-1].get(ego)
    for y_, x_, d_ in aliases:

        direction_worlds[d_][y_, x_] = -np.inf

    panel_axes = {}

    for i, ax in enumerate(axes):

        masked_array = np.ma.masked_where(
            direction_worlds[i] == -np.inf, direction_worlds[i])
        cmap = mcm.Greys
        cmap.set_bad(color='red')

        ax.imshow(masked_array, cmap=cmap)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_linestyle('dotted')
        ax.spines['right'].set_linestyle('dotted')
        ax.spines['bottom'].set_linestyle('dotted')
        ax.spines['left'].set_linestyle('dotted')
        ax.set_title(f"Direction {i}")
        panel_axes[f'direction_{i}'] = (ax,)

    if save_combined:
        fig.savefig(path.joinpath(f"aliasing_{ego}.svg"))
        plt.close(fig)
        return None

    return fig, panel_axes
    
    

def visualise_Q_values(path, model, ego_state, allo_state, directions, actions):

    weight = model.weight
    size = model.env.size
    ego_weight = weight.copy()
    allo_weight = weight.copy()
    ego_weight[1:model.allo_dim +
               1] = np.zeros_like(ego_weight[1:model.allo_dim+1])
    allo_weight[model.allo_dim +
                1:] = np.zeros_like(allo_weight[model.allo_dim+1:])

    allo_values = model.q_w(allo_weight, allo_state,
                            ego_state, direction=2, action="all")

    ego_values = model.q_w(ego_weight, allo_state,
                           ego_state, direction=2, action="all")

    action_values = model.q_w(
        weight, allo_state, ego_state, direction=2, action="all")

    fig_, ax_ = plt.subplots(1, 2, figsize=(10, 5))
    barchart1 = ax_[0].bar(np.arange(4), allo_values, bottom=0, color='orange')
    barchart2 = ax_[1].bar(np.arange(4), action_values,
                           bottom=0, color='orange')

    ax_[0].set_title("Egocentric Allo Q-values")
    ax_[1].set_title("Action Values")
    ax_[0].set_xticks(np.arange(4))
    ax_[0].set_xticklabels(actions, color='b')

    ax_[1].set_xticks(np.arange(4))
    ax_[1].set_xticklabels(actions, color='b')
    plt.savefig(path.joinpath("Q_values.svg"))
    plt.close(fig_)
    

def visualise_Q_values_new(path, model, ego_state, allo_state, directions, actions):

    weight = model.weight
    size = model.env.size
    ego_weight = weight.copy()
    allo_weight = weight.copy()
    ego_weight[1:model.allo_dim +
               1] = np.zeros_like(ego_weight[1:model.allo_dim+1])
    allo_weight[model.allo_dim +
                1:] = np.zeros_like(allo_weight[model.allo_dim+1:])

    allo_values = model.q_w(allo_weight, allo_state,
                            ego_state, direction=0, action="all")
    
    allo_values_as_ego = model.q_w(allo_weight, allo_state,
                            ego_state, direction=2, action="all")
                                
    ego_values = model.q_w(ego_weight, allo_state,
                           ego_state, direction=2, action="all")

    action_values = model.q_w(
        weight, allo_state, ego_state, direction=2, action="all")

    fig_, ax_ = plt.subplots(2, 2, figsize=(10, 5))
    ax_ = ax_.flatten()
    
    y, x = model.env.get_2d_pos(allo_state)
    
    
    
    const = 0.05
    
    relative_sizes_allo = allo_values - np.min(allo_values)
    relative_sizes_allo = relative_sizes_allo / np.max(relative_sizes_allo) + const
    
    relative_sizes_allo_ego = allo_values_as_ego - np.min(allo_values_as_ego)
    # relative_sizes_allo_ego = relative_sizes_allo_ego / np.max(relative_sizes_allo_ego) + const
    
    relative_sizes_ego = ego_values - np.min(ego_values)
    # relative_sizes_ego = relative_sizes_ego / np.max(relative_sizes_ego) + const
    
    # relative_sizes_action = relative_sizes_ego + relative_sizes_allo_ego
    
    relative_sizes_action = action_values - np.min(action_values)
    
    max_size = np.max(np.concatenate([relative_sizes_allo_ego, relative_sizes_ego, relative_sizes_action]))
    
    
    relative_sizes_action = relative_sizes_action / max_size + const
    relative_sizes_allo_ego = relative_sizes_allo_ego / max_size + const
    relative_sizes_ego = relative_sizes_ego / max_size + const
    
    square = np.zeros((5, 5))
    ax_[0].imshow(square, cmap='Greys')
    
    add_allo_arrows(2, 2, ax_[0], relative_sizes_allo)
    
    ax_[0].set_xticks([])
    ax_[0].set_yticks([])
    
    ax_[0].set_ylabel('$a^A$', color='b')
                                  
    generate_egocentric_plot(ax_[1], model.env, x, y, d=2, relative_sizes=relative_sizes_allo_ego)
    
    generate_egocentric_plot(ax_[2], model.env, x, y, d=2, relative_sizes=relative_sizes_ego)
    
    generate_egocentric_plot(ax_[3], model.env, x, y, d=2, relative_sizes=relative_sizes_action)

    ax_[0].set_title("Allo Q-values")
    ax_[1].set_title("Allo-as-Ego Q-values")
    ax_[2].set_title("Ego Q-values")
    ax_[3].set_title("Action Values")
    
    plt.savefig(path.joinpath("Q_values.svg"))
    plt.close(fig_)

def compare_value_after_switch(structure=None, model=None, env_switch_every=1000, path=None, worlds=None):
    if path:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
    else:
        path = pathlib.Path('.')

    if path.joinpath(
            f'compare_value_after_switch_data_{env_switch_every}.pkl').exists() and not structure and not model:
        with open(path.joinpath(f'compare_value_after_switch_data_{env_switch_every}.pkl'), 'rb') as f:
            data = pickle.load(f)
            unlesioned_state_values, vmin, vmax, lesioned_state_values, lmin, lmax = data['unlesioned_state_values'], \
                data[
                    'vmin'], data['vmax'], data['lesioned_state_values'], data['lmin'], data['lmax']
    else:
        if structure and model:
            model2 = deepcopy(model)
            structure = remove_empty_dicts(structure)
            
            # worlds = structure[list(structure.keys())[
            #     0]]['unlesioned']['worlds']
            
            # for world in worlds:
            #     model.switch_world(world)
            #     model2.switch_world(world)
                
            world1 = structure[list(structure.keys())[
                0]]['unlesioned']['worlds'][1]
           
            model2.switch_world(world1)
            
            model_unlesioned = deepcopy(model2)
            model_lesioned = deepcopy(model2)

            les_allo_sr_ss_y, les_allo_sr_ss_x = get_parameter_values('allo_SR.SR_ss', structure,
                                                                            prefix='lesionLEC')
            # les_allo_sr_sas_y, les_allo_sr_sas_x = get_parameter_values('allo_SR.SR_sas', structure,
            #                                                                 prefix='lesionLEC')
            
            unles_allo_sr_ss_y, unles_allo_sr_ss_x = get_parameter_values('allo_SR.SR_ss', structure,
                                                                                prefix='unlesioned')
            # unles_allo_sr_sas_y, unles_allo_sr_sas_x = get_parameter_values('allo_SR.SR_sas', structure,
            #                                                                     prefix='unlesioned')
            
            les_ego_sr_ss_y, les_ego_sr_ss_x = get_parameter_values(
                'ego_SR.SR_ss', structure, prefix='lesionLEC')
            # les_ego_sr_sas_y, les_ego_sr_sas_x = get_parameter_values(
            #     'ego_SR.SR_sas', structure, prefix='lesionLEC')
            unles_ego_sr_ss_y, unles_ego_sr_ss_x = get_parameter_values('ego_SR.SR_ss', structure,
                                                                              prefix='unlesioned')
            # unles_ego_sr_sas_y, unles_ego_sr_sas_x = get_parameter_values('ego_SR.SR_sas', structure,
            #                                                                     prefix='unlesioned')

            lesioned_weights_y, lesioned_weights_x = get_parameter_values(
                'weight', structure, prefix='lesionLEC')
            unlesioned_weights_y, unlesioned_weights_x = get_parameter_values(
                'weight', structure, prefix='unlesioned')

            import bisect
            index = bisect.bisect_left(
                unles_allo_sr_ss_x[0], env_switch_every + 10)

            model_unlesioned.ego_SR.SR_ss = unles_ego_sr_ss_y[0][index]
            model_unlesioned.allo_SR.SR_ss = unles_allo_sr_ss_y[0][index]
            model_unlesioned.weight = unlesioned_weights_y[0][index]
            
            
            # model_unlesioned_allo.SR_sas = unles_allo_sr_sas_y[0][index]
            

            model_lesioned.ego_SR.SR_ss = les_ego_sr_ss_y[0][index] #idk why this weird size but shouldn't matter
            # because weights are zero
            # model_lesioned.ego_SR.SR_ss = np.zeros_like(
            #     model_unlesioned.ego_SR.SR_ss)
            model_lesioned.allo_SR.SR_ss = les_allo_sr_ss_y[0][index]
            model_lesioned.weight = lesioned_weights_y[0][index]
            
            # model_lesioned_ego.SR_sas = les_ego_sr_sas_y[0][index]
            # model_lesioned_allo.SR_sas = les_allo_sr_sas_y[0][index]
            

            unlesioned_state_values, vmin, vmax = get_value_functions(model_unlesioned,
                                                                      split=False)
            lesioned_state_values, lmin, lmax = get_value_functions(
                model_lesioned, split=False)
            

            data = {'unlesioned_state_values': unlesioned_state_values, 'vmin': vmin, 'vmax': vmax,
                    'lesioned_state_values': lesioned_state_values, 'lmin': lmin, 'lmax': lmax}
            with open(path.joinpath(f'compare_value_after_switch_data_{env_switch_every}.pkl'), 'wb') as f:
                pickle.dump(data, f)
        else:
            raise NotImplementedError

    fig = plt.figure(figsize=(10, 5))
    grids = fig.add_gridspec(nrows=2, ncols=4, wspace=0.1)
    for d in range(4):
        ax = fig.add_subplot(grids[0, d])
        generate_value_plot(
            ax, unlesioned_state_values[:, :, d], vmin, vmax, colorbar=True if d == 3 else False)
        ax = fig.add_subplot(grids[1, d])
        generate_value_plot(
            ax, lesioned_state_values[:, :, d], lmin, lmax, colorbar=True if d == 3 else False)

    plt.savefig(path / 'compare_value_after_switch.png')
