"""
@author: Daniel Shani
"""
from matplotlib.patches import FancyArrow
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import argparse
import ast
import json
import os
import pathlib
import pickle
import sys

import matplotlib.cm as mcm
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter1d

from helper_functions_ import find_most_recent, load_structure, load_recent_model
from parameters import DotDict
from parameters import parameters

from scipy.special import softmax

from figure_functions_ import (
    generate_value_fig,
    generate_generalisation_plot,
    generate_task_plot,
    generate_lesion_plot_,
    generate_allo_sr_fig,
    generate_ego_sr_fig,
    generate_aliasing_plot,
    generate_schematic,
    _load_chunked_occupancy_summary,
    _plot_chunked_occupancy,
)

from structure_functions_ import *

def get_analysis_parser():
    parser = argparse.ArgumentParser(description='Analysis')

    def _str2bool(value):
        if isinstance(value, bool):
            return value
        value_lower = value.lower()
        if value_lower in {"true", "1", "yes", "y"}:
            return True
        if value_lower in {"false", "0", "no", "n"}:
            return False
        raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{value}'")
    for key in ["seed", "date", "run", "recent", "dict_params", "load_params", "save_dirs", "single_seed"]:
        parser.add_argument('--' + key, type=ast.literal_eval, default=None
                            )
    parser.add_argument('--job_id', type=str, default=None)
    parser.add_argument('--ego_state', type=ast.literal_eval, default=None)
    parser.add_argument('--allo_state', type=ast.literal_eval, default=None)
    parser.add_argument('--path_start', type=ast.literal_eval, default=None)
    parser.add_argument('--heatmap_path_num', type=int, default=100)
    parser.add_argument('--lesion_test', type=_str2bool, default=False)
    parser.add_argument('--unlesion_test', type=_str2bool, default=False)
    parser.add_argument('--use_cached_data', type=_str2bool, default=False)
    parser.add_argument('--env_switch_every', type=int, default=1000)
    parser.add_argument('--full_plots', type=_str2bool, default=False)
    parser.add_argument('--metrics_only', type=_str2bool, default=False)
    parser.add_argument('--generalisation_only', type=_str2bool, default=False)
    parser.add_argument('--room_compare_only', type=_str2bool, default=False)

    parser.add_argument('--debug', type=_str2bool, default=False)
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Maximum threads for parallel data loading')
    parser.add_argument('--delay', type=int, default=None)
    parser.add_argument('--figures_root', type=str, default=None,
                        help='Override the root directory for analysis figures')
    parser.add_argument('--model-analysis', action='store_true', default=False,
                        help='Limit analysis to model-centric figures without recomputing occupancy data')

    return parser

def save_params(model, path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    with open(path / 'params.txt', 'w') as f:
        f.write(str(model.env.pars))
        
        
def get_ratio_values(struct_all_seeds):
    
    ratio_y, unlesioned_x = get_parameter_values('ratio', struct_all_seeds, prefix=None)
    
    
    
    return ratio_y, unlesioned_x

def calculate_mean_ratio(struct_all_seeds, sigma=5, save_path=None):
    ratio_y, lesioned_x = get_ratio_values(struct_all_seeds)
    
    ratio_y = np.array(ratio_y)
    lesioned_x = np.array(lesioned_x)
    
    ratio_y_mean = np.mean(ratio_y, axis=0)
    ratio_y_sem = np.std(ratio_y, axis=0, ddof=1) / np.sqrt(ratio_y.shape[0])

    # Calculate mean and SEM for lesioned data
    lesioned_x_mean = np.mean(lesioned_x, axis=0)
    

    # Sort the data based on x-values
    ratio_sorted = sorted(zip(lesioned_x_mean, ratio_y_mean, ratio_y_sem))

    # Unzip the sorted data
    x, ratio_mean, ratio_sem = map(np.array, zip(*ratio_sorted))

    # Apply Gaussian smoothing
    ratio_mean_smooth = gaussian_filter1d(ratio_mean, sigma)
    ratio_sem_smooth = gaussian_filter1d(ratio_sem, sigma)
    
    if save_path:
        with open(save_path / 'ratio_mean.pkl', 'wb') as f:
            pickle.dump(ratio_mean_smooth, f)
        with open(save_path / 'ratio_sem.pkl', 'wb') as f:
            pickle.dump(ratio_sem_smooth, f)
        with open(save_path / 'x.pkl', 'wb') as f:
            pickle.dump(x, f)
    
    return ratio_mean_smooth, ratio_sem_smooth, x

from fractions import Fraction
        
def lesion_histogram(struct_all_seeds, path=None, ax=None):
    
    ratio_y, x = get_ratio_values(struct_all_seeds)
    ratio_y = np.array(ratio_y)
    x = np.array(x)

    if ratio_y.size == 0 or x.size == 0:
        raise ValueError('No ratio data available to plot.')

    # ``x`` stores the episode indices at which ratios were logged. Use these
    # values so the histogram's x-axis reflects the true training duration even
    # when logging is downsampled.
    if x.ndim == 1:
        episode_numbers = np.sort(x)
    else:
        episode_numbers = np.sort(np.mean(x, axis=0))

    N = ratio_y.shape[0]
    num_episodes = ratio_y.shape[1]
    hist_bins = [0, 0.01, 0.1, 0.2, 0.5, 2/3, 3/4, 1, 4/3, 3/2, 2, 5, 10, 100, np.inf]
    
    h = []
    for t in range(num_episodes):
        h.append(np.histogram(ratio_y[:, t], weights=np.ones(N) / N, bins=hist_bins)[0])
    
    h = np.array(h)
    
    if not ax:
        fig, ax = plt.subplots()
    ax.imshow(
        h.T,
        aspect='auto',
        cmap='viridis',
        extent=[episode_numbers[0], episode_numbers[-1], len(hist_bins) - 1, 0],
    )
    ax.set_xlim(episode_numbers[0], episode_numbers[-1])
    ax.set_xticks(np.linspace(episode_numbers[0], episode_numbers[-1], num=6, dtype=int))
    ax.set_yticks(range(len(hist_bins) - 1))
    ax.set_yticklabels([f'{hist_bins[i]:.2f} - {hist_bins[i+1]:.2f}' for i in range(len(hist_bins) - 1)])
    
    # We change the fontsize of minor ticks label 
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=5)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Ratio (Unlesioned Steps / Lesioned Steps)')
    ax.set_title(f'Ratio Histogram: N={N}')
    
    # colorbar
    cbar = plt.colorbar(mappable=ax.images[0], ax=ax)
    cbar.set_label('Rate')
    
    if not ax:
        fig.savefig(path / 'ratio_histogram.png')
        plt.close(fig)
        
def room_comparisons(model=None, struct_single_seed=None, struct_all_seeds=None, sigma=1, path=None, fsize_label=12,
                                 env_switch_every=1000, delay=1, seednum=None, save_combined=True):
    """Generate the room comparison figure.

    Parameters
    ----------
    save_combined : bool, optional
        Controls whether the combined figure is saved immediately (the
        previous behaviour) or whether the caller receives the figure and a
        mapping of panel names to axes so that individual panels can be saved
        elsewhere. Defaults to ``True``.
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
    if not seed_keys:
        raise ValueError("struct_all_seeds must contain at least one seed entry")

    worlds = struct_all_seeds[seed_keys[0]]['unlesioned']['worlds']
    all_worlds = [struct_all_seeds[key]['unlesioned']['worlds'] for key in seed_keys]

    world = worlds[0]

    panel_axes = {}

    def register_panel(name, ax):
        panel_axes.setdefault(name, []).append(ax)

    # A: Task schematic
    task_axes = generate_task_plot(worlds, grids[0, :5], fig, return_axes=True)
    for idx, task_ax in enumerate(task_axes):
        register_panel(f'task_world_{idx}', task_ax)
    # B: Lesion Plot
    ax = fig.add_subplot(grids[1, :5])
    (y_les, y_les_sem, x_les), (y_un, y_un_sem, x_un) = get_lesion_values_(struct_all_seeds, sigma=sigma)
    
    N_les = y_les.shape[0]
    N_un = y_un.shape[0]
    

    # hole_times_unles = get_hole_times(
    #     struct_all_seeds, prefix='unlesioned', worlds=worlds)
    # hole_times_les = get_hole_times(
    #     struct_all_seeds, prefix='lesionLEC', worlds=worlds)

    inputs = [(y_les, y_les_sem, x_les), (y_un, y_un_sem, x_un)]
    generate_lesion_plot_(ax, inputs=inputs, labels=['allocentric','ego+allo'],env_switch_every=env_switch_every)
    register_panel('lesion_learning', ax)
    
    # hole_times_unles = gaussian_filter1d(hole_times_unles, sigma=10)
    # hole_times_les = gaussian_filter1d(hole_times_les, sigma=10)

    bottom_grid = grids[2, :5].subgridspec(2, 1, hspace=0.35)

    barrier_ax = fig.add_subplot(bottom_grid[0, 0])
    chunk_size_target = 25
    chunk_summary = _load_chunked_occupancy_summary(path, expected_chunk_size=chunk_size_target)
    plotted_chunked = False
    if chunk_summary is not None:
        chunk_data, chunk_size_value = chunk_summary
        plotted_chunked = _plot_chunked_occupancy(barrier_ax, chunk_data, chunk_size_value, env_switch_every)

    if not plotted_chunked:
        barrier_ax.cla()
        try:
            hole_times_unles_mean, hole_times_unles_sem, x_hole_unles = get_hole_times(
                struct_all_seeds,
                prefix='unlesioned',
                worlds=worlds,
                all_worlds=all_worlds,
                switch_every=env_switch_every,
            )
            hole_times_les_mean, hole_times_les_sem, x_hole_les = get_hole_times(
                struct_all_seeds,
                prefix='lesionLEC',
                worlds=worlds,
                all_worlds=all_worlds,
                switch_every=env_switch_every,
            )
        except ValueError:
            barrier_ax.set_xticks([])
            barrier_ax.set_yticks([])
            barrier_ax.text(
                0.5,
                0.5,
                'No path data available',
                ha='center',
                va='center',
                fontsize='small',
                transform=barrier_ax.transAxes,
            )
        else:
            hole_times_unles_mean = gaussian_filter1d(hole_times_unles_mean, sigma=10)
            hole_times_les_mean = gaussian_filter1d(hole_times_les_mean, sigma=10)

            hole_times_unles_sem = gaussian_filter1d(hole_times_unles_sem, sigma=10)
            hole_times_les_sem = gaussian_filter1d(hole_times_les_sem, sigma=10)

            generate_lesion_plot_(
                barrier_ax,
                inputs=[
                    (hole_times_les_mean, hole_times_les_sem, x_hole_les[0]),
                    (hole_times_unles_mean, hole_times_unles_sem, x_hole_unles[0]),
                ],
                labels=['allocentric', 'allocentric + egocentric'],
                env_switch_every=env_switch_every,
            )
            barrier_ax.set_yscale("linear")
            barrier_ax.set_ylabel('Proportion of time spent inside barrier', fontsize='xx-small')

    register_panel('barrier_time', barrier_ax)

    ax2 = fig.add_subplot(bottom_grid[1, 0])

    lesion_histogram(struct_all_seeds, path=path, ax=ax2)
    register_panel('lesion_histogram', ax2)
    
    
    # ax2.set_ylabel('Proportion of time spent inside barrier',
    #                fontsize='xx-small')

    # x_axis = np.arange(0, len(hole_times_unles))

    # ax2.plot(x_axis, hole_times_les, color='r')
    # ax2.plot(x_axis, hole_times_unles, color='b')

    # switch_color = '0.7'

    # for i in range(1, 6):
    #     ax2.axvline(x=i * env_switch_every, color=switch_color, linestyle='--')

    # E path comparison after switch:

    # world = worlds[1]

    # inner_grid = grids[:, 5].subgridspec(2, 1, height_ratios=[1, 1])

    # path_unlesioned, un_ep = get_path(struct_all_seeds, prefix='unlesioned',
    #                                   episode=env_switch_every + delay, path_start=None)
    # path_lesioned, les_ep = get_path(struct_all_seeds, prefix='lesionLEC',
    #                                  episode=env_switch_every + delay, path_start=None)
    # heatmap_data_unlesioned = get_heatmap_data(path_unlesioned, world=world)
    # heatmap_data_lesioned = get_heatmap_data(path_lesioned, world=world)

    # vmin = np.min([np.min(heatmap_data_unlesioned),
    #               np.min(heatmap_data_lesioned)])
    # vmax = np.max([np.max(heatmap_data_unlesioned),
    #               np.max(heatmap_data_lesioned)])
    # lims = [vmin, vmax]

    # ax = fig.add_subplot(inner_grid[0, 0])

    # plot_track(path_unlesioned, ax, world=world, color='r', label=f"Allo+Ego, Episode: {un_ep} ", lims=lims,
    #            fontsize='xx-small')

    # ax = fig.add_subplot(inner_grid[1, 0])
    # plot_track(path_lesioned, ax, world=world, color='r', label=f'Allo, Episode: {les_ep}', lims=lims,
    #            fontsize='xx-small')

    if save_combined:
        if seednum is None:
            stem = f'room_compare_{sigma}_{env_switch_every}_{delay}'
        else:
            stem = f'room_compare_{sigma}_{env_switch_every}_{delay}_{seednum}'
        fig.savefig(path.joinpath(f'{stem}.png'), dpi=1200)
        fig.savefig(path.joinpath(f'{stem}.svg'), format='svg')
        plt.close(fig)
        return None

    return fig, {name: tuple(axes) for name, axes in panel_axes.items()}
