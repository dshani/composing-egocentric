"""
@author: Daniel Shani
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import itertools

import matplotlib.cm as mcm
import matplotlib.patches as patches

from structure_functions_ import add_egocentric_SR_grid
from fractions import Fraction


def get_occupancy_plot(model, path):
    """Returns a size x size matrix of the num of times spent in each
    position in the environment."""
    size = model.env.size
    occupancy = np.zeros((size, size))
    for loc in path:
        state = loc[0]
        if state is not None:
            x, y = model.env.get_2d_pos(state)
            occupancy[x, y] += 1
    return occupancy

def show_ego_state(i, model, local=False, ax=None, cmap='Greys'):
    if local:

        vmax, vmin = np.max(model.env.world), np.min(model.env.world)
        ego_state = model.env.ego_bins[i]

        if ax:
            ax.imshow(
                np.reshape(
                    ego_state, (model.env.pars.horizon + 1,
                                2 * model.env.pars.horizon + 1)), vmin=vmin,
                vmax=vmax, cmap=cmap)
        else:
            plt.imshow(
                np.reshape(
                    ego_state, (model.env.pars.horizon + 1,
                                2 * model.env.pars.horizon + 1)), vmin=vmin,
                vmax=vmax, cmap=cmap)
            plt.xticks([])
            plt.yticks([])
            plt.show()
    else:
        ego_state = model.env.ego_bins[i]
        print(ego_state)
        view = np.zeros(
            (2 * model.env.pars.horizon + 1, model.env.pars.horizon + 1))
        for j in range(3):
            dist = int(ego_state[2 * j])
            colour = ego_state[2 * j + 1]
            if j == 0:
                view[model.env.pars.horizon, dist] = \
                    colour
            elif j == 2:
                view[model.env.pars.horizon - dist, 0] = colour
            elif j == 1:
                view[model.env.pars.horizon + dist, 0] = colour

        plt.figure(figsize=(3, 3))
        plt.imshow(np.rot90(view))
        plt.show()