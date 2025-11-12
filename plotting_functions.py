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


# def get_directional_occupancies(paths, model, gamma):
#     occupancies_allo = np.zeros(
#         [model.env.allo_dim, model.env.size, model.env.size, 4])
#     occupancies_ego = np.zeros(
#         [model.env.ego_dim, model.env.ego_dim])

#     for path in paths:
#         for index, [allo, ego, direction, action] in enumerate(path):
#             if allo is not None:
#                 for j, [allo_, ego_, d_, a_] in enumerate(path[index:]):
#                     if allo_ is not None:
#                         occupancies_allo[allo, model.env.get_2d_pos(allo_)[0],
#                         model.env.get_2d_pos(allo_)[1], direction] += gamma ** j

#                         occupancies_ego[ego, ego_] += gamma ** j

#     occupancies_ego_mapped = np.zeros(
#         (model.env.ego_dim, model.env.size,
#          model.env.size, 4))
#     for ego in range(model.env.ego_dim):
#         for ego_ in range(model.env.ego_dim):
#             for x, y, k in model.env.ego_to_allo[-1][ego_]:
#                 occupancies_ego_mapped[ego, x, y, k] += occupancies_ego[
#                     ego, ego_]

#     return occupancies_allo, occupancies_ego_mapped


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


# def get_occupancy_plots(model, paths):
#     """Produces heatmap of the occupancy of each state in the environment."""
#     for path in paths:
#         occupancy = get_occupancy_plot(model, path)
#         plt.imshow(occupancy, cmap='jet')
#         plt.colorbar()
#         plt.show()


# def plot_eigenfunctions(eigenfunctions, k=None, ego=False, backward=False):
#     raise NotImplementedError


# def plot_policy(agent):
#     raise NotImplementedError


# def plot_path(agent):
#     # path = np.stack(agent.path, 1)
#     # _plot_path(path, agent)
#     raise NotImplementedError


# def _plot_path(_path, agent):
#     # path = np.copy(_path)
#     # # path[:, 2] = agent.env.size - path[:, 2] - 1
#     # path[:, 1] = agent.env.size - path[:, 1] - 1
#     # plt.scatter(path[:, 2], path[:, 1])
#     # plt.xlim(0, agent.env.size)
#     # plt.ylim(0, agent.env.size)
#     # plt.show()
#     raise NotImplementedError


# def plot_all_paths(paths, agent, num=1):
#     for index, _path in enumerate(paths):
#         if index % num == 0:
#             path = np.array(_path)
#             _plot_path(path, agent)


# def variances_over_path(path, model, weight=None):
#     # todo: update to add w_0
#     if weight is None:
#         weight = model.weight

#     w_allo = weight.copy()
#     w_ego = weight.copy()
#     w_ego[: model.allo_dim] = np.zeros_like(w_ego[:model.allo_dim])
#     w_allo[model.allo_dim:] = np.zeros_like(w_allo[model.allo_dim:])

#     full, egos, allos = [], [], []

#     for loc in path:

#         state, ego, direction, action = loc
#         if state is not None:
#             state = int(state)
#             ego = int(ego)

#             full.append(model.q_w(weight, state, ego, direction, action))
#             egos.append(model.q_w(w_ego, state, ego, direction, action))
#             allos.append(model.q_w(w_allo, state, ego, direction, action))

#     return np.mean(full), np.std(full), np.mean(allos), np.std(allos), \
#         np.mean(egos), np.std(egos)


# def plot_value_variances(model, paths, weights=None, sigma=1):
#     variances = []
#     if not weights:
#         weights = [model.weight for _ in range(len(paths))]
#     assert len(weights) == len(paths)
#     for path, weight in zip(paths, weights):
#         variances.append(variances_over_path(path, model, weight))
#     variances = np.stack(variances)
#     x = np.arange(
#         variances.shape[0] * model.pars.save_every,
#         step=model.pars.save_every)
#     for i in range(variances.shape[1]):
#         variances[:, i] = gaussian_filter(variances[:, i], sigma=sigma)
#     for i in range(2):
#         plt.subplot(2, 1, i + 1)
#         plt.plot(x, variances[:, i::2])
#         if i == 0:
#             plt.legend(['Full mean', 'Allocentric mean', 'Egocentric mean'])
#         else:
#             plt.legend(['Full s.d.', 'Allocentric s.d.', 'Egocentric s.d.'])

#     plt.xlabel('Path Number')
#     plt.show()


# def plot_steps(acc, sigma=1, xlim=None, ylim=None):
#     acc = gaussian_filter(acc, sigma=sigma)
#     plt.plot(acc)
#     if xlim:
#         plt.xlim(xlim)
#     if ylim:
#         plt.ylim(top=ylim)

#     plt.xlabel("Episode")
#     plt.ylabel("Steps Taken")
#     plt.show()


# def plot_q(agent, env=None, path=None):
#     if env is None:
#         env = agent.env
#     """Produce 16 plots of [x,y] Q-values for each direction and action."""
#     size = env.size
#     state_values = np.zeros((4, size, size, 4))
#     w = agent.weight
#     for d in range(4):
#         for a in range(4):
#             for i_ in range(size):
#                 for j_ in range(size):
#                     if env.world[i_, j_] <= 0:
#                         state = env.get_1d_pos([i_, j_])
#                         ego = env.get_ego_obs([i_, j_], d)
#                         state_values[d, i_, j_, a] = agent.q_w(
#                             w, state, ego,
#                             d, a)
#     state_values[state_values == 0] = np.mean(state_values[state_values != 0])
#     show_4x4_policy(state_values, path)


# def plot_diff_q(agent, env=None, path=None):
#     if env is None:
#         env = agent.env
#     size = env.size
#     allo_state_values = np.zeros((4, size, size, 4))
#     ego_state_values = np.zeros((4, size, size, 4))
#     w = agent.weight
#     w_allo = w.copy()
#     w_ego = w.copy()
#     w_allo[agent.allo_dim+1:] = np.zeros_like(w_allo[agent.allo_dim+1:])
#     w_allo[0] = 0
#     w_ego[:agent.allo_dim+1] = np.zeros_like(w_ego[:agent.allo_dim+1])
#     for d in range(4):
#         for a in range(4):
#             for i_ in range(size):
#                 for j_ in range(size):
#                     if env.world[i_, j_] <= 0:
#                         state = env.get_1d_pos([i_, j_])
#                         ego = env.get_ego_obs([i_, j_], d)
#                         allo_state_values[d, i_, j_, a] = \
#                             agent.q_w(w_allo, state, ego, d, a)
#                         ego_state_values[d, i_, j_, a] = \
#                             agent.q_w(w_ego, state, ego, d, a)

#     allo_state_values[allo_state_values == 0] = np.mean(
#         allo_state_values[allo_state_values != 0])
#     ego_state_values[ego_state_values == 0] = np.mean(
#         ego_state_values[ego_state_values != 0])
#     print("Allocentric")
#     show_4x4_policy(allo_state_values, path)
#     print("Egocentric")
#     show_4x4_policy(ego_state_values, )


# def plot_v(agent):
#     size = agent.env.size
#     state_values = np.zeros((size, size, 4))
#     w = agent.weight
#     for d in range(4):
#         for i_ in range(size):
#             for j_ in range(size):
#                 if agent.env.world[i_, j_] <= 0:
#                     state = agent.env.get_1d_pos([i_, j_])
#                     ego = agent.env.get_ego_obs([i_, j_], d)
#                     state_values[i_, j_, d] = agent.q_w(w, state, ego, d)
#     state_values[state_values == 0] = np.mean(state_values[state_values != 0])
#     show_2x2(state_values)


# def plot_diff_v(agent):
#     size = agent.env.size
#     allo_state_values = np.zeros((size, size, 4))
#     ego_state_values = np.zeros((size, size, 4))
#     w = agent.weight
#     w_allo = w.copy()
#     w_ego = w.copy()
#     w_allo[agent.allo_dim:] = np.zeros_like(w_allo[agent.allo_dim:])
#     w_ego[:agent.allo_dim] = np.zeros_like(w_ego[:agent.allo_dim])
#     for d in range(4):
#         for i_ in range(size):
#             for j_ in range(size):
#                 if agent.env.world[i_, j_] <= 0:
#                     state = agent.env.get_1d_pos([i_, j_])
#                     ego = agent.env.get_ego_obs([i_, j_], d)
#                     allo_state_values[i_, j_, d] = agent.q_w(
#                         w_allo, state, ego, d)
#                     ego_state_values[i_, j_, d] = agent.q_w(
#                         w_ego, state, ego, d)

#     allo_state_values[allo_state_values == 0] = np.mean(
#         allo_state_values[allo_state_values != 0])
#     ego_state_values[ego_state_values == 0] = np.mean(
#         ego_state_values[ego_state_values != 0])
#     print("Allocentric")
#     show_2x2(allo_state_values)
#     print("Egocentric")
#     show_2x2(ego_state_values)


# def show_4x4(Q):
#     vmin = np.min(Q)
#     vmax = np.max(Q)

#     values = [[a, l, Q[l, :, :, a]] for a in range(4) for l in range(4)]

#     fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))

#     for ax, value in zip(axes.flat, values):
#         im = ax.imshow(value[2], vmin=vmin, vmax=vmax)
#         ax.title.set_text(f"a: {value[0]}, d: {value[1]}")

#     plt.tight_layout()
#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#     fig.colorbar(im, cax=cbar_ax)


# def show_4x4_policy(Q, path=None):
#     vmin = np.min(Q, axis=(1, 2, 3))
#     vmax = np.max(Q, axis=(1, 2, 3))
#     print(vmin.shape)

#     values = [[a, l, Q[l, :, :, a]] for a in range(4) for l in range(4)]

#     fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))

#     for ax, value in zip(axes.flat, values):
#         im = ax.imshow(value[2], vmin=vmin[value[1]], vmax=vmax[value[1]])
#         ax.title.set_text(f"a: {value[0]}, d: {value[1]}")

#     plt.tight_layout()
#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#     fig.colorbar(im, cax=cbar_ax)
#     plt.show()

#     if path:
#         fig.savefig(path)


# def generate_q_plot(agent):
#     size = agent.env.size
#     state_values = np.zeros((size, size, 4, 4))
#     allo_state_values = np.zeros((size, size, 4, 4))
#     ego_state_values = np.zeros((size, size, 4, 4))
#     w = agent.weight
#     w_allo = w.copy()
#     w_ego = w.copy()
#     w_allo[agent.allo_dim:] = np.zeros_like(w_allo[agent.allo_dim:])
#     w_ego[:agent.allo_dim] = np.zeros_like(w_ego[:agent.allo_dim:])
#     for a in range(4):
#         for d in range(4):
#             for i_ in range(size):
#                 for j_ in range(size):
#                     if agent.env.world[i_, j_] <= 0:
#                         state = agent.env.get_1d_pos([i_, j_])
#                         ego = agent.env.get_ego_obs([i_, j_], d)
#                         state_values[i_, j_, d, a] = agent.q_w(
#                             w, state, ego,
#                             d, a)
#                         allo_state_values[i_, j_, d, a] = agent.q_w(
#                             w_allo, state, ego,
#                             d, a)
#                         ego_state_values[i_, j_, d, a] = agent.q_w(
#                             w_ego, state, ego,
#                             d, a)

#     state_values[state_values == 0] = np.mean(state_values[state_values != 0])
#     allo_state_values[allo_state_values == 0] = np.mean(
#         allo_state_values[allo_state_values != 0])
#     ego_state_values[ego_state_values == 0] = np.mean(
#         ego_state_values[ego_state_values != 0])

#     fig, ax = plt.subplots(
#         nrows=12, ncols=4, figsize=(10, 30),
#         gridspec_kw={"width_ratios": [1, 1, 1, 1]})

#     # outer = [[full], [allo], [ego]]
#     # fig, axes = plt.subplot_mosaic(outer, figsize=(10, 30))
#     for d in range(4):
#         for a in range(4):
#             ax[a, d].imshow(
#                 state_values[:, :, d, a], cmap="RdBu",
#                 vmin=np.min(state_values), vmax=np.max(state_values))
#             ax[a + 4, d].imshow(
#                 allo_state_values[:, :, d, a], cmap="RdBu",
#                 vmin=np.min(allo_state_values), vmax=np.max(
#                     allo_state_values))
#             ax[a + 8, d].imshow(
#                 ego_state_values[:, :, d, a], cmap="RdBu",
#                 vmin=np.min(
#                     ego_state_values), vmax=np.max(ego_state_values)
#                 )
#             ax[a, d].set_xticks([])
#             ax[a, d].set_yticks([])
#             ax[a + 4, d].set_xticks([])
#             ax[a + 4, d].set_yticks([])
#             ax[a + 8, d].set_xticks([])
#             ax[a + 8, d].set_yticks([])
#             if a == 0:
#                 ax[a, d].set_title(f"d: {d}")
#             if d == 0:
#                 ax[a, d].set_ylabel(f"$Q_w(s,{a})$")
#                 ax[a + 4, d].set_ylabel(f"$Q_w(s^a,{a})$")
#                 ax[a + 8, d].set_ylabel(f"$Q_w(s^e, {a})$")

#     im1 = ax[8, 0].imshow(
#         ego_state_values[:, :, 0, 0], cmap="RdBu",
#         vmin=np.min(ego_state_values), vmax=np.max(
#             ego_state_values))
#     cax1 = fig.add_axes([1., 0.01, 0.02, 0.3])
#     clb1 = fig.colorbar(im1, cax=cax1)

#     im2 = ax[4, 0].imshow(
#         allo_state_values[:, :, 0, 0], cmap="RdBu",
#         vmin=np.min(allo_state_values), vmax=np.max(
#             allo_state_values))
#     cax2 = fig.add_axes([1., 0.35, 0.02, 0.3])
#     clb2 = fig.colorbar(im2, cax=cax2)

#     im3 = ax[0, 0].imshow(
#         state_values[:, :, 0, 0], cmap="RdBu",
#         vmin=np.min(state_values), vmax=np.max(
#             state_values))
#     cax3 = fig.add_axes([1., 0.68, 0.02, 0.3])
#     clb3 = fig.colorbar(im3, cax=cax3)

#     plt.tight_layout()


# def generate_value_plot(agent):
#     gridspec = {'width_ratios': [1, 1, 1, 1, 0.1]}
#     fig, ax = plt.subplots(3, 5, figsize=(10, 10), gridspec_kw=gridspec)

#     size = agent.env.size
#     state_values = np.zeros((size, size, 4))
#     w = agent.weight
#     for d in range(4):
#         for i_ in range(size):
#             for j_ in range(size):
#                 if agent.env.world[i_, j_] <= 0:
#                     state = agent.env.get_1d_pos([i_, j_])
#                     ego = agent.env.get_ego_obs([i_, j_], d)
#                     state_values[i_, j_, d] = agent.q_w(w, state, ego, d)
#     state_values[state_values == 0] = np.mean(state_values[state_values != 0])

#     allo_state_values = np.zeros((size, size, 4))
#     ego_state_values = np.zeros((size, size, 4))
#     w = agent.weight
#     w_allo = w.copy()
#     w_ego = w.copy()
#     w_allo[agent.allo_dim:] = np.zeros_like(w_allo[agent.allo_dim:])
#     w_ego[:agent.allo_dim] = np.zeros_like(w_ego[:agent.allo_dim])
#     for d in range(4):
#         for i_ in range(size):
#             for j_ in range(size):
#                 if agent.env.world[i_, j_] <= 0:
#                     state = agent.env.get_1d_pos([i_, j_])
#                     ego = agent.env.get_ego_obs([i_, j_], d)
#                     allo_state_values[i_, j_, d] = agent.q_w(
#                         w_allo, state, ego, d)
#                     ego_state_values[i_, j_, d] = agent.q_w(
#                         w_ego, state, ego, d)

#     allo_state_values[allo_state_values == 0] = np.mean(
#         allo_state_values[allo_state_values != 0])
#     ego_state_values[ego_state_values == 0] = np.mean(
#         ego_state_values[ego_state_values != 0])

#     values = [state_values, allo_state_values, ego_state_values]
#     for k in range(3):
#         for d in range(4):
#             im = ax[k, d].imshow(
#                 values[k][:, :, d], vmin=np.min(values[k]),
#                 vmax=np.max(values[k]), cmap="RdBu")
#             if k == 0:
#                 ax[k, d].title.set_text(f"d: {d}")
#             ax[k, d].set_xticks([])
#             ax[k, d].set_yticks([])

#         cax = ax[k, 4]
#         plt.colorbar(im, cax=cax)

#     ax[0, 0].set_ylabel(f"Full Value Function")
#     ax[1, 0].set_ylabel(f"Allocentric")
#     ax[2, 0].set_ylabel(f"Egocentric")
#     plt.tight_layout()


# def show_2x2(state_values, vmin=None, vmax=None, cmap=None):
#     if cmap is None:
#         cmap = "plasma"

#     if vmin is None:
#         vmin = np.min(state_values)
#     if vmax is None:
#         vmax = np.max(state_values)

#     fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 4))

#     for d in range(2):
#         for d_ in range(2):
#             im = axes[d, d_].imshow(
#                 state_values[:, :, 2 * d + d_],
#                 vmin=vmin, vmax=vmax, cmap=cmap)
#             axes[d, d_].title.set_text(f"d: {2 * d + d_}")
#             axes[d, d_].set_xticks([])
#             axes[d, d_].set_yticks([])

#     plt.tight_layout()
#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#     fig.colorbar(im, cax=cbar_ax)


# def allo_basis_in_allo_coords(model, SR_sas=None):
#     if SR_sas is not None:
#         model.allo_SR.SR_sas = SR_sas
#     for i in range(model.env.allo_dim):
#         if model.env.world[model.env.get_2d_pos(i)[0],
#         model.env.get_2d_pos(i)[1]] <= 0:
#             basis = model.allo_SR.SR_sas[i]
#             plot_allo_basis_in_allo_coords(basis, model)


# def plot_allo_basis_in_allo_coords(basis, model):
#     array = np.zeros((model.env.size, model.env.size))
#     for j in range(basis.shape[0]):
#         x, y = model.env.allo_bins[j]
#         array[x, y] = basis[j]

#     plt.imshow(array)
#     plt.xticks([])
#     plt.yticks([])
#     plt.colorbar()
#     plt.show()


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


# def ego_basis_in_allo_coords(model, SR_sas=None):
#     if SR_sas is not None:
#         model.ego_SR.SR_sas = SR_sas

#     for i in range(model.ego_dim):
#         show_ego_state(i, model, model.env.pars.local)
#         basis = np.max(model.ego_SR.SR_sas, 1)[i]
#         plot_ego_basis_in_allo_coords(basis, model, local=model.env.pars.local)


# def ego_basis_in_ego_coords(model, SR_sas=None):
#     if SR_sas is not None:
#         model.ego_SR.SR_sas = SR_sas

#     for i in range(model.ego_dim):
#         print(i)
#         show_ego_state(i, model, model.env.pars.local)
#         basis = np.max(model.ego_SR.SR_sas, 1)[i]
#         plot_ego_basis_in_ego_coords(basis, model, local=model.env.pars.local)


# def plot_ego_basis_in_ego_coords(normed_basis, model, local=False):
#     # currently doesn't distinguish between colour walls
#     if local:

#         hist, bin_edges = np.histogram(normed_basis, bins=5)
#         fig, axs = plt.subplots(
#             hist.shape[0] - hist.shape[0] // 2,
#             np.max(hist[hist.shape[0] // 2:]), figsize=(5, 5))
#         if len(axs.shape) == 2:
#             for k in range(axs.shape[0]):
#                 for j in range(axs.shape[1]):
#                     if j != 0:
#                         axs[k, j].axis("off")

#         k = 0
#         for i in range(hist.shape[0] - 1, hist.shape[0] // 2 - 1, -1):

#             if hist[i]:

#                 ego_states = np.where(
#                     np.logical_and(
#                         normed_basis > bin_edges[i],
#                         normed_basis <= bin_edges[i + 1]))[0]

#                 # sort by size of normed basis
#                 if len(ego_states) > 1:

#                     for j in range(len(ego_states)):
#                         state = ego_states[j]
#                         show_ego_state(state, model, local=True, ax=axs[k, j])
#                         # axs[k, j].imshow(
#                         #     np.reshape(
#                         #         model.env.ego_bins[state],
#                         #         (model.env.pars.horizon + 1,
#                         #          2 * model.env.pars.horizon + 1)), vmax=vmax, vmin=vmin)
#                 else:
#                     state = ego_states[0]
#                     if len(axs.shape) == 2:
#                         show_ego_state(state, model, local=True, ax=axs[k, 0])
#                     else:
#                         show_ego_state(state, model, local=True, ax=axs[k])

#             if len(axs.shape) == 2:
#                 axs[k, 0].set_ylabel(
#                     str(round(bin_edges[i], 3)) + " - " + str(
#                         round(
#                             bin_edges[
#                                 i +
#                                 1],
#                             3)),
#                     fontsize='xx-small')
#                 axs[k, 0].xaxis.set_visible(False)
#                 axs[k, 0].tick_params(left=False, labelleft=False)
#             else:
#                 axs[k].set_ylabel(
#                     str(round(bin_edges[i], 3)) + " - " + str(
#                         round(
#                             bin_edges[
#                                 i +
#                                 1],
#                             3)),
#                     fontsize='xx-small')
#                 axs[k].xaxis.set_visible(False)
#                 axs[k].tick_params(left=False, labelleft=False)

#             k += 1

#     else:
#         raise NotImplementedError


# def allo_basis_in_ego_coords(model, SR_sas=None):
#     if SR_sas is not None:
#         model.allo_SR.SR_sas = SR_sas

#     for i in range(model.ego_dim):
#         basis = np.max(model.ego_SR.SR_sas, 1)[i]
#         plot_allo_basis_in_ego_coords(basis, model, local=model.env.pars.local)


# def plot_allo_basis_in_ego_coords(normed_basis, model, local=False):
#     if local:

#         vmax, vmin = np.max(model.env.world), np.min(model.env.world)

#         hist, bin_edges = np.histogram(normed_basis, bins=10)
#         print(hist, bin_edges)
#         num_rows = 2 * (hist.shape[0] - hist.shape[0] // 2)
#         num_cols = 2 * (np.max(hist[hist.shape[0] // 2:]))
#         fig, axs = plt.subplots(num_rows, num_cols, figsize=(100,100))

#         if len(axs.shape) == 2:
#             for k in range(axs.shape[0]):
#                 for j in range(axs.shape[1]):
#                     axs[k, j].axis("off")
#         k = 0

#         for i in range(hist.shape[0] - 1, hist.shape[0] // 2 - 1, -1):
#             if hist[i]:
#                 allo_states = np.where(
#                     np.logical_and(
#                         normed_basis > bin_edges[i],
#                         normed_basis <= bin_edges[i + 1]))[0]
#                 if len(allo_states) > 1:

#                     for j in range(len(allo_states)):
#                         state = allo_states[j]
#                         for d in range(4):
#                             model.env.get_egocentric_view(
#                                 model.env.world,
#                                 model.env.get_2d_pos(state), d,
#                                 display=True,
#                                 ax=axs[k + (
#                                         d // 2) * num_rows // 2,
#                                        j + (d % 2) * num_cols // 2])

#                         # view = np.reshape(
#                         #     model.env.get_egocentric_view(
#                         #         model.env.world,
#                         #         model.env.get_2d_pos(state), d)[1],
#                         #     (model.env.pars.horizon + 1,
#                         #      2 * model.env.pars.horizon + 1))
#                         # axs[k, j].imshow(view, vmax=vmax, vmin=vmin)
#                 else:
#                     state = allo_states[0]
#                     for d in range(4):

#                         if len(axs.shape) == 2:
#                             model.env.get_egocentric_view(
#                                 model.env.world, model.env.get_2d_pos(state),
#                                 d, display=True, ax=axs[
#                                     k + (d // 2) * num_rows / 2, 0 + (
#                                                 d % 2) * num_cols // 2])
#                     else:
#                         for d in range(4):
#                             model.env.get_egocentric_view(
#                                 model.env.world, model.env.get_2d_pos(state),
#                                 d, display=True,
#                                 ax=axs[k + (d // 2) * num_rows / 2])

#             for d in range(4):
#                 axs[k + (d // 2) * num_rows // 2, (
#                             d % 2) * num_cols // 2].axis("on")

#                 axs[k + (d // 2) * num_rows // 2, (
#                             d % 2) * num_cols // 2].set_ylabel(
#                     str(round(bin_edges[i], 2)) + " - " + str(
#                         round(bin_edges[i + 1], 2)), fontsize='xx-small')
#                 axs[(d // 2) * num_rows // 2, (d % 2) * num_cols // \
#                                                    2].set_title(
#                     f"d = {d}")
#                 # axs[k, 0].set_ylabel(
#                 #     str(int(bin_edges[i])) + " - " + str(
#                 #         int(bin_edges[i + 1])))
#                 axs[k, 0].xaxis.set_visible(False)
#                 axs[k+(d//2)*num_rows//2, (d%2)*num_cols//2].tick_params(
#                     left=False,
#                                                        labelleft=False)

#             k += 1

#         plt.tight_layout()
#         plt.show()
#     else:
#         raise NotImplementedError


# def plot_allo_basis_in_ego_coords_old(normed_basis, model, local=False):
#     if local:

#         vmax, vmin = np.max(model.env.world), np.min(model.env.world)

#         hist, bin_edges = np.histogram(normed_basis, bins=10)
#         print(hist, bin_edges)
#         for d in range(4):
#             print(d)
#             fig, axs = plt.subplots(
#                 hist.shape[0] - hist.shape[0] // 2,
#                 np.max(hist[hist.shape[0] // 2:]), figsize=(15, 5))
#             if len(axs.shape) == 2:
#                 for k in range(axs.shape[0]):
#                     for j in range(axs.shape[1]):
#                         if j != 0:
#                             axs[k, j].axis("off")

#             fig.text(0.0, 1.0, f"d = {d}", ha='center')

#             k = 0

#             for i in range(hist.shape[0] - 1, hist.shape[0] // 2 - 1, -1):
#                 if hist[i]:
#                     allo_states = np.where(
#                         np.logical_and(
#                             normed_basis > bin_edges[i],
#                             normed_basis <= bin_edges[i + 1]))[0]
#                     if len(allo_states) > 1:

#                         for j in range(len(allo_states)):
#                             state = allo_states[j]
#                             model.env.get_egocentric_view(
#                                 model.env.world,
#                                 model.env.get_2d_pos(state), d,
#                                 display=True, ax=axs[k, j])

#                             # view = np.reshape(
#                             #     model.env.get_egocentric_view(
#                             #         model.env.world,
#                             #         model.env.get_2d_pos(state), d)[1],
#                             #     (model.env.pars.horizon + 1,
#                             #      2 * model.env.pars.horizon + 1))
#                             # axs[k, j].imshow(view, vmax=vmax, vmin=vmin)
#                     else:
#                         state = allo_states[0]
#                         if len(axs.shape) == 2:
#                             model.env.get_egocentric_view(
#                                 model.env.world, model.env.get_2d_pos(state),
#                                 d, display=True, ax=axs[k, 0])
#                         else:
#                             model.env.get_egocentric_view(
#                                 model.env.world, model.env.get_2d_pos(state),
#                                 d, display=True, ax=axs[k])

#                 if len(axs.shape) == 2:
#                     axs[k, 0].set_ylabel(
#                         str(round(bin_edges[i], 2)) + " - " + str(
#                             round(bin_edges[i + 1], 2)))
#                     # axs[k, 0].set_ylabel(
#                     #     str(int(bin_edges[i])) + " - " + str(
#                     #         int(bin_edges[i + 1])))
#                     axs[k, 0].xaxis.set_visible(False)
#                     axs[k, 0].tick_params(left=False, labelleft=False)
#                 else:
#                     axs[k].set_ylabel(
#                         str(round(bin_edges[i], 2)) + " - " + str(
#                             round(bin_edges[i + 1], 2)))
#                     # axs[k].set_ylabel(
#                     #     str(int(bin_edges[i])) + " - " + str(
#                     #         int(bin_edges[i + 1])))
#                     axs[k].xaxis.set_visible(False)
#                     axs[k].tick_params(left=False, labelleft=False)

#                 k += 1

#             plt.tight_layout()
#             plt.show()
#     else:
#         raise NotImplementedError


# def plot_ego_basis_in_allo_coords(basis, model, local=False, cmap=None):
#     """
#     Args:
#         basis: array of shape (N_ego, ) which designates the basis' weight at each egocentric state.
#         model: agent
#         local: (bool)
#     Returns:

#     """

#     array = np.zeros((model.env.size, model.env.size, 4))
#     for x in range(model.env.size):
#         for y in range(model.env.size):
#             if model.env.world[x, y] <= 0:
#                 for d in range(4):
#                     egocentric_state = model.env.allo_to_ego[-1][(x, y, d)]
#                     array[x, y, d] += basis[egocentric_state]
#     show_2x2(array, cmap=cmap)


# def average_ego_basis(model, SR_sas=None):
#     if SR_sas is not None:
#         model.ego_SR.SR_sas = SR_sas


#     if model.env.pars.local:
#         raise NotImplementedError
#     else:
#         basis = np.max(model.ego_SR.SR_sas, 1)
#         basis = np.mean(basis, 0)

#         plot_ego_basis_in_allo_coords(basis, model, local=False)


# def average_allo_basis(model, SR_sas=None):
#     if SR_sas is not None:
#         model.allo_SR.SR_sas = SR_sas

#     print(len(model.env.ego_bins))

#     if model.env.pars.local:
#         raise NotImplementedError
#     else:
#         basis = np.mean(
#             [model.allo_SR.SR_sas[i] for i in range(
#                 model.env.allo_dim) if
#              model.env.world[model.env.get_2d_pos(i)[0],
#              model.env.get_2d_pos(i)[
#                  1]] <= 0], 0)

#         plot_allo_basis_in_allo_coords(basis, model)

# def knierim_plot(model, world=None, SR_index = 0, path = None):
#     if world is None:
#         world = model.env.world
#     fig = plt.figure(figsize=(10, 5))
    
#     grids = fig.add_gridspec(1, 2)
    
#     # quiver plot of SR
    
#     X, Y = np.meshgrid(np.arange(world.shape[1]), np.arange(world.shape[0]))
#     U = np.zeros_like(X)
#     V = np.zeros_like(Y)
#     linewidths = np.zeros_like(X)
#     head_widths = np.zeros_like(X)
#     head_lengths = np.zeros_like(X)
    
#     denominators = [0 for _ in range(model.ego_dim)]
#     for index_ in range(model.ego_dim):
#         ego_SR = model.ego_SR.SR_ss[index_]
#         for state in range(len(denominators)):
#             denominators[state] += ego_SR[state]
            
            
#     length_scale = 1.2
#     width_scale = 0.094
#     head_width_scale = 0.94
#     head_length_scale = 0.03
#     temp = 7
#     for i in range(world.shape[0]):
#         for j in range(world.shape[1]):
#             if world[i, j] <= 0:
#                 ego_states = [model.env.get_ego_obs([i, j], d) for d in range(4)]
#                 SR_values_normed = [model.ego_SR.SR_ss[SR_index][state]/denominators[state] for state in ego_states] #rescale this??
#                 max_d = np.argmax(SR_values_normed)
#                 max_SR = SR_values_normed[max_d] / np.sum(SR_values_normed)
#                 max_SR_2 = SR_values_normed[max_d] - np.min(SR_values_normed)
#                 max_SR_3 = max_SR - np.min(SR_values_normed)/np.sum(SR_values_normed)
#                 # max_SR = SR_values_normed[max_d]
#                 U[i, j] = np.cos(max_d * np.pi / 2) 
#                 V[i, j] = np.sin(max_d * np.pi / 2)  
                
#                 head_widths[i, j] = np.exp(temp*max_SR)
                    
    
#     head_widths = head_widths / np.max(head_widths)
    
    
#     ax = fig.add_subplot(grids[0, 0])
    
#     masked_array = np.ma.masked_where(world == -1, world)
#     cmap = mcm.Greys
#     cmap.set_bad(color='white')

#     ax.imshow(masked_array, cmap=cmap)
    
    
#     base_length = 0.
#     base_head_length = 0.4
#     base_head_width = 0.4
    
#     for i in range(world.shape[0]):
#         for j in range(world.shape[1]):
#             if U[i, j] != 0 or V[i, j] != 0:
#                 # Compute the arrow properties
#                 dx = U[i, j]*(head_widths[i, j]*length_scale+base_length)
#                 dy = V[i, j]*(head_widths[i, j]*length_scale+base_length)
#                 magnitude = np.hypot(dx, dy)

#                 # Adjust arrow style parameters
#                 arrow_style = patches.ArrowStyle.Simple(
#                     head_length=base_head_length + 0.6 * magnitude * head_widths[i, j],
#                     head_width=base_head_width + 0.8 * magnitude * head_widths[i, j],
#                 )

#                 arrow = patches.FancyArrowPatch(
#                     (X[i, j], Y[i, j]),
#                     (X[i, j] + dx, Y[i, j] + dy),
#                     arrowstyle=arrow_style,
#                     mutation_scale=10 * magnitude * head_widths[i, j],  # Scale mutation for visibility
#                     color='r',
#                     linewidth=linewidths[i, j]
#                 )
#                 ax.add_patch(arrow)
                
#     ax.set_title('Ego SR')
    
#     add_egocentric_SR_grid(grids[0, 1], fig, model, SR_index, coords=[-1, 0], proportion=Fraction(1,4))

#     if path:
#         plt.savefig(path / f'SR_quiver_{SR_index}.png')
#     else:
#         plt.savefig(f'SR_quiver_{SR_index}.png')
        
#     plt.close()