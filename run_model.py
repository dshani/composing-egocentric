"""
@author: Daniel Shani
"""
import ast
import os
import itertools
import pickle
import random
from collections import deque
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np


import gridworlds
import gridworlds as gd
from agent import BasisLearner
from environment_functions import Environment
from helper_functions_ import make_directories, make_logger, rgetattr
from parameters import get_parameter_parser
from parameters import parameters

from plotting_functions import knierim_plot
from structure_functions_ import get_value_functions


def _as_iterable(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return value
    return [value]


def _normalise_save_params(pars):
    """Return ``pars.save_params`` as a concrete list.

    ``save_params`` can reach ``train_agent`` either as an actual iterable or as
    a string originating from an environment variable.  The original training
    loop assumed the iterable form and iterated directly over the attribute
    which, when given a string, produced per-character dictionary keys such as
    ``'['`` or ``'w'``.  To keep the rest of the code agnostic to how the
    parameter was specified, we canonicalise it here.
    """

    raw = getattr(pars, "save_params", None)
    if raw is None:
        return []

    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return [raw]
        else:
            raw = parsed

    if isinstance(raw, (list, tuple, set)):
        return list(raw)

    return [raw]


def _build_param_save_schedule(pars):
    schedule_cfg = getattr(pars, 'save_param_schedule', None)
    env_switch_every = getattr(pars, 'env_switch_every', None) or 0
    total_episodes = getattr(pars, 'num_episodes', None)
    total_episodes = int(total_episodes) if total_episodes is not None else None

    # When no explicit schedule is provided, generate a minimal default that
    # captures snapshots around environment switches. This supports downstream
    # correlation analyses which expect pre/post switch timepoints.
    if not schedule_cfg:
        if env_switch_every > 0 and total_episodes and total_episodes > 0:
            default_params = {'weight', 'ego_SR.SR_ss', 'allo_SR.SR_ss', 'value_snapshots'}
            requested_params = set(_as_iterable(getattr(pars, 'save_params', [])))
            target_params = default_params & requested_params
            if target_params:
                schedule: dict[str, set[int]] = {}
                # Offsets chosen to match analysis pre/post windows
                offsets = (-10, 10)
                switch_count = total_episodes // env_switch_every
                for param in target_params:
                    episodes: set[int] = set()
                    for switch_index in range(1, switch_count + 1):
                        base_episode = switch_index * env_switch_every
                        for offset in offsets:
                            target = base_episode + int(offset)
                            if 0 <= target < total_episodes:
                                episodes.add(target)
                    if episodes:
                        schedule[param] = episodes
                if schedule:
                    return schedule
        return {}

    if not isinstance(schedule_cfg, dict):
        raise TypeError(
            "save_param_schedule must be a dictionary mapping parameter names"
            " to configuration dictionaries or iterables of episode indices"
        )

    # Custom schedule provided by user/config
    env_switch_every = getattr(pars, 'env_switch_every', None) or 0
    total_episodes = getattr(pars, 'num_episodes', None)
    total_episodes = int(total_episodes) if total_episodes is not None else None

    schedule = {}
    for param, config in schedule_cfg.items():
        if config is None:
            continue

        episodes = set()

        if isinstance(config, dict):
            explicit = []
            for key in ('episodes', 'timesteps', 'absolute'):
                if key in config and config[key] is not None:
                    explicit.extend(_as_iterable(config[key]))

            for value in explicit:
                episodes.add(int(value))

            offsets = config.get('env_switch_offsets')
            if offsets is not None:
                if env_switch_every <= 0:
                    raise ValueError(
                        "save_param_schedule specifies 'env_switch_offsets' but"
                        " env_switch_every is not positive"
                    )
                if total_episodes is None or total_episodes <= 0:
                    raise ValueError(
                        "save_param_schedule requires a positive num_episodes"
                        " when using 'env_switch_offsets'"
                    )

                switch_count = total_episodes // env_switch_every
                for switch_index in range(1, switch_count + 1):
                    base_episode = switch_index * env_switch_every
                    for offset in _as_iterable(offsets):
                        target = base_episode + int(offset)
                        if 0 <= target < total_episodes:
                            episodes.add(target)
        else:
            for value in _as_iterable(config):
                episodes.add(int(value))

        if episodes:
            schedule[param] = episodes

    return schedule


def _compute_directional_value_snapshot(agent):
    """Return a copy of the directional-mean value grid for ``agent``."""

    state_values, *_ = get_value_functions(agent, split=True)
    mean_grid = np.mean(state_values, axis=2)
    return np.array(mean_grid, dtype=np.float32, copy=True)


def train_agent(pars, save=False, plot=False, seed_path=None):
    """
    Main function for the BasisLearner.
    3 Main Steps:

    0. Make directories

    1. Initialize Environment
        Creates an environment from a gridworld.
        Calculates the transition matrix and basis functions.
        There are two sets of basis functions: one using the allocentric
        view and one using the egocentric view.
        The basis functions are the state-action successor representations,
        defined as M[s'|s,a] = I[s'=s] + gamma M[s'|z(s,a)],
        where z(s,a) is the state that comes after state s and action a.
        The allocentric basis function ignores barriers and
        therefore uses a flat SR for an empty grid.

    2. Initialize Agent
        Creates an agent.
        The agent uses linear function approximation with
        the two sets of basis in order to approximate the Q-function.
        The agent uses a softmax policy to select actions.
        After each step the agent updates its Q-function.
        There is the option of using an adaptive LR -
        it is currently turned off.

    3. Run Agent
        At the beginning of each episode, the agent is started in a
        random location.
        It receives its allocentric and egocentric state representations and
        uses them to act.
        When the agent chooses the action that takes it into the
        reward state, it receives its reward.
        After the episode, the path is logged and the weights are saved.
        The agent is then reset and the episode starts over.

    Args:
        pars (DotDict): parameters - see parameters.py
        save (bool): whether to save the weights

    Returns:
        total_steps (list): list of average steps per episode at each log step
    """
    if seed_path is None:
        seed_path = make_directories(pars=pars)

    if pars.seed is not None:
        np.random.seed(pars.seed)
        random.seed(pars.seed)
    world_type = pars.worlds_type.strip("'")

    total_steps = []
    weight = None
    verbose = pars.verbose
    steps = []
    heatmaps = []
    msg = "Creating Environment"

    print(msg)

    if world_type in ("full_resample", "random_rooms7"):
        worlds = [
            gd.grid_rep_color,
            gd.grid_rep_color_3,
            gd.grid_rep_color_4,
            gd.grid_rep_color_5,
            gd.grid_rep_color_6,
        ]
        

    # aligned barriers
    elif world_type == "random_rooms0":
        worlds = [gd.generate_random_gridworld_(pars.size, N=20, reward=["tr", "br"][i%2], align=[[1, 0], [3,0]][i%2]) for i in range(5)]
        
    # changing barrier sizes, aligned
    elif world_type == "random_rooms2":
        worlds = [gd.generate_random_gridworld_(pars.size, N=20, reward=["tr", "br"][i%2], align=[[1, 0], [3,0]][i%2], width = [3,4,5]) for i in range(5)]
        
    # changing barrier sizes, unaligned
    elif world_type == "random_rooms3":
        worlds = [gd.generate_random_gridworld_(pars.size, N=20, reward=["tr", "br"][i%2], align=False, width = [3,4,5]) for i in range(5)]


    elif world_type in {"random_rooms12", "randomrooms12"}:
        worlds = [gd.generate_scattered_obstacles_grid(pars.size, reward=["tr", "br"][i % 2]) for i in range(5)]
        
    elif world_type == "random_rooms13":
        worlds = [gd.generate_random_gridworld_(pars.size, N=20, reward="br", align=[3,0])] + [gd.generate_random_gridworld_(pars.size, N=20, reward=["tr", "br"][i%2], align=False) for i in range(4)]
        
    else:
        raise ValueError(f"Unknown worlds_type: {world_type}")

    

    tangibles = [True for _ in range(5)]
    if pars.tangibles:
        for i in pars.tangibles:
            tangibles[i] = False


    world_index = 0
    tan_index = 0
    # world = gd.grid00
    world = worlds[world_index]
    env = Environment(pars=pars, gridworld=world)
    if plot:
        env.show_world()

    use_precomputed_starts = getattr(pars, "precompute_reset_starts", True)

    def _build_reset_queue():
        rng = np.random.default_rng(pars.seed)
        saved_state = np.random.get_state()
        planning_env = Environment(pars=pars, gridworld=worlds[0])
        np.random.set_state(saved_state)

        def sample_start(plan_env):
            while True:
                if plan_env.pars.random_starts:
                    position_index = int(rng.integers(plan_env.size ** 2))
                    position_2d = plan_env.allo_bins[position_index]
                    direction = int(rng.integers(4))
                else:
                    position_2d = [1, 1]
                    direction = 0
                if plan_env.world[position_2d[0], position_2d[1]] == 0:
                    return tuple(position_2d), direction

        queue_entries = []

        queue_entries.append(sample_start(planning_env))

        if pars.env_switch:
            if pars.tangibles:
                for index, tangible in enumerate(tangibles):
                    planning_env.switch_world(
                        worlds[index % len(worlds)], tangible=tangible, switch_SRs=True)
                    queue_entries.append(sample_start(planning_env))
                planning_env.switch_world(
                    worlds[0], tangible=tangibles[0], switch_SRs=True)
                queue_entries.append(sample_start(planning_env))
            else:
                for grid in worlds:
                    planning_env.switch_world(grid, switch_SRs=True)
                    queue_entries.append(sample_start(planning_env))
                planning_env.switch_world(worlds[0], switch_SRs=True)
                queue_entries.append(sample_start(planning_env))

        world_index = 0
        tan_index = 0
        for episode in range(pars.num_episodes):
            if pars.env_switch and episode % pars.env_switch_every == 0 and episode > 0:
                world_index = (world_index + 1) % len(worlds)
                tan_index = (tan_index + 1) % len(tangibles)
                planning_env.switch_world(
                    worlds[world_index], tangible=tangibles[tan_index], switch_SRs=False)
                queue_entries.append(sample_start(planning_env))
            queue_entries.append(sample_start(planning_env))

        np.random.set_state(saved_state)
        return deque(queue_entries)

    if use_precomputed_starts:
        env.reset_queue = _build_reset_queue()
    msg = "Creating Agent"
    # env.show_world()

    if plot:
        env.produce_schematic()

    print(msg)

    pars.save_params = _normalise_save_params(pars)

    agent = BasisLearner(env, pars=pars)
    param_save_schedule = _build_param_save_schedule(pars)

    if agent.pars.env_switch:
        if pars.tangibles:
            for index, tan_ in enumerate(tangibles):
                world_ = worlds[index % len(worlds)]
                agent.switch_world(world_, tan_)
            agent.switch_world(worlds[0], tangibles[0])
        else:
            for world in worlds:
                agent.switch_world(world)
            agent.switch_world(worlds[0])

    max_size = max(world.shape[0] for world in worlds)
    if "heatmaps" in pars.save_params:
        agent.set_heatmap(max_size)
    if "paths" in pars.save_params and pars.save_paths:
        agent.collect_paths = True

    if save:
        print("Saving in directory: ", seed_path.absolute())
        logger_sums = make_logger(seed_path, 'summaries')

        save_model_flag = getattr(pars, 'save_model', True)

        if save_model_flag:
            filename = seed_path.joinpath('model', 'model.sav')
            with open(filename, 'wb') as file:
                pickle.dump(agent, file)

            env_path = seed_path.joinpath('worlds')
            filename = env_path.joinpath('worlds.sav')
            with open(filename, 'wb') as file:
                pickle.dump(worlds, file)
            filename = env_path.joinpath('pars.sav')
            with open(filename, 'wb') as file:
                pickle.dump(pars, file)
                
        else:
            print("Not saving model")

        save_dict = {}
        for param in pars.save_params:
            save_dict[param] = []
        save_dict["accuracies"] = []
        # Maintain a consolidated list of accuracies for quick loading later
        accuracies_all = []


        
    if world_type == "knierim":
        os.mkdir(seed_path.joinpath('knierim'))
        path = seed_path.joinpath('knierim')
        
        for state in range(agent.env.ego_dim):
            knierim_plot(agent, path=path, SR_index=state)
        return 0

    for episode in range(agent.pars.num_episodes):

        if agent.pars.env_switch and episode \
                % agent.pars.env_switch_every == 0 and episode > 0:
            if verbose:
                msg = "Switching Environment"
                print(msg)
            if save:
                logger_sums.info(msg)
            world_index = (world_index + 1) % len(worlds)
            tan_index = (tan_index + 1) % len(tangibles)
            if plot:
                plt.imshow(worlds[world_index])
                plt.show()

            agent.switch_world(worlds[world_index], tangibles[tan_index], switch_SRs=False)

        # agent.env.switch_world_less(worlds[world_index])

        # allo_dim, ego_dim = agent.env.switch_world(worlds[world_index])
        # agent.update_sizes(allo_dim, ego_dim)

        if agent.pars.regression_test:
            if episode > agent.pars.regression_test_episodes:
                break

        if pars.regress:
            if weight is not None:
                print(f"w_diff:{np.linalg.norm(weight - agent.weight)}")

        agent.reset()
        if verbose:
            print(f"Start: {list(agent.env.position_2d)}")

        fwd_count = 0

        for i in itertools.count():

            next_state, next_ego, direction, _, done, action = \
                agent.train_step((episode + 1) > agent.pars.SR_burnin)
            if action == 0:
                fwd_count += 1

            if done or i > agent.pars.max_steps:

                if verbose:
                    print(f"End: {agent.env.position_2d}")
                steps.append(fwd_count)

                if "heatmaps" in pars.save_params:
                    heatmaps.append(deepcopy(agent.heatmap))
                    agent.clear_heatmap(max_size)
                if "paths" in pars.save_params and pars.save_paths:
                    agent.paths.append([])
                break


        if (episode + 1) % agent.pars.update_SR_every == 0 and \
                (episode + 1) > agent.pars.SR_burnin:
            agent.update_SR()
            msg = f"Updated SR"
            if verbose:
                print(msg)
            if save:
                logger_sums.info(msg)

        ### SAVING ###

        log_event = save and ((episode + 1) % agent.pars.log_every == 0)
        regression_event = agent.pars.regression_test
        log_trigger = log_event or regression_event

        param_should_save = {}
        schedule_triggered = False
        for param in pars.save_params:
            if param in param_save_schedule:
                schedule_set = param_save_schedule[param]
                should_save = episode in schedule_set
                if should_save:
                    schedule_triggered = True
                    schedule_set.discard(episode)
                param_should_save[param] = should_save
            else:
                param_should_save[param] = log_trigger

        if log_trigger or schedule_triggered:
            value_snapshot_cache = None

            if log_trigger:
                msg = f"Episode:{episode + 1}, Average Steps: {np.mean(steps)}"
                logger_sums.info(msg)
                print(msg)

                avg_steps = deepcopy(np.mean(steps))
                save_dict["accuracies"].append((deepcopy(episode), avg_steps))
                if save:
                    accuracies_all.append((deepcopy(episode), avg_steps))
                    np.savez(seed_path.joinpath("save_dict", "save_dict_accuracies.npz"),
                             accuracies=np.array(accuracies_all, dtype=object))
                steps = []

                if "heatmaps" in pars.save_params:
                    save_dict.setdefault("heatmaps", []).append(
                        (deepcopy(episode), deepcopy(np.mean(heatmaps, axis=0)))
                    )
                    heatmaps = []

                if "paths" in pars.save_params and pars.save_paths:
                    save_dict.setdefault("paths", []).append(
                        (deepcopy(episode), deepcopy(agent.paths[:-1]))
                    )
                    agent.clear_paths()

            for param in pars.save_params:
                if not param_should_save.get(param):
                    continue
                bucket = save_dict.setdefault(param, [])

                if param == "value_snapshots":
                    if value_snapshot_cache is None:
                        value_snapshot_cache = _compute_directional_value_snapshot(agent)
                    bucket.append(
                        (deepcopy(episode), np.array(value_snapshot_cache, copy=True))
                    )
                    continue

                attr_value = rgetattr(agent, param)
                if attr_value is not None:
                    bucket.append((deepcopy(episode), deepcopy(attr_value)))

        # print(np.array(save_dict['weight']).shape)

        if (save and (episode + 1) % agent.pars.save_every == 0) or \
                agent.pars.regression_test:

            msg = f"Saving {save_dict.keys()}"
            print(msg)
            np.save(seed_path.joinpath("save_dict", "save_dict_" + str(episode + 1)), save_dict, allow_pickle=True)
            save_dict = {}
            for param in pars.save_params:
                save_dict[param] = []
            save_dict["accuracies"] = []



    if save:
        np.savez(seed_path.joinpath("save_dict", "save_dict_accuracies.npz"),
                 accuracies=np.array(accuracies_all, dtype=object))

    msg = "Training Finished"
    if verbose:
        print(msg)
    if save:
        logger_sums.info(msg)
    return total_steps


if __name__ == '__main__':

    parser = get_parameter_parser()
    args, _ = parser.parse_known_args()
    for arg in vars(args):
        if getattr(args, arg) is not None:
            parameters[arg] = getattr(args, arg)

    if parameters.ego_q_learning:
        parameters.egamma = 0
        parameters.SR_lr_e = 0

    parameters.compare = None

    seed_path = make_directories(seed=parameters.seed, pars=parameters,
                                 comparison=parameters.compare)
    # print(parameters)
    train_agent(parameters, save=True, plot=True, seed_path=seed_path)
