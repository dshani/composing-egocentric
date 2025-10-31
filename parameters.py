"""
@author: Daniel Shani
"""

import argparse
import ast

import numpy as np

from helper_functions_ import DotDict

parameters = {

    # Experiments:
    "normalize": False,
    "worlds_type": "full_resample" ,  # "rotation", "resampled_barriers", "full_resample"
    # Sparsity Parameters (see Langford et al. 2009)
    "grav": 0.,  # 1.0e-5
    "theta": 0.,  # 3e-4
    "rounding_param": 100,
    "SR_init": "empty",  # "empty", "full"
    # RL Parameters
    "agamma": 0.94,  ## allocentric SR gamma
    "egamma": 0.98,  # too high leads to less smooth value function ## egocentric SR gamma
    "gamma": 0.99,  ## value function gamma
    "lr": 0.5,
    "SR_lr_a": 0.001,  # 0.1
    "SR_lr_e": 0.001,  # 0.1
    "temperature": 100,  # softmax temperature
    "explore_param": 0.01,
    "forgetting_param": 1.,  # 0.9999
    "horizon": 2,  # 5 (2 local)

    "update_SR_every": 100,  # 100
    "SR_burnin": 200,

    # Train Parameters
    "save_every": 500,
    "log_every": 5,
    "num_episodes": 10000,
    "max_steps": 50000,
    "env_switch": True,
    "env_switch_every": 1000,  # 1000
    "save_model": True,

    # Environment Parameters
    "size": 20,
    "step_cost": 0.,
    "end_reward": 1.0,
    "wall_cost": 0.0,
    "random_start": True,
    "reward_location": "corner",  # "middle" / "corner"
    "random_starts": True,
    "random_reward_location": False,
    "reward_switch": 1000,
    "see_reward": False,
    # Start-state handling
    "precompute_reset_starts": True,

    # Adaptive LR -- Adam (beta1 = 0.9, beta2 = 0.999, eps = 1e-8) or RMSProp
    # (beta1 = 0.9, beta2 = 0, eps = 1e-6)
    "eta": 5e-06,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1.0e-8,

    "lesionLEC": False,
    "lesionMEC": False,
    "regress": False,

    "local": True,
    "seed": 1,
    "regression_test": False,
    "regression_test_episodes": 3,
    "new_allo": True,
    "transparent": False,
    "opacity": 1,  # integer between 0 and 3

    # "save_params": ["weight", "allo_SR.SR_sas", "ego_SR.SR_sas", "allo_SR.SR_ss", "ego_SR.SR_ss", "grad", "heatmaps",
    #                 "paths"],
    # ``value_snapshots`` stores compact directional-mean value grids for the
    # requested episodes and can be combined with ``param_time_slices`` to limit
    # storage to a handful of checkpoints required for the correlation plots.
    "save_params": ["weight"],
    "save_paths": True,
    "save_param_schedule": None,
    "compare": 'lesion',  # "lesion", "gamma", None
    "verbose": False,
    "tangibles" : None,
    "ego_q_learning": False,
}


def get_parameter_parser():
    parser = argparse.ArgumentParser(description='Parameters')
    for key in parameters.keys():
        # multiple inputs are added as list
        if key == 'ego_q_learning':
            parser.add_argument('--ego_q_learning', action='store_true')
        else:
            parser.add_argument('--' + key, type=ast.literal_eval, default=None)

    parser.add_argument('--job_id', type=str, default=None)
    parser.add_argument('--lesion_only', type=bool, default=False)
    parser.add_argument('--unlesion_only', type=bool, default=False)
    parser.add_argument('--knierim', type=bool, default=False)
    parser.add_argument('--LEC_only', type=bool, default=False)

    


    return parser


parameters = DotDict(parameters)
