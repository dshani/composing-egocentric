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

from figure_functions_ import *

from structure_functions_ import *

from run_analysis_ import get_analysis_parser
if __name__ == '__main__':
    

    parser = get_analysis_parser()
    args0, _ = parser.parse_known_args()
    

    for arg in vars(args0):
        if getattr(args0, arg) is not None:
            parameters[arg] = getattr(args0, arg)
    parameters.compare = 'lesion'

    if args0.save_dirs is None:
        save_dirs = [pathlib.Path("Results").resolve()]
    else:
        save_dirs = [pathlib.Path(save_dir).resolve()
                     for save_dir in args0.save_dirs]

    if args0.recent is None:
        args0.recent = -2

    run_path = None
    figure_path = pathlib.Path('.')
    
    

    if args0.job_id is not None:
        run_path_file = f'./Results/job_ids/{str(int(args0.job_id))}/run_path.txt'
        figure_path = pathlib.Path(
            f'./Results/job_ids/{str(int(args0.job_id))}/figures').resolve()
        # set path as the run path
        with open(run_path_file, 'r') as f:
            run_path = f.read()
            run_path = pathlib.Path(run_path).resolve()

        arg_path = f'./Results/job_ids/{str(int(args0.job_id))}/args.txt'
        with open(arg_path, 'r') as f:
            # load all arguments
            new_args = DotDict(json.load(f))

        # set non None arguments of new_args to args0
        for arg in vars(new_args):
            if getattr(new_args, arg) is not None:
                setattr(args0, arg, getattr(new_args, arg))

    if args0.lesion_test:
        struct_all_seeds = load_structure(args0.run, args0.date, args0.seed, save_dirs,
                                        dict_params=['accuracies'], compare='lesion',
                                        seeds_path=run_path,
                                        max_workers=args0.max_workers)
        lesioned_y, lesioned_x = get_parameter_values(
            'accuracies', struct_all_seeds, prefix='lesionLEC')
        accuracy_mean = np.mean(lesioned_y)
        print(f"args0: {new_args}, ", f"accuracy: {accuracy_mean}")
        sys.exit()
    if args0.unlesion_test:
        struct_all_seeds = load_structure(args0.run, args0.date, args0.seed, save_dirs,
                                        dict_params=['accuracies'], compare='lesion',
                                        seeds_path=run_path,
                                        max_workers=args0.max_workers)
        y, x = get_parameter_values(
            'accuracies', struct_all_seeds, prefix='unlesioned')
        accuracy_mean = np.mean(y)
        print(f"args0: {new_args}, accuracy: {accuracy_mean}")
        sys.exit()

    if args0.job_id is None:
        date = args0.date if args0.date is not None else max(
            os.listdir(save_dirs[0]))
        compare = parameters.compare if parameters.compare is not None else 'run'
        run = args0.run if args0.run is not None else \
            find_most_recent(os.listdir(save_dirs[0].joinpath(date)), must_contain=[parameters.compare],
                             recent=-1)[0]

        figure_path = save_dirs[0].joinpath(date, run, "figures")

    t0 = 0
    t1 = 1000
    t2 = 2000
    sigma = 1
    env_switch_every = 1000
    heatmap_path_num = 10
    prefix = 'unlesioned'
    
    if not figure_path.exists():
        figure_path.mkdir(parents=True)

    print(f"figure path: {figure_path}")    
        
    struct_all_seeds = load_structure(args0.run, args0.date, args0.seed, save_dirs,
                                    dict_params=['accuracies'], compare='lesion',
                                    seeds_path=run_path, lesionMEC=True,
                                    max_workers=args0.max_workers)
    struct_all_seeds = remove_empty_dicts(struct_all_seeds)

    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    inputs = get_lesion_values_(struct_all_seeds, sigma=sigma, lesionMEC=True)
    (y_les, y_les_sem, x_les), (y_un, y_un_sem, x_un), (y_mec, y_mec_sem, x_mec) = inputs

    N_les = y_les.shape[0]
    N_un = y_un.shape[0]
    N_mec = y_mec.shape[0]
    

    generate_lesion_plot_(ax, inputs=inputs, labels=['allocentric','allocentric+egocentric', 'egocentric'], env_switch_every=1000)
    fig.savefig(figure_path.joinpath('lesionMEC_comparison.png'))
    plt.close(fig)
    
