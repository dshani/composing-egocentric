"""
@author: Daniel Shani
"""
import json
import os

from helper_functions_ import make_directories
from parameters import parameters, get_parameter_parser
from run_model import train_agent

if __name__ == '__main__':

    parser = get_parameter_parser()
    args, _ = parser.parse_known_args()
    for arg in vars(args):
        if getattr(args, arg) is not None:
            parameters[arg] = getattr(args, arg)
    parameters.compare = 'lesion'

    if parameters.ego_q_learning:
        parameters.egamma = 0
        parameters.SR_lr_e = 0

    print("Making directories")
    SEED_PATH = make_directories(seed=parameters.seed, pars=parameters, comparison=parameters.compare, job_id=args.job_id)
    print(f"Run Path: {str(SEED_PATH.parent)}")
    print(f"Seed Path: {str(SEED_PATH)}")
    print(f"Arguments: {args}")

    if args.job_id is not None:
        os.makedirs(f'./Results/job_ids/{str(int(args.job_id) + 1)}', exist_ok=True)
        run_path_file = f'./Results/job_ids/{str(int(args.job_id) + 1)}/run_path.txt'
        args_path = f'./Results/job_ids/{str(int(args.job_id) + 1)}/args.txt'
        params_path = f'./Results/job_ids/{str(int(args.job_id) + 1)}/params.txt'
        code_path = f'./Results/job_ids/{str(int(args.job_id) + 1)}/code/'

        os.makedirs(code_path, exist_ok=True)
        # save all but Results and slurm folders and images:
        os.system(f'rsync -a --exclude="Results" --exclude="slurm" --exclude="*.png" --exclude="*.jpg" --exclude="*.ipynb" ../* {code_path}')

        with open(run_path_file, 'w') as f:
            f.write(str(SEED_PATH.parent.absolute()))

        with open(args_path, 'w') as f:
            # dump all arguments except seed and job_id
            json.dump({k: v for k, v in args.__dict__.items() if (k not in ['seed', 'job_id'] and v is not None)}, f)

        with open(params_path, 'w') as f:
            # dump all parameters
            json.dump(parameters, f)

    if args.lesion_only:
        parameters.lesionLEC = True
        path = SEED_PATH.joinpath("lesionLEC")
        train_agent(parameters, save=True, seed_path=path)
    elif args.unlesion_only:
        parameters.lesionLEC = False
        path = SEED_PATH.joinpath("unlesioned")
        train_agent(parameters, save=True, seed_path=path)
        
    elif args.knierim:
        print("Running Knierim comparison")
        parameters.lesionLEC = False
        parameters.worlds_type = "knierim"
        path = SEED_PATH.joinpath("unlesioned")
        train_agent(parameters, save=True,
                    seed_path=path)
        
        
    else:
        print("Running lesion comparison")

        
        if args.LEC_only:
            path = SEED_PATH.joinpath("lesionMEC")
            parameters.lesionLEC = False
            parameters.lesionMEC = True
            train_agent(parameters, save=True, seed_path=path)
            print("MEC lesioned done")
        
        parameters.lesionLEC = False
        parameters.lesionMEC = False
        path = SEED_PATH.joinpath("unlesioned")
        train_agent(parameters, save=True, seed_path=path)
        print("Unlesioned done")
        parameters.lesionLEC = True
        path = SEED_PATH.joinpath("lesionLEC")
        train_agent(parameters, save=True, seed_path=path)
        print("Lesioned done")
