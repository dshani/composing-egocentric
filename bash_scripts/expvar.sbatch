#!/bin/bash
#
#SBATCH --job-name=barrier
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=16G  #memory
#SBATCH -t 3:00:00
#SBATCH -o ./slurm/slurm.%A_%a.out
#SBATCH -e ./slurm/slurm.%A_%a.err
#SBATCH --array 0-29
#
hostname; date

source ~/.bashrc  #this was because conda wasn't a command for me
micromamba activate new_ego

sleep $((10*SLURM_ARRAY_TASK_ID))

# Optional environment variables:
#   PARAM_NAME / PARAM_VALUE - override the parameter swept in the run
#   SAVE_PARAMS              - override the list passed to --save_params (default matches
#                              the historical behaviour of persisting SR weights, moments,
#                              and paths)

# Use PARAM_NAME and PARAM_VALUE passed from the environment
PARAM_NAME=${PARAM_NAME:-"worlds_type"}
PARAM_VALUE=${PARAM_VALUE:-"random_rooms"}

# Allow overriding the list of parameters saved during training
SAVE_PARAMS=${SAVE_PARAMS:-"['weight','m','v','value_snapshots','paths']"}

# Construct the parameter string for the Python script
PYTHON_PARAM="--${PARAM_NAME}=${PARAM_VALUE}"

python -u ../run_experiments.py ${PYTHON_PARAM} --lr="0.5" --save_params="${SAVE_PARAMS}" --agamma="0.94" --egamma="0.98" --gamma="0.98" --SR_lr_a="0.0078125" --SR_lr_e="0.035" --save_every="500" --log_every="5" --env_switch_every="1000" --num_episodes="5000" --max_steps="50000" --seed=$SLURM_ARRAY_TASK_ID --job_id=$SLURM_ARRAY_JOB_ID --SR_init="'empty'"
