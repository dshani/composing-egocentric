#!/bin/bash
#
#SBATCH --job-name=horizon_exp
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=32G  #memory
#SBATCH -t 24:00:00
#SBATCH -o ./slurm/slurm.%A_%a.out
#SBATCH -e ./slurm/slurm.%A_%a.err
#SBATCH --array 0-29
#
hostname; date

source ~/.bashrc  #this was because conda wasn't a command for me
micromamba activate new_ego

sleep $((10*SLURM_ARRAY_TASK_ID))

# Use PARAM_NAME/PARAM_VALUE and PARAM_NAME2/PARAM_VALUE2 passed from the environment
PARAM_NAME=${PARAM_NAME:-"horizon"}
PARAM_VALUE=${PARAM_VALUE:-"2"}
PARAM_NAME2=${PARAM_NAME2:-"worlds_type"}
PARAM_VALUE2=${PARAM_VALUE2:-random_rooms0}
SR_LR_E=${SR_LR_E:-"0.035"}
EGAMMA=${EGAMMA:-"0.98"}

# Construct the parameter string for the Python script
PYTHON_PARAM="--${PARAM_NAME}=${PARAM_VALUE} --${PARAM_NAME2}=\"'${PARAM_VALUE2}'\""

python -u ../run_experiments.py ${PYTHON_PARAM} --lr="0.5" --save_params="[]" --agamma="0.94" --egamma="${EGAMMA}" --gamma="0.98" --SR_lr_a="0.0078125" --SR_lr_e="${SR_LR_E}" --save_every="500" --log_every="5" --env_switch_every="1000" --num_episodes="5000" --max_steps="50000" --seed=$SLURM_ARRAY_TASK_ID --job_id=$SLURM_ARRAY_JOB_ID --SR_init="'empty'" --save_paths="False" --save_model="False"
