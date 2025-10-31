#!/bin/bash
#
#SBATCH --job-name=barrier
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=32G  #memory
#SBATCH -t 24:00:00
#SBATCH -o ./slurm/slurm.%A_%a.out
#SBATCH -e ./slurm/slurm.%A_%a.err
#
hostname; date

source ~/.bashrc  #this was because conda wasn't a command for me
micromamba activate new_ego

sleep $((10*SLURM_ARRAY_TASK_ID))

python -u ../run_experiments.py --lr="0.5" --save_params="[]" --worlds_type="'full_resample'" --agamma="0.94" --egamma="0.98" --gamma="0.98" --SR_lr_a="0.0078125" --SR_lr_e="0.035" --save_every="500" --log_every="5" --env_switch_every="1000" --num_episodes="50" --max_steps="50000" --seed=0 --job_id=88888 --SR_init="'empty'" --save_paths="False" --save_model="False"
