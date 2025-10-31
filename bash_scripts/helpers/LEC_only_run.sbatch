#!/bin/bash
#
#SBATCH --job-name=barrier
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

python -u ../run_experiments.py --LEC_only="True" --lr="0.5" --save_params="['weight','m','v','allo_SR.SR_ss','ego_SR.SR_ss', 'paths']" --worlds_type="'full_resample'" --agamma="0.94" --egamma="0.98" --gamma="0.98" --SR_lr_a="0.0078125" --SR_lr_e="0.035" --save_every="500" --log_every="5" --env_switch_every="1000" --num_episodes="5000" --max_steps="50000" --seed=$SLURM_ARRAY_TASK_ID --job_id=$SLURM_ARRAY_JOB_ID --SR_init="'empty'"
