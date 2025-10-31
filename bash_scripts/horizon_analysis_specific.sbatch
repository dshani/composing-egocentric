#!/bin/bash
#
#SBATCH --job-name=horizon_analysis
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=64G  #memory
#SBATCH -t 24:00:00
#SBATCH -o ./slurm/slurm.%j.out
#SBATCH -e ./slurm/slurm.%j.err
#
hostname; date

source ~/.bashrc  #this was because conda wasn't a command for me
micromamba activate new_ego

# Allow overriding the worker count; default to a conservative value
MAX_WORKERS=${MAX_WORKERS:-4}

python -u ../run_analysis_.py --job_id=$SLURM_JOB_ID --full_plots=False --single_seed=None --max_workers=$MAX_WORKERS --metrics_only=True

# Record horizon, room type, and worker settings alongside the generated figures
if [ -n "$HORIZON" ] || [ -n "$WORLDS_TYPE" ] || [ -n "$MAX_WORKERS" ]; then
    mkdir -p ./Results/job_ids/$SLURM_JOB_ID
fi
if [ -n "$HORIZON" ]; then
    echo "$HORIZON" > ./Results/job_ids/$SLURM_JOB_ID/horizon.txt
fi
if [ -n "$WORLDS_TYPE" ]; then
    echo "$WORLDS_TYPE" > ./Results/job_ids/$SLURM_JOB_ID/worlds_type.txt
fi
if [ -n "$MAX_WORKERS" ]; then
    echo "$MAX_WORKERS" > ./Results/job_ids/$SLURM_JOB_ID/max_workers.txt
fi
