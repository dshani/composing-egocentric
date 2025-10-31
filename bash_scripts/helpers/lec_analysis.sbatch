#!/bin/bash
#
#
#SBATCH --job-name=analysis
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=32G  #memory
#SBATCH -t 24:00:00
#SBATCH -o ./slurm/slurm.%j.out
#SBATCH -e ./slurm/slurm.%j.err
#
hostname; date

source ~/.bashrc  #this was because conda wasn't a command for me
micromamba activate new_ego

python -u ../lesionMECplot.py --job_id=$SLURM_JOB_ID

