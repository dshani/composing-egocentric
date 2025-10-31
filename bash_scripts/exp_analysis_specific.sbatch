#!/bin/bash
#
#SBATCH --job-name=analysis
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

# python -u ../run_analysis_.py --job_id=1265388 --full_plots=True --single_seed=1
# python -u ../run_analysis_.py --job_id=1297791 --full_plots=True --single_seed=1 
# python -u ../run_analysis_.py --job_id=1298019 --full_plots=True --single_seed=1 
# python -u ../run_analysis_.py --job_id=1301821 --full_plots=True --single_seed 2 --delay 46
python -u ../run_analysis_.py --job_id=1301524 --full_plots=True --single_seed 2 --delay 46 --generalisation_only=True
