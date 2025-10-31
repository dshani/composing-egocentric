#!/bin/bash
#
#SBATCH --job-name=plot_mean_steps
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o ./slurm/slurm.%j.out
#SBATCH -e ./slurm/slurm.%j.err
#
hostname; date

source ~/.bashrc
micromamba activate new_ego

# Run from the repository layout assuming the working directory is bash_scripts/
# (same convention as other sbatch files in this folder)
PYTHON_BIN="$(command -v python || command -v python3)"
"$PYTHON_BIN" -u ../plot_mean_steps_by_horizon.py --worlds_type "$WORLDS_TYPE"

