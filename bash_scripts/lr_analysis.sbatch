#!/bin/bash
#
#SBATCH --job-name=lr_analysis
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

# Validate RUN_ID and locate run_path.txt
if [ -z "$RUN_ID" ]; then
  echo "RUN_ID is not set. This script must be launched with --export=RUN_ID=<job_id>." >&2
  exit 1
fi

# Primary location (matches make_directories: job_ids/{job_id+1})
JOB_DIR_PLUS_ONE=$((RUN_ID + 1))
RUN_PATH_FILE_P1="./Results/job_ids/${JOB_DIR_PLUS_ONE}/run_path.txt"
RUN_PATH_FILE="./Results/job_ids/${RUN_ID}/run_path.txt"

if [ -f "$RUN_PATH_FILE_P1" ]; then
  run_path=$(< "$RUN_PATH_FILE_P1")
elif [ -f "$RUN_PATH_FILE" ]; then
  run_path=$(< "$RUN_PATH_FILE")
else
  echo "Could not find run_path.txt in either: $RUN_PATH_FILE_P1 or $RUN_PATH_FILE" >&2
  echo "Available job_ids directories:" >&2
  ls -1 ./Results/job_ids | tail -n 50 >&2
  exit 1
fi

if [ ! -d "$run_path" ]; then
  echo "Resolved run_path does not exist: $run_path" >&2
  exit 1
fi

python -u ../analyse_nonadaptive_comparison.py --run_path "$run_path" --sigma 1 $EXTRA_ARGS
