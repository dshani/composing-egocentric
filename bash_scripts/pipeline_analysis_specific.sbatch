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

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 JOB_ID [ANALYSIS_ARGS...]" >&2
    exit 1
fi

job_id=$1
shift

analysis_entry=${ANALYSIS_SCRIPT:-../analyse_full_pipeline.py}

cmd=(python -u "$analysis_entry" --job_id="$job_id")

if [[ $# -gt 0 ]]; then
    cmd+=("$@")
fi

printf 'INFO: Running analysis with command:\n'
printf '  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
