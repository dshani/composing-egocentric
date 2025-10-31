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

analysis_args_str=${ANALYSIS_ARGS:="--generalisation_only=True"}
# shellcheck disable=SC2206  # we intentionally split on whitespace to get argv entries
analysis_args=(${analysis_args_str})

analysis_entry=${ANALYSIS_SCRIPT:-../run_analysis_.py}

job_id_override=${ANALYSIS_JOB_ID_OVERRIDE:-}
if [[ -n "$job_id_override" ]]; then
  printf 'INFO: Overriding analysis job_id with %s\n' "$job_id_override"
  cmd=(python -u "$analysis_entry" --job_id="$job_id_override")
else
  cmd=(python -u "$analysis_entry" --job_id="$SLURM_JOB_ID")
fi
cmd+=("${analysis_args[@]}")

printf 'INFO: Running analysis with command:\n'
printf '  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
