#!/bin/bash
#
#SBATCH --job-name=occupancy_analysis
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o ./slurm/slurm.%j.out
#SBATCH -e ./slurm/slurm.%j.err
#

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

hostname; date

source ~/.bashrc
micromamba activate new_ego

if [[ -z "${RUN_ID:-}" ]]; then
  echo "RUN_ID is not set. Launch this analysis with --export=RUN_ID=<job_id>." >&2
  exit 1
fi

RESULTS_DIR=${RESULTS_DIR:-"./Results/job_ids"}
ENV_SWITCH_EVERY=${ENV_SWITCH_EVERY:-1000}
MAX_WORKERS=${MAX_WORKERS:-8}
BARRIER_THICKNESS=${BARRIER_THICKNESS:-1}

cmd=(
  python -u "$SCRIPT_DIR/../analyse_barrier_occupancy.py" \
    --job-id "$RUN_ID" \
    --results-dir "$RESULTS_DIR" \
    --env-switch-every "$ENV_SWITCH_EVERY" \
    --max-workers "$MAX_WORKERS" \
    --barrier-thickness "$BARRIER_THICKNESS"

)

if [[ -n "${WORLD_METADATA:-}" ]]; then
  cmd+=(--world-metadata "$WORLD_METADATA")
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206  # we intentionally split on IFS
  extra_args=(${EXTRA_ARGS})
  cmd+=("${extra_args[@]}")
fi

printf 'INFO: Running occupancy analysis with command:\n'
printf '  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
