#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --include_lesion)
      EXTRA_ARGS="--include_lesion"
      shift
      ;;
    *)
      echo "Usage: $0 [--include_lesion]" >&2
      exit 1
      ;;
  esac
done

if [[ -n "$EXTRA_ARGS" ]]; then
  export EXTRA_ARGS
fi

export_args="ALL"
if [[ -n "$EXTRA_ARGS" ]]; then
  export_args="${export_args},EXTRA_ARGS=$EXTRA_ARGS"
fi

run_output=$(sbatch --export="$export_args" "$SCRIPT_DIR/lr_run.sbatch")
run_job_id=$(echo "$run_output" | awk '{print $4}')
echo "Submitted lr run job array: $run_job_id"

analysis_export_args="ALL,RUN_ID=$run_job_id"
if [[ -n "$EXTRA_ARGS" ]]; then
  analysis_export_args="${analysis_export_args},EXTRA_ARGS=$EXTRA_ARGS"
fi

analysis_output=$(sbatch --dependency=afterok:$run_job_id --export="$analysis_export_args" "$SCRIPT_DIR/lr_analysis.sbatch")
analysis_job_id=$(echo "$analysis_output" | awk '{print $4}')
echo "Submitted lr analysis job: $analysis_job_id (depends on $run_job_id)"
