#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN_SCRIPT="$SCRIPT_DIR/expvar.sbatch"
ANALYSIS_SCRIPT="$SCRIPT_DIR/exp_analysis.sbatch"
PARAM_NAME="worlds_type"

PARAM_VALUE="'value_plot_random_rooms'"
SAVE_PARAMS="['paths']"  # single quotes preserved via double-quoted assignment

train_output=$(sbatch --export=PARAM_NAME="$PARAM_NAME",PARAM_VALUE="$PARAM_VALUE",SAVE_PARAMS="$SAVE_PARAMS" "$TRAIN_SCRIPT")
train_job_id=$(echo "$train_output" | awk '{print $4}')

if [[ -z "$train_job_id" ]]; then
  echo "Failed to parse training job ID from: $train_output" >&2
  exit 1
fi

echo "Submitted value plot training job: $train_job_id"

analysis_output=$(sbatch --dependency=aftercorr:"$train_job_id" "$ANALYSIS_SCRIPT")
analysis_job_id=$(echo "$analysis_output" | awk '{print $4}')

if [[ -z "$analysis_job_id" ]]; then
  echo "Failed to parse analysis job ID from: $analysis_output" >&2
  exit 1
fi

echo "Queued value plot generalisation analysis job: $analysis_job_id (depends on $train_job_id)"
