#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# Configure the parameter grid. Override via environment variables before invoking
# this script, e.g. `HORIZON_GRID="2 4" bash bash_scripts/horizon_param_search.sh`.
IFS=' ' read -r -a HORIZON_GRID <<< "${HORIZON_GRID:-3 4 5}"
# IFS=' ' read -r -a HORIZON_GRID <<< "${HORIZON_GRID:-2}"
IFS=' ' read -r -a EGAMMA_GRID <<< "${EGAMMA_GRID:-0.6 0.65 0.7 0.75 0.8}"
# IFS=' ' read -r -a EGAMMA_GRID <<< "${EGAMMA_GRID:-0.98}"
IFS=' ' read -r -a SR_LR_E_GRID <<< "${SR_LR_E_GRID:-0.1 0.2 0.5 1.0}"
# IFS=' ' read -r -a SR_LR_E_GRID <<< "${SR_LR_E_GRID:-0.001}"
DEFAULT_RESULTS_FILE="$SCRIPT_DIR/Results/hparam_mean_steps.csv"
RESULTS_FILE=${RESULTS_FILE:-"$DEFAULT_RESULTS_FILE"}
RESULTS_FILE=$(python - "$RESULTS_FILE" <<'PY'
import os
import sys

path = os.path.expanduser(sys.argv[1])
print(os.path.abspath(path))
PY
)

SUGGESTIONS_FILE=${SUGGESTIONS_FILE:-}

if [[ -n "$SUGGESTIONS_FILE" && ! -f "$SUGGESTIONS_FILE" ]]; then
  echo "Suggestion file $SUGGESTIONS_FILE not found; falling back to full grid." >&2
  SUGGESTIONS_FILE=""
fi

if [[ -n "$SUGGESTIONS_FILE" ]]; then
  SUGGESTIONS_FILE=$(python - "$SUGGESTIONS_FILE" <<'PY'
import os
import sys

path = os.path.expanduser(sys.argv[1])
print(os.path.abspath(path))
PY
)
fi

declare -A HORIZON_SUGGESTIONS=()

if [[ -n "$SUGGESTIONS_FILE" ]]; then
  map_output=$(python - "$SUGGESTIONS_FILE" <<'PY'
import json
import math
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as fh:
    data = json.load(fh)

for entry in data.get("horizons", []):
    suggestions = entry.get("suggestions") or []
    if not suggestions:
        continue
    horizon_value = entry.get("horizon")
    if isinstance(horizon_value, (int, float)) and math.isfinite(horizon_value):
        if isinstance(horizon_value, float) and horizon_value.is_integer():
            horizon = str(int(horizon_value))
        else:
            horizon = str(horizon_value)
    else:
        horizon = str(horizon_value)
    for suggestion in suggestions:
        egamma = suggestion.get("egamma")
        sr_lr_e = suggestion.get("SR_lr_e")
        if egamma is None or sr_lr_e is None:
            continue
        print(f"{horizon} {egamma} {sr_lr_e}")
PY
)
  if [[ -n "$map_output" ]]; then
    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      read -r horizon egamma sr_lr_e <<< "$line"
      if [[ -z "${horizon:-}" || -z "${egamma:-}" || -z "${sr_lr_e:-}" ]]; then
        continue
      fi
      if [[ -n "${HORIZON_SUGGESTIONS[$horizon]:-}" ]]; then
        HORIZON_SUGGESTIONS[$horizon]+=$'\n'"$egamma $sr_lr_e"
      else
        HORIZON_SUGGESTIONS[$horizon]="$egamma $sr_lr_e"
      fi
    done <<< "$map_output"
    echo "Loaded Bayesian suggestions from $SUGGESTIONS_FILE" >&2
  else
    echo "No Bayesian suggestions found in $SUGGESTIONS_FILE; using grid search." >&2
  fi
fi

TRAIN_TEMPLATE="$SCRIPT_DIR/helpers/hparam_array.sbatch"
ANALYSIS_TEMPLATE="$SCRIPT_DIR/helpers/hparam_analysis.sbatch"

if [[ ! -f "$TRAIN_TEMPLATE" ]]; then
  echo "Could not find training template at $TRAIN_TEMPLATE" >&2
  exit 1
fi

if [[ ! -f "$ANALYSIS_TEMPLATE" ]]; then
  echo "Could not find analysis template at $ANALYSIS_TEMPLATE" >&2
  exit 1
fi

submit_hparam_jobs() {
  local horizon="$1"
  local egamma="$2"
  local sr_lr_e="$3"

  local -a export_args=("ALL" "HORIZON=${horizon}" "EGAMMA=${egamma}" "SR_LR_E=${sr_lr_e}")
  local train_output
  train_output=$(sbatch --parsable --export="$(IFS=,; echo "${export_args[*]}")" "$TRAIN_TEMPLATE")
  local train_job_id=${train_output%%.*}
  # Print explicit per-task log files for the fixed array 0-4
  local -a train_out_paths=(
    "/ceph/behrens/dshani/revisions_code/ego_release/bash_scripts/slurm/hparam_train.${train_job_id}_0.out"
    "/ceph/behrens/dshani/revisions_code/ego_release/bash_scripts/slurm/hparam_train.${train_job_id}_1.out"
    "/ceph/behrens/dshani/revisions_code/ego_release/bash_scripts/slurm/hparam_train.${train_job_id}_2.out"
    "/ceph/behrens/dshani/revisions_code/ego_release/bash_scripts/slurm/hparam_train.${train_job_id}_3.out"
    "/ceph/behrens/dshani/revisions_code/ego_release/bash_scripts/slurm/hparam_train.${train_job_id}_4.out"
  )
  local -a train_err_paths=(
    "/ceph/behrens/dshani/revisions_code/ego_release/bash_scripts/slurm/hparam_train.${train_job_id}_0.err"
    "/ceph/behrens/dshani/revisions_code/ego_release/bash_scripts/slurm/hparam_train.${train_job_id}_1.err"
    "/ceph/behrens/dshani/revisions_code/ego_release/bash_scripts/slurm/hparam_train.${train_job_id}_2.err"
    "/ceph/behrens/dshani/revisions_code/ego_release/bash_scripts/slurm/hparam_train.${train_job_id}_3.err"
    "/ceph/behrens/dshani/revisions_code/ego_release/bash_scripts/slurm/hparam_train.${train_job_id}_4.err"
  )
  echo "Submitted training array ${train_job_id} for horizon=${horizon}, egamma=${egamma}, SR_lr_e=${sr_lr_e}"
  echo "  logs: out: ${train_out_paths[*]} | err: ${train_err_paths[*]}"

  local -a analysis_export=("ALL" "HORIZON=${horizon}" "EGAMMA=${egamma}" "SR_LR_E=${sr_lr_e}" \
    "TRAIN_JOB_ID=${train_job_id}" "RESULTS_FILE=${RESULTS_FILE}")
  local analysis_output
  analysis_output=$(sbatch --parsable --dependency=afterok:${train_job_id} \
    --export="$(IFS=,; echo "${analysis_export[*]}")" "$ANALYSIS_TEMPLATE")
  local analysis_job_id=${analysis_output%%.*}
  local analysis_out="/ceph/behrens/dshani/revisions_code/ego_release/bash_scripts/slurm/hparam_analysis.${analysis_job_id}.out"
  local analysis_err="/ceph/behrens/dshani/revisions_code/ego_release/bash_scripts/slurm/hparam_analysis.${analysis_job_id}.err"
  echo "  ↳ Analysis job ${analysis_job_id} will summarise array ${train_job_id}"
  echo "     logs: ${analysis_out} | ${analysis_err}"
}

for horizon in "${HORIZON_GRID[@]}"; do
  if [[ -n "${HORIZON_SUGGESTIONS[$horizon]:-}" ]]; then
    while IFS= read -r suggestion; do
      [[ -z "$suggestion" ]] && continue
      read -r egamma sr_lr_e <<< "$suggestion"
      submit_hparam_jobs "$horizon" "$egamma" "$sr_lr_e"
    done <<< "${HORIZON_SUGGESTIONS[$horizon]}"
    continue
  fi

  for egamma in "${EGAMMA_GRID[@]}"; do
    for sr_lr_e in "${SR_LR_E_GRID[@]}"; do
      submit_hparam_jobs "$horizon" "$egamma" "$sr_lr_e"
    done
  done
done
