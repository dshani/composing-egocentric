#!/bin/bash
# enable tracing after SBATCH headers are read
#SBATCH --job-name=hparam-analyse
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 02:00:00
#SBATCH -o /ceph/behrens/dshani/revisions_code/ego_release/bash_scripts/slurm/hparam_analysis.%j.out
#SBATCH -e /ceph/behrens/dshani/revisions_code/ego_release/bash_scripts/slurm/hparam_analysis.%j.err

set -euo pipefail
set -x

:	"${HORIZON:?HORIZON must be provided via --export}"
:	"${EGAMMA:?EGAMMA must be provided via --export}"
:	"${SR_LR_E:?SR_LR_E must be provided via --export}"
:	"${TRAIN_JOB_ID:?TRAIN_JOB_ID must point to the completed array job}"

RESULTS_FILE=${RESULTS_FILE:-"./Results/hparam_mean_steps.csv"}
SEEDS="${SEEDS:-0 1 2 3 4}"
ENABLE_BAYES_OPT=${ENABLE_BAYES_OPT:-1}
BAYES_OPT_GRID_SIZE=${BAYES_OPT_GRID_SIZE:-100}
BAYES_OPT_TOP_K=${BAYES_OPT_TOP_K:-5}
BAYES_OPT_OUTPUT=${BAYES_OPT_OUTPUT:-"./Results/hparam_bayes_opt.json"}
BAYES_OPT_LOG_LR=${BAYES_OPT_LOG_LR:-1}
BAYES_OPT_EGAMMA_BOUNDS=${BAYES_OPT_EGAMMA_BOUNDS:-}
BAYES_OPT_SR_LR_BOUNDS=${BAYES_OPT_SR_LR_BOUNDS:-}
BAYES_OPT_EXTRA_ARGS=${BAYES_OPT_EXTRA_ARGS:-}
BAYES_OPT_INDENT=${BAYES_OPT_INDENT:-2}
BAYES_OPT_XI=${BAYES_OPT_XI:-0.01}
BAYES_OPT_N_RESTARTS=${BAYES_OPT_N_RESTARTS:-8}
BAYES_OPT_RANDOM_STATE=${BAYES_OPT_RANDOM_STATE:-0}
 
SCRIPT_DIR="/ceph/behrens/dshani/revisions_code/ego_release/bash_scripts/helpers"
PROJECT_ROOT="/ceph/behrens/dshani/revisions_code/ego_release"
BASH_SCRIPTS_DIR="/ceph/behrens/dshani/revisions_code/ego_release/bash_scripts"

source ~/.bashrc || true
if ! command -v micromamba >/dev/null 2>&1; then
  if [[ -x "$HOME/micromamba/bin/micromamba" ]]; then
    export PATH="$HOME/micromamba/bin:$PATH"
  fi
fi
if command -v micromamba >/dev/null 2>&1; then eval "$(micromamba shell hook -s bash)"; fi
if command -v micromamba >/dev/null 2>&1; then micromamba activate new_ego; fi

cd "$BASH_SCRIPTS_DIR"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "INFO: SCRIPT_DIR=$SCRIPT_DIR" >&2
echo "INFO: BASH_SCRIPTS_DIR=$BASH_SCRIPTS_DIR" >&2
echo "INFO: PROJECT_ROOT=$PROJECT_ROOT" >&2
echo "INFO: RESULTS_FILE=$RESULTS_FILE TRAIN_JOB_ID=$TRAIN_JOB_ID HORIZON=$HORIZON EGAMMA=$EGAMMA SR_LR_E=$SR_LR_E" >&2
echo "INFO: Analysis logs at /ceph/behrens/dshani/revisions_code/ego_release/bash_scripts/slurm/hparam_analysis.$SLURM_JOB_ID.{out,err}" >&2
echo "INFO: Reading job run_path from $BASH_SCRIPTS_DIR/Results/job_ids/$((TRAIN_JOB_ID + 1))/run_path.txt" >&2
echo "INFO: Appending results to $BASH_SCRIPTS_DIR/Results/hparam_mean_steps.csv (override with RESULTS_FILE)" >&2

# Print a readable message on any failure
trap 'ec=$?; echo "ERROR: analysis failed (exit $ec)" >&2; \
      echo "ERROR: Check run_path and inputs. TRAIN_JOB_ID=$TRAIN_JOB_ID HORIZON=$HORIZON EGAMMA=$EGAMMA SR_LR_E=$SR_LR_E" >&2; \
      echo "ERROR: Expected run_path: $BASH_SCRIPTS_DIR/Results/job_ids/$((TRAIN_JOB_ID + 1))/run_path.txt" >&2; \
      exit $ec' ERR

which python >&2 || true
python --version >&2 || true

echo "INFO: Running: python -u $SCRIPT_DIR/collect_hparam_results.py --project-root $BASH_SCRIPTS_DIR --job-id $TRAIN_JOB_ID ..." >&2
python -u "$SCRIPT_DIR/collect_hparam_results.py" \
  --project-root "$BASH_SCRIPTS_DIR" \
  --job-id "$TRAIN_JOB_ID" \
  --horizon "$HORIZON" \
  --egamma "$EGAMMA" \
  --sr-lr-e "$SR_LR_E" \
  --results-file "$RESULTS_FILE" \
  --seeds $SEEDS

if [[ "$ENABLE_BAYES_OPT" != "0" ]]; then
  BAYES_OPT_CMD=(
    python -u "$SCRIPT_DIR/bayesian_optimize_hparams.py"
    --results-file "$RESULTS_FILE"
    --grid-size "$BAYES_OPT_GRID_SIZE"
    --top-k "$BAYES_OPT_TOP_K"
    --output "$BAYES_OPT_OUTPUT"
    --xi "$BAYES_OPT_XI"
    --n-restarts "$BAYES_OPT_N_RESTARTS"
    --random-state "$BAYES_OPT_RANDOM_STATE"
    --indent "$BAYES_OPT_INDENT"
  )
  if [[ "$BAYES_OPT_LOG_LR" != "0" ]]; then
    BAYES_OPT_CMD+=(--log-lr)
  fi
  if [[ -n "$BAYES_OPT_EGAMMA_BOUNDS" ]]; then
    IFS=' ' read -r EGAMMA_MIN EGAMMA_MAX <<< "$BAYES_OPT_EGAMMA_BOUNDS"
    if [[ -z "${EGAMMA_MIN:-}" || -z "${EGAMMA_MAX:-}" ]]; then
      echo "ERROR: BAYES_OPT_EGAMMA_BOUNDS must contain two floats" >&2
      exit 1
    fi
    BAYES_OPT_CMD+=(--egamma-bounds "$EGAMMA_MIN" "$EGAMMA_MAX")
  fi
  if [[ -n "$BAYES_OPT_SR_LR_BOUNDS" ]]; then
    IFS=' ' read -r SR_LR_MIN SR_LR_MAX <<< "$BAYES_OPT_SR_LR_BOUNDS"
    if [[ -z "${SR_LR_MIN:-}" || -z "${SR_LR_MAX:-}" ]]; then
      echo "ERROR: BAYES_OPT_SR_LR_BOUNDS must contain two floats" >&2
      exit 1
    fi
    BAYES_OPT_CMD+=(--sr-lr-bounds "$SR_LR_MIN" "$SR_LR_MAX")
  fi
  if [[ -n "$BAYES_OPT_EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=($BAYES_OPT_EXTRA_ARGS)
    BAYES_OPT_CMD+=("${EXTRA_ARGS[@]}")
  fi
  echo "INFO: Running Bayesian optimisation helper: ${BAYES_OPT_CMD[*]}" >&2
  set +e
  "${BAYES_OPT_CMD[@]}"
  BAYES_STATUS=$?
  set -e
  if [[ $BAYES_STATUS -ne 0 ]]; then
    echo "WARNING: bayesian_optimize_hparams.py failed with exit code $BAYES_STATUS" >&2
  fi
fi
