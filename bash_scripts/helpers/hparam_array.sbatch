#!/bin/bash
#SBATCH --job-name=hparam-train
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 24:00:00
#SBATCH -o /ceph/behrens/dshani/revisions_code/ego_release/bash_scripts/slurm/hparam_train.%A_%a.out
#SBATCH -e /ceph/behrens/dshani/revisions_code/ego_release/bash_scripts/slurm/hparam_train.%A_%a.err
#SBATCH --array=0-4

sleep $((10*SLURM_ARRAY_TASK_ID))

set -euo pipefail
set -x

echo "DEBUG: starting hparam_array.sbatch on $(hostname) at $(date)" >&2
:	"${HORIZON:?HORIZON must be provided via --export}"
:	"${EGAMMA:?EGAMMA must be provided via --export}"
:	"${SR_LR_E:?SR_LR_E must be provided via --export}"

LOCAL=${LOCAL:-True}

echo "DEBUG: HORIZON=$HORIZON EGAMMA=$EGAMMA SR_LR_E=$SR_LR_E LOCAL=$LOCAL TASK=$SLURM_ARRAY_TASK_ID JOB=$SLURM_ARRAY_JOB_ID" >&2

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="/ceph/behrens/dshani/revisions_code/ego_release"
BASH_SCRIPTS_DIR="/ceph/behrens/dshani/revisions_code/ego_release/bash_scripts"

echo "DEBUG: SCRIPT_DIR=$SCRIPT_DIR BASH_SCRIPTS_DIR=$BASH_SCRIPTS_DIR PROJECT_ROOT=$PROJECT_ROOT PWD=$PWD" >&2

echo "INFO: Training logs will appear at $BASH_SCRIPTS_DIR/slurm/hparam_train.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.{out,err}" >&2
echo "INFO: Results base directory is $BASH_SCRIPTS_DIR/Results" >&2

source ~/.bashrc || true
micromamba activate new_ego || true

which python >&2 || true
python --version >&2 || true

cd "$BASH_SCRIPTS_DIR"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "DEBUG: launching run_experiments.py" >&2

# Run the training with lightweight artefact saving so the optimiser only needs
# the accuracy/step statistics.
python -u "$PROJECT_ROOT/run_experiments.py" \
  --unlesion_only=True \
  --save_params="[]" \
  --save_paths=False \
  --save_model=False \
  --worlds_type="'full_resample'" \
  --agamma="0.94" \
  --gamma="0.98" \
  --SR_lr_a="0.0078125" \
  --lr="0.5" \
  --save_every="500" \
  --log_every="5" \
  --env_switch_every="1000" \
  --num_episodes="2500" \
  --max_steps="50000" \
  --seed=$SLURM_ARRAY_TASK_ID \
  --job_id=$SLURM_ARRAY_JOB_ID \
  --horizon="$HORIZON" \
  --egamma="$EGAMMA" \
  --SR_lr_e="$SR_LR_E" \
  --local="$LOCAL" \
  --SR_init="'empty'"

echo "DEBUG: python finished with exit $?" >&2
echo "INFO: After training, check run_path at $BASH_SCRIPTS_DIR/Results/job_ids/$((SLURM_ARRAY_JOB_ID + 1))/run_path.txt" >&2
echo "INFO: To tail logs: tail -f $BASH_SCRIPTS_DIR/slurm/hparam_train.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" >&2

# Emit only the final mean step statistic so optimisation logs stay compact.
python - "$PROJECT_ROOT" <<'PY'
import os
from pathlib import Path
import numpy as np
import sys

project_root = Path(sys.argv[1])
array_job_id = int(os.environ["SLURM_ARRAY_JOB_ID"])
seed_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
job_dir = project_root / "Results" / "job_ids" / str(array_job_id + 1)
run_path_file = job_dir / "run_path.txt"

metric = None
if run_path_file.exists():
    run_path = Path(run_path_file.read_text().strip())
    acc_path = run_path / f"seed_{seed_id}" / "unlesioned" / "save_dict" / "save_dict_accuracies.npz"
    if acc_path.exists():
        with np.load(acc_path, allow_pickle=True) as data:
            if "accuracies" in data.files:
                accuracies = data["accuracies"]
            else:
                accuracies = None
        if accuracies is not None and len(accuracies) > 0:
            last_entry = accuracies[-1]
            # Entries are stored as (episode, mean_steps)
            try:
                metric = float(last_entry[1])
            except (TypeError, IndexError, ValueError):
                pass

if metric is None:
    print("FINAL_AVG_STEPS NA")
else:
    print(f"FINAL_AVG_STEPS {metric:.6f}")
PY
