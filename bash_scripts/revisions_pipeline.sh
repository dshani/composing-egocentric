#!/bin/bash

# Default settings
chunk_size=25
room_numbers=( 0 2 3 7 12 13)
# room_numbers=(0)
# room_numbers=(7)
analysis_only=false
model_analysis=false
# Limit path plots to all seeds and all delays < N in debug folder
path_scan=""
# Anchor job_ids under bash_scripts so analyse_full_pipeline.py (run from this dir) resolves ./Results/job_ids
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
results_dir="${script_dir}/Results/job_ids"
full_analysis_rooms=( 7 13 )
model_analysis_rooms=( 7 )
debug_analysis=false
select_max_delay=""
select_objective=""

# Select Python interpreter (allow override with $PYTHON)
PYTHON_BIN="${PYTHON:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "Neither 'python' nor 'python3' found on PATH. Set PYTHON env var to your interpreter." >&2
        exit 127
    fi
fi

find_latest_job_id() {
    local target_world="$1"
    local search_root="$2"
    "$PYTHON_BIN" - "$target_world" "$search_root" <<'PY'
import json
import sys
from pathlib import Path

target = sys.argv[1]
root = Path(sys.argv[2])
if not root.exists():
    sys.exit(3)

candidates = []
for entry in root.iterdir():
    if not entry.is_dir():
        continue
    name = entry.name
    if not name.isdigit():
        continue
    args_path = entry / "args.txt"
    if not args_path.exists():
        continue
    try:
        with args_path.open() as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        continue
    # Only consider runs matching worlds_type and exclude LEC-only runs
    if data.get("worlds_type") == target and not data.get("LEC_only", False):
        try:
            candidates.append(int(name))
        except ValueError:
            continue

if not candidates:
    sys.exit(4)

print(max(candidates))
PY
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --chunk-size)
            if [[ -z "$2" ]]; then
                echo "--chunk-size requires a value" >&2
                exit 1
            fi
            chunk_size="$2"
            shift 2
            ;;
        --select-max-delay)
            if [[ -z "$2" || "$2" == --* ]]; then
                echo "--select-max-delay requires a value" >&2
                exit 1
            fi
            select_max_delay="$2"
            shift 2
            ;;
        --select-max-delay=*)
            select_max_delay="${1#*=}"
            shift
            ;;
        --select-objective)
            if [[ -z "$2" || "$2" == --* ]]; then
                echo "--select-objective requires a value" >&2
                exit 1
            fi
            select_objective="$2"
            shift 2
            ;;
        --select-objective=*)
            select_objective="${1#*=}"
            shift
            ;;
        --room-numbers|--room_numbers)
            shift
            if [[ $# -eq 0 || "$1" == --* ]]; then
                echo "--room-numbers requires at least one value" >&2
                exit 1
            fi
            room_numbers=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                room_numbers+=("$1")
                shift
            done
            ;;
        --analysis-only)
            analysis_only=true
            shift
            ;;
        --full-analysis-rooms)
            shift
            if [[ $# -eq 0 || "$1" == --* ]]; then
                echo "--full-analysis-rooms requires at least one value" >&2
                exit 1
            fi
            while [[ $# -gt 0 && "$1" != --* ]]; do
                full_analysis_rooms+=("$1")
                shift
            done
            ;;
        --debug-analysis)
            debug_analysis=true
            shift
            ;;
        --model-analysis)
            model_analysis=true
            shift
            ;;
        --path-scan)
            if [[ -z "$2" || "$2" == --* ]]; then
                echo "--path-scan requires an integer value" >&2
                exit 1
            fi
            path_scan="$2"
            shift 2
            ;;
        --path-scan=*)
            path_scan="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

analysis_job_ids=()

# Ensure required directories exist for Slurm outputs and job metadata
mkdir -p "${script_dir}/slurm"
mkdir -p "${results_dir}"

for room_num in "${room_numbers[@]}"; do
    # Determine the parameter name and value
    if [ -z "$room_num" ]; then
        world_type="random_rooms0"
    else
        world_type="random_rooms${room_num}"
    fi
    PARAM_VALUE="'${world_type}'"
    PARAM_NAME="worlds_type"


    # Determine if this room should receive full analysis treatment
    is_full_analysis=false
    for candidate in "${full_analysis_rooms[@]}"; do
        if [[ "$candidate" == "$room_num" ]]; then
            is_full_analysis=true
            break
        fi
    done

    is_model_analysis=false
    # Enable model-analysis either when the global flag is set, or when this room is listed
    if [[ "$model_analysis" == true ]]; then
        is_model_analysis=true
    else
        for candidate in "${model_analysis_rooms[@]}"; do
            if [[ "$candidate" == "$room_num" ]]; then
                is_model_analysis=true
                break
            fi
        done
    fi


    analysis_args="--chunk-size=${chunk_size}"

    if [[ "$is_full_analysis" == true ]]; then
        analysis_args+=" --generalisation_only=False --barrier-thickness=0"
    else
        analysis_args+=" --room_compare_only=True"
    fi

    if [[ "$is_full_analysis" == true ]]; then
        analysis_args+=" --produce-schematic --episode-glm"
    fi

    if [[ "$analysis_only" != true ]]; then
        if [[ "$debug_analysis" == true ]]; then
            analysis_args+=" --debug=True --figures_root=./Results/figures/debug"
        else
            analysis_args+=" --figures_root=./Results/figures"
        fi
    fi

    if [[ "$is_model_analysis" == true ]]; then
        analysis_args+=" --model-analysis --room_compare_only=False"
    fi

    # If a path scan is requested, force debug output and model-analysis, and pass the scan bound
    if [[ -n "$path_scan" ]]; then
        # Ensure debug figures folder is used even if --debug-analysis was not passed
        if [[ "$analysis_only" != true ]]; then
            if [[ "$debug_analysis" != true ]]; then
                analysis_args+=" --figures_root=./Results/figures/debug"
            fi
        fi
        # Ensure model-analysis mode is enabled for path-only exports
        if [[ "$is_model_analysis" != true ]]; then
            is_model_analysis=true
            analysis_args+=" --model-analysis --room_compare_only=False"
        fi
        analysis_args+=" --path-scan=${path_scan}"
    fi

    # For standard model-analysis (no path-scan), pin seed and delay for path plot
    if [[ "$is_model_analysis" == true && -z "$path_scan" ]]; then
        analysis_args+=" --single_seed=1 --delay=6"
    fi

    if [[ -n "$select_max_delay" ]]; then
        analysis_args+=" --select-max-delay=${select_max_delay}"
    fi

    if [[ -n "$select_objective" ]]; then
        analysis_args+=" --select-objective=${select_objective}"
    fi

    analysis_exports=(
        "ANALYSIS_ARGS=$analysis_args"
        "ANALYSIS_SCRIPT=${script_dir}/../analyse_full_pipeline.py"
    )

    if [[ "$analysis_only" == true ]]; then
        latest_job_id=$(find_latest_job_id "$world_type" "$results_dir")
        status=$?
        latest_job_id=${latest_job_id//$'\n'/}
        if [[ $status -ne 0 ]]; then
            case $status in
                3)
                    echo "Results directory not found: $results_dir" >&2
                    ;;
                4)
                    echo "No completed job found for worlds_type '$world_type' in $results_dir" >&2
                    ;;
                *)
                    echo "Failed to locate job metadata for '$world_type' (exit code $status)" >&2
                    ;;
            esac
            exit 1
        fi

        # args.txt and run_path.txt are written into the analysis slot directory already
        # (job_ids/<train_job_id+1>). Our finder returns that slot ID, so use it directly.
        echo "Re-running analysis for $world_type using existing analysis slot $latest_job_id"
        analysis_exports+=("ANALYSIS_JOB_ID_OVERRIDE=$latest_job_id")
        rand_an_output=$(sbatch -D "${script_dir}" --export="$(IFS=,; echo "${analysis_exports[*]}")" "${script_dir}/exp_analysis.sbatch")
        rand_an_id=$(echo "$rand_an_output" | awk '{print $4}')
    else
        train_exports=("PARAM_NAME=$PARAM_NAME" "PARAM_VALUE=$PARAM_VALUE")

        if [[ "$is_model_analysis" == true ]]; then
            train_exports+=("SAVE_PARAMS=\"['weight','m','v','value_snapshots','ego_SR.SR_ss','allo_SR.SR_ss','paths']\"")
        elif [[ "$is_full_analysis" != true ]]; then
            train_exports+=("SAVE_PARAMS=\"['paths']\"")
        else
            train_exports+=("SAVE_PARAMS=\"['weight','m','v','value_snapshots','paths']\"")
        fi

        rand_exp=$(sbatch -D "${script_dir}" --export="$(IFS=,; echo "${train_exports[*]}")" "${script_dir}/expvar.sbatch")
        rand_id=$(echo "$rand_exp" | awk '{print $4}')
        metadata_job_id=$((rand_id + 1))
        echo "Scheduling analysis for job slot $metadata_job_id"
        analysis_exports+=("ANALYSIS_JOB_ID_OVERRIDE=$metadata_job_id")

        rand_an_output=$(sbatch -D "${script_dir}" --dependency=aftercorr:"$rand_id" --export="$(IFS=,; echo "${analysis_exports[*]}")" "${script_dir}/exp_analysis.sbatch")
        rand_an_id=$(echo "$rand_an_output" | awk '{print $4}')
    fi

    # Append the analysis job ID to the array
    analysis_job_ids+=("$rand_an_id")

done
# Create a colon-separated list of analysis job IDs for the dependency
dependency_ids=$(IFS=:; echo "${analysis_job_ids[*]}")

# Create a string representation of the job IDs list for Python
job_ids_list=$(printf ",%s" "${analysis_job_ids[@]}")
job_ids_list="[${job_ids_list:1}]"

echo "Job IDs list: $job_ids_list"
