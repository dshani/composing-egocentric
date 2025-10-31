#!/bin/bash

# Set GENERATE_SCRIPTS=1 or pass --generate-scripts to also write per-(horizon,room)
# analysis scripts to bash_scripts/helpers/generated_analysis/. Jobs are still
# submitted via sbatch so the pipeline runs as usual.

# Arrays of horizon values and room identifiers
horizon_values=( 2 3 4 5 )
room_numbers=( 0 2 3 6 ) # 7

# # debug
# horizon_values=( 4 5 )
# room_numbers=( 7 )

# Flag controlling whether to generate analysis scripts alongside submission
GENERATE_SCRIPTS=${GENERATE_SCRIPTS:-0}
for arg in "$@"; do
    case $arg in
        --generate-scripts)
            GENERATE_SCRIPTS=1
            shift
            ;;
    esac
done

# Limit worker threads for analysis jobs
MAX_WORKERS=${MAX_WORKERS:-4}
DEFAULT_MAX_WORKERS=$MAX_WORKERS

analysis_job_ids=()
analysis_scripts=()
declare -A deps=()

# Directory for generated analysis scripts (only created if generation is enabled)
if [ "$GENERATE_SCRIPTS" -eq 1 ]; then
    SCRIPT_BASE_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
    GENERATED_DIR="$SCRIPT_BASE_DIR/helpers/generated_analysis"
    mkdir -p "$GENERATED_DIR"
fi

for room_num in "${room_numbers[@]}"; do
    if [ -z "$room_num" ]; then
        PARAM_VALUE2="random_rooms0"
        room_label=0
    else
        PARAM_VALUE2="random_rooms${room_num}"
        room_label=$room_num
    fi
    PARAM_NAME2="worlds_type"
    # Export the raw PARAM_VALUE2 without pre-quoting (PARAM_VALUE2_QUOTED removed)

    for horizon in "${horizon_values[@]}"; do
        PARAM_NAME="horizon"
        PARAM_VALUE="$horizon"

        exp_output=$(sbatch --export=PARAM_NAME="$PARAM_NAME",PARAM_VALUE="$PARAM_VALUE",PARAM_NAME2="$PARAM_NAME2",PARAM_VALUE2="$PARAM_VALUE2" horizon_expvar.sbatch)
        exp_id=$(echo "$exp_output" | awk '{print $4}')

        an_output=$(sbatch --dependency=aftercorr:"$exp_id" --export=HORIZON="$horizon",WORLDS_TYPE="$PARAM_VALUE2",MAX_WORKERS="$MAX_WORKERS" horizon_analysis.sbatch)
        an_id=$(echo "$an_output" | awk '{print $4}')

        if [ "$GENERATE_SCRIPTS" -eq 1 ]; then
            script_path="$GENERATED_DIR/analyze_h${horizon}_r${room_label}.sh"
            cat > "$script_path" <<SCRIPT
#!/bin/bash
# Auto-generated analysis script for horizon $horizon and room $PARAM_VALUE2
MAX_WORKERS=\${MAX_WORKERS:-$DEFAULT_MAX_WORKERS}
SCRIPT_DIR=\$(cd -- "\$(dirname -- "\${BASH_SOURCE[0]}")" &>/dev/null && pwd)
sbatch --export=HORIZON=$horizon,WORLDS_TYPE="$PARAM_VALUE2",MAX_WORKERS=\$MAX_WORKERS,JOB_ID=$an_id "\$SCRIPT_DIR"/../../horizon_analysis.sbatch
SCRIPT
            chmod +x "$script_path"
            analysis_scripts+=("$script_path")
        fi

        analysis_job_ids+=("$an_id")
        deps[$room_label]="${deps[$room_label]}:$an_id"
    done
done

if [ "$GENERATE_SCRIPTS" -eq 1 ]; then
    printf '%s\n' "${analysis_scripts[@]}"
fi

dependency_ids=$(IFS=:; echo "${analysis_job_ids[*]}")

job_ids_list=$(printf ",%s" "${analysis_job_ids[@]}")
job_ids_list="[${job_ids_list:1}]"

echo "Job IDs list: $job_ids_list"

for room_num in "${room_numbers[@]}"; do
    if [ -z "$room_num" ]; then
        world="random_rooms0"
        room_label=0
    else
        world="random_rooms${room_num}"
        room_label=$room_num
    fi
    dep_ids=${deps[$room_label]}
    dep_ids=${dep_ids#:}
    if [ -n "$dep_ids" ]; then
        sbatch --dependency=aftercorr:${dep_ids} --export=WORLDS_TYPE="$world" plot_mean_steps.sbatch
    fi
done

