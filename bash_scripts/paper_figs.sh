#!/bin/bash

set -euo pipefail

# Create final figures for the paper by orchestrating the revisions pipeline and
# collecting specific panels into a clean folder tree.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

TMP_ROOT="${repo_root}/paper_tmp"
OUT_ROOT="${repo_root}/figures"

ROOMS=(0 2 3 7 12 13)
MODEL_ANALYSIS_ROOMS=(7)
FULL_ANALYSIS_ROOMS=(7 13)

ANALYSIS_ONLY=false
ASSEMBLE_ONLY=false

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--analysis-only]

Options:
  --analysis-only   Re-run analysis only (no new training), using latest jobs.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --analysis-only)
      ANALYSIS_ONLY=true
      shift
      ;;
    --assemble-only)
      ASSEMBLE_ONLY=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

mkdir -p "$TMP_ROOT"
mkdir -p "$OUT_ROOT"/fig2 "$OUT_ROOT"/fig3 "$OUT_ROOT"/fig4 "$OUT_ROOT"/fig5

info() { echo "[paper_figs] $*"; }
warn() { echo "[paper_figs][WARN] $*" >&2; }

run_pipeline() {
  local pipeline_sh="$script_dir/revisions_pipeline.sh"
  if [[ ! -f "$pipeline_sh" ]]; then
    warn "Missing script: $pipeline_sh"
    exit 1
  fi

  info "Submitting analysis jobs via revisions_pipeline.sh"
  # Build CLI for rooms, and pass explicit full-analysis rooms
  local room_args=(--room-numbers)
  for r in "${ROOMS[@]}"; do room_args+=("$r"); done
  local fa_args=(--full-analysis-rooms)
  for r in "${FULL_ANALYSIS_ROOMS[@]}"; do fa_args+=("$r"); done

  local output
  local extra=()
  if [[ "$ANALYSIS_ONLY" == true ]]; then
    extra+=(--analysis-only)
  fi

  # Do NOT pass --model-analysis globally; the pipeline will enable model-analysis
  # only for rooms listed in its model_analysis_rooms default (currently room 7).
  if ! output=$(bash "$pipeline_sh" "${room_args[@]}" "${fa_args[@]}" "${extra[@]}" 2>&1); then
    echo "$output" >&2
    warn "Pipeline submission failed"
    exit 1
  fi
  echo "$output"
}

parse_job_ids() {
  # Primary: parse explicit summary line from the pipeline
  grep -Eo "Job IDs list: \[[^]]*\]" | grep -Eo "[0-9]+" | tr '\n' ' ' | sed -e 's/  */ /g' -e 's/^ *//' -e 's/ *$//'
}

parse_any_job_ids() {
  # Fallback: parse generic sbatch submission lines if summary missing
  local buf="$1"
  local from_summary from_submitted combined
  from_summary=$(echo "$buf" | parse_job_ids || true)
  from_submitted=$(echo "$buf" | grep -Eo "Submitted batch job [0-9]+" | awk '{print $4}' | tr '\n' ' ' | sed -e 's/  */ /g' -e 's/^ *//' -e 's/ *$//' || true)
  combined="$from_summary $from_submitted"
  echo "$combined" | awk '{for(i=1;i<=NF;i++) if(!seen[$i]++){printf("%s%s", $i, (i<NF?" ":""))}}'
}

wait_for_jobs() {
  local ids=("$@")
  if [[ ${#ids[@]} -eq 0 ]]; then
    warn "No job IDs to wait for; continuing"
    return 0
  fi
  if ! command -v squeue >/dev/null 2>&1; then
    warn "squeue not found; sleeping 60s before continuing"
    sleep 60
    return 0
  fi
  info "Waiting for jobs to complete: ${ids[*]}"
  while true; do
    local pending
    pending=$(squeue -h -j "$(IFS=,; echo "${ids[*]}")" | wc -l || true)
    if [[ "$pending" -eq 0 ]]; then
      break
    fi
    sleep 10
  done
}

# Find the newest panels directory for a given room label
find_panels_dir_for_room() {
  local room_label="$1" # e.g. random_rooms7
  local best_dir=""
  local best_mtime=0
  # Search common roots
  while IFS= read -r -d '' d; do
    local mtime
    mtime=$(stat -c %Y "$d" 2>/dev/null || echo 0)
    if (( mtime > best_mtime )); then
      best_mtime=$mtime
      best_dir="$d"
    fi
  done < <(find "$repo_root" -type d -path "*/room_plots/${room_label}/panels" -print0 2>/dev/null)

  if [[ -z "$best_dir" ]]; then
    return 1
  fi
  echo "$best_dir"
}

copy_room_panels_to_tmp() {
  local room="$1" # numeric
  local label="random_rooms${room}"
  local src
  if ! src=$(find_panels_dir_for_room "$label"); then
    warn "Panels not found for $label"
    return 1
  fi
  # Try to extract job_id if path is under Results/job_ids/<id>/figures
  local jid
  jid=$(echo "$src" | sed -n 's#.*Results/job_ids/\([0-9][0-9]*\)/figures/.*#\1#p')
  if [[ -n "$jid" ]]; then
    info "Using panels for $label from job_id $jid: $src"
  else
    info "Using panels for $label from: $src"
  fi
  local dst="$TMP_ROOT/${label}/panels"
  mkdir -p "$dst"
  rsync -a --delete "$src/" "$dst/"
}

copy_dir_files_flat() {
  local src_dir="$1"; shift
  local dst_dir="$1"; shift
  mkdir -p "$dst_dir"
  if [[ -d "$src_dir" ]]; then
    find "$src_dir" -maxdepth 1 -type f -print0 2>/dev/null | xargs -0 -I{} cp -f "{}" "$dst_dir/" 2>/dev/null || true
  else
    warn "Missing directory: $src_dir"
  fi
}

copy_if_exists() {
  local src="$1"; shift
  local dst="$1"; shift
  if [[ -f "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp -f "$src" "$dst"
  else
    warn "Missing file: $src"
  fi
}

copy_glob() {
  local pattern="$1"; shift
  local dst_dir="$1"; shift
  mkdir -p "$dst_dir"
  shopt -s nullglob
  local any=0
  for f in $pattern; do
    any=1
    cp -f "$f" "$dst_dir/"
  done
  shopt -u nullglob
  if [[ $any -eq 0 ]]; then
    warn "No matches for pattern: $pattern"
  fi
}

sanitize_names_in_dir() {
  local dir="$1"; shift
  shopt -s nullglob
  for src in "$dir"/*; do
    [[ -f "$src" ]] || continue
    local base name ext new
    base=$(basename -- "$src")
    name="${base%.*}"; ext="${base##*.}"
    new=$(echo "$name" | sed -E 's/_k_[0-9]+//g; s/_n_[0-9]+//g; s/__+/_/g; s/_+$//')
    if [[ "$new.$ext" != "$base" ]]; then
      mv -f "$src" "$dir/$new.$ext"
    fi
  done
  shopt -u nullglob
}

assemble_figures() {
  # Ensure all panels from rooms are copied into tmp
  for r in "${ROOMS[@]}"; do
    copy_room_panels_to_tmp "$r" || true
  done

  # fig2
  local r7="$TMP_ROOT/random_rooms7/panels"
  copy_dir_files_flat "$r7/aliasing_76" "$OUT_ROOT/fig2"
  copy_dir_files_flat "$r7/ego_SR_76" "$OUT_ROOT/fig2"
  copy_dir_files_flat "$r7/aliasing_9" "$OUT_ROOT/fig2"
  copy_dir_files_flat "$r7/ego_SR_9" "$OUT_ROOT/fig2"
  copy_dir_files_flat "$r7/aliasing_4" "$OUT_ROOT/fig2"
  copy_dir_files_flat "$r7/ego_SR_4" "$OUT_ROOT/fig2"
  sanitize_names_in_dir "$OUT_ROOT/fig2"

  # fig3
  local gen7="$r7/generalisation"
  copy_if_exists "$gen7/generalisation_1_1000_6_path_unlesioned.png" "$OUT_ROOT/fig3/generalisation_1_1000_6_path_unlesioned.png"
  copy_if_exists "$gen7/generalisation_1_1000_6_path_lesioned.png" "$OUT_ROOT/fig3/generalisation_1_1000_6_path_lesioned.png"
  copy_if_exists "$gen7/generalisation_1_1000_6_task_world_0.png" "$OUT_ROOT/fig3/generalisation_1_1000_6_task_world_0.png"
  copy_if_exists "$gen7/generalisation_1_1000_6_task_world_1.png" "$OUT_ROOT/fig3/generalisation_1_1000_6_task_world_1.png"
  copy_if_exists "$gen7/generalisation_1_1000_6_task_world_2.png" "$OUT_ROOT/fig3/generalisation_1_1000_6_task_world_2.png"
  copy_if_exists "$gen7/generalisation_1_1000_6_task_world_3.png" "$OUT_ROOT/fig3/generalisation_1_1000_6_task_world_3.png"

  local r13="$TMP_ROOT/random_rooms13/panels/generalisation"
  # Try exact filenames; if missing, fallback to newest matching pattern
  if [[ -f "$r13/generalisation_1_1000_497_lesion_learning.png" ]]; then
    cp -f "$r13/generalisation_1_1000_497_lesion_learning.png" "$OUT_ROOT/fig3/"
  else
    local latest_lesion
    latest_lesion=$(ls -1t "$r13"/generalisation_1_1000_*_lesion_learning.png 2>/dev/null | head -n1 || true)
    [[ -n "$latest_lesion" ]] && cp -f "$latest_lesion" "$OUT_ROOT/fig3/generalisation_1_1000_497_lesion_learning.png" || warn "Missing lesion_learning for room 13"
  fi
  if [[ -f "$r13/generalisation_1_1000_497_barrier_time.png" ]]; then
    cp -f "$r13/generalisation_1_1000_497_barrier_time.png" "$OUT_ROOT/fig3/"
  else
    local latest_barrier
    latest_barrier=$(ls -1t "$r13"/generalisation_1_1000_*_barrier_time.png 2>/dev/null | head -n1 || true)
    [[ -n "$latest_barrier" ]] && cp -f "$latest_barrier" "$OUT_ROOT/fig3/generalisation_1_1000_497_barrier_time.png" || warn "Missing barrier_time for room 13"
  fi

  # Rename fig3 files to drop the leading generalisation_*_*_*_ prefix
  shopt -s nullglob
  for f in "$OUT_ROOT/fig3"/generalisation_*_path_unlesioned.png \
           "$OUT_ROOT/fig3"/generalisation_*_path_lesioned.png \
           "$OUT_ROOT/fig3"/generalisation_*_task_world_*.png \
           "$OUT_ROOT/fig3"/generalisation_*_barrier_time.png \
           "$OUT_ROOT/fig3"/generalisation_*_lesion_learning.png; do
    base=$(basename -- "$f")
    new=$(echo "$base" | sed -E 's/^generalisation_[0-9]+_[0-9]+_[0-9]+_//')
    if [[ "$base" != "$new" ]]; then mv -f "$f" "$OUT_ROOT/fig3/$new"; fi
  done
  shopt -u nullglob

  # fig4
  copy_glob "$r7/value_functions/value_functions_1_post*.png" "$OUT_ROOT/fig4"
  copy_glob "$r7/value_functions/value_functions_1_pre*.png" "$OUT_ROOT/fig4"
  sanitize_names_in_dir "$OUT_ROOT/fig4"

  # fig5
  for r in 0 2 3 12; do
    local dst="$OUT_ROOT/fig5/random_rooms${r}/room_comparisons"
    local panels_src
    panels_src=$(find_panels_dir_for_room "random_rooms${r}" || true)
    if [[ -n "$panels_src" && -d "$panels_src/room_comparisons" ]]; then
      local jid
      jid=$(echo "$panels_src" | sed -n 's#.*Results/job_ids/\([0-9][0-9]*\)/figures/.*#\1#p')
      if [[ -n "$jid" ]]; then
        info "fig5: random_rooms${r} from job_id $jid: $panels_src/room_comparisons"
      else
        info "fig5: random_rooms${r} from: $panels_src/room_comparisons"
      fi
      mkdir -p "$dst"
      rsync -a "$panels_src/room_comparisons/" "$dst/"
    else
      # Fallback to tmp if present
      local tmp_src="$TMP_ROOT/random_rooms${r}/panels/room_comparisons"
      if [[ -d "$tmp_src" ]]; then
        info "fig5: random_rooms${r} from tmp: $tmp_src"
        mkdir -p "$dst"
        rsync -a "$tmp_src/" "$dst/"
      else
        warn "Missing room_comparisons for random_rooms${r}"
      fi
    fi
  done
}

main() {
  if [[ "$ASSEMBLE_ONLY" == true ]]; then
    info "Assemble-only mode: collecting panels and building figures"
    assemble_figures
    info "Cleaning up tmp folder: $TMP_ROOT"
    rm -rf "$TMP_ROOT"
    info "All done. Figures are under: $OUT_ROOT"
    return 0
  fi

  info "Starting paper-figure assembly (pipeline + dependent assemble job)"
  local output job_ids
  output=$(run_pipeline)
  info "Pipeline submitted. Capturing job IDs..."
  job_ids=($(parse_any_job_ids "$output"))
  if [[ ${#job_ids[@]} -gt 0 ]]; then
    info "Parsed job IDs: ${job_ids[*]}"
  else
    warn "No job IDs parsed from pipeline output. Output was:\n$output"
  fi

  if [[ ${#job_ids[@]} -gt 0 ]] && command -v sbatch >/dev/null 2>&1; then
    local dep_ids
    dep_ids=$(IFS=:; echo "${job_ids[*]}")
    info "Submitting assemble job with dependency afterok:${dep_ids}"
    local assemble_job
    assemble_job=$(sbatch --parsable -J paper_assemble \
      --dependency=afterok:${dep_ids} \
      -D "$repo_root" \
      --export=ALL \
      --wrap "bash '$script_dir/paper_figs.sh' --assemble-only")
    info "Assemble job submitted: ${assemble_job}"
    echo "Assembly scheduled; exiting. Check job ${assemble_job} for progress."
  else
    warn "No job IDs parsed or sbatch unavailable; assembling synchronously."
    assemble_figures
    info "Cleaning up tmp folder: $TMP_ROOT"
    rm -rf "$TMP_ROOT"
    info "All done. Figures are under: $OUT_ROOT"
  fi
}

main "$@"


