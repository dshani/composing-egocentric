"""End-to-end analysis pipeline combining occupancy, GLM, and figure exports."""
from __future__ import annotations

import ast
import csv
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from analyse_chunked_occupancy_glm import (
    _fit_chunk_models,
    _join_records,
    _load_episode_starts,
    _load_feature,
    _load_world_breakdown,
    _populate_previous_lag_features,
    _feature_name_union,
    _write_joined_dataset,
    _write_summary,
    _plot_coefficients,
    _plot_coefficients_with_significance_bars,
    _plot_pvalues,
    _compute_world_means,
    _demean_feature,
    fit_episode_level_glm,
)
from analyse_hole_wall_chunked_occupancy import (
    _SampleConfig,
    _analyse_target,
    derive_occupancy_targets,
)
from analyse_glm_steps import _build_dataset
from compare_glm_barrier_effects import GlmFit, _fit_glm_for_prefix, _plot_coefficients as _plot_barrier_coefficients
from figure_functions_ import (
    generate_aliasing_plot,
    generate_allo_sr_fig,
    generate_generalisation_plot,
    generate_schematic,
    generate_value_fig,
    generate_ego_sr_fig,
    save_panel_crops,
)
from helper_functions_ import find_most_recent, load_recent_model, load_structure
from parameters import DotDict, parameters
from run_analysis_ import get_analysis_parser, room_comparisons
from structure_functions_ import remove_empty_dicts


_VALUE_CORRELATION_PRE_WINDOW = 10
_VALUE_CORRELATION_DELAY = 10

def _resolve_paths(args) -> Tuple[List[Path], Path, Path]:
    if args.save_dirs is None:
        save_dirs = [Path("Results").resolve()]
    else:
        save_dirs = [Path(save_dir).expanduser().resolve() for save_dir in args.save_dirs]

    run_path: Optional[Path] = None
    figure_path = Path(".")
    figures_root_override = getattr(args, "figures_root", None)
    figures_root = None
    if figures_root_override:
        figures_root = Path(figures_root_override).expanduser().resolve()

    if args.job_id is not None:
        job_id = int(args.job_id)
        job_dir = Path("./Results/job_ids") / str(job_id)
        run_path_file = job_dir / "run_path.txt"
        if not run_path_file.exists():
            raise FileNotFoundError(f"Could not locate {run_path_file}")
        run_path = Path(run_path_file.read_text().strip()).expanduser().resolve()
        if figures_root is not None:
            figure_path = figures_root
        else:
            figure_path = (job_dir / "figures").resolve()

        args_file = job_dir / "args.txt"
        if args_file.exists():
            loaded_args = DotDict(json.loads(args_file.read_text()))
            for key, value in vars(loaded_args).items():
                if getattr(args, key, None) is None and value is not None:
                    setattr(args, key, value)
    else:
        save_root = save_dirs[0]
        date = args.date if args.date is not None else max(os.listdir(save_root))
        compare = parameters.compare if parameters.compare is not None else "run"
        if args.run is not None:
            run = args.run
        else:
            candidates = os.listdir(save_root / date)
            run = find_most_recent(candidates, must_contain=[compare], recent=-1)[0]
        run_path = (save_root / date / run / compare).resolve()
        if figures_root is not None:
            figure_path = figures_root
        else:
            figure_path = (save_root / date / run / "figures").resolve()

    if run_path is None:
        raise RuntimeError("Unable to determine run directory")

    return save_dirs, run_path, figure_path


def _extract_worlds_type(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    text = path.read_text()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            payload = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
    if isinstance(payload, dict):
        worlds_type = payload.get("worlds_type")
        if worlds_type is not None:
            return str(worlds_type)
    return None


def _determine_room_label(args) -> str:
    worlds_type = getattr(args, "worlds_type", None)
    if worlds_type:
        return str(worlds_type)

    job_id = getattr(args, "job_id", None)
    if job_id is not None:
        job_dir = Path("./Results/job_ids") / str(int(job_id))
        for filename in ("args.txt", "params.txt"):
            label = _extract_worlds_type(job_dir / filename)
            if label:
                return label

    fallback = getattr(parameters, "worlds_type", None)
    if fallback:
        return str(fallback)

    return "default"


def _load_structures(
    args,
    save_dirs: Sequence[Path],
    run_path: Path,
) -> Tuple[Dict[str, object], Optional[Dict[str, object]], Optional[object]]:
    base_params = ["paths", "accuracies", "worlds"]
    dict_params = list(base_params)

    def _extend_params(values):
        if values is None:
            return
        if isinstance(values, (list, tuple, set)):
            iterable = values
        else:
            iterable = [values]
        for item in iterable:
            if item not in dict_params:
                dict_params.append(item)

    _extend_params(getattr(args, "dict_params", None))
    value_snapshot_times = None
    if not getattr(args, "room_compare_only", False):
        _extend_params([
            "value_snapshots",
            "weight",
            "ego_SR.SR_ss",
            "allo_SR.SR_ss",
        ])
        switch_every = getattr(args, "env_switch_every", None) or 0
        pre_time = max(switch_every - _VALUE_CORRELATION_PRE_WINDOW, 0)
        post_time = switch_every + _VALUE_CORRELATION_DELAY
        value_snapshot_times = [pre_time, post_time]

    struct_all_seeds = load_structure(
        args.run,
        args.date,
        args.seed,
        save_dirs,
        dict_params=dict_params,
        compare="lesion",
        seeds_path=run_path,
        debug=args.debug,
        load_worlds=True,
        max_workers=args.max_workers,
        param_time_slices=(
            {
                key: value_snapshot_times
                for key in ("value_snapshots", "ego_SR.SR_ss", "allo_SR.SR_ss", "weight")
                if key in dict_params
            }
            if value_snapshot_times is not None
            else None
        ),
    )
    struct_all_seeds = remove_empty_dicts(struct_all_seeds)

    struct_single_seed = None
    model = None
    target_seed = args.single_seed
    if target_seed is None:
        # Attempt to infer from first structure key
        if struct_all_seeds:
            try:
                target_seed = int(next(iter(struct_all_seeds)))
            except (ValueError, TypeError):
                target_seed = next(iter(struct_all_seeds))
    if target_seed is not None:
        struct_single_seed, model = load_recent_model(
            args.run,
            args.date,
            target_seed,
            save_dirs,
            args.recent,
            dict_params=args.dict_params,
            load_params=args.load_params,
            seeds_path=run_path,
            compare="lesion",
            max_workers=args.max_workers,
        )
    return struct_all_seeds, struct_single_seed, model


def _run_chunked_occupancy(
    structure: Mapping[str, object],
    run_path: Path,
    job_id: Optional[int],
    env_switch_every: int,
    chunk_size: int,
    barrier_thickness: int,
    wall_thickness: int,
    output_root: Path,
    max_workers: Optional[int],
) -> Dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    if env_switch_every and env_switch_every > 0:
        # Begin sampling once the agent has entered the second environment while
        # keeping the upper bound open so later worlds contribute additional lag
        # regressors as soon as the data exists.
        window_start = env_switch_every + 1
        episode_window = (window_start, None)
    else:
        episode_window = (1001, None)

    sample_config = _SampleConfig(
        episode_window=episode_window,
        agent_prefixes=frozenset({"lesionLEC"}),
        min_start_distance=None,
        occupancy_thresholds=None,
        max_samples=None,
    )

    target_dirs: Dict[str, Path] = {}
    for target in derive_occupancy_targets(structure):
        target_dir = output_root / target
        _analyse_target(
            structure=structure,
            metadata=None,
            run_path=run_path,
            job_id=job_id,
            env_switch_every=env_switch_every,
            chunk_size=chunk_size,
            occupancy_target=target,
            barrier_thickness=barrier_thickness,
            wall_thickness=wall_thickness,
            output_dir=target_dir,
            max_workers=max_workers,
            enable_parallel=False,
            sample_config=sample_config,
        )
        target_dirs[target] = target_dir
    return target_dirs


def _ensure_zero_barrier_current_metrics(
    structure: Mapping[str, object],
    run_path: Path,
    job_id: Optional[int],
    env_switch_every: int,
    chunk_size: int,
    wall_thickness: int,
    output_root: Path,
    max_workers: Optional[int],
) -> Path:
    """Ensure barrier-0 current-hole metrics exist for generalisation plots."""

    output_dir = output_root / "current_hole_locations_barrier0"
    metrics_path = output_dir / "chunked_occupancy_metrics.json"
    if metrics_path.exists():
        return output_dir

    if not structure:
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    if env_switch_every and env_switch_every > 0:
        window_start = env_switch_every + 1
        episode_window = (window_start, None)
    else:
        episode_window = (1001, None)

    sample_config = _SampleConfig(
        episode_window=episode_window,
        agent_prefixes=frozenset({"lesionLEC"}),
        min_start_distance=None,
        occupancy_thresholds=None,
        max_samples=None,
    )

    _analyse_target(
        structure=structure,
        metadata=None,
        run_path=run_path,
        job_id=job_id,
        env_switch_every=env_switch_every,
        chunk_size=chunk_size,
        occupancy_target="current_hole_locations",
        barrier_thickness=0,
        wall_thickness=wall_thickness,
        output_dir=output_dir,
        max_workers=max_workers,
        enable_parallel=False,
        sample_config=sample_config,
    )

    return output_dir


def _ensure_zero_barrier_previous_metrics(
    structure: Mapping[str, object],
    run_path: Path,
    job_id: Optional[int],
    env_switch_every: int,
    chunk_size: int,
    wall_thickness: int,
    output_root: Path,
    max_workers: Optional[int],
) -> Path:
    """Ensure barrier-0 previous-hole metrics exist for selection."""

    output_dir = output_root / "previous_hole_locations_barrier0"
    metrics_path = output_dir / "chunked_occupancy_metrics.json"
    if metrics_path.exists():
        return output_dir

    if not structure:
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    if env_switch_every and env_switch_every > 0:
        window_start = env_switch_every + 1
        episode_window = (window_start, None)
    else:
        episode_window = (1001, None)

    sample_config = _SampleConfig(
        episode_window=episode_window,
        agent_prefixes=frozenset({"lesionLEC", "unlesioned"}),
        min_start_distance=None,
        occupancy_thresholds=None,
        max_samples=None,
    )

    _analyse_target(
        structure=structure,
        metadata=None,
        run_path=run_path,
        job_id=job_id,
        env_switch_every=env_switch_every,
        chunk_size=chunk_size,
        occupancy_target="previous_hole_locations",
        barrier_thickness=0,
        wall_thickness=wall_thickness,
        output_dir=output_dir,
        max_workers=max_workers,
        enable_parallel=False,
        sample_config=sample_config,
    )

    return output_dir


def _select_sample(
    sample_json: Path,
    max_delay: Optional[int] = None,
    objective: str = "lesion_fraction",
    prev_json: Optional[Path] = None,
) -> Tuple[Optional[str], Optional[int]]:
    if not sample_json.exists():
        return None, None
    payload = json.loads(sample_json.read_text())
    records = payload.get("sample_paths", {}).get("records", [])
    episodes_cur = payload.get("episodes", {}) or {}

    # Build episode maps for current holes (unlesioned only for diff)
    cur_un_map_by_seed: Dict[str, Dict[int, float]] = {}
    for seed_key, per_prefix in episodes_cur.items():
        seed_key_str = str(seed_key)
        if isinstance(per_prefix, dict) and "unlesioned" in per_prefix:
            m: Dict[int, float] = {}
            for row in per_prefix["unlesioned"] or []:
                try:
                    ep = int(row.get("episode"))
                    frac = float(row.get("occupancy_fraction"))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(frac):
                    m[ep] = frac
            if m:
                cur_un_map_by_seed[seed_key_str] = m

    # Optionally load previous-hole episodes (both prefixes)
    prev_un_map_by_seed: Dict[str, Dict[int, float]] = {}
    prev_les_map_by_seed: Dict[str, Dict[int, float]] = {}
    if prev_json and prev_json.exists():
        prev_payload = json.loads(prev_json.read_text())
        episodes_prev = prev_payload.get("episodes", {}) or {}
        for seed_key, per_prefix in episodes_prev.items():
            seed_key_str = str(seed_key)
            if not isinstance(per_prefix, dict):
                continue
            for prefix_name, rows in per_prefix.items():
                if not rows:
                    continue
                target = prev_les_map_by_seed if prefix_name == "lesionLEC" else prev_un_map_by_seed
                m = target.setdefault(seed_key_str, {})
                for row in rows:
                    try:
                        ep = int(row.get("episode"))
                        frac = float(row.get("occupancy_fraction"))
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(frac):
                        m[ep] = frac

    best_record = None
    best_score = -math.inf

    for record in records:
        if record.get("agent") != "lesionLEC":
            continue
        try:
            distance = float(record.get("start_distance", "nan"))
            lesion_frac_cur = float(record.get("occupancy_fraction", "nan"))
            delay_val = int(record.get("delay"))
            ep = int(record.get("episode"))
            seed_key = str(record.get("seed"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(distance) or not math.isfinite(lesion_frac_cur):
            continue
        if distance <= 30:
            continue
        if max_delay is not None and delay_val > max_delay:
            continue

        un_frac_cur = cur_un_map_by_seed.get(seed_key, {}).get(ep)

        if objective == "lesion_fraction":
            score = lesion_frac_cur
        elif objective == "fraction_diff":
            if un_frac_cur is None or not math.isfinite(un_frac_cur):
                continue
            score = lesion_frac_cur - un_frac_cur
        elif objective == "sum_diff_opposite":
            if un_frac_cur is None or not math.isfinite(un_frac_cur):
                continue
            cur_diff = lesion_frac_cur - un_frac_cur
            les_prev = prev_les_map_by_seed.get(seed_key, {}).get(ep)
            un_prev = prev_un_map_by_seed.get(seed_key, {}).get(ep)
            if les_prev is not None and un_prev is not None and math.isfinite(les_prev) and math.isfinite(un_prev):
                prev_diff = (un_prev - les_prev)
            else:
                prev_diff = 0.0
            score = cur_diff + prev_diff
        else:  # "lesion_cur_minus_prev"
            les_prev = prev_les_map_by_seed.get(seed_key, {}).get(ep)
            if les_prev is not None and math.isfinite(les_prev):
                score = lesion_frac_cur - les_prev
            else:
                score = lesion_frac_cur

        if score > best_score:
            best_score = score
            best_record = record

    if not best_record:
        return None, None
    seed_value = best_record.get("seed")
    delay_value = best_record.get("delay")
    try:
        delay_int = int(delay_value)
    except (TypeError, ValueError):
        delay_int = None
    return seed_value, delay_int


def _run_glm_pipeline(
    occupancy_dirs: Mapping[str, Path],
    output_dir: Path,
    positive_label: str = "lesionLEC",
    workers: int = 1,
    demean_by_world: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    current_dir = occupancy_dirs["current_hole_locations"]
    previous_dir = occupancy_dirs["previous_hole_locations"]
    walls_dir = occupancy_dirs["walls"]

    current_map = _load_feature(current_dir, "current-hole")
    previous_map = _load_feature(previous_dir, "previous-hole")
    walls_map = _load_feature(walls_dir, "walls")
    episode_map = _load_episode_starts(current_dir)

    joined_records = _join_records(current_map, previous_map, walls_map, episode_map)

    # Add flexible per-lag features f_holes_world_t_minus_k using the
    # breakdowns computed for previous-hole occupancies.
    try:
        previous_breakdown, _ = _load_world_breakdown(previous_dir)
    except FileNotFoundError:
        previous_breakdown = {}
    if previous_breakdown:
        _populate_previous_lag_features(joined_records, previous_breakdown)

    if demean_by_world:
        current_breakdown, current_totals = _load_world_breakdown(current_dir)
        previous_breakdown, previous_totals = _load_world_breakdown(previous_dir)
        walls_breakdown, walls_totals = _load_world_breakdown(walls_dir)

        current_means = _compute_world_means(current_totals)
        previous_means = _compute_world_means(previous_totals)
        walls_means = _compute_world_means(walls_totals)

        for record in joined_records:
            key = (record.seed, record.agent, record.chunk_index)

            current_value = record.get_feature("f_current")
            record.set_feature(
                "f_current",
                _demean_feature(
                    current_value,
                    current_breakdown.get(key),
                    current_means,
                ),
            )

            previous_value = record.get_feature("f_previous")
            record.set_feature(
                "f_previous",
                _demean_feature(
                    previous_value,
                    previous_breakdown.get(key),
                    previous_means,
                ),
            )

            wall_value = record.get_feature("f_walls")
            record.set_feature(
                "f_walls",
                _demean_feature(
                    wall_value,
                    walls_breakdown.get(key),
                    walls_means,
                ),
            )

    dataset_path = output_dir / "chunked_occupancy_features.csv"
    _write_joined_dataset(dataset_path, joined_records)

    # Determine feature sets for model fitting and reporting
    feature_names, extra_feature_names = _feature_name_union(joined_records)

    results = _fit_chunk_models(
        joined_records,
        positive_label=positive_label,
        inverse_regularisation=1.0,
        max_iter=1000,
        centre_within_seed=False,
        feature_names=feature_names,
        extra_feature_names=extra_feature_names,
        worker_count=max(1, workers),
    )

    summary_path = output_dir / "chunk_glm_coefficients.csv"
    _write_summary(
        summary_path,
        results,
        centred=False,
        demeaned=demean_by_world,
        extra_feature_names=extra_feature_names,
    )

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_coefficients(
        figures_dir / "chunk_glm_coefficients.png",
        results,
        centred=False,
        demeaned=demean_by_world,
        positive_label=positive_label,
        significance_level=0.05,
        extra_feature_names=extra_feature_names,
    )
    _plot_coefficients_with_significance_bars(
        figures_dir / "chunk_glm_coefficients_all_values.png",
        results,
        centred=False,
        demeaned=demean_by_world,
        positive_label=positive_label,
        significance_level=0.05,
        extra_feature_names=extra_feature_names,
    )
    _plot_pvalues(
        figures_dir / "chunk_glm_pvalues.png",
        results,
        centred=False,
        demeaned=demean_by_world,
        significance_level=0.05,
        extra_feature_names=extra_feature_names,
    )


def _save_combined_and_panels(
    fig,
    axes_map,
    base_path: Path,
    panel_dir: Path,
    dpi: int = 1200,
    save_combined: bool = True,
) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    stem = base_path.stem
    if save_combined:
        import matplotlib as mpl

        # Match ``run_analysis_.py`` by rendering SVG text as paths so glyphs do not
        # overlap or disappear in downstream consumers such as Inkscape.
        with mpl.rc_context({"svg.fonttype": "path"}):
            fig.savefig(base_path.with_suffix(".png"), dpi=dpi)
            fig.savefig(base_path.with_suffix(".svg"), format="svg")
    save_panel_crops(fig, axes_map, panel_dir, stem)
    plt_close(fig)


def plt_close(fig) -> None:
    import matplotlib.pyplot as plt  # Local import to keep tests lightweight

    plt.close(fig)


def _maybe_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _write_barrier_glm_summary(path: Path, fits: Sequence[GlmFit]) -> None:
    fieldnames = [
        "prefix",
        "label",
        "feature",
        "coefficient",
        "coefficient_std",
        "dataset_size",
        "num_seeds",
        "deviance",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for fit in fits:
            for feature in list(fit.feature_names) + ["intercept"]:
                if feature == "intercept":
                    coef = fit.intercept
                    std = fit.intercept_std
                else:
                    coef = fit.coefficients.get(feature)
                    std = fit.coefficient_stds.get(feature)
                writer.writerow(
                    {
                        "prefix": fit.prefix,
                        "label": fit.label,
                        "feature": feature,
                        "coefficient": "" if coef is None else f"{coef:.10g}",
                        "coefficient_std": ""
                        if std is None or not math.isfinite(std)
                        else f"{std:.10g}",
                        "dataset_size": fit.dataset_size,
                        "num_seeds": fit.num_seeds,
                        "deviance": f"{fit.deviance:.10g}",
                    }
                )


def _run_barrier_glm(
    structure: Mapping[str, object],
    env_switch_every: int,
    output_dir: Path,
) -> None:
    if not structure:
        logging.warning("No structure data available for barrier GLM analysis")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    prefixes = ["unlesioned", "lesionLEC"]
    labels = ["Unlesioned", "Lesion LEC"]

    fits: List[GlmFit] = []
    for prefix, label in zip(prefixes, labels):
        dataset = _build_dataset(structure, prefix, env_switch_every)
        if not dataset:
            logging.warning("No barrier dataset produced for prefix %s", prefix)
            continue
        dataset_path = output_dir / f"{prefix}_barrier_dataset.json"
        dataset_path.write_text(json.dumps(dataset, indent=2))
        fit = _fit_glm_for_prefix(
            structure=structure,
            prefix=prefix,
            env_switch_every=env_switch_every,
            include_switch_number=False,
            alpha=0.0,
            max_iter=1000,
            label=label,
        )
        fits.append(fit)

    if not fits:
        logging.warning("Barrier GLM fits were not produced; skipping plots")
        return

    _write_barrier_glm_summary(output_dir / "barrier_glm_coefficients.csv", fits)
    _plot_barrier_coefficients(fits, output_dir / "barrier_glm_coefficients.png")


def main() -> None:
    parser = get_analysis_parser()
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--barrier-thickness", type=int, default=1)
    parser.add_argument("--wall-thickness", type=int, default=1)
    parser.add_argument(
        "--episode-glm",
        action="store_true",
        help="Fit and plot episode-level GLMs per chunk in addition to chunk-level GLMs.",
    )
    parser.add_argument(
        "--produce-schematic",
        action="store_true",
        help="Export a maze schematic alongside other room figures.",
    )
    parser.add_argument(
        "--select-max-delay",
        type=int,
        default=None,
        help="When auto-selecting sample paths, only consider candidates with delay <= this value.",
    )
    parser.add_argument(
        "--select-objective",
        choices=[
            "lesion_fraction",
            "fraction_diff",
            "sum_diff_opposite",
            "lesion_cur_minus_prev",
        ],
        default="lesion_fraction",
        help=(
            "Auto-selection objective: lesion_fraction; fraction_diff (lesioned - unlesioned current); "
            "sum_diff_opposite ((lesioned - unlesioned) current + (unlesioned - lesioned) previous); "
            "lesion_cur_minus_prev (lesioned current - lesioned previous)."
        ),
    )
    parser.add_argument(
        "--path-scan",
        type=int,
        default=None,
        help=(
            "In model-analysis mode, export only path panels for both agents "
            "for all seeds and all delays < N (this value) into the debug figures folder."
        ),
    )
    args = parser.parse_args()

    if getattr(args, "metrics_only", False):
        raise SystemExit("analyse_full_pipeline.py does not support --metrics_only")

    for arg_name in vars(args):
        arg_value = getattr(args, arg_name)
        if arg_value is not None:
            parameters[arg_name] = arg_value
    parameters.compare = "lesion"

    if args.recent is None:
        args.recent = -2

    save_dirs, run_path, figure_path = _resolve_paths(args)
    figure_path.mkdir(parents=True, exist_ok=True)

    room_label = _determine_room_label(args)
    room_root = figure_path / "room_plots" / room_label
    room_root.mkdir(parents=True, exist_ok=True)
    logging.info("Saving room-specific outputs to %s", room_root)

    struct_all_seeds, struct_single_seed, model = _load_structures(args, save_dirs, run_path)

    job_id_value = int(args.job_id) if args.job_id is not None else None

    _ensure_zero_barrier_current_metrics(
        structure=struct_all_seeds,
        run_path=run_path,
        job_id=job_id_value,
        env_switch_every=args.env_switch_every,
        chunk_size=args.chunk_size,
        wall_thickness=args.wall_thickness,
        output_root=room_root / "chunked_occupancy",
        max_workers=args.max_workers,
    )
    prev_barrier0_dir = _ensure_zero_barrier_previous_metrics(
        structure=struct_all_seeds,
        run_path=run_path,
        job_id=job_id_value,
        env_switch_every=args.env_switch_every,
        chunk_size=args.chunk_size,
        wall_thickness=args.wall_thickness,
        output_root=room_root / "chunked_occupancy",
        max_workers=args.max_workers,
    )

    occupancy_dirs = {}
    sample_seed = args.single_seed
    sample_delay = args.delay

    if args.model_analysis and sample_seed is None:
        sample_seed = 1

    run_occupancy = not args.room_compare_only and not args.model_analysis

    if run_occupancy:
        occupancy_dirs = _run_chunked_occupancy(
            struct_all_seeds,
            run_path,
            job_id_value,
            args.env_switch_every,
            args.chunk_size,
            args.barrier_thickness,
            args.wall_thickness,
            room_root / "chunked_occupancy",
            args.max_workers,
        )
        sample_json = occupancy_dirs["current_hole_locations"] / "chunked_occupancy_metrics.json"
        prev_json = occupancy_dirs["previous_hole_locations"] / "chunked_occupancy_metrics.json"
        seed_from_occupancy, delay_from_occupancy = _select_sample(
            sample_json,
            getattr(args, "select_max_delay", None),
            getattr(args, "select_objective", "lesion_fraction"),
            prev_json=prev_json,
        )
        if seed_from_occupancy is not None:
            sample_seed = seed_from_occupancy
        if delay_from_occupancy is not None:
            sample_delay = delay_from_occupancy

        _run_glm_pipeline(
            occupancy_dirs,
            room_root / "chunked_occupancy_glm",
            positive_label="lesionLEC",
            workers=args.max_workers or 1,
            demean_by_world=False,
        )
        if args.episode_glm:
            fit_episode_level_glm(
                occupancy_dirs["current_hole_locations"],
                occupancy_dirs["previous_hole_locations"],
                occupancy_dirs["walls"],
                room_root / "chunked_occupancy_glm" / "episode_level",
                positive_label="lesionLEC",
                max_iter=1000,
            )
        _run_glm_pipeline(
            occupancy_dirs,
            room_root / "chunked_occupancy_glm_world_demeaned",
            positive_label="lesionLEC",
            workers=args.max_workers or 1,
            demean_by_world=True,
        )
        if args.episode_glm:
            fit_episode_level_glm(
                occupancy_dirs["current_hole_locations"],
                occupancy_dirs["previous_hole_locations"],
                occupancy_dirs["walls"],
                room_root / "chunked_occupancy_glm_world_demeaned" / "episode_level",
                positive_label="lesionLEC",
                max_iter=1000,
            )
        _run_barrier_glm(struct_all_seeds, args.env_switch_every, room_root / "barrier_glm")

    selected_seed_int = _maybe_int(sample_seed)
    if selected_seed_int is not None:
        args.single_seed = selected_seed_int
    elif sample_seed is not None:
        args.single_seed = sample_seed

    if args.model_analysis:
        # Try to derive (seed, delay) from barrier-0 metrics even when not running full occupancy.
        if sample_delay is None:
            try:
                barrier0_json = room_root / "chunked_occupancy" / "current_hole_locations_barrier0" / "chunked_occupancy_metrics.json"
                prev_barrier0_json = room_root / "chunked_occupancy" / "previous_hole_locations_barrier0" / "chunked_occupancy_metrics.json"
                seed_from_barrier0, delay_from_barrier0 = _select_sample(
                    barrier0_json,
                    getattr(args, "select_max_delay", None),
                    getattr(args, "select_objective", "lesion_fraction"),
                    prev_json=prev_barrier0_json,
                )
                if seed_from_barrier0 is not None:
                    sample_seed = seed_from_barrier0
                if delay_from_barrier0 is not None:
                    sample_delay = delay_from_barrier0
            except Exception:
                # Fall back silently if metrics are missing or malformed
                pass

        if sample_delay is None:
            sample_delay = args.delay if args.delay is not None else 9
        generalisation_delay = sample_delay
        value_delay = 1
    else:
        if sample_delay is None:
            sample_delay = args.delay if args.delay is not None else 9
        generalisation_delay = sample_delay
        value_delay = _VALUE_CORRELATION_DELAY

    if args.generalisation_only or not args.room_compare_only:
        if struct_single_seed is None or model is None:
            struct_single_seed, model = load_recent_model(
                args.run,
                args.date,
                args.single_seed,
                save_dirs,
                args.recent,
                dict_params=args.dict_params,
                load_params=args.load_params,
                seeds_path=run_path,
                compare="lesion",
                max_workers=args.max_workers,
            )

    panel_root = room_root / "panels"

    # Path-scan: iterate all seeds and delays (< N) to export only path panels, then exit
    if args.model_analysis and getattr(args, "path_scan", None):
        try:
            delay_limit = int(args.path_scan)
        except Exception:
            delay_limit = None
        if delay_limit is not None and delay_limit > 0:
            # Load the full set of seeds regardless of debug flag to avoid truncation
            try:
                struct_all_seeds_full = load_structure(
                    args.run,
                    args.date,
                    args.seed,
                    save_dirs,
                    dict_params=["paths", "accuracies", "worlds"],
                    compare="lesion",
                    seeds_path=run_path,
                    debug=False,
                    load_worlds=True,
                    max_workers=args.max_workers,
                )
                struct_all_seeds_full = remove_empty_dicts(struct_all_seeds_full)
            except Exception:
                struct_all_seeds_full = struct_all_seeds
            seed_keys = list(struct_all_seeds_full.keys())
            for seed_key in seed_keys:
                # Normalise seed type to int when possible for filenames and loading
                try:
                    seed_id = int(seed_key)
                except Exception:
                    seed_id = seed_key
                # Load per-seed model/context
                try:
                    struct_single_seed_scan, model_scan = load_recent_model(
                        args.run,
                        args.date,
                        seed_id,
                        save_dirs,
                        args.recent,
                        dict_params=args.dict_params,
                        load_params=args.load_params,
                        seeds_path=run_path,
                        compare="lesion",
                        max_workers=args.max_workers,
                    )
                except Exception:
                    logging.warning("Skipping seed %s due to load failure", seed_key)
                    continue
                if model_scan is None:
                    logging.warning("Skipping seed %s: model not available", seed_key)
                    continue
                for delay_value in range(1, delay_limit):
                    try:
                        fig, axes_map = generate_generalisation_plot(
                            model=model_scan,
                            struct_all_seeds=struct_all_seeds_full,
                            struct_single_seed=struct_single_seed_scan,
                            sigma=1,
                            path=room_root,
                            env_switch_every=args.env_switch_every,
                            delay=delay_value,
                            seednum=seed_id,
                            save_combined=False,
                        )
                    except Exception:
                        logging.warning(
                            "Failed to generate path panels for seed %s delay %s",
                            seed_key,
                            delay_value,
                        )
                        continue
                    export_axes_map = {
                        name: axes for name, axes in axes_map.items() if name.startswith("path_")
                    }
                    if not export_axes_map:
                        logging.info(
                            "No path panels returned for seed %s delay %s; skipping",
                            seed_key,
                            delay_value,
                        )
                        continue
                    stem = f"generalisation_1_{args.env_switch_every}_{delay_value}_{seed_id}"
                    base = room_root / stem
                    _save_combined_and_panels(
                        fig,
                        export_axes_map,
                        base,
                        panel_root / stem,
                        dpi=300,
                        save_combined=False,
                    )
        return

    def export_generalisation():
        if model is None:
            return
        fig, axes_map = generate_generalisation_plot(
            model=model,
            struct_all_seeds=struct_all_seeds,
            struct_single_seed=struct_single_seed,
            sigma=1,
            path=room_root,
            env_switch_every=args.env_switch_every,
            delay=generalisation_delay,
            seednum=args.single_seed,
            save_combined=False,
        )
        stem = f"generalisation_1_{args.env_switch_every}_{generalisation_delay}"
        base = room_root / stem
        save_combined = not args.model_analysis
        export_axes_map = axes_map
        if args.model_analysis:
            export_axes_map = {
                name: axes
                for name, axes in axes_map.items()
                if name.startswith("path_") or name.startswith("task_world_")
            }
            if not export_axes_map:
                logging.warning(
                    "Model analysis requested but no path panels were returned from the generalisation figure"
                )
        _save_combined_and_panels(
            fig,
            export_axes_map,
            base,
            panel_root / "generalisation",
            save_combined=save_combined,
        )

    def export_room_compare():
        fig, axes_map = room_comparisons(
            model=model,
            struct_single_seed=struct_single_seed,
            struct_all_seeds=struct_all_seeds,
            sigma=1,
            path=room_root,
            env_switch_every=args.env_switch_every,
            delay=0,
            seednum=args.single_seed,
            save_combined=False,
        )
        stem = f"room_compare_1_{args.env_switch_every}_0"
        base = room_root / stem
        _save_combined_and_panels(
            fig,
            axes_map,
            base,
            panel_root / "room_comparisons",
            save_combined=not args.room_compare_only,
        )

    def export_value_fig():
        if model is None:
            return
        result = generate_value_fig(
            model=model,
            struct_single_seed=struct_single_seed,
            struct_all_seeds=struct_all_seeds,
            sigma=0.1,
            path=room_root,
            env_switch_every=args.env_switch_every,
            delay=value_delay,
            seednum=args.single_seed,
            save_combined=False,
        )
        if result is None:
            logging.info(
                "Value-function summary figure skipped; required parameter snapshots were not available."
            )
            return

        fig, axes_map, extra_figures = result
        stem = f"value_functions_{value_delay}"
        base = room_root / stem
        if fig is not None:
            _save_combined_and_panels(fig, axes_map, base, panel_root / "value_functions")
        else:
            logging.info(
                "Value-function heatmaps skipped; continuing with correlation outputs only."
            )
        for extra_stem, extra_fig, extra_axes_map in extra_figures:
            extra_base = room_root / extra_stem
            _save_combined_and_panels(
                extra_fig,
                extra_axes_map,
                extra_base,
                panel_root / "value_functions",
            )
            logging.info(
                "Saved %s.{png,svg} and panel crops under %s",
                extra_base,
                panel_root / "value_functions",
            )
        if not extra_figures:
            logging.info(
                "Value-function correlation figure was not generated; "
                "no valid parameter snapshots were available for correlation analysis.",
            )

    def export_sr_figures():
        if model is None:
            return
        ego_results = generate_ego_sr_fig(
            model=model,
            struct_single_seed=struct_single_seed,
            struct_all_seeds=struct_all_seeds,
            path=room_root,
            states=[4, 6, 7, 9, 76],
            save_combined=False,
        )
        for state, (fig, axes_map) in ego_results.items():
            stem = f"ego_SR_{state}"
            base = room_root / stem
            _save_combined_and_panels(fig, axes_map, base, panel_root / stem, dpi=300)

        allo_results = generate_allo_sr_fig(
            model=model,
            struct_single_seed=struct_single_seed,
            struct_all_seeds=struct_all_seeds,
            path=room_root,
            states=[200],
            save_combined=False,
        )
        for state, (fig, axes_map) in allo_results.items():
            stem = f"allo_SR_{state}"
            base = room_root / stem
            _save_combined_and_panels(fig, axes_map, base, panel_root / stem, dpi=300)

    def export_aliasing():
        if model is None:
            return
        for ego in [4, 6, 7, 9, 76]:
            fig, axes_map = generate_aliasing_plot(
                model.env,
                ego,
                path=room_root,
                world=None,
                save_combined=False,
            )
            stem = f"aliasing_{ego}"
            base = room_root / stem
            _save_combined_and_panels(fig, axes_map, base, panel_root / stem, dpi=300)

    def export_schematic():
        if model is None:
            return
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        try:
            generate_schematic(ax, model.env)
            base = room_root / "schematic"
            base.parent.mkdir(parents=True, exist_ok=True)
            with mpl.rc_context({"svg.fonttype": "path"}):
                fig.savefig(base.with_suffix(".png"), dpi=300)
                fig.savefig(base.with_suffix(".svg"), format="svg")
        finally:
            plt.close(fig)

    if args.generalisation_only:
        if args.produce_schematic and not args.model_analysis:
            export_schematic()
        export_generalisation()
        return

    if not args.room_compare_only:
        export_generalisation()
    if not args.model_analysis:
        export_room_compare()
    if args.produce_schematic and not args.model_analysis:
        export_schematic()
    if not args.room_compare_only:
        export_value_fig()
        export_sr_figures()
        export_aliasing()


if __name__ == "__main__":
    main()
