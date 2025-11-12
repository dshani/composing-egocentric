#!/usr/bin/env python3
"""Fit GLMs for unlesioned and lesioned agents and compare barrier betas."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance

from analyse_glm_steps import _build_dataset


@dataclass
class GlmFit:
    """Summary of a fitted GLM for a specific agent prefix."""

    prefix: str
    label: str
    feature_names: Sequence[str]
    coefficients: Dict[str, float]
    coefficient_stds: Dict[str, float]
    intercept: float
    intercept_std: Optional[float]

    deviance: float
    dataset_size: int
    num_seeds: int


def _prepare_features(
    dataset: Sequence[Dict[str, float]],
    include_switch_number: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Create design matrix and response vector from the aggregated dataset."""

    feature_columns: List[np.ndarray] = []
    feature_names: List[str] = []

    if include_switch_number:
        feature_columns.append(
            np.array([row["world_index"] for row in dataset], dtype=float)
        )
        feature_names.append("world_index")

    barrier_counts = np.array([row["barrier_count"] for row in dataset], dtype=float)
    feature_columns.append(barrier_counts)
    feature_names.append("barrier_count")


    X = np.column_stack(feature_columns)
    y = np.array([row["average_steps"] for row in dataset], dtype=float)
    return X, y, feature_names


def _fit_glm_for_prefix(
    structure: Dict,
    prefix: str,
    env_switch_every: int,
    include_switch_number: bool,
    alpha: float,
    max_iter: int,
    label: str,
) -> GlmFit:
    """Aggregate data for a prefix and fit a Poisson GLM."""

    dataset = _build_dataset(structure, prefix, env_switch_every)
    if not dataset:
        raise RuntimeError(f"No trajectory data found for prefix '{prefix}'")

    X, y, feature_names = _prepare_features(dataset, include_switch_number)

    model = PoissonRegressor(alpha=alpha, fit_intercept=True, max_iter=max_iter)
    model.fit(X, y)

    predictions = model.predict(X)
    deviance = mean_poisson_deviance(y, predictions)

    seeds = sorted({row["seed"] for row in dataset})
    coefficients = {name: float(coef) for name, coef in zip(feature_names, model.coef_)}

    per_feature_samples: Dict[str, List[float]] = {name: [] for name in feature_names}
    intercept_samples: List[float] = []

    for seed in seeds:
        seed_rows = [row for row in dataset if row["seed"] == seed]
        if len(seed_rows) < len(feature_names):
            continue
        X_seed, y_seed, _ = _prepare_features(seed_rows, include_switch_number)
        if X_seed.size == 0 or y_seed.size == 0:
            continue
        seed_model = PoissonRegressor(
            alpha=alpha,
            fit_intercept=True,
            max_iter=max_iter,
        )
        try:
            seed_model.fit(X_seed, y_seed)
        except Exception:
            continue
        intercept_samples.append(float(seed_model.intercept_))
        for name, coef in zip(feature_names, seed_model.coef_):
            per_feature_samples[name].append(float(coef))

    coefficient_stds = {}
    for name in feature_names:
        samples = per_feature_samples[name]
        if len(samples) >= 2:
            coefficient_stds[name] = float(np.std(samples, ddof=1))
        elif samples:
            coefficient_stds[name] = 0.0
        else:
            coefficient_stds[name] = float("nan")

    if len(intercept_samples) >= 2:
        intercept_std = float(np.std(intercept_samples, ddof=1))
    elif intercept_samples:
        intercept_std = 0.0
    else:
        intercept_std = None


    return GlmFit(
        prefix=prefix,
        label=label,
        feature_names=feature_names,
        coefficients=coefficients,
        coefficient_stds=coefficient_stds,
        intercept=float(model.intercept_),
        intercept_std=intercept_std,

        deviance=float(deviance),
        dataset_size=len(dataset),
        num_seeds=len(seeds),
    )


def _plot_coefficients(fits: Sequence[GlmFit], destination: Path) -> None:
    """Create a grouped point plot with error bars comparing feature coefficients.

    Error bars correspond to the across-seed standard deviation of the feature
    coefficients for each agent condition.
    """


    if not fits:
        raise ValueError("At least one fit is required to plot coefficients")

    colour_overrides = {
        "unlesioned": "tab:blue",
        "lesionlec": "tab:red",
    }


    feature_names = list(fits[0].feature_names)
    for fit in fits[1:]:
        if list(fit.feature_names) != feature_names:
            raise ValueError("All fits must use the same feature ordering to plot")

    positions = np.arange(len(feature_names), dtype=float)
    offset_width = 0.15 if len(fits) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(6, 4))
    for index, fit in enumerate(fits):
        offsets = positions + (index - (len(fits) - 1) / 2) * offset_width
        heights = np.array([fit.coefficients[name] for name in feature_names], dtype=float)
        errors = np.array(
            [fit.coefficient_stds.get(name, np.nan) for name in feature_names],
            dtype=float,
        )
        errors = np.where(np.isfinite(errors), errors, 0.0)
        prefix_key = fit.prefix.lower()
        label_key = fit.label.lower()
        colour = colour_overrides.get(prefix_key)
        if colour is None:
            colour = colour_overrides.get(label_key)

        ax.errorbar(
            offsets,
            heights,
            yerr=errors,
            fmt="o",
            capsize=4,
            label=fit.label,
            color=colour,
        )


    ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([name.replace("_", " ") for name in feature_names])
    ax.set_ylabel("Coefficient (log link)")
    ax.set_title("Barrier GLM coefficients by agent condition")
    ax.legend()
    fig.tight_layout()

    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=300)
    plt.close(fig)


# def parse_args() -> argparse.Namespace:
#     """Parse command line arguments."""

#     parser = argparse.ArgumentParser(
#         description=(
#             "Fit Poisson GLMs for unlesioned and lesioned agents to predict average "
#             "steps from barrier counts (optionally including switch number)."
#         )
#     )
#     parser.add_argument(
#         "--job-id",
#         type=int,
#         help="Training job identifier used to resolve run_path.txt",
#     )
#     parser.add_argument(
#         "--run-path",
#         type=Path,
#         default=None,
#         help="Override the run directory instead of reading run_path.txt",
#     )
#     parser.add_argument(
#         "--results-dir",
#         type=Path,
#         default=Path("Results/job_ids"),
#         help="Directory that contains job metadata (default: Results/job_ids)",
#     )
#     parser.add_argument(
#         "--prefixes",
#         nargs="+",
#         default=["unlesioned", "lesionLEC"],
#         help="Agent prefixes to compare (default: unlesioned lesionLEC)",
#     )
#     parser.add_argument(
#         "--labels",
#         nargs="+",
#         default=None,
#         help="Human readable labels for prefixes (defaults to the prefix names)",
#     )
#     parser.add_argument(
#         "--env-switch-every",
#         type=int,
#         default=1000,
#         help="Number of episodes between world switches",
#     )
#     parser.add_argument(
#         "--include-switch-number",
#         action="store_true",
#         help="Include the inferred world switch number as a regressor",
#     )
#     parser.add_argument(
#         "--alpha",
#         type=float,
#         default=0.0,
#         help="L2 regularisation strength for the Poisson GLM",
#     )
#     parser.add_argument(
#         "--max-iter",
#         type=int,
#         default=1000,
#         help="Maximum solver iterations for the Poisson GLM",
#     )
#     parser.add_argument(
#         "--figure-path",
#         type=Path,
#         default=None,
#         help="Destination for the coefficient comparison figure",
#     )
#     parser.add_argument(
#         "--max-workers",
#         type=int,
#         default=8,
#         help="Thread pool size for loading trajectories",
#     )
#     return parser.parse_args()


# def main() -> None:
#     args = parse_args()

#     if args.run_path is None and args.job_id is None:
#         raise SystemExit("Specify either --run-path or --job-id.")

#     results_dir = args.results_dir.resolve()
#     results_dir.mkdir(parents=True, exist_ok=True)

#     job_dir: Optional[Path] = None
#     if args.run_path is not None:
#         run_path = args.run_path.expanduser().resolve()
#         if args.job_id is not None:
#             job_dir = results_dir / str(args.job_id)
#     else:
#         assert args.job_id is not None
#         run_path, job_dir = _resolve_run_path(args.job_id, results_dir)

#     if job_dir is None and args.job_id is not None:
#         job_dir = results_dir / str(args.job_id)
#     if job_dir is None:
#         raise SystemExit("Unable to determine output directory for metrics.")
#     job_dir.mkdir(parents=True, exist_ok=True)

#     if not run_path.exists():
#         raise FileNotFoundError(f"Resolved run directory does not exist: {run_path}")

#     if len(run_path.parents) < 2:
#         raise FileNotFoundError(
#             "Run directory is missing parent hierarchy required for load_structure"
#         )

#     save_root = run_path.parents[1]

#     structure = load_structure(
#         run=None,
#         date=None,
#         seed=None,
#         save_dirs=[save_root],
#         compare="lesion",
#         dict_params=["paths"],
#         seeds_path=run_path,
#         max_workers=args.max_workers,
#     )
#     if not structure:
#         raise RuntimeError("No data was loaded from the run directory")

#     if args.labels is not None and len(args.labels) != len(args.prefixes):
#         raise SystemExit("Number of labels must match number of prefixes")

#     if args.labels is None:
#         labels = list(args.prefixes)
#     else:
#         labels = list(args.labels)

#     fits: List[GlmFit] = []
#     for prefix, label in zip(args.prefixes, labels):
#         print(f"Fitting GLM for prefix '{prefix}' ({label})...")
#         fit = _fit_glm_for_prefix(
#             structure=structure,
#             prefix=prefix,
#             env_switch_every=args.env_switch_every,
#             include_switch_number=args.include_switch_number,
#             alpha=args.alpha,
#             max_iter=args.max_iter,
#             label=label,
#         )
#         fits.append(fit)

#         print(
#             f"  Analysed {fit.dataset_size} world segments across {fit.num_seeds} seeds."
#         )
#         print("  Coefficients (log link):")
#         for name in fit.feature_names:
#             coef = fit.coefficients[name]
#             std = fit.coefficient_stds.get(name)
#             if std is None or np.isnan(std):
#                 print(f"    {name:>13}: {coef: .6f}")
#             else:
#                 print(f"    {name:>13}: {coef: .6f} ± {std: .6f}")
#         if fit.intercept_std is None:
#             print(f"    Intercept    : {fit.intercept: .6f}")
#         else:
#             print(f"    Intercept    : {fit.intercept: .6f} ± {fit.intercept_std: .6f}")

#         print(f"  Mean Poisson deviance on training data: {fit.deviance:.6f}\n")

#     if not fits:
#         raise RuntimeError("No GLM fits were produced")

#     figure_path = (
#         args.figure_path
#         if args.figure_path is not None
#         else job_dir / "barrier_glm_coefficients.png"
#     )
#     _plot_coefficients(fits, figure_path)
#     print(f"Saved coefficient comparison figure to {figure_path}")


# if __name__ == "__main__":
#     main()

