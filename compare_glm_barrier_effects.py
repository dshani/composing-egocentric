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
