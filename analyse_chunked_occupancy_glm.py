r"""Chunk-wise logistic regression over combined occupancy metrics.

This script joins the per-target outputs produced by
``analyse_hole_wall_chunked_occupancy.py`` and, for every chunk index, fits a
logistic-regression generalised linear model of the form

.. math::

   \log \frac{P(y=1 \mid x)}{1 - P(y=1 \mid x)} =
       \beta_0 + \beta_1 f_{\text{current}} + \beta_2 f_{\text{previous}} +
       \beta_3 f_{\text{walls}},

where ``y`` indicates whether the row belongs to the positive agent label
(default: ``lesionLEC``) and the features correspond to the occupancy fractions
for the three masks.  The per-chunk fits report standard errors, Wald
``z``-scores, and two-sided ``p``-values alongside the coefficients so the
exported summaries and plots highlight statistically significant effects. When
``--centre-within-seed`` is enabled the features are centred within each
``(seed, chunk_index)`` group so the model focuses on the lesion vs. control
contrast instead of cross-seed level shifts.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError


_BASE_FEATURE_NAMES = ("f_current", "f_previous", "f_walls")
_DEFAULT_POSITIVE_LABEL = "lesionLEC"
_SIGNIFICANCE_LEVEL = 0.05


@dataclass
class Record:
    """Joined occupancy metrics for a single seed/agent/chunk combination."""

    seed: str
    agent: str
    chunk_index: int
    features: Dict[str, float]
    episode_start: int | None = None
    chunk_size: int | None = None

    def get_feature(self, name: str) -> float:
        value = self.features.get(name)
        if value is None:
            return 0.0
        return float(value)

    def has_feature(self, name: str) -> bool:
        return name in self.features

    def set_feature(self, name: str, value: float) -> None:
        self.features[name] = float(value)

    def to_row(self, feature_names: Sequence[str]) -> List[str]:
        base: List[str] = [
            self.seed,
            self.agent,
            str(self.chunk_index),
            str(self.episode_start) if self.episode_start is not None else "",
            str(self.chunk_size) if self.chunk_size is not None else "",
        ]
        for name in feature_names:
            value = self.get_feature(name)
            base.append(f"{value:.10g}" if math.isfinite(value) else "nan")
        return base


def _copy_record(record: Record) -> Record:
    return Record(
        seed=record.seed,
        agent=record.agent,
        chunk_index=record.chunk_index,
        features=dict(record.features),
        episode_start=record.episode_start,
        chunk_size=record.chunk_size,
    )


def _feature_name_union(records: Sequence[Record]) -> Tuple[List[str], List[str]]:
    """Return full feature list (base first) and the extra feature subset."""

    all_names = set(_BASE_FEATURE_NAMES)
    for record in records:
        all_names.update(record.features.keys())

    base = list(_BASE_FEATURE_NAMES)
    extras = sorted(name for name in all_names if name not in _BASE_FEATURE_NAMES)
    return base + extras, extras


def _chunk_feature_subset(
    records: Sequence[Record], feature_names: Sequence[str]
) -> List[str]:
    """Return the features with explicit data in ``records`` (base always kept)."""

    active: List[str] = []
    for name in feature_names:
        if name in _BASE_FEATURE_NAMES:
            # Always include current and walls; include previous only if it is
            # explicitly present in at least one record (pre-switch chunks omit it).
            if name in {"f_current", "f_walls"}:
                active.append(name)
                continue
            if any(record.has_feature(name) for record in records):
                active.append(name)
                continue
            # Skip f_previous when it is not explicitly present in the records
            # for this chunk (e.g., before the first environment switch).
            continue
        if any(record.has_feature(name) for record in records):
            active.append(name)
    return active


@dataclass(frozen=True)
class WorldBreakdownSummary:
    """Episode-level occupancy aggregates for a single world within a chunk."""

    world_index: int
    episode_fraction_sum: float
    episode_fraction_count: int
    world_mod_index: Optional[int] = None
    world_count: Optional[int] = None
    lag_breakdown: Dict[int, Tuple[float, int, Optional[int]]] = field(default_factory=dict)


@dataclass
class ChunkResult:
    chunk_index: int
    episode_start: int | None
    chunk_size: int | None
    n_samples: int
    n_positive: int
    n_negative: int
    intercept: float
    intercept_se: float
    intercept_z: float
    intercept_p: float
    coef_current: float
    coef_current_se: float
    coef_current_z: float
    coef_current_p: float
    coef_previous: float
    coef_previous_se: float
    coef_previous_z: float
    coef_previous_p: float
    coef_walls: float
    coef_walls_se: float
    coef_walls_z: float
    coef_walls_p: float
    accuracy: float
    log_loss_value: float
    dropped_nonfinite: int
    dropped_unpaired: int
    extra_coefficients: Dict[str, Dict[str, float]]

    def to_row(
        self,
        centred: bool,
        demeaned: bool,
        extra_feature_names: Sequence[str],
    ) -> List[str]:
        row = [
            str(self.chunk_index),
            str(self.episode_start) if self.episode_start is not None else "",
            str(self.chunk_size) if self.chunk_size is not None else "",
            str(self.n_samples),
            str(self.n_positive),
            str(self.n_negative),
            f"{self.intercept:.10g}",
            f"{self.intercept_se:.10g}",
            f"{self.intercept_z:.10g}",
            f"{self.intercept_p:.10g}",
            f"{self.coef_current:.10g}",
            f"{self.coef_current_se:.10g}",
            f"{self.coef_current_z:.10g}",
            f"{self.coef_current_p:.10g}",
            f"{self.coef_previous:.10g}",
            f"{self.coef_previous_se:.10g}",
            f"{self.coef_previous_z:.10g}",
            f"{self.coef_previous_p:.10g}",
            f"{self.coef_walls:.10g}",
            f"{self.coef_walls_se:.10g}",
            f"{self.coef_walls_z:.10g}",
            f"{self.coef_walls_p:.10g}",
            f"{self.accuracy:.10g}",
            f"{self.log_loss_value:.10g}",
            str(self.dropped_nonfinite),
            str(self.dropped_unpaired),
            "1" if centred else "0",
            "1" if demeaned else "0",
        ]

        for name in extra_feature_names:
            stats = self.extra_coefficients.get(name)
            if not stats:
                row.extend(["nan", "nan", "nan", "nan"])
                continue
            row.extend(
                [
                    f"{stats.get('coef', float('nan')):.10g}",
                    f"{stats.get('se', float('nan')):.10g}",
                    f"{stats.get('z', float('nan')):.10g}",
                    f"{stats.get('p', float('nan')):.10g}",
                ]
            )

        return row




def _load_feature(directory: Path, feature_name: str) -> Dict[Tuple[str, str, int], float]:
    csv_path = directory / "chunked_occupancy_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not locate {csv_path}")
    logging.info("Loading %s from %s", feature_name, csv_path)

    records: Dict[Tuple[str, str, int], float] = {}
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                chunk_index = int(row["chunk_index"])
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                logging.warning("Skipping row with invalid chunk_index: %s (%s)", row, exc)
                continue
            seed = row.get("seed", "").strip()
            agent = row.get("agent", "").strip()
            key = (seed, agent, chunk_index)
            value_str = row.get("occupancy_fraction", "nan")
            try:
                value = float(value_str)
            except (TypeError, ValueError):
                value = float("nan")
            records[key] = value
    return records


def _load_episode_level(directory: Path) -> Dict[Tuple[str, str, int], List[Tuple[int, float, Dict[int, float]]]]:
    """Load per-episode features.

    Returns mapping (seed, agent, chunk_index) -> list of (episode, value, {lag_k: value}).
    For current/walls targets the lag map is empty; for previous-hole it contains
    any available lag_k fields (k>=1, with k=1 representing f_previous).
    """
    json_path = directory / "chunked_occupancy_metrics.json"
    if not json_path.exists():
        return {}
    try:
        payload = json.loads(json_path.read_text())
    except Exception:
        return {}
    result: Dict[Tuple[str, str, int], List[Tuple[int, float, Dict[int, float]]]] = {}
    seeds = payload.get("seeds", {})
    if not isinstance(seeds, dict):
        return {}
    for seed_key, seed_block in seeds.items():
        if not isinstance(seed_block, dict):
            continue
        episode_map = seed_block.get("episode_records", {})
        if not isinstance(episode_map, dict):
            continue
        for agent_label, records in episode_map.items():
            if not isinstance(records, list):
                continue
            for rec in records:
                try:
                    chunk_index = int(rec.get("chunk_index", "nan"))
                except Exception:
                    continue
                current = rec.get("occupancy_fraction", float("nan"))
                if not isinstance(current, (int, float)):
                    continue
                episode = rec.get("episode")
                try:
                    episode_int = int(episode)
                except Exception:
                    continue
                lags: Dict[int, float] = {}
                for key, value in rec.items():
                    if isinstance(key, str) and key.startswith("lag_"):
                        try:
                            k = int(key.split("_", 1)[1])
                        except Exception:
                            continue
                        if isinstance(value, (int, float)) and math.isfinite(value):
                            lags[k] = float(value)
                result.setdefault((str(seed_key), str(agent_label), chunk_index), []).append((episode_int, float(current), lags))
    return result


def fit_episode_level_glm(
    current_dir: Path,
    previous_dir: Path,
    walls_dir: Path,
    output_dir: Path,
    positive_label: str,
    max_iter: int = 1000,
) -> None:
    """Fit per-episode GLMs inside each chunk using episode-level features.

    Note: This requires episode-level records to be present in the JSON. It
    fits a simple model with current, previous (if available), walls, and any
    lag_k features provided by the previous-hole analysis.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    current_epi = _load_episode_level(current_dir)
    prev_epi = _load_episode_level(previous_dir)
    walls_epi = _load_episode_level(walls_dir)

    # Group by chunk
    by_chunk: Dict[int, List[Tuple[int, str, str, int, float, float, Dict[int, float]]]] = {}
    # Require current features (always produced) but allow previous-hole and wall
    # occupancies to be absent so pre-switch episodes can still be modelled.
    keys = set(current_epi) | set(walls_epi)
    for seed, agent, chunk_index in keys:
        current_list = current_epi.get((seed, agent, chunk_index), [])
        if not current_list:
            continue
        prev_list = prev_epi.get((seed, agent, chunk_index), [])
        walls_list = walls_epi.get((seed, agent, chunk_index), [])
        # Build episode->value maps for alignment
        cur_map = {ep: val for ep, val, _ in current_list}
        prev_val_map = {ep: val for ep, val, _ in prev_list}
        prev_lag_map_map = {ep: lag_map for ep, _, lag_map in prev_list}
        wal_map = {ep: val for ep, val, _ in walls_list}
        # Previous-hole or wall occupancies may be missing for pre-switch episodes;
        # iterate over episodes with current occupancy data and attach whatever
        # additional metrics are available.
        for ep in sorted(cur_map):
            cur_val = float(cur_map.get(ep, float("nan")))
            if not math.isfinite(cur_val):
                continue
            prev_val = float(prev_val_map.get(ep, float("nan")))
            wal_val = float(wal_map.get(ep, float("nan")))
            lag_map = dict(prev_lag_map_map.get(ep, {}))
            if math.isfinite(prev_val):
                lag_map[1] = float(prev_val)
            by_chunk.setdefault(chunk_index, []).append(
                (chunk_index, str(seed), str(agent), int(ep), cur_val, wal_val, lag_map)
            )

    # Determine all extra lag feature names across chunks (k >= 2)
    extras_all: List[int] = []
    extras_set: set[int] = set()
    for rows in by_chunk.values():
        for _, _seed, _agent, _ep, _cur, _wal, lag_map in rows:
            for k in lag_map.keys():
                if isinstance(k, int) and k >= 2 and k not in extras_set:
                    extras_set.add(k)
    extras_all = sorted(extras_set)
    extra_cols = [f"f_holes_world_t_minus_{k}" for k in extras_all]

    fieldnames = [
        "chunk_index",
        "episode_start",
        "chunk_size",
        "n_episodes",
        "n_positive",
        "n_negative",
        "intercept",
        "intercept_se",
        "intercept_z",
        "intercept_p",
        "coef_current",
        "coef_current_se",
        "coef_current_z",
        "coef_current_p",
        "coef_previous",
        "coef_previous_se",
        "coef_previous_z",
        "coef_previous_p",
        "coef_walls",
        "coef_walls_se",
        "coef_walls_z",
        "coef_walls_p",
    ]
    for col in extra_cols:
        fieldnames.extend([
            f"coef_{col}",
            f"coef_{col}_se",
            f"coef_{col}_z",
            f"coef_{col}_p",
        ])
    summary_path = output_dir / "episode_glm_coefficients.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for chunk_index in sorted(by_chunk):
            rows = by_chunk[chunk_index]
            if not rows:
                continue
            # Build per-episode dataset across seeds and agents
            X_cols: List[str] = ["f_current", "f_previous", "f_walls", *extra_cols]
            X: List[List[float]] = []
            y: List[int] = []
            episodes: List[int] = []
            for _, seed, agent, ep, cur, wal, lag_map in rows:
                if not math.isfinite(cur):
                    continue
                f_prev = float(lag_map.get(1, float("nan")))
                wal_val = float(wal)
                row = [
                    float(cur),
                    f_prev if math.isfinite(f_prev) else float("nan"),
                    wal_val if math.isfinite(wal_val) else float("nan"),
                ]
                for k in extras_all:
                    val = lag_map.get(k, float("nan"))
                    row.append(
                        float(val)
                        if isinstance(val, (int, float)) and math.isfinite(val)
                        else float("nan")
                    )
                X.append(row)
                y.append(1 if agent == positive_label else 0)
                episodes.append(ep)

            if not X:
                continue

            design = pd.DataFrame(X, columns=X_cols)
            # Drop columns that are entirely NaN (e.g., f_previous or f_walls pre-switch)
            design = design.dropna(axis=1, how="all")
            # Drop rows with any NaN feature
            valid_mask = np.isfinite(design.values).all(axis=1)
            design = design.loc[valid_mask]
            y_arr = np.asarray([label for mask, label in zip(valid_mask, y) if mask], dtype=float)
            ep_arr = [ep for mask, ep in zip(valid_mask, episodes) if mask]

            base_episode_list = ep_arr if ep_arr else episodes
            episode_start_val: int | str = int(min(base_episode_list)) if base_episode_list else ""
            chunk_size_val: int | str = (
                int(len(set(ep_arr)))
                if ep_arr
                else (int(len(set(episodes))) if episodes else "")
            )
            if y_arr.size:
                n_positive = int(np.sum(y_arr == 1))
                n_negative = int(np.sum(y_arr == 0))
                n_episodes_val = int(len(y_arr))
            else:
                n_positive = int(sum(1 for label in y if label == 1))
                n_negative = int(sum(1 for label in y if label == 0))
                n_episodes_val = int(len(y))

            row_out = {
                "chunk_index": chunk_index,
                "episode_start": episode_start_val,
                "chunk_size": chunk_size_val,
                "n_episodes": n_episodes_val,
                "n_positive": n_positive,
                "n_negative": n_negative,
                "intercept": float("nan"),
                "intercept_se": float("nan"),
                "intercept_z": float("nan"),
                "intercept_p": float("nan"),
                "coef_current": float("nan"),
                "coef_current_se": float("nan"),
                "coef_current_z": float("nan"),
                "coef_current_p": float("nan"),
                "coef_previous": float("nan"),
                "coef_previous_se": float("nan"),
                "coef_previous_z": float("nan"),
                "coef_previous_p": float("nan"),
                "coef_walls": float("nan"),
                "coef_walls_se": float("nan"),
                "coef_walls_z": float("nan"),
                "coef_walls_p": float("nan"),
            }
            for k in extras_all:
                name = f"f_holes_world_t_minus_{k}"
                row_out[f"coef_{name}"] = float("nan")
                row_out[f"coef_{name}_se"] = float("nan")
                row_out[f"coef_{name}_z"] = float("nan")
                row_out[f"coef_{name}_p"] = float("nan")

            if design.empty or len(set(y_arr)) < 2:
                writer.writerow(row_out)
                continue

            design.insert(0, "const", 1.0)
            try:
                model = sm.Logit(y_arr, design)
                fitted = model.fit(disp=0, maxiter=int(max_iter))
            except Exception:
                writer.writerow(row_out)
                continue

            params = fitted.params
            bse = fitted.bse
            z = fitted.tvalues
            p = fitted.pvalues

            row_out.update(
                {
                    "intercept": float(params.get("const", float("nan"))),
                    "intercept_se": float(bse.get("const", float("nan"))),
                    "intercept_z": float(z.get("const", float("nan"))),
                    "intercept_p": float(p.get("const", float("nan"))),
                    "coef_current": float(params.get("f_current", float("nan"))),
                    "coef_current_se": float(bse.get("f_current", float("nan"))),
                    "coef_current_z": float(z.get("f_current", float("nan"))),
                    "coef_current_p": float(p.get("f_current", float("nan"))),
                    "coef_previous": float(params.get("f_previous", float("nan"))),
                    "coef_previous_se": float(bse.get("f_previous", float("nan"))),
                    "coef_previous_z": float(z.get("f_previous", float("nan"))),
                    "coef_previous_p": float(p.get("f_previous", float("nan"))),
                    "coef_walls": float(params.get("f_walls", float("nan"))),
                    "coef_walls_se": float(bse.get("f_walls", float("nan"))),
                    "coef_walls_z": float(z.get("f_walls", float("nan"))),
                    "coef_walls_p": float(p.get("f_walls", float("nan"))),
                }
            )
            for k in extras_all:
                name = f"f_holes_world_t_minus_{k}"
                row_out[f"coef_{name}"] = float(params.get(name, float("nan")))
                row_out[f"coef_{name}_se"] = float(bse.get(name, float("nan")))
                row_out[f"coef_{name}_z"] = float(z.get(name, float("nan")))
                row_out[f"coef_{name}_p"] = float(p.get(name, float("nan")))

            writer.writerow(row_out)

    # Plot episode-level GLM results
    df = pd.read_csv(summary_path)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def _scatter_significant(ax, name: str, pcol: str, label: str, color: str) -> bool:
        mask = df[pcol].apply(lambda v: isinstance(v, (int, float)) and math.isfinite(v) and v < _SIGNIFICANCE_LEVEL)
        sig = df[mask]
        if sig.empty:
            return False
        x = sig["episode_start"].fillna(sig["chunk_index"]) if "episode_start" in sig else sig["chunk_index"]
        ax.scatter(x, sig[name], label=label, color=color)
        return True

    # Significant-only plot
    fig, ax = plt.subplots(figsize=(10, 5))
    has_any = False
    color_map = {"current": "C0", "previous": "C1", "walls": "C2"}
    has_any |= _scatter_significant(ax, "coef_current", "coef_current_p", _beta_label("coef_current"), color_map["current"]) 
    has_any |= _scatter_significant(ax, "coef_previous", "coef_previous_p", _beta_label("coef_previous"), color_map["previous"]) 
    has_any |= _scatter_significant(ax, "coef_walls", "coef_walls_p", _beta_label("coef_walls"), color_map["walls"]) 
    # Extra lag features
    extra_coef_cols = [c for c in df.columns if c.startswith("coef_f_holes_world_t_minus_") and not c.endswith(("_se","_z","_p"))]
    extra_cycle = iter([f"C{idx}" for idx in range(3, 10)])
    for col in sorted(extra_coef_cols):
        pcol = f"{col}_p"
        label = _beta_label(col.replace("coef_", ""))
        color = next(extra_cycle, "black")
        has_any |= _scatter_significant(ax, col, pcol, label, color)
    if has_any:
        ax.set_xlabel("Episode (chunk start)")
        ax.set_ylabel(f"Coefficient value (log-odds for {positive_label})")
        ax.set_title("Episode-level GLM coefficients (significant only)\n" f"p < {_SIGNIFICANCE_LEVEL:.3g}")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "episode_glm_coefficients.png", dpi=200)
    plt.close(fig)

    # All-values with significance bars
    fig, ax = plt.subplots(figsize=(10, 6))
    x_all = df["episode_start"].fillna(df["chunk_index"]) if "episode_start" in df else df["chunk_index"]
    ax.scatter(x_all, df["coef_current"], color=color_map["current"], label=_beta_label("coef_current"), alpha=0.85)
    ax.scatter(x_all, df["coef_previous"], color=color_map["previous"], label=_beta_label("coef_previous"), alpha=0.85)
    ax.scatter(x_all, df["coef_walls"], color=color_map["walls"], label=_beta_label("coef_walls"), alpha=0.85)
    for col in sorted(extra_coef_cols):
        ax.scatter(x_all, df[col], label=_beta_label(col.replace("coef_", "")), alpha=0.85)
    for name, pcol, color in [("coef_current", "coef_current_p", color_map["current"]), ("coef_previous", "coef_previous_p", color_map["previous"]), ("coef_walls", "coef_walls_p", color_map["walls"])]:
        sig_rows = df[df[pcol].apply(lambda v: isinstance(v, (int, float)) and math.isfinite(v) and v < _SIGNIFICANCE_LEVEL)]
        sig_x = (sig_rows["episode_start"].fillna(sig_rows["chunk_index"]) if "episode_start" in sig_rows else sig_rows["chunk_index"]).unique()
        for x in sig_x:
            ax.hlines(ax.get_ylim()[1]*0.95, x-0.4, x+0.4, colors=color, linewidth=4)
    # Significance bars for extras
    for col in sorted(extra_coef_cols):
        pcol = f"{col}_p"
        color = next(extra_cycle, "black")
        sig_rows = df[df[pcol].apply(lambda v: isinstance(v, (int, float)) and math.isfinite(v) and v < _SIGNIFICANCE_LEVEL)]
        sig_x = (sig_rows["episode_start"].fillna(sig_rows["chunk_index"]) if "episode_start" in sig_rows else sig_rows["chunk_index"]).unique()
        for x in sig_x:
            ax.hlines(ax.get_ylim()[1]*0.95, x-0.4, x+0.4, colors=color, linewidth=4)
    ax.set_xlabel("Episode (chunk start)")
    ax.set_ylabel(f"Coefficient value (log-odds for {positive_label})")
    ax.set_title("Episode-level GLM coefficients (all values)\nSolid bars: p < " f"{_SIGNIFICANCE_LEVEL:.3g}")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "episode_glm_coefficients_all_values.png", dpi=200)
    plt.close(fig)

    # p-value plot
    fig, ax = plt.subplots(figsize=(10, 5))
    def _plot_p(ax, name: str, pcol: str, label: str, color: str) -> None:
        cols = ["episode_start", "chunk_index", pcol] if "episode_start" in df else ["chunk_index", pcol]
        vals = df[cols].dropna()
        vals = vals[vals[pcol] > 0]
        if vals.empty:
            return
        xs = vals["episode_start"] if "episode_start" in vals else vals["chunk_index"]
        ax.plot(xs, -np.log10(vals[pcol]), marker="o", label=label, color=color)
    _plot_p(ax, "coef_current", "coef_current_p", _pvalue_label("coef_current"), color_map["current"]) 
    _plot_p(ax, "coef_previous", "coef_previous_p", _pvalue_label("coef_previous"), color_map["previous"]) 
    _plot_p(ax, "coef_walls", "coef_walls_p", _pvalue_label("coef_walls"), color_map["walls"]) 
    # Extras p-values
    extra_cycle = iter([f"C{idx}" for idx in range(3, 10)])
    for col in sorted(extra_coef_cols):
        pcol = f"{col}_p"
        label = _pvalue_label(col.replace("coef_", ""))
        color = next(extra_cycle, "black")
        _plot_p(ax, col, pcol, label, color)
    ax.axhline(-math.log10(_SIGNIFICANCE_LEVEL), color="red", linestyle="--", label=f"p = {_SIGNIFICANCE_LEVEL:.3g}")
    ax.set_xlabel("Chunk index")
    ax.set_ylabel("$-\\log_{10} p$")
    ax.set_title("Episode-level coefficient significance")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "episode_glm_pvalues.png", dpi=200)
    plt.close(fig)


def _load_episode_starts(
    directory: Path,
) -> Dict[Tuple[str, str, int], Tuple[int | None, int | None]]:
    csv_path = directory / "chunked_occupancy_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not locate {csv_path}")
    logging.info("Loading episode start indices from %s", csv_path)

    starts: Dict[Tuple[str, str, int], Tuple[int | None, int | None]] = {}
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                chunk_index = int(row.get("chunk_index", ""))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
            seed = row.get("seed", "").strip()
            agent = row.get("agent", "").strip()
            key = (seed, agent, chunk_index)
            episode_start: int | None
            chunk_size: int | None
            try:
                episode_start = int(row.get("episode_start", ""))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                episode_start = None
            try:
                episode_end = int(row.get("episode_end", ""))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                episode_end = None
            if (
                episode_start is not None
                and episode_end is not None
                and episode_end >= episode_start
            ):
                chunk_size = int(episode_end - episode_start + 1)
            else:
                chunk_size = None
            starts[key] = (episode_start, chunk_size)
    return starts


def _load_world_breakdown(
    directory: Path,
) -> Tuple[
    Dict[Tuple[str, str, int], List[WorldBreakdownSummary]],
    Dict[int, Tuple[float, int]],
]:
    json_path = directory / "chunked_occupancy_metrics.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Could not locate {json_path}; world demeaning requires the JSON summary"
        )

    with json_path.open("r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse {json_path}: {exc}") from exc

    record_breakdown: Dict[Tuple[str, str, int], List[WorldBreakdownSummary]] = {}
    world_totals: Dict[int, Tuple[float, int]] = {}

    seeds = payload.get("seeds", {})
    if not isinstance(seeds, dict):
        logging.warning("World summary missing 'seeds' section in %s", json_path)
        return record_breakdown, world_totals

    for seed_key, seed_block in seeds.items():
        if not isinstance(seed_block, dict):
            continue
        chunk_metrics = seed_block.get("chunk_metrics", {})
        if not isinstance(chunk_metrics, dict):
            continue
        for agent_label, chunk_map in chunk_metrics.items():
            if not isinstance(chunk_map, dict):
                continue
            for chunk_key, chunk_record in chunk_map.items():
                chunk_index: int | None = None
                if isinstance(chunk_key, (int, str)):
                    try:
                        chunk_index = int(chunk_key)
                    except (TypeError, ValueError):
                        chunk_index = None
                if chunk_index is None and isinstance(chunk_record, dict):
                    try:
                        chunk_index = int(chunk_record.get("chunk_index"))
                    except (TypeError, ValueError):
                        chunk_index = None
                if chunk_index is None:
                    continue

                world_entries = []
                if isinstance(chunk_record, dict):
                    entries = chunk_record.get("world_breakdown", [])
                    if isinstance(entries, list):
                        world_entries = entries

                breakdown_list: List[WorldBreakdownSummary] = []
                for entry in world_entries:
                    if not isinstance(entry, dict):
                        continue
                    world_index_raw = entry.get("world_index")
                    sum_raw = entry.get("episode_fraction_sum")
                    count_raw = entry.get("episode_fraction_count")
                    world_mod_raw = entry.get("world_mod_index")
                    world_count_raw = entry.get("world_count")
                    try:
                        world_index = int(world_index_raw)
                        count = int(count_raw)
                        sum_value = float(sum_raw)
                    except (TypeError, ValueError):
                        continue
                    if count < 0 or not math.isfinite(sum_value):
                        continue
                    world_mod_index: Optional[int]
                    if isinstance(world_mod_raw, (int, float)):
                        world_mod_index = int(world_mod_raw)
                    else:
                        world_mod_index = None
                    world_count_value: Optional[int]
                    if isinstance(world_count_raw, (int, float)):
                        world_count_value = int(world_count_raw)
                    else:
                        world_count_value = None
                    lag_breakdown: Dict[int, Tuple[float, int, Optional[int]]] = {}
                    lag_entries = entry.get("lag_breakdown", [])
                    if isinstance(lag_entries, list):
                        for lag_entry in lag_entries:
                            if not isinstance(lag_entry, dict):
                                continue
                            previous_world_raw = lag_entry.get("previous_world_index")
                            lag_sum_raw = lag_entry.get("episode_fraction_sum")
                            lag_count_raw = lag_entry.get("episode_fraction_count")
                            lag_offset_raw = lag_entry.get("lag_offset")
                            try:
                                previous_world = int(previous_world_raw)
                                lag_sum = float(lag_sum_raw)
                                lag_count = int(lag_count_raw)
                            except (TypeError, ValueError):
                                continue
                            if lag_count < 0 or not math.isfinite(lag_sum):
                                continue
                            if isinstance(lag_offset_raw, (int, float)):
                                lag_offset: Optional[int] = int(lag_offset_raw)
                            else:
                                lag_offset = None
                            lag_breakdown[previous_world] = (lag_sum, lag_count, lag_offset)

                    breakdown_list.append(
                        WorldBreakdownSummary(
                            world_index=world_index,
                            episode_fraction_sum=sum_value,
                            episode_fraction_count=count,
                            world_mod_index=world_mod_index,
                            world_count=world_count_value,
                            lag_breakdown=lag_breakdown,
                        )
                    )
                    if count > 0:
                        total_sum, total_count = world_totals.get(world_index, (0.0, 0))
                        world_totals[world_index] = (
                            total_sum + sum_value,
                            total_count + count,
                        )

                key = (str(seed_key).strip(), str(agent_label).strip(), chunk_index)
                record_breakdown[key] = breakdown_list

    return record_breakdown, world_totals


def _compute_world_means(
    world_totals: Mapping[int, Tuple[float, int]]
) -> Dict[int, float]:
    means: Dict[int, float] = {}
    for world_index, (total_sum, total_count) in world_totals.items():
        if total_count > 0 and math.isfinite(total_sum):
            means[world_index] = float(total_sum) / float(total_count)
    return means


def _demean_feature(
    value: float,
    breakdown: Sequence[WorldBreakdownSummary] | None,
    world_means: Mapping[int, float],
) -> float:
    if breakdown is None or not breakdown:
        return value
    if not math.isfinite(value):
        return value

    total_count = sum(
        entry.episode_fraction_count for entry in breakdown if entry.episode_fraction_count > 0
    )
    if total_count <= 0:
        return value

    expected_sum = 0.0
    for entry in breakdown:
        count = entry.episode_fraction_count
        if count <= 0:
            continue
        mean = world_means.get(entry.world_index)
        if mean is None or not math.isfinite(mean):
            continue
        expected_sum += float(mean) * float(count)

    demeaned = (float(value) * float(total_count) - expected_sum) / float(total_count)
    return float(demeaned)


def _is_specific_lag_dir_name(name: str) -> bool:
    return name.startswith("t_minus_") and name.endswith("_hole_locations")


def _resolve_lag_features_dir(candidate: Path | None) -> Path | None:
    """Return a directory to load lag breakdowns from.

    Preference order:
    1) If candidate is provided and contains the JSON, use it.
    2) If candidate looks like a specific lag folder (t_minus_k_hole_locations),
       try the sibling 'previous_hole_locations' if it contains the JSON.
    Otherwise return None.
    """
    if candidate is None:
        return None

    json_path = candidate / "chunked_occupancy_metrics.json"
    if json_path.exists():
        return candidate

    name = candidate.name
    if _is_specific_lag_dir_name(name):
        sibling = candidate.parent / "previous_hole_locations"
        sibling_json = sibling / "chunked_occupancy_metrics.json"
        if sibling_json.exists():
            return sibling

    return None


def _populate_previous_lag_features(
    records: Sequence[Record],
    breakdown_map: Mapping[Tuple[str, str, int], Sequence[WorldBreakdownSummary]],
) -> None:
    def _resolve_offset(
        explicit_offset: Optional[int],
        world_count: Optional[int],
        world_mod_index: Optional[int],
        previous_world: int,
    ) -> Optional[int]:
        if explicit_offset is not None and explicit_offset > 0:
            return int(explicit_offset)
        if world_count is None or world_count <= 0:
            return None
        if world_mod_index is None:
            return None
        offset = (int(world_mod_index) - int(previous_world)) % int(world_count)
        if offset <= 0:
            return None
        return int(offset)

    for record in records:
        key = (record.seed, record.agent, record.chunk_index)
        breakdown = breakdown_map.get(key)
        if not breakdown:
            continue
        lag_totals: Dict[int, Tuple[float, int]] = {}
        for entry in breakdown:
            contributions: List[Tuple[Optional[int], float, int]] = []
            if entry.lag_breakdown:
                for previous_world, stats in entry.lag_breakdown.items():
                    if len(stats) == 3:
                        lag_sum, lag_count, lag_offset = stats
                    else:
                        lag_sum, lag_count = stats[:2]
                        lag_offset = None
                    if lag_count <= 0:
                        continue
                    offset = _resolve_offset(
                        lag_offset,
                        entry.world_count,
                        entry.world_mod_index,
                        previous_world,
                    )
                    contributions.append((offset, float(lag_sum), int(lag_count)))
            else:
                if entry.world_index > 0 and entry.episode_fraction_count > 0:
                    contributions.append(
                        (
                            1,
                            float(entry.episode_fraction_sum),
                            int(entry.episode_fraction_count),
                        )
                    )

            for offset, total_sum, total_count in contributions:
                if offset is None or offset <= 0:
                    continue
                if total_count <= 0:
                    continue
                existing_sum, existing_count = lag_totals.get(offset, (0.0, 0))
                lag_totals[offset] = (
                    existing_sum + float(total_sum),
                    existing_count + int(total_count),
                )

        has_lag1 = False
        for lag_offset, (total_sum, total_count) in lag_totals.items():
            if total_count <= 0:
                continue
            mean = float(total_sum) / float(total_count)
            if not math.isfinite(mean):
                continue
            if lag_offset == 1:
                record.set_feature("f_previous", mean)
                has_lag1 = True
                continue
            feature_name = f"f_holes_world_t_minus_{int(lag_offset)}"
            record.set_feature(feature_name, mean)

        # If no lag-1 information was available, remove f_previous so the
        # pre-switch chunk can be fitted without that regressor.
        if not has_lag1 and "f_previous" in record.features:
            try:
                del record.features["f_previous"]
            except Exception:
                pass


def _join_records(
    current_map: Dict[Tuple[str, str, int], float],
    previous_map: Dict[Tuple[str, str, int], float],
    walls_map: Dict[Tuple[str, str, int], float],
    episode_start_map: Dict[Tuple[str, str, int], Tuple[int | None, int | None]],
) -> List[Record]:
    shared_keys = set(current_map) & set(walls_map)
    missing_current = set(walls_map) - set(current_map)
    missing_previous = shared_keys - set(previous_map)
    missing_walls = set(current_map) - set(walls_map)

    if missing_current:
        logging.warning("%d keys missing from current-hole metrics", len(missing_current))
    if missing_previous:
        logging.info(
            "%d keys missing from previous-hole metrics; dropping f_previous for those chunks",
            len(missing_previous),
        )
    if missing_walls:
        logging.warning("%d keys missing from wall metrics", len(missing_walls))

    records: List[Record] = []
    for seed, agent, chunk_index in sorted(shared_keys):
        episode_info = episode_start_map.get((seed, agent, chunk_index), (None, None))
        episode_start, chunk_size = episode_info
        features = {
            "f_current": float(current_map[(seed, agent, chunk_index)]),
            "f_walls": float(walls_map[(seed, agent, chunk_index)]),
        }
        if (seed, agent, chunk_index) in previous_map:
            features["f_previous"] = float(previous_map[(seed, agent, chunk_index)])
        records.append(
            Record(
                seed=seed,
                agent=agent,
                chunk_index=int(chunk_index),
                episode_start=episode_start,
                chunk_size=chunk_size,
                features=features,
            )
        )
    logging.info("Joined %d records across all targets", len(records))
    return records


def _filter_finite(
    records: List[Record], feature_names: Sequence[str]
) -> Tuple[List[Record], int]:
    filtered: List[Record] = []
    dropped = 0
    for record in records:
        values = [record.get_feature(name) for name in feature_names]
        if all(math.isfinite(value) for value in values):
            filtered.append(record)
        else:
            dropped += 1
    return filtered, dropped


def _centre_within_seed(
    records: List[Record], feature_names: Sequence[str]
) -> Tuple[List[Record], int]:
    grouped: Dict[Tuple[str, int], List[int]] = {}
    for idx, record in enumerate(records):
        grouped.setdefault((record.seed, record.chunk_index), []).append(idx)

    dropped_indices: set[int] = set()
    for key, indices in grouped.items():
        if len(indices) < 2:
            dropped_indices.update(indices)
            continue
        mean_vector = np.zeros(len(feature_names), dtype=float)
        for idx in indices:
            vector = np.array(
                [records[idx].get_feature(name) for name in feature_names], dtype=float
            )
            mean_vector += vector
        mean_vector /= float(len(indices))
        for idx in indices:
            centred_values = (
                np.array(
                    [records[idx].get_feature(name) for name in feature_names], dtype=float
                )
                - mean_vector
            )
            for name, value in zip(feature_names, centred_values):
                records[idx].set_feature(name, float(value))
    kept: List[Record] = []
    for idx, record in enumerate(records):
        if idx not in dropped_indices:
            kept.append(record)
    return kept, len(dropped_indices)


def _write_joined_dataset(path: Path, records: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        records_list = list(records)
        feature_names, _ = _feature_name_union(records_list)
        writer.writerow(
            ["seed", "agent", "chunk_index", "episode_start", "chunk_size", *feature_names]
        )
        for record in records_list:
            writer.writerow(record.to_row(feature_names))
    logging.info("Wrote joined dataset to %s", path)


def _write_summary(
    path: Path,
    results: Sequence[ChunkResult],
    centred: bool,
    demeaned: bool,
    extra_feature_names: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = [
            "chunk_index",
            "episode_start",
            "chunk_size",
            "n_samples",
            "n_positive",
            "n_negative",
            "intercept",
            "intercept_se",
            "intercept_z",
            "intercept_p",
            "coef_current",
            "coef_current_se",
            "coef_current_z",
            "coef_current_p",
            "coef_previous",
            "coef_previous_se",
            "coef_previous_z",
            "coef_previous_p",
            "coef_walls",
            "coef_walls_se",
            "coef_walls_z",
            "coef_walls_p",
            "accuracy",
            "log_loss",
            "dropped_nonfinite",
            "dropped_unpaired",
            "centred_within_seed",
            "demeaned_by_world",
        ]
        for name in extra_feature_names:
            header.extend(
                [
                    f"coef_{name}",
                    f"coef_{name}_se",
                    f"coef_{name}_z",
                    f"coef_{name}_p",
                ]
            )
        writer.writerow(header)
        for result in results:
            writer.writerow(result.to_row(centred, demeaned, extra_feature_names))
    logging.info("Saved GLM summary to %s", path)


def _result_x_value(result: ChunkResult) -> float:
    if result.episode_start is not None:
        return float(result.episode_start)
    if result.chunk_size is not None:
        return float(result.chunk_index * result.chunk_size)
    return float(result.chunk_index)


def _holes_feature_display(name: str) -> Optional[str]:
    if name in {"coef_current", "coef_current_p", "f_current"}:
        return "holes\\_world\\_t"
    if name in {"coef_previous", "coef_previous_p", "f_previous"}:
        return "holes\\_world\\_t-1"
    if name in {"coef_walls", "coef_walls_p", "f_walls"}:
        return "walls"
    if name.startswith("f_holes_world_t_minus_"):
        suffix = name.rsplit("_", 1)[-1]
        return f"holes\\_world\\_t-{suffix}"
    return None


def _beta_label(name: str) -> str:
    display = _holes_feature_display(name)
    if display is None:
        display = name.replace("_", "\\_")
    return f"$\\beta_{{{display}}}$"


def _pvalue_label(name: str) -> str:
    display = _holes_feature_display(name)
    if display is None:
        display = name.replace("_", "\\_")
    return f"$-\\log_{{10}} p(\\beta_{{{display}}})$"


def _build_feature_colors(extra_feature_names: Sequence[str]) -> Dict[str, str]:
    color_map = {
        "coef_current": "C0",
        "coef_previous": "C1",
        "coef_walls": "C2",
    }
    extra_cycle = itertools.cycle([f"C{idx}" for idx in range(3, 10)])
    for name in extra_feature_names:
        color_map.setdefault(name, next(extra_cycle))
    return color_map


def _episode_axis_limits(
    results: Sequence[ChunkResult],
) -> tuple[bool, tuple[float, float]]:
    """Determine axis bounds and whether to label by episode number."""

    use_episode_axis = any(
        result.episode_start is not None or result.chunk_size is not None
        for result in results
    )

    if not use_episode_axis:
        max_index = max((float(result.chunk_index) for result in results), default=0.0)
        return False, (0.0, max_index)

    max_episode = 0.0
    for result in results:
        if result.episode_start is not None:
            start = float(result.episode_start)
        elif result.chunk_size is not None:
            start = float(result.chunk_index * result.chunk_size)
        else:
            continue

        if result.chunk_size is not None and result.chunk_size > 0:
            end = start + float(result.chunk_size - 1)
        else:
            end = start
        max_episode = max(max_episode, end)

    return True, (0.0, max_episode)


def _plot_coefficients(
    path: Path,
    results: Sequence[ChunkResult],
    centred: bool,
    demeaned: bool,
    positive_label: str,
    significance_level: float,
    extra_feature_names: Sequence[str] | None = (),
) -> None:
    if not results:
        logging.warning("No chunk results to plot; skipping %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    extra_feature_names = tuple(extra_feature_names or ())
    plt.figure(figsize=(10, 5))
    color_map = _build_feature_colors(extra_feature_names)
    base_specs = [
        ("coef_current", "coef_current_p"),
        ("coef_previous", "coef_previous_p"),
        ("coef_walls", "coef_walls_p"),
    ]
    use_episode_axis, (x_min, x_max) = _episode_axis_limits(results)

    has_points = False
    for value_attr, p_attr in base_specs:
        xs = [
            _result_x_value(result)
            for result in results
            if getattr(result, p_attr) < significance_level
        ]
        ys = [
            getattr(result, value_attr)
            for result in results
            if getattr(result, p_attr) < significance_level
        ]
        if xs:
            has_points = True
            plt.scatter(
                xs,
                ys,
                label=_beta_label(value_attr),
                color=color_map.get(value_attr),
            )

    for name in extra_feature_names:
        xs: List[float] = []
        ys: List[float] = []
        for result in results:
            stats = result.extra_coefficients.get(name)
            if not stats:
                continue
            p_value = stats.get("p")
            if p_value is None or not math.isfinite(p_value):
                continue
            if p_value >= significance_level:
                continue
            coef_value = stats.get("coef")
            if coef_value is None or not math.isfinite(coef_value):
                continue
            xs.append(_result_x_value(result))
            ys.append(float(coef_value))
        if xs:
            has_points = True
            plt.scatter(
                xs,
                ys,
                label=_beta_label(name),
                color=color_map.get(name),
            )

    if not has_points:
        logging.warning(
            "No coefficients passed the significance threshold (p < %.3g); skipping plot %s",
            significance_level,
            path,
        )
        plt.close()
        return

    if use_episode_axis:
        plt.xlabel("Episode (chunk start)")
    else:
        plt.xlabel("Chunk index")

    plt.xlim(x_min, x_max)
    plt.ylabel(f"Coefficient value (log-odds for {positive_label})")
    title = "Chunk-wise logistic regression coefficients (significant only)"
    modifiers: List[str] = []
    if centred:
        modifiers.append("seed-centred")
    if demeaned:
        modifiers.append("world-demeaned")
    if modifiers:
        title += " (" + ", ".join(modifiers) + ")"
    title += f"\np < {significance_level:.3g}"
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    logging.info("Saved coefficient plot to %s", path)


def _plot_coefficients_with_significance_bars(
    path: Path,
    results: Sequence[ChunkResult],
    centred: bool,
    demeaned: bool,
    positive_label: str,
    significance_level: float,
    extra_feature_names: Sequence[str] | None = (),
) -> None:
    if not results:
        logging.warning("No chunk results to plot; skipping %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)


    extra_feature_names = tuple(extra_feature_names or ())
    fig, ax = plt.subplots(figsize=(10, 6))
    color_map = _build_feature_colors(extra_feature_names)
    use_episode_axis, (x_min, x_max) = _episode_axis_limits(results)
    base_specs = [
        ("coef_current", "coef_current_p"),
        ("coef_previous", "coef_previous_p"),
        ("coef_walls", "coef_walls_p"),
    ]


    entries: List[Dict[str, object]] = []
    all_xs: List[float] = []
    all_ys: List[float] = []

    def _append_entry(
        name: str,
        label: str,
        color: Optional[str],
        xs: List[float],
        ys: List[float],
        significant_xs: List[float],
    ) -> None:
        if not xs:
            return
        entries.append(
            {
                "name": name,
                "label": label,
                "color": color,
                "xs": xs,
                "ys": ys,
                "significant_xs": significant_xs,
            }
        )
        all_xs.extend(xs)
        all_ys.extend(ys)

    for value_attr, p_attr in base_specs:
        xs: List[float] = []
        ys: List[float] = []
        significant_xs: List[float] = []
        for result in results:
            value = getattr(result, value_attr, None)
            if value is None or not math.isfinite(value):
                continue
            x_value = _result_x_value(result)
            xs.append(x_value)
            ys.append(float(value))
            p_value = getattr(result, p_attr, None)
            if p_value is not None and math.isfinite(p_value) and p_value < significance_level:
                significant_xs.append(x_value)
        _append_entry(value_attr, _beta_label(value_attr), color_map.get(value_attr), xs, ys, significant_xs)

    for name in extra_feature_names:
        xs = []
        ys = []
        significant_xs = []
        for result in results:
            stats = result.extra_coefficients.get(name)
            if not stats:
                continue
            coef_value = stats.get("coef")
            if coef_value is None or not math.isfinite(coef_value):
                continue
            x_value = _result_x_value(result)
            xs.append(x_value)
            ys.append(float(coef_value))
            p_value = stats.get("p")
            if p_value is not None and math.isfinite(p_value) and p_value < significance_level:
                significant_xs.append(x_value)
        _append_entry(name, _beta_label(name), color_map.get(name), xs, ys, significant_xs)

    if not entries:
        logging.warning("No coefficients available to plot at %s", path)
        plt.close(fig)
        return

    for entry in entries:
        ax.scatter(
            entry["xs"],
            entry["ys"],
            color=entry["color"],
            label=entry["label"],
            alpha=0.85,
        )

    if not all_ys:
        logging.warning("No coefficient values were finite for plot %s", path)
        plt.close(fig)
        return

    y_min = min(all_ys)
    y_max = max(all_ys)
    y_range = y_max - y_min
    padding = 0.05 * y_range if y_range > 0 else 0.5
    base_top = y_max + padding

    sorted_xs = sorted(set(all_xs))
    positive_diffs = [b - a for a, b in zip(sorted_xs, sorted_xs[1:]) if b - a > 0]
    bar_width = min(positive_diffs) * 0.8 if positive_diffs else 0.8
    if bar_width <= 0:
        bar_width = 0.8

    bar_gap = max(padding, 0.5)
    total_top = base_top + bar_gap * len(entries) + padding
    ax.set_ylim(y_min - padding, total_top)

    for idx, entry in enumerate(entries):
        bar_y = base_top + bar_gap * (idx + 0.2)
        for x_value in sorted(set(entry["significant_xs"])):
            ax.hlines(
                bar_y,
                x_value - bar_width / 2,
                x_value + bar_width / 2,
                colors=entry["color"],
                linewidth=4,
            )

    if use_episode_axis:
        ax.set_xlabel("Episode (chunk start)")
    else:
        ax.set_xlabel("Chunk index")

    ax.set_xlim(x_min, x_max)
    ax.set_ylabel(f"Coefficient value (log-odds for {positive_label})")
    title = "Chunk-wise logistic regression coefficients (all values)"
    modifiers: List[str] = []
    if centred:
        modifiers.append("seed-centred")
    if demeaned:
        modifiers.append("world-demeaned")
    if modifiers:
        title += " (" + ", ".join(modifiers) + ")"
    title += f"\nSolid bars: p < {significance_level:.3g}"
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D  # Imported lazily to avoid global dependency.

    significance_handle = Line2D([0, 1], [0, 0], color="black", linewidth=4)
    handles.append(significance_handle)
    labels.append(f"Significant (p < {significance_level:.3g})")
    ax.legend(handles, labels)

    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logging.info("Saved coefficient plot with significance bars to %s", path)


def _plot_pvalues(
    path: Path,
    results: Sequence[ChunkResult],
    centred: bool,
    demeaned: bool,
    significance_level: float,
    extra_feature_names: Sequence[str] | None = (),
) -> None:
    if not results:
        logging.warning("No chunk results to plot; skipping %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    extra_feature_names = tuple(extra_feature_names or ())
    plt.figure(figsize=(10, 5))
    color_map = _build_feature_colors(extra_feature_names)
    base_specs = [
        ("coef_current", "coef_current_p"),
        ("coef_previous", "coef_previous_p"),
        ("coef_walls", "coef_walls_p"),
    ]
    use_episode_axis, (x_min, x_max) = _episode_axis_limits(results)

    plotted = False
    for base_name, p_attr in base_specs:
        xs = [
            _result_x_value(result)
            for result in results
            if getattr(result, p_attr) > 0
        ]
        if not xs:
            continue
        pvals = [
            getattr(result, p_attr)
            for result in results
            if getattr(result, p_attr) > 0
        ]
        transformed = [-math.log10(value) for value in pvals]
        plt.plot(
            xs,
            transformed,
            marker="o",
            label=_pvalue_label(base_name),
            color=color_map.get(base_name),
        )
        plotted = True

    for name in extra_feature_names:
        xs: List[float] = []
        pvals: List[float] = []
        for result in results:
            stats = result.extra_coefficients.get(name)
            if not stats:
                continue
            p_value = stats.get("p")
            if p_value is None or not math.isfinite(p_value) or p_value <= 0:
                continue
            xs.append(_result_x_value(result))
            pvals.append(float(p_value))
        if not xs:
            continue
        transformed = [-math.log10(value) for value in pvals]
        plt.plot(
            xs,
            transformed,
            marker="o",
            label=_pvalue_label(name),
            color=color_map.get(name),
        )
        plotted = True

    if not plotted:
        logging.warning("No valid p-values to plot at %s", path)
        plt.close()
        return

    threshold = -math.log10(significance_level)
    plt.axhline(threshold, color="red", linestyle="--", label=f"p = {significance_level:.3g}")
    if use_episode_axis:
        plt.xlabel("Episode (chunk start)")
    else:
        plt.xlabel("Chunk index")
    plt.xlim(x_min, x_max)
    plt.ylabel("$-\\log_{10} p$")
    title = "Chunk-wise coefficient significance"
    modifiers: List[str] = []
    if centred:
        modifiers.append("seed-centred")
    if demeaned:
        modifiers.append("world-demeaned")
    if modifiers:
        title += " (" + ", ".join(modifiers) + ")"
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    logging.info("Saved p-value plot to %s", path)


def _fit_single_chunk(
    chunk_index: int,
    chunk_records: Sequence[Record],
    positive_label: str,
    max_iter: int,
    centre_within_seed: bool,
    feature_names: Sequence[str],
    extra_feature_names: Sequence[str],
) -> ChunkResult | None:
    chunk_records = [_copy_record(rec) for rec in chunk_records]

    # Precompute metadata for placeholder rows
    meta_episode_starts = [
        record.episode_start for record in chunk_records if record.episode_start is not None
    ]
    meta_episode_start = int(min(meta_episode_starts)) if meta_episode_starts else None
    meta_chunk_sizes = [
        record.chunk_size for record in chunk_records if record.chunk_size is not None
    ]
    meta_chunk_size = int(min(meta_chunk_sizes)) if meta_chunk_sizes else None

    def _placeholder_result(
        *,
        n_samples: int,
        n_positive: int,
        n_negative: int,
        dropped_nonfinite: int,
        dropped_unpaired: int,
    ) -> ChunkResult:
        nan = float("nan")
        return ChunkResult(
            chunk_index=chunk_index,
            episode_start=meta_episode_start,
            chunk_size=meta_chunk_size,
            n_samples=int(n_samples),
            n_positive=int(n_positive),
            n_negative=int(n_negative),
            intercept=nan,
            intercept_se=nan,
            intercept_z=nan,
            intercept_p=nan,
            coef_current=nan,
            coef_current_se=nan,
            coef_current_z=nan,
            coef_current_p=nan,
            coef_previous=nan,
            coef_previous_se=nan,
            coef_previous_z=nan,
            coef_previous_p=nan,
            coef_walls=nan,
            coef_walls_se=nan,
            coef_walls_z=nan,
            coef_walls_p=nan,
            accuracy=nan,
            log_loss_value=nan,
            dropped_nonfinite=int(dropped_nonfinite),
            dropped_unpaired=int(dropped_unpaired),
            extra_coefficients={name: {"coef": nan, "se": nan, "z": nan, "p": nan} for name in extra_feature_names},
        )

    # Restrict the design matrix to features that exist within this chunk
    chunk_feature_names = _chunk_feature_subset(chunk_records, feature_names)
    if not chunk_feature_names:
        chunk_feature_names = [
            name for name in _BASE_FEATURE_NAMES if name in feature_names
        ]
    if not chunk_feature_names:
        logging.warning(
            "Chunk %s has no recognised features; returning placeholder result",
            chunk_index,
        )
        return _placeholder_result(
            n_samples=0,
            n_positive=0,
            n_negative=0,
            dropped_nonfinite=0,
            dropped_unpaired=0,
        )

    # Drop rows with non-finite features
    chunk_records, dropped_nonfinite = _filter_finite(chunk_records, chunk_feature_names)
    if not chunk_records:
        logging.warning("Chunk %s has no finite records after filtering; skipping", chunk_index)
        return _placeholder_result(
            n_samples=0,
            n_positive=0,
            n_negative=0,
            dropped_nonfinite=dropped_nonfinite,
            dropped_unpaired=0,
        )

    # Compute episode metadata for fitted rows
    episode_starts = [
        record.episode_start for record in chunk_records if record.episode_start is not None
    ]
    episode_start = int(min(episode_starts)) if episode_starts else meta_episode_start
    chunk_sizes = [
        record.chunk_size for record in chunk_records if record.chunk_size is not None
    ]
    chunk_size = int(min(chunk_sizes)) if chunk_sizes else meta_chunk_size

    dropped_unpaired = 0
    if centre_within_seed:
        chunk_records, dropped_unpaired = _centre_within_seed(
            chunk_records, chunk_feature_names
        )
        if not chunk_records:
            logging.warning(
                "Chunk %s has no records after seed centring; skipping", chunk_index
            )
            return _placeholder_result(
                n_samples=0,
                n_positive=0,
                n_negative=0,
                dropped_nonfinite=dropped_nonfinite,
                dropped_unpaired=dropped_unpaired,
            )

    labels = [1 if record.agent == positive_label else 0 for record in chunk_records]
    class_counts = {label: labels.count(label) for label in set(labels)}
    if len(class_counts) < 2:
        logging.warning(
            "Chunk %s does not contain both classes after preprocessing; skipping", chunk_index
        )
        return _placeholder_result(
            n_samples=len(chunk_records),
            n_positive=int(class_counts.get(1, 0)),
            n_negative=int(class_counts.get(0, 0)),
            dropped_nonfinite=dropped_nonfinite,
            dropped_unpaired=dropped_unpaired,
        )

    feature_matrix = np.array(
        [[record.get_feature(name) for name in chunk_feature_names] for record in chunk_records],
        dtype=float,
    )
    labels_array = np.asarray(labels, dtype=float)
    design = pd.DataFrame(feature_matrix, columns=list(chunk_feature_names))
    design.insert(0, "const", 1.0)

    try:
        model = sm.Logit(labels_array, design)
        fitted = model.fit(disp=0, maxiter=int(max_iter))
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError) as exc:
        logging.warning(
            "Chunk %s: failed to fit logistic model (%s); returning placeholder",
            chunk_index,
            exc,
        )
        return _placeholder_result(
            n_samples=len(chunk_records),
            n_positive=int(class_counts.get(1, 0)),
            n_negative=int(class_counts.get(0, 0)),
            dropped_nonfinite=dropped_nonfinite,
            dropped_unpaired=dropped_unpaired,
        )

    probabilities = fitted.predict(design)
    predictions = (probabilities >= 0.5).astype(int)
    accuracy = accuracy_score(labels, predictions)
    probability_matrix = np.column_stack([1.0 - probabilities, probabilities])
    log_loss_value = log_loss(labels, probability_matrix)

    params = fitted.params
    bse = fitted.bse
    zscores = fitted.tvalues
    pvalues = fitted.pvalues

    result = ChunkResult(
        chunk_index=chunk_index,
        episode_start=episode_start,
        chunk_size=chunk_size,
        n_samples=len(chunk_records),
        n_positive=int(class_counts.get(1, 0)),
        n_negative=int(class_counts.get(0, 0)),
        intercept=float(params["const"]),
        intercept_se=float(bse["const"]),
        intercept_z=float(zscores["const"]),
        intercept_p=float(pvalues["const"]),
        coef_current=float(params["f_current"]),
        coef_current_se=float(bse["f_current"]),
        coef_current_z=float(zscores["f_current"]),
        coef_current_p=float(pvalues["f_current"]),
        coef_previous=float(params.get("f_previous", float("nan"))),
        coef_previous_se=float(bse.get("f_previous", float("nan"))),
        coef_previous_z=float(zscores.get("f_previous", float("nan"))),
        coef_previous_p=float(pvalues.get("f_previous", float("nan"))),
        coef_walls=float(params["f_walls"]),
        coef_walls_se=float(bse["f_walls"]),
        coef_walls_z=float(zscores["f_walls"]),
        coef_walls_p=float(pvalues["f_walls"]),
        accuracy=float(accuracy),
        log_loss_value=float(log_loss_value),
        dropped_nonfinite=dropped_nonfinite,
        dropped_unpaired=dropped_unpaired,
        extra_coefficients={
            name: {
                "coef": float(params.get(name, float("nan"))),
                "se": float(bse.get(name, float("nan"))),
                "z": float(zscores.get(name, float("nan"))),
                "p": float(pvalues.get(name, float("nan"))),
            }
            for name in extra_feature_names
        },
    )
    logging.info(
        "Chunk %s: fitted with %d samples (pos=%d, neg=%d, dropped_na=%d, dropped_unpaired=%d)",
        chunk_index,
        result.n_samples,
        result.n_positive,
        result.n_negative,
        dropped_nonfinite,
        dropped_unpaired,
    )
    return result


def _fit_chunk_models(
    records: Sequence[Record],
    positive_label: str,
    inverse_regularisation: float,
    max_iter: int,
    centre_within_seed: bool,
    feature_names: Sequence[str],
    extra_feature_names: Sequence[str],
    worker_count: int = 1,
) -> List[ChunkResult]:
    _ = inverse_regularisation  # Retained for CLI compatibility; statsmodels fit is unregularised.
    results: List[ChunkResult] = []
    by_chunk: Dict[int, List[Record]] = {}
    for record in records:
        by_chunk.setdefault(record.chunk_index, []).append(record)

    chunk_indices = sorted(by_chunk)

    if worker_count <= 1:
        for chunk_index in chunk_indices:
            chunk_copy = [_copy_record(rec) for rec in by_chunk[chunk_index]]
            result = _fit_single_chunk(
                chunk_index,
                chunk_copy,
                positive_label,
                max_iter,
                centre_within_seed,
                feature_names,
                extra_feature_names,
            )
            if result is not None:
                results.append(result)
    else:
        futures = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for chunk_index in chunk_indices:
                chunk_copy = [_copy_record(rec) for rec in by_chunk[chunk_index]]
                futures.append(
                    executor.submit(
                        _fit_single_chunk,
                        chunk_index,
                        chunk_copy,
                        positive_label,
                        max_iter,
                        centre_within_seed,
                        feature_names,
                        extra_feature_names,
                    )
                )
            for future in futures:
                result = future.result()
                if result is not None:
                    results.append(result)

    results.sort(key=lambda chunk: chunk.chunk_index)
    return results


# def main(argv: Sequence[str] | None = None) -> None:
#     args = parse_args(argv)
#     logging.basicConfig(
#         level=getattr(logging, str(args.log_level).upper(), logging.INFO),
#         format="%(levelname)s:%(name)s:%(message)s",
#     )

#     current_map = _load_feature(args.current_dir, "current-hole")
#     previous_map = _load_feature(args.previous_dir, "previous-hole")
#     walls_map = _load_feature(args.walls_dir, "walls")
#     episode_start_map = _load_episode_starts(args.current_dir)

#     joined_records = _join_records(current_map, previous_map, walls_map, episode_start_map)

#     current_breakdown: Dict[Tuple[str, str, int], List[WorldBreakdownSummary]] = {}
#     current_totals: Dict[int, Tuple[float, int]] = {}
#     previous_breakdown: Dict[Tuple[str, str, int], List[WorldBreakdownSummary]] = {}
#     previous_totals: Dict[int, Tuple[float, int]] = {}
#     walls_breakdown: Dict[Tuple[str, str, int], List[WorldBreakdownSummary]] = {}
#     walls_totals: Dict[int, Tuple[float, int]] = {}

#     previous_load_failed = False
#     try:
#         previous_breakdown, previous_totals = _load_world_breakdown(args.previous_dir)
#     except FileNotFoundError:
#         logging.warning(
#             "Could not load previous-hole world breakdown; lag regressors will be unavailable"
#         )
#         previous_breakdown = {}
#         previous_totals = {}
#         previous_load_failed = True

#     # Determine the source directory for per-lag breakdowns used to create
#     # flexible features f_holes_world_t_minus_k.
#     lag_candidate = args.lag_features_dir or args.previous_dir
#     lag_dir = _resolve_lag_features_dir(lag_candidate)
#     lag_breakdown: Dict[Tuple[str, str, int], List[WorldBreakdownSummary]] = {}
#     if lag_dir is not None:
#         try:
#             lag_breakdown, _ = _load_world_breakdown(lag_dir)
#             logging.info("Using lag breakdowns from %s", lag_dir)
#         except FileNotFoundError:
#             logging.warning(
#                 "Lag features JSON missing in %s; falling back to --previous-dir if available",
#                 lag_dir,
#             )
#             lag_breakdown = {}
#     if not lag_breakdown and args.lag_features_dir is not None and args.lag_features_dir != lag_dir:
#         # Explicit lag dir provided but unresolved; try it directly as last resort
#         try:
#             lag_breakdown, _ = _load_world_breakdown(args.lag_features_dir)
#             logging.info("Using lag breakdowns from %s", args.lag_features_dir)
#         except FileNotFoundError:
#             lag_breakdown = {}

#     # Fallback to the breakdown from --previous-dir if nothing else worked
#     if not lag_breakdown:
#         if previous_breakdown:
#             logging.info("Using lag breakdowns from --previous-dir (%s)", args.previous_dir)
#             lag_breakdown = previous_breakdown
#         else:
#             logging.warning(
#                 "No lag breakdowns available; flexible lag features will not be added"
#             )

#     if lag_breakdown:
#         _populate_previous_lag_features(joined_records, lag_breakdown)

#     if args.demean_by_world:
#         logging.info("Applying world-level demeaning to chunk features")
#         try:
#             current_breakdown, current_totals = _load_world_breakdown(args.current_dir)
#         except FileNotFoundError:
#             logging.warning(
#                 "Current-hole world breakdown missing; current-hole features will not be demeaned"
#             )
#             current_breakdown = {}
#             current_totals = {}

#         if not previous_breakdown and not previous_totals and not previous_load_failed:
#             try:
#                 previous_breakdown, previous_totals = _load_world_breakdown(args.previous_dir)
#             except FileNotFoundError:
#                 previous_breakdown = {}
#                 previous_totals = {}
#                 previous_load_failed = True

#         try:
#             walls_breakdown, walls_totals = _load_world_breakdown(args.walls_dir)
#         except FileNotFoundError:
#             logging.warning(
#                 "Wall world breakdown missing; wall features will not be demeaned"
#             )
#             walls_breakdown = {}
#             walls_totals = {}

#         current_means = _compute_world_means(current_totals)
#         previous_means = _compute_world_means(previous_totals)
#         walls_means = _compute_world_means(walls_totals)

#         if not (current_means or previous_means or walls_means):
#             logging.warning(
#                 "World demeaning requested but no per-world statistics were available; "
#                 "features will remain unchanged"
#             )

#         for record in joined_records:
#             key = (record.seed, record.agent, record.chunk_index)
#             current_value = record.get_feature("f_current")
#             previous_value = record.get_feature("f_previous")
#             walls_value = record.get_feature("f_walls")

#             record.set_feature(
#                 "f_current",
#                 _demean_feature(current_value, current_breakdown.get(key), current_means),
#             )
#             record.set_feature(
#                 "f_previous",
#                 _demean_feature(previous_value, previous_breakdown.get(key), previous_means),
#             )
#             record.set_feature(
#                 "f_walls",
#                 _demean_feature(walls_value, walls_breakdown.get(key), walls_means),
#             )

#     output_dir = args.output_dir
#     output_dir.mkdir(parents=True, exist_ok=True)

#     dataset_path = output_dir / "chunked_occupancy_features.csv"
#     _write_joined_dataset(dataset_path, joined_records)

#     feature_names, extra_feature_names = _feature_name_union(joined_records)

#     results = _fit_chunk_models(
#         joined_records,
#         positive_label=args.positive_label,
#         inverse_regularisation=args.inverse_regularisation,
#         max_iter=args.max_iter,
#         centre_within_seed=args.centre_within_seed,
#         feature_names=feature_names,
#         extra_feature_names=extra_feature_names,
#         worker_count=max(1, int(args.workers)),
#     )

#     summary_path = output_dir / "chunk_glm_coefficients.csv"
#     _write_summary(
#         summary_path,
#         results,
#         args.centre_within_seed,
#         args.demean_by_world,
#         extra_feature_names,
#     )

#     figure_path = output_dir / "figures" / "chunk_glm_coefficients.png"
#     _plot_coefficients(
#         figure_path,
#         results,
#         args.centre_within_seed,
#         args.demean_by_world,
#         args.positive_label,
#         _SIGNIFICANCE_LEVEL,
#         extra_feature_names,
#     )

#     figure_all_path = output_dir / "figures" / "chunk_glm_coefficients_all_values.png"
#     _plot_coefficients_with_significance_bars(
#         figure_all_path,
#         results,
#         args.centre_within_seed,
#         args.demean_by_world,
#         args.positive_label,
#         _SIGNIFICANCE_LEVEL,
#         extra_feature_names,
#     )

#     pvalue_path = output_dir / "figures" / "chunk_glm_pvalues.png"
#     _plot_pvalues(
#         pvalue_path,
#         results,
#         args.centre_within_seed,
#         args.demean_by_world,
#         _SIGNIFICANCE_LEVEL,
#         extra_feature_names,
#     )


# if __name__ == "__main__":  # pragma: no cover - CLI entry point
#     main()
