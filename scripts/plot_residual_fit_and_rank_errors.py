#!/usr/bin/env python3
"""
Build paper-friendly diagnostics from leakage-free runs:
1) Full-data residual fit scatter (observed residual vs predicted residual)
2) Heldout prediction-vs-observed scatter by protocol/space
"""

from __future__ import annotations

import argparse
import colorsys
import itertools
import json
import math
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd


CV_SELECTION_SPECS: dict[str, dict[str, object]] = {
    "loto_pair_win": {
        "summary_file": "prediction_loto_holdout_placement_summary.csv",
        "metric_cols": ["pairwise_win_rate", "pairwise_win_rate_micro"],
        "maximize": True,
    },
    "loto_rank_pct_err": {
        "summary_file": "prediction_loto_holdout_placement_summary.csv",
        "metric_cols": ["abs_rank_pct_error", "abs_rank_pct_error_micro"],
        "maximize": False,
    },
    "loto_rank_spearman": {
        "summary_file": "prediction_loto_holdout_placement_summary.csv",
        "metric_cols": ["rank_spearman", "rank_spearman_fisher", "rank_spearman_micro"],
        "maximize": True,
    },
    "loto_rank_spearman_micro": {
        "summary_file": "prediction_loto_holdout_placement_summary.csv",
        "metric_cols": ["rank_spearman_micro", "rank_spearman", "rank_spearman_fisher"],
        "maximize": True,
    },
    "joint_pair_win": {
        "summary_file": "prediction_jointood_holdout_placement_summary.csv",
        "metric_cols": ["pairwise_win_rate", "pairwise_win_rate_micro"],
        "maximize": True,
    },
    "joint_rank_pct_err": {
        "summary_file": "prediction_jointood_holdout_placement_summary.csv",
        "metric_cols": ["abs_rank_pct_error", "abs_rank_pct_error_micro"],
        "maximize": False,
    },
    "joint_rank_spearman_micro": {
        "summary_file": "prediction_jointood_holdout_placement_summary.csv",
        "metric_cols": ["rank_spearman_micro", "rank_spearman", "rank_spearman_fisher"],
        "maximize": True,
    },
    "lobo_top1": {
        "summary_file": "prediction_lobo_rank_summary.csv",
        "metric_cols": ["top1"],
        "maximize": True,
    },
    "lobo_spearman": {
        "summary_file": "prediction_lobo_rank_summary.csv",
        "metric_cols": ["spearman"],
        "maximize": True,
    },
    "lobo_pred_spearman_global": {
        "summary_file": "prediction_lobo_rows.csv",
        "metric_cols": [],
        "derived_metric": "spearman",
        "derived_true_col": "target",
        "derived_pred_col": "prediction",
        "maximize": True,
    },
    "lobo_pred_spearman_macrofold": {
        "summary_file": "prediction_lobo_rows.csv",
        "metric_cols": [],
        "derived_metric": "spearman",
        "derived_true_col": "target",
        "derived_pred_col": "prediction",
        "derived_group_col": "fold",
        "derived_group_agg": "mean",
        "maximize": True,
    },
    "lobo_pred_spearman_contextmean": {
        "summary_file": "prediction_lobo_rows.csv",
        "metric_cols": [],
        "derived_metric": "spearman",
        "derived_true_col": "target",
        "derived_pred_col": "prediction",
        "derived_collapse_cols": ["train_dataset", "benchmark"],
        "derived_collapse_agg": "mean",
        "maximize": True,
    },
    "loto_pred_spearman_global": {
        "summary_file": "prediction_loto_rows.csv",
        "metric_cols": [],
        "derived_metric": "spearman",
        "derived_true_col": "target",
        "derived_pred_col": "prediction",
        "maximize": True,
    },
    "loto_pred_spearman_macrofold": {
        "summary_file": "prediction_loto_rows.csv",
        "metric_cols": [],
        "derived_metric": "spearman",
        "derived_true_col": "target",
        "derived_pred_col": "prediction",
        "derived_group_col": "fold",
        "derived_group_agg": "mean",
        "maximize": True,
    },
    "loto_pred_spearman_contextmean": {
        "summary_file": "prediction_loto_rows.csv",
        "metric_cols": [],
        "derived_metric": "spearman",
        "derived_true_col": "target",
        "derived_pred_col": "prediction",
        "derived_collapse_cols": ["train_dataset", "benchmark"],
        "derived_collapse_agg": "mean",
        "maximize": True,
    },
    "jointood_pred_spearman_global": {
        "summary_file": "prediction_jointood_rows.csv",
        "metric_cols": [],
        "derived_metric": "spearman",
        "derived_true_col": "target",
        "derived_pred_col": "prediction",
        "maximize": True,
    },
    "jointood_pred_spearman_macrofold": {
        "summary_file": "prediction_jointood_rows.csv",
        "metric_cols": [],
        "derived_metric": "spearman",
        "derived_true_col": "target",
        "derived_pred_col": "prediction",
        "derived_group_col": "fold",
        "derived_group_agg": "mean",
        "maximize": True,
    },
    "jointood_pred_spearman_contextmean": {
        "summary_file": "prediction_jointood_rows.csv",
        "metric_cols": [],
        "derived_metric": "spearman",
        "derived_true_col": "target",
        "derived_pred_col": "prediction",
        "derived_collapse_cols": ["train_dataset", "benchmark"],
        "derived_collapse_agg": "mean",
        "maximize": True,
    },
    "lobo_pred_mae_global": {
        "summary_file": "prediction_lobo_rows.csv",
        "metric_cols": [],
        "derived_metric": "mae",
        "derived_true_col": "target",
        "derived_pred_col": "prediction",
        "maximize": False,
    },
    "loto_pred_mae_global": {
        "summary_file": "prediction_loto_rows.csv",
        "metric_cols": [],
        "derived_metric": "mae",
        "derived_true_col": "target",
        "derived_pred_col": "prediction",
        "maximize": False,
    },
    "jointood_pred_mae_global": {
        "summary_file": "prediction_jointood_rows.csv",
        "metric_cols": [],
        "derived_metric": "mae",
        "derived_true_col": "target",
        "derived_pred_col": "prediction",
        "maximize": False,
    },
    "lobo_cindex": {
        "summary_file": "prediction_lobo_rank_summary.csv",
        "metric_cols": ["pairwise_cindex"],
        "maximize": True,
    },
    "lobo_rank_pct_err": {
        "summary_file": "prediction_lobo_rank_summary.csv",
        "metric_cols": ["mean_abs_rank_pct_error", "median_abs_rank_pct_error"],
        "maximize": False,
    },
}

SPECIAL_CV_SELECTION_KEYS: set[str] = {
    "lobo_rankfirst_antidegen",
    "loto_rankfirst_antidegen",
    "jointood_rankfirst_antidegen",
    "rankfirst_combo",
}
HELDOUT_MODEL_CV_RANKFIRST_KEY = "heldout_model_cv_rankfirst_antidegen"
RANKFIRST_MIN_DISPERSION_RATIO = 0.5

RANKFIRST_PROTOCOL_RULES: dict[str, dict[str, object]] = {
    "lobo": {
        "primary_summary": "prediction_lobo_rank_summary.csv",
        "primary_col": "spearman",
        "tie_cols": [("pairwise_cindex", True), ("regret", False)],
        "overall_cols": ["benchmark", "model_family_encoder"],
        "dispersion_summary": "prediction_lobo_summary.csv",
        "dispersion_overall_cols": ["benchmark"],
    },
    "loto": {
        "primary_summary": "prediction_loto_holdout_placement_summary.csv",
        "primary_col": "rank_spearman_micro",
        "tie_cols": [("pairwise_win_rate_micro", True), ("abs_rank_pct_error_micro", False)],
        "overall_cols": ["fold"],
        "dispersion_summary": "prediction_loto_summary.csv",
        "dispersion_overall_cols": ["train_dataset"],
    },
    "jointood": {
        "primary_summary": "prediction_jointood_holdout_placement_summary.csv",
        "primary_col": "rank_spearman_micro",
        "tie_cols": [("pairwise_win_rate_micro", True), ("abs_rank_pct_error_micro", False)],
        "overall_cols": ["fold"],
        "dispersion_summary": "prediction_jointood_summary.csv",
        "dispersion_overall_cols": ["joint_holdout", "train_dataset", "benchmark"],
    },
}


HELDOUT_PROTOCOL_ALIASES: dict[str, str] = {
    "lobo": "lobo",
    "loto": "loto",
    "jointood": "jointood",
    "joint_ood": "jointood",
    "joint-ood": "jointood",
    "model": "model",
    "model_only": "model",
    "lomo": "model",
    "training": "training",
    "train": "training",
    "train_only": "training",
    "eval": "eval",
    "evaluation": "eval",
    "benchmark": "eval",
    "eval_only": "eval",
    "model_train_benchmark": "training",
    "model_benchmark": "eval",
    "triple": "triple",
    "stress": "triple",
    "model_train_benchmark_disjoint": "triple",
    "trainset_disjoint": "trainset_disjoint",
    "model_benchmark_trainset_disjoint": "trainset_disjoint",
}

DEFAULT_HELDOUT_PROTOCOL_METRICS: dict[str, str] = {
    # Rank-first default selectors with anti-degeneracy guard.
    "lobo": "lobo_rankfirst_antidegen",
    "loto": "loto_rankfirst_antidegen",
    "jointood": "jointood_rankfirst_antidegen",
}

DEFAULT_HELDOUT_MODEL_CV_PROTOCOL_METRICS: dict[str, str] = {
    # For model-only protocol, prefer rank-first anti-degeneracy candidate selection.
    "model": HELDOUT_MODEL_CV_RANKFIRST_KEY,
    # For other heldout_model_cv protocols, Spearman remains the default.
    "training": "spearman",
    "eval": "spearman",
    "triple": "spearman",
    "trainset_disjoint": "spearman",
}

HELDOUT_SPACE_ALIASES: dict[str, str] = {
    "model": "model_space",
    "model_space": "model_space",
    "z": "model_space",
    "zscore": "model_space",
    "residual": "residual",
    "resid": "residual",
    "absolute": "absolute",
    "abs": "absolute",
}


def _parse_csv_list(text: str | None) -> list[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def _resolve_heldout_protocols(text: str | None) -> list[str]:
    vals = _parse_csv_list(text)
    out: list[str] = []
    seen = set()
    for raw in vals:
        key = str(raw).strip().lower()
        if not key:
            continue
        canon = HELDOUT_PROTOCOL_ALIASES.get(key, key)
        if canon not in seen:
            out.append(canon)
            seen.add(canon)
    return out


def _collapse_overlapping_heldout_protocols(protocols: list[str]) -> tuple[list[str], list[str]]:
    vals = [str(x).strip().lower() for x in list(protocols or []) if str(x).strip()]
    has_lobo = "lobo" in vals
    has_model = "model" in vals
    out: list[str] = []
    seen = set()
    dropped: list[str] = []
    for p in vals:
        if p == "eval" and has_lobo:
            dropped.append("eval (dropped; overlaps with lobo)")
            continue
        if p == "lomo" and has_model:
            dropped.append("lomo (dropped; overlaps with model)")
            continue
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out, dropped


def _resolve_heldout_metric_map(text: str | None) -> dict[str, str]:
    out = dict(DEFAULT_HELDOUT_PROTOCOL_METRICS)
    out.update(DEFAULT_HELDOUT_MODEL_CV_PROTOCOL_METRICS)
    for k, v in _parse_kv_csv(text).items():
        pk = HELDOUT_PROTOCOL_ALIASES.get(str(k).strip().lower(), str(k).strip().lower())
        mv = str(v).strip()
        if not pk or not mv:
            continue
        out[pk] = mv
    return out


def _resolve_heldout_plot_spaces(text: str | None) -> list[str]:
    vals = _parse_csv_list(text)
    if not vals:
        vals = ["model_space"]
    out: list[str] = []
    seen = set()
    for raw in vals:
        key = str(raw).strip().lower()
        if not key:
            continue
        canon = HELDOUT_SPACE_ALIASES.get(key, key)
        if canon not in {"model_space", "residual", "absolute"}:
            continue
        if canon not in seen:
            out.append(canon)
            seen.add(canon)
    return out or ["model_space"]


def _resolve_color_modes(color_by_arg: str | None) -> list[str]:
    raw = str(color_by_arg or "").strip()
    if not raw:
        return [""]
    if raw.lower() in {"both", "dual"}:
        return ["train_dataset", "benchmark"]
    vals = _parse_csv_list(raw)
    if not vals:
        return [""]
    out = []
    seen = set()
    for v in vals:
        key = str(v).strip()
        if key and key not in seen:
            out.append(key)
            seen.add(key)
    return out or [""]


def _parse_kv_csv(text: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not text:
        return out
    for chunk in str(text).split(","):
        part = chunk.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k:
            out[k] = v
    return out


def _safe_context_suffix(context_kv: dict[str, str]) -> str:
    if not context_kv:
        return ""
    parts: list[str] = []
    for k, v in context_kv.items():
        token = f"{k}_{v}".replace("/", "_").replace(" ", "_").replace("-", "_")
        parts.append(token)
    return "__".join(parts)


def _first_non_null(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def _collapse_rows(df: pd.DataFrame, group_cols: list[str], numeric_agg: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    group_cols = [c for c in group_cols if c in df.columns]
    if len(group_cols) < 2:
        return df
    agg_mode = str(numeric_agg or "median").strip().lower()
    if agg_mode not in {"mean", "median"}:
        agg_mode = "median"
    agg_map = {}
    for col in df.columns:
        if col in group_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            agg_map[col] = agg_mode
        else:
            agg_map[col] = _first_non_null
    grouped = df.groupby(group_cols, dropna=False, sort=False)
    out = grouped.agg(agg_map).reset_index()
    out["cell_n_rows"] = grouped.size().to_numpy()
    return out


def _collapse_points(df: pd.DataFrame, group_cols: list[str], numeric_agg: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    group_cols = [c for c in group_cols if c in df.columns]
    if len(group_cols) < 1:
        return df
    agg_mode = str(numeric_agg or "median").strip().lower()
    if agg_mode not in {"mean", "median"}:
        agg_mode = "median"
    agg_map = {}
    for col in df.columns:
        if col in group_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            agg_map[col] = agg_mode
        else:
            agg_map[col] = _first_non_null
    grouped = df.groupby(group_cols, dropna=False, sort=False)
    out = grouped.agg(agg_map).reset_index()
    out["_collapsed_n_rows"] = grouped.size().to_numpy()
    return out


def _zscore(col: pd.Series) -> pd.Series:
    std = float(col.std(ddof=0))
    if not np.isfinite(std) or std <= 0.0:
        return pd.Series(np.zeros(len(col), dtype=float), index=col.index)
    mean = float(col.mean())
    return (col - mean) / std


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x0 = x - np.mean(x)
    y0 = y - np.mean(y)
    denom = np.linalg.norm(x0) * np.linalg.norm(y0)
    if denom == 0:
        return float("nan")
    return float(np.dot(x0, y0) / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return _pearson(xr, yr)


def _approx_n_unique(values: np.ndarray, abs_tol: float = 1e-8, rel_tol: float = 1e-3) -> int:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0
    arr = np.sort(arr)
    n_unique = 1
    prev = float(arr[0])
    for v in arr[1:]:
        vv = float(v)
        tol = max(float(abs_tol), float(rel_tol) * max(1.0, abs(prev), abs(vv)))
        if abs(vv - prev) > tol:
            n_unique += 1
            prev = vv
    return int(n_unique)


def _compute_dispersion_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-3,
    std_ratio_warn: float = 0.35,
    unique_frac_warn: float = 0.35,
) -> dict[str, float | int | str | bool]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[valid]
    yp = yp[valid]
    n = int(len(yt))
    if n == 0:
        return {
            "n_rows": 0,
            "y_true_std": float("nan"),
            "y_pred_std": float("nan"),
            "std_ratio": float("nan"),
            "y_true_range": float("nan"),
            "y_pred_range": float("nan"),
            "range_ratio": float("nan"),
            "y_true_iqr": float("nan"),
            "y_pred_iqr": float("nan"),
            "iqr_ratio": float("nan"),
            "pred_unique_approx": 0,
            "pred_unique_frac_approx": float("nan"),
            "pred_tie_frac_approx": float("nan"),
            "dispersion_flag": True,
            "dispersion_note": "no valid rows",
        }

    y_true_std = float(np.std(yt, ddof=0))
    y_pred_std = float(np.std(yp, ddof=0))
    std_ratio = float(y_pred_std / y_true_std) if y_true_std > 0 else float("nan")
    y_true_range = float(np.max(yt) - np.min(yt))
    y_pred_range = float(np.max(yp) - np.min(yp))
    range_ratio = float(y_pred_range / y_true_range) if y_true_range > 0 else float("nan")
    y_true_iqr = float(np.quantile(yt, 0.75) - np.quantile(yt, 0.25))
    y_pred_iqr = float(np.quantile(yp, 0.75) - np.quantile(yp, 0.25))
    iqr_ratio = float(y_pred_iqr / y_true_iqr) if y_true_iqr > 0 else float("nan")
    pred_unique_approx = int(_approx_n_unique(yp, abs_tol=abs_tol, rel_tol=rel_tol))
    pred_unique_frac_approx = float(pred_unique_approx / n) if n > 0 else float("nan")
    pred_tie_frac_approx = float(1.0 - pred_unique_frac_approx) if n > 0 else float("nan")

    flags: list[str] = []
    if np.isfinite(std_ratio) and std_ratio < float(std_ratio_warn):
        flags.append(f"low_std_ratio<{std_ratio_warn:g}")
    if np.isfinite(pred_unique_frac_approx) and pred_unique_frac_approx < float(unique_frac_warn):
        flags.append(f"low_unique_frac<{unique_frac_warn:g}")
    if np.isfinite(range_ratio) and range_ratio < float(std_ratio_warn):
        flags.append(f"low_range_ratio<{std_ratio_warn:g}")
    if not flags and np.isfinite(iqr_ratio) and iqr_ratio < float(std_ratio_warn):
        flags.append(f"low_iqr_ratio<{std_ratio_warn:g}")

    dispersion_flag = bool(flags)
    note = ",".join(flags) if flags else "ok"
    return {
        "n_rows": n,
        "y_true_std": y_true_std,
        "y_pred_std": y_pred_std,
        "std_ratio": std_ratio,
        "y_true_range": y_true_range,
        "y_pred_range": y_pred_range,
        "range_ratio": range_ratio,
        "y_true_iqr": y_true_iqr,
        "y_pred_iqr": y_pred_iqr,
        "iqr_ratio": iqr_ratio,
        "pred_unique_approx": pred_unique_approx,
        "pred_unique_frac_approx": pred_unique_frac_approx,
        "pred_tie_frac_approx": pred_tie_frac_approx,
        "dispersion_flag": dispersion_flag,
        "dispersion_note": note,
    }


def _transform_prediction(
    y_pred: np.ndarray,
    mode: str = "none",
    eps: float = 1e-9,
) -> tuple[np.ndarray, dict]:
    mode = str(mode or "none").strip().lower()
    out = np.asarray(y_pred, dtype=float).copy()
    info = {
        "prediction_transform": mode,
        "prediction_transform_mean": float("nan"),
        "prediction_transform_std": float("nan"),
    }
    if mode == "zscore":
        mean = float(np.mean(out))
        std = float(np.std(out, ddof=0))
        info["prediction_transform_mean"] = mean
        info["prediction_transform_std"] = std
        if np.isfinite(std) and std > float(eps):
            out = (out - mean) / std
        else:
            out = out - mean
    return out, info


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    X_aug = np.column_stack([np.ones(len(X), dtype=float), X])
    penalty = np.eye(X_aug.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    return np.linalg.solve(X_aug.T @ X_aug + float(alpha) * penalty, X_aug.T @ y)


def _resolve_predictors(meta: dict, predictors_arg: str | None) -> list[str]:
    if predictors_arg:
        return _parse_csv_list(predictors_arg)
    raw = meta.get("predictors", [])
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        return _parse_csv_list(raw)
    return []


def _resolve_context_cols(meta: dict, context_arg: str | None) -> list[str]:
    if context_arg:
        return _parse_csv_list(context_arg)
    return _parse_csv_list(meta.get("cv_residual_context_cols", ""))


def _load_metadata(run_dir: Path) -> dict:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _selection_metric_choices() -> list[str]:
    return sorted(set(CV_SELECTION_SPECS.keys()).union(SPECIAL_CV_SELECTION_KEYS))


def _select_overall_rows(df: pd.DataFrame, preferred_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    matched_any = False
    for col in preferred_cols:
        if col not in work.columns:
            continue
        mask = work[col].astype(str) == "__overall__"
        if bool(mask.any()):
            work = work[mask].copy()
            matched_any = True
    if matched_any and not work.empty:
        return work

    fallback_cols = [
        c
        for c in ["fold", "benchmark", "train_dataset", "joint_holdout", "model_family_encoder"]
        if c in df.columns
    ]
    for col in fallback_cols:
        mask = df[col].astype(str) == "__overall__"
        if bool(mask.any()):
            return df[mask].copy()
    return df.copy()


def _metric_mean_from_rows(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))


def _metric_value_for_sort(value: float, maximize: bool) -> float:
    if not np.isfinite(value):
        return float("inf")
    return -float(value) if maximize else float(value)


def _read_protocol_rank_metrics(
    run_dir: Path,
    protocol: str,
) -> dict[str, object] | None:
    rules = RANKFIRST_PROTOCOL_RULES.get(str(protocol).strip().lower())
    if not rules:
        return None
    primary_summary = run_dir / str(rules["primary_summary"])
    dispersion_summary = run_dir / str(rules["dispersion_summary"])
    if not primary_summary.exists() or not dispersion_summary.exists():
        return None
    try:
        primary_df = pd.read_csv(primary_summary)
        dispersion_df = pd.read_csv(dispersion_summary)
    except Exception:
        return None
    if primary_df.empty or dispersion_df.empty:
        return None

    primary_rows = _select_overall_rows(primary_df, [str(c) for c in list(rules["overall_cols"])])
    dispersion_rows = _select_overall_rows(dispersion_df, [str(c) for c in list(rules["dispersion_overall_cols"])])

    primary_col = str(rules["primary_col"])
    tie_cols = [(str(c), bool(maximize)) for c, maximize in list(rules["tie_cols"])]
    primary_value = _metric_mean_from_rows(primary_rows, primary_col)
    tie_values = {col: _metric_mean_from_rows(primary_rows, col) for col, _ in tie_cols}

    if "target_std" not in dispersion_rows.columns or "pred_std" not in dispersion_rows.columns:
        dispersion_ratio = float("nan")
    else:
        target_std = pd.to_numeric(dispersion_rows["target_std"], errors="coerce")
        pred_std = pd.to_numeric(dispersion_rows["pred_std"], errors="coerce")
        tvals = target_std.to_numpy(dtype=float)
        ratio = (pred_std.to_numpy(dtype=float) / tvals)
        ratio = ratio[np.isfinite(ratio) & np.isfinite(tvals) & (tvals > 0.0)]
        dispersion_ratio = float(np.mean(ratio)) if len(ratio) > 0 else float("nan")

    out: dict[str, object] = {
        "protocol": str(protocol),
        "primary_col": primary_col,
        "primary_value": float(primary_value),
        "dispersion_ratio": float(dispersion_ratio),
        "primary_summary_path": primary_summary,
        "dispersion_summary_path": dispersion_summary,
    }
    for col, _ in tie_cols:
        out[f"tie__{col}"] = float(tie_values.get(col, float("nan")))
    return out


def _select_best_run_dir_rankfirst_protocol(
    run_root: Path,
    protocol: str,
    min_dispersion_ratio: float = RANKFIRST_MIN_DISPERSION_RATIO,
) -> dict[str, object]:
    proto = str(protocol).strip().lower()
    rules = RANKFIRST_PROTOCOL_RULES.get(proto)
    if not rules:
        raise ValueError(f"Unsupported rank-first protocol: {protocol}")
    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")

    candidates: list[dict[str, object]] = []
    for summary_path in run_root.rglob(str(rules["primary_summary"])):
        run_dir = summary_path.parent
        if not (run_dir / "auc_with_features.csv").exists():
            continue
        metrics = _read_protocol_rank_metrics(run_dir, proto)
        if not metrics:
            continue
        disp_ratio = float(metrics["dispersion_ratio"])
        if (not np.isfinite(disp_ratio)) or (disp_ratio < float(min_dispersion_ratio)):
            continue

        primary_col = str(metrics["primary_col"])
        primary_value = float(metrics["primary_value"])
        tie_1_col, tie_1_maximize = list(rules["tie_cols"])[0]
        tie_2_col, tie_2_maximize = list(rules["tie_cols"])[1]
        tie_1 = float(metrics.get(f"tie__{tie_1_col}", float("nan")))
        tie_2 = float(metrics.get(f"tie__{tie_2_col}", float("nan")))

        primary_missing = 1 if (proto == "jointood" and not np.isfinite(primary_value)) else 0
        if proto != "jointood" and not np.isfinite(primary_value):
            continue

        candidates.append(
            {
                "run_dir": run_dir,
                "metric_key": f"{proto}_rankfirst_antidegen",
                "metric_col_used": primary_col,
                "metric_value": primary_value,
                "primary_missing": primary_missing,
                "tie_1_col": str(tie_1_col),
                "tie_1_value": tie_1,
                "tie_2_col": str(tie_2_col),
                "tie_2_value": tie_2,
                "dispersion_ratio": disp_ratio,
                "summary_path": metrics["primary_summary_path"],
                "dispersion_summary_path": metrics["dispersion_summary_path"],
                "maximize": True,
                "sort_key": (
                    int(primary_missing),
                    _metric_value_for_sort(primary_value, maximize=True),
                    _metric_value_for_sort(tie_1, maximize=bool(tie_1_maximize)),
                    _metric_value_for_sort(tie_2, maximize=bool(tie_2_maximize)),
                    str(run_dir),
                ),
            }
        )

    if not candidates:
        raise FileNotFoundError(
            f"No candidate runs passed rank-first anti-degeneracy selection "
            f"for protocol='{proto}' under {run_root}."
        )
    candidates = sorted(candidates, key=lambda r: r["sort_key"])
    best = candidates[0].copy()
    best["n_candidates"] = len(candidates)
    best["selection_mode"] = "rank_first_anti_degeneracy"
    best["min_dispersion_ratio"] = float(min_dispersion_ratio)
    return best


def _select_best_run_dir_rankfirst_combo(
    run_root: Path,
    min_dispersion_ratio: float = RANKFIRST_MIN_DISPERSION_RATIO,
) -> dict[str, object]:
    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")

    candidates: list[dict[str, object]] = []
    for auc_path in run_root.rglob("auc_with_features.csv"):
        run_dir = auc_path.parent
        lobo = _read_protocol_rank_metrics(run_dir, "lobo")
        loto = _read_protocol_rank_metrics(run_dir, "loto")
        joint = _read_protocol_rank_metrics(run_dir, "jointood")
        if not lobo or not loto or not joint:
            continue

        lobo_s = float(lobo.get("primary_value", float("nan")))
        loto_s = float(loto.get("primary_value", float("nan")))
        joint_s = float(joint.get("primary_value", float("nan")))
        ratios = np.asarray(
            [
                float(lobo.get("dispersion_ratio", float("nan"))),
                float(loto.get("dispersion_ratio", float("nan"))),
                float(joint.get("dispersion_ratio", float("nan"))),
            ],
            dtype=float,
        )
        if not (np.isfinite(lobo_s) and np.isfinite(loto_s) and np.isfinite(joint_s)):
            continue
        if not np.all(np.isfinite(ratios)):
            continue
        if bool(np.any(ratios < float(min_dispersion_ratio))):
            continue

        score = float(0.5 * loto_s + 0.3 * lobo_s + 0.2 * joint_s)
        avg_ratio = float(np.mean(ratios))
        ratio_dist_to_one = float(abs(avg_ratio - 1.0))
        candidates.append(
            {
                "run_dir": run_dir,
                "metric_key": "rankfirst_combo",
                "metric_col_used": "0.5*loto_rank_spearman_micro + 0.3*lobo_spearman + 0.2*joint_rank_spearman_micro",
                "metric_value": score,
                "lobo_spearman": lobo_s,
                "loto_rank_spearman_micro": loto_s,
                "jointood_rank_spearman_micro": joint_s,
                "lobo_dispersion_ratio": float(ratios[0]),
                "loto_dispersion_ratio": float(ratios[1]),
                "jointood_dispersion_ratio": float(ratios[2]),
                "avg_dispersion_ratio": avg_ratio,
                "avg_dispersion_ratio_dist_to_one": ratio_dist_to_one,
                "summary_path": run_dir / "prediction_loto_holdout_placement_summary.csv",
                "maximize": True,
                "sort_key": (
                    -score,
                    ratio_dist_to_one,
                    -avg_ratio,
                    str(run_dir),
                ),
            }
        )

    if not candidates:
        raise FileNotFoundError(
            f"No candidate runs passed rank-first combo selection under {run_root}."
        )
    candidates = sorted(candidates, key=lambda r: r["sort_key"])
    best = candidates[0].copy()
    best["n_candidates"] = len(candidates)
    best["selection_mode"] = "rank_first_combo"
    best["min_dispersion_ratio"] = float(min_dispersion_ratio)
    return best


def _resolve_selection_spec(metric_key: str) -> dict[str, object]:
    key = str(metric_key or "").strip()
    if key not in CV_SELECTION_SPECS:
        valid = ", ".join(_selection_metric_choices())
        raise ValueError(f"Unknown --best-cv-metric '{key}'. Valid: {valid}")
    return CV_SELECTION_SPECS[key]


def _extract_metric_value(
    summary_path: Path,
    metric_cols: list[str],
    derived_metric: str | None = None,
    derived_true_col: str | None = None,
    derived_pred_col: str | None = None,
    derived_group_col: str | None = None,
    derived_group_agg: str | None = None,
    derived_collapse_cols: list[str] | None = None,
    derived_collapse_agg: str | None = None,
) -> tuple[float, str | None]:
    try:
        df = pd.read_csv(summary_path)
    except Exception:
        return float("nan"), None
    if df.empty:
        return float("nan"), None

    if derived_metric:
        true_col = str(derived_true_col or "target")
        pred_col = str(derived_pred_col or "prediction")
        if true_col not in df.columns or pred_col not in df.columns:
            return float("nan"), None
        metric_label = str(derived_metric).strip().lower()

        def _metric_from_arrays(y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> float:
            if len(y_true_arr) < 2:
                return float("nan")
            if metric_label == "spearman":
                return _spearman(y_true_arr, y_pred_arr)
            if metric_label == "pearson":
                return _pearson(y_true_arr, y_pred_arr)
            if metric_label == "rmse":
                return float(np.sqrt(np.mean((y_pred_arr - y_true_arr) ** 2)))
            if metric_label == "mae":
                return float(np.mean(np.abs(y_pred_arr - y_true_arr)))
            return float("nan")

        # Optional context collapse: evaluate metric after collapsing repeated rows
        # by context keys (e.g., train_dataset+benchmark) to match context-level predictors.
        collapse_cols = [str(c) for c in list(derived_collapse_cols or []) if str(c) in df.columns]
        if collapse_cols:
            cagg = str(derived_collapse_agg or "mean").strip().lower()
            if cagg not in {"mean", "median"}:
                cagg = "mean"
            agg_fn = "median" if cagg == "median" else "mean"
            tmp = df.copy()
            tmp[true_col] = pd.to_numeric(tmp[true_col], errors="coerce")
            tmp[pred_col] = pd.to_numeric(tmp[pred_col], errors="coerce")
            tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=[true_col, pred_col])
            if tmp.empty:
                return float("nan"), None
            coll = (
                tmp.groupby(collapse_cols, dropna=False)
                .agg(**{true_col: (true_col, agg_fn), pred_col: (pred_col, agg_fn)})
                .reset_index()
            )
            y_true_c = pd.to_numeric(coll[true_col], errors="coerce").to_numpy(dtype=float)
            y_pred_c = pd.to_numeric(coll[pred_col], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(y_true_c) & np.isfinite(y_pred_c)
            if int(np.sum(valid)) < 2:
                return float("nan"), None
            value = _metric_from_arrays(y_true_c[valid], y_pred_c[valid])
            if not np.isfinite(value):
                return float("nan"), None
            return float(value), f"{metric_label}_collapsed({'+'.join(collapse_cols)},{agg_fn})"

        if derived_group_col and str(derived_group_col) in df.columns:
            group_col = str(derived_group_col)
            gvals = []
            for _, g in df.groupby(group_col, dropna=False):
                yt = pd.to_numeric(g[true_col], errors="coerce").to_numpy(dtype=float)
                yp = pd.to_numeric(g[pred_col], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(yt) & np.isfinite(yp)
                if int(np.sum(valid)) < 2:
                    continue
                mv = _metric_from_arrays(yt[valid], yp[valid])
                if np.isfinite(mv):
                    gvals.append(float(mv))
            if not gvals:
                return float("nan"), None
            agg_mode = str(derived_group_agg or "mean").strip().lower()
            if agg_mode == "median":
                value = float(np.median(np.asarray(gvals, dtype=float)))
            else:
                value = float(np.mean(np.asarray(gvals, dtype=float)))
            if not np.isfinite(value):
                return float("nan"), None
            return value, f"{metric_label}_macro({group_col},{agg_mode})"

        y_true = pd.to_numeric(df[true_col], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        if int(np.sum(valid)) < 2:
            return float("nan"), None
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        dkey = str(derived_metric).strip().lower()
        value = _metric_from_arrays(y_true, y_pred)
        if not np.isfinite(value):
            return float("nan"), None
        return float(value), f"{dkey}({true_col},{pred_col})"

    work = df.copy()
    if "fold" in work.columns:
        overall = work[work["fold"].astype(str) == "__overall__"]
        if not overall.empty:
            work = overall

    for col in metric_cols:
        if col not in work.columns:
            continue
        vals = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        if len(vals) == 1:
            return float(vals[0]), col
        return float(np.mean(vals)), col
    return float("nan"), None


def _select_best_run_dir(
    run_root: Path,
    metric_key: str,
    min_dispersion_ratio: float = RANKFIRST_MIN_DISPERSION_RATIO,
) -> dict[str, object]:
    key = str(metric_key or "").strip()
    if key == "lobo_rankfirst_antidegen":
        return _select_best_run_dir_rankfirst_protocol(
            run_root,
            protocol="lobo",
            min_dispersion_ratio=float(min_dispersion_ratio),
        )
    if key == "loto_rankfirst_antidegen":
        return _select_best_run_dir_rankfirst_protocol(
            run_root,
            protocol="loto",
            min_dispersion_ratio=float(min_dispersion_ratio),
        )
    if key == "jointood_rankfirst_antidegen":
        return _select_best_run_dir_rankfirst_protocol(
            run_root,
            protocol="jointood",
            min_dispersion_ratio=float(min_dispersion_ratio),
        )
    if key == "rankfirst_combo":
        return _select_best_run_dir_rankfirst_combo(
            run_root,
            min_dispersion_ratio=float(min_dispersion_ratio),
        )

    spec = _resolve_selection_spec(metric_key)
    summary_file = str(spec["summary_file"])
    metric_cols = [str(x) for x in list(spec["metric_cols"])]
    derived_metric = str(spec.get("derived_metric", "") or "").strip() or None
    derived_true_col = str(spec.get("derived_true_col", "") or "").strip() or None
    derived_pred_col = str(spec.get("derived_pred_col", "") or "").strip() or None
    derived_group_col = str(spec.get("derived_group_col", "") or "").strip() or None
    derived_group_agg = str(spec.get("derived_group_agg", "") or "").strip() or None
    raw_collapse_cols = spec.get("derived_collapse_cols", []) or []
    derived_collapse_cols = [str(x) for x in list(raw_collapse_cols)]
    derived_collapse_agg = str(spec.get("derived_collapse_agg", "") or "").strip() or None
    maximize = bool(spec["maximize"])

    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")

    rows: list[dict[str, object]] = []
    for summary_path in run_root.rglob(summary_file):
        run_dir = summary_path.parent
        if not (run_dir / "auc_with_features.csv").exists():
            continue
        metric_value, metric_col_used = _extract_metric_value(
            summary_path,
            metric_cols,
            derived_metric=derived_metric,
            derived_true_col=derived_true_col,
            derived_pred_col=derived_pred_col,
            derived_group_col=derived_group_col,
            derived_group_agg=derived_group_agg,
            derived_collapse_cols=derived_collapse_cols,
            derived_collapse_agg=derived_collapse_agg,
        )
        if not np.isfinite(metric_value):
            continue
        rows.append(
            {
                "run_dir": run_dir,
                "summary_path": summary_path,
                "metric_value": float(metric_value),
                "metric_col_used": metric_col_used,
            }
        )

    if not rows:
        raise FileNotFoundError(
            f"No candidate runs with usable metric '{metric_key}' "
            f"under {run_root} (expected summary: {summary_file})."
        )

    rows = sorted(
        rows,
        key=lambda r: (
            -float(r["metric_value"]) if maximize else float(r["metric_value"]),
            str(r["run_dir"]),
        ),
    )
    best = rows[0].copy()
    best["metric_key"] = metric_key
    best["maximize"] = maximize
    best["n_candidates"] = len(rows)
    return best


def _derive_repeat_group_cols(meta: dict) -> list[str]:
    cols = ["train_dataset", "benchmark"]
    ranking_group = str(meta.get("ranking_group", "") or "").strip()
    if ranking_group:
        cols.append(ranking_group)
    cols.extend(_parse_csv_list(str(meta.get("ranking_context_cols", "") or "")))
    cols.extend(_parse_csv_list(str(meta.get("pairwise_group_cols", "") or "")))
    if bool(meta.get("cv_residualize_target_by_context", False)):
        cols.extend(_parse_csv_list(str(meta.get("cv_residual_context_cols", "") or "")))
    cols.append("model_family_encoder")
    return list(dict.fromkeys([c for c in cols if c]))


def _build_full_residual_fit(
    run_dir: Path,
    target: str,
    predictors: list[str],
    context_cols: list[str],
    ridge_alpha: float,
    top_k: int,
    passthrough_cols: list[str] | None = None,
    repeat_agg_mode: str = "none",
    repeat_group_cols: list[str] | None = None,
    context_target_transform: str = "residual",
    context_target_zscore_eps: float = 1e-9,
    source_df: pd.DataFrame | None = None,
):
    if source_df is None:
        auc_path = run_dir / "auc_with_features.csv"
        if not auc_path.exists():
            raise FileNotFoundError(f"Missing file: {auc_path}")
        df = pd.read_csv(auc_path)
    else:
        df = source_df.copy()

    if target not in df.columns:
        raise ValueError(f"Target not found in auc_with_features.csv: {target}")

    missing_pred = [p for p in predictors if p not in df.columns]
    predictors = [p for p in predictors if p in df.columns]
    if not predictors:
        raise ValueError("No valid predictors found in auc_with_features.csv.")
    if missing_pred:
        print("Dropped missing predictors:", ", ".join(missing_pred))

    context_cols = [c for c in context_cols if c in df.columns]
    if not context_cols:
        raise ValueError("No valid context columns for residualization.")

    passthrough_cols = [c for c in (passthrough_cols or []) if c not in context_cols and c not in predictors and c != target]
    needed = list(dict.fromkeys(context_cols + predictors + [target] + passthrough_cols + list(repeat_group_cols or [])))
    fit_df = df[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if fit_df.empty:
        raise ValueError("No complete rows after filtering.")
    before_rows = int(len(fit_df))
    applied_repeat_agg = "none"
    used_repeat_group_cols: list[str] = []
    mode = str(repeat_agg_mode or "none").strip().lower()
    if mode in {"mean", "median"}:
        group_cols = [c for c in list(repeat_group_cols or []) if c in fit_df.columns]
        if len(group_cols) >= 2:
            fit_df = _collapse_rows(fit_df, group_cols=group_cols, numeric_agg=mode)
            applied_repeat_agg = mode
            used_repeat_group_cols = group_cols
        else:
            print(
                "Warning: repeat aggregation requested but insufficient grouping columns; "
                "skipping."
            )

    target_mode = str(context_target_transform or "residual").strip().lower()
    context_mean = fit_df.groupby(context_cols, dropna=False)[target].transform("mean")
    fit_df["target_ctx_mean"] = context_mean
    fit_df["target_resid_raw"] = fit_df[target] - context_mean
    fit_df["target_resid"] = fit_df["target_resid_raw"]
    context_std = fit_df.groupby(context_cols, dropna=False)[target].transform(
        lambda s: float(np.std(s.to_numpy(dtype=float), ddof=0))
    )
    std_ok = np.isfinite(context_std.to_numpy(dtype=float)) & (
        context_std.to_numpy(dtype=float) > float(context_target_zscore_eps)
    )
    target_zscore_fallback_count = int((~std_ok).sum())
    safe_std = context_std.copy()
    safe_std.loc[~std_ok] = 1.0
    fit_df["target_ctx_std"] = safe_std
    if target_mode == "zscore":
        fit_df["target_resid"] = fit_df["target_resid_raw"] / safe_std

    z_cols = []
    for p in predictors:
        zc = f"{p}__z"
        fit_df[zc] = _zscore(fit_df[p].astype(float))
        z_cols.append(zc)

    X = fit_df[z_cols].to_numpy(dtype=float)
    y = fit_df["target_resid"].to_numpy(dtype=float)

    coef = _ridge_fit(X, y, ridge_alpha)
    coefs = np.asarray(coef[1:], dtype=float)
    pred_names = list(predictors)

    if top_k > 0 and top_k < len(pred_names):
        idx = np.argsort(np.abs(coefs))[::-1][:top_k]
        keep = sorted(int(i) for i in idx)
        pred_names = [pred_names[i] for i in keep]
        z_keep = [z_cols[i] for i in keep]
        X = fit_df[z_keep].to_numpy(dtype=float)
        coef = _ridge_fit(X, y, ridge_alpha)
        coefs = np.asarray(coef[1:], dtype=float)
        z_cols = z_keep

    yhat = coef[0] + X @ coefs
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    pear = _pearson(y, yhat)
    spear = _spearman(y, yhat)

    coef_df = pd.DataFrame(
        {
            "predictor": pred_names,
            "coef_standardized": coefs,
            "abs_coef": np.abs(coefs),
        }
    ).sort_values("abs_coef", ascending=False)

    agg_info = {
        "repeat_aggregation": applied_repeat_agg,
        "repeat_group_cols": used_repeat_group_cols,
        "rows_before_repeat_agg": before_rows,
        "rows_after_repeat_agg": int(len(fit_df)),
        "context_target_transform": target_mode,
        "context_target_zscore_eps": float(context_target_zscore_eps),
        "context_target_zscore_fallback_rows": int(target_zscore_fallback_count),
    }
    return fit_df, y, yhat, rmse, pear, spear, coef_df, agg_info


def _predictor_family_for_name(name: str) -> str:
    raw = str(name or "").strip()
    low = raw.lower()
    if low == "flow_mmd":
        return "flow_mmd"
    if low in {"dino_mmd", "feature_mmd"}:
        return "appearance_mmd"
    if low.endswith("_mmd"):
        return "other_mmd"
    if low.startswith("flow_") or low.startswith("hof_"):
        return "flow"
    if low.startswith("dino_"):
        return "appearance"
    return "other"


def _predictor_family_buckets(predictors: list[str]) -> dict[str, list[str]]:
    buckets = {
        "flow": [],
        "appearance": [],
        "flow_mmd": [],
        "appearance_mmd": [],
        "other_mmd": [],
        "other": [],
    }
    for p in [str(x) for x in predictors]:
        fam = _predictor_family_for_name(p)
        if fam not in buckets:
            fam = "other"
        if p not in buckets[fam]:
            buckets[fam].append(p)
    return buckets


def _first_n_from_family(
    buckets: dict[str, list[str]],
    family: str,
    n: int,
) -> list[str]:
    n_int = int(n)
    if n_int <= 0:
        return []
    return list(buckets.get(family, [])[ :n_int])


def _predictor_grid_default_specs(buckets: dict[str, list[str]]) -> list[str]:
    n_flow = len(buckets.get("flow", []))
    n_app = len(buckets.get("appearance", []))
    specs: list[str] = []
    seen: set[str] = set()
    candidates = [
        "flow_only_f1",
        "flow_only_f2",
        "appearance_only_a1",
        "appearance_only_a2",
        "hybrid_f1_a1",
        "hybrid_f2_a2",
        "full",
    ]
    for tok in candidates:
        if tok in seen:
            continue
        if tok in {"flow_only_f1", "flow_only_f2"} and n_flow < int(tok[-1]):
            continue
        if tok in {"appearance_only_a1", "appearance_only_a2"} and n_app < int(tok[-1]):
            continue
        if tok == "hybrid_f1_a1" and (n_flow < 1 or n_app < 1):
            continue
        if tok == "hybrid_f2_a2" and (n_flow < 2 or n_app < 2):
            continue
        specs.append(tok)
        seen.add(tok)
    if not specs:
        specs.append("full")
    return specs


def _resolve_predictor_grid_specs(
    raw_specs: list[str],
    predictors: list[str],
) -> tuple[list[dict[str, object]], list[str]]:
    buckets = _predictor_family_buckets(predictors)
    if not raw_specs:
        raw_specs = _predictor_grid_default_specs(buckets)
    out: list[dict[str, object]] = []
    warnings: list[str] = []
    seen: set[str] = set()

    def _subset_from_token(token: str) -> list[str]:
        t = str(token or "").strip().lower()
        if t == "full":
            return list(dict.fromkeys([str(p) for p in predictors]))
        m = re.fullmatch(r"flow_only_f(\d+)", t)
        if m:
            k = int(m.group(1))
            return _first_n_from_family(buckets, "flow", k)
        m = re.fullmatch(r"appearance_only_a(\d+)", t)
        if m:
            k = int(m.group(1))
            return _first_n_from_family(buckets, "appearance", k)
        m = re.fullmatch(r"hybrid_f(\d+)_a(\d+)", t)
        if m:
            n_flow = int(m.group(1))
            n_app = int(m.group(2))
            return _first_n_from_family(buckets, "flow", n_flow) + _first_n_from_family(
                buckets, "appearance", n_app
            )
        return []

    def _label_from_token(token: str, selected: list[str]) -> str:
        t = str(token or "").strip().lower()
        if t == "full":
            return "All signals"
        m = re.fullmatch(r"flow_only_f(\d+)", t)
        if m:
            return f"Flow-only ({int(m.group(1))})"
        m = re.fullmatch(r"appearance_only_a(\d+)", t)
        if m:
            return f"Appearance-only ({int(m.group(1))})"
        m = re.fullmatch(r"hybrid_f(\d+)_a(\d+)", t)
        if m:
            return f"Hybrid ({int(m.group(1))}F/{int(m.group(2))}A)"
        return token

    for token in raw_specs:
        token = str(token or "").strip()
        if not token or token in seen:
            continue
        subset = _subset_from_token(token)
        if not subset:
            warnings.append(f"Skipping predictor-grid spec '{token}': no matching predictors.")
            continue
        label = _label_from_token(token, subset)
        out.append(
            {
                "spec": token,
                "label": label,
                "predictors": subset,
                "n_predictors": int(len(subset)),
            }
        )
        seen.add(token)
    if not out:
        warnings.append("No valid predictor-grid specs were available; falling back to full predictor set.")
        out = [
            {
                "spec": "full",
                "label": "All signals",
                "predictors": list(dict.fromkeys([str(p) for p in predictors])),
                "n_predictors": int(len(predictors)),
            }
        ]
    return out, warnings


def _predictor_grid_panel_family(spec: str) -> str:
    t = str(spec or "").strip().lower()
    if t == "full":
        return "full"
    if t.startswith("flow_only_"):
        return "flow"
    if t.startswith("appearance_only_"):
        return "appearance"
    if t.startswith("hybrid_"):
        return "hybrid"
    return "other"


def _organize_predictor_grid_rows_by_family(
    panel_rows: list[dict[str, object]],
    requested_cols: int,
) -> tuple[list[dict[str, object]], int, list[str]]:
    if not panel_rows:
        return [], max(int(requested_cols), 1), []

    requested_n_cols = max(int(requested_cols), 1)
    family_order = ["flow", "appearance", "hybrid", "other"]
    rows_by_family: dict[str, list[dict[str, object]]] = {family: [] for family in family_order}
    full_rows: list[dict[str, object]] = []

    for row in panel_rows:
        family = _predictor_grid_panel_family(row.get("spec", ""))
        if family == "full":
            full_rows.append(row)
            continue
        if family not in rows_by_family:
            family = "other"
        rows_by_family[family].append(row)

    active_families = [f for f in family_order if rows_by_family[f]]

    arranged: list[dict[str, object]] = []
    if active_families:
        max_family_len = max(len(rows_by_family[f]) for f in active_families)
        for ridx in range(max_family_len):
            for fam in active_families:
                if ridx < len(rows_by_family[fam]):
                    arranged.append(rows_by_family[fam][ridx])

    arranged.extend(full_rows)

    desired_cols = len(active_families) if active_families else requested_n_cols
    if desired_cols <= 0:
        desired_cols = 1
    n_cols_out = requested_n_cols
    warnings: list[str] = []
    if requested_n_cols < desired_cols:
        warnings.append(
            f"Increased predictor-grid columns from {requested_n_cols} to {desired_cols} to keep flow/appearance/hybrid family columns."
        )
        n_cols_out = desired_cols
    return arranged, n_cols_out, warnings


def _parse_predictor_grid_spec_token(token: str) -> dict[str, int | str] | None:
    t = str(token or "").strip().lower()
    if t == "full":
        return {"kind": "full"}
    m = re.fullmatch(r"flow_only_f(\d+)", t)
    if m:
        return {"kind": "flow_only", "n_flow": int(m.group(1))}
    m = re.fullmatch(r"appearance_only_a(\d+)", t)
    if m:
        return {"kind": "appearance_only", "n_app": int(m.group(1))}
    m = re.fullmatch(r"hybrid_f(\d+)_a(\d+)", t)
    if m:
        return {"kind": "hybrid", "n_flow": int(m.group(1)), "n_app": int(m.group(2))}
    return None


def _predictor_grid_score_tuple(
    rmse: float,
    pear: float,
    spear: float,
    metric: str,
) -> tuple[float, float, float]:
    m = str(metric or "pearson").strip().lower()
    if m == "spearman":
        return (float(spear), -float(rmse), float(pear))
    if m == "rmse":
        return (-float(rmse), float(pear), float(spear))
    return (float(pear), -float(rmse), float(spear))


def _maybe_select_best_predictor_subset_for_grid_spec(
    token: str,
    base_subset: list[str],
    all_predictors: list[str],
    run_dir: Path,
    target: str,
    context_cols: list[str],
    ridge_alpha: float,
    passthrough_cols: list[str],
    repeat_agg_mode: str,
    repeat_group_cols: list[str],
    context_target_transform: str,
    context_target_zscore_eps: float,
    score_metric: str,
    max_combos: int,
    source_df: pd.DataFrame | None = None,
) -> tuple[list[str], str | None]:
    parsed = _parse_predictor_grid_spec_token(token)
    if parsed is None:
        return list(base_subset), None
    kind = str(parsed.get("kind", "")).strip().lower()
    if kind in {"", "full"}:
        return list(base_subset), None

    buckets = _predictor_family_buckets([str(x) for x in all_predictors])
    flow_pool = list(buckets.get("flow", []))
    app_pool = list(buckets.get("appearance", []))
    candidates: list[list[str]] = []
    token_label = str(token or "").strip()

    if kind == "flow_only":
        n_flow = int(parsed.get("n_flow", 0))
        if n_flow <= 0 or len(flow_pool) < n_flow:
            return list(base_subset), f"{token_label}: best-subset search skipped (insufficient flow predictors)."
        n_combos = math.comb(len(flow_pool), n_flow)
        if int(max_combos) > 0 and n_combos > int(max_combos):
            return list(base_subset), (
                f"{token_label}: best-subset search skipped ({n_combos} combos > limit {int(max_combos)})."
            )
        candidates = [list(c) for c in itertools.combinations(flow_pool, n_flow)]
    elif kind == "appearance_only":
        n_app = int(parsed.get("n_app", 0))
        if n_app <= 0 or len(app_pool) < n_app:
            return list(base_subset), f"{token_label}: best-subset search skipped (insufficient appearance predictors)."
        n_combos = math.comb(len(app_pool), n_app)
        if int(max_combos) > 0 and n_combos > int(max_combos):
            return list(base_subset), (
                f"{token_label}: best-subset search skipped ({n_combos} combos > limit {int(max_combos)})."
            )
        candidates = [list(c) for c in itertools.combinations(app_pool, n_app)]
    elif kind == "hybrid":
        n_flow = int(parsed.get("n_flow", 0))
        n_app = int(parsed.get("n_app", 0))
        if n_flow <= 0 or n_app <= 0 or len(flow_pool) < n_flow or len(app_pool) < n_app:
            return list(base_subset), f"{token_label}: best-subset search skipped (insufficient hybrid predictors)."
        n_combos = math.comb(len(flow_pool), n_flow) * math.comb(len(app_pool), n_app)
        if int(max_combos) > 0 and n_combos > int(max_combos):
            return list(base_subset), (
                f"{token_label}: best-subset search skipped ({n_combos} combos > limit {int(max_combos)})."
            )
        for flow_choice in itertools.combinations(flow_pool, n_flow):
            for app_choice in itertools.combinations(app_pool, n_app):
                candidates.append(list(flow_choice) + list(app_choice))
    else:
        return list(base_subset), None

    best_subset = list(base_subset)
    best_score: tuple[float, float, float] | None = None
    best_rmse = float("inf")
    best_pear = float("nan")
    best_spear = float("nan")
    for subset in candidates:
        _, y_t, y_p, rmse_v, pear_v, spear_v, _, _ = _build_full_residual_fit(
            run_dir=run_dir,
            target=target,
            predictors=list(subset),
            context_cols=context_cols,
            ridge_alpha=ridge_alpha,
            top_k=0,
            passthrough_cols=passthrough_cols,
            repeat_agg_mode=repeat_agg_mode,
            repeat_group_cols=repeat_group_cols,
            context_target_transform=context_target_transform,
            context_target_zscore_eps=float(context_target_zscore_eps),
            source_df=source_df,
        )
        score = _predictor_grid_score_tuple(rmse=rmse_v, pear=pear_v, spear=spear_v, metric=score_metric)
        if best_score is None or score > best_score:
            best_score = score
            best_subset = list(subset)
            best_rmse = float(rmse_v)
            best_pear = float(pear_v)
            best_spear = float(spear_v)

    if best_score is None:
        return list(base_subset), f"{token_label}: best-subset search failed; using default subset."
    picked = ",".join(str(x) for x in best_subset)
    note = (
        f"{token_label}: best-subset ({str(score_metric).lower()}) picked [{picked}] "
        f"(rmse={best_rmse:.2f}, pear={best_pear:+.3f}, sr={best_spear:+.3f})."
    )
    return best_subset, note


def _prepare_plot_space_arrays(
    fit_df: pd.DataFrame,
    target: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    plot_space: str,
    context_target_transform: str,
    prediction_transform: str = "none",
    prediction_transform_eps: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str | bool]]:
    y_true_plot = np.asarray(y_true, dtype=float)
    y_pred_plot = np.asarray(y_pred, dtype=float)
    if str(plot_space).strip().lower() == "residual":
        y_true_plot = fit_df["target_resid_raw"].to_numpy(dtype=float)
        if str(context_target_transform).strip().lower() == "zscore":
            y_pred_plot = y_pred_plot * fit_df["target_ctx_std"].to_numpy(dtype=float)
    elif str(plot_space).strip().lower() == "absolute":
        y_true_plot = fit_df[target].to_numpy(dtype=float)
        if str(context_target_transform).strip().lower() == "zscore":
            y_pred_plot = fit_df["target_ctx_mean"].to_numpy(dtype=float) + (
                y_pred_plot * fit_df["target_ctx_std"].to_numpy(dtype=float)
            )
        else:
            y_pred_plot = fit_df["target_ctx_mean"].to_numpy(dtype=float) + y_pred_plot
    elif str(plot_space).strip().lower() == "zscore":
        ctx_std = fit_df["target_ctx_std"].to_numpy(dtype=float)
        ctx_std_safe = np.asarray(ctx_std, dtype=float).copy()
        ctx_std_safe[~np.isfinite(ctx_std_safe) | (ctx_std_safe <= 0.0)] = 1.0
        if str(context_target_transform).strip().lower() == "zscore":
            y_true_plot = fit_df["target_resid"].to_numpy(dtype=float)
        else:
            y_true_plot = fit_df["target_resid_raw"].to_numpy(dtype=float) / ctx_std_safe
            y_pred_plot = y_pred_plot / ctx_std_safe

    y_pred_plot_t, transform_info = _transform_prediction(
        y_pred_plot,
        mode=prediction_transform,
        eps=prediction_transform_eps,
    )
    return y_true_plot, y_pred_plot_t, transform_info


def _adjust_color_saturation(color: tuple[float, float, float, float], saturation: float) -> tuple[float, float, float, float]:
    r, g, b, a = mcolors.to_rgba(color)
    factor = float(saturation)
    if not np.isfinite(factor):
        factor = 1.0
    factor = max(factor, 0.0)
    if abs(factor - 1.0) <= 1e-6:
        return (r, g, b, a)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s = min(max(s * factor, 0.0), 1.0)
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return (r2, g2, b2, a)


def _category_color_map(categories: list[str], saturation: float = 1.0) -> dict[str, tuple[float, float, float, float]]:
    n = len(categories)
    if n <= 20:
        cmap = plt.get_cmap("tab20")
        vals = [cmap(i) for i in range(n)]
    else:
        cmap = plt.get_cmap("gist_ncar")
        vals = [cmap(i / max(n - 1, 1)) for i in range(n)]
    vals = [_adjust_color_saturation(v, saturation) for v in vals]
    return {cat: vals[i] for i, cat in enumerate(categories)}


def _category_marker_map(categories: list[str]) -> dict[str, str]:
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "8", "*"]
    out: dict[str, str] = {}
    for i, cat in enumerate(categories):
        out[cat] = marker_cycle[i % len(marker_cycle)]
    return out


_BASE_DATASET_DISPLAY_NAMES: dict[str, str] = {
    "flyingthings": "FlyingThings",
    "pointodyssey": "PointOdyssey",
    "spair": "SPair",
    "sintel": "Sintel",
    "imagenet2dwarp": "ImageNet 2D Warp",
}

_BASE_DATASET_SHORT_DISPLAY_NAMES: dict[str, str] = {
    "flyingthings": "FlyThings",
    "pointodyssey": "PointOdy",
    "spair": "SPair",
    "sintel": "Sintel",
    "imagenet2dwarp": "ImageNet-2DWarp",
}

_SYNTHETIC_VARIANT_DISPLAY_NAMES: dict[str, str] = {
    "2d_warp": "2D Warp",
    "small_zoom": "Small Zoom",
    "large_zoom": "Large Zoom",
    "random_flipping": "Random Flipping",
}


def _titleize_identifier(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw
    parts = re.split(r"[_\s]+", raw)
    token_map = {
        "2d": "2D",
        "3d": "3D",
        "sdf": "SDF",
        "mmd": "MMD",
    }
    out: list[str] = []
    for p in parts:
        if not p:
            continue
        low = p.lower()
        if low in token_map:
            out.append(token_map[low])
        elif p.isupper():
            out.append(p)
        else:
            out.append(p[:1].upper() + p[1:])
    return " ".join(out)


def _pretty_column_label(name: str | None) -> str:
    key = str(name or "").strip()
    if not key:
        return "group"
    if key == "train_dataset":
        return "Train Dataset"
    if key == "benchmark":
        return "Evaluation"
    return _titleize_identifier(key)


def _short_synthetic_label(synthetic_label: str) -> str:
    text = str(synthetic_label or "").strip()
    if not text:
        return text
    if text.lower().startswith("sdf"):
        return "SDF"
    return text


def _render_parenthetical(base: str, detail: str, wrap: bool) -> str:
    if not str(detail).strip():
        return base
    if wrap:
        return f"{base}\n({detail})"
    return f"{base} ({detail})"


def _wrap_label_two_lines(text: str, min_chars: int = 14) -> str:
    raw = str(text or "").strip()
    if not raw or "\n" in raw or len(raw) < int(min_chars):
        return raw
    split_chars = [" ", "-", "/"]
    mid = len(raw) / 2.0
    best_idx = -1
    best_cost = float("inf")
    for i, ch in enumerate(raw):
        if ch not in split_chars:
            continue
        # Avoid splitting on very early/late separators.
        if i < 3 or i > len(raw) - 4:
            continue
        cost = abs(i - mid)
        if cost < best_cost:
            best_idx = i
            best_cost = cost
    if best_idx < 0:
        return raw
    if raw[best_idx] == " ":
        return raw[:best_idx] + "\n" + raw[best_idx + 1 :]
    return raw[: best_idx + 1] + "\n" + raw[best_idx + 1 :]


def _format_train_dataset_label(raw_value: str, synthetic_label: str, mix_label_style: str = "full") -> str:
    raw = str(raw_value or "").strip()
    if not raw:
        return raw
    style = str(mix_label_style or "full").strip().lower()
    use_short = style in {"short", "short_wrap"}
    use_wrap = style == "short_wrap"
    synthetic_base = _short_synthetic_label(synthetic_label) if use_short else synthetic_label
    if raw == "synthetic":
        return _wrap_label_two_lines(synthetic_base) if use_wrap else synthetic_base
    if raw.startswith("synthetic_"):
        suffix = raw[len("synthetic_") :]
        suffix_label = _SYNTHETIC_VARIANT_DISPLAY_NAMES.get(suffix, _titleize_identifier(suffix))
        out = _render_parenthetical(str(synthetic_base), str(suffix_label), use_wrap)
        return _wrap_label_two_lines(out) if use_wrap else out
    m = re.fullmatch(r"([a-z0-9]+)_synthetic_(\d{1,3})_(\d{1,3})", raw)
    if m:
        base_raw = m.group(1)
        pct_a = int(m.group(2))
        pct_b = int(m.group(3))
        if use_short:
            base_label = _BASE_DATASET_SHORT_DISPLAY_NAMES.get(base_raw, _BASE_DATASET_DISPLAY_NAMES.get(base_raw, _titleize_identifier(base_raw)))
            detail = f"{pct_a}/{pct_b}%"
        else:
            base_label = _BASE_DATASET_DISPLAY_NAMES.get(base_raw, _titleize_identifier(base_raw))
            detail = f"{pct_a}%/{pct_b}% mix"
        out = _render_parenthetical(f"{base_label}/{synthetic_base}", detail, use_wrap)
        return _wrap_label_two_lines(out) if use_wrap else out
    if raw in _BASE_DATASET_DISPLAY_NAMES:
        if use_short:
            out = _BASE_DATASET_SHORT_DISPLAY_NAMES.get(raw, _BASE_DATASET_DISPLAY_NAMES[raw])
        else:
            out = _BASE_DATASET_DISPLAY_NAMES[raw]
        return _wrap_label_two_lines(out) if use_wrap else out
    out = _titleize_identifier(raw)
    return _wrap_label_two_lines(out) if use_wrap else out


_BENCHMARK_DISPLAY_NAMES: dict[str, str] = {
    "flyingthings": "FlyingThings",
    "kitti2012": "KITTI-2012",
    "kitti2015": "KITTI-2015",
    "pointodyssey": "PointOdyssey",
    "middlebury": "Middlebury",
    "pfpascal": "PF-PASCAL",
    "pfwillow": "PF-WILLOW",
    "spair": "SPair",
    "tss": "TSS",
    "sintel": "Sintel",
    "imagenet2dwarp": "ImageNet 2D Warp",
}

_BENCHMARK_SHORT_DISPLAY_NAMES: dict[str, str] = {
    "flyingthings": "FlyThings",
    "kitti2012": "KITTI-2012",
    "kitti2015": "KITTI-2015",
    "pointodyssey": "PointOdy",
    "middlebury": "Middlebury",
    "pfpascal": "PF-PASCAL",
    "pfwillow": "PF-WILLOW",
    "spair": "SPair",
    "tss": "TSS",
    "sintel": "Sintel",
    "imagenet2dwarp": "ImageNet-2DWarp",
}


def _normalize_key(raw: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(raw or "").strip().lower())


def _format_benchmark_label(raw_value: str, synthetic_label: str, mix_label_style: str = "full") -> str:
    raw = str(raw_value or "").strip()
    if not raw:
        return raw
    style = str(mix_label_style or "full").strip().lower()
    use_short = style in {"short", "short_wrap"}
    use_wrap = style == "short_wrap"
    if raw == "synthetic" or raw.startswith("synthetic_"):
        return _format_train_dataset_label(raw, synthetic_label, mix_label_style=mix_label_style)
    key = _normalize_key(raw)
    if use_short and key in _BENCHMARK_SHORT_DISPLAY_NAMES:
        out = _BENCHMARK_SHORT_DISPLAY_NAMES[key]
        return _wrap_label_two_lines(out) if use_wrap else out
    if key in _BENCHMARK_DISPLAY_NAMES:
        out = _BENCHMARK_DISPLAY_NAMES[key]
        return _wrap_label_two_lines(out) if use_wrap else out
    out = _titleize_identifier(raw)
    return _wrap_label_two_lines(out) if use_wrap else out


def _display_group_values(
    values: pd.Series,
    label_name: str | None,
    pretty_dataset_labels: bool = False,
    synthetic_label: str = "SDF-Fractal3D",
    mix_label_style: str = "full",
) -> pd.Series:
    out = values.astype(str)
    if not pretty_dataset_labels:
        return out
    if str(label_name or "").strip() == "train_dataset":
        return out.map(lambda v: _format_train_dataset_label(v, synthetic_label, mix_label_style=mix_label_style))
    if str(label_name or "").strip() == "benchmark":
        return out.map(lambda v: _format_benchmark_label(v, synthetic_label, mix_label_style=mix_label_style))
    return out


def _compute_axis_limits(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pad_frac: float = 0.05,
    clip_quantile: float = 0.0,
) -> tuple[float, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    vals = np.concatenate([yt, yp], axis=0)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0, 1.0

    lo = float(np.min(vals))
    hi = float(np.max(vals))
    q = float(clip_quantile)
    if 0.0 < q < 0.5 and vals.size >= 8:
        lo_q = float(np.quantile(vals, q))
        hi_q = float(np.quantile(vals, 1.0 - q))
        if np.isfinite(lo_q) and np.isfinite(hi_q) and hi_q > lo_q:
            lo = lo_q
            hi = hi_q

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        center = float(np.nanmean(vals)) if vals.size else 0.0
        return center - 1.0, center + 1.0

    pad = float(pad_frac) * (hi - lo if hi > lo else 1.0)
    return lo - pad, hi + pad


def _compute_axis_limits_1d(
    values: np.ndarray,
    pad_frac: float = 0.05,
    clip_quantile: float = 0.0,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return -1.0, 1.0
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    q = float(clip_quantile)
    if 0.0 < q < 0.5 and arr.size >= 8:
        lo_q = float(np.quantile(arr, q))
        hi_q = float(np.quantile(arr, 1.0 - q))
        if np.isfinite(lo_q) and np.isfinite(hi_q) and hi_q > lo_q:
            lo = lo_q
            hi = hi_q
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        center = float(np.nanmean(arr)) if arr.size else 0.0
        return center - 1.0, center + 1.0
    pad = float(pad_frac) * (hi - lo if hi > lo else 1.0)
    return lo - pad, hi + pad


def _center_axis_limits_zero(lo: float, hi: float) -> tuple[float, float]:
    m = float(max(abs(float(lo)), abs(float(hi))))
    if not np.isfinite(m) or m <= 0.0:
        m = 1.0
    return -m, m


def _save_figure(
    fig: plt.Figure,
    out_path: Path,
    dpi: int = 220,
    tight_bbox: bool = False,
    bbox_extra_artists: list[object] | None = None,
) -> None:
    if tight_bbox:
        fig.savefig(
            out_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.03,
            bbox_extra_artists=list(bbox_extra_artists or []),
        )
    else:
        fig.savefig(out_path, dpi=dpi)


def _heldout_protocol_cv_targets(protocol: str) -> list[str]:
    p = str(protocol or "").strip().lower()
    if p == "model":
        return ["model_only"]
    if p == "training":
        return ["model_train_benchmark"]
    if p == "eval":
        return ["model_benchmark"]
    if p == "triple":
        return ["model_train_benchmark_disjoint"]
    if p == "trainset_disjoint":
        return ["model_benchmark_trainset_disjoint"]
    return [p]


def _is_model_cv_protocol(protocol: str) -> bool:
    return str(protocol or "").strip().lower() in {
        "model",
        "training",
        "eval",
        "triple",
        "trainset_disjoint",
    }


def _normalize_model_cv_metric_name(metric_key: str | None) -> str:
    mk = str(metric_key or "").strip().lower()
    if mk.startswith("heldout_model_cv_"):
        mk = mk[len("heldout_model_cv_") :]
    return mk


def _metric_is_higher_better(metric_name: str) -> bool:
    name = str(metric_name or "").strip().lower()
    higher_better_tokens = [
        "pearson",
        "spearman",
        "kendall",
        "cindex",
        "top",
        "win",
        "auc",
        "r2",
    ]
    lower_better_tokens = [
        "mae",
        "rmse",
        "mse",
        "abs_err",
        "pct_err",
        "regret",
        "loss",
        "error",
    ]
    if any(tok in name for tok in higher_better_tokens):
        return True
    if any(tok in name for tok in lower_better_tokens):
        return False
    return False


def _candidate_dispersion_ratio(
    df: pd.DataFrame,
    true_col: str,
    pred_col: str,
) -> float:
    if true_col not in df.columns or pred_col not in df.columns:
        return float("nan")
    yt = pd.to_numeric(df[true_col], errors="coerce").to_numpy(dtype=float)
    yp = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[valid]
    yp = yp[valid]
    if len(yt) < 2:
        return float("nan")
    ystd = float(np.std(yt, ddof=0))
    pstd = float(np.std(yp, ddof=0))
    if not np.isfinite(ystd) or ystd <= 0.0 or not np.isfinite(pstd):
        return float("nan")
    return float(pstd / ystd)


def _select_best_candidate_from_heldout_model_cv_rankfirst(
    heldout_model_cv_dir: Path,
    cv_protocols: list[str],
    head: str,
    min_dispersion_ratio: float = RANKFIRST_MIN_DISPERSION_RATIO,
) -> tuple[str | None, str | None, float | None, str | None]:
    summary_path = heldout_model_cv_dir / "heldout_model_cv_summary.csv"
    pred_rows_path = heldout_model_cv_dir / "heldout_model_cv_pred_rows.csv"
    if not summary_path.exists():
        return None, None, None, "heldout_model_cv_summary.csv missing"
    if not pred_rows_path.exists():
        return None, None, None, "heldout_model_cv_pred_rows.csv missing"
    try:
        summary_df = pd.read_csv(summary_path)
        pred_df = pd.read_csv(pred_rows_path)
    except Exception as ex:
        return None, None, None, f"failed to read heldout_model_cv CSVs: {ex}"
    if summary_df.empty or pred_df.empty:
        return None, None, None, "heldout_model_cv summary/pred rows empty"

    def _apply_common_filters(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "protocol" in out.columns:
            out = out[out["protocol"].astype(str).isin(cv_protocols)].copy()
        head_key = str(head or "").strip()
        if head_key and "head" in out.columns:
            out = out[out["head"].astype(str) == head_key].copy()
        if "status" in out.columns:
            ok = out["status"].astype(str).str.lower() == "ok"
            if ok.any():
                out = out[ok].copy()
        return out

    work = _apply_common_filters(summary_df)
    if work.empty:
        return None, None, None, f"no summary rows for protocols={','.join(cv_protocols)}"
    required_cols = ["candidate_id", "rank_spearman", "rank_pairwise_cindex", "rank_regret"]
    missing = [c for c in required_cols if c not in work.columns]
    if missing:
        return None, None, None, f"missing summary columns for rank-first: {','.join(missing)}"

    cand_summary = (
        work.groupby("candidate_id", dropna=False)[["rank_spearman", "rank_pairwise_cindex", "rank_regret"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    if cand_summary.empty:
        return None, None, None, "no candidate rows after rank aggregation"

    pred_work = _apply_common_filters(pred_df)
    if pred_work.empty:
        return None, None, None, "no pred rows after protocol/head filtering"
    if "candidate_id" not in pred_work.columns:
        return None, None, None, "candidate_id column missing from pred rows"

    # Prefer residual heldout columns for model-only visualization selection.
    if "target_eval" in pred_work.columns and "prediction" in pred_work.columns:
        true_col, pred_col = "target_eval", "prediction"
    elif "target_model_space" in pred_work.columns and "prediction_model_space" in pred_work.columns:
        true_col, pred_col = "target_model_space", "prediction_model_space"
    elif "target_absolute" in pred_work.columns and "prediction_absolute" in pred_work.columns:
        true_col, pred_col = "target_absolute", "prediction_absolute"
    else:
        true_col, pred_col = _extract_true_pred_columns(pred_work)
    if true_col is None or pred_col is None:
        return None, None, None, "unable to identify true/pred columns for pred rows"

    ratio_rows: list[dict[str, object]] = []
    for cid, g in pred_work.groupby("candidate_id", dropna=False):
        ratio = _candidate_dispersion_ratio(g, true_col=true_col, pred_col=pred_col)
        ratio_rows.append({"candidate_id": str(cid), "dispersion_ratio": float(ratio)})
    ratio_df = pd.DataFrame(ratio_rows)
    if ratio_df.empty:
        return None, None, None, "no candidate-level dispersion ratios available"

    cand_summary["candidate_id"] = cand_summary["candidate_id"].astype(str)
    score_df = cand_summary.merge(ratio_df, on="candidate_id", how="left")
    score_df = score_df[np.isfinite(pd.to_numeric(score_df["dispersion_ratio"], errors="coerce"))].copy()
    score_df = score_df[
        pd.to_numeric(score_df["dispersion_ratio"], errors="coerce") >= float(min_dispersion_ratio)
    ].copy()
    if score_df.empty:
        return None, None, None, (
            f"no candidates passed dispersion filter pred_std/target_std>={float(min_dispersion_ratio):g}"
        )

    for col in ["rank_spearman", "rank_pairwise_cindex", "rank_regret"]:
        score_df[col] = pd.to_numeric(score_df[col], errors="coerce")
    score_df["rank_primary_missing"] = ~np.isfinite(score_df["rank_spearman"].to_numpy(dtype=float))
    score_df["sort_1"] = score_df["rank_primary_missing"].astype(int)
    score_df["sort_2"] = score_df["rank_spearman"].where(np.isfinite(score_df["rank_spearman"]), -np.inf)
    score_df["sort_3"] = score_df["rank_pairwise_cindex"].where(
        np.isfinite(score_df["rank_pairwise_cindex"]), -np.inf
    )
    score_df["sort_4"] = score_df["rank_regret"].where(np.isfinite(score_df["rank_regret"]), np.inf)
    score_df = score_df.sort_values(
        by=["sort_1", "sort_2", "sort_3", "sort_4", "candidate_id"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)
    if score_df.empty:
        return None, None, None, "no candidates after rank-first sorting"
    best = score_df.iloc[0]
    return (
        str(best["candidate_id"]),
        "rank_spearman",
        float(best["rank_spearman"]) if np.isfinite(float(best["rank_spearman"])) else float("nan"),
        None,
    )


def _select_best_candidate_from_heldout_model_cv_summary(
    heldout_model_cv_dir: Path,
    cv_protocols: list[str],
    head: str,
    metric_key: str | None,
    min_dispersion_ratio: float = RANKFIRST_MIN_DISPERSION_RATIO,
) -> tuple[str | None, str | None, float | None, str | None]:
    metric_key_raw = str(metric_key or "").strip().lower()
    metric_key_norm = _normalize_model_cv_metric_name(metric_key)
    if metric_key_raw in {HELDOUT_MODEL_CV_RANKFIRST_KEY.lower(), "rankfirst_antidegen"} or metric_key_norm in {
        "rankfirst_antidegen",
    }:
        return _select_best_candidate_from_heldout_model_cv_rankfirst(
            heldout_model_cv_dir=heldout_model_cv_dir,
            cv_protocols=cv_protocols,
            head=head,
            min_dispersion_ratio=float(min_dispersion_ratio),
        )

    summary_path = heldout_model_cv_dir / "heldout_model_cv_summary.csv"
    if not summary_path.exists():
        return None, None, None, "heldout_model_cv_summary.csv missing"
    try:
        summary_df = pd.read_csv(summary_path)
    except Exception as ex:
        return None, None, None, f"failed to read heldout_model_cv_summary.csv: {ex}"
    if summary_df.empty:
        return None, None, None, "heldout_model_cv_summary.csv empty"

    work = summary_df.copy()
    if "protocol" in work.columns:
        work = work[work["protocol"].astype(str).isin(cv_protocols)].copy()
    if work.empty:
        return None, None, None, f"no rows for protocols={','.join(cv_protocols)}"

    head_key = str(head or "").strip()
    if head_key and "head" in work.columns:
        work = work[work["head"].astype(str) == head_key].copy()
    if work.empty:
        return None, None, None, f"no rows for head={head_key}"

    if "status" in work.columns:
        ok = work["status"].astype(str).str.lower() == "ok"
        if ok.any():
            work = work[ok].copy()
    if work.empty:
        return None, None, None, "no status=ok rows"

    metric_name = _normalize_model_cv_metric_name(metric_key)
    if not metric_name:
        metric_name = "mae"
    if metric_name not in work.columns:
        return None, metric_name, None, f"metric column '{metric_name}' missing"

    work[metric_name] = pd.to_numeric(work[metric_name], errors="coerce")
    work = work[np.isfinite(work[metric_name].to_numpy(dtype=float))].copy()
    if work.empty:
        return None, metric_name, None, f"metric column '{metric_name}' has no finite values"

    if "candidate_id" not in work.columns:
        return None, metric_name, None, "candidate_id column missing"

    cand_scores = (
        work.groupby("candidate_id", dropna=False)[metric_name]
        .mean()
        .reset_index()
        .sort_values(metric_name, ascending=not _metric_is_higher_better(metric_name))
        .reset_index(drop=True)
    )
    if cand_scores.empty:
        return None, metric_name, None, "no candidate scores after aggregation"

    best_row = cand_scores.iloc[0]
    return str(best_row["candidate_id"]), metric_name, float(best_row[metric_name]), None


def _heldout_protocol_detail_files(protocol: str) -> list[str]:
    p = str(protocol or "").strip().lower()
    if p == "lobo":
        return [
            "prediction_lobo_rows.csv",
            "prediction_lobo_holdout_placement_detail.csv",
            "prediction_lobo_rank_detail.csv",
        ]
    if p == "loto":
        return [
            "prediction_loto_rows.csv",
            "prediction_loto_holdout_placement_detail.csv",
            "prediction_loto_rank_detail.csv",
        ]
    if p == "lomo":
        return [
            "prediction_lomo_rows.csv",
            "prediction_lomo_holdout_placement_detail.csv",
            "prediction_lomo_rank_detail.csv",
        ]
    if p == "jointood":
        return [
            "prediction_jointood_rows.csv",
            "prediction_jointood_holdout_placement_detail.csv",
            "prediction_jointood_rank_detail.csv",
        ]
    if p == "model":
        return [
            "prediction_model_holdout_placement_detail.csv",
            "prediction_model_holdout_rank_detail.csv",
            "prediction_model_holdout_rank_error_detail.csv",
        ]
    if p == "training":
        return [
            "prediction_training_holdout_placement_detail.csv",
            "prediction_train_holdout_placement_detail.csv",
            "prediction_trainset_holdout_placement_detail.csv",
            "prediction_training_holdout_rank_detail.csv",
            "prediction_train_holdout_rank_detail.csv",
        ]
    if p == "eval":
        return [
            "prediction_eval_holdout_placement_detail.csv",
            "prediction_benchmark_holdout_placement_detail.csv",
            "prediction_eval_holdout_rank_detail.csv",
            "prediction_benchmark_holdout_rank_detail.csv",
        ]
    return [
        f"prediction_{p}_holdout_placement_detail.csv",
        f"prediction_{p}_rank_detail.csv",
    ]


def _extract_true_pred_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    true_cols = [
        "heldout_true",
        "true_mean",
        "target_eval",
        "target_absolute",
        "target_model_space",
        "target",
        "truth",
        "y_true",
        "y",
        "auc_normalized_observed",
    ]
    pred_cols = [
        "heldout_pred",
        "prediction_absolute",
        "prediction_model_space",
        "prediction",
        "target_pred",
        "pred_mean",
        "pred",
        "y_pred",
        "yhat",
    ]
    true_col = next((c for c in true_cols if c in df.columns), None)
    pred_col = next((c for c in pred_cols if c in df.columns), None)
    return true_col, pred_col


def _load_protocol_points_from_detail_file(detail_path: Path, protocol: str) -> pd.DataFrame:
    df = pd.read_csv(detail_path)
    if df.empty:
        return pd.DataFrame()
    true_col, pred_col = _extract_true_pred_columns(df)
    if true_col is None or pred_col is None:
        return pd.DataFrame()
    out = df.copy()
    out["y_true"] = pd.to_numeric(out[true_col], errors="coerce")
    out["y_pred"] = pd.to_numeric(out[pred_col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_true", "y_pred"]).copy()
    if out.empty:
        return pd.DataFrame()
    out["_protocol"] = str(protocol)
    out["_source_file"] = str(detail_path.name)
    out["_y_true_col"] = str(true_col)
    out["_y_pred_col"] = str(pred_col)
    return out


def _load_protocol_points_from_heldout_model_cv(
    heldout_model_cv_dir: Path,
    protocol: str,
    head: str,
    metric_key: str | None = None,
    rankfirst_min_dispersion_ratio: float = RANKFIRST_MIN_DISPERSION_RATIO,
) -> pd.DataFrame:
    pred_rows_path = heldout_model_cv_dir / "heldout_model_cv_pred_rows.csv"
    if not pred_rows_path.exists():
        return pd.DataFrame()

    try:
        pred_df = pd.read_csv(pred_rows_path)
    except Exception:
        return pd.DataFrame()
    if pred_df.empty:
        return pd.DataFrame()

    cv_protocols = _heldout_protocol_cv_targets(protocol)
    if "protocol" in pred_df.columns:
        pred_df = pred_df[pred_df["protocol"].astype(str).isin(cv_protocols)].copy()
    if pred_df.empty:
        return pd.DataFrame()

    head_key = str(head or "").strip()
    if head_key and "head" in pred_df.columns:
        pred_df = pred_df[pred_df["head"].astype(str) == head_key].copy()
    if pred_df.empty:
        return pd.DataFrame()

    best_candidate_id = None
    selected_metric_col = None
    selected_metric_value = None
    selection_note = None

    metric_key_norm = str(metric_key or "").strip()
    if metric_key_norm:
        best_candidate_id, selected_metric_col, selected_metric_value, selection_note = (
            _select_best_candidate_from_heldout_model_cv_summary(
                heldout_model_cv_dir=heldout_model_cv_dir,
                cv_protocols=cv_protocols,
                head=head_key,
                metric_key=metric_key_norm,
                min_dispersion_ratio=float(rankfirst_min_dispersion_ratio),
            )
        )

    # Fallback to precomputed best table when summary-based selection is unavailable.
    if best_candidate_id is None:
        best_path = heldout_model_cv_dir / "heldout_model_cv_best_by_protocol_head.csv"
        if best_path.exists():
            try:
                best_df = pd.read_csv(best_path)
                if "protocol" in best_df.columns:
                    best_df = best_df[best_df["protocol"].astype(str).isin(cv_protocols)].copy()
                if head_key and "head" in best_df.columns:
                    best_df = best_df[best_df["head"].astype(str) == head_key].copy()
                if not best_df.empty and "candidate_id" in best_df.columns:
                    best_candidate_id = str(best_df.iloc[0]["candidate_id"])
            except Exception:
                best_candidate_id = None

    if best_candidate_id is not None and "candidate_id" in pred_df.columns:
        chosen = pred_df[pred_df["candidate_id"].astype(str) == best_candidate_id].copy()
        if not chosen.empty:
            pred_df = chosen

    if "target_absolute" in pred_df.columns and "prediction_absolute" in pred_df.columns:
        true_col, pred_col = "target_absolute", "prediction_absolute"
    elif "target_model_space" in pred_df.columns and "prediction_model_space" in pred_df.columns:
        true_col, pred_col = "target_model_space", "prediction_model_space"
    elif "target_eval" in pred_df.columns and "prediction" in pred_df.columns:
        true_col, pred_col = "target_eval", "prediction"
    else:
        true_col, pred_col = _extract_true_pred_columns(pred_df)
    if true_col is None or pred_col is None:
        return pd.DataFrame()
    pred_df["y_true"] = pd.to_numeric(pred_df[true_col], errors="coerce")
    pred_df["y_pred"] = pd.to_numeric(pred_df[pred_col], errors="coerce")
    pred_df = pred_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_true", "y_pred"]).copy()
    if pred_df.empty:
        return pd.DataFrame()
    pred_df["_protocol"] = str(protocol)
    pred_df["_source_file"] = "heldout_model_cv_pred_rows.csv"
    pred_df["_y_true_col"] = str(true_col)
    pred_df["_y_pred_col"] = str(pred_col)
    pred_df["_cv_protocols"] = ",".join(cv_protocols)
    if best_candidate_id is not None:
        pred_df["_cv_candidate_id"] = best_candidate_id
    if selected_metric_col:
        pred_df["_cv_selected_metric"] = str(selected_metric_col)
    if selected_metric_value is not None and np.isfinite(float(selected_metric_value)):
        pred_df["_cv_selected_metric_value"] = float(selected_metric_value)
    if selection_note:
        pred_df["_cv_selection_note"] = str(selection_note)
    return pred_df


def _load_heldout_protocol_points(
    run_dir: Path,
    protocol: str,
    heldout_model_cv_dir: Path | None = None,
    heldout_model_cv_head: str = "ridge",
    heldout_model_cv_metric_key: str | None = None,
    rankfirst_min_dispersion_ratio: float = RANKFIRST_MIN_DISPERSION_RATIO,
) -> tuple[pd.DataFrame, str | None]:
    for fname in _heldout_protocol_detail_files(protocol):
        path = run_dir / fname
        if not path.exists():
            continue
        try:
            pts = _load_protocol_points_from_detail_file(path, protocol=protocol)
        except Exception:
            pts = pd.DataFrame()
        if not pts.empty:
            return pts, fname

    if heldout_model_cv_dir is not None and str(protocol).strip().lower() in {
        "model",
        "training",
        "eval",
        "triple",
        "trainset_disjoint",
    }:
        pts = _load_protocol_points_from_heldout_model_cv(
            heldout_model_cv_dir=heldout_model_cv_dir,
            protocol=protocol,
            head=heldout_model_cv_head,
            metric_key=heldout_model_cv_metric_key,
            rankfirst_min_dispersion_ratio=float(rankfirst_min_dispersion_ratio),
        )
        if not pts.empty:
            return pts, f"{heldout_model_cv_dir}/heldout_model_cv_pred_rows.csv"

    return pd.DataFrame(), None


def _load_context_stats_for_run(
    run_dir: Path,
    target_col: str,
    context_cols: list[str],
    cache: dict[tuple[str, str, str], pd.DataFrame],
) -> pd.DataFrame:
    ctx_key = (str(run_dir), str(target_col), ",".join(context_cols))
    if ctx_key in cache:
        return cache[ctx_key]
    auc_path = run_dir / "auc_with_features.csv"
    if not auc_path.exists():
        cache[ctx_key] = pd.DataFrame()
        return cache[ctx_key]
    try:
        header = pd.read_csv(auc_path, nrows=0)
    except Exception:
        cache[ctx_key] = pd.DataFrame()
        return cache[ctx_key]
    present_ctx = [c for c in context_cols if c in header.columns]
    if target_col not in header.columns or not present_ctx:
        cache[ctx_key] = pd.DataFrame()
        return cache[ctx_key]
    try:
        df = pd.read_csv(auc_path, usecols=present_ctx + [target_col])
    except Exception:
        cache[ctx_key] = pd.DataFrame()
        return cache[ctx_key]
    if df.empty:
        cache[ctx_key] = pd.DataFrame()
        return cache[ctx_key]
    grp = df.groupby(present_ctx, dropna=False)[target_col]
    stats = grp.agg(
        _ctx_mean="mean",
        _ctx_std=lambda s: float(np.std(pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float), ddof=0)),
    ).reset_index()
    stats["_ctx_std"] = pd.to_numeric(stats["_ctx_std"], errors="coerce")
    bad = ~np.isfinite(stats["_ctx_std"].to_numpy(dtype=float)) | (stats["_ctx_std"].to_numpy(dtype=float) <= 0.0)
    if bad.any():
        stats.loc[bad, "_ctx_std"] = 1.0
    cache[ctx_key] = stats
    return cache[ctx_key]


def _project_heldout_points_to_space(
    pts: pd.DataFrame,
    protocol: str,
    space: str,
    protocol_run_dir: Path,
    context_stats_cache: dict[tuple[str, str, str], pd.DataFrame],
) -> tuple[pd.DataFrame, str | None]:
    if pts.empty:
        return pd.DataFrame(), "no points"
    p = str(protocol or "").strip().lower()
    s = str(space or "").strip().lower()
    out = pts.copy()

    def _finalize(df: pd.DataFrame, true_col: str, pred_col: str) -> pd.DataFrame:
        if true_col not in df.columns or pred_col not in df.columns:
            return pd.DataFrame()
        work = df.copy()
        work["y_true"] = pd.to_numeric(work[true_col], errors="coerce")
        work["y_pred"] = pd.to_numeric(work[pred_col], errors="coerce")
        work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_true", "y_pred"]).copy()
        if work.empty:
            return pd.DataFrame()
        work["_y_true_col"] = str(true_col)
        work["_y_pred_col"] = str(pred_col)
        work["_heldout_space"] = s
        return work

    # Explicit absolute/model-space columns from heldout_model_cv rows.
    if s == "absolute":
        if "target_absolute" in out.columns and "prediction_absolute" in out.columns:
            return _finalize(out, "target_absolute", "prediction_absolute"), None
    if s == "model_space":
        if "target_model_space" in out.columns and "prediction_model_space" in out.columns:
            return _finalize(out, "target_model_space", "prediction_model_space"), None
    if s == "residual":
        if "target_eval" in out.columns and "prediction" in out.columns and p in {"model", "training", "eval"}:
            return _finalize(out, "target_eval", "prediction"), None
        if "target_model_space" in out.columns and "prediction_model_space" in out.columns and p in {"model", "training", "eval"}:
            return _finalize(out, "target_model_space", "prediction_model_space"), None

    # For leakage-free heldout rows, target/prediction are model-space values.
    if "target" in out.columns and "prediction" in out.columns:
        if s == "model_space":
            return _finalize(out, "target", "prediction"), None

        meta = _load_metadata(protocol_run_dir)
        target_col = str(meta.get("target_col", "auc_normalized_observed") or "auc_normalized_observed")
        target_mode = str(meta.get("cv_residual_target_transform", "residual") or "residual").strip().lower()
        if target_mode not in {"residual", "zscore"}:
            target_mode = "residual"
        context_cols = [c for c in _resolve_context_cols(meta, None) if c in out.columns]
        abs_truth_col = next(
            (
                c
                for c in ["target_absolute", "auc_normalized", "auc", target_col]
                if c in out.columns and pd.to_numeric(out[c], errors="coerce").notna().any()
            ),
            None,
        )
        if target_mode == "zscore":
            if not context_cols:
                return pd.DataFrame(), (
                    "cannot project zscore model-space heldout rows to residual/absolute without context columns"
                )
            stats = _load_context_stats_for_run(
                run_dir=protocol_run_dir,
                target_col=target_col,
                context_cols=context_cols,
                cache=context_stats_cache,
            )
            if stats.empty:
                return pd.DataFrame(), (
                    "missing context mean/std stats for heldout residual/absolute projection"
                )
            out = out.merge(stats, on=context_cols, how="left")
            if out["_ctx_std"].isna().all() or out["_ctx_mean"].isna().all():
                return pd.DataFrame(), "failed to match heldout rows to context stats for projection"
            out["target_residual_model"] = pd.to_numeric(out["target"], errors="coerce") * pd.to_numeric(
                out["_ctx_std"], errors="coerce"
            )
            out["prediction_residual_model"] = pd.to_numeric(out["prediction"], errors="coerce") * pd.to_numeric(
                out["_ctx_std"], errors="coerce"
            )
            if abs_truth_col is not None:
                out["target_absolute_from_data"] = pd.to_numeric(out[abs_truth_col], errors="coerce")
                out["target_residual_from_data"] = pd.to_numeric(
                    out["target_absolute_from_data"], errors="coerce"
                ) - pd.to_numeric(out["_ctx_mean"], errors="coerce")
            out["target_residual"] = (
                pd.to_numeric(out["target_residual_from_data"], errors="coerce")
                if "target_residual_from_data" in out.columns
                else pd.to_numeric(out["target_residual_model"], errors="coerce")
            )
            out["prediction_residual"] = pd.to_numeric(out["prediction_residual_model"], errors="coerce")
            out["target_absolute_from_model"] = pd.to_numeric(out["_ctx_mean"], errors="coerce") + pd.to_numeric(
                out["target_residual_model"], errors="coerce"
            )
            out["prediction_absolute_from_model"] = pd.to_numeric(out["_ctx_mean"], errors="coerce") + pd.to_numeric(
                out["prediction_residual_model"], errors="coerce"
            )
        else:
            out["target_residual"] = pd.to_numeric(out["target"], errors="coerce")
            out["prediction_residual"] = pd.to_numeric(out["prediction"], errors="coerce")
            if context_cols:
                stats = _load_context_stats_for_run(
                    run_dir=protocol_run_dir,
                    target_col=target_col,
                    context_cols=context_cols,
                    cache=context_stats_cache,
                )
                if not stats.empty:
                    out = out.merge(stats, on=context_cols, how="left")
            if abs_truth_col is not None:
                out["target_absolute_from_data"] = pd.to_numeric(out[abs_truth_col], errors="coerce")
            if "_ctx_mean" in out.columns:
                out["target_absolute_from_model"] = pd.to_numeric(out["_ctx_mean"], errors="coerce") + pd.to_numeric(
                    out["target_residual"], errors="coerce"
                )
                out["prediction_absolute_from_model"] = pd.to_numeric(
                    out["_ctx_mean"], errors="coerce"
                ) + pd.to_numeric(out["prediction_residual"], errors="coerce")
                if "target_absolute_from_data" in out.columns:
                    out["target_residual"] = pd.to_numeric(out["target_absolute_from_data"], errors="coerce") - pd.to_numeric(
                        out["_ctx_mean"], errors="coerce"
                    )
            elif target_col in out.columns:
                # Recover context mean row-wise when target is residual (not zscore).
                ctx_mean_row = pd.to_numeric(out[target_col], errors="coerce") - pd.to_numeric(
                    out["target_residual"], errors="coerce"
                )
                out["target_absolute_from_model"] = pd.to_numeric(
                    out["target_absolute_from_data"] if "target_absolute_from_data" in out.columns else out[target_col],
                    errors="coerce",
                )
                out["prediction_absolute_from_model"] = ctx_mean_row + pd.to_numeric(
                    out["prediction_residual"], errors="coerce"
                )

        if s == "residual":
            return _finalize(out, "target_residual", "prediction_residual"), None
        if s == "absolute":
            true_abs_col = None
            for cand in [
                "target_absolute_from_data",
                "target_absolute",
                "auc_normalized",
                "auc",
                target_col,
                "target_absolute_from_model",
            ]:
                if cand in out.columns and pd.to_numeric(out[cand], errors="coerce").notna().any():
                    true_abs_col = cand
                    break
            if true_abs_col is None:
                true_abs_col = "target_absolute_from_model"
            return _finalize(out, true_abs_col, "prediction_absolute_from_model"), None

    # Conservative fallback: keep currently loaded pair for model-space only.
    if s == "model_space" and "y_true" in out.columns and "y_pred" in out.columns:
        work = _finalize(out, "y_true", "y_pred")
        if not work.empty:
            return work, "fell back to loaded y_true/y_pred"
    return pd.DataFrame(), f"unsupported heldout space projection '{s}' for protocol '{protocol}'"


def _plot_residual_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rmse: float,
    pear: float,
    spear: float,
    out_path: Path,
    color_values: pd.Series | None = None,
    color_label: str | None = None,
    marker_values: pd.Series | None = None,
    marker_label: str | None = None,
    centroid_values: pd.Series | None = None,
    centroid_label: str | None = None,
    ellipse_values: pd.Series | None = None,
    ellipse_label: str | None = None,
    ellipse_n_std: float = 1.25,
    ellipse_min_points: int = 3,
    ellipse_face_alpha: float = 0.10,
    ellipse_edge_alpha: float = 0.95,
    ellipse_equal_area: bool = False,
    draw_points: bool = True,
    color_map_csv: Path | None = None,
    identifier_text: str | None = None,
    x_label: str = "Predicted residual",
    y_label: str = "Observed residual",
    title_base: str = "Full-data residual fit (context-aware)",
    diagnostics: dict[str, float | int | str | bool] | None = None,
    fit_line_group_values: pd.Series | None = None,
    fit_line_group_label: str | None = None,
    fit_line_group_min_points: int = 8,
    fit_line_group_max_groups: int = 24,
    fit_line_group_show_legend: bool = False,
    axis_clip_quantile: float = 0.0,
    axis_pad_frac: float = 0.05,
    axis_match_xy_limits: bool = True,
    show_fit_line: bool = True,
    show_diagnostics: bool = True,
    show_identity_line: bool = True,
    show_title: bool = True,
    legend_ncol: int | None = None,
    show_color_in_title: bool = True,
    tight_bbox: bool = False,
    pretty_dataset_labels: bool = False,
    synthetic_label: str = "SDF-Fractal3D",
    marker_size: float = 15.0,
    point_alpha: float | None = None,
    font_scale: float = 1.0,
    legend_font_scale: float = 1.0,
    color_saturation: float = 1.0,
    mix_label_style: str = "full",
    center_axes_zero: bool = False,
):
    fig, ax = plt.subplots(figsize=(9.2, 9.2))
    resolved_marker_size = max(float(marker_size), 1.0)
    resolved_font_scale = max(float(font_scale), 0.5)
    resolved_legend_font_scale = resolved_font_scale * max(float(legend_font_scale), 0.5)
    resolved_point_alpha = None
    if point_alpha is not None:
        try:
            resolved_point_alpha = float(np.clip(float(point_alpha), 0.02, 1.0))
        except Exception:
            resolved_point_alpha = None
    try:
        resolved_ellipse_face_alpha = float(np.clip(float(ellipse_face_alpha), 0.0, 1.0))
    except Exception:
        resolved_ellipse_face_alpha = 0.10
    try:
        resolved_ellipse_edge_alpha = float(np.clip(float(ellipse_edge_alpha), 0.0, 1.0))
    except Exception:
        resolved_ellipse_edge_alpha = 0.95
    has_groups = color_values is not None and len(color_values) == len(y_true)
    has_marker_groups = marker_values is not None and len(marker_values) == len(y_true)
    has_centroid_groups = centroid_values is not None and len(centroid_values) == len(y_true)
    has_ellipse_groups = ellipse_values is not None and len(ellipse_values) == len(y_true)
    cats: list[str] = []
    marker_cats: list[str] = []
    centroid_cats: list[str] = []
    ellipse_cats: list[str] = []
    legend_title = _pretty_column_label(color_label)
    marker_legend_title = _pretty_column_label(marker_label)
    centroid_legend_title = _pretty_column_label(centroid_label)
    ellipse_legend_title = _pretty_column_label(ellipse_label)
    marker_map: dict[str, str] = {}
    centroid_map: dict[str, str] = {}
    ellipse_cmap: dict[str, tuple[float, float, float, float]] = {}
    groups = None
    marker_groups = None
    centroid_groups = None
    ellipse_groups = None
    cmap: dict[str, tuple[float, float, float, float]] = {}
    if has_groups:
        group_series = _display_group_values(
            color_values,
            label_name=color_label,
            pretty_dataset_labels=pretty_dataset_labels,
            synthetic_label=synthetic_label,
            mix_label_style=mix_label_style,
        )
        groups = group_series.to_numpy(dtype=object)
        cats = sorted(group_series.dropna().unique().tolist())
        cmap = _category_color_map(cats, saturation=color_saturation)
        if color_map_csv is not None:
            map_df = pd.DataFrame(
                {
                    "category": cats,
                    "color_hex": [mcolors.to_hex(cmap[c]) for c in cats],
                }
            )
            map_df.to_csv(color_map_csv, index=False)
    if has_marker_groups:
        marker_series = _display_group_values(
            marker_values,
            label_name=marker_label,
            pretty_dataset_labels=pretty_dataset_labels,
            synthetic_label=synthetic_label,
            mix_label_style=mix_label_style,
        )
        marker_groups = marker_series.to_numpy(dtype=object)
        marker_cats = sorted(marker_series.dropna().unique().tolist())
        marker_map = _category_marker_map(marker_cats)
    if has_centroid_groups:
        centroid_series = _display_group_values(
            centroid_values,
            label_name=centroid_label,
            pretty_dataset_labels=pretty_dataset_labels,
            synthetic_label=synthetic_label,
            mix_label_style=mix_label_style,
        )
        centroid_groups = centroid_series.to_numpy(dtype=object)
        centroid_cats = sorted(centroid_series.dropna().unique().tolist())
        centroid_map = _category_marker_map(centroid_cats)
    if has_ellipse_groups:
        ellipse_series = _display_group_values(
            ellipse_values,
            label_name=ellipse_label,
            pretty_dataset_labels=pretty_dataset_labels,
            synthetic_label=synthetic_label,
            mix_label_style=mix_label_style,
        )
        ellipse_groups = ellipse_series.to_numpy(dtype=object)
        ellipse_cats = sorted(ellipse_series.dropna().unique().tolist())
        if has_groups and str(color_label or "").strip() == str(ellipse_label or "").strip() and cmap:
            ellipse_cmap = dict(cmap)
        else:
            ellipse_cmap = _category_color_map(ellipse_cats, saturation=color_saturation)

    if draw_points:
        if has_groups and has_marker_groups and groups is not None and marker_groups is not None:
            for cat in cats:
                cat_mask = groups == cat
                for mcat in marker_cats:
                    mask = cat_mask & (marker_groups == mcat)
                    if not np.any(mask):
                        continue
                    ax.scatter(
                        y_pred[mask],
                        y_true[mask],
                        s=resolved_marker_size,
                        alpha=(resolved_point_alpha if resolved_point_alpha is not None else 0.68),
                        color=cmap[cat],
                        marker=marker_map[mcat],
                        edgecolors="none",
                    )
        elif has_groups and groups is not None:
            for cat in cats:
                mask = groups == cat
                ax.scatter(
                    y_pred[mask],
                    y_true[mask],
                    s=resolved_marker_size,
                    alpha=(resolved_point_alpha if resolved_point_alpha is not None else 0.65),
                    color=cmap[cat],
                    edgecolors="none",
                )
        elif has_marker_groups and marker_groups is not None:
            for mcat in marker_cats:
                mask = marker_groups == mcat
                ax.scatter(
                    y_pred[mask],
                    y_true[mask],
                    s=resolved_marker_size,
                    alpha=(resolved_point_alpha if resolved_point_alpha is not None else 0.55),
                    color="#1f77b4",
                    marker=marker_map[mcat],
                    edgecolors="none",
                )
        else:
            ax.scatter(
                y_pred,
                y_true,
                s=resolved_marker_size,
                alpha=(resolved_point_alpha if resolved_point_alpha is not None else 0.45),
                color="#1f77b4",
                edgecolors="none",
            )

    # Optional group ellipses to summarize cluster centers/spread (e.g., by benchmark).
    ellipse_handles: list[Line2D] = []
    ellipse_specs: list[dict[str, object]] = []
    if has_ellipse_groups and ellipse_groups is not None and ellipse_cats:
        nstd = max(float(ellipse_n_std), 0.1)
        min_pts = max(int(ellipse_min_points), 3)
        for ecat in ellipse_cats:
            mask = ellipse_groups == ecat
            if not np.any(mask):
                continue
            x = np.asarray(y_pred[mask], dtype=float)
            y = np.asarray(y_true[mask], dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            x = x[valid]
            y = y[valid]
            if len(x) < min_pts:
                continue
            cov = np.cov(np.vstack([x, y]), ddof=1)
            if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
                continue
            eigvals, eigvecs = np.linalg.eigh(cov)
            if not np.all(np.isfinite(eigvals)):
                continue
            eigvals = np.maximum(eigvals, 0.0)
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            if eigvals[0] <= 0.0:
                continue
            width = 2.0 * nstd * float(np.sqrt(eigvals[0]))
            height = 2.0 * nstd * float(np.sqrt(eigvals[1]))
            if not (np.isfinite(width) and np.isfinite(height)) or width <= 0.0 or height <= 0.0:
                continue
            angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
            cx = float(np.mean(x))
            cy = float(np.mean(y))
            rgba = ellipse_cmap.get(ecat, (0.25, 0.25, 0.25, 1.0))
            edge_rgba = (rgba[0], rgba[1], rgba[2], resolved_ellipse_edge_alpha)
            face_rgba = (rgba[0], rgba[1], rgba[2], resolved_ellipse_face_alpha)
            ellipse_specs.append(
                {
                    "cat": str(ecat),
                    "cx": cx,
                    "cy": cy,
                    "width": width,
                    "height": height,
                    "angle": angle,
                    "edge_rgba": edge_rgba,
                    "face_rgba": face_rgba,
                }
            )
        if bool(ellipse_equal_area) and len(ellipse_specs) >= 2:
            areas = np.asarray(
                [float(spec["width"]) * float(spec["height"]) for spec in ellipse_specs],
                dtype=float,
            )
            finite_areas = areas[np.isfinite(areas) & (areas > 0.0)]
            if finite_areas.size > 0:
                target_area = float(np.median(finite_areas))
                for spec in ellipse_specs:
                    area = float(spec["width"]) * float(spec["height"])
                    if not np.isfinite(area) or area <= 0.0:
                        continue
                    scale = float(np.sqrt(target_area / area))
                    # Keep normalization moderate so ellipses stay comparable but not identical.
                    scale = float(np.clip(scale, 0.70, 1.40))
                    spec["width"] = float(spec["width"]) * scale
                    spec["height"] = float(spec["height"]) * scale
        for spec in ellipse_specs:
            ax.add_patch(
                Ellipse(
                    (float(spec["cx"]), float(spec["cy"])),
                    width=float(spec["width"]),
                    height=float(spec["height"]),
                    angle=float(spec["angle"]),
                    facecolor=spec["face_rgba"],
                    edgecolor=spec["edge_rgba"],
                    linewidth=1.6,
                    zorder=4,
                )
            )
            ellipse_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=spec["edge_rgba"],
                    linewidth=1.8,
                    label=str(spec["cat"]),
                )
            )

    if bool(axis_match_xy_limits):
        lo, hi = _compute_axis_limits(
            y_true=y_true,
            y_pred=y_pred,
            pad_frac=float(axis_pad_frac),
            clip_quantile=axis_clip_quantile,
        )
        if bool(center_axes_zero):
            lo, hi = _center_axis_limits_zero(lo, hi)
        xlo, xhi = lo, hi
        ylo, yhi = lo, hi
    else:
        xlo, xhi = _compute_axis_limits_1d(
            y_pred,
            pad_frac=float(axis_pad_frac),
            clip_quantile=axis_clip_quantile,
        )
        ylo, yhi = _compute_axis_limits_1d(
            y_true,
            pad_frac=float(axis_pad_frac),
            clip_quantile=axis_clip_quantile,
        )
        if bool(center_axes_zero):
            xlo, xhi = _center_axis_limits_zero(xlo, xhi)
            ylo, yhi = _center_axis_limits_zero(ylo, yhi)
        lo = float(min(xlo, ylo))
        hi = float(max(xhi, yhi))

    if show_identity_line:
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5, color="black", label="y=x")
    if show_fit_line and len(y_true) >= 2:
        m, b = np.polyfit(y_pred, y_true, deg=1)
        ax.plot([xlo, xhi], [m * xlo + b, m * xhi + b], linewidth=1.7, color="#ff7f0e", label="fit line")

    fit_group_lines_drawn = 0
    fit_group_lines_total = 0
    fit_group_lines_truncated = False
    if fit_line_group_values is not None and len(fit_line_group_values) == len(y_true):
        gser = pd.Series(fit_line_group_values)
        valid = gser.notna().to_numpy()
        if valid.any():
            gvals = gser[valid].astype(str)
            y_true_g = y_true[valid]
            y_pred_g = y_pred[valid]
            cats_all = gvals.value_counts(dropna=False).index.tolist()
            fit_group_lines_total = int(len(cats_all))
            cats = cats_all
            max_groups = int(max(fit_line_group_max_groups, 0))
            if max_groups > 0 and len(cats) > max_groups:
                cats = cats[:max_groups]
                fit_group_lines_truncated = True
            cmap_lines = _category_color_map(list(cats), saturation=color_saturation)
            for cat in cats:
                mask = (gvals.to_numpy() == cat)
                n_cat = int(np.sum(mask))
                if n_cat < max(int(fit_line_group_min_points), 2):
                    continue
                x_cat = y_pred_g[mask]
                y_cat = y_true_g[mask]
                if len(np.unique(x_cat)) < 2:
                    continue
                try:
                    m_cat, b_cat = np.polyfit(x_cat, y_cat, deg=1)
                except Exception:
                    continue
                x0 = float(np.min(x_cat))
                x1 = float(np.max(x_cat))
                if not (np.isfinite(x0) and np.isfinite(x1)) or x1 <= x0:
                    continue
                ax.plot(
                    [x0, x1],
                    [m_cat * x0 + b_cat, m_cat * x1 + b_cat],
                    linewidth=1.35,
                    alpha=0.85,
                    color=cmap_lines[cat],
                    linestyle="-",
                    label=(f"{fit_line_group_label}: {cat}" if fit_line_group_show_legend else None),
                )
                fit_group_lines_drawn += 1

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    if bool(axis_match_xy_limits):
        # Keep a fixed square plotting area when x/y limits are intentionally matched.
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1)
        else:
            ax.set_aspect("equal", adjustable="box")
    else:
        # Preserve equal data units (y=x remains interpretable) while allowing a tall/wide panel.
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(x_label, fontsize=11.0 * resolved_font_scale)
    ax.set_ylabel(y_label, fontsize=11.0 * resolved_font_scale)
    title = title_base
    if has_groups and color_label and show_color_in_title:
        title += f"\ncolor: {color_label}"
    if bool(show_title) and str(title).strip():
        ax.set_title(title, loc="center", fontsize=12.5 * resolved_font_scale)
    ax.tick_params(axis="both", labelsize=9.0 * resolved_font_scale)
    if identifier_text:
        fig.text(
            0.995,
            0.01,
            identifier_text,
            ha="right",
            va="bottom",
            fontsize=8.0 * resolved_font_scale,
            color="#444444",
        )
    resolved_ncol = int(legend_ncol) if legend_ncol is not None else (1 if len(cats) <= 18 else 2)
    legend_anchor_x = 1.02
    legend_artists: list[object] = []
    color_legend = None
    marker_legend = None
    color_handles: list[Line2D] = []
    marker_handles: list[Line2D] = []
    if has_groups:
        color_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=cmap[cat],
                markeredgecolor="none",
                markersize=max(4.2, np.sqrt(resolved_marker_size) * 0.95),
                label=str(cat),
            )
            for cat in cats
        ]
    if has_marker_groups:
        marker_handles = [
            Line2D(
                [0],
                [0],
                marker=marker_map[mcat],
                linestyle="None",
                markerfacecolor="#666666",
                markeredgecolor="none",
                markersize=max(4.5, np.sqrt(resolved_marker_size) * 0.95),
                label=str(mcat),
            )
            for mcat in marker_cats
        ]
    ellipse_same_as_color = bool(
        has_groups
        and has_ellipse_groups
        and str(color_label or "").strip()
        and str(color_label or "").strip() == str(ellipse_label or "").strip()
    )
    suppress_ellipse_legend = bool(ellipse_same_as_color and ellipse_handles)
    if marker_handles and color_handles:
        combined_handles: list[Line2D] = [
            Line2D([0], [0], linestyle="None", color="none", label=str(marker_legend_title))
        ]
        combined_handles.extend(marker_handles)
        combined_handles.append(Line2D([0], [0], linestyle="None", color="none", label=""))
        combined_handles.append(Line2D([0], [0], linestyle="None", color="none", label=str(legend_title)))
        combined_handles.extend(color_handles)
        marker_legend = ax.legend(
            handles=combined_handles,
            loc="upper left",
            bbox_to_anchor=(legend_anchor_x, 1.0),
            borderaxespad=0.0,
            frameon=True,
            fontsize=8.0 * resolved_legend_font_scale,
            ncol=1,
        )
        ax.add_artist(marker_legend)
        legend_artists.append(marker_legend)
        color_legend = marker_legend
    elif marker_handles:
        marker_legend = ax.legend(
            handles=marker_handles,
            loc="upper left",
            bbox_to_anchor=(legend_anchor_x, 1.0),
            borderaxespad=0.0,
            frameon=True,
            fontsize=8.0 * resolved_legend_font_scale,
            ncol=1,
            title=marker_legend_title,
            title_fontsize=9.0 * resolved_legend_font_scale,
        )
        ax.add_artist(marker_legend)
        legend_artists.append(marker_legend)
    elif color_handles:
        color_legend = ax.legend(
            handles=color_handles,
            loc="upper left",
            bbox_to_anchor=(legend_anchor_x, 1.0),
            borderaxespad=0.0,
            frameon=True,
            fontsize=8.0 * resolved_legend_font_scale,
            ncol=resolved_ncol,
            title=legend_title,
            title_fontsize=9.0 * resolved_legend_font_scale,
        )
        ax.add_artist(color_legend)
        legend_artists.append(color_legend)
    if not marker_handles and not color_handles:
        if show_identity_line or show_fit_line:
            fallback_legend = ax.legend(
                loc="upper left",
                bbox_to_anchor=(legend_anchor_x, 1.0),
                borderaxespad=0.0,
                frameon=True,
                fontsize=8.5 * resolved_legend_font_scale,
            )
            legend_artists.append(fallback_legend)
    if ellipse_handles and not suppress_ellipse_legend:
        ellipse_anchor_y = 1.0
        if color_legend is not None:
            try:
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                bbox_disp = color_legend.get_window_extent(renderer=renderer)
                (_, y0_ax) = ax.transAxes.inverted().transform((bbox_disp.x0, bbox_disp.y0))
                ellipse_anchor_y = float(max(0.02, min(1.0, y0_ax - 0.02)))
            except Exception:
                ellipse_anchor_y = 0.42
        ellipse_legend = ax.legend(
            handles=ellipse_handles,
            loc="upper left",
            bbox_to_anchor=(legend_anchor_x, ellipse_anchor_y),
            borderaxespad=0.0,
            frameon=True,
            fontsize=8.0 * resolved_legend_font_scale,
            ncol=1,
            title=ellipse_legend_title,
            title_fontsize=9.0 * resolved_legend_font_scale,
        )
        legend_artists.append(ellipse_legend)
    if has_centroid_groups and centroid_groups is not None:
        centroid_handles: list[Line2D] = []
        centroid_size = max(7.0, np.sqrt(resolved_marker_size) * 1.65)
        for ccat in centroid_cats:
            mask = centroid_groups == ccat
            if not np.any(mask):
                continue
            x_c = float(np.mean(y_pred[mask]))
            y_c = float(np.mean(y_true[mask]))
            marker = centroid_map.get(ccat, "X")
            ax.scatter(
                [x_c],
                [y_c],
                s=resolved_marker_size * 3.2,
                marker=marker,
                facecolor="white",
                edgecolor="#111111",
                linewidths=1.2,
                zorder=5,
            )
            centroid_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    linestyle="None",
                    markerfacecolor="white",
                    markeredgecolor="#111111",
                    markeredgewidth=1.0,
                    markersize=centroid_size,
                    label=str(ccat),
                )
            )
        if centroid_handles:
            centroid_legend = ax.legend(
                handles=centroid_handles,
                loc="upper left",
                bbox_to_anchor=(legend_anchor_x, 0.02 if (color_legend is not None or has_marker_groups) else 1.0),
                borderaxespad=0.0,
                frameon=True,
                fontsize=8.0 * resolved_legend_font_scale,
                ncol=1,
                title=centroid_legend_title,
                title_fontsize=9.0 * resolved_legend_font_scale,
            )
            legend_artists.append(centroid_legend)
    if show_diagnostics:
        info_lines = [f"RMSE={rmse:.2f}", f"Pearson={pear:+.3f}", f"Spearman={spear:+.3f}"]
        if diagnostics:
            n_rows = diagnostics.get("n_rows")
            std_ratio = diagnostics.get("std_ratio")
            pred_unique_frac = diagnostics.get("pred_unique_frac_approx")
            if isinstance(n_rows, (int, np.integer, float, np.floating)) and np.isfinite(float(n_rows)):
                info_lines.append(f"n={int(float(n_rows))}")
            if isinstance(std_ratio, (float, np.floating, int, np.integer)) and np.isfinite(float(std_ratio)):
                info_lines.append(f"std_ratio={float(std_ratio):.2f}")
            if isinstance(pred_unique_frac, (float, np.floating, int, np.integer)) and np.isfinite(float(pred_unique_frac)):
                info_lines.append(f"pred_unique_frac~={float(pred_unique_frac):.2f}")
            if diagnostics.get("dispersion_flag"):
                note = str(diagnostics.get("dispersion_note", "dispersion_flagged"))
                info_lines.append(f"flag={note}")
        if fit_line_group_values is not None and fit_group_lines_total > 0:
            label = str(fit_line_group_label or "group")
            line_msg = f"group_fit_lines({label})={fit_group_lines_drawn}/{fit_group_lines_total}"
            if fit_group_lines_truncated:
                line_msg += f" (top {int(max(fit_line_group_max_groups, 0))} by count)"
            info_lines.append(line_msg)
        ax.text(
            0.03,
            0.97,
            "\n".join(info_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10.0 * resolved_font_scale,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )
    # Fixed margins keep canvas geometry stable across plots regardless of legend length.
    fig.subplots_adjust(left=0.12, right=0.66, bottom=0.11, top=0.94)
    _save_figure(
        fig,
        out_path,
        dpi=220,
        tight_bbox=tight_bbox,
        bbox_extra_artists=legend_artists,
    )
    plt.close(fig)


def _plot_predictor_grid_residuals(
    panel_rows: list[dict[str, object]],
    out_path: Path,
    global_color_map: dict[str, tuple[float, float, float, float]],
    x_label: str,
    y_label: str,
    title: str | None = None,
    identifier_text: str | None = None,
    n_cols: int = 3,
    legend_ncol: int | None = None,
    axis_clip_quantile: float = 0.0,
    axis_pad_frac: float = 0.05,
    axis_match_xy_limits: bool = True,
    show_fit_line: bool = True,
    show_diagnostics: bool = True,
    show_identity_line: bool = True,
    show_title: bool = True,
    tight_bbox: bool = False,
    marker_size: float = 16.0,
    point_alpha: float | None = None,
    font_scale: float = 1.0,
    legend_font_scale: float = 1.0,
    color_saturation: float = 1.0,
    center_axes_zero: bool = False,
    legend_title: str = "Train Dataset",
) -> None:
    if not panel_rows:
        return

    n_cols = max(int(n_cols), 1)
    n_panels = int(len(panel_rows))
    n_rows = int(int(np.ceil(n_panels / float(n_cols))))
    fig_w = max(4.1 * n_cols, 8.5)
    fig_h = max(3.6 * n_rows, 3.8)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    resolved_font_scale = max(float(font_scale), 0.5)
    resolved_marker_size = max(float(marker_size), 2.0)
    resolved_legend_font_scale = resolved_font_scale * max(float(legend_font_scale), 0.5)
    resolved_point_alpha = None
    if point_alpha is not None:
        try:
            resolved_point_alpha = float(np.clip(float(point_alpha), 0.02, 1.0))
        except Exception:
            resolved_point_alpha = None

    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []
    for row in panel_rows:
        y_t = np.asarray(row.get("y_true"), dtype=float)
        y_p = np.asarray(row.get("y_pred"), dtype=float)
        all_y_true.append(y_t)
        all_y_pred.append(y_p)
    if not all_y_true:
        return
    all_true = np.concatenate(all_y_true, axis=0)
    all_pred = np.concatenate(all_y_pred, axis=0)
    if str(axis_match_xy_limits):
        lo, hi = _compute_axis_limits(
            y_true=all_true,
            y_pred=all_pred,
            pad_frac=float(axis_pad_frac),
            clip_quantile=float(axis_clip_quantile),
        )
        if bool(center_axes_zero):
            lo, hi = _center_axis_limits_zero(lo, hi)
        xlim = (lo, hi)
        ylim = (lo, hi)
    else:
        xlim = _compute_axis_limits_1d(
            values=all_pred,
            pad_frac=float(axis_pad_frac),
            clip_quantile=float(axis_clip_quantile),
        )
        ylim = _compute_axis_limits_1d(
            values=all_true,
            pad_frac=float(axis_pad_frac),
            clip_quantile=float(axis_clip_quantile),
        )
        if bool(center_axes_zero):
            xlim = _center_axis_limits_zero(xlim[0], xlim[1])
            ylim = _center_axis_limits_zero(ylim[0], ylim[1])

    legend_cats = [c for c in sorted(global_color_map.keys())]
    legend_handles: list[Line2D] = []
    for cat in legend_cats:
        rgba = global_color_map[cat]
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=tuple(rgba),
                markeredgecolor="none",
                markersize=max(5.0, np.sqrt(resolved_marker_size) * 0.95),
                label=str(cat),
            )
        )

    panel_iter = iter(panel_rows)
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r][c]
            try:
                panel = next(panel_iter)
            except StopIteration:
                ax.set_visible(False)
                continue

            y_true_panel = np.asarray(panel.get("y_true"), dtype=float)
            y_pred_panel = np.asarray(panel.get("y_pred"), dtype=float)
            label = str(panel.get("label", "")).strip()
            color_values = panel.get("color_values")
            color_values_arr = np.asarray(color_values, dtype=object)
            if color_values is None or color_values_arr.shape[0] != len(y_true_panel):
                ax.scatter(
                    y_pred_panel,
                    y_true_panel,
                    s=resolved_marker_size,
                    alpha=(resolved_point_alpha if resolved_point_alpha is not None else 0.52),
                    color="#1f77b4",
                    edgecolors="none",
                )
            else:
                cv = pd.Series(color_values_arr, copy=False).astype(str).to_numpy()
                local_cats = sorted(pd.Series(cv).dropna().unique().tolist())
                for cat in legend_cats:
                    if cat not in local_cats:
                        continue
                    mask = cv == cat
                    if not np.any(mask):
                        continue
                    ax.scatter(
                        y_pred_panel[mask],
                        y_true_panel[mask],
                        s=resolved_marker_size,
                        alpha=(resolved_point_alpha if resolved_point_alpha is not None else 0.52),
                        color=global_color_map.get(cat, (0.3, 0.3, 0.3, 1.0)),
                        edgecolors="none",
                    )

            if show_identity_line:
                ax.plot(
                    [xlim[0], xlim[1]],
                    [xlim[0], xlim[1]],
                    linestyle="--",
                    linewidth=1.1,
                    color="black",
                )
            if show_fit_line and len(y_true_panel) >= 2:
                try:
                    m_fit, b_fit = np.polyfit(y_pred_panel, y_true_panel, deg=1)
                    ax.plot(
                        [xlim[0], xlim[1]],
                        [m_fit * xlim[0] + b_fit, m_fit * xlim[1] + b_fit],
                        linewidth=1.3,
                        color="#ff7f0e",
                    )
                except Exception:
                    pass

            if show_diagnostics:
                rmse_p = float(np.sqrt(np.mean((y_pred_panel - y_true_panel) ** 2)))
                pear_p = _pearson(y_true_panel, y_pred_panel)
                spear_p = _spearman(y_true_panel, y_pred_panel)
                ax.text(
                    0.03,
                    0.02,
                    f"n={len(y_true_panel)}\nRMSE={rmse_p:.2f}\nPear={pear_p:+.3f}\nSr={spear_p:+.3f}",
                    transform=ax.transAxes,
                    va="bottom",
                    ha="left",
                    fontsize=8.5 * resolved_font_scale,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85),
                )

            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(label, loc="center", fontsize=10.2 * resolved_font_scale)
            if r == n_rows - 1:
                ax.set_xlabel(x_label, fontsize=9.5 * resolved_font_scale)
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel(y_label, fontsize=9.5 * resolved_font_scale)
            else:
                ax.set_yticklabels([])
            ax.tick_params(axis="both", labelsize=8.0 * resolved_font_scale)

    if str(title).strip() and show_title:
        fig.suptitle(str(title), x=0.5, y=0.995, ha="center", fontsize=12.5 * resolved_font_scale)
    if identifier_text:
        fig.text(
            0.995,
            0.005,
            identifier_text,
            ha="right",
            va="bottom",
            fontsize=8.0 * resolved_font_scale,
            color="#444444",
        )

    legend_ncol_local = int(legend_ncol) if legend_ncol is not None else 1

    # Reserve right-side margin for shared legend and keep legend visually close
    # to the plot grid instead of at the far edge of the canvas.
    right_margin = 0.70 if len(legend_handles) > 16 else 0.76
    legend_anchor_x = min(right_margin + 0.015, 0.90)
    legend_anchor_y = 0.97
    legend = fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(legend_anchor_x, legend_anchor_y),
        borderaxespad=0.0,
        frameon=True,
        fontsize=8.5 * resolved_legend_font_scale,
        title=legend_title,
        title_fontsize=9.3 * resolved_legend_font_scale,
        ncol=legend_ncol_local,
    )

    # Keep canvas geometry stable and reserve a compact legend lane.
    fig.subplots_adjust(
        left=0.09,
        right=right_margin,
        bottom=0.08,
        top=0.92,
        wspace=0.26,
        hspace=0.24,
    )
    _save_figure(
        fig,
        out_path,
        dpi=220,
        tight_bbox=True,
        bbox_extra_artists=[legend],
    )
    plt.close(fig)


def _plot_residual_scatter_aggregated(
    agg_df: pd.DataFrame,
    out_path: Path,
    color_label: str | None = None,
    identifier_text: str | None = None,
    title: str = "Aggregated residual fit (unique feature rows)",
    xlabel: str = "Predicted residual",
    ylabel: str = "Observed residual (group mean)",
    axis_clip_quantile: float = 0.0,
    show_fit_line: bool = True,
    show_diagnostics: bool = True,
    show_identity_line: bool = True,
    legend_ncol: int | None = None,
    tight_bbox: bool = False,
    pretty_dataset_labels: bool = False,
    synthetic_label: str = "SDF-Fractal3D",
    marker_size: float = 18.0,
    font_scale: float = 1.0,
    color_saturation: float = 1.0,
    mix_label_style: str = "full",
    center_axes_zero: bool = False,
):
    fig, ax = plt.subplots(figsize=(11.0, 6.6))
    resolved_marker_size = max(float(marker_size), 2.0)
    resolved_font_scale = max(float(font_scale), 0.5)
    color_series = _display_group_values(
        agg_df["_color"],
        label_name=color_label,
        pretty_dataset_labels=pretty_dataset_labels,
        synthetic_label=synthetic_label,
        mix_label_style=mix_label_style,
    )
    cats = sorted(color_series.dropna().unique().tolist())
    cmap = _category_color_map(cats, saturation=color_saturation) if cats else {}

    base = agg_df["n_rep"].to_numpy(dtype=float)
    sizes = resolved_marker_size + 4.0 * np.sqrt(np.maximum(base, 1.0))
    for cat in cats:
        m = (color_series == cat).to_numpy()
        ax.scatter(
            agg_df.loc[m, "predicted_mean"],
            agg_df.loc[m, "observed_mean"],
            s=sizes[m],
            alpha=0.70,
            color=cmap[cat],
            edgecolors="none",
            label=cat,
        )

    y_true = agg_df["observed_mean"].to_numpy(dtype=float)
    y_pred = agg_df["predicted_mean"].to_numpy(dtype=float)
    lo, hi = _compute_axis_limits(
        y_true=y_true,
        y_pred=y_pred,
        pad_frac=0.05,
        clip_quantile=axis_clip_quantile,
    )
    if bool(center_axes_zero):
        lo, hi = _center_axis_limits_zero(lo, hi)
    if show_identity_line:
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5, color="black", label="y=x")
    if show_fit_line and len(y_true) >= 2:
        m, b = np.polyfit(y_pred, y_true, deg=1)
        ax.plot([lo, hi], [m * lo + b, m * hi + b], linewidth=1.7, color="#ff7f0e", label="fit line")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel, fontsize=11.0 * resolved_font_scale)
    ax.set_ylabel(ylabel, fontsize=11.0 * resolved_font_scale)
    if str(title).strip():
        ax.set_title(title, loc="center", fontsize=12.5 * resolved_font_scale)
    ax.tick_params(axis="both", labelsize=9.0 * resolved_font_scale)
    if identifier_text:
        fig.text(
            0.995,
            0.01,
            identifier_text,
            ha="right",
            va="bottom",
            fontsize=8.0 * resolved_font_scale,
            color="#444444",
        )

    if show_diagnostics:
        pear = _pearson(y_true, y_pred)
        spear = _spearman(y_true, y_pred)
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        ax.text(
            0.03,
            0.97,
            f"Groups={len(agg_df)}\nRMSE={rmse:.2f}\nPearson={pear:+.3f}\nSpearman={spear:+.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10.0 * resolved_font_scale,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fontsize=8.0 * resolved_font_scale,
        ncol=(int(legend_ncol) if legend_ncol is not None else (1 if len(cats) <= 18 else 2)),
        title=_pretty_column_label(color_label),
        title_fontsize=9.0 * resolved_font_scale,
    )
    fig.tight_layout(rect=(0.0, 0.02, 0.74, 1.0))
    _save_figure(fig, out_path, dpi=220, tight_bbox=tight_bbox)
    plt.close(fig)


def _plot_perfect_side_by_side(
    y_true: np.ndarray,
    train_values: pd.Series | None,
    benchmark_values: pd.Series | None,
    out_path: Path,
    identifier_text: str | None = None,
    show_identity_line: bool = True,
    legend_ncol: int | None = None,
    tight_bbox: bool = False,
    pretty_dataset_labels: bool = False,
    synthetic_label: str = "SDF-Fractal3D",
):
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.2), sharex=True, sharey=True)
    y = np.asarray(y_true, dtype=float)
    lo = float(np.min(y))
    hi = float(np.max(y))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    lo -= pad
    hi += pad

    panels = [
        ("train_dataset", train_values),
        ("benchmark", benchmark_values),
    ]
    for ax, (label, series) in zip(axes, panels):
        if series is None or len(series) != len(y):
            ax.scatter(y, y, s=12, alpha=0.45, color="#1f77b4", edgecolors="none")
        else:
            display_series = _display_group_values(
                series,
                label_name=label,
                pretty_dataset_labels=pretty_dataset_labels,
                synthetic_label=synthetic_label,
            )
            groups = display_series.to_numpy()
            cats = sorted(display_series.dropna().unique().tolist())
            cmap = _category_color_map(cats)
            for cat in cats:
                m = groups == cat
                ax.scatter(
                    y[m],
                    y[m],
                    s=13,
                    alpha=0.62,
                    color=cmap[cat],
                    edgecolors="none",
                    label=cat,
                )
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0.0,
                frameon=True,
                fontsize=8,
                ncol=(int(legend_ncol) if legend_ncol is not None else (1 if len(cats) <= 12 else 2)),
                title=_pretty_column_label(label),
                title_fontsize=9,
            )
        if show_identity_line:
            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.4, color="black")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Perfect predicted residual (y_true)")
        ax.set_title(f"Perfect target view | color={label}", loc="center")
        ax.grid(alpha=0.18, linewidth=0.5)
    axes[0].set_ylabel("Observed residual (y_true)")
    fig.suptitle("What perfect prediction would look like", x=0.5, ha="center")
    if identifier_text:
        fig.text(0.995, 0.985, identifier_text, ha="right", va="top", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    _save_figure(fig, out_path, dpi=220, tight_bbox=tight_bbox)
    plt.close(fig)


def _plot_rank_error_scatter(
    detail_df: pd.DataFrame,
    out_path: Path,
    title: str,
    identifier_text: str | None = None,
    summary_stats: dict[str, float | int | str | bool | None] | None = None,
):
    df = detail_df.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["true_rank_pct", "pred_rank_pct", "rank_error"])
    if df.empty:
        raise ValueError("Rank detail has no usable rows.")

    # rank_error > 0: predicted worse rank than true rank (underestimation)
    exact = df["rank_error"] == 0
    under = df["rank_error"] > 0
    over = df["rank_error"] < 0

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(
        df.loc[over, "true_rank_pct"],
        df.loc[over, "pred_rank_pct"],
        s=12,
        alpha=0.55,
        color="#1f77b4",
        label=f"Overestimate ({int(over.sum())})",
        edgecolors="none",
    )
    ax.scatter(
        df.loc[under, "true_rank_pct"],
        df.loc[under, "pred_rank_pct"],
        s=12,
        alpha=0.55,
        color="#d62728",
        label=f"Underestimate ({int(under.sum())})",
        edgecolors="none",
    )
    ax.scatter(
        df.loc[exact, "true_rank_pct"],
        df.loc[exact, "pred_rank_pct"],
        s=14,
        alpha=0.75,
        color="#2ca02c",
        label=f"Exact ({int(exact.sum())})",
        edgecolors="none",
    )

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("True rank percentile")
    ax.set_ylabel("Predicted rank percentile")
    ax.set_title(title)
    if identifier_text:
        ax.text(
            0.995,
            1.01,
            identifier_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#444444",
        )
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    if "pairwise_win_rate" in df.columns:
        pair_win = float(pd.to_numeric(df["pairwise_win_rate"], errors="coerce").mean())
    else:
        pair_win = float(np.mean((pd.to_numeric(df["rank_error"], errors="coerce") == 0).to_numpy(dtype=bool)))
    if "abs_rank_pct_error" in df.columns:
        rank_pct_err = float(pd.to_numeric(df["abs_rank_pct_error"], errors="coerce").mean())
    else:
        rank_pct_err = float(
            np.mean(
                np.abs(
                    pd.to_numeric(df["pred_rank_pct"], errors="coerce").to_numpy(dtype=float)
                    - pd.to_numeric(df["true_rank_pct"], errors="coerce").to_numpy(dtype=float)
                )
            )
        )
    stat_lines = [f"pair_win={pair_win:.2f}", f"rank_pct_err={rank_pct_err:.2f}"]
    if summary_stats:
        n_groups = summary_stats.get("rank_n_groups")
        n_reliable = summary_stats.get("rank_n_groups_reliable")
        small_frac = summary_stats.get("rank_small_group_frac")
        pred_tie_frac = summary_stats.get("rank_pred_tied_group_frac")
        if (
            isinstance(n_groups, (float, np.floating, int, np.integer))
            and isinstance(n_reliable, (float, np.floating, int, np.integer))
            and np.isfinite(float(n_groups))
            and np.isfinite(float(n_reliable))
        ):
            stat_lines.append(f"reliable_groups={int(float(n_reliable))}/{int(float(n_groups))}")
        if isinstance(small_frac, (float, np.floating, int, np.integer)) and np.isfinite(float(small_frac)):
            stat_lines.append(f"small_group_frac={float(small_frac):.2f}")
        if isinstance(pred_tie_frac, (float, np.floating, int, np.integer)) and np.isfinite(float(pred_tie_frac)):
            stat_lines.append(f"pred_tied_group_frac={float(pred_tie_frac):.2f}")
        warn = str(summary_stats.get("rank_reliability_warning", "") or "").strip()
        if warn:
            stat_lines.append(f"warning={warn}")
    ax.text(
        0.03,
        0.03,
        "\n".join(stat_lines),
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _resolve_rank_group_cols_for_points(
    df: pd.DataFrame,
    requested_group_cols: list[str] | None = None,
) -> list[str]:
    req = [str(c) for c in list(requested_group_cols or []) if str(c)]
    if req:
        out = [c for c in req if c in df.columns]
        return out

    # Auto-heuristics for heldout protocol rows.
    candidates = [
        ["fold_id"],
        ["fold"],
        ["joint_holdout"],
        ["benchmark", "model_family_encoder"],
        ["benchmark"],
    ]
    for cols in candidates:
        if not all(c in df.columns for c in cols):
            continue
        try:
            sizes = df.groupby(cols, dropna=False).size()
        except Exception:
            continue
        if not sizes.empty and int((sizes >= 2).sum()) > 0:
            return cols
    return []


def _build_rank_alignment_rows(
    df: pd.DataFrame,
    true_col: str = "y_true",
    pred_col: str = "y_pred",
    group_cols: list[str] | None = None,
    min_group_size: int = 3,
    min_group_unique_values: int = 2,
    approx_unique_abs_tol: float = 1e-8,
    approx_unique_rel_tol: float = 1e-3,
) -> tuple[pd.DataFrame, dict[str, float | int | str | None]]:
    def _empty_summary(group_cols_text: str) -> dict[str, float | int | str | None]:
        return {
            "rank_n_groups": 0,
            "rank_spearman_macro": float("nan"),
            "rank_top1": float("nan"),
            "rank_abs_rank_pct_error": float("nan"),
            "rank_group_cols": group_cols_text,
            "rank_n_groups_reliable": 0,
            "rank_reliable_frac": float("nan"),
            "rank_spearman_macro_reliable": float("nan"),
            "rank_top1_reliable": float("nan"),
            "rank_abs_rank_pct_error_reliable": float("nan"),
            "rank_group_size_min": float("nan"),
            "rank_group_size_median": float("nan"),
            "rank_group_size_max": float("nan"),
            "rank_small_group_frac": float("nan"),
            "rank_pred_tied_group_frac": float("nan"),
            "rank_true_tied_group_frac": float("nan"),
            "rank_pred_unique_approx_median": float("nan"),
            "rank_true_unique_approx_median": float("nan"),
            "rank_reliability_warning": "",
        }

    work = df.copy()
    if true_col not in work.columns or pred_col not in work.columns:
        return pd.DataFrame(), _empty_summary("")
    work[true_col] = pd.to_numeric(work[true_col], errors="coerce")
    work[pred_col] = pd.to_numeric(work[pred_col], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[true_col, pred_col]).copy()
    if work.empty:
        return pd.DataFrame(), _empty_summary(",".join(list(group_cols or [])))

    gcols = [c for c in list(group_cols or []) if c in work.columns]
    if gcols:
        grouped = work.groupby(gcols, dropna=False, sort=False)
        group_iter = ((k, g.copy()) for k, g in grouped)
    else:
        group_iter = [("__all__", work.copy())]

    out_rows: list[pd.DataFrame] = []
    group_spears: list[float] = []
    group_top1: list[float] = []
    group_sizes: list[int] = []
    group_pred_unique_approx: list[int] = []
    group_true_unique_approx: list[int] = []
    reliable_group_spears: list[float] = []
    reliable_group_top1: list[float] = []
    reliable_group_abs_rank_pct_errors: list[float] = []
    for gkey, gdf in group_iter:
        if len(gdf) < 2:
            continue
        gdf = gdf.copy()
        group_size = int(len(gdf))
        pred_unique_approx = int(
            _approx_n_unique(
                gdf[pred_col].to_numpy(dtype=float),
                abs_tol=float(approx_unique_abs_tol),
                rel_tol=float(approx_unique_rel_tol),
            )
        )
        true_unique_approx = int(
            _approx_n_unique(
                gdf[true_col].to_numpy(dtype=float),
                abs_tol=float(approx_unique_abs_tol),
                rel_tol=float(approx_unique_rel_tol),
            )
        )
        is_small_group = group_size < int(min_group_size)
        pred_tied_group = pred_unique_approx < int(min_group_unique_values)
        true_tied_group = true_unique_approx < int(min_group_unique_values)
        is_reliable_group = (not is_small_group) and (not pred_tied_group) and (not true_tied_group)
        gdf["_rank_group_size"] = group_size
        gdf["_rank_group_pred_unique_approx"] = pred_unique_approx
        gdf["_rank_group_true_unique_approx"] = true_unique_approx
        gdf["_rank_group_is_reliable"] = bool(is_reliable_group)

        # Higher score = better rank (rank 1).
        gdf["true_rank"] = gdf[true_col].rank(method="average", ascending=False)
        gdf["pred_rank"] = gdf[pred_col].rank(method="average", ascending=False)
        denom = float(max(len(gdf) - 1, 1))
        gdf["true_rank_pct"] = (gdf["true_rank"] - 1.0) / denom
        gdf["pred_rank_pct"] = (gdf["pred_rank"] - 1.0) / denom
        gdf["rank_error"] = gdf["pred_rank"] - gdf["true_rank"]
        gdf["abs_rank_pct_error"] = np.abs(gdf["pred_rank_pct"] - gdf["true_rank_pct"])
        if isinstance(gkey, tuple):
            gdf["_rank_group"] = "|".join(str(x) for x in gkey)
        else:
            gdf["_rank_group"] = str(gkey)
        out_rows.append(gdf)

        group_sizes.append(group_size)
        group_pred_unique_approx.append(pred_unique_approx)
        group_true_unique_approx.append(true_unique_approx)

        g_spear = _spearman(
            gdf[true_col].to_numpy(dtype=float),
            gdf[pred_col].to_numpy(dtype=float),
        )
        if np.isfinite(g_spear):
            group_spears.append(float(g_spear))
            if is_reliable_group:
                reliable_group_spears.append(float(g_spear))

        true_top = set(gdf.index[gdf["true_rank"] <= 1.0 + 1e-12].tolist())
        pred_top = set(gdf.index[gdf["pred_rank"] <= 1.0 + 1e-12].tolist())
        top1_hit = 1.0 if len(true_top.intersection(pred_top)) > 0 else 0.0
        group_top1.append(top1_hit)
        if is_reliable_group:
            reliable_group_top1.append(top1_hit)
            reliable_group_abs_rank_pct_errors.append(
                float(np.mean(gdf["abs_rank_pct_error"].to_numpy(dtype=float)))
            )

    if not out_rows:
        return pd.DataFrame(), _empty_summary(",".join(gcols))

    out = pd.concat(out_rows, axis=0, ignore_index=True)
    n_groups = int(out["_rank_group"].nunique())
    n_reliable_groups = int(sum(bool(x) for x in out.groupby("_rank_group", dropna=False)["_rank_group_is_reliable"].first().tolist()))
    reliable_frac = float(n_reliable_groups / n_groups) if n_groups > 0 else float("nan")
    small_group_frac = (
        float(np.mean(np.asarray(group_sizes, dtype=float) < float(min_group_size)))
        if group_sizes
        else float("nan")
    )
    pred_tied_group_frac = (
        float(np.mean(np.asarray(group_pred_unique_approx, dtype=float) < float(min_group_unique_values)))
        if group_pred_unique_approx
        else float("nan")
    )
    true_tied_group_frac = (
        float(np.mean(np.asarray(group_true_unique_approx, dtype=float) < float(min_group_unique_values)))
        if group_true_unique_approx
        else float("nan")
    )

    reliability_warn = ""
    if n_groups <= 0:
        reliability_warn = "no_rank_groups"
    elif n_reliable_groups <= 0:
        reliability_warn = "no_reliable_groups"
    elif np.isfinite(reliable_frac) and reliable_frac < 0.5:
        reliability_warn = "majority_groups_unreliable"

    summary = {
        "rank_n_groups": n_groups,
        "rank_spearman_macro": float(np.mean(group_spears)) if group_spears else float("nan"),
        "rank_top1": float(np.mean(group_top1)) if group_top1 else float("nan"),
        "rank_abs_rank_pct_error": float(np.mean(out["abs_rank_pct_error"].to_numpy(dtype=float))),
        "rank_group_cols": ",".join(gcols),
        "rank_n_groups_reliable": n_reliable_groups,
        "rank_reliable_frac": reliable_frac,
        "rank_spearman_macro_reliable": (
            float(np.mean(reliable_group_spears)) if reliable_group_spears else float("nan")
        ),
        "rank_top1_reliable": float(np.mean(reliable_group_top1)) if reliable_group_top1 else float("nan"),
        "rank_abs_rank_pct_error_reliable": (
            float(np.mean(reliable_group_abs_rank_pct_errors))
            if reliable_group_abs_rank_pct_errors
            else float("nan")
        ),
        "rank_group_size_min": float(np.min(group_sizes)) if group_sizes else float("nan"),
        "rank_group_size_median": float(np.median(group_sizes)) if group_sizes else float("nan"),
        "rank_group_size_max": float(np.max(group_sizes)) if group_sizes else float("nan"),
        "rank_small_group_frac": small_group_frac,
        "rank_pred_tied_group_frac": pred_tied_group_frac,
        "rank_true_tied_group_frac": true_tied_group_frac,
        "rank_pred_unique_approx_median": (
            float(np.median(group_pred_unique_approx)) if group_pred_unique_approx else float("nan")
        ),
        "rank_true_unique_approx_median": (
            float(np.median(group_true_unique_approx)) if group_true_unique_approx else float("nan")
        ),
        "rank_reliability_warning": reliability_warn,
    }
    return out, summary


def _plot_single_context_scatter(
    df: pd.DataFrame,
    out_path: Path,
    context_desc: str,
    color_col: str | None = "train_dataset",
    identifier_text: str | None = None,
    axis_clip_quantile: float = 0.0,
    show_fit_line: bool = True,
    show_diagnostics: bool = True,
    show_identity_line: bool = True,
    legend_ncol: int | None = None,
    tight_bbox: bool = False,
    pretty_dataset_labels: bool = False,
    synthetic_label: str = "SDF-Fractal3D",
    center_axes_zero: bool = False,
):
    work = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["y", "yhat"]).copy()
    if work.empty:
        raise ValueError("Single-context plot has no usable rows.")
    y_true = work["y"].to_numpy(dtype=float)
    y_pred = work["yhat"].to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    pear = _pearson(y_true, y_pred)
    spear = _spearman(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    has_color = color_col is not None and color_col in work.columns
    if has_color:
        display_series = _display_group_values(
            work[color_col],
            label_name=color_col,
            pretty_dataset_labels=pretty_dataset_labels,
            synthetic_label=synthetic_label,
        )
        groups = display_series.to_numpy()
        cats = sorted(display_series.dropna().unique().tolist())
        cmap = _category_color_map(cats)
        for cat in cats:
            m = groups == cat
            ax.scatter(
                y_pred[m],
                y_true[m],
                s=20,
                alpha=0.70,
                color=cmap[cat],
                edgecolors="none",
                label=cat,
            )
    else:
        ax.scatter(y_pred, y_true, s=20, alpha=0.60, color="#1f77b4", edgecolors="none")

    lo, hi = _compute_axis_limits(
        y_true=y_true,
        y_pred=y_pred,
        pad_frac=0.08,
        clip_quantile=axis_clip_quantile,
    )
    if bool(center_axes_zero):
        lo, hi = _center_axis_limits_zero(lo, hi)
    if show_identity_line:
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.4, color="black", label="y=x")
    if show_fit_line and len(y_true) >= 2:
        m, b = np.polyfit(y_pred, y_true, deg=1)
        ax.plot([lo, hi], [m * lo + b, m * hi + b], linewidth=1.8, color="#ff7f0e", label="fit line")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted residual")
    ax.set_ylabel("Observed residual")
    ax.set_title(f"Single context fit | {context_desc}", loc="center")
    if identifier_text:
        ax.text(
            0.995,
            1.01,
            identifier_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#444444",
        )
    if show_diagnostics:
        ax.text(
            0.03,
            0.97,
            f"n={len(work)}\nRMSE={rmse:.2f}\nPearson={pear:+.3f}\nSpearman={spear:+.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )
    if has_color:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
            fontsize=8,
            ncol=(int(legend_ncol) if legend_ncol is not None else 1),
            title=_pretty_column_label(color_col),
            title_fontsize=9,
        )
        fig.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    else:
        ax.legend(loc="upper left", frameon=True)
        fig.tight_layout()
    _save_figure(fig, out_path, dpi=220, tight_bbox=tight_bbox)
    plt.close(fig)


def _plot_residual_density_hexbin(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    identifier_text: str | None = None,
    x_label: str = "Predicted residual",
    y_label: str = "Observed residual",
    title: str = "Residual fit density (all rows)",
    axis_clip_quantile: float = 0.0,
    show_fit_line: bool = True,
    show_identity_line: bool = True,
    tight_bbox: bool = False,
    center_axes_zero: bool = False,
):
    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    hb = ax.hexbin(
        y_pred,
        y_true,
        gridsize=45,
        mincnt=1,
        cmap="viridis",
        linewidths=0.0,
    )
    cbar = fig.colorbar(hb, ax=ax, fraction=0.047, pad=0.02)
    cbar.set_label("count")

    lo, hi = _compute_axis_limits(
        y_true=y_true,
        y_pred=y_pred,
        pad_frac=0.05,
        clip_quantile=axis_clip_quantile,
    )
    if bool(center_axes_zero):
        lo, hi = _center_axis_limits_zero(lo, hi)
    if show_identity_line:
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.3, color="black", label="y=x")
    if show_fit_line and len(y_true) >= 2:
        m, b = np.polyfit(y_pred, y_true, deg=1)
        ax.plot([lo, hi], [m * lo + b, m * hi + b], linewidth=1.6, color="#ff7f0e", label="fit line")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, loc="center")
    if identifier_text:
        fig.text(0.995, 0.01, identifier_text, ha="right", va="bottom", fontsize=8, color="#444444")
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 1.0))
    _save_figure(fig, out_path, dpi=220, tight_bbox=tight_bbox)
    plt.close(fig)


def _plot_train_benchmark_median_iqr(
    fit_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    identifier_text: str | None = None,
    x_label: str = "Predicted residual",
    y_label: str = "Observed residual",
    title: str = "Train×benchmark medians (IQR bars)",
    axis_clip_quantile: float = 0.0,
    show_fit_line: bool = True,
    show_diagnostics: bool = True,
    show_identity_line: bool = True,
    legend_ncol: int | None = None,
    tight_bbox: bool = False,
    pretty_dataset_labels: bool = False,
    synthetic_label: str = "SDF-Fractal3D",
    marker_size: float = 22.0,
    font_scale: float = 1.0,
    color_saturation: float = 1.0,
    mix_label_style: str = "full",
    center_axes_zero: bool = False,
):
    if "train_dataset" not in fit_df.columns or "benchmark" not in fit_df.columns:
        return
    work = fit_df.copy()
    work["observed"] = y_true
    work["predicted"] = y_pred

    agg = (
        work.groupby(["train_dataset", "benchmark"], dropna=False)
        .agg(
            obs_med=("observed", "median"),
            obs_q25=("observed", lambda s: float(np.quantile(s.to_numpy(dtype=float), 0.25))),
            obs_q75=("observed", lambda s: float(np.quantile(s.to_numpy(dtype=float), 0.75))),
            pred_med=("predicted", "median"),
            pred_q25=("predicted", lambda s: float(np.quantile(s.to_numpy(dtype=float), 0.25))),
            pred_q75=("predicted", lambda s: float(np.quantile(s.to_numpy(dtype=float), 0.75))),
            n=("observed", "size"),
        )
        .reset_index()
    )
    if agg.empty:
        return

    agg = agg.copy()
    agg["_train_display"] = _display_group_values(
        agg["train_dataset"],
        label_name="train_dataset",
        pretty_dataset_labels=pretty_dataset_labels,
        synthetic_label=synthetic_label,
        mix_label_style=mix_label_style,
    )
    cats = sorted(agg["_train_display"].dropna().unique().tolist())
    cmap = _category_color_map(cats, saturation=color_saturation)
    resolved_marker_size = max(float(marker_size), 2.0)
    resolved_font_scale = max(float(font_scale), 0.5)

    fig, ax = plt.subplots(figsize=(11.0, 6.8))
    for _, row in agg.iterrows():
        cat = str(row["_train_display"])
        color = cmap.get(cat, (0.2, 0.2, 0.2, 1.0))
        x = float(row["pred_med"])
        y = float(row["obs_med"])
        x_lo = max(0.0, x - float(row["pred_q25"]))
        x_hi = max(0.0, float(row["pred_q75"]) - x)
        y_lo = max(0.0, y - float(row["obs_q25"]))
        y_hi = max(0.0, float(row["obs_q75"]) - y)
        ax.errorbar(
            x,
            y,
            xerr=np.array([[x_lo], [x_hi]]),
            yerr=np.array([[y_lo], [y_hi]]),
            fmt="none",
            ecolor=color,
            elinewidth=0.8,
            alpha=0.38,
            capsize=1.8,
            zorder=1,
        )

    for cat in cats:
        sub = agg[agg["_train_display"].astype(str) == cat]
        if sub.empty:
            continue
        sizes = resolved_marker_size + 3.0 * np.sqrt(sub["n"].to_numpy(dtype=float))
        ax.scatter(
            sub["pred_med"],
            sub["obs_med"],
            s=sizes,
            alpha=0.78,
            color=cmap[cat],
            edgecolors="none",
            label=cat,
            zorder=2,
        )

    y_true_g = agg["obs_med"].to_numpy(dtype=float)
    y_pred_g = agg["pred_med"].to_numpy(dtype=float)
    lo, hi = _compute_axis_limits(
        y_true=y_true_g,
        y_pred=y_pred_g,
        pad_frac=0.05,
        clip_quantile=axis_clip_quantile,
    )
    if bool(center_axes_zero):
        lo, hi = _center_axis_limits_zero(lo, hi)
    if show_identity_line:
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.3, color="black", label="y=x")
    if show_fit_line and len(y_true_g) >= 2:
        m, b = np.polyfit(y_pred_g, y_true_g, deg=1)
        ax.plot([lo, hi], [m * lo + b, m * hi + b], linewidth=1.6, color="#ff7f0e", label="fit line")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"{x_label} (train×benchmark median)", fontsize=11.0 * resolved_font_scale)
    ax.set_ylabel(f"{y_label} (train×benchmark median)", fontsize=11.0 * resolved_font_scale)
    if str(title).strip():
        ax.set_title(title, loc="center", fontsize=12.5 * resolved_font_scale)
    ax.tick_params(axis="both", labelsize=9.0 * resolved_font_scale)
    if show_diagnostics:
        rmse = float(np.sqrt(np.mean((y_pred_g - y_true_g) ** 2)))
        pear = _pearson(y_true_g, y_pred_g)
        spear = _spearman(y_true_g, y_pred_g)
        ax.text(
            0.03,
            0.97,
            f"Groups={len(agg)}\nRMSE={rmse:.2f}\nPearson={pear:+.3f}\nSpearman={spear:+.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10.0 * resolved_font_scale,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )
    if identifier_text:
        fig.text(
            0.995,
            0.01,
            identifier_text,
            ha="right",
            va="bottom",
            fontsize=8.0 * resolved_font_scale,
            color="#444444",
        )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fontsize=8.0 * resolved_font_scale,
        ncol=(int(legend_ncol) if legend_ncol is not None else (1 if len(cats) <= 18 else 2)),
        title=_pretty_column_label("train_dataset"),
        title_fontsize=9.0 * resolved_font_scale,
    )
    fig.tight_layout(rect=(0.0, 0.02, 0.74, 1.0))
    _save_figure(fig, out_path, dpi=220, tight_bbox=tight_bbox)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None, help="Leakage-free run directory. If this path is a run root (no auc_with_features.csv), best-CV auto-selection is applied.")
    parser.add_argument("--run-root", default=None, help="Root directory containing many run dirs; script auto-selects the best CV run.")
    parser.add_argument(
        "--best-cv-metric",
        default="loto_pair_win",
        choices=_selection_metric_choices(),
        help="Metric used to select the best run when using --run-root (or --run-dir pointing at a root).",
    )
    parser.add_argument(
        "--rankfirst-min-dispersion-ratio",
        type=float,
        default=RANKFIRST_MIN_DISPERSION_RATIO,
        help=(
            "Minimum pred_std/target_std required by rank-first anti-degeneracy selectors "
            f"(default: {RANKFIRST_MIN_DISPERSION_RATIO:g})."
        ),
    )
    parser.add_argument("--target", default=None, help="Target column (default from metadata or auc_normalized_observed).")
    parser.add_argument("--predictors", default=None, help="Comma-separated predictors (default from metadata).")
    parser.add_argument("--context-cols", default=None, help="Residual context cols CSV (default from metadata).")
    parser.add_argument("--ridge-alpha", type=float, default=None, help="Ridge alpha (default from metadata).")
    parser.add_argument("--top-k", type=int, default=8, help="Keep top-k predictors by abs standardized coef (0=keep all).")
    parser.add_argument(
        "--rank-detail-file",
        default="prediction_loto_holdout_placement_detail.csv",
        help="Deprecated and ignored (kept for CLI compatibility).",
    )
    parser.add_argument(
        "--color-by",
        default="train_dataset",
        help="Column in auc_with_features.csv to color residual scatter by (empty to disable).",
    )
    parser.add_argument(
        "--predictor-grid",
        action="store_true",
        help="Generate a compact predictor-grid figure (multiple subset fits in one panel set).",
    )
    parser.add_argument(
        "--predictor-grid-spec",
        default="",
        help=(
            "Optional CSV specs for predictor-grid rows/columns. Examples: "
            "flow_only_f1,appearance_only_a2,hybrid_f2_a2,full."
        ),
    )
    parser.add_argument(
        "--predictor-grid-columns",
        type=int,
        default=3,
        help="Number of columns in predictor-grid output.",
    )
    parser.add_argument(
        "--predictor-grid-spec-selection",
        choices=["first", "best"],
        default="first",
        help=(
            "How family specs pick predictors. "
            "'first' keeps existing order-based behavior. "
            "'best' searches candidate subsets for flow/appearance/hybrid specs."
        ),
    )
    parser.add_argument(
        "--predictor-grid-best-score",
        choices=["pearson", "spearman", "rmse"],
        default="pearson",
        help="Score used by --predictor-grid-spec-selection=best.",
    )
    parser.add_argument(
        "--predictor-grid-best-max-combos",
        type=int,
        default=5000,
        help=(
            "Maximum candidate subsets evaluated per predictor-grid spec when "
            "--predictor-grid-spec-selection=best (0 disables limit)."
        ),
    )
    parser.add_argument(
        "--predictor-grid-only",
        action="store_true",
        help=(
            "When set, skip standard full-data single-panel residual plots and emit only "
            "the predictor-grid figure."
        ),
    )
    parser.add_argument(
        "--predictor-grid-output-name",
        default="residual_fit_predictor_grid",
        help="Output filename suffix for predictor-grid plot.",
    )
    parser.add_argument(
        "--predictor-grid-family-layout",
        action="store_true",
        help=(
            "Arrange predictor-grid panels by predictor family columns (Flow, Appearance, Hybrid) "
            "instead of row-wise wrapping. Full panels are appended after family columns."
        ),
    )
    parser.add_argument(
        "--predictor-grid-zscore",
        action="store_true",
        help="Also emit a predictor-grid plot in context-normalized residual (z-score) space.",
    )
    parser.add_argument(
        "--predictor-grid-zscore-output-name",
        default="residual_fit_predictor_grid_zscore",
        help="Output filename suffix for predictor-grid z-score plot.",
    )
    parser.add_argument(
        "--paper-ready",
        action="store_true",
        help=(
            "Enable paper-oriented plotting defaults: cleaner dataset labels, tighter legend-safe export, "
            "hidden fit diagnostics, no y=x reference line, single-column legends, "
            "and 1%% axis clipping (unless overridden)."
        ),
    )
    parser.add_argument(
        "--pretty-dataset-labels",
        action="store_true",
        help="Use cleaned/capitalized train-dataset labels in legends.",
    )
    parser.add_argument(
        "--paper-synthetic-label",
        default="SDF-Fractal3D",
        help="Display label used for synthetic datasets when --pretty-dataset-labels is enabled.",
    )
    parser.add_argument(
        "--axis-clip-quantile",
        type=float,
        default=0.0,
        help=(
            "Optional symmetric quantile clipping for axes (e.g., 0.01 keeps [1%%,99%%] range). "
            "Default: 0 (disabled)."
        ),
    )
    parser.add_argument(
        "--axis-pad-frac",
        type=float,
        default=0.05,
        help=(
            "Fractional padding added to plot limits after clipping "
            "(default: 0.05, set near 0.0 to trim internal whitespace)."
        ),
    )
    parser.add_argument(
        "--axis-independent-limits",
        action="store_true",
        help=(
            "Use independent x/y limits while preserving equal data units; "
            "allows tall/wide panels instead of forced square limits."
        ),
    )
    parser.add_argument(
        "--axis-center-zero",
        action="store_true",
        help="Force symmetric axis limits centered at 0 for scatter-style plots.",
    )
    parser.add_argument(
        "--hide-fit-diagnostics",
        action="store_true",
        help="Hide RMSE/Pearson/Spearman diagnostic text overlays on scatter plots.",
    )
    parser.add_argument(
        "--hide-fit-line",
        action="store_true",
        help="Hide regression fit lines (keeps y=x reference line).",
    )
    parser.add_argument(
        "--hide-title",
        action="store_true",
        help="Hide plot titles (useful when titling from LaTeX).",
    )
    parser.add_argument(
        "--tight-bbox",
        action="store_true",
        help="Save figures with bbox_inches='tight' to avoid legend clipping.",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=0.0,
        help="Base scatter marker size (0 = auto/default for mode).",
    )
    parser.add_argument(
        "--point-alpha",
        type=float,
        default=0.0,
        help="Scatter point alpha in [0,1] (0 = auto/default for mode).",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=0.0,
        help="Global font scale factor for scatter/legend text (0 = auto/default for mode).",
    )
    parser.add_argument(
        "--legend-font-scale",
        type=float,
        default=1.0,
        help="Additional legend-only font scale multiplier (default: 1.0).",
    )
    parser.add_argument(
        "--color-saturation",
        type=float,
        default=0.0,
        help="Category color saturation factor (0 = auto/default for mode).",
    )
    parser.add_argument(
        "--mix-label-style",
        choices=["auto", "full", "short", "short_wrap"],
        default="auto",
        help=(
            "Formatting style for synthetic mix legend labels: "
            "full, short, short_wrap, or auto (mode-dependent)."
        ),
    )
    parser.add_argument("--output-dir", default=None, help="Output directory (default: <run-dir>/paper_plots).")
    parser.add_argument(
        "--repeat-aggregation",
        choices=["auto", "none", "mean", "median"],
        default="auto",
        help=(
            "Aggregate repeated rows before residual fit. "
            "'auto' uses run_metadata cv_repeat_aggregation; default: auto."
        ),
    )
    parser.add_argument(
        "--repeat-group-cols",
        default="",
        help=(
            "CSV grouping columns for repeat aggregation. "
            "Default derives from run_metadata (train_dataset, benchmark, ranking/pairwise/context cols)."
        ),
    )
    parser.add_argument(
        "--context-target-transform",
        choices=["residual", "zscore"],
        default="residual",
        help=(
            "Context target transform for visualization fit: "
            "'residual' uses y-mean(context), 'zscore' uses (y-mean)/std(context)."
        ),
    )
    parser.add_argument(
        "--context-target-zscore-eps",
        type=float,
        default=1e-9,
        help="Small epsilon threshold for context std when --context-target-transform=zscore.",
    )
    parser.add_argument(
        "--context-target-plot-space",
        choices=["model_space", "residual", "absolute", "zscore"],
        default="model_space",
        help=(
            "Space used for plotting and plot-metrics. "
            "model_space: fit target space (residual or target-zscore), "
            "residual: raw residual space, "
            "zscore: context-normalized residual space, "
            "absolute: original target space."
        ),
    )
    parser.add_argument(
        "--prediction-transform",
        choices=["none", "zscore"],
        default="none",
        help="Optional transform applied to predictions before plotting/plot-metrics.",
    )
    parser.add_argument(
        "--prediction-transform-eps",
        type=float,
        default=1e-9,
        help="Small epsilon threshold used by --prediction-transform zscore.",
    )
    parser.add_argument(
        "--single-context",
        default="",
        help=(
            "Optional context filter as CSV key=value pairs "
            "(e.g., benchmark=kitti2012,model_family_encoder=raft_TF)."
        ),
    )
    parser.add_argument(
        "--single-context-color-by",
        default="train_dataset",
        help="Color column for single-context plot (default: train_dataset).",
    )
    parser.add_argument(
        "--heldout-protocols",
        default="",
        help=(
            "Optional CSV protocols for heldout-only prediction-vs-truth scatters. "
            "Common aliases: lobo,loto,lomo,jointood,model,training,eval."
        ),
    )
    parser.add_argument(
        "--heldout-color-by",
        default="train_dataset",
        help="Color column for heldout protocol scatters (default: train_dataset).",
    )
    parser.add_argument(
        "--heldout-shape-by",
        default="",
        help="Optional shape column for heldout protocol scatters (example: benchmark).",
    )
    parser.add_argument(
        "--heldout-centroid-by",
        default="",
        help="Optional heldout column used to plot large centroid markers per group (example: benchmark).",
    )
    parser.add_argument(
        "--heldout-ellipse-by",
        default="",
        help="Optional heldout column used to draw covariance ellipses per group (example: benchmark).",
    )
    parser.add_argument(
        "--heldout-ellipse-n-std",
        type=float,
        default=1.25,
        help="Ellipse radius in standard deviations for --heldout-ellipse-by (default: 1.25).",
    )
    parser.add_argument(
        "--heldout-ellipse-min-points",
        type=int,
        default=3,
        help="Minimum points in a group before drawing ellipse (default: 3).",
    )
    parser.add_argument(
        "--heldout-ellipse-face-alpha",
        type=float,
        default=0.10,
        help="Face alpha for heldout ellipses in [0,1] (default: 0.10).",
    )
    parser.add_argument(
        "--heldout-ellipse-edge-alpha",
        type=float,
        default=0.95,
        help="Edge alpha for heldout ellipses in [0,1] (default: 0.95).",
    )
    parser.add_argument(
        "--heldout-ellipse-equal-area",
        action="store_true",
        help="Normalize heldout ellipse areas toward a common size for cleaner visual comparison.",
    )
    parser.add_argument(
        "--heldout-ellipse-only",
        action="store_true",
        help="For heldout plots, draw group ellipses without plotting individual points.",
    )
    parser.add_argument(
        "--heldout-fit-lines-by",
        default="",
        help=(
            "Optional heldout column to fit subgroup regression lines on top of heldout scatters "
            "(example: benchmark)."
        ),
    )
    parser.add_argument(
        "--heldout-fit-lines-protocols",
        default="lobo,jointood",
        help=(
            "CSV heldout protocols to apply subgroup fit lines when --heldout-fit-lines-by is set "
            "(default: lobo,jointood)."
        ),
    )
    parser.add_argument(
        "--heldout-fit-lines-min-points",
        type=int,
        default=8,
        help="Minimum points per subgroup to draw a heldout fit line (default: 8).",
    )
    parser.add_argument(
        "--heldout-fit-lines-max-groups",
        type=int,
        default=24,
        help=(
            "Maximum subgroup lines to draw for heldout fit-lines; keep top groups by count. "
            "Use 0 to disable the cap (default: 24)."
        ),
    )
    parser.add_argument(
        "--heldout-fit-lines-show-legend",
        action="store_true",
        help="Show subgroup fit-line labels in legend (can be crowded).",
    )
    parser.add_argument(
        "--heldout-center-benchmark",
        action="store_true",
        help=(
            "Benchmark-center heldout y_true/y_pred before plotting/metrics for selected protocols."
        ),
    )
    parser.add_argument(
        "--heldout-center-benchmark-protocols",
        default="lobo,jointood",
        help=(
            "CSV protocols to benchmark-center when --heldout-center-benchmark is enabled "
            "(default: lobo,jointood)."
        ),
    )
    parser.add_argument(
        "--heldout-single-context",
        default="",
        help=(
            "Optional heldout context filter as CSV key=value pairs "
            "(example: model_family_encoder=raft_TF)."
        ),
    )
    parser.add_argument(
        "--heldout-collapse-aggregation",
        choices=["none", "mean", "median"],
        default="none",
        help="Optional aggregation for heldout points before plotting (default: none).",
    )
    parser.add_argument(
        "--heldout-collapse-group-cols",
        default="",
        help=(
            "CSV grouping columns for heldout point aggregation. "
            "Special value 'auto' uses train_dataset,benchmark."
        ),
    )
    parser.add_argument(
        "--heldout-plot-spaces",
        default="model_space",
        help=(
            "CSV heldout plotting spaces. Choices: model_space,residual,absolute. "
            "Example: residual,absolute"
        ),
    )
    parser.add_argument(
        "--heldout-model-cv-dir",
        default="",
        help=(
            "Optional heldout_model_cv output dir. Used for protocol aliases "
            "model/training/eval/triple/trainset_disjoint when run-level detail CSVs are unavailable."
        ),
    )
    parser.add_argument(
        "--heldout-model-cv-head",
        default="ridge",
        help="Head filter for --heldout-model-cv-dir pred rows (default: ridge).",
    )
    parser.add_argument(
        "--heldout-save-points",
        action="store_true",
        help="Also write per-protocol heldout scatter points as CSV.",
    )
    parser.add_argument(
        "--heldout-rank-plots",
        action="store_true",
        help="Deprecated and ignored (kept for CLI compatibility).",
    )
    parser.add_argument(
        "--heldout-rank-group-cols",
        default="auto",
        help=(
            "CSV group columns used to rank within heldout subsets. "
            "Use 'auto' to infer from available columns (default)."
        ),
    )
    parser.add_argument(
        "--heldout-rank-min-group-size",
        type=int,
        default=3,
        help="Minimum group size for a heldout rank group to be considered reliable (default: 3).",
    )
    parser.add_argument(
        "--heldout-rank-min-unique-values",
        type=int,
        default=2,
        help=(
            "Minimum approximate unique values required in both y_true and y_pred within a rank group "
            "for reliable rank metrics (default: 2)."
        ),
    )
    parser.add_argument(
        "--heldout-approx-unique-abs-tol",
        type=float,
        default=1e-8,
        help="Absolute tolerance used for approximate uniqueness diagnostics (default: 1e-8).",
    )
    parser.add_argument(
        "--heldout-approx-unique-rel-tol",
        type=float,
        default=1e-3,
        help="Relative tolerance used for approximate uniqueness diagnostics (default: 1e-3).",
    )
    parser.add_argument(
        "--heldout-dispersion-warn-std-ratio",
        type=float,
        default=0.35,
        help="Warn when heldout y_pred std / y_true std drops below this threshold (default: 0.35).",
    )
    parser.add_argument(
        "--heldout-dispersion-warn-unique-frac",
        type=float,
        default=0.35,
        help="Warn when approximate unique prediction fraction drops below this threshold (default: 0.35).",
    )
    parser.add_argument(
        "--heldout-best-per-protocol",
        action="store_true",
        help=(
            "For heldout protocols backed by leakage-free run dirs "
            "(e.g., lobo/loto/jointood), select best run separately per protocol. "
            "For heldout-model-CV protocols (model/training/eval/triple/trainset_disjoint), "
            "select the best candidate_id per protocol from heldout_model_cv_summary.csv."
        ),
    )
    parser.add_argument(
        "--heldout-protocol-metrics",
        default="",
        help=(
            "Optional protocol->best-cv-metric CSV overrides for "
            "--heldout-best-per-protocol (example: lobo=lobo_top1,loto=loto_pair_win)."
        ),
    )
    args = parser.parse_args()
    if bool(args.heldout_rank_plots):
        print(
            "Warning: --heldout-rank-plots is deprecated and ignored; "
            "this script now emits only prediction-vs-observed plots."
        )
    args.heldout_rank_plots = False

    if args.run_dir and args.run_root:
        raise ValueError("Pass only one of --run-dir or --run-root.")

    selection_info = None
    selection_root: Path | None = None
    if args.run_root:
        run_root = Path(args.run_root)
        selection_root = run_root
        selection_info = _select_best_run_dir(
            run_root,
            metric_key=args.best_cv_metric,
            min_dispersion_ratio=float(args.rankfirst_min_dispersion_ratio),
        )
        run_dir = Path(selection_info["run_dir"])
    elif args.run_dir:
        candidate = Path(args.run_dir)
        if not candidate.exists():
            raise FileNotFoundError(f"Run path not found: {candidate}")
        if (candidate / "auc_with_features.csv").exists():
            run_dir = candidate
        else:
            selection_root = candidate
            selection_info = _select_best_run_dir(
                candidate,
                metric_key=args.best_cv_metric,
                min_dispersion_ratio=float(args.rankfirst_min_dispersion_ratio),
            )
            run_dir = Path(selection_info["run_dir"])
    else:
        raise ValueError("Pass --run-dir or --run-root.")

    if not run_dir.exists():
        raise FileNotFoundError(f"Resolved run dir not found: {run_dir}")

    if selection_info is not None:
        plot_identifier = (
            f"best={Path(selection_info['run_dir']).name} | "
            f"{selection_info['metric_key']}={selection_info['metric_value']:.4f}"
        )
    else:
        plot_identifier = f"run={run_dir.name}"

    meta = _load_metadata(run_dir)
    target = args.target or meta.get("target", "auc_normalized_observed")
    predictors = _resolve_predictors(meta, args.predictors)
    context_cols = _resolve_context_cols(meta, args.context_cols)
    ridge_alpha = float(args.ridge_alpha if args.ridge_alpha is not None else meta.get("ridge_alpha", 10.0))
    color_modes = _resolve_color_modes(args.color_by)
    grid_plot_enabled = bool(args.predictor_grid)
    predictor_grid_warnings: list[str] = []
    if grid_plot_enabled and str(args.color_by).strip().lower() != "train_dataset":
        predictor_grid_warnings.append(
            "Predictor-grid mode currently forces color-by=train_dataset for clean shared legends."
        )
        color_modes = ["train_dataset"]
    if grid_plot_enabled:
        predictor_grid_specs_raw = _parse_csv_list(args.predictor_grid_spec)
    else:
        predictor_grid_specs_raw = []
    predictor_grid_spec_selection = str(args.predictor_grid_spec_selection or "first").strip().lower()
    predictor_grid_best_score = str(args.predictor_grid_best_score or "pearson").strip().lower()
    predictor_grid_best_max_combos = max(int(args.predictor_grid_best_max_combos), 0)
    color_by = color_modes[0] if color_modes else ""
    repeat_mode = str(args.repeat_aggregation or "auto").strip().lower()
    if repeat_mode == "auto":
        repeat_mode = str(meta.get("cv_repeat_aggregation", "none") or "none").strip().lower()
    if args.repeat_group_cols:
        repeat_group_cols = _parse_csv_list(args.repeat_group_cols)
    else:
        repeat_group_cols = _derive_repeat_group_cols(meta)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        if selection_root is not None:
            out_dir = selection_root / f"paper_plots_best_cv_{args.best_cv_metric}"
        else:
            out_dir = run_dir / "paper_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    paper_ready = bool(args.paper_ready)
    pretty_dataset_labels = bool(args.pretty_dataset_labels or paper_ready or grid_plot_enabled)
    synthetic_label = str(args.paper_synthetic_label or "").strip() or "SDF-Fractal3D"
    axis_clip_quantile = float(args.axis_clip_quantile)
    if paper_ready and axis_clip_quantile <= 0.0:
        axis_clip_quantile = 0.01
    axis_clip_quantile = min(max(axis_clip_quantile, 0.0), 0.49)
    axis_pad_frac = min(max(float(args.axis_pad_frac), 0.0), 0.30)
    axis_match_xy_limits = not bool(args.axis_independent_limits)
    center_axes_zero = bool(args.axis_center_zero)
    show_fit_diagnostics = not bool(args.hide_fit_diagnostics or paper_ready)
    show_fit_line = not bool(args.hide_fit_line)
    show_identity_line = not bool(paper_ready)
    show_title = not bool(args.hide_title)
    legend_ncol = 1 if paper_ready else None
    show_color_in_title = not bool(paper_ready)
    tight_bbox = bool(args.tight_bbox or paper_ready)
    plot_identifier_for_figure = None if paper_ready else plot_identifier
    marker_size = float(args.marker_size)
    if marker_size <= 0.0:
        marker_size = 22.0 if paper_ready else 15.0
    point_alpha = float(args.point_alpha)
    if point_alpha <= 0.0:
        point_alpha = None
    font_scale = float(args.font_scale)
    if font_scale <= 0.0:
        font_scale = 1.30 if paper_ready else 1.0
    legend_font_scale = max(float(args.legend_font_scale), 0.5)
    color_saturation = float(args.color_saturation)
    if color_saturation <= 0.0:
        color_saturation = 1.28 if paper_ready else 1.0
    mix_label_style_raw = str(args.mix_label_style or "auto").strip().lower()
    if mix_label_style_raw == "auto":
        mix_label_style = "short_wrap" if paper_ready else "full"
    else:
        mix_label_style = mix_label_style_raw

    passthrough = [c for c in color_modes if c]
    passthrough.extend(["train_dataset", "benchmark"])
    passthrough = list(dict.fromkeys(passthrough))
    fit_df, y_true, y_pred, rmse, pear, spear, coef_df, agg_info = _build_full_residual_fit(
        run_dir=run_dir,
        target=target,
        predictors=predictors,
        context_cols=context_cols,
        ridge_alpha=ridge_alpha,
        top_k=max(int(args.top_k), 0),
        passthrough_cols=passthrough,
        repeat_agg_mode=repeat_mode,
        repeat_group_cols=repeat_group_cols,
        context_target_transform=args.context_target_transform,
        context_target_zscore_eps=float(args.context_target_zscore_eps),
    )
    plot_space = str(args.context_target_plot_space).strip().lower()
    y_true_base, y_pred_plot, pred_tinfo = _prepare_plot_space_arrays(
        fit_df=fit_df,
        target=target,
        y_true=y_true,
        y_pred=y_pred,
        plot_space=plot_space,
        context_target_transform=str(args.context_target_transform),
        prediction_transform=str(args.prediction_transform),
        prediction_transform_eps=float(args.prediction_transform_eps),
    )
    rmse_plot = float(np.sqrt(np.mean((y_pred_plot - y_true_base) ** 2)))
    pear_plot = _pearson(y_true_base, y_pred_plot)
    spear_plot = _spearman(y_true_base, y_pred_plot)
    if plot_space == "absolute":
        x_label = f"Predicted {target}"
        y_label = f"Observed {target}"
        title_base = "Full-data fit (context-aware)"
        agg_x_label = f"Predicted {target} (group mean)"
        agg_y_label = f"Observed {target} (group mean)"
    elif plot_space == "residual":
        x_label = "Predicted residual"
        y_label = "Observed residual"
        title_base = "Full-data residual fit (context-aware)"
        agg_x_label = "Predicted residual"
        agg_y_label = "Observed residual (group mean)"
    elif plot_space == "zscore":
        x_label = "Predicted context-normalized residual (z)"
        y_label = "Observed context-normalized residual (z)"
        title_base = "Full-data context-normalized residual fit"
        agg_x_label = "Predicted context-normalized residual (z)"
        agg_y_label = "Observed context-normalized residual (z)"
    else:
        if str(args.context_target_transform).strip().lower() == "zscore":
            x_label = "Predicted context-normalized residual (z)"
            y_label = "Observed context-normalized residual (z)"
            title_base = "Full-data normalized residual fit (context-aware)"
            agg_x_label = "Predicted normalized residual (z)"
            agg_y_label = "Observed normalized residual (z, group mean)"
        else:
            x_label = "Predicted residual"
            y_label = "Observed residual"
            title_base = "Full-data residual fit (context-aware)"
            agg_x_label = "Predicted residual"
            agg_y_label = "Observed residual (group mean)"

    predictor_grid_specs_resolved: list[dict[str, object]] = []
    predictor_grid_panel_rows: list[dict[str, object]] = []
    predictor_grid_panel_rows_zscore: list[dict[str, object]] = []
    predictor_grid_name_raw = str(args.predictor_grid_output_name).strip() or "residual_fit_predictor_grid"
    predictor_grid_output = out_dir / (
        predictor_grid_name_raw
        if str(predictor_grid_name_raw).lower().endswith(".png")
        else f"{predictor_grid_name_raw}.png"
    )
    predictor_grid_zscore_enabled = bool(args.predictor_grid_zscore)
    predictor_grid_zscore_output = out_dir / (
        str(args.predictor_grid_zscore_output_name).strip()
        if str(args.predictor_grid_zscore_output_name).strip().lower().endswith(".png")
        else f"{str(args.predictor_grid_zscore_output_name).strip() or 'residual_fit_predictor_grid_zscore'}.png"
    )
    predictor_grid_generated = False
    predictor_grid_zscore_generated = False
    predictor_grid_plot_columns = max(int(args.predictor_grid_columns), 1)
    predictor_grid_source_df: pd.DataFrame | None = None
    if grid_plot_enabled and predictor_grid_spec_selection == "best":
        auc_path = run_dir / "auc_with_features.csv"
        if auc_path.exists():
            predictor_grid_source_df = pd.read_csv(auc_path)
    if grid_plot_enabled:
        if "train_dataset" not in fit_df.columns:
            predictor_grid_warnings.append(
                "Predictor-grid requested, but fit data is missing train_dataset; unable to color by dataset."
            )
        predictor_grid_specs_resolved, resolve_warnings = _resolve_predictor_grid_specs(
            raw_specs=predictor_grid_specs_raw,
            predictors=predictors,
        )
        predictor_grid_warnings.extend(resolve_warnings)
        if not predictor_grid_specs_resolved:
            predictor_grid_warnings.append("No predictor-grid panels could be built.")
        else:
            grid_color_values: list[np.ndarray] = []
            for spec in predictor_grid_specs_resolved:
                spec_predictors = [str(x) for x in spec.get("predictors", [])]
                if predictor_grid_spec_selection == "best":
                    spec_predictors, select_note = _maybe_select_best_predictor_subset_for_grid_spec(
                        token=str(spec.get("spec", "")),
                        base_subset=spec_predictors,
                        all_predictors=predictors,
                        run_dir=run_dir,
                        target=target,
                        context_cols=context_cols,
                        ridge_alpha=ridge_alpha,
                        passthrough_cols=passthrough,
                        repeat_agg_mode=repeat_mode,
                        repeat_group_cols=repeat_group_cols,
                        context_target_transform=args.context_target_transform,
                        context_target_zscore_eps=float(args.context_target_zscore_eps),
                        score_metric=predictor_grid_best_score,
                        max_combos=predictor_grid_best_max_combos,
                        source_df=predictor_grid_source_df,
                    )
                    if select_note:
                        predictor_grid_warnings.append(select_note)
                if not spec_predictors:
                    predictor_grid_warnings.append(
                        f"Skipping predictor-grid spec '{spec.get('spec', '')}': empty predictor subset."
                    )
                    continue
                grid_fit_df, g_y_true, g_y_pred, g_rmse, g_pear, g_spear, _, _ = _build_full_residual_fit(
                    run_dir=run_dir,
                    target=target,
                    predictors=spec_predictors,
                    context_cols=context_cols,
                    ridge_alpha=ridge_alpha,
                    top_k=0,
                    passthrough_cols=passthrough,
                    repeat_agg_mode=repeat_mode,
                    repeat_group_cols=repeat_group_cols,
                    context_target_transform=args.context_target_transform,
                    context_target_zscore_eps=float(args.context_target_zscore_eps),
                    source_df=predictor_grid_source_df,
                )
                g_true_plot, g_pred_plot, _ = _prepare_plot_space_arrays(
                    fit_df=grid_fit_df,
                    target=target,
                    y_true=g_y_true,
                    y_pred=g_y_pred,
                    plot_space=plot_space,
                    context_target_transform=str(args.context_target_transform),
                    prediction_transform=str(args.prediction_transform),
                    prediction_transform_eps=float(args.prediction_transform_eps),
                )
                if predictor_grid_zscore_enabled:
                    g_true_plot_z, g_pred_plot_z, _ = _prepare_plot_space_arrays(
                        fit_df=grid_fit_df,
                        target=target,
                        y_true=g_y_true,
                        y_pred=g_y_pred,
                        plot_space="zscore",
                        context_target_transform=str(args.context_target_transform),
                        prediction_transform=str(args.prediction_transform),
                        prediction_transform_eps=float(args.prediction_transform_eps),
                    )
                g_color = None
                if "train_dataset" in grid_fit_df.columns:
                    g_color = _display_group_values(
                        grid_fit_df["train_dataset"],
                        label_name="train_dataset",
                        pretty_dataset_labels=pretty_dataset_labels,
                        synthetic_label=synthetic_label,
                        mix_label_style=mix_label_style,
                    ).to_numpy()
                    grid_color_values.append(np.asarray(g_color, dtype=object))
                g_label = str(spec.get("label", "")).strip() or "All signals"
                predictor_grid_panel_rows.append(
                    {
                        "label": f"{g_label} [n={len(spec_predictors)}]",
                        "spec": str(spec.get("spec", "")),
                        "y_true": g_true_plot,
                        "y_pred": g_pred_plot,
                        "rmse": float(np.sqrt(np.mean((g_pred_plot - g_true_plot) ** 2))),
                        "pearson": float(_pearson(g_true_plot, g_pred_plot)),
                        "spearman": float(_spearman(g_true_plot, g_pred_plot)),
                        "color_values": g_color,
                    }
                )
                if predictor_grid_zscore_enabled:
                    predictor_grid_panel_rows_zscore.append(
                        {
                            "label": f"{g_label} [n={len(spec_predictors)}]",
                            "spec": str(spec.get("spec", "")),
                            "y_true": g_true_plot_z,
                            "y_pred": g_pred_plot_z,
                            "rmse": float(np.sqrt(np.mean((g_pred_plot_z - g_true_plot_z) ** 2))),
                            "pearson": float(_pearson(g_true_plot_z, g_pred_plot_z)),
                            "spearman": float(_spearman(g_true_plot_z, g_pred_plot_z)),
                            "color_values": g_color,
                        }
                    )
            if predictor_grid_panel_rows:
                if bool(args.predictor_grid_family_layout):
                    predictor_grid_panel_rows, predictor_grid_plot_columns, layout_warnings = (
                        _organize_predictor_grid_rows_by_family(
                            panel_rows=predictor_grid_panel_rows,
                            requested_cols=predictor_grid_plot_columns,
                        )
                    )
                    predictor_grid_warnings.extend(layout_warnings)
                    if predictor_grid_zscore_enabled and predictor_grid_panel_rows_zscore:
                        predictor_grid_panel_rows_zscore, z_cols, z_warnings = _organize_predictor_grid_rows_by_family(
                            panel_rows=predictor_grid_panel_rows_zscore,
                            requested_cols=predictor_grid_plot_columns,
                        )
                        predictor_grid_plot_columns = max(int(predictor_grid_plot_columns), int(z_cols))
                        predictor_grid_warnings.extend(z_warnings)
                        predictor_grid_warnings.extend([x for x in z_warnings if x not in layout_warnings])
                grid_color_categories: list[str] = []
                for cv in grid_color_values:
                    cats = pd.Series(cv).dropna().astype(str).unique().tolist()
                    grid_color_categories.extend([str(c) for c in cats])
                grid_color_categories = sorted(set(grid_color_categories))
                grid_color_map = _category_color_map(
                    grid_color_categories,
                    saturation=color_saturation,
                ) if grid_color_categories else {}
                _plot_predictor_grid_residuals(
                    panel_rows=predictor_grid_panel_rows,
                    out_path=predictor_grid_output,
                    global_color_map=grid_color_map,
                    x_label=x_label,
                    y_label=y_label,
                    title="Predictor subset ablation (target-to-predicted residuals)" if plot_space == "residual" else "Predictor subset ablation",
                    n_cols=predictor_grid_plot_columns,
                    axis_clip_quantile=axis_clip_quantile,
                    axis_pad_frac=axis_pad_frac,
                    axis_match_xy_limits=axis_match_xy_limits,
                    show_fit_line=show_fit_line,
                    show_diagnostics=show_fit_diagnostics,
                    show_identity_line=show_identity_line,
                    show_title=show_title,
                    tight_bbox=tight_bbox,
                    marker_size=marker_size,
                    point_alpha=point_alpha,
                    font_scale=font_scale,
                    legend_font_scale=legend_font_scale,
                    color_saturation=color_saturation,
                    center_axes_zero=center_axes_zero,
                    legend_title=_pretty_column_label("train_dataset"),
                )
                predictor_grid_generated = True
            if predictor_grid_zscore_enabled and predictor_grid_panel_rows_zscore:
                _plot_predictor_grid_residuals(
                    panel_rows=predictor_grid_panel_rows_zscore,
                    out_path=predictor_grid_zscore_output,
                    global_color_map=grid_color_map,
                    x_label="Predicted context-normalized residual (z)",
                    y_label="Observed context-normalized residual (z)",
                    title="Predictor subset ablation (context-normalized residuals)",
                    n_cols=predictor_grid_plot_columns,
                    axis_clip_quantile=axis_clip_quantile,
                    axis_pad_frac=axis_pad_frac,
                    axis_match_xy_limits=axis_match_xy_limits,
                    show_fit_line=show_fit_line,
                    show_diagnostics=show_fit_diagnostics,
                    show_identity_line=show_identity_line,
                    show_title=show_title,
                    tight_bbox=tight_bbox,
                    marker_size=marker_size,
                    point_alpha=point_alpha,
                    font_scale=font_scale,
                    legend_font_scale=legend_font_scale,
                    color_saturation=color_saturation,
                    center_axes_zero=center_axes_zero,
                    legend_title=_pretty_column_label("train_dataset"),
                )
                predictor_grid_zscore_generated = True

    coef_csv = out_dir / "full_fit_top_coefficients.csv"
    coef_df.to_csv(coef_csv, index=False)
    metrics_json = out_dir / "residual_fit_metrics.json"
    metrics_payload = {
        "n_rows": int(len(y_true)),
        "rmse": float(rmse_plot),
        "pearson": float(pear_plot),
        "spearman": float(spear_plot),
        "target": target,
        "predictors": predictors,
        "context_cols": context_cols,
        "ridge_alpha": ridge_alpha,
        "top_k": max(int(args.top_k), 0),
        "color_by_requested": args.color_by,
        "color_modes_generated": color_modes,
        "paper_ready": bool(paper_ready),
        "pretty_dataset_labels": bool(pretty_dataset_labels),
        "paper_synthetic_label": str(synthetic_label),
        "axis_clip_quantile": float(axis_clip_quantile),
        "axis_pad_frac": float(axis_pad_frac),
        "axis_match_xy_limits": bool(axis_match_xy_limits),
        "axis_center_zero": bool(center_axes_zero),
        "show_fit_diagnostics": bool(show_fit_diagnostics),
        "show_fit_line": bool(show_fit_line),
        "show_identity_line": bool(show_identity_line),
        "show_title": bool(show_title),
        "legend_ncol": int(legend_ncol) if legend_ncol is not None else None,
        "show_color_in_title": bool(show_color_in_title),
        "tight_bbox": bool(tight_bbox),
        "marker_size": float(marker_size),
        "point_alpha": (float(point_alpha) if point_alpha is not None else None),
        "font_scale": float(font_scale),
        "legend_font_scale": float(legend_font_scale),
        "color_saturation": float(color_saturation),
        "mix_label_style": str(mix_label_style),
        "repeat_aggregation": agg_info["repeat_aggregation"],
        "repeat_group_cols": agg_info["repeat_group_cols"],
        "rows_before_repeat_agg": agg_info["rows_before_repeat_agg"],
        "rows_after_repeat_agg": agg_info["rows_after_repeat_agg"],
        "context_target_transform": agg_info["context_target_transform"],
        "context_target_plot_space": plot_space,
        "context_target_zscore_eps": agg_info["context_target_zscore_eps"],
        "context_target_zscore_fallback_rows": agg_info["context_target_zscore_fallback_rows"],
        "prediction_transform": pred_tinfo["prediction_transform"],
        "prediction_transform_mean": pred_tinfo["prediction_transform_mean"],
        "prediction_transform_std": pred_tinfo["prediction_transform_std"],
        "heldout_protocols_requested": _resolve_heldout_protocols(args.heldout_protocols),
        "heldout_best_per_protocol": bool(args.heldout_best_per_protocol),
        "heldout_protocol_metric_map": _resolve_heldout_metric_map(args.heldout_protocol_metrics),
        "heldout_plot_spaces_requested": _resolve_heldout_plot_spaces(args.heldout_plot_spaces),
        "heldout_color_by": str(args.heldout_color_by),
        "heldout_shape_by": str(args.heldout_shape_by),
        "heldout_centroid_by": str(args.heldout_centroid_by),
        "heldout_ellipse_by": str(args.heldout_ellipse_by),
        "heldout_ellipse_n_std": float(args.heldout_ellipse_n_std),
        "heldout_ellipse_min_points": int(args.heldout_ellipse_min_points),
        "heldout_ellipse_face_alpha": float(args.heldout_ellipse_face_alpha),
        "heldout_ellipse_edge_alpha": float(args.heldout_ellipse_edge_alpha),
        "heldout_ellipse_equal_area": bool(args.heldout_ellipse_equal_area),
        "heldout_ellipse_only": bool(args.heldout_ellipse_only),
        "heldout_fit_lines_by": str(args.heldout_fit_lines_by or ""),
        "heldout_fit_lines_protocols": _resolve_heldout_protocols(args.heldout_fit_lines_protocols),
        "heldout_fit_lines_min_points": int(args.heldout_fit_lines_min_points),
        "heldout_fit_lines_max_groups": int(args.heldout_fit_lines_max_groups),
        "heldout_fit_lines_show_legend": bool(args.heldout_fit_lines_show_legend),
        "heldout_center_benchmark": bool(args.heldout_center_benchmark),
        "heldout_center_benchmark_protocols": _resolve_heldout_protocols(
            args.heldout_center_benchmark_protocols
        ),
        "heldout_single_context": _parse_kv_csv(args.heldout_single_context),
        "heldout_collapse_aggregation": str(args.heldout_collapse_aggregation),
        "heldout_collapse_group_cols_requested": str(args.heldout_collapse_group_cols),
        "heldout_rank_plots": bool(args.heldout_rank_plots),
        "heldout_rank_group_cols_requested": str(args.heldout_rank_group_cols),
        "heldout_rank_min_group_size": int(args.heldout_rank_min_group_size),
        "heldout_rank_min_unique_values": int(args.heldout_rank_min_unique_values),
        "heldout_approx_unique_abs_tol": float(args.heldout_approx_unique_abs_tol),
        "heldout_approx_unique_rel_tol": float(args.heldout_approx_unique_rel_tol),
        "heldout_dispersion_warn_std_ratio": float(args.heldout_dispersion_warn_std_ratio),
        "heldout_dispersion_warn_unique_frac": float(args.heldout_dispersion_warn_unique_frac),
        "rankfirst_min_dispersion_ratio": float(args.rankfirst_min_dispersion_ratio),
        "heldout_model_cv_dir": str(args.heldout_model_cv_dir or ""),
        "heldout_model_cv_head": str(args.heldout_model_cv_head or ""),
        "raw_rmse": float(rmse),
        "raw_pearson": float(pear),
        "raw_spearman": float(spear),
        "selected_run_from_root": str(selection_info["run_dir"]) if selection_info else None,
        "selection_metric_key": str(selection_info["metric_key"]) if selection_info else None,
        "selection_metric_value": float(selection_info["metric_value"]) if selection_info else None,
        "selection_metric_column": str(selection_info["metric_col_used"]) if selection_info else None,
        "selection_n_candidates": int(selection_info["n_candidates"]) if selection_info else None,
        "predictor_grid_enabled": bool(grid_plot_enabled),
        "predictor_grid_only": bool(args.predictor_grid_only),
        "predictor_grid_specs_raw": [str(x) for x in predictor_grid_specs_raw],
        "predictor_grid_specs_resolved": [
            {
                "spec": str(row.get("spec", "")),
                "label": str(row.get("label", "")),
                "predictors": [str(x) for x in row.get("predictors", [])],
                "n_predictors": int(row.get("n_predictors") or 0),
            }
            for row in predictor_grid_specs_resolved
        ],
        "predictor_grid_output": str(predictor_grid_output),
        "predictor_grid_columns": int(predictor_grid_plot_columns),
        "predictor_grid_family_layout": bool(args.predictor_grid_family_layout),
        "predictor_grid_spec_selection": str(predictor_grid_spec_selection),
        "predictor_grid_best_score": str(predictor_grid_best_score),
        "predictor_grid_best_max_combos": int(predictor_grid_best_max_combos),
        "predictor_grid_generated": bool(predictor_grid_generated),
        "predictor_grid_zscore_enabled": bool(args.predictor_grid_zscore),
        "predictor_grid_zscore_output": str(predictor_grid_zscore_output),
        "predictor_grid_zscore_generated": bool(predictor_grid_zscore_generated),
        "predictor_grid_warnings": [str(x) for x in predictor_grid_warnings],
    }
    metrics_json.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True))

    residual_plots = []
    aggregated_plots = []
    train_median_plot = None
    density_hexbin_plot = None
    train_benchmark_median_iqr_plot = None
    color_maps = []
    if not bool(args.predictor_grid_only):
        # write one residual plot per requested color mode
        for color_mode in color_modes:
            color_values = None
            color_map_csv = None
            color_suffix = ""
            if len(color_modes) > 1:
                color_suffix = f"__{color_mode or 'none'}"
            if color_mode and color_mode in fit_df.columns:
                color_values = fit_df[color_mode]
                color_map_csv = out_dir / f"residual_fit_color_map{color_suffix}.csv"
                color_maps.append(color_map_csv)
            residual_plot = out_dir / f"residual_fit_scatter{color_suffix}.png"
            _plot_residual_scatter(
                y_true_base,
                y_pred_plot,
                rmse_plot,
                pear_plot,
                spear_plot,
                residual_plot,
                color_values=color_values,
                color_label=color_mode if color_values is not None else None,
                color_map_csv=color_map_csv,
                identifier_text=plot_identifier_for_figure,
                x_label=x_label,
                y_label=y_label,
                title_base=title_base,
                axis_clip_quantile=axis_clip_quantile,
                axis_pad_frac=axis_pad_frac,
                axis_match_xy_limits=axis_match_xy_limits,
                show_fit_line=show_fit_line,
                show_diagnostics=show_fit_diagnostics,
                show_identity_line=show_identity_line,
                show_title=show_title,
                legend_ncol=legend_ncol,
                show_color_in_title=show_color_in_title,
                tight_bbox=tight_bbox,
                pretty_dataset_labels=pretty_dataset_labels,
                synthetic_label=synthetic_label,
                marker_size=marker_size,
                point_alpha=point_alpha,
                font_scale=font_scale,
                legend_font_scale=legend_font_scale,
                color_saturation=color_saturation,
                mix_label_style=mix_label_style,
                center_axes_zero=center_axes_zero,
            )
            residual_plots.append(residual_plot)

            # Also write an aggregated view: one point per unique predictor feature row.
            feat_cols = [f"{p}__z" for p in coef_df["predictor"].tolist() if f"{p}__z" in fit_df.columns]
            if feat_cols:
                agg_work = fit_df.copy()
                agg_work["observed_residual"] = y_true_base
                agg_work["predicted_residual"] = y_pred_plot
                color_col = color_mode if (color_mode and color_mode in agg_work.columns) else None
                group_cols = feat_cols + ([color_col] if color_col else [])
                agg_df = (
                    agg_work.groupby(group_cols, dropna=False)
                    .agg(
                        observed_mean=("observed_residual", "mean"),
                        predicted_mean=("predicted_residual", "mean"),
                        n_rep=("observed_residual", "size"),
                    )
                    .reset_index()
                )
                agg_df["_color"] = agg_df[color_col].astype(str) if color_col else "all"
                agg_plot = out_dir / f"residual_fit_scatter_aggregated{color_suffix}.png"
                _plot_residual_scatter_aggregated(
                    agg_df=agg_df,
                    out_path=agg_plot,
                    color_label=color_col or "group",
                    identifier_text=plot_identifier_for_figure,
                    title="Aggregated residual fit (unique feature rows)",
                    xlabel=agg_x_label,
                    ylabel=agg_y_label,
                    axis_clip_quantile=axis_clip_quantile,
                    show_fit_line=show_fit_line,
                    show_diagnostics=show_fit_diagnostics,
                    show_identity_line=show_identity_line,
                    legend_ncol=legend_ncol,
                    tight_bbox=tight_bbox,
                    pretty_dataset_labels=pretty_dataset_labels,
                    synthetic_label=synthetic_label,
                    marker_size=max(marker_size + 4.0, marker_size * 1.15),
                    font_scale=font_scale,
                    color_saturation=color_saturation,
                    mix_label_style=mix_label_style,
                    center_axes_zero=center_axes_zero,
                )
                aggregated_plots.append(agg_plot)

    # Extra view: collapse to one point per train_dataset using medians.
    if not bool(args.predictor_grid_only) and "train_dataset" in fit_df.columns:
        train_work = fit_df.copy()
        train_work["observed_residual"] = y_true_base
        train_work["predicted_residual"] = y_pred_plot
        train_med_df = (
            train_work.groupby("train_dataset", dropna=False)
            .agg(
                observed_mean=("observed_residual", "median"),
                predicted_mean=("predicted_residual", "median"),
                n_rep=("observed_residual", "size"),
            )
            .reset_index()
        )
        train_med_df["_color"] = train_med_df["train_dataset"].astype(str)
        train_median_plot = out_dir / "residual_fit_scatter_train_dataset_median.png"
        _plot_residual_scatter_aggregated(
            agg_df=train_med_df,
            out_path=train_median_plot,
            color_label="train_dataset",
            identifier_text=plot_identifier_for_figure,
            title="Aggregated residual fit (median by train_dataset)",
            xlabel="Predicted residual (train_dataset median)",
            ylabel="Observed residual (train_dataset median)",
            axis_clip_quantile=axis_clip_quantile,
            show_fit_line=show_fit_line,
            show_diagnostics=show_fit_diagnostics,
            show_identity_line=show_identity_line,
            legend_ncol=legend_ncol,
            tight_bbox=tight_bbox,
            pretty_dataset_labels=pretty_dataset_labels,
            synthetic_label=synthetic_label,
            marker_size=max(marker_size + 4.0, marker_size * 1.15),
            font_scale=font_scale,
            color_saturation=color_saturation,
            mix_label_style=mix_label_style,
            center_axes_zero=center_axes_zero,
        )

    if not bool(args.predictor_grid_only):
        density_hexbin_plot = out_dir / "residual_fit_density_hexbin.png"
        _plot_residual_density_hexbin(
            y_true=y_true_base,
            y_pred=y_pred_plot,
            out_path=density_hexbin_plot,
            identifier_text=plot_identifier_for_figure,
            x_label=x_label,
            y_label=y_label,
            title="Residual fit density (all rows)",
            axis_clip_quantile=axis_clip_quantile,
            show_fit_line=show_fit_line,
            show_identity_line=show_identity_line,
            tight_bbox=tight_bbox,
            center_axes_zero=center_axes_zero,
        )

    if not bool(args.predictor_grid_only) and "train_dataset" in fit_df.columns and "benchmark" in fit_df.columns:
        train_benchmark_median_iqr_plot = out_dir / "residual_fit_scatter_train_benchmark_median_iqr.png"
        _plot_train_benchmark_median_iqr(
            fit_df=fit_df,
            y_true=y_true_base,
            y_pred=y_pred_plot,
            out_path=train_benchmark_median_iqr_plot,
            identifier_text=plot_identifier_for_figure,
            x_label=x_label,
            y_label=y_label,
            title="Train×benchmark medians with IQR (color=train_dataset)",
            axis_clip_quantile=axis_clip_quantile,
            show_fit_line=show_fit_line,
            show_diagnostics=show_fit_diagnostics,
            show_identity_line=show_identity_line,
            legend_ncol=legend_ncol,
            tight_bbox=tight_bbox,
            pretty_dataset_labels=pretty_dataset_labels,
            synthetic_label=synthetic_label,
            marker_size=max(marker_size + 5.0, marker_size * 1.20),
            font_scale=font_scale,
            color_saturation=color_saturation,
            mix_label_style=mix_label_style,
            center_axes_zero=center_axes_zero,
        )

    # Side-by-side ideal reference: perfect prediction (y_pred == y_true), colored two ways.
    perfect_plot = out_dir / "perfect_prediction_side_by_side.png"
    if not bool(args.predictor_grid_only):
        _plot_perfect_side_by_side(
            y_true=y_true_base,
            train_values=fit_df["train_dataset"] if "train_dataset" in fit_df.columns else None,
            benchmark_values=fit_df["benchmark"] if "benchmark" in fit_df.columns else None,
            out_path=perfect_plot,
            identifier_text=plot_identifier_for_figure,
            show_identity_line=show_identity_line,
            legend_ncol=legend_ncol,
            tight_bbox=tight_bbox,
            pretty_dataset_labels=pretty_dataset_labels,
            synthetic_label=synthetic_label,
        )

    rank_plot = None

    single_context_plot = None
    context_kv = _parse_kv_csv(args.single_context)
    if context_kv:
        ctx = fit_df.copy()
        ctx["y"] = y_true_base
        ctx["yhat"] = y_pred_plot
        for k, v in context_kv.items():
            if k not in ctx.columns:
                raise ValueError(f"--single-context key not in data columns: {k}")
            ctx = ctx[ctx[k].astype(str) == str(v)]
        if ctx.empty:
            raise ValueError(
                "No rows matched --single-context filter: "
                + ", ".join([f"{k}={v}" for k, v in context_kv.items()])
            )
        context_desc = ", ".join([f"{k}={v}" for k, v in context_kv.items()])
        safe_name = "__".join([f"{k}_{v}" for k, v in context_kv.items()])
        safe_name = safe_name.replace("/", "_").replace(" ", "_")
        single_context_plot = out_dir / f"single_context_scatter__{safe_name}.png"
        _plot_single_context_scatter(
            df=ctx,
            out_path=single_context_plot,
            context_desc=context_desc,
            color_col=(args.single_context_color_by or "").strip() or None,
            identifier_text=plot_identifier_for_figure,
            axis_clip_quantile=axis_clip_quantile,
            show_fit_line=show_fit_line,
            show_diagnostics=show_fit_diagnostics,
            show_identity_line=show_identity_line,
            legend_ncol=legend_ncol,
            tight_bbox=tight_bbox,
            pretty_dataset_labels=pretty_dataset_labels,
            synthetic_label=synthetic_label,
        )

    heldout_protocol_plots: list[Path] = []
    heldout_protocol_points_csvs: list[Path] = []
    heldout_protocol_skipped: list[str] = []
    heldout_protocol_warnings: list[str] = []
    heldout_protocol_metrics_csv: Path | None = None
    heldout_metric_rows: list[dict[str, object]] = []
    heldout_rank_plots: list[Path] = []
    heldout_rank_rows_csvs: list[Path] = []
    heldout_protocols_requested = _resolve_heldout_protocols(args.heldout_protocols)
    heldout_protocols, heldout_protocols_collapsed = _collapse_overlapping_heldout_protocols(
        heldout_protocols_requested
    )
    heldout_center_protocols = set()
    if bool(args.heldout_center_benchmark):
        heldout_center_protocols = set(
            _resolve_heldout_protocols(args.heldout_center_benchmark_protocols)
        )
    heldout_plot_spaces = _resolve_heldout_plot_spaces(args.heldout_plot_spaces)
    heldout_fit_lines_by = str(args.heldout_fit_lines_by or "").strip()
    heldout_fit_line_protocols = set(_resolve_heldout_protocols(args.heldout_fit_lines_protocols))
    heldout_fit_line_min_points = max(int(args.heldout_fit_lines_min_points), 2)
    heldout_fit_line_max_groups = max(int(args.heldout_fit_lines_max_groups), 0)
    heldout_collapse_mode = str(args.heldout_collapse_aggregation or "none").strip().lower()
    heldout_collapse_group_cols = _parse_csv_list(args.heldout_collapse_group_cols)
    if len(heldout_collapse_group_cols) == 1 and heldout_collapse_group_cols[0].strip().lower() == "auto":
        heldout_collapse_group_cols = ["train_dataset", "benchmark"]
    heldout_metric_map = _resolve_heldout_metric_map(args.heldout_protocol_metrics)
    heldout_context_kv = _parse_kv_csv(args.heldout_single_context)
    heldout_context_desc = ", ".join([f"{k}={v}" for k, v in heldout_context_kv.items()])
    heldout_context_suffix = _safe_context_suffix(heldout_context_kv)
    heldout_rank_group_cols_requested = _parse_csv_list(args.heldout_rank_group_cols)
    if len(heldout_rank_group_cols_requested) == 1 and heldout_rank_group_cols_requested[0].strip().lower() == "auto":
        heldout_rank_group_cols_requested = []
    heldout_model_cv_dir = Path(args.heldout_model_cv_dir) if str(args.heldout_model_cv_dir).strip() else None
    heldout_selection_root = selection_root
    protocol_best_cache: dict[str, dict[str, object]] = {}
    context_stats_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    if heldout_model_cv_dir is not None and not (heldout_model_cv_dir / "heldout_model_cv_pred_rows.csv").exists():
        heldout_protocol_warnings.append(
            "heldout_model_cv_pred_rows.csv not found under "
            f"{heldout_model_cv_dir}; model/training/eval/triple/trainset_disjoint scatters need this file "
            "(rerun heldout model CV with --save-pred-rows)."
        )
    for msg in heldout_protocols_collapsed:
        heldout_protocol_warnings.append(f"Auto-collapsed heldout protocol: {msg}")
    if heldout_protocols:
        for protocol in heldout_protocols:
            protocol_run_dir = run_dir
            protocol_metric_key = heldout_metric_map.get(protocol)
            protocol_metric_value = None
            if args.heldout_best_per_protocol and protocol_metric_key:
                if _is_model_cv_protocol(protocol):
                    if heldout_model_cv_dir is None:
                        heldout_protocol_warnings.append(
                            f"Cannot best-select heldout-model-CV protocol '{protocol}' "
                            "without --heldout-model-cv-dir."
                        )
                else:
                    if heldout_selection_root is None:
                        heldout_protocol_warnings.append(
                            f"Cannot best-select per protocol '{protocol}' without a run root; "
                            "pass --run-root (or --run-dir as a root path)."
                        )
                    else:
                        if protocol_metric_key not in protocol_best_cache:
                            try:
                                protocol_best_cache[protocol_metric_key] = _select_best_run_dir(
                                    heldout_selection_root,
                                    metric_key=protocol_metric_key,
                                    min_dispersion_ratio=float(args.rankfirst_min_dispersion_ratio),
                                )
                            except Exception as ex:
                                heldout_protocol_warnings.append(
                                    f"Best selection failed for protocol '{protocol}' "
                                    f"metric '{protocol_metric_key}': {ex}"
                                )
                                protocol_best_cache[protocol_metric_key] = {}
                        sel = protocol_best_cache.get(protocol_metric_key, {})
                        if sel.get("run_dir") is not None:
                            protocol_run_dir = Path(sel["run_dir"])
                        if sel.get("metric_value") is not None:
                            protocol_metric_value = float(sel["metric_value"])

            pts, source_desc = _load_heldout_protocol_points(
                run_dir=protocol_run_dir,
                protocol=protocol,
                heldout_model_cv_dir=heldout_model_cv_dir,
                heldout_model_cv_head=args.heldout_model_cv_head,
                heldout_model_cv_metric_key=(
                    protocol_metric_key
                    if args.heldout_best_per_protocol and _is_model_cv_protocol(protocol)
                    else None
                ),
                rankfirst_min_dispersion_ratio=float(args.rankfirst_min_dispersion_ratio),
            )
            if pts.empty:
                heldout_protocol_skipped.append(protocol)
                continue

            for heldout_space in heldout_plot_spaces:
                pts_space, space_note = _project_heldout_points_to_space(
                    pts=pts,
                    protocol=protocol,
                    space=heldout_space,
                    protocol_run_dir=protocol_run_dir,
                    context_stats_cache=context_stats_cache,
                )
                if pts_space.empty:
                    heldout_protocol_skipped.append(f"{protocol}:{heldout_space}")
                    if space_note:
                        heldout_protocol_warnings.append(
                            f"Skipped heldout plot protocol={protocol} space={heldout_space}: {space_note}"
                        )
                    continue

                pts_plot = pts_space
                if heldout_context_kv:
                    missing_context_cols = [k for k in heldout_context_kv.keys() if k not in pts_plot.columns]
                    if missing_context_cols:
                        heldout_protocol_skipped.append(f"{protocol}:{heldout_space}")
                        heldout_protocol_warnings.append(
                            "Skipped heldout plot due to missing heldout context columns "
                            f"(protocol={protocol}, space={heldout_space}, "
                            f"missing={','.join(missing_context_cols)})."
                        )
                        continue
                    for k, v in heldout_context_kv.items():
                        pts_plot = pts_plot[pts_plot[k].astype(str) == str(v)].copy()
                    if pts_plot.empty:
                        heldout_protocol_skipped.append(f"{protocol}:{heldout_space}")
                        heldout_protocol_warnings.append(
                            "Skipped heldout plot due to no rows after heldout context filter "
                            f"(protocol={protocol}, space={heldout_space}, "
                            f"context={heldout_context_desc})."
                        )
                        continue
                centered_by_benchmark = False
                if protocol in heldout_center_protocols:
                    if "benchmark" not in pts_plot.columns:
                        heldout_protocol_warnings.append(
                            "Heldout benchmark-centering requested but 'benchmark' column is missing "
                            f"(protocol={protocol}, space={heldout_space}); using raw values."
                        )
                    else:
                        y_true_center = (
                            pts_plot.groupby("benchmark", dropna=False)["y_true"]
                            .transform("mean")
                            .to_numpy(dtype=float)
                        )
                        y_pred_center = (
                            pts_plot.groupby("benchmark", dropna=False)["y_pred"]
                            .transform("mean")
                            .to_numpy(dtype=float)
                        )
                        pts_plot = pts_plot.copy()
                        pts_plot["y_true"] = pd.to_numeric(pts_plot["y_true"], errors="coerce") - y_true_center
                        pts_plot["y_pred"] = pd.to_numeric(pts_plot["y_pred"], errors="coerce") - y_pred_center
                        pts_plot = (
                            pts_plot.replace([np.inf, -np.inf], np.nan)
                            .dropna(subset=["y_true", "y_pred"])
                            .copy()
                        )
                        if pts_plot.empty:
                            heldout_protocol_skipped.append(f"{protocol}:{heldout_space}")
                            heldout_protocol_warnings.append(
                                "Skipped heldout plot after benchmark-centering removed all rows "
                                f"(protocol={protocol}, space={heldout_space})."
                            )
                            continue
                        centered_by_benchmark = True
                collapse_cols_used: list[str] = []
                rows_before_collapse = int(len(pts_plot))
                rows_after_collapse = rows_before_collapse
                if heldout_collapse_mode in {"mean", "median"}:
                    collapse_cols_used = [c for c in heldout_collapse_group_cols if c in pts_plot.columns]
                    if collapse_cols_used:
                        pts_plot = _collapse_points(
                            pts_plot,
                            group_cols=collapse_cols_used,
                            numeric_agg=heldout_collapse_mode,
                        )
                        rows_after_collapse = int(len(pts_plot))
                    else:
                        heldout_protocol_warnings.append(
                            "Heldout collapse requested but no collapse group columns were found for "
                            f"protocol={protocol} space={heldout_space}; plotting raw rows."
                        )

                y_true_h = pts_plot["y_true"].to_numpy(dtype=float)
                y_pred_h = pts_plot["y_pred"].to_numpy(dtype=float)
                rmse_h = float(np.sqrt(np.mean((y_pred_h - y_true_h) ** 2)))
                pear_h = _pearson(y_true_h, y_pred_h)
                spear_h = _spearman(y_true_h, y_pred_h)
                dispersion_diag = _compute_dispersion_diagnostics(
                    y_true_h,
                    y_pred_h,
                    abs_tol=float(args.heldout_approx_unique_abs_tol),
                    rel_tol=float(args.heldout_approx_unique_rel_tol),
                    std_ratio_warn=float(args.heldout_dispersion_warn_std_ratio),
                    unique_frac_warn=float(args.heldout_dispersion_warn_unique_frac),
                )
                if bool(dispersion_diag.get("dispersion_flag")):
                    heldout_protocol_warnings.append(
                        "Heldout dispersion flagged "
                        f"(protocol={protocol}, space={heldout_space}, "
                        f"std_ratio={dispersion_diag.get('std_ratio')}, "
                        f"pred_unique_frac~={dispersion_diag.get('pred_unique_frac_approx')}, "
                        f"note={dispersion_diag.get('dispersion_note')})"
                    )

                color_col = str(args.heldout_color_by or "").strip()
                color_values = pts_plot[color_col] if color_col and color_col in pts_plot.columns else None
                shape_col = str(args.heldout_shape_by or "").strip()
                shape_values = pts_plot[shape_col] if shape_col and shape_col in pts_plot.columns else None
                centroid_col = str(args.heldout_centroid_by or "").strip()
                centroid_values = (
                    pts_plot[centroid_col]
                    if centroid_col and centroid_col in pts_plot.columns
                    else None
                )
                ellipse_col = str(args.heldout_ellipse_by or "").strip()
                ellipse_values = (
                    pts_plot[ellipse_col]
                    if ellipse_col and ellipse_col in pts_plot.columns
                    else None
                )
                fit_lines_enabled = bool(
                    heldout_fit_lines_by
                    and protocol in heldout_fit_line_protocols
                )
                fit_line_values = None
                fit_line_label = None
                if fit_lines_enabled:
                    if heldout_fit_lines_by in pts_plot.columns:
                        fit_line_values = pts_plot[heldout_fit_lines_by]
                        fit_line_label = heldout_fit_lines_by
                    else:
                        heldout_protocol_warnings.append(
                            "Heldout fit-lines requested but fit-line column is missing "
                            f"(protocol={protocol}, space={heldout_space}, column={heldout_fit_lines_by})."
                        )
                suffix_parts = [f"{heldout_space}_{protocol}"]
                if heldout_context_suffix:
                    suffix_parts.append(heldout_context_suffix)
                plot_suffix = "__".join(suffix_parts).replace("/", "_").replace(" ", "_").replace("-", "_")
                out_plot = out_dir / f"heldout_{plot_suffix}_fit_scatter.png"
                if paper_ready:
                    title = "Heldout Predictions vs Observed Residuals"
                    heldout_x_label = "Predicted Heldout Residual"
                    heldout_y_label = "Observed Residual"
                else:
                    title = f"Heldout-only fit | protocol={protocol} | space={heldout_space}"
                    if source_desc:
                        title += f"\nsource: {source_desc}"
                    if heldout_context_desc:
                        title += f"\ncontext: {heldout_context_desc}"
                    if centered_by_benchmark:
                        title += "\nbenchmark-centered: yes"
                    if heldout_collapse_mode in {"mean", "median"} and collapse_cols_used:
                        title += (
                            f"\ncollapse={heldout_collapse_mode} on "
                            f"{','.join(collapse_cols_used)} ({rows_before_collapse}->{rows_after_collapse})"
                        )
                    heldout_x_label = f"Predicted heldout score ({heldout_space})"
                    heldout_y_label = f"Observed heldout score ({heldout_space})"
                _plot_residual_scatter(
                    y_true=y_true_h,
                    y_pred=y_pred_h,
                    rmse=rmse_h,
                    pear=pear_h,
                    spear=spear_h,
                    out_path=out_plot,
                    color_values=color_values,
                    color_label=(color_col if color_values is not None else None),
                    marker_values=shape_values,
                    marker_label=(shape_col if shape_values is not None else None),
                    centroid_values=centroid_values,
                    centroid_label=(centroid_col if centroid_values is not None else None),
                    ellipse_values=ellipse_values,
                    ellipse_label=(ellipse_col if ellipse_values is not None else None),
                    ellipse_n_std=float(args.heldout_ellipse_n_std),
                    ellipse_min_points=int(args.heldout_ellipse_min_points),
                    ellipse_face_alpha=float(args.heldout_ellipse_face_alpha),
                    ellipse_edge_alpha=float(args.heldout_ellipse_edge_alpha),
                    ellipse_equal_area=bool(args.heldout_ellipse_equal_area),
                    draw_points=(not bool(args.heldout_ellipse_only)),
                    color_map_csv=None,
                    identifier_text=plot_identifier_for_figure,
                    x_label=heldout_x_label,
                    y_label=heldout_y_label,
                    title_base=title,
                    diagnostics=dispersion_diag,
                    fit_line_group_values=fit_line_values,
                    fit_line_group_label=fit_line_label,
                    fit_line_group_min_points=heldout_fit_line_min_points,
                    fit_line_group_max_groups=heldout_fit_line_max_groups,
                    fit_line_group_show_legend=bool(args.heldout_fit_lines_show_legend),
                    axis_clip_quantile=axis_clip_quantile,
                    axis_pad_frac=axis_pad_frac,
                    axis_match_xy_limits=axis_match_xy_limits,
                    show_fit_line=show_fit_line,
                    show_diagnostics=show_fit_diagnostics,
                    show_identity_line=show_identity_line,
                    show_title=show_title,
                    legend_ncol=legend_ncol,
                    show_color_in_title=show_color_in_title,
                    tight_bbox=tight_bbox,
                    pretty_dataset_labels=pretty_dataset_labels,
                    synthetic_label=synthetic_label,
                    marker_size=marker_size,
                    point_alpha=point_alpha,
                    font_scale=font_scale,
                    legend_font_scale=legend_font_scale,
                    color_saturation=color_saturation,
                    mix_label_style=mix_label_style,
                    center_axes_zero=center_axes_zero,
                )
                heldout_protocol_plots.append(out_plot)
                cv_metric_value_from_pts = None
                if "_cv_selected_metric_value" in pts_plot.columns:
                    cv_vals = pd.to_numeric(pts_plot["_cv_selected_metric_value"], errors="coerce")
                    cv_vals = cv_vals[np.isfinite(cv_vals.to_numpy(dtype=float))]
                    if not cv_vals.empty:
                        cv_metric_value_from_pts = float(cv_vals.iloc[0])
                cv_candidate_id = (
                    str(pts_plot["_cv_candidate_id"].iloc[0]) if "_cv_candidate_id" in pts_plot.columns else None
                )
                selected_run_dir_value = str(protocol_run_dir)
                if "path" in pts_plot.columns:
                    path_vals = pts_plot["path"].dropna().astype(str)
                    if not path_vals.empty and str(path_vals.iloc[0]).strip():
                        selected_run_dir_value = str(path_vals.iloc[0]).strip()
                rank_summary = {
                    "rank_n_groups": float("nan"),
                    "rank_spearman_macro": float("nan"),
                    "rank_top1": float("nan"),
                    "rank_abs_rank_pct_error": float("nan"),
                    "rank_group_cols": "",
                    "rank_n_groups_reliable": float("nan"),
                    "rank_reliable_frac": float("nan"),
                    "rank_spearman_macro_reliable": float("nan"),
                    "rank_top1_reliable": float("nan"),
                    "rank_abs_rank_pct_error_reliable": float("nan"),
                    "rank_group_size_min": float("nan"),
                    "rank_group_size_median": float("nan"),
                    "rank_group_size_max": float("nan"),
                    "rank_small_group_frac": float("nan"),
                    "rank_pred_tied_group_frac": float("nan"),
                    "rank_true_tied_group_frac": float("nan"),
                    "rank_pred_unique_approx_median": float("nan"),
                    "rank_true_unique_approx_median": float("nan"),
                    "rank_reliability_warning": "",
                }
                if bool(args.heldout_rank_plots):
                    rank_group_cols = _resolve_rank_group_cols_for_points(
                        pts_plot,
                        requested_group_cols=heldout_rank_group_cols_requested,
                    )
                    rank_rows_df, rank_summary = _build_rank_alignment_rows(
                        pts_plot,
                        true_col="y_true",
                        pred_col="y_pred",
                        group_cols=rank_group_cols,
                        min_group_size=max(int(args.heldout_rank_min_group_size), 2),
                        min_group_unique_values=max(int(args.heldout_rank_min_unique_values), 1),
                        approx_unique_abs_tol=float(args.heldout_approx_unique_abs_tol),
                        approx_unique_rel_tol=float(args.heldout_approx_unique_rel_tol),
                    )
                    rank_warn = str(rank_summary.get("rank_reliability_warning", "") or "").strip()
                    if rank_warn:
                        heldout_protocol_warnings.append(
                            "Heldout rank reliability warning "
                            f"(protocol={protocol}, space={heldout_space}, warning={rank_warn}, "
                            f"groups={rank_summary.get('rank_n_groups')}, "
                            f"reliable_groups={rank_summary.get('rank_n_groups_reliable')})"
                        )
                    if not rank_rows_df.empty:
                        rank_plot = out_dir / f"heldout_{plot_suffix}_rank_alignment_scatter.png"
                        rank_title = (
                            f"Heldout ranking alignment | protocol={protocol} | space={heldout_space}"
                        )
                        if rank_summary.get("rank_group_cols"):
                            rank_title += f"\nrank groups: {rank_summary['rank_group_cols']}"
                        _plot_rank_error_scatter(
                            rank_rows_df,
                            rank_plot,
                            rank_title,
                            identifier_text=plot_identifier_for_figure,
                            summary_stats=rank_summary,
                        )
                        heldout_rank_plots.append(rank_plot)
                        if args.heldout_save_points:
                            rank_rows_csv = out_dir / f"heldout_{plot_suffix}_rank_alignment_rows.csv"
                            keep_rank_cols = [
                                c
                                for c in [
                                    "_rank_group",
                                    "benchmark",
                                    "train_dataset",
                                    "model_family_encoder",
                                    "fold",
                                    "fold_id",
                                    "y_true",
                                    "y_pred",
                                    "true_rank",
                                    "pred_rank",
                                    "true_rank_pct",
                                    "pred_rank_pct",
                                    "rank_error",
                                    "abs_rank_pct_error",
                                    "_rank_group_size",
                                    "_rank_group_pred_unique_approx",
                                    "_rank_group_true_unique_approx",
                                    "_rank_group_is_reliable",
                                ]
                                if c in rank_rows_df.columns
                            ]
                            rank_rows_df[keep_rank_cols].to_csv(rank_rows_csv, index=False)
                            heldout_rank_rows_csvs.append(rank_rows_csv)
                heldout_metric_rows.append(
                    {
                        "protocol": protocol,
                        "space": heldout_space,
                        "heldout_context": heldout_context_desc if heldout_context_desc else None,
                        "benchmark_centered": bool(centered_by_benchmark),
                        "n_rows": int(rows_after_collapse),
                        "n_rows_before_collapse": int(rows_before_collapse),
                        "n_rows_after_collapse": int(rows_after_collapse),
                        "collapse_aggregation": heldout_collapse_mode,
                        "collapse_group_cols": ",".join(collapse_cols_used),
                        "rmse": float(rmse_h),
                        "pearson": float(pear_h),
                        "spearman": float(spear_h),
                        "y_true_std": dispersion_diag.get("y_true_std"),
                        "y_pred_std": dispersion_diag.get("y_pred_std"),
                        "std_ratio": dispersion_diag.get("std_ratio"),
                        "y_true_range": dispersion_diag.get("y_true_range"),
                        "y_pred_range": dispersion_diag.get("y_pred_range"),
                        "range_ratio": dispersion_diag.get("range_ratio"),
                        "y_true_iqr": dispersion_diag.get("y_true_iqr"),
                        "y_pred_iqr": dispersion_diag.get("y_pred_iqr"),
                        "iqr_ratio": dispersion_diag.get("iqr_ratio"),
                        "pred_unique_approx": dispersion_diag.get("pred_unique_approx"),
                        "pred_unique_frac_approx": dispersion_diag.get("pred_unique_frac_approx"),
                        "pred_tie_frac_approx": dispersion_diag.get("pred_tie_frac_approx"),
                        "dispersion_flag": dispersion_diag.get("dispersion_flag"),
                        "dispersion_note": dispersion_diag.get("dispersion_note"),
                        "color_by": color_col if color_col else None,
                        "source": source_desc,
                        "y_true_col": (
                            str(pts_plot["_y_true_col"].iloc[0]) if "_y_true_col" in pts_plot.columns else None
                        ),
                        "y_pred_col": (
                            str(pts_plot["_y_pred_col"].iloc[0]) if "_y_pred_col" in pts_plot.columns else None
                        ),
                        "best_cv_metric": protocol_metric_key if args.heldout_best_per_protocol else None,
                        "best_cv_metric_value": (
                            cv_metric_value_from_pts
                            if cv_metric_value_from_pts is not None
                            else protocol_metric_value
                        ),
                        "selected_run_dir": selected_run_dir_value,
                        "selected_candidate_id": cv_candidate_id,
                        "rank_n_groups": rank_summary.get("rank_n_groups"),
                        "rank_spearman_macro": rank_summary.get("rank_spearman_macro"),
                        "rank_top1": rank_summary.get("rank_top1"),
                        "rank_abs_rank_pct_error": rank_summary.get("rank_abs_rank_pct_error"),
                        "rank_group_cols": rank_summary.get("rank_group_cols"),
                        "rank_n_groups_reliable": rank_summary.get("rank_n_groups_reliable"),
                        "rank_reliable_frac": rank_summary.get("rank_reliable_frac"),
                        "rank_spearman_macro_reliable": rank_summary.get("rank_spearman_macro_reliable"),
                        "rank_top1_reliable": rank_summary.get("rank_top1_reliable"),
                        "rank_abs_rank_pct_error_reliable": rank_summary.get(
                            "rank_abs_rank_pct_error_reliable"
                        ),
                        "rank_group_size_min": rank_summary.get("rank_group_size_min"),
                        "rank_group_size_median": rank_summary.get("rank_group_size_median"),
                        "rank_group_size_max": rank_summary.get("rank_group_size_max"),
                        "rank_small_group_frac": rank_summary.get("rank_small_group_frac"),
                        "rank_pred_tied_group_frac": rank_summary.get("rank_pred_tied_group_frac"),
                        "rank_true_tied_group_frac": rank_summary.get("rank_true_tied_group_frac"),
                        "rank_pred_unique_approx_median": rank_summary.get(
                            "rank_pred_unique_approx_median"
                        ),
                        "rank_true_unique_approx_median": rank_summary.get(
                            "rank_true_unique_approx_median"
                        ),
                        "rank_reliability_warning": rank_summary.get("rank_reliability_warning"),
                    }
                )

                if args.heldout_save_points:
                    keep_cols = [
                        c
                        for c in [
                            "benchmark",
                            "train_dataset",
                            "model_family_encoder",
                            "fold",
                            "fold_id",
                            "candidate_id",
                            "head",
                            "protocol",
                            "_cv_protocols",
                            "_cv_candidate_id",
                            "_cv_selected_metric",
                            "_cv_selected_metric_value",
                            "_cv_selection_note",
                            "y_true",
                            "y_pred",
                            "_heldout_space",
                            "_source_file",
                            "_y_true_col",
                            "_y_pred_col",
                        ]
                        if c in pts_plot.columns
                    ]
                    out_points = out_dir / f"heldout_{plot_suffix}_fit_points.csv"
                    pts_plot[keep_cols].to_csv(out_points, index=False)
                    heldout_protocol_points_csvs.append(out_points)

        if heldout_metric_rows:
            heldout_protocol_metrics_csv = out_dir / "heldout_protocol_fit_metrics.csv"
            pd.DataFrame(heldout_metric_rows).to_csv(heldout_protocol_metrics_csv, index=False)

    metrics_payload["heldout_protocols_resolved"] = [str(x) for x in heldout_protocols]
    metrics_payload["heldout_protocols_collapsed"] = [str(x) for x in heldout_protocols_collapsed]
    metrics_payload["heldout_plot_spaces_resolved"] = [str(x) for x in heldout_plot_spaces]
    metrics_payload["heldout_center_benchmark_resolved"] = bool(args.heldout_center_benchmark)
    metrics_payload["heldout_center_benchmark_protocols_resolved"] = sorted(list(heldout_center_protocols))
    metrics_payload["heldout_centroid_by_resolved"] = str(args.heldout_centroid_by or "")
    metrics_payload["heldout_ellipse_by_resolved"] = str(args.heldout_ellipse_by or "")
    metrics_payload["heldout_ellipse_n_std_resolved"] = float(args.heldout_ellipse_n_std)
    metrics_payload["heldout_ellipse_min_points_resolved"] = int(args.heldout_ellipse_min_points)
    metrics_payload["heldout_ellipse_face_alpha_resolved"] = float(args.heldout_ellipse_face_alpha)
    metrics_payload["heldout_ellipse_edge_alpha_resolved"] = float(args.heldout_ellipse_edge_alpha)
    metrics_payload["heldout_ellipse_equal_area_resolved"] = bool(args.heldout_ellipse_equal_area)
    metrics_payload["heldout_ellipse_only_resolved"] = bool(args.heldout_ellipse_only)
    metrics_payload["heldout_fit_lines_by_resolved"] = str(heldout_fit_lines_by or "")
    metrics_payload["heldout_fit_lines_protocols_resolved"] = sorted(list(heldout_fit_line_protocols))
    metrics_payload["heldout_fit_lines_min_points_resolved"] = int(heldout_fit_line_min_points)
    metrics_payload["heldout_fit_lines_max_groups_resolved"] = int(heldout_fit_line_max_groups)
    metrics_payload["heldout_fit_lines_show_legend_resolved"] = bool(args.heldout_fit_lines_show_legend)
    metrics_payload["heldout_single_context_resolved"] = heldout_context_kv
    metrics_payload["heldout_collapse_aggregation_resolved"] = heldout_collapse_mode
    metrics_payload["heldout_collapse_group_cols_resolved"] = [str(x) for x in heldout_collapse_group_cols]
    metrics_payload["heldout_protocol_plots"] = [str(x) for x in heldout_protocol_plots]
    metrics_payload["heldout_protocol_points_csvs"] = [str(x) for x in heldout_protocol_points_csvs]
    metrics_payload["heldout_rank_plots"] = [str(x) for x in heldout_rank_plots]
    metrics_payload["heldout_rank_rows_csvs"] = [str(x) for x in heldout_rank_rows_csvs]
    metrics_payload["heldout_protocol_metrics_csv"] = (
        str(heldout_protocol_metrics_csv) if heldout_protocol_metrics_csv is not None else None
    )
    metrics_payload["heldout_protocol_skipped"] = [str(x) for x in heldout_protocol_skipped]
    metrics_payload["heldout_protocol_warnings"] = [str(x) for x in heldout_protocol_warnings]
    metrics_payload["heldout_protocol_metrics_rows"] = [
        {
            k: (
                float(v)
                if isinstance(v, (np.floating, float))
                else int(v)
                if isinstance(v, (np.integer, int))
                else str(v)
                if isinstance(v, Path)
                else v
            )
            for k, v in row.items()
        }
        for row in heldout_metric_rows
    ]
    metrics_json.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True))

    selection_meta_json = None
    selection_readme_txt = None
    if selection_info is not None:
        selection_meta_json = out_dir / "best_cv_selection_metadata.json"
        selection_payload = {
            "selection_mode": "root_auto_select",
            "selection_root": str(selection_root) if selection_root is not None else None,
            "selected_run_dir": str(selection_info["run_dir"]),
            "selection_metric_key": str(selection_info["metric_key"]),
            "selection_metric_value": float(selection_info["metric_value"]),
            "selection_metric_column": str(selection_info["metric_col_used"]),
            "selection_direction": "max" if bool(selection_info["maximize"]) else "min",
            "selection_summary_path": str(selection_info["summary_path"]),
            "selection_n_candidates": int(selection_info["n_candidates"]),
            "plot_output_dir": str(out_dir),
            "rank_detail_file": str(args.rank_detail_file),
            "resolved_target": str(target),
            "resolved_predictors": [str(x) for x in predictors],
            "resolved_context_cols": [str(x) for x in context_cols],
            "resolved_ridge_alpha": float(ridge_alpha),
            "resolved_top_k": int(max(int(args.top_k), 0)),
            "resolved_context_target_plot_space": plot_space,
            "paper_ready": bool(paper_ready),
            "pretty_dataset_labels": bool(pretty_dataset_labels),
            "paper_synthetic_label": str(synthetic_label),
            "axis_clip_quantile": float(axis_clip_quantile),
            "axis_pad_frac": float(axis_pad_frac),
            "axis_match_xy_limits": bool(axis_match_xy_limits),
            "show_fit_diagnostics": bool(show_fit_diagnostics),
            "show_fit_line": bool(show_fit_line),
            "show_identity_line": bool(show_identity_line),
            "show_title": bool(show_title),
            "legend_ncol": int(legend_ncol) if legend_ncol is not None else None,
            "show_color_in_title": bool(show_color_in_title),
            "tight_bbox": bool(tight_bbox),
            "marker_size": float(marker_size),
            "point_alpha": (float(point_alpha) if point_alpha is not None else None),
            "font_scale": float(font_scale),
            "legend_font_scale": float(legend_font_scale),
            "color_saturation": float(color_saturation),
            "mix_label_style": str(mix_label_style),
            "repeat_aggregation_applied": str(agg_info["repeat_aggregation"]),
            "repeat_group_cols_applied": [str(x) for x in agg_info["repeat_group_cols"]],
            "full_fit_rmse": float(rmse_plot),
            "full_fit_pearson": float(pear_plot),
            "full_fit_spearman": float(spear_plot),
            "predictor_grid_enabled": bool(grid_plot_enabled),
            "predictor_grid_only": bool(args.predictor_grid_only),
            "predictor_grid_specs_raw": [str(x) for x in predictor_grid_specs_raw],
            "predictor_grid_specs_resolved": [
                {
                    "spec": str(row.get("spec", "")),
                    "label": str(row.get("label", "")),
                    "predictors": [str(x) for x in row.get("predictors", [])],
                    "n_predictors": int(row.get("n_predictors") or 0),
                }
                for row in predictor_grid_specs_resolved
            ],
            "predictor_grid_output": str(predictor_grid_output),
            "predictor_grid_columns": int(predictor_grid_plot_columns),
            "predictor_grid_family_layout": bool(args.predictor_grid_family_layout),
            "predictor_grid_spec_selection": str(predictor_grid_spec_selection),
            "predictor_grid_best_score": str(predictor_grid_best_score),
            "predictor_grid_best_max_combos": int(predictor_grid_best_max_combos),
            "predictor_grid_generated": bool(predictor_grid_generated),
            "predictor_grid_zscore_enabled": bool(args.predictor_grid_zscore),
            "predictor_grid_zscore_output": str(predictor_grid_zscore_output),
            "predictor_grid_zscore_generated": bool(predictor_grid_zscore_generated),
            "predictor_grid_warnings": [str(x) for x in predictor_grid_warnings],
            "heldout_protocols_requested": [str(x) for x in heldout_protocols_requested],
            "heldout_protocols_resolved": [str(x) for x in heldout_protocols],
            "heldout_protocols_collapsed": [str(x) for x in heldout_protocols_collapsed],
            "heldout_plot_spaces": [str(x) for x in heldout_plot_spaces],
            "heldout_best_per_protocol": bool(args.heldout_best_per_protocol),
            "heldout_protocol_metric_map": {str(k): str(v) for k, v in heldout_metric_map.items()},
            "heldout_model_cv_dir": str(heldout_model_cv_dir) if heldout_model_cv_dir is not None else None,
            "heldout_model_cv_head": str(args.heldout_model_cv_head),
            "heldout_center_benchmark": bool(args.heldout_center_benchmark),
            "heldout_center_benchmark_protocols": sorted(list(heldout_center_protocols)),
            "heldout_centroid_by": str(args.heldout_centroid_by or ""),
            "heldout_ellipse_by": str(args.heldout_ellipse_by or ""),
            "heldout_ellipse_n_std": float(args.heldout_ellipse_n_std),
            "heldout_ellipse_min_points": int(args.heldout_ellipse_min_points),
            "heldout_ellipse_face_alpha": float(args.heldout_ellipse_face_alpha),
            "heldout_ellipse_edge_alpha": float(args.heldout_ellipse_edge_alpha),
            "heldout_ellipse_equal_area": bool(args.heldout_ellipse_equal_area),
            "heldout_ellipse_only": bool(args.heldout_ellipse_only),
            "heldout_fit_lines_by": str(heldout_fit_lines_by),
            "heldout_fit_lines_protocols": sorted(list(heldout_fit_line_protocols)),
            "heldout_fit_lines_min_points": int(heldout_fit_line_min_points),
            "heldout_fit_lines_max_groups": int(heldout_fit_line_max_groups),
            "heldout_fit_lines_show_legend": bool(args.heldout_fit_lines_show_legend),
            "heldout_single_context": heldout_context_kv,
            "heldout_collapse_aggregation": heldout_collapse_mode,
            "heldout_collapse_group_cols": [str(x) for x in heldout_collapse_group_cols],
            "heldout_rank_plots": bool(args.heldout_rank_plots),
            "heldout_rank_group_cols": [str(x) for x in heldout_rank_group_cols_requested],
            "heldout_rank_min_group_size": int(args.heldout_rank_min_group_size),
            "heldout_rank_min_unique_values": int(args.heldout_rank_min_unique_values),
            "heldout_approx_unique_abs_tol": float(args.heldout_approx_unique_abs_tol),
            "heldout_approx_unique_rel_tol": float(args.heldout_approx_unique_rel_tol),
            "heldout_dispersion_warn_std_ratio": float(args.heldout_dispersion_warn_std_ratio),
            "heldout_dispersion_warn_unique_frac": float(args.heldout_dispersion_warn_unique_frac),
            "rankfirst_min_dispersion_ratio": float(args.rankfirst_min_dispersion_ratio),
            "heldout_protocol_metrics_csv": (
                str(heldout_protocol_metrics_csv) if heldout_protocol_metrics_csv is not None else None
            ),
            "heldout_rank_plots_files": [str(x) for x in heldout_rank_plots],
            "heldout_rank_rows_csv_files": [str(x) for x in heldout_rank_rows_csvs],
            "heldout_protocol_rows": heldout_metric_rows,
            "heldout_protocol_skipped": [str(x) for x in heldout_protocol_skipped],
            "heldout_protocol_warnings": [str(x) for x in heldout_protocol_warnings],
        }
        selection_meta_json.write_text(json.dumps(selection_payload, indent=2, sort_keys=True))

        selection_readme_txt = out_dir / "best_cv_selection_README.txt"
        rerun_cmd = (
            "python scripts/plot_residual_fit_and_rank_errors.py "
            f"--run-dir {selection_root} "
            f"--best-cv-metric {args.best_cv_metric}"
        )
        rerun_cmd += f" --rankfirst-min-dispersion-ratio {float(args.rankfirst_min_dispersion_ratio)}"
        rerun_cmd += f" --context-target-transform {args.context_target_transform}"
        rerun_cmd += f" --context-target-plot-space {args.context_target_plot_space}"
        rerun_cmd += f" --prediction-transform {args.prediction_transform}"
        rerun_cmd += f" --prediction-transform-eps {args.prediction_transform_eps}"
        if paper_ready:
            rerun_cmd += " --paper-ready"
        if args.pretty_dataset_labels:
            rerun_cmd += " --pretty-dataset-labels"
        if str(synthetic_label).strip() and str(synthetic_label).strip() != "SDF-Fractal3D":
            rerun_cmd += f" --paper-synthetic-label {synthetic_label}"
        if float(axis_clip_quantile) > 0:
            rerun_cmd += f" --axis-clip-quantile {float(axis_clip_quantile)}"
        rerun_cmd += f" --axis-pad-frac {float(axis_pad_frac)}"
        if not bool(axis_match_xy_limits):
            rerun_cmd += " --axis-independent-limits"
        rerun_cmd += f" --marker-size {float(marker_size)}"
        if point_alpha is not None:
            rerun_cmd += f" --point-alpha {float(point_alpha)}"
        rerun_cmd += f" --font-scale {float(font_scale)}"
        rerun_cmd += f" --legend-font-scale {float(legend_font_scale)}"
        rerun_cmd += f" --color-saturation {float(color_saturation)}"
        if str(mix_label_style).strip():
            rerun_cmd += f" --mix-label-style {mix_label_style}"
        if bool(args.hide_fit_diagnostics):
            rerun_cmd += " --hide-fit-diagnostics"
        if bool(args.hide_fit_line):
            rerun_cmd += " --hide-fit-line"
        if bool(args.hide_title):
            rerun_cmd += " --hide-title"
        if bool(args.tight_bbox):
            rerun_cmd += " --tight-bbox"
        if str(args.context_cols or "").strip():
            rerun_cmd += f" --context-cols {args.context_cols}"
        if str(args.repeat_aggregation or "").strip():
            rerun_cmd += f" --repeat-aggregation {args.repeat_aggregation}"
        if str(args.repeat_group_cols or "").strip():
            rerun_cmd += f" --repeat-group-cols {args.repeat_group_cols}"
        if str(args.color_by or "").strip():
            rerun_cmd += f" --color-by {args.color_by}"
        if bool(args.predictor_grid):
            rerun_cmd += " --predictor-grid"
            if predictor_grid_specs_raw:
                rerun_cmd += f" --predictor-grid-spec {','.join(predictor_grid_specs_raw)}"
            rerun_cmd += f" --predictor-grid-columns {max(int(args.predictor_grid_columns), 1)}"
            rerun_cmd += f" --predictor-grid-spec-selection {predictor_grid_spec_selection}"
            rerun_cmd += f" --predictor-grid-best-score {predictor_grid_best_score}"
            rerun_cmd += f" --predictor-grid-best-max-combos {int(predictor_grid_best_max_combos)}"
            rerun_cmd += f" --predictor-grid-output-name {str(args.predictor_grid_output_name).strip() or 'residual_fit_predictor_grid'}"
            if bool(args.predictor_grid_family_layout):
                rerun_cmd += " --predictor-grid-family-layout"
            if bool(args.predictor_grid_zscore):
                rerun_cmd += " --predictor-grid-zscore"
                rerun_cmd += f" --predictor-grid-zscore-output-name {str(args.predictor_grid_zscore_output_name).strip() or 'residual_fit_predictor_grid_zscore'}"
            if bool(args.predictor_grid_only):
                rerun_cmd += " --predictor-grid-only"
        if str(args.output_dir or "").strip():
            rerun_cmd += f" --output-dir {args.output_dir}"
        if heldout_protocols:
            rerun_cmd += f" --heldout-protocols {','.join(heldout_protocols_requested)}"
            rerun_cmd += f" --heldout-plot-spaces {','.join(heldout_plot_spaces)}"
            if str(args.heldout_color_by or "").strip():
                rerun_cmd += f" --heldout-color-by {args.heldout_color_by}"
            if str(args.heldout_shape_by or "").strip():
                rerun_cmd += f" --heldout-shape-by {args.heldout_shape_by}"
            if str(args.heldout_centroid_by or "").strip():
                rerun_cmd += f" --heldout-centroid-by {args.heldout_centroid_by}"
            if str(args.heldout_ellipse_by or "").strip():
                rerun_cmd += f" --heldout-ellipse-by {args.heldout_ellipse_by}"
                rerun_cmd += f" --heldout-ellipse-n-std {float(args.heldout_ellipse_n_std)}"
                rerun_cmd += f" --heldout-ellipse-min-points {int(args.heldout_ellipse_min_points)}"
                rerun_cmd += f" --heldout-ellipse-face-alpha {float(args.heldout_ellipse_face_alpha)}"
                rerun_cmd += f" --heldout-ellipse-edge-alpha {float(args.heldout_ellipse_edge_alpha)}"
                if bool(args.heldout_ellipse_equal_area):
                    rerun_cmd += " --heldout-ellipse-equal-area"
                if bool(args.heldout_ellipse_only):
                    rerun_cmd += " --heldout-ellipse-only"
            if bool(args.heldout_center_benchmark):
                rerun_cmd += " --heldout-center-benchmark"
                rerun_cmd += (
                    f" --heldout-center-benchmark-protocols "
                    f"{args.heldout_center_benchmark_protocols}"
                )
            if heldout_fit_lines_by:
                rerun_cmd += f" --heldout-fit-lines-by {heldout_fit_lines_by}"
                rerun_cmd += (
                    f" --heldout-fit-lines-protocols "
                    f"{','.join(sorted(list(heldout_fit_line_protocols)))}"
                )
                rerun_cmd += f" --heldout-fit-lines-min-points {int(heldout_fit_line_min_points)}"
                rerun_cmd += f" --heldout-fit-lines-max-groups {int(heldout_fit_line_max_groups)}"
                if bool(args.heldout_fit_lines_show_legend):
                    rerun_cmd += " --heldout-fit-lines-show-legend"
            if heldout_context_desc:
                rerun_cmd += f" --heldout-single-context {args.heldout_single_context}"
            rerun_cmd += f" --heldout-collapse-aggregation {heldout_collapse_mode}"
            if heldout_collapse_group_cols:
                rerun_cmd += f" --heldout-collapse-group-cols {','.join(heldout_collapse_group_cols)}"
            if args.heldout_rank_plots:
                rerun_cmd += " --heldout-rank-plots"
            if str(args.heldout_rank_group_cols or "").strip():
                rerun_cmd += f" --heldout-rank-group-cols {args.heldout_rank_group_cols}"
            if args.heldout_best_per_protocol:
                rerun_cmd += " --heldout-best-per-protocol"
            if str(args.heldout_protocol_metrics or "").strip():
                rerun_cmd += f" --heldout-protocol-metrics {args.heldout_protocol_metrics}"
            if heldout_model_cv_dir is not None:
                rerun_cmd += f" --heldout-model-cv-dir {heldout_model_cv_dir}"
                rerun_cmd += f" --heldout-model-cv-head {args.heldout_model_cv_head}"
            if args.heldout_save_points:
                rerun_cmd += " --heldout-save-points"
            rerun_cmd += f" --heldout-rank-min-group-size {int(args.heldout_rank_min_group_size)}"
            rerun_cmd += f" --heldout-rank-min-unique-values {int(args.heldout_rank_min_unique_values)}"
            rerun_cmd += f" --heldout-approx-unique-abs-tol {float(args.heldout_approx_unique_abs_tol)}"
            rerun_cmd += f" --heldout-approx-unique-rel-tol {float(args.heldout_approx_unique_rel_tol)}"
            rerun_cmd += f" --heldout-dispersion-warn-std-ratio {float(args.heldout_dispersion_warn_std_ratio)}"
            rerun_cmd += f" --heldout-dispersion-warn-unique-frac {float(args.heldout_dispersion_warn_unique_frac)}"

        readme_lines = [
            "Best-CV selection reference",
            "",
            f"Selection root: {selection_root}",
            f"Selected run: {selection_info['run_dir']}",
            f"Selection metric: {selection_info['metric_key']}",
            f"Metric column used: {selection_info['metric_col_used']}",
            f"Metric value: {selection_info['metric_value']:.6f}",
            f"Direction: {'maximize' if bool(selection_info['maximize']) else 'minimize'}",
            f"Summary file: {selection_info['summary_path']}",
            f"Candidates scanned: {selection_info['n_candidates']}",
            "",
            "How this plot was run:",
            rerun_cmd,
        ]

        if heldout_protocols:
            readme_lines += [
                "",
                "Heldout protocol selection:",
                f"- Requested protocols: {','.join(heldout_protocols_requested)}",
                f"- Resolved protocols: {','.join(heldout_protocols)}",
                f"- Requested spaces: {','.join(heldout_plot_spaces)}",
                f"- Heldout color by: {args.heldout_color_by if str(args.heldout_color_by).strip() else '(none)'}",
                f"- Heldout shape by: {args.heldout_shape_by if str(args.heldout_shape_by).strip() else '(none)'}",
                f"- Heldout centroid by: {args.heldout_centroid_by if str(args.heldout_centroid_by).strip() else '(none)'}",
                f"- Heldout ellipse by: {args.heldout_ellipse_by if str(args.heldout_ellipse_by).strip() else '(none)'}",
                f"- Heldout ellipse n-std: {float(args.heldout_ellipse_n_std)}",
                f"- Heldout ellipse min points: {int(args.heldout_ellipse_min_points)}",
                f"- Heldout ellipse face alpha: {float(args.heldout_ellipse_face_alpha)}",
                f"- Heldout ellipse edge alpha: {float(args.heldout_ellipse_edge_alpha)}",
                f"- Heldout ellipse equal area: {'yes' if bool(args.heldout_ellipse_equal_area) else 'no'}",
                f"- Heldout ellipse only: {'yes' if bool(args.heldout_ellipse_only) else 'no'}",
                f"- Benchmark-centering: {'yes' if bool(args.heldout_center_benchmark) else 'no'}",
                f"- Benchmark-centering protocols: {','.join(sorted(list(heldout_center_protocols)))}",
                f"- Heldout fit-lines by: {heldout_fit_lines_by if heldout_fit_lines_by else '(none)'}",
                f"- Heldout fit-line protocols: {','.join(sorted(list(heldout_fit_line_protocols)))}",
                f"- Heldout fit-line min points: {int(heldout_fit_line_min_points)}",
                f"- Heldout fit-line max groups: {int(heldout_fit_line_max_groups)}",
                f"- Heldout fit-line legend: {'yes' if bool(args.heldout_fit_lines_show_legend) else 'no'}",
                f"- Heldout context filter: {heldout_context_desc if heldout_context_desc else '(none)'}",
                f"- Heldout collapse: {heldout_collapse_mode}",
                f"- Heldout collapse group cols: {','.join(heldout_collapse_group_cols)}",
                f"- Heldout rank plots: {'yes' if args.heldout_rank_plots else 'no'}",
                f"- Heldout rank group cols: {','.join(heldout_rank_group_cols_requested) if heldout_rank_group_cols_requested else 'auto'}",
                f"- Heldout rank min group size: {int(args.heldout_rank_min_group_size)}",
                f"- Heldout rank min unique values: {int(args.heldout_rank_min_unique_values)}",
                f"- Heldout approx unique abs tol: {float(args.heldout_approx_unique_abs_tol)}",
                f"- Heldout approx unique rel tol: {float(args.heldout_approx_unique_rel_tol)}",
                f"- Heldout dispersion warn std ratio: {float(args.heldout_dispersion_warn_std_ratio)}",
                f"- Heldout dispersion warn unique frac: {float(args.heldout_dispersion_warn_unique_frac)}",
                f"- Rank-first min dispersion ratio: {float(args.rankfirst_min_dispersion_ratio)}",
                f"- Best per protocol: {'yes' if args.heldout_best_per_protocol else 'no'}",
                f"- Protocol metric map: {heldout_metric_map}",
                f"- Heldout model CV dir: {heldout_model_cv_dir if heldout_model_cv_dir is not None else ''}",
                f"- Heldout protocol metrics CSV: {heldout_protocol_metrics_csv if heldout_protocol_metrics_csv is not None else ''}",
            ]
            if heldout_protocols_collapsed:
                readme_lines.append(
                    f"- Auto-collapsed overlaps: {', '.join(heldout_protocols_collapsed)}"
                )
            if heldout_metric_rows:
                readme_lines.append("- Per-protocol resolved selections:")
                for row in heldout_metric_rows:
                    readme_lines.append(
                        "  "
                        + (
                            f"{row.get('protocol')} | space={row.get('space') or ''} "
                            f"| metric={row.get('best_cv_metric') or ''} "
                            f"| metric_value={row.get('best_cv_metric_value') if row.get('best_cv_metric_value') is not None else ''} "
                            f"| run={row.get('selected_run_dir') or ''} "
                            f"| candidate_id={row.get('selected_candidate_id') or ''} "
                            f"| source={row.get('source') or ''} "
                            f"| y={row.get('y_true_col')}/{row.get('y_pred_col')} "
                            f"| n={row.get('n_rows')} "
                            f"| std_ratio={row.get('std_ratio') if row.get('std_ratio') is not None else ''} "
                            f"| pred_unique_frac~={row.get('pred_unique_frac_approx') if row.get('pred_unique_frac_approx') is not None else ''} "
                            f"| rank_groups={row.get('rank_n_groups') if row.get('rank_n_groups') is not None else ''} "
                            f"| rank_groups_reliable={row.get('rank_n_groups_reliable') if row.get('rank_n_groups_reliable') is not None else ''} "
                            f"| rank_spear_macro={row.get('rank_spearman_macro') if row.get('rank_spearman_macro') is not None else ''} "
                            f"| rank_top1={row.get('rank_top1') if row.get('rank_top1') is not None else ''}"
                        )
                    )
            if heldout_protocol_skipped:
                readme_lines.append(f"- Skipped protocols: {','.join(heldout_protocol_skipped)}")
            for warn in heldout_protocol_warnings:
                readme_lines.append(f"- Warning: {warn}")

        readme_lines += [
            "",
            "Output files:",
            f"- {out_dir / 'residual_fit_scatter*.png'}",
            f"- {out_dir / 'residual_fit_scatter_aggregated*.png'}",
            f"- {out_dir / 'residual_fit_scatter_train_dataset_median.png'}",
            f"- {out_dir / 'residual_fit_density_hexbin.png'}",
            f"- {out_dir / 'residual_fit_scatter_train_benchmark_median_iqr.png'}",
            f"- {out_dir / 'full_fit_top_coefficients.csv'}",
            f"- {out_dir / 'residual_fit_metrics.json'}",
            f"- {out_dir / 'perfect_prediction_side_by_side.png'}",
        ]
        if grid_plot_enabled and predictor_grid_generated:
            grid_output_name = str(Path(predictor_grid_output).name)
            readme_lines.append(f"- {out_dir / grid_output_name}")
        if grid_plot_enabled and predictor_grid_zscore_generated:
            z_output_name = str(Path(predictor_grid_zscore_output).name)
            readme_lines.append(f"- {out_dir / z_output_name}")
        selection_readme_txt.write_text("\n".join(readme_lines) + "\n")

    if bool(args.predictor_grid) and predictor_grid_generated:
        print(f"Wrote: {predictor_grid_output}")
    if bool(args.predictor_grid) and args.predictor_grid_zscore and predictor_grid_zscore_generated:
        print(f"Wrote: {predictor_grid_zscore_output}")
    for p in residual_plots:
        print(f"Wrote: {p}")
    for p in aggregated_plots:
        print(f"Wrote: {p}")
    if train_median_plot is not None:
        print(f"Wrote: {train_median_plot}")
    if density_hexbin_plot is not None:
        print(f"Wrote: {density_hexbin_plot}")
    if train_benchmark_median_iqr_plot is not None:
        print(f"Wrote: {train_benchmark_median_iqr_plot}")
    if rank_plot is not None:
        print(f"Wrote: {rank_plot}")
    print(f"Wrote: {coef_csv}")
    print(f"Wrote: {metrics_json}")
    if not bool(args.predictor_grid_only):
        print(f"Wrote: {perfect_plot}")
    if selection_info:
        direction = "max" if bool(selection_info["maximize"]) else "min"
        print(
            "Auto-selected run: "
            f"{selection_info['run_dir']} "
            f"({selection_info['metric_key']}={selection_info['metric_value']:.6f}, "
            f"column={selection_info['metric_col_used']}, direction={direction}, "
            f"candidates={selection_info['n_candidates']})"
        )
    print(
        "Full-fit plot metrics: "
        f"rmse={rmse_plot:.4f} pearson={pear_plot:+.6f} spearman={spear_plot:+.6f}"
    )
    if str(args.prediction_transform).strip().lower() != "none":
        print(
            "Prediction transform: "
            f"{pred_tinfo['prediction_transform']} "
            f"(mean={pred_tinfo['prediction_transform_mean']:.6f}, "
            f"std={pred_tinfo['prediction_transform_std']:.6f})"
        )
    if agg_info["repeat_aggregation"] != "none":
        print(
            "Repeat aggregation: "
            f"{agg_info['repeat_aggregation']} on {agg_info['repeat_group_cols']} "
            f"({agg_info['rows_before_repeat_agg']} -> {agg_info['rows_after_repeat_agg']})"
        )
    if agg_info["context_target_transform"] == "zscore":
        print(
            "Context target z-score active: "
            f"eps={agg_info['context_target_zscore_eps']} "
            f"fallback_rows={agg_info['context_target_zscore_fallback_rows']}"
        )
    for p in color_maps:
        print(f"Wrote: {p}")
    for p in heldout_protocol_plots:
        print(f"Wrote: {p}")
    for p in heldout_protocol_points_csvs:
        print(f"Wrote: {p}")
    for p in heldout_rank_plots:
        print(f"Wrote: {p}")
    for p in heldout_rank_rows_csvs:
        print(f"Wrote: {p}")
    if heldout_protocol_metrics_csv is not None:
        print(f"Wrote: {heldout_protocol_metrics_csv}")
    if heldout_protocol_skipped:
        print(
            "Heldout protocol plots skipped (missing usable detail rows): "
            + ", ".join(heldout_protocol_skipped)
        )
    for msg in heldout_protocol_warnings:
        print(f"Warning: {msg}")
    for msg in predictor_grid_warnings:
        print(f"Warning: {msg}")
    if single_context_plot is not None:
        print(f"Wrote: {single_context_plot}")
    if selection_meta_json is not None:
        print(f"Wrote: {selection_meta_json}")
    if selection_readme_txt is not None:
        print(f"Wrote: {selection_readme_txt}")


if __name__ == "__main__":
    main()
