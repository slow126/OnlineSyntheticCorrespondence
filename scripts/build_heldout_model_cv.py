#!/usr/bin/env python3
"""
Held-out model-group CV on existing leakage-free row tables.

Protocols:
  1) model_only: leave-one-model-group-out
  2) model_train_benchmark: leave-one-(model_group, train_dataset, benchmark)-out
  3) model_benchmark: leave-one-(model_group, benchmark)-out (all train datasets)
  4) model_train_benchmark_disjoint: test one (model_group, train_dataset, benchmark)
     cell; train on rows with different model_group AND train_dataset AND benchmark.
  5) model_benchmark_trainset_disjoint: test one (model_group, benchmark) context
     with a held-out subset T of train datasets (|T|=k); train on rows with
     different model_group AND different benchmark AND train_dataset not in T.

Heads:
  - ols
  - ridge
  - pairwise_rank (pairwise logistic on within-benchmark option pairs)

This runs retroactively from existing prediction_jointood_rows.csv files and
does not rerun correspondence training.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


CONTROL_PREFIXES = (
    "log_n_samples_",
    "log_avg_flows_",
    "n_samples_",
    "avg_flows_",
    "enc_",
    "mf_",
)

TOKEN_EQUIV_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"^flow_train_to_eval_auc$"), "flow_train_to_eval_quantile"),
    (re.compile(r"^flow_eval_to_train_auc$"), "flow_eval_to_train_quantile"),
    (re.compile(r"^flow_train_to_eval_eps_at\d+$"), "flow_train_to_eval_quantile"),
    (re.compile(r"^flow_eval_to_train_eps_at\d+$"), "flow_eval_to_train_quantile"),
]


@dataclass
class EvalResult:
    summary: Dict[str, object]
    pred_rows: pd.DataFrame
    fold_rows: pd.DataFrame


def _split_predictors(text: object) -> List[str]:
    if text is None:
        return []
    raw = str(text).strip()
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _canonical_token(token: str) -> str:
    t = token.strip()
    for pat, rep in TOKEN_EQUIV_PATTERNS:
        if pat.match(t):
            return rep
    return t


def _signal_tokens(predictors: Sequence[str]) -> List[str]:
    out: List[str] = []
    for tok in predictors:
        if tok.startswith(CONTROL_PREFIXES):
            continue
        out.append(_canonical_token(tok))
    return sorted(out)


def _token_lane_counts(tokens: Sequence[str]) -> Tuple[int, int]:
    motion = 0
    appearance = 0
    for t in tokens:
        if t.startswith("flow_") or t.startswith("hof_") or t == "flow_mmd":
            motion += 1
        elif t.startswith("dino_") or t in {"dino_mmd", "feature_mmd"}:
            appearance += 1
    return motion, appearance


def _lane_from_tokens(tokens: Sequence[str]) -> str:
    m, a = _token_lane_counts(tokens)
    if m > 0 and a == 0:
        return "motion_only"
    if a > 0 and m == 0:
        return "appearance_only"
    if m > 0 and a > 0:
        return "hybrid"
    return "other"


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return math.nan
    x = x[finite]
    y = y[finite]
    x = x - np.mean(x)
    y = y - np.mean(y)
    x_scale = float(np.max(np.abs(x))) if x.size else 0.0
    y_scale = float(np.max(np.abs(y))) if y.size else 0.0
    if x_scale == 0.0 or y_scale == 0.0:
        return math.nan
    x = x / x_scale
    y = y / y_scale
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0:
        return math.nan
    return float(np.dot(x, y) / denom)


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return math.nan
    rx = pd.Series(x[finite]).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y[finite]).rank(method="average").to_numpy(dtype=float)
    return _pearson_corr(rx, ry)


def _kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return math.nan
    xx = np.asarray(x[finite], dtype=float)
    yy = np.asarray(y[finite], dtype=float)
    n = int(len(xx))
    if n < 2:
        return math.nan

    n_conc = 0
    n_disc = 0
    n_tie_x = 0
    n_tie_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xx[i] - xx[j]
            dy = yy[i] - yy[j]
            tie_x = dx == 0.0
            tie_y = dy == 0.0
            if tie_x:
                n_tie_x += 1
            if tie_y:
                n_tie_y += 1
            if tie_x or tie_y:
                continue
            if dx * dy > 0.0:
                n_conc += 1
            elif dx * dy < 0.0:
                n_disc += 1

    n0 = n * (n - 1) / 2.0
    denom = math.sqrt(max(n0 - n_tie_x, 0.0) * max(n0 - n_tie_y, 0.0))
    if denom == 0.0:
        return math.nan
    return float((n_conc - n_disc) / denom)


def _pairwise_cindex(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.sum() < 2:
        return math.nan
    yt = np.asarray(y_true[finite], dtype=float)
    yp = np.asarray(y_pred[finite], dtype=float)
    n = int(len(yt))
    if n < 2:
        return math.nan

    comparable = 0
    score = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dy_true = yt[i] - yt[j]
            if dy_true == 0.0:
                continue
            comparable += 1
            dy_pred = yp[i] - yp[j]
            if dy_pred == 0.0:
                score += 0.5
            elif dy_true * dy_pred > 0.0:
                score += 1.0
    if comparable == 0:
        return math.nan
    return float(score / comparable)


def _mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.sum() == 0:
        return math.nan, math.nan
    diff = y_true[finite] - y_pred[finite]
    mae = float(np.mean(np.abs(diff)))
    scale = float(np.max(np.abs(diff))) if diff.size else 0.0
    if not np.isfinite(scale):
        rmse = math.nan
    elif scale == 0.0:
        rmse = 0.0
    else:
        rmse = float(scale * np.sqrt(np.mean((diff / scale) ** 2)))
    return mae, rmse


def _fit_context_target_residualizer(
    train_df: pd.DataFrame,
    target_col: str,
    context_cols: Sequence[str],
    transform: str = "residual",
    std_eps: float = 1e-9,
) -> Optional[Dict[str, object]]:
    if train_df is None or train_df.empty:
        return None
    cols = [c for c in context_cols if c in train_df.columns]
    if not cols:
        return None
    mode = str(transform or "residual").strip().lower()
    if mode not in {"residual", "zscore"}:
        mode = "residual"

    mean_col = "__target_context_mean__"
    means = (
        train_df.groupby(cols, dropna=False)[target_col]
        .mean()
        .reset_index()
        .rename(columns={target_col: mean_col})
    )
    std_col = "__target_context_std__"
    stds = None
    global_std = float(np.nanstd(train_df[target_col].to_numpy(dtype=float), ddof=0))
    if not np.isfinite(global_std) or global_std <= float(std_eps):
        global_std = 1.0
    if mode == "zscore":
        stds = (
            train_df.groupby(cols, dropna=False)[target_col]
            .std(ddof=0)
            .reset_index()
            .rename(columns={target_col: std_col})
        )
    global_mean = float(np.nanmean(train_df[target_col].to_numpy(dtype=float)))
    return {
        "context_cols": cols,
        "transform": mode,
        "std_eps": float(std_eps),
        "mean_col": mean_col,
        "std_col": std_col,
        "means": means,
        "stds": stds,
        "global_mean": global_mean,
        "global_std": global_std,
    }


def _apply_context_target_residualizer(
    df: pd.DataFrame,
    target_col: str,
    residualizer: Optional[Dict[str, object]],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if df is None:
        return pd.DataFrame(), np.asarray([], dtype=float), np.asarray([], dtype=float)
    if residualizer is None:
        out = df.copy()
        return out, np.zeros(len(out), dtype=float), np.ones(len(out), dtype=float)

    cols = [c for c in residualizer.get("context_cols", []) if c in df.columns]
    if not cols:
        out = df.copy()
        return out, np.zeros(len(out), dtype=float), np.ones(len(out), dtype=float)

    means = residualizer.get("means")
    mean_col = str(residualizer.get("mean_col") or "__target_context_mean__")
    stds = residualizer.get("stds")
    std_col = str(residualizer.get("std_col") or "__target_context_std__")
    global_mean = float(residualizer.get("global_mean", np.nan))
    global_std = float(residualizer.get("global_std", 1.0))
    transform = str(residualizer.get("transform") or "residual").strip().lower()
    std_eps = float(residualizer.get("std_eps", 1e-9))
    if not np.isfinite(global_std) or global_std <= std_eps:
        global_std = 1.0

    if means is None or means.empty:
        out = df.copy()
        offsets = np.full(len(out), global_mean, dtype=float)
        scales = np.full(len(out), global_std if transform == "zscore" else 1.0, dtype=float)
        if transform == "zscore":
            out[target_col] = (out[target_col].to_numpy(dtype=float) - offsets) / scales
        else:
            out[target_col] = out[target_col].to_numpy(dtype=float) - offsets
        return out, offsets, scales

    out = df.copy()
    out["__target_resid_row__"] = np.arange(len(out))
    out = out.merge(means[cols + [mean_col]], on=cols, how="left")
    if transform == "zscore" and stds is not None and not stds.empty:
        out = out.merge(stds[cols + [std_col]], on=cols, how="left")
    out = out.sort_values("__target_resid_row__", kind="mergesort")

    offsets = out[mean_col].fillna(global_mean).to_numpy(dtype=float)
    scales = np.ones(len(out), dtype=float)
    drop_cols = ["__target_resid_row__", mean_col]
    if transform == "zscore":
        if std_col in out.columns:
            scales = out[std_col].fillna(global_std).to_numpy(dtype=float)
        else:
            scales = np.full(len(out), global_std, dtype=float)
        scales = np.where((~np.isfinite(scales)) | (scales <= std_eps), global_std, scales)
        scales = np.where((~np.isfinite(scales)) | (scales <= std_eps), 1.0, scales)
        out[target_col] = (out[target_col].to_numpy(dtype=float) - offsets) / scales
        if std_col in out.columns:
            drop_cols.append(std_col)
    else:
        out[target_col] = out[target_col].to_numpy(dtype=float) - offsets
    out = out.drop(columns=drop_cols, errors="ignore")
    return out, offsets, scales


def _residualize_target_by_context(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    context_cols: Sequence[str],
    transform: str = "residual",
    std_eps: float = 1e-9,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[Dict[str, object]]]:
    train_df = train_df.copy()
    test_df = test_df.copy()
    residualizer = _fit_context_target_residualizer(
        train_df=train_df,
        target_col=target_col,
        context_cols=context_cols,
        transform=transform,
        std_eps=std_eps,
    )
    if residualizer is None:
        return (
            train_df,
            test_df,
            np.zeros(len(train_df), dtype=float),
            np.zeros(len(test_df), dtype=float),
            np.ones(len(train_df), dtype=float),
            np.ones(len(test_df), dtype=float),
            None,
        )
    train_out, train_offsets, train_scales = _apply_context_target_residualizer(
        df=train_df,
        target_col=target_col,
        residualizer=residualizer,
    )
    test_out, test_offsets, test_scales = _apply_context_target_residualizer(
        df=test_df,
        target_col=target_col,
        residualizer=residualizer,
    )
    return (
        train_out,
        test_out,
        train_offsets,
        test_offsets,
        train_scales,
        test_scales,
        residualizer,
    )


def _sample_fewshot_calibration_mask(
    df: pd.DataFrame,
    context_cols: Optional[Sequence[str]] = None,
    k: int = 0,
    rng: Optional[np.random.RandomState] = None,
    allow_backoff: bool = True,
) -> np.ndarray:
    n_rows = int(len(df)) if df is not None else 0
    mask = np.zeros(n_rows, dtype=bool)
    if n_rows <= 1:
        return mask
    k_int = int(k)
    if k_int <= 0:
        return mask

    rng = rng or np.random.RandomState(0)
    cols = [c for c in (context_cols or []) if c in df.columns]
    if cols and bool(allow_backoff):
        for cand_cols in [cols[:kk] for kk in range(len(cols), 0, -1)]:
            counts = (
                df[cand_cols]
                .assign(__row__=1)
                .groupby(cand_cols, dropna=False)["__row__"]
                .sum()
            )
            if not counts.empty and int(counts.max()) > 1:
                cols = cand_cols
                break
    if not cols:
        k_eff = min(k_int, n_rows - 1)
        if k_eff <= 0:
            return mask
        chosen = rng.choice(np.arange(n_rows), size=k_eff, replace=False)
        mask[chosen] = True
        return mask

    work = df[cols].copy()
    work["__row__"] = np.arange(n_rows)
    for _, grp in work.groupby(cols, dropna=False, sort=False):
        rows = grp["__row__"].to_numpy(dtype=int)
        if rows.size <= 1:
            continue
        k_eff = min(k_int, rows.size - 1)
        if k_eff <= 0:
            continue
        chosen = rng.choice(rows, size=k_eff, replace=False)
        mask[chosen] = True
    return mask


def _fit_context_prediction_calibrator(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    context_cols: Optional[Sequence[str]] = None,
    std_eps: float = 1e-9,
    min_group_size: int = 2,
    allow_backoff: bool = True,
) -> Optional[Dict[str, object]]:
    if df is None or len(df) == 0:
        return None
    cols = [c for c in (context_cols or []) if c in df.columns]
    work = df[cols].copy() if cols else pd.DataFrame(index=df.index)
    work["__pred__"] = np.asarray(y_pred, dtype=float)
    work["__true__"] = np.asarray(y_true, dtype=float)
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["__pred__", "__true__"])
    if work.empty:
        return None

    global_pred_mean = float(np.nanmean(work["__pred__"].to_numpy(dtype=float)))
    global_true_mean = float(np.nanmean(work["__true__"].to_numpy(dtype=float)))
    global_pred_std = float(np.nanstd(work["__pred__"].to_numpy(dtype=float), ddof=0))
    global_true_std = float(np.nanstd(work["__true__"].to_numpy(dtype=float), ddof=0))
    if not np.isfinite(global_pred_std) or global_pred_std <= float(std_eps):
        global_pred_std = 1.0
    if not np.isfinite(global_true_std) or global_true_std <= float(std_eps):
        global_true_std = 1.0

    min_n = max(int(min_group_size), 1)
    levels: List[Dict[str, object]] = []
    if cols:
        candidate_cols = [cols]
        if bool(allow_backoff):
            candidate_cols = [cols[:k] for k in range(len(cols), 0, -1)]
        for level_cols in candidate_cols:
            stats = (
                work.groupby(level_cols, dropna=False)
                .agg(
                    pred_mean=("__pred__", "mean"),
                    pred_std=("__pred__", lambda s: float(np.nanstd(s.to_numpy(dtype=float), ddof=0))),
                    true_mean=("__true__", "mean"),
                    true_std=("__true__", lambda s: float(np.nanstd(s.to_numpy(dtype=float), ddof=0))),
                    group_n=("__pred__", "size"),
                )
                .reset_index()
            )
            stats = stats[stats["group_n"] >= min_n].copy()
            if stats.empty:
                continue
            levels.append(
                {
                    "context_cols": list(level_cols),
                    "stats": stats[level_cols + ["pred_mean", "pred_std", "true_mean", "true_std"]],
                }
            )
            if not bool(allow_backoff):
                break
    stats = levels[0]["stats"] if levels else None
    effective_context_cols = levels[0]["context_cols"] if levels else []
    return {
        "context_cols": cols,
        "effective_context_cols": effective_context_cols,
        "levels": levels,
        "stats": stats,
        "global_pred_mean": global_pred_mean,
        "global_true_mean": global_true_mean,
        "global_pred_std": global_pred_std,
        "global_true_std": global_true_std,
        "std_eps": float(std_eps),
        "min_group_size": int(min_n),
        "allow_backoff": bool(allow_backoff),
    }


def _apply_context_prediction_calibrator(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    calibrator: Optional[Dict[str, object]],
) -> np.ndarray:
    pred = np.asarray(y_pred, dtype=float)
    if calibrator is None or df is None or len(pred) == 0:
        return pred

    cols = [c for c in (calibrator.get("context_cols") or []) if c in df.columns]
    std_eps = float(calibrator.get("std_eps", 1e-9))
    gpm = float(calibrator.get("global_pred_mean", 0.0))
    gtm = float(calibrator.get("global_true_mean", 0.0))
    gps = float(calibrator.get("global_pred_std", 1.0))
    gts = float(calibrator.get("global_true_std", 1.0))
    if not np.isfinite(gps) or gps <= std_eps:
        gps = 1.0
    if not np.isfinite(gts) or gts <= std_eps:
        gts = 1.0

    levels = calibrator.get("levels") or []
    n_rows = len(pred)
    pred_mean = np.full(n_rows, gpm, dtype=float)
    true_mean = np.full(n_rows, gtm, dtype=float)
    pred_std = np.full(n_rows, gps, dtype=float)
    true_std = np.full(n_rows, gts, dtype=float)
    assigned = np.zeros(n_rows, dtype=bool)

    # Backward compatibility for calibrators saved before hierarchical levels existed.
    if not levels:
        stats = calibrator.get("stats")
        if cols and stats is not None and len(stats) > 0:
            levels = [{"context_cols": cols, "stats": stats}]

    for level in levels:
        level_cols = [c for c in (level.get("context_cols") or []) if c in df.columns]
        stats = level.get("stats")
        if not level_cols or stats is None or len(stats) == 0:
            continue
        work = df[level_cols].copy()
        work["__pred_row__"] = np.arange(len(work))
        work = work.merge(stats, on=level_cols, how="left")
        work = work.sort_values("__pred_row__", kind="mergesort")
        cand_pred_mean = work["pred_mean"].to_numpy(dtype=float)
        cand_true_mean = work["true_mean"].to_numpy(dtype=float)
        cand_pred_std = work["pred_std"].to_numpy(dtype=float)
        cand_true_std = work["true_std"].to_numpy(dtype=float)
        valid = np.isfinite(cand_pred_mean) & np.isfinite(cand_true_mean)
        take = (~assigned) & valid
        if not np.any(take):
            continue
        pred_mean[take] = cand_pred_mean[take]
        true_mean[take] = cand_true_mean[take]
        pred_std[take] = cand_pred_std[take]
        true_std[take] = cand_true_std[take]
        assigned[take] = True

    pred_std = np.where((~np.isfinite(pred_std)) | (pred_std <= std_eps), gps, pred_std)
    true_std = np.where((~np.isfinite(true_std)) | (true_std <= std_eps), gts, true_std)
    return ((pred - pred_mean) / pred_std) * true_std + true_mean


def _inv_one_plus_exp(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    pos = values >= 0
    if np.any(pos):
        exp_neg = np.exp(-values[pos])
        out[pos] = exp_neg / (1.0 + exp_neg)
    if np.any(~pos):
        exp_pos = np.exp(values[~pos])
        out[~pos] = 1.0 / (1.0 + exp_pos)
    return out


def _fit_linear(
    train_df: pd.DataFrame,
    predictors: Sequence[str],
    target_col: str,
    model: str,
    ridge_alpha: float,
    standardize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = train_df[list(predictors)].to_numpy(dtype=float)
    y = train_df[target_col].to_numpy(dtype=float)
    mean = np.zeros(x.shape[1], dtype=float)
    std = np.ones(x.shape[1], dtype=float)
    if standardize:
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std == 0] = 1.0
        x = (x - mean) / std
    x = np.column_stack([np.ones(len(x)), x])
    if model == "ridge":
        alpha = float(ridge_alpha)
        penalty = np.eye(x.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        coef = np.linalg.solve(x.T @ x + alpha * penalty, x.T @ y)
    else:
        coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return coef, mean, std


def _predict_linear(
    test_df: pd.DataFrame,
    predictors: Sequence[str],
    coef: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    standardize: bool = True,
) -> np.ndarray:
    x = test_df[list(predictors)].to_numpy(dtype=float)
    if standardize:
        x = (x - mean) / std
    x = np.column_stack([np.ones(len(x)), x])
    return x @ coef


def _fit_pairwise_rank(
    train_df: pd.DataFrame,
    predictors: Sequence[str],
    target_col: str,
    group_cols: Sequence[str],
    option_col: str,
    ridge_alpha: float,
    max_pairs_per_group: int,
    max_iter: int,
    lr: float,
    seed: int,
    standardize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gcols = [c for c in group_cols if c]
    agg_cols = list(dict.fromkeys(gcols + [option_col]))
    grouped = (
        train_df.groupby(agg_cols, dropna=False)[list(predictors) + [target_col]]
        .mean()
        .reset_index(drop=False)
    )
    grouped = grouped.dropna(subset=list(predictors) + [target_col]).copy()
    if grouped.empty:
        n = len(predictors)
        return np.zeros(n, dtype=float), np.zeros(n, dtype=float), np.ones(n, dtype=float)

    x = grouped[list(predictors)].to_numpy(dtype=float)
    y = grouped[target_col].to_numpy(dtype=float)

    mean = np.zeros(x.shape[1], dtype=float)
    std = np.ones(x.shape[1], dtype=float)
    if standardize:
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std == 0] = 1.0
        x = (x - mean) / std

    diffs: List[np.ndarray] = []
    labels: List[float] = []
    rng = np.random.default_rng(int(seed))
    work = grouped.reset_index(drop=True).copy()

    if gcols:
        group_iter = work.groupby(gcols, dropna=False)
    else:
        group_iter = [(None, work)]
    for _, sub in group_iter:
        n = len(sub)
        if n < 2:
            continue
        idx = sub.index.to_numpy(dtype=int)
        pairs: List[Tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((idx[i], idx[j]))
        if max_pairs_per_group > 0 and len(pairs) > int(max_pairs_per_group):
            take = rng.choice(len(pairs), size=int(max_pairs_per_group), replace=False)
            pairs = [pairs[t] for t in take]
        for i, j in pairs:
            yi = y[i]
            yj = y[j]
            if yi == yj:
                continue
            label = 1.0 if yi > yj else -1.0
            diffs.append(x[i] - x[j])
            labels.append(label)

    if not diffs:
        n = x.shape[1]
        return np.zeros(n, dtype=float), mean, std

    d = np.asarray(diffs, dtype=float)
    lbl = np.asarray(labels, dtype=float)
    w = np.zeros(d.shape[1], dtype=float)
    reg = float(ridge_alpha)

    for _ in range(int(max_iter)):
        margins = lbl * (d @ w)
        grad = -(lbl[:, None] * d) * _inv_one_plus_exp(margins)[:, None]
        grad = grad.mean(axis=0)
        if reg > 0:
            grad = grad + reg * w
        if not np.all(np.isfinite(grad)):
            break
        norm = float(np.linalg.norm(grad))
        if np.isfinite(norm) and norm > 1e3:
            grad = grad * (1e3 / norm)
        step = float(lr) * grad
        if not np.all(np.isfinite(step)):
            break
        w = w - step
    return w, mean, std


def _predict_pairwise_rank(
    test_df: pd.DataFrame,
    predictors: Sequence[str],
    coef: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    standardize: bool = True,
) -> np.ndarray:
    x = test_df[list(predictors)].to_numpy(dtype=float)
    if standardize:
        x = (x - mean) / std
    return x @ coef


def _compute_ranking_summary(
    pred_df: pd.DataFrame,
    option_col: str,
    benchmark_col: str,
    target_col: str,
    grouping: str = "fold_benchmark",
    context_cols: Optional[Sequence[str]] = None,
    topk_frac: float = 0.2,
    topk_min: int = 1,
) -> Dict[str, float]:
    if pred_df.empty:
        return {
            "rank_n_benchmarks": 0.0,
            "rank_top1": math.nan,
            "rank_top3": math.nan,
            "rank_topk": math.nan,
            "rank_regret": math.nan,
            "rank_spearman": math.nan,
            "rank_kendall_tau": math.nan,
            "rank_pairwise_cindex": math.nan,
            "rank_abs_err": math.nan,
            "rank_pct_err": math.nan,
        }

    required = [benchmark_col, option_col, "prediction", target_col]
    miss = [c for c in required if c not in pred_df.columns]
    if miss:
        return {
            "rank_n_benchmarks": 0.0,
            "rank_top1": math.nan,
            "rank_top3": math.nan,
            "rank_topk": math.nan,
            "rank_regret": math.nan,
            "rank_spearman": math.nan,
            "rank_kendall_tau": math.nan,
            "rank_pairwise_cindex": math.nan,
            "rank_abs_err": math.nan,
            "rank_pct_err": math.nan,
        }

    rows = pred_df.dropna(subset=required).copy()
    if rows.empty:
        return {
            "rank_n_benchmarks": 0.0,
            "rank_top1": math.nan,
            "rank_top3": math.nan,
            "rank_topk": math.nan,
            "rank_regret": math.nan,
            "rank_spearman": math.nan,
            "rank_kendall_tau": math.nan,
            "rank_pairwise_cindex": math.nan,
            "rank_abs_err": math.nan,
            "rank_pct_err": math.nan,
        }

    group_cols: List[str]
    if grouping == "fold_benchmark" and "fold_id" in rows.columns:
        group_cols = ["fold_id", benchmark_col]
    else:
        group_cols = [benchmark_col]
    for col in (context_cols or []):
        if col in rows.columns and col not in group_cols:
            group_cols.append(col)

    per_bench: List[Dict[str, float]] = []
    for _, sub in rows.groupby(group_cols, dropna=False):
        grouped = (
            sub.groupby(option_col, dropna=False)
            .agg(pred_mean=("prediction", "mean"), true_mean=(target_col, "mean"))
            .reset_index(drop=False)
        )
        if len(grouped) < 2:
            continue
        grouped = grouped.sort_values("pred_mean", ascending=False).reset_index(drop=True)
        grouped["rank_true"] = grouped["true_mean"].rank(ascending=False, method="min")
        grouped["rank_pred"] = grouped["pred_mean"].rank(ascending=False, method="min")
        n_opt = int(len(grouped))
        denom = float(max(n_opt - 1, 1))
        true_rank_pct = (grouped["rank_true"] - 1.0) / denom
        pred_rank_pct = (grouped["rank_pred"] - 1.0) / denom

        pred_best = grouped.iloc[0]
        true_best_val = float(grouped["true_mean"].max())
        pred_best_true = float(pred_best["true_mean"])
        pred_best_true_rank = int(pred_best["rank_true"])
        k = int(math.ceil(float(topk_frac) * n_opt))
        k = max(int(topk_min), k)
        k = min(k, n_opt)

        spearman = _spearman_corr(
            grouped["pred_mean"].to_numpy(dtype=float),
            grouped["true_mean"].to_numpy(dtype=float),
        )
        kendall_tau = _kendall_tau_b(
            grouped["pred_mean"].to_numpy(dtype=float),
            grouped["true_mean"].to_numpy(dtype=float),
        )
        pair_cindex = _pairwise_cindex(
            grouped["true_mean"].to_numpy(dtype=float),
            grouped["pred_mean"].to_numpy(dtype=float),
        )
        per_bench.append(
            {
                "top1": float(pred_best_true_rank <= 1),
                "top3": float(pred_best_true_rank <= 3),
                "topk": float(pred_best_true_rank <= k),
                "regret": float(true_best_val - pred_best_true),
                "spearman": spearman,
                "kendall_tau": kendall_tau,
                "pairwise_cindex": pair_cindex,
                "rank_abs_err": float((grouped["rank_pred"] - grouped["rank_true"]).abs().mean()),
                "rank_pct_err": float((pred_rank_pct - true_rank_pct).abs().mean()),
            }
        )

    if not per_bench:
        return {
            "rank_n_benchmarks": 0.0,
            "rank_top1": math.nan,
            "rank_top3": math.nan,
            "rank_topk": math.nan,
            "rank_regret": math.nan,
            "rank_spearman": math.nan,
            "rank_kendall_tau": math.nan,
            "rank_pairwise_cindex": math.nan,
            "rank_abs_err": math.nan,
            "rank_pct_err": math.nan,
        }

    out = pd.DataFrame(per_bench)
    return {
        "rank_n_benchmarks": float(len(out)),
        "rank_top1": float(out["top1"].mean()),
        "rank_top3": float(out["top3"].mean()),
        "rank_topk": float(out["topk"].mean()),
        "rank_regret": float(out["regret"].mean()),
        "rank_spearman": float(out["spearman"].mean()),
        "rank_kendall_tau": float(out["kendall_tau"].mean()),
        "rank_pairwise_cindex": float(out["pairwise_cindex"].mean()),
        "rank_abs_err": float(out["rank_abs_err"].mean()),
        "rank_pct_err": float(out["rank_pct_err"].mean()),
    }


def _model_group_series(df: pd.DataFrame, preferred_col: str) -> pd.Series:
    if preferred_col in df.columns:
        s = df[preferred_col].astype(str)
        if s.notna().any():
            return s
    parts: List[pd.Series] = []
    if "model_family" in df.columns:
        parts.append(df["model_family"].astype(str))
    if "pretrained" in df.columns:
        parts.append(df["pretrained"].astype(str))
    if "freeze" in df.columns:
        parts.append(df["freeze"].astype(str))
    if parts:
        out = parts[0]
        for p in parts[1:]:
            out = out + "__" + p
        return out
    return pd.Series(["unknown"] * len(df), index=df.index, dtype=object)


def _ensure_encoder_config_column(df: pd.DataFrame) -> pd.DataFrame:
    if "encoder_config" in df.columns:
        return df
    if "pretrained" in df.columns and "freeze" in df.columns:
        out = df.copy()
        pre = out["pretrained"].map({True: "T", False: "F"})
        frz = out["freeze"].map({True: "T", False: "F"})
        cfg = pre.fillna("U") + frz.fillna("U")
        out["encoder_config"] = cfg.where(pre.notna() & frz.notna(), "unknown")
        return out
    return df


def _ensure_model_family_encoder_column(df: pd.DataFrame) -> pd.DataFrame:
    if "model_family_encoder" in df.columns:
        return df
    if "model_family" not in df.columns:
        return df
    out = _ensure_encoder_config_column(df)
    out = out.copy()
    if "encoder_config" in out.columns:
        out["model_family_encoder"] = out.apply(
            lambda row: (
                f"{row['model_family']}_{row['encoder_config']}"
                if pd.notna(row.get("encoder_config"))
                and str(row.get("encoder_config")) not in {"", "unknown", "nan"}
                else str(row["model_family"])
            ),
            axis=1,
        )
    else:
        out["model_family_encoder"] = out["model_family"].astype(str)
    return out


def _ensure_option_col(df: pd.DataFrame, option_col: str) -> Tuple[pd.DataFrame, str]:
    if option_col in df.columns:
        return df, option_col

    if option_col == "train_dataset_encoder":
        out = _ensure_encoder_config_column(df)
        if "train_dataset" in out.columns and "encoder_config" in out.columns:
            out = out.copy()
            out["train_dataset_encoder"] = (
                out["train_dataset"].astype(str) + "__" + out["encoder_config"].astype(str)
            )
            return out, "train_dataset_encoder"

    if option_col == "model_family_encoder":
        out = _ensure_model_family_encoder_column(df)
        if "model_family_encoder" in out.columns:
            return out, "model_family_encoder"

    if option_col == "train_dataset_model_family_encoder":
        out = _ensure_model_family_encoder_column(df)
        if "train_dataset" in out.columns and "model_family_encoder" in out.columns:
            out = out.copy()
            out["train_dataset_model_family_encoder"] = (
                out["train_dataset"].astype(str) + "__" + out["model_family_encoder"].astype(str)
            )
            return out, "train_dataset_model_family_encoder"

    return df, option_col


def _resolve_pairwise_group_cols(df: pd.DataFrame, group_cols_csv: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    return _resolve_group_cols(df=df, group_cols_csv=group_cols_csv, default_cols=["benchmark"])


def _resolve_context_cols(df: pd.DataFrame, group_cols_csv: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    return _resolve_group_cols(df=df, group_cols_csv=group_cols_csv, default_cols=[])


def _resolve_group_cols(
    df: pd.DataFrame,
    group_cols_csv: str,
    default_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    raw = [c.strip() for c in str(group_cols_csv).split(",") if c.strip()]
    if not raw and default_cols is not None:
        raw = [str(c).strip() for c in default_cols if str(c).strip()]
    out = df
    resolved: List[str] = []
    missing: List[str] = []
    for col in raw:
        if col in out.columns:
            resolved.append(col)
            continue
        if col == "encoder_config":
            out = _ensure_encoder_config_column(out)
        elif col == "model_family_encoder":
            out = _ensure_model_family_encoder_column(out)
        if col in out.columns:
            resolved.append(col)
        else:
            missing.append(col)
    return out, resolved, missing


def _metric_is_lower_better(metric_name: str) -> bool:
    name = str(metric_name).strip().lower()
    lower_tokens = ("mae", "rmse", "regret", "abs_err", "pct_err", "loss", "error")
    return any(tok in name for tok in lower_tokens)


def _prepare_selection_sort_columns(
    df: pd.DataFrame,
    primary_metric: str,
    tiebreak_metric: str,
) -> Tuple[pd.DataFrame, bool, bool]:
    out = df.copy()
    for col in [primary_metric, tiebreak_metric]:
        if col not in out.columns:
            out[col] = math.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    primary_asc = _metric_is_lower_better(primary_metric)
    tie_asc = _metric_is_lower_better(tiebreak_metric)
    out["_sel_primary"] = out[primary_metric]
    out["_sel_tiebreak"] = out[tiebreak_metric]
    out["_sel_primary"] = out["_sel_primary"].fillna(np.inf if primary_asc else -np.inf)
    out["_sel_tiebreak"] = out["_sel_tiebreak"].fillna(np.inf if tie_asc else -np.inf)
    return out, primary_asc, tie_asc


def _collect_candidate_pool(
    run_roots: Sequence[Path],
    include_pairwise_candidates: bool,
    dedup_primary_metric: str,
    dedup_tiebreak_metric: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for root in run_roots:
        summary_path = root / "method_summary.csv"
        if not summary_path.exists():
            print(f"Warning: missing summary: {summary_path}")
            continue
        try:
            df = pd.read_csv(summary_path)
        except Exception as exc:
            print(f"Warning: failed to read {summary_path}: {exc}")
            continue
        if df.empty:
            continue
        if not include_pairwise_candidates:
            df = df[~df["method"].astype(str).str.endswith("_pairwise")].copy()
        has_rank_sp = pd.to_numeric(df.get("jointood_rank_spearman"), errors="coerce").notna()
        has_rank_c = pd.to_numeric(df.get("jointood_rank_pairwise_cindex"), errors="coerce").notna()
        has_mae = pd.to_numeric(df.get("jointood_mae"), errors="coerce").notna()
        df = df[has_rank_sp | has_rank_c | has_mae].copy()
        if df.empty:
            continue

        for _, r in df.iterrows():
            path = Path(str(r.get("path", "")))
            predictors = _split_predictors(r.get("predictors"))
            if not predictors:
                continue
            signal_tokens = _signal_tokens(predictors)
            lane = _lane_from_tokens(signal_tokens)
            rows.append(
                {
                    "variant": root.name,
                    "method": str(r.get("method", "")),
                    "path": str(path),
                    "predictors": ",".join(predictors),
                    "n_predictors_total": int(len(predictors)),
                    "signal_tokens": ",".join(signal_tokens),
                    "signature": "|".join(signal_tokens),
                    "signal_k": int(len(signal_tokens)),
                    "lane": lane,
                    "jointood_mae": float(r.get("jointood_mae")),
                    "jointood_spearman": float(r.get("jointood_spearman", math.nan)),
                    "jointood_regret": float(r.get("jointood_regret", math.nan)),
                    "jointood_rank_spearman": float(r.get("jointood_rank_spearman", math.nan)),
                    "jointood_rank_kendall_tau": float(r.get("jointood_rank_kendall_tau", math.nan)),
                    "jointood_rank_pairwise_cindex": float(r.get("jointood_rank_pairwise_cindex", math.nan)),
                    "jointood_rank_abs_err": float(r.get("jointood_rank_abs_err", math.nan)),
                    "jointood_rank_pct_err": float(r.get("jointood_rank_pct_err", math.nan)),
                }
            )
    if not rows:
        return pd.DataFrame()
    pool = pd.DataFrame(rows)
    work, pri_asc, tie_asc = _prepare_selection_sort_columns(
        pool,
        primary_metric=dedup_primary_metric,
        tiebreak_metric=dedup_tiebreak_metric,
    )
    work = work.sort_values(
        ["signature", "_sel_primary", "_sel_tiebreak"],
        ascending=[True, pri_asc, tie_asc],
    )
    best = work.groupby("signature", dropna=False).head(1).copy().reset_index(drop=True)
    best = best.drop(columns=["_sel_primary", "_sel_tiebreak"], errors="ignore")
    return best


def _select_candidates(
    pool: pd.DataFrame,
    lanes: Sequence[str],
    top_n_per_lane: int,
    min_signal_k: int,
    max_signal_k: int,
    selection_primary_metric: str,
    selection_tiebreak_metric: str,
) -> pd.DataFrame:
    sel: List[pd.DataFrame] = []
    for lane in lanes:
        sub = pool[
            (pool["lane"] == lane)
            & (pool["signal_k"] >= int(min_signal_k))
            & (pool["signal_k"] <= int(max_signal_k))
        ].copy()
        if sub.empty:
            continue
        work, pri_asc, tie_asc = _prepare_selection_sort_columns(
            sub,
            primary_metric=selection_primary_metric,
            tiebreak_metric=selection_tiebreak_metric,
        )
        sub = work.sort_values(
            ["_sel_primary", "_sel_tiebreak"],
            ascending=[pri_asc, tie_asc],
        ).drop(columns=["_sel_primary", "_sel_tiebreak"], errors="ignore")
        if top_n_per_lane > 0:
            sub = sub.head(int(top_n_per_lane))
        sel.append(sub)
    if not sel:
        return pd.DataFrame()
    out = pd.concat(sel, ignore_index=True)
    out = out.reset_index(drop=True)
    out.insert(0, "candidate_id", [f"c{i+1:03d}" for i in range(len(out))])
    return out


def _build_folds(
    df: pd.DataFrame,
    protocol: str,
    model_group_col: str,
    train_col: str,
    benchmark_col: str,
    holdout_train_k: int,
    max_hard_folds: int,
    rng: np.random.Generator,
) -> List[Tuple[str, np.ndarray, Optional[np.ndarray]]]:
    folds: List[Tuple[str, np.ndarray, Optional[np.ndarray]]] = []
    if protocol == "model_only":
        groups = sorted(set(df[model_group_col].astype(str).dropna().unique()))
        for g in groups:
            mask = df[model_group_col].astype(str) == str(g)
            folds.append((f"model={g}", mask.to_numpy(dtype=bool), None))
        return folds

    if protocol == "model_train_benchmark":
        uniq = (
            df[[model_group_col, train_col, benchmark_col]]
            .dropna()
            .drop_duplicates()
            .copy()
            .reset_index(drop=True)
        )
        if uniq.empty:
            return folds
        if max_hard_folds > 0 and len(uniq) > int(max_hard_folds):
            take = rng.choice(len(uniq), size=int(max_hard_folds), replace=False)
            uniq = uniq.iloc[np.sort(take)].reset_index(drop=True)
        for _, r in uniq.iterrows():
            g = str(r[model_group_col])
            t = str(r[train_col])
            b = str(r[benchmark_col])
            mask = (
                (df[model_group_col].astype(str) == g)
                & (df[train_col].astype(str) == t)
                & (df[benchmark_col].astype(str) == b)
            )
            folds.append((f"model={g}|train={t}|benchmark={b}", mask.to_numpy(dtype=bool), None))
        return folds

    if protocol == "model_benchmark":
        uniq = (
            df[[model_group_col, benchmark_col]]
            .dropna()
            .drop_duplicates()
            .copy()
            .reset_index(drop=True)
        )
        if uniq.empty:
            return folds
        if max_hard_folds > 0 and len(uniq) > int(max_hard_folds):
            take = rng.choice(len(uniq), size=int(max_hard_folds), replace=False)
            uniq = uniq.iloc[np.sort(take)].reset_index(drop=True)
        for _, r in uniq.iterrows():
            g = str(r[model_group_col])
            b = str(r[benchmark_col])
            mask = (
                (df[model_group_col].astype(str) == g)
                & (df[benchmark_col].astype(str) == b)
            )
            folds.append((f"model={g}|benchmark={b}", mask.to_numpy(dtype=bool), None))
        return folds

    if protocol == "model_train_benchmark_disjoint":
        uniq = (
            df[[model_group_col, train_col, benchmark_col]]
            .dropna()
            .drop_duplicates()
            .copy()
            .reset_index(drop=True)
        )
        if uniq.empty:
            return folds
        if max_hard_folds > 0 and len(uniq) > int(max_hard_folds):
            take = rng.choice(len(uniq), size=int(max_hard_folds), replace=False)
            uniq = uniq.iloc[np.sort(take)].reset_index(drop=True)
        mg = df[model_group_col].astype(str)
        tr = df[train_col].astype(str)
        bm = df[benchmark_col].astype(str)
        for _, r in uniq.iterrows():
            g = str(r[model_group_col])
            t = str(r[train_col])
            b = str(r[benchmark_col])
            test_mask = (mg == g) & (tr == t) & (bm == b)
            train_mask = (mg != g) & (tr != t) & (bm != b)
            folds.append(
                (
                    f"model={g}|train={t}|benchmark={b}|disjoint=1",
                    test_mask.to_numpy(dtype=bool),
                    train_mask.to_numpy(dtype=bool),
                )
            )
        return folds

    if protocol == "model_benchmark_trainset_disjoint":
        k = max(int(holdout_train_k), 2)
        mg = df[model_group_col].astype(str)
        tr = df[train_col].astype(str)
        bm = df[benchmark_col].astype(str)

        contexts: List[Tuple[str, str, Tuple[str, ...]]] = []
        uniq_ctx = (
            df[[model_group_col, benchmark_col, train_col]]
            .dropna()
            .drop_duplicates()
            .copy()
        )
        for (g_raw, b_raw), sub in uniq_ctx.groupby([model_group_col, benchmark_col], dropna=False):
            g = str(g_raw)
            b = str(b_raw)
            train_vals = sorted(set(sub[train_col].astype(str).dropna().unique()))
            if len(train_vals) < k:
                continue
            for combo in itertools.combinations(train_vals, k):
                contexts.append((g, b, tuple(combo)))

        if not contexts:
            return folds

        if max_hard_folds > 0 and len(contexts) > int(max_hard_folds):
            take = rng.choice(len(contexts), size=int(max_hard_folds), replace=False)
            contexts = [contexts[i] for i in np.sort(take)]

        for g, b, holdout_train_sets in contexts:
            holdout_set = set(holdout_train_sets)
            test_mask = (mg == g) & (bm == b) & (tr.isin(holdout_set))
            train_mask = (mg != g) & (bm != b) & (~tr.isin(holdout_set))
            holdout_tag = "+".join(holdout_train_sets)
            if len(holdout_tag) > 120:
                holdout_tag = holdout_tag[:117] + "..."
            folds.append(
                (
                    f"model={g}|benchmark={b}|trainset_k={k}|holdout={holdout_tag}",
                    test_mask.to_numpy(dtype=bool),
                    train_mask.to_numpy(dtype=bool),
                )
            )
        return folds

    raise ValueError(f"Unsupported protocol: {protocol}")


def _evaluate_one(
    cand: pd.Series,
    head: str,
    protocol: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> EvalResult:
    run_path = Path(str(cand["path"]))
    if args.row_source == "raw":
        rows_path = run_path / "auc_with_features.csv"
    else:
        rows_path = run_path / "prediction_jointood_rows.csv"
    predictors = _split_predictors(cand["predictors"])

    out_summary: Dict[str, object] = {
        "candidate_id": str(cand["candidate_id"]),
        "variant": str(cand["variant"]),
        "method": str(cand["method"]),
        "path": str(cand["path"]),
        "lane": str(cand["lane"]),
        "signal_k": int(cand["signal_k"]),
        "n_predictors_total": int(cand["n_predictors_total"]),
        "head": head,
        "protocol": protocol,
        "status": "ok",
    }

    if not rows_path.exists():
        out_summary["status"] = "missing_rows_csv"
        return EvalResult(out_summary, pd.DataFrame(), pd.DataFrame())
    try:
        df = pd.read_csv(rows_path)
    except Exception as exc:
        out_summary["status"] = f"read_error:{exc}"
        return EvalResult(out_summary, pd.DataFrame(), pd.DataFrame())

    df, option_col_used = _ensure_option_col(df, str(args.option_col))
    df, pairwise_group_cols, pairwise_group_missing = _resolve_pairwise_group_cols(
        df, str(args.pairwise_group_cols)
    )
    df, residual_context_cols, residual_context_missing = _resolve_context_cols(
        df, str(args.cv_residual_context_cols)
    )
    df, rank_context_cols, rank_context_missing = _resolve_context_cols(
        df, str(args.rank_context_cols)
    )
    df, fewshot_context_cols, fewshot_context_missing = _resolve_context_cols(
        df, str(args.cv_fewshot_context_calibration_cols)
    )
    out_summary["option_col_requested"] = str(args.option_col)
    out_summary["option_col_used"] = str(option_col_used)
    out_summary["pairwise_group_cols_requested"] = str(args.pairwise_group_cols)
    out_summary["pairwise_group_cols_used"] = ",".join(pairwise_group_cols)
    out_summary["cv_residualize_target_by_context"] = bool(args.cv_residualize_target_by_context)
    out_summary["cv_residual_context_cols_requested"] = str(args.cv_residual_context_cols)
    out_summary["cv_residual_context_cols_used"] = ",".join(residual_context_cols)
    out_summary["cv_residual_target_transform"] = str(args.cv_residual_target_transform)
    out_summary["cv_residual_eval_space"] = str(args.cv_residual_eval_space)
    out_summary["cv_fewshot_context_calibration"] = bool(args.cv_fewshot_context_calibration)
    out_summary["cv_fewshot_context_calibration_cols_requested"] = str(args.cv_fewshot_context_calibration_cols)
    out_summary["cv_fewshot_context_calibration_cols_used"] = ",".join(fewshot_context_cols)
    out_summary["cv_fewshot_context_calibration_k"] = int(args.cv_fewshot_context_calibration_k)
    out_summary["cv_fewshot_context_calibration_min_group_size"] = int(
        args.cv_fewshot_context_calibration_min_group_size
    )
    out_summary["cv_fewshot_context_calibration_backoff"] = bool(
        args.cv_fewshot_context_calibration_backoff
    )
    out_summary["rank_context_cols_requested"] = str(args.rank_context_cols)
    out_summary["rank_context_cols_used"] = ",".join(rank_context_cols)

    target_candidates = [args.target_col, "auc_normalized_observed", "target"]
    target_col = next((c for c in target_candidates if c in df.columns), None)
    if target_col is None:
        out_summary["status"] = "missing_target_col"
        return EvalResult(out_summary, pd.DataFrame(), pd.DataFrame())

    required_cols = list(predictors) + [target_col, args.train_col, args.benchmark_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        out_summary["status"] = "missing_columns:" + ",".join(missing)
        return EvalResult(out_summary, pd.DataFrame(), pd.DataFrame())
    if head == "pairwise_rank" and option_col_used not in df.columns:
        out_summary["status"] = f"missing_option_col:{option_col_used}"
        return EvalResult(out_summary, pd.DataFrame(), pd.DataFrame())
    if head == "pairwise_rank" and pairwise_group_missing:
        out_summary["status"] = "missing_pairwise_group_cols:" + ",".join(pairwise_group_missing)
        return EvalResult(out_summary, pd.DataFrame(), pd.DataFrame())
    if head == "pairwise_rank" and not pairwise_group_cols:
        out_summary["status"] = "empty_pairwise_group_cols"
        return EvalResult(out_summary, pd.DataFrame(), pd.DataFrame())
    if bool(args.cv_residualize_target_by_context) and residual_context_missing:
        out_summary["status"] = "missing_residual_context_cols:" + ",".join(residual_context_missing)
        return EvalResult(out_summary, pd.DataFrame(), pd.DataFrame())
    if bool(args.cv_residualize_target_by_context) and not residual_context_cols:
        out_summary["status"] = "empty_residual_context_cols"
        return EvalResult(out_summary, pd.DataFrame(), pd.DataFrame())
    if rank_context_missing:
        out_summary["status"] = "missing_rank_context_cols:" + ",".join(rank_context_missing)
        return EvalResult(out_summary, pd.DataFrame(), pd.DataFrame())
    if bool(args.cv_fewshot_context_calibration) and fewshot_context_missing:
        out_summary["status"] = "missing_fewshot_context_cols:" + ",".join(fewshot_context_missing)
        return EvalResult(out_summary, pd.DataFrame(), pd.DataFrame())

    df = df.copy()
    df["__model_group__"] = _model_group_series(df, args.model_group_col)
    df = df.dropna(subset=required_cols + ["__model_group__"]).copy()
    if df.empty:
        out_summary["status"] = "empty_after_dropna"
        return EvalResult(out_summary, pd.DataFrame(), pd.DataFrame())

    # Keep only finite numeric rows for predictors + target.
    for c in predictors + [target_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=predictors + [target_col]).copy()
    if df.empty:
        out_summary["status"] = "empty_after_numeric_filter"
        return EvalResult(out_summary, pd.DataFrame(), pd.DataFrame())

    folds = _build_folds(
        df=df,
        protocol=protocol,
        model_group_col="__model_group__",
        train_col=args.train_col,
        benchmark_col=args.benchmark_col,
        holdout_train_k=int(args.holdout_train_k),
        max_hard_folds=int(args.max_hard_folds),
        rng=rng,
    )
    if not folds:
        out_summary["status"] = "no_folds"
        return EvalResult(out_summary, pd.DataFrame(), pd.DataFrame())

    pred_rows: List[pd.DataFrame] = []
    fold_rows: List[Dict[str, object]] = []
    eval_space = str(args.cv_residual_eval_space).strip().lower()
    for fold_idx, (fold_name, test_mask, train_mask) in enumerate(folds):
        test_df = df.loc[test_mask].copy()
        if train_mask is None:
            train_df = df.loc[~test_mask].copy()
        else:
            train_df = df.loc[train_mask].copy()
        if train_df.empty or test_df.empty:
            continue
        if len(train_df) < max(int(args.min_train_rows), len(predictors) + 2):
            continue

        train_fit_df = train_df.copy()
        test_fit_df = test_df.copy()
        test_offsets = np.zeros(len(test_fit_df), dtype=float)
        test_scales = np.ones(len(test_fit_df), dtype=float)
        if bool(args.cv_residualize_target_by_context):
            (
                train_fit_df,
                test_fit_df,
                _train_offsets,
                test_offsets,
                _train_scales,
                test_scales,
                _residualizer,
            ) = _residualize_target_by_context(
                train_df=train_df,
                test_df=test_df,
                target_col=target_col,
                context_cols=residual_context_cols,
                transform=str(args.cv_residual_target_transform),
                std_eps=float(args.cv_residual_target_std_eps),
            )
            del _train_offsets, _train_scales, _residualizer

        if head in {"ols", "ridge"}:
            coef, mean, std = _fit_linear(
                train_df=train_fit_df,
                predictors=predictors,
                target_col=target_col,
                model=head,
                ridge_alpha=float(args.ridge_alpha),
                standardize=True,
            )
            y_pred_model = _predict_linear(
                test_df=test_fit_df,
                predictors=predictors,
                coef=coef,
                mean=mean,
                std=std,
                standardize=True,
            )
        elif head == "pairwise_rank":
            coef, mean, std = _fit_pairwise_rank(
                train_df=train_fit_df,
                predictors=predictors,
                target_col=target_col,
                group_cols=pairwise_group_cols,
                option_col=option_col_used,
                ridge_alpha=float(args.ridge_alpha),
                max_pairs_per_group=int(args.max_pairs_per_group),
                max_iter=int(args.pairwise_max_iter),
                lr=float(args.pairwise_lr),
                seed=int(args.seed) + fold_idx,
                standardize=True,
            )
            y_pred_model = _predict_pairwise_rank(
                test_df=test_fit_df,
                predictors=predictors,
                coef=coef,
                mean=mean,
                std=std,
                standardize=True,
            )
        else:
            out_summary["status"] = f"unsupported_head:{head}"
            return EvalResult(out_summary, pd.DataFrame(), pd.DataFrame())

        y_true_absolute = test_df[target_col].to_numpy(dtype=float)
        y_true_model = test_fit_df[target_col].to_numpy(dtype=float)
        y_pred_absolute = y_pred_model * test_scales + test_offsets
        if eval_space == "absolute":
            y_true_eval = y_true_absolute
            y_pred_eval = y_pred_absolute
        else:
            y_true_eval = y_true_model
            y_pred_eval = y_pred_model

        fewshot_calibration_mask = np.zeros(len(test_df), dtype=bool)
        if bool(args.cv_fewshot_context_calibration):
            calib_df = test_df
            calib_y_true = y_true_eval
            calib_y_pred = y_pred_eval
            k_shot = int(args.cv_fewshot_context_calibration_k)
            if k_shot > 0:
                fold_rng = np.random.RandomState(int(args.cv_fewshot_context_calibration_seed) + int(fold_idx))
                fewshot_calibration_mask = _sample_fewshot_calibration_mask(
                    test_df,
                    context_cols=fewshot_context_cols,
                    k=k_shot,
                    rng=fold_rng,
                    allow_backoff=bool(args.cv_fewshot_context_calibration_backoff),
                )
                if np.any(fewshot_calibration_mask):
                    calib_idx = np.where(fewshot_calibration_mask)[0]
                    calib_df = test_df.iloc[calib_idx]
                    calib_y_true = y_true_eval[calib_idx]
                    calib_y_pred = y_pred_eval[calib_idx]
            calibrator = _fit_context_prediction_calibrator(
                calib_df,
                calib_y_true,
                calib_y_pred,
                context_cols=fewshot_context_cols,
                std_eps=float(args.cv_fewshot_context_calibration_std_eps),
                min_group_size=int(args.cv_fewshot_context_calibration_min_group_size),
                allow_backoff=bool(args.cv_fewshot_context_calibration_backoff),
            )
            y_pred_eval = _apply_context_prediction_calibrator(
                test_df,
                y_pred_eval,
                calibrator,
            )
        eval_mask = (
            ~fewshot_calibration_mask
            if int(args.cv_fewshot_context_calibration_k) > 0 and bool(args.cv_fewshot_context_calibration)
            else np.ones(len(test_df), dtype=bool)
        )
        if not np.any(eval_mask):
            continue
        test_eval_df = test_df.iloc[np.where(eval_mask)[0]]
        y_true_eval = y_true_eval[eval_mask]
        y_pred_eval = y_pred_eval[eval_mask]
        y_true_absolute_eval = y_true_absolute[eval_mask]
        y_true_model_eval = y_true_model[eval_mask]
        y_pred_model_eval = y_pred_model[eval_mask]
        y_pred_absolute_eval = y_pred_absolute[eval_mask]
        if eval_space == "absolute":
            y_pred_absolute_eval = y_pred_eval
        else:
            y_pred_model_eval = y_pred_eval

        mae, rmse = _mae_rmse(y_true_eval, y_pred_eval)
        pearson = _pearson_corr(y_true_eval, y_pred_eval)
        spearman = _spearman_corr(y_true_eval, y_pred_eval)
        fold_rows.append(
            {
                "candidate_id": str(cand["candidate_id"]),
                "variant": str(cand["variant"]),
                "method": str(cand["method"]),
                "lane": str(cand["lane"]),
                "signal_k": int(cand["signal_k"]),
                "n_predictors_total": int(cand["n_predictors_total"]),
                "head": head,
                "protocol": protocol,
                "fold_id": fold_name,
                "n_train": int(len(train_df)),
                "n_test": int(len(test_eval_df)),
                "n_calibration": int(fewshot_calibration_mask.sum()),
                "eval_space": eval_space,
                "mae": mae,
                "rmse": rmse,
                "pearson": pearson,
                "spearman": spearman,
            }
        )

        tmp = test_eval_df[[args.train_col, args.benchmark_col, "__model_group__", target_col]].copy()
        if option_col_used in test_eval_df.columns and option_col_used not in tmp.columns:
            tmp[option_col_used] = test_eval_df[option_col_used].values
        for ctx_col in rank_context_cols:
            if ctx_col in test_eval_df.columns and ctx_col not in tmp.columns:
                tmp[ctx_col] = test_eval_df[ctx_col].values
        tmp = tmp.rename(columns={target_col: args.target_col})
        tmp["target_absolute"] = y_true_absolute_eval
        tmp["target_model_space"] = y_true_model_eval
        tmp["prediction_absolute"] = y_pred_absolute_eval
        tmp["prediction_model_space"] = y_pred_model_eval
        tmp["target_eval"] = y_true_eval
        tmp["prediction"] = y_pred_eval
        tmp["fold_id"] = fold_name
        tmp["candidate_id"] = str(cand["candidate_id"])
        tmp["head"] = head
        tmp["protocol"] = protocol
        tmp["eval_space"] = eval_space
        tmp["is_calibration_row"] = 0
        pred_rows.append(tmp)

    if not pred_rows:
        out_summary["status"] = "no_scored_folds"
        return EvalResult(out_summary, pd.DataFrame(), pd.DataFrame())

    pred_df = pd.concat(pred_rows, ignore_index=True)
    fold_df = pd.DataFrame(fold_rows)

    if "target_eval" not in pred_df.columns:
        out_summary["status"] = "missing_target_eval_col"
        return EvalResult(out_summary, pred_df, fold_df)
    y_true = pred_df["target_eval"].to_numpy(dtype=float)
    y_pred = pred_df["prediction"].to_numpy(dtype=float)
    mae, rmse = _mae_rmse(y_true, y_pred)
    pearson = _pearson_corr(y_true, y_pred)
    spearman = _spearman_corr(y_true, y_pred)
    rank = _compute_ranking_summary(
        pred_df=pred_df,
        option_col=option_col_used,
        benchmark_col=args.benchmark_col,
        target_col="target_eval",
        grouping=str(args.rank_grouping),
        context_cols=rank_context_cols,
        topk_frac=float(args.rank_topk_frac),
        topk_min=int(args.rank_topk_min),
    )

    out_summary.update(
        {
            "n_rows_scored": int(len(pred_df)),
            "n_folds_requested": int(len(folds)),
            "n_folds_scored": int(len(fold_df)),
            "n_calibration": (
                float(pd.to_numeric(fold_df["n_calibration"], errors="coerce").sum())
                if "n_calibration" in fold_df.columns
                else math.nan
            ),
            "n_model_groups": int(pred_df["__model_group__"].nunique(dropna=True)),
            "target_col_used": str(target_col),
            "target_eval_col_used": "target_eval",
            "eval_space": eval_space,
            "mae": mae,
            "rmse": rmse,
            "pearson": pearson,
            "spearman": spearman,
            "fold_mae_mean": float(pd.to_numeric(fold_df["mae"], errors="coerce").mean()),
            "fold_mae_std": float(pd.to_numeric(fold_df["mae"], errors="coerce").std(ddof=0)),
            "fold_rmse_mean": float(pd.to_numeric(fold_df["rmse"], errors="coerce").mean()),
            "fold_rmse_std": float(pd.to_numeric(fold_df["rmse"], errors="coerce").std(ddof=0)),
            "fold_spearman_mean": float(pd.to_numeric(fold_df["spearman"], errors="coerce").mean()),
            "fold_spearman_std": float(pd.to_numeric(fold_df["spearman"], errors="coerce").std(ddof=0)),
        }
    )
    out_summary.update(rank)
    return EvalResult(out_summary, pred_df, fold_df)


def _permutation_baseline(
    pred_df: pd.DataFrame,
    target_col: str,
    option_col: str,
    benchmark_col: str,
    rank_grouping: str,
    rank_context_cols: Optional[Sequence[str]],
    n_permutations: int,
    rng: np.random.Generator,
    mode: str,
) -> Dict[str, float]:
    out: Dict[str, float] = {
        "n_rows": 0.0,
        "n_rank_groups": 0.0,
        "n_permutations_used": 0.0,
        "obs_rank_spearman": math.nan,
        "obs_rank_kendall_tau": math.nan,
        "obs_rank_pairwise_cindex": math.nan,
        "perm_rank_spearman_mean": math.nan,
        "perm_rank_spearman_std": math.nan,
        "perm_rank_kendall_tau_mean": math.nan,
        "perm_rank_kendall_tau_std": math.nan,
        "perm_rank_pairwise_cindex_mean": math.nan,
        "perm_rank_pairwise_cindex_std": math.nan,
        "delta_rank_spearman_obs_minus_perm": math.nan,
        "delta_rank_kendall_tau_obs_minus_perm": math.nan,
        "delta_rank_pairwise_cindex_obs_minus_perm": math.nan,
        "p_value_rank_spearman_higher_than_perm": math.nan,
        "p_value_rank_kendall_tau_higher_than_perm": math.nan,
        "p_value_rank_pairwise_cindex_higher_than_perm": math.nan,
    }
    if n_permutations <= 0:
        return out
    if pred_df.empty or "prediction" not in pred_df.columns or target_col not in pred_df.columns:
        return out

    required = [benchmark_col, option_col, "prediction", target_col]
    missing = [c for c in required if c not in pred_df.columns]
    if missing:
        return out
    rows = pred_df.dropna(subset=required).copy()
    if rows.empty:
        return out

    group_cols: List[str]
    if rank_grouping == "fold_benchmark" and "fold_id" in rows.columns:
        group_cols = ["fold_id", benchmark_col]
    else:
        group_cols = [benchmark_col]
    for col in (rank_context_cols or []):
        if col in rows.columns and col not in group_cols:
            group_cols.append(col)

    option_group_cols = group_cols + [option_col]
    grouped = (
        rows.groupby(option_group_cols, dropna=False)
        .agg(pred_mean=("prediction", "mean"), true_mean=(target_col, "mean"))
        .reset_index(drop=False)
    )
    if grouped.empty:
        return out

    out["n_rows"] = float(len(rows))

    per_group: List[Tuple[float, float, float]] = []
    for _, sub in grouped.groupby(group_cols, dropna=False):
        if len(sub) < 2:
            continue
        pred = sub["pred_mean"].to_numpy(dtype=float)
        true = sub["true_mean"].to_numpy(dtype=float)
        per_group.append(
            (
                _spearman_corr(pred, true),
                _kendall_tau_b(pred, true),
                _pairwise_cindex(true, pred),
            )
        )
    if not per_group:
        return out

    obs_arr = np.asarray(per_group, dtype=float)
    obs_rank_s = float(np.nanmean(obs_arr[:, 0]))
    obs_rank_k = float(np.nanmean(obs_arr[:, 1]))
    obs_rank_c = float(np.nanmean(obs_arr[:, 2]))
    out["obs_rank_spearman"] = obs_rank_s
    out["obs_rank_kendall_tau"] = obs_rank_k
    out["obs_rank_pairwise_cindex"] = obs_rank_c
    out["n_rank_groups"] = float(len(obs_arr))

    if mode == "global":
        perm_group_cols: List[str] = []
    elif mode == "fold" and "fold_id" in grouped.columns:
        perm_group_cols = ["fold_id"]
    else:
        perm_group_cols = list(group_cols)

    idx_groups: List[np.ndarray]
    if not perm_group_cols:
        idx_groups = [np.arange(len(grouped), dtype=int)]
    else:
        key_series = grouped[perm_group_cols].astype(str).agg("||".join, axis=1)
        idx_groups = [idx.to_numpy(dtype=int) for _, idx in key_series.groupby(key_series).groups.items()]

    if idx_groups:
        singleton_frac = float(sum(len(g) <= 1 for g in idx_groups)) / float(len(idx_groups))
        if singleton_frac > 0.8:
            idx_groups = [np.arange(len(grouped), dtype=int)]

    true_base = grouped["true_mean"].to_numpy(dtype=float)
    perm_s: List[float] = []
    perm_k: List[float] = []
    perm_c: List[float] = []
    for _ in range(int(n_permutations)):
        true_perm = true_base.copy()
        for idx in idx_groups:
            if len(idx) <= 1:
                continue
            true_perm[idx] = true_perm[idx][rng.permutation(len(idx))]
        grouped["__true_perm__"] = true_perm

        vals: List[Tuple[float, float, float]] = []
        for _, sub in grouped.groupby(group_cols, dropna=False):
            if len(sub) < 2:
                continue
            pred = sub["pred_mean"].to_numpy(dtype=float)
            true = sub["__true_perm__"].to_numpy(dtype=float)
            vals.append(
                (
                    _spearman_corr(pred, true),
                    _kendall_tau_b(pred, true),
                    _pairwise_cindex(true, pred),
                )
            )
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        perm_s.append(float(np.nanmean(arr[:, 0])))
        perm_k.append(float(np.nanmean(arr[:, 1])))
        perm_c.append(float(np.nanmean(arr[:, 2])))

    perm_s_arr = np.asarray(perm_s, dtype=float)
    perm_k_arr = np.asarray(perm_k, dtype=float)
    perm_c_arr = np.asarray(perm_c, dtype=float)
    out["n_permutations_used"] = float(len(perm_s_arr))

    if np.isfinite(perm_s_arr).any():
        finite = perm_s_arr[np.isfinite(perm_s_arr)]
        out["perm_rank_spearman_mean"] = float(np.nanmean(perm_s_arr))
        out["perm_rank_spearman_std"] = float(np.nanstd(perm_s_arr))
        out["delta_rank_spearman_obs_minus_perm"] = float(obs_rank_s - out["perm_rank_spearman_mean"])
        out["p_value_rank_spearman_higher_than_perm"] = float(
            (1 + np.sum(finite >= float(obs_rank_s))) / (len(finite) + 1)
        )
    if np.isfinite(perm_k_arr).any():
        finite = perm_k_arr[np.isfinite(perm_k_arr)]
        out["perm_rank_kendall_tau_mean"] = float(np.nanmean(perm_k_arr))
        out["perm_rank_kendall_tau_std"] = float(np.nanstd(perm_k_arr))
        out["delta_rank_kendall_tau_obs_minus_perm"] = float(obs_rank_k - out["perm_rank_kendall_tau_mean"])
        out["p_value_rank_kendall_tau_higher_than_perm"] = float(
            (1 + np.sum(finite >= float(obs_rank_k))) / (len(finite) + 1)
        )
    if np.isfinite(perm_c_arr).any():
        finite = perm_c_arr[np.isfinite(perm_c_arr)]
        out["perm_rank_pairwise_cindex_mean"] = float(np.nanmean(perm_c_arr))
        out["perm_rank_pairwise_cindex_std"] = float(np.nanstd(perm_c_arr))
        out["delta_rank_pairwise_cindex_obs_minus_perm"] = float(obs_rank_c - out["perm_rank_pairwise_cindex_mean"])
        out["p_value_rank_pairwise_cindex_higher_than_perm"] = float(
            (1 + np.sum(finite >= float(obs_rank_c))) / (len(finite) + 1)
        )

    return out


def _build_comparison_tables(summary_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rank_sort_cols = [
        "rank_spearman",
        "rank_pairwise_cindex",
        "rank_kendall_tau",
        "rank_regret",
        "spearman",
        "mae",
    ]
    rank_sort_asc = [False, False, False, True, False, True]

    def _sort_for_rank(df: pd.DataFrame, prefix_cols: Sequence[str]) -> pd.DataFrame:
        cols = list(prefix_cols) + rank_sort_cols
        asc = [True] * len(prefix_cols) + rank_sort_asc
        return df.sort_values(cols, ascending=asc, na_position="last")

    def _winner_by_rank(a: pd.Series, b: pd.Series) -> str:
        checks: List[Tuple[str, bool]] = [
            ("rank_spearman", True),
            ("rank_pairwise_cindex", True),
            ("rank_kendall_tau", True),
            ("rank_regret", False),
            ("spearman", True),
            ("mae", False),
        ]
        for col, higher_is_better in checks:
            av = pd.to_numeric(pd.Series([a.get(col)]), errors="coerce").iloc[0]
            bv = pd.to_numeric(pd.Series([b.get(col)]), errors="coerce").iloc[0]
            if pd.isna(av) and pd.isna(bv):
                continue
            if pd.isna(av):
                return "B better"
            if pd.isna(bv):
                return "A better"
            if float(av) == float(bv):
                continue
            if higher_is_better:
                return "A better" if float(av) > float(bv) else "B better"
            return "A better" if float(av) < float(bv) else "B better"
        return "tie"

    lane_comp_cols = [
        "protocol",
        "head",
        "signal_k",
        "comparison",
        "lane_a",
        "lane_b",
        "method_a",
        "method_b",
        "mae_a",
        "mae_b",
        "delta_mae_b_minus_a",
        "spearman_a",
        "spearman_b",
        "delta_spearman_a_minus_b",
        "rank_spearman_a",
        "rank_spearman_b",
        "delta_rank_spearman_a_minus_b",
        "rank_kendall_tau_a",
        "rank_kendall_tau_b",
        "delta_rank_kendall_tau_a_minus_b",
        "rank_pairwise_cindex_a",
        "rank_pairwise_cindex_b",
        "delta_rank_pairwise_cindex_a_minus_b",
        "rank_regret_a",
        "rank_regret_b",
        "delta_regret_b_minus_a",
        "selection_basis",
        "verdict",
    ]
    mmd_comp_cols = [
        "protocol",
        "head",
        "signal_k",
        "method_asym",
        "method_mmd",
        "mae_asym",
        "mae_mmd",
        "delta_mae_mmd_minus_asym",
        "spearman_asym",
        "spearman_mmd",
        "delta_spearman_asym_minus_mmd",
        "rank_spearman_asym",
        "rank_spearman_mmd",
        "delta_rank_spearman_asym_minus_mmd",
        "rank_kendall_tau_asym",
        "rank_kendall_tau_mmd",
        "delta_rank_kendall_tau_asym_minus_mmd",
        "rank_pairwise_cindex_asym",
        "rank_pairwise_cindex_mmd",
        "delta_rank_pairwise_cindex_asym_minus_mmd",
        "rank_regret_asym",
        "rank_regret_mmd",
        "delta_regret_mmd_minus_asym",
        "selection_basis",
        "verdict",
    ]

    if summary_df.empty:
        empty = pd.DataFrame()
        return (
            empty,
            empty,
            empty,
            empty,
            pd.DataFrame(columns=lane_comp_cols),
            pd.DataFrame(columns=mmd_comp_cols),
        )

    ok = summary_df[summary_df["status"] == "ok"].copy()
    if ok.empty:
        empty = pd.DataFrame()
        return (
            empty,
            empty,
            empty,
            empty,
            pd.DataFrame(columns=lane_comp_cols),
            pd.DataFrame(columns=mmd_comp_cols),
        )

    ranked_all = _sort_for_rank(ok, ["protocol", "head", "signal_k"]).reset_index(drop=True)

    best_by_protocol_head = (
        _sort_for_rank(ok, ["protocol", "head"])
        .drop_duplicates(["protocol", "head"], keep="first")
        .sort_values(["protocol", "head"])
        .reset_index(drop=True)
    )

    best_by_protocol_head_lane = (
        _sort_for_rank(ok, ["protocol", "head", "lane"])
        .drop_duplicates(["protocol", "head", "lane"], keep="first")
        .sort_values(["protocol", "head", "lane"])
        .reset_index(drop=True)
    )

    best_by_protocol_head_lane_k = (
        _sort_for_rank(ok, ["protocol", "head", "lane", "signal_k"])
        .drop_duplicates(["protocol", "head", "lane", "signal_k"], keep="first")
        .sort_values(["protocol", "head", "lane", "signal_k"])
        .reset_index(drop=True)
    )

    lane_rows: List[Dict[str, object]] = []
    pairs = [
        ("motion_only", "appearance_only", "motion_vs_appearance"),
        ("hybrid", "motion_only", "hybrid_vs_motion"),
        ("hybrid", "appearance_only", "hybrid_vs_appearance"),
    ]
    for (protocol, head, k), sub in best_by_protocol_head_lane_k.groupby(["protocol", "head", "signal_k"], dropna=False):
        by_lane = {str(r["lane"]): r for _, r in sub.iterrows()}
        for a_lane, b_lane, tag in pairs:
            if a_lane not in by_lane or b_lane not in by_lane:
                continue
            a = by_lane[a_lane]
            b = by_lane[b_lane]
            d_mae = float(b["mae"] - a["mae"])
            d_sp = float(a["spearman"] - b["spearman"])
            d_rank_sp = float(a["rank_spearman"] - b["rank_spearman"])
            d_rank_k = float(a["rank_kendall_tau"] - b["rank_kendall_tau"])
            d_rank_c = float(a["rank_pairwise_cindex"] - b["rank_pairwise_cindex"])
            d_reg = float(b["rank_regret"] - a["rank_regret"])
            verdict = _winner_by_rank(a, b)
            lane_rows.append(
                {
                    "protocol": protocol,
                    "head": head,
                    "signal_k": int(k),
                    "comparison": tag,
                    "lane_a": a_lane,
                    "lane_b": b_lane,
                    "method_a": str(a["method"]),
                    "method_b": str(b["method"]),
                    "mae_a": float(a["mae"]),
                    "mae_b": float(b["mae"]),
                    "delta_mae_b_minus_a": d_mae,
                    "spearman_a": float(a["spearman"]),
                    "spearman_b": float(b["spearman"]),
                    "delta_spearman_a_minus_b": d_sp,
                    "rank_spearman_a": float(a["rank_spearman"]),
                    "rank_spearman_b": float(b["rank_spearman"]),
                    "delta_rank_spearman_a_minus_b": d_rank_sp,
                    "rank_kendall_tau_a": float(a["rank_kendall_tau"]),
                    "rank_kendall_tau_b": float(b["rank_kendall_tau"]),
                    "delta_rank_kendall_tau_a_minus_b": d_rank_k,
                    "rank_pairwise_cindex_a": float(a["rank_pairwise_cindex"]),
                    "rank_pairwise_cindex_b": float(b["rank_pairwise_cindex"]),
                    "delta_rank_pairwise_cindex_a_minus_b": d_rank_c,
                    "rank_regret_a": float(a["rank_regret"]),
                    "rank_regret_b": float(b["rank_regret"]),
                    "delta_regret_b_minus_a": d_reg,
                    "selection_basis": "rank_spearman>rank_pairwise_cindex>rank_kendall_tau>rank_regret",
                    "verdict": verdict,
                }
            )
    lane_budget_comparisons = (
        pd.DataFrame(lane_rows).sort_values(["protocol", "head", "signal_k", "comparison"])
        if lane_rows
        else pd.DataFrame(columns=lane_comp_cols)
    )

    mmd_rows: List[Dict[str, object]] = []
    for (protocol, head, k), sub in ok.groupby(["protocol", "head", "signal_k"], dropna=False):
        methods = sub["method"].astype(str)
        mmd = sub[methods.str.contains("mmd", na=False)].copy()
        asym = sub[~methods.str.contains("mmd", na=False)].copy()
        if mmd.empty or asym.empty:
            continue
        m = _sort_for_rank(mmd, []).iloc[0]
        a = _sort_for_rank(asym, []).iloc[0]
        d_mae = float(m["mae"] - a["mae"])
        d_sp = float(a["spearman"] - m["spearman"])
        d_rank_sp = float(a["rank_spearman"] - m["rank_spearman"])
        d_rank_k = float(a["rank_kendall_tau"] - m["rank_kendall_tau"])
        d_rank_c = float(a["rank_pairwise_cindex"] - m["rank_pairwise_cindex"])
        d_reg = float(m["rank_regret"] - a["rank_regret"])
        rank_verdict = _winner_by_rank(a, m)
        verdict = "asym better" if rank_verdict == "A better" else ("mmd better" if rank_verdict == "B better" else "tie")
        mmd_rows.append(
            {
                "protocol": protocol,
                "head": head,
                "signal_k": int(k),
                "method_asym": str(a["method"]),
                "method_mmd": str(m["method"]),
                "mae_asym": float(a["mae"]),
                "mae_mmd": float(m["mae"]),
                "delta_mae_mmd_minus_asym": d_mae,
                "spearman_asym": float(a["spearman"]),
                "spearman_mmd": float(m["spearman"]),
                "delta_spearman_asym_minus_mmd": d_sp,
                "rank_spearman_asym": float(a["rank_spearman"]),
                "rank_spearman_mmd": float(m["rank_spearman"]),
                "delta_rank_spearman_asym_minus_mmd": d_rank_sp,
                "rank_kendall_tau_asym": float(a["rank_kendall_tau"]),
                "rank_kendall_tau_mmd": float(m["rank_kendall_tau"]),
                "delta_rank_kendall_tau_asym_minus_mmd": d_rank_k,
                "rank_pairwise_cindex_asym": float(a["rank_pairwise_cindex"]),
                "rank_pairwise_cindex_mmd": float(m["rank_pairwise_cindex"]),
                "delta_rank_pairwise_cindex_asym_minus_mmd": d_rank_c,
                "rank_regret_asym": float(a["rank_regret"]),
                "rank_regret_mmd": float(m["rank_regret"]),
                "delta_regret_mmd_minus_asym": d_reg,
                "selection_basis": "rank_spearman>rank_pairwise_cindex>rank_kendall_tau>rank_regret",
                "verdict": verdict,
            }
        )
    mmd_budget_comparisons = (
        pd.DataFrame(mmd_rows).sort_values(["protocol", "head", "signal_k"])
        if mmd_rows
        else pd.DataFrame(columns=mmd_comp_cols)
    )

    return (
        ranked_all,
        best_by_protocol_head,
        best_by_protocol_head_lane,
        best_by_protocol_head_lane_k,
        lane_budget_comparisons,
        mmd_budget_comparisons,
    )


def _subset_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if scope == "all":
        return out
    if "head" in out.columns and scope in {
        "ols_ridge",
        "model_only_ols_ridge",
        "model_train_benchmark_ols_ridge",
        "model_benchmark_ols_ridge",
        "model_train_benchmark_disjoint_ols_ridge",
        "model_benchmark_trainset_disjoint_ols_ridge",
    }:
        out = out[out["head"].astype(str).isin(["ols", "ridge"])].copy()
    if "protocol" in out.columns:
        if scope in {"model_only_all_heads", "model_only_ols_ridge"}:
            out = out[out["protocol"].astype(str) == "model_only"].copy()
        elif scope in {"model_train_benchmark_all_heads", "model_train_benchmark_ols_ridge"}:
            out = out[out["protocol"].astype(str) == "model_train_benchmark"].copy()
        elif scope in {"model_benchmark_all_heads", "model_benchmark_ols_ridge"}:
            out = out[out["protocol"].astype(str) == "model_benchmark"].copy()
        elif scope in {"model_train_benchmark_disjoint_all_heads", "model_train_benchmark_disjoint_ols_ridge"}:
            out = out[out["protocol"].astype(str) == "model_train_benchmark_disjoint"].copy()
        elif scope in {"model_benchmark_trainset_disjoint_all_heads", "model_benchmark_trainset_disjoint_ols_ridge"}:
            out = out[out["protocol"].astype(str) == "model_benchmark_trainset_disjoint"].copy()
    return out


def _claim_row(
    df: pd.DataFrame,
    scope: str,
    claim: str,
    delta_col: str,
    eps: float,
) -> Dict[str, object]:
    out: Dict[str, object] = {
        "scope": scope,
        "claim": claim,
        "tie_epsilon": float(eps),
        "n_total": 0,
        "n_support": 0,
        "n_against": 0,
        "n_near_tie": 0,
        "support_rate": math.nan,
        "mean_delta_primary": math.nan,
        "median_delta_primary": math.nan,
        "mean_delta_rank_kendall_tau": math.nan,
        "median_delta_rank_kendall_tau": math.nan,
        "mean_delta_rank_pairwise_cindex": math.nan,
        "median_delta_rank_pairwise_cindex": math.nan,
    }
    if df.empty or delta_col not in df.columns:
        return out
    work = df.copy()
    work[delta_col] = pd.to_numeric(work[delta_col], errors="coerce")
    work = work[work[delta_col].notna()].copy()
    if work.empty:
        return out

    n = len(work)
    support = int((work[delta_col] > float(eps)).sum())
    against = int((work[delta_col] < -float(eps)).sum())
    near_tie = int(n - support - against)
    out.update(
        {
            "n_total": int(n),
            "n_support": support,
            "n_against": against,
            "n_near_tie": near_tie,
            "support_rate": float(support / n),
            "mean_delta_primary": float(work[delta_col].mean()),
            "median_delta_primary": float(work[delta_col].median()),
        }
    )

    kendall_col = None
    if "delta_rank_kendall_tau_a_minus_b" in work.columns:
        kendall_col = "delta_rank_kendall_tau_a_minus_b"
    elif "delta_rank_kendall_tau_asym_minus_mmd" in work.columns:
        kendall_col = "delta_rank_kendall_tau_asym_minus_mmd"
    if kendall_col:
        work[kendall_col] = pd.to_numeric(work[kendall_col], errors="coerce")
        vals = work[kendall_col].dropna()
        if not vals.empty:
            out["mean_delta_rank_kendall_tau"] = float(vals.mean())
            out["median_delta_rank_kendall_tau"] = float(vals.median())

    cidx_col = None
    if "delta_rank_pairwise_cindex_a_minus_b" in work.columns:
        cidx_col = "delta_rank_pairwise_cindex_a_minus_b"
    elif "delta_rank_pairwise_cindex_asym_minus_mmd" in work.columns:
        cidx_col = "delta_rank_pairwise_cindex_asym_minus_mmd"
    if cidx_col:
        work[cidx_col] = pd.to_numeric(work[cidx_col], errors="coerce")
        vals = work[cidx_col].dropna()
        if not vals.empty:
            out["mean_delta_rank_pairwise_cindex"] = float(vals.mean())
            out["median_delta_rank_pairwise_cindex"] = float(vals.median())
    return out


def _build_claims_summary(
    lane_budget_comparisons: pd.DataFrame,
    mmd_budget_comparisons: pd.DataFrame,
    mae_tie_epsilon: float,
) -> pd.DataFrame:
    scopes = [
        "all",
        "ols_ridge",
        "model_only_ols_ridge",
        "model_train_benchmark_ols_ridge",
        "model_train_benchmark_all_heads",
        "model_benchmark_ols_ridge",
        "model_benchmark_all_heads",
        "model_train_benchmark_disjoint_ols_ridge",
        "model_train_benchmark_disjoint_all_heads",
        "model_benchmark_trainset_disjoint_ols_ridge",
        "model_benchmark_trainset_disjoint_all_heads",
    ]
    lane_claims = [
        ("motion_beats_appearance", "motion_vs_appearance"),
        ("hybrid_beats_motion", "hybrid_vs_motion"),
        ("hybrid_beats_appearance", "hybrid_vs_appearance"),
    ]

    rows: List[Dict[str, object]] = []
    for scope in scopes:
        lane_sub = _subset_scope(lane_budget_comparisons, scope)
        for claim_name, comp in lane_claims:
            if lane_sub.empty:
                rows.append(_claim_row(lane_sub, scope, claim_name, "delta_rank_spearman_a_minus_b", mae_tie_epsilon))
                continue
            rows.append(
                _claim_row(
                    lane_sub[lane_sub["comparison"].astype(str) == comp].copy(),
                    scope,
                    claim_name,
                    "delta_rank_spearman_a_minus_b",
                    mae_tie_epsilon,
                )
            )
        mmd_sub = _subset_scope(mmd_budget_comparisons, scope)
        rows.append(
            _claim_row(
                mmd_sub,
                scope,
                "asym_beats_mmd",
                "delta_rank_spearman_asym_minus_mmd",
                mae_tie_epsilon,
            )
        )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(["scope", "claim"]).reset_index(drop=True)


def _build_permutation_summary(permutation_df: pd.DataFrame) -> pd.DataFrame:
    if permutation_df.empty:
        return pd.DataFrame()
    scopes = [
        "all",
        "ols_ridge",
        "model_only_ols_ridge",
        "model_train_benchmark_ols_ridge",
        "model_train_benchmark_all_heads",
        "model_benchmark_ols_ridge",
        "model_benchmark_all_heads",
        "model_train_benchmark_disjoint_ols_ridge",
        "model_train_benchmark_disjoint_all_heads",
        "model_benchmark_trainset_disjoint_ols_ridge",
        "model_benchmark_trainset_disjoint_all_heads",
    ]
    rows: List[Dict[str, object]] = []
    for scope in scopes:
        sub = _subset_scope(permutation_df, scope)
        if sub.empty:
            rows.append(
                {
                    "scope": scope,
                    "n_total": 0,
                    "n_sig_rank_spearman": 0,
                    "n_sig_rank_kendall_tau": 0,
                    "n_sig_rank_pairwise_cindex": 0,
                    "n_sig_all_rank_metrics": 0,
                    "sig_rate_rank_spearman": math.nan,
                    "sig_rate_rank_kendall_tau": math.nan,
                    "sig_rate_rank_pairwise_cindex": math.nan,
                    "sig_rate_all_rank_metrics": math.nan,
                    "mean_delta_rank_spearman_obs_minus_perm": math.nan,
                    "mean_delta_rank_kendall_tau_obs_minus_perm": math.nan,
                    "mean_delta_rank_pairwise_cindex_obs_minus_perm": math.nan,
                }
            )
            continue
        p_sp = pd.to_numeric(sub["p_value_rank_spearman_higher_than_perm"], errors="coerce")
        p_k = pd.to_numeric(sub["p_value_rank_kendall_tau_higher_than_perm"], errors="coerce")
        p_c = pd.to_numeric(sub["p_value_rank_pairwise_cindex_higher_than_perm"], errors="coerce")
        d_sp = pd.to_numeric(sub["delta_rank_spearman_obs_minus_perm"], errors="coerce")
        d_k = pd.to_numeric(sub["delta_rank_kendall_tau_obs_minus_perm"], errors="coerce")
        d_c = pd.to_numeric(sub["delta_rank_pairwise_cindex_obs_minus_perm"], errors="coerce")
        valid_sp = p_sp.notna()
        valid_k = p_k.notna()
        valid_c = p_c.notna()
        sig_sp = valid_sp & (p_sp <= 0.05)
        sig_k = valid_k & (p_k <= 0.05)
        sig_c = valid_c & (p_c <= 0.05)
        all_rank = sig_sp & sig_k & sig_c
        n = len(sub)
        rows.append(
            {
                "scope": scope,
                "n_total": int(n),
                "n_sig_rank_spearman": int(sig_sp.sum()),
                "n_sig_rank_kendall_tau": int(sig_k.sum()),
                "n_sig_rank_pairwise_cindex": int(sig_c.sum()),
                "n_sig_all_rank_metrics": int(all_rank.sum()),
                "sig_rate_rank_spearman": float(sig_sp.sum() / n) if n > 0 else math.nan,
                "sig_rate_rank_kendall_tau": float(sig_k.sum() / n) if n > 0 else math.nan,
                "sig_rate_rank_pairwise_cindex": float(sig_c.sum() / n) if n > 0 else math.nan,
                "sig_rate_all_rank_metrics": float(all_rank.sum() / n) if n > 0 else math.nan,
                "mean_delta_rank_spearman_obs_minus_perm": float(d_sp.mean()) if d_sp.notna().any() else math.nan,
                "mean_delta_rank_kendall_tau_obs_minus_perm": float(d_k.mean()) if d_k.notna().any() else math.nan,
                "mean_delta_rank_pairwise_cindex_obs_minus_perm": float(d_c.mean()) if d_c.notna().any() else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("scope").reset_index(drop=True)


def _build_permutation_claim_subset(
    permutation_df: pd.DataFrame,
    lane_budget_comparisons: pd.DataFrame,
    mmd_budget_comparisons: pd.DataFrame,
) -> pd.DataFrame:
    if permutation_df.empty:
        return pd.DataFrame()
    keys: set[Tuple[str, str, int, str]] = set()

    if not lane_budget_comparisons.empty:
        cols = ["protocol", "head", "signal_k", "method_a", "method_b"]
        work = lane_budget_comparisons[cols].copy()
        for _, r in work.iterrows():
            k = int(r["signal_k"])
            keys.add((str(r["protocol"]), str(r["head"]), k, str(r["method_a"])))
            keys.add((str(r["protocol"]), str(r["head"]), k, str(r["method_b"])))

    if not mmd_budget_comparisons.empty:
        cols = ["protocol", "head", "signal_k", "method_asym", "method_mmd"]
        work = mmd_budget_comparisons[cols].copy()
        for _, r in work.iterrows():
            k = int(r["signal_k"])
            keys.add((str(r["protocol"]), str(r["head"]), k, str(r["method_asym"])))
            keys.add((str(r["protocol"]), str(r["head"]), k, str(r["method_mmd"])))

    if not keys:
        return pd.DataFrame()

    sub = permutation_df.copy()
    sub = sub[
        sub.apply(
            lambda r: (
                str(r.get("protocol", "")),
                str(r.get("head", "")),
                int(r.get("signal_k", -1)),
                str(r.get("method", "")),
            )
            in keys,
            axis=1,
        )
    ].copy()
    return sub.reset_index(drop=True)


def _partition_permutation_signal(permutation_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if permutation_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    work = permutation_df.copy()
    work["_p_sp"] = pd.to_numeric(work["p_value_rank_spearman_higher_than_perm"], errors="coerce")
    work["_p_k"] = pd.to_numeric(work["p_value_rank_kendall_tau_higher_than_perm"], errors="coerce")
    work["_p_c"] = pd.to_numeric(work["p_value_rank_pairwise_cindex_higher_than_perm"], errors="coerce")
    work["_delta_sp"] = pd.to_numeric(work["delta_rank_spearman_obs_minus_perm"], errors="coerce")
    work["_delta_k"] = pd.to_numeric(work["delta_rank_kendall_tau_obs_minus_perm"], errors="coerce")
    work["_delta_c"] = pd.to_numeric(work["delta_rank_pairwise_cindex_obs_minus_perm"], errors="coerce")

    signal_mask = (
        (work["_delta_sp"] > 0.0)
        & (work["_delta_k"] > 0.0)
        & (work["_delta_c"] > 0.0)
        & (work["_p_sp"] <= 0.05)
        & (work["_p_k"] <= 0.05)
        & (work["_p_c"] <= 0.05)
    )

    work["signal_label"] = np.where(signal_mask, "signal", "no_signal")
    work["signal_reason"] = np.where(signal_mask, "signal_all_rank_metrics", "mixed_or_missing")

    no_signal_mask = ~signal_mask
    no_signal_reason = np.select(
        [
            no_signal_mask & (work["_delta_sp"] <= 0.0) & (work["_delta_k"] <= 0.0) & (work["_delta_c"] <= 0.0),
            no_signal_mask & (work["_delta_sp"] <= 0.0),
            no_signal_mask & (work["_delta_k"] <= 0.0),
            no_signal_mask & (work["_delta_c"] <= 0.0),
            no_signal_mask & (work["_p_sp"] > 0.05) & (work["_p_k"] > 0.05) & (work["_p_c"] > 0.05),
            no_signal_mask & (work["_p_sp"] > 0.05),
            no_signal_mask & (work["_p_k"] > 0.05),
            no_signal_mask & (work["_p_c"] > 0.05),
        ],
        [
            "worse_than_perm_all_rank_metrics",
            "worse_than_perm_spearman",
            "worse_than_perm_kendall_tau",
            "worse_than_perm_pairwise_cindex",
            "not_significant_all_rank_metrics",
            "not_significant_spearman",
            "not_significant_kendall_tau",
            "not_significant_pairwise_cindex",
        ],
        default="mixed_or_missing",
    )
    work.loc[no_signal_mask, "signal_reason"] = no_signal_reason[no_signal_mask]

    work["_joint_p"] = np.maximum(np.maximum(work["_p_sp"].fillna(1.0), work["_p_k"].fillna(1.0)), work["_p_c"].fillna(1.0))
    work["_delta_sum"] = (
        work["_delta_sp"].fillna(-1.0e9)
        + work["_delta_k"].fillna(-1.0e9)
        + work["_delta_c"].fillna(-1.0e9)
    )

    reason_rank = {
        "worse_than_perm_all_rank_metrics": 0,
        "worse_than_perm_spearman": 1,
        "worse_than_perm_kendall_tau": 2,
        "worse_than_perm_pairwise_cindex": 3,
        "not_significant_all_rank_metrics": 4,
        "not_significant_spearman": 5,
        "not_significant_kendall_tau": 6,
        "not_significant_pairwise_cindex": 7,
        "mixed_or_missing": 8,
    }
    work["_reason_rank"] = work["signal_reason"].map(reason_rank).fillna(99).astype(int)

    signal_df = work[signal_mask].copy()
    signal_df = signal_df.sort_values(
        ["_joint_p", "_delta_sum", "protocol", "head", "lane", "signal_k", "method"],
        ascending=[True, False, True, True, True, True, True],
        na_position="last",
    )

    no_signal_df = work[~signal_mask].copy()
    no_signal_df = no_signal_df.sort_values(
        ["_reason_rank", "_delta_sum", "_joint_p", "protocol", "head", "lane", "signal_k", "method"],
        ascending=[True, True, False, True, True, True, True, True],
        na_position="last",
    )

    drop_cols = ["_p_sp", "_p_k", "_p_c", "_delta_sp", "_delta_k", "_delta_c", "_joint_p", "_delta_sum", "_reason_rank"]
    signal_df = signal_df.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
    no_signal_df = no_signal_df.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
    return signal_df, no_signal_df


def _write_markdown_report(
    out_path: Path,
    args: argparse.Namespace,
    selected: pd.DataFrame,
    summary_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
    permutation_signal_df: pd.DataFrame,
    permutation_no_signal_df: pd.DataFrame,
    claims_summary_df: pd.DataFrame,
    permutation_summary_df: pd.DataFrame,
    permutation_claims_summary_df: pd.DataFrame,
    ranked_all: pd.DataFrame,
    best_by_protocol_head: pd.DataFrame,
    best_by_protocol_head_lane: pd.DataFrame,
    best_by_protocol_head_lane_k: pd.DataFrame,
    lane_budget_comparisons: pd.DataFrame,
    mmd_budget_comparisons: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# Held-Out Model CV Report")
    lines.append("")
    lines.append(f"- Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Output dir: `{out_path.parent}`")
    lines.append(f"- Protocols: `{args.protocols}`")
    lines.append(f"- Heads: `{args.heads}`")
    lines.append(f"- Row source: `{args.row_source}`")
    lines.append(f"- Max hard folds: `{args.max_hard_folds}`")
    lines.append(f"- Holdout train-set size k: `{args.holdout_train_k}`")
    lines.append(f"- Rank grouping: `{args.rank_grouping}`")
    lines.append(f"- CV residualize target by context: `{args.cv_residualize_target_by_context}`")
    lines.append(f"- CV residual context cols: `{args.cv_residual_context_cols}`")
    lines.append(f"- CV residual transform: `{args.cv_residual_target_transform}`")
    lines.append(f"- CV residual eval space: `{args.cv_residual_eval_space}`")
    lines.append(f"- CV few-shot context calibration: `{args.cv_fewshot_context_calibration}`")
    lines.append(f"- CV few-shot context cols: `{args.cv_fewshot_context_calibration_cols}`")
    lines.append(f"- CV few-shot k: `{args.cv_fewshot_context_calibration_k}`")
    lines.append(f"- CV few-shot min group size: `{args.cv_fewshot_context_calibration_min_group_size}`")
    lines.append(f"- CV few-shot backoff: `{args.cv_fewshot_context_calibration_backoff}`")
    lines.append(f"- Max pairs/group (pairwise): `{args.max_pairs_per_group}`")
    lines.append(f"- Permutation samples: `{args.permutation_samples}`")
    lines.append(f"- Permutation mode: `{args.permutation_mode}`")
    lines.append(f"- Signal-k range: `[ {args.min_signal_k}, {args.max_signal_k} ]`")
    lines.append(
        f"- Candidate dedup metric: `{args.candidate_dedup_primary_metric}` (tie: `{args.candidate_dedup_tiebreak_metric}`)"
    )
    lines.append(
        f"- Candidate selection metric: `{args.candidate_selection_primary_metric}` (tie: `{args.candidate_selection_tiebreak_metric}`)"
    )
    lines.append("")

    lines.append("## Selected Candidates")
    if selected.empty:
        lines.append("- none")
    else:
        show = selected[
            [
                "candidate_id",
                "lane",
                "variant",
                "method",
                "signal_k",
                "n_predictors_total",
                "jointood_mae",
                "jointood_spearman",
            ]
        ].copy()
        lines.append(show.to_markdown(index=False))
    lines.append("")

    lines.append("## Best Per Protocol/Head (Rank-Selected)")
    if best_by_protocol_head.empty:
        lines.append("- none")
    else:
        show_cols = [
            "protocol",
            "head",
            "candidate_id",
            "lane",
            "signal_k",
            "method",
            "mae",
            "rmse",
            "spearman",
            "rank_regret",
            "rank_spearman",
            "rank_kendall_tau",
            "rank_pairwise_cindex",
            "rank_abs_err",
            "rank_pct_err",
            "n_calibration",
            "n_rows_scored",
            "n_folds_scored",
        ]
        lines.append(best_by_protocol_head[show_cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Best Per Protocol/Head/Lane (Rank-Selected)")
    if best_by_protocol_head_lane.empty:
        lines.append("- none")
    else:
        show_cols = [
            "protocol",
            "head",
            "lane",
            "candidate_id",
            "signal_k",
            "method",
            "mae",
            "spearman",
            "rank_spearman",
            "rank_kendall_tau",
            "rank_pairwise_cindex",
            "rank_regret",
            "rank_abs_err",
            "rank_pct_err",
            "n_calibration",
            "n_folds_scored",
        ]
        lines.append(best_by_protocol_head_lane[show_cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Best Per Protocol/Head/Lane/Signal-k (Rank-Selected, Budget Matched Winners)")
    if best_by_protocol_head_lane_k.empty:
        lines.append("- none")
    else:
        show_cols = [
            "protocol",
            "head",
            "lane",
            "signal_k",
            "candidate_id",
            "method",
            "mae",
            "spearman",
            "rank_spearman",
            "rank_kendall_tau",
            "rank_pairwise_cindex",
            "rank_regret",
            "rank_abs_err",
            "rank_pct_err",
            "n_calibration",
            "n_folds_scored",
        ]
        lines.append(best_by_protocol_head_lane_k[show_cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Parameter-Matched Lane Comparisons (by Signal-k)")
    if lane_budget_comparisons.empty:
        lines.append("- none")
    else:
        show_cols = [
            "protocol",
            "head",
            "signal_k",
            "comparison",
            "method_a",
            "method_b",
            "mae_a",
            "mae_b",
            "delta_mae_b_minus_a",
            "spearman_a",
            "spearman_b",
            "delta_spearman_a_minus_b",
            "rank_spearman_a",
            "rank_spearman_b",
            "delta_rank_spearman_a_minus_b",
            "rank_kendall_tau_a",
            "rank_kendall_tau_b",
            "delta_rank_kendall_tau_a_minus_b",
            "rank_pairwise_cindex_a",
            "rank_pairwise_cindex_b",
            "delta_rank_pairwise_cindex_a_minus_b",
            "rank_regret_a",
            "rank_regret_b",
            "delta_regret_b_minus_a",
            "selection_basis",
            "verdict",
        ]
        lines.append(lane_budget_comparisons[show_cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Parameter-Matched Asym vs MMD (by Signal-k)")
    if mmd_budget_comparisons.empty:
        lines.append("- none")
    else:
        show_cols = [
            "protocol",
            "head",
            "signal_k",
            "method_asym",
            "method_mmd",
            "mae_asym",
            "mae_mmd",
            "delta_mae_mmd_minus_asym",
            "spearman_asym",
            "spearman_mmd",
            "delta_spearman_asym_minus_mmd",
            "rank_spearman_asym",
            "rank_spearman_mmd",
            "delta_rank_spearman_asym_minus_mmd",
            "rank_kendall_tau_asym",
            "rank_kendall_tau_mmd",
            "delta_rank_kendall_tau_asym_minus_mmd",
            "rank_pairwise_cindex_asym",
            "rank_pairwise_cindex_mmd",
            "delta_rank_pairwise_cindex_asym_minus_mmd",
            "rank_regret_asym",
            "rank_regret_mmd",
            "delta_regret_mmd_minus_asym",
            "selection_basis",
            "verdict",
        ]
        lines.append(mmd_budget_comparisons[show_cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Collapsed Claim Verification")
    lines.append(f"- Rank-Spearman tie epsilon for support/against: `{args.claim_mae_tie_epsilon}`")
    if claims_summary_df.empty:
        lines.append("- none")
    else:
        show_cols = [
            "scope",
            "claim",
            "n_total",
            "n_support",
            "n_against",
            "n_near_tie",
            "support_rate",
            "mean_delta_primary",
            "median_delta_primary",
            "mean_delta_rank_kendall_tau",
            "median_delta_rank_kendall_tau",
            "mean_delta_rank_pairwise_cindex",
            "median_delta_rank_pairwise_cindex",
        ]
        lines.append(claims_summary_df[show_cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Permutation Baseline (Collapsed Summary)")
    if permutation_summary_df.empty:
        lines.append("- disabled or none")
    else:
        show_cols = [
            "scope",
            "n_total",
            "n_sig_rank_spearman",
            "n_sig_rank_kendall_tau",
            "n_sig_rank_pairwise_cindex",
            "n_sig_all_rank_metrics",
            "sig_rate_rank_spearman",
            "sig_rate_rank_kendall_tau",
            "sig_rate_rank_pairwise_cindex",
            "sig_rate_all_rank_metrics",
            "mean_delta_rank_spearman_obs_minus_perm",
            "mean_delta_rank_kendall_tau_obs_minus_perm",
            "mean_delta_rank_pairwise_cindex_obs_minus_perm",
        ]
        lines.append(permutation_summary_df[show_cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Permutation Baseline (Claim Methods Only)")
    if permutation_claims_summary_df.empty:
        lines.append("- disabled or none")
    else:
        show_cols = [
            "scope",
            "n_total",
            "n_sig_rank_spearman",
            "n_sig_rank_kendall_tau",
            "n_sig_rank_pairwise_cindex",
            "n_sig_all_rank_metrics",
            "sig_rate_rank_spearman",
            "sig_rate_rank_kendall_tau",
            "sig_rate_rank_pairwise_cindex",
            "sig_rate_all_rank_metrics",
            "mean_delta_rank_spearman_obs_minus_perm",
            "mean_delta_rank_kendall_tau_obs_minus_perm",
            "mean_delta_rank_pairwise_cindex_obs_minus_perm",
        ]
        lines.append(permutation_claims_summary_df[show_cols].to_markdown(index=False))
    lines.append("")

    lines.append("## All Successful Runs (Sorted)")
    if ranked_all.empty:
        lines.append("- none")
    else:
        show_cols = [
            "protocol",
            "head",
            "lane",
            "signal_k",
            "candidate_id",
            "method",
            "mae",
            "rmse",
            "spearman",
            "rank_regret",
            "rank_spearman",
            "rank_kendall_tau",
            "rank_pairwise_cindex",
            "rank_abs_err",
            "rank_pct_err",
            "n_calibration",
            "n_folds_scored",
        ]
        lines.append(ranked_all[show_cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Permutation Baseline (Rank Metrics, Within-Group Target Shuffle)")
    if permutation_df.empty:
        lines.append("- disabled or none")
    else:
        show_cols = [
            "protocol",
            "head",
            "lane",
            "signal_k",
            "candidate_id",
            "method",
            "n_rows",
            "n_rank_groups",
            "n_permutations_used",
            "obs_rank_spearman",
            "perm_rank_spearman_mean",
            "delta_rank_spearman_obs_minus_perm",
            "p_value_rank_spearman_higher_than_perm",
            "obs_rank_kendall_tau",
            "perm_rank_kendall_tau_mean",
            "delta_rank_kendall_tau_obs_minus_perm",
            "p_value_rank_kendall_tau_higher_than_perm",
            "obs_rank_pairwise_cindex",
            "perm_rank_pairwise_cindex_mean",
            "delta_rank_pairwise_cindex_obs_minus_perm",
            "p_value_rank_pairwise_cindex_higher_than_perm",
        ]
        lines.append(permutation_df[show_cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Permutation Signal vs No-Signal (Sorted)")
    lines.append(
        "- Signal criteria: all three rank deltas (`spearman`, `kendall_tau`, `pairwise_cindex`) are `> 0` and all three permutation p-values are `<= 0.05`."
    )
    if permutation_df.empty:
        lines.append("- disabled or none")
    else:
        n_total = len(permutation_df)
        n_sig = len(permutation_signal_df)
        n_no = len(permutation_no_signal_df)
        lines.append(f"- Signal rows: `{n_sig}/{n_total}` ({(n_sig / n_total):.1%})")
        lines.append(f"- No-signal rows: `{n_no}/{n_total}` ({(n_no / n_total):.1%})")
        lines.append("")

        show_cols = [
            "protocol",
            "head",
            "lane",
            "signal_k",
            "candidate_id",
            "method",
            "n_rows",
            "obs_rank_spearman",
            "perm_rank_spearman_mean",
            "delta_rank_spearman_obs_minus_perm",
            "p_value_rank_spearman_higher_than_perm",
            "obs_rank_kendall_tau",
            "perm_rank_kendall_tau_mean",
            "delta_rank_kendall_tau_obs_minus_perm",
            "p_value_rank_kendall_tau_higher_than_perm",
            "obs_rank_pairwise_cindex",
            "perm_rank_pairwise_cindex_mean",
            "delta_rank_pairwise_cindex_obs_minus_perm",
            "p_value_rank_pairwise_cindex_higher_than_perm",
            "signal_reason",
        ]
        lines.append("### Signal Rows (Sorted strongest first)")
        if permutation_signal_df.empty:
            lines.append("- none")
        else:
            lines.append(permutation_signal_df[show_cols].to_markdown(index=False))
        lines.append("")

        lines.append("### No-Signal Rows (Sorted worst first)")
        if permutation_no_signal_df.empty:
            lines.append("- none")
        else:
            lines.append(permutation_no_signal_df[show_cols].to_markdown(index=False))
        lines.append("")

    lines.append("## Red Flags")
    flags: List[str] = []
    if not summary_df.empty:
        bad = summary_df[summary_df["status"] != "ok"]
        if not bad.empty:
            flags.append(f"{len(bad)} runs failed or were skipped; see `heldout_model_cv_summary.csv`.")
        ok = summary_df[summary_df["status"] == "ok"]
        if not ok.empty:
            low_folds = ok[ok["n_folds_scored"] < 3]
            if not low_folds.empty:
                flags.append(f"{len(low_folds)} runs have <3 scored folds.")
            weak_rank = ok[pd.to_numeric(ok["rank_spearman"], errors="coerce") < 0.05]
            if not weak_rank.empty:
                flags.append(f"{len(weak_rank)} runs have near-zero rank Spearman (<0.05).")
    if not permutation_df.empty:
        weak_sp = permutation_df[
            pd.to_numeric(permutation_df["p_value_rank_spearman_higher_than_perm"], errors="coerce") > 0.05
        ]
        weak_k = permutation_df[
            pd.to_numeric(permutation_df["p_value_rank_kendall_tau_higher_than_perm"], errors="coerce") > 0.05
        ]
        weak_c = permutation_df[
            pd.to_numeric(permutation_df["p_value_rank_pairwise_cindex_higher_than_perm"], errors="coerce") > 0.05
        ]
        if not weak_sp.empty:
            flags.append(f"{len(weak_sp)} runs are not significantly better than permutation on rank Spearman (p>0.05).")
        if not weak_k.empty:
            flags.append(f"{len(weak_k)} runs are not significantly better than permutation on rank Kendall tau (p>0.05).")
        if not weak_c.empty:
            flags.append(f"{len(weak_c)} runs are not significantly better than permutation on rank pairwise C-index (p>0.05).")
    if not flags:
        lines.append("- No immediate red flags.")
    else:
        for f in flags:
            lines.append(f"- {f}")
    lines.append("")

    out_path.write_text("\n".join(lines))


def _default_run_roots() -> List[Path]:
    base = Path("analysis_comprehensive_runs")
    names = [
        "hof_motion_v3_density_jointood_full_ridge_a10_no_family_v1",
        "hof_motion_v3_density_jointood_full_ridge_a10_no_family_no_density_v1",
        "hof_motion_v3_density_jointood_full_ols_no_family_v1",
        "hof_motion_v3_density_jointood_full_ols_no_family_no_density_v1",
    ]
    return [base / n for n in names]


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out model-group CV on existing joint-OOD row tables.")
    parser.add_argument("--run-roots", default="", help="Comma-separated run roots. Default: no-family four-pack.")
    parser.add_argument("--output-dir", default="analysis_comprehensive_runs/heldout_model_cv_v1")
    parser.add_argument(
        "--row-source",
        default="raw",
        choices=["raw", "prediction"],
        help="Use raw `auc_with_features.csv` (recommended) or collapsed `prediction_jointood_rows.csv`.",
    )
    parser.add_argument("--heads", default="ols,ridge,pairwise_rank", help="CSV list of heads.")
    parser.add_argument(
        "--protocols",
        default="model_benchmark_trainset_disjoint",
        help=(
            "CSV list: model_only,model_train_benchmark,model_benchmark,"
            "model_train_benchmark_disjoint,model_benchmark_trainset_disjoint"
        ),
    )
    parser.add_argument("--lanes", default="motion_only,appearance_only,hybrid", help="CSV lane filter.")
    parser.add_argument("--top-n-per-lane", type=int, default=2, help="Best candidates per lane from each pool.")
    parser.add_argument("--min-signal-k", type=int, default=1, help="Minimum signal predictor count for candidate selection.")
    parser.add_argument("--max-signal-k", type=int, default=8, help="Maximum signal predictor count for candidate selection.")
    parser.add_argument(
        "--candidate-dedup-primary-metric",
        default="jointood_mae",
        help="Metric used to pick best variant per signature in candidate pool.",
    )
    parser.add_argument(
        "--candidate-dedup-tiebreak-metric",
        default="jointood_spearman",
        help="Tie-break metric for per-signature candidate pool dedup.",
    )
    parser.add_argument(
        "--candidate-selection-primary-metric",
        default="jointood_mae",
        help="Primary metric used to rank candidates within each lane.",
    )
    parser.add_argument(
        "--candidate-selection-tiebreak-metric",
        default="jointood_spearman",
        help="Tie-break metric used to rank candidates within each lane.",
    )
    parser.add_argument("--include-pairwise-candidates", action="store_true", help="Allow *_pairwise candidates in selection.")
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument(
        "--target-col",
        default="auc_normalized_observed",
        help="Target column name preference; script falls back to auc_normalized_observed/target if needed.",
    )
    parser.add_argument(
        "--cv-residualize-target-by-context",
        dest="cv_residualize_target_by_context",
        action="store_true",
        help=(
            "Residualize target per context within each fold using train-fold stats only "
            "(supports mean-only residual or z-score)."
        ),
    )
    parser.add_argument(
        "--no-cv-residualize-target-by-context",
        dest="cv_residualize_target_by_context",
        action="store_false",
        help="Disable fold-wise context target residualization.",
    )
    parser.set_defaults(cv_residualize_target_by_context=False)
    parser.add_argument(
        "--cv-residual-context-cols",
        default="",
        help=(
            "CSV context columns for target residualization/z-score "
            "(e.g., benchmark,model_family_encoder)."
        ),
    )
    parser.add_argument(
        "--cv-residual-target-transform",
        choices=["residual", "zscore"],
        default="residual",
        help="Context target transform mode: subtract mean (`residual`) or z-score (`zscore`).",
    )
    parser.add_argument(
        "--cv-residual-target-std-eps",
        type=float,
        default=1e-9,
        help="Epsilon floor for context std when --cv-residual-target-transform=zscore.",
    )
    parser.add_argument(
        "--cv-residual-eval-space",
        choices=["residual", "absolute"],
        default="residual",
        help=(
            "Metric space for predictions: `residual` keeps transformed target space; "
            "`absolute` maps predictions back using fold-safe context stats."
        ),
    )
    parser.add_argument(
        "--cv-fewshot-context-calibration",
        dest="cv_fewshot_context_calibration",
        action="store_true",
        help=(
            "Apply context-aware calibration on heldout fold predictions "
            "(uses fold-internal calibration rows when k>0)."
        ),
    )
    parser.add_argument(
        "--no-cv-fewshot-context-calibration",
        dest="cv_fewshot_context_calibration",
        action="store_false",
        help="Disable few-shot context calibration.",
    )
    parser.set_defaults(cv_fewshot_context_calibration=False)
    parser.add_argument(
        "--cv-fewshot-context-calibration-cols",
        default="",
        help=(
            "CSV context columns for few-shot calibration "
            "(e.g., benchmark,model_family_encoder)."
        ),
    )
    parser.add_argument(
        "--cv-fewshot-context-calibration-std-eps",
        type=float,
        default=1e-9,
        help="Epsilon floor for prediction/target std inside few-shot calibration.",
    )
    parser.add_argument(
        "--cv-fewshot-context-calibration-min-group-size",
        type=int,
        default=2,
        help="Minimum calibration samples per context group.",
    )
    parser.add_argument(
        "--cv-fewshot-context-calibration-backoff",
        dest="cv_fewshot_context_calibration_backoff",
        action="store_true",
        help="Allow hierarchical context backoff for few-shot calibration.",
    )
    parser.add_argument(
        "--no-cv-fewshot-context-calibration-backoff",
        dest="cv_fewshot_context_calibration_backoff",
        action="store_false",
        help="Disable context backoff in few-shot calibration.",
    )
    parser.set_defaults(cv_fewshot_context_calibration_backoff=True)
    parser.add_argument(
        "--cv-fewshot-context-calibration-k",
        type=int,
        default=0,
        help=(
            "Calibration shots per context group from each heldout fold. "
            "0 means use all heldout rows for calibration (oracle-style)."
        ),
    )
    parser.add_argument(
        "--cv-fewshot-context-calibration-seed",
        type=int,
        default=0,
        help="Random seed base for few-shot calibration row sampling.",
    )
    parser.add_argument("--train-col", default="train_dataset")
    parser.add_argument("--benchmark-col", default="benchmark")
    parser.add_argument("--option-col", default="train_dataset")
    parser.add_argument("--model-group-col", default="model_family_encoder")
    parser.add_argument("--min-train-rows", type=int, default=64)

    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--max-pairs-per-group", type=int, default=2000)
    parser.add_argument("--pairwise-max-iter", type=int, default=150)
    parser.add_argument("--pairwise-lr", type=float, default=0.1)
    parser.add_argument(
        "--pairwise-group-cols",
        default="benchmark",
        help="CSV columns for independent pairwise ranking groups (default: benchmark).",
    )

    parser.add_argument(
        "--max-hard-folds",
        type=int,
        default=150,
        help=(
            "For model_train_benchmark/model_benchmark/model_train_benchmark_disjoint/"
            "model_benchmark_trainset_disjoint; <=0 uses all folds."
        ),
    )
    parser.add_argument(
        "--holdout-train-k",
        type=int,
        default=3,
        help="For model_benchmark_trainset_disjoint: number of held-out train datasets per fold (min 2).",
    )
    parser.add_argument(
        "--rank-grouping",
        default="fold_benchmark",
        choices=["fold_benchmark", "benchmark"],
        help="How to aggregate ranking summaries: fold_benchmark (default) or benchmark (legacy pooled).",
    )
    parser.add_argument(
        "--rank-context-cols",
        default="",
        help=(
            "Optional CSV of additional grouping columns for rank metrics/permutation "
            "(e.g., __model_group__ or model_family_encoder)."
        ),
    )
    parser.add_argument("--rank-topk-frac", type=float, default=0.2)
    parser.add_argument("--rank-topk-min", type=int, default=1)
    parser.add_argument(
        "--permutation-samples",
        type=int,
        default=0,
        help="Target-shuffle permutations per run for null baseline (0 disables).",
    )
    parser.add_argument(
        "--permutation-mode",
        default="benchmark",
        choices=["benchmark", "fold", "global"],
        help="Permutation grouping: benchmark (recommended), fold, or global.",
    )
    parser.add_argument(
        "--claim-mae-tie-epsilon",
        type=float,
        default=0.1,
        help="Treat |delta MAE| <= epsilon as near-tie in collapsed claim summaries.",
    )

    parser.add_argument("--save-pred-rows", action="store_true", help="Write OOF prediction rows CSV.")
    parser.add_argument("--save-fold-rows", action="store_true", help="Write per-fold metrics CSV.")
    args = parser.parse_args()

    if args.run_roots.strip():
        run_roots = [Path(x.strip()) for x in args.run_roots.split(",") if x.strip()]
    else:
        run_roots = _default_run_roots()

    heads = [x.strip() for x in args.heads.split(",") if x.strip()]
    protocols = [x.strip() for x in args.protocols.split(",") if x.strip()]
    lanes = [x.strip() for x in args.lanes.split(",") if x.strip()]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = _collect_candidate_pool(
        run_roots=run_roots,
        include_pairwise_candidates=bool(args.include_pairwise_candidates),
        dedup_primary_metric=str(args.candidate_dedup_primary_metric),
        dedup_tiebreak_metric=str(args.candidate_dedup_tiebreak_metric),
    )
    if pool.empty:
        raise SystemExit("No candidate methods collected from run roots.")
    pool.to_csv(out_dir / "candidate_pool.csv", index=False)

    selected = _select_candidates(
        pool=pool,
        lanes=lanes,
        top_n_per_lane=int(args.top_n_per_lane),
        min_signal_k=int(args.min_signal_k),
        max_signal_k=int(args.max_signal_k),
        selection_primary_metric=str(args.candidate_selection_primary_metric),
        selection_tiebreak_metric=str(args.candidate_selection_tiebreak_metric),
    )
    if selected.empty:
        raise SystemExit("No candidates selected.")
    selected.to_csv(out_dir / "selected_candidates.csv", index=False)

    rng = np.random.default_rng(int(args.seed))
    summary_rows: List[Dict[str, object]] = []
    permutation_rows: List[Dict[str, object]] = []
    pred_frames: List[pd.DataFrame] = []
    fold_frames: List[pd.DataFrame] = []

    total = len(selected) * len(heads) * len(protocols)
    done = 0
    for _, cand in selected.iterrows():
        for head in heads:
            for protocol in protocols:
                done += 1
                print(
                    f"[{done}/{total}] candidate={cand['candidate_id']} method={cand['method']} "
                    f"head={head} protocol={protocol}"
                )
                res = _evaluate_one(cand=cand, head=head, protocol=protocol, args=args, rng=rng)
                summary_rows.append(res.summary)
                if (
                    int(args.permutation_samples) > 0
                    and res.summary.get("status") == "ok"
                    and not res.pred_rows.empty
                ):
                    perm_row = _permutation_baseline(
                        pred_df=res.pred_rows,
                        target_col="target_eval",
                        option_col=str(res.summary.get("option_col_used", args.option_col)),
                        benchmark_col=str(args.benchmark_col),
                        rank_grouping=str(args.rank_grouping),
                        rank_context_cols=[
                            c.strip()
                            for c in str(res.summary.get("rank_context_cols_used", "")).split(",")
                            if c.strip()
                        ],
                        n_permutations=int(args.permutation_samples),
                        rng=rng,
                        mode=str(args.permutation_mode),
                    )
                    perm_row.update(
                        {
                            "candidate_id": str(cand["candidate_id"]),
                            "variant": str(cand["variant"]),
                            "method": str(cand["method"]),
                            "lane": str(cand["lane"]),
                            "signal_k": int(cand["signal_k"]),
                            "n_predictors_total": int(cand["n_predictors_total"]),
                            "head": head,
                            "protocol": protocol,
                            "status": str(res.summary.get("status", "ok")),
                        }
                    )
                    permutation_rows.append(perm_row)
                if args.save_pred_rows and not res.pred_rows.empty:
                    pred_frames.append(res.pred_rows)
                if args.save_fold_rows and not res.fold_rows.empty:
                    fold_frames.append(res.fold_rows)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        [
            "status",
            "protocol",
            "head",
            "rank_spearman",
            "rank_pairwise_cindex",
            "rank_kendall_tau",
            "rank_regret",
            "mae",
        ],
        ascending=[True, True, True, False, False, False, True, True],
        na_position="last",
    )
    summary_df.to_csv(out_dir / "heldout_model_cv_summary.csv", index=False)
    permutation_df = pd.DataFrame(permutation_rows)
    if not permutation_df.empty:
        permutation_df = permutation_df.sort_values(
            ["protocol", "head", "lane", "signal_k", "p_value_rank_spearman_higher_than_perm"],
            na_position="last",
        ).reset_index(drop=True)
    permutation_df.to_csv(out_dir / "heldout_model_cv_permutation_baseline.csv", index=False)
    permutation_signal_df, permutation_no_signal_df = _partition_permutation_signal(permutation_df)
    permutation_signal_df.to_csv(out_dir / "heldout_model_cv_permutation_signal_rows.csv", index=False)
    permutation_no_signal_df.to_csv(out_dir / "heldout_model_cv_permutation_no_signal_rows.csv", index=False)

    if args.save_pred_rows and pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_csv(out_dir / "heldout_model_cv_pred_rows.csv", index=False)
    if args.save_fold_rows and fold_frames:
        pd.concat(fold_frames, ignore_index=True).to_csv(out_dir / "heldout_model_cv_fold_rows.csv", index=False)

    (
        ranked_all,
        best_by_protocol_head,
        best_by_protocol_head_lane,
        best_by_protocol_head_lane_k,
        lane_budget_comparisons,
        mmd_budget_comparisons,
    ) = _build_comparison_tables(summary_df)
    ranked_all.to_csv(out_dir / "heldout_model_cv_ranked_all.csv", index=False)
    best_by_protocol_head.to_csv(out_dir / "heldout_model_cv_best_by_protocol_head.csv", index=False)
    best_by_protocol_head_lane.to_csv(out_dir / "heldout_model_cv_best_by_protocol_head_lane.csv", index=False)
    best_by_protocol_head_lane_k.to_csv(
        out_dir / "heldout_model_cv_best_by_protocol_head_lane_signalk.csv", index=False
    )
    lane_budget_comparisons.to_csv(
        out_dir / "heldout_model_cv_parameter_matched_lane_comparisons.csv", index=False
    )
    mmd_budget_comparisons.to_csv(
        out_dir / "heldout_model_cv_parameter_matched_asym_vs_mmd.csv", index=False
    )
    claims_summary_df = _build_claims_summary(
        lane_budget_comparisons=lane_budget_comparisons,
        mmd_budget_comparisons=mmd_budget_comparisons,
        mae_tie_epsilon=float(args.claim_mae_tie_epsilon),
    )
    claims_summary_df.to_csv(out_dir / "heldout_model_cv_claims_summary.csv", index=False)
    permutation_summary_df = _build_permutation_summary(permutation_df)
    permutation_summary_df.to_csv(out_dir / "heldout_model_cv_permutation_summary.csv", index=False)
    permutation_claim_subset = _build_permutation_claim_subset(
        permutation_df=permutation_df,
        lane_budget_comparisons=lane_budget_comparisons,
        mmd_budget_comparisons=mmd_budget_comparisons,
    )
    permutation_claims_summary_df = _build_permutation_summary(permutation_claim_subset)
    permutation_claims_summary_df.to_csv(
        out_dir / "heldout_model_cv_permutation_claims_summary.csv", index=False
    )

    cfg = {
        "run_roots": [str(p) for p in run_roots],
        "heads": heads,
        "protocols": protocols,
        "row_source": str(args.row_source),
        "lanes": lanes,
        "top_n_per_lane": int(args.top_n_per_lane),
        "min_signal_k": int(args.min_signal_k),
        "max_signal_k": int(args.max_signal_k),
        "candidate_dedup_primary_metric": str(args.candidate_dedup_primary_metric),
        "candidate_dedup_tiebreak_metric": str(args.candidate_dedup_tiebreak_metric),
        "candidate_selection_primary_metric": str(args.candidate_selection_primary_metric),
        "candidate_selection_tiebreak_metric": str(args.candidate_selection_tiebreak_metric),
        "seed": int(args.seed),
        "target_col": str(args.target_col),
        "cv_residualize_target_by_context": bool(args.cv_residualize_target_by_context),
        "cv_residual_context_cols": str(args.cv_residual_context_cols),
        "cv_residual_target_transform": str(args.cv_residual_target_transform),
        "cv_residual_target_std_eps": float(args.cv_residual_target_std_eps),
        "cv_residual_eval_space": str(args.cv_residual_eval_space),
        "cv_fewshot_context_calibration": bool(args.cv_fewshot_context_calibration),
        "cv_fewshot_context_calibration_cols": str(args.cv_fewshot_context_calibration_cols),
        "cv_fewshot_context_calibration_std_eps": float(args.cv_fewshot_context_calibration_std_eps),
        "cv_fewshot_context_calibration_min_group_size": int(
            args.cv_fewshot_context_calibration_min_group_size
        ),
        "cv_fewshot_context_calibration_backoff": bool(args.cv_fewshot_context_calibration_backoff),
        "cv_fewshot_context_calibration_k": int(args.cv_fewshot_context_calibration_k),
        "cv_fewshot_context_calibration_seed": int(args.cv_fewshot_context_calibration_seed),
        "train_col": str(args.train_col),
        "benchmark_col": str(args.benchmark_col),
        "option_col": str(args.option_col),
        "model_group_col": str(args.model_group_col),
        "ridge_alpha": float(args.ridge_alpha),
        "max_pairs_per_group": int(args.max_pairs_per_group),
        "pairwise_max_iter": int(args.pairwise_max_iter),
        "pairwise_lr": float(args.pairwise_lr),
        "pairwise_group_cols": str(args.pairwise_group_cols),
        "max_hard_folds": int(args.max_hard_folds),
        "holdout_train_k": int(args.holdout_train_k),
        "rank_grouping": str(args.rank_grouping),
        "rank_context_cols": str(args.rank_context_cols),
        "permutation_samples": int(args.permutation_samples),
        "permutation_mode": str(args.permutation_mode),
        "claim_mae_tie_epsilon": float(args.claim_mae_tie_epsilon),
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    _write_markdown_report(
        out_path=out_dir / "heldout_model_cv_report.md",
        args=args,
        selected=selected,
        summary_df=summary_df,
        permutation_df=permutation_df,
        permutation_signal_df=permutation_signal_df,
        permutation_no_signal_df=permutation_no_signal_df,
        claims_summary_df=claims_summary_df,
        permutation_summary_df=permutation_summary_df,
        permutation_claims_summary_df=permutation_claims_summary_df,
        ranked_all=ranked_all,
        best_by_protocol_head=best_by_protocol_head,
        best_by_protocol_head_lane=best_by_protocol_head_lane,
        best_by_protocol_head_lane_k=best_by_protocol_head_lane_k,
        lane_budget_comparisons=lane_budget_comparisons,
        mmd_budget_comparisons=mmd_budget_comparisons,
    )
    print(f"Wrote held-out model CV outputs to: {out_dir}")


if __name__ == "__main__":
    main()
