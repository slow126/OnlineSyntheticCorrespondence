#!/usr/bin/env python3
"""
Build a final utility sweep across multiple run roots.

Outputs:
  - Candidate/summary CSVs
  - Selected finalists at exact-k and <=k by lane
  - Residual diagnostics and post-hoc calibration for selected finalists
  - Human-readable markdown report
  - Fit/tradeoff plots
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_BOOTSTRAP_SAMPLES = 400
DEFAULT_PERMUTATION_SAMPLES = 300


CONTROL_PREFIXES = (
    "log_n_samples_",
    "log_avg_flows_",
    "n_samples_",
    "avg_flows_",
    "enc_",
    "mf_",
)

# Collapse near-duplicate flow quantile variants to a common canonical token.
TOKEN_EQUIV_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"^flow_train_to_eval_auc$"), "flow_train_to_eval_quantile"),
    (re.compile(r"^flow_eval_to_train_auc$"), "flow_eval_to_train_quantile"),
    (re.compile(r"^flow_train_to_eval_eps_at\d+$"), "flow_train_to_eval_quantile"),
    (re.compile(r"^flow_eval_to_train_eps_at\d+$"), "flow_eval_to_train_quantile"),
]


@dataclass
class ResidualDiag:
    n: float = math.nan
    mae: float = math.nan
    rmse: float = math.nan
    medae: float = math.nan
    p90ae: float = math.nan
    bias: float = math.nan
    cal_slope: float = math.nan
    cal_intercept: float = math.nan
    spearman: float = math.nan
    pearson: float = math.nan


@dataclass
class CalibrationFit:
    slope: float = math.nan
    intercept: float = math.nan


def _canonical_token(token: str) -> str:
    t = token.strip()
    for pat, rep in TOKEN_EQUIV_PATTERNS:
        if pat.match(t):
            return rep
    return t


def _is_control_token(token: str) -> bool:
    return token.startswith(CONTROL_PREFIXES)


def _split_predictors(text: object) -> List[str]:
    if text is None:
        return []
    raw = str(text).strip()
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _signal_tokens(text: object) -> List[str]:
    out: List[str] = []
    for tok in _split_predictors(text):
        if _is_control_token(tok):
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


def _model_for_root(root: Path) -> str:
    name = root.name
    if "pairwise" in name:
        return "pairwise_rank"
    if "_ridge_" in name:
        return "ridge"
    if "_ols_" in name:
        return "ols"
    return "ridge" if "ridge" in name else "ols"


def _as_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _safe_spearman(pred: pd.Series, target: pd.Series) -> float:
    if len(pred) < 3:
        return math.nan
    return float(pred.corr(target, method="spearman"))


def _safe_pearson(pred: pd.Series, target: pd.Series) -> float:
    if len(pred) < 3:
        return math.nan
    return float(pred.corr(target, method="pearson"))


def _kendall_tau_b(pred: pd.Series, target: pd.Series) -> float:
    x = pd.to_numeric(pred, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return math.nan
    x = x[finite]
    y = y[finite]
    n = int(len(x))
    if n < 2:
        return math.nan
    n_conc = 0
    n_disc = 0
    n_tie_x = 0
    n_tie_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
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


def _pairwise_cindex(target: pd.Series, pred: pd.Series) -> float:
    y_true = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(pred, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.sum() < 2:
        return math.nan
    y_true = y_true[finite]
    y_pred = y_pred[finite]
    n = int(len(y_true))
    if n < 2:
        return math.nan
    comparable = 0
    score = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dy_true = y_true[i] - y_true[j]
            if dy_true == 0.0:
                continue
            comparable += 1
            dy_pred = y_pred[i] - y_pred[j]
            if dy_pred == 0.0:
                score += 0.5
            elif dy_true * dy_pred > 0.0:
                score += 1.0
    if comparable == 0:
        return math.nan
    return float(score / comparable)


def _mode_stat(values: np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return math.nan
    vc = pd.Series(np.round(arr, 6)).value_counts()
    if vc.empty:
        return math.nan
    top = vc[vc == vc.iloc[0]].index.to_numpy(dtype=float)
    if top.size == 0:
        return math.nan
    return float(np.mean(top))


def _bootstrap_ci(
    values: np.ndarray,
    stat: str,
    n_boot: int = 400,
    seed: int = 0,
) -> Tuple[float, float]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    n = int(arr.size)
    if n < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    draws: List[float] = []
    for _ in range(max(int(n_boot), 50)):
        sample = arr[rng.integers(0, n, size=n)]
        if stat == "mean":
            val = float(np.mean(sample))
        elif stat == "median":
            val = float(np.median(sample))
        else:
            val = _mode_stat(sample)
        if np.isfinite(val):
            draws.append(val)
    if len(draws) < 20:
        return math.nan, math.nan
    q = np.quantile(np.asarray(draws, dtype=float), [0.025, 0.975])
    return float(q[0]), float(q[1])


def _rank_metrics_from_jointood_rows(
    rows: pd.DataFrame,
    option_col: str = "train_dataset",
    benchmark_col: str = "benchmark",
) -> Dict[str, float]:
    out = {
        "jointood_rank_spearman": math.nan,
        "jointood_rank_kendall_tau": math.nan,
        "jointood_rank_pairwise_cindex": math.nan,
        "jointood_rank_pct_err": math.nan,
    }
    if rows.empty:
        return out
    required = [option_col, benchmark_col, "prediction", "target"]
    if any(c not in rows.columns for c in required):
        return out

    work = rows.dropna(subset=required).copy()
    if work.empty:
        return out

    per_bench: List[Dict[str, float]] = []
    for _, sub in work.groupby(benchmark_col, dropna=False):
        grouped = (
            sub.groupby(option_col, dropna=False)
            .agg(pred_mean=("prediction", "mean"), true_mean=("target", "mean"))
            .reset_index(drop=False)
        )
        if len(grouped) < 2:
            continue
        grouped["rank_true"] = grouped["true_mean"].rank(ascending=False, method="min")
        grouped["rank_pred"] = grouped["pred_mean"].rank(ascending=False, method="min")
        n_opt = int(len(grouped))
        denom = float(max(n_opt - 1, 1))
        true_rank_pct = (grouped["rank_true"] - 1.0) / denom
        pred_rank_pct = (grouped["rank_pred"] - 1.0) / denom
        per_bench.append(
            {
                "rank_spearman": _safe_spearman(grouped["pred_mean"], grouped["true_mean"]),
                "rank_kendall_tau": _kendall_tau_b(grouped["pred_mean"], grouped["true_mean"]),
                "rank_pairwise_cindex": _pairwise_cindex(grouped["true_mean"], grouped["pred_mean"]),
                "rank_pct_err": float((pred_rank_pct - true_rank_pct).abs().mean()),
            }
        )
    if not per_bench:
        return out
    df = pd.DataFrame(per_bench)
    out["jointood_rank_spearman"] = _as_float(pd.to_numeric(df["rank_spearman"], errors="coerce").mean())
    out["jointood_rank_kendall_tau"] = _as_float(pd.to_numeric(df["rank_kendall_tau"], errors="coerce").mean())
    out["jointood_rank_pairwise_cindex"] = _as_float(
        pd.to_numeric(df["rank_pairwise_cindex"], errors="coerce").mean()
    )
    out["jointood_rank_pct_err"] = _as_float(pd.to_numeric(df["rank_pct_err"], errors="coerce").mean())
    return out


def _compute_residual_diag(pred: pd.Series, tgt: pd.Series) -> ResidualDiag:
    pred = pd.to_numeric(pred, errors="coerce")
    tgt = pd.to_numeric(tgt, errors="coerce")
    mask = pred.notna() & tgt.notna()
    pred = pred[mask]
    tgt = tgt[mask]
    if pred.empty:
        return ResidualDiag()

    err = pred - tgt
    ae = err.abs()
    slope = math.nan
    intercept = math.nan
    if pred.nunique() >= 2:
        try:
            slope, intercept = np.polyfit(pred.to_numpy(), tgt.to_numpy(), 1)
        except Exception:
            slope, intercept = math.nan, math.nan

    return ResidualDiag(
        n=float(len(pred)),
        mae=float(ae.mean()),
        rmse=float(np.sqrt((err**2).mean())),
        medae=float(ae.median()),
        p90ae=float(ae.quantile(0.90)),
        bias=float(err.mean()),
        cal_slope=float(slope) if np.isfinite(slope) else math.nan,
        cal_intercept=float(intercept) if np.isfinite(intercept) else math.nan,
        spearman=_safe_spearman(pred, tgt),
        pearson=_safe_pearson(pred, tgt),
    )


def _load_jointood_rows(method_path: Path) -> pd.DataFrame:
    rows_path = method_path / "prediction_jointood_rows.csv"
    if not rows_path.exists():
        return pd.DataFrame(columns=["prediction", "target"])
    try:
        df = pd.read_csv(rows_path)
    except Exception:
        return pd.DataFrame(columns=["prediction", "target"])
    if "prediction" not in df.columns or "target" not in df.columns:
        return pd.DataFrame(columns=["prediction", "target"])
    keep = ["prediction", "target"]
    for c in ("joint_holdout", "train_dataset", "benchmark", "fold", "run_id"):
        if c in df.columns:
            keep.append(c)
    out = df[keep].copy()
    out["prediction"] = pd.to_numeric(out["prediction"], errors="coerce")
    out["target"] = pd.to_numeric(out["target"], errors="coerce")
    out = out.dropna(subset=["prediction", "target"]).copy()
    return out


def _load_jointood_rank_detail(method_path: Path) -> pd.DataFrame:
    detail_path = method_path / "prediction_jointood_rank_detail.csv"
    if not detail_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(detail_path)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    if "true_rank_pct" not in out.columns and {"true_rank", "n_options"}.issubset(out.columns):
        true_rank = pd.to_numeric(out["true_rank"], errors="coerce")
        n_opts = pd.to_numeric(out["n_options"], errors="coerce")
        denom = (n_opts - 1.0).replace(0.0, np.nan)
        out["true_rank_pct"] = (true_rank - 1.0) / denom
    if "pred_rank_pct" not in out.columns and {"pred_rank", "n_options"}.issubset(out.columns):
        pred_rank = pd.to_numeric(out["pred_rank"], errors="coerce")
        n_opts = pd.to_numeric(out["n_options"], errors="coerce")
        denom = (n_opts - 1.0).replace(0.0, np.nan)
        out["pred_rank_pct"] = (pred_rank - 1.0) / denom
    if "true_rank_pct" not in out.columns or "pred_rank_pct" not in out.columns:
        return pd.DataFrame()

    out["true_rank_pct"] = pd.to_numeric(out["true_rank_pct"], errors="coerce")
    out["pred_rank_pct"] = pd.to_numeric(out["pred_rank_pct"], errors="coerce")
    out = out.dropna(subset=["true_rank_pct", "pred_rank_pct"]).copy()
    if out.empty:
        return pd.DataFrame()

    out["abs_rank_pct_error"] = (out["pred_rank_pct"] - out["true_rank_pct"]).abs()
    if "abs_rank_error" in out.columns:
        out["abs_rank_error"] = pd.to_numeric(out["abs_rank_error"], errors="coerce")
    return out


def _load_jointood_rank_summary(method_path: Path) -> pd.DataFrame:
    summary_path = method_path / "prediction_jointood_rank_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(summary_path)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    for c in (
        "top1",
        "top3",
        "topk",
        "topk_k",
        "topk_frac",
        "regret",
        "spearman",
        "kendall_tau",
        "pairwise_cindex",
        "mean_abs_rank_pct_error",
        "pred_best_true_rank_pct",
    ):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _load_residual_diag(method_path: Path) -> ResidualDiag:
    df = _load_jointood_rows(method_path)
    if df.empty:
        return ResidualDiag()
    return _compute_residual_diag(df["prediction"], df["target"])


def _collect_pool(run_roots: Sequence[Path], objective: str = "absolute") -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    rank_fallback_cache: Dict[str, Dict[str, float]] = {}
    for root in run_roots:
        summary_path = root / "method_summary.csv"
        if not summary_path.exists():
            print(f"Warning: missing summary: {summary_path}")
            continue
        try:
            df = pd.read_csv(summary_path)
        except Exception:
            print(f"Warning: failed to read summary: {summary_path}")
            continue

        model = _model_for_root(root)
        if "model" in df.columns:
            model_mask = df["model"].astype(str) == model
            if model_mask.any():
                df = df[model_mask].copy()
            else:
                print(
                    f"Warning: no rows matched inferred model='{model}' for {summary_path}; "
                    "using all models in this summary."
                )
        df = df[pd.to_numeric(df.get("jointood_mae"), errors="coerce").notna()].copy()
        if df.empty:
            continue

        # Avoid duplicate rows per signature within a run by taking the best row
        # under the active selection objective.
        working: List[Dict[str, object]] = []
        for _, r in df.iterrows():
            method = str(r.get("method", ""))
            path = Path(str(r.get("path", "")))
            method_path = path
            if not method_path.exists() and not method_path.is_absolute():
                alt = root / method_path
                if alt.exists():
                    method_path = alt
            tokens = _signal_tokens(r.get("predictors"))
            if not tokens:
                continue
            signature = "|".join(tokens)
            lane = _lane_from_tokens(tokens)
            motion_count, appearance_count = _token_lane_counts(tokens)
            row = {
                "run_root": str(root),
                "variant": root.name,
                "model": model,
                "method": method,
                "path": str(method_path),
                "family": str(r.get("family", "")),
                "symmetry": str(r.get("symmetry", "")),
                "signal_tokens": ",".join(tokens),
                "signature": signature,
                "signal_k": int(len(tokens)),
                "motion_k": int(motion_count),
                "appearance_k": int(appearance_count),
                "lane": lane,
                "jointood_mae": _as_float(r.get("jointood_mae")),
                "jointood_spearman": _as_float(r.get("jointood_spearman", np.nan)),
                "jointood_rank_spearman": _as_float(r.get("jointood_rank_spearman", np.nan)),
                "jointood_rank_kendall_tau": _as_float(r.get("jointood_rank_kendall_tau", np.nan)),
                "jointood_rank_pairwise_cindex": _as_float(
                    r.get("jointood_rank_pairwise_cindex", np.nan)
                ),
                "jointood_rank_pct_err": _as_float(r.get("jointood_rank_pct_err", np.nan)),
                "loto_mae": _as_float(r.get("loto_mae", np.nan)),
                "lobo_mae": _as_float(r.get("lobo_mae", np.nan)),
            }

            rank_keys = (
                "jointood_rank_spearman",
                "jointood_rank_kendall_tau",
                "jointood_rank_pairwise_cindex",
                "jointood_rank_pct_err",
            )
            if any(not np.isfinite(_as_float(row[k])) for k in rank_keys):
                cache_key = str(method_path)
                if cache_key not in rank_fallback_cache:
                    rank_rows = _load_jointood_rows(method_path)
                    rank_fallback_cache[cache_key] = _rank_metrics_from_jointood_rows(rank_rows)
                fallback = rank_fallback_cache[cache_key]
                for k in rank_keys:
                    if not np.isfinite(_as_float(row[k])):
                        row[k] = _as_float(fallback.get(k, np.nan))

            working.append(row)
        if not working:
            continue
        wf = pd.DataFrame(working)
        wf = _sort_pool_rows_for_objective(wf, objective=objective)
        best = wf.drop_duplicates(["variant", "signature"], keep="first").copy()
        rows.append(best)

    if not rows:
        return pd.DataFrame()
    pool = pd.concat(rows, ignore_index=True)
    return pool


def _aggregate_signatures(pool: pd.DataFrame) -> pd.DataFrame:
    agg = (
        pool.groupby(["signature", "signal_tokens", "signal_k", "motion_k", "appearance_k", "lane"], dropna=False)
        .agg(
            n_runs=("variant", "nunique"),
            variants=("variant", lambda s: ",".join(sorted(set(map(str, s))))),
            methods=("method", lambda s: ",".join(sorted(set(map(str, s))))),
            mean_jointood_mae=("jointood_mae", "mean"),
            std_jointood_mae=("jointood_mae", "std"),
            mean_jointood_spearman=("jointood_spearman", "mean"),
            std_jointood_spearman=("jointood_spearman", "std"),
            mean_jointood_rank_spearman=("jointood_rank_spearman", "mean"),
            std_jointood_rank_spearman=("jointood_rank_spearman", "std"),
            mean_jointood_rank_kendall_tau=("jointood_rank_kendall_tau", "mean"),
            std_jointood_rank_kendall_tau=("jointood_rank_kendall_tau", "std"),
            mean_jointood_rank_pairwise_cindex=("jointood_rank_pairwise_cindex", "mean"),
            std_jointood_rank_pairwise_cindex=("jointood_rank_pairwise_cindex", "std"),
            mean_jointood_rank_pct_err=("jointood_rank_pct_err", "mean"),
            std_jointood_rank_pct_err=("jointood_rank_pct_err", "std"),
            mean_loto_mae=("loto_mae", "mean"),
            mean_lobo_mae=("lobo_mae", "mean"),
        )
        .reset_index()
    )
    agg["std_jointood_mae"] = agg["std_jointood_mae"].fillna(0.0)
    agg["std_jointood_spearman"] = agg["std_jointood_spearman"].fillna(0.0)
    agg["std_jointood_rank_spearman"] = agg["std_jointood_rank_spearman"].fillna(0.0)
    agg["std_jointood_rank_kendall_tau"] = agg["std_jointood_rank_kendall_tau"].fillna(0.0)
    agg["std_jointood_rank_pairwise_cindex"] = agg["std_jointood_rank_pairwise_cindex"].fillna(0.0)
    agg["std_jointood_rank_pct_err"] = agg["std_jointood_rank_pct_err"].fillna(0.0)
    return agg


def _sort_for_selection(cand: pd.DataFrame, objective: str) -> pd.DataFrame:
    if objective == "ranking":
        order = [
            ("mean_jointood_rank_pairwise_cindex", False),
            ("mean_jointood_rank_spearman", False),
            ("mean_jointood_rank_kendall_tau", False),
            ("mean_jointood_rank_pct_err", True),
            ("mean_jointood_mae", True),
        ]
    else:
        order = [
            ("mean_jointood_mae", True),
            ("mean_jointood_spearman", False),
            ("mean_jointood_rank_pairwise_cindex", False),
        ]
    cols = [c for c, _ in order if c in cand.columns]
    if not cols:
        return cand
    asc = [a for c, a in order if c in cand.columns]
    return cand.sort_values(cols, ascending=asc)


def _sort_pool_rows_for_objective(cand: pd.DataFrame, objective: str) -> pd.DataFrame:
    if objective == "ranking":
        order = [
            ("jointood_rank_pairwise_cindex", False),
            ("jointood_rank_spearman", False),
            ("jointood_rank_kendall_tau", False),
            ("jointood_rank_pct_err", True),
            ("jointood_mae", True),
        ]
    else:
        order = [
            ("jointood_mae", True),
            ("jointood_spearman", False),
            ("jointood_rank_pairwise_cindex", False),
        ]
    cols = [c for c, _ in order if c in cand.columns]
    if not cols:
        return cand
    asc = [a for c, a in order if c in cand.columns]
    return cand.sort_values(cols, ascending=asc, na_position="last")


def _select_ladder(
    agg: pd.DataFrame,
    ks: Sequence[int],
    lanes: Sequence[str],
    mode: str,
    min_runs: int,
    objective: str = "absolute",
) -> pd.DataFrame:
    out: List[Dict[str, object]] = []
    for lane in lanes:
        lane_df = agg[(agg["lane"] == lane) & (agg["n_runs"] >= min_runs)].copy()
        for k in ks:
            if mode == "exact":
                cand = lane_df[lane_df["signal_k"] == k].copy()
            else:
                cand = lane_df[lane_df["signal_k"] <= k].copy()
            if cand.empty:
                continue
            cand = _sort_for_selection(cand, objective=objective)
            best = cand.iloc[0]
            row = best.to_dict()
            row["k_target"] = int(k)
            row["selection_mode"] = mode
            out.append(row)
    return pd.DataFrame(out)


def _select_unbounded(
    agg: pd.DataFrame,
    lanes: Sequence[str],
    min_runs: int,
    top_n: int,
    objective: str = "absolute",
) -> pd.DataFrame:
    out: List[Dict[str, object]] = []
    n_keep = max(int(top_n), 1)
    for lane in lanes:
        lane_df = agg[(agg["lane"] == lane) & (agg["n_runs"] >= min_runs)].copy()
        if lane_df.empty:
            continue
        lane_df = _sort_for_selection(lane_df, objective=objective).head(n_keep)
        for rank_idx, (_, row) in enumerate(lane_df.iterrows(), start=1):
            rec = row.to_dict()
            k_val = _as_float(row.get("signal_k"))
            rec["k_target"] = int(k_val) if np.isfinite(k_val) else -1
            rec["selection_mode"] = "unbounded"
            rec["lane_rank"] = int(rank_idx)
            out.append(rec)
    return pd.DataFrame(out)


def _attach_residual_diagnostics(
    selected: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    if selected.empty:
        return selected
    diag_rows: List[Dict[str, object]] = []
    cache: Dict[str, ResidualDiag] = {}
    for _, srow in selected.iterrows():
        sig = str(srow["signature"])
        subset = pool[pool["signature"] == sig].copy()
        if subset.empty:
            continue
        vals: Dict[str, List[float]] = {
            "diag_n": [],
            "diag_mae": [],
            "diag_rmse": [],
            "diag_medae": [],
            "diag_p90ae": [],
            "diag_bias": [],
            "diag_cal_slope": [],
            "diag_cal_intercept": [],
            "diag_spearman": [],
            "diag_pearson": [],
        }
        for _, prow in subset.iterrows():
            p = str(prow["path"])
            if p not in cache:
                cache[p] = _load_residual_diag(Path(p))
            d = cache[p]
            vals["diag_n"].append(d.n)
            vals["diag_mae"].append(d.mae)
            vals["diag_rmse"].append(d.rmse)
            vals["diag_medae"].append(d.medae)
            vals["diag_p90ae"].append(d.p90ae)
            vals["diag_bias"].append(d.bias)
            vals["diag_cal_slope"].append(d.cal_slope)
            vals["diag_cal_intercept"].append(d.cal_intercept)
            vals["diag_spearman"].append(d.spearman)
            vals["diag_pearson"].append(d.pearson)

        merged = dict(srow)
        for k, arr in vals.items():
            arr_f = np.array([x for x in arr if np.isfinite(x)], dtype=float)
            merged[f"{k}_mean"] = float(np.mean(arr_f)) if arr_f.size else math.nan
            merged[f"{k}_std"] = float(np.std(arr_f)) if arr_f.size else math.nan
        diag_rows.append(merged)
    return pd.DataFrame(diag_rows)


def _collect_signature_rows(
    signature: str,
    pool: pd.DataFrame,
    cache: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    sub = pool[pool["signature"] == signature].copy()
    if sub.empty:
        return pd.DataFrame(columns=["prediction", "target", "variant", "method", "path"])
    frames: List[pd.DataFrame] = []
    for _, r in sub.iterrows():
        p = str(r["path"])
        if p not in cache:
            cache[p] = _load_jointood_rows(Path(p))
        rows = cache[p]
        if rows.empty:
            continue
        tmp = rows.copy()
        tmp["variant"] = str(r["variant"])
        tmp["method"] = str(r["method"])
        tmp["path"] = p
        frames.append(tmp)
    if not frames:
        return pd.DataFrame(columns=["prediction", "target", "variant", "method", "path"])
    return pd.concat(frames, ignore_index=True)


def _collect_signature_rank_detail(
    signature: str,
    pool: pd.DataFrame,
    cache: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    sub = pool[pool["signature"] == signature].copy()
    if sub.empty:
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    for _, r in sub.iterrows():
        p = str(r["path"])
        if p not in cache:
            cache[p] = _load_jointood_rank_detail(Path(p))
        detail = cache[p]
        if detail.empty:
            continue
        tmp = detail.copy()
        tmp["variant"] = str(r["variant"])
        tmp["method"] = str(r["method"])
        tmp["path"] = p
        frames.append(tmp)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _collect_signature_rank_summary(
    signature: str,
    pool: pd.DataFrame,
    cache: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    sub = pool[pool["signature"] == signature].copy()
    if sub.empty:
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    for _, r in sub.iterrows():
        p = str(r["path"])
        if p not in cache:
            cache[p] = _load_jointood_rank_summary(Path(p))
        summary = cache[p]
        if summary.empty:
            continue
        tmp = summary.copy()
        tmp["variant"] = str(r["variant"])
        tmp["method"] = str(r["method"])
        tmp["path"] = p
        frames.append(tmp)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _fit_linear_calibration(pred: pd.Series, tgt: pd.Series) -> CalibrationFit:
    pred = pd.to_numeric(pred, errors="coerce")
    tgt = pd.to_numeric(tgt, errors="coerce")
    mask = pred.notna() & tgt.notna()
    pred = pred[mask]
    tgt = tgt[mask]
    if pred.nunique() < 2 or pred.empty:
        return CalibrationFit()
    try:
        slope, intercept = np.polyfit(pred.to_numpy(), tgt.to_numpy(), 1)
    except Exception:
        return CalibrationFit()
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return CalibrationFit()
    return CalibrationFit(float(slope), float(intercept))


def _strict_oof_linear_calibration(
    rows: pd.DataFrame,
    variant_col: str = "variant",
    pred_col: str = "prediction",
    target_col: str = "target",
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Strict calibration: for each variant, fit on all other variants and apply to held-out variant.
    Returns calibrated predictions and per-fold fit stats.
    """
    if rows.empty or variant_col not in rows.columns:
        return pd.Series(np.nan, index=rows.index, dtype=float), pd.DataFrame()

    pred_cal = pd.Series(np.nan, index=rows.index, dtype=float)
    fit_rows: List[Dict[str, object]] = []
    variants = sorted(set(map(str, rows[variant_col].dropna().unique())))
    for v in variants:
        tr = rows[rows[variant_col].astype(str) != v]
        te = rows[rows[variant_col].astype(str) == v]
        fit = _fit_linear_calibration(tr[pred_col], tr[target_col])
        fit_rows.append(
            {
                "heldout_variant": v,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "slope": fit.slope,
                "intercept": fit.intercept,
            }
        )
        if np.isfinite(fit.slope) and np.isfinite(fit.intercept):
            pred_cal.loc[te.index] = fit.slope * te[pred_col] + fit.intercept
    return pred_cal, pd.DataFrame(fit_rows)


def _collapse_fit_points(rows: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    if rows.empty or pred_col not in rows.columns or "target" not in rows.columns:
        return pd.DataFrame(columns=[pred_col, "target", "n_rows"])
    if "joint_holdout" in rows.columns:
        grp = ["joint_holdout"]
    elif "train_dataset" in rows.columns and "benchmark" in rows.columns:
        grp = ["train_dataset", "benchmark"]
    elif "fold" in rows.columns:
        grp = ["fold"]
    else:
        grp = []
    if not grp:
        out = rows[[pred_col, "target"]].copy()
        out["n_rows"] = 1
        return out
    agg = (
        rows.groupby(grp, dropna=False)
        .agg(
            **{
                pred_col: (pred_col, "mean"),
                "target": ("target", "mean"),
                "n_rows": ("target", "size"),
            }
        )
        .reset_index()
    )
    return agg


def _synthetic_mask(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series([], dtype=bool)
    mask = pd.Series(False, index=rows.index)
    for c in ("train_dataset", "benchmark", "joint_holdout"):
        if c in rows.columns:
            vals = rows[c].astype(str).str.lower()
            mask = mask | vals.str.contains("synthetic", na=False)
    return mask


def _real_only_mask(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series([], dtype=bool)
    return ~_synthetic_mask(rows)


def _bootstrap_metric_ci(
    pred: pd.Series,
    tgt: pd.Series,
    metric: str,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 123,
) -> Tuple[float, float]:
    pred = pd.to_numeric(pred, errors="coerce")
    tgt = pd.to_numeric(tgt, errors="coerce")
    m = pred.notna() & tgt.notna()
    pred = pred[m].to_numpy(dtype=float)
    tgt = tgt[m].to_numpy(dtype=float)
    n = pred.shape[0]
    if n < 5 or n_boot <= 0:
        return math.nan, math.nan

    rng = np.random.default_rng(seed)
    vals: List[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        p = pred[idx]
        t = tgt[idx]
        if metric == "mae":
            vals.append(float(np.mean(np.abs(p - t))))
        elif metric == "rmse":
            vals.append(float(np.sqrt(np.mean((p - t) ** 2))))
        elif metric == "spearman":
            pv = pd.Series(p).corr(pd.Series(t), method="spearman")
            vals.append(float(pv) if np.isfinite(pv) else math.nan)
        elif metric == "pearson":
            pv = pd.Series(p).corr(pd.Series(t), method="pearson")
            vals.append(float(pv) if np.isfinite(pv) else math.nan)
        else:
            return math.nan, math.nan
    arr = np.array([x for x in vals if np.isfinite(x)], dtype=float)
    if arr.size == 0:
        return math.nan, math.nan
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def _shuffle_within_variant_target(rows: pd.DataFrame, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    out = pd.Series(index=rows.index, dtype=float)
    if "variant" not in rows.columns:
        vals = rows["target"].to_numpy()
        idx = np.arange(vals.shape[0])
        rng.shuffle(idx)
        out.loc[rows.index] = vals[idx]
        return out
    for v, g in rows.groupby("variant", dropna=False):
        vals = g["target"].to_numpy()
        idx = np.arange(vals.shape[0])
        rng.shuffle(idx)
        out.loc[g.index] = vals[idx]
    return out


def _attach_bootstrap_uncertainty(
    selected_cal: pd.DataFrame,
    pool: pd.DataFrame,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> pd.DataFrame:
    if selected_cal.empty:
        return selected_cal
    out_rows: List[Dict[str, object]] = []
    row_cache: Dict[str, pd.DataFrame] = {}
    for i, srow in selected_cal.iterrows():
        merged = dict(srow)
        sig = str(srow["signature"])
        rows = _collect_signature_rows(sig, pool, row_cache)
        if rows.empty:
            merged["raw_mae_ci95_lo"] = math.nan
            merged["raw_mae_ci95_hi"] = math.nan
            merged["strict_oof_mae_ci95_lo"] = math.nan
            merged["strict_oof_mae_ci95_hi"] = math.nan
            merged["raw_rmse_ci95_lo"] = math.nan
            merged["raw_rmse_ci95_hi"] = math.nan
            merged["strict_oof_rmse_ci95_lo"] = math.nan
            merged["strict_oof_rmse_ci95_hi"] = math.nan
            out_rows.append(merged)
            continue

        pred = pd.to_numeric(rows["prediction"], errors="coerce")
        tgt = pd.to_numeric(rows["target"], errors="coerce")
        m = pred.notna() & tgt.notna()
        rows = rows.loc[m].copy()
        rows["prediction"] = pred[m]
        rows["target"] = tgt[m]
        strict_pred, _ = _strict_oof_linear_calibration(rows)
        sm = strict_pred.notna() & rows["target"].notna()
        strict_pred = strict_pred[sm]
        strict_tgt = rows.loc[sm, "target"]

        lo, hi = _bootstrap_metric_ci(rows["prediction"], rows["target"], metric="mae", n_boot=n_boot, seed=2026 + i)
        merged["raw_mae_ci95_lo"] = lo
        merged["raw_mae_ci95_hi"] = hi
        lo, hi = _bootstrap_metric_ci(strict_pred, strict_tgt, metric="mae", n_boot=n_boot, seed=3026 + i)
        merged["strict_oof_mae_ci95_lo"] = lo
        merged["strict_oof_mae_ci95_hi"] = hi
        lo, hi = _bootstrap_metric_ci(rows["prediction"], rows["target"], metric="rmse", n_boot=n_boot, seed=4026 + i)
        merged["raw_rmse_ci95_lo"] = lo
        merged["raw_rmse_ci95_hi"] = hi
        lo, hi = _bootstrap_metric_ci(strict_pred, strict_tgt, metric="rmse", n_boot=n_boot, seed=5026 + i)
        merged["strict_oof_rmse_ci95_lo"] = lo
        merged["strict_oof_rmse_ci95_hi"] = hi
        out_rows.append(merged)
    return pd.DataFrame(out_rows)


def _build_subgroup_robustness(
    selected_exact_cal: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    if selected_exact_cal.empty:
        return pd.DataFrame()
    row_cache: Dict[str, pd.DataFrame] = {}
    out_rows: List[Dict[str, object]] = []
    for i, srow in selected_exact_cal.iterrows():
        sig = str(srow["signature"])
        rows = _collect_signature_rows(sig, pool, row_cache)
        if rows.empty:
            continue
        pred = pd.to_numeric(rows["prediction"], errors="coerce")
        tgt = pd.to_numeric(rows["target"], errors="coerce")
        m = pred.notna() & tgt.notna()
        rows = rows.loc[m].copy()
        rows["prediction"] = pred[m]
        rows["target"] = tgt[m]
        if rows.empty:
            continue
        synth_m = _synthetic_mask(rows)
        groups = {
            "all": rows.index,
            "synthetic_only": rows.index[synth_m],
            "real_only": rows.index[~synth_m],
        }
        for gname, gidx in groups.items():
            if len(gidx) < 5:
                continue
            sub = rows.loc[gidx].copy()
            strict_pred, _ = _strict_oof_linear_calibration(sub)
            sm = strict_pred.notna() & sub["target"].notna()
            sp = strict_pred[sm]
            st = sub.loc[sm, "target"]
            raw = _compute_residual_diag(sub["prediction"], sub["target"])
            strict = _compute_residual_diag(sp, st)
            lo_raw, hi_raw = _bootstrap_metric_ci(sub["prediction"], sub["target"], "mae", seed=7000 + i)
            lo_str, hi_str = _bootstrap_metric_ci(sp, st, "mae", seed=8000 + i)
            out_rows.append(
                {
                    "lane": srow["lane"],
                    "k_target": int(srow["k_target"]),
                    "signal_k": int(srow["signal_k"]),
                    "signature": sig,
                    "subgroup": gname,
                    "n_rows": int(len(sub)),
                    "raw_mae": raw.mae,
                    "raw_rmse": raw.rmse,
                    "strict_oof_mae": strict.mae,
                    "strict_oof_rmse": strict.rmse,
                    "strict_oof_delta_mae": (raw.mae - strict.mae) if np.isfinite(raw.mae) and np.isfinite(strict.mae) else math.nan,
                    "strict_oof_delta_rmse": (raw.rmse - strict.rmse) if np.isfinite(raw.rmse) and np.isfinite(strict.rmse) else math.nan,
                    "raw_mae_ci95_lo": lo_raw,
                    "raw_mae_ci95_hi": hi_raw,
                    "strict_oof_mae_ci95_lo": lo_str,
                    "strict_oof_mae_ci95_hi": hi_str,
                }
            )
    return pd.DataFrame(out_rows)


def _run_sanity_checks(
    selected_exact_cal: pd.DataFrame,
    pool: pd.DataFrame,
    n_perm: int = DEFAULT_PERMUTATION_SAMPLES,
) -> pd.DataFrame:
    if selected_exact_cal.empty:
        return pd.DataFrame()
    row_cache: Dict[str, pd.DataFrame] = {}
    out: List[Dict[str, object]] = []
    for i, srow in selected_exact_cal.iterrows():
        sig = str(srow["signature"])
        rows = _collect_signature_rows(sig, pool, row_cache)
        if rows.empty:
            continue
        pred = pd.to_numeric(rows["prediction"], errors="coerce")
        tgt = pd.to_numeric(rows["target"], errors="coerce")
        m = pred.notna() & tgt.notna()
        rows = rows.loc[m].copy()
        rows["prediction"] = pred[m]
        rows["target"] = tgt[m]
        if len(rows) < 20:
            continue
        strict_pred, _ = _strict_oof_linear_calibration(rows)
        sm = strict_pred.notna() & rows["target"].notna()
        sp = strict_pred[sm]
        st = rows.loc[sm, "target"]
        obs_mae = float(np.mean(np.abs(sp.to_numpy() - st.to_numpy())))
        obs_spear = float(pd.Series(sp).corr(pd.Series(st), method="spearman"))
        perm_mae: List[float] = []
        perm_spear: List[float] = []
        for j in range(int(n_perm)):
            shuf = rows.copy()
            shuf["target"] = _shuffle_within_variant_target(shuf, seed=100000 + 1000 * i + j)
            strict_p, _ = _strict_oof_linear_calibration(shuf)
            mm = strict_p.notna() & shuf["target"].notna()
            if mm.sum() < 10:
                continue
            p = strict_p[mm].to_numpy()
            t = shuf.loc[mm, "target"].to_numpy()
            perm_mae.append(float(np.mean(np.abs(p - t))))
            sv = pd.Series(p).corr(pd.Series(t), method="spearman")
            perm_spear.append(float(sv) if np.isfinite(sv) else math.nan)
        arr_mae = np.array([x for x in perm_mae if np.isfinite(x)], dtype=float)
        arr_sp = np.array([x for x in perm_spear if np.isfinite(x)], dtype=float)
        p_mae = float((1 + np.sum(arr_mae <= obs_mae)) / (1 + arr_mae.size)) if arr_mae.size else math.nan
        p_sp = float((1 + np.sum(arr_sp >= obs_spear)) / (1 + arr_sp.size)) if arr_sp.size else math.nan
        out.append(
            {
                "lane": srow["lane"],
                "k_target": int(srow["k_target"]),
                "signal_k": int(srow["signal_k"]),
                "signature": sig,
                "n_rows": int(len(rows)),
                "obs_strict_oof_mae": obs_mae,
                "perm_mae_mean": float(np.mean(arr_mae)) if arr_mae.size else math.nan,
                "perm_mae_std": float(np.std(arr_mae)) if arr_mae.size else math.nan,
                "p_value_mae_lower_than_perm": p_mae,
                "obs_strict_oof_spearman": obs_spear,
                "perm_spearman_mean": float(np.mean(arr_sp)) if arr_sp.size else math.nan,
                "perm_spearman_std": float(np.std(arr_sp)) if arr_sp.size else math.nan,
                "p_value_spearman_higher_than_perm": p_sp,
                "n_permutations_used": int(min(arr_mae.size, arr_sp.size)),
            }
        )
    return pd.DataFrame(out)


def _attach_calibrated_diagnostics(
    selected_diag: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    if selected_diag.empty:
        return selected_diag
    out_rows: List[Dict[str, object]] = []
    row_cache: Dict[str, pd.DataFrame] = {}
    for _, srow in selected_diag.iterrows():
        sig = str(srow["signature"])
        rows = _collect_signature_rows(sig, pool, row_cache)
        merged = dict(srow)
        if rows.empty:
            merged["pooled_n"] = math.nan
            merged["pooled_raw_mae"] = math.nan
            merged["pooled_raw_rmse"] = math.nan
            merged["pooled_cal_mae"] = math.nan
            merged["pooled_cal_rmse"] = math.nan
            merged["pooled_cal_delta_mae"] = math.nan
            merged["pooled_cal_delta_rmse"] = math.nan
            merged["cal_linear_slope"] = math.nan
            merged["cal_linear_intercept"] = math.nan
            out_rows.append(merged)
            continue

        pred = rows["prediction"]
        tgt = rows["target"]
        raw_diag = _compute_residual_diag(pred, tgt)
        strict_pred, strict_fits = _strict_oof_linear_calibration(rows)
        strict_mask = strict_pred.notna() & tgt.notna()
        strict_diag = _compute_residual_diag(strict_pred[strict_mask], tgt[strict_mask])
        fit = _fit_linear_calibration(pred, tgt)

        if np.isfinite(fit.slope) and np.isfinite(fit.intercept):
            pred_cal = fit.slope * pred + fit.intercept
            cal_diag = _compute_residual_diag(pred_cal, tgt)
        else:
            cal_diag = ResidualDiag()

        merged["pooled_n"] = raw_diag.n
        merged["pooled_raw_mae"] = raw_diag.mae
        merged["pooled_raw_rmse"] = raw_diag.rmse
        merged["pooled_raw_medae"] = raw_diag.medae
        merged["pooled_raw_p90ae"] = raw_diag.p90ae
        merged["pooled_raw_bias"] = raw_diag.bias
        merged["pooled_raw_spearman"] = raw_diag.spearman
        merged["pooled_raw_pearson"] = raw_diag.pearson

        merged["cal_linear_slope"] = fit.slope
        merged["cal_linear_intercept"] = fit.intercept
        merged["pooled_cal_mae"] = cal_diag.mae
        merged["pooled_cal_rmse"] = cal_diag.rmse
        merged["pooled_cal_medae"] = cal_diag.medae
        merged["pooled_cal_p90ae"] = cal_diag.p90ae
        merged["pooled_cal_bias"] = cal_diag.bias
        merged["pooled_cal_spearman"] = cal_diag.spearman
        merged["pooled_cal_pearson"] = cal_diag.pearson
        merged["pooled_cal_delta_mae"] = (
            raw_diag.mae - cal_diag.mae
            if np.isfinite(raw_diag.mae) and np.isfinite(cal_diag.mae)
            else math.nan
        )
        merged["pooled_cal_delta_rmse"] = (
            raw_diag.rmse - cal_diag.rmse
            if np.isfinite(raw_diag.rmse) and np.isfinite(cal_diag.rmse)
            else math.nan
        )
        merged["strict_oof_cal_n"] = float(strict_mask.sum())
        merged["strict_oof_cal_mae"] = strict_diag.mae
        merged["strict_oof_cal_rmse"] = strict_diag.rmse
        merged["strict_oof_cal_medae"] = strict_diag.medae
        merged["strict_oof_cal_p90ae"] = strict_diag.p90ae
        merged["strict_oof_cal_bias"] = strict_diag.bias
        merged["strict_oof_cal_spearman"] = strict_diag.spearman
        merged["strict_oof_cal_pearson"] = strict_diag.pearson
        merged["strict_oof_delta_mae"] = (
            raw_diag.mae - strict_diag.mae
            if np.isfinite(raw_diag.mae) and np.isfinite(strict_diag.mae)
            else math.nan
        )
        merged["strict_oof_delta_rmse"] = (
            raw_diag.rmse - strict_diag.rmse
            if np.isfinite(raw_diag.rmse) and np.isfinite(strict_diag.rmse)
            else math.nan
        )
        if strict_fits.empty:
            merged["strict_oof_fit_slope_mean"] = math.nan
            merged["strict_oof_fit_slope_std"] = math.nan
            merged["strict_oof_fit_intercept_mean"] = math.nan
            merged["strict_oof_fit_intercept_std"] = math.nan
        else:
            merged["strict_oof_fit_slope_mean"] = float(pd.to_numeric(strict_fits["slope"], errors="coerce").mean())
            merged["strict_oof_fit_slope_std"] = float(pd.to_numeric(strict_fits["slope"], errors="coerce").std(ddof=0))
            merged["strict_oof_fit_intercept_mean"] = float(pd.to_numeric(strict_fits["intercept"], errors="coerce").mean())
            merged["strict_oof_fit_intercept_std"] = float(pd.to_numeric(strict_fits["intercept"], errors="coerce").std(ddof=0))
        out_rows.append(merged)
    return pd.DataFrame(out_rows)


def _attach_base_variant_calibrated_diagnostics(
    selected_diag: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    """Calibrated diagnostics restricted to effect/control-rich base variants."""
    if selected_diag.empty:
        return selected_diag
    out_rows: List[Dict[str, object]] = []
    row_cache: Dict[str, pd.DataFrame] = {}
    for _, srow in selected_diag.iterrows():
        sig = str(srow["signature"])
        sub_pool = pool[
            (pool["signature"] == sig)
            & (pool["variant"].astype(str).str.contains("_base_"))
        ].copy()
        merged = dict(srow)
        if sub_pool.empty:
            merged["base_pooled_n"] = math.nan
            merged["base_pooled_raw_mae"] = math.nan
            merged["base_pooled_raw_rmse"] = math.nan
            merged["base_pooled_cal_mae"] = math.nan
            merged["base_pooled_cal_rmse"] = math.nan
            merged["base_pooled_cal_delta_mae"] = math.nan
            merged["base_pooled_cal_delta_rmse"] = math.nan
            merged["base_cal_linear_slope"] = math.nan
            merged["base_cal_linear_intercept"] = math.nan
            out_rows.append(merged)
            continue

        frames: List[pd.DataFrame] = []
        for _, r in sub_pool.iterrows():
            p = str(r["path"])
            if p not in row_cache:
                row_cache[p] = _load_jointood_rows(Path(p))
            rows = row_cache[p]
            if rows.empty:
                continue
            tmp = rows.copy()
            tmp["variant"] = str(r["variant"])
            frames.append(tmp)
        if not frames:
            merged["base_pooled_n"] = math.nan
            merged["base_pooled_raw_mae"] = math.nan
            merged["base_pooled_raw_rmse"] = math.nan
            merged["base_pooled_cal_mae"] = math.nan
            merged["base_pooled_cal_rmse"] = math.nan
            merged["base_pooled_cal_delta_mae"] = math.nan
            merged["base_pooled_cal_delta_rmse"] = math.nan
            merged["base_cal_linear_slope"] = math.nan
            merged["base_cal_linear_intercept"] = math.nan
            out_rows.append(merged)
            continue

        rows_all = pd.concat(frames, ignore_index=True)
        pred = rows_all["prediction"]
        tgt = rows_all["target"]
        raw_diag = _compute_residual_diag(pred, tgt)
        strict_pred, strict_fits = _strict_oof_linear_calibration(rows_all)
        strict_mask = strict_pred.notna() & tgt.notna()
        strict_diag = _compute_residual_diag(strict_pred[strict_mask], tgt[strict_mask])
        fit = _fit_linear_calibration(pred, tgt)
        if np.isfinite(fit.slope) and np.isfinite(fit.intercept):
            pred_cal = fit.slope * pred + fit.intercept
            cal_diag = _compute_residual_diag(pred_cal, tgt)
        else:
            cal_diag = ResidualDiag()

        merged["base_pooled_n"] = raw_diag.n
        merged["base_pooled_raw_mae"] = raw_diag.mae
        merged["base_pooled_raw_rmse"] = raw_diag.rmse
        merged["base_cal_linear_slope"] = fit.slope
        merged["base_cal_linear_intercept"] = fit.intercept
        merged["base_pooled_cal_mae"] = cal_diag.mae
        merged["base_pooled_cal_rmse"] = cal_diag.rmse
        merged["base_pooled_cal_delta_mae"] = (
            raw_diag.mae - cal_diag.mae
            if np.isfinite(raw_diag.mae) and np.isfinite(cal_diag.mae)
            else math.nan
        )
        merged["base_pooled_cal_delta_rmse"] = (
            raw_diag.rmse - cal_diag.rmse
            if np.isfinite(raw_diag.rmse) and np.isfinite(cal_diag.rmse)
            else math.nan
        )
        merged["base_strict_oof_cal_n"] = float(strict_mask.sum())
        merged["base_strict_oof_cal_mae"] = strict_diag.mae
        merged["base_strict_oof_cal_rmse"] = strict_diag.rmse
        merged["base_strict_oof_delta_mae"] = (
            raw_diag.mae - strict_diag.mae
            if np.isfinite(raw_diag.mae) and np.isfinite(strict_diag.mae)
            else math.nan
        )
        merged["base_strict_oof_delta_rmse"] = (
            raw_diag.rmse - strict_diag.rmse
            if np.isfinite(raw_diag.rmse) and np.isfinite(strict_diag.rmse)
            else math.nan
        )
        if strict_fits.empty:
            merged["base_strict_oof_fit_slope_mean"] = math.nan
            merged["base_strict_oof_fit_intercept_mean"] = math.nan
        else:
            merged["base_strict_oof_fit_slope_mean"] = float(pd.to_numeric(strict_fits["slope"], errors="coerce").mean())
            merged["base_strict_oof_fit_intercept_mean"] = float(pd.to_numeric(strict_fits["intercept"], errors="coerce").mean())
        out_rows.append(merged)
    return pd.DataFrame(out_rows)


def _plot_finalist_fit_scatter(out_dir: Path, selected_exact_cal: pd.DataFrame, pool: pd.DataFrame) -> None:
    if selected_exact_cal.empty:
        return
    plot_dir = out_dir / "fit_scatter_exact_k"
    plot_dir.mkdir(parents=True, exist_ok=True)
    row_cache: Dict[str, pd.DataFrame] = {}

    for _, row in selected_exact_cal.sort_values(["lane", "k_target"]).iterrows():
        sig = str(row["signature"])
        lane = str(row["lane"])
        k = int(row["k_target"])
        rows = _collect_signature_rows(sig, pool, row_cache)
        if rows.empty:
            continue
        pred = pd.to_numeric(rows["prediction"], errors="coerce")
        tgt = pd.to_numeric(rows["target"], errors="coerce")
        mask = pred.notna() & tgt.notna()
        pred = pred[mask]
        tgt = tgt[mask]
        rows = rows.loc[mask].copy()
        if pred.empty:
            continue

        strict_pred, _ = _strict_oof_linear_calibration(rows)
        strict_mask = strict_pred.notna() & tgt.notna()
        pred_cal = strict_pred[strict_mask]
        tgt_cal = tgt[strict_mask]
        if pred_cal.empty:
            pred_cal = pred.copy()
            tgt_cal = tgt.copy()

        raw = _compute_residual_diag(pred, tgt)
        cal = _compute_residual_diag(pred_cal, tgt_cal)

        lo = float(min(pred.min(), tgt.min(), pred_cal.min()))
        hi = float(max(pred.max(), tgt.max(), pred_cal.max()))
        pad = 0.05 * (hi - lo + 1e-8)
        x0, x1 = lo - pad, hi + pad

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), squeeze=False)
        ax0, ax1 = axes[0, 0], axes[0, 1]

        ax0.scatter(pred, tgt, s=14, alpha=0.35, color="#1f77b4")
        ax0.plot([x0, x1], [x0, x1], "k--", linewidth=1.0)
        ax0.set_xlim(x0, x1)
        ax0.set_ylim(x0, x1)
        ax0.set_title(f"Raw: MAE={raw.mae:.2f}, RMSE={raw.rmse:.2f}")
        ax0.set_xlabel("Predicted")
        ax0.set_ylabel("Observed")
        ax0.grid(True, alpha=0.25)

        ax1.scatter(pred_cal, tgt_cal, s=14, alpha=0.35, color="#2ca02c")
        ax1.plot([x0, x1], [x0, x1], "k--", linewidth=1.0)
        ax1.set_xlim(x0, x1)
        ax1.set_ylim(x0, x1)
        ax1.set_title(f"Strict OOF Cal: MAE={cal.mae:.2f}, RMSE={cal.rmse:.2f}")
        ax1.set_xlabel("Strict OOF calibrated prediction")
        ax1.set_ylabel("Observed")
        ax1.grid(True, alpha=0.25)

        fig.suptitle(f"{lane} | k={k} | n={len(pred)}", fontsize=11)
        fig.tight_layout()
        out = plot_dir / f"{lane}_k{k}_actual_vs_pred.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)


def _plot_finalist_fit_scatter_cell_collapsed(
    out_dir: Path,
    selected_exact_cal: pd.DataFrame,
    pool: pd.DataFrame,
) -> None:
    if selected_exact_cal.empty:
        return
    plot_dir = out_dir / "fit_scatter_cell_collapsed_exact_k"
    plot_dir.mkdir(parents=True, exist_ok=True)
    row_cache: Dict[str, pd.DataFrame] = {}

    for _, row in selected_exact_cal.sort_values(["lane", "k_target"]).iterrows():
        sig = str(row["signature"])
        lane = str(row["lane"])
        k = int(row["k_target"])
        rows = _collect_signature_rows(sig, pool, row_cache)
        if rows.empty:
            continue
        pred = pd.to_numeric(rows["prediction"], errors="coerce")
        tgt = pd.to_numeric(rows["target"], errors="coerce")
        mask = pred.notna() & tgt.notna()
        rows = rows.loc[mask].copy()
        rows["prediction"] = pred[mask]
        rows["target"] = tgt[mask]
        if rows.empty:
            continue

        strict_pred, _ = _strict_oof_linear_calibration(rows)
        rows["prediction_strict_oof_cal"] = strict_pred

        raw_coll = _collapse_fit_points(rows, pred_col="prediction")
        cal_coll = _collapse_fit_points(rows.dropna(subset=["prediction_strict_oof_cal"]), pred_col="prediction_strict_oof_cal")
        if raw_coll.empty:
            continue
        if cal_coll.empty:
            cal_coll = raw_coll.rename(columns={"prediction": "prediction_strict_oof_cal"})

        raw = _compute_residual_diag(raw_coll["prediction"], raw_coll["target"])
        cal = _compute_residual_diag(cal_coll["prediction_strict_oof_cal"], cal_coll["target"])

        lo = float(min(raw_coll["prediction"].min(), raw_coll["target"].min(), cal_coll["prediction_strict_oof_cal"].min()))
        hi = float(max(raw_coll["prediction"].max(), raw_coll["target"].max(), cal_coll["prediction_strict_oof_cal"].max()))
        pad = 0.05 * (hi - lo + 1e-8)
        x0, x1 = lo - pad, hi + pad

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), squeeze=False)
        ax0, ax1 = axes[0, 0], axes[0, 1]

        sz0 = np.clip(raw_coll["n_rows"].to_numpy() * 8.0, 14.0, 120.0)
        sz1 = np.clip(cal_coll["n_rows"].to_numpy() * 8.0, 14.0, 120.0)
        ax0.scatter(raw_coll["prediction"], raw_coll["target"], s=sz0, alpha=0.35, color="#1f77b4")
        ax0.plot([x0, x1], [x0, x1], "k--", linewidth=1.0)
        ax0.set_xlim(x0, x1)
        ax0.set_ylim(x0, x1)
        ax0.set_title(f"Cell-collapsed Raw: MAE={raw.mae:.2f}, RMSE={raw.rmse:.2f}")
        ax0.set_xlabel("Predicted")
        ax0.set_ylabel("Observed")
        ax0.grid(True, alpha=0.25)

        ax1.scatter(cal_coll["prediction_strict_oof_cal"], cal_coll["target"], s=sz1, alpha=0.35, color="#2ca02c")
        ax1.plot([x0, x1], [x0, x1], "k--", linewidth=1.0)
        ax1.set_xlim(x0, x1)
        ax1.set_ylim(x0, x1)
        ax1.set_title(f"Cell-collapsed Strict OOF: MAE={cal.mae:.2f}, RMSE={cal.rmse:.2f}")
        ax1.set_xlabel("Strict OOF calibrated prediction")
        ax1.set_ylabel("Observed")
        ax1.grid(True, alpha=0.25)

        fig.suptitle(
            f"{lane} | k={k} | collapsed cells={len(raw_coll)} (from rows={len(rows)})",
            fontsize=11,
        )
        fig.tight_layout()
        out = plot_dir / f"{lane}_k{k}_cell_collapsed_actual_vs_pred.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)


def _plot_finalist_ranking_views(
    out_dir: Path,
    selected_exact_cal: pd.DataFrame,
    pool: pd.DataFrame,
) -> None:
    if selected_exact_cal.empty:
        return
    plot_dir = out_dir / "rank_views_exact_k"
    plot_dir.mkdir(parents=True, exist_ok=True)
    binned_plot_dir = out_dir / "rank_binned_exact_k"
    binned_plot_dir.mkdir(parents=True, exist_ok=True)
    binned_paper_dir = out_dir / "rank_binned_paper_exact_k"
    binned_paper_dir.mkdir(parents=True, exist_ok=True)
    detail_cache: Dict[str, pd.DataFrame] = {}
    summary_cache: Dict[str, pd.DataFrame] = {}

    for _, row in selected_exact_cal.sort_values(["lane", "k_target"]).iterrows():
        sig = str(row["signature"])
        lane = str(row["lane"])
        k = int(row["k_target"])
        detail = _collect_signature_rank_detail(sig, pool, detail_cache)
        summary = _collect_signature_rank_summary(sig, pool, summary_cache)
        if detail.empty and summary.empty:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), squeeze=False)
        ax0, ax1, ax2 = axes[0, 0], axes[0, 1], axes[0, 2]

        if not detail.empty:
            x = pd.to_numeric(detail["pred_rank_pct"], errors="coerce")
            y = pd.to_numeric(detail["true_rank_pct"], errors="coerce")
            m = x.notna() & y.notna()
            x = x[m].clip(0.0, 1.0)
            y = y[m].clip(0.0, 1.0)
            err = (x - y).abs()
            binned_stats = pd.DataFrame()
            if len(x) >= 10:
                bins = pd.cut(x, bins=np.linspace(0.0, 1.0, 11), include_lowest=True, duplicates="drop")
                bsrc = pd.DataFrame({"x": x, "y": y, "bin": bins})
                rows_stat: List[Dict[str, float]] = []
                for _, g in bsrc.groupby("bin", dropna=False, observed=False):
                    if g.empty:
                        continue
                    xv = pd.to_numeric(g["x"], errors="coerce").dropna()
                    yv = pd.to_numeric(g["y"], errors="coerce").dropna()
                    if xv.empty or yv.empty:
                        continue
                    yv_np = yv.to_numpy(dtype=float)
                    y_mode = _mode_stat(yv_np)
                    med_lo, med_hi = _bootstrap_ci(yv_np, stat="median", seed=17 + int(len(rows_stat)))
                    mean_lo, mean_hi = _bootstrap_ci(yv_np, stat="mean", seed=117 + int(len(rows_stat)))
                    mode_lo, mode_hi = _bootstrap_ci(yv_np, stat="mode", seed=217 + int(len(rows_stat)))
                    rows_stat.append(
                        {
                            "x_center": float(xv.median()),
                            "n": float(len(yv_np)),
                            "y_median": float(yv.median()),
                            "y_median_ci_lo": med_lo,
                            "y_median_ci_hi": med_hi,
                            "y_mean": float(yv.mean()),
                            "y_mean_ci_lo": mean_lo,
                            "y_mean_ci_hi": mean_hi,
                            "y_mode": y_mode,
                            "y_mode_ci_lo": mode_lo,
                            "y_mode_ci_hi": mode_hi,
                        }
                    )
                if rows_stat:
                    binned_stats = pd.DataFrame(rows_stat).sort_values("x_center")

            ax0.scatter(x, y, s=10, alpha=0.25, color="#1f77b4")
            ax0.plot([0.0, 1.0], [0.0, 1.0], "k--", linewidth=1.0, alpha=0.75)
            if not binned_stats.empty:
                ax0.plot(
                    binned_stats["x_center"],
                    binned_stats["y_median"],
                    color="#d62728",
                    linewidth=1.8,
                    label="Binned median",
                )
                ax0.plot(
                    binned_stats["x_center"],
                    binned_stats["y_mean"],
                    color="#2ca02c",
                    linewidth=1.5,
                    linestyle="--",
                    label="Binned mean",
                )
                ax0.plot(
                    binned_stats["x_center"],
                    binned_stats["y_mode"],
                    color="#9467bd",
                    linewidth=1.5,
                    linestyle=":",
                    label="Binned mode",
                )
                ax0.legend(loc="upper left", fontsize=8, framealpha=0.9)
            ax0.set_xlim(-0.02, 1.02)
            ax0.set_ylim(-0.02, 1.02)
            ax0.set_xlabel("Predicted rank percentile (lower better)")
            ax0.set_ylabel("Observed rank percentile (lower better)")
            tau = _kendall_tau_b(x, y)
            cidx = _pairwise_cindex(y, x)
            mae_rank = float(err.mean()) if len(err) else math.nan
            ax0.set_title(
                "Rank-vs-rank\n"
                f"tau={tau:.2f}, c-index={cidx:.2f}, mean |rank err|={mae_rank:.2f}"
            )
            ax0.grid(True, alpha=0.25)

            xs = np.sort(err.to_numpy())
            ys = np.arange(1, len(xs) + 1, dtype=float) / float(len(xs))
            ax1.plot(xs, ys, color="#ff7f0e", linewidth=2.0)
            ax1.axvline(0.10, color="k", linestyle="--", linewidth=0.9, alpha=0.7)
            ax1.axvline(0.20, color="k", linestyle=":", linewidth=0.9, alpha=0.7)
            p10 = float((err <= 0.10).mean())
            p20 = float((err <= 0.20).mean())
            ax1.text(
                0.52,
                0.08,
                f"within 10%: {p10:.2f}\nwithin 20%: {p20:.2f}",
                transform=ax1.transAxes,
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )
            ax1.set_xlim(0.0, 1.0)
            ax1.set_ylim(0.0, 1.0)
            ax1.set_xlabel("|rank percentile error|")
            ax1.set_ylabel("CDF")
            ax1.set_title("Ranking error distribution")
            ax1.grid(True, alpha=0.25)

            if not binned_stats.empty:
                fig_b, axes_b = plt.subplots(1, 3, figsize=(13.5, 4.0), squeeze=False)
                b_specs = [
                    ("y_median", "Binned median", "#d62728", "-"),
                    ("y_mean", "Binned mean", "#2ca02c", "--"),
                    ("y_mode", "Binned mode", "#9467bd", ":"),
                ]
                for i, (col, label, color, ls) in enumerate(b_specs):
                    axb = axes_b[0, i]
                    axb.scatter(x, y, s=10, alpha=0.18, color="#1f77b4")
                    axb.plot([0.0, 1.0], [0.0, 1.0], "k--", linewidth=1.0, alpha=0.7)
                    axb.plot(binned_stats["x_center"], binned_stats[col], color=color, linewidth=2.0, linestyle=ls)
                    axb.set_xlim(-0.02, 1.02)
                    axb.set_ylim(-0.02, 1.02)
                    axb.set_title(label)
                    axb.set_xlabel("Predicted rank percentile")
                    if i == 0:
                        axb.set_ylabel("Observed rank percentile")
                    axb.grid(True, alpha=0.25)
                fig_b.suptitle(f"{lane} | k={k} | binned rank trends", fontsize=11)
                fig_b.tight_layout()
                out_b = binned_plot_dir / f"{lane}_k{k}_binned_median_mean_mode.png"
                fig_b.savefig(out_b, dpi=180)
                plt.close(fig_b)

                fig_p, axes_p = plt.subplots(1, 3, figsize=(13.5, 4.0), squeeze=False)
                p_specs = [
                    ("y_median", "y_median_ci_lo", "y_median_ci_hi", "Binned median", "#d62728", "-"),
                    ("y_mean", "y_mean_ci_lo", "y_mean_ci_hi", "Binned mean", "#2ca02c", "--"),
                    ("y_mode", "y_mode_ci_lo", "y_mode_ci_hi", "Binned mode", "#9467bd", ":"),
                ]
                for i, (col, c_lo, c_hi, title, color, ls) in enumerate(p_specs):
                    axp = axes_p[0, i]
                    xc = pd.to_numeric(binned_stats["x_center"], errors="coerce").to_numpy(dtype=float)
                    yy = pd.to_numeric(binned_stats[col], errors="coerce").to_numpy(dtype=float)
                    lo = pd.to_numeric(binned_stats[c_lo], errors="coerce").to_numpy(dtype=float)
                    hi = pd.to_numeric(binned_stats[c_hi], errors="coerce").to_numpy(dtype=float)
                    n_bin = pd.to_numeric(binned_stats["n"], errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)

                    mask = np.isfinite(xc) & np.isfinite(yy)
                    if mask.any():
                        axp.plot(xc[mask], yy[mask], color=color, linewidth=2.0, linestyle=ls, alpha=0.9)
                        ci_mask = mask & np.isfinite(lo) & np.isfinite(hi)
                        if ci_mask.any():
                            yerr = np.vstack(
                                [
                                    np.maximum(yy[ci_mask] - lo[ci_mask], 0.0),
                                    np.maximum(hi[ci_mask] - yy[ci_mask], 0.0),
                                ]
                            )
                            axp.errorbar(
                                xc[ci_mask],
                                yy[ci_mask],
                                yerr=yerr,
                                fmt="o",
                                color=color,
                                ecolor=color,
                                elinewidth=1.2,
                                capsize=3,
                                markersize=5.5,
                                alpha=0.95,
                            )
                        no_ci_mask = mask & ~(np.isfinite(lo) & np.isfinite(hi))
                        if no_ci_mask.any():
                            axp.plot(
                                xc[no_ci_mask],
                                yy[no_ci_mask],
                                marker="o",
                                linestyle="None",
                                color=color,
                                markersize=5.5,
                                alpha=0.95,
                            )

                    axp.plot([0.0, 1.0], [0.0, 1.0], "k--", linewidth=1.0, alpha=0.7)
                    axp.set_xlim(-0.02, 1.02)
                    axp.set_ylim(-0.02, 1.02)
                    axp.set_title(title)
                    axp.set_xlabel("Predicted rank percentile")
                    if i == 0:
                        axp.set_ylabel("Observed rank percentile")
                    axp.grid(True, alpha=0.25)
                    if n_bin.size:
                        axp.text(
                            0.02,
                            0.03,
                            f"bin n: min={int(np.min(n_bin))}, median={int(np.median(n_bin))}, max={int(np.max(n_bin))}",
                            transform=axp.transAxes,
                            fontsize=8,
                            color="#444444",
                        )

                fig_p.suptitle(
                    f"{lane} | k={k} | binned rank trends (points with 95% bootstrap CI; no raw scatter)",
                    fontsize=11,
                )
                fig_p.tight_layout()
                out_p = binned_paper_dir / f"{lane}_k{k}_binned_points_ci.png"
                fig_p.savefig(out_p, dpi=220)
                plt.close(fig_p)
        else:
            ax0.text(0.5, 0.5, "No rank-detail rows", ha="center", va="center", fontsize=10)
            ax1.text(0.5, 0.5, "No rank-detail rows", ha="center", va="center", fontsize=10)
            ax0.set_axis_off()
            ax1.set_axis_off()

        best_true_rank_pct = pd.Series(dtype=float)
        if not summary.empty and "pred_best_true_rank_pct" in summary.columns:
            best_true_rank_pct = pd.to_numeric(summary["pred_best_true_rank_pct"], errors="coerce").dropna()
        if best_true_rank_pct.empty and not detail.empty:
            group_cols = [c for c in ("benchmark", "model_family_encoder") if c in detail.columns]
            if group_cols:
                idx = detail.groupby(group_cols, dropna=False)["pred_rank_pct"].idxmin()
                approx = detail.loc[idx, "true_rank_pct"]
                best_true_rank_pct = pd.to_numeric(approx, errors="coerce").dropna()

        if best_true_rank_pct.empty:
            ax2.text(0.5, 0.5, "No top-k hit data", ha="center", va="center", fontsize=10)
            ax2.set_axis_off()
        else:
            best_true_rank_pct = best_true_rank_pct.clip(0.0, 1.0)
            q = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50], dtype=float)
            hit = np.array([float((best_true_rank_pct <= qi).mean()) for qi in q], dtype=float)
            ax2.plot(q, hit, marker="o", linewidth=2.0, color="#2ca02c", label="Model")
            ax2.plot(q, q, linestyle="--", linewidth=1.0, color="k", alpha=0.7, label="Random")
            q20_idx = int(np.where(np.isclose(q, 0.20))[0][0])
            lift20 = float(hit[q20_idx] - q[q20_idx])
            ax2.text(
                0.52,
                0.08,
                f"Hit@20%: {hit[q20_idx]:.2f}\nLift vs random: {lift20:+.2f}",
                transform=ax2.transAxes,
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )
            ax2.set_xlim(0.0, 0.5)
            ax2.set_ylim(0.0, 1.0)
            ax2.set_xlabel("Allowed top-q percentile")
            ax2.set_ylabel("Best-choice hit rate")
            ax2.set_title("Top-q hit curve")
            ax2.grid(True, alpha=0.25)
            ax2.legend(loc="lower right", fontsize=8, framealpha=0.9)

        n_cells = len(best_true_rank_pct) if not best_true_rank_pct.empty else 0
        n_rows = len(detail) if not detail.empty else 0
        fig.suptitle(f"{lane} | k={k} | ranking views | cells={n_cells}, detail rows={n_rows}", fontsize=11)
        fig.tight_layout()
        out = plot_dir / f"{lane}_k{k}_ranking_views.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)


def _plot_finalist_fit_scatter_synthetic(
    out_dir: Path,
    selected_exact_cal: pd.DataFrame,
    pool: pd.DataFrame,
) -> None:
    if selected_exact_cal.empty:
        return
    plot_dir = out_dir / "fit_scatter_synthetic_exact_k"
    plot_dir.mkdir(parents=True, exist_ok=True)
    row_cache: Dict[str, pd.DataFrame] = {}
    for _, row in selected_exact_cal.sort_values(["lane", "k_target"]).iterrows():
        sig = str(row["signature"])
        lane = str(row["lane"])
        k = int(row["k_target"])
        rows = _collect_signature_rows(sig, pool, row_cache)
        if rows.empty:
            continue
        pred = pd.to_numeric(rows["prediction"], errors="coerce")
        tgt = pd.to_numeric(rows["target"], errors="coerce")
        m = pred.notna() & tgt.notna()
        rows = rows.loc[m].copy()
        rows["prediction"] = pred[m]
        rows["target"] = tgt[m]
        rows = rows.loc[_synthetic_mask(rows)].copy()
        if len(rows) < 10:
            continue
        strict_pred, _ = _strict_oof_linear_calibration(rows)
        sm = strict_pred.notna() & rows["target"].notna()
        pred_cal = strict_pred[sm]
        tgt_cal = rows.loc[sm, "target"]
        if pred_cal.empty:
            pred_cal = rows["prediction"]
            tgt_cal = rows["target"]

        raw = _compute_residual_diag(rows["prediction"], rows["target"])
        cal = _compute_residual_diag(pred_cal, tgt_cal)
        lo = float(min(rows["prediction"].min(), rows["target"].min(), pred_cal.min(), tgt_cal.min()))
        hi = float(max(rows["prediction"].max(), rows["target"].max(), pred_cal.max(), tgt_cal.max()))
        pad = 0.05 * (hi - lo + 1e-8)
        x0, x1 = lo - pad, hi + pad

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), squeeze=False)
        ax0, ax1 = axes[0, 0], axes[0, 1]
        ax0.scatter(rows["prediction"], rows["target"], s=14, alpha=0.35, color="#1f77b4")
        ax0.plot([x0, x1], [x0, x1], "k--", linewidth=1.0)
        ax0.set_xlim(x0, x1)
        ax0.set_ylim(x0, x1)
        ax0.set_title(f"Synthetic Raw: MAE={raw.mae:.2f}, RMSE={raw.rmse:.2f}")
        ax0.set_xlabel("Predicted")
        ax0.set_ylabel("Observed")
        ax0.grid(True, alpha=0.25)

        ax1.scatter(pred_cal, tgt_cal, s=14, alpha=0.35, color="#2ca02c")
        ax1.plot([x0, x1], [x0, x1], "k--", linewidth=1.0)
        ax1.set_xlim(x0, x1)
        ax1.set_ylim(x0, x1)
        ax1.set_title(f"Synthetic Strict OOF: MAE={cal.mae:.2f}, RMSE={cal.rmse:.2f}")
        ax1.set_xlabel("Strict OOF calibrated prediction")
        ax1.set_ylabel("Observed")
        ax1.grid(True, alpha=0.25)

        fig.suptitle(f"{lane} | k={k} | synthetic rows n={len(rows)}", fontsize=11)
        fig.tight_layout()
        out = plot_dir / f"{lane}_k{k}_synthetic_actual_vs_pred.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)


def _plot_finalist_fit_scatter_synthetic_cell_collapsed(
    out_dir: Path,
    selected_exact_cal: pd.DataFrame,
    pool: pd.DataFrame,
) -> None:
    if selected_exact_cal.empty:
        return
    plot_dir = out_dir / "fit_scatter_synthetic_cell_collapsed_exact_k"
    plot_dir.mkdir(parents=True, exist_ok=True)
    row_cache: Dict[str, pd.DataFrame] = {}
    for _, row in selected_exact_cal.sort_values(["lane", "k_target"]).iterrows():
        sig = str(row["signature"])
        lane = str(row["lane"])
        k = int(row["k_target"])
        rows = _collect_signature_rows(sig, pool, row_cache)
        if rows.empty:
            continue
        pred = pd.to_numeric(rows["prediction"], errors="coerce")
        tgt = pd.to_numeric(rows["target"], errors="coerce")
        m = pred.notna() & tgt.notna()
        rows = rows.loc[m].copy()
        rows["prediction"] = pred[m]
        rows["target"] = tgt[m]
        rows = rows.loc[_synthetic_mask(rows)].copy()
        if len(rows) < 10:
            continue
        strict_pred, _ = _strict_oof_linear_calibration(rows)
        rows["prediction_strict_oof_cal"] = strict_pred

        raw_coll = _collapse_fit_points(rows, pred_col="prediction")
        cal_coll = _collapse_fit_points(rows.dropna(subset=["prediction_strict_oof_cal"]), pred_col="prediction_strict_oof_cal")
        if raw_coll.empty:
            continue
        if cal_coll.empty:
            cal_coll = raw_coll.rename(columns={"prediction": "prediction_strict_oof_cal"})

        raw = _compute_residual_diag(raw_coll["prediction"], raw_coll["target"])
        cal = _compute_residual_diag(cal_coll["prediction_strict_oof_cal"], cal_coll["target"])
        lo = float(min(raw_coll["prediction"].min(), raw_coll["target"].min(), cal_coll["prediction_strict_oof_cal"].min()))
        hi = float(max(raw_coll["prediction"].max(), raw_coll["target"].max(), cal_coll["prediction_strict_oof_cal"].max()))
        pad = 0.05 * (hi - lo + 1e-8)
        x0, x1 = lo - pad, hi + pad

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), squeeze=False)
        ax0, ax1 = axes[0, 0], axes[0, 1]
        sz0 = np.clip(raw_coll["n_rows"].to_numpy() * 8.0, 14.0, 120.0)
        sz1 = np.clip(cal_coll["n_rows"].to_numpy() * 8.0, 14.0, 120.0)
        ax0.scatter(raw_coll["prediction"], raw_coll["target"], s=sz0, alpha=0.35, color="#1f77b4")
        ax0.plot([x0, x1], [x0, x1], "k--", linewidth=1.0)
        ax0.set_xlim(x0, x1)
        ax0.set_ylim(x0, x1)
        ax0.set_title(f"Synthetic Cell Raw: MAE={raw.mae:.2f}, RMSE={raw.rmse:.2f}")
        ax0.set_xlabel("Predicted")
        ax0.set_ylabel("Observed")
        ax0.grid(True, alpha=0.25)

        ax1.scatter(cal_coll["prediction_strict_oof_cal"], cal_coll["target"], s=sz1, alpha=0.35, color="#2ca02c")
        ax1.plot([x0, x1], [x0, x1], "k--", linewidth=1.0)
        ax1.set_xlim(x0, x1)
        ax1.set_ylim(x0, x1)
        ax1.set_title(f"Synthetic Cell Strict OOF: MAE={cal.mae:.2f}, RMSE={cal.rmse:.2f}")
        ax1.set_xlabel("Strict OOF calibrated prediction")
        ax1.set_ylabel("Observed")
        ax1.grid(True, alpha=0.25)

        fig.suptitle(
            f"{lane} | k={k} | synthetic collapsed cells={len(raw_coll)} (rows={len(rows)})",
            fontsize=11,
        )
        fig.tight_layout()
        out = plot_dir / f"{lane}_k{k}_synthetic_cell_collapsed_actual_vs_pred.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)


def _plot_finalist_fit_scatter_real_only(
    out_dir: Path,
    selected_exact_cal: pd.DataFrame,
    pool: pd.DataFrame,
) -> None:
    if selected_exact_cal.empty:
        return
    plot_dir = out_dir / "fit_scatter_real_exact_k"
    plot_dir.mkdir(parents=True, exist_ok=True)
    row_cache: Dict[str, pd.DataFrame] = {}
    for _, row in selected_exact_cal.sort_values(["lane", "k_target"]).iterrows():
        sig = str(row["signature"])
        lane = str(row["lane"])
        k = int(row["k_target"])
        rows = _collect_signature_rows(sig, pool, row_cache)
        if rows.empty:
            continue
        pred = pd.to_numeric(rows["prediction"], errors="coerce")
        tgt = pd.to_numeric(rows["target"], errors="coerce")
        m = pred.notna() & tgt.notna()
        rows = rows.loc[m].copy()
        rows["prediction"] = pred[m]
        rows["target"] = tgt[m]
        rows = rows.loc[_real_only_mask(rows)].copy()
        if len(rows) < 10:
            continue
        strict_pred, _ = _strict_oof_linear_calibration(rows)
        sm = strict_pred.notna() & rows["target"].notna()
        pred_cal = strict_pred[sm]
        tgt_cal = rows.loc[sm, "target"]
        if pred_cal.empty:
            pred_cal = rows["prediction"]
            tgt_cal = rows["target"]

        raw = _compute_residual_diag(rows["prediction"], rows["target"])
        cal = _compute_residual_diag(pred_cal, tgt_cal)
        lo = float(min(rows["prediction"].min(), rows["target"].min(), pred_cal.min(), tgt_cal.min()))
        hi = float(max(rows["prediction"].max(), rows["target"].max(), pred_cal.max(), tgt_cal.max()))
        pad = 0.05 * (hi - lo + 1e-8)
        x0, x1 = lo - pad, hi + pad

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), squeeze=False)
        ax0, ax1 = axes[0, 0], axes[0, 1]
        ax0.scatter(rows["prediction"], rows["target"], s=14, alpha=0.35, color="#1f77b4")
        ax0.plot([x0, x1], [x0, x1], "k--", linewidth=1.0)
        ax0.set_xlim(x0, x1)
        ax0.set_ylim(x0, x1)
        ax0.set_title(f"Real-only Raw: MAE={raw.mae:.2f}, RMSE={raw.rmse:.2f}")
        ax0.set_xlabel("Predicted")
        ax0.set_ylabel("Observed")
        ax0.grid(True, alpha=0.25)

        ax1.scatter(pred_cal, tgt_cal, s=14, alpha=0.35, color="#2ca02c")
        ax1.plot([x0, x1], [x0, x1], "k--", linewidth=1.0)
        ax1.set_xlim(x0, x1)
        ax1.set_ylim(x0, x1)
        ax1.set_title(f"Real-only Strict OOF: MAE={cal.mae:.2f}, RMSE={cal.rmse:.2f}")
        ax1.set_xlabel("Strict OOF calibrated prediction")
        ax1.set_ylabel("Observed")
        ax1.grid(True, alpha=0.25)

        fig.suptitle(f"{lane} | k={k} | real-only rows n={len(rows)}", fontsize=11)
        fig.tight_layout()
        out = plot_dir / f"{lane}_k{k}_real_actual_vs_pred.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)


def _plot_finalist_fit_scatter_real_only_cell_collapsed(
    out_dir: Path,
    selected_exact_cal: pd.DataFrame,
    pool: pd.DataFrame,
) -> None:
    if selected_exact_cal.empty:
        return
    plot_dir = out_dir / "fit_scatter_real_cell_collapsed_exact_k"
    plot_dir.mkdir(parents=True, exist_ok=True)
    row_cache: Dict[str, pd.DataFrame] = {}
    for _, row in selected_exact_cal.sort_values(["lane", "k_target"]).iterrows():
        sig = str(row["signature"])
        lane = str(row["lane"])
        k = int(row["k_target"])
        rows = _collect_signature_rows(sig, pool, row_cache)
        if rows.empty:
            continue
        pred = pd.to_numeric(rows["prediction"], errors="coerce")
        tgt = pd.to_numeric(rows["target"], errors="coerce")
        m = pred.notna() & tgt.notna()
        rows = rows.loc[m].copy()
        rows["prediction"] = pred[m]
        rows["target"] = tgt[m]
        rows = rows.loc[_real_only_mask(rows)].copy()
        if len(rows) < 10:
            continue
        strict_pred, _ = _strict_oof_linear_calibration(rows)
        rows["prediction_strict_oof_cal"] = strict_pred

        raw_coll = _collapse_fit_points(rows, pred_col="prediction")
        cal_coll = _collapse_fit_points(rows.dropna(subset=["prediction_strict_oof_cal"]), pred_col="prediction_strict_oof_cal")
        if raw_coll.empty:
            continue
        if cal_coll.empty:
            cal_coll = raw_coll.rename(columns={"prediction": "prediction_strict_oof_cal"})

        raw = _compute_residual_diag(raw_coll["prediction"], raw_coll["target"])
        cal = _compute_residual_diag(cal_coll["prediction_strict_oof_cal"], cal_coll["target"])
        lo = float(min(raw_coll["prediction"].min(), raw_coll["target"].min(), cal_coll["prediction_strict_oof_cal"].min()))
        hi = float(max(raw_coll["prediction"].max(), raw_coll["target"].max(), cal_coll["prediction_strict_oof_cal"].max()))
        pad = 0.05 * (hi - lo + 1e-8)
        x0, x1 = lo - pad, hi + pad

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), squeeze=False)
        ax0, ax1 = axes[0, 0], axes[0, 1]
        sz0 = np.clip(raw_coll["n_rows"].to_numpy() * 8.0, 14.0, 120.0)
        sz1 = np.clip(cal_coll["n_rows"].to_numpy() * 8.0, 14.0, 120.0)
        ax0.scatter(raw_coll["prediction"], raw_coll["target"], s=sz0, alpha=0.35, color="#1f77b4")
        ax0.plot([x0, x1], [x0, x1], "k--", linewidth=1.0)
        ax0.set_xlim(x0, x1)
        ax0.set_ylim(x0, x1)
        ax0.set_title(f"Real-only Cell Raw: MAE={raw.mae:.2f}, RMSE={raw.rmse:.2f}")
        ax0.set_xlabel("Predicted")
        ax0.set_ylabel("Observed")
        ax0.grid(True, alpha=0.25)

        ax1.scatter(cal_coll["prediction_strict_oof_cal"], cal_coll["target"], s=sz1, alpha=0.35, color="#2ca02c")
        ax1.plot([x0, x1], [x0, x1], "k--", linewidth=1.0)
        ax1.set_xlim(x0, x1)
        ax1.set_ylim(x0, x1)
        ax1.set_title(f"Real-only Cell Strict OOF: MAE={cal.mae:.2f}, RMSE={cal.rmse:.2f}")
        ax1.set_xlabel("Strict OOF calibrated prediction")
        ax1.set_ylabel("Observed")
        ax1.grid(True, alpha=0.25)

        fig.suptitle(
            f"{lane} | k={k} | real-only collapsed cells={len(raw_coll)} (rows={len(rows)})",
            fontsize=11,
        )
        fig.tight_layout()
        out = plot_dir / f"{lane}_k{k}_real_cell_collapsed_actual_vs_pred.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)


def _plot_effect_control_fit_scatter(out_dir: Path, selected_exact_cal: pd.DataFrame, pool: pd.DataFrame) -> None:
    """Effect/control-specific fit plots using only base variants."""
    if selected_exact_cal.empty:
        return
    plot_dir = out_dir / "fit_scatter_effects_controls_exact_k"
    plot_dir.mkdir(parents=True, exist_ok=True)
    row_cache: Dict[str, pd.DataFrame] = {}

    for _, row in selected_exact_cal.sort_values(["lane", "k_target"]).iterrows():
        sig = str(row["signature"])
        lane = str(row["lane"])
        k = int(row["k_target"])
        sub = pool[
            (pool["signature"] == sig)
            & (pool["variant"].astype(str).str.contains("_base_"))
        ].copy()
        if sub.empty:
            continue

        frames: List[pd.DataFrame] = []
        for _, r in sub.iterrows():
            p = str(r["path"])
            if p not in row_cache:
                row_cache[p] = _load_jointood_rows(Path(p))
            rows = row_cache[p]
            if rows.empty:
                continue
            frames.append(rows.copy())
        if not frames:
            continue
        rows_all = pd.concat(frames, ignore_index=True)
        pred = pd.to_numeric(rows_all["prediction"], errors="coerce")
        tgt = pd.to_numeric(rows_all["target"], errors="coerce")
        mask = pred.notna() & tgt.notna()
        rows_all = rows_all.loc[mask].copy()
        rows_all["prediction"] = pred[mask]
        rows_all["target"] = tgt[mask]
        if rows_all.empty:
            continue

        pred = rows_all["prediction"]
        tgt = rows_all["target"]
        strict_pred, _ = _strict_oof_linear_calibration(rows_all)
        strict_mask = strict_pred.notna() & tgt.notna()
        pred_cal = strict_pred[strict_mask]
        tgt_cal = tgt[strict_mask]
        if pred_cal.empty:
            pred_cal = pred.copy()
            tgt_cal = tgt.copy()

        raw = _compute_residual_diag(pred, tgt)
        cal = _compute_residual_diag(pred_cal, tgt_cal)

        lo = float(min(pred.min(), tgt.min(), pred_cal.min()))
        hi = float(max(pred.max(), tgt.max(), pred_cal.max()))
        pad = 0.05 * (hi - lo + 1e-8)
        x0, x1 = lo - pad, hi + pad

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), squeeze=False)
        ax0, ax1 = axes[0, 0], axes[0, 1]
        ax0.scatter(pred, tgt, s=14, alpha=0.35, color="#1f77b4")
        ax0.plot([x0, x1], [x0, x1], "k--", linewidth=1.0)
        ax0.set_xlim(x0, x1)
        ax0.set_ylim(x0, x1)
        ax0.set_title(f"Base-only Raw: MAE={raw.mae:.2f}, RMSE={raw.rmse:.2f}")
        ax0.set_xlabel("Predicted")
        ax0.set_ylabel("Observed")
        ax0.grid(True, alpha=0.25)

        ax1.scatter(pred_cal, tgt_cal, s=14, alpha=0.35, color="#2ca02c")
        ax1.plot([x0, x1], [x0, x1], "k--", linewidth=1.0)
        ax1.set_xlim(x0, x1)
        ax1.set_ylim(x0, x1)
        ax1.set_title(f"Base-only Strict OOF: MAE={cal.mae:.2f}, RMSE={cal.rmse:.2f}")
        ax1.set_xlabel("Strict OOF calibrated prediction")
        ax1.set_ylabel("Observed")
        ax1.grid(True, alpha=0.25)

        fig.suptitle(f"{lane} | k={k} | base variants only | n={len(pred)}", fontsize=11)
        fig.tight_layout()
        out = plot_dir / f"{lane}_k{k}_base_only_actual_vs_pred.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)


def _plot_effect_control_fit_scatter_cell_collapsed(
    out_dir: Path,
    selected_exact_cal: pd.DataFrame,
    pool: pd.DataFrame,
) -> None:
    if selected_exact_cal.empty:
        return
    plot_dir = out_dir / "fit_scatter_effects_controls_cell_collapsed_exact_k"
    plot_dir.mkdir(parents=True, exist_ok=True)
    row_cache: Dict[str, pd.DataFrame] = {}

    for _, row in selected_exact_cal.sort_values(["lane", "k_target"]).iterrows():
        sig = str(row["signature"])
        lane = str(row["lane"])
        k = int(row["k_target"])
        sub = pool[
            (pool["signature"] == sig)
            & (pool["variant"].astype(str).str.contains("_base_"))
        ].copy()
        if sub.empty:
            continue

        frames: List[pd.DataFrame] = []
        for _, r in sub.iterrows():
            p = str(r["path"])
            if p not in row_cache:
                row_cache[p] = _load_jointood_rows(Path(p))
            rows = row_cache[p]
            if rows.empty:
                continue
            tmp = rows.copy()
            tmp["variant"] = str(r["variant"])
            frames.append(tmp)
        if not frames:
            continue
        rows_all = pd.concat(frames, ignore_index=True)
        pred = pd.to_numeric(rows_all["prediction"], errors="coerce")
        tgt = pd.to_numeric(rows_all["target"], errors="coerce")
        mask = pred.notna() & tgt.notna()
        rows_all = rows_all.loc[mask].copy()
        rows_all["prediction"] = pred[mask]
        rows_all["target"] = tgt[mask]
        if rows_all.empty:
            continue

        strict_pred, _ = _strict_oof_linear_calibration(rows_all)
        rows_all["prediction_strict_oof_cal"] = strict_pred

        raw_coll = _collapse_fit_points(rows_all, pred_col="prediction")
        cal_coll = _collapse_fit_points(rows_all.dropna(subset=["prediction_strict_oof_cal"]), pred_col="prediction_strict_oof_cal")
        if raw_coll.empty:
            continue
        if cal_coll.empty:
            cal_coll = raw_coll.rename(columns={"prediction": "prediction_strict_oof_cal"})

        raw = _compute_residual_diag(raw_coll["prediction"], raw_coll["target"])
        cal = _compute_residual_diag(cal_coll["prediction_strict_oof_cal"], cal_coll["target"])

        lo = float(min(raw_coll["prediction"].min(), raw_coll["target"].min(), cal_coll["prediction_strict_oof_cal"].min()))
        hi = float(max(raw_coll["prediction"].max(), raw_coll["target"].max(), cal_coll["prediction_strict_oof_cal"].max()))
        pad = 0.05 * (hi - lo + 1e-8)
        x0, x1 = lo - pad, hi + pad

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), squeeze=False)
        ax0, ax1 = axes[0, 0], axes[0, 1]
        sz0 = np.clip(raw_coll["n_rows"].to_numpy() * 8.0, 14.0, 120.0)
        sz1 = np.clip(cal_coll["n_rows"].to_numpy() * 8.0, 14.0, 120.0)

        ax0.scatter(raw_coll["prediction"], raw_coll["target"], s=sz0, alpha=0.35, color="#1f77b4")
        ax0.plot([x0, x1], [x0, x1], "k--", linewidth=1.0)
        ax0.set_xlim(x0, x1)
        ax0.set_ylim(x0, x1)
        ax0.set_title(f"Base-only Cell Raw: MAE={raw.mae:.2f}, RMSE={raw.rmse:.2f}")
        ax0.set_xlabel("Predicted")
        ax0.set_ylabel("Observed")
        ax0.grid(True, alpha=0.25)

        ax1.scatter(cal_coll["prediction_strict_oof_cal"], cal_coll["target"], s=sz1, alpha=0.35, color="#2ca02c")
        ax1.plot([x0, x1], [x0, x1], "k--", linewidth=1.0)
        ax1.set_xlim(x0, x1)
        ax1.set_ylim(x0, x1)
        ax1.set_title(f"Base-only Cell Strict OOF: MAE={cal.mae:.2f}, RMSE={cal.rmse:.2f}")
        ax1.set_xlabel("Strict OOF calibrated prediction")
        ax1.set_ylabel("Observed")
        ax1.grid(True, alpha=0.25)

        fig.suptitle(
            f"{lane} | k={k} | base-only collapsed cells={len(raw_coll)} (rows={len(rows_all)})",
            fontsize=11,
        )
        fig.tight_layout()
        out = plot_dir / f"{lane}_k{k}_base_only_cell_collapsed_actual_vs_pred.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)


def _plot_motion_vs_appearance_tradeoff(out_dir: Path, selected_exact_cal: pd.DataFrame) -> None:
    if selected_exact_cal.empty:
        return
    mot = selected_exact_cal[selected_exact_cal["lane"] == "motion_only"].copy()
    app = selected_exact_cal[selected_exact_cal["lane"] == "appearance_only"].copy()
    if mot.empty or app.empty:
        return
    m_by_k = {int(r["k_target"]): r for _, r in mot.iterrows()}
    a_by_k = {int(r["k_target"]): r for _, r in app.iterrows()}
    ks = sorted(set(m_by_k.keys()) & set(a_by_k.keys()))
    if not ks:
        return

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    x_raw: List[float] = []
    y_raw: List[float] = []
    x_cal: List[float] = []
    y_cal: List[float] = []

    for k in ks:
        mr = m_by_k[k]
        ar = a_by_k[k]
        x0 = float(mr.get("pooled_raw_mae", math.nan))
        y0 = float(ar.get("pooled_raw_mae", math.nan))
        x1 = float(mr.get("pooled_cal_mae", math.nan))
        y1 = float(ar.get("pooled_cal_mae", math.nan))
        if np.isfinite(x0) and np.isfinite(y0):
            x_raw.append(x0)
            y_raw.append(y0)
            ax.annotate(f"k={k}", (x0, y0), textcoords="offset points", xytext=(4, 4), fontsize=8, color="#1f77b4")
        if np.isfinite(x1) and np.isfinite(y1):
            x_cal.append(x1)
            y_cal.append(y1)
            ax.annotate(f"k={k}", (x1, y1), textcoords="offset points", xytext=(4, -10), fontsize=8, color="#d62728")

    if x_raw:
        ax.scatter(x_raw, y_raw, marker="o", s=70, color="#1f77b4", alpha=0.85, label="Raw")
    if x_cal:
        ax.scatter(x_cal, y_cal, marker="s", s=70, color="#d62728", alpha=0.85, label="Calibrated")

    vals = [*x_raw, *y_raw, *x_cal, *y_cal]
    if vals:
        lo = min(vals)
        hi = max(vals)
        pad = 0.05 * (hi - lo + 1e-8)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1.0, alpha=0.7)

    ax.set_xlabel("Motion-only MAE (lower better)")
    ax.set_ylabel("Appearance-only MAE (lower better)")
    ax.set_title("Motion vs Appearance by k (Joint-OOD MAE)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "motion_vs_appearance_pareto_by_k.png", dpi=180)
    plt.close(fig)


def _select_best_largek_model(pool: pd.DataFrame, variant_pattern: str) -> Optional[pd.Series]:
    sub = pool[pool["variant"].astype(str).str.contains(variant_pattern)].copy()
    if sub.empty:
        return None
    kmax = int(sub["signal_k"].max())
    cand = sub[sub["signal_k"] == kmax].copy()
    if cand.empty:
        return None
    cand = cand.sort_values(["jointood_mae", "jointood_spearman"], ascending=[True, False])
    return cand.iloc[0]


def _plot_best_largek_effects_vs_noeffects(out_dir: Path, pool: pd.DataFrame) -> pd.DataFrame:
    selected: List[Dict[str, object]] = []
    specs = [
        ("effects_present", "_base_"),
        ("effects_absent", "_no_family_no_density_"),
    ]
    for label, pat in specs:
        row = _select_best_largek_model(pool, pat)
        if row is None:
            continue
        selected.append(
            {
                "setting": label,
                "variant": str(row["variant"]),
                "model": str(row.get("model", "")),
                "method": str(row.get("method", "")),
                "path": str(row.get("path", "")),
                "signal_k": int(row.get("signal_k", 0)),
                "jointood_mae": float(row.get("jointood_mae", np.nan)),
                "jointood_spearman": float(row.get("jointood_spearman", np.nan)),
                "jointood_rank_spearman": float(row.get("jointood_rank_spearman", np.nan)),
                "jointood_rank_kendall_tau": float(row.get("jointood_rank_kendall_tau", np.nan)),
                "jointood_rank_pairwise_cindex": float(row.get("jointood_rank_pairwise_cindex", np.nan)),
                "jointood_rank_pct_err": float(row.get("jointood_rank_pct_err", np.nan)),
                "signature": str(row.get("signature", "")),
                "signal_tokens": str(row.get("signal_tokens", "")),
            }
        )

    sel_df = pd.DataFrame(selected)
    if sel_df.empty:
        return sel_df

    n = len(sel_df)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4.3 * n), squeeze=False)

    for i, (_, r) in enumerate(sel_df.iterrows()):
        path = Path(str(r["path"]))
        rows = _load_jointood_rows(path)
        if rows.empty:
            continue
        pred = pd.to_numeric(rows["prediction"], errors="coerce")
        tgt = pd.to_numeric(rows["target"], errors="coerce")
        mask = pred.notna() & tgt.notna()
        rows = rows.loc[mask].copy()
        rows["prediction"] = pred[mask]
        rows["target"] = tgt[mask]
        if rows.empty:
            continue

        # Strict OOF calibrator for selected model: fit on other variants in same setting/signature.
        setting = str(r["setting"])
        pat = "_base_" if setting == "effects_present" else "_no_family_no_density_"
        same = pool[
            (pool["variant"].astype(str).str.contains(pat))
            & (pool["signature"].astype(str) == str(r["signature"]))
        ].copy()
        frames: List[pd.DataFrame] = []
        for _, prow in same.iterrows():
            rp = _load_jointood_rows(Path(str(prow["path"])))
            if rp.empty:
                continue
            tmp = rp.copy()
            tmp["variant"] = str(prow["variant"])
            frames.append(tmp)
        raw_tgt = rows["target"]
        raw = _compute_residual_diag(rows["prediction"], raw_tgt)
        fit = CalibrationFit()
        pred_cal = rows["prediction"].copy()
        tgt_cal = raw_tgt.copy()
        if frames:
            all_rows = pd.concat(frames, ignore_index=True)
            strict_all, strict_fits = _strict_oof_linear_calibration(all_rows)
            all_rows = all_rows.copy()
            all_rows["prediction_strict_oof_cal"] = strict_all
            this_variant = str(r["variant"])
            take = all_rows[
                (all_rows["variant"].astype(str) == this_variant)
                & all_rows["prediction_strict_oof_cal"].notna()
            ]
            if not take.empty:
                pred_cal = pd.to_numeric(take["prediction_strict_oof_cal"], errors="coerce")
                tgt_cal = pd.to_numeric(take["target"], errors="coerce")
                fit_s = strict_fits[strict_fits["heldout_variant"].astype(str) == this_variant]
                if not fit_s.empty:
                    fit = CalibrationFit(
                        float(pd.to_numeric(fit_s["slope"], errors="coerce").iloc[0]),
                        float(pd.to_numeric(fit_s["intercept"], errors="coerce").iloc[0]),
                    )
        cal = _compute_residual_diag(pred_cal, tgt_cal)

        sel_df.loc[i, "raw_mae"] = raw.mae
        sel_df.loc[i, "raw_rmse"] = raw.rmse
        sel_df.loc[i, "strict_oof_cal_mae"] = cal.mae
        sel_df.loc[i, "strict_oof_cal_rmse"] = cal.rmse
        sel_df.loc[i, "strict_oof_delta_mae"] = raw.mae - cal.mae if np.isfinite(raw.mae) and np.isfinite(cal.mae) else math.nan
        sel_df.loc[i, "strict_oof_delta_rmse"] = raw.rmse - cal.rmse if np.isfinite(raw.rmse) and np.isfinite(cal.rmse) else math.nan
        sel_df.loc[i, "strict_oof_fit_slope"] = fit.slope
        sel_df.loc[i, "strict_oof_fit_intercept"] = fit.intercept

        lo = float(min(rows["prediction"].min(), raw_tgt.min(), pred_cal.min(), tgt_cal.min()))
        hi = float(max(rows["prediction"].max(), raw_tgt.max(), pred_cal.max(), tgt_cal.max()))
        pad = 0.05 * (hi - lo + 1e-8)
        x0, x1 = lo - pad, hi + pad

        ax0 = axes[i, 0]
        ax1 = axes[i, 1]
        ax0.scatter(rows["prediction"], raw_tgt, s=14, alpha=0.35, color="#1f77b4")
        ax0.plot([x0, x1], [x0, x1], "k--", linewidth=1.0)
        ax0.set_xlim(x0, x1)
        ax0.set_ylim(x0, x1)
        ax0.set_xlabel("Predicted")
        ax0.set_ylabel("Observed")
        ax0.set_title(f"{r['setting']} raw | MAE={raw.mae:.2f}, RMSE={raw.rmse:.2f}")
        ax0.grid(True, alpha=0.25)

        ax1.scatter(pred_cal, tgt_cal, s=14, alpha=0.35, color="#2ca02c")
        ax1.plot([x0, x1], [x0, x1], "k--", linewidth=1.0)
        ax1.set_xlim(x0, x1)
        ax1.set_ylim(x0, x1)
        ax1.set_xlabel("Calibrated prediction")
        ax1.set_ylabel("Observed")
        ax1.set_title(f"{r['setting']} strict OOF | MAE={cal.mae:.2f}, RMSE={cal.rmse:.2f}")
        ax1.grid(True, alpha=0.25)

        ax0.text(
            0.02,
            0.98,
            f"variant={r['variant']}\nmethod={r['method']}\nk={int(r['signal_k'])}",
            transform=ax0.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.8"),
        )

    fig.suptitle("Best Large-k: Effects+Controls vs No-Effects", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "best_largek_effects_vs_noeffects_actual_vs_pred.png", dpi=180)
    plt.close(fig)
    return sel_df


def _build_red_flags(agg: pd.DataFrame, selected_exact_diag: pd.DataFrame, min_runs: int, mae_std_flag: float) -> List[str]:
    flags: List[str] = []
    if agg.empty:
        flags.append("No candidates found after filtering.")
        return flags

    sparse = agg[agg["n_runs"] < min_runs]
    if not sparse.empty:
        flags.append(f"{len(sparse)} signatures have low coverage (n_runs < {min_runs}).")

    unstable = agg[agg["std_jointood_mae"] > mae_std_flag]
    if not unstable.empty:
        flags.append(
            f"{len(unstable)} signatures show high Joint-OOD variance (std_mae > {mae_std_flag:.2f})."
        )

    if not selected_exact_diag.empty and "diag_cal_slope_mean" in selected_exact_diag.columns:
        bad_cal = selected_exact_diag[
            selected_exact_diag["diag_cal_slope_mean"].notna()
            & ((selected_exact_diag["diag_cal_slope_mean"] < 0.5) | (selected_exact_diag["diag_cal_slope_mean"] > 1.5))
        ]
        if not bad_cal.empty:
            flags.append(f"{len(bad_cal)} selected models have weak calibration slope (outside [0.5, 1.5]).")

    if not selected_exact_diag.empty and "strict_oof_delta_mae" in selected_exact_diag.columns:
        worse = selected_exact_diag[
            selected_exact_diag["strict_oof_delta_mae"].notna()
            & (selected_exact_diag["strict_oof_delta_mae"] < 0.0)
        ]
        if not worse.empty:
            flags.append(f"{len(worse)} selected models get worse MAE after strict OOF calibration.")

    lane_counts = agg.sort_values("mean_jointood_mae").head(20)["lane"].value_counts()
    if not lane_counts.empty and (lane_counts.max() / lane_counts.sum()) > 0.85:
        flags.append("Top candidates are strongly lane-imbalanced (>85% in one lane).")

    return flags


def _write_markdown_report(
    out_path: Path,
    run_roots: Sequence[Path],
    agg: pd.DataFrame,
    selected_exact_diag: pd.DataFrame,
    selected_upto_diag: pd.DataFrame,
    selected_exact_base: pd.DataFrame,
    largek_compare: pd.DataFrame,
    subgroup_robustness: pd.DataFrame,
    sanity_checks: pd.DataFrame,
    red_flags: Sequence[str],
    selection_objective: str,
) -> None:
    lines: List[str] = []
    lines.append("# Final Utility Sweep")
    lines.append("")
    lines.append("## Inputs")
    for r in run_roots:
        lines.append(f"- `{r}`")
    lines.append("")
    if selection_objective == "ranking":
        lines.append(
            "Primary selector: **ranking-first** (pairwise c-index, rank Spearman, rank Kendall; "
            "lower rank pct error and MAE as tie-breakers)."
        )
    else:
        lines.append("Primary selector: **Joint-OOD MAE** (lower is better), tie-break: Joint-OOD Spearman.")
    lines.append(
        "Ranking diagnostics tracked: Joint-OOD rank Spearman / Kendall tau / pairwise c-index (higher better), rank pct error (lower better)."
    )
    lines.append("Calibration protocol: **strict OOF by variant** (fit calibrator on other variants, apply to held-out variant).")
    lines.append("")

    if not agg.empty:
        lines.append("## Top Stable Signatures")
        show = _sort_for_selection(agg.copy(), objective=selection_objective).head(20).copy()
        lines.append(show[[
            "lane",
            "signal_k",
            "n_runs",
            "mean_jointood_mae",
            "std_jointood_mae",
            "mean_jointood_spearman",
            "mean_jointood_rank_spearman",
            "mean_jointood_rank_kendall_tau",
            "mean_jointood_rank_pairwise_cindex",
            "mean_jointood_rank_pct_err",
            "signal_tokens",
        ]].to_markdown(index=False))
        lines.append("")

    if not selected_exact_diag.empty:
        lines.append("## Finalists by Exact-k")
        lines.append(selected_exact_diag[[
            "lane",
            "k_target",
            "signal_k",
            "mean_jointood_mae",
            "std_jointood_mae",
            "mean_jointood_spearman",
            "mean_jointood_rank_spearman",
            "mean_jointood_rank_kendall_tau",
            "mean_jointood_rank_pairwise_cindex",
            "mean_jointood_rank_pct_err",
            "signal_tokens",
            "diag_mae_mean",
            "diag_rmse_mean",
            "diag_medae_mean",
            "diag_p90ae_mean",
            "diag_cal_slope_mean",
            "diag_cal_intercept_mean",
            "pooled_raw_mae",
            "strict_oof_cal_mae",
            "strict_oof_delta_mae",
            "strict_oof_mae_ci95_lo",
            "strict_oof_mae_ci95_hi",
        ]].to_markdown(index=False))
        lines.append("")

    if not selected_upto_diag.empty:
        lines.append("## Finalists by <=k")
        lines.append(selected_upto_diag[[
            "lane",
            "k_target",
            "signal_k",
            "mean_jointood_mae",
            "std_jointood_mae",
            "mean_jointood_spearman",
            "mean_jointood_rank_spearman",
            "mean_jointood_rank_kendall_tau",
            "mean_jointood_rank_pairwise_cindex",
            "mean_jointood_rank_pct_err",
            "signal_tokens",
            "diag_mae_mean",
            "diag_rmse_mean",
            "diag_medae_mean",
            "diag_p90ae_mean",
            "diag_cal_slope_mean",
            "diag_cal_intercept_mean",
            "pooled_raw_mae",
            "strict_oof_cal_mae",
            "strict_oof_delta_mae",
            "strict_oof_mae_ci95_lo",
            "strict_oof_mae_ci95_hi",
        ]].to_markdown(index=False))
        lines.append("")

    if not selected_exact_diag.empty:
        lines.append("## Calibration Summary (Exact-k)")
        cal = selected_exact_diag[[
            "lane",
            "k_target",
            "signal_k",
            "cal_linear_slope",
            "cal_linear_intercept",
            "pooled_raw_mae",
            "pooled_cal_mae",
            "pooled_cal_delta_mae",
            "pooled_raw_rmse",
            "pooled_cal_rmse",
            "pooled_cal_delta_rmse",
            "strict_oof_cal_mae",
            "strict_oof_delta_mae",
            "strict_oof_cal_rmse",
            "strict_oof_delta_rmse",
        ]].sort_values(["lane", "k_target"])
        lines.append(cal.to_markdown(index=False))
        lines.append("")

    if not selected_exact_base.empty:
        lines.append("## Effects+Controls View (Base Variants Only, Exact-k)")
        base_cols = [
            "lane",
            "k_target",
            "signal_k",
            "base_cal_linear_slope",
            "base_cal_linear_intercept",
            "base_pooled_raw_mae",
            "base_pooled_cal_mae",
            "base_pooled_cal_delta_mae",
            "base_pooled_raw_rmse",
            "base_pooled_cal_rmse",
            "base_pooled_cal_delta_rmse",
            "base_strict_oof_cal_mae",
            "base_strict_oof_delta_mae",
            "base_strict_oof_cal_rmse",
            "base_strict_oof_delta_rmse",
        ]
        base_cols = [c for c in base_cols if c in selected_exact_base.columns]
        if base_cols:
            sort_cols = [c for c in ("lane", "k_target") if c in base_cols]
            if sort_cols:
                base_tab = selected_exact_base[base_cols].sort_values(sort_cols)
            else:
                base_tab = selected_exact_base[base_cols]
            lines.append(base_tab.to_markdown(index=False))
        else:
            lines.append("- No base-variant calibrated columns were available for this selection.")
        lines.append("")

    if not largek_compare.empty:
        lines.append("## Best Large-k Model Comparison (Effects vs No-Effects)")
        show = largek_compare[[
            "setting",
            "variant",
            "method",
            "signal_k",
            "jointood_mae",
            "jointood_spearman",
            "jointood_rank_spearman",
            "jointood_rank_kendall_tau",
            "jointood_rank_pairwise_cindex",
            "jointood_rank_pct_err",
            "raw_mae",
            "strict_oof_cal_mae",
            "strict_oof_delta_mae",
            "strict_oof_fit_slope",
            "strict_oof_fit_intercept",
            "signal_tokens",
        ]].copy()
        lines.append(show.to_markdown(index=False))
        lines.append("")

    if not subgroup_robustness.empty:
        lines.append("## Subgroup Robustness (All vs Synthetic vs Non-Synthetic)")
        show = subgroup_robustness[[
            "lane",
            "k_target",
            "subgroup",
            "n_rows",
            "raw_mae",
            "strict_oof_mae",
            "strict_oof_delta_mae",
            "raw_mae_ci95_lo",
            "raw_mae_ci95_hi",
            "strict_oof_mae_ci95_lo",
            "strict_oof_mae_ci95_hi",
        ]].sort_values(["lane", "k_target", "subgroup"])
        lines.append(show.to_markdown(index=False))
        lines.append("")

    if not sanity_checks.empty:
        lines.append("## Sanity Checks (Permutation Within Variant)")
        show = sanity_checks[[
            "lane",
            "k_target",
            "n_rows",
            "obs_strict_oof_mae",
            "perm_mae_mean",
            "p_value_mae_lower_than_perm",
            "obs_strict_oof_spearman",
            "perm_spearman_mean",
            "p_value_spearman_higher_than_perm",
            "n_permutations_used",
        ]].sort_values(["lane", "k_target"])
        lines.append(show.to_markdown(index=False))
        lines.append("")

    lines.append("## Red Flags")
    if red_flags:
        for f in red_flags:
            lines.append(f"- {f}")
    else:
        lines.append("- No immediate red flags from sweep-level diagnostics.")
    lines.append("")
    lines.append("## Plots")
    lines.append("- `motion_vs_appearance_pareto_by_k.png`")
    lines.append("- `fit_scatter_exact_k/<lane>_k<k>_actual_vs_pred.png`")
    lines.append("- `fit_scatter_cell_collapsed_exact_k/<lane>_k<k>_cell_collapsed_actual_vs_pred.png`")
    lines.append("- `rank_views_exact_k/<lane>_k<k>_ranking_views.png`")
    lines.append("- `rank_binned_exact_k/<lane>_k<k>_binned_median_mean_mode.png`")
    lines.append("- `rank_binned_paper_exact_k/<lane>_k<k>_binned_points_ci.png`")
    lines.append("- `fit_scatter_synthetic_exact_k/<lane>_k<k>_synthetic_actual_vs_pred.png`")
    lines.append("- `fit_scatter_synthetic_cell_collapsed_exact_k/<lane>_k<k>_synthetic_cell_collapsed_actual_vs_pred.png`")
    lines.append("- `fit_scatter_real_exact_k/<lane>_k<k>_real_actual_vs_pred.png`")
    lines.append("- `fit_scatter_real_cell_collapsed_exact_k/<lane>_k<k>_real_cell_collapsed_actual_vs_pred.png`")
    lines.append("- `fit_scatter_effects_controls_exact_k/<lane>_k<k>_base_only_actual_vs_pred.png`")
    lines.append("- `fit_scatter_effects_controls_cell_collapsed_exact_k/<lane>_k<k>_base_only_cell_collapsed_actual_vs_pred.png`")
    lines.append("- `best_largek_effects_vs_noeffects_actual_vs_pred.png`")
    lines.append("")
    out_path.write_text("\n".join(lines))


def _default_run_roots() -> List[Path]:
    base = Path("analysis_comprehensive_runs")
    names = [
        "hof_motion_v3_density_jointood_full_ridge_a10_base_v1",
        "hof_motion_v3_density_jointood_full_ridge_a10_no_family_v1",
        "hof_motion_v3_density_jointood_full_ridge_a10_no_family_no_density_v1",
        "hof_motion_v3_density_jointood_full_ols_base_v1",
        "hof_motion_v3_density_jointood_full_ols_no_family_v1",
        "hof_motion_v3_density_jointood_full_ols_no_family_no_density_v1",
    ]
    return [base / n for n in names]


def _resolve_selection_objective(selection_objective: str, run_roots: Sequence[Path]) -> str:
    if selection_objective in {"absolute", "ranking"}:
        return selection_objective
    inferred = sorted({_model_for_root(r) for r in run_roots})
    if len(inferred) == 1 and inferred[0] == "pairwise_rank":
        return "ranking"
    return "absolute"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final utility sweep and diagnostics.")
    parser.add_argument("--run-roots", default="", help="Comma-separated run roots (default: six-pack roots).")
    parser.add_argument("--output-dir", default="analysis_comprehensive_runs/final_utility_sweep", help="Output directory.")
    parser.add_argument("--k-values", default="1,2,3,4,5,6,7,8", help="Comma-separated k values.")
    parser.add_argument("--min-runs", type=int, default=3, help="Minimum run coverage for selection.")
    parser.add_argument(
        "--selection-policy",
        choices=["ladder", "unbounded", "both"],
        default="ladder",
        help=(
            "Selection mode: ladder (exact/<=k), unbounded (ignore k ladder), "
            "or both (write both sets; diagnostics use ladder finalists)."
        ),
    )
    parser.add_argument(
        "--unbounded-top-n",
        type=int,
        default=10,
        help="Top-N per lane to export for unbounded selection mode.",
    )
    parser.add_argument(
        "--selection-objective",
        choices=["auto", "absolute", "ranking"],
        default="auto",
        help=(
            "Finalist objective: auto (ranking for pairwise runs, absolute otherwise), "
            "or force absolute/ranking."
        ),
    )
    parser.add_argument("--mae-std-flag", type=float, default=2.0, help="Red-flag threshold for std Joint-OOD MAE.")
    args = parser.parse_args()

    if args.run_roots.strip():
        run_roots = [Path(x.strip()) for x in args.run_roots.split(",") if x.strip()]
    else:
        run_roots = _default_run_roots()
    selection_objective = _resolve_selection_objective(args.selection_objective, run_roots)
    print(f"Selection objective resolved to: {selection_objective}")
    ks = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    ks = sorted(set(ks))
    lanes = ["motion_only", "appearance_only", "hybrid"]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = _collect_pool(run_roots, objective=selection_objective)
    if pool.empty:
        raise SystemExit("No candidate rows collected from run roots.")
    pool.to_csv(out_dir / "candidate_pool_per_run.csv", index=False)

    agg = _aggregate_signatures(pool)
    agg.to_csv(out_dir / "signature_summary.csv", index=False)

    selected_unbounded = _select_unbounded(
        agg,
        lanes=lanes,
        min_runs=args.min_runs,
        top_n=args.unbounded_top_n,
        objective=selection_objective,
    )
    selected_unbounded.to_csv(out_dir / "selected_unbounded_any_k.csv", index=False)

    if args.selection_policy == "ladder":
        selected_exact = _select_ladder(
            agg,
            ks=ks,
            lanes=lanes,
            mode="exact",
            min_runs=args.min_runs,
            objective=selection_objective,
        )
        selected_upto = _select_ladder(
            agg,
            ks=ks,
            lanes=lanes,
            mode="upto",
            min_runs=args.min_runs,
            objective=selection_objective,
        )
    elif args.selection_policy == "unbounded":
        if selected_unbounded.empty or "lane_rank" not in selected_unbounded.columns:
            selected_exact = selected_unbounded.copy()
        else:
            selected_exact = selected_unbounded[selected_unbounded["lane_rank"] == 1].copy()
        selected_upto = pd.DataFrame(columns=selected_exact.columns)
    else:
        selected_exact = _select_ladder(
            agg,
            ks=ks,
            lanes=lanes,
            mode="exact",
            min_runs=args.min_runs,
            objective=selection_objective,
        )
        selected_upto = _select_ladder(
            agg,
            ks=ks,
            lanes=lanes,
            mode="upto",
            min_runs=args.min_runs,
            objective=selection_objective,
        )
    selected_exact.to_csv(out_dir / "selected_exact_k.csv", index=False)
    selected_upto.to_csv(out_dir / "selected_upto_k.csv", index=False)

    selected_exact_diag = _attach_residual_diagnostics(selected_exact, pool)
    selected_upto_diag = _attach_residual_diagnostics(selected_upto, pool)
    selected_exact_cal = _attach_calibrated_diagnostics(selected_exact_diag, pool)
    selected_upto_cal = _attach_calibrated_diagnostics(selected_upto_diag, pool)
    selected_exact_cal = _attach_bootstrap_uncertainty(selected_exact_cal, pool)
    selected_upto_cal = _attach_bootstrap_uncertainty(selected_upto_cal, pool)
    selected_exact_base = _attach_base_variant_calibrated_diagnostics(selected_exact_cal, pool)
    subgroup_robustness = _build_subgroup_robustness(selected_exact_cal, pool)
    sanity_checks = _run_sanity_checks(selected_exact_cal, pool)
    selected_exact_diag.to_csv(out_dir / "selected_exact_k_with_diagnostics.csv", index=False)
    selected_upto_diag.to_csv(out_dir / "selected_upto_k_with_diagnostics.csv", index=False)
    selected_exact_cal.to_csv(out_dir / "selected_exact_k_with_calibrated_diagnostics.csv", index=False)
    selected_upto_cal.to_csv(out_dir / "selected_upto_k_with_calibrated_diagnostics.csv", index=False)
    selected_exact_base.to_csv(out_dir / "selected_exact_k_base_variants_calibrated_diagnostics.csv", index=False)
    subgroup_robustness.to_csv(out_dir / "subgroup_robustness_selected_exact_k.csv", index=False)
    sanity_checks.to_csv(out_dir / "sanity_checks_permutation_selected_exact_k.csv", index=False)

    _plot_motion_vs_appearance_tradeoff(out_dir, selected_exact_cal)
    _plot_finalist_fit_scatter(out_dir, selected_exact_cal, pool)
    _plot_finalist_fit_scatter_cell_collapsed(out_dir, selected_exact_cal, pool)
    _plot_finalist_ranking_views(out_dir, selected_exact_cal, pool)
    _plot_finalist_fit_scatter_synthetic(out_dir, selected_exact_cal, pool)
    _plot_finalist_fit_scatter_synthetic_cell_collapsed(out_dir, selected_exact_cal, pool)
    _plot_finalist_fit_scatter_real_only(out_dir, selected_exact_cal, pool)
    _plot_finalist_fit_scatter_real_only_cell_collapsed(out_dir, selected_exact_cal, pool)
    _plot_effect_control_fit_scatter(out_dir, selected_exact_cal, pool)
    _plot_effect_control_fit_scatter_cell_collapsed(out_dir, selected_exact_cal, pool)
    largek_compare = _plot_best_largek_effects_vs_noeffects(out_dir, pool)
    largek_compare.to_csv(out_dir / "best_largek_effects_vs_noeffects_summary.csv", index=False)

    flags = _build_red_flags(agg, selected_exact_cal, min_runs=args.min_runs, mae_std_flag=args.mae_std_flag)
    _write_markdown_report(
        out_dir / "final_utility_sweep_report.md",
        run_roots=run_roots,
        agg=agg,
        selected_exact_diag=selected_exact_cal,
        selected_upto_diag=selected_upto_cal,
        selected_exact_base=selected_exact_base,
        largek_compare=largek_compare,
        subgroup_robustness=subgroup_robustness,
        sanity_checks=sanity_checks,
        red_flags=flags,
        selection_objective=selection_objective,
    )
    print(f"Wrote final utility sweep outputs to: {out_dir}")


if __name__ == "__main__":
    main()
