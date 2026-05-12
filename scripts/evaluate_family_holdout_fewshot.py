#!/usr/bin/env python3
"""Evaluate leave-one-family-out utility ranking with few-shot calibration curves.

This script is designed to run on top of final sweep artifacts:
- candidate_pool_per_run.csv
- selected_* CSV containing `signature`

For each selected signature:
1) Refit a utility model from dataset features -> target (ridge/ols),
   training on all families except one held-out family.
2) Evaluate uncalibrated ranking metrics on held-out family.
3) Fit few-shot affine calibration y = a*u + b using K sampled points from
   the held-out family, then evaluate ranking metrics on the remaining points.

Outputs:
- family_holdout_uncalibrated.csv
- family_holdout_fewshot_curve_per_family.csv
- family_holdout_fewshot_curve_summary.csv
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TOKEN_EQUIV_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"^flow_train_to_eval_auc$"), "flow_train_to_eval_quantile"),
    (re.compile(r"^flow_eval_to_train_auc$"), "flow_eval_to_train_quantile"),
    (re.compile(r"^flow_train_to_eval_eps_at\d+$"), "flow_train_to_eval_quantile"),
    (re.compile(r"^flow_eval_to_train_eps_at\d+$"), "flow_eval_to_train_quantile"),
]


def _as_float(v: object) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _canonical_token(token: str) -> str:
    t = token.strip()
    for pat, rep in TOKEN_EQUIV_PATTERNS:
        if pat.match(t):
            return rep
    return t


def _token_list_from_signature(signature: str) -> List[str]:
    if not signature:
        return []
    raw = [t.strip() for t in str(signature).split("|") if t.strip()]
    return [_canonical_token(t) for t in raw]


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


def _fit_linear(
    train_df: pd.DataFrame,
    predictors: Sequence[str],
    target_col: str,
    model: str,
    ridge_alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = train_df[list(predictors)].to_numpy(dtype=float)
    y = train_df[target_col].to_numpy(dtype=float)

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0.0] = 1.0
    xz = (x - mean) / std

    xz_i = np.column_stack([np.ones(len(xz)), xz])
    if model == "ridge":
        alpha = float(ridge_alpha)
        penalty = np.eye(xz_i.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        coef = np.linalg.solve(xz_i.T @ xz_i + alpha * penalty, xz_i.T @ y)
    elif model == "ols":
        coef, _, _, _ = np.linalg.lstsq(xz_i, y, rcond=None)
    else:
        raise ValueError(f"Unsupported model: {model}")
    return coef, mean, std


def _predict_linear(
    df: pd.DataFrame,
    predictors: Sequence[str],
    coef: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    x = df[list(predictors)].to_numpy(dtype=float)
    xz = (x - mean) / std
    xz_i = np.column_stack([np.ones(len(xz)), xz])
    return xz_i @ coef


def _fit_affine(u: pd.Series, y: pd.Series) -> Tuple[float, float]:
    uu = pd.to_numeric(u, errors="coerce")
    yy = pd.to_numeric(y, errors="coerce")
    m = uu.notna() & yy.notna()
    uu = uu[m]
    yy = yy[m]
    if len(uu) == 0:
        return math.nan, math.nan
    if uu.nunique() < 2:
        return 0.0, float(yy.mean())
    try:
        slope, intercept = np.polyfit(uu.to_numpy(dtype=float), yy.to_numpy(dtype=float), 1)
        if np.isfinite(slope) and np.isfinite(intercept):
            return float(slope), float(intercept)
    except Exception:
        pass
    return math.nan, math.nan


def _rank_metrics_per_benchmark(
    df: pd.DataFrame,
    train_col: str,
    benchmark_col: str,
    target_col: str,
    pred_col: str,
    min_options_per_benchmark: int,
) -> Dict[str, float]:
    required = [train_col, benchmark_col, target_col, pred_col]
    work = df.dropna(subset=required).copy()
    if work.empty:
        return {
            "n_benchmarks": 0.0,
            "rank_spearman": math.nan,
            "rank_kendall_tau": math.nan,
            "rank_pairwise_cindex": math.nan,
            "regret_at1": math.nan,
            "regret_at3": math.nan,
            "hitrate_at1": math.nan,
            "hitrate_at3": math.nan,
        }

    rows: List[Dict[str, float]] = []
    for _, sub in work.groupby(benchmark_col, dropna=False):
        agg = (
            sub.groupby(train_col, dropna=False)
            .agg(true_mean=(target_col, "mean"), pred_mean=(pred_col, "mean"))
            .reset_index(drop=False)
        )
        n_opt = int(len(agg))
        if n_opt < max(2, int(min_options_per_benchmark)):
            continue
        true = agg["true_mean"].to_numpy(dtype=float)
        pred = agg["pred_mean"].to_numpy(dtype=float)

        spearman = float(pd.Series(pred).corr(pd.Series(true), method="spearman"))
        kendall = _kendall_tau_b(pred, true)
        cidx = _pairwise_cindex(true, pred)

        true_best = float(np.max(true))
        agg_pred_sort = agg.sort_values("pred_mean", ascending=False).reset_index(drop=True)
        agg_true_sort = agg.sort_values("true_mean", ascending=False).reset_index(drop=True)

        k1 = 1
        k3 = min(3, n_opt)

        pred_top1 = agg_pred_sort.head(k1)
        pred_top3 = agg_pred_sort.head(k3)
        true_top1 = set(agg_true_sort.head(k1)[train_col].astype(str))
        true_top3 = set(agg_true_sort.head(k3)[train_col].astype(str))

        pred_top1_set = set(pred_top1[train_col].astype(str))
        pred_top3_set = set(pred_top3[train_col].astype(str))

        regret1 = true_best - float(pred_top1["true_mean"].max())
        regret3 = true_best - float(pred_top3["true_mean"].max())
        hit1 = float(len(true_top1 & pred_top1_set)) / float(k1)
        hit3 = float(len(true_top3 & pred_top3_set)) / float(k3)

        rows.append(
            {
                "rank_spearman": spearman,
                "rank_kendall_tau": kendall,
                "rank_pairwise_cindex": cidx,
                "regret_at1": regret1,
                "regret_at3": regret3,
                "hitrate_at1": hit1,
                "hitrate_at3": hit3,
            }
        )

    if not rows:
        return {
            "n_benchmarks": 0.0,
            "rank_spearman": math.nan,
            "rank_kendall_tau": math.nan,
            "rank_pairwise_cindex": math.nan,
            "regret_at1": math.nan,
            "regret_at3": math.nan,
            "hitrate_at1": math.nan,
            "hitrate_at3": math.nan,
        }

    out = pd.DataFrame(rows)
    return {
        "n_benchmarks": float(len(out)),
        "rank_spearman": float(pd.to_numeric(out["rank_spearman"], errors="coerce").mean()),
        "rank_kendall_tau": float(pd.to_numeric(out["rank_kendall_tau"], errors="coerce").mean()),
        "rank_pairwise_cindex": float(pd.to_numeric(out["rank_pairwise_cindex"], errors="coerce").mean()),
        "regret_at1": float(pd.to_numeric(out["regret_at1"], errors="coerce").mean()),
        "regret_at3": float(pd.to_numeric(out["regret_at3"], errors="coerce").mean()),
        "hitrate_at1": float(pd.to_numeric(out["hitrate_at1"], errors="coerce").mean()),
        "hitrate_at3": float(pd.to_numeric(out["hitrate_at3"], errors="coerce").mean()),
    }


def _ensure_family_col(df: pd.DataFrame, family_col: str) -> pd.DataFrame:
    if family_col in df.columns:
        return df
    out = df.copy()
    if "model_family_encoder" in out.columns:
        out[family_col] = out["model_family_encoder"].astype(str)
        return out
    if "model_family" in out.columns and "encoder_config" in out.columns:
        out[family_col] = out["model_family"].astype(str) + "_" + out["encoder_config"].astype(str)
        return out
    if "model_family" in out.columns:
        out[family_col] = out["model_family"].astype(str)
        return out
    out[family_col] = "unknown"
    return out


def _candidate_cols_for_quantile(token: str, cols: Sequence[str]) -> List[str]:
    cset = set(cols)
    if token == "flow_train_to_eval_quantile":
        pref = [
            "flow_train_to_eval_auc_at95",
            "flow_train_to_eval_auc_at50",
            "flow_train_to_eval_auc",
            "flow_train_to_eval_eps_at50",
            "flow_train_to_eval_eps_at95",
        ]
        out = [c for c in pref if c in cset]
        if out:
            return out
        pat = re.compile(r"^flow_train_to_eval_(auc|auc_at\d+|eps_at\d+)$")
        return [c for c in cols if pat.match(c)]
    if token == "flow_eval_to_train_quantile":
        pref = [
            "flow_eval_to_train_auc_at95",
            "flow_eval_to_train_auc_at50",
            "flow_eval_to_train_auc",
            "flow_eval_to_train_eps_at50",
            "flow_eval_to_train_eps_at95",
        ]
        out = [c for c in pref if c in cset]
        if out:
            return out
        pat = re.compile(r"^flow_eval_to_train_(auc|auc_at\d+|eps_at\d+)$")
        return [c for c in cols if pat.match(c)]
    return []


def _resolve_predictors_from_tokens(tokens: Sequence[str], df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    resolved: List[str] = []
    missing: List[str] = []
    cols = list(df.columns)
    cset = set(cols)

    for tok in tokens:
        if tok in cset:
            resolved.append(tok)
            continue
        cands = _candidate_cols_for_quantile(tok, cols)
        if cands:
            # Choose the candidate with largest non-null support.
            best = sorted(
                cands,
                key=lambda c: int(pd.to_numeric(df[c], errors="coerce").notna().sum()),
                reverse=True,
            )[0]
            resolved.append(best)
            continue
        missing.append(tok)

    # Preserve order while dropping duplicates.
    seen = set()
    uniq: List[str] = []
    for c in resolved:
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)
    return uniq, missing


def _resolve_target_column(
    df: pd.DataFrame,
    requested_target_col: str,
) -> Optional[str]:
    candidates = [
        str(requested_target_col),
        "auc_normalized_observed",
        "target",
        "auc_normalized",
        "peak_pck",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _load_signature_rows(
    signature: str,
    pool: pd.DataFrame,
    train_col: str,
    benchmark_col: str,
    family_col: str,
    target_col: str,
    rows_filename: str,
) -> pd.DataFrame:
    sub = pool[pool["signature"].astype(str) == str(signature)].copy()
    if sub.empty:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for _, r in sub.iterrows():
        p = Path(str(r.get("path", "")))
        rows_path = p / str(rows_filename)
        if not rows_path.exists():
            continue
        try:
            df = pd.read_csv(rows_path)
        except Exception:
            continue
        if df.empty:
            continue
        df = _ensure_family_col(df, family_col)
        tgt = _resolve_target_column(df, target_col)
        if tgt is None:
            continue
        if tgt != target_col:
            df = df.rename(columns={tgt: target_col})
        keep = [c for c in [train_col, benchmark_col, family_col, target_col] if c in df.columns]
        if len(keep) < 4:
            continue
        # Keep full columns for later feature resolution.
        df["__variant__"] = str(r.get("variant", ""))
        df["__method_path__"] = str(p)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _evaluate_signature_family_holdout(
    sig_row: pd.Series,
    pool: pd.DataFrame,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    signature = str(sig_row["signature"])
    lane = str(sig_row.get("lane", ""))
    signal_k = _as_float(sig_row.get("signal_k"))
    k_target = _as_float(sig_row.get("k_target"))

    raw = _load_signature_rows(
        signature=signature,
        pool=pool,
        train_col=args.train_col,
        benchmark_col=args.benchmark_col,
        family_col=args.family_col,
        target_col=args.target_col,
        rows_filename=args.rows_filename,
    )
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    tokens = _token_list_from_signature(signature)
    predictors, missing_tokens = _resolve_predictors_from_tokens(tokens, raw)
    if args.verbose and missing_tokens:
        print(f"[{signature}] missing tokens: {','.join(missing_tokens)}")
    if len(predictors) == 0:
        if args.verbose:
            print(f"[{signature}] no usable predictors after token resolution")
        return pd.DataFrame(), pd.DataFrame()

    keep = [args.train_col, args.benchmark_col, args.family_col, args.target_col] + predictors
    work = raw[keep].copy()
    for c in [args.target_col] + predictors:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=[args.train_col, args.benchmark_col, args.family_col, args.target_col] + predictors).copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()

    agg = (
        work.groupby([args.train_col, args.benchmark_col, args.family_col], dropna=False)[predictors + [args.target_col]]
        .mean()
        .reset_index(drop=False)
    )
    if len(agg) < max(10, int(args.min_family_rows)):
        return pd.DataFrame(), pd.DataFrame()

    families = sorted(set(agg[args.family_col].astype(str).dropna().unique()))
    if len(families) < 2:
        return pd.DataFrame(), pd.DataFrame()

    k_values = sorted(set(int(x) for x in str(args.k_values).split(",") if x.strip()))

    uncal_rows: List[Dict[str, object]] = []
    curve_rows: List[Dict[str, object]] = []

    for fam in families:
        fam_s = agg[args.family_col].astype(str)
        train_df = agg[fam_s != fam].copy()
        test_df = agg[fam_s == fam].copy()

        if len(train_df) < max(int(args.min_train_rows), len(predictors) + 2):
            continue
        if len(test_df) < int(args.min_family_rows):
            continue

        try:
            coef, mean, std = _fit_linear(
                train_df=train_df,
                predictors=predictors,
                target_col=args.target_col,
                model=args.model,
                ridge_alpha=float(args.ridge_alpha),
            )
        except Exception:
            continue

        test_df = test_df.copy()
        test_df["utility_u"] = _predict_linear(test_df, predictors, coef, mean, std)

        uncal = _rank_metrics_per_benchmark(
            df=test_df,
            train_col=args.train_col,
            benchmark_col=args.benchmark_col,
            target_col=args.target_col,
            pred_col="utility_u",
            min_options_per_benchmark=int(args.min_options_per_benchmark),
        )
        if int(uncal.get("n_benchmarks", 0.0)) <= 0:
            continue

        base_row: Dict[str, object] = {
            "signature": signature,
            "lane": lane,
            "signal_k": int(signal_k) if np.isfinite(signal_k) else math.nan,
            "k_target": int(k_target) if np.isfinite(k_target) else math.nan,
            "family": fam,
            "n_predictors": int(len(predictors)),
            "predictors": ",".join(predictors),
            "n_train_rows": int(len(train_df)),
            "n_test_rows": int(len(test_df)),
            "n_cal_repeats": 0,
            "k_calibration": 0,
        }
        base_row.update(uncal)
        uncal_rows.append(base_row)

        # Include K=0 in curve output.
        k0 = dict(base_row)
        k0["n_cal_repeats"] = 1
        for m in ["rank_spearman", "rank_kendall_tau", "rank_pairwise_cindex", "regret_at1", "regret_at3", "hitrate_at1", "hitrate_at3"]:
            k0[f"{m}_mean"] = _as_float(uncal.get(m))
            k0[f"{m}_std"] = 0.0
        curve_rows.append(k0)

        test_idx = test_df.index.to_numpy(dtype=int)
        for k in k_values:
            if k <= 0:
                continue
            k_eff = min(int(k), max(len(test_df) - 1, 0))
            if k_eff <= 0:
                continue

            rep_metrics: List[Dict[str, float]] = []
            for _ in range(int(args.n_repeats)):
                cal_idx = rng.choice(test_idx, size=k_eff, replace=False)
                cal_df = test_df.loc[cal_idx].copy()
                eval_df = test_df.drop(index=cal_idx).copy()
                if len(eval_df) < 2:
                    continue

                a, b = _fit_affine(cal_df["utility_u"], cal_df[args.target_col])
                if not np.isfinite(a) or not np.isfinite(b):
                    continue

                eval_df["yhat_cal"] = a * eval_df["utility_u"] + b
                mm = _rank_metrics_per_benchmark(
                    df=eval_df,
                    train_col=args.train_col,
                    benchmark_col=args.benchmark_col,
                    target_col=args.target_col,
                    pred_col="yhat_cal",
                    min_options_per_benchmark=int(args.min_options_per_benchmark),
                )
                if int(mm.get("n_benchmarks", 0.0)) <= 0:
                    continue
                rep_metrics.append(mm)

            if not rep_metrics:
                continue
            rep_df = pd.DataFrame(rep_metrics)

            crow: Dict[str, object] = {
                "signature": signature,
                "lane": lane,
                "signal_k": int(signal_k) if np.isfinite(signal_k) else math.nan,
                "k_target": int(k_target) if np.isfinite(k_target) else math.nan,
                "family": fam,
                "n_predictors": int(len(predictors)),
                "predictors": ",".join(predictors),
                "n_train_rows": int(len(train_df)),
                "n_test_rows": int(len(test_df)),
                "k_calibration": int(k),
                "k_calibration_effective": int(k_eff),
                "n_cal_repeats": int(len(rep_df)),
                "n_benchmarks_mean": float(pd.to_numeric(rep_df["n_benchmarks"], errors="coerce").mean()),
            }
            for m in [
                "rank_spearman",
                "rank_kendall_tau",
                "rank_pairwise_cindex",
                "regret_at1",
                "regret_at3",
                "hitrate_at1",
                "hitrate_at3",
            ]:
                vals = pd.to_numeric(rep_df[m], errors="coerce")
                crow[f"{m}_mean"] = float(vals.mean())
                crow[f"{m}_std"] = float(vals.std(ddof=0))
            curve_rows.append(crow)

    return pd.DataFrame(uncal_rows), pd.DataFrame(curve_rows)


def _aggregate_curve(curve_df: pd.DataFrame) -> pd.DataFrame:
    if curve_df.empty:
        return pd.DataFrame()

    metrics = [
        "rank_spearman",
        "rank_kendall_tau",
        "rank_pairwise_cindex",
        "regret_at1",
        "regret_at3",
        "hitrate_at1",
        "hitrate_at3",
    ]

    group_cols = [c for c in ["signature", "lane", "signal_k", "k_target", "k_calibration"] if c in curve_df.columns]

    agg_rows: List[Dict[str, object]] = []
    for key, sub in curve_df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        out: Dict[str, object] = {group_cols[i]: key[i] for i in range(len(group_cols))}
        out["n_families"] = int(sub["family"].nunique()) if "family" in sub.columns else int(len(sub))
        out["n_rows"] = int(len(sub))
        for m in metrics:
            col = f"{m}_mean"
            if col not in sub.columns:
                continue
            vals = pd.to_numeric(sub[col], errors="coerce")
            out[f"{m}_across_family_mean"] = float(vals.mean())
            out[f"{m}_across_family_std"] = float(vals.std(ddof=0))
        agg_rows.append(out)

    out_df = pd.DataFrame(agg_rows)
    sort_cols = [c for c in ["lane", "signature", "k_calibration"] if c in out_df.columns]
    if sort_cols:
        out_df = out_df.sort_values(sort_cols)
    return out_df


def main() -> None:
    ap = argparse.ArgumentParser(description="Family-holdout + few-shot calibration evaluator.")
    ap.add_argument("--selected-csv", required=True, help="Selected signatures CSV (must contain signature).")
    ap.add_argument("--candidate-pool-csv", required=True, help="candidate_pool_per_run.csv")
    ap.add_argument("--output-dir", default="", help="Output dir (default: selected CSV parent).")
    ap.add_argument(
        "--rows-filename",
        default="auc_with_features.csv",
        help=(
            "Per-method row table filename under each method path "
            "(default: auc_with_features.csv; alternative: prediction_jointood_rows.csv)"
        ),
    )

    ap.add_argument("--model", choices=["ridge", "ols"], default="ridge")
    ap.add_argument("--ridge-alpha", type=float, default=1.0)

    ap.add_argument("--train-col", default="train_dataset")
    ap.add_argument("--benchmark-col", default="benchmark")
    ap.add_argument("--family-col", default="model_family")
    ap.add_argument(
        "--target-col",
        default="auc_normalized_observed",
        help="Primary target column. Falls back to auc_normalized_observed/target/auc_normalized/peak_pck if missing.",
    )

    ap.add_argument("--k-values", default="0,2,4,8,16")
    ap.add_argument("--n-repeats", type=int, default=80)
    ap.add_argument("--min-train-rows", type=int, default=40)
    ap.add_argument("--min-family-rows", type=int, default=20)
    ap.add_argument("--min-options-per-benchmark", type=int, default=2)
    ap.add_argument("--max-signatures", type=int, default=0, help="Optional cap for debugging.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    selected_path = Path(args.selected_csv)
    pool_path = Path(args.candidate_pool_csv)
    if not selected_path.exists():
        raise SystemExit(f"Missing selected CSV: {selected_path}")
    if not pool_path.exists():
        raise SystemExit(f"Missing candidate pool CSV: {pool_path}")

    out_dir = Path(args.output_dir) if args.output_dir else selected_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(selected_path)
    pool = pd.read_csv(pool_path)

    if "signature" not in selected.columns:
        raise SystemExit("selected CSV must include column: signature")
    if "signature" not in pool.columns or "path" not in pool.columns:
        raise SystemExit("candidate pool CSV must include columns: signature,path")

    sig_cols = [c for c in ["signature", "lane", "signal_k", "k_target"] if c in selected.columns]
    sig_meta = selected[sig_cols].drop_duplicates(subset=["signature"]).reset_index(drop=True)
    if int(args.max_signatures) > 0:
        sig_meta = sig_meta.head(int(args.max_signatures)).copy()

    rng = np.random.default_rng(int(args.seed))

    uncal_all: List[pd.DataFrame] = []
    curve_all: List[pd.DataFrame] = []
    for _, srow in sig_meta.iterrows():
        if args.verbose:
            print(f"Evaluating signature: {srow['signature']}")
        uncal_df, curve_df = _evaluate_signature_family_holdout(srow, pool, args, rng)
        if not uncal_df.empty:
            uncal_all.append(uncal_df)
        if not curve_df.empty:
            curve_all.append(curve_df)

    uncal = pd.concat(uncal_all, ignore_index=True) if uncal_all else pd.DataFrame()
    curve = pd.concat(curve_all, ignore_index=True) if curve_all else pd.DataFrame()
    summary = _aggregate_curve(curve)

    uncal_out = out_dir / "family_holdout_uncalibrated.csv"
    curve_out = out_dir / "family_holdout_fewshot_curve_per_family.csv"
    summary_out = out_dir / "family_holdout_fewshot_curve_summary.csv"

    uncal.to_csv(uncal_out, index=False)
    curve.to_csv(curve_out, index=False)
    summary.to_csv(summary_out, index=False)

    print(f"Wrote: {uncal_out}")
    print(f"Wrote: {curve_out}")
    print(f"Wrote: {summary_out}")


if __name__ == "__main__":
    main()
