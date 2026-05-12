#!/usr/bin/env python3
"""
Compute drop-one predictor sensitivity on existing LOBO rows for a single run.

This reuses the fold assignments already present in prediction_lobo_rows.csv and
fits ridge models per fold for:
  - full predictor set
  - one model dropping each predictor

Outputs:
  - dropone_sensitivity_lobo.csv
  - dropone_sensitivity_lobo.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = (
    "analysis_comprehensive_runs/"
    "ridge_resid_weighted_ridge_a10_no_family_no_density_zscore_zeroshot_v3/"
    "density_joint/leakage_free_combo_flow_eps_raw_single__dino_rnorm_k5"
)
DEFAULT_OUT_DIR = "analysis/utility_guided_design_tables"

PREDICTOR_DISPLAY = {
    "flow_eval_to_train_eps1p5px": r"$F_{E\to T}^{\mathrm{cov}@1.5px}$",
    "flow_train_to_eval_eps1px": r"$F_{T\to E}^{\mathrm{cov}@1.0px}$",
    "dino_eval_to_train_mean_dist": r"$A_{E\to T}^{\mathrm{dist}}$",
    "dino_train_to_eval_mean_dist": r"$A_{T\to E}^{\mathrm{dist}}$",
}


def _pairwise_cindex(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    n = 0
    c = 0.0
    m = y.shape[0]
    for i in range(m):
        yi = y[i]
        pi = p[i]
        if not (np.isfinite(yi) and np.isfinite(pi)):
            continue
        for j in range(i + 1, m):
            yj = y[j]
            pj = p[j]
            if not (np.isfinite(yj) and np.isfinite(pj)):
                continue
            if yi == yj:
                continue
            n += 1
            dy = yi - yj
            dp = pi - pj
            if dp == 0:
                c += 0.5
            elif dy * dp > 0:
                c += 1.0
    return float(c / n) if n > 0 else float("nan")


def _ridge_predict_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
    eps: float = 1e-12,
) -> np.ndarray:
    x_mean = np.nanmean(x_train, axis=0)
    x_std = np.nanstd(x_train, axis=0)
    x_std = np.where(np.isfinite(x_std) & (x_std > eps), x_std, 1.0)

    xtr = (x_train - x_mean) / x_std
    xte = (x_test - x_mean) / x_std
    y_mean = float(np.nanmean(y_train))
    yctr = y_train - y_mean

    p = xtr.shape[1]
    xtx = xtr.T @ xtr
    lhs = xtx + alpha * np.eye(p, dtype=float)
    rhs = xtr.T @ yctr
    coef = np.linalg.solve(lhs, rhs)
    return y_mean + (xte @ coef)


def _evaluate_rank_metrics(df: pd.DataFrame, pred_col: str) -> Dict[str, float]:
    cidx_vals: List[float] = []
    sp_vals: List[float] = []
    top1_vals: List[float] = []
    groups = df.groupby(["benchmark", "model_family_encoder"], dropna=False)
    for _, g in groups:
        y = pd.to_numeric(g["target"], errors="coerce")
        p = pd.to_numeric(g[pred_col], errors="coerce")
        if y.notna().sum() < 2 or p.notna().sum() < 2:
            continue
        cidx_vals.append(_pairwise_cindex(y.to_numpy(dtype=float), p.to_numpy(dtype=float)))
        sp_vals.append(float(y.corr(p, method="spearman")))
        true_best_idx = y.idxmax()
        pred_best_idx = p.idxmax()
        top1_vals.append(1.0 if true_best_idx == pred_best_idx else 0.0)
    return {
        "pairwise_cindex_macro": float(np.nanmean(cidx_vals)) if cidx_vals else float("nan"),
        "spearman_macro": float(np.nanmean(sp_vals)) if sp_vals else float("nan"),
        "top1_macro": float(np.nanmean(top1_vals)) if top1_vals else float("nan"),
        "n_groups": int(len(cidx_vals)),
    }


def _to_tex(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Drop-one sensitivity on LOBO held-out rows} for the selected 2F+2A run. Models are refit on the original LOBO fold splits; metrics are macro-averaged over $(\mathrm{benchmark}, \mathrm{model\_family\_encoder})$ groups.}",
        r"\label{tab:dropone_sensitivity_lobo}",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\begin{tabular}{l r r r}",
        r"\toprule",
        r"\textbf{Model} & \textbf{pairwise c-index} & $\boldsymbol{\Delta}$\textbf{ c-index} & \textbf{Spearman} \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        model_tex = str(r["model_tex"])
        cidx = f"{float(r['pairwise_cindex_macro']):.3f}"
        dc = f"{float(r['delta_cindex_vs_full']):+.3f}"
        sp = f"{float(r['spearman_macro']):.3f}"
        lines.append(f"{model_tex} & {cidx} & {dc} & {sp} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--prediction-source", default="lobo", choices=["lobo", "loto", "jointood"])
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = run_dir / "run_metadata.json"
    pred_path = run_dir / f"prediction_{args.prediction_source}_rows.csv"
    if not meta_path.exists():
        raise SystemExit(f"Missing metadata: {meta_path}")
    if not pred_path.exists():
        raise SystemExit(f"Missing prediction rows: {pred_path}")

    meta = json.loads(meta_path.read_text())
    predictors: List[str] = [str(x) for x in meta.get("predictors", [])]
    if len(predictors) < 2:
        raise SystemExit(f"Need at least 2 predictors, got: {predictors}")
    alpha = float(meta.get("ridge_alpha", 10.0))

    df = pd.read_csv(pred_path)
    required = ["fold", "target", "benchmark", "model_family_encoder"] + predictors
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    work = df[required].copy()
    for c in ["target"] + predictors:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=["fold", "target"] + predictors).reset_index(drop=True)
    if work.empty:
        raise SystemExit("No valid rows after numeric filtering.")

    folds = list(pd.unique(work["fold"].astype(str)))
    scenarios: List[Tuple[str, Sequence[str]]] = [("full", predictors)]
    for p in predictors:
        scenarios.append((f"drop:{p}", [x for x in predictors if x != p]))

    rows = []
    for name, use_predictors in scenarios:
        pred = np.full(work.shape[0], np.nan, dtype=float)
        for fold in folds:
            tr = work["fold"].astype(str) != fold
            te = ~tr
            x_train = work.loc[tr, use_predictors].to_numpy(dtype=float)
            y_train = work.loc[tr, "target"].to_numpy(dtype=float)
            x_test = work.loc[te, use_predictors].to_numpy(dtype=float)
            pred[te.to_numpy()] = _ridge_predict_fold(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                alpha=alpha,
            )
        eval_df = work.copy()
        eval_df["pred_refit"] = pred
        m = _evaluate_rank_metrics(eval_df, "pred_refit")
        dropped = "" if name == "full" else name.split("drop:", 1)[1]
        if name == "full":
            model_tex = r"\textbf{Full (2F+2A)}"
        else:
            model_tex = "Drop " + PREDICTOR_DISPLAY.get(dropped, dropped)
        rows.append(
            {
                "scenario": name,
                "dropped_predictor": dropped,
                "model_tex": model_tex,
                **m,
            }
        )

    out = pd.DataFrame(rows)
    full_cidx = float(out.loc[out["scenario"] == "full", "pairwise_cindex_macro"].iloc[0])
    out["delta_cindex_vs_full"] = out["pairwise_cindex_macro"] - full_cidx
    out["__ord"] = np.where(out["scenario"] == "full", -1, 0)
    out["__abs_delta"] = np.abs(out["delta_cindex_vs_full"])
    out = out.sort_values(by=["__ord", "__abs_delta"], ascending=[True, False]).reset_index(drop=True)
    out = out.drop(columns=["__ord", "__abs_delta"])

    csv_path = out_dir / "dropone_sensitivity_lobo.csv"
    tex_path = out_dir / "dropone_sensitivity_lobo.tex"
    out.to_csv(csv_path, index=False)
    tex_path.write_text(_to_tex(out))
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {tex_path}")


if __name__ == "__main__":
    main()
