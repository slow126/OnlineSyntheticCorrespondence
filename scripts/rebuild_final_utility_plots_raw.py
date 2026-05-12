#!/usr/bin/env python3
"""
Rebuild final utility plots from raw auc_with_features rows (non-collapsed),
with optional cell-collapsed companions for comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import build_heldout_model_cv as hcv


def _load_metadata(path: Path) -> Dict[str, object]:
    meta_path = path / "run_metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {}


def _fit_predict_jointood_raw(
    method_path: Path,
    seed: int,
    max_pairs_per_group: int,
    pairwise_max_iter: int,
    pairwise_lr: float,
) -> pd.DataFrame:
    meta = _load_metadata(method_path)
    predictors = list(meta.get("predictors", []))
    if not predictors:
        raise ValueError(f"No predictors in run_metadata: {method_path}")

    target_col = str(meta.get("target") or "auc_normalized_observed")
    model = str(meta.get("prediction_model") or meta.get("linear_model") or "ols")
    if model not in {"ols", "ridge", "pairwise_rank"}:
        model = "ols"
    ridge_alpha = float(meta.get("ridge_alpha") or 1.0)
    standardize = bool(meta.get("standardize", True))
    option_col = str(meta.get("ranking_group") or "train_dataset")

    raw_path = method_path / "auc_with_features.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw rows: {raw_path}")
    df = pd.read_csv(raw_path)
    if target_col not in df.columns and "auc_normalized_observed" in df.columns:
        target_col = "auc_normalized_observed"

    req = predictors + [target_col, "train_dataset", "benchmark"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns for {method_path.name}: {miss}")

    work = df.copy()
    for c in predictors + [target_col]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=req).copy()
    if work.empty:
        raise ValueError(f"No valid rows after filtering: {method_path}")

    pred_rows: List[pd.DataFrame] = []
    fold_idx = 0
    train_groups = sorted(work["train_dataset"].dropna().astype(str).unique())
    benchmark_groups = sorted(work["benchmark"].dropna().astype(str).unique())
    for train_group in train_groups:
        for benchmark in benchmark_groups:
            fold_idx += 1
            te = work[
                (work["train_dataset"].astype(str) == train_group)
                & (work["benchmark"].astype(str) == benchmark)
            ].copy()
            tr = work[
                (work["train_dataset"].astype(str) != train_group)
                & (work["benchmark"].astype(str) != benchmark)
            ].copy()
            if tr.empty or te.empty:
                continue
            if len(tr) <= len(predictors) + 1:
                continue

            if model in {"ols", "ridge"}:
                coef, mean, std = hcv._fit_linear(
                    train_df=tr,
                    predictors=predictors,
                    target_col=target_col,
                    model=model,
                    ridge_alpha=ridge_alpha,
                    standardize=standardize,
                )
                y_pred = hcv._predict_linear(
                    test_df=te,
                    predictors=predictors,
                    coef=coef,
                    mean=mean,
                    std=std,
                    standardize=standardize,
                )
            else:
                if option_col not in tr.columns:
                    option_col = "train_dataset"
                coef, mean, std = hcv._fit_pairwise_rank(
                    train_df=tr,
                    predictors=predictors,
                    target_col=target_col,
                    group_col="benchmark",
                    option_col=option_col,
                    ridge_alpha=ridge_alpha,
                    max_pairs_per_group=max_pairs_per_group,
                    max_iter=pairwise_max_iter,
                    lr=pairwise_lr,
                    seed=seed + fold_idx,
                    standardize=standardize,
                )
                y_pred = hcv._predict_pairwise_rank(
                    test_df=te,
                    predictors=predictors,
                    coef=coef,
                    mean=mean,
                    std=std,
                    standardize=standardize,
                )

            out = te.copy()
            out["target"] = out[target_col].astype(float)
            out["prediction"] = y_pred.astype(float)
            out["joint_holdout"] = f"{train_group}__{benchmark}"
            pred_rows.append(out)

    if not pred_rows:
        raise ValueError(f"No scored folds for: {method_path}")
    pred_df = pd.concat(pred_rows, ignore_index=True)
    return pred_df


def _collapse_cells(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["train_dataset", "benchmark"], dropna=False)
        .agg(prediction=("prediction", "mean"), target=("target", "mean"), n_rows=("target", "size"))
        .reset_index(drop=False)
    )


def _is_synthetic(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for c in ("train_dataset", "benchmark"):
        if c in df.columns:
            mask = mask | df[c].astype(str).str.lower().str.contains("synthetic", na=False)
    return mask


def _metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"n": 0.0, "mae": math.nan, "rmse": math.nan, "pearson": math.nan, "spearman": math.nan}
    y = pd.to_numeric(df["target"], errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(df["prediction"], errors="coerce").to_numpy(dtype=float)
    mae, rmse = hcv._mae_rmse(y, p)
    pear = hcv._pearson_corr(y, p)
    spear = hcv._spearman_corr(y, p)
    return {"n": float(len(df)), "mae": mae, "rmse": rmse, "pearson": pear, "spearman": spear}


def _scatter_plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if df.empty:
        return
    x = pd.to_numeric(df["prediction"], errors="coerce")
    y = pd.to_numeric(df["target"], errors="coerce")
    m = x.notna() & y.notna()
    x = x[m]
    y = y[m]
    if x.empty:
        return
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = 0.05 * (hi - lo + 1e-8)
    x0, x1 = lo - pad, hi + pad

    met = _metrics(pd.DataFrame({"prediction": x, "target": y}))
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.scatter(x, y, s=14, alpha=0.35, color="#1f77b4")
    ax.plot([x0, x1], [x0, x1], "k--", linewidth=1.0)
    ax.set_xlim(x0, x1)
    ax.set_ylim(x0, x1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title(
        f"{title}\nN={int(met['n'])} MAE={met['mae']:.2f} RMSE={met['rmse']:.2f} "
        f"Pearson={met['pearson']:.2f} Spearman={met['spearman']:.2f}"
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _pick_representative_paths(selected: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, s in selected.iterrows():
        sig = str(s["signature"])
        sub = pool[pool["signature"] == sig].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["jointood_mae", "jointood_spearman"], ascending=[True, False])
        b = sub.iloc[0]
        rows.append(
            {
                "lane": str(s["lane"]),
                "k_target": int(s["k_target"]),
                "signature": sig,
                "path": str(b["path"]),
                "variant": str(b["variant"]),
                "method": str(b["method"]),
                "jointood_mae": float(b["jointood_mae"]),
                "jointood_spearman": float(b["jointood_spearman"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild final utility fit plots from raw rows.")
    parser.add_argument("--sweep-dir", default="analysis_comprehensive_runs/final_utility_sweep_v1")
    parser.add_argument("--output-dir", default=None, help="Default: <sweep-dir>/raw_refit_plots")
    parser.add_argument("--max-pairs-per-group", type=int, default=2000)
    parser.add_argument("--pairwise-max-iter", type=int, default=150)
    parser.add_argument("--pairwise-lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_dir = Path(args.output_dir) if args.output_dir else (sweep_dir / "raw_refit_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    sel_path = sweep_dir / "selected_exact_k_with_calibrated_diagnostics.csv"
    pool_path = sweep_dir / "candidate_pool_per_run.csv"
    if not sel_path.exists() or not pool_path.exists():
        raise SystemExit("Missing selected_exact_k_with_calibrated_diagnostics.csv or candidate_pool_per_run.csv")
    selected = pd.read_csv(sel_path)
    pool = pd.read_csv(pool_path)

    picks = _pick_representative_paths(selected, pool)
    if picks.empty:
        raise SystemExit("No representative methods picked.")
    picks.to_csv(out_dir / "selected_exact_k_representative_methods.csv", index=False)

    metric_rows: List[Dict[str, object]] = []
    for _, r in picks.sort_values(["lane", "k_target"]).iterrows():
        lane = str(r["lane"])
        k = int(r["k_target"])
        method = str(r["method"])
        path = Path(str(r["path"]))
        print(f"refit raw: lane={lane} k={k} method={method}")
        try:
            pred = _fit_predict_jointood_raw(
                method_path=path,
                seed=int(args.seed),
                max_pairs_per_group=int(args.max_pairs_per_group),
                pairwise_max_iter=int(args.pairwise_max_iter),
                pairwise_lr=float(args.pairwise_lr),
            )
        except Exception as exc:
            print(f"  skip ({exc})")
            continue

        pred.to_csv(out_dir / f"{lane}_k{k}_raw_predictions.csv", index=False)
        cell = _collapse_cells(pred)
        cell.to_csv(out_dir / f"{lane}_k{k}_cell_collapsed_predictions.csv", index=False)
        syn_mask = _is_synthetic(pred)
        pred_syn = pred[syn_mask].copy()
        cell_syn = _collapse_cells(pred_syn) if not pred_syn.empty else pd.DataFrame()

        _scatter_plot(
            pred,
            out_dir / "fit_scatter_raw_exact_k" / f"{lane}_k{k}_raw_actual_vs_pred.png",
            f"{lane} k={k} raw refit ({method})",
        )
        _scatter_plot(
            cell,
            out_dir / "fit_scatter_raw_cell_collapsed_exact_k" / f"{lane}_k{k}_raw_cell_collapsed_actual_vs_pred.png",
            f"{lane} k={k} raw refit cell-collapsed ({method})",
        )
        _scatter_plot(
            pred_syn,
            out_dir / "fit_scatter_raw_synthetic_exact_k" / f"{lane}_k{k}_raw_synthetic_actual_vs_pred.png",
            f"{lane} k={k} raw refit synthetic ({method})",
        )
        _scatter_plot(
            cell_syn,
            out_dir / "fit_scatter_raw_synthetic_cell_collapsed_exact_k" / f"{lane}_k{k}_raw_synthetic_cell_collapsed_actual_vs_pred.png",
            f"{lane} k={k} raw refit synthetic cell-collapsed ({method})",
        )

        m_raw = _metrics(pred)
        m_cell = _metrics(cell)
        m_syn = _metrics(pred_syn)
        m_syn_cell = _metrics(cell_syn)
        metric_rows.append(
            {
                "lane": lane,
                "k_target": k,
                "method": method,
                "variant": str(r["variant"]),
                "path": str(path),
                "raw_n": m_raw["n"],
                "raw_mae": m_raw["mae"],
                "raw_rmse": m_raw["rmse"],
                "raw_pearson": m_raw["pearson"],
                "raw_spearman": m_raw["spearman"],
                "cell_n": m_cell["n"],
                "cell_mae": m_cell["mae"],
                "cell_rmse": m_cell["rmse"],
                "cell_pearson": m_cell["pearson"],
                "cell_spearman": m_cell["spearman"],
                "syn_raw_n": m_syn["n"],
                "syn_raw_mae": m_syn["mae"],
                "syn_raw_spearman": m_syn["spearman"],
                "syn_cell_n": m_syn_cell["n"],
                "syn_cell_mae": m_syn_cell["mae"],
                "syn_cell_spearman": m_syn_cell["spearman"],
            }
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["lane", "k_target"])
    metrics_df.to_csv(out_dir / "raw_refit_plot_metrics.csv", index=False)

    report_lines = [
        "# Raw Refit Plot Rebuild",
        "",
        f"- sweep_dir: `{sweep_dir}`",
        f"- output_dir: `{out_dir}`",
        "- row source: `auc_with_features.csv`",
        "- CV protocol: leave-one-(train_dataset, benchmark)-out",
        "",
        "## Plot dirs",
        f"- `{out_dir / 'fit_scatter_raw_exact_k'}`",
        f"- `{out_dir / 'fit_scatter_raw_cell_collapsed_exact_k'}`",
        f"- `{out_dir / 'fit_scatter_raw_synthetic_exact_k'}`",
        f"- `{out_dir / 'fit_scatter_raw_synthetic_cell_collapsed_exact_k'}`",
        "",
        "## Metrics",
        "",
    ]
    if not metrics_df.empty:
        report_lines.append(
            metrics_df[
                [
                    "lane",
                    "k_target",
                    "method",
                    "raw_n",
                    "raw_mae",
                    "raw_spearman",
                    "cell_n",
                    "cell_mae",
                    "cell_spearman",
                    "syn_raw_n",
                    "syn_raw_mae",
                    "syn_raw_spearman",
                ]
            ].to_markdown(index=False)
        )
    else:
        report_lines.append("- no successful refits")
    (out_dir / "raw_refit_plot_report.md").write_text("\n".join(report_lines))
    print(f"Wrote raw refit plots to: {out_dir}")


if __name__ == "__main__":
    main()
