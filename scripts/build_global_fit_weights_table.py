#!/usr/bin/env python3
"""
Build a compact LaTeX table for global predictor weights and collinearity.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = (
    "analysis_comprehensive_runs/"
    "ridge_resid_weighted_ridge_a10_no_family_no_density_zscore_zeroshot_v3/"
    "density_joint/leakage_free_combo_flow_eps_raw_single__dino_rnorm_k5"
)
DEFAULT_OUT_DIR = "analysis/utility_guided_design_tables"

DISPLAY_NAMES = {
    "flow_eval_to_train_eps1p5px": r"$F_{E\to T}^{\mathrm{cov}@1.5px}$",
    "flow_train_to_eval_eps1px": r"$F_{T\to E}^{\mathrm{cov}@1.0px}$",
    "dino_eval_to_train_mean_dist": r"$A_{E\to T}^{\mathrm{dist}}$",
    "dino_train_to_eval_mean_dist": r"$A_{T\to E}^{\mathrm{dist}}$",
}

FAMILY = {
    "flow_eval_to_train_eps1p5px": "Flow",
    "flow_train_to_eval_eps1px": "Flow",
    "dino_eval_to_train_mean_dist": "Appearance",
    "dino_train_to_eval_mean_dist": "Appearance",
}


def _parse_coefficients(path: Path) -> Dict[str, float]:
    text = path.read_text()
    coeffs: Dict[str, float] = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z0-9_]+):\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*$", line.strip())
        if not m:
            continue
        key = m.group(1)
        if key in {"Rows", "Target", "Intercept", "R2", "Predictors"}:
            continue
        coeffs[key] = float(m.group(2))
    return coeffs


def _load_collinearity(path: Path) -> pd.DataFrame:
    corr = pd.read_csv(path, index_col=0)
    corr.index = corr.index.astype(str)
    corr.columns = corr.columns.astype(str)
    return corr


def _max_abs_corr(corr: pd.DataFrame, key: str) -> float:
    if key not in corr.index:
        return float("nan")
    row = pd.to_numeric(corr.loc[key], errors="coerce")
    row = row[row.index != key]
    col = pd.to_numeric(corr[key], errors="coerce") if key in corr.columns else pd.Series(dtype=float)
    col = col[col.index != key]
    both = pd.concat([row, col], axis=0)
    if both.empty:
        return float("nan")
    vals = np.abs(pd.to_numeric(both, errors="coerce").to_numpy(dtype=float))
    vals = vals[np.isfinite(vals)]
    if vals.size == 0 or not np.isfinite(vals).any():
        return float("nan")
    return float(np.nanmax(vals))


def _fmt(x: float, digits: int = 3) -> str:
    if not np.isfinite(x):
        return "--"
    return f"{x:.{digits}f}"


def build_table(df: pd.DataFrame) -> str:
    lines: List[str] = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Global standardized predictor weights} for the 4-predictor utility model (2 flow + 2 appearance). Coefficients are ridge weights in z-scored feature space; $\max |r|$ reports the largest absolute pairwise predictor correlation for each predictor.}",
        r"\label{tab:global_predictor_weights_4pred}",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\begin{tabular}{l l r r r}",
        r"\toprule",
        r"\textbf{Predictor} & \textbf{Type} & $\boldsymbol{\beta}_{\mathrm{std}}$ & $|\beta|$ rank & $\max |r|$ \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['predictor_tex']} & {r['family']} & {r['beta_std']:+.3f} & {int(r['abs_rank'])} & {_fmt(float(r['max_abs_corr']), 3)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coeff_path = run_dir / "regression_summary.txt"
    corr_path = run_dir / "predictor_colinearity_triangle.csv"
    coeffs = _parse_coefficients(coeff_path)
    corr = _load_collinearity(corr_path)

    keys = [
        "flow_eval_to_train_eps1p5px",
        "flow_train_to_eval_eps1px",
        "dino_eval_to_train_mean_dist",
        "dino_train_to_eval_mean_dist",
    ]
    missing = [k for k in keys if k not in coeffs]
    if missing:
        raise SystemExit(f"Missing expected coefficients: {missing}")

    rows = []
    for key in keys:
        rows.append(
            {
                "predictor_key": key,
                "predictor_tex": DISPLAY_NAMES.get(key, key),
                "family": FAMILY.get(key, "Other"),
                "beta_std": float(coeffs[key]),
                "max_abs_corr": _max_abs_corr(corr, key),
            }
        )
    df = pd.DataFrame(rows)
    df["abs_beta"] = np.abs(df["beta_std"])
    df["abs_rank"] = df["abs_beta"].rank(method="dense", ascending=False).astype(int)
    df = df.sort_values("abs_rank").reset_index(drop=True)

    csv_path = out_dir / "global_predictor_weights_4pred.csv"
    tex_path = out_dir / "global_predictor_weights_4pred.tex"
    df.to_csv(csv_path, index=False)
    tex_path.write_text(build_table(df))
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {tex_path}")


if __name__ == "__main__":
    main()
