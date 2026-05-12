#!/usr/bin/env python3
"""
Build ranking-first paper tables from method_summary.csv.

Default protocol is LOTO and default metrics prioritize ranking fidelity:
- rank_spearman (primary, higher is better)
- rank_pairwise_cindex (secondary, higher is better)
- spearman (target-level monotonicity, higher is better)
- mae / rmse (calibration error, lower is better)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def _to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _select_cols(df: pd.DataFrame, scope: str) -> List[str]:
    base = ["method", "family", "symmetry", "notes", "n_predictors", "n_predictors_base", "target", "model"]
    metrics = [
        f"{scope}_mae",
        f"{scope}_rmse",
        f"{scope}_spearman",
        f"{scope}_rank_spearman",
        f"{scope}_rank_pairwise_cindex",
        f"{scope}_rank_kendall_tau",
        f"{scope}_regret",
    ]
    return [c for c in base + metrics if c in df.columns]


def _sort_table(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    order = []
    asc = []
    for c, direction in [
        (f"{scope}_rank_spearman", False),
        (f"{scope}_rank_pairwise_cindex", False),
        (f"{scope}_spearman", False),
        (f"{scope}_mae", True),
        (f"{scope}_rmse", True),
    ]:
        if c in df.columns:
            order.append(c)
            asc.append(direction)
    if not order:
        return df
    return df.sort_values(order, ascending=asc)


def _best_by_group(df: pd.DataFrame, group_col: str, scope: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return df.iloc[0:0].copy()
    work = _sort_table(df, scope)
    return work.groupby(group_col, dropna=False, as_index=False).head(1).reset_index(drop=True)


def _group_mean(df: pd.DataFrame, group_col: str, scope: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return df.iloc[0:0].copy()
    metrics = [
        c
        for c in [
            f"{scope}_mae",
            f"{scope}_rmse",
            f"{scope}_spearman",
            f"{scope}_rank_spearman",
            f"{scope}_rank_pairwise_cindex",
            f"{scope}_rank_kendall_tau",
            f"{scope}_regret",
        ]
        if c in df.columns
    ]
    if not metrics:
        return df.iloc[0:0].copy()
    out = (
        df.groupby(group_col, dropna=False)[metrics]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(metrics[0], ascending=True)
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ranking-first tables from method_summary.csv")
    parser.add_argument("--summary", required=True, help="Path to method_summary.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--scope",
        default="loto",
        choices=["loto", "lobo", "jointood"],
        help="Protocol scope prefix for metric columns",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise SystemExit(f"Missing summary CSV: {summary_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)
    if df.empty:
        raise SystemExit(f"No rows in summary CSV: {summary_path}")

    cols = _select_cols(df, args.scope)
    work = df[cols].copy()
    metric_cols = [c for c in cols if c.startswith(f"{args.scope}_")]
    work = _to_numeric(work, metric_cols)

    all_sorted = _sort_table(work, args.scope)
    best_family = _best_by_group(work, "family", args.scope)
    best_symmetry = _best_by_group(work, "symmetry", args.scope)
    mean_family = _group_mean(work, "family", args.scope)
    mean_symmetry = _group_mean(work, "symmetry", args.scope)

    prefix = f"rank_first_{args.scope}"
    all_sorted.to_csv(out_dir / f"{prefix}_all_methods.csv", index=False)
    best_family.to_csv(out_dir / f"{prefix}_best_by_family.csv", index=False)
    best_symmetry.to_csv(out_dir / f"{prefix}_best_by_symmetry.csv", index=False)
    mean_family.to_csv(out_dir / f"{prefix}_family_means.csv", index=False)
    mean_symmetry.to_csv(out_dir / f"{prefix}_symmetry_means.csv", index=False)

    print(f"Wrote tables under {out_dir}")


if __name__ == "__main__":
    main()
