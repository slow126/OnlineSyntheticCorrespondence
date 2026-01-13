#!/usr/bin/env python3
"""
Summarize predictor signal consistency across model/encoder regimes.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = (
    "analysis/comprehensive/univariate_all_predictors_per_encoder_options/auc_with_features.csv"
)


def _normalize_encoder_value(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"unknown", "nan"}:
        return None
    return text


def add_model_family_encoder_group(df):
    if "model_family" not in df.columns:
        if "model_family_encoder" in df.columns:
            return df
        raise ValueError("model_family column missing; cannot build group labels.")
    enc = df["encoder_config"] if "encoder_config" in df.columns else None
    groups = []
    for idx, row in df.iterrows():
        model_family = str(row["model_family"])
        enc_val = _normalize_encoder_value(enc.iloc[idx]) if enc is not None else None
        if model_family == "catspp" and enc_val:
            groups.append(f"{model_family}_{enc_val}")
        else:
            groups.append(model_family)
    df = df.copy()
    df["model_family_encoder"] = groups
    return df


def infer_predictors(df):
    predictors = []
    for col in df.columns:
        if col.startswith("flow_") or col.startswith("dino_"):
            predictors.append(col)
    return predictors


def compute_group_corrs(df, predictors, target, group_col):
    rows = []
    for group, sub in df.groupby(group_col):
        sub = sub.dropna(subset=[target])
        if sub.empty:
            continue
        for pred in predictors:
            if pred not in sub.columns:
                continue
            values = sub[[pred, target]].dropna()
            if len(values) < 3:
                continue
            if values[pred].nunique() < 2 or values[target].nunique() < 2:
                continue
            pearson = values[pred].corr(values[target], method="pearson")
            spearman = values[pred].corr(values[target], method="spearman")
            rows.append({
                "group": group,
                "predictor": pred,
                "n": len(values),
                "pearson_r": pearson,
                "spearman_r": spearman,
            })
    return pd.DataFrame(rows)


def summarize_consistency(per_group_df):
    if per_group_df.empty:
        return pd.DataFrame()
    rows = []
    for pred, sub in per_group_df.groupby("predictor"):
        spearman = sub["spearman_r"].dropna()
        pearson = sub["pearson_r"].dropna()
        if spearman.empty and pearson.empty:
            continue
        pos = (spearman > 0).sum()
        neg = (spearman < 0).sum()
        rows.append({
            "predictor": pred,
            "groups": int(sub["group"].nunique()),
            "spearman_pos": int(pos),
            "spearman_neg": int(neg),
            "spearman_consistency": float(max(pos, neg) / max(pos + neg, 1)),
            "spearman_mean": float(spearman.mean()) if not spearman.empty else np.nan,
            "spearman_mean_abs": float(spearman.abs().mean()) if not spearman.empty else np.nan,
            "pearson_mean": float(pearson.mean()) if not pearson.empty else np.nan,
            "pearson_mean_abs": float(pearson.abs().mean()) if not pearson.empty else np.nan,
        })
    return pd.DataFrame(rows).sort_values(
        ["spearman_consistency", "spearman_mean_abs"], ascending=False
    )


def main():
    parser = argparse.ArgumentParser(
        description="Summarize predictor consistency across model/encoder regimes."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV path.")
    parser.add_argument("--target", default=None, help="Target column name.")
    parser.add_argument(
        "--predictors",
        default=None,
        help="Comma-separated predictor list (defaults to flow_/dino_ columns).",
    )
    parser.add_argument(
        "--group-col",
        default="model_family_encoder",
        help="Grouping column to use (default: model_family_encoder).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory for CSV summaries.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = add_model_family_encoder_group(df)

    target = args.target
    if not target:
        target = "auc_delta_rank" if "auc_delta_rank" in df.columns else "auc_delta"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in input.")

    predictors = (
        [p.strip() for p in args.predictors.split(",") if p.strip()]
        if args.predictors
        else infer_predictors(df)
    )
    if not predictors:
        raise ValueError("No predictors found.")

    group_col = args.group_col
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in input.")

    per_group = compute_group_corrs(df, predictors, target, group_col)
    summary = summarize_consistency(per_group)

    print("Groups:", sorted(per_group["group"].unique()) if not per_group.empty else [])
    print("\nTop predictors by sign consistency (Spearman):")
    if summary.empty:
        print("No summary rows produced.")
    else:
        print(summary.head(15).to_string(index=False))

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        per_group.to_csv(out_dir / "per_group_correlations.csv", index=False)
        summary.to_csv(out_dir / "predictor_consistency_summary.csv", index=False)


if __name__ == "__main__":
    main()
