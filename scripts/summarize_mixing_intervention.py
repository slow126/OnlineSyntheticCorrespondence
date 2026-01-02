#!/usr/bin/env python3
"""
Summarize synthetic mixing intervention effects vs base datasets.

Computes delta in a chosen metric for each mixed dataset vs its base dataset
within each benchmark and encoder regime, then aggregates overall and by family.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

FLOW_FAMILY = ["kitti2012", "kitti2015", "middlebury", "flyingthings", "pointodyssey"]
SEMANTIC_FAMILY = ["spair", "pfpascal", "pfwillow", "tss"]


def _is_mix_dataset(name: str) -> bool:
    if not name or not isinstance(name, str):
        return False
    if name.startswith("synthetic"):
        return False
    return "_synthetic" in name


def _base_dataset(name: str) -> Optional[str]:
    if not _is_mix_dataset(name):
        return None
    if "_synthetic_" in name:
        return name.split("_synthetic_", 1)[0]
    if name.endswith("_synthetic"):
        return name[: -len("_synthetic")]
    return name.split("_synthetic", 1)[0]


def _summarize_group(df: pd.DataFrame, metric: str) -> Dict[str, float]:
    values = pd.to_numeric(df[metric], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {"n": 0, "mean": np.nan, "median": np.nan}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(values.median()),
    }


def _compute_mix_deltas(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if "train_dataset" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["train_dataset"] = df["train_dataset"].astype(str)
    df["benchmark"] = df["benchmark"].astype(str).str.lower()
    df["base_dataset"] = df["train_dataset"].apply(_base_dataset)
    mix_df = df[df["base_dataset"].notna()].copy()
    if mix_df.empty:
        return pd.DataFrame()

    key_cols = ["benchmark"]
    if "encoder_config" in df.columns:
        key_cols.append("encoder_config")
    elif "pretrained" in df.columns and "freeze" in df.columns:
        key_cols.extend(["pretrained", "freeze"])

    base_df = df[df["train_dataset"].notna()].copy()
    base_df = base_df.rename(columns={metric: f"{metric}_base"})
    base_df = base_df[key_cols + ["train_dataset", f"{metric}_base"]]
    mix_df = mix_df[key_cols + ["train_dataset", "base_dataset", metric]]

    merged = mix_df.merge(
        base_df,
        left_on=key_cols + ["base_dataset"],
        right_on=key_cols + ["train_dataset"],
        how="left",
        suffixes=("", "_base"),
    )
    if "mix_dataset" not in merged.columns:
        if "train_dataset_x" in merged.columns:
            merged = merged.rename(columns={"train_dataset_x": "mix_dataset"})
        elif "train_dataset" in merged.columns:
            merged = merged.rename(columns={"train_dataset": "mix_dataset"})
    if "train_dataset_base" in merged.columns:
        merged = merged.drop(columns=["train_dataset_base"])
    merged["delta"] = merged[metric] - merged[f"{metric}_base"]
    return merged


def _aggregate_mix_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (mix, base), sub in df.groupby(["mix_dataset", "base_dataset"], dropna=False):
        overall = _summarize_group(sub, "delta")
        flow = _summarize_group(sub[sub["benchmark"].isin(FLOW_FAMILY)], "delta")
        semantic = _summarize_group(sub[sub["benchmark"].isin(SEMANTIC_FAMILY)], "delta")
        frac_improved = float((sub["delta"] > 0).mean()) if not sub.empty else np.nan
        rows.append({
            "mix_dataset": mix,
            "base_dataset": base,
            "n_all": overall["n"],
            "mean_delta": overall["mean"],
            "median_delta": overall["median"],
            "frac_improved": frac_improved,
            "n_flow": flow["n"],
            "mean_delta_flow": flow["mean"],
            "median_delta_flow": flow["median"],
            "frac_improved_flow": float((sub[sub["benchmark"].isin(FLOW_FAMILY)]["delta"] > 0).mean())
            if flow["n"] else np.nan,
            "n_semantic": semantic["n"],
            "mean_delta_semantic": semantic["mean"],
            "median_delta_semantic": semantic["median"],
            "frac_improved_semantic": float((sub[sub["benchmark"].isin(SEMANTIC_FAMILY)]["delta"] > 0).mean())
            if semantic["n"] else np.nan,
        })
    return pd.DataFrame(rows)


def _write_summary_txt(output_path: Path, df: pd.DataFrame, metric: str) -> None:
    lines = []
    lines.append(f"MIXING INTERVENTION SUMMARY ({metric})")
    lines.append("=" * 80)
    if df.empty:
        lines.append("No mix datasets found or no matches to base datasets.")
        output_path.write_text("\n".join(lines))
        return
    has_target = "target" in df.columns
    has_variant = "variant" in df.columns
    header = ""
    if has_target:
        header += f"{'target':<14} "
    if has_variant:
        header += f"{'variant':<12} "
    header += (
        f"{'mix_dataset':<28} {'base':<12} "
        f"{'n':>4} {'mean':>7} {'med':>7} {'frac+':>7} "
        f"{'nF':>4} {'meanF':>7} {'fracF':>7} "
        f"{'nS':>4} {'meanS':>7} {'fracS':>7}"
    )
    lines.append("")
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in df.iterrows():
        prefix = ""
        if has_target:
            prefix += f"{row['target']:<14} "
        if has_variant:
            prefix += f"{row['variant']:<12} "
        lines.append(
            f"{prefix}{row['mix_dataset']:<28} {row['base_dataset']:<12} "
            f"{int(row['n_all']):>4} {row['mean_delta']:>7.2f} {row['median_delta']:>7.2f} {row['frac_improved']:>7.2f} "
            f"{int(row['n_flow']):>4} {row['mean_delta_flow']:>7.2f} {row['frac_improved_flow']:>7.2f} "
            f"{int(row['n_semantic']):>4} {row['mean_delta_semantic']:>7.2f} {row['frac_improved_semantic']:>7.2f}"
        )

    group_cols = []
    if has_target:
        group_cols.append("target")
    if has_variant:
        group_cols.append("variant")
    if not group_cols:
        group_cols = ["__all__"]
        df = df.assign(__all__="all")

    lines.append("")
    lines.append("Summary across mix datasets (weighted by n):")
    summary_header = ""
    if "__all__" not in group_cols:
        summary_header = f"{'target':<14} {'variant':<12} "
    summary_header += f"{'mix_count':>9} {'mean':>7} {'frac+':>7} {'meanF':>7} {'fracF':>7} {'meanS':>7} {'fracS':>7}"
    lines.append(summary_header)
    lines.append("-" * len(summary_header))
    for key, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        n_all = sub["n_all"].sum()
        n_flow = sub["n_flow"].sum()
        n_sem = sub["n_semantic"].sum()
        mean_delta = np.nan
        frac_improved = np.nan
        mean_flow = np.nan
        frac_flow = np.nan
        mean_sem = np.nan
        frac_sem = np.nan
        if n_all > 0:
            mean_delta = float((sub["mean_delta"] * sub["n_all"]).sum() / n_all)
            frac_improved = float((sub["frac_improved"] * sub["n_all"]).sum() / n_all)
        if n_flow > 0:
            mean_flow = float((sub["mean_delta_flow"] * sub["n_flow"]).sum() / n_flow)
            frac_flow = float((sub["frac_improved_flow"] * sub["n_flow"]).sum() / n_flow)
        if n_sem > 0:
            mean_sem = float((sub["mean_delta_semantic"] * sub["n_semantic"]).sum() / n_sem)
            frac_sem = float((sub["frac_improved_semantic"] * sub["n_semantic"]).sum() / n_sem)
        mix_count = int(sub["mix_dataset"].nunique())
        prefix = ""
        if "__all__" not in group_cols:
            prefix = f"{key[0]:<14} {key[1]:<12} "
        lines.append(
            f"{prefix}{mix_count:>9} {mean_delta:>7.2f} {frac_improved:>7.2f} "
            f"{mean_flow:>7.2f} {frac_flow:>7.2f} {mean_sem:>7.2f} {frac_sem:>7.2f}"
        )

    group_cols_base = [c for c in group_cols if c != "__all__"]
    if "base_dataset" not in group_cols_base:
        group_cols_base.append("base_dataset")
    best_rows = []
    for key, sub in df.groupby(group_cols_base, dropna=False):
        if sub.empty:
            continue
        best = sub.sort_values("mean_delta", ascending=False).iloc[0]
        if not isinstance(key, tuple):
            key = (key,)
        best_rows.append((key, best))
    if best_rows:
        lines.append("")
        lines.append("Best mix per base dataset (by mean delta):")
        header = ""
        if "__all__" not in group_cols_base:
            if "target" in df.columns:
                header += f"{'target':<14} "
            if "variant" in df.columns:
                header += f"{'variant':<12} "
        header += (
            f"{'base':<12} {'mix_dataset':<28} "
            f"{'mean':>7} {'frac+':>7} {'meanF':>7} {'fracF':>7} {'meanS':>7} {'fracS':>7}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for key, row in best_rows:
            idx = 0
            prefix = ""
            if "__all__" not in group_cols_base:
                if "target" in df.columns:
                    prefix += f"{key[idx]:<14} "
                    idx += 1
                if "variant" in df.columns:
                    prefix += f"{key[idx]:<12} "
                    idx += 1
            base_val = key[idx] if idx < len(key) else row["base_dataset"]
            lines.append(
                f"{prefix}{str(base_val):<12} {row['mix_dataset']:<28} "
                f"{row['mean_delta']:>7.2f} {row['frac_improved']:>7.2f} "
                f"{row['mean_delta_flow']:>7.2f} {row['frac_improved_flow']:>7.2f} "
                f"{row['mean_delta_semantic']:>7.2f} {row['frac_improved_semantic']:>7.2f}"
            )

    output_path.write_text("\n".join(lines))


def run_for_dir(run_dir: Path, metric: str, output_name: str) -> Optional[pd.DataFrame]:
    auc_path = run_dir / "auc_with_features.csv"
    if not auc_path.exists():
        return None
    df = pd.read_csv(auc_path)
    if metric not in df.columns:
        return None
    deltas = _compute_mix_deltas(df, metric)
    summary = _aggregate_mix_summary(deltas)
    if summary.empty:
        return None
    summary_path = run_dir / f"{output_name}.csv"
    summary.to_csv(summary_path, index=False)
    _write_summary_txt(run_dir / f"{output_name}.txt", summary, metric)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize mixing intervention effects.")
    parser.add_argument(
        "--base-dir",
        default="analysis/leakage_free_local_fast_dino_faiss",
        help="Base analysis directory containing target subfolders.",
    )
    parser.add_argument(
        "--targets",
        default="auc_delta,auc_delta_rank,peak_pck,peak_pck_rank",
        help="Comma-separated list of target subdirectories.",
    )
    parser.add_argument(
        "--variants",
        default="combined",
        help="Comma-separated list of ablation variants.",
    )
    parser.add_argument(
        "--metric",
        default="peak_pck",
        help="Metric column to compare mix vs base (default: peak_pck).",
    )
    parser.add_argument(
        "--write-combined",
        action="store_true",
        help="Also write a combined summary across targets (default: off).",
    )
    parser.add_argument(
        "--output-name",
        default="mix_intervention_summary",
        help="Base filename for summary outputs.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional directory to write a single summary table (e.g., output dir root).",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    combined_rows = []
    root_rows = []
    for target in targets:
        target_dir = base_dir / target
        if not target_dir.exists():
            print(f"Skipping {target}: directory not found ({target_dir})")
            continue
        for variant in variants:
            run_dir = target_dir / variant
            if not run_dir.exists():
                continue
            summary = run_for_dir(run_dir, args.metric, args.output_name)
            if summary is None or summary.empty:
                continue
            summary = summary.copy()
            summary.insert(0, "variant", variant)
            summary.insert(0, "target", target)
            combined_rows.append(summary)
            if args.output_root:
                root_rows.append(summary)

    if args.write_combined and combined_rows:
        combined_df = pd.concat(combined_rows, ignore_index=True)
        combined_df = combined_df.drop_duplicates()
        combined_path = base_dir / f"{args.output_name}_combined.csv"
        combined_df.to_csv(combined_path, index=False)
        _write_summary_txt(base_dir / f"{args.output_name}_combined.txt", combined_df, args.metric)

    if args.output_root and root_rows:
        root_df = pd.concat(root_rows, ignore_index=True).drop_duplicates()
        root_dir = Path(args.output_root)
        root_dir.mkdir(parents=True, exist_ok=True)
        root_csv = root_dir / f"{args.output_name}.csv"
        root_txt = root_dir / f"{args.output_name}.txt"
        root_df.to_csv(root_csv, index=False)
        _write_summary_txt(root_txt, root_df, args.metric)


if __name__ == "__main__":
    main()
