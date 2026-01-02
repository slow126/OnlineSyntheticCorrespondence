#!/usr/bin/env python3
"""
Summarize metric shifts between base, mix, and synthetic datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


DEFAULT_METRICS = [
    "flow_train_to_eval_coverage_logit",
    "flow_eval_to_train_coverage_logit",
    "flow_train_to_eval_mean_dist_over_radius_eval",
    "flow_eval_to_train_mean_dist_over_radius_train",
    "flow_mean_dist_asymmetry",
    "dino_train_to_eval_coverage_logit",
    "dino_eval_to_train_coverage_logit",
    "dino_train_to_eval_mean_dist_over_radius_eval",
    "dino_eval_to_train_mean_dist_over_radius_train",
    "dino_mean_dist_asymmetry",
    "flow_mmd",
    "dino_mmd",
]

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


def _mean_metrics(df: pd.DataFrame, metrics: List[str]) -> dict:
    if df.empty:
        return {m: np.nan for m in metrics}
    out = {}
    for m in metrics:
        if m not in df.columns:
            out[m] = np.nan
            continue
        values = pd.to_numeric(df[m], errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[m] = float(values.mean()) if not values.dropna().empty else np.nan
    return out


def _metric_family(name: str) -> str:
    if name.startswith("flow_") or name == "flow_mmd":
        return "flow"
    if name.startswith("dino_") or name.startswith("resnet_") or name in ("dino_mmd", "feature_mmd"):
        return "semantic"
    return "other"


def _direction_sign(name: str) -> int:
    lname = name.lower()
    if "coverage" in lname:
        if "outside" in lname:
            return -1
        return 1
    if "mmd" in lname:
        return -1
    if "mean_dist" in lname or "median_dist" in lname or "p90_dist" in lname:
        return -1
    if "asymmetry" in lname:
        return -1
    return 1


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x = x.astype(float)
    y = y.astype(float)
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return _pearson(rx, ry)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize metric shifts for mixed datasets.")
    parser.add_argument(
        "--auc-table",
        default="analysis/leakage_free_local_fast_dino_faiss/auc_delta/combined/auc_with_features.csv",
        help="Path to auc_with_features.csv.",
    )
    parser.add_argument(
        "--metrics",
        default="",
        help="Comma-separated list of metric columns (default: standard predictor set).",
    )
    parser.add_argument(
        "--performance-metric",
        default="peak_pck",
        help="Performance metric used for mix vs base delta (default: peak_pck).",
    )
    parser.add_argument(
        "--synthetic-dataset",
        default="synthetic",
        help="Synthetic dataset name to compare against (default: synthetic).",
    )
    parser.add_argument(
        "--by-encoder",
        action="store_true",
        help="Include encoder_config (or pretrained/freeze) in the grouping.",
    )
    parser.add_argument(
        "--output-csv",
        default="analysis/leakage_free_local_fast_dino_faiss/mix_metric_shifts.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-txt",
        default="analysis/leakage_free_local_fast_dino_faiss/mix_metric_shifts.txt",
        help="Output TXT path.",
    )
    args = parser.parse_args()

    auc_path = Path(args.auc_table)
    if not auc_path.exists():
        raise SystemExit(f"Missing auc table: {auc_path}")

    df = pd.read_csv(auc_path)
    df["benchmark"] = df["benchmark"].astype(str)
    df["train_dataset"] = df["train_dataset"].astype(str)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metrics:
        metrics = DEFAULT_METRICS
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        raise SystemExit("No metrics found in auc table.")
    perf_metric = args.performance_metric
    if perf_metric not in df.columns:
        raise SystemExit(f"Performance metric not found in auc table: {perf_metric}")

    mix_datasets = sorted({name for name in df["train_dataset"].unique() if _is_mix_dataset(name)})
    if not mix_datasets:
        raise SystemExit("No mixed datasets found.")

    group_cols = ["benchmark"]
    if args.by_encoder:
        if "encoder_config" in df.columns:
            group_cols.append("encoder_config")
        elif "pretrained" in df.columns and "freeze" in df.columns:
            group_cols.extend(["pretrained", "freeze"])

    rows = []
    for mix in mix_datasets:
        base = _base_dataset(mix)
        if base is None:
            continue
        for keys, sub in df.groupby(group_cols, dropna=False):
            sub = sub.copy()
            mix_df = sub[sub["train_dataset"] == mix]
            base_df = sub[sub["train_dataset"] == base]
            synth_df = sub[sub["train_dataset"] == args.synthetic_dataset]
            if mix_df.empty and base_df.empty and synth_df.empty:
                continue
            row = {}
            if not isinstance(keys, tuple):
                keys = (keys,)
            for col, val in zip(group_cols, keys):
                row[col] = val
            row["mix_dataset"] = mix
            row["base_dataset"] = base
            row["synthetic_dataset"] = args.synthetic_dataset
            row["n_mix"] = int(len(mix_df))
            row["n_base"] = int(len(base_df))
            row["n_synthetic"] = int(len(synth_df))

            mix_vals = _mean_metrics(mix_df, metrics)
            base_vals = _mean_metrics(base_df, metrics)
            synth_vals = _mean_metrics(synth_df, metrics)
            perf_mix = _mean_metrics(mix_df, [perf_metric])[perf_metric]
            perf_base = _mean_metrics(base_df, [perf_metric])[perf_metric]
            perf_synth = _mean_metrics(synth_df, [perf_metric])[perf_metric]
            for m in metrics:
                row[f"{m}_mix"] = mix_vals[m]
                row[f"{m}_base"] = base_vals[m]
                row[f"{m}_synthetic"] = synth_vals[m]
                row[f"{m}_delta_mix_base"] = mix_vals[m] - base_vals[m]
                row[f"{m}_delta_mix_synthetic"] = mix_vals[m] - synth_vals[m]
            row["performance_metric"] = perf_metric
            row["performance_mix"] = perf_mix
            row["performance_base"] = perf_base
            row["performance_synthetic"] = perf_synth
            row["performance_delta_mix_base"] = perf_mix - perf_base
            row["performance_delta_mix_synthetic"] = perf_mix - perf_synth
            rows.append(row)

    if not rows:
        raise SystemExit("No rows to write.")

    out_df = pd.DataFrame(rows)
    out_df.sort_values(group_cols + ["mix_dataset"], inplace=True)
    out_df.to_csv(args.output_csv, index=False)

    lines = []
    lines.append("MIX METRIC SHIFTS (base vs mix vs synthetic)")
    lines.append("=" * 80)
    lines.append(f"AUC table: {auc_path}")
    lines.append(f"Metrics: {', '.join(metrics)}")
    lines.append("")

    def _format_delta(value: float) -> str:
        if np.isnan(value):
            return "   n/a"
        return f"{value:+7.3f}"

    metric_families = {m: _metric_family(m) for m in metrics}
    flow_metrics = [m for m in metrics if metric_families[m] == "flow"]
    semantic_metrics = [m for m in metrics if metric_families[m] == "semantic"]

    signed = []
    for _, row in out_df.iterrows():
        signed_row = {
            "benchmark": row.get("benchmark"),
            "mix_dataset": row.get("mix_dataset"),
            "base_dataset": row.get("base_dataset"),
            "performance_delta_mix_base": row.get("performance_delta_mix_base"),
        }
        for m in metrics:
            delta = row.get(f"{m}_delta_mix_base")
            if pd.isna(delta):
                signed_row[m] = np.nan
            else:
                signed_row[m] = delta * _direction_sign(m)
        signed.append(signed_row)
    signed_df = pd.DataFrame(signed)
    signed_df["overall_shift"] = signed_df[metrics].mean(axis=1, skipna=True)
    if flow_metrics:
        signed_df["flow_shift"] = signed_df[flow_metrics].mean(axis=1, skipna=True)
    if semantic_metrics:
        signed_df["semantic_shift"] = signed_df[semantic_metrics].mean(axis=1, skipna=True)

    def _corr_block(label: str, df_block: pd.DataFrame, col: str) -> str:
        if df_block.empty or col not in df_block.columns:
            return f"{label}: n=0 (n/a)"
        sub = df_block[[col, "performance_delta_mix_base"]].dropna()
        if sub.empty:
            return f"{label}: n=0 (n/a)"
        x = sub[col].to_numpy(dtype=float)
        y = sub["performance_delta_mix_base"].to_numpy(dtype=float)
        return (
            f"{label}: n={len(sub)} "
            f"Pearson={_pearson(x, y):.3f}, Spearman={_spearman(x, y):.3f}"
        )

    lines.append("Direction-adjusted shift vs performance (mix-base):")
    lines.append(f"Metric: {perf_metric} (positive shift = better alignment)")
    lines.append("  " + _corr_block("Overall shift", signed_df, "overall_shift"))
    if flow_metrics:
        lines.append("  " + _corr_block("Flow shift", signed_df, "flow_shift"))
    if semantic_metrics:
        lines.append("  " + _corr_block("Semantic shift", signed_df, "semantic_shift"))
    flow_df = signed_df[signed_df["benchmark"].isin(FLOW_FAMILY)]
    sem_df = signed_df[signed_df["benchmark"].isin(SEMANTIC_FAMILY)]
    lines.append("  " + _corr_block("Overall shift (flow benchmarks)", flow_df, "overall_shift"))
    lines.append("  " + _corr_block("Overall shift (semantic benchmarks)", sem_df, "overall_shift"))
    lines.append("")

    if args.by_encoder:
        group_cols = ["benchmark", "encoder_config"] if "encoder_config" in out_df.columns else ["benchmark", "pretrained", "freeze"]
    else:
        group_cols = ["benchmark"]

    for key, sub in out_df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        header_parts = [f"{col}={val}" for col, val in zip(group_cols, key)]
        lines.append("")
        lines.append("Benchmark: " + ", ".join(header_parts))
        lines.append("-" * 80)
        for _, row in sub.iterrows():
            lines.append(
                f"Mix: {row['mix_dataset']} (base={row['base_dataset']}, n_mix={int(row['n_mix'])}, "
                f"n_base={int(row['n_base'])}, n_synth={int(row['n_synthetic'])})"
            )
            lines.append(f"{'metric':<38} {'d_mix-base':>10} {'d_mix-synth':>12}")
            lines.append("-" * 80)
            for m in metrics:
                lines.append(
                    f"{m:<38} {_format_delta(row[f'{m}_delta_mix_base']):>10} {_format_delta(row[f'{m}_delta_mix_synthetic']):>12}"
                )
            if flow_metrics or semantic_metrics:
                def _count_pos(metric_list, col):
                    if not metric_list:
                        return (0, 0)
                    vals = [row[f"{m}_{col}"] for m in metric_list]
                    vals = [v for v in vals if not np.isnan(v)]
                    pos = sum(1 for v in vals if v > 0)
                    return (pos, len(vals))

                flow_pos, flow_total = _count_pos(flow_metrics, "delta_mix_base")
                sem_pos, sem_total = _count_pos(semantic_metrics, "delta_mix_base")
                lines.append(
                    f"Delta mix-base positive: flow {flow_pos}/{flow_total}, semantic {sem_pos}/{sem_total}"
                )
                flow_pos, flow_total = _count_pos(flow_metrics, "delta_mix_synthetic")
                sem_pos, sem_total = _count_pos(semantic_metrics, "delta_mix_synthetic")
                lines.append(
                    f"Delta mix-synth positive: flow {flow_pos}/{flow_total}, semantic {sem_pos}/{sem_total}"
                )
            lines.append("")

    lines.append(f"Wrote CSV: {args.output_csv}")
    Path(args.output_txt).write_text("\n".join(lines))
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_txt}")


if __name__ == "__main__":
    main()
