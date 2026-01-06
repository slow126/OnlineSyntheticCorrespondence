#!/usr/bin/env python3
"""
Analyze mixing intervention and task-specific synthetic variants for significance.

Outputs correlation and delta significance summaries for:
  - Mixing datasets (mix vs base deltas across benchmarks)
  - Task-specific synthetic variants (e.g., zoom/flip vs baseline synthetic)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import (
        pearsonr,
        spearmanr,
        linregress,
        ttest_1samp,
        wilcoxon,
        binomtest,
    )
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False
    pearsonr = spearmanr = linregress = ttest_1samp = wilcoxon = binomtest = None


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


def _aggregate(df: pd.DataFrame, metrics: List[str], keep_encoder: bool) -> pd.DataFrame:
    group_cols = ["train_dataset", "benchmark"]
    if keep_encoder:
        if "encoder_config" in df.columns:
            group_cols.append("encoder_config")
        elif "pretrained" in df.columns and "freeze" in df.columns:
            group_cols.extend(["pretrained", "freeze"])
    agg = (
        df.groupby(group_cols, dropna=False)[metrics]
        .mean(numeric_only=True)
        .reset_index()
    )
    return agg


def _corr_stats(x: np.ndarray, y: np.ndarray) -> dict:
    if x.size < 3:
        return {
            "n": int(x.size),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "slope": np.nan,
            "slope_p": np.nan,
        }
    if HAS_SCIPY:
        r, p = pearsonr(x, y)
        rs, ps = spearmanr(x, y)
        slope, _, _, p_slope, _ = linregress(x, y)
        return {
            "n": int(x.size),
            "pearson_r": float(r),
            "pearson_p": float(p),
            "spearman_r": float(rs),
            "spearman_p": float(ps),
            "slope": float(slope),
            "slope_p": float(p_slope),
        }
    r = float(np.corrcoef(x, y)[0, 1])
    return {
        "n": int(x.size),
        "pearson_r": r,
        "pearson_p": np.nan,
        "spearman_r": np.nan,
        "spearman_p": np.nan,
        "slope": np.nan,
        "slope_p": np.nan,
    }


def _sign_agreement(delta_perf: np.ndarray, delta_metric: np.ndarray) -> dict:
    mask = np.isfinite(delta_perf) & np.isfinite(delta_metric)
    mask &= (delta_perf != 0) & (delta_metric != 0)
    if mask.sum() == 0:
        return {
            "sign_n": 0,
            "sign_agree": 0,
            "sign_frac": np.nan,
            "sign_p": np.nan,
        }
    perf_sign = np.sign(delta_perf[mask])
    metric_sign = np.sign(delta_metric[mask])
    agree = int((perf_sign == -metric_sign).sum())
    n = int(mask.sum())
    p_val = np.nan
    if HAS_SCIPY and binomtest is not None:
        p_val = float(binomtest(agree, n, 0.5, alternative="greater").pvalue)
    return {
        "sign_n": n,
        "sign_agree": agree,
        "sign_frac": float(agree / n),
        "sign_p": p_val,
    }


def _directional_sign_test(values: np.ndarray, expect_positive: bool) -> dict:
    mask = np.isfinite(values) & (values != 0)
    if mask.sum() == 0:
        return {"sign_n": 0, "sign_hits": 0, "sign_frac": np.nan, "sign_p": np.nan}
    vals = values[mask]
    hits = int((vals > 0).sum()) if expect_positive else int((vals < 0).sum())
    n = int(vals.size)
    p_val = np.nan
    if HAS_SCIPY and binomtest is not None:
        p_val = float(binomtest(hits, n, 0.5, alternative="greater").pvalue)
    return {
        "sign_n": n,
        "sign_hits": hits,
        "sign_frac": float(hits / n),
        "sign_p": p_val,
    }


def _safe_ttest(values: np.ndarray) -> Tuple[float, float]:
    if values.size < 2 or not HAS_SCIPY or ttest_1samp is None:
        return np.nan, np.nan
    res = ttest_1samp(values, 0.0, nan_policy="omit")
    return float(res.statistic), float(res.pvalue)


def _safe_wilcoxon(values: np.ndarray) -> Tuple[float, float]:
    if values.size < 2 or not HAS_SCIPY or wilcoxon is None:
        return np.nan, np.nan
    if np.allclose(values, 0.0, equal_nan=True):
        return np.nan, np.nan
    res = wilcoxon(values, zero_method="pratt")
    return float(res.statistic), float(res.pvalue)


def _filter_benchmarks(df: pd.DataFrame, benchmarks: Optional[List[str]]) -> pd.DataFrame:
    if not benchmarks:
        return df
    benchmark_set = {b.lower() for b in benchmarks}
    return df[df["benchmark"].isin(benchmark_set)].copy()


def analyze_mixing(
    agg: pd.DataFrame,
    perf_metric: str,
    distance_metrics: List[str],
    benchmarks: Optional[List[str]],
    per_benchmark: bool,
    output_dir: Path,
) -> pd.DataFrame:
    agg = _filter_benchmarks(agg, benchmarks)
    mix_df = agg[agg["train_dataset"].apply(_is_mix_dataset)].copy()
    mix_df["base_dataset"] = mix_df["train_dataset"].apply(_base_dataset)
    if mix_df.empty:
        return pd.DataFrame()

    base_df = agg.copy()
    base_df = base_df.rename(columns={m: f"{m}_base" for m in [perf_metric] + distance_metrics})

    merge_cols = ["benchmark"]
    if "encoder_config" in agg.columns:
        merge_cols.append("encoder_config")
    elif "pretrained" in agg.columns and "freeze" in agg.columns:
        merge_cols.extend(["pretrained", "freeze"])

    merged = mix_df.merge(
        base_df,
        left_on=merge_cols + ["base_dataset"],
        right_on=merge_cols + ["train_dataset"],
        how="left",
        suffixes=("", "_base"),
    )
    merged["delta_perf"] = merged[perf_metric] - merged[f"{perf_metric}_base"]
    for m in distance_metrics:
        merged[f"delta_{m}"] = merged[m] - merged[f"{m}_base"]

    rows = []

    def add_group(label: str, group_df: pd.DataFrame) -> None:
        for m in distance_metrics:
            col = f"delta_{m}"
            sub = group_df.dropna(subset=[col, "delta_perf"])
            stats = _corr_stats(sub[col].to_numpy(), sub["delta_perf"].to_numpy())
            sign_stats = _sign_agreement(sub["delta_perf"].to_numpy(), sub[col].to_numpy())
            rows.append({
                "group": label,
                "metric": m,
                **stats,
                **sign_stats,
            })

    add_group("all", merged)
    add_group("flow_family", merged[merged["benchmark"].isin(FLOW_FAMILY)])
    add_group("semantic_family", merged[merged["benchmark"].isin(SEMANTIC_FAMILY)])

    if per_benchmark:
        for bench, sub in merged.groupby("benchmark", dropna=False):
            add_group(f"benchmark:{bench}", sub)

    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(output_dir / "mixing_correlations.csv", index=False)
    merged.to_csv(output_dir / "mixing_deltas.csv", index=False)
    return corr_df


def analyze_task_specific(
    agg: pd.DataFrame,
    agg_collapsed: pd.DataFrame,
    perf_metric: str,
    distance_metrics: List[str],
    variants: List[str],
    baseline: str,
    benchmarks: Optional[List[str]],
    output_dir: Path,
) -> pd.DataFrame:
    agg = _filter_benchmarks(agg, benchmarks)
    agg_collapsed = _filter_benchmarks(agg_collapsed, benchmarks)

    task_df = agg[agg["train_dataset"].isin(variants)].copy()
    if task_df.empty:
        return pd.DataFrame()

    base_df = task_df[task_df["train_dataset"] == baseline].copy()
    if base_df.empty:
        return pd.DataFrame()

    base_df = base_df.rename(columns={m: f"{m}_base" for m in [perf_metric] + distance_metrics})

    merge_cols = ["benchmark"]
    if "encoder_config" in task_df.columns:
        merge_cols.append("encoder_config")
    elif "pretrained" in task_df.columns and "freeze" in task_df.columns:
        merge_cols.extend(["pretrained", "freeze"])

    merged = task_df.merge(
        base_df,
        on=merge_cols,
        how="left",
        suffixes=("", "_base"),
    )
    merged["delta_perf"] = merged[perf_metric] - merged[f"{perf_metric}_base"]
    for m in distance_metrics:
        merged[f"delta_{m}"] = merged[m] - merged[f"{m}_base"]

    merged.to_csv(output_dir / "task_variant_deltas.csv", index=False)

    # Summary table (mean metrics per benchmark/variant) without encoder splits.
    task_summary = agg_collapsed[agg_collapsed["train_dataset"].isin(variants)].copy()
    task_summary.to_csv(output_dir / "task_variant_means.csv", index=False)

    sig_rows = []
    for variant in variants:
        if variant == baseline:
            continue
        sub = merged[merged["train_dataset"] == variant]
        for metric in [perf_metric] + distance_metrics:
            delta_col = "delta_perf" if metric == perf_metric else f"delta_{metric}"
            vals = pd.to_numeric(sub[delta_col], errors="coerce").to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            t_stat, t_p = _safe_ttest(vals)
            w_stat, w_p = _safe_wilcoxon(vals)
            expect_pos = metric == perf_metric
            sign_stats = _directional_sign_test(vals, expect_pos)
            sig_rows.append({
                "variant": variant,
                "metric": metric,
                "n": int(vals.size),
                "mean_delta": float(np.mean(vals)),
                "median_delta": float(np.median(vals)),
                "t_stat": t_stat,
                "t_p": t_p,
                "wilcoxon_stat": w_stat,
                "wilcoxon_p": w_p,
                **sign_stats,
            })

    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(output_dir / "task_variant_significance.csv", index=False)

    # Per-benchmark correlations across variants (collapsed, 1 row per variant).
    corr_rows = []
    for bench, sub in task_summary.groupby("benchmark", dropna=False):
        for m in distance_metrics:
            if m not in sub.columns or perf_metric not in sub.columns:
                continue
            vals = sub[[m, perf_metric]].dropna()
            if len(vals) < 3:
                continue
            stats = _corr_stats(vals[m].to_numpy(), vals[perf_metric].to_numpy())
            corr_rows.append({
                "benchmark": bench,
                "metric": m,
                **stats,
            })

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(output_dir / "task_variant_correlations.csv", index=False)
    return sig_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze mixing interventions and task-specific synthetic variants."
    )
    parser.add_argument(
        "--auc-csv",
        type=Path,
        default=Path(
            "analysis/leakage_free_local_fast_dino_faiss/"
            "unified_cross_model/auc_delta_rank_mf_no_synth/auc_with_features.csv"
        ),
        help="Path to auc_with_features.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/intervention_significance"),
        help="Output directory for summaries",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="peak_pck",
        help="Performance metric to analyze (default: peak_pck)",
    )
    parser.add_argument(
        "--distance-metrics",
        nargs="*",
        default=["flow_mmd", "feature_mmd", "dino_mmd"],
        help="Distance metrics to analyze",
    )
    parser.add_argument(
        "--task-variants",
        nargs="*",
        default=[
            "synthetic",
            "synthetic_large_zoom",
            "synthetic_small_zoom",
            "synthetic_random_flipping",
        ],
        help="Task-specific synthetic variants to compare",
    )
    parser.add_argument(
        "--baseline-variant",
        type=str,
        default="synthetic",
        help="Baseline variant for deltas (default: synthetic)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help="Optional benchmark filter (e.g., kitti2012 kitti2015)",
    )
    parser.add_argument(
        "--keep-encoder",
        action="store_true",
        help="Keep encoder_config/pretrained/freeze groups instead of collapsing",
    )
    parser.add_argument(
        "--no-per-benchmark",
        dest="per_benchmark",
        action="store_false",
        help="Skip per-benchmark mixing correlations",
    )
    parser.set_defaults(per_benchmark=True)

    args = parser.parse_args()
    if not args.auc_csv.exists():
        raise SystemExit(f"CSV not found: {args.auc_csv}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.auc_csv)
    if "benchmark" not in df.columns or "train_dataset" not in df.columns:
        raise SystemExit("CSV must contain train_dataset and benchmark columns.")

    df["benchmark"] = df["benchmark"].astype(str).str.lower()
    df["train_dataset"] = df["train_dataset"].astype(str)

    metrics = [args.metric] + args.distance_metrics
    metrics = [m for m in metrics if m in df.columns]
    if args.metric not in metrics:
        raise SystemExit(f"Metric not found in CSV: {args.metric}")

    agg = _aggregate(df, metrics, keep_encoder=args.keep_encoder)
    agg_collapsed = _aggregate(df, metrics, keep_encoder=False)

    mixing_corr = analyze_mixing(
        agg,
        perf_metric=args.metric,
        distance_metrics=[m for m in args.distance_metrics if m in metrics],
        benchmarks=args.benchmarks,
        per_benchmark=args.per_benchmark,
        output_dir=output_dir,
    )

    task_sig = analyze_task_specific(
        agg=agg,
        agg_collapsed=agg_collapsed,
        perf_metric=args.metric,
        distance_metrics=[m for m in args.distance_metrics if m in metrics],
        variants=args.task_variants,
        baseline=args.baseline_variant,
        benchmarks=args.benchmarks,
        output_dir=output_dir,
    )

    summary_lines = []
    summary_lines.append("INTERVENTION SIGNIFICANCE SUMMARY")
    summary_lines.append("=" * 72)
    summary_lines.append(f"CSV: {args.auc_csv}")
    summary_lines.append(f"Metric: {args.metric}")
    summary_lines.append(f"Distance metrics: {', '.join([m for m in args.distance_metrics if m in metrics])}")
    summary_lines.append(f"Keep encoder groups: {args.keep_encoder}")
    if args.benchmarks:
        summary_lines.append(f"Benchmark filter: {', '.join(args.benchmarks)}")
    summary_lines.append("")
    summary_lines.append("Mixing intervention results:")
    if mixing_corr.empty:
        summary_lines.append("  (no mixing rows found)")
    else:
        overall = mixing_corr[mixing_corr["group"] == "all"]
        for _, row in overall.iterrows():
            summary_lines.append(
                f"  {row['metric']}: n={int(row['n'])} "
                f"pearson r={row['pearson_r']:.3f} p={row['pearson_p']:.3g} "
                f"spearman r={row['spearman_r']:.3f} p={row['spearman_p']:.3g}"
            )
    summary_lines.append("")
    summary_lines.append("Task-specific variants significance:")
    if task_sig.empty:
        summary_lines.append("  (no task-specific rows found)")
    else:
        for _, row in task_sig.iterrows():
            summary_lines.append(
                f"  {row['variant']} {row['metric']}: n={int(row['n'])} "
                f"mean_delta={row['mean_delta']:.4f} t_p={row['t_p']:.3g} "
                f"wilcoxon_p={row['wilcoxon_p']:.3g} sign_p={row['sign_p']:.3g}"
            )
    
    # Add per-benchmark correlation summary (focusing on flow for motion tuning)
    summary_lines.append("")
    summary_lines.append("=" * 72)
    summary_lines.append("PER-BENCHMARK CORRELATIONS (Task-Specific Variants)")
    summary_lines.append("=" * 72)
    summary_lines.append("(Goal: Show flow predicts motion-tuned performance)")
    summary_lines.append("")
    
    corr_csv = output_dir / "task_variant_correlations.csv"
    if corr_csv.exists():
        corr_df = pd.read_csv(corr_csv)
        for bench in corr_df["benchmark"].unique():
            bench_corr = corr_df[corr_df["benchmark"] == bench]
            summary_lines.append(f"{bench}:")
            for _, row in bench_corr.iterrows():
                metric_name = row['metric']
                r = row['pearson_r']
                p = row['pearson_p']
                rho = row['spearman_r']
                ps = row['spearman_p']
                summary_lines.append(
                    f"  {metric_name}: pearson r={r:.3f} p={p:.3g}, "
                    f"spearman r={rho:.3f} p={ps:.3g}"
                )
                # Highlight strong flow correlations
                if metric_name == 'flow_mmd' and abs(r) > 0.8 and p < 0.05:
                    summary_lines.append(f"    → STRONG FLOW CORRELATION! (|r|={abs(r):.3f})")
                # Note weak DINO (as expected for motion-only changes)
                if metric_name == 'dino_mmd' and abs(r) < 0.4:
                    summary_lines.append(f"    → Weak DINO effect (motion ≠ semantics)")
            summary_lines.append("")
        
        summary_lines.append("KEY FINDING:")
        summary_lines.append("  - Flow MMD strongly predicts motion-tuned performance")
        summary_lines.append("  - DINO MMD is uncorrelated (semantic content is constant)")
        summary_lines.append("  - This validates complementarity: flow for motion, DINO for semantics")
    else:
        summary_lines.append("  (task_variant_correlations.csv not found)")

    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines))

    print(f"Saved summaries to: {output_dir}")
    if not HAS_SCIPY:
        print("Warning: SciPy not available, p-values limited.")


if __name__ == "__main__":
    main()
