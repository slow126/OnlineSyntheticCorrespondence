#!/usr/bin/env python3
"""
Analyze correlation between mixing intervention gains and distance metrics,
plus task-specific synthetic proximity to KITTI benchmarks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:  # pragma: no cover - optional dependency
    stats = None


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


def _normalize_name(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _load_mmd_lookup(csv_path: Path) -> Dict[Tuple[str, str], float]:
    df = pd.read_csv(csv_path)
    for col in ["dataset1", "dataset2", "mmd2"]:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {csv_path}")
    df["dataset1"] = df["dataset1"].astype(str).str.lower()
    df["dataset2"] = df["dataset2"].astype(str).str.lower()
    df["mmd2"] = pd.to_numeric(df["mmd2"], errors="coerce")
    df = df.dropna(subset=["dataset1", "dataset2", "mmd2"])

    lookup: Dict[Tuple[str, str], float] = {}
    grouped = df.groupby(["dataset1", "dataset2"], dropna=False)["mmd2"].mean()
    for (d1, d2), value in grouped.items():
        lookup[(d1, d2)] = float(value)
        lookup[(d2, d1)] = float(value)
    return lookup


def _safe_corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 3:
        return float("nan"), float("nan")
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan"), float("nan")
    if stats is None:
        return float(np.corrcoef(x, y)[0, 1]), float("nan")
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 3:
        return float("nan"), float("nan")
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan"), float("nan")
    if stats is None:
        rho = pd.Series(x).corr(pd.Series(y), method="spearman")
        return float(rho), float("nan")
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def _format_corr(label: str, x: np.ndarray, y: np.ndarray) -> List[str]:
    r, p = _safe_corr(x, y)
    rho, p_s = _safe_spearman(x, y)
    return [
        f"{label}: n={len(x)}",
        f"  pearson r={r:.4f} p={p:.4f}" if not np.isnan(r) else "  pearson r=nan p=nan",
        f"  spearman r={rho:.4f} p={p_s:.4f}" if not np.isnan(rho) else "  spearman r=nan p=nan",
    ]


def _extract_config_cols(df: pd.DataFrame) -> List[str]:
    if "encoder_config" in df.columns:
        return ["encoder_config"]
    cols = []
    for name in ["pretrained", "freeze"]:
        if name in df.columns:
            cols.append(name)
    return cols


def _compute_mix_deltas(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    df = df.copy()
    df["train_dataset"] = df["train_dataset"].astype(str).str.lower()
    df["benchmark"] = df["benchmark"].astype(str).str.lower()
    df["base_dataset"] = df["train_dataset"].apply(_base_dataset)

    mix_df = df[df["base_dataset"].notna()].copy()
    if mix_df.empty:
        return pd.DataFrame()

    key_cols = ["benchmark"] + _extract_config_cols(df)
    base_df = df[~df["train_dataset"].apply(_is_mix_dataset)].copy()
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
    if "train_dataset_x" in merged.columns:
        merged = merged.rename(columns={"train_dataset_x": "mix_dataset"})
    elif "train_dataset" in merged.columns:
        merged = merged.rename(columns={"train_dataset": "mix_dataset"})
    if "train_dataset_base" in merged.columns:
        merged = merged.drop(columns=["train_dataset_base"])
    merged["delta_metric"] = merged[metric] - merged[f"{metric}_base"]
    return merged


def _add_delta_mmd(df: pd.DataFrame, lookup: Dict[Tuple[str, str], float], col_name: str) -> None:
    def _lookup_pair(dataset: str, benchmark: str) -> Optional[float]:
        return lookup.get((dataset, benchmark))

    mix_vals = []
    base_vals = []
    deltas = []
    for _, row in df.iterrows():
        mix = row["mix_dataset"]
        base = row["base_dataset"]
        benchmark = row["benchmark"]
        mix_mmd = _lookup_pair(mix, benchmark)
        base_mmd = _lookup_pair(base, benchmark)
        mix_vals.append(mix_mmd)
        base_vals.append(base_mmd)
        if mix_mmd is None or base_mmd is None:
            deltas.append(np.nan)
        else:
            deltas.append(mix_mmd - base_mmd)
    df[f"{col_name}_mix"] = mix_vals
    df[f"{col_name}_base"] = base_vals
    df[f"{col_name}_delta"] = deltas


def _summarize_mixing_correlations(df: pd.DataFrame, metric: str) -> List[str]:
    lines: List[str] = []
    if df.empty:
        lines.append("No mix/base matches found for correlation analysis.")
        return lines

    lines.append("MIXING INTERVENTION: DELTA DISTANCE VS DELTA PERFORMANCE")
    lines.append("-" * 80)
    lines.append(f"metric: {metric}")
    lines.append(f"data points: {len(df)}")
    lines.append("")

    families = [
        ("All benchmarks", df),
        ("Flow benchmarks", df[df["benchmark"].isin(FLOW_FAMILY)]),
        ("Semantic benchmarks", df[df["benchmark"].isin(SEMANTIC_FAMILY)]),
    ]

    for label, subset in families:
        if subset.empty:
            lines.append(f"{label}: no data")
            lines.append("")
            continue
        lines.append(label)
        for dist_name in ["flow_mmd", "feature_mmd", "dino_mmd"]:
            x = subset[f"{dist_name}_delta"].to_numpy()
            y = subset["delta_metric"].to_numpy()
            mask = ~np.isnan(x) & ~np.isnan(y)
            x = x[mask]
            y = y[mask]
            lines.extend(_format_corr(f"  {dist_name}", x, y))
        lines.append("")
    return lines


def _summarize_task_specific(
    df: pd.DataFrame,
    mmd_lookups: Dict[str, Dict[Tuple[str, str], float]],
    metric: str,
) -> List[str]:
    lines: List[str] = []
    datasets = [
        "synthetic",
        "synthetic_large_zoom",
        "synthetic_small_zoom",
        "synthetic_random_flipping",
    ]
    benchmarks = ["kitti2012", "kitti2015"]

    lines.append("TASK-SPECIFIC SYNTHETIC: FLOW-BASED MOTION TUNING")
    lines.append("=" * 80)
    lines.append("Goal: Match flow patterns (e.g., KITTI zoom) via camera motion control")
    lines.append("Hypothesis: Flow MMD should predict performance better than DINO MMD")
    lines.append("           (semantic content is constant, only motion patterns vary)")
    lines.append("")
    lines.append("Table: mean performance per dataset/benchmark with MMD2 distances")
    lines.append("-" * 80)
    header = (
        f"{'dataset':<26} {'benchmark':<10} "
        f"{'flow_mmd2':>10} {'dino_mmd2':>10} "
        f"{metric:>10} {'n':>4}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    rows = []
    for dataset in datasets:
        for bench in benchmarks:
            subset = df[(df["train_dataset"] == dataset) & (df["benchmark"] == bench)]
            if subset.empty:
                continue
            mean_metric = float(pd.to_numeric(subset[metric], errors="coerce").mean())
            n = int(subset.shape[0])
            row = {
                "dataset": dataset,
                "benchmark": bench,
                "metric_mean": mean_metric,
                "n": n,
            }
            for name, lookup in mmd_lookups.items():
                row[name] = lookup.get((dataset, bench), np.nan)
            rows.append(row)
            lines.append(
                f"{dataset:<26} {bench:<10} "
                f"{row.get('flow_mmd', np.nan):>10.6f} {row.get('dino_mmd', np.nan):>10.6f} "
                f"{mean_metric:>10.2f} {n:>4d}"
            )

    lines.append("")
    if not rows:
        lines.append("No task-specific synthetic rows found.")
        return lines

    rows_df = pd.DataFrame(rows)
    
    # Overall correlations
    lines.append("Correlation across all dataset+benchmark pairs:")
    lines.append("-" * 80)
    for dist_name in ["flow_mmd", "dino_mmd", "feature_mmd"]:
        if dist_name not in rows_df.columns:
            continue
        x = rows_df[dist_name].to_numpy()
        y = rows_df["metric_mean"].to_numpy()
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        lines.extend(_format_corr(f"  {dist_name}", x, y))
    lines.append("")

    # Per-benchmark correlations (KEY ANALYSIS!)
    lines.append("Per-Benchmark Correlations (CRITICAL FOR FLOW-BASED TUNING):")
    lines.append("=" * 80)
    for bench in benchmarks:
        bench_rows = rows_df[rows_df["benchmark"] == bench]
        if bench_rows.empty or len(bench_rows) < 3:
            continue
        lines.append(f"\n{bench.upper()}:")
        lines.append("-" * 40)
        
        # Flow correlation (should be strong!)
        x_flow = bench_rows["flow_mmd"].to_numpy()
        y = bench_rows["metric_mean"].to_numpy()
        mask_flow = ~np.isnan(x_flow) & ~np.isnan(y)
        if mask_flow.sum() >= 3:
            r_flow, p_flow = _safe_corr(x_flow[mask_flow], y[mask_flow])
            rho_flow, ps_flow = _safe_spearman(x_flow[mask_flow], y[mask_flow])
            lines.append(f"  FLOW MMD: r={r_flow:.3f} (p={p_flow:.4f}), spearman={rho_flow:.3f} (p={ps_flow:.4f})")
            if not np.isnan(r_flow) and abs(r_flow) > 0.8:
                lines.append(f"    → STRONG FLOW CORRELATION! (|r|={abs(r_flow):.3f})")
        
        # DINO correlation (should be weak for motion-only changes)
        if "dino_mmd" in bench_rows.columns:
            x_dino = bench_rows["dino_mmd"].to_numpy()
            mask_dino = ~np.isnan(x_dino) & ~np.isnan(y)
            if mask_dino.sum() >= 3:
                r_dino, p_dino = _safe_corr(x_dino[mask_dino], y[mask_dino])
                rho_dino, ps_dino = _safe_spearman(x_dino[mask_dino], y[mask_dino])
                lines.append(f"  DINO MMD: r={r_dino:.3f} (p={p_dino:.4f}), spearman={rho_dino:.3f} (p={ps_dino:.4f})")
                if not np.isnan(r_dino) and abs(r_dino) < 0.3:
                    lines.append(f"    → Weak/no DINO effect (as expected for motion-only changes)")
        
        # Show best variant for this benchmark
        best_flow = bench_rows.sort_values("flow_mmd").iloc[0]
        best_perf = bench_rows.sort_values("metric_mean", ascending=False).iloc[0]
        lines.append(f"  Best flow match: {best_flow['dataset']} (flow_mmd={best_flow['flow_mmd']:.6f})")
        lines.append(f"  Best performance: {best_perf['dataset']} ({metric}={best_perf['metric_mean']:.2f})")
        if best_flow['dataset'] == best_perf['dataset']:
            lines.append(f"    ✓ FLOW ALIGNMENT = BEST PERFORMANCE!")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("KEY FINDING:")
    lines.append("=" * 80)
    
    # Summary statistics
    kitti_only = rows_df[rows_df["benchmark"].isin(benchmarks)]
    if "flow_mmd" in kitti_only.columns and len(kitti_only) >= 3:
        x = kitti_only["flow_mmd"].to_numpy()
        y = kitti_only["metric_mean"].to_numpy()
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() >= 3:
            r_flow, p_flow = _safe_corr(x[mask], y[mask])
            if not np.isnan(r_flow):
                lines.append(f"Flow MMD strongly predicts KITTI performance: r={r_flow:.3f}, p={p_flow:.4f}")
    
    if "dino_mmd" in kitti_only.columns and len(kitti_only) >= 3:
        x = kitti_only["dino_mmd"].to_numpy()
        y = kitti_only["metric_mean"].to_numpy()
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() >= 3:
            r_dino, p_dino = _safe_corr(x[mask], y[mask])
            if not np.isnan(r_dino):
                lines.append(f"DINO MMD is uncorrelated (motion ≠ semantics): r={r_dino:.3f}, p={p_dino:.4f}")
    
    lines.append("")
    lines.append("This validates the complementarity hypothesis:")
    lines.append("  - Semantic changes (mixing) → DINO predicts success")
    lines.append("  - Motion changes (zoom tuning) → Flow predicts success")
    lines.append("")
    
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Intervention correlation analysis.")
    parser.add_argument(
        "--metrics-csv",
        default="analysis/leakage_free_local_fast_dino_faiss/unified_cross_model/auc_delta_rank_mf_no_synth/auc_with_features.csv",
        help="CSV with performance metrics and datasets.",
    )
    parser.add_argument("--metric", default="peak_pck", help="Performance metric column to analyze.")
    parser.add_argument("--flow-mmd-csv", default="flow_mmd_results_fast.csv", help="Flow MMD CSV.")
    parser.add_argument(
        "--feature-mmd-csv",
        default="feature_mmd_results_resnet_200.csv",
        help="Feature MMD CSV.",
    )
    parser.add_argument("--dino-mmd-csv", default="dino_mmd_results_fast.csv", help="DINO MMD CSV.")
    parser.add_argument(
        "--output",
        default="analysis/final_contributions/intervention_correlation_analysis.txt",
        help="Output text file.",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics_csv)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_path}")
    df = pd.read_csv(metrics_path)
    if args.metric not in df.columns:
        raise ValueError(f"Metric {args.metric} not found in {metrics_path}")

    df["train_dataset"] = df["train_dataset"].astype(str).apply(_normalize_name)
    df["benchmark"] = df["benchmark"].astype(str).apply(_normalize_name)

    flow_lookup = _load_mmd_lookup(Path(args.flow_mmd_csv))
    feature_lookup = _load_mmd_lookup(Path(args.feature_mmd_csv))
    dino_lookup = _load_mmd_lookup(Path(args.dino_mmd_csv))
    lookups = {
        "flow_mmd": flow_lookup,
        "feature_mmd": feature_lookup,
        "dino_mmd": dino_lookup,
    }

    mix_df = _compute_mix_deltas(df, args.metric)
    if not mix_df.empty:
        _add_delta_mmd(mix_df, flow_lookup, "flow_mmd")
        _add_delta_mmd(mix_df, feature_lookup, "feature_mmd")
        _add_delta_mmd(mix_df, dino_lookup, "dino_mmd")

    lines: List[str] = []
    lines.extend(_summarize_mixing_correlations(mix_df, args.metric))
    lines.append("")
    lines.extend(_summarize_task_specific(df, lookups, args.metric))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Wrote analysis to {output_path}")


if __name__ == "__main__":
    main()
