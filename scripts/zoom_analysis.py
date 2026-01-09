#!/usr/bin/env python3
"""
Analyze metric shifts for synthetic zoom datasets vs base synthetic.

This script compares synthetic_large_zoom and synthetic_small_zoom to a base
synthetic dataset across selected benchmarks. It reports whether predictor
metrics move in a direction consistent with performance improvements.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_AUC_TABLE = (
    "analysis/leakage_free_local_fast_dino_faiss/"
    "all/peak_pck_rank_trimmed/full_interactions/auc_with_features.csv"
)
DEFAULT_BENCHMARKS = ["kitti2012", "kitti2015", "pfpascal", "pfwillow", "tss"]
DEFAULT_ZOOM_DATASETS = ["synthetic_large_zoom", "synthetic_small_zoom"]
DEFAULT_BASE_DATASET = "synthetic"


EXCLUDE_EXACT = {
    "run_id",
    "run_name",
    "train_dataset",
    "step_tag",
    "logsteps",
    "pretrained",
    "freeze",
    "timestamp",
    "model_family",
    "benchmark",
    "auc_steps",
    "auc_points",
    "peak_training_steps",
    "peak_epoch",
    "final_training_steps",
    "final_epoch",
    "baseline_value",
    "baseline_train_dataset",
    "encoder_config",
}

EXCLUDE_SUBSTRINGS = [
    "pck",
    "auc",
    "rank",
    "epoch",
    "steps",
    "baseline",
]


def _rename_coverage_predictor(name: str) -> str:
    if not isinstance(name, str):
        return name
    replacements = {
        "flow_eval_to_train_coverage": "flow_eval_to_train_over_train_precision",
        "flow_eval_to_train_coverage_logit": "flow_eval_to_train_over_train_precision_logit",
        "flow_train_to_eval_coverage": "flow_train_to_eval_over_eval_recall",
        "flow_train_to_eval_coverage_logit": "flow_train_to_eval_over_eval_recall_logit",
        "resnet_eval_to_train_coverage": "resnet_eval_to_train_over_train_precision",
        "resnet_eval_to_train_coverage_logit": "resnet_eval_to_train_over_train_precision_logit",
        "resnet_train_to_eval_coverage": "resnet_train_to_eval_over_eval_recall",
        "resnet_train_to_eval_coverage_logit": "resnet_train_to_eval_over_eval_recall_logit",
        "dino_eval_to_train_coverage": "dino_eval_to_train_over_train_precision",
        "dino_eval_to_train_coverage_logit": "dino_eval_to_train_over_train_precision_logit",
        "dino_train_to_eval_coverage": "dino_train_to_eval_over_eval_recall",
        "dino_train_to_eval_coverage_logit": "dino_train_to_eval_over_eval_recall_logit",
    }
    return replacements.get(name, name)


def _resolve_paths(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        expanded = list(Path(".").glob(pattern))
        if expanded:
            paths.extend(expanded)
        else:
            p = Path(pattern)
            if p.exists():
                paths.append(p)
    unique = sorted({p.resolve() for p in paths})
    if not unique:
        raise SystemExit("No auc_with_features.csv files found from provided patterns.")
    return unique


def _direction_sign(name: str) -> Optional[int]:
    lname = name.lower()
    if "coverage" in lname or "recall" in lname or "precision" in lname:
        if "outside" in lname:
            return -1
        return 1
    if "outside" in lname:
        return -1
    if "mmd" in lname:
        return -1
    if "kl_div" in lname or "kld" in lname:
        return -1
    if "mean_dist" in lname or "median_dist" in lname or "p90_dist" in lname:
        return -1
    if "dist_over_radius" in lname:
        return -1
    if "asymmetry" in lname:
        return -1
    return None


def _expected_direction(sign: Optional[int]) -> str:
    if sign is None:
        return "unknown"
    return "increase" if sign > 0 else "decrease"


def _is_interaction(col: str) -> bool:
    return col.startswith("enc_") or col.startswith("mf_") or "__enc_" in col or "__mf_" in col


def _auto_predictors(
    df: pd.DataFrame,
    performance_metric: str,
    include_interactions: bool,
    include_radius: bool,
    include_k: bool,
) -> List[str]:
    predictors: List[str] = []
    for col in df.columns:
        if col == performance_metric:
            continue
        if col in EXCLUDE_EXACT:
            continue
        if any(token in col for token in EXCLUDE_SUBSTRINGS):
            continue
        if not include_interactions and _is_interaction(col):
            continue
        if not include_radius and "radius" in col:
            continue
        if not include_k and (col.endswith("_k") or col in ("flow_k", "resnet_k", "dino_k")):
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            predictors.append(col)
    return sorted(set(predictors))


def _parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    return float(values.mean()) if values.notna().any() else float("nan")


def _format_ratio(numer: int, denom: int) -> str:
    if denom == 0:
        return "n/a"
    return f"{numer}/{denom} ({numer / denom:.2f})"


def _format_value(value: float) -> str:
    if np.isnan(value):
        return "  n/a"
    return f"{value: .4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare synthetic zoom datasets to base synthetic and summarize predictor shifts.")
    parser.add_argument(
        "--auc-table",
        nargs="+",
        default=[DEFAULT_AUC_TABLE],
        help="One or more auc_with_features.csv paths or glob patterns.",
    )
    parser.add_argument(
        "--benchmarks",
        default=",".join(DEFAULT_BENCHMARKS),
        help="Comma-separated list of benchmarks to include.",
    )
    parser.add_argument(
        "--zoom-datasets",
        default=",".join(DEFAULT_ZOOM_DATASETS),
        help="Comma-separated list of zoom datasets to compare.",
    )
    parser.add_argument(
        "--base-dataset",
        default=DEFAULT_BASE_DATASET,
        help="Base synthetic dataset name.",
    )
    parser.add_argument(
        "--performance-metric",
        default="peak_pck",
        help="Performance metric column to compare.",
    )
    parser.add_argument(
        "--predictors",
        default="",
        help="Comma-separated list of predictor columns (default: auto-detect).",
    )
    parser.add_argument(
        "--group-cols",
        default="",
        help="Comma-separated list of extra group-by columns (e.g., model_family,encoder_config).",
    )
    parser.add_argument(
        "--include-interactions",
        action="store_true",
        help="Include interaction columns like __enc_ and __mf_.",
    )
    parser.add_argument(
        "--include-radius",
        action="store_true",
        help="Include radius-related columns in predictors.",
    )
    parser.add_argument(
        "--include-k",
        action="store_true",
        help="Include k-related columns in predictors.",
    )
    parser.add_argument(
        "--rename-coverage",
        action="store_true",
        help="Rename coverage predictors to explicit precision/recall labels in outputs.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-9,
        help="Tolerance for detecting non-zero deltas.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/zoom_analysis",
        help="Output directory for CSV/TXT summaries.",
    )
    args = parser.parse_args()

    auc_paths = _resolve_paths(args.auc_table)
    frames = []
    for path in auc_paths:
        df = pd.read_csv(path)
        df["source_path"] = str(path)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    if "benchmark" not in df.columns or "train_dataset" not in df.columns:
        raise SystemExit("Missing required columns: benchmark and train_dataset.")

    benchmarks = _parse_csv_list(args.benchmarks)
    zoom_datasets = _parse_csv_list(args.zoom_datasets)
    if not benchmarks:
        raise SystemExit("No benchmarks provided.")
    if not zoom_datasets:
        raise SystemExit("No zoom datasets provided.")

    perf_col = args.performance_metric
    if perf_col not in df.columns:
        raise SystemExit(f"Performance metric column not found: {perf_col}")

    group_cols = ["benchmark"]
    group_cols.extend(_parse_csv_list(args.group_cols))

    df = df.copy()
    df["benchmark"] = df["benchmark"].astype(str)
    df["train_dataset"] = df["train_dataset"].astype(str)

    df = df[df["benchmark"].isin(benchmarks)]
    df = df[df["train_dataset"].isin([args.base_dataset] + zoom_datasets)]
    if df.empty:
        raise SystemExit("No rows remaining after filtering benchmarks/datasets.")

    if args.predictors:
        predictors = _parse_csv_list(args.predictors)
    else:
        predictors = _auto_predictors(
            df,
            perf_col,
            include_interactions=args.include_interactions,
            include_radius=args.include_radius,
            include_k=args.include_k,
        )

    if not predictors:
        raise SystemExit("No predictors selected.")

    for col in [perf_col] + predictors:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    rows = []
    for key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        group_info = dict(zip(group_cols, key))

        base_df = group[group["train_dataset"] == args.base_dataset]
        if base_df.empty:
            continue
        base_perf = _safe_mean(base_df[perf_col])

        for zoom in zoom_datasets:
            zoom_df = group[group["train_dataset"] == zoom]
            if zoom_df.empty:
                continue
            zoom_perf = _safe_mean(zoom_df[perf_col])
            perf_delta = zoom_perf - base_perf
            improved = perf_delta > args.epsilon
            for predictor in predictors:
                base_val = _safe_mean(base_df[predictor])
                zoom_val = _safe_mean(zoom_df[predictor])
                delta = zoom_val - base_val
                sign = _direction_sign(predictor)
                signed_delta = delta * sign if sign is not None else float("nan")
                movement = "unknown"
                aligned = None
                if sign is not None and not np.isnan(delta):
                    if signed_delta > args.epsilon:
                        movement = "toward_expected"
                    elif signed_delta < -args.epsilon:
                        movement = "away_from_expected"
                    else:
                        movement = "flat"
                if (
                    sign is not None
                    and not np.isnan(perf_delta)
                    and abs(perf_delta) > args.epsilon
                    and not np.isnan(signed_delta)
                ):
                    aligned = (perf_delta > 0 and signed_delta > args.epsilon) or (
                        perf_delta < 0 and signed_delta < -args.epsilon
                    )
                row = {
                    **group_info,
                    "zoom_dataset": zoom,
                    "base_dataset": args.base_dataset,
                    "performance_metric": perf_col,
                    "performance_zoom": zoom_perf,
                    "performance_base": base_perf,
                    "performance_delta": perf_delta,
                    "improved": improved,
                    "predictor": predictor,
                    "predictor_base": base_val,
                    "predictor_zoom": zoom_val,
                    "predictor_delta": delta,
                    "direction_sign": sign,
                    "expected_direction": _expected_direction(sign),
                    "signed_delta": signed_delta,
                    "movement": movement,
                    "aligned_with_perf": aligned,
                }
                rows.append(row)

    if not rows:
        raise SystemExit("No comparison rows produced. Check dataset/benchmark filters.")

    rows_df = pd.DataFrame(rows)
    if args.rename_coverage:
        rows_df["predictor_raw"] = rows_df["predictor"]
        rows_df["predictor"] = rows_df["predictor"].apply(_rename_coverage_predictor)

    summary_rows = []
    for (zoom, predictor), sub in rows_df.groupby(["zoom_dataset", "predictor"], dropna=False):
        sub = sub.copy()
        predictor_raw = None
        if "predictor_raw" in sub.columns:
            predictor_raw = sub["predictor_raw"].iloc[0]
        improved = sub["improved"] == True
        aligned = sub["aligned_with_perf"] == True
        has_dir = sub["direction_sign"].notna()
        n_improved = int(improved.sum())
        n_aligned_improved = int((improved & aligned).sum())
        n_total = int((has_dir).sum())
        n_aligned_total = int((aligned & has_dir).sum())
        summary_rows.append(
            {
                "zoom_dataset": zoom,
                "predictor": predictor,
                "predictor_raw": predictor_raw,
                "direction_sign": sub["direction_sign"].iloc[0],
                "n_total": n_total,
                "n_aligned_total": n_aligned_total,
                "aligned_rate_total": n_aligned_total / n_total if n_total else float("nan"),
                "n_improved": n_improved,
                "n_aligned_improved": n_aligned_improved,
                "aligned_rate_improved": (
                    n_aligned_improved / n_improved if n_improved else float("nan")
                ),
                "mean_perf_delta": float(sub["performance_delta"].mean()),
                "mean_predictor_delta": float(sub["predictor_delta"].mean()),
                "mean_signed_delta": float(sub["signed_delta"].mean()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    bench_rows = []
    bench_groups = ["zoom_dataset"] + group_cols
    for key, sub in rows_df.groupby(bench_groups, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        info = dict(zip(bench_groups, key))
        perf_delta = sub["performance_delta"].iloc[0]
        valid = sub[sub["direction_sign"].notna()]
        bench_rows.append(
            {
                **info,
                "performance_delta": perf_delta,
                "n_predictors": int(len(valid)),
                "n_toward_expected": int((valid["movement"] == "toward_expected").sum()),
                "n_away_expected": int((valid["movement"] == "away_from_expected").sum()),
                "n_flat": int((valid["movement"] == "flat").sum()),
                "n_unknown": int((valid["movement"] == "unknown").sum()),
                "mean_signed_delta": float(valid["signed_delta"].mean()),
            }
        )

    bench_df = pd.DataFrame(bench_rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = out_dir / "zoom_analysis_rows.csv"
    summary_csv = out_dir / "zoom_analysis_summary.csv"
    bench_csv = out_dir / "zoom_analysis_benchmark_summary.csv"
    rows_df.to_csv(rows_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    bench_df.to_csv(bench_csv, index=False)

    lines: List[str] = []
    lines.append("ZOOM ANALYSIS")
    lines.append("=" * 80)
    lines.append("AUC tables:")
    for path in auc_paths:
        lines.append(f"  - {path}")
    lines.append(f"Benchmarks: {', '.join(benchmarks)}")
    lines.append(f"Base dataset: {args.base_dataset}")
    lines.append(f"Zoom datasets: {', '.join(zoom_datasets)}")
    lines.append(f"Performance metric: {perf_col}")
    lines.append(f"Predictors: {len(predictors)}")
    lines.append("")
    lines.append("Definitions:")
    lines.append("  delta = zoom - base")
    lines.append("  expected_direction: increase for coverage/recall/precision, decrease for distances/MMD/outside")
    lines.append("  signed_delta = delta * direction_sign (+1 increase, -1 decrease); positive means toward expected")
    lines.append("  movement: toward_expected / away_from_expected / flat (within epsilon) / unknown")
    lines.append("  aligned_with_perf: signed_delta moves in same direction as performance_delta")
    if args.rename_coverage:
        lines.append("  coverage rename: eval_to_train => over_train_precision, train_to_eval => over_eval_recall")
    lines.append("")

    for zoom in zoom_datasets:
        sub = rows_df[rows_df["zoom_dataset"] == zoom]
        perf_rows = sub.drop_duplicates(bench_groups)
        improved = perf_rows[perf_rows["performance_delta"] > args.epsilon]
        improved_benchmarks = improved["benchmark"].tolist() if "benchmark" in improved else []
        improved_benchmarks = sorted(set(improved_benchmarks))
        lines.append("")
        lines.append(f"Zoom dataset: {zoom}")
        lines.append("-" * 80)
        if improved_benchmarks:
            lines.append(f"Improved benchmarks: {', '.join(improved_benchmarks)}")
        else:
            lines.append("Improved benchmarks: none")

        zoom_summary = summary_df[summary_df["zoom_dataset"] == zoom].copy()
        zoom_summary = zoom_summary.sort_values(
            ["n_aligned_improved", "mean_predictor_delta"],
            ascending=[False, False],
            key=lambda col: col.abs() if col.name == "mean_predictor_delta" else col,
        )
        lines.append("Predictor summary (sorted by mean_signed_delta):")
        if zoom_summary.empty:
            lines.append("  (none)")
        else:
            max_pred_len = max(len(str(val)) for val in zoom_summary["predictor"])
            pred_width = max(32, min(64, max_pred_len + 2))
            lines.append(
                f"{'predictor':<{pred_width}}  {'aligned_impr':>14}  {'aligned_total':>14}  {'mean_delta':>12}"
            )
            lines.append("-" * 80)
            for _, row in zoom_summary.iterrows():
                lines.append(
                    f"{row['predictor']:<{pred_width}}  "
                    f"{_format_ratio(int(row['n_aligned_improved']), int(row['n_improved'])):>14}  "
                    f"{_format_ratio(int(row['n_aligned_total']), int(row['n_total'])):>14}  "
                    f"{row['mean_predictor_delta']:+12.4f}"
                )

    lines.append("")
    lines.append("Per-benchmark predictor shifts")
    lines.append("=" * 80)
    for key, bench_df in rows_df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        header_parts = [f"{col}={val}" for col, val in zip(group_cols, key)]
        lines.append("")
        lines.append("Benchmark: " + ", ".join(header_parts))
        lines.append("-" * 80)
        for zoom in zoom_datasets:
            sub = bench_df[bench_df["zoom_dataset"] == zoom]
            if sub.empty:
                continue
            max_pred_len = max(len(str(val)) for val in sub["predictor"])
            pred_width = max(32, min(64, max_pred_len + 2))
            perf_delta = sub["performance_delta"].iloc[0]
            improved = perf_delta > args.epsilon
            lines.append(
                f"Zoom dataset: {zoom} | perf_delta={perf_delta:+.4f} | improved={improved}"
            )
            lines.append(
                f"{'predictor':<{pred_width}}  {'expect':>8}  {'base':>10}  {'zoom':>10}  {'delta':>10}  {'signed':>10}  {'movement':>16}"
            )
            lines.append("-" * 80)
            sub = sub.sort_values("predictor")
            for _, row in sub.iterrows():
                lines.append(
                    f"{row['predictor']:<{pred_width}}  "
                    f"{row['expected_direction']:>8}  "
                    f"{_format_value(row['predictor_base']):>10}  "
                    f"{_format_value(row['predictor_zoom']):>10}  "
                    f"{_format_value(row['predictor_delta']):>10}  "
                    f"{_format_value(row['signed_delta']):>10}  "
                    f"{row['movement']:>16}"
                )
            lines.append("")

    lines.append("")
    lines.append(f"Wrote: {rows_csv}")
    lines.append(f"Wrote: {summary_csv}")
    lines.append(f"Wrote: {bench_csv}")

    out_txt = out_dir / "zoom_analysis.txt"
    out_txt.write_text("\n".join(lines))

    print(f"Wrote {rows_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {bench_csv}")
    print(f"Wrote {out_txt}")


if __name__ == "__main__":
    main()
