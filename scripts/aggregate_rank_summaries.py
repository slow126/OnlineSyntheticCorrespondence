#!/usr/bin/env python3
"""
Aggregate LOBO/LOTO rank summaries without blurring model-family differences.

Outputs:
  - rank_rows.csv: all per-benchmark rows with metadata
  - rank_overall_by_run.csv: __overall__ rows per run
  - rank_overall_by_group.csv: robust aggregates (median/IQR) by fold/model/encoder
  - rank_by_benchmark_by_group.csv: robust aggregates per benchmark by fold/model/encoder
  - rank_pred_best_counts.csv: counts of predicted-best options per group/benchmark
  - rank_true_best_counts.csv: counts of true-best options per group/benchmark
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_name(value):
    if value is None:
        return None
    return str(value).strip().lower().replace("+", "_")


def infer_model_family(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    if "raft" in parts:
        return "raft"
    if any("flowformer" in p for p in parts):
        return "flowformer"
    if "catspp" in parts or "cats" in parts:
        return "catspp"
    if "unified_cross_model" in parts:
        return "mixed"
    return "unknown"


def infer_encoder_from_path(path: Path) -> tuple[str, str]:
    parts = list(path.parts)
    if "by_encoder" not in parts:
        return "all", "all"
    idx = parts.index("by_encoder")
    if idx + 1 >= len(parts):
        return "all", "all"
    tag = parts[idx + 1]
    lower = tag.lower()
    pre = "T" if "pretrainedtrue" in lower else "F" if "pretrainedfalse" in lower else "U"
    frz = "T" if "freezetrue" in lower else "F" if "freezefalse" in lower else "U"
    if pre in {"T", "F"} and frz in {"T", "F"}:
        return tag, f"{pre}{frz}"
    return tag, "all"


def discover_files(inputs, filename):
    seen = set()
    for raw in inputs:
        path = Path(raw)
        if path.is_file() and path.name == filename:
            if path not in seen:
                seen.add(path)
                yield path
        elif path.is_dir():
            for match in path.rglob(filename):
                if match not in seen:
                    seen.add(match)
                    yield match


def load_rank_rows(paths, fold_label):
    rows = []
    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "benchmark" not in df.columns:
            continue
        model_family = infer_model_family(path)
        encoder_tag, encoder_config = infer_encoder_from_path(path)
        df = df.copy()
        df["benchmark"] = df["benchmark"].astype(str)
        df["fold"] = fold_label
        df["summary_path"] = str(path)
        df["run_dir"] = str(path.parent)
        df["model_family"] = model_family
        df["encoder_tag"] = encoder_tag
        df["encoder_config"] = encoder_config
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_prediction_overall(paths, fold_label):
    rows = []
    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        name_col = None
        if "benchmark" in df.columns:
            name_col = "benchmark"
        elif "train_dataset" in df.columns:
            name_col = "train_dataset"
        if name_col is None:
            continue
        overall = df[df[name_col] == "__overall__"].copy()
        if overall.empty:
            continue
        model_family = infer_model_family(path)
        encoder_tag, encoder_config = infer_encoder_from_path(path)
        overall["fold"] = fold_label
        overall["summary_path"] = str(path)
        overall["run_dir"] = str(path.parent)
        overall["model_family"] = model_family
        overall["encoder_tag"] = encoder_tag
        overall["encoder_config"] = encoder_config
        rows.append(overall)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_groups(df, group_cols, output_path):
    if df.empty:
        return pd.DataFrame()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return pd.DataFrame()
    summary = df.groupby(group_cols)[numeric_cols].agg(["median", "mean", "std", "min", "max"])
    summary = summary.reset_index()
    summary.columns = [
        "_".join([str(c) for c in col if c]).rstrip("_")
        for col in summary.columns.to_flat_index()
    ]
    run_counts = df.groupby(group_cols)["run_dir"].nunique().reset_index(name="n_runs")
    summary = summary.merge(run_counts, on=group_cols, how="left")
    summary.to_csv(output_path, index=False)
    return summary


def summarize_two_stage(df, group_cols, output_path):
    if df.empty:
        return pd.DataFrame()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return pd.DataFrame()
    grouped = df.groupby(group_cols)[numeric_cols].median().reset_index()
    summary = grouped.groupby([group_cols[0]])[numeric_cols].agg(["median", "mean", "std", "min", "max"])
    summary = summary.reset_index()
    summary.columns = [
        "_".join([str(c) for c in col if c]).rstrip("_")
        for col in summary.columns.to_flat_index()
    ]
    group_counts = grouped.groupby(group_cols[0]).size().reset_index(name="n_groups")
    summary = summary.merge(group_counts, on=group_cols[0], how="left")
    summary.to_csv(output_path, index=False)
    return summary


def bootstrap_ci(values, stat_fn, n_boot=2000, alpha=0.05, seed=17):
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(vals, size=vals.size, replace=True)
        stats.append(stat_fn(sample))
    stats = np.asarray(stats, dtype=float)
    stats = stats[~np.isnan(stats)]
    if stats.size == 0:
        return np.nan, np.nan, np.nan
    lo = np.quantile(stats, alpha / 2.0)
    hi = np.quantile(stats, 1.0 - alpha / 2.0)
    return float(stat_fn(vals)), float(lo), float(hi)


def format_metric_line(name, values):
    median, lo, hi = bootstrap_ci(values, np.nanmedian)
    mean = float(np.nanmean(values)) if len(values) else np.nan
    pos_frac = float(np.mean(np.asarray(values) > 0)) if len(values) else np.nan
    return f"{name}: median={median:.3f} (CI [{lo:.3f},{hi:.3f}]), mean={mean:.3f}, pos_frac={pos_frac:.2f}"


def main():
    parser = argparse.ArgumentParser(description="Aggregate rank summaries across runs.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["analysis"],
        help="Directories (or files) to search for rank summary CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/rank_aggregate",
        help="Directory to write aggregate CSVs.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lobo_paths = list(discover_files(args.inputs, "prediction_lobo_rank_summary.csv"))
    loto_paths = list(discover_files(args.inputs, "prediction_loto_rank_summary.csv"))
    lobo_pred_paths = list(discover_files(args.inputs, "prediction_lobo_summary.csv"))
    loto_pred_paths = list(discover_files(args.inputs, "prediction_loto_summary.csv"))

    lobo_df = load_rank_rows(lobo_paths, "lobo")
    loto_df = load_rank_rows(loto_paths, "loto")
    all_rows = pd.concat([lobo_df, loto_df], ignore_index=True)
    if all_rows.empty:
        print("No rank summaries found.")
        return

    all_rows.to_csv(out_dir / "rank_rows.csv", index=False)

    overall_rows = all_rows[all_rows["benchmark"] == "__overall__"].copy()
    if not overall_rows.empty:
        overall_rows.to_csv(out_dir / "rank_overall_by_run.csv", index=False)

    group_cols = ["fold", "model_family", "encoder_config"]
    if not overall_rows.empty:
        summarize_groups(overall_rows, group_cols, out_dir / "rank_overall_by_group.csv")
        summarize_groups(overall_rows, ["fold"], out_dir / "rank_overall_by_fold.csv")
        summarize_two_stage(
            overall_rows,
            ["fold", "model_family", "encoder_config"],
            out_dir / "rank_overall_by_fold_two_stage.csv",
        )

    bench_rows = all_rows[all_rows["benchmark"] != "__overall__"].copy()
    if not bench_rows.empty:
        summarize_groups(
            bench_rows,
            group_cols + ["benchmark"],
            out_dir / "rank_by_benchmark_by_group.csv",
        )

        pred_counts = (
            bench_rows.groupby(group_cols + ["benchmark", "pred_best_option"])
            .size()
            .reset_index(name="pred_best_count")
        )
        pred_counts.to_csv(out_dir / "rank_pred_best_counts.csv", index=False)

        true_counts = (
            bench_rows.groupby(group_cols + ["benchmark", "true_best_option"])
            .size()
            .reset_index(name="true_best_count")
        )
        true_counts.to_csv(out_dir / "rank_true_best_counts.csv", index=False)

    pred_overall_rows = pd.concat(
        [
            load_prediction_overall(lobo_pred_paths, "lobo"),
            load_prediction_overall(loto_pred_paths, "loto"),
        ],
        ignore_index=True,
    )
    if not pred_overall_rows.empty:
        pred_overall_rows.to_csv(out_dir / "pred_overall_by_run.csv", index=False)
        summarize_groups(
            pred_overall_rows,
            group_cols,
            out_dir / "pred_overall_by_group.csv",
        )
        summarize_groups(
            pred_overall_rows,
            ["fold"],
            out_dir / "pred_overall_by_fold.csv",
        )
        summarize_two_stage(
            pred_overall_rows,
            ["fold", "model_family", "encoder_config"],
            out_dir / "pred_overall_by_fold_two_stage.csv",
        )

    summary_lines = []
    summary_lines.append("Aggregate LOBO/LOTO summary (across all model families and encoder regimes)")
    summary_lines.append("======================================================================")
    if not overall_rows.empty:
        summary_lines.append("")
        summary_lines.append("Ranking metrics (from prediction_*_rank_summary.csv, __overall__ rows):")
        for fold in ("lobo", "loto"):
            sub = overall_rows[overall_rows["fold"] == fold]
            summary_lines.append(f"  {fold.upper()} (n_runs={sub['run_dir'].nunique()}):")
            for metric in ("top1", "top3", "topk", "spearman", "regret", "mean_abs_rank_error"):
                if metric not in sub.columns:
                    continue
                summary_lines.append(
                    "    "
                    + format_metric_line(
                        metric, sub[metric].dropna().to_numpy(dtype=float)
                    )
                )
        summary_lines.append("")
        summary_lines.append("Ranking metrics (two-stage: median per family/regime, then aggregate):")
        grouped = overall_rows.groupby(["fold", "model_family", "encoder_config"])
        grouped_med = grouped.median(numeric_only=True).reset_index()
        for fold in ("lobo", "loto"):
            sub = grouped_med[grouped_med["fold"] == fold]
            summary_lines.append(f"  {fold.upper()} (n_groups={len(sub)}):")
            for metric in ("top1", "top3", "topk", "spearman", "regret", "mean_abs_rank_error"):
                if metric not in sub.columns:
                    continue
                summary_lines.append(
                    "    "
                    + format_metric_line(
                        metric, sub[metric].dropna().to_numpy(dtype=float)
                    )
                )
    if not pred_overall_rows.empty:
        summary_lines.append("")
        summary_lines.append("Prediction metrics (from prediction_*_summary.csv, __overall__ rows):")
        for fold in ("lobo", "loto"):
            sub = pred_overall_rows[pred_overall_rows["fold"] == fold]
            summary_lines.append(f"  {fold.upper()} (n_runs={sub['run_dir'].nunique()}):")
            for metric in ("pearson", "spearman", "mae", "rmse"):
                if metric not in sub.columns:
                    continue
                summary_lines.append(
                    "    "
                    + format_metric_line(
                        metric, sub[metric].dropna().to_numpy(dtype=float)
                    )
                )
        summary_lines.append("")
        summary_lines.append("Prediction metrics (two-stage: median per family/regime, then aggregate):")
        grouped = pred_overall_rows.groupby(["fold", "model_family", "encoder_config"])
        grouped_med = grouped.median(numeric_only=True).reset_index()
        for fold in ("lobo", "loto"):
            sub = grouped_med[grouped_med["fold"] == fold]
            summary_lines.append(f"  {fold.upper()} (n_groups={len(sub)}):")
            for metric in ("pearson", "spearman", "mae", "rmse"):
                if metric not in sub.columns:
                    continue
                summary_lines.append(
                    "    "
                    + format_metric_line(
                        metric, sub[metric].dropna().to_numpy(dtype=float)
                    )
                )

    if summary_lines:
        (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
