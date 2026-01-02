#!/usr/bin/env python3
"""
Aggregate training dataset performance across benchmarks and encoder regimes.

Outputs:
  - group_rank_table.csv: per (benchmark, encoder_regime, dataset) score + rank
  - aggregate_by_dataset.csv: aggregate stats (treat mixes as distinct)
  - aggregate_by_dataset_group.csv: aggregate stats (collapse mix ratios)
  - mix_vs_base_summary.csv: mix dataset improvement vs base dataset
  - mix_vs_base_by_base_dataset.csv: mix improvement aggregated by base dataset
"""

import argparse
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def parse_best_performance_from_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        return {}

    best_performance = {}
    try:
        with summary_path.open("r") as f:
            lines = f.readlines()

        in_best_section = False
        for line in lines:
            line = line.strip()
            if "BEST PERFORMANCE PER BENCHMARK:" in line:
                in_best_section = True
                continue
            if in_best_section:
                if line.startswith("-") and len(line) > 10:
                    continue
                if line.startswith("MOTION-AWARE") or line.startswith("TRAINING CONFIGURATION"):
                    break
                if ":" in line and "%" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        benchmark = parts[0].strip()
                        value_part = parts[1].strip()
                        match = re.search(r"(\d+\.?\d*)%", value_part)
                        if match:
                            best_performance[benchmark] = float(match.group(1))
    except Exception as exc:
        print(f"Warning: Could not parse best performance from {summary_path}: {exc}")
        return {}

    return best_performance


def parse_dir_dataset(directory_name: str) -> str | None:
    param_keywords = ["stride", "sequence_length", "freeze", "pretrained", "eval", "steps", "logsteps"]
    parts = directory_name.split("_")
    if not parts:
        return None

    if parts[0].lower() == "synthetic" and len(parts) >= 2:
        dataset_parts = [parts[0]]
        for i in range(1, len(parts)):
            part = parts[i].lower()
            if (
                part in param_keywords
                or part.startswith("stride")
                or part.startswith("sequence")
                or part.startswith("freeze")
                or part.startswith("pretrained")
                or part.startswith("steps")
                or part.startswith("logsteps")
                or (part.isdigit() and i > 1)
            ):
                break
            dataset_parts.append(parts[i])
        return "_".join(dataset_parts)

    mixed_with_percent = re.match(
        r"^([a-zA-Z]+)_([a-zA-Z]+)_(\d+)_(\d+)(?:_|$)", directory_name
    )
    if mixed_with_percent:
        dataset1, dataset2, p1, p2 = mixed_with_percent.groups()
        return f"{dataset1}_{dataset2}_{p1}_{p2}"

    if len(parts) >= 2:
        part2 = parts[1].lower()
        is_part2_param = (
            part2 in param_keywords
            or part2.startswith("stride")
            or part2.startswith("sequence")
            or part2.startswith("freeze")
            or part2.startswith("pretrained")
            or part2.startswith("steps")
            or part2.startswith("logsteps")
            or part2.isdigit()
        )
        if not is_part2_param:
            return f"{parts[0]}_{parts[1]}"

    return parts[0]


def parse_training_dataset(summary_path: Path, snapshot_name: str) -> str | None:
    dataset = None
    if summary_path.exists():
        try:
            with summary_path.open("r") as f:
                for line in f:
                    if line.startswith("Train dataset:"):
                        dataset = line.split("Train dataset:", 1)[1].strip().lower()
                        dataset = dataset.replace("+", "_")
                        break
        except Exception as exc:
            print(f"Warning: Could not parse training dataset from {summary_path}: {exc}")

    dir_dataset = parse_dir_dataset(snapshot_name)
    if dir_dataset:
        dir_dataset = dir_dataset.lower()

    if dataset is None and dir_dataset:
        return dir_dataset

    if dataset and dir_dataset:
        if dir_dataset.startswith("synthetic_") and dataset == "synthetic":
            return dir_dataset
        if (
            "_synthetic_" in dir_dataset
            and re.search(r"_\d+_\d+$", dir_dataset)
            and ("synthetic" in dataset or "+" in dataset)
        ):
            return dir_dataset

    return dataset or dir_dataset


def parse_encoder_regime(snapshot_name: str) -> str:
    name = snapshot_name.lower()
    pretrained = None
    freeze = None

    if "pretrainedtrue" in name:
        pretrained = "pretrainedTrue"
    elif "pretrainedfalse" in name:
        pretrained = "pretrainedFalse"

    if "freezetrue" in name:
        freeze = "freezeTrue"
    elif "freezefalse" in name:
        freeze = "freezeFalse"

    if pretrained and freeze:
        return f"{pretrained}_{freeze}"
    if pretrained or freeze:
        return f"{pretrained or 'pretrainedUnknown'}_{freeze or 'freezeUnknown'}"
    return "unknown"


def is_snapshot_dir(path: Path) -> bool:
    return (path / "training_summary.txt").exists() or (path / "validation_results.csv").exists()


def collect_snapshot_dirs(snapshot_dirs: list[str], max_depth: int = 3) -> list[str]:
    found = []

    def walk(root: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            for subdir in sorted(root.iterdir()):
                if not subdir.is_dir():
                    continue
                if is_snapshot_dir(subdir):
                    found.append(str(subdir))
                else:
                    walk(subdir, depth + 1)
        except PermissionError:
            return

    for root in snapshot_dirs:
        root_path = Path(root).expanduser()
        if root_path.exists() and root_path.is_dir():
            walk(root_path, 0)
        else:
            print(f"Warning: Snapshot directory not found: {root}")

    # de-duplicate
    seen = set()
    unique = []
    for item in found:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def collapse_mix_ratio(name: str) -> str:
    match = re.match(r"^(.*_synthetic)_(\d+_\d+)$", name)
    if match:
        return match.group(1)
    return name


def parse_mix_info(name: str):
    match = re.match(r"^(.+)_synthetic(?:_(\d+_\d+))?$", name)
    if not match:
        return None, None
    base = match.group(1)
    if base == "synthetic":
        return None, None
    ratio = match.group(2) or "50_50"
    return base, ratio


def build_rank_table(df, score_col, topk_frac, topk_min):
    rows = []
    for (benchmark, regime), sub in df.groupby(["benchmark", "encoder_regime"]):
        sub = sub.copy()
        sub["rank"] = sub[score_col].rank(ascending=False, method="min")
        n_options = len(sub)
        denom = max(n_options - 1, 1)
        sub["rank_pct"] = (sub["rank"] - 1.0) / denom
        best_score = float(sub[score_col].max())
        worst_score = float(sub[score_col].min())
        score_range = best_score - worst_score
        sub["regret"] = best_score - sub[score_col]
        sub["rel_regret"] = sub["regret"] / best_score if best_score != 0 else np.nan
        sub["score_vs_best"] = sub[score_col] / best_score if best_score != 0 else np.nan
        sub["score_gap_to_best"] = best_score - sub[score_col]
        if score_range > 0:
            sub["score_range_norm"] = (sub[score_col] - worst_score) / score_range
        else:
            sub["score_range_norm"] = np.nan

        k = max(int(topk_min), int(math.ceil(topk_frac * n_options)))
        k = min(k, n_options)
        sub["top1"] = (sub["rank"] == 1).astype(int)
        sub["top3"] = (sub["rank"] <= 3).astype(int)
        sub["topk"] = (sub["rank"] <= k).astype(int)
        sub["topk_k"] = k
        sub["topk_frac"] = topk_frac

        rows.append(sub)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def aggregate_by_dataset(rank_df, dataset_col):
    agg = rank_df.groupby(dataset_col).agg(
        n_groups=("rank", "count"),
        avg_rank=("rank", "mean"),
        median_rank=("rank", "median"),
        avg_rank_pct=("rank_pct", "mean"),
        median_rank_pct=("rank_pct", "median"),
        top1_rate=("top1", "mean"),
        top3_rate=("top3", "mean"),
        topk_rate=("topk", "mean"),
        mean_regret=("regret", "mean"),
        mean_rel_regret=("rel_regret", "mean"),
        mean_score=("score", "mean"),
        mean_score_vs_best=("score_vs_best", "mean"),
        mean_score_gap_to_best=("score_gap_to_best", "mean"),
        mean_score_range_norm=("score_range_norm", "mean"),
    ).reset_index()
    return agg.sort_values(["avg_rank_pct", "avg_rank"], ascending=[True, True])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate training set performance across benchmarks and encoder regimes."
    )
    parser.add_argument(
        "--snapshots-dir",
        nargs="+",
        required=True,
        help="Snapshot root directories to scan (supports multiple).",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/training_set_aggregate",
        help="Output directory for aggregate CSVs.",
    )
    parser.add_argument(
        "--score",
        choices=["mean", "max", "median"],
        default="mean",
        help="Score to rank datasets within each benchmark+regime.",
    )
    parser.add_argument(
        "--topk-frac",
        type=float,
        default=0.2,
        help="Top-k fraction for topk rate (default: 0.2 = top-20%%).",
    )
    parser.add_argument(
        "--topk-min",
        type=int,
        default=1,
        help="Minimum k for top-k evaluation (default: 1).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Max recursion depth when scanning snapshot dirs (default: 3).",
    )
    parser.add_argument(
        "--summary-max-rows",
        type=int,
        default=0,
        help="Max rows per summary table (0 = show all, default: 0).",
    )
    args = parser.parse_args()

    snapshot_dirs = collect_snapshot_dirs(args.snapshots_dir, max_depth=args.max_depth)
    if not snapshot_dirs:
        print("No snapshot directories found.")
        return

    records = []
    for snapshot_dir in snapshot_dirs:
        snap_path = Path(snapshot_dir)
        summary_path = snap_path / "training_summary.txt"
        dataset = parse_training_dataset(summary_path, snap_path.name)
        if not dataset:
            continue
        regime = parse_encoder_regime(snap_path.name)
        best_perf = parse_best_performance_from_summary(summary_path)
        if not best_perf:
            continue
        for benchmark, best_pck in best_perf.items():
            records.append({
                "benchmark": benchmark,
                "encoder_regime": regime,
                "dataset": dataset,
                "best_pck": float(best_pck),
                "snapshot": snap_path.name,
            })

    if not records:
        print("No benchmark results found.")
        return

    df = pd.DataFrame(records)
    grouped = df.groupby(["benchmark", "encoder_regime", "dataset"]).agg(
        mean=("best_pck", "mean"),
        median=("best_pck", "median"),
        max=("best_pck", "max"),
        n_runs=("best_pck", "count"),
    ).reset_index()
    grouped["score"] = grouped[args.score]

    rank_table = build_rank_table(
        grouped,
        score_col="score",
        topk_frac=args.topk_frac,
        topk_min=args.topk_min,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rank_table.to_csv(output_dir / "group_rank_table.csv", index=False)

    agg_distinct = aggregate_by_dataset(rank_table, "dataset")
    agg_distinct.to_csv(output_dir / "aggregate_by_dataset.csv", index=False)

    rank_table["dataset_group"] = rank_table["dataset"].apply(collapse_mix_ratio)
    agg_grouped = aggregate_by_dataset(rank_table, "dataset_group")
    agg_grouped.to_csv(output_dir / "aggregate_by_dataset_group.csv", index=False)

    mix_df = rank_table.copy()
    mix_df[["base_dataset", "mix_ratio"]] = mix_df["dataset"].apply(
        lambda name: pd.Series(parse_mix_info(name))
    )
    mix_df = mix_df.dropna(subset=["base_dataset"])
    mix_summary = None
    base_summary = None
    if not mix_df.empty:
        base_scores = rank_table[[
            "benchmark",
            "encoder_regime",
            "dataset",
            "score",
        ]].rename(columns={"dataset": "base_dataset", "score": "base_score"})
        merged = mix_df.merge(
            base_scores,
            on=["benchmark", "encoder_regime", "base_dataset"],
            how="inner",
        )
        if not merged.empty:
            merged["delta"] = merged["score"] - merged["base_score"]
            merged["rel_delta"] = merged["delta"] / merged["base_score"].replace(0, np.nan)
            merged["win"] = (merged["delta"] > 0).astype(int)

            mix_summary = merged.groupby(
                ["dataset", "base_dataset", "mix_ratio"]
            ).agg(
                n_groups=("delta", "count"),
                mean_delta=("delta", "mean"),
                median_delta=("delta", "median"),
                mean_rel_delta=("rel_delta", "mean"),
                win_rate=("win", "mean"),
            ).reset_index().sort_values("mean_delta", ascending=False)
            mix_summary.to_csv(output_dir / "mix_vs_base_summary.csv", index=False)

            base_summary = merged.groupby("base_dataset").agg(
                n_groups=("delta", "count"),
                mean_delta=("delta", "mean"),
                median_delta=("delta", "median"),
                mean_rel_delta=("rel_delta", "mean"),
                win_rate=("win", "mean"),
            ).reset_index().sort_values("mean_delta", ascending=False)
            base_summary.to_csv(output_dir / "mix_vs_base_by_base_dataset.csv", index=False)
    else:
        print("No mix datasets detected for mix vs base summary.")

    summary_path = output_dir / "aggregate_summary.txt"
    lines = []
    lines.append("TRAINING SET AGGREGATE SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Total snapshots parsed: {len(snapshot_dirs)}")
    lines.append(f"Grouped entries: {len(rank_table)}")
    lines.append("")
    lines.append(f"Ranking score: {args.score}")
    lines.append(f"Top-k fraction: {args.topk_frac:.2f}")
    lines.append("")

    if not agg_distinct.empty:
        lines.append("Top datasets (distinct mixes)")
        lines.append("-" * 80)
        for label, sort_col, context in [
            (
                "Highest mean_score_vs_best",
                "mean_score_vs_best",
                "Average (score / best_score) across benchmark+regime groups; 1.0 means consistently near-best.",
            ),
            (
                "Lowest mean_score_gap_to_best",
                "mean_score_gap_to_best",
                "Average absolute gap to the best score within each benchmark+regime; lower is better.",
            ),
            (
                "Highest top1_rate",
                "top1_rate",
                "Fraction of benchmark+regime groups where the dataset is ranked #1.",
            ),
            (
                "Highest topk_rate",
                "topk_rate",
                "Fraction of benchmark+regime groups where the dataset falls within top-k (default top-20%).",
            ),
        ]:
            sub = agg_distinct.sort_values(sort_col, ascending=(sort_col != "mean_score_vs_best" and sort_col != "top1_rate" and sort_col != "topk_rate"))
            if sort_col == "mean_score_gap_to_best":
                sub = sub.sort_values(sort_col, ascending=True)
            lines.append(label + ":")
            lines.append("  " + context)
            lines.append(
                "  "
                + f"{'dataset':<28} {'score_vs_best':>13} {'gap_to_best':>12} "
                + f"{'top1':>6} {'topk':>6}"
            )
            max_rows = None if args.summary_max_rows <= 0 else args.summary_max_rows
            view = sub if max_rows is None else sub.head(max_rows)
            for _, row in view.iterrows():
                lines.append(
                    "  "
                    + f"{row['dataset']:<28} "
                    + f"{row['mean_score_vs_best']:>13.4f} "
                    + f"{row['mean_score_gap_to_best']:>12.4f} "
                    + f"{row['top1_rate']:>6.2f} "
                    + f"{row['topk_rate']:>6.2f}"
                )
            lines.append("")

    if not agg_grouped.empty:
        lines.append("Top datasets (mixes collapsed)")
        lines.append("-" * 80)
        lines.append(
            "Context: mix ratios like *_30_70 and *_70_30 are collapsed into a single synthetic-mix group."
        )
        sub = agg_grouped.sort_values("mean_score_vs_best", ascending=False)
        lines.append(
            "  "
            + f"{'dataset_group':<28} {'score_vs_best':>13} {'gap_to_best':>12} "
            + f"{'top1':>6} {'topk':>6}"
        )
        max_rows = None if args.summary_max_rows <= 0 else args.summary_max_rows
        view = sub if max_rows is None else sub.head(max_rows)
        for _, row in view.iterrows():
            lines.append(
                "  "
                + f"{row['dataset_group']:<28} "
                + f"{row['mean_score_vs_best']:>13.4f} "
                + f"{row['mean_score_gap_to_best']:>12.4f} "
                + f"{row['top1_rate']:>6.2f} "
                + f"{row['topk_rate']:>6.2f}"
            )
        lines.append("")

    if mix_summary is not None and not mix_summary.empty:
        lines.append("Mix vs base (best improvements)")
        lines.append("-" * 80)
        lines.append(
            "Context: compares each mix to its base dataset within the same benchmark+regime; "
            "mean_delta is absolute PCK gain, win_rate is the fraction of groups where the mix wins."
        )
        lines.append(
            "  "
            + f"{'mix':<28} {'base':<18} {'ratio':<8} "
            + f"{'mean_delta':>11} {'rel_delta':>11} {'win_rate':>9}"
        )
        max_rows = None if args.summary_max_rows <= 0 else args.summary_max_rows
        view = mix_summary.sort_values("mean_delta", ascending=False)
        if max_rows is not None:
            view = view.head(max_rows)
        for _, row in view.iterrows():
            lines.append(
                "  "
                + f"{row['dataset']:<28} "
                + f"{row['base_dataset']:<18} "
                + f"{row['mix_ratio']:<8} "
                + f"{row['mean_delta']:>11.4f} "
                + f"{row['mean_rel_delta']:>11.4f} "
                + f"{row['win_rate']:>9.2f}"
            )
        lines.append("")

    if base_summary is not None and not base_summary.empty:
        lines.append("Base datasets that benefit most from mixing")
        lines.append("-" * 80)
        lines.append(
            "Context: aggregates all synthetic mixes for each base dataset to show overall mix benefit."
        )
        lines.append(
            "  "
            + f"{'base_dataset':<22} {'mean_delta':>11} {'rel_delta':>11} {'win_rate':>9}"
        )
        max_rows = None if args.summary_max_rows <= 0 else args.summary_max_rows
        view = base_summary.sort_values("mean_delta", ascending=False)
        if max_rows is not None:
            view = view.head(max_rows)
        for _, row in view.iterrows():
            lines.append(
                "  "
                + f"{row['base_dataset']:<22} "
                + f"{row['mean_delta']:>11.4f} "
                + f"{row['mean_rel_delta']:>11.4f} "
                + f"{row['win_rate']:>9.2f}"
            )
        lines.append("")

    summary_path.write_text("\n".join(lines))
    print(f"Saved summary to: {summary_path}")
    print(f"Saved aggregates to: {output_dir}")


if __name__ == "__main__":
    main()
