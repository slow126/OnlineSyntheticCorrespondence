#!/usr/bin/env python3
"""
Plot smoothness metrics (TV, Laplacian) with explicit mix labels.
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def extract_gt_reference(df):
    gt_cols = [
        "mean_tv_gt",
        "std_tv_gt",
        "mean_laplacian_gt",
        "std_laplacian_gt",
        "num_samples_gt",
    ]
    if not all(col in df.columns for col in gt_cols):
        return {}
    gt_df = df.dropna(subset=["mean_tv_gt", "mean_laplacian_gt"])
    if gt_df.empty:
        return {}
    grouped = gt_df.groupby("benchmark", dropna=False)[gt_cols].mean().reset_index()
    return grouped.set_index("benchmark").to_dict(orient="index")


def resolve_validation_results_path(checkpoint_path):
    if checkpoint_path is None or pd.isna(checkpoint_path):
        return None
    path = Path(checkpoint_path)
    candidates = [
        path.parent / "validation_results.csv",
        path.parent.parent / "validation_results.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_best_pck(validation_path, benchmark, cache):
    if validation_path is None:
        return None
    key = (str(validation_path), benchmark)
    if key in cache:
        return cache[key]
    try:
        val_df = pd.read_csv(validation_path)
    except Exception:
        cache[key] = None
        return None
    if "benchmark" not in val_df.columns or "pck" not in val_df.columns:
        cache[key] = None
        return None
    bench_rows = val_df[val_df["benchmark"] == benchmark]
    if bench_rows.empty:
        cache[key] = None
        return None
    best_pck = bench_rows["pck"].max()
    cache[key] = best_pck
    return best_pck


def add_pck_column(df):
    cache = {}
    pck_values = []
    for _, row in df.iterrows():
        validation_path = resolve_validation_results_path(row.get("checkpoint_path"))
        pck = load_best_pck(validation_path, row.get("benchmark"), cache)
        pck_values.append(pck)
    df = df.copy()
    df["best_pck"] = pck_values
    return df


def parse_checkpoint_name(name):
    text = str(name).lower()
    condition = "unknown"
    mix_ratio = None

    if "synthetic_2d" in text or "synthetic2d" in text or "synthetic2dwarp" in text or "synthetic_2dwarp" in text:
        condition = "synthetic_2dwarp"
    elif text.startswith("synthetic"):
        condition = "synthetic_only"
    elif text.startswith("imagenet2dwarp") or text.startswith("2dwarp"):
        condition = "2dwarp_only"
        mix_ratio = "100_0"
    elif "spair_only" in text or (
        text.startswith("spair_")
        and "synthetic" not in text
        and "2d_warp" not in text
        and "2dwarp" not in text
        and "imagenet" not in text
    ):
        condition = "spair_only"
    elif "synthetic" in text:
        condition = "spair_synthetic"
        for ratio in ("30_70", "50_50", "70_30"):
            if ratio in text:
                mix_ratio = ratio
                break
    elif "2d_warp" in text or "2dwarp" in text or "imagenet2dwarp" in text:
        condition = "spair_2dwarp"
        for ratio in ("30_70", "50_50", "70_30", "100_0"):
            if ratio in text:
                mix_ratio = ratio
                break
        if mix_ratio is None and "imagenet2dwarp" in text:
            mix_ratio = "100_0"

    model_type = "raft" if "raft" in text else "cats"

    pretrained = None
    freeze = None
    if "pretrainedtrue" in text:
        pretrained = True
    elif "pretrainedfalse" in text:
        pretrained = False
    if "freezetrue" in text:
        freeze = True
    elif "freezefalse" in text:
        freeze = False

    return condition, mix_ratio, model_type, pretrained, freeze


def format_mix_label(condition, mix_ratio, name):
    name = str(name).lower()
    if condition == "spair_only":
        return "SPAIR only"
    if condition == "synthetic_only":
        return "Synthetic only"
    if condition == "synthetic_2dwarp":
        return "Synthetic 2D Warp only"
    if condition == "2dwarp_only":
        return "2D Warp only"
    if condition == "spair_synthetic":
        if mix_ratio:
            spair_pct, synth_pct = mix_ratio.split("_")
            return f"SPAIR {spair_pct}% + Synthetic {synth_pct}%"
        return "SPAIR + Synthetic"
    if condition == "spair_2dwarp":
        if "imagenet2dwarp" in name:
            return "2D Warp 100% (no SPAIR)"
        if mix_ratio:
            spair_pct, warp_pct = mix_ratio.split("_")
            return f"SPAIR {spair_pct}% + 2D Warp {warp_pct}%"
        return "SPAIR + 2D Warp"
    return condition


def encoder_group_label(model_type, pretrained, freeze):
    if model_type == "raft":
        return "RAFT"
    if pretrained is None or freeze is None:
        return "CatsPP"
    pre = "T" if pretrained else "F"
    frz = "T" if freeze else "F"
    return f"CatsPP {pre}{frz}"


def sort_key(condition, mix_ratio, name):
    cond_order = {
        "spair_only": 0,
        "spair_synthetic": 1,
        "spair_2dwarp": 2,
        "synthetic_only": 3,
        "2dwarp_only": 4,
        "synthetic_2dwarp": 5,
    }
    base = cond_order.get(condition, 9)
    if condition == "spair_only":
        return (base, 0)
    if condition == "spair_2dwarp" and "imagenet2dwarp" in str(name).lower():
        return (base, 100)
    if mix_ratio and "_" in mix_ratio:
        try:
            spair_pct = int(mix_ratio.split("_")[0])
        except ValueError:
            spair_pct = 99
        return (base, spair_pct)
    return (base, 99)


def aggregate(df):
    rows = []
    for _, row in df.iterrows():
        condition, mix_ratio, model_type, pretrained, freeze = parse_checkpoint_name(
            row["checkpoint_name"]
        )
        label = format_mix_label(condition, mix_ratio, row["checkpoint_name"])
        group = encoder_group_label(model_type, pretrained, freeze)
        rows.append({
            "benchmark": row["benchmark"],
            "encoder_group": group,
            "condition": condition,
            "mix_ratio": mix_ratio,
            "label": label,
            "sort_key": sort_key(condition, mix_ratio, row["checkpoint_name"]),
            "mean_tv": row["mean_tv"],
            "mean_laplacian": row["mean_laplacian"],
            "best_pck": row.get("best_pck"),
        })
    plot_df = pd.DataFrame(rows)
    agg = (
        plot_df.groupby(
            ["benchmark", "encoder_group", "condition", "mix_ratio", "label", "sort_key"],
            dropna=False,
        )
        .agg(
            mean_tv=("mean_tv", "mean"),
            std_tv=("mean_tv", "std"),
            mean_laplacian=("mean_laplacian", "mean"),
            std_laplacian=("mean_laplacian", "std"),
            mean_pck=("best_pck", "mean"),
            std_pck=("best_pck", "std"),
            n=("mean_tv", "size"),
        )
        .reset_index()
    )
    return agg


def build_raw_table(df):
    rows = []
    for _, row in df.iterrows():
        condition, mix_ratio, model_type, pretrained, freeze = parse_checkpoint_name(
            row["checkpoint_name"]
        )
        label = format_mix_label(condition, mix_ratio, row["checkpoint_name"])
        group = encoder_group_label(model_type, pretrained, freeze)
        rows.append({
            "benchmark": row["benchmark"],
            "checkpoint_name": row["checkpoint_name"],
            "encoder_group": group,
            "condition": condition,
            "mix_ratio": mix_ratio,
            "label": label,
            "mean_tv": row["mean_tv"],
            "mean_laplacian": row["mean_laplacian"],
            "best_pck": row.get("best_pck"),
            "n": 1,
        })
    return pd.DataFrame(rows)


def plot_metrics(df, benchmark, output_path, dual_pck=False):
    bench_df = df[df["benchmark"] == benchmark].copy()
    if bench_df.empty:
        print(f"No rows found for benchmark '{benchmark}'.")
        return

    encoder_order = ["CatsPP FF", "CatsPP FT", "CatsPP TF", "CatsPP TT", "RAFT"]
    groups = [g for g in encoder_order if g in bench_df["encoder_group"].unique()]
    if not groups:
        groups = sorted(bench_df["encoder_group"].unique())

    condition_colors = {
        "spair_only": "#d62728",
        "spair_synthetic": "#2ca02c",
        "spair_2dwarp": "#ff7f0e",
        "synthetic_only": "#1f77b4",
        "2dwarp_only": "#9467bd",
        "synthetic_2dwarp": "#8c564b",
    }

    fig, axes = plt.subplots(len(groups), 2, figsize=(16, 4 * len(groups)))
    if len(groups) == 1:
        axes = [axes]

    fig.suptitle(
        f"Flow Smoothness by Encoder Regime ({benchmark})",
        fontsize=16,
        fontweight="bold",
    )

    gt_reference = df.attrs.get("gt_reference", {})
    gt_values = gt_reference.get(benchmark, {}) if gt_reference else {}

    for row_idx, group in enumerate(groups):
        group_df = bench_df[bench_df["encoder_group"] == group].copy()
        group_df = group_df.sort_values("sort_key")

        for col_idx, metric in enumerate(["mean_tv", "mean_laplacian"]):
            ax = axes[row_idx][col_idx]
            values = group_df[metric].to_list()
            labels = group_df["label"].to_list()
            colors = [
                condition_colors.get(cond, "#7f7f7f")
                for cond in group_df["condition"]
            ]
            x_pos = np.arange(len(values))
            bar_width = 0.38 if dual_pck else 0.6
            smooth_positions = x_pos - (bar_width / 2 if dual_pck else 0)
            bars = ax.bar(
                smooth_positions,
                values,
                width=bar_width,
                color=colors,
                alpha=0.8,
                edgecolor="black",
                linewidth=1.0,
            )
            ax.set_xticks(list(x_pos))
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
            ylabel = "Total Variation (lower = smoother)" if metric == "mean_tv" else "Laplacian Magnitude (lower = smoother)"
            title = "Total Variation" if metric == "mean_tv" else "Laplacian Smoothness"
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f"{group} - {title}", fontsize=12, fontweight="bold")
            ax.grid(True, axis="y", alpha=0.3)
            for idx, value in enumerate(values):
                if value is None or pd.isna(value):
                    continue
                ax.text(smooth_positions[idx], value * 1.02, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
            if dual_pck and "mean_pck" in group_df.columns:
                pck_values = pd.to_numeric(group_df["mean_pck"], errors="coerce").to_numpy()
                if np.isfinite(pck_values).any():
                    ax2 = ax.twinx()
                    pck_positions = x_pos + bar_width / 2
                    pck_bar = ax2.bar(
                        pck_positions,
                        pck_values,
                        width=bar_width,
                        color="#4d4d4d",
                        alpha=0.55,
                        edgecolor="black",
                        linewidth=0.8,
                        hatch="//",
                        label="PCK",
                    )
                    max_pck = np.nanmax(pck_values)
                    y_max = max_pck * 1.2 if max_pck > 0 else 1.0
                    ax2.set_ylim(0, y_max)
                    ax2.set_ylabel("PCK (higher = better)", fontsize=10)
                    ax2.legend(handles=[pck_bar], loc="upper right", fontsize=8, frameon=False)
            if gt_values:
                gt_metric = "mean_tv_gt" if metric == "mean_tv" else "mean_laplacian_gt"
                gt_value = gt_values.get(gt_metric)
                if gt_value is not None and not pd.isna(gt_value):
                    ax.axhline(gt_value, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
                    ax.text(
                        0.99,
                        gt_value,
                        "GT",
                        transform=ax.get_yaxis_transform(),
                        ha="right",
                        va="bottom",
                        fontsize=8,
                        color="black",
                    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")
    plt.close()


def write_table(df, benchmark, output_path):
    bench_df = df[df["benchmark"] == benchmark].copy()
    if bench_df.empty:
        print(f"No rows found for benchmark '{benchmark}'.")
        return

    table_df = bench_df.sort_values(["encoder_group", "sort_key"]).copy()
    table_cols = [
        "encoder_group",
        "condition",
        "mix_ratio",
        "label",
        "mean_tv",
        "std_tv",
        "mean_laplacian",
        "std_laplacian",
        "n",
    ]
    if "mean_pck" in table_df.columns:
        table_cols += ["mean_pck", "std_pck"]
    table_df = table_df[table_cols]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(output_path, index=False)
    print(f"Saved table to: {output_path}")


def write_summary_txt(df, benchmark, output_path):
    bench_df = df[df["benchmark"] == benchmark].copy()
    if bench_df.empty:
        print(f"No rows found for benchmark '{benchmark}'.")
        return

    encoder_order = ["CatsPP FF", "CatsPP FT", "CatsPP TF", "CatsPP TT", "RAFT"]
    row_order = [g for g in encoder_order if g in bench_df["encoder_group"].unique()]
    row_order += sorted([g for g in bench_df["encoder_group"].unique() if g not in row_order])

    col_order = (
        bench_df[["label", "sort_key"]]
        .drop_duplicates()
        .sort_values("sort_key")["label"]
        .to_list()
    )

    def pivot_metric(metric):
        table = bench_df.pivot_table(
            index="encoder_group",
            columns="label",
            values=metric,
            aggfunc="mean",
        )
        return table.reindex(index=row_order, columns=col_order)

    pck_table = pivot_metric("mean_pck")
    baseline_label = "SPAIR only"
    baseline_pck = (
        pck_table[baseline_label]
        if baseline_label in pck_table.columns
        else pd.Series(index=pck_table.index, dtype=float)
    )

    def format_table(metric_table):
        def format_row(row):
            pck_row = pck_table.loc[row.name] if row.name in pck_table.index else None
            baseline = baseline_pck.get(row.name, float("nan"))
            formatted = {}
            for col, value in row.items():
                if pd.isna(value):
                    formatted[col] = ""
                    continue
                pck_val = float("nan")
                if pck_row is not None and col in pck_row.index:
                    pck_val = pck_row[col]
                if pd.isna(pck_val) or pd.isna(baseline):
                    formatted[col] = f"{value:.6f}"
                else:
                    delta = pck_val - baseline
                    formatted[col] = f"{value:.6f} ({delta:+.2f})"
            return pd.Series(formatted)

        return metric_table.apply(format_row, axis=1)

    tv_table = format_table(pivot_metric("mean_tv"))
    lap_table = format_table(pivot_metric("mean_laplacian"))

    def format_stat(value):
        if value is None or pd.isna(value):
            return "n/a"
        return f"{value:.6f}"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"Flow smoothness summary (benchmark: {benchmark})\n\n")
        mask_status = df.attrs.get("mask_by_gt", "unknown")
        f.write(f"Label mask (valid GT pixels only): {mask_status}\n\n")
        gt_reference = df.attrs.get("gt_reference", {})
        gt_values = gt_reference.get(benchmark, {}) if gt_reference else {}
        if gt_values:
            tv_mean = gt_values.get("mean_tv_gt")
            tv_std = gt_values.get("std_tv_gt")
            lap_mean = gt_values.get("mean_laplacian_gt")
            lap_std = gt_values.get("std_laplacian_gt")
            num_samples = gt_values.get("num_samples_gt")
            samples_text = f"{int(num_samples)}" if num_samples is not None and not pd.isna(num_samples) else "n/a"
            f.write("Ground truth smoothness (ideal reference)\n")
            f.write(f"  TV: {format_stat(tv_mean)} ± {format_stat(tv_std)} (n={samples_text})\n")
            f.write(
                f"  Laplacian: {format_stat(lap_mean)} ± {format_stat(lap_std)} (n={samples_text})\n\n"
            )
        f.write("Values show metric; PCK delta in parentheses vs SPAIR only per encoder group.\n\n")
        f.write("Total Variation (lower = smoother)\n")
        f.write(tv_table.to_string())
        f.write("\n\n")
        f.write("Laplacian Magnitude (lower = smoother)\n")
        f.write(lap_table.to_string())
        f.write("\n")
    print(f"Saved summary to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot TV/Laplacian smoothness with explicit mix labels."
    )
    parser.add_argument(
        "--input",
        default="analysis/sparse_regularization_kitti/smoothness_raw_results.csv",
        help="Path to smoothness_raw_results.csv",
    )
    parser.add_argument(
        "--benchmark",
        default="kitti2015",
        help="Benchmark to plot (default: kitti2015).",
    )
    parser.add_argument(
        "--output",
        default="analysis/sparse_regularization_kitti/smoothness_comparison_all_encoders.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--dual-pck",
        action="store_true",
        help="Plot paired bars with a secondary PCK axis.",
    )
    parser.add_argument(
        "--table-output",
        default=None,
        help="Optional CSV table output path.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional summary TXT output path.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if df.empty:
        print("No rows found in input.")
        return
    mask_status = "unknown"
    if "mask_by_gt" in df.columns:
        mask_vals = df["mask_by_gt"].dropna().unique().tolist()
        if len(mask_vals) == 1:
            mask_status = "on" if bool(mask_vals[0]) else "off"
        elif len(mask_vals) > 1:
            mask_status = "mixed"
    gt_reference = extract_gt_reference(df)
    df = add_pck_column(df)
    agg = aggregate(df)
    agg.attrs["gt_reference"] = gt_reference
    agg.attrs["mask_by_gt"] = mask_status
    plot_metrics(agg, args.benchmark, args.output, dual_pck=args.dual_pck)
    table_output = args.table_output
    if not table_output:
        table_output = str(Path(args.output).with_suffix("")) + "_table.csv"
    if "checkpoint_name" in df.columns:
        raw_table = build_raw_table(df)
        raw_table = raw_table[raw_table["benchmark"] == args.benchmark]
        raw_table.to_csv(table_output, index=False)
        agg_table_output = str(Path(table_output).with_suffix("")) + "_agg.csv"
        write_table(agg, args.benchmark, agg_table_output)
    else:
        write_table(agg, args.benchmark, table_output)
    summary_output = args.summary_output
    if not summary_output:
        summary_output = str(Path(args.output).with_suffix("")) + "_summary.txt"
    write_summary_txt(agg, args.benchmark, summary_output)


if __name__ == "__main__":
    main()
