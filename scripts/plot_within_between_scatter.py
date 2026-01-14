#!/usr/bin/env python3
"""
Plot within- vs between-benchmark scatter for predictors vs target.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_INPUT = "analysis/comprehensive/stability_precision_recall_raw_no_spair/auc_with_features.csv"
DEFAULT_OUTPUT = "analysis/comprehensive/within_between_scatter.png"


def _parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return np.nan
    return float(np.dot(x, y) / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return _pearson(rx, ry)


def _fit_line(x: np.ndarray, y: np.ndarray):
    if len(x) < 2:
        return None
    coef = np.polyfit(x, y, 1)
    return coef


def _resolve_mode_column(pred: str, mode: str, df: pd.DataFrame) -> str:
    if mode == "raw":
        return pred
    if "train_to_eval_mean_dist" in pred:
        candidate = pred.replace("mean_dist", "mean_dist_over_radius_eval")
    elif "eval_to_train_mean_dist" in pred:
        candidate = pred.replace("mean_dist", "mean_dist_over_radius_train")
    else:
        candidate = pred
    return candidate if candidate in df.columns else pred


def _prepare_mode_df(
    df: pd.DataFrame, predictors: List[str], mode: str
) -> tuple[pd.DataFrame, List[str], dict]:
    df_mode = df.copy()
    pred_cols = []
    label_map = {}
    for pred in predictors:
        src_col = _resolve_mode_column(pred, mode, df_mode)
        if src_col not in df_mode.columns:
            continue
        if mode == "log1p":
            out_col = f"{src_col}_log1p"
            values = pd.to_numeric(df_mode[src_col], errors="coerce")
            values = values.where(values >= 0)
            df_mode[out_col] = np.log1p(values)
            pred_cols.append(out_col)
            label_map[out_col] = f"log1p({src_col})"
        else:
            pred_cols.append(src_col)
            label_map[src_col] = src_col
    return df_mode, pred_cols, label_map


def _within_transform(df: pd.DataFrame, cols: List[str], mode: str) -> pd.DataFrame:
    if mode == "demean":
        group_means = df.groupby("benchmark", dropna=False)[cols].transform("mean")
        return df[cols] - group_means
    if mode == "zscore":
        group_means = df.groupby("benchmark", dropna=False)[cols].transform("mean")
        group_stds = df.groupby("benchmark", dropna=False)[cols].transform(
            lambda s: s.std(ddof=0)
        )
        group_stds = group_stds.replace(0, np.nan)
        return (df[cols] - group_means) / group_stds
    return df[cols]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot within- vs between-benchmark scatter."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="auc_with_features.csv with predictors and target.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output image path (suffix added per mode).",
    )
    parser.add_argument(
        "--predictors",
        default="flow_train_to_eval_mean_dist,flow_eval_to_train_mean_dist",
        help="Comma-separated predictors to plot.",
    )
    parser.add_argument(
        "--target",
        default="peak_pck",
        help="Target column.",
    )
    parser.add_argument(
        "--axis-modes",
        default="raw,norm,log1p",
        help="Comma-separated modes: raw,norm,log1p.",
    )
    parser.add_argument(
        "--label-benchmarks",
        action="store_true",
        help="Annotate between-benchmark points with benchmark names.",
    )
    parser.add_argument(
        "--within-mode",
        choices=["demean", "zscore"],
        default="demean",
        help="Within-benchmark normalization for the bottom row.",
    )
    parser.add_argument(
        "--within-color",
        choices=["single", "benchmark"],
        default="single",
        help="Color within-benchmark points by a single color or by benchmark.",
    )
    parser.add_argument(
        "--within-legend",
        action="store_true",
        help="Show benchmark legend for within-benchmark plots.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Point transparency for within-benchmark scatter.",
    )
    args = parser.parse_args()

    predictors = _parse_csv_list(args.predictors)
    if not predictors:
        raise SystemExit("No predictors provided.")
    modes = _parse_csv_list(args.axis_modes)
    if not modes:
        raise SystemExit("No axis modes provided.")

    df = pd.read_csv(args.input)
    df = df.copy()
    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")
    df = df.dropna(subset=["benchmark", args.target])
    if df.empty:
        raise SystemExit("No valid rows after filtering.")

    base_out = Path(args.output)
    suffix = base_out.suffix or ".png"
    stem = base_out.stem

    for mode in modes:
        mode_df, pred_cols, label_map = _prepare_mode_df(df, predictors, mode)
        if not pred_cols:
            print(f"Skipping mode '{mode}': no matching predictors found.")
            continue
        mode_df = mode_df.dropna(subset=pred_cols + [args.target, "benchmark"])
        if mode_df.empty:
            print(f"Skipping mode '{mode}': no valid rows after filtering.")
            continue

        between = (
            mode_df.groupby("benchmark", dropna=False)[[args.target] + pred_cols]
            .mean()
            .reset_index()
        )
        within_df = mode_df.copy()
        within_cols = [args.target] + pred_cols
        within_df[within_cols] = _within_transform(within_df, within_cols, args.within_mode)

        n_preds = len(pred_cols)
        fig, axes = plt.subplots(
            2, n_preds, figsize=(5.2 * n_preds, 8.2), squeeze=False
        )

        bench_list = sorted(within_df["benchmark"].dropna().unique().tolist())
        cmap = plt.get_cmap("tab10")
        bench_colors = {b: cmap(i % 10) for i, b in enumerate(bench_list)}

        for idx, pred in enumerate(pred_cols):
            label = label_map.get(pred, pred)
            # Between-benchmark scatter
            ax = axes[0, idx]
            x = between[pred].to_numpy()
            y = between[args.target].to_numpy()
            ax.scatter(x, y, s=70, color="tab:blue", edgecolor="black", linewidth=0.4)
            if args.label_benchmarks:
                for _, row in between.iterrows():
                    ax.annotate(
                        str(row["benchmark"]),
                        (row[pred], row[args.target]),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=8,
                    )
            coef = _fit_line(x, y)
            if coef is not None:
                xs = np.linspace(np.min(x), np.max(x), 100)
                ax.plot(xs, coef[0] * xs + coef[1], color="black", linewidth=1.2)
            ax.set_title(f"Between benchmarks: {label}")
            ax.set_xlabel(label)
            ax.set_ylabel(args.target)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
            pear = _pearson(x, y)
            spear = _spearman(x, y)
            ax.text(
                0.02,
                0.98,
                f"Pearson={pear:+.2f}\nSpearman={spear:+.2f}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )

            # Within-benchmark scatter
            ax = axes[1, idx]
            if args.within_color == "benchmark":
                for bench in bench_list:
                    sub = within_df[within_df["benchmark"] == bench]
                    xw = sub[pred].to_numpy()
                    yw = sub[args.target].to_numpy()
                    ax.scatter(
                        xw,
                        yw,
                        s=18,
                        color=bench_colors[bench],
                        alpha=args.alpha,
                        edgecolor="none",
                        label=bench,
                    )
                    coef = _fit_line(xw, yw)
                    if coef is not None:
                        xs = np.linspace(np.min(xw), np.max(xw), 100)
                        ax.plot(xs, coef[0] * xs + coef[1], color=bench_colors[bench], linewidth=1.1)
                if args.within_legend:
                    ax.legend(loc="upper right", fontsize=7, frameon=True)
            else:
                xw = within_df[pred].to_numpy()
                yw = within_df[args.target].to_numpy()
                ax.scatter(xw, yw, s=18, color="tab:orange", alpha=args.alpha, edgecolor="none")
                coef = _fit_line(xw, yw)
                if coef is not None:
                    xs = np.linspace(np.min(xw), np.max(xw), 100)
                    ax.plot(xs, coef[0] * xs + coef[1], color="black", linewidth=1.2)
            ax.axhline(0, color="0.6", linewidth=0.8)
            ax.axvline(0, color="0.6", linewidth=0.8)
            mode_label = "z-scored" if args.within_mode == "zscore" else "demeaned"
            ax.set_title(f"Within benchmark ({mode_label}): {label}")
            ax.set_xlabel(f"{label} ({mode_label})")
            ax.set_ylabel(f"{args.target} ({mode_label})")
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
            pear = _pearson(xw, yw)
            spear = _spearman(xw, yw)
            ax.text(
                0.02,
                0.98,
                f"Pearson={pear:+.2f}\nSpearman={spear:+.2f}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )

        fig.tight_layout()
        out_path = base_out.with_name(f"{stem}_{mode}{suffix}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=220)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
