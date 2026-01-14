#!/usr/bin/env python3
"""
Pareto-style precision/recall scatter with color = performance metric.

Uses coverage-based precision/recall columns (flow/dino) from auc_with_features.csv.
Optionally de-mean or z-score the color metric within benchmark/encoder/model groups.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_INPUT = "analysis/comprehensive/univariate_all_predictors/auc_with_features.csv"


def _parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_pr_cols(
    df: pd.DataFrame,
    prefix: str,
    variant: str,
) -> Tuple[Optional[str], Optional[str]]:
    recall_raw = [
        f"{prefix}_train_to_eval_over_eval_recall",
        f"{prefix}_train_to_eval_coverage",
    ]
    precision_raw = [
        f"{prefix}_eval_to_train_over_train_precision",
        f"{prefix}_eval_to_train_coverage",
    ]
    recall_logit = [
        f"{prefix}_train_to_eval_over_eval_recall_logit",
        f"{prefix}_train_to_eval_coverage_logit",
    ]
    precision_logit = [
        f"{prefix}_eval_to_train_over_train_precision_logit",
        f"{prefix}_eval_to_train_coverage_logit",
    ]

    if variant == "logit":
        recall_candidates = recall_logit + recall_raw
        precision_candidates = precision_logit + precision_raw
    elif variant == "raw":
        recall_candidates = recall_raw + recall_logit
        precision_candidates = precision_raw + precision_logit
    else:
        recall_candidates = recall_raw + recall_logit
        precision_candidates = precision_raw + precision_logit

    recall_col = next((c for c in recall_candidates if c in df.columns), None)
    precision_col = next((c for c in precision_candidates if c in df.columns), None)
    return recall_col, precision_col


def _resolve_distance_cols(
    df: pd.DataFrame,
    prefix: str,
    norm_mode: str,
) -> Tuple[Optional[str], Optional[str]]:
    if norm_mode == "train":
        x_candidates = [
            f"{prefix}_train_to_eval_mean_dist_over_radius_train",
            f"{prefix}_train_to_eval_mean_dist_over_radius_eval",
            f"{prefix}_train_to_eval_mean_dist",
        ]
        y_candidates = [
            f"{prefix}_eval_to_train_mean_dist_over_radius_train",
            f"{prefix}_eval_to_train_mean_dist_over_radius_eval",
            f"{prefix}_eval_to_train_mean_dist",
        ]
    elif norm_mode == "swap":
        x_candidates = [
            f"{prefix}_train_to_eval_mean_dist_over_radius_train",
            f"{prefix}_train_to_eval_mean_dist_over_radius_eval",
            f"{prefix}_train_to_eval_mean_dist",
        ]
        y_candidates = [
            f"{prefix}_eval_to_train_mean_dist_over_radius_eval",
            f"{prefix}_eval_to_train_mean_dist_over_radius_train",
            f"{prefix}_eval_to_train_mean_dist",
        ]
    elif norm_mode == "eval":
        x_candidates = [
            f"{prefix}_train_to_eval_mean_dist_over_radius_eval",
            f"{prefix}_train_to_eval_mean_dist_over_radius_train",
            f"{prefix}_train_to_eval_mean_dist",
        ]
        y_candidates = [
            f"{prefix}_eval_to_train_mean_dist_over_radius_eval",
            f"{prefix}_eval_to_train_mean_dist_over_radius_train",
            f"{prefix}_eval_to_train_mean_dist",
        ]
    elif norm_mode == "mixed":
        x_candidates = [
            f"{prefix}_train_to_eval_mean_dist_over_radius_eval",
            f"{prefix}_train_to_eval_mean_dist_over_radius_train",
            f"{prefix}_train_to_eval_mean_dist",
        ]
        y_candidates = [
            f"{prefix}_eval_to_train_mean_dist_over_radius_train",
            f"{prefix}_eval_to_train_mean_dist_over_radius_eval",
            f"{prefix}_eval_to_train_mean_dist",
        ]
    else:
        x_candidates = [
            f"{prefix}_train_to_eval_mean_dist_over_radius_eval",
            f"{prefix}_train_to_eval_mean_dist_over_radius_train",
            f"{prefix}_train_to_eval_mean_dist",
        ]
        y_candidates = [
            f"{prefix}_eval_to_train_mean_dist_over_radius_train",
            f"{prefix}_eval_to_train_mean_dist_over_radius_eval",
            f"{prefix}_eval_to_train_mean_dist",
        ]
    x_col = next((c for c in x_candidates if c in df.columns), None)
    y_col = next((c for c in y_candidates if c in df.columns), None)
    return x_col, y_col


def _select_group_cols(df: pd.DataFrame, group_cols: Iterable[str]) -> List[str]:
    return [col for col in group_cols if col in df.columns]


def _normalize_color(
    df: pd.DataFrame,
    metric_col: str,
    mode: str,
    group_cols: List[str],
) -> pd.Series:
    values = pd.to_numeric(df[metric_col], errors="coerce")
    if mode == "raw" or not group_cols:
        return values
    grouped = df.groupby(group_cols, dropna=False)[metric_col]
    means = grouped.transform("mean")
    if mode == "demean":
        return values - means
    if mode == "zscore":
        stds = grouped.transform(lambda s: s.std(ddof=0))
        stds = stds.replace(0, np.nan)
        return (values - means) / stds
    return values


def _pareto_front(x: np.ndarray, y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "min":
        order = np.argsort(x)
        best_y = np.inf
        front_idx: List[int] = []
        for idx in order:
            if not np.isfinite(x[idx]) or not np.isfinite(y[idx]):
                continue
            if y[idx] <= best_y:
                front_idx.append(idx)
                best_y = y[idx]
        return np.array(front_idx, dtype=int)
    order = np.argsort(-x)
    best_y = -np.inf
    front_idx: List[int] = []
    for idx in order:
        if not np.isfinite(x[idx]) or not np.isfinite(y[idx]):
            continue
        if y[idx] >= best_y:
            front_idx.append(idx)
            best_y = y[idx]
    return np.array(front_idx, dtype=int)


def _pick_cmap(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "viridis"
    if finite.min() < 0 < finite.max():
        return "coolwarm"
    return "viridis"


def _maybe_set_unit_axes(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    finite_x = x[np.isfinite(x)]
    finite_y = y[np.isfinite(y)]
    if finite_x.size and finite_y.size:
        if finite_x.min() >= -0.05 and finite_x.max() <= 1.05:
            ax.set_xlim(0.0, 1.0)
        if finite_y.min() >= -0.05 and finite_y.max() <= 1.05:
            ax.set_ylim(0.0, 1.0)


def _safe_filename(token: str) -> str:
    return token.replace("/", "_").replace(" ", "_")


def _is_logit_column(col: str) -> bool:
    return isinstance(col, str) and col.endswith("_logit")


def _logit_transform(values: np.ndarray, eps: float) -> np.ndarray:
    clipped = np.clip(values, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def _format_group_label(cols: List[str], values: Tuple[object, ...]) -> str:
    if not cols:
        return "all"
    parts = []
    for col, val in zip(cols, values):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            text = "unknown"
        else:
            text = str(val)
        parts.append(f"{col}={text}")
    return ", ".join(parts)


def _group_subdir(cols: List[str], values: Tuple[object, ...]) -> str:
    if not cols:
        return "all"
    parts = []
    for col, val in zip(cols, values):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            text = "unknown"
        else:
            text = str(val)
        parts.append(f"{col}-{text}")
    return "__".join(_safe_filename(part) for part in parts)


def _parse_quantile_limits(value: str) -> Optional[Tuple[float, float]]:
    raw = value.strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("axis-quantile-limits expects two comma-separated values (e.g. 0.02,0.98)")
    low, high = (float(parts[0]), float(parts[1]))
    if not (0.0 <= low < high <= 1.0):
        raise ValueError("axis-quantile-limits must satisfy 0 <= low < high <= 1")
    return low, high


def _apply_axis_quantile_limits(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    limits: Optional[Tuple[float, float]],
) -> None:
    if limits is None:
        return
    low, high = limits
    finite_x = x[np.isfinite(x)]
    finite_y = y[np.isfinite(y)]
    if finite_x.size:
        ax.set_xlim(np.quantile(finite_x, low), np.quantile(finite_x, high))
    if finite_y.size:
        ax.set_ylim(np.quantile(finite_y, low), np.quantile(finite_y, high))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot pareto-style precision/recall scatter with performance color."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to auc_with_features.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/pareto_pr",
        help="Directory to save plots and pareto CSVs.",
    )
    parser.add_argument(
        "--metrics",
        default="peak_pck,auc,auc_delta",
        help="Comma-separated performance metrics for coloring.",
    )
    parser.add_argument(
        "--representations",
        default="flow,dino",
        help="Comma-separated representations to plot.",
    )
    parser.add_argument(
        "--axis-mode",
        choices=["coverage", "distance"],
        default="coverage",
        help="Use coverage-based precision/recall or normalized distances on axes.",
    )
    parser.add_argument(
        "--coverage-variant",
        choices=["auto", "raw", "logit"],
        default="auto",
        help="Choose raw or logit coverage columns when axis-mode=coverage.",
    )
    parser.add_argument(
        "--distance-norm",
        choices=["auto", "train", "eval", "mixed", "swap", "both", "grid"],
        default="auto",
        help="Choose distance normalization for axes when axis-mode=distance.",
    )
    parser.add_argument(
        "--x-col",
        default="",
        help="Override x-axis column (applies to all representations).",
    )
    parser.add_argument(
        "--y-col",
        default="",
        help="Override y-axis column (applies to all representations).",
    )
    parser.add_argument(
        "--pareto-mode",
        choices=["max", "min"],
        default="",
        help="Override Pareto direction: max (upper-right) or min (lower-left).",
    )
    parser.add_argument(
        "--axis-transform",
        choices=["none", "log1p", "logit"],
        default="none",
        help="Transform x/y axes (e.g., log1p for distance axes).",
    )
    parser.add_argument(
        "--axis-logit-eps",
        type=float,
        default=1e-6,
        help="Epsilon for logit axis transform.",
    )
    parser.add_argument(
        "--axis-quantile-limits",
        default="",
        help="Optional quantile zoom for axes (e.g. 0.02,0.98).",
    )
    parser.add_argument(
        "--split-by",
        default="",
        help="Comma-separated columns to split plots into subdirectories.",
    )
    parser.add_argument(
        "--color-modes",
        default="raw,demean",
        help="Comma-separated color modes: raw,demean,zscore.",
    )
    parser.add_argument(
        "--demean-by",
        default="benchmark,model_family,encoder_config",
        help="Comma-separated grouping columns for demeaning/zscore.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=28.0,
        help="Scatter point size.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.75,
        help="Scatter alpha.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output figure DPI.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")
    df = pd.read_csv(input_path)

    metrics = _parse_csv_list(args.metrics)
    representations = _parse_csv_list(args.representations)
    color_modes = _parse_csv_list(args.color_modes)
    group_cols = _select_group_cols(df, _parse_csv_list(args.demean_by))
    x_override = args.x_col.strip() or None
    y_override = args.y_col.strip() or None
    axis_limits = _parse_quantile_limits(args.axis_quantile_limits)
    split_cols = _select_group_cols(df, _parse_csv_list(args.split_by))
    distance_norms = [args.distance_norm]
    if args.distance_norm == "both":
        distance_norms = ["train", "eval"]
    if args.distance_norm == "grid":
        distance_norms = ["train", "eval", "mixed", "swap"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if split_cols:
        grouped = df.groupby(split_cols, dropna=False)
    else:
        grouped = [((), df)]

    for group_key, group_df in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_label = _format_group_label(split_cols, group_key)
        group_dir = output_dir / _group_subdir(split_cols, group_key)
        group_dir.mkdir(parents=True, exist_ok=True)

        for rep in representations:
            for norm_mode in distance_norms:
                if x_override and y_override:
                    x_col, y_col = x_override, y_override
                    axis_label_x = x_col
                    axis_label_y = y_col
                    default_pareto = "max"
                    norm_choice = norm_mode if args.axis_mode == "distance" else "coverage"
                elif args.axis_mode == "distance":
                    norm_choice = norm_mode if norm_mode != "auto" else "mixed"
                    x_col, y_col = _resolve_distance_cols(group_df, rep, norm_choice)
                    axis_label_x = f"{rep} train->eval normalized distance ({norm_choice})"
                    axis_label_y = f"{rep} eval->train normalized distance ({norm_choice})"
                    default_pareto = "min"
                else:
                    norm_choice = "coverage"
                    x_col, y_col = _resolve_pr_cols(group_df, rep, args.coverage_variant)
                    axis_label_x = f"{rep} recall (train->eval)"
                    axis_label_y = f"{rep} precision (eval->train)"
                    default_pareto = "max"

                if x_col is None or y_col is None:
                    print(f"Skipping {rep}: missing axis columns for mode {args.axis_mode}.")
                    continue

                pareto_mode = args.pareto_mode.strip() or default_pareto
                if args.axis_mode == "distance":
                    norm_tag = norm_choice
                else:
                    norm_tag = f"coverage_{args.coverage_variant}"

                for metric in metrics:
                    if metric not in group_df.columns:
                        print(f"Skipping metric {metric}: not in input.")
                        continue

                    for mode in color_modes:
                        color = _normalize_color(group_df, metric, mode, group_cols)
                        plot_df = group_df.copy()
                        plot_df["_x"] = pd.to_numeric(group_df[x_col], errors="coerce")
                        plot_df["_y"] = pd.to_numeric(group_df[y_col], errors="coerce")
                        plot_df["_color"] = pd.to_numeric(color, errors="coerce")
                        plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
                        plot_df = plot_df.dropna(subset=["_x", "_y", "_color"])
                        if plot_df.empty:
                            print(f"No valid rows for {rep} {metric} {mode} ({group_label}).")
                            continue

                        x = plot_df["_x"].to_numpy(dtype=float)
                        y = plot_df["_y"].to_numpy(dtype=float)
                        c = plot_df["_color"].to_numpy(dtype=float)
                        if args.axis_transform == "log1p":
                            x = np.log1p(x)
                            y = np.log1p(y)
                        elif args.axis_transform == "logit" and args.axis_mode == "coverage":
                            if not _is_logit_column(x_col):
                                x = _logit_transform(x, args.axis_logit_eps)
                            if not _is_logit_column(y_col):
                                y = _logit_transform(y, args.axis_logit_eps)

                        fig, ax = plt.subplots(figsize=(6.6, 5.6))
                        cmap = _pick_cmap(c)
                        sc = ax.scatter(
                            x,
                            y,
                            c=c,
                            s=args.point_size,
                            alpha=args.alpha,
                            cmap=cmap,
                            edgecolors="none",
                        )

                        front_idx = _pareto_front(x, y, pareto_mode)
                        if front_idx.size:
                            front = plot_df.iloc[front_idx].copy()
                            front = front.sort_values("_x")
                            ax.plot(
                                front["_x"],
                                front["_y"],
                                color="black",
                                linewidth=1.2,
                                alpha=0.9,
                                label="Pareto front",
                            )
                            ax.legend(loc="lower right")

                            front_out = front.copy()
                            front_out["color_metric"] = metric
                            front_out["color_mode"] = mode
                            front_out["representation"] = rep
                            front_out["x_col"] = x_col
                            front_out["y_col"] = y_col
                            front_out["pareto_mode"] = pareto_mode
                            front_out["distance_norm"] = norm_tag
                            front_out.rename(
                                columns={"_x": "x_value", "_y": "y_value", "_color": "color_value"},
                                inplace=True,
                            )
                            pareto_path = group_dir / (
                                f"{rep}_pr_{metric}_{mode}_{args.axis_mode}_{norm_tag}_pareto.csv"
                            )
                            front_out.to_csv(pareto_path, index=False)

                        color_label = f"{metric} ({mode})"
                        if mode != "raw" and group_cols:
                            color_label += f" | group={'/'.join(group_cols)}"
                        cbar = fig.colorbar(sc, ax=ax)
                        cbar.set_label(color_label)

                        xlabel = axis_label_x
                        ylabel = axis_label_y
                        if args.axis_transform == "log1p":
                            xlabel = f"log1p({xlabel})"
                            ylabel = f"log1p({ylabel})"
                        elif args.axis_transform == "logit" and args.axis_mode == "coverage":
                            if not _is_logit_column(x_col):
                                xlabel = f"logit({xlabel})"
                            if not _is_logit_column(y_col):
                                ylabel = f"logit({ylabel})"
                        ax.set_xlabel(xlabel)
                        ax.set_ylabel(ylabel)
                        ax.set_title(
                            f"{rep} {args.axis_mode} scatter colored by {metric} [{mode}]"
                        )
                        if split_cols:
                            ax.set_title(
                                f"{rep} {args.axis_mode} scatter colored by {metric} [{mode}]"
                                f"\n{group_label}"
                            )
                        ax.grid(True, linestyle="--", alpha=0.3)
                        _maybe_set_unit_axes(ax, x, y)
                        _apply_axis_quantile_limits(ax, x, y, axis_limits)

                        out_name = (
                            f"{_safe_filename(rep)}_pr_{_safe_filename(metric)}_"
                            f"{_safe_filename(mode)}_{_safe_filename(args.axis_mode)}_"
                            f"{_safe_filename(norm_tag)}.png"
                        )
                        out_path = group_dir / out_name
                        fig.tight_layout()
                        fig.savefig(out_path, dpi=args.dpi)
                        plt.close(fig)


if __name__ == "__main__":
    main()
