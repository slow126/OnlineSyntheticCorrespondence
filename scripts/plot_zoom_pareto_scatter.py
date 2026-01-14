#!/usr/bin/env python3
"""
Pareto-style scatter for zoom analysis deltas.

Default: x=train_to_eval_mean_dist, y=eval_to_train_mean_dist, color=performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


DEFAULT_INPUT = "analysis/zoom_variants/h3_asymmetry.csv"
DEFAULT_OUTPUT_DIR = "analysis/zoom_variants"
DEFAULT_OUTPUT_NAME = "zoom_pareto_scatter_raw.png"
DEFAULT_VARIANTS = [
    "synthetic_large_zoom",
    "synthetic_small_zoom",
    "synthetic_random_flipping",
]


VARIANT_MARKERS = {
    "synthetic_large_zoom": "o",
    "synthetic_small_zoom": "s",
    "synthetic_random_flipping": "^",
}


def _parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _normalize_color(
    df: pd.DataFrame,
    color_col: str,
    mode: str,
    group_cols: List[str],
) -> pd.Series:
    values = pd.to_numeric(df[color_col], errors="coerce")
    if mode == "raw":
        return values
    if mode == "auto":
        if "delta" in color_col:
            return values
        mode = "demean"
    if not group_cols:
        return values
    groups = [col for col in group_cols if col in df.columns]
    if not groups:
        return values
    grouped = df.groupby(groups, dropna=False)[color_col]
    means = grouped.transform("mean")
    if mode == "demean":
        return values - means
    if mode == "zscore":
        stds = grouped.transform(lambda s: s.std(ddof=0))
        stds = stds.replace(0, np.nan)
        return (values - means) / stds
    return values


def _normalize_values(values: np.ndarray) -> TwoSlopeNorm:
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size == 0:
        return TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    max_abs = float(np.max(np.abs(finite_vals)))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0
    return TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)


def _pareto_front(x: np.ndarray, y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "max":
        order = np.argsort(-x)
        best_y = -np.inf
        front_idx = []
        for idx in order:
            if not np.isfinite(x[idx]) or not np.isfinite(y[idx]):
                continue
            if y[idx] > best_y:
                front_idx.append(idx)
                best_y = y[idx]
        return np.array(front_idx, dtype=int)
    order = np.argsort(x)
    best_y = np.inf
    front_idx = []
    for idx in order:
        if not np.isfinite(x[idx]) or not np.isfinite(y[idx]):
            continue
        if y[idx] < best_y:
            front_idx.append(idx)
            best_y = y[idx]
    return np.array(front_idx, dtype=int)


def _default_label(col: str) -> str:
    if "train_to_eval" in col:
        base = "Train->Eval"
    elif "eval_to_train" in col:
        base = "Eval->Train"
    else:
        base = col
    if "mean_dist" in col:
        return f"{base} Distance (raw)"
    if "norm" in col:
        return f"{base} Distance (normalized)"
    if "kl" in col:
        return f"{base} KL"
    return base


def _plot_scatter(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    color_col: str,
    color_mode: str,
    color_group_cols: List[str],
    title: str,
    output_path: Path,
    show_labels: bool,
    alpha: float,
    pareto_mode: str,
    pareto_group: str,
    x_label: Optional[str],
    y_label: Optional[str],
) -> None:
    x_vals = df[x_col].to_numpy()
    y_vals = df[y_col].to_numpy()
    color_series = _normalize_color(df, color_col, color_mode, color_group_cols)
    color_vals = color_series.to_numpy()

    fig, ax = plt.subplots(figsize=(9.2, 7.8))
    cmap = plt.get_cmap("RdBu_r")
    norm = _normalize_values(color_vals)

    for variant, sub in df.groupby("variant", dropna=False):
        marker = VARIANT_MARKERS.get(str(variant), "D")
        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=70,
            marker=marker,
            c=cmap(norm(color_series.loc[sub.index].to_numpy())),
            edgecolor="black",
            linewidth=0.4,
            alpha=alpha,
            label=str(variant),
            zorder=3,
        )
        if show_labels:
            for _, row in sub.iterrows():
                ax.annotate(
                    str(row.get("benchmark", "")),
                    (row[x_col], row[y_col]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                    color="black",
                )

    if pareto_group != "none":
        groups = [("all", df)] if pareto_group == "all" else list(df.groupby(pareto_group, dropna=False))
        for _, sub in groups:
            x = sub[x_col].to_numpy()
            y = sub[y_col].to_numpy()
            front_idx = _pareto_front(x, y, pareto_mode)
            if front_idx.size == 0:
                continue
            front = sub.iloc[front_idx].sort_values(x_col)
            ax.plot(
                front[x_col],
                front[y_col],
                color="black",
                linewidth=1.4,
                alpha=0.8,
                zorder=2,
            )

    ax.axhline(0, color="0.6", linewidth=1.0, zorder=1)
    ax.axvline(0, color="0.6", linewidth=1.0, zorder=1)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    ax.set_xlabel(x_label or _default_label(x_col))
    ax.set_ylabel(y_label or _default_label(y_col))
    ax.set_title(title)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar_label = color_col
    if color_mode in ("demean", "zscore") or (color_mode == "auto" and "delta" not in color_col):
        suffix = "demeaned" if color_mode != "zscore" else "z-scored"
        cbar_label = f"{color_col} ({suffix})"
    cbar.set_label(cbar_label)

    ax.legend(loc="upper right", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pareto-style scatter for zoom analysis deltas."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV path.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the plot.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="Output filename (png/pdf).",
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated list of variants to plot.",
    )
    parser.add_argument(
        "--axis-mode",
        choices=["raw", "norm", "delta"],
        default="raw",
        help="Choose raw, normalized, or delta distance axes.",
    )
    parser.add_argument(
        "--x-col",
        default="auto",
        help="Column for x-axis (default: auto based on axis-mode).",
    )
    parser.add_argument(
        "--y-col",
        default="auto",
        help="Column for y-axis (default: auto based on axis-mode).",
    )
    parser.add_argument(
        "--color-col",
        default="auto",
        help="Column for color (default: performance or delta_performance).",
    )
    parser.add_argument(
        "--color-normalize",
        choices=["auto", "raw", "demean", "zscore"],
        default="auto",
        help="Normalize color values (auto=use deltas raw, else demean by group).",
    )
    parser.add_argument(
        "--color-group",
        default="benchmark",
        help="Comma-separated columns to group for color normalization.",
    )
    parser.add_argument(
        "--pareto-mode",
        choices=["min", "max"],
        default="min",
        help="Pareto front direction (min = lower-left, max = upper-right).",
    )
    parser.add_argument(
        "--pareto-group",
        choices=["none", "all", "variant", "benchmark"],
        default="variant",
        help="Group for Pareto front computation.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Disable benchmark text labels.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help="Alpha for points.",
    )
    parser.add_argument(
        "--x-label",
        default="",
        help="Override x-axis label.",
    )
    parser.add_argument(
        "--y-label",
        default="",
        help="Override y-axis label.",
    )
    parser.add_argument(
        "--split-by-variant",
        action="store_true",
        help="Also save one plot per variant.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    variants = _parse_csv_list(args.variants)
    if variants:
        df = df[df["variant"].isin(variants)]
    if df.empty:
        raise SystemExit("No rows found after filtering variants.")

    if args.axis_mode == "raw":
        x_default = "train_to_eval_mean_dist"
        y_default = "eval_to_train_mean_dist"
        color_default = "performance"
        title = "Zoom Pareto Scatter (Raw Distances)"
    elif args.axis_mode == "norm":
        x_default = "train_to_eval_norm_by_eval_log1p"
        y_default = "eval_to_train_norm_by_eval_log1p"
        color_default = "performance"
        title = "Zoom Pareto Scatter (Normalized Distances)"
    else:
        x_default = "delta_train_to_eval_norm_by_eval_log1p"
        y_default = "delta_eval_to_train_norm_by_eval_log1p"
        color_default = "delta_performance"
        title = "Zoom Pareto Scatter (Delta Distances)"

    x_col = x_default if args.x_col == "auto" else args.x_col
    y_col = y_default if args.y_col == "auto" else args.y_col
    color_col = color_default if args.color_col == "auto" else args.color_col

    if x_col not in df.columns:
        x_col = _resolve_col(
            df,
            [
                x_default,
                "train_to_eval_mean_dist",
                "train_to_eval_norm_by_eval_log1p",
                "delta_train_to_eval_norm_by_eval_log1p",
                "delta_train_to_eval_mean_dist",
            ],
        )
    if y_col not in df.columns:
        y_col = _resolve_col(
            df,
            [
                y_default,
                "eval_to_train_mean_dist",
                "eval_to_train_norm_by_eval_log1p",
                "delta_eval_to_train_norm_by_eval_log1p",
                "delta_eval_to_train_mean_dist",
            ],
        )
    if x_col is None or y_col is None:
        raise SystemExit("Could not resolve x/y columns.")
    if color_col not in df.columns:
        raise SystemExit(f"Color column not found: {color_col}")

    df = df.copy()
    for col in [x_col, y_col, color_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col, color_col, "benchmark", "variant"])
    if df.empty:
        raise SystemExit("No valid rows after numeric conversion.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name
    _plot_scatter(
        df,
        x_col=x_col,
        y_col=y_col,
        color_col=color_col,
        color_mode=args.color_normalize,
        color_group_cols=_parse_csv_list(args.color_group),
        title=title,
        output_path=out_path,
        show_labels=not args.no_labels,
        alpha=args.alpha,
        pareto_mode=args.pareto_mode,
        pareto_group=args.pareto_group,
        x_label=args.x_label or None,
        y_label=args.y_label or None,
    )

    if args.split_by_variant:
        stem = out_path.stem
        suffix = out_path.suffix or ".png"
        for variant in variants:
            sub = df[df["variant"] == variant]
            if sub.empty:
                continue
            safe_name = variant.replace("synthetic_", "")
            variant_path = out_dir / f"{stem}_{safe_name}{suffix}"
            _plot_scatter(
                sub,
                x_col=x_col,
                y_col=y_col,
                color_col=color_col,
                color_mode=args.color_normalize,
                color_group_cols=_parse_csv_list(args.color_group),
                title=f"Zoom Pareto Scatter: {variant}",
                output_path=variant_path,
                show_labels=not args.no_labels,
                alpha=args.alpha,
                pareto_mode=args.pareto_mode,
                pareto_group="all",
                x_label=args.x_label or None,
                y_label=args.y_label or None,
            )


if __name__ == "__main__":
    main()
