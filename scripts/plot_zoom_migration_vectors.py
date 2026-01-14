#!/usr/bin/env python3
"""
Plot geometric migration vectors for zoom/flip variants vs baseline synthetic.

Arrows originate at the baseline (0, 0) and point to the delta in distances:
  x = delta_train_to_eval_* (alignment)
  y = delta_eval_to_train_* (coverage)
Color encodes delta_performance (positive = gain, negative = loss).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


DEFAULT_INPUT = "analysis/zoom_variants/h3_asymmetry.csv"
DEFAULT_OUTPUT_DIR = "analysis/zoom_variants"
DEFAULT_OUTPUT_NAME = "geometric_migration_vectors.png"
DEFAULT_VARIANTS = [
    "synthetic_large_zoom",
    "synthetic_small_zoom",
    "synthetic_random_flipping",
]


BENCHMARK_LABELS: Dict[str, str] = {
    "kitti2012": "KITTI12",
    "kitti2015": "KITTI15",
    "pfpascal": "PF-Pascal",
    "pfwillow": "PF-Willow",
    "spair": "SPair",
    "tss": "TSS",
}


VARIANT_MARKERS: Dict[str, str] = {
    "synthetic_large_zoom": "o",
    "synthetic_small_zoom": "s",
    "synthetic_random_flipping": "^",
}


VARIANT_LABEL_OFFSETS: Dict[str, tuple[int, int]] = {
    "synthetic_large_zoom": (4, 4),
    "synthetic_small_zoom": (4, -6),
    "synthetic_random_flipping": (-8, 4),
}


def _parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _symmetric_limits(values: np.ndarray, pad: float = 0.15) -> tuple[float, float]:
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size == 0:
        return -1.0, 1.0
    bound = np.max(np.abs(finite_vals))
    if bound == 0:
        bound = 1.0
    bound *= 1.0 + pad
    return -bound, bound


def _normalize_values(values: np.ndarray) -> TwoSlopeNorm:
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size == 0:
        return TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    max_abs = float(np.max(np.abs(finite_vals)))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0
    return TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)


def _plot_vectors(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    color_col: str,
    variants: List[str],
    title: str,
    output_path: Path,
    show_labels: bool,
    alpha: float,
    jitter: float,
) -> None:
    x_vals = df[x_col].to_numpy()
    y_vals = df[y_col].to_numpy()
    color_vals = df[color_col].to_numpy()

    fig, ax = plt.subplots(figsize=(9.5, 8.0))

    norm = _normalize_values(color_vals)
    cmap = plt.get_cmap("RdBu_r")

    ax.scatter(
        [0],
        [0],
        marker="*",
        s=160,
        color="black",
        zorder=4,
        label="Baseline (synthetic)",
    )

    counts = df.groupby(["variant", "benchmark"], dropna=False).size()
    drawn_arrows = set()
    labeled = set()
    rng = np.random.default_rng(0)

    for _, row in df.iterrows():
        x = float(row[x_col])
        y = float(row[y_col])
        perf = float(row[color_col])
        variant = str(row["variant"])
        benchmark = str(row["benchmark"])
        key = (variant, benchmark)

        color = cmap(norm(perf))
        marker = VARIANT_MARKERS.get(variant, "D")
        if key not in drawn_arrows:
            neutral = counts.get(key, 1) > 1
            arrow_color = "0.3" if neutral else color
            arrow_alpha = min(alpha, 0.6) if neutral else alpha
            ax.annotate(
                "",
                xy=(x, y),
                xytext=(0, 0),
                arrowprops=dict(
                    arrowstyle="->",
                    color=arrow_color,
                    linewidth=1.5,
                    alpha=arrow_alpha,
                ),
                zorder=2,
            )
            drawn_arrows.add(key)
        jx = rng.uniform(-jitter, jitter) if jitter > 0 else 0.0
        jy = rng.uniform(-jitter, jitter) if jitter > 0 else 0.0
        ax.scatter(
            [x + jx],
            [y + jy],
            s=80,
            marker=marker,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            alpha=alpha,
            zorder=3,
        )
        if show_labels and key not in labeled:
            label = BENCHMARK_LABELS.get(benchmark, benchmark)
            offset = VARIANT_LABEL_OFFSETS.get(variant, (4, 4))
            ax.annotate(
                label,
                xy=(x, y),
                xytext=offset,
                textcoords="offset points",
                fontsize=8.5,
                color="black",
            )
            labeled.add(key)

    ax.axhline(0, color="0.6", linewidth=1.0, zorder=1)
    ax.axvline(0, color="0.6", linewidth=1.0, zorder=1)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    ax.set_xlabel(
        "Delta Train->Eval Distance (alignment; negative = closer/better)"
    )
    ax.set_ylabel(
        "Delta Eval->Train Distance (coverage; negative = better)"
    )
    ax.set_title(title)

    x_min, x_max = _symmetric_limits(x_vals)
    y_min, y_max = _symmetric_limits(y_vals)
    if jitter > 0:
        x_min -= jitter
        x_max += jitter
        y_min -= jitter
        y_max += jitter
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Delta performance")

    variant_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=VARIANT_MARKERS.get(name, "D"),
            color="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.9,
            markersize=8,
            label=name,
        )
        for name in variants
        if name in df["variant"].unique()
    ]
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(variant_handles)
    labels.extend([h.get_label() for h in variant_handles])
    ax.legend(handles, labels, loc="upper right", frameon=True, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot geometric migration vectors for zoom/flip variants."
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
        "--x-col",
        default="delta_train_to_eval_norm_by_eval_log1p",
        help="Column for x-axis delta (train->eval distance).",
    )
    parser.add_argument(
        "--y-col",
        default="delta_eval_to_train_norm_by_eval_log1p",
        help="Column for y-axis delta (eval->train distance).",
    )
    parser.add_argument(
        "--color-col",
        default="delta_performance",
        help="Column for color (performance delta).",
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
        help="Alpha for points/arrows (lower = more transparent).",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.0,
        help="Uniform jitter applied to points to reveal overlaps.",
    )
    parser.add_argument(
        "--split-by-variant",
        action="store_true",
        help="Also save one plot per variant alongside the combined plot.",
    )
    parser.add_argument(
        "--raw-x-col",
        default="delta_train_to_eval_mean_dist",
        help="Column for raw x-axis delta (train->eval mean distance).",
    )
    parser.add_argument(
        "--raw-y-col",
        default="delta_eval_to_train_mean_dist",
        help="Column for raw y-axis delta (eval->train mean distance).",
    )
    parser.add_argument(
        "--raw-suffix",
        default="_raw",
        help="Suffix to append for raw-distance plots.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    variants = _parse_csv_list(args.variants)
    if variants:
        df = df[df["variant"].isin(variants)]
    if df.empty:
        raise SystemExit("No rows found after filtering variants.")

    x_col = args.x_col
    y_col = args.y_col
    color_col = args.color_col
    if x_col not in df.columns:
        x_col = _resolve_col(
            df,
            [
                "delta_train_to_eval_norm_by_eval_log1p",
                "delta_train_to_eval_norm_by_train_log1p",
                "delta_train_to_eval_kl",
            ],
        )
    if y_col not in df.columns:
        y_col = _resolve_col(
            df,
            [
                "delta_eval_to_train_norm_by_eval_log1p",
                "delta_eval_to_train_norm_by_train_log1p",
                "delta_eval_to_train_kl",
            ],
        )
    if x_col is None or y_col is None:
        raise SystemExit("Could not resolve x/y columns for migration vectors.")
    if color_col not in df.columns:
        raise SystemExit(f"Color column not found: {color_col}")

    df = df.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df[color_col] = pd.to_numeric(df[color_col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col, color_col, "benchmark", "variant"])
    if df.empty:
        raise SystemExit("No valid rows after numeric conversion.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name
    _plot_vectors(
        df,
        x_col=x_col,
        y_col=y_col,
        color_col=color_col,
        variants=variants,
        title="Geometric Migration Vectors: Synthetic -> Zoom/Flip",
        output_path=out_path,
        show_labels=not args.no_labels,
        alpha=args.alpha,
        jitter=args.jitter,
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
            _plot_vectors(
                sub,
                x_col=x_col,
                y_col=y_col,
                color_col=color_col,
                variants=[variant],
                title=f"Geometric Migration Vectors: {variant}",
                output_path=variant_path,
                show_labels=not args.no_labels,
                alpha=args.alpha,
                jitter=args.jitter,
            )
        raw_x_col = args.raw_x_col
        raw_y_col = args.raw_y_col
        if raw_x_col in df.columns and raw_y_col in df.columns:
            raw_title = "Geometric Migration Vectors (Raw Distances)"
            raw_path = out_dir / f"{stem}{args.raw_suffix}{suffix}"
            _plot_vectors(
                df,
                x_col=raw_x_col,
                y_col=raw_y_col,
                color_col=color_col,
                variants=variants,
                title=raw_title,
                output_path=raw_path,
                show_labels=not args.no_labels,
                alpha=args.alpha,
                jitter=args.jitter,
            )
            for variant in variants:
                sub = df[df["variant"] == variant]
                if sub.empty:
                    continue
                safe_name = variant.replace("synthetic_", "")
                variant_path = out_dir / f"{stem}_{safe_name}{args.raw_suffix}{suffix}"
                _plot_vectors(
                    sub,
                    x_col=raw_x_col,
                    y_col=raw_y_col,
                    color_col=color_col,
                    variants=[variant],
                    title=f"{raw_title}: {variant}",
                    output_path=variant_path,
                    show_labels=not args.no_labels,
                    alpha=args.alpha,
                    jitter=args.jitter,
                )
        else:
            missing = [c for c in (raw_x_col, raw_y_col) if c not in df.columns]
            print(f"Raw distance columns missing, skipping raw plots: {missing}")


if __name__ == "__main__":
    main()
