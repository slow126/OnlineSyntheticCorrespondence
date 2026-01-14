#!/usr/bin/env python3
"""
Plot a dataset-by-dataset MMD heatmap from a precomputed CSV.

Supports train/train (square) and train/val (rectangular) matrices.
Low MMD = green, high MMD = red by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _sorted_unique(items: List[str]) -> List[str]:
    return sorted({str(item) for item in items})


def _build_matrix(
    df: pd.DataFrame,
    split1: str,
    split2: str,
    value_col: str,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    filtered = df[(df["split1"] == split1) & (df["split2"] == split2)].copy()
    if filtered.empty:
        raise SystemExit(f"No rows found for split1={split1} split2={split2}")

    rows = _sorted_unique(filtered["dataset1"].tolist())
    cols = _sorted_unique(filtered["dataset2"].tolist())

    pivot = filtered.pivot_table(
        index="dataset1", columns="dataset2", values=value_col, aggfunc="mean"
    )
    pivot = pivot.reindex(index=rows, columns=cols)

    if split1 == split2:
        # Fill symmetric entries if they exist in the opposite direction.
        swapped = filtered.rename(columns={"dataset1": "dataset2", "dataset2": "dataset1"})
        swapped = swapped.pivot_table(
            index="dataset1", columns="dataset2", values=value_col, aggfunc="mean"
        )
        swapped = swapped.reindex(index=rows, columns=cols)
        pivot = pivot.combine_first(swapped)

    return pivot, rows, cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MMD matrix heatmap.")
    parser.add_argument("--csv", default="dino_mmd_results_fast.csv", help="MMD results CSV.")
    parser.add_argument("--split1", default="train", help="Split for dataset1.")
    parser.add_argument("--split2", default="train", help="Split for dataset2.")
    parser.add_argument(
        "--value",
        default="mmd",
        choices=["mmd", "mmd2"],
        help="Value column to plot.",
    )
    parser.add_argument(
        "--triangle",
        choices=["upper", "lower", "none"],
        default="none",
        help="Mask triangle for square matrices.",
    )
    parser.add_argument("--vmin", type=float, default=None, help="Color scale min.")
    parser.add_argument("--vmax", type=float, default=None, help="Color scale max.")
    parser.add_argument(
        "--output",
        default="analysis/comprehensive/dino_mmd_matrix_train_train.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    for col in ["dataset1", "dataset2", "split1", "split2", args.value]:
        if col not in df.columns:
            raise SystemExit(f"Missing column: {col}")

    matrix, rows, cols = _build_matrix(df, args.split1, args.split2, args.value)
    values = matrix.to_numpy(dtype=float)

    mask = None
    if args.triangle != "none" and matrix.shape[0] == matrix.shape[1]:
        mask = np.zeros_like(values, dtype=bool)
        if args.triangle == "upper":
            mask = np.triu(mask, k=1)
        else:
            mask = np.tril(mask, k=-1)
        values = np.ma.array(values, mask=mask)

    vmin = args.vmin if args.vmin is not None else np.nanmin(values)
    vmax = args.vmax if args.vmax is not None else np.nanmax(values)

    fig_w = max(6.5, 0.5 * len(cols))
    fig_h = max(5.5, 0.5 * len(rows))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(values, cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(rows)

    title = f"MMD ({args.split1} vs {args.split2})"
    if args.value == "mmd2":
        title = f"MMD² ({args.split1} vs {args.split2})"
    ax.set_title(title)
    ax.set_xlabel(f"dataset2 ({args.split2})")
    ax.set_ylabel(f"dataset1 ({args.split1})")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar_label = "MMD" if args.value == "mmd" else "MMD²"
    cbar.set_label(cbar_label)

    fig.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
