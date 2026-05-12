#!/usr/bin/env python3
"""
Visualize HOF log-polar histograms exported by scripts/hof_diagnostic_bins.py.

Input format: .npz with keys:
  - hist: (N, angle_bins, mag_bins)
  - sample_id: (N,) object array
  - index: (N,) int array

Outputs:
  - mean.png (average histogram)
  - grid.png (grid of sample histograms)
  - grid_samples.csv (grid ordering)
  - optional per-sample images
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _load_hist_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "hist" not in data:
        raise ValueError(f"{path}: missing 'hist'")
    hist = data["hist"].astype(np.float32, copy=False)
    sample_id = data["sample_id"] if "sample_id" in data else np.arange(hist.shape[0]).astype(object)
    index = data["index"] if "index" in data else np.arange(hist.shape[0]).astype(np.int64)
    return hist, sample_id, index


def _normalize_hist(hist: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    sums = hist.sum(axis=(1, 2), keepdims=True)
    return hist / np.maximum(sums, eps)


def _plot_polar_hist(
    ax,
    hist: np.ndarray,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
    mag_edges: Optional[np.ndarray] = None,
):
    angle_bins, mag_bins = hist.shape

    theta_edges = np.linspace(0.0, 2.0 * np.pi, angle_bins + 1, endpoint=True)
    theta = theta_edges[:-1]
    width = theta_edges[1] - theta_edges[0]

    if mag_edges is None:
        r_edges = np.linspace(0.0, 1.0, mag_bins + 1, endpoint=True)
    else:
        # Normalize mag edges to [0, 1] for visualization
        r_edges = np.asarray(mag_edges, dtype=np.float32)
        r_edges = r_edges - r_edges.min()
        denom = r_edges.max() if r_edges.max() > 0 else 1.0
        r_edges = r_edges / denom

    for j in range(mag_bins):
        bottom = r_edges[j]
        height = r_edges[j + 1] - r_edges[j]
        colors = plt.get_cmap(cmap)(
            np.clip((hist[:, j] - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
        )
        ax.bar(theta, [height] * angle_bins, width=width, bottom=bottom, color=colors, edgecolor="none")

    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-+" else "_" for c in s)


def _write_grid_csv(path: Path, rows: List[Tuple[int, str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["grid_index", "sample_id", "index"])
        writer.writerows(rows)


def _plot_grid(
    out_path: Path,
    hist: np.ndarray,
    sample_id: np.ndarray,
    index: np.ndarray,
    n: int,
    seed: int,
    vmin: float,
    vmax: float,
    mag_edges: Optional[np.ndarray],
    show_ids: bool,
):
    n_total = hist.shape[0]
    if n_total == 0:
        return
    n = min(n, n_total)

    rng = np.random.default_rng(seed)
    if n_total > n:
        sel = rng.choice(n_total, size=n, replace=False)
    else:
        sel = np.arange(n_total)

    grid = int(math.ceil(math.sqrt(n)))
    fig = plt.figure(figsize=(grid * 2.3, grid * 2.3))

    rows = []
    for i, idx in enumerate(sel):
        ax = fig.add_subplot(grid, grid, i + 1, projection="polar")
        _plot_polar_hist(ax, hist[idx], vmin=vmin, vmax=vmax, mag_edges=mag_edges)
        if show_ids:
            sid = str(sample_id[idx])
            ax.set_title(_sanitize_name(sid)[:24], fontsize=6)
        rows.append((i, str(sample_id[idx]), int(index[idx])))

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    _write_grid_csv(out_path.with_suffix(".csv"), rows)


def _plot_mean(out_path: Path, hist: np.ndarray, vmin: float, vmax: float, mag_edges: Optional[np.ndarray]):
    if hist.shape[0] == 0:
        return
    mean_hist = hist.mean(axis=0)
    fig = plt.figure(figsize=(3.0, 3.0))
    ax = fig.add_subplot(1, 1, 1, projection="polar")
    _plot_polar_hist(ax, mean_hist, vmin=vmin, vmax=vmax, mag_edges=mag_edges)
    ax.set_title("Mean HOF", fontsize=10)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_per_sample(
    out_dir: Path,
    hist: np.ndarray,
    sample_id: np.ndarray,
    index: np.ndarray,
    max_samples: int,
    vmin: float,
    vmax: float,
    mag_edges: Optional[np.ndarray],
):
    n = hist.shape[0]
    if n == 0:
        return
    n = min(n, max_samples)
    for i in range(n):
        fig = plt.figure(figsize=(3.0, 3.0))
        ax = fig.add_subplot(1, 1, 1, projection="polar")
        _plot_polar_hist(ax, hist[i], vmin=vmin, vmax=vmax, mag_edges=mag_edges)
        title = f"{sample_id[i]}"
        ax.set_title(title[:32], fontsize=8)
        out_name = f"sample_{i:04d}__{_sanitize_name(str(sample_id[i]))[:32]}.png"
        fig.tight_layout(pad=0.3)
        fig.savefig(out_dir / out_name, dpi=200)
        plt.close(fig)


def _collect_inputs(input_path: Optional[str], input_dir: Optional[str], pattern: str) -> List[Path]:
    if input_path:
        return [Path(input_path)]
    if input_dir:
        base = Path(input_dir)
        # Recursive search so we can point at analysis/hof_diag and find nested *_hist.npz
        return sorted(base.rglob(pattern))
    raise ValueError("Provide --input or --input-dir")


def main():
    parser = argparse.ArgumentParser(description="Visualize HOF log-polar histograms")
    parser.add_argument("--input", default=None, help="Path to *_hist.npz file")
    parser.add_argument("--input-dir", default=None, help="Directory containing *_hist.npz files")
    parser.add_argument("--pattern", default="*_hist.npz", help="Glob pattern for --input-dir")
    parser.add_argument("--out-dir", default="gaussian_splat/output_hof_hist", help="Output directory")
    parser.add_argument("--grid", type=int, default=16, help="Number of samples in grid")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for grid sampling")
    parser.add_argument("--normalize", action="store_true", help="Normalize each histogram to sum 1")
    parser.add_argument("--per-sample", action="store_true", help="Write per-sample images")
    parser.add_argument("--per-sample-max", type=int, default=64, help="Max per-sample images to write")
    parser.add_argument("--show-ids", action="store_true", help="Show sample_id titles in grid")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap")
    parser.add_argument("--mag-edges", default=None, help="Comma-separated mag edges (e.g., 0,0.01,0.03,0.08,0.25)")
    args = parser.parse_args()

    mag_edges = None
    if args.mag_edges:
        mag_edges = np.array([float(x) for x in args.mag_edges.split(",")], dtype=np.float32)

    inputs = _collect_inputs(args.input, args.input_dir, args.pattern)
    out_root = _ensure_dir(Path(args.out_dir))

    for path in inputs:
        hist, sample_id, index = _load_hist_npz(path)
        if args.normalize:
            hist = _normalize_hist(hist)

        vmax = float(np.max(hist)) if hist.size else 1.0
        vmin = 0.0

        tag = f"{path.parent.name}__{path.stem}"
        out_dir = _ensure_dir(out_root / _sanitize_name(tag))

        _plot_mean(out_dir / "mean.png", hist, vmin=vmin, vmax=vmax, mag_edges=mag_edges)
        _plot_grid(
            out_dir / "grid.png",
            hist,
            sample_id,
            index,
            n=args.grid,
            seed=args.seed,
            vmin=vmin,
            vmax=vmax,
            mag_edges=mag_edges,
            show_ids=args.show_ids,
        )
        if args.per_sample:
            _plot_per_sample(
                out_dir,
                hist,
                sample_id,
                index,
                max_samples=args.per_sample_max,
                vmin=vmin,
                vmax=vmax,
                mag_edges=mag_edges,
            )


if __name__ == "__main__":
    main()
