#!/usr/bin/env python3
"""
Compare flow distributions across multiple datasets side-by-side.

Creates a grid showing key characteristics for each dataset:
  - Flow magnitude distribution
  - Flow direction distribution
  - Flow space hexbin
  - Statistics

Example:
  python visualize_flow_comparison.py \
    --input_dir /mnt/nvme_1tb_b/coverage_vectors \
    --datasets "flyingthings_train_flow.npy,kitti2015_val_flow.npy,spair_test_flow.npy" \
    --out_path comparison.png
"""

import argparse
import os
import pickle
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_any(path: str) -> np.ndarray:
    """Load flow vectors from various file formats."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npz":
        z = np.load(path, allow_pickle=True)
        for k in ["flows", "xydxdy", "data", "arr_0"]:
            if k in z:
                arr = z[k]
                return _concat_ragged(arr)
        keys = list(z.keys())
        if not keys:
            raise ValueError(f"{path}: empty npz")
        return _concat_ragged(z[keys[0]])

    if ext == ".npy":
        arr = np.load(path, allow_pickle=True)
        return _concat_ragged(arr)

    if ext == ".pkl":
        with open(path, "rb") as f:
            arr = pickle.load(f)
        return _concat_ragged(arr)

    if ext == ".pt":
        import torch
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            for k in ["flows", "xydxdy", "data"]:
                if k in obj:
                    return _concat_ragged(obj[k])
            arr = next(iter(obj.values()))
            return _concat_ragged(arr)
        return _concat_ragged(obj)

    raise ValueError(f"Unsupported file extension: {ext} ({path})")


def _to_numpy(a):
    if hasattr(a, "detach") and hasattr(a, "cpu"):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def _concat_ragged(arr) -> np.ndarray:
    arr = _to_numpy(arr)
    if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 4:
        return arr.astype(np.float32, copy=False)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        chunks = [_to_numpy(x) for x in arr.tolist()]
    elif isinstance(arr, (list, tuple)):
        chunks = [_to_numpy(x) for x in arr]
    else:
        arr2 = np.asarray(arr)
        if arr2.ndim == 2 and arr2.shape[1] == 4:
            return arr2.astype(np.float32, copy=False)
        raise ValueError(f"Could not interpret ragged array")

    chunks2 = []
    for c in chunks:
        c = np.asarray(c)
        if c.size == 0:
            continue
        if c.ndim != 2 or c.shape[1] != 4:
            raise ValueError(f"Chunk must be (N,4), got {c.shape}")
        chunks2.append(c.astype(np.float32, copy=False))

    if not chunks2:
        return np.zeros((0, 4), dtype=np.float32)
    return np.concatenate(chunks2, axis=0)


def subsample_rows(arr: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if n <= 0 or arr.shape[0] <= n:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.shape[0], size=n, replace=False)
    return arr[idx]


def load_dataset_flows(input_dir: str, filename: str, subsample: int, seed: int):
    """Load and subsample a single dataset."""
    path = os.path.join(input_dir, filename)
    flows = load_any(path)
    flows = subsample_rows(flows, subsample, seed=seed)
    
    dx = flows[:, 2]
    dy = flows[:, 3]
    mag = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    return {
        'dx': dx,
        'dy': dy,
        'mag': mag,
        'angle': angle,
        'n_total': flows.shape[0],
        'name': os.path.splitext(filename)[0]
    }


def make_comparison_figure(
    datasets: List[dict],
    out_path: str,
    dpi: int,
):
    """Create comparison figure with datasets in rows."""
    n_datasets = len(datasets)
    
    # Create figure: 4 columns (mag dist, direction rose, flow space, stats)
    fig = plt.figure(figsize=(20, 4.5 * n_datasets))
    gs = GridSpec(n_datasets, 4, figure=fig, hspace=0.35, wspace=0.25)
    
    for i, data in enumerate(datasets):
        dx = data['dx']
        dy = data['dy']
        mag = data['mag']
        angle = data['angle']
        name = data['name']
        
        # ===== Column 1: Magnitude distribution =====
        ax_mag = fig.add_subplot(gs[i, 0])
        ax_mag.hist(mag, bins=80, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        mag_mean = np.mean(mag)
        mag_median = np.median(mag)
        ax_mag.axvline(mag_mean, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mag_mean:.2f}')
        ax_mag.axvline(mag_median, color='orange', linestyle='--', linewidth=2,
                       label=f'Median: {mag_median:.2f}')
        ax_mag.set_xlabel('Magnitude (pixels)')
        ax_mag.set_ylabel('Count')
        ax_mag.set_title(f'{name}\nMagnitude Distribution')
        ax_mag.legend(fontsize=9)
        ax_mag.grid(alpha=0.3)
        
        # ===== Column 2: Direction rose =====
        ax_rose = fig.add_subplot(gs[i, 1], projection='polar')
        n_bins = 36
        angle_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        angle_hist, _ = np.histogram(angle, bins=angle_bins, weights=mag)
        
        theta = (angle_bins[:-1] + angle_bins[1:]) / 2
        width = 2 * np.pi / n_bins
        bars = ax_rose.bar(theta, angle_hist, width=width, alpha=0.7, 
                           edgecolor='black', linewidth=0.5)
        
        # Color by direction
        colors = plt.cm.hsv(np.linspace(0, 1, n_bins))
        for bar, color in zip(bars, colors):
            bar.set_facecolor(color)
        
        ax_rose.set_title(f'{name}\nDirection Rose', y=1.08)
        ax_rose.set_theta_zero_location('E')
        ax_rose.set_theta_direction(1)
        
        # ===== Column 3: Flow space hexbin =====
        ax_hex = fig.add_subplot(gs[i, 2])
        dx_lim = np.percentile(np.abs(dx), 99)
        dy_lim = np.percentile(np.abs(dy), 99)
        
        hb = ax_hex.hexbin(dx, dy, gridsize=40, cmap='YlOrRd', mincnt=1,
                           extent=[-dx_lim, dx_lim, -dy_lim, dy_lim])
        ax_hex.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax_hex.axvline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax_hex.set_xlabel('dx (pixels)')
        ax_hex.set_ylabel('dy (pixels)')
        ax_hex.set_title(f'{name}\nFlow Space Density')
        ax_hex.set_aspect('equal')
        plt.colorbar(hb, ax=ax_hex, label='Count', pad=0.02)
        
        # ===== Column 4: Statistics =====
        ax_stats = fig.add_subplot(gs[i, 3])
        ax_stats.axis('off')
        
        stats_text = f"""
STATISTICS

N: {data['n_total']:,}

Magnitude:
  Mean:   {np.mean(mag):.3f} px
  Median: {np.median(mag):.3f} px
  Std:    {np.std(mag):.3f} px
  P95:    {np.percentile(mag, 95):.3f} px
  Max:    {np.max(mag):.3f} px

Components:
  dx: {np.mean(dx):+.3f} ± {np.std(dx):.3f}
  dy: {np.mean(dy):+.3f} ± {np.std(dy):.3f}
        """
        
        ax_stats.text(0.05, 0.95, stats_text.strip(), transform=ax_stats.transAxes,
                      fontsize=10, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle('Flow Distribution Comparison', fontsize=18, fontweight='bold', y=0.995)
    
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Compare flow distributions across datasets")
    ap.add_argument("--input_dir", required=True, help="Directory containing flow files")
    ap.add_argument("--datasets", required=True, 
                    help="Comma-separated list of filenames to compare")
    ap.add_argument("--out_path", required=True, help="Output PNG path")
    ap.add_argument("--subsample", type=int, default=50000,
                    help="Max vectors per dataset")
    ap.add_argument("--dpi", type=int, default=150, help="Output DPI")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    dataset_files = [f.strip() for f in args.datasets.split(',')]
    
    print(f"[INFO] Loading {len(dataset_files)} datasets...")
    print()
    
    datasets = []
    for fname in dataset_files:
        print(f"  Loading {fname}...")
        try:
            data = load_dataset_flows(args.input_dir, fname, args.subsample, args.seed)
            datasets.append(data)
            print(f"    -> {data['n_total']:,} vectors")
        except Exception as e:
            print(f"    [ERROR] {e}")
            continue
    
    if not datasets:
        print("[ERROR] No datasets loaded successfully")
        return
    
    print()
    print("[INFO] Creating comparison figure...")
    make_comparison_figure(datasets, args.out_path, args.dpi)
    print("\n[DONE]")


if __name__ == "__main__":
    main()
