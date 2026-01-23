#!/usr/bin/env python3
"""
Alternative flow visualizations that are easier to interpret.

Produces multiple visualization types:
  1. Vector field (quiver plot) - see actual flow vectors
  2. Flow magnitude heatmap - see motion intensity
  3. Flow direction rose - circular histogram of directions
  4. Magnitude distribution - histogram of flow magnitudes
  5. Hexbin density - 2D density in flow space
  6. Flow statistics summary

Example:
  python visualize_flow_alternatives.py \
    --input_dir /mnt/nvme_1tb_b/coverage_vectors \
    --pattern "*_flow.npy" \
    --out_dir ./alternative_vis \
    --subsample 50000
"""

import argparse
import os
import glob
import pickle
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
                    arr = obj[k]
                    return _concat_ragged(arr)
            arr = next(iter(obj.values()))
            return _concat_ragged(arr)
        return _concat_ragged(obj)

    raise ValueError(f"Unsupported file extension: {ext} ({path})")


def _to_numpy(a):
    if hasattr(a, "detach") and hasattr(a, "cpu"):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def _concat_ragged(arr) -> np.ndarray:
    """Handle ragged arrays and convert to (N,4) format."""
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
    """Subsample rows from array."""
    if n <= 0 or arr.shape[0] <= n:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.shape[0], size=n, replace=False)
    return arr[idx]


def compute_flow_stats(dx: np.ndarray, dy: np.ndarray) -> dict:
    """Compute statistical summary of flow vectors."""
    mag = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx) * 180 / np.pi  # degrees
    
    return {
        'n_vectors': len(dx),
        'mag_mean': float(np.mean(mag)),
        'mag_median': float(np.median(mag)),
        'mag_std': float(np.std(mag)),
        'mag_p95': float(np.percentile(mag, 95)),
        'mag_max': float(np.max(mag)),
        'dx_mean': float(np.mean(dx)),
        'dy_mean': float(np.mean(dy)),
        'dx_std': float(np.std(dx)),
        'dy_std': float(np.std(dy)),
    }


def make_alternative_visualizations(
    flows: np.ndarray,
    dataset_name: str,
    out_path: str,
    subsample: int,
    quiver_subsample: int,
    seed: int,
    dpi: int,
):
    """Generate multiple alternative flow visualizations."""
    if flows.shape[0] == 0:
        print(f"[WARN] {dataset_name}: no flows found, skipping")
        return

    # Subsample for visualization
    flows = subsample_rows(flows, subsample, seed=seed)
    x = flows[:, 0]
    y = flows[:, 1]
    dx = flows[:, 2]
    dy = flows[:, 3]
    
    mag = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)  # radians
    
    # Compute statistics
    stats = compute_flow_stats(dx, dy)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # ========== 1. Quiver plot (vector field) ==========
    ax1 = fig.add_subplot(gs[0, 0])
    quiver_flows = subsample_rows(flows, quiver_subsample, seed=seed)
    qx, qy, qdx, qdy = quiver_flows[:, 0], quiver_flows[:, 1], quiver_flows[:, 2], quiver_flows[:, 3]
    qmag = np.sqrt(qdx**2 + qdy**2)
    
    q = ax1.quiver(qx, qy, qdx, qdy, qmag, 
                   cmap='viridis', alpha=0.6, scale_units='xy', 
                   angles='xy', scale=1, width=0.003)
    ax1.set_title(f'Vector Field (quiver)\n{len(qx):,} vectors')
    ax1.set_xlabel('x (pixels)')
    ax1.set_ylabel('y (pixels)')
    ax1.set_aspect('equal', adjustable='datalim')
    ax1.invert_yaxis()
    plt.colorbar(q, ax=ax1, label='Magnitude')
    
    # ========== 2. Magnitude heatmap (2D histogram) ==========
    ax2 = fig.add_subplot(gs[0, 1])
    # Determine spatial extent
    x_bins = min(100, int(np.ptp(x) / 10) + 1)
    y_bins = min(100, int(np.ptp(y) / 10) + 1)
    
    H, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=mag)
    counts, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
    
    # Average magnitude per bin
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_mag = H / np.maximum(counts, 1)
        avg_mag[counts == 0] = 0
    
    im2 = ax2.imshow(avg_mag.T, origin='lower', aspect='auto', cmap='hot',
                     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    ax2.set_title('Average Flow Magnitude Heatmap')
    ax2.set_xlabel('x (pixels)')
    ax2.set_ylabel('y (pixels)')
    plt.colorbar(im2, ax=ax2, label='Avg Magnitude')
    
    # ========== 3. Flow direction rose (circular histogram) ==========
    ax3 = fig.add_subplot(gs[0, 2], projection='polar')
    n_bins = 36  # 10-degree bins
    angle_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    angle_hist, _ = np.histogram(angle, bins=angle_bins, weights=mag)
    
    # Plot as bar chart
    theta = (angle_bins[:-1] + angle_bins[1:]) / 2
    width = 2 * np.pi / n_bins
    bars = ax3.bar(theta, angle_hist, width=width, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Color by direction
    colors = plt.cm.hsv(np.linspace(0, 1, n_bins))
    for bar, color in zip(bars, colors):
        bar.set_facecolor(color)
    
    ax3.set_title('Flow Direction Rose\n(magnitude-weighted)', y=1.08)
    ax3.set_theta_zero_location('E')
    ax3.set_theta_direction(1)
    
    # ========== 4. Magnitude distribution ==========
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(mag, bins=100, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    ax4.axvline(stats['mag_mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mag_mean']:.2f}")
    ax4.axvline(stats['mag_median'], color='orange', linestyle='--', linewidth=2, label=f"Median: {stats['mag_median']:.2f}")
    ax4.axvline(stats['mag_p95'], color='green', linestyle='--', linewidth=2, label=f"P95: {stats['mag_p95']:.2f}")
    ax4.set_xlabel('Flow Magnitude (pixels)')
    ax4.set_ylabel('Count')
    ax4.set_title('Magnitude Distribution')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # ========== 5. Hexbin density in flow space ==========
    ax5 = fig.add_subplot(gs[1, 1])
    # Robust range
    dx_lim = np.percentile(np.abs(dx), 99)
    dy_lim = np.percentile(np.abs(dy), 99)
    
    hb = ax5.hexbin(dx, dy, gridsize=50, cmap='YlOrRd', mincnt=1,
                    extent=[-dx_lim, dx_lim, -dy_lim, dy_lim])
    ax5.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax5.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax5.set_xlabel('dx (pixels)')
    ax5.set_ylabel('dy (pixels)')
    ax5.set_title('Flow Space Density (hexbin)')
    ax5.set_aspect('equal')
    plt.colorbar(hb, ax=ax5, label='Count')
    
    # ========== 6. Statistics summary ==========
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    stats_text = f"""
    FLOW STATISTICS
    
    N vectors: {stats['n_vectors']:,}
    
    Magnitude:
      Mean:   {stats['mag_mean']:.3f} px
      Median: {stats['mag_median']:.3f} px
      Std:    {stats['mag_std']:.3f} px
      P95:    {stats['mag_p95']:.3f} px
      Max:    {stats['mag_max']:.3f} px
    
    Components:
      dx mean: {stats['dx_mean']:+.3f} px
      dy mean: {stats['dy_mean']:+.3f} px
      dx std:  {stats['dx_std']:.3f} px
      dy std:  {stats['dy_std']:.3f} px
    """
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Main title
    fig.suptitle(f'{dataset_name} - Flow Distribution Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Alternative flow visualizations")
    ap.add_argument("--input_dir", required=True, help="Directory containing flow files")
    ap.add_argument("--pattern", default="*.npy", help="Glob pattern for files")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--subsample", type=int, default=50000, 
                    help="Max vectors for histograms/density plots")
    ap.add_argument("--quiver_subsample", type=int, default=2000,
                    help="Max vectors for quiver plot (keep low for readability)")
    ap.add_argument("--dpi", type=int, default=150, help="Output DPI")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    # Find files
    pattern_path = os.path.join(args.input_dir, args.pattern)
    files = sorted(glob.glob(pattern_path))

    if not files:
        print(f"[ERROR] No files found matching: {pattern_path}")
        return

    print(f"[INFO] Found {len(files)} files")
    print(f"[INFO] Output directory: {args.out_dir}")
    print()

    os.makedirs(args.out_dir, exist_ok=True)

    for fpath in files:
        dataset_name = os.path.splitext(os.path.basename(fpath))[0]
        out_name = f"{dataset_name}_alternatives.png"
        out_path = os.path.join(args.out_dir, out_name)

        print(f"[PROCESSING] {dataset_name}...")
        try:
            flows = load_any(fpath)
            print(f"  Loaded {flows.shape[0]:,} vectors")

            make_alternative_visualizations(
                flows=flows,
                dataset_name=dataset_name,
                out_path=out_path,
                subsample=args.subsample,
                quiver_subsample=args.quiver_subsample,
                seed=args.seed,
                dpi=args.dpi,
            )
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n[DONE]")


if __name__ == "__main__":
    main()
