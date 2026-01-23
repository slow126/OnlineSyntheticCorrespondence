#!/usr/bin/env python3
"""
Aggregate flow Gaussian-splat visualization across multiple datasets.

This script loads flow vectors from multiple datasets, aggregates them together,
and produces a single combined visualization showing the overall distribution
across all datasets.

Example:
  python visualize_aggregated_splats.py \
    --input_dir /mnt/nvme_1tb_b/coverage_vectors \
    --pattern "*_flow.npy" \
    --out_path aggregated_splat.png \
    --K 1500 \
    --subsample_per_dataset 500000
"""

import argparse
import os
import glob
import math
import pickle
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt


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
        raise ValueError(f"Could not interpret ragged array of type {type(arr)}")

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


def infer_hw(xy: np.ndarray) -> Tuple[int, int]:
    """Infer image dimensions from x,y coordinates."""
    if xy.shape[0] == 0:
        return 512, 512
    x = xy[:, 0]
    y = xy[:, 1]
    w = int(np.ceil(np.max(x) + 1))
    h = int(np.ceil(np.max(y) + 1))
    w = int(np.ceil(w * 1.02))
    h = int(np.ceil(h * 1.02))
    return h, w


def fit_endpoint_clusters(q: np.ndarray, K: int, seed: int = 0):
    """Cluster flow endpoints using MiniBatch K-Means."""
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(
        n_clusters=K,
        random_state=seed,
        batch_size=8192,
        n_init="auto",
        max_iter=200
    )
    labels = km.fit_predict(q)
    centers = km.cluster_centers_.astype(np.float32)
    return labels, centers


def robust_cov_2d(points: np.ndarray) -> np.ndarray:
    """Compute robust 2D covariance matrix."""
    if points.shape[0] < 3:
        return np.eye(2, dtype=np.float32) * 4.0

    mu = points.mean(axis=0, keepdims=True)
    z = points - mu
    cov = (z.T @ z) / max(points.shape[0] - 1, 1)

    cov = cov.astype(np.float32)
    eps = 1e-3
    cov[0, 0] = max(cov[0, 0], eps)
    cov[1, 1] = max(cov[1, 1], eps)

    det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
    if det <= eps:
        cov = cov + np.eye(2, dtype=np.float32) * 1.0
    return cov


def splat_gaussians_2d(
    H: int,
    W: int,
    mus: np.ndarray,
    covs: np.ndarray,
    weights: np.ndarray,
    max_radius_px: int = 64,
) -> np.ndarray:
    """Rasterize Gaussian mixture onto image grid."""
    dens = np.zeros((H, W), dtype=np.float32)

    for k in range(mus.shape[0]):
        mx, my = mus[k]
        if not (np.isfinite(mx) and np.isfinite(my)):
            continue

        cov = covs[k]
        w = float(weights[k])
        if w <= 0:
            continue

        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-6)
        sigmax = float(math.sqrt(vals[1]))
        sigmay = float(math.sqrt(vals[0]))
        rx = int(min(max_radius_px, max(3, math.ceil(3.0 * sigmax))))
        ry = int(min(max_radius_px, max(3, math.ceil(3.0 * sigmay))))

        cx = int(round(mx))
        cy = int(round(my))

        x0 = max(0, cx - rx)
        x1 = min(W, cx + rx + 1)
        y0 = max(0, cy - ry)
        y1 = min(H, cy + ry + 1)
        if x1 <= x0 or y1 <= y0:
            continue

        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)
        dX = X - mx
        dY = Y - my
        D = np.stack([dX, dY], axis=-1)

        inv = np.linalg.inv(cov).astype(np.float32)
        m2 = (
            D[..., 0] * (inv[0, 0] * D[..., 0] + inv[0, 1] * D[..., 1]) +
            D[..., 1] * (inv[1, 0] * D[..., 0] + inv[1, 1] * D[..., 1])
        )

        patch = np.exp(-0.5 * m2).astype(np.float32)
        dens[y0:y1, x0:x1] += (w * patch)

    return dens


def tone_map(d: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply log-scale tone mapping."""
    return np.log1p(d) / (np.log1p(d.max() + eps) + eps)


def flowspace_density(dxdy: np.ndarray, bins: int = 512, clip_q: float = 0.995) -> np.ndarray:
    """Compute flow-space density using histogram + blur."""
    dx = dxdy[:, 0]
    dy = dxdy[:, 1]
    ax = np.quantile(np.abs(dx), clip_q)
    ay = np.quantile(np.abs(dy), clip_q)
    ax = float(max(ax, 1e-3))
    ay = float(max(ay, 1e-3))

    H, xedges, yedges = np.histogram2d(
        dy, dx,
        bins=bins,
        range=[[-ay, ay], [-ax, ax]]
    )
    H = H.astype(np.float32)

    for _ in range(3):
        H = (H + np.roll(H, 1, 0) + np.roll(H, -1, 0)) / 3.0
        H = (H + np.roll(H, 1, 1) + np.roll(H, -1, 1)) / 3.0

    return tone_map(H)


def load_and_aggregate_datasets(
    file_paths: List[str],
    subsample_per_dataset: int,
    seed: int = 0
) -> np.ndarray:
    """Load multiple datasets and aggregate them together."""
    all_flows = []
    
    for fpath in file_paths:
        dataset_name = os.path.splitext(os.path.basename(fpath))[0]
        try:
            flows = load_any(fpath)
            print(f"  Loaded {flows.shape[0]:,} vectors from {dataset_name}")
            
            if flows.shape[0] > 0:
                flows_sub = subsample_rows(flows, subsample_per_dataset, seed=seed)
                all_flows.append(flows_sub)
                print(f"    -> Subsampled to {flows_sub.shape[0]:,} vectors")
        except Exception as e:
            print(f"  [ERROR] Failed to load {fpath}: {e}")
            continue
    
    if not all_flows:
        raise ValueError("No datasets loaded successfully")
    
    aggregated = np.concatenate(all_flows, axis=0)
    print(f"\n[INFO] Total aggregated vectors: {aggregated.shape[0]:,}")
    return aggregated


def make_aggregated_figure(
    flows: np.ndarray,
    out_path: str,
    H: Optional[int],
    W: Optional[int],
    K: int,
    subsample_final: int,
    seed: int,
    max_radius_px: int,
    flow_bins: int,
    dpi: int,
    n_datasets: int,
):
    """Generate combined visualization from aggregated flows."""
    if flows.shape[0] == 0:
        print("[WARN] No flows to visualize")
        return

    # Final subsampling if needed
    flows = subsample_rows(flows, subsample_final, seed=seed)
    print(f"[INFO] Using {flows.shape[0]:,} vectors for visualization")

    x = flows[:, 0]
    y = flows[:, 1]
    dx = flows[:, 2]
    dy = flows[:, 3]

    if H is None or W is None:
        H2, W2 = infer_hw(flows[:, :2])
        print(f"[INFO] Inferred dimensions: {H2}x{W2}")
    else:
        H2, W2 = H, W

    # Endpoints
    q = np.stack([x + dx, y + dy], axis=1).astype(np.float32)
    q[:, 0] = np.clip(q[:, 0], 0, W2 - 1)
    q[:, 1] = np.clip(q[:, 1], 0, H2 - 1)

    # Cluster
    print(f"[INFO] Clustering into K={K} Gaussians...")
    labels, centers = fit_endpoint_clusters(q, K=K, seed=seed)

    # Compute cluster statistics
    covs = np.zeros((K, 2, 2), dtype=np.float32)
    weights = np.zeros((K,), dtype=np.float32)

    for k in range(K):
        idx = (labels == k)
        nk = int(idx.sum())
        if nk == 0:
            covs[k] = np.eye(2, dtype=np.float32) * 4.0
            weights[k] = 0.0
            continue
        pts = q[idx]
        covs[k] = robust_cov_2d(pts)
        weights[k] = float(nk)

    weights = np.log1p(weights).astype(np.float32)

    # Render
    print("[INFO] Rendering Gaussian splats...")
    dens_xy = splat_gaussians_2d(
        H=H2, W=W2,
        mus=centers, covs=covs, weights=weights,
        max_radius_px=max_radius_px
    )
    dens_xy_tm = tone_map(dens_xy)

    print("[INFO] Computing flow-space density...")
    dens_uv = flowspace_density(np.stack([dx, dy], axis=1), bins=flow_bins)

    # Plot
    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(dens_xy_tm, origin="upper", cmap="viridis")
    ax0.set_title(f"Aggregated Endpoint Footprint (Gaussian splats)\n{n_datasets} datasets, {flows.shape[0]:,} vectors")
    ax0.set_axis_off()

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(dens_uv, origin="upper", cmap="plasma")
    ax1.set_title(f"Aggregated Flow-space Density (dx, dy)\n{n_datasets} datasets")
    ax1.set_axis_off()

    fig.suptitle("Multi-Dataset Aggregated Flow Visualization", y=1.02, fontsize=14, fontweight='bold')
    
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Generate aggregated Gaussian splat visualization across multiple datasets")
    ap.add_argument("--input_dir", required=True, help="Directory containing flow files")
    ap.add_argument("--pattern", default="*_flow.npy", help="Glob pattern for dataset files")
    ap.add_argument("--out_path", required=True, help="Output PNG file path")
    ap.add_argument("--height", type=int, default=None, help="Image height (infer if not given)")
    ap.add_argument("--width", type=int, default=None, help="Image width (infer if not given)")
    ap.add_argument("--K", type=int, default=1500, help="Number of Gaussian clusters")
    ap.add_argument("--subsample_per_dataset", type=int, default=500000, 
                    help="Max vectors per dataset before aggregation")
    ap.add_argument("--subsample_final", type=int, default=5000000,
                    help="Max total vectors after aggregation (0=no limit)")
    ap.add_argument("--max_radius_px", type=int, default=64, help="Max splat radius")
    ap.add_argument("--flow_bins", type=int, default=512, help="Flow-space histogram bins")
    ap.add_argument("--dpi", type=int, default=250, help="Output DPI")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    # Find files
    pattern_path = os.path.join(args.input_dir, args.pattern)
    files = sorted(glob.glob(pattern_path))

    if not files:
        print(f"[ERROR] No files found matching: {pattern_path}")
        return

    print(f"[INFO] Found {len(files)} datasets matching pattern: {args.pattern}")
    print(f"[INFO] Subsample per dataset: {args.subsample_per_dataset:,}")
    print(f"[INFO] Final subsample: {args.subsample_final:,}")
    print()

    # Load and aggregate
    print("[LOADING DATASETS]")
    flows_agg = load_and_aggregate_datasets(
        file_paths=files,
        subsample_per_dataset=args.subsample_per_dataset,
        seed=args.seed
    )

    # Generate figure
    print("\n[GENERATING VISUALIZATION]")
    make_aggregated_figure(
        flows=flows_agg,
        out_path=args.out_path,
        H=args.height,
        W=args.width,
        K=args.K,
        subsample_final=args.subsample_final,
        seed=args.seed,
        max_radius_px=args.max_radius_px,
        flow_bins=args.flow_bins,
        dpi=args.dpi,
        n_datasets=len(files),
    )

    print("\n[DONE]")


if __name__ == "__main__":
    main()
