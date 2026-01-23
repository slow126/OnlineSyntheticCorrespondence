#!/usr/bin/env python3
"""
Flow Gaussian-splat visualizations for ragged flow vectors [x, y, dx, dy].

Produces per-dataset figures:
  (A) Endpoint footprint: cluster endpoints q=(x+dx, y+dy), render elliptical Gaussian splats.
  (B) Flow-space density: density over (dx,dy) using a histogram + Gaussian blur (no canceling).

Input formats supported (per dataset file):
  - .npz containing an array under one of: 'flows', 'xydxdy', 'data'
  - .npy containing either (N,4) or object array of ragged chunks
  - .pt (torch) containing tensor or dict with 'flows'
  - .pkl containing array-like
Ragged is handled by concatenating all chunks.

Example:
  python visualize_flow_splats.py \
    --input_dir /path/to/flow_vectors \
    --pattern "*.npz" \
    --out_dir /path/to/out \
    --K 800 \
    --subsample 2000000 \
    --height 384 --width 1248 \
    --dpi 200

If you don't pass --height/--width, it will infer from max x/y in the data.
"""

import argparse
import os
import glob
import math
import pickle
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

def load_any(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npz":
        z = np.load(path, allow_pickle=True)
        for k in ["flows", "xydxdy", "data", "arr_0"]:
            if k in z:
                arr = z[k]
                return _concat_ragged(arr)
        # fallback: first key
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
            # if dict of tensors/lists, try first value
            arr = next(iter(obj.values()))
            return _concat_ragged(arr)
        return _concat_ragged(obj)

    raise ValueError(f"Unsupported file extension: {ext} ({path})")


def _to_numpy(a):
    # torch tensor?
    if hasattr(a, "detach") and hasattr(a, "cpu"):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def _concat_ragged(arr) -> np.ndarray:
    arr = _to_numpy(arr)

    # If it's already (N,4)
    if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 4:
        return arr.astype(np.float32, copy=False)

    # If it's object array / list of chunks
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        chunks = [ _to_numpy(x) for x in arr.tolist() ]
    elif isinstance(arr, (list, tuple)):
        chunks = [ _to_numpy(x) for x in arr ]
    else:
        # Try to coerce to array
        arr2 = np.asarray(arr)
        if arr2.ndim == 2 and arr2.shape[1] == 4:
            return arr2.astype(np.float32, copy=False)
        raise ValueError(f"Could not interpret ragged array of type {type(arr)} with shape {getattr(arr,'shape',None)}")

    chunks2 = []
    for c in chunks:
        c = np.asarray(c)
        if c.size == 0:
            continue
        if c.ndim != 2 or c.shape[1] != 4:
            raise ValueError(f"Chunk must be (N,4), got {c.shape}")
        chunks2.append(c.astype(np.float32, copy=False))

    if not chunks2:
        return np.zeros((0,4), dtype=np.float32)

    return np.concatenate(chunks2, axis=0)


def infer_hw(xy: np.ndarray) -> Tuple[int, int]:
    # infer width/height from x,y max + 1, with a little padding
    if xy.shape[0] == 0:
        return 512, 512
    x = xy[:, 0]
    y = xy[:, 1]
    w = int(np.ceil(np.max(x) + 1))
    h = int(np.ceil(np.max(y) + 1))
    # pad a bit
    w = int(np.ceil(w * 1.02))
    h = int(np.ceil(h * 1.02))
    return h, w


def subsample_rows(arr: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if n <= 0 or arr.shape[0] <= n:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.shape[0], size=n, replace=False)
    return arr[idx]


def fit_endpoint_clusters(q: np.ndarray, K: int, seed: int = 0):
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
    # points: (M,2)
    if points.shape[0] < 3:
        return np.eye(2, dtype=np.float32) * 4.0

    mu = points.mean(axis=0, keepdims=True)
    z = points - mu
    cov = (z.T @ z) / max(points.shape[0] - 1, 1)

    # Regularize: ensure positive definite
    cov = cov.astype(np.float32)
    eps = 1e-3
    cov[0, 0] = max(cov[0, 0], eps)
    cov[1, 1] = max(cov[1, 1], eps)

    # Add small diagonal if near-singular
    det = cov[0,0]*cov[1,1] - cov[0,1]*cov[1,0]
    if det <= eps:
        cov = cov + np.eye(2, dtype=np.float32) * 1.0
    return cov


def splat_gaussians_2d(
    H: int,
    W: int,
    mus: np.ndarray,        # (K,2) in pixel coords (x,y)
    covs: np.ndarray,       # (K,2,2)
    weights: np.ndarray,    # (K,)
    max_radius_px: int = 64,
) -> np.ndarray:
    """
    Rasterize a mixture of 2D Gaussians onto an image grid.
    Returns density map (H,W) float32.
    """
    dens = np.zeros((H, W), dtype=np.float32)

    for k in range(mus.shape[0]):
        mx, my = mus[k]
        if not (np.isfinite(mx) and np.isfinite(my)):
            continue

        cov = covs[k]
        w = float(weights[k])
        if w <= 0:
            continue

        # Compute ellipse extent via eigenvalues (sigma ~ sqrt(lambda))
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-6)
        sigmax = float(math.sqrt(vals[1]))
        sigmay = float(math.sqrt(vals[0]))
        # 3-sigma extent, clipped
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

        # grid
        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)
        dX = X - mx
        dY = Y - my
        D = np.stack([dX, dY], axis=-1)  # (h,w,2)

        # inverse covariance
        inv = np.linalg.inv(cov).astype(np.float32)

        # mahalanobis distance: D^T inv D
        m2 = (
            D[..., 0] * (inv[0, 0] * D[..., 0] + inv[0, 1] * D[..., 1]) +
            D[..., 1] * (inv[1, 0] * D[..., 0] + inv[1, 1] * D[..., 1])
        )

        patch = np.exp(-0.5 * m2).astype(np.float32)

        dens[y0:y1, x0:x1] += (w * patch)

    return dens


def tone_map(d: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # log compression to make dense regions visible without saturation
    return np.log1p(d) / (np.log1p(d.max() + eps) + eps)


def flowspace_density(dxdy: np.ndarray, bins: int = 512, clip_q: float = 0.995) -> np.ndarray:
    # robust range based on quantiles to avoid long-tail dominating
    dx = dxdy[:, 0]
    dy = dxdy[:, 1]
    ax = np.quantile(np.abs(dx), clip_q)
    ay = np.quantile(np.abs(dy), clip_q)
    ax = float(max(ax, 1e-3))
    ay = float(max(ay, 1e-3))

    # histogram over [-ax,ax] x [-ay,ay]
    H, xedges, yedges = np.histogram2d(
        dy, dx,  # y first so image row corresponds to dy
        bins=bins,
        range=[[-ay, ay], [-ax, ax]]
    )
    H = H.astype(np.float32)

    # simple separable blur without scipy (cheap box blur repeated)
    for _ in range(3):
        H = (H + np.roll(H, 1, 0) + np.roll(H, -1, 0)) / 3.0
        H = (H + np.roll(H, 1, 1) + np.roll(H, -1, 1)) / 3.0

    return tone_map(H)


def make_figure_for_dataset(
    flows: np.ndarray,
    dataset_name: str,
    out_path: str,
    H: Optional[int],
    W: Optional[int],
    K: int,
    subsample: int,
    seed: int,
    max_radius_px: int,
    flow_bins: int,
    dpi: int,
):
    if flows.shape[0] == 0:
        print(f"[WARN] {dataset_name}: no flows found, skipping")
        return

    flows = subsample_rows(flows, subsample, seed=seed)
    x = flows[:, 0]
    y = flows[:, 1]
    dx = flows[:, 2]
    dy = flows[:, 3]

    if H is None or W is None:
        H2, W2 = infer_hw(flows[:, :2])
    else:
        H2, W2 = H, W

    # endpoints q = (x+dx, y+dy)
    q = np.stack([x + dx, y + dy], axis=1).astype(np.float32)

    # keep endpoints inside reasonable bounds (optional; helps if flows go off-image a lot)
    # Here: just clip to image bounds for clustering stability
    q[:, 0] = np.clip(q[:, 0], 0, W2 - 1)
    q[:, 1] = np.clip(q[:, 1], 0, H2 - 1)

    # cluster endpoints -> one splat per cluster
    labels, centers = fit_endpoint_clusters(q, K=K, seed=seed)

    # compute cov + weight per cluster (in endpoint space)
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

    # normalize weights (log-ish) so a few huge clusters don't dominate
    weights = np.log1p(weights).astype(np.float32)

    dens_xy = splat_gaussians_2d(
        H=H2, W=W2,
        mus=centers, covs=covs, weights=weights,
        max_radius_px=max_radius_px
    )
    dens_xy_tm = tone_map(dens_xy)

    dens_uv = flowspace_density(np.stack([dx, dy], axis=1), bins=flow_bins)

    # Plot
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(dens_xy_tm, origin="upper", cmap="viridis")
    ax0.set_title(f"{dataset_name}  |  Endpoint footprint (Gaussian splats)")
    ax0.set_axis_off()

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(dens_uv, origin="upper", cmap="plasma")
    ax1.set_title(f"{dataset_name}  |  Flow-space density (dx, dy)")
    ax1.set_axis_off()

    fig.suptitle("Flow distribution visualization (no mean-canceling)", y=1.02)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Generate Gaussian splat visualizations for flow vectors")
    ap.add_argument("--input_dir", required=True, help="Directory containing per-dataset files (.npz/.npy/.pt/.pkl)")
    ap.add_argument("--pattern", default="*.npy", help="Glob pattern for dataset files, e.g. '*.npz' or '*.npy'")
    ap.add_argument("--out_dir", required=True, help="Output directory for PNGs")
    ap.add_argument("--height", type=int, default=None, help="Image height (pixels); infer if not given")
    ap.add_argument("--width", type=int, default=None, help="Image width (pixels); infer if not given")
    ap.add_argument("--K", type=int, default=800, help="Number of Gaussian clusters for endpoint footprint")
    ap.add_argument("--subsample", type=int, default=2000000, help="Max rows to use per dataset (0=no limit)")
    ap.add_argument("--max_radius_px", type=int, default=64, help="Max radius (pixels) for each Gaussian splat")
    ap.add_argument("--flow_bins", type=int, default=512, help="Histogram bins for flow-space density")
    ap.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for subsampling and clustering")
    args = ap.parse_args()

    # Find all matching files
    pattern_path = os.path.join(args.input_dir, args.pattern)
    files = sorted(glob.glob(pattern_path))
    
    if not files:
        print(f"[ERROR] No files found matching pattern: {pattern_path}")
        return

    print(f"[INFO] Found {len(files)} files matching pattern: {args.pattern}")
    print(f"[INFO] Output directory: {args.out_dir}")
    print(f"[INFO] K={args.K}, subsample={args.subsample}, seed={args.seed}")
    print()

    os.makedirs(args.out_dir, exist_ok=True)

    for fpath in files:
        dataset_name = os.path.splitext(os.path.basename(fpath))[0]
        out_name = f"{dataset_name}_splat.png"
        out_path = os.path.join(args.out_dir, out_name)

        print(f"[PROCESSING] {dataset_name} ...")
        try:
            flows = load_any(fpath)
            print(f"  Loaded {flows.shape[0]} flow vectors from {fpath}")
            
            make_figure_for_dataset(
                flows=flows,
                dataset_name=dataset_name,
                out_path=out_path,
                H=args.height,
                W=args.width,
                K=args.K,
                subsample=args.subsample,
                seed=args.seed,
                max_radius_px=args.max_radius_px,
                flow_bins=args.flow_bins,
                dpi=args.dpi,
            )
        except Exception as e:
            print(f"  [ERROR] Failed to process {fpath}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print()
    print("[DONE] All datasets processed!")


if __name__ == "__main__":
    main()
