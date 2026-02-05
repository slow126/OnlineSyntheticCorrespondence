#!/usr/bin/env python3
"""
Flow Gaussian-splat visualizations for ragged flow vectors [x, y, dx, dy].

Produces per-dataset figures:
  (A) Endpoint footprint: cluster endpoints q=(x+dx, y+dy), render elliptical Gaussian splats.
  (B) Directional splats: grid bins; multiple oriented splats per bin; color by flow direction.
  (C) Flow-space density: density over (dx,dy) using a histogram + Gaussian blur (no canceling).

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
import matplotlib.colors as mcolors

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


def flow_direction_colors(
    dx: np.ndarray,
    dy: np.ndarray,
    mag: np.ndarray,
    weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    # Map direction to hue; scale saturation by magnitude and (optional) density weight.
    angle = np.arctan2(dy, dx)
    hue = (angle + np.pi) / (2 * np.pi)
    mag_norm = mag / max(np.percentile(mag, 95), 1e-6)
    sat = np.clip(0.25 + 0.75 * mag_norm, 0.0, 1.0)
    if weight is not None:
        w = weight.astype(np.float32)
        w = w / max(np.percentile(w, 95), 1e-6)
        sat = np.clip(sat * (0.35 + 0.65 * w), 0.0, 1.0)
    val = np.ones_like(hue, dtype=np.float32)
    hsv = np.stack([hue, sat, val], axis=-1).astype(np.float32)
    return mcolors.hsv_to_rgb(hsv)


def add_direction_legend(ax, size: float = 0.22):
    # Draw a small HSV color wheel to indicate direction -> hue mapping.
    inset = ax.inset_axes([0.02, 0.02, size, size])
    n = 200
    ys, xs = np.mgrid[-1:1:complex(0, n), -1:1:complex(0, n)]
    r = np.sqrt(xs**2 + ys**2)
    # Use image-coordinate convention: +y is down.
    ang = np.arctan2(-ys, xs)
    hue = (ang + np.pi) / (2 * np.pi)
    sat = np.clip(r, 0.0, 1.0)
    val = np.ones_like(hue)
    hsv = np.stack([hue, sat, val], axis=-1).astype(np.float32)
    rgb = mcolors.hsv_to_rgb(hsv)
    rgb[r > 1.0] = 1.0
    inset.imshow(rgb, origin="lower")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("dir", fontsize=8, pad=2)

    # Cardinal labels (0° = +x to the right).
    inset.text(0.5, 1.02, "N", ha="center", va="bottom", fontsize=7, transform=inset.transAxes)
    inset.text(1.02, 0.5, "E", ha="left", va="center", fontsize=7, transform=inset.transAxes)
    inset.text(0.5, -0.02, "S", ha="center", va="top", fontsize=7, transform=inset.transAxes)
    inset.text(-0.02, 0.5, "W", ha="right", va="center", fontsize=7, transform=inset.transAxes)


def oriented_cov_from_flow(dx: float, dy: float, base_sigma: float, max_sigma: float) -> np.ndarray:
    mag = float(math.sqrt(dx * dx + dy * dy))
    if mag < 1e-3:
        return np.eye(2, dtype=np.float32) * (base_sigma ** 2)

    ux = dx / mag
    uy = dy / mag
    # Elongate along flow direction, clamp to avoid giant splats.
    sigma_parallel = min(max_sigma, base_sigma + 0.4 * mag)
    sigma_perp = max(1.0, base_sigma * 0.6)

    R = np.array([[ux, -uy], [uy, ux]], dtype=np.float32)
    D = np.diag([sigma_parallel ** 2, sigma_perp ** 2]).astype(np.float32)
    return R @ D @ R.T


def build_grid_bin_splats(
    x: np.ndarray,
    y: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    H: int,
    W: int,
    grid: int,
    k_dir: int,
    min_bin: int,
    base_sigma: float,
    max_sigma: float,
):
    from sklearn.cluster import KMeans

    gx = np.clip((x / max(W, 1e-6) * grid).astype(np.int32), 0, grid - 1)
    gy = np.clip((y / max(H, 1e-6) * grid).astype(np.int32), 0, grid - 1)
    bin_id = gy * grid + gx

    mus = []
    covs = []
    weights = []
    colors = []

    for b in range(grid * grid):
        idx = np.nonzero(bin_id == b)[0]
        if idx.size < min_bin:
            continue

        dx_b = dx[idx]
        dy_b = dy[idx]
        xy_b = np.stack([x[idx], y[idx]], axis=1)

        k = int(min(k_dir, idx.size))
        if k <= 1:
            dxk = float(np.mean(dx_b))
            dyk = float(np.mean(dy_b))
            magk = float(np.mean(np.sqrt(dx_b ** 2 + dy_b ** 2)))
            muk = xy_b.mean(axis=0)
            covs.append(oriented_cov_from_flow(dxk, dyk, base_sigma=base_sigma, max_sigma=max_sigma))
            mus.append(muk.astype(np.float32))
            weights.append(float(idx.size))
            colors.append(flow_direction_colors(
                np.array([dxk], dtype=np.float32),
                np.array([dyk], dtype=np.float32),
                np.array([magk], dtype=np.float32),
                np.array([float(idx.size)], dtype=np.float32),
            )[0])
            continue

        km = KMeans(n_clusters=k, n_init="auto", max_iter=50, random_state=0)
        labels = km.fit_predict(np.stack([dx_b, dy_b], axis=1))

        for kk in range(k):
            idk = (labels == kk)
            nk = int(idk.sum())
            if nk == 0:
                continue
            dxk = float(np.mean(dx_b[idk]))
            dyk = float(np.mean(dy_b[idk]))
            magk = float(np.mean(np.sqrt(dx_b[idk] ** 2 + dy_b[idk] ** 2)))
            muk = xy_b[idk].mean(axis=0)
            covs.append(oriented_cov_from_flow(dxk, dyk, base_sigma=base_sigma, max_sigma=max_sigma))
            mus.append(muk.astype(np.float32))
            weights.append(float(nk))
            colors.append(flow_direction_colors(
                np.array([dxk], dtype=np.float32),
                np.array([dyk], dtype=np.float32),
                np.array([magk], dtype=np.float32),
                np.array([float(nk)], dtype=np.float32),
            )[0])

    if not mus:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )

    return (
        np.stack(mus, axis=0),
        np.stack(covs, axis=0),
        np.log1p(np.asarray(weights, dtype=np.float32)),
        np.stack(colors, axis=0),
    )


def build_cluster_splats(
    x: np.ndarray,
    y: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    K: int,
    seed: int,
    base_sigma: float,
    max_sigma: float,
):
    labels_xy, centers_xy = fit_endpoint_clusters(np.stack([x, y], axis=1), K=K, seed=seed)
    covs_xy = np.zeros((K, 2, 2), dtype=np.float32)
    weights_xy = np.zeros((K,), dtype=np.float32)
    colors_xy = np.zeros((K, 3), dtype=np.float32)

    for k in range(K):
        idx = (labels_xy == k)
        nk = int(idx.sum())
        if nk == 0:
            covs_xy[k] = np.eye(2, dtype=np.float32) * 4.0
            weights_xy[k] = 0.0
            colors_xy[k] = np.array([0.2, 0.2, 0.2], dtype=np.float32)
            continue

        dxk = float(np.mean(dx[idx]))
        dyk = float(np.mean(dy[idx]))
        magk = float(np.mean(np.sqrt(dx[idx] ** 2 + dy[idx] ** 2)))

        covs_xy[k] = oriented_cov_from_flow(dxk, dyk, base_sigma=base_sigma, max_sigma=max_sigma)
        weights_xy[k] = float(nk)
        colors_xy[k] = flow_direction_colors(
            np.array([dxk], dtype=np.float32),
            np.array([dyk], dtype=np.float32),
            np.array([magk], dtype=np.float32),
            np.array([float(nk)], dtype=np.float32),
        )[0]

    weights_xy = np.log1p(weights_xy).astype(np.float32)
    return centers_xy, covs_xy, weights_xy, colors_xy


def build_joint_cluster_splats(
    x: np.ndarray,
    y: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    K: int,
    seed: int,
    base_sigma: float,
    max_sigma: float,
    xy_scale: float,
    flow_scale: float,
):
    from sklearn.cluster import MiniBatchKMeans

    feats = np.stack([x, y, dx, dy], axis=1).astype(np.float32)
    std = np.std(feats, axis=0) + 1e-6
    norm = feats / std
    norm[:, 0:2] *= xy_scale
    norm[:, 2:4] *= flow_scale

    km = MiniBatchKMeans(
        n_clusters=K,
        random_state=seed,
        batch_size=8192,
        n_init="auto",
        max_iter=200,
    )
    labels = km.fit_predict(norm)

    mus = np.zeros((K, 2), dtype=np.float32)
    covs = np.zeros((K, 2, 2), dtype=np.float32)
    weights = np.zeros((K,), dtype=np.float32)
    colors = np.zeros((K, 3), dtype=np.float32)

    for k in range(K):
        idx = (labels == k)
        nk = int(idx.sum())
        if nk == 0:
            covs[k] = np.eye(2, dtype=np.float32) * 4.0
            weights[k] = 0.0
            colors[k] = np.array([0.2, 0.2, 0.2], dtype=np.float32)
            continue

        dxk = float(np.mean(dx[idx]))
        dyk = float(np.mean(dy[idx]))
        magk = float(np.mean(np.sqrt(dx[idx] ** 2 + dy[idx] ** 2)))
        muk = np.array([np.mean(x[idx]), np.mean(y[idx])], dtype=np.float32)

        covs[k] = oriented_cov_from_flow(dxk, dyk, base_sigma=base_sigma, max_sigma=max_sigma)
        mus[k] = muk
        weights[k] = float(nk)
        colors[k] = flow_direction_colors(
            np.array([dxk], dtype=np.float32),
            np.array([dyk], dtype=np.float32),
            np.array([magk], dtype=np.float32),
            np.array([float(nk)], dtype=np.float32),
        )[0]

    weights = np.log1p(weights).astype(np.float32)
    return mus, covs, weights, colors


def splat_gaussians_2d(
    H: int,
    W: int,
    mus: np.ndarray,        # (K,2) in pixel coords (x,y)
    covs: np.ndarray,       # (K,2,2)
    weights: np.ndarray,    # (K,)
    max_radius_px: int = 64,
    soft_edge: float = 0.15,
    support_sigma: float = 3.0,
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
        rx = int(max(3, math.ceil(support_sigma * sigmax)))
        ry = int(max(3, math.ceil(support_sigma * sigmay)))
        if max_radius_px > 0:
            rx = min(max_radius_px, rx)
            ry = min(max_radius_px, ry)

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
        if soft_edge > 0:
            r = np.sqrt((dX / max(rx, 1.0)) ** 2 + (dY / max(ry, 1.0)) ** 2)
            t = np.clip((r - (1.0 - soft_edge)) / max(soft_edge, 1e-6), 0.0, 1.0)
            window = 0.5 * (1.0 + np.cos(np.pi * t))
            patch *= window.astype(np.float32)

        dens[y0:y1, x0:x1] += (w * patch)

    return dens


def splat_gaussians_color(
    H: int,
    W: int,
    mus: np.ndarray,        # (K,2)
    covs: np.ndarray,       # (K,2,2)
    weights: np.ndarray,    # (K,)
    colors: np.ndarray,     # (K,3) RGB in [0,1]
    max_radius_px: int = 64,
    soft_edge: float = 0.15,
    support_sigma: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    dens = np.zeros((H, W), dtype=np.float32)
    accum = np.zeros((H, W, 3), dtype=np.float32)

    for k in range(mus.shape[0]):
        mx, my = mus[k]
        if not (np.isfinite(mx) and np.isfinite(my)):
            continue

        cov = covs[k]
        w = float(weights[k])
        if w <= 0:
            continue

        vals, _ = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-6)
        sigmax = float(math.sqrt(vals[1]))
        sigmay = float(math.sqrt(vals[0]))
        rx = int(max(3, math.ceil(support_sigma * sigmax)))
        ry = int(max(3, math.ceil(support_sigma * sigmay)))
        if max_radius_px > 0:
            rx = min(max_radius_px, rx)
            ry = min(max_radius_px, ry)

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
        if soft_edge > 0:
            r = np.sqrt((dX / max(rx, 1.0)) ** 2 + (dY / max(ry, 1.0)) ** 2)
            t = np.clip((r - (1.0 - soft_edge)) / max(soft_edge, 1e-6), 0.0, 1.0)
            window = 0.5 * (1.0 + np.cos(np.pi * t))
            patch *= window.astype(np.float32)
        patch_w = w * patch
        dens[y0:y1, x0:x1] += patch_w
        accum[y0:y1, x0:x1] += patch_w[..., None] * colors[k]

    dens_tm = tone_map(dens)
    rgb = accum / (dens[..., None] + 1e-6)
    rgb = np.clip(rgb * dens_tm[..., None], 0.0, 1.0)
    return rgb, dens_tm


def tone_map(d: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # log compression to make dense regions visible without saturation
    return np.log1p(d) / (np.log1p(d.max() + eps) + eps)


def flowspace_density(
    dxdy: np.ndarray,
    bins: int = 512,
    clip_q: float = 0.995,
    fixed_ax: Optional[float] = None,
    fixed_ay: Optional[float] = None,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    # robust range based on quantiles to avoid long-tail dominating
    dx = dxdy[:, 0]
    dy = dxdy[:, 1]
    if fixed_ax is None:
        ax = np.quantile(np.abs(dx), clip_q)
    else:
        ax = float(fixed_ax)
    if fixed_ay is None:
        ay = np.quantile(np.abs(dy), clip_q)
    else:
        ay = float(fixed_ay)
    ax = float(max(ax, 1e-3))
    ay = float(max(ay, 1e-3))

    # histogram over [-ax,ax] x [-ay,ay]
    H, xedges, yedges = np.histogram2d(
        dy, dx,  # y first so image row corresponds to dy
        bins=bins,
        range=[[-ay, ay], [-ax, ax]]
    )
    H = H.astype(np.float32)

    extent = (-ax, ax, -ay, ay)
    return tone_map(H), extent


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
    grid: int,
    k_dir: int,
    min_bin: int,
    dir_base_sigma: float,
    dir_max_sigma: float,
    dir_mode: str,
    joint_xy_scale: float,
    joint_flow_scale: float,
    soft_edge: float,
    support_sigma: float,
    flow_range: Optional[float],
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
        max_radius_px=max_radius_px,
        soft_edge=soft_edge,
        support_sigma=support_sigma
    )
    dens_xy_tm = tone_map(dens_xy)

    # Directional splats: grid bins or single splat per spatial cluster
    if dir_mode == "grid":
        mus_xy, covs_xy, weights_xy, colors_xy = build_grid_bin_splats(
            x=x, y=y, dx=dx, dy=dy,
            H=H2, W=W2,
            grid=grid, k_dir=k_dir, min_bin=min_bin,
            base_sigma=dir_base_sigma, max_sigma=dir_max_sigma
        )
        dir_title = "Directional splats (grid bins, hue=direction)"
    elif dir_mode == "cluster":
        mus_xy, covs_xy, weights_xy, colors_xy = build_cluster_splats(
            x=x, y=y, dx=dx, dy=dy,
            K=K, seed=seed,
            base_sigma=dir_base_sigma, max_sigma=dir_max_sigma
        )
        dir_title = "Directional splats (spatial clusters, hue=direction)"
    else:
        mus_xy, covs_xy, weights_xy, colors_xy = build_joint_cluster_splats(
            x=x, y=y, dx=dx, dy=dy,
            K=K, seed=seed,
            base_sigma=dir_base_sigma, max_sigma=dir_max_sigma,
            xy_scale=joint_xy_scale, flow_scale=joint_flow_scale
        )
        dir_title = "Directional splats (joint clusters, hue=direction)"
    rgb_xy, dens_xy_dir = splat_gaussians_color(
        H=H2, W=W2,
        mus=mus_xy, covs=covs_xy, weights=weights_xy,
        colors=colors_xy,
        max_radius_px=max_radius_px,
        soft_edge=soft_edge,
        support_sigma=support_sigma
    )

    dens_uv, dens_extent = flowspace_density(
        np.stack([dx, dy], axis=1),
        bins=flow_bins,
        fixed_ax=flow_range,
        fixed_ay=flow_range,
    )

    # Plot
    fig = plt.figure(figsize=(17, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(dens_xy_tm, origin="upper", cmap="viridis")
    ax0.set_title(f"{dataset_name}  |  Endpoint footprint (Gaussian splats)")
    ax0.set_axis_off()

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(rgb_xy, origin="upper")
    ax1.set_title(f"{dataset_name}  |  {dir_title}")
    ax1.set_axis_off()
    add_direction_legend(ax1)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(dens_uv, origin="upper", cmap="plasma", extent=dens_extent)
    ax2.set_title(f"{dataset_name}  |  Flow-space density (dx, dy)")
    ax2.set_xlabel("dx (pixels)")
    ax2.set_ylabel("dy (pixels)")
    ax2.set_aspect("equal", adjustable="box")

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
    ap.add_argument("--soft_edge", type=float, default=0.15, help="Feather splat edges (0=hard, 0.2=soft)")
    ap.add_argument("--support_sigma", type=float, default=3.0, help="Gaussian support in sigmas (higher = wider)")
    ap.add_argument("--flow_range", type=float, default=None, help="Fixed flow-space range (pixels) for dx/dy axes")
    ap.add_argument("--flow_bins", type=int, default=512, help="Histogram bins for flow-space density")
    ap.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for subsampling and clustering")
    ap.add_argument("--grid", type=int, default=32, help="Grid size for directional splats (grid x grid)")
    ap.add_argument("--k_dir", type=int, default=3, help="Splats per grid bin (direction clusters)")
    ap.add_argument("--min_bin", type=int, default=20, help="Minimum vectors per bin to emit splats")
    ap.add_argument("--dir_base_sigma", type=float, default=3.0, help="Directional splat base sigma (pixels)")
    ap.add_argument("--dir_max_sigma", type=float, default=48.0, help="Directional splat max sigma (pixels)")
    ap.add_argument(
        "--dir_mode",
        choices=["grid", "cluster", "joint"],
        default="grid",
        help="Directional splats mode: grid bins, spatial clusters, or joint (x,y,dx,dy) clusters",
    )
    ap.add_argument("--joint_xy_scale", type=float, default=1.0, help="Joint clustering scale for x,y")
    ap.add_argument("--joint_flow_scale", type=float, default=1.5, help="Joint clustering scale for dx,dy")
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
                grid=args.grid,
                k_dir=args.k_dir,
                min_bin=args.min_bin,
                dir_base_sigma=args.dir_base_sigma,
                dir_max_sigma=args.dir_max_sigma,
                dir_mode=args.dir_mode,
                joint_xy_scale=args.joint_xy_scale,
                joint_flow_scale=args.joint_flow_scale,
                soft_edge=args.soft_edge,
                support_sigma=args.support_sigma,
                flow_range=args.flow_range,
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
