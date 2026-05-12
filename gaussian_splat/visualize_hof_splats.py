#!/usr/bin/env python3
"""
Visualize HOF fingerprints with Gaussian splat-style flow plots.

This script:
1) Reads diagnostic CSVs (from scripts/hof_diagnostic_bins.py)
2) Converts HOF fingerprints into synthetic flow vectors (x, y, dx, dy)
3) Renders splat visualizations using visualize_flow_splats.py

It does NOT modify existing gaussian_splat tools.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import renderer from existing tool (no modification required)
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import visualize_flow_splats as vfs  # type: ignore


def _load_manifest_from_cache_path(cache_path: Path) -> Optional[Dict]:
    # cache_path: /.../hof_cache/{dataset}/{split}/{index}.npz
    cache_dir = cache_path.parent
    manifest = cache_dir / "manifest.json"
    if not manifest.exists():
        return None
    with manifest.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_hof_config_from_manifest(manifest: Optional[Dict]) -> Optional[Dict]:
    if not manifest:
        return None
    return manifest.get("hof_config")


def _split_fingerprint(
    fp: np.ndarray,
    grid_hw: Tuple[int, int],
    angle_bins: int,
    mag_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    gh, gw = grid_hw
    per_cell = 1 + angle_bins * mag_bins
    expected = gh * gw * per_cell
    if fp.shape[0] != expected:
        raise ValueError(f"Unexpected fingerprint dim={fp.shape[0]} (expected {expected})")
    cells = fp.reshape(gh * gw, per_cell)
    occ = cells[:, 0].reshape(gh, gw)
    hist = cells[:, 1:].reshape(gh, gw, angle_bins, mag_bins)
    return occ, hist


def _infer_hw_from_samples(cache_paths: List[str], max_probe: int = 50) -> Tuple[int, int]:
    hs, ws = [], []
    for p in cache_paths[:max_probe]:
        with np.load(p) as data:
            h = int(data.get("height", 0))
            w = int(data.get("width", 0))
        if h > 0 and w > 0:
            hs.append(h)
            ws.append(w)
    if not hs or not ws:
        return 512, 512
    return int(np.median(hs)), int(np.median(ws))


def _sample_counts_from_hist(
    weights: np.ndarray,
    n_total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    # weights shape: (angle_bins, mag_bins)
    flat = weights.reshape(-1).astype(np.float64, copy=False)
    if n_total <= 0:
        return np.zeros_like(flat, dtype=np.int64).reshape(weights.shape)

    # sanitize
    flat = np.where(np.isfinite(flat) & (flat > 0), flat, 0.0)
    total = float(flat.sum())
    if total <= 0:
        return np.zeros_like(flat, dtype=np.int64).reshape(weights.shape)

    probs = flat / total
    # Renormalize to avoid numerical drift > 1.0
    s = float(probs.sum())
    if not np.isfinite(s) or s <= 0:
        return np.zeros_like(flat, dtype=np.int64).reshape(weights.shape)
    probs = probs / s
    # Clip and renormalize one more time to be safe
    probs = np.clip(probs, 0.0, 1.0)
    s2 = float(probs.sum())
    if s2 <= 0:
        return np.zeros_like(flat, dtype=np.int64).reshape(weights.shape)
    probs = probs / s2

    counts = rng.multinomial(n_total, probs.astype(np.float64, copy=False))
    return counts.reshape(weights.shape)


def _flow_vectors_from_hof(
    cache_paths: List[str],
    grid_hw: Tuple[int, int],
    angle_bins: int,
    mag_edges: np.ndarray,
    out_hw: Tuple[int, int],
    samples_per_cell: int,
    max_samples: Optional[int],
    max_flows: Optional[int],
    seed: int,
    min_valid_count: int,
    cell_jitter: bool,
    cell_jitter_frac: float,
) -> np.ndarray:
    gh, gw = grid_hw
    mag_bins = len(mag_edges) - 1
    mag_mid = 0.5 * (mag_edges[:-1] + mag_edges[1:])

    rng = np.random.default_rng(seed)
    if max_samples is not None and len(cache_paths) > max_samples:
        cache_paths = list(rng.choice(cache_paths, size=max_samples, replace=False))

    H_out, W_out = out_hw
    flows: List[np.ndarray] = []

    # Precompute cell centers and bounds in pixel coords
    cell_w = float(W_out) / float(gw)
    cell_h = float(H_out) / float(gh)
    xs = (np.arange(gw) + 0.5) * cell_w
    ys = (np.arange(gh) + 0.5) * cell_h
    cx = np.tile(xs[None, :], (gh, 1)).astype(np.float32)
    cy = np.tile(ys[:, None], (1, gw)).astype(np.float32)

    # Jitter bounds
    cell_jitter_frac = float(np.clip(cell_jitter_frac, 0.1, 1.0))
    pad_x = 0.5 * (1.0 - cell_jitter_frac) * cell_w
    pad_y = 0.5 * (1.0 - cell_jitter_frac) * cell_h

    angle_edges = np.linspace(0.0, 2.0 * np.pi, angle_bins + 1, endpoint=True)

    for path in cache_paths:
        with np.load(path) as data:
            fp = data["fingerprint"]
            valid_count = int(data.get("valid_count", 0))
        if valid_count < min_valid_count:
            continue

        occ, hist = _split_fingerprint(fp, grid_hw, angle_bins, mag_bins)

        # For each cell, sample flows from histogram
        for i in range(gh):
            for j in range(gw):
                occ_ij = float(occ[i, j])
                if occ_ij <= 0:
                    continue
                n_cell = int(round(occ_ij * samples_per_cell))
                if n_cell <= 0:
                    continue

                weights = hist[i, j] * occ_ij
                counts = _sample_counts_from_hist(weights, n_cell, rng)
                if counts.sum() == 0:
                    continue

                # Convert counts into flow vectors
                for a in range(angle_bins):
                    for m in range(mag_bins):
                        c = int(counts[a, m])
                        if c <= 0:
                            continue
                        # jitter angle within bin
                        theta0 = angle_edges[a]
                        theta1 = angle_edges[a + 1]
                        theta = rng.uniform(theta0, theta1, size=c)

                        # jitter magnitude within bin
                        mag0 = mag_edges[m]
                        mag1 = mag_edges[m + 1]
                        mag = rng.uniform(mag0, mag1, size=c)

                        dx = (mag * np.cos(theta) * W_out).astype(np.float32)
                        dy = (mag * np.sin(theta) * H_out).astype(np.float32)

                        if cell_jitter:
                            x0 = j * cell_w + pad_x
                            x1 = (j + 1) * cell_w - pad_x
                            y0 = i * cell_h + pad_y
                            y1 = (i + 1) * cell_h - pad_y
                            x = rng.uniform(x0, x1, size=c).astype(np.float32)
                            y = rng.uniform(y0, y1, size=c).astype(np.float32)
                        else:
                            x = np.full(c, cx[i, j], dtype=np.float32)
                            y = np.full(c, cy[i, j], dtype=np.float32)
                        flows.append(np.stack([x, y, dx, dy], axis=1))

    if not flows:
        return np.zeros((0, 4), dtype=np.float32)

    flows_arr = np.concatenate(flows, axis=0)
    if max_flows is not None and flows_arr.shape[0] > max_flows:
        idx = rng.choice(flows_arr.shape[0], size=max_flows, replace=False)
        flows_arr = flows_arr[idx]
    return flows_arr.astype(np.float32, copy=False)


def _collect_csvs(input_path: Optional[str], input_dir: Optional[str], pattern: str) -> List[Path]:
    if input_path:
        return [Path(input_path)]
    if input_dir:
        base = Path(input_dir)
        return sorted(base.rglob(pattern))
    raise ValueError("Provide --csv or --csv-dir")


def _write_placeholder_image(out_path: Path, title: str, subtitle: str, dpi: int = 200) -> None:
    fig = plt.figure(figsize=(6.0, 5.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axis_off()
    ax.set_facecolor("#f2f2f2")
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=14, weight="bold")
    ax.text(0.5, 0.45, subtitle, ha="center", va="center", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_hof_figure(
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
    show_endpoint: bool,
    show_flowspace: bool,
    legend_outside: bool,
    title_mode: str,
) -> None:
    if flows.shape[0] == 0:
        print(f"[WARN] {dataset_name}: no flows found, skipping")
        return

    flows = vfs.subsample_rows(flows, subsample, seed=seed)
    x = flows[:, 0]
    y = flows[:, 1]
    dx = flows[:, 2]
    dy = flows[:, 3]

    if H is None or W is None:
        H2, W2 = vfs.infer_hw(flows[:, :2])
    else:
        H2, W2 = H, W

    panels = []
    titles = []
    dir_panel_index = None

    if show_endpoint:
        q = np.stack([x + dx, y + dy], axis=1).astype(np.float32)
        q[:, 0] = np.clip(q[:, 0], 0, W2 - 1)
        q[:, 1] = np.clip(q[:, 1], 0, H2 - 1)

        labels, centers = vfs.fit_endpoint_clusters(q, K=K, seed=seed)
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
            covs[k] = vfs.robust_cov_2d(pts)
            weights[k] = float(nk)
        weights = np.log1p(weights).astype(np.float32)
        dens_xy = vfs.splat_gaussians_2d(
            H=H2,
            W=W2,
            mus=centers,
            covs=covs,
            weights=weights,
            max_radius_px=max_radius_px,
            soft_edge=soft_edge,
            support_sigma=support_sigma,
        )
        dens_xy = vfs.tone_map(dens_xy)
        panels.append(("image", dens_xy, {"cmap": "viridis"}))
        titles.append("Endpoint footprint (Gaussian splats)")

    if dir_mode == "grid":
        mus_xy, covs_xy, weights_xy, colors_xy = vfs.build_grid_bin_splats(
            x=x,
            y=y,
            dx=dx,
            dy=dy,
            H=H2,
            W=W2,
            grid=grid,
            k_dir=k_dir,
            min_bin=min_bin,
            base_sigma=dir_base_sigma,
            max_sigma=dir_max_sigma,
        )
        dir_title = "Directional splats (grid bins)"
    elif dir_mode == "cluster":
        mus_xy, covs_xy, weights_xy, colors_xy = vfs.build_cluster_splats(
            x=x,
            y=y,
            dx=dx,
            dy=dy,
            K=K,
            seed=seed,
            base_sigma=dir_base_sigma,
            max_sigma=dir_max_sigma,
        )
        dir_title = "Directional splats (spatial clusters)"
    else:
        mus_xy, covs_xy, weights_xy, colors_xy = vfs.build_joint_cluster_splats(
            x=x,
            y=y,
            dx=dx,
            dy=dy,
            K=K,
            seed=seed,
            base_sigma=dir_base_sigma,
            max_sigma=dir_max_sigma,
            xy_scale=joint_xy_scale,
            flow_scale=joint_flow_scale,
        )
        dir_title = "Directional splats (joint clusters)"

    rgb_xy, _ = vfs.splat_gaussians_color(
        H=H2,
        W=W2,
        mus=mus_xy,
        covs=covs_xy,
        weights=weights_xy,
        colors=colors_xy,
        max_radius_px=max_radius_px,
        soft_edge=soft_edge,
        support_sigma=support_sigma,
    )
    panels.append(("image", rgb_xy, {}))
    titles.append(dir_title)
    dir_panel_index = len(panels) - 1

    if show_flowspace:
        dens_uv, dens_extent = vfs.flowspace_density(
            np.stack([dx, dy], axis=1),
            bins=flow_bins,
            fixed_ax=flow_range,
            fixed_ay=flow_range,
        )
        panels.append(("flowspace", dens_uv, {"extent": dens_extent, "cmap": "plasma"}))
        titles.append("Flow-space density (dx, dy)")

    if legend_outside and dir_panel_index is not None:
        panels.append(("legend", None, {}))
        titles.append("Direction legend")

    ncols = len(panels)
    if ncols == 0:
        print(f"[WARN] {dataset_name}: no panels to render")
        return

    width_ratios = []
    for ptype, _, _ in panels:
        if ptype == "legend":
            width_ratios.append(0.35)
        elif ptype == "flowspace":
            width_ratios.append(1.0)
        else:
            width_ratios.append(1.1)

    fig = plt.figure(figsize=(6.0 * sum(width_ratios), 5.0), constrained_layout=True)
    gs = fig.add_gridspec(1, ncols, width_ratios=width_ratios)

    for i, (ptype, img, kw) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        if ptype == "legend":
            ax.set_axis_off()
            vfs.add_direction_legend(ax)
        elif ptype == "flowspace":
            ax.imshow(img, origin="upper", **kw)
            ax.set_xlabel("dx (pixels)")
            ax.set_ylabel("dy (pixels)")
            ax.set_aspect("equal", adjustable="box")
        else:
            ax.imshow(img, origin="upper", **kw)
            ax.set_axis_off()
            if (not legend_outside) and (dir_panel_index is not None) and i == dir_panel_index:
                vfs.add_direction_legend(ax)
        if title_mode == "full":
            ax.set_title(f"{dataset_name} | {titles[i]}")
        elif title_mode == "short":
            ax.set_title(f"{titles[i]}")

    if title_mode in {"short", "full"}:
        fig.suptitle(dataset_name, y=1.02, fontsize=12)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Gaussian-splat visualization for HOF diagnostics")
    ap.add_argument("--csv", default=None, help="Diagnostic CSV file (e.g., eval_out_train.csv)")
    ap.add_argument("--csv-dir", default=None, help="Directory containing diagnostic CSVs")
    ap.add_argument("--pattern", default="*out*train.csv", help="Glob pattern for --csv-dir")
    ap.add_argument("--out-dir", default="gaussian_splat/output_hof_splats", help="Output directory for PNGs")
    ap.add_argument("--flows-out-dir", default="gaussian_splat/hof_flow_vectors", help="Save flow vectors NPZs here")
    ap.add_argument("--height", type=int, default=None, help="Output height in pixels")
    ap.add_argument("--width", type=int, default=None, help="Output width in pixels")
    ap.add_argument("--samples-per-cell", type=int, default=12, help="Flow samples per cell (scaled by occupancy)")
    ap.add_argument("--max-samples", type=int, default=200, help="Max HOF samples to aggregate per CSV")
    ap.add_argument("--max-flows", type=int, default=2000000, help="Max flow vectors per CSV")
    ap.add_argument("--min-valid-count", type=int, default=1, help="Minimum valid_count to include sample")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite cached flow vectors")
    ap.add_argument("--no-endpoint", action="store_true", help="Disable endpoint footprint panel")
    ap.add_argument("--no-flowspace", action="store_true", help="Disable flow-space density panel")
    ap.add_argument("--no-cell-jitter", action="store_true", help="Disable within-cell jitter")
    ap.add_argument("--cell-jitter-frac", type=float, default=1.0, help="Jitter span as fraction of cell size (0.1-1.0)")
    ap.add_argument("--legend-inside", action="store_true", help="Draw direction legend inside the image (default: outside)")
    ap.add_argument(
        "--title-mode",
        choices=["short", "full", "none"],
        default="short",
        help="Title mode: short (panel titles + dataset suptitle), full (dataset|panel), none",
    )

    # Splat renderer options (passed through to visualize_flow_splats)
    ap.add_argument("--K", type=int, default=800)
    ap.add_argument("--subsample", type=int, default=2000000)
    ap.add_argument("--max_radius_px", type=int, default=64)
    ap.add_argument("--soft_edge", type=float, default=0.15)
    ap.add_argument("--support_sigma", type=float, default=3.0)
    ap.add_argument("--flow_range", type=float, default=None)
    ap.add_argument("--flow_bins", type=int, default=512)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--grid", type=int, default=None, help="Directional splat grid (defaults to HOF grid)")
    ap.add_argument("--k_dir", type=int, default=3)
    ap.add_argument("--min_bin", type=int, default=20)
    ap.add_argument("--dir_base_sigma", type=float, default=3.0)
    ap.add_argument("--dir_max_sigma", type=float, default=48.0)
    ap.add_argument("--dir_mode", choices=["grid", "cluster", "joint"], default="grid")
    ap.add_argument("--joint_xy_scale", type=float, default=1.0)
    ap.add_argument("--joint_flow_scale", type=float, default=1.5)
    args = ap.parse_args()

    csvs = _collect_csvs(args.csv, args.csv_dir, args.pattern)
    if not csvs:
        print("[ERROR] No CSVs found.")
        return

    out_dir = Path(args.out_dir)
    flows_dir = Path(args.flows_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flows_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        if "cache_path" not in df.columns:
            print(f"[WARN] {csv_path}: missing cache_path column, skipping")
            continue

        cache_paths = df["cache_path"].dropna().astype(str).tolist()
        if not cache_paths:
            print(f"[WARN] {csv_path}: no cache paths, writing placeholder")
            tag = csv_path.parent.name + "__" + csv_path.stem
            out_path = Path(args.out_dir) / f"{tag}_splat.png"
            _write_placeholder_image(out_path, tag, "No samples in this bin", dpi=args.dpi)
            continue

        manifest = _load_manifest_from_cache_path(Path(cache_paths[0]))
        hof_cfg = _parse_hof_config_from_manifest(manifest)
        if not hof_cfg:
            print(f"[WARN] {csv_path}: no hof_config found in manifest, writing placeholder")
            tag = csv_path.parent.name + "__" + csv_path.stem
            out_path = Path(args.out_dir) / f"{tag}_splat.png"
            _write_placeholder_image(out_path, tag, "Missing HOF config", dpi=args.dpi)
            continue

        grid_hw = tuple(hof_cfg.get("grid_hw", (32, 32)))
        angle_bins = int(hof_cfg.get("angle_bins", 8))
        mag_edges = np.asarray(hof_cfg.get("mag_edges", [0.0, 0.01, 0.03, 0.08, 0.25]), dtype=np.float32)

        H = args.height
        W = args.width
        if H is None or W is None:
            Hm, Wm = _infer_hw_from_samples(cache_paths)
            H = H or Hm
            W = W or Wm

        if args.grid is None:
            grid = int(grid_hw[0])
        else:
            grid = int(args.grid)

        tag = csv_path.parent.name + "__" + csv_path.stem
        flow_npz = flows_dir / f"{tag}.npz"

        if flow_npz.exists() and not args.overwrite:
            flows = vfs.load_any(str(flow_npz))
        else:
            flows = _flow_vectors_from_hof(
                cache_paths=cache_paths,
                grid_hw=grid_hw,
                angle_bins=angle_bins,
                mag_edges=mag_edges,
                out_hw=(H, W),
                samples_per_cell=args.samples_per_cell,
                max_samples=args.max_samples,
                max_flows=args.max_flows,
                seed=args.seed,
                min_valid_count=args.min_valid_count,
                cell_jitter=(not args.no_cell_jitter),
                cell_jitter_frac=args.cell_jitter_frac,
            )
            np.savez_compressed(flow_npz, flows=flows)

        out_path = out_dir / f"{tag}_splat.png"
        if flows.shape[0] == 0:
            _write_placeholder_image(out_path, tag, "No valid flows generated", dpi=args.dpi)
            continue
        make_hof_figure(
            flows=flows,
            dataset_name=tag,
            out_path=str(out_path),
            H=H,
            W=W,
            K=args.K,
            subsample=args.subsample,
            seed=args.seed,
            max_radius_px=args.max_radius_px,
            flow_bins=args.flow_bins,
            dpi=args.dpi,
            grid=grid,
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
            show_endpoint=(not args.no_endpoint),
            show_flowspace=(not args.no_flowspace),
            legend_outside=(not args.legend_inside),
            title_mode=args.title_mode,
        )


if __name__ == "__main__":
    main()
