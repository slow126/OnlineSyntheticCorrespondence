#!/usr/bin/env python3
"""
Export clean directional splat panels directly from joint support cache NPZs.

Input cache format:
  gaussian_splat/joint_flow_support_vectors/<run_label>/<pair>/<pair>__<bin>.npz
where <bin> is one of:
  train_in_eval, train_out_eval, eval_in_train, eval_out_train

These NPZ files already contain precomputed support-bin flows, so no NN search is run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib.colors as mcolors
import numpy as np
from PIL import Image

import visualize_flow_splats as vfs


VALID_BINS = ("train_in_eval", "train_out_eval", "eval_in_train", "eval_out_train")


def _save_rgb_image(rgb: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(arr).save(out_path)


def _make_colorwheel(size: int = 1024) -> np.ndarray:
    n = max(int(size), 128)
    ys, xs = np.mgrid[-1:1:complex(0, n), -1:1:complex(0, n)]
    r = np.sqrt(xs**2 + ys**2)
    ang = np.arctan2(-ys, xs)
    hue = (ang + np.pi) / (2.0 * np.pi)
    sat = np.clip(r, 0.0, 1.0)
    val = np.ones_like(hue, dtype=np.float32)
    hsv = np.stack([hue, sat, val], axis=-1).astype(np.float32)
    rgb = mcolors.hsv_to_rgb(hsv)
    rgb[r > 1.0] = 1.0
    return rgb


def _build_direction_panel(
    flows: np.ndarray,
    *,
    height: int | None,
    width: int | None,
    K: int,
    subsample: int,
    seed: int,
    max_radius_px: int,
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
) -> np.ndarray:
    flows = flows.astype(np.float32, copy=False)
    finite = np.isfinite(flows).all(axis=1)
    flows = flows[finite]
    if flows.shape[0] == 0:
        raise ValueError("No valid flow vectors")
    n_vec = int(flows.shape[0])

    flows = vfs.subsample_rows(flows, subsample, seed=seed)
    x = flows[:, 0]
    y = flows[:, 1]
    dx = flows[:, 2]
    dy = flows[:, 3]

    if height is None or width is None:
        H, W = vfs.infer_hw(flows[:, :2])
    else:
        H, W = int(height), int(width)

    if dir_mode == "grid":
        mus, covs, weights, colors = vfs.build_grid_bin_splats(
            x=x, y=y, dx=dx, dy=dy, H=H, W=W, grid=grid, k_dir=k_dir, min_bin=min_bin,
            base_sigma=dir_base_sigma, max_sigma=dir_max_sigma
        )
    elif dir_mode == "cluster":
        K_eff = max(1, min(int(K), n_vec))
        mus, covs, weights, colors = vfs.build_cluster_splats(
            x=x, y=y, dx=dx, dy=dy, K=K_eff, seed=seed,
            base_sigma=dir_base_sigma, max_sigma=dir_max_sigma
        )
    else:
        K_eff = max(1, min(int(K), n_vec))
        mus, covs, weights, colors = vfs.build_joint_cluster_splats(
            x=x, y=y, dx=dx, dy=dy, K=K_eff, seed=seed,
            base_sigma=dir_base_sigma, max_sigma=dir_max_sigma,
            xy_scale=joint_xy_scale, flow_scale=joint_flow_scale
        )

    rgb, _ = vfs.splat_gaussians_color(
        H=H, W=W, mus=mus, covs=covs, weights=weights, colors=colors,
        max_radius_px=max_radius_px, soft_edge=soft_edge, support_sigma=support_sigma
    )
    return rgb


def _empty_panel(height: int | None, width: int | None) -> np.ndarray:
    h = int(height) if height is not None else 512
    w = int(width) if width is not None else 512
    return np.zeros((h, w, 3), dtype=np.float32)


def _discover_pairs(run_dir: Path) -> List[str]:
    out: List[str] = []
    for p in sorted(run_dir.iterdir()):
        if p.is_dir():
            out.append(p.name)
    return out


def _parse_bins(vals: Iterable[str]) -> List[str]:
    bins = []
    for b in vals:
        if b not in VALID_BINS:
            raise ValueError(f"Invalid bin '{b}'. Expected one of: {', '.join(VALID_BINS)}")
        bins.append(b)
    return bins


def main() -> None:
    ap = argparse.ArgumentParser(description="Render directional splats from precomputed joint support NPZ caches")
    ap.add_argument("--cache-root", default="gaussian_splat/joint_flow_support_vectors", help="Support cache root")
    ap.add_argument("--run-label", default="space_joint__a1p0__t2e1p0__e2t1p5", help="Cache run label")
    ap.add_argument("--pairs", nargs="+", default=None, help="Pair names (default: all pairs under run label)")
    ap.add_argument("--bins", nargs="+", default=list(VALID_BINS), help="Subset of bins to export")
    ap.add_argument(
        "--out-dir",
        default="ECCV26___Beyond_Realism__Aligning_Flow_Statistics_for_Dense_Correspondence_Pre_training/figures/eccv26/section4_support_panels_final",
        help="Output directory",
    )

    # Render parameters
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--K", type=int, default=800)
    ap.add_argument("--subsample", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-radius-px", type=int, default=64)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--k-dir", type=int, default=3)
    ap.add_argument("--min-bin", type=int, default=20)
    ap.add_argument("--dir-base-sigma", type=float, default=3.0)
    ap.add_argument("--dir-max-sigma", type=float, default=48.0)
    ap.add_argument("--dir-mode", choices=["grid", "cluster", "joint"], default="joint")
    ap.add_argument("--joint-xy-scale", type=float, default=1.0)
    ap.add_argument("--joint-flow-scale", type=float, default=1.5)
    ap.add_argument("--soft-edge", type=float, default=0.15)
    ap.add_argument("--support-sigma", type=float, default=3.0)
    ap.add_argument("--wheel-size", type=int, default=1024)
    ap.add_argument("--no-colorwheel", action="store_true", help="Do not export standalone direction colorwheel")
    args = ap.parse_args()

    bins = _parse_bins(args.bins)
    run_dir = Path(args.cache_root) / args.run_label
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing cache run directory: {run_dir}")

    if args.pairs is None:
        pairs = _discover_pairs(run_dir)
    else:
        pairs = list(args.pairs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_done = 0
    n_missing = 0
    for pair in pairs:
        pair_dir = run_dir / pair
        if not pair_dir.exists():
            print(f"[WARN] missing pair dir: {pair_dir}")
            n_missing += 1
            continue
        for bin_name in bins:
            npz_name = f"{pair}__{bin_name}.npz"
            in_path = pair_dir / npz_name
            if not in_path.exists():
                print(f"[WARN] missing bin file: {in_path}")
                n_missing += 1
                continue
            z = np.load(in_path, allow_pickle=True)
            if "flows" not in z:
                print(f"[WARN] no 'flows' key in {in_path}")
                n_missing += 1
                continue
            flows = np.asarray(z["flows"], dtype=np.float32)
            if flows.ndim != 2 or flows.shape[1] != 4:
                print(f"[WARN] bad flow shape in {in_path}: {flows.shape}")
                n_missing += 1
                continue

            print(f"[PROCESSING] {pair} :: {bin_name} ({flows.shape[0]:,} vectors)")
            if flows.shape[0] == 0:
                print(f"[WARN] empty bin; writing black panel for {pair} :: {bin_name}")
                rgb = _empty_panel(args.height, args.width)
            else:
                rgb = _build_direction_panel(
                    flows,
                    height=args.height,
                    width=args.width,
                    K=args.K,
                    subsample=args.subsample,
                    seed=args.seed,
                    max_radius_px=args.max_radius_px,
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
                )
            out_path = out_dir / pair / f"{pair}__{bin_name}__directional_splat.png"
            _save_rgb_image(rgb, out_path)
            print(f"[SAVED] {out_path}")
            n_done += 1

    if not args.no_colorwheel:
        wheel_path = out_dir / "legend__direction_colorwheel.png"
        _save_rgb_image(_make_colorwheel(args.wheel_size), wheel_path)
        print(f"[SAVED] {wheel_path}")
    print(f"[DONE] exported={n_done}, missing={n_missing}")


if __name__ == "__main__":
    main()
