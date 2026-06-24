#!/usr/bin/env python3
"""
Export clean directional splat panels (no titles/axes) from cached flow vectors.

This script regenerates splats from cached flow files (e.g., /mnt/.../*_flow.npy),
then saves one directional panel per dataset plus one standalone direction colorwheel.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.colors as mcolors
import numpy as np
from PIL import Image

import visualize_flow_splats as vfs


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    flow_file: str
    out_file: str
    label: str
    group: str


DATASETS: List[DatasetSpec] = [
    DatasetSpec("synthetic_train", "synthetic_train_flow.npy", "train__sdf-fractal3d__directional_splat.png", "SDF-Fractal3D", "train"),
    DatasetSpec("sintel_train", "sintel_train_flow.npy", "train__sintel__directional_splat.png", "Sintel", "train"),
    DatasetSpec("pointodyssey_train", "pointodyssey_train_flow.npy", "train__pointodyssey__directional_splat.png", "PointOdyssey", "train"),
    DatasetSpec("flyingthings_train", "flyingthings_train_flow.npy", "train__flyingthings__directional_splat.png", "FlyingThings", "train"),
    DatasetSpec("imagenet2dwarp_train", "imagenet2dwarp_train_flow.npy", "train__imagenet2dwarp__directional_splat.png", "ImageNet2DWarp", "train"),
    DatasetSpec("spair_train", "spair_train_flow.npy", "train__spair__directional_splat.png", "SPair", "train"),
    DatasetSpec("flyingthings_test", "flyingthings_test_flow.npy", "benchmark__flyingthings_test__directional_splat.png", "FlyingThings-test", "benchmark"),
    DatasetSpec("kitti2012_val", "kitti2012_val_flow.npy", "benchmark__kitti2012__directional_splat.png", "KITTI-2012", "benchmark"),
    DatasetSpec("kitti2015_val", "kitti2015_val_flow.npy", "benchmark__kitti2015__directional_splat.png", "KITTI-2015", "benchmark"),
    DatasetSpec("middlebury_val", "middlebury_val_flow.npy", "benchmark__middlebury__directional_splat.png", "Middlebury", "benchmark"),
    DatasetSpec("pfpascal_test", "pfpascal_test_flow.npy", "benchmark__pfpascal__directional_splat.png", "PF-PASCAL", "benchmark"),
    DatasetSpec("pfwillow_test", "pfwillow_test_flow.npy", "benchmark__pfwillow__directional_splat.png", "PF-WILLOW", "benchmark"),
    DatasetSpec("pointodyssey_test", "pointodyssey_test_flow.npy", "benchmark__pointodyssey_test__directional_splat.png", "PointOdyssey-test", "benchmark"),
    DatasetSpec("spair_test", "spair_test_flow.npy", "benchmark__spair_test__directional_splat.png", "SPair-test", "benchmark"),
    DatasetSpec("tss_val", "tss_val_flow.npy", "benchmark__tss__directional_splat.png", "TSS", "benchmark"),
    # ACCV 2026 additions: MOVi-F + the motion-tuned intervention sources
    DatasetSpec("movi_f_train", "movi_f_train_flow.npy", "train__movi-f__directional_splat.png", "MOVi-F", "train"),
    DatasetSpec("trial19_train", "kitti2015_hq_trial19_train_flow.npy", "train__trial19__directional_splat.png", "trial19 (motion-tuned)", "train"),
    DatasetSpec("lowtex_matte_train", "kitti2015_lowtex_matte_train_flow.npy", "train__lowtex-matte__directional_splat.png", "lowtex-matte", "train"),
    DatasetSpec("kitti_recovered_train", "kitti_recovered_gso_hq_train_flow.npy", "train__kitti-recovered__directional_splat.png", "KITTI-recovered (motion-tuned)", "train"),
    DatasetSpec("flyingthings_recovered_train", "flyingthings_recovered_hq_train_flow.npy", "train__flyingthings-recovered__directional_splat.png", "FlyingThings-recovered (motion-tuned)", "train"),
]


def _save_rgb_image(rgb: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(arr).save(out_path)


def _autocrop_to_content(rgb: np.ndarray, *, thresh: float, pad_frac: float) -> np.ndarray:
    """Trim the uniform black border (canvas padding + flow-free edges).

    Motion that leaves the frame, or short central motion sitting inside the
    auto canvas padding, leaves a dead black margin. Cropping to the content
    bounding box makes each panel fill its tile like the dense panels do,
    rather than floating in a black box.
    """
    # rgb is float in [0, 1]; thresh is expressed on the 0-255 scale to match
    # the composite builder's crop_content().
    lum = rgb.max(axis=2) * 255.0
    mask = lum > float(thresh)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return rgb
    H, W = rgb.shape[:2]
    pad = int(round(min(H, W) * float(pad_frac)))
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(H, int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(W, int(xs.max()) + 1 + pad)
    return rgb[y0:y1, x0:x1]


def _pad_to_square(rgb: np.ndarray) -> np.ndarray:
    """Center the panel in a square black canvas so grid tiles stay aligned."""
    h, w = rgb.shape[:2]
    if h == w:
        return rgb
    s = max(h, w)
    out = np.zeros((s, s, rgb.shape[2]), dtype=rgb.dtype)
    y0 = (s - h) // 2
    x0 = (s - w) // 2
    out[y0:y0 + h, x0:x0 + w] = rgb
    return out


def _apply_edge_fade(rgb: np.ndarray, fade_frac: float) -> np.ndarray:
    if fade_frac <= 0:
        return rgb
    h, w = rgb.shape[:2]
    fade_px = int(round(min(h, w) * float(fade_frac)))
    if fade_px <= 1:
        return rgb

    yy, xx = np.mgrid[0:h, 0:w]
    d_left = xx
    d_right = (w - 1) - xx
    d_top = yy
    d_bottom = (h - 1) - yy
    d_edge = np.minimum(np.minimum(d_left, d_right), np.minimum(d_top, d_bottom)).astype(np.float32)
    t = np.clip(d_edge / float(fade_px), 0.0, 1.0)
    # Smoothstep fade-in from border toward interior.
    mask = t * t * (3.0 - 2.0 * t)
    return (rgb * mask[..., None]).astype(np.float32)


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _circular_mean(theta: np.ndarray) -> float:
    s = float(np.mean(np.sin(theta)))
    c = float(np.mean(np.cos(theta)))
    return float(np.arctan2(s, c))


def _build_fan_panel(
    *,
    x: np.ndarray,
    y: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    H: int,
    W: int,
    K: int,
    seed: int,
    fan_min_count: int,
    fan_mag_q_low: float,
    fan_mag_q_high: float,
    fan_angle_q: float,
    fan_radius_scale: float,
    fan_max_radius: float,
) -> np.ndarray:
    labels_xy, centers_xy = vfs.fit_endpoint_clusters(np.stack([x, y], axis=1), K=K, seed=seed)

    dens = np.zeros((H, W), dtype=np.float32)
    accum = np.zeros((H, W, 3), dtype=np.float32)

    for k in range(int(K)):
        idx = np.nonzero(labels_xy == k)[0]
        nk = int(idx.size)
        if nk < max(1, int(fan_min_count)):
            continue

        dxi = dx[idx]
        dyi = dy[idx]
        mags = np.sqrt(dxi * dxi + dyi * dyi).astype(np.float32)
        if mags.size == 0:
            continue

        # Robust directional center and spread (circular).
        theta = np.arctan2(dyi, dxi).astype(np.float32)
        theta_mu = _circular_mean(theta)
        dtheta = np.abs(_wrap_angle(theta - theta_mu))

        q_low = float(np.clip(fan_mag_q_low, 0.0, 1.0))
        q_high = float(np.clip(fan_mag_q_high, q_low + 1e-3, 1.0))
        r0 = float(np.quantile(mags, q_low))
        r1 = float(np.quantile(mags, q_high))
        r0 = max(0.0, r0 * float(fan_radius_scale))
        r1 = max(r0 + 1.0, r1 * float(fan_radius_scale))
        if fan_max_radius > 0:
            r1 = min(r1, float(fan_max_radius))
            r0 = min(r0, max(0.0, r1 - 1.0))

        ang_q = float(np.clip(fan_angle_q, 0.5, 0.99))
        ang_half = float(np.quantile(dtheta, ang_q))
        ang_half = max(np.deg2rad(5.0), min(ang_half, np.deg2rad(80.0)))

        r_mid = 0.5 * (r0 + r1)
        r_sigma = max(1.0, 0.35 * (r1 - r0))
        a_sigma = max(np.deg2rad(2.0), 0.45 * ang_half)

        color = vfs.flow_direction_colors(
            np.array([np.mean(dxi)], dtype=np.float32),
            np.array([np.mean(dyi)], dtype=np.float32),
            np.array([np.mean(mags)], dtype=np.float32),
            np.array([float(nk)], dtype=np.float32),
        )[0]
        weight = float(np.log1p(nk))

        cx, cy = float(centers_xy[k, 0]), float(centers_xy[k, 1])
        r_lim = r1 + (3.0 * r_sigma)
        rx = int(np.ceil(r_lim))
        ry = int(np.ceil(r_lim))
        x0 = max(0, int(np.floor(cx)) - rx)
        x1 = min(W, int(np.floor(cx)) + rx + 1)
        y0 = max(0, int(np.floor(cy)) - ry)
        y1 = min(H, int(np.floor(cy)) + ry + 1)
        if x1 <= x0 or y1 <= y0:
            continue

        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)
        dX = X - cx
        dY = Y - cy
        rr = np.sqrt(dX * dX + dY * dY).astype(np.float32)
        aa = np.arctan2(dY, dX).astype(np.float32)
        da = _wrap_angle(aa - theta_mu)

        radial = np.exp(-0.5 * ((rr - r_mid) / r_sigma) ** 2).astype(np.float32)
        angular = np.exp(-0.5 * (da / a_sigma) ** 2).astype(np.float32)

        # Hard support bounds keep each glyph as a directional fan rather than full disk.
        r_gate = (rr >= max(0.0, r0 - 2.0 * r_sigma)) & (rr <= (r1 + 2.0 * r_sigma))
        a_gate = np.abs(da) <= max(ang_half, 2.2 * a_sigma)
        patch = radial * angular
        patch *= (r_gate & a_gate).astype(np.float32)

        patch_w = weight * patch
        dens[y0:y1, x0:x1] += patch_w
        accum[y0:y1, x0:x1] += patch_w[..., None] * color

    dens_tm = vfs.tone_map(dens)
    rgb = accum / (dens[..., None] + 1e-6)
    rgb = np.clip(rgb * dens_tm[..., None], 0.0, 1.0).astype(np.float32)
    return rgb


def _build_direction_panel(
    flows: np.ndarray,
    *,
    height: int | None,
    width: int | None,
    canvas_pad_px: int,
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
    edge_fade_frac: float,
    fan_min_count: int,
    fan_mag_q_low: float,
    fan_mag_q_high: float,
    fan_angle_q: float,
    fan_radius_scale: float,
    fan_max_radius: float,
    autocrop: bool,
    autocrop_thresh: float,
    autocrop_pad_frac: float,
    square: bool,
) -> np.ndarray:
    if flows.shape[0] == 0:
        raise ValueError("No flow vectors found")

    flows = vfs.subsample_rows(flows, subsample, seed=seed)
    x = flows[:, 0]
    y = flows[:, 1]
    dx = flows[:, 2]
    dy = flows[:, 3]

    if height is None or width is None:
        H, W = vfs.infer_hw(flows[:, :2])
    else:
        H, W = int(height), int(width)

    # Add canvas padding to avoid clipping elongated splats at image borders.
    if canvas_pad_px < 0:
        # Auto padding tracks maximum Gaussian support radius.
        pad = int(np.ceil(float(support_sigma) * float(dir_max_sigma))) + 4
    else:
        pad = int(canvas_pad_px)
    pad = max(0, pad)
    if pad > 0:
        x = x + float(pad)
        y = y + float(pad)
        H = H + (2 * pad)
        W = W + (2 * pad)

    if dir_mode == "grid":
        mus, covs, weights, colors = vfs.build_grid_bin_splats(
            x=x, y=y, dx=dx, dy=dy, H=H, W=W, grid=grid, k_dir=k_dir, min_bin=min_bin,
            base_sigma=dir_base_sigma, max_sigma=dir_max_sigma
        )
    elif dir_mode == "cluster":
        mus, covs, weights, colors = vfs.build_cluster_splats(
            x=x, y=y, dx=dx, dy=dy, K=K, seed=seed,
            base_sigma=dir_base_sigma, max_sigma=dir_max_sigma
        )
    elif dir_mode == "joint":
        mus, covs, weights, colors = vfs.build_joint_cluster_splats(
            x=x, y=y, dx=dx, dy=dy, K=K, seed=seed,
            base_sigma=dir_base_sigma, max_sigma=dir_max_sigma,
            xy_scale=joint_xy_scale, flow_scale=joint_flow_scale
        )
    elif dir_mode == "fan":
        rgb = _build_fan_panel(
            x=x,
            y=y,
            dx=dx,
            dy=dy,
            H=H,
            W=W,
            K=K,
            seed=seed,
            fan_min_count=fan_min_count,
            fan_mag_q_low=fan_mag_q_low,
            fan_mag_q_high=fan_mag_q_high,
            fan_angle_q=fan_angle_q,
            fan_radius_scale=fan_radius_scale,
            fan_max_radius=fan_max_radius,
        )
    else:
        raise ValueError(f"Unsupported dir_mode: {dir_mode}")

    if dir_mode in {"grid", "cluster", "joint"}:
        rgb, _ = vfs.splat_gaussians_color(
            H=H,
            W=W,
            mus=mus,
            covs=covs,
            weights=weights,
            colors=colors,
            max_radius_px=max_radius_px,
            soft_edge=soft_edge,
            support_sigma=support_sigma,
        )

    rgb = _apply_edge_fade(rgb, edge_fade_frac)
    if autocrop:
        rgb = _autocrop_to_content(rgb, thresh=autocrop_thresh, pad_frac=autocrop_pad_frac)
        if square:
            rgb = _pad_to_square(rgb)
    return rgb


def _make_colorwheel(size: int = 1024) -> np.ndarray:
    n = max(int(size), 128)
    ys, xs = np.mgrid[-1:1:complex(0, n), -1:1:complex(0, n)]
    r = np.sqrt(xs**2 + ys**2)
    # match the splat encoding flow_direction_colors: angle = atan2(dy, dx) in
    # IMAGE coordinates (y points down). ys already increases downward, so use
    # it directly (no negation) — otherwise the wheel is vertically flipped.
    ang = np.arctan2(ys, xs)
    hue = (ang + np.pi) / (2.0 * np.pi)
    sat = np.clip(r, 0.0, 1.0)
    val = np.ones_like(hue, dtype=np.float32)
    hsv = np.stack([hue, sat, val], axis=-1).astype(np.float32)
    rgb = mcolors.hsv_to_rgb(hsv)
    rgb[r > 1.0] = 1.0
    return rgb


def main() -> None:
    ap = argparse.ArgumentParser(description="Export clean directional splat panels from cached flow vectors")
    ap.add_argument("--flow-dir", default="/mnt/nvme_1tb_b/coverage_vectors", help="Directory with *_flow.npy files")
    ap.add_argument("--out-dir", default="gaussian_splat/output_final_direction_panels", help="Output directory")
    ap.add_argument("--datasets", nargs="+", default=[d.key for d in DATASETS], help="Dataset keys to export")

    # Renderer options, matched to visualize_flow_splats defaults / run_splat_vis.sh
    ap.add_argument("--height", type=int, default=None, help="Force image height; otherwise infer from flows")
    ap.add_argument("--width", type=int, default=None, help="Force image width; otherwise infer from flows")
    ap.add_argument(
        "--canvas-pad-px",
        type=int,
        default=-1,
        help="Canvas padding in pixels before rendering (-1 = auto from support_sigma*dir_max_sigma)",
    )
    ap.add_argument("--K", type=int, default=800, help="Cluster count for cluster/joint dir modes")
    ap.add_argument("--subsample", type=int, default=2_000_000, help="Max vectors per dataset")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--max-radius-px", type=int, default=64, help="Max splat radius")
    ap.add_argument("--grid", type=int, default=32, help="Directional grid size")
    ap.add_argument("--k-dir", type=int, default=3, help="Direction clusters per grid bin")
    ap.add_argument("--min-bin", type=int, default=20, help="Minimum vectors per grid bin")
    ap.add_argument("--dir-base-sigma", type=float, default=3.0, help="Directional base sigma")
    ap.add_argument("--dir-max-sigma", type=float, default=48.0, help="Directional max sigma")
    ap.add_argument("--dir-mode", choices=["grid", "cluster", "joint", "fan"], default="cluster", help="Directional mode")
    ap.add_argument("--joint-xy-scale", type=float, default=1.0, help="Joint mode XY scale")
    ap.add_argument("--joint-flow-scale", type=float, default=1.5, help="Joint mode flow scale")
    ap.add_argument("--soft-edge", type=float, default=0.15, help="Edge feathering")
    ap.add_argument("--support-sigma", type=float, default=3.0, help="Gaussian support in sigma")
    ap.add_argument("--wheel-size", type=int, default=1024, help="Standalone colorwheel image size (px)")
    ap.add_argument("--fan-min-count", type=int, default=48, help="Minimum vectors per spatial cluster for fan mode")
    ap.add_argument("--fan-mag-q-low", type=float, default=0.15, help="Low magnitude quantile for fan radial band")
    ap.add_argument("--fan-mag-q-high", type=float, default=0.90, help="High magnitude quantile for fan radial band")
    ap.add_argument("--fan-angle-q", type=float, default=0.80, help="Angular spread quantile for fan mode")
    ap.add_argument("--fan-radius-scale", type=float, default=1.0, help="Scale factor on fan radial distances")
    ap.add_argument("--fan-max-radius", type=float, default=96.0, help="Maximum fan radius in pixels (<=0 disables)")
    ap.add_argument(
        "--edge-fade-frac",
        type=float,
        default=0.0,
        help="Fraction of min image dimension used to softly fade panel edges (e.g., 0.05)",
    )
    ap.add_argument(
        "--no-autocrop",
        dest="autocrop",
        action="store_false",
        help="Disable trimming the uniform black border around panel content",
    )
    ap.set_defaults(autocrop=True)
    ap.add_argument("--autocrop-thresh", type=float, default=12.0, help="0-255 luminance below which a pixel is treated as empty border")
    ap.add_argument("--autocrop-pad-frac", type=float, default=0.02, help="Fraction of min dim kept as breathing room around cropped content")
    ap.add_argument(
        "--no-square",
        dest="square",
        action="store_false",
        help="Do not pad cropped panels back to square (square keeps grid tiles aligned)",
    )
    ap.set_defaults(square=True)
    args = ap.parse_args()

    flow_dir = Path(args.flow_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec_by_key: Dict[str, DatasetSpec] = {d.key: d for d in DATASETS}
    selected: List[DatasetSpec] = []
    for key in args.datasets:
        if key not in spec_by_key:
            raise ValueError(f"Unknown dataset key: {key}")
        selected.append(spec_by_key[key])

    manifest = {
        "flow_dir": str(flow_dir),
        "out_dir": str(out_dir),
        "datasets": [],
        "render_args": {
            "height": args.height,
            "width": args.width,
            "canvas_pad_px": args.canvas_pad_px,
            "K": args.K,
            "subsample": args.subsample,
            "seed": args.seed,
            "max_radius_px": args.max_radius_px,
            "grid": args.grid,
            "k_dir": args.k_dir,
            "min_bin": args.min_bin,
            "dir_base_sigma": args.dir_base_sigma,
            "dir_max_sigma": args.dir_max_sigma,
            "dir_mode": args.dir_mode,
            "joint_xy_scale": args.joint_xy_scale,
            "joint_flow_scale": args.joint_flow_scale,
            "soft_edge": args.soft_edge,
            "support_sigma": args.support_sigma,
            "wheel_size": args.wheel_size,
            "edge_fade_frac": args.edge_fade_frac,
            "fan_min_count": args.fan_min_count,
            "fan_mag_q_low": args.fan_mag_q_low,
            "fan_mag_q_high": args.fan_mag_q_high,
            "fan_angle_q": args.fan_angle_q,
            "fan_radius_scale": args.fan_radius_scale,
            "fan_max_radius": args.fan_max_radius,
            "autocrop": args.autocrop,
            "autocrop_thresh": args.autocrop_thresh,
            "autocrop_pad_frac": args.autocrop_pad_frac,
            "square": args.square,
        },
    }

    for spec in selected:
        in_path = flow_dir / spec.flow_file
        if not in_path.exists():
            print(f"[WARN] missing input file: {in_path}")
            continue

        print(f"[PROCESSING] {spec.key} from {in_path}")
        flows = vfs.load_any(str(in_path))
        rgb = _build_direction_panel(
            flows,
            height=args.height,
            width=args.width,
            K=args.K,
            subsample=args.subsample,
            seed=args.seed,
            canvas_pad_px=args.canvas_pad_px,
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
            edge_fade_frac=args.edge_fade_frac,
            fan_min_count=args.fan_min_count,
            fan_mag_q_low=args.fan_mag_q_low,
            fan_mag_q_high=args.fan_mag_q_high,
            fan_angle_q=args.fan_angle_q,
            fan_radius_scale=args.fan_radius_scale,
            fan_max_radius=args.fan_max_radius,
            autocrop=args.autocrop,
            autocrop_thresh=args.autocrop_thresh,
            autocrop_pad_frac=args.autocrop_pad_frac,
            square=args.square,
        )
        out_path = out_dir / spec.out_file
        _save_rgb_image(rgb, out_path)
        print(f"[SAVED] {out_path}")

        manifest["datasets"].append(
            {
                "key": spec.key,
                "label": spec.label,
                "group": spec.group,
                "input_flow_file": str(in_path),
                "output_png": str(out_path),
                "num_vectors_loaded": int(flows.shape[0]),
                "output_shape_hw": [int(rgb.shape[0]), int(rgb.shape[1])],
            }
        )

    wheel = _make_colorwheel(size=args.wheel_size)
    wheel_path = out_dir / "legend__direction_colorwheel.png"
    _save_rgb_image(wheel, wheel_path)
    print(f"[SAVED] {wheel_path}")
    manifest["direction_colorwheel_png"] = str(wheel_path)

    manifest_path = out_dir / "_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[SAVED] {manifest_path}")

    print("[DONE] Final directional panels exported.")


if __name__ == "__main__":
    main()
