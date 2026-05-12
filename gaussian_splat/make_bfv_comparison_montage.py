#!/usr/bin/env python3
"""
Build a clean BFV comparison montage from existing per-dataset splat figures.

This script extracts only the middle directional splat panel from images in:
  gaussian_splat/output_splats/<dataset>_flow_splat.png
and arranges them into a paper-ready montage.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


DEFAULT_DATASETS = ["synthetic_train", "kitti2015_val", "spair_train", "sintel_train"]

PRETTY_LABELS: Dict[str, str] = {
    "synthetic_train": "SDF-Fractal3D",
    "kitti2015_val": "KITTI-15",
    "kitti2012_val": "KITTI-12",
    "spair_train": "SPair",
    "spair_test": "SPair-test",
    "sintel_train": "Sintel",
    "pointodyssey_train": "PointOdyssey",
    "pointodyssey_test": "PointOdyssey-test",
    "flyingthings_train": "FlyingThings",
    "flyingthings_test": "FlyingThings-test",
    "imagenet2dwarp_train": "ImageNet 2D Warps",
    "pfpascal_test": "PF-PASCAL",
    "pfwillow_test": "PF-WILLOW",
    "middlebury_val": "Middlebury",
    "tss_val": "TSS",
}

ALIASES: Dict[str, str] = {
    "sdf-fractal3d": "synthetic_train",
    "sdf_fractal3d": "synthetic_train",
    "sdf fractal 3d": "synthetic_train",
    "synthetic": "synthetic_train",
    "synthetic train": "synthetic_train",
    "flyingthings train": "flyingthings_train",
    "flyingthings test": "flyingthings_test",
    "kitti-2015": "kitti2015_val",
    "kitti2015": "kitti2015_val",
    "kitti 2015": "kitti2015_val",
    "kitti-2012": "kitti2012_val",
    "kitti2012": "kitti2012_val",
    "kitti 2012": "kitti2012_val",
    "spair": "spair_train",
    "spair train": "spair_train",
    "spair test": "spair_test",
    "sintel": "sintel_train",
    "sintel train": "sintel_train",
    "pointodyssey": "pointodyssey_train",
    "point odyssey": "pointodyssey_train",
    "pointodyssey train": "pointodyssey_train",
    "pointodyssey test": "pointodyssey_test",
    "flyingthings": "flyingthings_train",
    "flying things": "flyingthings_train",
    "imagenet2dwarp": "imagenet2dwarp_train",
    "imagenet 2d warp": "imagenet2dwarp_train",
    "imagenet 2d warps": "imagenet2dwarp_train",
    "imagenet-2d-warp": "imagenet2dwarp_train",
    "imagenet-2d-warps": "imagenet2dwarp_train",
    "middlebury": "middlebury_val",
    "pfpascal": "pfpascal_test",
    "pfwillow": "pfwillow_test",
    "tss": "tss_val",
}


def _resolve_dataset_name(name: str) -> str:
    raw = name.strip().lower()
    normalized = raw.replace("_", " ").replace("-", " ")
    if raw in ALIASES:
        return ALIASES[raw]
    if normalized in ALIASES:
        return ALIASES[normalized]
    return raw


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _contiguous_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    in_seg = False
    start = 0
    for i, on in enumerate(mask):
        if on and not in_seg:
            start = i
            in_seg = True
        elif in_seg and not on:
            segments.append((start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((start, len(mask) - 1))
    return segments


def _extract_direction_panel(img: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape

    nonwhite_row = ((img < 0.98).any(axis=2)).mean(axis=1)
    row_idx = np.where(nonwhite_row > 0.08)[0]
    if row_idx.size == 0:
        y0, y1 = int(0.08 * h), int(0.92 * h)
    else:
        y0, y1 = int(row_idx[0]), int(row_idx[-1])

    # Detect coarse panel blocks in x by non-white coverage.
    col_nonwhite = ((img[y0 : y1 + 1] < 0.98).any(axis=2)).mean(axis=0)
    raw_segments = _contiguous_segments(col_nonwhite > 0.12)
    # Keep only broad segments to ignore thin legends/ticks.
    broad_segments = [s for s in raw_segments if (s[1] - s[0] + 1) >= int(0.08 * w)]
    broad_segments = sorted(broad_segments, key=lambda s: s[0])

    # In typical outputs: 3 broad panels -> [endpoint, direction, flowspace].
    # For 2-panel outputs (no endpoint): [direction, flowspace].
    if len(broad_segments) >= 3:
        x0, x1 = broad_segments[1]
    elif len(broad_segments) == 2:
        x0, x1 = broad_segments[0]
    elif len(broad_segments) == 1:
        x0, x1 = broad_segments[0]
    else:
        # Fallback: central region in standard multi-panel layout.
        x0 = int(round(0.26 * w))
        x1 = int(round(0.58 * w))

    crop = img[y0 : y1 + 1, x0 : x1 + 1]

    # Trim residual white margins around the extracted panel.
    nonwhite = (crop < 0.98).any(axis=2)
    valid_rows = np.where(nonwhite.mean(axis=1) > 0.01)[0]
    valid_cols = np.where(nonwhite.mean(axis=0) > 0.01)[0]
    if valid_rows.size > 0 and valid_cols.size > 0:
        crop = crop[valid_rows[0] : valid_rows[-1] + 1, valid_cols[0] : valid_cols[-1] + 1]

    return crop


def _center_crop_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0 : y0 + side, x0 : x0 + side]


def _build_source_path(image_dir: Path, dataset: str) -> Path:
    if dataset.endswith(".png"):
        return image_dir / dataset
    if dataset.endswith("_flow_splat"):
        return image_dir / f"{dataset}.png"
    return image_dir / f"{dataset}_flow_splat.png"


def _panel_label(dataset: str) -> str:
    if dataset in PRETTY_LABELS:
        return PRETTY_LABELS[dataset]
    return dataset.replace("_", " ")


def _parse_layout(layout: str, n_panels: int) -> Tuple[int, int]:
    layout = layout.lower()
    if layout == "auto":
        ncols = int(math.ceil(math.sqrt(n_panels)))
        nrows = int(math.ceil(n_panels / ncols))
        return nrows, ncols
    if "x" not in layout:
        raise ValueError("--layout must be 'auto' or in the form 'RxC' (e.g., 2x3)")
    left, right = layout.split("x", 1)
    try:
        nrows = int(left)
        ncols = int(right)
    except ValueError as exc:
        raise ValueError("--layout must be 'auto' or in the form 'RxC' (e.g., 2x3)") from exc
    if nrows <= 0 or ncols <= 0:
        raise ValueError("layout rows/cols must be positive")
    if nrows * ncols < n_panels:
        raise ValueError(f"layout {layout} has only {nrows*ncols} cells for {n_panels} panels")
    return nrows, ncols


def _draw_direction_colorwheel(ax: plt.Axes, title: str = "Direction") -> None:
    n = 240
    ys, xs = np.mgrid[-1:1:complex(0, n), -1:1:complex(0, n)]
    r = np.sqrt(xs**2 + ys**2)
    ang = np.arctan2(-ys, xs)
    hue = (ang + np.pi) / (2.0 * np.pi)
    sat = np.clip(r, 0.0, 1.0)
    val = np.ones_like(hue, dtype=np.float32)
    hsv = np.stack([hue, sat, val], axis=-1).astype(np.float32)
    rgb = mcolors.hsv_to_rgb(hsv)
    rgb[r > 1.0] = 1.0

    ax.imshow(rgb, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8, pad=2.0)


def _add_direction_colorwheel(fig: plt.Figure, size: float = 0.10, title: str = "Direction") -> None:
    size = float(max(0.04, min(size, 0.25)))
    margin = 0.012
    ax = fig.add_axes([1.0 - size - margin, 1.0 - size - margin, size, size])
    _draw_direction_colorwheel(ax, title=title)


def _add_group_dividers(
    fig: plt.Figure,
    row_axes: Dict[int, plt.Axes],
    nrows: int,
    ncols: int,
    n_panels: int,
    group_sizes: Sequence[int],
    group_labels: Sequence[str] | None,
    divider_color: str,
    divider_lw: float,
    group_label_size: float,
) -> None:
    if len(group_sizes) == 0:
        return

    if sum(group_sizes) != n_panels:
        raise ValueError("--group-sizes must sum to number of panels")

    if group_labels is not None and len(group_labels) != len(group_sizes):
        raise ValueError("--group-labels length must match --group-sizes length")

    # Require intermediate group boundaries to end at a row boundary for clean dividers.
    cumulative = 0
    row_ends: List[int] = []
    for gi, sz in enumerate(group_sizes):
        cumulative += sz
        is_last = gi == (len(group_sizes) - 1)
        if (not is_last) and (cumulative % ncols != 0):
            raise ValueError("Each group in --group-sizes must end at a row boundary for the selected --layout")
        if cumulative % ncols == 0:
            row_ends.append(cumulative // ncols - 1)

    fig.canvas.draw()
    used_rows = sorted(row_axes.keys())
    if not used_rows:
        return

    x0 = min(row_axes[r].get_position().x0 for r in used_rows)
    x1 = max(row_axes[r].get_position().x1 for r in used_rows)

    # Divider between consecutive groups.
    for i in range(len(row_ends) - 1):
        r_top_group_end = row_ends[i]
        r_bottom_group_start = r_top_group_end + 1
        if r_top_group_end not in row_axes or r_bottom_group_start not in row_axes:
            continue
        y_top_gap = row_axes[r_top_group_end].get_position().y0
        y_bottom_gap = row_axes[r_bottom_group_start].get_position().y1
        y_div = 0.5 * (y_top_gap + y_bottom_gap)
        fig.add_artist(
            Line2D([x0, x1], [y_div, y_div], transform=fig.transFigure, color=divider_color, linewidth=divider_lw)
        )

    if group_labels is None:
        return

    row_starts = [0] + [r + 1 for r in row_ends[:-1]]
    for label, r0 in zip(group_labels, row_starts):
        if r0 not in row_axes:
            continue
        pos = row_axes[r0].get_position()
        fig.text(x0, min(0.995, pos.y1 + 0.01), label, ha="left", va="bottom", fontsize=group_label_size)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Create a clean BFV multi-dataset montage from existing splat images")
    ap.add_argument(
        "--image-dir",
        default=str(script_dir / "output_splats"),
        help="Directory containing <dataset>_flow_splat.png files",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset ids (or aliases like 'sdf fractal 3d', flyingthings, 'imagenet 2d warps')",
    )
    ap.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional panel labels (must match number of datasets)",
    )
    ap.add_argument("--layout", default="auto", help="Panel layout: 'auto' or 'RxC' (e.g., 1x6, 2x3, 2x2)")
    ap.add_argument("--out", default=str(script_dir / "output_splats" / "section3_bfv_comparison_4panel.png"))
    ap.add_argument("--dpi", type=int, default=300, help="Output DPI")
    ap.add_argument("--cell-size", type=float, default=2.45, help="Per-panel size in inches")
    ap.add_argument("--label-mode", choices=["top", "bottom", "none"], default="bottom", help="Panel label position")
    ap.add_argument("--label-size", type=float, default=10.0, help="Panel label size")
    ap.add_argument("--label-pad", type=float, default=0.06, help="Panel label padding in axis units")
    ap.add_argument("--show-colorwheel", action="store_true", help="Add direction colorwheel legend")
    ap.add_argument(
        "--wheel-placement",
        choices=["overlay", "bottom-right"],
        default="overlay",
        help="Place colorwheel as overlay or in the bottom-right empty grid cell",
    )
    ap.add_argument("--wheel-size", type=float, default=0.10, help="Colorwheel size as figure fraction")
    ap.add_argument("--wheel-title", default="Direction", help="Colorwheel title")
    ap.add_argument("--group-sizes", nargs="+", type=int, default=None, help="Group panel counts (e.g., 6 9)")
    ap.add_argument("--group-labels", nargs="+", default=None, help="Optional group labels")
    ap.add_argument("--divider-color", default="black", help="Group divider line color")
    ap.add_argument("--divider-lw", type=float, default=1.0, help="Group divider line width")
    ap.add_argument("--group-label-size", type=float, default=11.0, help="Group label text size")
    ap.add_argument("--pad-inches", type=float, default=0.02, help="Savefig padding")
    args = ap.parse_args()

    resolved = [_resolve_dataset_name(d) for d in args.datasets]
    if args.labels is not None and len(args.labels) != len(resolved):
        raise ValueError("--labels must have the same length as --datasets")

    image_dir = Path(args.image_dir)
    source_paths = [_build_source_path(image_dir, d) for d in resolved]
    missing = [p for p in source_paths if not p.exists()]
    if missing:
        missing_str = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing source image(s):\n{missing_str}")

    panels: List[np.ndarray] = []
    for path in source_paths:
        img = _load_rgb(path)
        panel = _extract_direction_panel(img)
        panel = _center_crop_square(panel)
        panels.append(panel)

    labels: Sequence[str]
    if args.labels is not None:
        labels = args.labels
    else:
        labels = [_panel_label(d) for d in resolved]

    nrows, ncols = _parse_layout(args.layout, len(panels))
    fig_w = args.cell_size * ncols
    fig_h = args.cell_size * nrows

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    gs = fig.add_gridspec(nrows, ncols)
    row_axes: Dict[int, plt.Axes] = {}

    wheel_cell_index = None
    if args.show_colorwheel and args.wheel_placement == "bottom-right":
        total_cells = nrows * ncols
        if total_cells <= len(panels):
            raise ValueError("--wheel-placement bottom-right requires at least one empty grid cell")
        wheel_cell_index = total_cells - 1

    for i in range(nrows * ncols):
        r = i // ncols
        c = i % ncols
        ax = fig.add_subplot(gs[r, c])
        if r not in row_axes:
            row_axes[r] = ax
        if i < len(panels):
            panel = panels[i]
            label = labels[i]
            ax.imshow(panel)
            ax.set_axis_off()
            if args.label_mode == "top":
                ax.set_title(label, fontsize=args.label_size, pad=2.5)
            elif args.label_mode == "bottom":
                ax.text(
                    0.5,
                    -abs(args.label_pad),
                    label,
                    ha="center",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=args.label_size,
                    clip_on=False,
                )
        elif wheel_cell_index is not None and i == wheel_cell_index:
            _draw_direction_colorwheel(ax, title=args.wheel_title)
        else:
            ax.set_axis_off()

    if args.group_sizes is not None:
        _add_group_dividers(
            fig=fig,
            row_axes=row_axes,
            nrows=nrows,
            ncols=ncols,
            n_panels=len(panels),
            group_sizes=args.group_sizes,
            group_labels=args.group_labels,
            divider_color=args.divider_color,
            divider_lw=args.divider_lw,
            group_label_size=args.group_label_size,
        )

    if args.show_colorwheel and args.wheel_placement == "overlay":
        _add_direction_colorwheel(fig, size=args.wheel_size, title=args.wheel_title)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=max(args.pad_inches, 0.0))
    plt.close(fig)
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
