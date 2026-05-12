#!/usr/bin/env python3
"""
Create a 2x2 montage of HOF splat visualizations for a train/eval pair.

Expected image filenames:
  {pair_name}__train_out_eval_splat.png
  {pair_name}__train_in_eval_splat.png
  {pair_name}__eval_in_train_splat.png
  {pair_name}__eval_out_train_splat.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_image(path: Path) -> np.ndarray:
    if not path.exists():
        return None
    return plt.imread(str(path))


def _placeholder(text: str, shape: Tuple[int, int] = (512, 512)) -> np.ndarray:
    h, w = shape
    img = np.ones((h, w, 3), dtype=np.float32)
    # simple gray box with text
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axis_off()
    ax.imshow(img * 0.95)
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12)
    fig.canvas.draw()
    # Agg canvas may expose buffer_rgba / tostring_argb depending on backend
    if hasattr(fig.canvas, "buffer_rgba"):
        buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        buf = buf[..., :3]
    else:
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        buf = buf[..., 1:]  # drop alpha channel
    plt.close(fig)
    return buf.astype(np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser(description="Make a 2x2 montage for HOF splat outputs")
    ap.add_argument("--pair-dir", required=True, help="Path to analysis/hof_diag/<pair>")
    ap.add_argument(
        "--image-dir",
        default=None,
        help="Directory containing splat images (defaults to gaussian_splat/output_hof_splats/<pair>)",
    )
    ap.add_argument("--out", default=None, help="Output montage path (png)")
    ap.add_argument("--dpi", type=int, default=200, help="Output DPI")
    ap.add_argument("--fig-height", type=float, default=7.0, help="Figure height in inches")
    ap.add_argument("--wspace", type=float, default=0.01, help="Horizontal gap between panels")
    ap.add_argument("--hspace", type=float, default=0.08, help="Vertical gap between panels")
    ap.add_argument("--title-size", type=float, default=11.0, help="Panel title font size")
    ap.add_argument("--title-pad", type=float, default=2.0, help="Panel title padding")
    ap.add_argument("--pad-inches", type=float, default=0.02, help="Savefig padding in inches")
    args = ap.parse_args()

    pair_dir = Path(args.pair_dir)
    pair_name = pair_dir.name

    if args.image_dir:
        image_dir = Path(args.image_dir)
    else:
        image_dir = Path("gaussian_splat/output_hof_splats") / pair_name

    targets: Dict[str, str] = {
        "Train outside eval": f"{pair_name}__train_out_eval_splat.png",
        "Train in eval": f"{pair_name}__train_in_eval_splat.png",
        "Eval in train": f"{pair_name}__eval_in_train_splat.png",
        "Eval outside train": f"{pair_name}__eval_out_train_splat.png",
    }

    imgs = {}
    for title, fname in targets.items():
        path = image_dir / fname
        img = _load_image(path)
        imgs[title] = img

    # Determine placeholder size
    sample_img = next((v for v in imgs.values() if v is not None), None)
    if sample_img is None:
        print(f"[ERROR] No images found in {image_dir}")
        return
    ph_shape = (sample_img.shape[0], sample_img.shape[1])
    panel_aspect = float(sample_img.shape[1]) / max(float(sample_img.shape[0]), 1.0)

    # Match figure aspect to panel image aspect to avoid letterboxing whitespace.
    fig_h = float(max(args.fig_height, 2.0))
    fig_w = fig_h * panel_aspect
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
    gs = fig.add_gridspec(2, 2)
    fig.subplots_adjust(
        left=0.005,
        right=0.995,
        bottom=0.005,
        top=0.965,
        wspace=max(args.wspace, 0.0),
        hspace=max(args.hspace, 0.0),
    )

    titles = list(targets.keys())
    for i, title in enumerate(titles):
        r = i // 2
        c = i % 2
        ax = fig.add_subplot(gs[r, c])
        img = imgs[title]
        if img is None:
            img = _placeholder(f"Missing:\n{title}", shape=ph_shape)
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(title, fontsize=args.title_size, pad=args.title_pad)

    out_path = Path(args.out) if args.out else image_dir / f"{pair_name}__montage.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=max(args.pad_inches, 0.0))
    plt.close(fig)
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
