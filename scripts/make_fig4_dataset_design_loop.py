#!/usr/bin/env python3
"""
Generate Fig. 4: dataset design loop
(generate -> measure -> score -> iterate).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _box(ax, xy, w, h, title, body, fc, ec="#333333"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.03",
        linewidth=1.4,
        facecolor=fc,
        edgecolor=ec,
    )
    patch.set_clip_on(False)
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.67,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#222222",
    )
    ax.text(
        x + w / 2,
        y + h * 0.35,
        body,
        ha="center",
        va="center",
        fontsize=9,
        color="#2f2f2f",
        linespacing=1.2,
    )
    return patch


def _arrow(ax, p0, p1, color="#444444"):
    arrow = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.8,
        color=color,
        connectionstyle="arc3,rad=0.0",
    )
    arrow.set_clip_on(False)
    ax.add_patch(arrow)


def make_figure(out_dir: Path, stem: str) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(11.5, 4.6), constrained_layout=False)
    # Wider limits prevent cut-offs at export in both PNG and PDF backends.
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.02, 1.05)
    ax.axis("off")

    # Palette tuned for print legibility.
    c_gen = "#dceeff"
    c_meas = "#e6f4df"
    c_score = "#fff1d6"
    c_iter = "#f7def0"

    w, h = 0.2, 0.34
    y = 0.34
    x1, x2, x3, x4 = 0.04, 0.29, 0.54, 0.79

    _box(
        ax,
        (x1, y),
        w,
        h,
        "1. Generate",
        "Create candidate\ntrain datasets $T$\n(mixes, augments,\nweights, filters)",
        c_gen,
    )
    _box(
        ax,
        (x2, y),
        w,
        h,
        "2. Measure",
        "Compute directional\nmismatch features\n$E\\!\\to\\!T$, $T\\!\\to\\!E$\n+ density controls",
        c_meas,
    )
    _box(
        ax,
        (x3, y),
        w,
        h,
        "3. Score",
        "Predict utility\n$U(T\\mid B)=\\hat{P}(T\\to B)$\n(target: AUC\nnormalized observed)",
        c_score,
    )
    _box(
        ax,
        (x4, y),
        w,
        h,
        "4. Iterate",
        "Select top candidates,\ninspect diagnostics\n(missing support vs\nextra mass), refine $T$",
        c_iter,
    )

    ymid = y + h / 2
    _arrow(ax, (x1 + w, ymid), (x2, ymid))
    _arrow(ax, (x2 + w, ymid), (x3, ymid))
    _arrow(ax, (x3 + w, ymid), (x4, ymid))

    # Loop-back arrow from Iterate to Generate.
    loop_arrow = FancyArrowPatch(
        (x4 + w * 0.92, y + h * 0.15),
        (x1 + w * 0.08, y + h * 0.15),
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.8,
        color="#7a3c6d",
        connectionstyle="arc3,rad=-0.55",
    )
    loop_arrow.set_clip_on(False)
    ax.add_patch(loop_arrow)
    ax.text(
        0.5,
        0.14,
        "Closed-loop data design guided by predicted transfer utility",
        ha="center",
        va="center",
        fontsize=10,
        color="#5a2d51",
    )

    ax.set_title("Fig. 4: Utility-guided dataset design loop", fontsize=13, pad=8)

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.08)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Fig. 4 dataset design loop diagram.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures/section5"),
        help="Output directory.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="fig4_dataset_design_loop",
        help="Output filename stem.",
    )
    args = parser.parse_args()

    png_path, pdf_path = make_figure(args.out_dir, args.stem)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
