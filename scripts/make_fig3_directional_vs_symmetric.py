#!/usr/bin/env python3
"""
Fig. 1 conceptual diagram (directional vs. symmetric distances).

- (a) Under-coverage: the target places mass on a mode the source never renders,
  so the coverage distance d_{T->S} (target -> source) rises.
- (b) Off-target mass: the mirror case; the off-target distance d_{S->T} (source ->
  target) rises while coverage is perfect.
- (c) The two directed distances swap between the cases, while a symmetric
  distance (their average) stays essentially unchanged.

Everything in panel (c) is in the same distance units (directed mean 1-NN),
so the bars are directly comparable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

# Color semantics kept consistent across all three panels.
C_SOURCE = "#1f77b4"   # Source / Train (blue)
C_TARGET = "#ff7f0e"   # Target (orange)
C_SYM = "#8c8c8c"      # Symmetric distance (grey)


def _sample_mixture(
    rng: np.random.Generator,
    centers: list[tuple[float, float]],
    n_per_center: list[int],
    sigma: float = 0.22,
) -> np.ndarray:
    chunks = []
    for (cx, cy), n in zip(centers, n_per_center):
        chunks.append(rng.normal(loc=(cx, cy), scale=sigma, size=(n, 2)))
    return np.concatenate(chunks, axis=0)


def _directed_mean_nn(query: np.ndarray, reference: np.ndarray) -> float:
    # Euclidean 1-NN distance from query points to reference points.
    d2 = np.sum((query[:, None, :] - reference[None, :, :]) ** 2, axis=-1)
    return float(np.mean(np.sqrt(np.min(d2, axis=1))))


def _panel_caption(ax: plt.Axes, letter: str, text: str) -> None:
    """subcaption-style '(a) text' placed UNDER the panel, per the ACCV style guide."""
    ax.text(
        0.5, -0.235, rf"$\mathbf{{({letter})}}$ {text}",
        transform=ax.transAxes, ha="center", va="top", fontsize=11,
    )


def _style_scatter_panel(
    ax: plt.Axes,
    source: np.ndarray,
    target: np.ndarray,
    letter: str,
    caption: str,
    d_bt: float,
    d_tb: float,
) -> None:
    ax.scatter(source[:, 0], source[:, 1], s=10, alpha=0.55, c=C_SOURCE, label="Source (train)")
    ax.scatter(target[:, 0], target[:, 1], s=10, alpha=0.55, c=C_TARGET, label="Target")
    _panel_caption(ax, letter, caption)
    ax.set_xlim(-2.25, 2.25)
    ax.set_ylim(-1.2, 2.4)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_xlabel("Feature axis 1")
    ax.set_ylabel("Feature axis 2")
    ax.text(
        0.03,
        0.97,
        f"Coverage  $d_{{T\\to S}}$: {d_bt:.3f}\nOff-target $d_{{S\\to T}}$: {d_tb:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )


def make_figure(seed: int, out_dir: Path, stem: str) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)

    # Case A: under-coverage (target has one extra mode absent from source)
    source_a = _sample_mixture(rng, centers=[(-1.0, 0.0), (1.0, 0.0)], n_per_center=[260, 260])
    target_a = _sample_mixture(
        rng, centers=[(-1.0, 0.0), (1.0, 0.0), (0.0, 1.7)], n_per_center=[170, 170, 170]
    )

    # Case B: off-target mass (swap source/target roles of Case A)
    source_b = target_a.copy()
    target_b = source_a.copy()

    # Directed coverage distances. d_{T->S} = target -> source (coverage gap);
    # d_{S->T} = source -> target (off-target mass).
    dbt_a = _directed_mean_nn(target_a, source_a)
    dtb_a = _directed_mean_nn(source_a, target_a)
    dbt_b = _directed_mean_nn(target_b, source_b)
    dtb_b = _directed_mean_nn(source_b, target_b)

    # Symmetric distance = average of the two directed terms (stays put under swap).
    sym_a = 0.5 * (dbt_a + dtb_a)
    sym_b = 0.5 * (dbt_b + dtb_b)

    fig = plt.figure(figsize=(13.2, 4.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.0])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    _style_scatter_panel(
        ax1, source_a, target_a, "a",
        "Under-coverage: target mode missing from source", dbt_a, dtb_a
    )
    _style_scatter_panel(
        ax2, source_b, target_b, "b",
        "Off-target mass: source-only regions absent from target", dbt_b, dtb_b
    )
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc="lower right", fontsize=8, framealpha=0.9)

    # ---- Panel (c): grouped distance bars ----------------------------------
    group_x = np.array([0.0, 1.0])
    w = 0.26
    dbt_vals = [dbt_a, dbt_b]
    dtb_vals = [dtb_a, dtb_b]
    sym_vals = [sym_a, sym_b]

    ax3.bar(group_x - w, dbt_vals, width=w, color=C_TARGET, alpha=0.9,
            label=r"Coverage  $d_{T\to S}$ (target$\to$source)")
    ax3.bar(group_x, dtb_vals, width=w, color=C_SOURCE, alpha=0.9,
            label=r"Off-target $d_{S\to T}$ (source$\to$target)")
    bars_sym = ax3.bar(group_x + w, sym_vals, width=w, color=C_SYM, alpha=0.9,
                       hatch="//", edgecolor="white", label="Symmetric (average)")

    ax3.set_xticks(group_x, ["Case (a)\nUnder-coverage", "Case (b)\nOff-target mass"], fontsize=9)
    ax3.set_ylabel("Directed mean 1-NN distance")
    ax3.grid(axis="y", alpha=0.25, linewidth=0.6)

    ymax = max(max(dbt_vals), max(dtb_vals), max(sym_vals))
    ax3.set_ylim(0, ymax * 1.45)

    # Dotted reference line across the two grey bars: they sit at the same level.
    grey_level = float(np.mean(sym_vals))
    ax3.plot([group_x[0] + w, group_x[1] + w], [sym_vals[0], sym_vals[1]],
             linestyle=":", color="#555555", linewidth=1.2, zorder=5)

    # Annotation moved low, in a small translucent box, with arrows to the grey bars.
    txt = ax3.annotate(
        "Symmetric distance stays\nsimilar (grey); the directed\nterms swap between cases.",
        xy=(0.5, 0.20), xycoords="axes fraction",
        ha="center", va="center", fontsize=8.5, style="italic",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
              "alpha": 0.6, "edgecolor": "#999999"},
    )

    ax3.legend(loc="upper right", fontsize=9,
               framealpha=0.9, ncol=1, handlelength=1.6, borderpad=0.5)

    _panel_caption(ax3, "c", "Directed terms swap; symmetric stays flat")

    fig.subplots_adjust(left=0.045, right=0.99, top=0.98, bottom=0.27, wspace=0.22)

    # Realize the text box patch so the arrow tails can clip to its edge (patchA),
    # rather than starting at the box center and crossing over the text.
    fig.canvas.draw()
    box_patch = txt.get_bbox_patch()
    for bar in bars_sym:
        ann = ax3.annotate(
            "", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0.5, 0.20), textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.3,
                        "connectionstyle": "arc3,rad=0.15", "shrinkB": 4,
                        "patchA": box_patch},
        )
        # White halo so the arrow stays legible where it crosses the grey bars.
        ann.arrow_patch.set_path_effects(
            [pe.withStroke(linewidth=3.0, foreground="white")]
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the conceptual Fig. 1.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("ACCV_2026/figures"),
        help="Output directory.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="concept_directional_vs_symmetric",
        help="Output filename stem (without extension).",
    )
    args = parser.parse_args()

    png_path, pdf_path = make_figure(seed=args.seed, out_dir=args.out_dir, stem=args.stem)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
