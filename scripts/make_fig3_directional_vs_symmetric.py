#!/usr/bin/env python3
"""
Make Fig. 3 conceptual diagram:
- Case A: under-coverage (eval has unsupported mode)
- Case B: extra mass (train has unsupported mode)
- Symmetric MMD stays unchanged under direction swap
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def _rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    d2 = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    return np.exp(-gamma * d2)


def _mmd2_biased(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    # Biased MMD^2 estimator (symmetric by construction).
    k_xx = _rbf_kernel(x, x, gamma).mean()
    k_yy = _rbf_kernel(y, y, gamma).mean()
    k_xy = _rbf_kernel(x, y, gamma).mean()
    return float(k_xx + k_yy - 2.0 * k_xy)


def _style_scatter_panel(
    ax: plt.Axes,
    train: np.ndarray,
    eval_: np.ndarray,
    title: str,
    e2t: float,
    t2e: float,
) -> None:
    ax.scatter(train[:, 0], train[:, 1], s=10, alpha=0.55, c="#1f77b4", label="Train")
    ax.scatter(eval_[:, 0], eval_[:, 1], s=10, alpha=0.55, c="#ff7f0e", label="Target")
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlim(-2.25, 2.25)
    ax.set_ylim(-1.2, 2.4)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_xlabel("Feature Axis 1")
    ax.set_ylabel("Feature Axis 2")
    ax.text(
        0.03,
        0.97,
        f"Target->Train mean NN: {e2t:.3f}\nTrain->Target mean NN: {t2e:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )


def make_figure(seed: int, out_dir: Path, stem: str) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)

    # Case A: under-coverage (eval has one extra mode absent from train)
    train_a = _sample_mixture(
        rng,
        centers=[(-1.0, 0.0), (1.0, 0.0)],
        n_per_center=[260, 260],
    )
    eval_a = _sample_mixture(
        rng,
        centers=[(-1.0, 0.0), (1.0, 0.0), (0.0, 1.7)],
        n_per_center=[170, 170, 170],
    )

    # Case B: extra mass (swap train/eval roles of Case A)
    train_b = eval_a.copy()
    eval_b = train_a.copy()

    e2t_a = _directed_mean_nn(eval_a, train_a)
    t2e_a = _directed_mean_nn(train_a, eval_a)
    e2t_b = _directed_mean_nn(eval_b, train_b)
    t2e_b = _directed_mean_nn(train_b, eval_b)

    mmd_a = _mmd2_biased(train_a, eval_a, gamma=0.8)
    mmd_b = _mmd2_biased(train_b, eval_b, gamma=0.8)

    fig = plt.figure(figsize=(13.2, 4.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.0])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    _style_scatter_panel(
        ax1,
        train=train_a,
        eval_=eval_a,
        title="A. Under-coverage (missing support)",
        e2t=e2t_a,
        t2e=t2e_a,
    )
    _style_scatter_panel(
        ax2,
        train=train_b,
        eval_=eval_b,
        title="B. Extra mass (train-only regions)",
        e2t=e2t_b,
        t2e=t2e_b,
    )

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc="lower right", fontsize=8, framealpha=0.9)

    x = np.arange(2)
    labels = ["Case A", "Case B"]

    ax3.bar(x, [mmd_a, mmd_b], width=0.55, color="#8c8c8c", alpha=0.9, label="Symmetric MMD^2")
    ax3.set_xticks(x, labels)
    ax3.set_ylabel("MMD^2 (symmetric)", color="#3a3a3a")
    ax3.set_title("C. Symmetric Collapse vs Directional Split", fontsize=11, pad=8)
    ax3.grid(axis="y", alpha=0.25, linewidth=0.6)

    ax3r = ax3.twinx()
    ax3r.plot(x, [e2t_a, e2t_b], "o-", color="#ff7f0e", linewidth=2.0, markersize=6, label="Target->Train")
    ax3r.plot(x, [t2e_a, t2e_b], "s-", color="#1f77b4", linewidth=2.0, markersize=6, label="Train->Target")
    ax3r.set_ylabel("Directed mean 1-NN distance", color="#3a3a3a")

    ax3.text(
        0.5,
        0.98,
        "MMD nearly unchanged;\ndirectional terms swap.",
        transform=ax3.transAxes,
        va="top",
        ha="center",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3r.get_legend_handles_labels()
    leg_bars = ax3.legend(
        h1,
        l1,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        fontsize=8,
        framealpha=0.9,
    )
    ax3.add_artist(leg_bars)
    ax3r.legend(
        h2,
        l2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.15),
        fontsize=8,
        framealpha=0.9,
        ncol=2,
    )

    fig.suptitle(
        "Conceptual two-mode mismatch: directional metrics separate failure modes, symmetric MMD does not",
        fontsize=12,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate conceptual Fig. 3 for Section 4.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures/section4"),
        help="Output directory.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="fig3_directional_vs_symmetric_concept",
        help="Output filename stem (without extension).",
    )
    args = parser.parse_args()

    png_path, pdf_path = make_figure(seed=args.seed, out_dir=args.out_dir, stem=args.stem)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
