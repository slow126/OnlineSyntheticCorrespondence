"""v5 figures: F2 (per-variant direction preference), F5 (absolute scatter),
F4 (gap-stratified accuracy vs retraining reproducibility).

    python scripts/transfer_analysis_v5/make_figures.py \
        --out-dir scripts/transfer_analysis_v5/results/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

V4 = Path("scripts/transfer_analysis_v4")
V5 = Path("scripts/transfer_analysis_v5")
CORE = V4 / "results_rule_v5core"

ARCH_COLOR = {"catspp": "#1f77b4", "glunet": "#d62728", "raft": "#2ca02c"}


def f2_direction_bars(out):
    m = pd.read_csv(V4 / "regime_direction_verification/master_table_mean_nn.csv")
    h = pd.read_csv(V5 / "results/rule_holdout_checks.csv")
    m = m.merge(h[["variant", "mean_level", "regime"]], on="variant")
    m = m.sort_values("mean_level")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(m))
    colors = ["#1f77b4" if r == "scratch" else "#d62728" for r in m.regime]
    ax.bar(x, m.d, yerr=[m.d - m.d_lo, m.d_hi - m.d],
           color=colors, alpha=0.85, capsize=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("|", "\n") for v in m.variant], fontsize=7)
    ax.set_ylabel("d = ρ(precision) − ρ(recall)")
    ax.set_title("Direction preference flips with regime "
                 "(variants ordered by mean transfer level →)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#1f77b4", label="scratch (law: precision)"),
                       Patch(color="#d62728", label="pretrained (law: recall)")],
              fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "F2_direction_preference.png", dpi=180)
    plt.close(fig)


def f5_absolute_scatter(out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    for ax, split in zip(axes, ["LOTO", "LOBO", "JOINT"]):
        rows = pd.read_csv(CORE / f"predictions/peak_pck/rows_{split}_motion_rule.csv")
        rows["pred"] = rows.L + rows.g
        rows["arch"] = rows.variant.str.split("|").str[0]
        for arch, g in rows.groupby("arch"):
            ax.scatter(g.pred, g.actual, s=9, alpha=0.55,
                       color=ARCH_COLOR.get(arch, "gray"), label=arch)
        lims = [min(rows.pred.min(), rows.actual.min()) - 2,
                max(rows.pred.max(), rows.actual.max()) + 2]
        ax.plot(lims, lims, "k--", lw=0.8)
        r = pearsonr(rows.pred, rows.actual)[0]
        mae = float(np.mean(np.abs(rows.pred - rows.actual)))
        ax.set_title(f"{split}:  r={r:.3f}, MAE={mae:.1f} PCK")
        ax.set_xlabel("predicted peak PCK  (L + regime-rule g)")
        ax.set_xlim(lims), ax.set_ylim(lims)
    axes[0].set_ylabel("actual peak PCK")
    axes[0].legend(fontsize=8)
    fig.suptitle("Absolute transfer prediction: level anchor + fit-free regime rule")
    fig.tight_layout()
    fig.savefig(out / "F5_absolute_scatter.png", dpi=180)
    plt.close(fig)


def f4_gap_curves(out):
    df = pd.read_csv(V5 / "results/pairwise_gap_rule.csv")
    order = ["0-1", "1-2", "2-5", "5-10", ">10"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for fam, label, style in [
        ("same_arch", "same-arch retraining (ceiling)", "k-o"),
        ("cross_arch", "cross-arch retraining", "k--s"),
    ]:
        s = df[(df.measure == "empirical_reproducibility") & (df.family == fam)]
        s = s.set_index("gap_bin").reindex(order)
        ax.plot(order, s.acc, style, label=label, alpha=0.7)
    for split, color in [("LOTO", "#1f77b4"), ("LOBO", "#d62728")]:
        s = df[(df.measure == "predictor_accuracy") & (df.split == split)
               & (df.family == "motion_rule")]
        s = s.set_index("gap_bin").reindex(order)
        ax.plot(order, s.acc, "-^", color=color, label=f"rule predictor ({split})")
    ax.axhline(0.5, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("true |peak PCK gap| between source pair")
    ax.set_ylabel("pairwise ordering accuracy")
    ax.set_title("Errors concentrate on pairs retraining can't rank either")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "F4_gap_stratified.png", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="scripts/transfer_analysis_v5/results/figures")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f2_direction_bars(out)
    f5_absolute_scatter(out)
    f4_gap_curves(out)
    print(f"wrote F2/F4/F5 -> {out}")


if __name__ == "__main__":
    main()
