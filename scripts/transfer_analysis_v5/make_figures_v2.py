"""v5 final-draft figures, round 2.

F2v2  — paired precision/recall bars per variant (replaces the confusing d-bars):
        two bars per variant; the LAW is visible as "left bar tall in the scratch
        group, right bar tall in the pretrained group".
F5c   — rank-signal scatter, z-scored within context (the honest fix for the
        "stinky" pooled-PCK residual scatter): rows = motion rule vs appearance
        (DINO) — appearance shows no structure; columns = LOTO/LOBO/JOINT.
F5d   — supplement: gain-calibrated residual scatter in PCK units (shrink-gain
        head; std ratio ~1), colored by REGIME.

    python scripts/transfer_analysis_v5/make_figures_v2.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

V4 = Path("scripts/transfer_analysis_v4")
V5 = Path("scripts/transfer_analysis_v5")
CORE = V4 / "results_rule_v5core"
OUT = V5 / "results/figures"
OUT.mkdir(parents=True, exist_ok=True)

REGIME_COLOR = {"scratch": "#1f77b4", "pretrained": "#d62728"}


def regime_of(v):
    arch, pre, _ = v.split("|")
    return "scratch" if (pre == "False" or arch == "raft") else "pretrained"


def f2v2_paired_bars():
    m = pd.read_csv(V4 / "regime_direction_verification/master_table_mean_nn.csv")
    h = pd.read_csv(V5 / "results/rule_holdout_checks.csv")
    m = m.merge(h[["variant", "regime"]], on="variant")
    # group: scratch first (sorted by precision), then pretrained (by recall)
    m["grp"] = (m.regime != "scratch").astype(int)
    m = m.sort_values(["grp", "variant"])
    x = np.arange(len(m))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - w / 2, m.rho_ab, w, color="#2c7fb8",
           label="precision direction (train→target: off-target mass)")
    ax.bar(x + w / 2, m.rho_ba, w, color="#de2d26",
           label="recall direction (target→train: missing support)")
    ax.axhline(0, color="k", lw=0.8)
    n_scratch = int((m.regime == "scratch").sum())
    ax.axvline(n_scratch - 0.5, color="gray", ls="--", lw=1)
    ax.text(n_scratch / 2 - 0.5, 0.78, "FROM-SCRATCH variants\n(law: precision governs)",
            ha="center", fontsize=9, color="#2c7fb8")
    ax.text(n_scratch + (len(m) - n_scratch) / 2 - 0.5, 0.78,
            "PRETRAINED variants\n(law: recall governs)",
            ha="center", fontsize=9, color="#de2d26")
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("|", "\n") for v in m.variant], fontsize=7)
    ax.set_ylabel("within-context Spearman ρ (fit-free)")
    ax.set_ylim(-0.35, 0.95)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Which direction of motion coverage predicts transfer flips with training regime")
    fig.tight_layout()
    fig.savefig(OUT / "F2_direction_preference.png", dpi=180)
    plt.close(fig)


def _zscatter(ax, rows, head_col, title):
    z = rows.copy()
    z["rz"] = (z.actual - z.L)
    z["rz"] = z.groupby("context_id")["rz"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9))
    z["gz"] = z.groupby("context_id")[head_col].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9))
    z["regime"] = [regime_of(v) for v in z.variant]
    for reg, c in REGIME_COLOR.items():
        s = z[z.regime == reg]
        ax.scatter(s.gz, s.rz, s=8, alpha=0.45, color=c, label=reg)
    rs = [spearmanr(g.actual, g[head_col]).statistic
          for _, g in rows.groupby("context_id")
          if g[head_col].std() > 1e-12 and g.train_dataset.nunique() >= 3]
    rho = np.nanmean([r for r in rs if np.isfinite(r)])
    lim = 3.2
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.7)
    ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim)
    ax.set_title(f"{title}   mean ctx ρ = {rho:+.2f}", fontsize=9)
    ax.axhline(0, color="gray", lw=0.4), ax.axvline(0, color="gray", lw=0.4)


def f5c_rank_signal():
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8))
    for j, split in enumerate(["LOTO", "LOBO", "JOINT"]):
        mr = pd.read_csv(CORE / f"predictions/peak_pck/rows_{split}_motion_rule.csv")
        ap = pd.read_csv(CORE / f"predictions/peak_pck/rows_{split}_appearance.csv")
        _zscatter(axes[0, j], mr, "g", f"motion rule — {split}")
        _zscatter(axes[1, j], ap, "g", f"appearance (DINO) — {split}")
    axes[0, 0].set_ylabel("actual residual (z within context)")
    axes[1, 0].set_ylabel("actual residual (z within context)")
    for j in range(3):
        axes[1, j].set_xlabel("predicted score (z within context)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Rank signal, scale-normalized: motion rule carries it; appearance does not")
    fig.tight_layout()
    fig.savefig(OUT / "F5c_rank_signal_motion_vs_dino.png", dpi=180)
    plt.close(fig)


def f5d_calibrated_supplement():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, split in zip(axes, ["LOTO", "LOBO", "JOINT"]):
        r = pd.read_csv(V5 / f"results/benchsim_rule/rows_{split}_all_variants.csv")
        resid = r.actual - r.L
        head = r.g_shrink_gain
        reg = [regime_of(v) for v in r.variant]
        for rg, c in REGIME_COLOR.items():
            m = np.array(reg) == rg
            ax.scatter(head[m], resid[m], s=8, alpha=0.45, color=c, label=rg)
        sr = float(np.std(head) / np.std(resid))
        lim = np.percentile(np.abs(resid), 99) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.7)
        ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim)
        ax.set_title(f"{split}: dispersion ratio = {sr:.2f}")
        ax.set_xlabel("calibrated g (per-regime shrink gain, PCK)")
        ax.axhline(0, color="gray", lw=0.4), ax.axvline(0, color="gray", lw=0.4)
    axes[0].set_ylabel("actual residual (PCK)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Supplement: leakage-clean per-regime gain calibration (PCK units)")
    fig.tight_layout()
    fig.savefig(OUT / "F5d_residual_calibrated_supplement.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    f2v2_paired_bars()
    f5c_rank_signal()
    f5d_calibrated_supplement()
    print(f"wrote F2 (v2), F5c, F5d -> {OUT}")
