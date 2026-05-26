"""Figures for v4 (per target × split × family × head).

Output naming:
  fig1_headline_bars__<target>.png       headline ctx_rho_g, ridge vs ranknet
  fig2_global_scatter__<target>.png      predicted L+g vs actual (ridge)
  fig3a_residual_scatter_g__<target>.png within-context residual (ridge)
  fig3a_residual_scatter_g_cal__<target>.png same but with gain calibration
  fig3b_residual_scatter_Lg__<target>.png full-model residual (ridge)
  fig4_controls__<target>.png            shuffle + random feature controls
  fig5_density_confound__<target>.png    density confound bars
  fig6_rank_scatter__<target>__<head>.png predicted RANK vs actual RANK
                                          (one panel per (split,family); head ∈
                                          ridge/ranknet)
  fig7_gain_cal_compare__<target>.png    abs_r comparison ridge vs gain-cal

Run:
    python scripts/transfer_analysis_v4/figures.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "font.size": 9,
})

FAMILY_COLOR = {
    "motion":         "#1b6ca8",
    "appearance":     "#d1495b",
    "both":           "#7a378b",
    "random":         "#9aa0a6",
    "density":        "#e08a3c",
    "motion_density": "#0f9b8e",
}
FAMILY_LABEL = {"motion": "motion (flow)", "appearance": "appearance (DINO)",
                "both": "both", "random": "random (control)",
                "density": "density (size + per-sample)",
                "motion_density": "motion + density"}
SPLITS = ["LOTO", "LOBO", "JOINT"]
FAMILIES = ["motion", "appearance", "both", "random"]
DENSITY_FAMILIES = ["density", "motion", "motion_density"]
HEAD_LABEL = {"g": "ridge", "g_zridge": "z-ridge", "g_rank": "ranknet", "g_gbm": "gbm"}


# ---------------------------------------------------------------------------
def fig_headline_bars(summary: pd.DataFrame, out_dir: Path, target: str) -> Path:
    """Bars: ridge / ranknet / gbm ρ_g per (split, family) — 3 heads side-by-side."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True,
                             constrained_layout=True)
    sub = summary[(summary["label"] == "main") &
                  (summary["target"] == target)]

    # Show whichever heads have non-trivial data in the summary
    available_heads = set(summary["head"].unique())
    candidates = [("g", "ridge", ""),
                  ("g_zridge", "z-ridge", "////"),
                  ("g_gbm", "gbm", ".."),
                  ("g_rank", "ranknet", "xx")]
    HEADS_SHOWN = []
    for k, lbl, hatch in candidates:
        if k in available_heads:
            sub_k = summary[(summary["head"] == k) & (summary["label"] == "main")]
            # require at least one finite non-zero value
            if sub_k["ctx_rho_g"].abs().max() > 1e-6:
                HEADS_SHOWN.append((k, lbl, hatch))
    if not HEADS_SHOWN:
        HEADS_SHOWN = [("g", "ridge", "")]
    width = max(0.85 / len(HEADS_SHOWN), 0.15)
    for ax, split in zip(axes, SPLITS):
        per_head = {}
        for h_key, _, _ in HEADS_SHOWN:
            per_head[h_key] = (sub[(sub["split"] == split) & (sub["head"] == h_key)]
                               .set_index("family").reindex(FAMILIES))
        if per_head["g"]["ctx_rho_g"].isna().all():
            ax.set_axis_off()
            continue
        x = np.arange(len(FAMILIES))
        colors = [FAMILY_COLOR[f] for f in FAMILIES]
        for i, (h_key, lbl, hatch) in enumerate(HEADS_SHOWN):
            offs = (i - (len(HEADS_SHOWN) - 1) / 2) * width
            s = per_head[h_key]
            heights = s["ctx_rho_g"].values
            err = np.vstack([heights - s["ctx_rho_g_lo"].values,
                             s["ctx_rho_g_hi"].values - heights])
            alpha = 1.0 if i == 0 else 0.6
            ax.bar(x + offs, heights, width, color=colors,
                   edgecolor="black", linewidth=0.4, hatch=hatch, alpha=alpha,
                   yerr=err, capsize=2, ecolor="black",
                   label=lbl if ax is axes[0] else None)
            for xi, h in zip(x + offs, heights):
                if np.isfinite(h):
                    ax.text(xi, h + (0.04 if h >= 0 else -0.06),
                            f"{h:+.2f}", ha="center",
                            va="bottom" if h >= 0 else "top",
                            fontsize=6, alpha=alpha)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([FAMILY_LABEL[f] for f in FAMILIES],
                           rotation=20, ha="right", fontsize=8)
        ax.set_title(f"{split}", fontsize=10, fontweight="bold")
        ax.set_ylim(-0.7, 0.8)
        ax.grid(axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.legend(loc="lower right", fontsize=8, frameon=False)

    axes[0].set_ylabel("within-context Spearman ρ (g only)")
    heads_label = " / ".join(lbl for _, lbl, _ in HEADS_SHOWN)
    fig.suptitle(f"Within-context ranking ρ — target = {target}\n"
                 f"heads: {heads_label}; 95% bootstrap CI.",
                 fontsize=11, fontweight="bold")
    path = out_dir / f"fig1_headline_bars__{target}.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def _benchmark_palette(benchmarks):
    cmap = plt.cm.tab20 if len(benchmarks) > 10 else plt.cm.tab10
    return {b: cmap(i % cmap.N) for i, b in enumerate(sorted(benchmarks))}


def fig_global_scatter(pred_dir: Path, out_dir: Path, target: str) -> Path:
    all_actuals = []
    cache = {}
    for split in SPLITS:
        for fam in FAMILIES:
            p = pred_dir / f"rows_{split}_{fam}.csv"
            if p.exists():
                df = pd.read_csv(p)
                cache[(split, fam)] = df
                all_actuals.append(df["actual"].values)
    if not all_actuals:
        return None
    all_actuals = np.concatenate(all_actuals)
    lo = float(np.quantile(all_actuals, 0.005))
    hi = float(np.quantile(all_actuals, 0.995))
    pad = 0.1 * (hi - lo)
    lo -= pad; hi += pad

    fig, axes = plt.subplots(len(SPLITS), len(FAMILIES),
                             figsize=(3.4 * len(FAMILIES), 3.3 * len(SPLITS)),
                             squeeze=False, constrained_layout=True)
    palette = None
    for ri, split in enumerate(SPLITS):
        for ci, fam in enumerate(FAMILIES):
            ax = axes[ri][ci]
            df = cache.get((split, fam))
            if df is None:
                ax.set_axis_off(); continue
            pred = (df["L"] + df["g"]).values
            act = df["actual"].values
            if palette is None:
                palette = _benchmark_palette(df["benchmark"].unique())
            for b in sorted(df["benchmark"].unique()):
                m = df["benchmark"].values == b
                ax.scatter(pred[m], act[m], s=14, alpha=0.6, linewidths=0,
                           color=palette.get(b, "#999"))
            ax.plot([lo, hi], [lo, hi], "--", color="black", lw=0.7, alpha=0.4)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            if len(pred) >= 3 and np.std(pred) > 0:
                r = pearsonr(pred, act)[0]
                ax.text(0.04, 0.95, f"pooled r = {r:+.2f}", transform=ax.transAxes,
                        va="top", fontsize=8,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7,
                                  pad=1.5))
            if ri == 0:
                ax.set_title(FAMILY_LABEL[fam], fontsize=9, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{split}\nactual", fontsize=9)
            if ri == len(SPLITS) - 1:
                ax.set_xlabel("predicted (L + g)", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(f"Predicted (L + g) vs actual — target = {target}\n"
                 f"r in each panel = **pooled Pearson r** across all rows (NOT the "
                 f"within-context claim).  All families look similar on LOBO because "
                 f"the band L carries ~0.7 of the fit by identity-borrowing — the "
                 f"feature signal lives in the within-context residuals (fig3/fig6).",
                 fontsize=10, fontweight="bold")
    path = out_dir / f"fig2_global_scatter__{target}.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def fig_residual_scatter(pred_dir: Path, out_dir: Path, target: str,
                         residual: str = "g") -> Path:
    """residual ∈ {'g','g_zridge','Lg'}."""
    cache = {}
    actual_resids_all = []
    for split in SPLITS:
        for fam in FAMILIES:
            p = pred_dir / f"rows_{split}_{fam}.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p).copy()
            df["actual_resid"] = (df["actual"]
                                  - df.groupby("context_id")["actual"].transform("mean"))
            df["Lg"] = df["L"] + df["g"]
            df["Lg_resid"] = (df["Lg"]
                              - df.groupby("context_id")["Lg"].transform("mean"))
            cache[(split, fam)] = df
            actual_resids_all.append(df["actual_resid"].values)
    if not cache:
        return None
    actual_resids_all = np.concatenate(actual_resids_all)
    yl = float(np.quantile(np.abs(actual_resids_all), 0.99)) * 1.1

    if residual == "g":
        xcol = "g"
        xlabel = "g (ridge predicted residual)"
        title = (f"Within-context residuals (RIDGE g) — target = {target}\n"
                 f"motion has a slope; appearance / random are flat clouds")
        suffix = "fig3a_residual_scatter_g"
    elif residual == "g_zridge":
        xcol = "g_zridge"
        xlabel = "g_zridge (within-context z-score ridge)"
        title = (f"Within-context residuals (Z-RIDGE) — target = {target}\n"
                 f"target z-scored per context before fitting → each context "
                 f"contributes equally to the loss → predictions naturally rescale "
                 f"to match each benchmark's actual spread")
        suffix = "fig3a_residual_scatter_g_zridge"
    else:
        xcol = "Lg_resid"
        xlabel = "(L+g) − ctx mean — full-model residual"
        title = (f"Within-context residuals (FULL MODEL L+g) — target = {target}\n"
                 f"includes identity-borrowing signal carried by L on LOBO "
                 f"(conflates feature signal with band)")
        suffix = "fig3b_residual_scatter_Lg"

    fig, axes = plt.subplots(len(SPLITS), len(FAMILIES),
                             figsize=(3.4 * len(FAMILIES), 3.3 * len(SPLITS)),
                             squeeze=False, constrained_layout=True)
    palette = None
    for ri, split in enumerate(SPLITS):
        all_x_split = np.concatenate([cache[(split, fam)][xcol].values
                                      for fam in FAMILIES if (split, fam) in cache])
        xl = float(np.quantile(np.abs(all_x_split), 0.99)) * 1.1 + 1e-6
        for ci, fam in enumerate(FAMILIES):
            ax = axes[ri][ci]
            df = cache.get((split, fam))
            if df is None:
                ax.set_axis_off(); continue
            x = df[xcol].values; y = df["actual_resid"].values
            if palette is None:
                palette = _benchmark_palette(df["benchmark"].unique())
            for b in sorted(df["benchmark"].unique()):
                m = df["benchmark"].values == b
                ax.scatter(x[m], y[m], s=14, alpha=0.65, linewidths=0,
                           color=palette.get(b, "#999"))
            ax.axhline(0, color="black", lw=0.5, alpha=0.4)
            ax.axvline(0, color="black", lw=0.5, alpha=0.4)
            rs = []
            score_col = ("g" if residual == "g"
                         else "g_zridge" if residual == "g_zridge"
                         else "Lg")
            for _, grp in df.groupby("context_id"):
                if grp["train_dataset"].nunique() < 3:
                    continue
                if grp[score_col].std() < 1e-12:
                    continue
                rho = spearmanr(grp["actual"], grp[score_col]).statistic
                if np.isfinite(rho):
                    rs.append(rho)
            ctx_rho = float(np.nanmean(rs)) if rs else float("nan")
            ax.text(0.04, 0.95, f"ctx ρ = {ctx_rho:+.2f}", transform=ax.transAxes,
                    va="top", fontsize=8, fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7,
                              pad=1.5))
            ax.set_xlim(-xl, xl); ax.set_ylim(-yl, yl)
            if ri == 0:
                ax.set_title(FAMILY_LABEL[fam], fontsize=9, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{split}\nactual residual", fontsize=9)
            if ri == len(SPLITS) - 1:
                ax.set_xlabel(xlabel, fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(title, fontsize=10, fontweight="bold")
    path = out_dir / f"{suffix}__{target}.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def fig_rank_scatter(pred_dir: Path, out_dir: Path, target: str, head: str) -> Path:
    """Predicted-rank vs actual-rank scatter, per (split, family). Within each
    context, rows are ranked 1..n; this strips magnitude noise so the ordering
    signal is visible directly. Diagonal line = perfect ordering.

    head ∈ {'g', 'g_cal', 'g_rank'}. ridge_cal will be identical to ridge by
    construction (rank-invariant positive rescale)."""
    cache = {}
    for split in SPLITS:
        for fam in FAMILIES:
            p = pred_dir / f"rows_{split}_{fam}.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p).copy()
            if head not in df.columns:
                continue
            # within-context rank (1..n)
            df["actual_rank"] = df.groupby("context_id")["actual"].rank(method="average")
            df["pred_rank"] = df.groupby("context_id")[head].rank(method="average")
            df["ctx_n"] = df.groupby("context_id")["train_dataset"].transform("nunique")
            cache[(split, fam)] = df
    if not cache:
        return None

    fig, axes = plt.subplots(len(SPLITS), len(FAMILIES),
                             figsize=(3.2 * len(FAMILIES), 3.1 * len(SPLITS)),
                             squeeze=False, constrained_layout=True)
    palette = None
    for ri, split in enumerate(SPLITS):
        for ci, fam in enumerate(FAMILIES):
            ax = axes[ri][ci]
            df = cache.get((split, fam))
            if df is None:
                ax.set_axis_off(); continue
            # Normalize rank to [0,1] so contexts with different n are comparable
            df_n = df.copy()
            df_n["actual_q"] = (df_n["actual_rank"] - 1) / (df_n["ctx_n"] - 1).clip(lower=1)
            df_n["pred_q"] = (df_n["pred_rank"] - 1) / (df_n["ctx_n"] - 1).clip(lower=1)
            if palette is None:
                palette = _benchmark_palette(df["benchmark"].unique())
            for b in sorted(df["benchmark"].unique()):
                m = df_n["benchmark"].values == b
                ax.scatter(df_n["pred_q"].values[m],
                           df_n["actual_q"].values[m],
                           s=18, alpha=0.55, linewidths=0,
                           color=palette.get(b, "#999"))
            ax.plot([0, 1], [0, 1], "--", color="black", lw=0.7, alpha=0.5)
            rs = []
            for _, grp in df.groupby("context_id"):
                if grp["train_dataset"].nunique() < 3:
                    continue
                if grp[head].std() < 1e-12:
                    continue
                rho = spearmanr(grp["actual"], grp[head]).statistic
                if np.isfinite(rho):
                    rs.append(rho)
            ctx_rho = float(np.nanmean(rs)) if rs else float("nan")
            ax.text(0.04, 0.95, f"ctx ρ = {ctx_rho:+.2f}",
                    transform=ax.transAxes, va="top", fontsize=8,
                    fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7,
                              pad=1.5))
            ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
            if ri == 0:
                ax.set_title(FAMILY_LABEL[fam], fontsize=9, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{split}\nactual rank (within-context)",
                              fontsize=9)
            if ri == len(SPLITS) - 1:
                ax.set_xlabel(f"predicted rank ({HEAD_LABEL[head]})", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(f"Within-context predicted vs actual rank "
                 f"({HEAD_LABEL[head]}) — target = {target}\n"
                 f"each context ranks its training sources from 1..n; diagonal = "
                 f"perfect order. Rank-space view removes magnitude noise.",
                 fontsize=10, fontweight="bold")
    path = out_dir / f"fig6_rank_scatter__{target}__{head}.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def fig_controls(summary: pd.DataFrame, out_dir: Path, target: str) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True,
                             constrained_layout=True)
    sub = summary[(summary["target"] == target) & (summary["head"] == "g")]
    main = sub[sub["label"] == "main"]
    sh = sub[sub["label"] == "shuffle"]

    for ax, split in zip(axes, SPLITS):
        rows = []
        for fam in FAMILIES:
            m = main[(main.split == split) & (main.family == fam)]
            if not m.empty:
                rows.append((f"{fam}\n(real)", float(m["ctx_rho_g"].iloc[0]),
                             float(m["ctx_rho_g_lo"].iloc[0]),
                             float(m["ctx_rho_g_hi"].iloc[0]),
                             FAMILY_COLOR[fam], False))
        for fam in ["motion", "random"]:
            s = sh[(sh.split == split) & (sh.family == fam)]
            if not s.empty:
                rows.append((f"{fam}\n(shuffled)", float(s["ctx_rho_g"].iloc[0]),
                             float(s["ctx_rho_g_lo"].iloc[0]),
                             float(s["ctx_rho_g_hi"].iloc[0]),
                             FAMILY_COLOR[fam], True))
        if not rows:
            ax.set_axis_off(); continue
        labels, vals, los, his, colors, is_sh = zip(*rows)
        x = np.arange(len(labels))
        err = np.vstack([np.asarray(vals) - np.asarray(los),
                         np.asarray(his) - np.asarray(vals)])
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.4,
                      yerr=err, capsize=3, ecolor="black", alpha=1.0)
        for bar, hatched in zip(bars, is_sh):
            if hatched:
                bar.set_hatch("////"); bar.set_alpha(0.6)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=15,
                                              ha="right")
        ax.set_title(split, fontsize=10, fontweight="bold")
        ax.set_ylim(-0.7, 0.8)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("ctx ρ (ridge g)")
    fig.suptitle(f"Controls — target = {target}\n"
                 f"hatched bars = target shuffled within context (should be ≈ 0)",
                 fontsize=10, fontweight="bold")
    path = out_dir / f"fig4_controls__{target}.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def fig_density_confound(summary: pd.DataFrame, out_dir: Path, target: str) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0), sharey=True,
                             constrained_layout=True)
    sub = summary[(summary["label"] == "main") &
                  (summary["target"] == target) &
                  (summary["head"] == "g")]
    for ax, split in zip(axes, SPLITS):
        s = sub[sub["split"] == split].set_index("family").reindex(DENSITY_FAMILIES)
        if s["ctx_rho_g"].isna().all():
            ax.set_axis_off(); continue
        x = np.arange(len(DENSITY_FAMILIES))
        heights = s["ctx_rho_g"].values
        err = np.vstack([heights - s["ctx_rho_g_lo"].values,
                         s["ctx_rho_g_hi"].values - heights])
        colors = [FAMILY_COLOR[f] for f in DENSITY_FAMILIES]
        ax.bar(x, heights, color=colors, edgecolor="black", linewidth=0.4,
               yerr=err, capsize=4, ecolor="black")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([FAMILY_LABEL[f] for f in DENSITY_FAMILIES],
                                              rotation=20, ha="right", fontsize=8)
        ax.set_title(split, fontsize=10, fontweight="bold")
        ax.set_ylim(-0.7, 0.8)
        ax.grid(axis="y", alpha=0.3)
        for xi, h in zip(x, heights):
            if np.isfinite(h):
                ax.text(xi, h + (0.04 if h >= 0 else -0.06),
                        f"{h:+.2f}", ha="center",
                        va="bottom" if h >= 0 else "top", fontsize=8,
                        fontweight="bold")
    axes[0].set_ylabel("within-context Spearman ρ (ridge g)")
    fig.suptitle(f"Density confound — target = {target}\n"
                 f"if density alone ≈ 0 and motion+density ≈ motion, motion survives partialling",
                 fontsize=11, fontweight="bold")
    path = out_dir / f"fig5_density_confound__{target}.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def fig_gain_cal_compare(summary: pd.DataFrame, out_dir: Path,
                         target: str) -> Path:
    """abs_r comparison: ridge vs z-ridge."""
    main = summary[(summary["label"] == "main") &
                   (summary["target"] == target) &
                   (summary["head"].isin(["g", "g_zridge"]))]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), sharey=True,
                             constrained_layout=True)
    width = 0.38
    for ax, split in zip(axes, SPLITS):
        rows_r = main[(main["split"] == split) & (main["head"] == "g")] \
            .set_index("family").reindex(FAMILIES)
        rows_c = main[(main["split"] == split) & (main["head"] == "g_zridge")] \
            .set_index("family").reindex(FAMILIES)
        if rows_r["abs_r_Lg"].isna().all():
            ax.set_axis_off(); continue
        x = np.arange(len(FAMILIES))
        h_r = rows_r["abs_r_Lg"].values
        err_r = np.vstack([h_r - rows_r["abs_r_Lg_lo"].values,
                           rows_r["abs_r_Lg_hi"].values - h_r])
        h_c = rows_c["abs_r_Lg"].values
        err_c = np.vstack([h_c - rows_c["abs_r_Lg_lo"].values,
                           rows_c["abs_r_Lg_hi"].values - h_c])
        colors = [FAMILY_COLOR[f] for f in FAMILIES]
        ax.bar(x - width / 2, h_r, width, color=colors, edgecolor="black",
               linewidth=0.4, yerr=err_r, capsize=2, ecolor="black",
               label="ridge")
        ax.bar(x + width / 2, h_c, width, color=colors, edgecolor="black",
               linewidth=0.4, hatch="////", alpha=0.55,
               yerr=err_c, capsize=2, ecolor="black", label="z-ridge")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([FAMILY_LABEL[f] for f in FAMILIES],
                           rotation=20, ha="right", fontsize=8)
        ax.set_title(split, fontsize=10, fontweight="bold")
        ax.set_ylim(-0.4, 1.0)
        ax.grid(axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.legend(loc="lower right", fontsize=8, frameon=False)

    axes[0].set_ylabel("abs r (L+g) — pooled calibration")
    fig.suptitle(f"Pooled calibration: ridge vs z-ridge — target = {target}\n"
                 f"z-ridge normalizes target by context std at fit time → predictions "
                 f"are naturally on-scale per benchmark",
                 fontsize=10, fontweight="bold")
    path = out_dir / f"fig7_zridge_compare__{target}.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def fig_per_context_rho_hist(pred_dir: Path, out_dir: Path, target: str,
                             head: str = "g") -> Path:
    """Per-context ρ histogram: every context (benchmark × variant) contributes
    one ρ value; this is the distribution behind the mean ctx_rho_g. Tells you
    *which* contexts the model wins on vs loses on, and how wide the spread is.

    Bars are colored by benchmark to show clustering — if all the spair
    contexts are at the negative end and synthetic at the positive end, that's
    a label-density story right there."""
    fig, axes = plt.subplots(len(SPLITS), len(FAMILIES),
                             figsize=(3.4 * len(FAMILIES), 2.6 * len(SPLITS)),
                             squeeze=False, sharex=True, constrained_layout=True)
    palette = None
    bins = np.linspace(-1.0, 1.0, 21)
    for ri, split in enumerate(SPLITS):
        for ci, fam in enumerate(FAMILIES):
            ax = axes[ri][ci]
            p = pred_dir / f"rows_{split}_{fam}.csv"
            if not p.exists():
                ax.set_axis_off(); continue
            df = pd.read_csv(p)
            if head not in df.columns:
                ax.set_axis_off(); continue
            if palette is None:
                palette = _benchmark_palette(df["benchmark"].unique())
            rho_by_bench = {}
            rho_all = []
            for ctx, grp in df.groupby("context_id"):
                if grp["train_dataset"].nunique() < 3:
                    continue
                if grp[head].std() < 1e-12:
                    continue
                rho = spearmanr(grp["actual"], grp[head]).statistic
                if not np.isfinite(rho):
                    continue
                bench = grp["benchmark"].iloc[0]
                rho_by_bench.setdefault(bench, []).append(rho)
                rho_all.append(rho)
            if not rho_all:
                ax.set_axis_off(); continue
            mean_rho = float(np.mean(rho_all))
            # Stacked bars per benchmark
            base = np.zeros(len(bins) - 1)
            for b in sorted(rho_by_bench):
                h, _ = np.histogram(rho_by_bench[b], bins=bins)
                ax.bar((bins[:-1] + bins[1:]) / 2, h, width=np.diff(bins),
                       bottom=base, color=palette.get(b, "#999"), alpha=0.85,
                       edgecolor="white", linewidth=0.4, label=b if ri == 0 and ci == 0 else None)
                base = base + h
            ax.axvline(0, color="black", lw=0.5, alpha=0.5)
            ax.axvline(mean_rho, color="red", lw=1.2, linestyle="--",
                       label=f"mean ρ = {mean_rho:+.2f}" if ri == 0 else None)
            n_pos = sum(1 for r in rho_all if r > 0)
            ax.text(0.04, 0.95,
                    f"n_ctx={len(rho_all)}\n{n_pos}/{len(rho_all)} > 0\n"
                    f"mean = {mean_rho:+.2f}",
                    transform=ax.transAxes, va="top", fontsize=7,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))
            ax.set_xlim(-1.05, 1.05)
            if ri == 0:
                ax.set_title(FAMILY_LABEL[fam], fontsize=9, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{split}\n# contexts", fontsize=9)
            if ri == len(SPLITS) - 1:
                ax.set_xlabel(f"per-context Spearman ρ ({HEAD_LABEL[head]})", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(f"Per-context ρ distribution ({HEAD_LABEL[head]}) — target = {target}\n"
                 f"each context contributes one ρ; histogram shows wins (right of 0) "
                 f"and losses (left of 0). The headline ρ is just the mean of this "
                 f"distribution.",
                 fontsize=10, fontweight="bold")
    path = out_dir / f"fig8_per_context_rho_hist__{target}__{head}.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def fig_topk_hit_rate(pred_dir: Path, out_dir: Path, target: str,
                      head: str = "g", ks=(1, 3, 5)) -> Path:
    """Top-K hit rate: for each context, how often does the model's top-K
    predicted training sources overlap with the actual top-K?

    Practical metric: if you used the predictor to pick training datasets for
    a new benchmark, how often would your top picks include the genuine top
    picks? Reported as | top_k_pred ∩ top_k_actual | / k, averaged across
    contexts."""
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(11, 4.2), sharey=True,
                             constrained_layout=True)
    width = 0.18
    for ax, split in zip(axes, SPLITS):
        bars_per_fam = []
        for fam in FAMILIES:
            p = pred_dir / f"rows_{split}_{fam}.csv"
            if not p.exists():
                bars_per_fam.append([np.nan] * len(ks)); continue
            df = pd.read_csv(p)
            if head not in df.columns:
                bars_per_fam.append([np.nan] * len(ks)); continue
            scores = []
            for ctx, grp in df.groupby("context_id"):
                if grp["train_dataset"].nunique() < max(ks) + 1:
                    continue
                n = len(grp)
                act_order = grp.sort_values("actual", ascending=False)["train_dataset"].tolist()
                pred_order = grp.sort_values(head, ascending=False)["train_dataset"].tolist()
                scores.append([
                    len(set(pred_order[:k]) & set(act_order[:k])) / k
                    for k in ks
                ])
            if not scores:
                bars_per_fam.append([np.nan] * len(ks)); continue
            mean_hit = np.mean(scores, axis=0)
            bars_per_fam.append(mean_hit.tolist())
        bars_arr = np.array(bars_per_fam)  # (n_fam, n_ks)
        x = np.arange(len(FAMILIES))
        for ki, k in enumerate(ks):
            offs = (ki - (len(ks) - 1) / 2) * width
            colors = [FAMILY_COLOR[f] for f in FAMILIES]
            ax.bar(x + offs, bars_arr[:, ki], width=width, color=colors,
                   edgecolor="black", linewidth=0.4,
                   alpha=0.5 + 0.25 * ki, label=f"top-{k}" if ax is axes[0] else None,
                   hatch={0: "", 1: "//", 2: "xx"}[ki])
            for xi, h in zip(x + offs, bars_arr[:, ki]):
                if np.isfinite(h):
                    ax.text(xi, h + 0.01, f"{h:.2f}", ha="center", va="bottom",
                            fontsize=6)
        # Random-chance baselines (top-k / n, with n averaged over contexts)
        # For k=1 with ~11 sources, random hit ≈ 1/11; for k=3 it's 3/11; for k=5 it's 5/11.
        # Use empirical n per (split, family) — assume ~11 sources for LOTO/LOBO, ~10 for JOINT.
        ax.axhline(1 / 11, color="grey", linestyle=":", lw=0.8, alpha=0.6)
        ax.axhline(3 / 11, color="grey", linestyle=":", lw=0.8, alpha=0.6)
        ax.axhline(5 / 11, color="grey", linestyle=":", lw=0.8, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([FAMILY_LABEL[f] for f in FAMILIES],
                           rotation=20, ha="right", fontsize=8)
        ax.set_title(split, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=7, frameon=False)
            ax.set_ylabel(f"top-K hit rate ({HEAD_LABEL[head]})")

    fig.suptitle(f"Top-K training-source hit rate ({HEAD_LABEL[head]}) — target = {target}\n"
                 f"for each (benchmark, variant) context, what fraction of the top-K "
                 f"predicted training datasets are also in the actual top-K?\n"
                 f"dotted lines = random-chance baselines (1/11, 3/11, 5/11)",
                 fontsize=10, fontweight="bold")
    path = out_dir / f"fig9_topk_hit_rate__{target}__{head}.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def fig_residual_hexbin(pred_dir: Path, out_dir: Path, target: str,
                        head: str = "g") -> Path:
    """Hexbin density of within-context (predicted residual, actual residual)
    overlaid with the per-benchmark scatter. Density shows where the bulk of
    points live; the diagonal density along the trend is the visual analog of
    the within-context Spearman."""
    fig, axes = plt.subplots(len(SPLITS), len(FAMILIES),
                             figsize=(3.4 * len(FAMILIES), 3.2 * len(SPLITS)),
                             squeeze=False, constrained_layout=True)
    # Pre-pass: collect actual residuals across all panels for unified y-scale.
    cache = {}
    actual_all = []
    for split in SPLITS:
        for fam in FAMILIES:
            p = pred_dir / f"rows_{split}_{fam}.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p).copy()
            if head not in df.columns:
                continue
            df["actual_resid"] = (df["actual"]
                                  - df.groupby("context_id")["actual"].transform("mean"))
            cache[(split, fam)] = df
            actual_all.append(df["actual_resid"].values)
    if not cache:
        return None
    actual_all = np.concatenate(actual_all)
    yl = float(np.quantile(np.abs(actual_all), 0.99)) * 1.1

    for ri, split in enumerate(SPLITS):
        all_x = np.concatenate([cache[(split, fam)][head].values
                                for fam in FAMILIES if (split, fam) in cache])
        xl = float(np.quantile(np.abs(all_x), 0.99)) * 1.1 + 1e-6
        for ci, fam in enumerate(FAMILIES):
            ax = axes[ri][ci]
            df = cache.get((split, fam))
            if df is None:
                ax.set_axis_off(); continue
            x = df[head].values; y = df["actual_resid"].values
            # Mask to in-range for hexbin (else outliers dominate the binning)
            m = (np.abs(x) <= xl) & (np.abs(y) <= yl)
            if m.sum() > 10:
                hb = ax.hexbin(x[m], y[m], gridsize=18, cmap="Blues",
                               mincnt=1, linewidths=0)
            ax.axhline(0, color="black", lw=0.5, alpha=0.4)
            ax.axvline(0, color="black", lw=0.5, alpha=0.4)
            rs = []
            for _, grp in df.groupby("context_id"):
                if grp["train_dataset"].nunique() < 3:
                    continue
                if grp[head].std() < 1e-12:
                    continue
                rho = spearmanr(grp["actual"], grp[head]).statistic
                if np.isfinite(rho):
                    rs.append(rho)
            ctx_rho = float(np.nanmean(rs)) if rs else float("nan")
            ax.text(0.04, 0.95, f"ctx ρ = {ctx_rho:+.2f}", transform=ax.transAxes,
                    va="top", fontsize=8, fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))
            ax.set_xlim(-xl, xl); ax.set_ylim(-yl, yl)
            if ri == 0:
                ax.set_title(FAMILY_LABEL[fam], fontsize=9, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{split}\nactual residual", fontsize=9)
            if ri == len(SPLITS) - 1:
                ax.set_xlabel(f"predicted residual ({HEAD_LABEL[head]})", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(f"Within-context residuals — HEXBIN density ({HEAD_LABEL[head]}) "
                 f"— target = {target}\n"
                 f"density view of fig3a; bulk-of-data diagonal trend is the visual "
                 f"analog of ctx ρ",
                 fontsize=10, fontweight="bold")
    path = out_dir / f"fig10_residual_hexbin__{target}__{head}.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="scripts/transfer_analysis_v4/results")
    args = ap.parse_args()
    root = Path(".").resolve()
    out_dir = root / args.results / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(root / args.results / "summary.csv")
    preds_root = root / args.results / "predictions"

    targets = sorted(summary["target"].unique())
    print(f"figures: targets={targets} -> {out_dir}")
    for target in targets:
        pred_dir = preds_root / target
        # fig6 (rank scatter) dropped — at N=10 sources/context the grid is
        # fully populated and unreadable. fig8 (per-context ρ hist) +
        # fig10 (hexbin) cover the same information more clearly.
        for fn in (
            lambda: fig_headline_bars(summary, out_dir, target),
            lambda: fig_global_scatter(pred_dir, out_dir, target),
            lambda: fig_residual_scatter(pred_dir, out_dir, target, residual="g"),
            lambda: fig_residual_scatter(pred_dir, out_dir, target, residual="g_zridge"),
            lambda: fig_residual_scatter(pred_dir, out_dir, target, residual="Lg"),
            lambda: fig_controls(summary, out_dir, target),
            lambda: fig_density_confound(summary, out_dir, target),
            lambda: fig_gain_cal_compare(summary, out_dir, target),
            lambda: fig_per_context_rho_hist(pred_dir, out_dir, target, head="g"),
            lambda: fig_topk_hit_rate(pred_dir, out_dir, target, head="g"),
            lambda: fig_residual_hexbin(pred_dir, out_dir, target, head="g"),
            lambda: fig_residual_hexbin(pred_dir, out_dir, target, head="g_zridge"),
        ):
            try:
                path = fn()
                if path is not None:
                    print(f"  {path.name}")
            except Exception as e:
                print(f"  (failed: {e})")


if __name__ == "__main__":
    main()
