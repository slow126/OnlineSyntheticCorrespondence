"""v3 figure suite — one shared style, regime-color-coded throughout.

Conventions (consistent across every figure):
  blue  = scratch regime / precision direction d(T->B)   (the law pairs them)
  red   = pretrained regime / recall direction d(B->T)
  gray  = controls, references, retraining baselines

F2  the law: (a) per-variant paired directional rho (dumbbell), grouped by
    regime; (b) the continuum: flip d vs mean transfer level (rho = -0.80).
F3  specificity control: flip statistic per estimator, motion vs DINO.
F4  gap-stratified pairwise accuracy vs retraining reproducibility.
F5  absolute prediction scatters; JOINT panel uses the two-way similarity
    anchor (joint_anchor_v2), NOT the retired degenerate anchor.
F6  closed-loop intervention: predicted vs actual margins + null control.

    python scripts/transfer_analysis_v5/make_figures_v3.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

V4 = Path("scripts/transfer_analysis_v4")
V5 = Path("scripts/transfer_analysis_v5")
CORE = V4 / "results_rule_v5core"
OUT = V5 / "results/figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#2b6cb0"   # scratch / precision d(T->B)
RED = "#c0392b"    # pretrained / recall d(B->T)
GRAY = "#6b7280"
LGRAY = "#c4c8cf"

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e5e7eb", "grid.linewidth": 0.7,
    "legend.frameon": False,
})


def vlabel(v: str) -> str:
    a, p, f = v.split("|")
    return f"{a} {'T' if p == 'True' else 'F'}/{'T' if f == 'True' else 'F'}"


def load_law():
    m = pd.read_csv(V4 / "regime_direction_verification/master_table_mean_nn.csv")
    h = pd.read_csv(V5 / "results/rule_holdout_checks.csv")
    return m.merge(h[["variant", "regime", "mean_level"]], on="variant")


def f2_law():
    """Panel a: interaction (crossover) plot — group-mean rho per direction,
    per-variant points jittered behind. Direct line-end labels; NO legend box
    and NO floating text inside the data region (every prior version had a
    collision). Panel b: the continuum, points labeled by cluster."""
    m = load_law()
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})

    groups = ["scratch", "pretrained"]
    gx = {"scratch": 0.0, "pretrained": 1.0}
    for col, c, name in [("rho_ab", BLUE, "rank by d(T→B)\noff-target mass"),
                         ("rho_ba", RED, "rank by d(B→T)\nmissing support")]:
        means = [m[m.regime == g][col].mean() for g in groups]
        ax.plot([0, 1], means, color=c, lw=2.6, zorder=3,
                marker="o", ms=10, mec="white", mew=1.2)
        # per-variant points, deterministic jitter, behind the mean line
        for g in groups:
            vals = m[m.regime == g].sort_values("variant")[col].values
            off = np.linspace(-0.07, 0.07, len(vals))
            sgn = 1 if col == "rho_ab" else -1
            ax.scatter(gx[g] + sgn * 0.16 + off, vals, s=26, color=c,
                       alpha=0.45, lw=0, zorder=2)
        ax.annotate(name, (1.0, means[1]), xytext=(1.30, means[1]),
                    color=c, fontsize=10, fontweight="bold", va="center")
        # endpoint values; BOLD where this direction dominates (scratch side
        # for precision, pretrained side for recall), muted where it doesn't
        dominant = {"rho_ab": "scratch", "rho_ba": "pretrained"}[col]
        other = {"rho_ab": "rho_ba", "rho_ba": "rho_ab"}[col]
        for g, v in zip(groups, means):
            dom = g == dominant
            upper = v >= m[m.regime == g][other].mean()
            ax.annotate(f"{v:+.2f}", (gx[g], v),
                        textcoords="offset points",
                        xytext=(-30, 12 if upper else -18),
                        ha="center", fontsize=11 if dom else 8.5,
                        fontweight="bold" if dom else "normal",
                        color=c, alpha=1.0 if dom else 0.55)
    ax.axhline(0, color="#9ca3af", lw=0.9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["FROM SCRATCH\n(5 variants)", "PRETRAINED\n(4 variants)"],
                       fontsize=10)
    ax.set_xlim(-0.35, 2.05)
    ax.set_ylim(-0.3, 0.8)
    ax.set_ylabel("within-context Spearman ρ (fit-free)")
    ax.set_title("(a) The governing direction flips with regime", loc="left")
    ax.grid(axis="y")
    ax.grid(False, axis="x")

    # panel b: the direction plane — each variant as a point in
    # (precision signal, recall signal) space; regimes separate across y=x
    for reg, c in [("scratch", BLUE), ("pretrained", RED)]:
        s = m[m.regime == reg]
        ax2.scatter(s.rho_ab, s.rho_ba, s=90, color=c, zorder=3,
                    edgecolor="white", lw=1.2, label=reg)
    lim = (-0.32, 0.85)
    ax2.plot(lim, lim, color="#9ca3af", lw=1, ls="--", zorder=1)
    ax2.fill_between(lim, lim, lim[1], color=RED, alpha=0.05, zorder=0)
    ax2.fill_between(lim, lim[0], lim, color=BLUE, alpha=0.05, zorder=0)
    ax2.annotate("missing support\ngoverns", (0.07, 0.76), fontsize=9.5,
                 color=RED, va="top", fontweight="bold")
    ax2.annotate("off-target mass\ngoverns", (0.55, -0.13), fontsize=9.5,
                 color=BLUE, fontweight="bold")
    ax2.axhline(0, color="#e5e7eb", lw=0.8)
    ax2.axvline(0, color="#e5e7eb", lw=0.8)
    ax2.set_xlim(lim), ax2.set_ylim(lim)
    ax2.set_aspect("equal")
    ax2.set_xlabel("ρ(transfer, −d(T→B))   [off-target-mass signal]")
    ax2.set_ylabel("ρ(transfer, −d(B→T))   [missing-support signal]")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.set_title("(b) Each variant in the direction plane", loc="left")

    fig.tight_layout()
    fig.savefig(OUT / "F2_direction_preference.png", bbox_inches="tight")
    plt.close(fig)


def f3_specificity():
    # flip statistic + permutation p, motion vs DINO — values from the
    # no-Middlebury verification reports (Appendix A/B, 2026-06-10)
    rows = [
        ("mean-NN", 0.899, 0.0286, 0.018, 0.4286),
        ("ε-coverage 4px", 0.652, 0.0286, 0.000, 1.0),
        ("ε-coverage 16px", 0.914, 0.0286, 0.000, 1.0),
        ("kNN-KL (k=20)", 0.298, 0.0286, -0.093, 0.4000),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    y = np.arange(len(rows))[::-1]
    for yi, (name, fm, pm, fd, pdn) in zip(y, rows):
        ax.plot([fd, fm], [yi, yi], color=LGRAY, lw=2, zorder=1)
        ax.scatter([fm], [yi], s=80, color="#1a7f5a", zorder=3)
        ax.scatter([fd], [yi], s=80, color=LGRAY, edgecolor=GRAY, zorder=3)
        ax.annotate(f"p={pm:.3f}".rstrip("0"), (fm, yi), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color="#1a7f5a")
    ax.axvline(0, color="#9ca3af", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_xlabel("regime flip statistic  (scratch − pretrained mean d)")
    ax.scatter([], [], s=80, color="#1a7f5a", label="motion (flow) space")
    ax.scatter([], [], s=80, color=LGRAY, edgecolor=GRAY,
               label="appearance (DINO) space")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("The flip exists only in motion space — every estimator agrees",
                 loc="left")
    ax.grid(axis="x")
    ax.grid(False, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "F3_motion_vs_dino_flip.png", bbox_inches="tight")
    plt.close(fig)


def f4_gap():
    df = pd.read_csv(V5 / "results/pairwise_gap_rule.csv")
    order = ["0-1", "1-2", "2-5", "5-10", ">10"]
    fig, ax = plt.subplots(figsize=(7, 4.3))
    # reproducibility ceiling: how often two independent retrainings agree
    for fam, label, ls, mk in [
        ("same_arch", "retraining, same architecture (ceiling)", "-", "o"),
        ("cross_arch", "retraining, different architecture", "--", "s"),
    ]:
        s = df[(df.measure == "empirical_reproducibility") & (df.family == fam)]
        s = s.set_index("gap_bin").reindex(order)
        ax.plot(order, s.acc, ls=ls, marker=mk, color=GRAY, ms=5.5, lw=1.5,
                label=label, zorder=2)
    # zero-shot source-selection predictors (held-out source, LOTO): our coverage
    # rule vs a direction-blind symmetric distance vs the appearance control
    SPLIT = "LOTO"
    pa = df[(df.measure == "predictor_accuracy") & (df.split == SPLIT)]
    for fam, label, c, mk, lw, z in [
        ("motion_recall", r"motion coverage  $-d_{B\to T}$  (ours)", RED, "o", 2.5, 5),
        ("motion_w2", "symmetric Wasserstein (flow $W_2$)", "#3a8f6a", "D", 1.6, 3),
        ("motion_precision", r"wrong direction: off-target mass  $-d_{T\to B}$", "#e08a1e", "v", 1.8, 4),
        ("appearance", "appearance (DINO) distance", "#8e44ad", "x", 1.8, 4),
    ]:
        s = pa[pa.family == fam].set_index("gap_bin").reindex(order)
        ax.plot(order, s.acc, marker=mk, color=c, ms=6.5, lw=lw, label=label,
                zorder=z)
    ax.axhline(0.5, color="#9ca3af", lw=0.8, ls=":")
    ax.annotate("coin flip", (0.02, 0.505), xycoords=("axes fraction", "data"),
                fontsize=8.5, color=GRAY, va="bottom")
    ax.set_ylim(0.38, 0.93)
    ax.set_xlabel("how different the two candidate training sets really are\n(true peak-PCK difference between them)")
    ax.set_ylabel("chance of picking the truly\nbetter training set")
    ax.set_title("Coverage tracks the retraining ceiling; other distances stall at chance",
                 loc="left")
    ax.legend(fontsize=8.8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "F4_gap_stratified.png", bbox_inches="tight")
    plt.close(fig)


def _calibrated_rows(split, head=None):
    """Best-calibration predictions per split. LOTO/LOBO: pipeline anchor +
    shrink-gain g. JOINT: two-way kernel anchor (joint_anchor.py — pulls
    observed cells through benchmark similarity x training-set similarity
    jointly) + PROFILESIM-calibrated g (the JOINT dispersion head).
    benchsim_policy rows = the regime-aware policy (scratch contexts use the
    summed-distance/sym calibration, pretrained use the recall arm; built by
    make_policy_rows.py); Middlebury-free."""
    if head is None:
        head = "g_profilesim_gain" if split == "JOINT" else "g_shrink_gain"
    rows = pd.read_csv(V5 / f"results/benchsim_policy/rows_{split}_all_variants.csv")
    rows = rows[rows.benchmark != "middlebury"].copy()
    if split == "JOINT":
        j = pd.read_csv(V5 / "results/joint_anchor_v2.csv").rename(
            columns={"src": "train_dataset", "bench": "benchmark"})
        rows = rows.merge(j[["train_dataset", "benchmark", "variant", "L2K"]],
                          on=["train_dataset", "benchmark", "variant"],
                          how="inner")
        rows = rows[np.isfinite(rows.L2K)]
        rows["pred"] = rows.L2K + rows[head]
    else:
        rows["pred"] = rows.L + rows[head]
    # design-defined: pretrained backbone-dependent matchers (all targets)
    # + RAFT from scratch (real-motion only)
    _a = rows.variant.str.split("|")
    _SEM = {"spair", "pfpascal", "pfwillow", "tss"}
    rows = rows[(_a.str[1] == "True")
                | ((_a.str[0] == "raft") & ~rows.benchmark.isin(_SEM))].copy()
    return rows


def f5_absolute(fname="F5_absolute_scatter.png", color_by="hybrid"):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5), sharey=True)
    panels = []
    for split in ["LOTO", "LOBO"]:
        panels.append((split, "anchor + calibrated coverage",
                       _calibrated_rows(split)))
    panels.append(("JOINT", "two-way kernel anchor + calibrated coverage",
                   _calibrated_rows("JOINT")))

    for ax, (split, sub, rows) in zip(axes, panels):
        rows = rows.copy()
        # restrict to the design-defined configurations (each architecture in its
        # intended regime): pretrained backbone-dependent matchers (all targets)
        # + RAFT from scratch (real-motion only)
        _a = rows.variant.str.split("|")
        _pre = _a.str[1] == "True"
        _raft = _a.str[0] == "raft"
        _SEM = {"spair", "pfpascal", "pfwillow", "tss"}
        rows = rows[_pre | (_raft & ~rows.benchmark.isin(_SEM))].copy()
        pre = rows.variant.str.split("|").str[1]
        arch = rows.variant.str.split("|").str[0]
        rows["regime"] = np.where((pre == "False") | (arch == "raft"),
                                  "scratch", "pretrained")
        # "hybrid" (default): dots colored by benchmark to show the LOTO
        # clustering / LOBO striation, with red/blue regime calibration lines
        # on top so the regime story stays legible.
        if color_by == "regime":
            groups = [("scratch", BLUE), ("pretrained", RED)]
            gcol = "regime"
        elif color_by == "trainset":
            keys = sorted(rows.train_dataset.unique())
            groups = [(k, plt.cm.tab20(i % 20)) for i, k in enumerate(keys)]
            gcol = "train_dataset"
        else:  # "hybrid" or "benchmark": dots by benchmark
            keys = sorted(rows.benchmark.unique())
            groups = [(k, plt.cm.tab10(i % 10)) for i, k in enumerate(keys)]
            gcol = "benchmark"
        dot_alpha = 0.25 if color_by == "regime" else 0.5
        for key, c in groups:
            s = rows[rows[gcol] == key]
            ax.scatter(s.pred, s.actual, s=8, alpha=dot_alpha, color=c, lw=0,
                       label=key)
        # per-regime linear fits on top of the benchmark-colored dots
        if color_by != "regime":
            for reg, c in [("scratch", BLUE), ("pretrained", RED)]:
                s = rows[rows.regime == reg]
                if len(s) < 12:
                    continue
                z = np.polyfit(s.pred, s.actual, 1)
                xs = np.linspace(s.pred.quantile(0.02), s.pred.quantile(0.98), 10)
                ax.plot(xs, np.polyval(z, xs), color=c, lw=2.8, zorder=5,
                        label=f"{reg} fit")
        ax.plot([0, 100], [0, 100], color="#9ca3af", lw=1, ls="--", zorder=1)
        r = pearsonr(rows.pred, rows.actual)[0]
        mae = float(np.mean(np.abs(rows.pred - rows.actual)))
        ax.set_title(f"{split}   r = {r:+.2f}   MAE = {mae:.1f}", loc="left")
        ax.annotate(sub, (0.03, 0.965), xycoords="axes fraction", fontsize=8.5,
                    color=GRAY, va="top")
        ax.set_xlabel("predicted peak PCK")
        ax.set_xlim(-3, 103), ax.set_ylim(-3, 103)
        ax.set_aspect("equal")
    axes[0].set_ylabel("actual peak PCK")
    if color_by == "regime":
        axes[0].legend(fontsize=9, loc="lower right", markerscale=2.2)
        fig.tight_layout()
    else:
        # dedup legend (benchmark dots from panel 0 + the two regime lines)
        h, l = axes[0].get_legend_handles_labels()
        seen, hh, ll = set(), [], []
        for handle, lab in zip(h, l):
            if lab not in seen:
                seen.add(lab); hh.append(handle); ll.append(lab)
        fig.legend(hh, ll, loc="center left", bbox_to_anchor=(0.91, 0.5),
                   fontsize=8, markerscale=2.0)
        fig.tight_layout(rect=(0, 0, 0.91, 1))
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)


def f5supp_full_calibrated(head="g_shrink_gain"):
    """F5supp — the FULL calibrated predictor in PCK units:
    pred = L + dispersion-calibrated g (shrink-gain head, the P6 winner;
    pass head='g_profilesim_gain' or 'g_benchsim_gain' to compare).
    Three held-out settings, colored by benchmark. Middlebury rows dropped
    (eval bug). NOTE: rows come from the benchsim_rule run, which predates the
    Middlebury exclusion — L/gains were fit with Middlebury present; flagged
    in the doc until context_scale_calibration is rerun."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    rows0 = pd.read_csv(V5 / "results/benchsim_rule/rows_LOTO_all_variants.csv")
    benches = sorted(b for b in rows0.benchmark.unique() if b != "middlebury")
    bench_c = {b: plt.cm.tab10(i % 10) for i, b in enumerate(benches)}
    for ax, split in zip(axes, ["LOTO", "LOBO", "JOINT"]):
        rows = _calibrated_rows(split, head)
        for b in benches:
            s = rows[rows.benchmark == b]
            ax.scatter(s.pred, s.actual, s=9, alpha=0.5, color=bench_c[b],
                       lw=0, label=b)
        ax.plot([0, 100], [0, 100], color="#9ca3af", lw=1, ls="--", zorder=1)
        r = pearsonr(rows.pred, rows.actual)[0]
        mae = float(np.mean(np.abs(rows.pred - rows.actual)))
        ax.set_title(f"{split}   r = {r:+.2f}   MAE = {mae:.1f}", loc="left")
        ax.set_xlabel("predicted peak PCK  (L + calibrated g)")
        ax.set_xlim(-3, 103), ax.set_ylim(-3, 103)
        ax.set_aspect("equal")
        if split == "JOINT":
            ax.annotate("two-way kernel anchor + profilesim-calibrated g",
                        (0.04, 0.97), xycoords="axes fraction", fontsize=8.5,
                        color=GRAY, va="top")
    axes[0].set_ylabel("actual peak PCK")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="center left",
               bbox_to_anchor=(0.91, 0.5), fontsize=8.5, markerscale=2.2)
    fig.tight_layout(rect=(0, 0, 0.91, 1))
    fig.savefig(OUT / "F5supp_lobo_joint.png", bbox_inches="tight")
    plt.close(fig)


def f5supp_decomposition(head="g_shrink_gain"):
    """F5supp2 — per-context linear fits that decompose the predictor.
    2 rows x 3 splits. TOP row: fit per BENCHMARK -> each is a within-context
    band whose slope is the within-context ranking rho (~0.5): "the rule buys
    order". BOTTOM row: fit per TRAINING SET -> each cuts across benchmark
    levels, slope ~1: "the anchor buys units". Big, supplement-only."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharex=True, sharey=True)
    rows0 = pd.read_csv(V5 / "results/benchsim_rule/rows_LOTO_all_variants.csv")
    benches = sorted(b for b in rows0.benchmark.unique() if b != "middlebury")
    bench_c = {b: plt.cm.tab10(i % 10) for i, b in enumerate(benches)}

    GROUPS = [("benchmark", "fit per BENCHMARK  (within-context ranking)"),
              ("train_dataset", "fit per TRAINING SET  (level / anchor)")]
    for r, (gcol, rowlabel) in enumerate(GROUPS):
        keys = (benches if gcol == "benchmark"
                else sorted(rows0.train_dataset.unique()))
        cmap = (bench_c if gcol == "benchmark"
                else {k: plt.cm.tab20(i % 20) for i, k in enumerate(keys)})
        for cidx, split in enumerate(["LOTO", "LOBO", "JOINT"]):
            ax = axes[r, cidx]
            rows = _calibrated_rows(split, head)
            ax.scatter(rows.pred, rows.actual, s=7, alpha=0.25, color="#c9ccd1",
                       lw=0, zorder=1)
            slopes = []
            for k in keys:
                s = rows[rows[gcol] == k]
                if len(s) < 6 or s.pred.std() < 1:
                    continue
                z = np.polyfit(s.pred, s.actual, 1)
                slopes.append(z[0])
                xs = np.linspace(s.pred.quantile(0.05), s.pred.quantile(0.95), 10)
                ax.plot(xs, np.polyval(z, xs), color=cmap[k], lw=2.0, zorder=3)
            ax.plot([0, 100], [0, 100], color="#9ca3af", lw=1, ls="--", zorder=2)
            ms = float(np.median(slopes)) if slopes else float("nan")
            ax.annotate(f"median slope = {ms:.2f}", (0.04, 0.93),
                        xycoords="axes fraction", fontsize=10, color="#374151")
            if r == 0:
                ax.set_title(split, loc="left", fontsize=12)
            ax.set_xlim(-3, 103), ax.set_ylim(-3, 103), ax.set_aspect("equal")
            if r == 1:
                ax.set_xlabel("predicted peak PCK")
        axes[r, 0].set_ylabel(f"actual peak PCK\n{rowlabel}", fontsize=10)
    fig.suptitle("Decomposing the predictor: per-benchmark fits expose the "
                 "ranking signal (slope $\\approx\\rho$), per-training-set fits "
                 "the level (slope $\\approx 1$)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "F5supp_decomposition.png", bbox_inches="tight")
    plt.close(fig)


def f5grid_color_views():
    """The ranking signal with the level anchor stripped.
    x = the rule score g, y = actual minus anchor, both z-scored within each
    context. 2x3 grid of IDENTICAL panels: columns = the three held-out
    settings; top row colored by benchmark, bottom row colored by training
    set — same data, two color stories."""
    data = {}
    for split in ["LOTO", "LOBO", "JOINT"]:
        rows = pd.read_csv(CORE / f"predictions/peak_pck/rows_{split}_motion_policy.csv")
        # design-defined: pretrained matchers (all targets) + RAFT scratch (real-motion)
        _a = rows.variant.str.split("|"); _SEM = {"spair", "pfpascal", "pfwillow", "tss"}
        rows = rows[(_a.str[1] == "True")
                    | ((_a.str[0] == "raft") & ~rows.benchmark.isin(_SEM))].copy()
        rows["resid"] = rows.actual - rows.L
        for c in ("g", "resid"):
            mu = rows.groupby("context_id")[c].transform("mean")
            sd = rows.groupby("context_id")[c].transform("std").replace(0, np.nan)
            rows[c + "_z"] = (rows[c] - mu) / sd
        data[split] = rows.dropna(subset=["g_z", "resid_z"])

    benches = sorted(data["LOTO"].benchmark.unique())
    sources = sorted(data["LOTO"].train_dataset.unique())
    bench_c = {b: plt.cm.tab10(i % 10) for i, b in enumerate(benches)}
    src_c = {s: plt.cm.tab20(i % 20) for i, s in enumerate(sources)}

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6),
                             sharex=True, sharey=True)
    for col, split in enumerate(["LOTO", "LOBO", "JOINT"]):
        rows = data[split]
        rho = np.mean([spearmanr(g.actual, g.g).statistic
                       for _, g in rows.groupby("context_id")
                       if g.train_dataset.nunique() >= 3 and g.g.std() > 1e-12])
        for r, (key, cmap) in enumerate([("benchmark", bench_c),
                                         ("train_dataset", src_c)]):
            ax = axes[r, col]
            for val, grp in rows.groupby(key):
                ax.scatter(grp.g_z, grp.resid_z, s=8, alpha=0.5,
                           color=cmap[val], lw=0, label=val)
            z = np.polyfit(rows.g_z, rows.resid_z, 1)
            xs = np.linspace(-2.6, 2.6, 10)
            ax.plot(xs, np.polyval(z, xs), color="#333", lw=1.6, ls="--",
                    zorder=3)
            if r == 0:
                ax.set_title(f"{split}   mean within-context ρ = {rho:+.2f}",
                             loc="left")
            ax.set_xlim(-2.8, 2.8), ax.set_ylim(-2.8, 2.8)
            ax.set_aspect("equal")
    for ax in axes[1]:
        ax.set_xlabel("rule score g (z within context)")
    axes[0, 0].set_ylabel("actual − anchor (z within context)\ncolored by BENCHMARK")
    axes[1, 0].set_ylabel("actual − anchor (z within context)\ncolored by TRAINING SET")
    # one legend per row, outside on the right
    h0, l0 = axes[0, 2].get_legend_handles_labels()
    by0 = dict(zip(l0, h0))
    fig.legend(by0.values(), by0.keys(), loc="center left",
               bbox_to_anchor=(0.92, 0.74), fontsize=8, markerscale=2.2)
    h1, l1 = axes[1, 2].get_legend_handles_labels()
    by1 = dict(zip(l1, h1))
    fig.legend(by1.values(), by1.keys(), loc="center left",
               bbox_to_anchor=(0.92, 0.28), fontsize=8, markerscale=2.2)
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    fig.savefig(OUT / "F5grid_color_views.png", bbox_inches="tight")
    plt.close(fig)


def f6_closed_loop():
    pts = [  # target, predicted margin, actual margin, seed noise, is_null
        ("KITTI-2015", 0.93, 0.87, 0.9, False),
        ("Middlebury", 1.03, 0.52, 1.9, False),
        ("FlyingThings\n(null control)", 0.05, -1.58, np.nan, True),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.axhspan(-1.0, 1.0, color="#f1f2f4", zorder=0)
    ax.annotate("typical anchor seed noise", (0.985, -0.93),
                xycoords=("axes fraction", "data"), ha="right", fontsize=8.5,
                color=GRAY)
    lims = [-2.4, 2.4]
    ax.plot(lims, lims, color="#9ca3af", lw=1, ls="--", zorder=1)
    ax.axhline(0, color="#9ca3af", lw=0.8)
    ax.axvline(0, color="#9ca3af", lw=0.8)
    for name, p, a, _, null in pts:
        c = GRAY if null else "#1a7f5a"
        ax.scatter([p], [a], s=110, color=c, zorder=3)
        ax.annotate(name, (p, a), textcoords="offset points", xytext=(10, -4),
                    fontsize=9.5, color=c)
    ax.set_xlabel("PREDICTED margin of tuned generator over anchor (PCK)")
    ax.set_ylabel("ACTUAL margin after training (PCK)")
    ax.set_title("Closed loop: predicted gains realize; the null stays null",
                 loc="left")
    ax.set_xlim(lims), ax.set_ylim(lims)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT / "F6_closed_loop.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    f2_law()
    f3_specificity()
    f4_gap()
    f5_absolute()  # hybrid: benchmark dots + regime lines
    f5_absolute(fname="F5_absolute_by_regime.png", color_by="regime")
    f5supp_full_calibrated()
    f5supp_decomposition()
    f5grid_color_views()
    f6_closed_loop()
    print(f"wrote F2/F3/F4/F5/F5supp/F5grid/F6 (v3) -> {OUT}")
