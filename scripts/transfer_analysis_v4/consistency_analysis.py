#!/usr/bin/env python
"""
Consistency analyses for the "motion >> appearance, but ceiling-limited" story.

Three FREE analyses (no new training) that reconcile the apparent inconsistencies
in the interventional grid into one model:

    perf(source, benchmark, variant)
        = benchmark_level                       (context level)
        + source_quality(source)                (target-AGNOSTIC main effect)
        + motion_match(source, benchmark)        (target-DEPENDENT interaction)
        + noise

ANALYSIS 1 -- crossover double-dissociation (same-generator kubric pair).
    Each recovered motion wins on the benchmark whose REGIME it was recovered
    from (camera->K2012 pure ego-motion, object->FlyingThings pure object),
    ties on mixed (K2015) / neither (Middlebury). Replicates in sign across two
    independent training configs (FF and TT). Numbers are from the Master Table.

ANALYSIS 2 -- double demeaning (the formal version of the model).
    Single demean = remove the context (benchmark|variant) mean -> the usual
    within-context residual (this is what rho_g correlates against).
    Double demean = ALSO remove the per-source main effect (two-way / fixed
    effects within each variant) -> the pure source x benchmark INTERACTION.
    Prediction: MOTION distance survives double demeaning (the interaction is
    real); APPEARANCE distance collapses to ~0 (its apparent signal was a source
    main effect: "good sources happen to look a certain way", not "closer in
    appearance -> better on THIS target").

ANALYSIS 3 -- win-rate vs gap curve (the quantitative "as strong as the ceiling").
    Over all within-context source pairs, P(motion distance picks the better
    source) as a function of the true PCK gap, overlaid with the same curve for
    an independent training run (cross-architecture replicate = the ceiling).
    If the motion curve tracks the ceiling curve across gaps, motion recovers
    essentially all the orderable signal at every resolution -- not just on
    average.

Outputs:
    figures -> /home/spencer/Documents/Obsidian/Correspondence/Attachments/consist_*.png
    tables  -> ./consistency_analysis_out/*.csv
Run:  conda activate cuda; python consistency_analysis.py
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

V3 = Path(__file__).resolve().parent.parent / "transfer_analysis_v3"
TABLE = V3 / "transfer_table.csv"
ATTACH = Path("/home/spencer/Documents/Obsidian/Correspondence/Attachments")
OUT = Path(__file__).resolve().parent / "consistency_analysis_out"
OUT.mkdir(exist_ok=True)

PURE = ["flyingthings", "imagenet2dwarp", "movi_f", "pointodyssey", "sintel",
        "spair", "synthetic", "synthetic_2d_warp", "synthetic_large_zoom",
        "synthetic_random_flipping", "synthetic_small_zoom"]
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------
def load() -> pd.DataFrame:
    df = pd.read_csv(TABLE)
    df = df[df.train_dataset.isin(PURE)].copy()
    df["variant"] = (df.model_family + "|" + df.pretrained.astype(str) + "|"
                     + df.freeze.astype(str))
    df["mot"] = df[["flow_mean_nn_eval_to_train_k1",
                    "flow_mean_nn_train_to_eval_k1"]].mean(1)
    df["app"] = df[["dino_mean_nn_eval_to_train_k1",
                    "dino_mean_nn_train_to_eval_k1"]].mean(1)
    df["y"] = df["peak_pck"]
    # drop variants with grossly incomplete coverage (raft|False|False = 10 rows)
    keep = df.variant.value_counts()
    keep = keep[keep >= 90].index
    df = df[df.variant.isin(keep)].copy()
    return df


# --------------------------------------------------------------------------
def two_way_resid(sub: pd.DataFrame, col: str) -> pd.Series:
    """Per-variant additive two-way (source + benchmark) demeaning.
    Returns the source x benchmark interaction residual:
        r = y - ybar_source - ybar_bench + ybar_grand   (within each variant)."""
    out = pd.Series(np.nan, index=sub.index)
    for v, g in sub.groupby("variant"):
        m = g.pivot_table(index="train_dataset", columns="benchmark", values=col)
        grand = np.nanmean(m.values)
        rmean = m.mean(1)            # source main effect (over benchmarks)
        cmean = m.mean(0)            # benchmark main effect (over sources)
        resid = m.sub(rmean, axis=0).sub(cmean, axis=1) + grand
        # map back to row index
        for idx, r in g.iterrows():
            out[idx] = resid.loc[r.train_dataset, r.benchmark]
    return out


def context_resid(sub: pd.DataFrame, col: str) -> pd.Series:
    """Single demean: subtract context (benchmark|variant) mean only."""
    return sub[col] - sub.groupby(["variant", "benchmark"])[col].transform("mean")


def analysis2(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("ANALYSIS 2 -- single vs double demeaning")
    print("=" * 70)
    # matched grid: rows where BOTH motion & appearance exist (drops 'synthetic'
    # benchmark which has no DINO mean_nn), pure sources.
    sub = df[df.mot.notna() & df.app.notna() & df.y.notna()].copy()
    # keep only (variant) cells that are full-ish grids
    print(f"matched rows: {len(sub)}  benchmarks: {sorted(sub.benchmark.unique())}")
    print(f"variants: {sorted(sub.variant.unique())}")

    res = {}
    for col in ["mot", "app", "y"]:
        sub[f"{col}_ctx"] = context_resid(sub, col)
        sub[f"{col}_tw"] = two_way_resid(sub, col)

    def srho(a, b):
        m = sub[a].notna() & sub[b].notna()
        return spearmanr(sub.loc[m, a], sub.loc[m, b]).correlation

    rows = []
    for fam, fcol in [("motion", "mot"), ("appearance", "app")]:
        single = srho(f"{fcol}_ctx", "y_ctx")
        double = srho(f"{fcol}_tw", "y_tw")
        rows.append(dict(family=fam, single_demean_rho=single, double_demean_rho=double))
        print(f"  {fam:11s}  single(context)={single:+.3f}   double(interaction)={double:+.3f}")

    # cluster bootstrap over the 11 sources for double-demean CI
    print("\n  cluster bootstrap (resample sources, 1000x) on DOUBLE-demean rho:")
    srcs = sub.train_dataset.unique()
    boot = {"motion": [], "appearance": []}
    for _ in range(1000):
        pick = RNG.choice(srcs, size=len(srcs), replace=True)
        bs = pd.concat([sub[sub.train_dataset == s] for s in pick], ignore_index=True)
        for col in ["mot", "app", "y"]:
            bs[f"{col}_tw"] = two_way_resid(bs, col)
        for fam, fcol in [("motion", "mot"), ("appearance", "app")]:
            m = bs[f"{fcol}_tw"].notna() & bs["y_tw"].notna()
            if m.sum() > 10:
                boot[fam].append(spearmanr(bs.loc[m, f"{fcol}_tw"], bs.loc[m, "y_tw"]).correlation)
    for fam in ["motion", "appearance"]:
        arr = np.array(boot[fam])
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        pneg = float(np.mean(arr < 0))
        for r in rows:
            if r["family"] == fam:
                r["double_ci_lo"], r["double_ci_hi"], r["double_P_negative"] = lo, hi, pneg
        print(f"    {fam:11s}  double rho 95% CI [{lo:+.3f}, {hi:+.3f}]   P(rho<0)={pneg:.3f}")

    pd.DataFrame(rows).to_csv(OUT / "analysis2_demean.csv", index=False)

    # ---- figure: 2x2 scatter (single vs double) x (motion vs appearance) ----
    fig, ax = plt.subplots(2, 2, figsize=(10, 9))
    panels = [
        ("mot_ctx", "y_ctx", "MOTION -- single demean\n(within-context residual)", "tab:blue"),
        ("app_ctx", "y_ctx", "APPEARANCE -- single demean", "tab:orange"),
        ("mot_tw", "y_tw", "MOTION -- double demean\n(pure source x benchmark interaction)", "tab:blue"),
        ("app_tw", "y_tw", "APPEARANCE -- double demean", "tab:orange"),
    ]
    for a, (xc, yc, title, color) in zip(ax.ravel(), panels):
        m = sub[xc].notna() & sub[yc].notna()
        x, y = sub.loc[m, xc], sub.loc[m, yc]
        rho = spearmanr(x, y).correlation
        a.scatter(x, y, s=14, alpha=0.5, color=color, edgecolor="none")
        a.axhline(0, lw=.6, color="k", alpha=.3); a.axvline(0, lw=.6, color="k", alpha=.3)
        a.set_title(f"{title}\nSpearman rho = {rho:+.3f}", fontsize=10)
        a.set_xlabel("distance residual (closer = left)")
        a.set_ylabel("PCK residual")
    fig.suptitle("Single (context) vs double (two-way) demeaning\n"
                 "double demean removes the per-source main effect -> pure interaction",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(ATTACH / "consist_double_demean.png", dpi=130)
    plt.close(fig)
    print(f"  -> {ATTACH/'consist_double_demean.png'}")
    return rows


# --------------------------------------------------------------------------
def analysis3(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("ANALYSIS 3 -- win-rate vs gap (ceiling-limited at every resolution)")
    print("=" * 70)
    sub = df[df.mot.notna() & df.app.notna() & df.y.notna()].copy()

    BINS = [0, 1, 2, 4, 8, 16, 32, 200]
    BLAB = ["0-1", "1-2", "2-4", "4-8", "8-16", "16-32", "32+"]

    # predictor curves: motion / appearance distance vs each variant's ordering
    pred_hits = {"motion": {}, "appearance": {}}
    for (v, b), g in sub.groupby(["variant", "benchmark"]):
        g = g.reset_index(drop=True)
        n = len(g)
        for i in range(n):
            for j in range(i + 1, n):
                gap = abs(g.y[i] - g.y[j])
                if g.y[i] == g.y[j]:
                    continue
                truth = g.y[i] > g.y[j]
                for fam, col in [("motion", "mot"), ("appearance", "app")]:
                    if g[col][i] == g[col][j]:
                        continue
                    pred = g[col][i] < g[col][j]   # smaller distance = predicted better
                    bk = np.digitize(gap, BINS) - 1
                    pred_hits[fam].setdefault(bk, []).append(int(pred == truth))

    # ceiling curve: cross-architecture replicate agreement
    ceil_hits = {}
    sub["arch"] = sub.variant.str.split("|").str[0]
    # build (source,benchmark)->{variant:pck}
    piv = sub.pivot_table(index=["train_dataset", "benchmark"],
                          columns="variant", values="y")
    variants = list(piv.columns)
    arch_of = {v: v.split("|")[0] for v in variants}
    for b, gb in sub.groupby("benchmark"):
        srcs = sorted(gb.train_dataset.unique())
        for vi, vA in enumerate(variants):
            for vB in variants:
                if vA == vB or arch_of[vA] == arch_of[vB]:
                    continue  # cross-architecture only = independent observer
                for a_i in range(len(srcs)):
                    for a_j in range(a_i + 1, len(srcs)):
                        s1, s2 = srcs[a_i], srcs[a_j]
                        try:
                            yA1, yA2 = piv.loc[(s1, b), vA], piv.loc[(s2, b), vA]
                            yB1, yB2 = piv.loc[(s1, b), vB], piv.loc[(s2, b), vB]
                        except KeyError:
                            continue
                        if any(pd.isna(x) for x in [yA1, yA2, yB1, yB2]):
                            continue
                        if yA1 == yA2 or yB1 == yB2:
                            continue
                        gap = abs(yA1 - yA2)       # truth-side gap (variant A)
                        truth = yA1 > yA2
                        pred = yB1 > yB2           # other run's ordering
                        bk = np.digitize(gap, BINS) - 1
                        ceil_hits.setdefault(bk, []).append(int(pred == truth))

    rows = []
    for bk in range(len(BLAB)):
        r = {"gap_bin": BLAB[bk]}
        for fam in ["motion", "appearance"]:
            h = pred_hits[fam].get(bk, [])
            r[f"{fam}_winrate"] = np.mean(h) if h else np.nan
            r[f"{fam}_n"] = len(h)
        h = ceil_hits.get(bk, [])
        r["ceiling_winrate"] = np.mean(h) if h else np.nan
        r["ceiling_n"] = len(h)
        rows.append(r)
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))
    tbl.to_csv(OUT / "analysis3_winrate.csv", index=False)

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(BLAB))
    ax.plot(x, tbl.ceiling_winrate, "-o", color="black", lw=2,
            label="ceiling: independent run (cross-arch)")
    ax.plot(x, tbl.motion_winrate, "-o", color="tab:blue", lw=2,
            label="motion distance")
    ax.plot(x, tbl.appearance_winrate, "-o", color="tab:orange", lw=2,
            label="appearance distance")
    ax.axhline(0.5, ls="--", color="gray", label="chance")
    for i, r in tbl.iterrows():
        if not np.isnan(r.motion_winrate):
            ax.annotate(f"n={int(r.motion_n)}", (i, r.motion_winrate),
                        fontsize=7, ha="center", va="bottom", color="tab:blue")
    ax.set_xticks(x); ax.set_xticklabels(BLAB)
    ax.set_xlabel("true PCK gap between the two sources")
    ax.set_ylabel("P(predictor picks the better source)")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Motion distance is ceiling-limited at every resolution\n"
                 "(tracks an independent training run; appearance ~ chance)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(ATTACH / "consist_winrate_vs_gap.png", dpi=130)
    plt.close(fig)
    print(f"  -> {ATTACH/'consist_winrate_vs_gap.png'}")
    return tbl


# --------------------------------------------------------------------------
def analysis1():
    print("\n" + "=" * 70)
    print("ANALYSIS 1 -- crossover double-dissociation (same-generator pair)")
    print("=" * 70)
    # From the Master Table. Same generator, gso assets, hq appearance.
    # camera-dominant = kitti_recovered_gso_hq ; object-dominant = kitti_badmotion_ft_gso_hq
    data = {
        "FF": {  # from-scratch (pretrained=False, freeze=False)
            "camera": dict(kitti2012=96.48, kitti2015=83.16, flyingthings=48.93, middlebury=52.92),
            "object": dict(kitti2012=89.49, kitti2015=84.20, flyingthings=54.39, middlebury=53.12),
        },
        "TT": {  # pretrained + frozen
            "camera": dict(kitti2012=98.46, kitti2015=96.58, flyingthings=73.49, middlebury=57.05),
            "object": dict(kitti2012=95.79, kitti2015=94.26, flyingthings=77.14, middlebury=56.23),
        },
    }
    benches = ["kitti2012", "kitti2015", "flyingthings", "middlebury"]
    blab = ["K2012\n(pure ego)", "K2015\n(mixed)", "FlyingThings\n(pure object)", "Middlebury\n(lateral)"]
    regime = ["camera", "mixed", "object", "lateral"]

    print("  Δ = camera-dominant minus object-dominant (same generator/assets/appearance):")
    for cfg in ["FF", "TT"]:
        ds = [data[cfg]["camera"][b] - data[cfg]["object"][b] for b in benches]
        print(f"   {cfg}: " + "  ".join(f"{b.split(chr(10))[0]}={d:+.2f}" for b, d in zip(blab, ds)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, cfg in zip(axes, ["FF", "TT"]):
        x = np.arange(len(benches)); w = 0.38
        cam = [data[cfg]["camera"][b] for b in benches]
        obj = [data[cfg]["object"][b] for b in benches]
        ax.bar(x - w/2, cam, w, label="camera-dominant motion", color="tab:blue")
        ax.bar(x + w/2, obj, w, label="object-dominant motion", color="tab:red")
        for xi, (c, o) in enumerate(zip(cam, obj)):
            win = "cam" if c > o else "obj"
            d = c - o
            ax.annotate(f"{'cam' if d>0 else 'obj'} {d:+.1f}", (xi, max(c, o)),
                        ha="center", va="bottom", fontsize=8,
                        color="tab:blue" if d > 0 else "tab:red")
        ax.set_xticks(x); ax.set_xticklabels(blab, fontsize=8)
        ax.set_title(f"{cfg}  ({'from-scratch' if cfg=='FF' else 'pretrained+frozen'})")
        ax.set_ylabel("PCK@0.05" if cfg == "FF" else "")
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Crossover double-dissociation: each recovered motion wins on its own regime\n"
                 "(same Kubric generator, GSO assets, HQ appearance -- only motion theta differs; sign replicates FF & TT)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(ATTACH / "consist_crossover.png", dpi=130)
    plt.close(fig)
    print(f"  -> {ATTACH/'consist_crossover.png'}")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    df = load()
    print(f"loaded {len(df)} pure rows, {df.variant.nunique()} variants, "
          f"{df.train_dataset.nunique()} sources, {df.benchmark.nunique()} benchmarks")
    analysis1()
    r2 = analysis2(df)
    r3 = analysis3(df)
    print("\nDONE. Figures in", ATTACH, " tables in", OUT)
