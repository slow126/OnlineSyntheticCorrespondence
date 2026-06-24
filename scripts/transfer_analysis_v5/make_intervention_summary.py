"""F7 — one digestible figure for the interventional studies (Act 2).

Four panels, replacing a wall of rank tables:
  (a) the pre-registered out-of-sample test: per benchmark, how well each
      direction ranks the from-scratch grid (matched direction should win,
      mismatched should fail, symmetric should hedge)
  (b) rule rank vs actual rank across the whole from-scratch grid
  (c) the trial19 story: same sources trained from scratch vs pretrained on
      KITTI-2015 — the winner changes with the regime
  (d) the appearance control: hq -> matte twins (identical motion), transfer
      barely moves while appearance distance moves enormously

Middlebury excluded (eval bug, 2026-06-10).

    python scripts/transfer_analysis_v5/make_intervention_summary.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

GRID = Path("/mnt/nvme_1tb_a/snapshots/transfer_grid")
LEWM = Path("/home/spencer/Projects/le-wm/outputs")
OUT = Path("scripts/transfer_analysis_v5/results/figures")

BLUE = "#2b6cb0"   # d(T->B) / scratch
RED = "#c0392b"    # d(B->T) / pretrained
GRAY = "#6b7280"
LGRAY = "#c4c8cf"
BENCH_LABEL = {"flyingthings": "FlyingThings", "kitti2012": "KITTI-2012",
               "kitti2015": "KITTI-2015"}

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e5e7eb", "grid.linewidth": 0.7,
    "legend.frameon": False,
})


EXPECTED_EPOCHS = 50  # grid horizon; runs with fewer logged epochs are still
                      # training — skip them so partial CSVs don't leak in.


def load():
    rows = []
    for d in sorted(GRID.iterdir()):
        f = d / "validation_results.csv"
        if not f.exists():
            continue
        v = pd.read_csv(f)
        if v["epoch"].nunique() < EXPECTED_EPOCHS:
            continue  # in-progress retrain; exclude until it completes
        src = d.name.rsplit("_pt", 1)[0]
        arm = "FF" if "_pt0_fz0" in d.name else "TT"
        for b, g in v.groupby("benchmark"):
            rows.append((src, arm, b, float(g["pck"].max())))
    # trial19 TT now harvested from the grid (trial19_pt1_fz1_harvested symlink);
    # former hardcoded ("trial19","TT","kitti2015",96.1158) removed 2026-06-10.
    pck = pd.DataFrame(rows, columns=["source", "arm", "benchmark", "peak_pck"])
    pck = pck[pck.benchmark != "middlebury"]  # eval bug
    dist = pd.read_csv(LEWM / "intervention_motion_distances_directional.csv")
    return pck.merge(dist, on=["source", "benchmark"], how="left")


def main():
    m = load()
    ff = m[(m.arm == "FF") & m.flow_mean_nn_a_to_b.notna()].copy()
    benches = ["flyingthings", "kitti2012", "kitti2015"]

    # Two panels, both genuinely supporting the actionability / specificity
    # story: (a) the regime-relative best-dataset flip (trial19), (b) the
    # appearance-stripped twins. The earlier per-benchmark OOS-direction bars
    # and the pooled rank scatter were removed: that result is one clean cell
    # (FlyingThings) and lives in Table 6, not a figure (the pooled view
    # blended the discriminative cell with confounded KITTI cells).
    fig, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(12.5, 4.7))

    # ---- (a) trial19: the regime flip on KITTI-2015 ------------------------
    k = m[m.benchmark == "kitti2015"]
    # only grid sources with a motion distance belong on this panel; the SDF
    # closed-loop source (synthetic_fractal_trial76) has none and is reported
    # separately, so exclude it rather than render a pr/re #0 glitch.
    tt_sources = k[(k.arm == "TT")
                   & k.flow_mean_nn_a_to_b.notna()].source.unique()
    nice = {"trial19": "trial19 (frozen dolly)",
            "kitti_recovered_gso_hq": "kitti gso hq",
            "kitti_recovered_gso_matte": "kitti gso matte",
            "kitti_badmotion_ft_gso_hq": "badmotion gso hq"}
    pts = []
    for src in tt_sources:
        f0 = k[(k.arm == "FF") & (k.source == src)].peak_pck
        t0 = k[(k.arm == "TT") & (k.source == src)].peak_pck
        if f0.empty or t0.empty:
            continue
        pts.append((src, f0.iloc[0], t0.iloc[0]))
    for src, f0, t0 in pts:
        hot = src == "trial19"
        ax_c.plot([0, 1], [f0, t0], color=BLUE if hot else LGRAY,
                  lw=2.6 if hot else 1.6, marker="o",
                  ms=7 if hot else 5, zorder=3 if hot else 2)
    # each source's directional profile within the plotted family:
    # pr rank = rank of d(T->B) (1 = most on-target mass)
    # re rank = rank of d(B->T) (1 = best target coverage)
    prof = (k[(k.arm == "TT")][["source", "flow_mean_nn_a_to_b",
                                "flow_mean_nn_b_to_a"]]
            .dropna().set_index("source"))
    pr_rank = prof.flow_mean_nn_a_to_b.rank().astype(int)
    re_rank = prof.flow_mean_nn_b_to_a.rank().astype(int)
    # dodge the right-hand labels: evenly spaced stack centered on the cluster
    pts.sort(key=lambda r: r[2])
    gap = 3.2
    center = np.mean([t0 for _, _, t0 in pts])
    start = center - (len(pts) - 1) * gap / 2
    ys = [start + i * gap for i in range(len(pts))]
    ax_c.set_ylim(min(f0 for _, f0, _ in pts) - 3, max(ys) + 3)
    for (src, _, t0), yl in zip(pts, ys):
        hot = src == "trial19"
        ax_c.annotate(nice.get(src, src), (1.21, yl), fontsize=8.5,
                      va="center", color=BLUE if hot else GRAY,
                      fontweight="bold" if hot else "normal")
        # profile tuple below the name; the source's stronger metric in bold
        pr, re = int(pr_rank.get(src, 0)), int(re_rank.get(src, 0))
        pr_dom = pr <= re
        ax_c.annotate(f"pr #{pr}", (1.21, yl - 1.45), fontsize=8, va="center",
                      color=BLUE, alpha=1.0 if pr_dom else 0.45,
                      fontweight="bold" if pr_dom else "normal")
        ax_c.annotate(f"re #{re}", (1.40, yl - 1.45), fontsize=8, va="center",
                      color=RED, alpha=1.0 if not pr_dom else 0.45,
                      fontweight="bold" if not pr_dom else "normal")
        ax_c.plot([1.02, 1.19], [t0, yl], color=LGRAY, lw=0.7, zorder=1)
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(["FROM SCRATCH", "PRETRAINED"], fontsize=10)
    ax_c.set_xlim(-0.15, 1.75)
    ax_c.set_ylabel("KITTI-2015 peak PCK")
    ax_c.set_title("(a) \"Best dataset\" is regime-relative: trial19 wins\n"
                   "from scratch, loses its edge once pretrained", loc="left")
    ax_c.grid(axis="y"), ax_c.grid(False, axis="x")

    # ---- (d) appearance swap at identical motion ---------------------------
    pairs = [("ft_recovered", "FF"), ("kitti_recovered", "FF"),
             ("kitti_recovered_gso", "FF"), ("kitti_recovered_gso", "TT")]
    k15 = m[m.benchmark == "kitti2015"]
    y = np.arange(len(pairs))[::-1]
    for yi, (stem, arm) in zip(y, pairs):
        hq = k15[(k15.source == f"{stem}_hq") & (k15.arm == arm)].peak_pck
        mt = k15[(k15.source == f"{stem}_matte") & (k15.arm == arm)].peak_pck
        if hq.empty or mt.empty:
            continue
        hq, mt = hq.iloc[0], mt.iloc[0]
        ax_d.plot([mt, hq], [yi, yi], color=LGRAY, lw=2, zorder=1)
        ax_d.scatter([hq], [yi], s=70, color="#1a7f5a", zorder=3)
        ax_d.scatter([mt], [yi], s=70, color=LGRAY, edgecolor=GRAY, zorder=3)
        ax_d.annotate(f"Δ {mt - hq:+.1f}", ((hq + mt) / 2, yi + 0.22),
                      ha="center", fontsize=8.5, color=GRAY)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels([f"{s} ({a})" for s, a in pairs], fontsize=9)
    ax_d.scatter([], [], s=70, color="#1a7f5a", label="full appearance (hq)")
    ax_d.scatter([], [], s=70, color=LGRAY, edgecolor=GRAY,
                 label="matte (appearance stripped)")
    ax_d.legend(fontsize=8.5, loc="lower left")
    ax_d.set_xlabel("KITTI-2015 peak PCK")
    ax_d.set_title("(b) Strip the appearance, keep the motion:\n"
                   "transfer barely moves (DINO distance moves ~70%)",
                   loc="left")
    ax_d.grid(axis="x"), ax_d.grid(False, axis="y")

    fig.tight_layout()
    fig.savefig(OUT / "F7_interventions_summary.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'F7_interventions_summary.png'}")


if __name__ == "__main__":
    main()
