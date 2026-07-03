"""RECOMMENDED ladder-overlap figure, single panel, both KITTI splits overlaid.

Same three quantities as protoA4 (coverage -d_{B->T}, transfer PCK, off-target
-d_{T->B}), now for BOTH KITTI splits on one plot, distinguished by SHADE
(dark = KITTI-2012, light = KITTI-2015). Joint 4-D mean-NN throughout (qnorm retired).

Story in one view: coverage tracks PCK (both peak together), and the optimum SHIFTS
right for the larger-motion target -- KITTI-2012 (14.5px) peaks at 1x, KITTI-2015
(19px) at 1.5x -- while off-target stays flat for both.

Output: ACCV_2026/figures/proto/protoA6_overlap_combined.png
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path("ACCV_2026/figures/proto"); OUT.mkdir(parents=True, exist_ok=True)
RUNGS = ["m025", "m050", "m100", "m150", "m200"]
XL = ["0.25x", "0.5x", "1x", "1.5x", "2x"]
x = np.arange(5)
# (bench, label, mean px, coverage shade, pck shade, offtarget shade, peak marker color)
TARGETS = [
    ("kitti2012", "KITTI-2012 (14.5px)", "#15406b", "#4a2570", "#8a1010"),
    ("kitti2015", "KITTI-2015 (19px)",   "#6aaed6", "#b08fd0", "#e8736b"),
]

cv = pd.read_csv("analysis/coverage_v2_flow_ladder.csv")
cv["rung"] = cv.train_dataset.astype(str).str.extract(r"(m\d+)")[0]
m = pd.read_csv("analysis/ladder_master_table.csv")

fig, ax = plt.subplots(figsize=(9.6, 6.4))
ax2 = ax.twinx()                                  # PCK
ax3 = ax.twinx(); ax3.spines["right"].set_position(("outward", 52))  # off-target
off_means = []
for bench, lab, cCov, cPck, cOff in TARGETS:
    g = cv[cv.eval_dataset.astype(str) == bench].groupby("rung").agg(
        dBT=("mean_nn_eval_to_train_k1", "mean"),
        dTB=("mean_nn_train_to_eval_k1", "mean")).reindex(RUNGS)
    dBT, dTB = g.dBT.values, g.dTB.values
    pck = m[m.reg == "TF"].pivot_table(index="rung", columns="app", values=bench,
                                       aggfunc="mean").reindex(RUNGS).mean(axis=1).values
    peak = int(np.argmin(dBT))
    off_means.append(dTB.mean())
    rc = spearmanr(-dBT, pck).correlation
    ax.plot(x, -dBT, "o-", color=cCov, lw=2.6, ms=8, zorder=3)
    ax2.plot(x, pck, "^--", color=cPck, lw=2.2, ms=8, zorder=3)
    ax3.plot(x, -dTB, "s:", color=cOff, lw=1.6, ms=6, alpha=0.85, zorder=2)
    # mark this target's optimum
    ax.axvline(peak, color=cCov, ls="--", lw=1.4, alpha=0.7, zorder=1)
    ax.text(peak, 0.96, f"opt {XL[peak]}", transform=ax.get_xaxis_transform(),
            color=cCov, fontsize=9, ha="center", va="top", weight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=cCov, alpha=0.8))
    print(f"{lab}: coverage peak {XL[peak]}, cov-PCK rho={rc:+.2f}")

ax.set_xticks(x); ax.set_xticklabels(XL)
ax.set_xlabel("source motion magnitude (x KITTI's)")
ax.set_ylabel("joint coverage  ($-d_{B\\to T}$, mean-NN)", color="#2b6cb0")
ax.tick_params(axis="y", labelcolor="#2b6cb0"); ax.margins(y=0.26)
ax2.set_ylabel("transfer peak PCK", color="#6a3d9a"); ax2.tick_params(axis="y", labelcolor="#6a3d9a")
ax3.set_ylabel("off-target  ($-d_{T\\to B}$)", color="#c0392b"); ax3.tick_params(axis="y", labelcolor="#c0392b")
ax3.set_ylim(-max(off_means) * 1.5, -min(off_means) * 0.5)   # generous -> flatness honest

# two-part legend: quantity (linestyle) + target (shade)
qleg = [Line2D([0], [0], color="#555", marker="o", ls="-", label="coverage  $-d_{B\\to T}$"),
        Line2D([0], [0], color="#555", marker="^", ls="--", label="transfer PCK"),
        Line2D([0], [0], color="#555", marker="s", ls=":", label="off-target  $-d_{T\\to B}$ (flat)")]
tleg = [Line2D([0], [0], color="#15406b", lw=3, label="KITTI-2012 (dark)"),
        Line2D([0], [0], color="#6aaed6", lw=3, label="KITTI-2015 (light)")]
l1 = ax.legend(handles=qleg, fontsize=9, loc="lower left", title="quantity (line style)")
ax.add_artist(l1)
ax.legend(handles=tleg, fontsize=9, loc="lower right", title="target (shade)")

ax.set_title("Overlap peaks with transfer, and the optimum shifts with the target's magnitude\n"
             "KITTI-2012 (smaller motion) peaks at 1x; KITTI-2015 (larger) at 1.5x; off-target flat for both",
             fontsize=11.5)
fig.tight_layout()
fig.savefig(OUT / "protoA6_overlap_combined.png", dpi=160, bbox_inches="tight")
print("wrote", OUT / "protoA6_overlap_combined.png")
