"""Figure-8 LEFT-PANEL replacement: joint mean-NN coverage/off-target, stacked by KITTI split.

Replaces the retired qnorm support-overlap left panel of make_ladder_fig.py with the
headline joint 4-D mean-NN metric. Two stacked sub-panels (KITTI-2012 top, KITTI-2015
bottom); coverage (-d_{B->T}, blue) is the inverted-U whose peak tracks the target's
magnitude, off-target (-d_{T->B}, red) is flat. PCK is NOT shown here -- it is the two
heatmaps to the right in Figure 8.

Output: ACCV_2026/figures/proto/protoA7_ladder_left.png
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("ACCV_2026/figures/proto"); OUT.mkdir(parents=True, exist_ok=True)
RUNGS = ["m025", "m050", "m100", "m150", "m200"]
XL = ["0.25x", "0.5x", "1x", "1.5x", "2x"]
x = np.arange(5)
BLUE, RED, GREEN = "#2b6cb0", "#c0392b", "#2a8f3f"
TARGETS = [("kitti2012", "KITTI-2012  (mean 14.5 px)"),
           ("kitti2015", "KITTI-2015  (mean 19 px)")]

cv = pd.read_csv("analysis/coverage_v2_flow_ladder.csv")
cv["rung"] = cv.train_dataset.astype(str).str.extract(r"(m\d+)")[0]

fig, axes = plt.subplots(2, 1, figsize=(5.6, 7.6), sharex=True)
for ax, (bench, lab) in zip(axes, TARGETS):
    g = cv[cv.eval_dataset.astype(str) == bench].groupby("rung").agg(
        dBT=("mean_nn_eval_to_train_k1", "mean"),
        dTB=("mean_nn_train_to_eval_k1", "mean")).reindex(RUNGS)
    dBT, dTB = g.dBT.values, g.dTB.values
    peak = int(np.argmin(dBT))

    # under / over-shoot shading relative to this target's optimum
    ax.axvspan(-0.4, peak - 0.5, color="#fdecea", zorder=0)          # under-coverage
    ax.axvspan(peak + 0.5, 4.4, color="#fff4e6", zorder=0)           # over-shoot
    ax.annotate("under", (-0.35, 0.04), xycoords=("data", "axes fraction"), fontsize=8.5, color="#a33")
    ax.annotate("over-shoot", (peak + 0.6, 0.04), xycoords=("data", "axes fraction"), fontsize=8.5, color="#b5670f")

    # coverage (left axis) -- the inverted-U
    ax.plot(x, -dBT, "o-", color=BLUE, lw=2.6, ms=8, zorder=3)
    ax.set_ylabel("coverage  $-d_{B\\to T}$", color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax.margins(y=0.22)
    ax.set_title(lab, fontsize=11, loc="left")

    # off-target (right axis) -- flat, generous +/-50% window so the flatness is honest
    axr = ax.twinx()
    axr.plot(x, -dTB, "s:", color=RED, lw=1.7, ms=6, alpha=0.9, zorder=2)
    axr.set_ylim(-dTB.mean() * 1.5, -dTB.mean() * 0.5)
    axr.set_ylabel("off-target  $-d_{T\\to B}$", color=RED)
    axr.tick_params(axis="y", labelcolor=RED)

axes[1].set_xticks(x); axes[1].set_xticklabels(XL)
axes[1].set_xlabel("source motion magnitude (x KITTI's mean flow)")
# one shared legend along the bottom, out of the panels
from matplotlib.lines import Line2D
fig.legend([Line2D([0], [0], color=BLUE, marker="o", lw=2.4),
            Line2D([0], [0], color=RED, marker="s", ls=":", lw=1.7)],
           ["coverage (target covered by source)", "off-target (source outside target)"],
           fontsize=8.4, loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Motion overlap vs magnitude\n(joint mean-NN; coverage peaks at the match, off-target flat)",
             fontsize=11.5)
fig.tight_layout(rect=[0, 0.06, 1, 0.95])
fig.savefig(OUT / "protoA7_ladder_left.png", dpi=170, bbox_inches="tight")
print("wrote", OUT / "protoA7_ladder_left.png")
