"""RECOMMENDED ladder-overlap figure: the two KITTI splits kept SEPARATE.

KITTI-2012 (mean ~14.5px) and KITTI-2015 (mean ~19px) have different motion
magnitudes, so their coverage/transfer optima sit at different rungs. Pooling them
averages two inverted-Us with offset peaks and smears the signal; split, the result
is stronger: the optimum TRACKS each target's magnitude (2012->1x, 2015->1.5x), and
joint-space coverage predicts the shift at rho=1.0 in BOTH.

All distances are joint 4-D mean-NN (the headline metric); qnorm is retired.

Each panel: coverage -d_{B->T} (blue) + transfer PCK (purple) co-peak at the marked
rung; off-target -d_{T->B} (red, generous 3rd axis) is flat -> over-shoot = lost
coverage, not off-target.

Output: ACCV_2026/figures/proto/protoA5_overlap_split.png
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("ACCV_2026/figures/proto"); OUT.mkdir(parents=True, exist_ok=True)
RUNGS = ["m025", "m050", "m100", "m150", "m200"]
XL = ["0.25x", "0.5x", "1x", "1.5x", "2x"]
x = np.arange(5)
TARGETS = [("kitti2012", "KITTI-2012", 14.5), ("kitti2015", "KITTI-2015", 18.9)]
BLUE, PURPLE, RED, GREEN = "#2b6cb0", "#6a3d9a", "#c0392b", "#2a8f3f"

cv = pd.read_csv("analysis/coverage_v2_flow_ladder.csv")
cv["rung"] = cv.train_dataset.astype(str).str.extract(r"(m\d+)")[0]
m = pd.read_csv("analysis/ladder_master_table.csv")

fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0))
for ax, (bench, lab, meanpx) in zip(axes, TARGETS):
    g = cv[cv.eval_dataset.astype(str) == bench].groupby("rung").agg(
        dBT=("mean_nn_eval_to_train_k1", "mean"),
        dTB=("mean_nn_train_to_eval_k1", "mean")).reindex(RUNGS)
    dBT, dTB = g.dBT.values, g.dTB.values
    pck = m[m.reg == "TF"].pivot_table(index="rung", columns="app", values=bench,
                                       aggfunc="mean").reindex(RUNGS).mean(axis=1).values
    cov = -dBT
    peak = int(np.argmin(dBT))
    rho_cov = spearmanr(-dBT, pck).correlation
    rho_off = spearmanr(-dTB, pck).correlation

    # shade under / over relative to THIS target's peak
    ax.axvspan(-0.4, peak - 0.5, color="#fdecea", alpha=0.5, zorder=0)
    ax.axvspan(peak + 0.5, 4.4, color="#fff4e6", alpha=0.6, zorder=0)
    ax.axvline(peak, color=GREEN, ls="--", lw=1.6, zorder=1)
    ax.text(peak, 1.0, f" optimum: {XL[peak]}", color=GREEN, fontsize=10, va="top", ha="left",
            transform=ax.get_xaxis_transform(), weight="bold")

    # coverage (left)
    ax.plot(x, cov, "o-", color=BLUE, lw=2.6, ms=8, label="coverage  $-d_{B\\to T}$", zorder=3)
    ax.set_ylabel("joint coverage  ($-$mean-NN)", color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax.margins(y=0.22)
    ax.set_xticks(x); ax.set_xticklabels(XL)
    ax.set_xlabel("source motion magnitude (x KITTI's)")
    ax.set_title(f"{lab}   (mean motion {meanpx:.0f} px)\n"
                 f"coverage & PCK peak at {XL[peak]}  -  cov$\\leftrightarrow$PCK $\\rho$={rho_cov:+.2f}, "
                 f"off-target $\\rho$={rho_off:+.2f}", fontsize=11)

    # PCK (right inner)
    ax2 = ax.twinx()
    ax2.plot(x, pck, "^-", color=PURPLE, lw=2.4, ms=8, label="transfer PCK", zorder=3)
    ax2.set_ylabel("peak PCK (pretrained)", color=PURPLE)
    ax2.tick_params(axis="y", labelcolor=PURPLE)

    # off-target (right outer, generous +/-50% window -> flatness honest)
    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("outward", 50))
    ax3.plot(x, -dTB, "s:", color=RED, lw=1.7, ms=6, alpha=0.9, label="off-target  $-d_{T\\to B}$", zorder=2)
    ax3.set_ylim(-dTB.mean() * 1.5, -dTB.mean() * 0.5)
    ax3.set_ylabel("off-target  ($-d_{T\\to B}$)", color=RED)
    ax3.tick_params(axis="y", labelcolor=RED)

    if bench == "kitti2012":   # one shared legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        h3, l3 = ax3.get_legend_handles_labels()
        ax.legend(h1 + h2 + h3, l1 + l2 + l3, fontsize=9, loc="lower center")

fig.suptitle("Source-target motion overlap across the magnitude ladder, per KITTI split (joint mean-NN)\n"
             "the optimum tracks the target's magnitude: KITTI-2012 (smaller) peaks at 1x, KITTI-2015 (larger) at 1.5x",
             fontsize=13, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / "protoA5_overlap_split.png", dpi=160, bbox_inches="tight")
print("wrote", OUT / "protoA5_overlap_split.png")
for bench, lab, _ in TARGETS:
    g = cv[cv.eval_dataset.astype(str) == bench].groupby("rung").mean_nn_eval_to_train_k1.mean().reindex(RUNGS)
    print(lab, "coverage dBT:", np.round(g.values, 4), "-> peak", XL[int(np.argmin(g.values))])
