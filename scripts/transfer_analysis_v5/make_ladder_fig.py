"""Controlled motion-magnitude x appearance grid -> KITTI transfer (fig:ladder).

Left: directed motion distances vs source magnitude, using the paper's headline
metric -- raw JOINT mean-NN distance d_{B->T} (coverage) and d_{T->B} (off-target),
the same metric as Table 1 (NOT the retired qnorm support-overlap, and NOT the
scale-normalized distance the original panel used which falsely fell monotonically).
KITTI-2012 and KITTI-2015 are plotted as separate lines. Coverage d_{B->T} dips to a
minimum at the matched magnitude (best coverage) and rises on either side; off-target
d_{T->B} stays flat (<=1% swing). Right two panels: peak-PCK heatmaps (inverted-U).

Output: ACCV_2026/figures/results/ladder_grid_final.png
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path("ACCV_2026/figures/results/ladder_grid_final.png")
RUNGS = ["m025", "m050", "m100", "m150", "m200"]
XLAB = ["0.25×", "0.5×", "1×", "1.5×", "2×"]

# --- directed mean-NN motion distances per rung (JOINT, KITTI target) ---
# Raw mean-NN (NOT qnorm): d_{B->T}=mean_nn_eval_to_train (coverage),
# d_{T->B}=mean_nn_train_to_eval (off-target). Per benchmark, averaged over apps.
cv = pd.read_csv("analysis/coverage_v2_flow_ladder.csv")
cv["rung"] = cv.train_dataset.astype(str).str.extract(r"(m\d+)")[0]
def _series(bench):
    g = cv[cv.eval_dataset == bench].groupby("rung").agg(
        dBT=("mean_nn_eval_to_train_k1", "mean"),
        dTB=("mean_nn_train_to_eval_k1", "mean")).reindex(RUNGS)
    return g["dBT"].values, g["dTB"].values
cov15, off15 = _series("kitti2015")
cov12, off12 = _series("kitti2012")

# --- transfer heatmaps (app x rung), averaged over scratch/pretrained ---
m = pd.read_csv("analysis/ladder_master_table.csv")
APPS = ["hq", "gsohq", "matte", "gsomatte"]
APPLAB = ["HDRI bg · KuBasic", "HDRI bg · GSO",
          "matte bg · KuBasic", "matte bg · GSO"]
APPDINO = {"hq": 0.58, "gsohq": 0.59, "matte": 1.08, "gsomatte": 1.04}
def heat(col):
    return m.pivot_table(index="app", columns="rung", values=col, aggfunc="mean").reindex(index=APPS, columns=RUNGS).values

fig = plt.figure(figsize=(15.9, 4.3))
# spacer column (index 1) gives the middle panel's appearance labels room AND
# clears the left panel's right-hand off-target axis, while the middle/right
# heatmaps stay close together.
gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 0.62, 1.15, 1.15], wspace=0.18)

# ---- Left: coverage / off-target mean-NN distance vs magnitude ----
# coverage d_{B->T} on the left axis (dips at the match); off-target d_{T->B} on a
# generous right axis so its flatness reads as a flat floor, not a co-moving line.
ax = fig.add_subplot(gs[0, 0])
ax2 = ax.twinx()
x = np.arange(5)
COV15, COV12 = "#1b4f86", "#5b9bd5"
OFF15, OFF12 = "#a3271b", "#e8806f"
ax.axvspan(-0.4, 1.5, color="#fdecea", alpha=0.5, zorder=0)   # under
ax.axvspan(2.5, 4.4, color="#fff4e6", alpha=0.6, zorder=0)    # over
ax.axvline(2, color="#2a8f3f", ls="--", lw=1.4, zorder=1)     # matched (1x)
ax.plot(x, cov15, "-o", color=COV15, lw=2.5, ms=7, zorder=4)
ax.plot(x, cov12, "-o", color=COV12, lw=2.5, ms=7, zorder=4, mfc="white", mew=1.7)
ax2.plot(x, off15, "--s", color=OFF15, lw=2.0, ms=6, zorder=3)
ax2.plot(x, off12, "--s", color=OFF12, lw=2.0, ms=6, zorder=3, mfc="white", mew=1.5)
ax.set_xticks(x); ax.set_xticklabels(XLAB)
ax.set_xlabel("source motion magnitude (× KITTI's mean flow)")
ax.set_ylim(0.0, 0.0115)
ax2.set_ylim(0.0, 0.058)
ax.set_ylabel(r"coverage  $d_{B\to T}$  (mean-NN, lower=better)", color=COV15)
ax2.set_ylabel(r"off-target  $d_{T\to B}$  (mean-NN)", color=OFF15)
ax.tick_params(axis="y", labelcolor=COV15); ax2.tick_params(axis="y", labelcolor=OFF15)
ax.set_title("Coverage dips at the match;\noff-target stays flat", loc="left", fontsize=10.5)
ax2.text(0.96, 0.80, "off-target flat (≤1%)", transform=ax.transAxes,
         fontsize=8.0, color=OFF15, ha="right", style="italic")
ax.annotate("under", (0.30, 0.0004), fontsize=8.3, color="#a33", ha="center")
ax.annotate("over-shoot", (3.4, 0.0004), fontsize=8.3, color="#b5670f", ha="center")
_h = [Line2D([0],[0], color=COV15, lw=2.3, marker="o", ms=6, label="coverage · KITTI-15"),
      Line2D([0],[0], color=COV12, lw=2.3, marker="o", ms=6, mfc="white", mew=1.5, label="coverage · KITTI-12"),
      Line2D([0],[0], color=OFF15, lw=2.0, ls="--", marker="s", ms=5, label="off-target · KITTI-15"),
      Line2D([0],[0], color=OFF12, lw=2.0, ls="--", marker="s", ms=5, mfc="white", mew=1.3, label="off-target · KITTI-12")]
ax.legend(handles=_h, fontsize=7.2, loc="upper center", ncol=2, columnspacing=0.9, handlelength=1.8)

# ---- Middle/Right: transfer heatmaps ----
for gi, (col, tgt) in enumerate([("kitti2012", "KITTI-2012"), ("kitti2015", "KITTI-2015")]):
    ax = fig.add_subplot(gs[0, gi + 2])
    H = heat(col)
    vmin, vmax = np.nanmin(H), np.nanmax(H)
    im = ax.imshow(H, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    for i in range(4):
        for j in range(5):
            v = H[i, j]
            best = (v == np.nanmax(H[:, j]))
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8.2,
                    color="white" if (v - vmin) / (vmax - vmin) < 0.5 else "#111",
                    weight="bold" if best else "normal")
    ax.set_xticks(range(5)); ax.set_xticklabels(XLAB)
    ax.set_yticks(range(4))
    if gi == 0:
        ax.set_yticklabels([f"{l}\n(DINO gap {APPDINO[a]:.2f})" for l, a in zip(APPLAB, APPS)], fontsize=7.6)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("source motion magnitude (× KITTI's)")
    ax.set_title(f"Transfer to {tgt} (peak PCK)", loc="left", fontsize=10.5)
    ax.axvline(2, color="white", lw=1.0, ls="--", alpha=0.7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

fig.savefig(OUT, bbox_inches="tight", dpi=200)
print("wrote", OUT)

# ---- Save individual panels for LaTeX subfigures ----
# Both panels use the SAME figure height and the SAME plot-area vertical band
# (bottom=BOT, top=TOP) and are saved WITHOUT bbox_inches="tight", so when included
# at a common height in LaTeX their plot areas line up exactly.
OUT_A  = Path("ACCV_2026/figures/results/ladder_panel_a.png")
OUT_BC = Path("ACCV_2026/figures/results/ladder_panel_bc.png")
FH = 3.7            # shared figure height (in)
BOT, TOP = 0.16, 0.86   # shared plot-area vertical band (fraction of FH)

# Panel (a): coverage / off-target line chart
fig_a = plt.figure(figsize=(5.0, FH))
ax_a = fig_a.add_axes([0.145, BOT, 0.715, TOP - BOT])
ax_a2 = ax_a.twinx()
ax_a.axvspan(-0.4, 1.5, color="#fdecea", alpha=0.5, zorder=0)
ax_a.axvspan(2.5, 4.4, color="#fff4e6", alpha=0.6, zorder=0)
ax_a.axvline(2, color="#2a8f3f", ls="--", lw=1.4, zorder=1)
ax_a.plot(x, cov15, "-o", color=COV15, lw=2.5, ms=7, zorder=4)
ax_a.plot(x, cov12, "-o", color=COV12, lw=2.5, ms=7, zorder=4, mfc="white", mew=1.7)
ax_a2.plot(x, off15, "--s", color=OFF15, lw=2.0, ms=6, zorder=3)
ax_a2.plot(x, off12, "--s", color=OFF12, lw=2.0, ms=6, zorder=3, mfc="white", mew=1.5)
ax_a.set_xticks(x); ax_a.set_xticklabels(XLAB)
ax_a.set_xlabel("source motion magnitude (× KITTI's mean flow)")
ax_a.set_ylim(0.0, 0.0115)
ax_a2.set_ylim(0.0, 0.058)
ax_a.set_ylabel(r"coverage  $d_{B\to T}$  (mean-NN, lower=better)", color=COV15)
ax_a2.set_ylabel(r"off-target  $d_{T\to B}$  (mean-NN)", color=OFF15)
ax_a.tick_params(axis="y", labelcolor=COV15)
ax_a2.tick_params(axis="y", labelcolor=OFF15)
ax_a.annotate("under", (0.30, 0.0004), fontsize=8.3, color="#a33", ha="center")
ax_a.annotate("over-shoot", (3.4, 0.0004), fontsize=8.3, color="#b5670f", ha="center")
ax_a.legend(handles=_h, fontsize=7.2, loc="upper center", ncol=2, columnspacing=0.9, handlelength=1.8)
fig_a.savefig(OUT_A, dpi=200)
plt.close(fig_a)

# Panel (b+c): KITTI-2012 and KITTI-2015 heatmaps, manually placed at the same
# vertical band [BOT, TOP] as panel (a) so plot areas align across subfigures.
fig_bc = plt.figure(figsize=(8.8, FH))
HMW = 0.30                                  # heatmap width fraction
hm_left = [0.185, 0.585]                    # left edges of the two heatmaps
for gi, (col, tgt) in enumerate([("kitti2012", "KITTI-2012"), ("kitti2015", "KITTI-2015")]):
    ax = fig_bc.add_axes([hm_left[gi], BOT, HMW, TOP - BOT])
    H = heat(col)
    vmin, vmax = np.nanmin(H), np.nanmax(H)
    im = ax.imshow(H, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    for i in range(4):
        for j in range(5):
            v = H[i, j]
            best = (v == np.nanmax(H[:, j]))
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8.2,
                    color="white" if (v - vmin) / (vmax - vmin) < 0.5 else "#111",
                    weight="bold" if best else "normal")
    ax.set_xticks(range(5)); ax.set_xticklabels(XLAB)
    ax.set_yticks(range(4))
    if gi == 0:
        ax.set_yticklabels([f"{l}\n" + (r"$\mathbf{DINO\ gap\ %.2f}$" % APPDINO[a])
                            for l, a in zip(APPLAB, APPS)], fontsize=7.6)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("source motion magnitude (× KITTI's)")
    ax.set_title(f"Transfer to {tgt} (peak PCK)", loc="left", fontsize=10)
    ax.axvline(2, color="white", lw=1.0, ls="--", alpha=0.7)
    cax = fig_bc.add_axes([hm_left[gi] + HMW + 0.008, BOT, 0.013, TOP - BOT])
    fig_bc.colorbar(im, cax=cax)
fig_bc.savefig(OUT_BC, dpi=200)
plt.close(fig_bc)

print("wrote", OUT_A, OUT_BC)
