"""PROTOTYPE v4: motion-coverage view of the 0.25x->2x ladder vs KITTI (JOINT metric).

Goal: show that source motion COVERAGE of the target rises as magnitude grows to
match the target (~1-1.5x) and then DROPS as it over-shoots (2x).

Key design point (why this is NOT a convex hull):
  A hull / single fixed-density contour shows binary EXTENT, which only grows with
  magnitude -- it can never show the 2x drop. The drop is a DENSITY effect: at 2x the
  source mass spreads so wide that its local density inside the target region thins
  out, raising the joint mean-NN distance d_{B->T}. So each panel renders the source as
  a filled DENSITY (bulk + faint tail kept), pinned to the real coverage number.

Row 1: per-rung flow-space [dx,dy] -- source filled density (orange) + faint extent
       outline (tail kept), fixed KITTI target outline (blue) + mean-flow ring.
       Each panel's border is colored by its coverage; the peak (1.5x) is framed bold.
Row 2: the headline curve -- joint coverage (-d_{B->T}) vs magnitude, an inverted-U
       peaking at 1.5x, with transfer PCK overlaid (they co-peak).

Output: ACCV_2026/figures/proto/protoA_ladder_motion_v2.png
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

CACHE = Path("/mnt/nvme_1tb_b/coverage_vectors")
OUT = Path("ACCV_2026/figures/proto"); OUT.mkdir(parents=True, exist_ok=True)
RUNGS = ["m025", "m050", "m100", "m150", "m200"]
RUNGLAB = ["0.25x", "0.5x", "1x", "1.5x", "2x"]
BENCH = "kitti2015"
rng = np.random.default_rng(0)

WIN = 62.0
EDGES = np.linspace(-WIN, WIN, 141)
xc = 0.5 * (EDGES[:-1] + EDGES[1:])


def load(name, n=150000):
    v = np.load(CACHE / f"{name}_flow.npy").astype(np.float32)  # pixel [x,y,dx,dy]
    return v[rng.choice(len(v), min(n, len(v)), replace=False)]


def density(fx, fy, sigma=1.5):
    h, _, _ = np.histogram2d(fx, fy, bins=[EDGES, EDGES])
    h = gaussian_filter(h.T, sigma)
    return h / h.max()


target = load(f"{BENCH}_val")
rungs = {r: load(f"kitti_{r}_hq_train") for r in RUNGS}
tgt_d = density(target[:, 2], target[:, 3])
kitti_mean = np.hypot(target[:, 2], target[:, 3]).mean()

# ---- real joint coverage per rung (the paper's metric) ----
cv = pd.read_csv("analysis/coverage_v2_flow_ladder.csv")
cv = cv[cv.eval_dataset.astype(str).str.contains(BENCH, case=False)].copy()
cv["rung"] = cv.train_dataset.astype(str).str.extract(r"(m\d+)")[0]
g = cv.groupby("rung").agg(dBT=("mean_nn_eval_to_train_k1", "mean")).reindex(RUNGS)
dBT = g.dBT.values
cov = -dBT
# normalized 0..1 coverage for per-panel border color (0=worst rung, 1=peak rung)
covn = (dBT.max() - dBT) / (dBT.max() - dBT.min())
peak = int(np.argmin(dBT))

# transfer PCK (pretrained encoder) to overlay on the curve
mt = pd.read_csv("analysis/ladder_master_table.csv")
pck = mt[mt.reg == "TF"].pivot_table(index="rung", columns="app", values=BENCH,
                                     aggfunc="mean").reindex(RUNGS).mean(axis=1).values

# border colormap: low coverage = grey, high = green
border_cmap = LinearSegmentedColormap.from_list("cov", ["#b8b8b8", "#7fb069", "#1a7a33"])

fig = plt.figure(figsize=(16, 7.0))
gs = gridspec.GridSpec(2, 5, height_ratios=[1.18, 1.0], hspace=0.46, wspace=0.10,
                       top=0.875, bottom=0.085, left=0.045, right=0.985)

theta = np.linspace(0, 2 * np.pi, 200)
ring_x, ring_y = kitti_mean * np.cos(theta), kitti_mean * np.sin(theta)

# fill levels (normalized density): faint tail kept at 0.01, bulk graded up to 1.0
fill_levels = [0.01, 0.04, 0.10, 0.22, 0.40, 0.65, 1.0]

for j, (r, lab) in enumerate(zip(RUNGS, RUNGLAB)):
    ax = fig.add_subplot(gs[0, j])
    sd = density(rungs[r][:, 2], rungs[r][:, 3])
    # source as filled density (bulk + faint tail), perceptually graded
    ax.contourf(xc, xc, sd, levels=fill_levels, cmap="Oranges", alpha=1.0, extend="max")
    # faint outline at the lowest level = honest extent / tail (the thing a hull would clip)
    ax.contour(xc, xc, sd, levels=[0.01], colors="#c2641a", linewidths=0.6, alpha=0.6)
    # fixed KITTI target reference: outline (full extent) + mid contour + mean-flow ring
    ax.contour(xc, xc, tgt_d, levels=[0.015, 0.2], colors="#15406b", linewidths=[0.9, 1.7])
    ax.plot(ring_x, ring_y, color="#15406b", ls=":", lw=1.1, alpha=0.75)

    ax.set_xlim(-WIN, WIN); ax.set_ylim(WIN, -WIN)   # image convention: +dy = down
    ax.set_aspect("equal")
    ax.set_xticks([-50, 0, 50]); ax.set_yticks([-50, 0, 50])
    ax.tick_params(labelsize=8)
    ax.axhline(0, color="k", lw=0.3, alpha=0.2); ax.axvline(0, color="k", lw=0.3, alpha=0.2)
    if j == 0:
        ax.set_ylabel("dy (px)", fontsize=10)
    ax.set_xlabel("dx (px)", fontsize=9)

    # title carries the magnitude + coverage readout (color-coded by coverage)
    smean = np.hypot(rungs[r][:, 2], rungs[r][:, 3]).mean()
    bc = border_cmap(covn[j])
    ax.set_title(f"{lab}   ($|f|$={smean:.0f}px,  coverage {covn[j]*100:.0f}%)",
                 fontsize=11.5, color=bc, weight="bold" if j == peak else "normal")

    # colored border encodes coverage; peak gets a bold frame + "peak" tag
    lw = 3.6 if j == peak else 2.2
    for sp in ax.spines.values():
        sp.set_edgecolor(bc); sp.set_linewidth(lw)
    if j == peak:
        ax.text(0.5, 0.04, "PEAK COVERAGE", transform=ax.transAxes, ha="center",
                va="bottom", fontsize=9, color="#1a7a33", weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1a7a33", alpha=0.9))
    if j == len(RUNGS) - 1:
        ax.text(0.5, 0.04, "over-shoots:\ndensity thins", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8, color="#a14a00", weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#a14a00", alpha=0.9))
    if j == 0:
        ax.text(0.5, 0.04, "under-fills:\ntoo small", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8, color="#666", weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#999", alpha=0.9))

fig.text(0.5, 0.985,
         "Source motion grows to match the target, then over-shoots: coverage rises to ~1.5x, then drops",
         ha="center", fontsize=15, weight="bold")
fig.text(0.5, 0.955,
         "small fan inside KITTI (0.25x)  →  fills KITTI (1–1.5x, peak coverage)  →  spreads past KITTI, "
         "local density thins (2x)",
         ha="center", fontsize=10.5, color="#444")
fig.legend(handles=[
    Line2D([0], [0], color="#15406b", lw=1.8, label="KITTI target (outline)"),
    Patch(facecolor="#ee9c3e", edgecolor="none", label="source motion (density)"),
    Line2D([0], [0], color="#15406b", ls=":", lw=1.3, label="KITTI mean-flow radius"),
], loc="upper center", fontsize=9.5, ncol=3, bbox_to_anchor=(0.5, 0.945), frameon=False)

# ---- Row 2: the coverage inverted-U (real joint metric), PCK overlaid ----
axc = fig.add_subplot(gs[1, :])
x = np.arange(5)
axc.axvspan(-0.4, 1.0, color="#f3f3f3", alpha=0.7, zorder=0)      # under-fill region
axc.axvspan(3.0, 4.4, color="#fff4e6", alpha=0.7, zorder=0)       # over-shoot region
axc.plot(x, cov, "o-", color="#1a7a33", lw=3.0, ms=11, zorder=4, label="joint coverage  $-d_{B\\to T}$")
axc.scatter([peak], [cov[peak]], s=320, facecolor="none", edgecolor="#1a7a33", lw=2.4, zorder=5)
axc.axvline(peak, color="#1a7a33", ls="--", lw=1.4, alpha=0.6, zorder=1)
axc.annotate("peak coverage\n(source best matches target)", xy=(peak, cov[peak]),
             xytext=(peak - 1.25, cov[peak] + (cov.max() - cov.min()) * 0.12),
             fontsize=10, color="#1a7a33", weight="bold",
             arrowprops=dict(arrowstyle="->", color="#1a7a33", lw=1.6))
axc.annotate("over-shoot:\ncoverage drops", xy=(4, cov[4]),
             xytext=(3.45, cov[4] - (cov.max() - cov.min()) * 0.30),
             fontsize=10, color="#a14a00", weight="bold",
             arrowprops=dict(arrowstyle="->", color="#a14a00", lw=1.6))
axc.set_xticks(x); axc.set_xticklabels(RUNGLAB, fontsize=12)
axc.set_xlabel("source motion magnitude  (× KITTI's)", fontsize=12)
axc.set_ylabel("joint coverage\n$-d_{B\\to T}$  (higher = better)", color="#1a7a33", fontsize=11)
axc.tick_params(axis="y", labelcolor="#1a7a33")
axc.margins(y=0.28)
axc.grid(axis="y", ls=":", alpha=0.35)

# overlay transfer PCK to show coverage tracks transfer
ax2 = axc.twinx()
ax2.plot(x, pck, "^--", color="#6a3d9a", lw=2.2, ms=9, zorder=3, alpha=0.9, label="transfer PCK")
ax2.set_ylabel("transfer PCK", color="#6a3d9a", fontsize=11)
ax2.tick_params(axis="y", labelcolor="#6a3d9a")
ax2.margins(y=0.28)

h1, l1 = axc.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
axc.legend(h1 + h2, l1 + l2, fontsize=10, loc="lower center", ncol=2, framealpha=0.9)
axc.set_title("Coverage of the target is an inverted-U: it rises to the match (~1.5x) and falls on over-shoot — "
              "and transfer PCK co-peaks", fontsize=11.5, loc="left")

fig.savefig(OUT / "protoA_ladder_motion_v2.png", dpi=160, bbox_inches="tight")
print("wrote", OUT / "protoA_ladder_motion_v2.png")
print("coverage -dBT:", np.round(cov, 5), "peak", RUNGLAB[peak])
print("coverage norm%:", np.round(covn * 100, 0))
print("PCK:", np.round(pck, 1))
