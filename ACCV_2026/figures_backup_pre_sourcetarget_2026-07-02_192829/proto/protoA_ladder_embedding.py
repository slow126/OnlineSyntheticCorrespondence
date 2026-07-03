"""PROTOTYPE v3: motion-space view of the 0.25x->2x magnitude ladder vs KITTI.

Fixes the v2 artifact: a fixed +/-0.16 normalized window clipped the largest
motions at 2x, and because the dolly fan is one-sided in dy (ground-plane motion)
the clip read as "dy collapses". Here everything is in ABSOLUTE PIXEL coordinates
with ONE wide window (+/-WIN px) that fits every rung, equal aspect, no clipping.
The motion is isotropic (std_dx ~ std_dy at every rung); the figure now shows that.

Row 1: flow-space [dx,dy] density per rung (orange), KITTI target fixed (blue).
       0.25x = small fan inside KITTI -> grows -> 2x = fan over-shoots KITTI's reach.
Row 2: flow-magnitude |f| distribution (px), KITTI filled, rungs swept light->dark.

Output: ACCV_2026/figures/proto/protoA_ladder_motion.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter

CACHE = Path("/mnt/nvme_1tb_b/coverage_vectors")
OUT = Path("ACCV_2026/figures/proto"); OUT.mkdir(parents=True, exist_ok=True)
RUNGS = ["m025", "m050", "m100", "m150", "m200"]
RUNGLAB = ["0.25x", "0.5x", "1x", "1.5x", "2x"]
rng = np.random.default_rng(0)
mag_cmap = LinearSegmentedColormap.from_list("mag", ["#f0a860", "#d2691e", "#7a1500"])

WIN = 62.0  # px display half-window: fits 2x (max 58px) and KITTI with no clipping


def load(name, n):
    v = np.load(CACHE / f"{name}_flow.npy").astype(np.float32)  # pixel [x,y,dx,dy]
    return v[rng.choice(len(v), min(n, len(v)), replace=False)]


N = 150000
target = load("kitti2015_val", N)
rungs = {r: load(f"kitti_{r}_hq_train", N) for r in RUNGS}

EDGES = np.linspace(-WIN, WIN, 141)
xc = 0.5 * (EDGES[:-1] + EDGES[1:])


def density(fx, fy):
    h, _, _ = np.histogram2d(fx, fy, bins=[EDGES, EDGES])
    h = gaussian_filter(h.T, 1.5)
    return h / h.max()


tgt_d = density(target[:, 2], target[:, 3])
kitti_mean = np.hypot(target[:, 2], target[:, 3]).mean()

fig = plt.figure(figsize=(16, 8.4))
gs = gridspec.GridSpec(2, 5, height_ratios=[2.0, 0.95], hspace=0.26, wspace=0.08)

# fixed "target reach" reference ring at KITTI's mean flow magnitude
theta = np.linspace(0, 2 * np.pi, 200)
ring_x, ring_y = kitti_mean * np.cos(theta), kitti_mean * np.sin(theta)

# ---- Row 1: rung as LOG-density (sparse tail kept), KITTI as contour outline ----
# log density shows every occupied bin, so the sparse high-dy fan tail is NOT clipped
# the way a fixed-level filled contour ("hull") clips it.
for j, (r, lab) in enumerate(zip(RUNGS, RUNGLAB)):
    ax = fig.add_subplot(gs[0, j])
    s = rungs[r]
    Hs, _, _ = np.histogram2d(s[:, 2], s[:, 3], bins=[EDGES, EDGES])
    Hs = np.ma.masked_where(Hs.T == 0, Hs.T)
    ax.pcolormesh(xc, xc, Hs, cmap="Oranges", norm=LogNorm(vmin=1, vmax=Hs.max()), shading="auto")
    # KITTI reference outline at a low + mid density level (its full extent, not a fill)
    ax.contour(xc, xc, tgt_d, levels=[0.015, 0.2], colors="#15406b", linewidths=[0.8, 1.5])
    ax.plot(ring_x, ring_y, color="#15406b", ls=":", lw=1.1, alpha=0.75)  # KITTI mean-flow ring
    ax.set_title(lab, fontsize=14)
    ax.set_xlim(-WIN, WIN); ax.set_ylim(WIN, -WIN)  # image convention: +dy = down
    ax.set_aspect("equal")
    ax.set_xticks([-50, 0, 50]); ax.set_yticks([-50, 0, 50])
    ax.tick_params(labelsize=8)
    ax.axhline(0, color="k", lw=0.3, alpha=0.25); ax.axvline(0, color="k", lw=0.3, alpha=0.25)
    if j == 0:
        ax.set_ylabel("dy (px)", fontsize=10)
    ax.set_xlabel("dx (px)", fontsize=9)
# shared legend proxies
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([0], [0], color="#15406b", lw=1.6, label="KITTI target (outline)"),
                    Patch(facecolor="#ee9c3e", edgecolor="none", label="source rung (log density)"),
                    Line2D([0], [0], color="#15406b", ls=":", lw=1.3, label="KITTI mean-flow radius")],
           loc="upper right", fontsize=9.5, ncol=3, bbox_to_anchor=(0.995, 1.005))
fig.text(0.5, 0.99, "Source motion (orange) vs KITTI target (blue) in flow space  -  absolute px, equal aspect",
         ha="center", fontsize=14, weight="bold")
fig.text(0.5, 0.96, "isotropic growth: small fan inside KITTI (0.25x)  ->  fills KITTI (~1.5x)  ->  over-shoots KITTI's reach (2x)",
         ha="center", fontsize=11, color="#444")

# ---- Row 2: flow-magnitude distribution (px) ----
axm = fig.add_subplot(gs[1, :])
mag_edges = np.linspace(0, WIN, 90)
mc = 0.5 * (mag_edges[:-1] + mag_edges[1:])
tm = np.hypot(target[:, 2], target[:, 3])
th, _ = np.histogram(tm, bins=mag_edges, density=True)
axm.fill_between(mc, th, color="#1f6fb2", alpha=0.32, label="KITTI target", zorder=1)
axm.plot(mc, th, color="#15406b", lw=2.0, zorder=2)
axm.axvline(kitti_mean, color="#15406b", ls="--", lw=1.3, alpha=0.85)
axm.text(kitti_mean + 0.6, axm.get_ylim()[1], f"KITTI mean {kitti_mean:.0f}px",
         color="#15406b", fontsize=9, va="top")
for i, (r, lab) in enumerate(zip(RUNGS, RUNGLAB)):
    sm = np.hypot(rungs[r][:, 2], rungs[r][:, 3])
    sh, _ = np.histogram(sm, bins=mag_edges, density=True)
    axm.plot(mc, sh, color=mag_cmap(i / 4), lw=2.3, label=f"source {lab}")
axm.set_xlabel("flow magnitude |f| (px)", fontsize=11)
axm.set_ylabel("density", fontsize=11)
axm.set_title("Source magnitude sweeps through the target: 0.25x under-fills, 2x over-shoots KITTI",
              fontsize=11.5, loc="left")
axm.legend(fontsize=9, ncol=6, loc="upper right")
axm.set_xlim(0, 52)

fig.savefig(OUT / "protoA_ladder_motion.png", dpi=160, bbox_inches="tight")
print("wrote", OUT / "protoA_ladder_motion.png")
print(f"KITTI mean |f| = {kitti_mean:.2f} px; rung means:",
      {r: round(float(np.hypot(rungs[r][:, 2], rungs[r][:, 3]).mean()), 1) for r in RUNGS})
