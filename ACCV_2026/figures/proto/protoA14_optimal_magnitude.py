"""WHY the center reddens at high magnitude: the optimal magnitude varies across the frame.

Each scene region has its own motion scale, so its own best source magnitude:
slow regions (horizon, mid-field) peak at ~1x and OVER-SHOOT beyond that; the fast road
needs the most magnitude (peaks at 1.5-2x). A single global magnitude cannot be optimal
everywhere -- to cover the road you over-shoot everything slower, which is why the
mid-field turns red at 1.5x/2x.

Left:  map of the OPTIMAL magnitude at each location (parabolic sub-rung minimum of the
       flow residual), painted on the camera frame -> a gradient that grows toward the road.
Right: each region's flow residual vs magnitude (normalized to its own best) -> the minima
       sit at different rungs: horizon/mid at 1x, road later.

Output: ACCV_2026/figures/proto/protoA14_optimal_magnitude.png
"""
from pathlib import Path
import warnings
import numpy as np
warnings.filterwarnings("ignore", message="Mean of empty slice")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic_2d

CACHE = Path("/mnt/nvme_1tb_b/coverage_vectors")
OUT = Path("ACCV_2026/figures/proto"); OUT.mkdir(parents=True, exist_ok=True)
W = H = 512
NB = 22
MINCOUNT = 12
rng = np.random.default_rng(1)
RUNGS = ["m025", "m050", "m100", "m150", "m200"]
RLAB = ["0.25×", "0.5×", "1×", "1.5×", "2×"]
TARGETS = [("kitti2015_val", "KITTI-2015"), ("kitti2012_val", "KITTI-2012")]
edges = np.linspace(0, 512, NB + 1)
xc = yc = 0.5 * (edges[:-1] + edges[1:])
YBANDS = [(140, 235, "horizon/top"), (235, 320, "upper-mid"), (320, 400, "lower-mid"), (400, 512, "road/bottom")]
BANDC = ["#3b7dd8", "#46a04a", "#e08a1e", "#c0392b"]


def load(n, N):
    v = np.load(CACHE / f"{n}_flow.npy").astype(np.float64)
    return v[rng.choice(len(v), min(N, len(v)), replace=False)]


def norm(v):
    o = np.empty_like(v); o[:, 0] = 2 * v[:, 0] / W - 1; o[:, 1] = 2 * v[:, 1] / H - 1
    o[:, 2] = 2 * v[:, 2] / W; o[:, 3] = 2 * v[:, 3] / H; return o


sraw = {r: load(f"kitti_{r}_hq_train", 80000) for r in RUNGS}
sn = {r: norm(sraw[r]) for r in RUNGS}

fig = plt.figure(figsize=(15.5, 9.4))
gs = gridspec.GridSpec(2, 2, width_ratios=[1.05, 1.0], left=0.085, right=0.93,
                       top=0.86, bottom=0.08, hspace=0.28, wspace=0.22)
cmap = plt.get_cmap("plasma").copy(); cmap.set_bad("#ededed")

for ti, (tname, tlab) in enumerate(TARGETS):
    traw = load(tname, 30000); tnn = norm(traw)
    res_pt = {}
    for r in RUNGS:
        d, idx = cKDTree(sn[r]).query(tnn, k=1)
        res_pt[r] = np.hypot(traw[:, 2] - sraw[r][idx, 2], traw[:, 3] - sraw[r][idx, 3])
    # per-cell residual stack + optimal (parabolic) rung index
    stack = np.stack([binned_statistic_2d(traw[:, 0], traw[:, 1], res_pt[r], "mean", bins=[edges, edges])[0] for r in RUNGS])
    cnt = binned_statistic_2d(traw[:, 0], traw[:, 1], res_pt["m100"], "count", bins=[edges, edges])[0]
    valid = (cnt >= MINCOUNT) & np.all(np.isfinite(stack), axis=0)
    k = np.argmin(np.where(np.isfinite(stack), stack, np.inf), axis=0)
    def tk(o):
        kk = np.clip(k + o, 0, 4)
        return np.take_along_axis(stack, kk[None], axis=0)[0]
    r0, rm, rp = tk(0), tk(-1), tk(1)
    denom = rm - 2 * r0 + rp
    interior = (k > 0) & (k < 4) & (denom > 1e-9)
    off = np.where(interior, 0.5 * (rm - rp) / np.where(denom == 0, 1, denom), 0.0)
    opt = np.where(valid, k + np.clip(off, -0.5, 0.5), np.nan)

    # ---- left: optimal-magnitude map ----
    axm = fig.add_subplot(gs[ti, 0])
    im = axm.pcolormesh(xc, yc, opt.T, cmap=cmap, vmin=0, vmax=4, shading="auto")
    axm.set_ylim(512, 0); axm.set_xlim(0, 512); axm.set_aspect("equal")
    axm.set_xticks([]); axm.set_yticks([])
    axm.set_ylabel(f"{tlab}\ncamera frame  sky↑ ↓road", fontsize=10, weight="bold", linespacing=1.4)
    if ti == 0:
        axm.set_title("Optimal source magnitude at each location", fontsize=12.5, weight="bold")
    axm.text(0.5, 0.94, "slow → low magnitude", transform=axm.transAxes, ha="center", va="top",
             fontsize=8, color="#222", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    axm.text(0.5, 0.07, "fast road → high magnitude", transform=axm.transAxes, ha="center", va="bottom",
             fontsize=8, color="#222", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    # ---- right: per-region residual vs magnitude (normalized to each region's best) ----
    axc = fig.add_subplot(gs[ti, 1])
    for (lo, hi, nm), col in zip(YBANDS, BANDC):
        m = (traw[:, 1] >= lo) & (traw[:, 1] < hi)
        if m.sum() < 100:
            continue
        curve = np.array([res_pt[r][m].mean() for r in RUNGS]); curve = curve / curve.min()
        kk = int(np.argmin(curve))
        axc.plot(range(5), curve, "o-", color=col, lw=2.2, ms=6, label=f"{nm}")
        axc.scatter([kk], [curve[kk]], s=120, facecolor="none", edgecolor=col, lw=2.2, zorder=5)
    axc.set_xticks(range(5)); axc.set_xticklabels(RLAB)
    axc.set_xlabel("source motion magnitude")
    axc.set_ylabel("flow residual\n(÷ each region's best)", fontsize=9.5)
    axc.grid(axis="y", ls=":", alpha=0.4)
    axc.legend(fontsize=8.5, loc="upper center", ncol=2, title="frame region (circle = its optimum)", title_fontsize=8)
    if ti == 0:
        axc.set_title("Each region peaks at a different magnitude", fontsize=12.5, weight="bold")

cax = fig.add_axes([0.94, 0.30, 0.014, 0.4])
cb = fig.colorbar(im, cax=cax); cb.set_ticks([0, 1, 2, 3, 4]); cb.set_ticklabels(RLAB)
cb.set_label("optimal magnitude", fontsize=10)

fig.suptitle("The optimal magnitude is not global — it grows from the slow horizon to the fast road, "
             "so any single magnitude over-shoots part of the scene", fontsize=14, weight="bold", y=0.955)
fig.text(0.5, 0.90, "This is why the mid-field reddens at 1.5×/2× in the coverage map: it already peaked at ~1× and is now over-shot, "
         "while the road still wants more.", ha="center", fontsize=10, color="#444")

fig.savefig(OUT / "protoA14_optimal_magnitude.png", dpi=155, bbox_inches="tight")
print("wrote", OUT / "protoA14_optimal_magnitude.png")
