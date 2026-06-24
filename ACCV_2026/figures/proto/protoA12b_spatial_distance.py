"""Spatial coverage map in absolute DISTANCE units (companion to ladder panel a).

Same as protoA12 but colored by the absolute motion-coverage distance (px) rather than
the relative-to-own-average delta:
  green = low distance  (well covered)
  red   = high distance (less covered)
Each panel shows its mean distance (px); per-target color scale + px colorbar.

This is the spatial version of panel (a)'s d_{B->T} curve: where the synthetic motion is
close to / far from KITTI's, per source magnitude.

Output: ACCV_2026/figures/proto/protoA12b_spatial_distance.png
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
from scipy.ndimage import gaussian_filter

CACHE = Path("/mnt/nvme_1tb_b/coverage_vectors")
OUT = Path("ACCV_2026/figures/proto"); OUT.mkdir(parents=True, exist_ok=True)
W = H = 512
NB = 26
MINCOUNT = 8
rng = np.random.default_rng(1)
RUNGS = ["m025", "m050", "m100", "m150", "m200"]
RLAB = ["0.25×", "0.5×", "1×", "1.5×", "2×"]
TARGETS = [("kitti2015_val", "KITTI-2015"), ("kitti2012_val", "KITTI-2012")]


def load_raw(name, n):
    v = np.load(CACHE / f"{name}_flow.npy").astype(np.float64)
    return v[rng.choice(len(v), min(n, len(v)), replace=False)]


def norm(v):
    o = np.empty_like(v); o[:, 0] = 2 * v[:, 0] / W - 1; o[:, 1] = 2 * v[:, 1] / H - 1
    o[:, 2] = 2 * v[:, 2] / W; o[:, 3] = 2 * v[:, 3] / H; return o


def nan_smooth(a, sigma=0.9):
    m = np.isfinite(a).astype(float); a0 = np.where(np.isfinite(a), a, 0.0)
    num = gaussian_filter(a0, sigma); den = gaussian_filter(m, sigma)
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0.15)


sraw = {r: load_raw(f"kitti_{r}_hq_train", 80000) for r in RUNGS}
sn = {r: norm(sraw[r]) for r in RUNGS}
edges = np.linspace(0, 512, NB + 1)
xc = yc = 0.5 * (edges[:-1] + edges[1:])

cmap = plt.get_cmap("RdYlGn_r").copy(); cmap.set_bad("#ededed")

fig = plt.figure(figsize=(19, 8.6))
gs = gridspec.GridSpec(2, 5, left=0.07, right=0.9, top=0.93, bottom=0.06, hspace=0.16, wspace=0.06)
axmap = [[None] * 5 for _ in range(2)]
for ti, (tname, tlab) in enumerate(TARGETS):
    traw = load_raw(tname, 24000); tnn = norm(traw)
    cells = {}
    for r in RUNGS:
        d, idx = cKDTree(sn[r]).query(tnn, k=1)
        fres = np.hypot(traw[:, 2] - sraw[r][idx, 2], traw[:, 3] - sraw[r][idx, 3])
        stat, _, _, _ = binned_statistic_2d(traw[:, 0], traw[:, 1], fres, "mean", bins=[edges, edges])
        cnt, _, _, _ = binned_statistic_2d(traw[:, 0], traw[:, 1], fres, "count", bins=[edges, edges])
        cells[r] = nan_smooth(np.where(cnt >= MINCOUNT, stat, np.nan))
    valid = np.all([np.isfinite(cells[r]) for r in RUNGS], axis=0)
    allv = np.concatenate([cells[r][valid] for r in RUNGS])
    vmin, vmax = np.floor(np.percentile(allv, 4)), np.ceil(np.percentile(allv, 96))

    for ci, r in enumerate(RUNGS):
        ax = fig.add_subplot(gs[ti, ci]); axmap[ti][ci] = ax
        im = ax.pcolormesh(xc, yc, np.where(valid, cells[r], np.nan).T, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        ax.set_ylim(512, 0); ax.set_xlim(0, 512); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.045, f"{np.nanmean(cells[r][valid]):.0f} px", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=9, weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#555", alpha=0.92))
        if ti == 0:
            ax.set_title(RLAB[ci], fontsize=15, weight="bold", pad=8)
    p0 = axmap[ti][0].get_position()
    fig.text(0.028, 0.5 * (p0.y0 + p0.y1), tlab, rotation=90, ha="center", va="center",
             fontsize=15, weight="bold", color="#222")
    p4 = axmap[ti][4].get_position()
    cax = fig.add_axes([0.905, p4.y0, 0.013, p4.y1 - p4.y0])
    cb = fig.colorbar(im, cax=cax); cb.set_ticks([vmin, (vmin + vmax) / 2, vmax])
    cb.set_ticklabels([f"{int(vmin)}", f"{int((vmin + vmax) / 2)}", f"{int(vmax)}"]); cb.ax.tick_params(labelsize=8)
    cax.text(2.1, 1.0, "less\ncovered", transform=cax.transAxes, fontsize=7.5, color="#9e1b1b", va="top", ha="left")
    cax.text(2.1, 0.0, "better\ncovered", transform=cax.transAxes, fontsize=7.5, color="#1a7a33", va="bottom", ha="left")
    cax.set_ylabel("motion-coverage distance (px)", fontsize=8)

fig.savefig(OUT / "protoA12b_spatial_distance.png", dpi=155, bbox_inches="tight")
print("wrote", OUT / "protoA12b_spatial_distance.png")
