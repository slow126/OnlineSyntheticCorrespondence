"""SPLAT-FIELD coverage: borrow the gaussian-splat aesthetic to show everything at once.

Each glyph is a motion "splat" placed on the frame:
  * position  = where in the frame (x, y)
  * elongation + orientation = the REAL KITTI motion there (direction & magnitude of mean flow)
  * color     = coverage -- how well the source reproduces that motion (relative to the
                spot's across-rung average; green = better than usual, red = worse)

The splat SHAPES are the same in every panel (they are KITTI's motion field); only the
COLOR changes across the magnitude rungs. So you read the scene's motion once, then watch
the road's long splats go red (under-shoot) -> green (match) -> red (over-shoot).

Output: ACCV_2026/figures/proto/protoA13_splat_coverage.png
"""
from pathlib import Path
import warnings
import numpy as np
warnings.filterwarnings("ignore", message="Mean of empty slice")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Ellipse
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter

CACHE = Path("/mnt/nvme_1tb_b/coverage_vectors")
OUT = Path("ACCV_2026/figures/proto"); OUT.mkdir(parents=True, exist_ok=True)
W = H = 512
NB = 17
CS = 512 / NB
MINCOUNT = 10
rng = np.random.default_rng(1)
RUNGS = ["m025", "m050", "m100", "m150", "m200"]
RLAB = ["0.25×", "0.5×", "1×", "1.5×", "2×"]
TARGETS = [("kitti2015_val", "KITTI-2015"), ("kitti2012_val", "KITTI-2012")]


def load_raw(name, n):
    v = np.load(CACHE / f"{name}_flow.npy").astype(np.float64)
    return v[rng.choice(len(v), min(n, len(v)), replace=False)]


def norm(v):
    o = np.empty_like(v)
    o[:, 0] = 2 * v[:, 0] / W - 1; o[:, 1] = 2 * v[:, 1] / H - 1
    o[:, 2] = 2 * v[:, 2] / W;     o[:, 3] = 2 * v[:, 3] / H
    return o


def binned(x, y, val, stat):
    s, _, _, _ = binned_statistic_2d(x, y, val, stat, bins=[edges, edges]); return s


sraw = {r: load_raw(f"kitti_{r}_hq_train", 80000) for r in RUNGS}
sn = {r: norm(sraw[r]) for r in RUNGS}
edges = np.linspace(0, 512, NB + 1)
ctr = 0.5 * (edges[:-1] + edges[1:])
GX, GY = np.meshgrid(ctr, ctr, indexing="ij")

cmap = LinearSegmentedColormap.from_list("rel", ["#0b6b2e", "#7fc97f", "#f4f4f4", "#ef8a62", "#9e1b1b"])

fig = plt.figure(figsize=(19.5, 9.6))
gs = gridspec.GridSpec(2, 5, left=0.065, right=0.9, top=0.85, bottom=0.04, hspace=0.12, wspace=0.05)
axmap = [[None] * 5 for _ in range(2)]
for ti, (tname, tlab) in enumerate(TARGETS):
    traw = load_raw(tname, 26000); tnn = norm(traw)
    cnt = binned(traw[:, 0], traw[:, 1], traw[:, 3], "count")
    mdx = binned(traw[:, 0], traw[:, 1], traw[:, 2], "mean")
    mdy = binned(traw[:, 0], traw[:, 1], traw[:, 3], "mean")
    mmag = binned(traw[:, 0], traw[:, 1], np.hypot(traw[:, 2], traw[:, 3]), "mean")
    # coverage: flow residual at matched position, per rung
    res = {}
    for r in RUNGS:
        d, idx = cKDTree(sn[r]).query(tnn, k=1)
        fres = np.hypot(traw[:, 2] - sraw[r][idx, 2], traw[:, 3] - sraw[r][idx, 3])
        res[r] = binned(traw[:, 0], traw[:, 1], fres, "mean")
    valid = (cnt >= MINCOUNT) & np.all([np.isfinite(res[r]) for r in RUNGS], axis=0)
    base = np.nanmean(np.stack([res[r] for r in RUNGS]), axis=0)
    delta = {r: res[r] - base for r in RUNGS}
    vmax = np.nanpercentile(np.abs([delta[r][valid] for r in RUNGS]), 92)
    dn = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    # splat length scaling (shared within target): longest motion ~ 1.8 cells
    magn = mmag[valid]; Lscale = 1.8 * CS / np.nanpercentile(magn, 92)
    absmean = {r: float(np.nanmean(res[r][valid])) for r in RUNGS}
    best = int(np.argmin([absmean[r] for r in RUNGS]))

    ii, jj = np.where(valid)
    ang = np.degrees(np.arctan2(-mdy, mdx))   # negate dy: y-axis is drawn inverted (image down)
    for ci, r in enumerate(RUNGS):
        ax = fig.add_subplot(gs[ti, ci]); axmap[ti][ci] = ax
        ax.set_facecolor("#fbfbfb")
        for i, j in zip(ii, jj):
            L = float(np.clip(mmag[i, j] * Lscale, 0.40 * CS, 2.4 * CS))
            e = Ellipse((GX[i, j], GY[i, j]), width=L, height=max(0.42 * L, 0.34 * CS),
                        angle=ang[i, j], facecolor=cmap(dn(delta[r][i, j])),
                        edgecolor="#555", lw=0.2, alpha=0.97)
            ax.add_patch(e)
        ax.set_xlim(0, 512); ax.set_ylim(512, 0); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.04, f"{absmean[r]:.1f}px", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9, weight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#555", alpha=0.9))
        if ti == 0:
            ax.set_title(RLAB[ci], fontsize=15, weight="bold", pad=8)
        if 0 < best < 4 and ci == best:
            for sp in ax.spines.values():
                sp.set_edgecolor("#0b6b2e"); sp.set_linewidth(3.4)
            ax.text(0.5, 0.95, "best", transform=ax.transAxes, ha="center", va="top",
                    fontsize=9.5, color="#0b6b2e", weight="bold")
    p0 = axmap[ti][0].get_position()
    fig.text(0.028, 0.5 * (p0.y0 + p0.y1), tlab, rotation=90, ha="center", va="center",
             fontsize=15, weight="bold", color="#222")
    p4 = axmap[ti][4].get_position()
    cax = fig.add_axes([0.905, p4.y0, 0.012, p4.y1 - p4.y0])
    import matplotlib.cm as cm
    cb = fig.colorbar(cm.ScalarMappable(norm=dn, cmap=cmap), cax=cax)
    cb.set_ticks([-vmax, 0, vmax]); cb.set_ticklabels(["better", "avg", "worse"]); cb.ax.tick_params(labelsize=7.5)

axmap[0][0].text(0.5, 0.93, "horizon", transform=axmap[0][0].transAxes, ha="center", va="top",
                 fontsize=8, color="#333", weight="bold", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))
axmap[0][0].text(0.5, 0.14, "road", transform=axmap[0][0].transAxes, ha="center", va="bottom",
                 fontsize=8, color="#333", weight="bold", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))

fig.suptitle("Motion-splat field: each splat is the real motion (elongation = direction & magnitude), "
             "colored by how well the source covers it", fontsize=14.5, weight="bold", y=0.95)
fig.text(0.5, 0.875,
         "Splat shapes = KITTI's motion field (fixed every panel).  Color = coverage vs that spot's across-rung average.  "
         "Long road splats: red at 0.25× (under) → green at match → red at 2× (over).  Number = abs. mean flow gap (px).",
         ha="center", fontsize=10.3, color="#444")

fig.savefig(OUT / "protoA13_splat_coverage.png", dpi=170, bbox_inches="tight")
print("wrote", OUT / "protoA13_splat_coverage.png")
