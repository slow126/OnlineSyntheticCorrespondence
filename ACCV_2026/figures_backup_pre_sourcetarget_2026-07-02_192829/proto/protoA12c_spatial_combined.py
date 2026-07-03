"""Compact spatial coverage map: both KITTI runs pooled into one row (relative px).

Same relative encoding as protoA12 (each spot vs. its own across-magnitude average,
signed px), but KITTI-2015 and KITTI-2012 are POOLED into a single KITTI target so the
panel is one row of five maps instead of two -- smaller, for use as Fig.8(c).

green = better covered than that spot's average across magnitudes
red   = less covered
Output: ACCV_2026/figures/proto/protoA12c_spatial_combined.png
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
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter

CACHE = Path("/mnt/nvme_1tb_b/coverage_vectors")
OUT = Path("ACCV_2026/figures/proto"); OUT.mkdir(parents=True, exist_ok=True)
W = H = 512
NB = 26
MINCOUNT = 12
rng = np.random.default_rng(1)
RUNGS = ["m025", "m050", "m100", "m150", "m200"]
RLAB = ["0.25×", "0.5×", "1×", "1.5×", "2×"]
TARGETS = ["kitti2015_val", "kitti2012_val"]


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


def load_full(name):
    return np.load(CACHE / f"{name}_flow.npy").astype(np.float64)


edges = np.linspace(0, 512, NB + 1)
xc = yc = 0.5 * (edges[:-1] + edges[1:])

# pool both KITTI targets into one
traw = np.concatenate([load_raw(t, 22000) for t in TARGETS], axis=0)
tnn = norm(traw)

cells = {}
for r in RUNGS:
    src = norm(load_full(f"kitti_{r}_hq_train"))            # FULL source -> d_{B->T} at the coverage-law scale
    d, _ = cKDTree(src).query(tnn, k=1)                     # d = joint mean-NN distance = d_{B->T}
    del src
    stat, _, _, _ = binned_statistic_2d(traw[:, 0], traw[:, 1], d, "mean", bins=[edges, edges])
    cnt, _, _, _ = binned_statistic_2d(traw[:, 0], traw[:, 1], d, "count", bins=[edges, edges])
    cells[r] = np.where(cnt >= MINCOUNT, stat, np.nan)
valid = np.all([np.isfinite(cells[r]) for r in RUNGS], axis=0)
best = np.nanmin(np.stack([cells[r] for r in RUNGS]), axis=0)   # each spot's BEST (lowest d_BT) magnitude
deltas = {r: nan_smooth(np.where(valid, np.maximum(cells[r] - best, 0.0), np.nan)) for r in RUNGS}
vmax = np.nanpercentile(np.concatenate([deltas[r][np.isfinite(deltas[r])] for r in RUNGS]), 96)

cmap = plt.get_cmap("RdYlGn_r").copy(); cmap.set_bad("#ededed")

YTOP = 120   # crop empty sky (no KITTI GT above ~y=120)
fig = plt.figure(figsize=(14.5, 3.2))
gs = gridspec.GridSpec(1, 5, left=0.012, right=0.93, top=0.86, bottom=0.04, wspace=0.05)
for ci, r in enumerate(RUNGS):
    ax = fig.add_subplot(gs[0, ci])
    im = ax.pcolormesh(xc, yc, deltas[r].T, cmap=cmap, vmin=0, vmax=vmax, shading="auto")
    ax.set_ylim(512, YTOP); ax.set_xlim(0, 512); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(RLAB[ci], fontsize=14, weight="bold", pad=6)

cax = fig.add_axes([0.94, 0.10, 0.011, 0.70])
cb = fig.colorbar(im, cax=cax); cb.set_ticks([0, vmax])
cb.set_ticklabels(["0", f"{vmax * 1e3:.0f}"]); cb.ax.tick_params(labelsize=8.5)
cax.text(3.1, 1.0, "less\ncovered", transform=cax.transAxes, fontsize=9, color="#9e1b1b", va="top", ha="left")
cax.text(3.1, 0.0, "better\ncovered", transform=cax.transAxes, fontsize=9, color="#0b6b2e", va="bottom", ha="left")
cax.set_xlabel(r"$d_{B\to T}$" "\n" r"$(\times10^{-3})$", fontsize=7.5, labelpad=3)

fig.savefig(OUT / "protoA12c_spatial_combined.png", dpi=170, bbox_inches="tight")
print("wrote", OUT / "protoA12c_spatial_combined.png")
