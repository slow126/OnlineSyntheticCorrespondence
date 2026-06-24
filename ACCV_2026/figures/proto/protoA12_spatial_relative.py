"""BOTH AXES AT ONCE, with the magnitude effect amplified (relative spatial map).

protoA11 painted ABSOLUTE joint coverage on the frame, but two things crush the rung-to-
rung signal: (1) the joint distance is position-dominated (a constant offset every rung),
and (2) the spatial range (road vs horizon, ~3x) dwarfs the cross-rung range (~12% for
KITTI-2015). So an absolute scale can't show "which magnitude is better, where".

This version fixes both:
  * metric = FLOW residual at the matched position (px): joint NN picks the same-place
    source, we color by ONLY the flow mismatch -> the constant position term is gone, units
    are interpretable pixels.
  * color = each cell RELATIVE to its own across-rung average -> the spatial baseline is
    gone too, leaving only the magnitude effect. Green = this magnitude covers that spot
    BETTER than its average; red = WORSE.

Result: 0.25x and 2x panels go red (under-/over-shoot), the matched rung goes green --
the inverted-U as a spatial pattern. Per-target color scale (KITTI-2015 varies less than
KITTI-2012, shown honestly via the absolute px label on each panel).

Output: ACCV_2026/figures/proto/protoA12_spatial_relative.png
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
NB = 30
MINCOUNT = 8
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


def nan_smooth(a, sigma=1.0):
    m = np.isfinite(a).astype(float); a0 = np.where(np.isfinite(a), a, 0.0)
    num = gaussian_filter(a0, sigma); den = gaussian_filter(m, sigma)
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0.15)
    return out


sraw = {r: load_raw(f"kitti_{r}_hq_train", 80000) for r in RUNGS}
sn = {r: norm(sraw[r]) for r in RUNGS}
edges = np.linspace(0, 512, NB + 1)
xc = yc = 0.5 * (edges[:-1] + edges[1:])

cmap = LinearSegmentedColormap.from_list("rel", ["#0b6b2e", "#7fc97f", "#f7f7f7", "#ef8a62", "#9e1b1b"])
cmap.set_bad("#ededed")

fig = plt.figure(figsize=(19, 8.6))
gs = gridspec.GridSpec(2, 5, left=0.07, right=0.9, top=0.93, bottom=0.06, hspace=0.16, wspace=0.06)
axmap = [[None] * 5 for _ in range(2)]
ims = []
for ti, (tname, tlab) in enumerate(TARGETS):
    traw = load_raw(tname, 24000); tnn = norm(traw)
    cells = {}
    for r in RUNGS:
        d, idx = cKDTree(sn[r]).query(tnn, k=1)                                 # joint NN -> same-place source
        fres = np.hypot(traw[:, 2] - sraw[r][idx, 2], traw[:, 3] - sraw[r][idx, 3])  # flow residual (px)
        stat, _, _, _ = binned_statistic_2d(traw[:, 0], traw[:, 1], fres, "mean", bins=[edges, edges])
        cnt, _, _, _ = binned_statistic_2d(traw[:, 0], traw[:, 1], fres, "count", bins=[edges, edges])
        cells[r] = np.where(cnt >= MINCOUNT, stat, np.nan)
    valid = np.all([np.isfinite(cells[r]) for r in RUNGS], axis=0)              # cells defined in every rung
    base = np.nanmean(np.stack([cells[r] for r in RUNGS]), axis=0)
    deltas = {r: np.where(valid, cells[r] - base, np.nan) for r in RUNGS}
    deltas = {r: nan_smooth(deltas[r], 0.9) for r in RUNGS}
    vmax = np.nanpercentile(np.abs(np.concatenate([deltas[r][np.isfinite(deltas[r])] for r in RUNGS])), 93)
    norm_div = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    for ci, r in enumerate(RUNGS):
        ax = fig.add_subplot(gs[ti, ci]); axmap[ti][ci] = ax
        im = ax.pcolormesh(xc, yc, deltas[r].T, cmap=cmap, norm=norm_div, shading="auto")
        ax.set_ylim(512, 0); ax.set_xlim(0, 512); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        if ti == 0:
            ax.set_title(RLAB[ci], fontsize=15, weight="bold", pad=8)
    p0 = axmap[ti][0].get_position()
    fig.text(0.028, 0.5 * (p0.y0 + p0.y1), tlab, rotation=90, ha="center", va="center",
             fontsize=15, weight="bold", color="#222")
    # per-row colorbar (scales differ per target)
    p4 = axmap[ti][4].get_position()
    cax = fig.add_axes([0.905, p4.y0, 0.012, p4.y1 - p4.y0])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([-vmax, 0, vmax])
    cb.set_ticklabels([f"$-${vmax:.1f}", "0", f"$+${vmax:.1f}"]); cb.ax.tick_params(labelsize=7.5)
    cax.text(3.3, 1.0, "less\ncovered", transform=cax.transAxes, fontsize=7.5, color="#9e1b1b", va="top", ha="left")
    cax.text(3.3, 0.0, "better\ncovered", transform=cax.transAxes, fontsize=7.5, color="#0b6b2e", va="bottom", ha="left")
    cax.set_xlabel("px vs.\nspot avg", fontsize=7, labelpad=2)

fig.savefig(OUT / "protoA12_spatial_relative.png", dpi=155, bbox_inches="tight")
print("wrote", OUT / "protoA12_spatial_relative.png")
