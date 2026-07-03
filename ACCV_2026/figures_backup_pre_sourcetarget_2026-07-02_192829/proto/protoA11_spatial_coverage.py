"""BOTH AXES AT ONCE: joint coverage painted onto the image frame.

The split figures (protoA8-10) spend the 2-D canvas on ONE position axis + its motion.
To show BOTH at once we give the canvas to position (x,y) = the actual frame, and encode
the FULL 2-D motion coverage (dx,dy) as color:

  for every KITTI point [x,y,dx,dy] we compute its 4-D reach to the nearest source sample
  (= d_{B->T}), then average that reach in each spatial cell of the frame.

  green cell = the source reproduces the real motion there (covered)
  red   cell = the source fails to reproduce the real motion there (uncovered gap)

Every dimension is represented: (x,y) -> where in the frame, (dx,dy) -> the coverage color.

Reading: the ROAD (bottom of frame) is the coverage bottleneck (largest, fastest motion);
it goes green as the source magnitude rises to the match, then reddens again on over-shoot.
The horizon (top) is easy throughout. Trade-off vs the split: this shows WHERE coverage
fails but blends the axes, so it cannot isolate WHICH motion component drives the drop.

Output: ACCV_2026/figures/proto/protoA11_spatial_coverage.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic_2d

CACHE = Path("/mnt/nvme_1tb_b/coverage_vectors")
OUT = Path("ACCV_2026/figures/proto"); OUT.mkdir(parents=True, exist_ok=True)
W = H = 512
NB = 20            # spatial grid cells per axis
MINCOUNT = 12      # mask cells with too few KITTI samples
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


src_n = {r: norm(load_raw(f"kitti_{r}_hq_train", 60000)) for r in RUNGS}
edges = np.linspace(0, 512, NB + 1)
xc = yc = 0.5 * (edges[:-1] + edges[1:])

# precompute spatial reach maps + mean reach per (target, rung)
maps = {}; means = {}
for tname, tlab in TARGETS:
    traw = load_raw(tname, 22000); tn = norm(traw)
    for r in RUNGS:
        reach, _ = cKDTree(src_n[r]).query(tn, k=1)
        stat, _, _, _ = binned_statistic_2d(traw[:, 0], traw[:, 1], reach, "mean", bins=[edges, edges])
        cnt, _, _, _ = binned_statistic_2d(traw[:, 0], traw[:, 1], reach, "count", bins=[edges, edges])
        stat = np.where(cnt >= MINCOUNT, stat, np.nan)
        maps[(tname, r)] = stat
        means[(tname, r)] = reach.mean()

allv = np.concatenate([m[~np.isnan(m)] for m in maps.values()])
vmin, vmax = np.percentile(allv, 4), np.percentile(allv, 94)

cmap = plt.get_cmap("RdYlGn_r").copy(); cmap.set_bad("#ececec")

fig = plt.figure(figsize=(18.5, 8.2))
gs = gridspec.GridSpec(2, 5, left=0.07, right=0.9, top=0.84, bottom=0.07, hspace=0.16, wspace=0.06)
axmap = [[None] * 5 for _ in range(2)]
for ti, (tname, tlab) in enumerate(TARGETS):
    best = int(np.argmin([means[(tname, r)] for r in RUNGS]))
    for ci, r in enumerate(RUNGS):
        ax = fig.add_subplot(gs[ti, ci]); axmap[ti][ci] = ax
        im = ax.pcolormesh(xc, yc, maps[(tname, r)].T, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        ax.set_ylim(512, 0); ax.set_xlim(0, 512); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.045, f"mean gap {means[(tname, r)]:.3f}", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=9, weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#555", alpha=0.9))
        if ti == 0:
            ax.set_title(RLAB[ci], fontsize=15, weight="bold", pad=8)
        if 0 < best < 4 and ci == best:
            for sp in ax.spines.values():
                sp.set_edgecolor("#1a7a33"); sp.set_linewidth(3.4)
            ax.text(0.5, 0.95, "best coverage", transform=ax.transAxes, ha="center", va="top",
                    fontsize=9, color="#0c5c22", weight="bold")
    # row label
    p = axmap[ti][0].get_position()
    fig.text(0.03, 0.5 * (p.y0 + p.y1), tlab, rotation=90, ha="center", va="center",
             fontsize=15, weight="bold", color="#222")

# horizon/road guide on the first panel (top of frame = horizon, bottom = road)
axmap[0][0].text(0.5, 0.94, "horizon (easy)", transform=axmap[0][0].transAxes, ha="center", va="top",
                 fontsize=8, color="#0c5c22", weight="bold",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
axmap[0][0].text(0.5, 0.18, "road (bottleneck)", transform=axmap[0][0].transAxes, ha="center", va="bottom",
                 fontsize=8, color="#7a1500", weight="bold",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

cax = fig.add_axes([0.915, 0.10, 0.013, 0.68])
cb = fig.colorbar(im, cax=cax)
cb.set_label("joint coverage gap  $d_{B\\to T}$  (reach to nearest source)", fontsize=10)
cax.text(0.5, 1.03, "uncovered", transform=cax.transAxes, ha="center", fontsize=8.5, color="#7a1500", weight="bold")
cax.text(0.5, -0.05, "covered", transform=cax.transAxes, ha="center", va="top", fontsize=8.5, color="#0c5c22", weight="bold")

fig.suptitle("Both axes at once: joint motion-coverage painted on the frame  "
             "(green = source reproduces the real motion, red = it does not)", fontsize=15.5, weight="bold", y=0.95)
fig.text(0.5, 0.88,
         "(x,y) = where in the frame · color = coverage of the full 2-D motion (dx,dy).  The road is the bottleneck: "
         "it greens as magnitude rises to the match, then reddens again on over-shoot.",
         ha="center", fontsize=10.5, color="#444")

fig.savefig(OUT / "protoA11_spatial_coverage.png", dpi=150, bbox_inches="tight")
print("wrote", OUT / "protoA11_spatial_coverage.png")
for tname, tlab in TARGETS:
    print(tlab, "mean gap:", {RLAB[i]: round(means[(tname, r)], 4) for i, r in enumerate(RUNGS)})
