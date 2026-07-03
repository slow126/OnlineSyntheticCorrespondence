"""PROTOTYPE: grid of raw-BFV UMAP panels, KITTI target vs each source rung.

ONE shared UMAP is fit on the raw normalized BFV [x,y,dx,dy] (no z-score) of the
target + all 5 rungs, so the target sits in the SAME place in every panel and the
panels are directly comparable. Each panel highlights the target (blue) against one
rung (orange), with the full manifold faint-gray for context.

Because the raw BFV is position-dominated (panel-set protoA2_umap_raw), the manifold
is laid out spatially; this grid shows how each rung's mass occupies that shared
spatial manifold relative to the fixed target.

Output: ACCV_2026/figures/proto/protoA2_umap_grid.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap

CACHE = Path("/mnt/nvme_1tb_b/coverage_vectors")
OUT = Path("ACCV_2026/figures/proto"); OUT.mkdir(parents=True, exist_ok=True)
W = H = 512
RUNGS = ["m025", "m050", "m100", "m150", "m200"]
RUNGLAB = ["0.25x", "0.5x", "1x", "1.5x", "2x"]
rng = np.random.default_rng(0)


def norm_bfv(v):
    o = v.astype(np.float32).copy()
    o[:, 0] = 2 * o[:, 0] / W - 1; o[:, 1] = 2 * o[:, 1] / H - 1
    o[:, 2] = 2 * o[:, 2] / W;     o[:, 3] = 2 * o[:, 3] / H
    return o


def load(name, n):
    v = norm_bfv(np.load(CACHE / f"{name}_flow.npy"))
    return v[rng.choice(len(v), min(n, len(v)), replace=False)]


N = 5000
target = load("kitti2015_val", N)
rungs = [load(f"kitti_{r}_hq_train", N) for r in RUNGS]
clouds = [target] + rungs
X = np.vstack(clouds)                         # raw normalized BFV, NO z-score
grp = np.concatenate([[i] * len(c) for i, c in enumerate(clouds)])

emb = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="euclidean",
                random_state=0).fit_transform(X)
tgt = emb[grp == 0]
xlo, xhi = emb[:, 0].min() - 1, emb[:, 0].max() + 1
ylo, yhi = emb[:, 1].min() - 1, emb[:, 1].max() + 1

fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
axes = axes.ravel()
for j, (r, lab) in enumerate(zip(RUNGS, RUNGLAB)):
    ax = axes[j]
    ax.scatter(emb[:, 0], emb[:, 1], s=2, c="#dddddd", alpha=0.35, linewidths=0)  # full manifold context
    src = emb[grp == j + 1]
    ax.scatter(src[:, 0], src[:, 1], s=4, c="#e8801a", alpha=0.45, label=f"source {lab}", linewidths=0)
    ax.scatter(tgt[:, 0], tgt[:, 1], s=4, c="#1f6fb2", alpha=0.45, label="KITTI target", linewidths=0)
    ax.set_title(f"target vs {lab}", fontsize=13)
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=3, fontsize=8.5, loc="upper right")

# 6th panel: reference — all rungs colored by magnitude, target on top
ax = axes[5]
from matplotlib.colors import LinearSegmentedColormap
rcmap = LinearSegmentedColormap.from_list("m", ["#ffd9a0", "#e8801a", "#7a1500"])
for j in range(5):
    s = emb[grp == j + 1]
    ax.scatter(s[:, 0], s[:, 1], s=2, c=[rcmap(j / 4)], alpha=0.35, linewidths=0, label=RUNGLAB[j])
ax.scatter(tgt[:, 0], tgt[:, 1], s=3, c="#1f6fb2", alpha=0.5, linewidths=0, label="KITTI")
ax.set_title("all rungs (light->dark) + target", fontsize=12)
ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
ax.set_xticks([]); ax.set_yticks([])
ax.legend(markerscale=3, fontsize=7.5, loc="upper right", ncol=2)

fig.suptitle("Shared raw-BFV UMAP (no z-score): KITTI target (blue) vs each source rung (orange)",
             fontsize=14, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "protoA2_umap_grid.png", dpi=150, bbox_inches="tight")
print("wrote", OUT / "protoA2_umap_grid.png")
