"""PROTOTYPE: UMAP of the magnitude ladder in the RAW normalized BFV space (NO z-score).

The paper's distances live in raw normalized BFV [x,y,dx,dy] (each dim mapped to
[-1,1] via 2*coord/512). Crucially the SCALES differ: position spans ~[-1,1] (range 2)
but flow at KITTI magnitudes spans ~[-0.1,0.1] (range ~0.2). So a raw Euclidean
embedding is dominated by spatial position. This script tests that directly:
embed raw BFV (faithful to the paper, no rescaling) and color 3 ways to see what
organizes the manifold.

  panel 1: by source (KITTI target vs the 5 rungs)   -> do rungs separate or overlap?
  panel 2: by spatial radius sqrt(x^2+y^2)            -> expect a clean gradient (position)
  panel 3: by flow magnitude |f|                       -> expect NO clean structure (motion buried)

Output: ACCV_2026/figures/proto/protoA2_umap_raw.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
clouds = [load("kitti2015_val", N)] + [load(f"kitti_{r}_hq_train", N) for r in RUNGS]
labels = ["KITTI (target)"] + RUNGLAB
X = np.vstack(clouds)                                   # RAW normalized BFV, NO z-score
grp = np.concatenate([[i] * len(c) for i, c in enumerate(clouds)])
spatial_r = np.hypot(X[:, 0], X[:, 1])                  # distance from image center
flow_mag = np.hypot(X[:, 2], X[:, 3])                   # motion magnitude

print("per-dim std (shows the scale imbalance that dominates raw Euclidean):")
print(f"  x={X[:,0].std():.3f} y={X[:,1].std():.3f} dx={X[:,2].std():.4f} dy={X[:,3].std():.4f}")

emb = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="euclidean",
                random_state=0).fit_transform(X)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# panel 1: by source
ax = axes[0]
rung_cmap = LinearSegmentedColormap.from_list("mag", ["#ffd9a0", "#e8801a", "#7a1500"])
colors = ["#1f6fb2"] + [rung_cmap(i / 4) for i in range(5)]
for i in [1, 2, 3, 4, 5, 0]:                            # target drawn last (on top)
    m = grp == i
    ax.scatter(emb[m, 0], emb[m, 1], s=3, alpha=0.4, c=[colors[i]], label=labels[i], linewidths=0)
ax.legend(markerscale=4, fontsize=9, loc="best")
ax.set_title("by source\n(target vs rungs)", fontsize=12)
ax.set_xticks([]); ax.set_yticks([])

# panel 2: by spatial radius
ax = axes[1]
sc = ax.scatter(emb[:, 0], emb[:, 1], s=3, alpha=0.5, c=spatial_r, cmap="viridis", linewidths=0)
fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="spatial radius sqrt(x^2+y^2)")
ax.set_title("by SPATIAL position\n(expect a clean gradient)", fontsize=12)
ax.set_xticks([]); ax.set_yticks([])

# panel 3: by flow magnitude
ax = axes[2]
sc = ax.scatter(emb[:, 0], emb[:, 1], s=3, alpha=0.5, c=flow_mag, cmap="magma",
                vmax=np.percentile(flow_mag, 99), linewidths=0)
fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="flow magnitude |f|")
ax.set_title("by FLOW magnitude\n(expect it buried / not organizing)", fontsize=12)
ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("UMAP of RAW normalized BFV [x,y,dx,dy] (no z-score) — what organizes the manifold?",
             fontsize=14, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "protoA2_umap_raw.png", dpi=150, bbox_inches="tight")
print("wrote", OUT / "protoA2_umap_raw.png")
