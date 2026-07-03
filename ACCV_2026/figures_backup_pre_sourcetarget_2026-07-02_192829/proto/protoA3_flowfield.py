"""PROTOTYPE: compare datasets by WHERE motion occurs + HOW MUCH / WHICH DIRECTION.

Instead of embedding the 4-D cloud (where spatial variance dominates and motion
washes out), condition on position: bin flow vectors into an image-space grid and
render each cell's MEAN flow as standard optical-flow color (hue = direction,
brightness = magnitude). A FIXED magnitude scale across all panels makes the ladder
read as "same spatial pattern + same hue, just brighter" -> over-shoot = too bright
relative to the target.

Row 1: target + each rung as a flow-color field (fixed scale).
Row 2: per-cell magnitude RATIO source/target (1=match, >1 over-shoot, <1 under) -
       localizes that the over-shoot is ~uniform across position (right where, wrong how-much).

Output: ACCV_2026/figures/proto/protoA3_flowfield.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

CACHE = Path("/mnt/nvme_1tb_b/coverage_vectors")
OUT = Path("ACCV_2026/figures/proto"); OUT.mkdir(parents=True, exist_ok=True)
RUNGS = ["m025", "m050", "m100", "m150", "m200"]
RUNGLAB = ["0.25x", "0.5x", "1x", "1.5x", "2x"]
G = 40                       # spatial grid GxG over the 512x512 frame
MAXMAG = 45.0                # fixed px magnitude for brightness normalization (shared)
MINCOUNT = 20                # mask cells with too few vectors
rng = np.random.default_rng(0)


def load(name):
    return np.load(CACHE / f"{name}_flow.npy").astype(np.float32)  # pixel [x,y,dx,dy]


def grid_mean_flow(v):
    """mean (dx,dy) per spatial cell + count, over the 512x512 frame."""
    ix = np.clip((v[:, 0] / 512 * G).astype(int), 0, G - 1)
    iy = np.clip((v[:, 1] / 512 * G).astype(int), 0, G - 1)
    flat = iy * G + ix
    cnt = np.bincount(flat, minlength=G * G).astype(float)
    sx = np.bincount(flat, weights=v[:, 2], minlength=G * G)
    sy = np.bincount(flat, weights=v[:, 3], minlength=G * G)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = (sx / cnt).reshape(G, G)
        my = (sy / cnt).reshape(G, G)
    cnt = cnt.reshape(G, G)
    return mx, my, cnt


def flow_to_rgb(U, V, cnt, maxmag):
    ang = (np.arctan2(V, U) + np.pi) / (2 * np.pi)
    mag = np.clip(np.hypot(U, V) / maxmag, 0, 1)
    hsv = np.stack([ang, np.ones_like(ang), mag], axis=-1)
    rgb = mcolors.hsv_to_rgb(np.nan_to_num(hsv))
    rgb[cnt < MINCOUNT] = 1.0   # mask sparse cells -> white
    return rgb


target = load("kitti2015_val")
tmx, tmy, tcnt = grid_mean_flow(target)
tmag = np.hypot(tmx, tmy)

fig, axes = plt.subplots(2, 6, figsize=(19, 6.6))

# ---- Row 1: flow-color fields (fixed scale) ----
panels = [("KITTI target", target)] + [(RUNGLAB[i], load(f"kitti_{RUNGS[i]}_hq_train")) for i in range(5)]
mags = {}
for j, (lab, v) in enumerate(panels):
    mx, my, cnt = grid_mean_flow(v)
    mags[j] = (np.hypot(mx, my), cnt)
    ax = axes[0, j]
    ax.imshow(flow_to_rgb(mx, my, cnt, MAXMAG), origin="upper")
    ax.set_title(lab + ("" if j == 0 else " source"), fontsize=11.5)
    ax.set_xticks([]); ax.set_yticks([])
axes[0, 0].set_ylabel("flow-color field\n(hue=dir, bright=mag)", fontsize=9)

# ---- Row 2: magnitude RATIO source/target per cell ----
axes[1, 0].axis("off")
axes[1, 0].text(0.5, 0.5, "magnitude ratio\nsource / target\n\n1 = match\n>1 over-shoot\n<1 under",
                ha="center", va="center", fontsize=10, transform=axes[1, 0].transAxes)
for j in range(1, 6):
    smag, scnt = mags[j]
    ratio = np.where((tcnt >= MINCOUNT) & (scnt >= MINCOUNT) & (tmag > 1e-3), smag / tmag, np.nan)
    ax = axes[1, j]
    im = ax.imshow(ratio, origin="upper", cmap="RdBu_r", norm=mcolors.LogNorm(vmin=0.25, vmax=4.0))
    med = np.nanmedian(ratio)
    ax.set_title(f"{RUNGLAB[j-1]}  (median {med:.2f}x)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axes[1, 1:].tolist(), fraction=0.012, pad=0.01, label="source/target |flow|")

fig.suptitle("Where motion occurs + how much / which direction:  flow-color field (top) and "
             "per-location magnitude ratio (bottom)", fontsize=13.5, weight="bold")
fig.savefig(OUT / "protoA3_flowfield.png", dpi=150, bbox_inches="tight")
print("wrote", OUT / "protoA3_flowfield.png")
print("per-cell magnitude ratio median (source/target):",
      {RUNGLAB[j-1]: round(float(np.nanmedian(np.where((tcnt >= MINCOUNT) & (mags[j][1] >= MINCOUNT) & (tmag > 1e-3),
       mags[j][0] / tmag, np.nan))), 2) for j in range(1, 6)})
