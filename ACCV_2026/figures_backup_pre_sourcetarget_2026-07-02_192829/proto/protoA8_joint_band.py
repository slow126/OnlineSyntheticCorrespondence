"""INTUITION figure: WHY over-shooting motion drops JOINT coverage (geometric, at-a-glance).

The coverage drop at 2x is a JOINT [position, motion] effect -- it does NOT live in the
flow marginal (there, bigger motion is monotonically better coverage). The faithful 2-D
carrier is (y, dy): vertical position in the frame vs vertical motion. In a driving scene
the ground plane couples them -- KITTI lies on a tilted BAND (corr(y,dy)=0.50): horizon
barely moves (dy~-3px), the road near the camera moves a lot (dy~+22px).

The synthetic source is itself a tight band (corr~0.91) whose STEEPNESS scales with the
magnitude knob. The source must land ON KITTI's band:
  0.25x  band too FLAT   -> never reaches the road's motion -> big gaps (under-fill)
  1-1.5x band MATCHED    -> source sits on KITTI's band      -> covered (peak)
  2x     band too STEEP  -> over-shoots the road's motion    -> points leave the band,
                            density inside it thins -> gaps re-open (coverage drops)

Red "reach" lines = each poorly-covered KITTI point to its nearest source sample = the
quantity d_{B->T} averages. This is the geometric why behind the inverted-U.

Output: ACCV_2026/figures/proto/protoA8_joint_band.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

CACHE = Path("/mnt/nvme_1tb_b/coverage_vectors")
OUT = Path("ACCV_2026/figures/proto"); OUT.mkdir(parents=True, exist_ok=True)
W = H = 512
rng = np.random.default_rng(1)

# full ladder sweep: band tilts up to the match, then over-shoots
RUNGS = [("m025", "0.25x"), ("m050", "0.5x"), ("m100", "1x"), ("m150", "1.5x"), ("m200", "2x")]
TAG = ["too flat", "filling in", "nearly on band", "ON THE BAND — peak", "too steep — over-shoots"]
TAGCOL = ["#8a8a8a", "#7f9a52", "#4f9a57", "#1a7a33", "#a14a00"]
PEAK = 3  # 1.5x


def load(name, n):
    v = np.load(CACHE / f"{name}_flow.npy").astype(np.float64)
    return v[rng.choice(len(v), min(n, len(v)), replace=False)]


# KITTI target: dense translucent context + a probe set for the reach lines
tgt_ctx = load("kitti2015_val", 9000)          # [x,y,dx,dy]
tgt_probe = load("kitti2015_val", 1400)

# normalized (y, dy) for honest nearest-neighbour reach (matches the paper's metric axes)
def ydy_norm(v):
    return np.column_stack([2 * v[:, 1] / H - 1, 2 * v[:, 3] / H])

tp_n = ydy_norm(tgt_probe)

DYLIM = (-28, 66)   # vertical-motion axis (px)
YLIM = (512, 70)    # frame position, inverted: top of frame at top

# --- clean KITTI band as a filled density (the "target manifold") ---
dy_edges = np.linspace(*DYLIM, 90)
y_edges = np.linspace(70, 512, 90)
dyc = 0.5 * (dy_edges[:-1] + dy_edges[1:]); yc = 0.5 * (y_edges[:-1] + y_edges[1:])
Ht, _, _ = np.histogram2d(tgt_ctx[:, 3], tgt_ctx[:, 1], bins=[dy_edges, y_edges])
Ht = gaussian_filter(Ht.T, 1.4); Ht /= Ht.max()


def centerline(v, ymid, hw=18):
    return np.array([np.nanmean(v[np.abs(v[:, 1] - ym) <= hw, 3])
                     if np.any(np.abs(v[:, 1] - ym) <= hw) else np.nan for ym in ymid])


ymid = np.linspace(110, 500, 22)
kit_dl = centerline(tgt_ctx, ymid)

fig, axes = plt.subplots(1, 5, figsize=(20, 6.1), sharex=True, sharey=True)
reaches = []
for k, ((r, lab), ax) in enumerate(zip(RUNGS, axes)):
    src = load(f"kitti_{r}_hq_train", 9000)
    src_dots = src[rng.choice(len(src), 600, replace=False)]
    d, idx = cKDTree(ydy_norm(src)).query(tp_n, k=1)
    reaches.append(d.mean())
    src_dl = centerline(src, ymid)

    # KITTI band: filled blue density (the target manifold), FIXED across panels
    ax.contourf(dyc, yc, Ht, levels=[0.03, 0.10, 0.30, 0.70, 1.0], cmap="Blues", alpha=0.45, zorder=1)

    # the GAP: wedge between source band and KITTI band -- red=under-shoot, orange=over-shoot
    ok = ~(np.isnan(kit_dl) | np.isnan(src_dl))
    ax.fill_betweenx(ymid[ok], src_dl[ok], kit_dl[ok], where=src_dl[ok] < kit_dl[ok],
                     color="#d62728", alpha=0.28, zorder=2, label="missing motion (under)")
    ax.fill_betweenx(ymid[ok], src_dl[ok], kit_dl[ok], where=src_dl[ok] >= kit_dl[ok],
                     color="#e8821e", alpha=0.30, zorder=2, label="wasted motion (over)")

    # source dots (density visible -> thinning at 2x) + the two band centerlines
    ax.scatter(src_dots[:, 3], src_dots[:, 1], s=9, c="#e8821e", alpha=0.40, edgecolors="none", zorder=3)
    ax.plot(kit_dl[ok], ymid[ok], color="#15406b", lw=3.0, zorder=5, label="KITTI band")
    ax.plot(src_dl[ok], ymid[ok], color="#b5530a", lw=3.0, ls="--", zorder=6, label="source band")

    ax.set_xlim(*DYLIM); ax.set_ylim(*YLIM)
    ax.axvline(0, color="k", lw=0.4, alpha=0.25)
    ax.set_title(f"{lab}\n{TAG[k]}", fontsize=12, color=TAGCOL[k],
                 weight="bold" if k == PEAK else "normal", pad=6, linespacing=1.3)
    ax.set_xlabel("vertical motion  dy  (px)", fontsize=10)
    ax.tick_params(labelsize=8.5)
    ax.text(0.95, 0.05, f"$d_{{B\\to T}}$ = {d.mean():.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color=TAGCOL[k],
            weight="bold", bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=TAGCOL[k], alpha=0.95))
    # frame the peak panel
    if k == PEAK:
        for sp in ax.spines.values():
            sp.set_edgecolor("#1a7a33"); sp.set_linewidth(3.2)

axes[0].set_ylabel("position in frame", fontsize=11)
axes[0].text(0.04, 0.97, "horizon (top): barely moves", transform=axes[0].transAxes,
             fontsize=8.2, color="#15406b", va="top", ha="left", style="italic")
axes[0].text(0.04, 0.03, "road / near camera: moves a lot", transform=axes[0].transAxes,
             fontsize=8.2, color="#15406b", va="bottom", ha="left", style="italic")

fig.legend(handles=[
    Line2D([0], [0], color="#15406b", lw=2.8, label="KITTI band  (motion grows toward the road)"),
    Line2D([0], [0], color="#b5530a", lw=2.8, ls="--", label="source band  (slope = magnitude knob)"),
    Patch(facecolor="#d62728", edgecolor="none", alpha=0.35, label="missing motion (under-shoot)"),
    Patch(facecolor="#e8821e", edgecolor="none", alpha=0.4, label="wasted motion (over-shoot)"),
], loc="lower center", ncol=4, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.01))

fig.suptitle("Coverage is JOINT: the source must put the right motion at the right place in the frame",
             fontsize=15.5, weight="bold", y=0.985)
fig.text(0.5, 0.915,
         "KITTI motion grows from horizon to road (the tilted band).  Scaling the source tilts ITS band: the gap "
         "closes as the slope rises to the match (1.5×), then re-opens on the other side as it over-shoots (2×).",
         ha="center", fontsize=10.5, color="#444")

fig.subplots_adjust(left=0.058, right=0.992, top=0.815, bottom=0.16, wspace=0.05)
fig.savefig(OUT / "protoA8_joint_band.png", dpi=160, bbox_inches="tight")
print("wrote", OUT / "protoA8_joint_band.png")
print("panel mean reach (y,dy norm):", dict(zip([l for _, l in RUNGS], np.round(reaches, 4))))
