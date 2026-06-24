"""BIG GRID: joint band-matching across {target} x {projection} x {magnitude rung}.

4 rows x 5 columns = 20 panels.
  Rows: KITTI-2015 (y,dy) | KITTI-2015 (x,dx) | KITTI-2012 (y,dy) | KITTI-2012 (x,dx)
  Cols: 0.25x 0.5x 1x 1.5x 2x

Story the grid makes at a glance:
  * (y,dy) rows  -> INVERTED-U: the gap closes to a peak then re-opens (over-shoot).
                   The peak SHIFTS with the target's motion scale:
                   KITTI-2012 (smaller motion) peaks at 1x, KITTI-2015 (larger) at 1.5x.
  * (x,dx) rows  -> MONOTONIC control: horizontal motion is ~symmetric, so over-shoot
                   still covers it -> no drop. The drop is SPECIFIC to the asymmetric,
                   bounded vertical ground-plane coupling, not generic.

Each panel: KITTI band (blue, fixed per row) vs source band (orange dashed, slope set by
magnitude); the wedge between them is the coverage gap (red=under, orange=over); orange
dots = source samples; number = joint reach d_{B->T} for that cell.

Output: ACCV_2026/figures/proto/protoA9_joint_grid.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

CACHE = Path("/mnt/nvme_1tb_b/coverage_vectors")
OUT = Path("ACCV_2026/figures/proto"); OUT.mkdir(parents=True, exist_ok=True)
W = H = 512
rng = np.random.default_rng(1)
RUNGS = ["m025", "m050", "m100", "m150", "m200"]
RLAB = ["0.25×", "0.5×", "1×", "1.5×", "2×"]


def load(name, n):
    v = np.load(CACHE / f"{name}_flow.npy").astype(np.float64)
    return v[rng.choice(len(v), min(n, len(v)), replace=False)]


def to_norm(v):  # [x,y,dx,dy] px -> normalized
    o = np.empty_like(v)
    o[:, 0] = 2 * v[:, 0] / W - 1; o[:, 1] = 2 * v[:, 1] / H - 1
    o[:, 2] = 2 * v[:, 2] / W;     o[:, 3] = 2 * v[:, 3] / H
    return o


def centerline(v, pos, mot, grid, hw=18):
    return np.array([np.nanmean(v[np.abs(v[:, pos] - g) <= hw, mot])
                     if np.any(np.abs(v[:, pos] - g) <= hw) else np.nan for g in grid])


# projection configs: orient 'V' = position on vertical axis (image-like rows),
#                      orient 'H' = position on horizontal axis (image-like cols)
PROJ = {
    "ydy": dict(pos=1, mot=3, cols=[1, 3], orient="V", motlab="vertical motion  dy (px)",
                poslim=(512, 70), motlim=(-28, 66), grid=np.linspace(110, 500, 22)),
    "xdx": dict(pos=0, mot=2, cols=[0, 2], orient="H", poslab="horizontal position  x  (left→right)",
                poslim=(0, 512), motlim=(-62, 62), grid=np.linspace(20, 500, 22)),
}
TARGETS = [("kitti2015_val", "KITTI-2015"), ("kitti2012_val", "KITTI-2012")]
ROWS = [(t, tl, p) for (t, tl) in TARGETS for p in ["ydy", "xdx"]]

# preload source rungs once
src_px = {r: load(f"kitti_{r}_hq_train", 9000) for r in RUNGS}
src_dots = {r: src_px[r][rng.choice(9000, 300, replace=False)] for r in RUNGS}
src_norm = {r: to_norm(src_px[r]) for r in RUNGS}

fig = plt.figure(figsize=(20.5, 15.5))
gs = gridspec.GridSpec(4, 5, left=0.085, right=0.915, top=0.915, bottom=0.05,
                       hspace=0.30, wspace=0.07)

axmap = [[None] * 5 for _ in range(4)]
for ri, (tname, tlab, pk) in enumerate(ROWS):
    P = PROJ[pk]
    tctx = load(tname, 9000)
    tprobe = load(tname, 1400)
    tprobe_n = to_norm(tprobe)[:, P["cols"]]
    kit_cl = centerline(tctx, P["pos"], P["mot"], P["grid"])

    # KITTI density on the plotted axes (A=horizontal var, B=vertical var)
    if P["orient"] == "V":     # horiz=mot(dy), vert=pos(y)
        A, B, Alim, Blim = tctx[:, P["mot"]], tctx[:, P["pos"]], P["motlim"], P["poslim"]
    else:                      # horiz=pos(x), vert=mot(dx)
        A, B, Alim, Blim = tctx[:, P["pos"]], tctx[:, P["mot"]], P["poslim"], P["motlim"]
    ae = np.linspace(min(Alim), max(Alim), 80); be = np.linspace(min(Blim), max(Blim), 80)
    ac = 0.5 * (ae[:-1] + ae[1:]); bc = 0.5 * (be[:-1] + be[1:])
    Hd, _, _ = np.histogram2d(A, B, bins=[ae, be]); Hd = gaussian_filter(Hd.T, 1.4); Hd /= Hd.max()

    # reach per rung -> peak / shape
    reach = np.array([cKDTree(src_norm[r][:, P["cols"]]).query(tprobe_n, k=1)[0].mean() for r in RUNGS])
    peak = int(np.argmin(reach))
    inverted = 0 < peak < 4

    for ci, r in enumerate(RUNGS):
        ax = fig.add_subplot(gs[ri, ci]); axmap[ri][ci] = ax
        src_cl = centerline(src_px[r], P["pos"], P["mot"], P["grid"])
        ok = ~(np.isnan(kit_cl) | np.isnan(src_cl))
        g = P["grid"]

        ax.contourf(ac, bc, Hd, levels=[0.03, 0.10, 0.30, 0.70, 1.0], cmap="Blues", alpha=0.45, zorder=1)
        if P["orient"] == "V":
            ax.fill_betweenx(g[ok], src_cl[ok], kit_cl[ok], where=src_cl[ok] < kit_cl[ok], color="#d62728", alpha=0.28, zorder=2)
            ax.fill_betweenx(g[ok], src_cl[ok], kit_cl[ok], where=src_cl[ok] >= kit_cl[ok], color="#e8821e", alpha=0.30, zorder=2)
            ax.scatter(src_dots[r][:, P["mot"]], src_dots[r][:, P["pos"]], s=7, c="#e8821e", alpha=0.4, edgecolors="none", zorder=3)
            ax.plot(kit_cl[ok], g[ok], color="#15406b", lw=2.6, zorder=5)
            ax.plot(src_cl[ok], g[ok], color="#b5530a", lw=2.6, ls="--", zorder=6)
            ax.set_xlim(*P["motlim"]); ax.set_ylim(*P["poslim"]); ax.axvline(0, color="k", lw=0.4, alpha=0.2)
        else:
            ax.fill_between(g[ok], src_cl[ok], kit_cl[ok], where=src_cl[ok] < kit_cl[ok], color="#d62728", alpha=0.28, zorder=2)
            ax.fill_between(g[ok], src_cl[ok], kit_cl[ok], where=src_cl[ok] >= kit_cl[ok], color="#e8821e", alpha=0.30, zorder=2)
            ax.scatter(src_dots[r][:, P["pos"]], src_dots[r][:, P["mot"]], s=7, c="#e8821e", alpha=0.4, edgecolors="none", zorder=3)
            ax.plot(g[ok], kit_cl[ok], color="#15406b", lw=2.6, zorder=5)
            ax.plot(g[ok], src_cl[ok], color="#b5530a", lw=2.6, ls="--", zorder=6)
            ax.set_xlim(*P["poslim"]); ax.set_ylim(*P["motlim"]); ax.axhline(0, color="k", lw=0.4, alpha=0.2)

        ax.tick_params(labelsize=7.5)
        is_best = (ci == peak)
        ncol = "#1a7a33" if is_best else "#444"
        ax.text(0.95, 0.05, f"{reach[ci]:.3f}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color=ncol, weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=ncol, alpha=0.9))
        if inverted and is_best:
            for sp in ax.spines.values():
                sp.set_edgecolor("#1a7a33"); sp.set_linewidth(3.0)
            ax.text(0.5, 0.96, "peak", transform=ax.transAxes, ha="center", va="top",
                    fontsize=8.5, color="#1a7a33", weight="bold")

        # column headers (top row only)
        if ri == 0:
            ax.set_title(RLAB[ci], fontsize=14, weight="bold", pad=10)
        # axis labels only on outer panels
        if ci == 0:
            ax.set_ylabel("position in frame\n(top→bottom)" if P["orient"] == "V" else "horizontal motion\ndx (px)", fontsize=8.5)
        if ri == 3:
            ax.set_xlabel(P.get("motlab") if P["orient"] == "V" else P.get("poslab"), fontsize=8.5)
        elif P["orient"] == "V":
            ax.set_xlabel(P["motlab"], fontsize=7.5, labelpad=1)

    # right-edge verdict per row
    rax = axmap[ri][4]
    if inverted:
        verdict = f"INVERTED-U\npeak {RLAB[peak]}\nover-shoot\n→ drop"
        vc = "#1a7a33"
    else:
        verdict = "MONOTONIC\n(control)\nno over-shoot\ndrop here"
        vc = "#7a4a1a"
    rax.text(1.06, 0.5, verdict, transform=rax.transAxes, ha="left", va="center",
             fontsize=9.5, color=vc, weight="bold")

# big rotated target labels on the far left, spanning each target's two rows
for ti, (_, tlab) in enumerate(TARGETS):
    p_top = axmap[2 * ti][0].get_position(); p_bot = axmap[2 * ti + 1][0].get_position()
    yc = 0.5 * (p_top.y1 + p_bot.y0)
    fig.text(0.018, yc, tlab, rotation=90, ha="center", va="center", fontsize=16, weight="bold", color="#222")
# projection tag per row, just left of the y-axis label
for ri, (_, _, pk) in enumerate(ROWS):
    pos = axmap[ri][0].get_position()
    tag = "vertical\n(y, dy)" if pk == "ydy" else "horizontal\n(x, dx)"
    fig.text(0.052, 0.5 * (pos.y0 + pos.y1), tag, rotation=90, ha="center", va="center",
             fontsize=10, weight="bold", color="#15406b" if pk == "ydy" else "#7a4a1a")

fig.legend(handles=[
    Line2D([0], [0], color="#15406b", lw=2.8, label="KITTI band (target)"),
    Line2D([0], [0], color="#b5530a", lw=2.8, ls="--", label="source band (slope = magnitude)"),
    Patch(facecolor="#d62728", edgecolor="none", alpha=0.35, label="missing motion (under-shoot)"),
    Patch(facecolor="#e8821e", edgecolor="none", alpha=0.4, label="wasted motion (over-shoot)"),
], loc="lower center", ncol=4, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.002))

fig.suptitle("Joint coverage is band-matching: the over-shoot drop is specific to the vertical (y,dy) coupling, "
             "and its peak shifts with the target",
             fontsize=15.5, weight="bold", y=0.97)
fig.text(0.5, 0.94,
         "(y,dy) rows: the gap closes to a peak then re-opens as motion over-shoots — peak at 1× for KITTI-2012, "
         "1.5× for the larger-motion KITTI-2015.   (x,dx) rows: monotonic (control) — no over-shoot drop in this range.",
         ha="center", fontsize=10.5, color="#444")

fig.savefig(OUT / "protoA9_joint_grid.png", dpi=150, bbox_inches="tight")
print("wrote", OUT / "protoA9_joint_grid.png")
