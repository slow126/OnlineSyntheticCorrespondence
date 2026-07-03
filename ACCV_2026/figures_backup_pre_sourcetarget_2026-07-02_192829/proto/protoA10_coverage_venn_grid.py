"""BIG VENN GRID: coverage as covered / uncovered / overshoot, {target}x{projection}x{rung}.

Replaces the mean-centerline + wedge of protoA9 with a faithful, Venn-like primitive:
for every KITTI point we ask "is there a source sample within radius eps?" (this IS the
d_{B->T} coverage metric). Three zones, like a 2-D Venn:
   GREEN  covered    = KITTI motion the source reaches            (overlap)
   RED    uncovered  = KITTI motion the source misses             (the coverage gap)
   ORANGE overshoot  = source motion KITTI never has              (wasted / off-target)
A faint blue outline = the KITTI "band" (its support) for context.

Rows: KITTI-2015 (y,dy) | KITTI-2015 (x,dx) | KITTI-2012 (y,dy) | KITTI-2012 (x,dx)
Cols: 0.25x 0.5x 1x 1.5x 2x

Honest reading: the down-then-up drop is STRONG for KITTI-2012 (y,dy) (red shrinks then
re-grows, orange blooms at 2x), WEAK for KITTI-2015 (y,dy), and the (x,dx) rows improve
MONOTONICALLY (red just keeps shrinking) -- the over-shoot drop is specific to the
asymmetric vertical ground-plane.

Output: ACCV_2026/figures/proto/protoA10_coverage_venn_grid.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

CACHE = Path("/mnt/nvme_1tb_b/coverage_vectors")
OUT = Path("ACCV_2026/figures/proto"); OUT.mkdir(parents=True, exist_ok=True)
W = H = 512
EPS = 0.020                       # coverage radius (normalized units); single value across grid
rng = np.random.default_rng(1)
RUNGS = ["m025", "m050", "m100", "m150", "m200"]
RLAB = ["0.25×", "0.5×", "1×", "1.5×", "2×"]
COV, UNC, OVR = "#3bb068", "#d62728", "#f08a1e"   # overlap green / gap red / waste orange


def load_raw(name, n):
    v = np.load(CACHE / f"{name}_flow.npy").astype(np.float64)
    return v[rng.choice(len(v), min(n, len(v)), replace=False)]


def norm(v):
    o = np.empty_like(v)
    o[:, 0] = 2 * v[:, 0] / W - 1; o[:, 1] = 2 * v[:, 1] / H - 1
    o[:, 2] = 2 * v[:, 2] / W;     o[:, 3] = 2 * v[:, 3] / H
    return o


PROJ = {
    "ydy": dict(pos=1, mot=3, cols=[1, 3], orient="V",
                motlim=(-30, 68), poslim=(512, 60), motlab="vertical motion  dy (px)"),
    "xdx": dict(pos=0, mot=2, cols=[0, 2], orient="H",
                poslim=(0, 512), motlim=(-66, 66), poslab="horizontal position  x  (left→right)"),
}
TARGETS = [("kitti2015_val", "KITTI-2015"), ("kitti2012_val", "KITTI-2012")]
ROWS = [(t, tl, p) for (t, tl) in TARGETS for p in ["ydy", "xdx"]]

# preload sources: full (for KD-trees) + a plotting subsample
src_full = {r: load_raw(f"kitti_{r}_hq_train", 60000) for r in RUNGS}
src_full_n = {r: norm(src_full[r]) for r in RUNGS}
src_plot_idx = {r: rng.choice(len(src_full[r]), 1400, replace=False) for r in RUNGS}

fig = plt.figure(figsize=(20.5, 15.6))
gs = gridspec.GridSpec(4, 5, left=0.085, right=0.905, top=0.915, bottom=0.055,
                       hspace=0.27, wspace=0.07)
axmap = [[None] * 5 for _ in range(4)]

for ri, (tname, tlab, pk) in enumerate(ROWS):
    P = PROJ[pk]; cols = P["cols"]
    tgt_full = load_raw(tname, 20000)
    tgt_full_n = norm(tgt_full)
    tidx = rng.choice(len(tgt_full), 1800, replace=False)
    tgt_plot, tgt_plot_n = tgt_full[tidx], tgt_full_n[tidx]
    ktree = cKDTree(tgt_full_n[:, cols])

    # KITTI band outline (its support) from a smoothed density
    if P["orient"] == "V":
        A, B, Al, Bl = tgt_full[:, P["mot"]], tgt_full[:, P["pos"]], P["motlim"], P["poslim"]
    else:
        A, B, Al, Bl = tgt_full[:, P["pos"]], tgt_full[:, P["mot"]], P["poslim"], P["motlim"]
    ae = np.linspace(min(Al), max(Al), 80); be = np.linspace(min(Bl), max(Bl), 80)
    ac = 0.5 * (ae[:-1] + ae[1:]); bc = 0.5 * (be[:-1] + be[1:])
    Hd, _, _ = np.histogram2d(A, B, bins=[ae, be]); Hd = gaussian_filter(Hd.T, 1.5); Hd /= Hd.max()

    covpct = []
    for ci, r in enumerate(RUNGS):
        ax = fig.add_subplot(gs[ri, ci]); axmap[ri][ci] = ax
        # coverage: KITTI -> nearest source within EPS
        dK, _ = cKDTree(src_full_n[r][:, cols]).query(tgt_plot_n[:, cols], k=1)
        covered = dK < EPS
        covpct.append(100 * covered.mean())
        # overshoot: plotted source -> nearest KITTI beyond EPS (motion KITTI never has)
        sp = src_full[r][src_plot_idx[r]]; spn = src_full_n[r][src_plot_idx[r]]
        dS, _ = ktree.query(spn[:, cols], k=1)
        over = dS >= EPS

        # faint KITTI band outline (support) for context
        ax.contour(ac, bc, Hd, levels=[0.05], colors="#15406b", linewidths=1.1, alpha=0.55, zorder=2)

        def XY(v):  # plotting coordinates for this projection
            return (v[:, P["mot"]], v[:, P["pos"]]) if P["orient"] == "V" else (v[:, P["pos"]], v[:, P["mot"]])

        ox, oy = XY(sp[over])
        cx, cy = XY(tgt_plot[covered]); ux, uy = XY(tgt_plot[~covered])
        ax.scatter(cx, cy, s=5, c=COV, alpha=0.5, edgecolors="none", zorder=3)      # covered (overlap)
        ax.scatter(ox, oy, s=7, c=OVR, alpha=0.75, edgecolors="none", zorder=4)     # overshoot (wasted)
        ax.scatter(ux, uy, s=8, c=UNC, alpha=0.85, edgecolors="none", zorder=5)     # uncovered (gap)

        if P["orient"] == "V":
            ax.set_xlim(*P["motlim"]); ax.set_ylim(*P["poslim"]); ax.axvline(0, color="k", lw=0.4, alpha=0.18)
        else:
            ax.set_xlim(*P["poslim"]); ax.set_ylim(*P["motlim"]); ax.axhline(0, color="k", lw=0.4, alpha=0.18)
        ax.tick_params(labelsize=7.5)
        ax.text(0.04, 0.05, f"{covpct[ci]:.0f}% covered", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=9, color=COV, weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COV, alpha=0.9))

        if ri == 0:
            ax.set_title(RLAB[ci], fontsize=14, weight="bold", pad=10)
        if ci == 0:
            ax.set_ylabel("position in frame\n(top→bottom)" if P["orient"] == "V" else "horizontal motion\ndx (px)", fontsize=8.5)
        if ri == 3:
            ax.set_xlabel(P.get("motlab") if P["orient"] == "V" else P.get("poslab"), fontsize=8.5)
        elif P["orient"] == "V":
            ax.set_xlabel(P["motlab"], fontsize=7.3, labelpad=1)

    cov = np.array(covpct); peak = int(np.argmax(cov)); inverted = 0 < peak < 4
    strong = inverted and (cov[peak] - min(cov[0], cov[4]) > 12)
    if strong:
        for sp_ in axmap[ri][peak].spines.values():
            sp_.set_edgecolor("#1a7a33"); sp_.set_linewidth(3.0)
        axmap[ri][peak].text(0.5, 0.96, "peak coverage", transform=axmap[ri][peak].transAxes,
                             ha="center", va="top", fontsize=8.5, color="#1a7a33", weight="bold")
    # right-edge verdict
    if inverted and strong:
        verdict, vc = f"INVERTED-U\npeak {RLAB[peak]}\nthen drops\n(red re-grows)", "#1a7a33"
    elif inverted:
        verdict, vc = f"weak peak\n~{RLAB[peak]}\n(≈ flat here)", "#6f7a52"
    else:
        verdict, vc = "MONOTONIC\n(control)\nkeeps\nimproving", "#7a4a1a"
    axmap[ri][4].text(1.06, 0.5, verdict, transform=axmap[ri][4].transAxes, ha="left", va="center",
                      fontsize=9.5, color=vc, weight="bold")

for ti, (_, tlab) in enumerate(TARGETS):
    pt = axmap[2 * ti][0].get_position(); pb = axmap[2 * ti + 1][0].get_position()
    fig.text(0.018, 0.5 * (pt.y1 + pb.y0), tlab, rotation=90, ha="center", va="center",
             fontsize=16, weight="bold", color="#222")
for ri, (_, _, pk) in enumerate(ROWS):
    pos = axmap[ri][0].get_position()
    tag = "vertical\n(y, dy)" if pk == "ydy" else "horizontal\n(x, dx)"
    fig.text(0.052, 0.5 * (pos.y0 + pos.y1), tag, rotation=90, ha="center", va="center",
             fontsize=10, weight="bold", color="#15406b" if pk == "ydy" else "#7a4a1a")

fig.legend(handles=[
    Line2D([0], [0], color="#15406b", lw=1.4, label="KITTI band (support outline)"),
    Line2D([0], [0], marker="o", ls="", color=COV, label="covered  (source reaches it)"),
    Line2D([0], [0], marker="o", ls="", color=UNC, label="uncovered  (coverage gap, $d_{B\\to T}$)"),
    Line2D([0], [0], marker="o", ls="", color=OVR, label="overshoot  (wasted source motion)"),
], loc="lower center", ncol=4, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.004), markerscale=1.6)

fig.suptitle(f"Coverage as a 2-D Venn: covered / uncovered / overshoot   (source within ε={EPS} of a KITTI point)",
             fontsize=16, weight="bold", y=0.972)
fig.text(0.5, 0.943,
         "(y,dy): the gap (red) shrinks to a peak then re-grows as motion over-shoots — strong for KITTI-2012 (peak 1×), "
         "weak for KITTI-2015.   (x,dx): red just keeps shrinking — monotonic (control).",
         ha="center", fontsize=10.5, color="#444")

fig.savefig(OUT / "protoA10_coverage_venn_grid.png", dpi=150, bbox_inches="tight")
print("wrote", OUT / "protoA10_coverage_venn_grid.png")
