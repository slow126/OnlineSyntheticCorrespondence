"""Compose the closed-loop, geometry-only source-design figure (fig:intervsplats)
as a single matplotlib render: rounded panels around the directional-splat
fingerprints, labelled arrows, a dashed feedback arc (the closed loop), a
geometry-only annotation, and a measured-gain badge.

Output: ACCV_2026/figures/results/F_pipeline_closedloop.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

ROOT = Path("ACCV_2026")
SPL = ROOT / "figures/splats"
OUT = ROOT / "figures/results/F_pipeline_closedloop.png"

INK = "#2b2b2b"
GRAYE = "#c2c7ce"
PANEL = "#f4f5f7"
TARGET = "#e9f1fb"
TARGETE = "#9bb8de"
GREEN = "#d6efdd"
GREENE = "#3a8f6a"
FEED = "#3a6ea5"

# layout in data units (equal aspect)
IMG_H = 22.0
YB, YT = 9.0, 9.0 + IMG_H            # image bottom / top
PAD = 1.7                            # panel padding around image

def panel_w(aspect):                 # width for IMG_H tall image
    return IMG_H * aspect

stages = [
    dict(img="train__movi-f__directional_splat.png",        asp=1.00,
         role="generic source",  name="MOVi-F",          fill=PANEL,  edge=GRAYE),
    dict(img="train__kitti-recovered__directional_splat.png", asp=1.00,
         role="tuned source",    name="KITTI-tuned",     fill=PANEL,  edge=GRAYE),
    dict(img="benchmark__kitti2015__directional_splat.png", asp=1.30,
         role="target",          name="KITTI-2015",      fill=TARGET, edge=TARGETE),
]

GAP1, GAP2, GAP3 = 22.0, 14.0, 12.0  # arrow gaps (GAP1 wide for search labels)
x = 3.0
for s in stages:
    s["w"] = panel_w(s["asp"]); s["x0"] = x; s["x1"] = x + s["w"]
    x = s["x1"] + (GAP1 if s is stages[0] else GAP2 if s is stages[1] else GAP3)
badge_x0 = stages[2]["x1"] + GAP3
badge_w = 21.0
badge_x1 = badge_x0 + badge_w
XMAX = badge_x1 + 3.0

fig, ax = plt.subplots(figsize=(13.6, 4.2))
ax.set_xlim(-1, XMAX); ax.set_ylim(0, 49); ax.set_aspect("equal"); ax.axis("off")

def rrect(x0, y0, x1, y1, fill, edge, lw=1.4, z=1):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                       boxstyle="round,pad=0,rounding_size=2.2",
                       fc=fill, ec=edge, lw=lw, zorder=z, mutation_aspect=1)
    ax.add_patch(p); return p

def arrow(x0, x1, y, color=INK, lw=2.2, style="-|>", ls="-", rad=0.0, z=4):
    a = FancyArrowPatch((x0, y), (x1, y), arrowstyle=style, mutation_scale=18,
                        lw=lw, color=color, zorder=z,
                        connectionstyle=f"arc3,rad={rad}", linestyle=ls)
    ax.add_patch(a); return a

# panels + images + labels
ROLE_Y = YT + PAD + 1.2
for s in stages:
    rrect(s["x0"] - PAD, YB - PAD, s["x1"] + PAD, YT + PAD, s["fill"], s["edge"])
    img = mpimg.imread(SPL / s["img"])
    ax.imshow(img, extent=[s["x0"], s["x1"], YB, YT], zorder=3, aspect="auto")
    cx = (s["x0"] + s["x1"]) / 2
    ax.text(cx, ROLE_Y, s["role"].upper(), ha="center", va="bottom",
            fontsize=9, color="#6b7280", weight="bold")
    ax.text(cx, YB - PAD - 1.8, s["name"], ha="center", va="top",
            fontsize=11.5, color=INK, weight="bold")

ymid = (YB + YT) / 2

# arrow 1: MOVi-F -> recovered  (the geometry-only TPE search)
a1x0, a1x1 = stages[0]["x1"] + PAD + 0.8, stages[1]["x0"] - PAD - 0.8
arrow(a1x0, a1x1, ymid, lw=2.6)
acx = (a1x0 + a1x1) / 2
ax.text(acx, ymid + 2.6, "TPE search", ha="center", va="bottom",
        fontsize=11, color=INK, weight="bold")
ax.text(acx, ymid - 3.0, "geometry-only", ha="center", va="top",
        fontsize=8.8, color="#b5670f", weight="bold")
ax.text(acx, ymid - 5.2, "(BFV only,\nno lighting pass)", ha="center", va="top",
        fontsize=8.0, color="#9a5a10", linespacing=1.35)

# arrow 2: recovered -> target  (fingerprint match)
a2x0, a2x1 = stages[1]["x1"] + PAD + 0.8, stages[2]["x0"] - PAD - 0.8
arrow(a2x0, a2x1, ymid)
ax.text((a2x0 + a2x1) / 2, ymid + 2.4, "BFV", ha="center", va="bottom",
        fontsize=9, color="#555")
ax.text((a2x0 + a2x1) / 2, ymid - 2.6, "match", ha="center", va="top",
        fontsize=9, color="#555")

# arrow 3: target -> badge  (train, measured gain)
a3x0, a3x1 = stages[2]["x1"] + PAD + 0.8, badge_x0 - 0.8
arrow(a3x0, a3x1, ymid, lw=2.6)
ax.text((a3x0 + a3x1) / 2, ymid + 2.4, "train", ha="center", va="bottom",
        fontsize=9.5, color="#555", weight="bold")

# gain badge
rrect(badge_x0, YB - PAD, badge_x1, YT + PAD, GREEN, GREENE, lw=1.6)
bcx = (badge_x0 + badge_x1) / 2
ax.text(bcx, YT - 2.2, "peak PCK", ha="center", va="top", fontsize=9, color="#2f6b48")
ax.text(bcx, ymid + 1.4, "+10.7", ha="center", va="center", fontsize=21,
        color="#1f5b39", weight="bold")
ax.text(bcx, YB + 3.7, "GLU-Net, KITTI-2015", ha="center", va="center",
        fontsize=8.2, color="#2f6b48")
ax.text(bcx, YB + 1.3, "vs. MOVi-F baseline", ha="center", va="center",
        fontsize=7.8, color="#3f7a57", style="italic")

# closed-loop feedback arc: target's BFV is the search objective.
# Routed with a high bulge so it clears every role tag; arrowhead drops onto
# the search arrow, label sits above the apex.
fb = FancyArrowPatch((stages[2]["x0"], YT + PAD), (acx, ymid + 6.0),
                     arrowstyle="-|>", mutation_scale=14, lw=1.6, color=FEED,
                     linestyle=(0, (5, 3)), zorder=6,
                     connectionstyle="arc3,rad=0.52")
ax.add_patch(fb)
ax.text((stages[2]["x0"] + acx) / 2, 46.0,
        "search objective: match the target's motion fingerprint",
        ha="center", va="bottom", fontsize=8.6, color=FEED, style="italic")

fig.savefig(OUT, bbox_inches="tight", dpi=200)
print("wrote", OUT)
