"""F8 — BFV construct-validity: render-free TPE search on the crude BFV
descriptor recovers the physically correct motion regime of each target.

Two targets (Middlebury dropped: eval-bugged). x = recovered camera-dolly
magnitude, y = recovered object vertical-motion fraction. KITTI lands
camera-centric (bottom-right), FlyingThings object-centric (top-left), with no
supervision beyond matching the 4D BFV motion histogram.

    python scripts/transfer_analysis_v5/make_recovery_figure.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "results/figures"
OUT.mkdir(parents=True, exist_ok=True)

GREEN = "#1a7f5a"   # camera-centric
ORANGE = "#d9822b"  # object-centric
GRAY = "#6b7280"

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

df = pd.read_csv(HERE / "recovered_theta.csv")
fig, ax = plt.subplots(figsize=(6.4, 4.4))

styles = {"KITTI-2015": (GREEN, "o"), "FlyingThings": (ORANGE, "s")}
for _, r in df.iterrows():
    c, mk = styles[r.target]
    ax.scatter(r.dolly_delta, r.vertical_frac, s=300, color=c, marker=mk,
               edgecolor="white", lw=1.5, zorder=3)

# direction guides
ax.annotate("", xy=(2.7, -0.06), xytext=(0.1, -0.06),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6))
ax.text(1.4, -0.13, "more CAMERA motion (dolly)", color=GREEN, fontsize=9.5,
        ha="center")
ax.annotate("", xy=(-0.18, 0.95), xytext=(-0.18, 0.05),
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6))
ax.text(-0.30, 0.5, "more OBJECT motion (vertical frac.)", color=ORANGE,
        fontsize=9.5, rotation=90, va="center")

# labels with the regime read-off
ax.annotate("KITTI-2015\nrecovered: forward dolly,\ngrounded objects "
            "$\\rightarrow$ camera-centric", (2.18, 0.0006),
            xytext=(1.15, 0.30), fontsize=9.5, color=GREEN,
            arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.8))
ax.annotate("FlyingThings\nrecovered: ~static camera,\nflying objects "
            "$\\rightarrow$ object-centric", (0.32, 0.92),
            xytext=(0.55, 0.62), fontsize=9.5, color=ORANGE,
            arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))

ax.axhline(0, color="#e5e7eb", lw=0.8, zorder=0)
ax.set_xlim(-0.45, 2.9)
ax.set_ylim(-0.18, 1.05)
ax.set_xlabel("recovered camera-dolly magnitude  $\\Delta$distance")
ax.set_ylabel("recovered object vertical-motion fraction")
ax.set_title("Matching the crude BFV descriptor alone recovers\n"
             "the physically correct motion regime", fontsize=11.5, loc="left")
ax.grid(True, color="#eef0f2", lw=0.7)
fig.tight_layout()
fig.savefig(OUT / "F8_bfv_recovery.png", bbox_inches="tight")
print(f"wrote {OUT / 'F8_bfv_recovery.png'}")
