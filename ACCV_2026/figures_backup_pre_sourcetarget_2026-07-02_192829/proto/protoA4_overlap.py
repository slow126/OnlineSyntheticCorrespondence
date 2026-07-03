"""PROTOTYPE: source-target motion OVERLAP across the magnitude ladder (JOINT space).

Now that the headline metric is confirmed joint 4-D [x,y,dx,dy] (to_joint_space,
alpha=1, sqL2), the overlap panel uses the JOINT directed mean-NN distances from the
paper's own ladder file (NOT the retired qnorm). Coverage (target covered by source)
is the inverted-U that tracks transfer; off-target is flat.

Left:  overlap curve -- joint coverage (-dBT) and off-target (-dTB) vs magnitude,
       with transfer PCK overlaid. coverage and PCK co-peak at the match.
Right: the mechanism -- |flow| magnitude distributions sliding through the target;
       overlap (histogram intersection) is maximal at the matched rung.

Output: ACCV_2026/figures/proto/protoA4_overlap.png
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CACHE = Path("/mnt/nvme_1tb_b/coverage_vectors")
OUT = Path("ACCV_2026/figures/proto"); OUT.mkdir(parents=True, exist_ok=True)
RUNGS = ["m025", "m050", "m100", "m150", "m200"]
XL = ["0.25x", "0.5x", "1x", "1.5x", "2x"]
mag_cmap = LinearSegmentedColormap.from_list("m", ["#f0a860", "#d2691e", "#7a1500"])
rng = np.random.default_rng(0)

# --- joint directed mean-NN per rung (paper's metric), target = KITTI-2015 ---
cv = pd.read_csv("analysis/coverage_v2_flow_ladder.csv")
cv = cv[cv.eval_dataset.astype(str).str.contains("kitti2015", case=False)].copy()
cv["rung"] = cv.train_dataset.astype(str).str.extract(r"(m\d+)")[0]
g = cv.groupby("rung").agg(dBT=("mean_nn_eval_to_train_k1", "mean"),
                           dTB=("mean_nn_train_to_eval_k1", "mean")).reindex(RUNGS)
dBT, dTB = g.dBT.values, g.dTB.values

# --- transfer PCK (pretrained encoder, KITTI-2015) ---
m = pd.read_csv("analysis/ladder_master_table.csv")
pck = m[m.reg == "TF"].pivot_table(index="rung", columns="app", values="kitti2015",
                                   aggfunc="mean").reindex(RUNGS).mean(axis=1).values

# --- |flow| magnitude distributions (px) ---
def load(name, n=150000):
    v = np.load(CACHE / f"{name}_flow.npy").astype(np.float32)
    return v[rng.choice(len(v), min(n, len(v)), replace=False)]
tmag = np.hypot(*load("kitti2015_val")[:, 2:].T)
rmag = {r: np.hypot(*load(f"kitti_{r}_hq_train")[:, 2:].T) for r in RUNGS}
edges = np.linspace(0, 55, 80); mc = 0.5 * (edges[:-1] + edges[1:])
th, _ = np.histogram(tmag, bins=edges, density=True)
# magnitude-overlap fraction (histogram intersection) per rung
ov = []
for r in RUNGS:
    sh, _ = np.histogram(rmag[r], bins=edges, density=True)
    ov.append(np.minimum(th, sh).sum() * (edges[1] - edges[0]))
ov = np.array(ov)

x = np.arange(5)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.6))

# ===== Left: overlap curve + transfer =====
axL.axvspan(-0.4, 1.5, color="#fdecea", alpha=0.5, zorder=0)   # under
axL.axvspan(2.5, 4.4, color="#fff4e6", alpha=0.6, zorder=0)    # over
axL.axvline(2, color="#2a8f3f", ls="--", lw=1.4, zorder=1)
cov = -dBT                                                     # higher = better coverage/overlap
axL.plot(x, cov, "o-", color="#2b6cb0", lw=2.6, ms=8, label="coverage  $-d_{B\\to T}$ (joint)", zorder=3)
axL.set_xticks(x); axL.set_xticklabels(XL)
axL.set_xlabel("source motion magnitude (x KITTI's)")
axL.set_ylabel("joint coverage / overlap  ($-$mean-NN, higher=better)", color="#2b6cb0")
axL.tick_params(axis="y", labelcolor="#2b6cb0")
axL.margins(y=0.20)
axL.set_title("Overlap peaks at the match, and tracks transfer", loc="left", fontsize=12)
# correlations vs transfer (5 rungs) -> quantify "coverage tracks, off-target doesn't"
rho_cov = spearmanr(-dBT, pck).correlation
rho_off = spearmanr(-dTB, pck).correlation
axL.text(0.03, 0.04, f"vs PCK:  coverage $\\rho$={rho_cov:+.2f}   off-target $\\rho$={rho_off:+.2f}\n"
         f"off-target {dTB.mean():.3f}, varies <{100*(dTB.max()-dTB.min())/dTB.mean():.1f}% -> over-shoot = lost coverage",
         transform=axL.transAxes, fontsize=7.8, color="#333",
         bbox=dict(boxstyle="round", fc="#f4f4f4", ec="#999", alpha=0.9))
# PCK on a 2nd axis
ax2 = axL.twinx()
ax2.plot(x, pck, "^-", color="#6a3d9a", lw=2.4, ms=8, label="transfer PCK (pretrained)", zorder=3)
ax2.set_ylabel("KITTI-2015 peak PCK", color="#6a3d9a")
ax2.tick_params(axis="y", labelcolor="#6a3d9a")
# off-target on a 3rd axis (generous +/-50% window -> its flatness is honest, not zoom-amplified)
ax3 = axL.twinx()
ax3.spines["right"].set_position(("outward", 52))
ax3.plot(x, -dTB, "s:", color="#c0392b", lw=1.8, ms=6, alpha=0.9, label="off-target  $-d_{T\\to B}$ (joint)", zorder=2)
ax3.set_ylim(-dTB.mean() * 1.5, -dTB.mean() * 0.5)
ax3.set_ylabel("off-target  $-d_{T\\to B}$", color="#c0392b")
ax3.tick_params(axis="y", labelcolor="#c0392b")
h1, l1 = axL.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
h3, l3 = ax3.get_legend_handles_labels()
axL.legend(h1 + h2 + h3, l1 + l2 + l3, fontsize=8.0, loc="upper center")

# ===== Right: magnitude distributions sliding through the target =====
axR.fill_between(mc, th, color="#1f6fb2", alpha=0.30, label="KITTI target", zorder=1)
axR.plot(mc, th, color="#15406b", lw=2.0, zorder=2)
for i, r in enumerate(RUNGS):
    sh, _ = np.histogram(rmag[r], bins=edges, density=True)
    axR.plot(mc, sh, color=mag_cmap(i / 4), lw=2.2, label=f"{XL[i]}  (overlap {ov[i]:.2f})")
# shade the matched-rung overlap
sh1, _ = np.histogram(rmag["m100"], bins=edges, density=True)
axR.fill_between(mc, np.minimum(th, sh1), color="#2a8f3f", alpha=0.25, zorder=0)
axR.set_xlabel("flow magnitude |f| (px)")
axR.set_ylabel("density")
axR.set_title("Why: the source distribution slides through the target\n(green = 1x overlap)", loc="left", fontsize=12)
axR.legend(fontsize=8.6, loc="upper right")
axR.set_xlim(0, 38)

fig.suptitle("Source-target motion overlap across the magnitude ladder (joint metric, target = KITTI-2015)",
             fontsize=13.5, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "protoA4_overlap.png", dpi=160, bbox_inches="tight")
print("wrote", OUT / "protoA4_overlap.png")
print("joint coverage dBT:", np.round(dBT, 5))
print("magnitude-overlap fraction:", dict(zip(XL, np.round(ov, 3))))
print("PCK:", np.round(pck, 1))
