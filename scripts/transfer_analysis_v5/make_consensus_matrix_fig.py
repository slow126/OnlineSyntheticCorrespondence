"""Rank-agreement matrix: is coverage's source ranking within the spread that the
individual retrained models themselves produce?

The consensus is an average over several independently-trained models that disagree
among themselves (the reproducibility ceiling). So rather than score coverage
against the consensus average, we ask: does coverage's rank for a source fall within
the [min, max] band of ranks the individual in-scope models assign it? If so, coverage
is within retraining noise -- indistinguishable from just training another model.

Each cell: coverage's predicted rank (large), the model-rank band lo-hi (small), and
the consensus peak-PCK (tiny reference). GREEN = coverage within the model band;
gray = outside, shaded by how far outside. Footer = per-target rank-rho.

Output: ACCV_2026/figures/results/F_consensus_matrix.png
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from matplotlib.patches import Patch

OUT = Path("ACCV_2026/figures/results/F_consensus_matrix.png")
T = pd.read_csv("scripts/transfer_analysis_v3/transfer_table_nomid.csv")
PURE = ["flyingthings","movi_f","pointodyssey","sintel","spair","synthetic",
        "synthetic_2d_warp","synthetic_large_zoom","synthetic_random_flipping",
        "synthetic_small_zoom","imagenet2dwarp"]
SEM = {"spair","pfpascal","pfwillow","tss"}
T = T[T.train_dataset.isin(PURE)].copy()
T["variant"] = T.model_family + "|" + T.pretrained.astype(str) + "|" + T.freeze.astype(str)
def designdefined(r):
    # Each architecture in its intended regime: pretrained backbone-dependent
    # matchers (all targets) + RAFT from scratch (real-motion only).
    fam, pt, b = r.model_family, bool(r.pretrained), r.benchmark
    if fam == "raft":
        return b not in SEM
    if not pt:                 # CATs++/GLU-Net from scratch = off-design
        return False
    return True
ins = T[T.apply(designdefined, axis=1)]
cons = ins.groupby(["train_dataset","benchmark"])["peak_pck"].mean().unstack()
cov  = T.groupby(["train_dataset","benchmark"])["flow_mean_nn_eval_to_train_k1"].first().unstack()

REAL = ["kitti2012","kitti2015","flyingthings","pointodyssey","synthetic"]
SEMC = ["spair","pfpascal","pfwillow","tss"]
COLS = REAL + SEMC
cons = cons.reindex(columns=COLS); cov = cov.reindex(columns=COLS)
order = cons[REAL].mean(axis=1).sort_values(ascending=False).index.tolist()
cons = cons.reindex(order); cov = cov.reindex(order)

CLEAN_SRC = {"flyingthings":"FlyingThings","movi_f":"MOVi-F","pointodyssey":"PointOdyssey",
             "sintel":"Sintel","spair":"SPair","synthetic":"Synthetic (ours)",
             "synthetic_2d_warp":"Synthetic+2D-warp","synthetic_large_zoom":"Synthetic+large-zoom",
             "synthetic_random_flipping":"Synthetic+flip","synthetic_small_zoom":"Synthetic+small-zoom",
             "imagenet2dwarp":"ImageNet-2D-warp"}
CLEAN_TGT = {"kitti2012":"KITTI-12","kitti2015":"KITTI-15","flyingthings":"FlyingThings",
             "pointodyssey":"PointOdyssey","synthetic":"Synthetic","spair":"SPair",
             "pfpascal":"PF-PASCAL","pfwillow":"PF-WILLOW","tss":"TSS"}

# per-target model-rank band [lo,hi] for each source, coverage rank, rank-rho
nR, nC = cons.shape
GREEN = np.array([0.30, 0.62, 0.33])
rgb = np.ones((nR, nC, 3))
covR = pd.DataFrame(index=cons.index, columns=COLS, dtype=float)
loB = pd.DataFrame(index=cons.index, columns=COLS, dtype=float)
hiB = pd.DataFrame(index=cons.index, columns=COLS, dtype=float)
col_rho = {}; within = 0; ncell = 0
for j, c in enumerate(COLS):
    sub = ins[ins.benchmark == c]
    R = {}
    for v, g in sub.groupby("variant"):
        R[v] = g.set_index("train_dataset")["peak_pck"].rank(ascending=False, method="min")
    R = pd.DataFrame(R)                                    # sources x models
    cc = cons[c].dropna(); cv = cov[c].reindex(cc.index)
    covr = cv.rank(ascending=True, method="min")
    col_rho[c] = spearmanr(-cv.values, cc.values).correlation
    for s in cc.index:
        i = list(cons.index).index(s)
        ranks = R.loc[s].dropna() if s in R.index else pd.Series(dtype=float)
        if ranks.empty: continue
        cr = covr[s]
        rv = ranks.values
        if len(rv) >= 4:                                  # robust band: trim lone outlier model
            med = np.median(rv)
            rv = np.delete(rv, int(np.argmax(np.abs(rv - med))))
        lo, hi = rv.min(), rv.max()
        covR.loc[s, c] = cr; loB.loc[s, c] = lo; hiB.loc[s, c] = hi
        ncell += 1
        if lo <= cr <= hi:
            rgb[i, j] = GREEN; within += 1
        else:
            out = min(cr - hi, lo - cr) if (cr > hi or cr < lo) else 0
            n = min(abs(out) / 5.0, 1.0)                   # ranks outside band
            shade = 0.95 - 0.33 * n
            rgb[i, j] = np.array([shade, shade, shade])

fig, ax = plt.subplots(figsize=(1.08*nC + 3.3, 0.46*nR + 1.7))
ax.imshow(rgb, aspect="auto")
ax.axvline(len(REAL) - 0.5, color="#333", lw=1.4)

for i, s in enumerate(cons.index):
    for j, c in enumerate(COLS):
        if np.isnan(covR.iloc[i, j]): continue
        lum = rgb[i, j].mean(); tcol = "white" if lum < 0.5 else "#1a1a1a"
        cr, lo, hi = int(covR.iloc[i, j]), int(loB.iloc[i, j]), int(hiB.iloc[i, j])
        band = f"{lo}" if lo == hi else f"{lo}–{hi}"
        ax.text(j-0.04, i-0.13, f"{cr}", ha="center", va="center", fontsize=9.2,
                color=tcol, weight="bold")
        ax.text(j-0.04, i+0.20, f"[{band}]", ha="center", va="center", fontsize=6.0,
                color=tcol, alpha=0.8)
        ax.text(j+0.40, i-0.32, f"{cons.iloc[i,j]:.0f}", ha="center", va="center",
                fontsize=5.2, color=tcol, alpha=0.55)

for j, c in enumerate(COLS):
    ax.text(j, nR+0.02, f"{col_rho[c]:+.2f}", ha="center", va="top", fontsize=8,
            color="#1f5b39" if col_rho[c] >= 0.5 else "#b54", weight="bold")
ax.text(-0.7, nR+0.02, r"rank $\rho$:", ha="right", va="top", fontsize=8.2, color="#444")

ax.set_xticks(range(nC)); ax.set_xticklabels([CLEAN_TGT[c] for c in COLS], rotation=35, ha="right", fontsize=9)
ax.set_yticks(range(nR)); ax.set_yticklabels([CLEAN_SRC[s] for s in cons.index], fontsize=9)
ax.set_xlabel("target benchmark", fontsize=10); ax.set_ylabel("training source", fontsize=10)
ax.tick_params(length=0)
ax.text((len(REAL)-1)/2, -1.05, "Real-motion Targets", ha="center", va="bottom",
        fontsize=9.5, weight="bold", color="#444")
ax.text(len(REAL)+(len(SEMC)-1)/2, -1.05, "Semantic Targets", ha="center", va="bottom",
        fontsize=9.5, weight="bold", color="#444")
leg = [Patch(fc=GREEN, label="coverage within the model band"),
       Patch(fc=(0.80,0.80,0.80), ec="#aaa", label="outside the band (shaded by distance)")]
ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8.2,
          frameon=False, handlelength=1.3, borderaxespad=0,
          title="cell: coverage rank, [model-rank band],\nconsensus PCK (tiny)",
          title_fontsize=8.0, alignment="left")
ax.set_ylim(nR+1.0, -1.5)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", dpi=200)
print("wrote", OUT, (nR, nC), "within-band:", round(100*within/ncell), "%")
