"""PROTOTYPE v2: model-to-model consensus, CHOSEN regimes only.

Models (7): RAFT (e2e), CATs++ TT/TF, GLU-Net TT/TF, FlowFormer TT/TF.
  T_ = pretrained backbone (no from-scratch matchers); _T frozen, _F fine-tuned.
  Only RAFT is the lone backbone-free model. FlowFormer is parsed from its
  separate results dir (not in transfer_table_nomid.csv).

Views:
  Left  -- 7x7 clustered heatmap of pairwise source-ranking agreement
           (mean Spearman over real-motion benchmarks, common sources).
           mean off-diagonal = empirical reproducibility ceiling.
  Right -- rank-tracking ("bump") on KITTI-2015: flat = agree, crossings = disagree.

Output: ACCV_2026/figures/proto/protoB_model_consensus.png
"""
from pathlib import Path
import glob, re
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from scipy.cluster.hierarchy import linkage, leaves_list
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("ACCV_2026/figures/proto"); OUT.mkdir(parents=True, exist_ok=True)
REAL = ["kitti2012", "kitti2015", "flyingthings", "pointodyssey", "synthetic"]

# ---- backbone matchers + RAFT from the main table (pretrained only, + RAFT) ----
d = pd.read_csv("scripts/transfer_analysis_v3/transfer_table_nomid.csv")
rows = []
for _, r in d.iterrows():
    if r.model_family == "raft":
        cfg = "RAFT"
    elif r.model_family in ("catspp", "glunet") and r.pretrained:
        cfg = f"{r.model_family}|{'TT' if r.freeze else 'TF'}"
    else:
        continue  # drop all from-scratch matcher configs
    rows.append((cfg, r.benchmark, r.train_dataset, r.peak_pck))

# ---- FlowFormer TT/TF from its own results dir (peak = max pck over epochs) ----
FF_DIR = "scripts/transfer_analysis_v5/flowformer_rc_results"
pat = re.compile(r"^(?P<src>.+?)_flowformer_steps100_pretrain(?P<pt>True|False)_freeze(?P<fz>True|False)_")
for path in glob.glob(f"{FF_DIR}/*/validation_results.csv"):
    name = Path(path).parent.name
    m = pat.match(name)
    if not m or m["pt"] != "True":
        continue  # FlowFormer pretrained (TT/TF) only
    cfg = f"flowformer|{'TT' if m['fz'] == 'True' else 'TF'}"
    src = m["src"]
    v = pd.read_csv(path)
    peak = v.groupby("benchmark")["pck"].max()
    for bm, pk in peak.items():
        rows.append((cfg, bm, src, float(pk)))

df = pd.DataFrame(rows, columns=["cfg", "benchmark", "train_dataset", "peak_pck"])
# dedupe (a few sources have 2 FlowFormer dirs): keep the best peak
df = df.groupby(["cfg", "benchmark", "train_dataset"], as_index=False).peak_pck.max()

CFGS = ["catspp|TT", "catspp|TF", "glunet|TT", "glunet|TF",
        "flowformer|TT", "flowformer|TF", "RAFT"]
NICE = {"catspp|TT": "CATs++ TT", "catspp|TF": "CATs++ TF",
        "glunet|TT": "GLU-Net TT", "glunet|TF": "GLU-Net TF",
        "flowformer|TT": "FlowFormer TT", "flowformer|TF": "FlowFormer TF",
        "RAFT": "RAFT (e2e)"}
n = len(CFGS)
sets = [set(df[(df.cfg == c) & (df.benchmark.isin(REAL))].train_dataset) for c in CFGS]
common = sorted(set.intersection(*sets))
print(f"common sources ({len(common)}):", common)


def rankvec(c, b):
    s = df[(df.cfg == c) & (df.benchmark == b)].set_index("train_dataset").peak_pck
    return s.reindex(common).values


M = np.full((n, n), np.nan)
for i, a in enumerate(CFGS):
    for j, b in enumerate(CFGS):
        rs = []
        for bm in REAL:
            va, vb = rankvec(a, bm), rankvec(b, bm)
            ok = ~(np.isnan(va) | np.isnan(vb))
            if ok.sum() >= 4:
                rs.append(spearmanr(va[ok], vb[ok]).correlation)
        M[i, j] = np.nanmean(rs) if rs else np.nan
order = list(leaves_list(linkage(M, method="average")))
CFGS_ORD = [CFGS[k] for k in order]          # clustered order, shared by both panels
Mo = M[np.ix_(order, order)]
labs = [NICE[CFGS[k]] for k in order]
off = M[~np.eye(n, dtype=bool)]
off_mean = np.nanmean(off)

# --- consensus: average-rank (Borda) vs average-pck-then-rank, per benchmark ---
from scipy.stats import rankdata as _rd
print("\n=== consensus comparison (how much do the two recipes disagree?) ===")
for bm in REAL:
    sc = np.vstack([rankvec(c, bm) for c in CFGS])          # models x sources
    ok = ~np.isnan(sc).any(0)
    avg_pck = np.nanmean(sc[:, ok], axis=0)                  # average pck, then rank
    rank_pck = _rd(-avg_pck)
    per_model_rank = np.vstack([_rd(-sc[m, ok]) for m in range(n)])
    avg_rank = per_model_rank.mean(0)                        # average ranks (Borda)
    rho = spearmanr(rank_pck, avg_rank).correlation
    print(f"  {bm:14s}: avg-rank vs avg-pck consensus  rho={rho:.3f}  (n={ok.sum()})")

fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.6), gridspec_kw={"width_ratios": [1.05, 1.1]})

# ---- Left: clustered agreement heatmap ----
ax = axes[0]
im = ax.imshow(Mo, cmap="RdYlGn", vmin=0.2, vmax=1.0)
ax.set_xticks(range(n)); ax.set_xticklabels(labs, rotation=40, ha="right", fontsize=9.5)
ax.set_yticks(range(n)); ax.set_yticklabels(labs, fontsize=9.5)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{Mo[i, j]:.2f}", ha="center", va="center", fontsize=9.5,
                color="black", weight="bold" if i != j else "normal")
ax.set_title(f"Pairwise source-ranking agreement (Spearman rho)\n"
             f"chosen regimes; mean off-diagonal = {off_mean:.2f}  (reproducibility ceiling)",
             fontsize=11.5)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean Spearman rho")

# ---- Right: rank-tracking (bump) on KITTI-2015, models in clustered order ----
ax = axes[1]
BM = "kitti2015"
ranks = {c: (len(common) + 1 - rankdata(rankvec(c, BM))) for c in CFGS}
# color sources by consensus (average) rank so the legend reads top->bottom
consensus = np.mean([ranks[c] for c in CFGS], axis=0)
src_order = np.argsort(consensus)
csrc = {si: plt.cm.turbo(k / (len(common) - 1)) for k, si in enumerate(src_order)}
x = np.arange(n)
for si, src in enumerate(common):
    y = [ranks[c][si] for c in CFGS_ORD]
    ax.plot(x, y, "-o", color=csrc[si], lw=1.5, ms=4.5, alpha=0.85)
    ax.text(-0.08, ranks[CFGS_ORD[0]][si], src, ha="right", va="center", fontsize=7, color=csrc[si])
    ax.text(n - 1 + 0.08, ranks[CFGS_ORD[-1]][si], src, ha="left", va="center", fontsize=7, color=csrc[si])
ax.set_xticks(x); ax.set_xticklabels([NICE[c] for c in CFGS_ORD], rotation=40, ha="right", fontsize=9)
ax.set_ylabel("source rank on KITTI-2015  (1 = best)", fontsize=10)
ax.set_ylim(len(common) + 0.5, 0.5)
ax.set_title("Where models agree (flat) vs disagree (crossings)\nsource ranking on KITTI-2015", fontsize=11.5)
ax.set_xlim(-1.4, n - 0.4)
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(OUT / "protoB_model_consensus.png", dpi=160, bbox_inches="tight")
print("wrote", OUT / "protoB_model_consensus.png")
print(f"mean off-diagonal = {off_mean:.3f}")
print(pd.DataFrame(M, index=CFGS, columns=CFGS).round(2).to_string())
