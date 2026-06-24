"""Does the SEMANTIC "precision" (d_{T->B}) signal in Table 1 come from the
motion (dx,dy) part of BFV, or just from the SPATIAL (x,y) sampling?

Hypothesis (Spencer, 2026-06-12): semantic benchmarks (SPair/PF-PASCAL/
PF-WILLOW/TSS) are sparsely keypoint-labelled, so their BFV (x,y) marginal is
a peculiar, object-centric point pattern. Because normalize_flow_vectors maps
x,y to [-1,1] (range 2) but scales dx,dy by only 2/W (range ~0.25 at KITTI
motion magnitudes), the JOINT 4D nearest-neighbour distance is dominated by
position. So the "semantics prefer precision (d_{T->B})" cell of Table 1 might
be a spatial-sampling artifact, not a motion fact.

Test: recompute the two directed mean-NN distances in three sub-spaces of the
SAME BFV cloud --- full [x,y,dx,dy], spatial-only [x,y], flow-only [dx,dy] ---
and re-run the stratified within-context Spearman cells (exactly Table 1's
cell()). If on semantic targets xy-only reproduces the +0.5 precision signal
and flow-only collapses, the hypothesis holds; on real-motion targets we
expect the opposite (flow carries it).

a = source (train, "T"); b = benchmark (eval, "B").
  d_{T->B} = mean_nn_a_to_b  (off-target / "precision")
  d_{B->T} = mean_nn_b_to_a  (missing-support / "recall" / coverage)

Output: scripts/transfer_analysis_v5/results/bfv_spatial_vs_flow_distances.csv
        (+ printed stratified summary)
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = "/home/spencer/Projects/OnlineSyntheticCorrespondence"
sys.path.insert(0, ROOT)
from scripts.coverage import spaces  # normalize_flow_vectors

VEC = "/mnt/nvme_1tb_b/coverage_vectors"
IMG_W = IMG_H = 512
N_SUB = 120_000          # per-dataset subsample (seed 0) -- ranking-stable
SEED = 0

PURE = ["flyingthings","imagenet2dwarp","movi_f","pointodyssey","sintel","spair",
        "synthetic","synthetic_2d_warp","synthetic_large_zoom",
        "synthetic_random_flipping","synthetic_small_zoom"]
SRC_SPLIT = {s: "train" for s in PURE}
# (benchmark -> split) for the stratified targets
BENCH = {"flyingthings":"test","kitti2012":"val","kitti2015":"val",
         "pointodyssey":"test","synthetic":"val",
         "spair":"test","pfpascal":"test","pfwillow":"test","tss":"val"}
REAL = ["kitti2012","kitti2015","flyingthings","pointodyssey","synthetic"]
SEM  = ["spair","pfpascal","pfwillow","tss"]
SUB = {"full":[0,1,2,3], "xy":[0,1], "flow":[2,3]}


def load_norm(name, split):
    p = os.path.join(VEC, f"{name}_{split}_flow.npy")
    if not os.path.exists(p):
        return None
    a = np.load(p, mmap_mode="r")
    rng = np.random.default_rng(SEED)
    n = min(N_SUB, len(a))
    idx = np.sort(rng.choice(len(a), n, replace=False))
    v = np.asarray(a[idx], dtype=np.float32)
    return spaces.normalize_flow_vectors(v, IMG_W, IMG_H)


WORKERS = 4  # be polite to the concurrent kubric CPU render; don't grab all cores

def directed_mean_nn(a, b):
    """mean_{p in a} dist(p, nearest in b)."""
    d, _ = cKDTree(b).query(a, k=1, workers=WORKERS)
    return float(np.mean(d))


def main():
    out = os.path.join(ROOT, "scripts/transfer_analysis_v5/results/bfv_spatial_vs_flow_distances.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # load all datasets once (normalized subsample held in memory)
    cache = {}
    for s in PURE:
        cache[("src", s)] = load_norm(s, SRC_SPLIT[s])
    for b in BENCH:
        cache[("bench", b)] = load_norm(b, BENCH[b])
    missing = [k for k, v in cache.items() if v is None]
    if missing:
        print("WARN missing caches:", missing, flush=True)

    rows = []
    for s in PURE:
        A = cache[("src", s)]
        if A is None:
            continue
        for b in BENCH:
            B = cache[("bench", b)]
            if B is None:
                continue
            rec = {"source": s, "benchmark": b}
            for tag, cols in SUB.items():
                Ai, Bi = A[:, cols], B[:, cols]
                rec[f"dTB_{tag}"] = directed_mean_nn(Ai, Bi)   # T->B  precision
                rec[f"dBT_{tag}"] = directed_mean_nn(Bi, Ai)   # B->T  recall
            rows.append(rec)
            print(f"  {s:>28s} -> {b:<14s} done", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(D)} pairs)\n", flush=True)

    # ---- merge peak_pck (canonical FF/TT) and re-run Table-1 cells ----
    t0 = pd.read_csv(os.path.join(ROOT, "scripts/transfer_analysis_v3/transfer_table.csv"))
    t = t0[t0.train_dataset.isin(PURE)].copy()
    def regime(r):
        if r.model_family == "raft":
            return "FF"
        if r.pretrained == False and r.freeze == False:
            return "FF"
        if r.pretrained == True and r.freeze == True:
            return "TT"
        return None
    t["regime"] = t.apply(regime, axis=1)
    t = t[t.regime.notna()]
    P = t.merge(D, left_on=["train_dataset", "benchmark"],
                right_on=["source", "benchmark"], how="inner")

    def cell(sub_df, bset, col):
        rs = []
        for _, g in sub_df[sub_df.benchmark.isin(bset)].groupby("benchmark"):
            g = g.dropna(subset=["peak_pck", col])
            if g.train_dataset.nunique() >= 3 and g[col].std() > 1e-12:
                r = spearmanr(g.peak_pck, -g[col]).statistic
                if np.isfinite(r):
                    rs.append(r)
        return np.mean(rs) if rs else np.nan

    # pool architectures within a regime (mean of per-arch cell means), like Table 1
    def regime_cell(reg, bset, col):
        vals = []
        for arch, ga in P[P.regime == reg].groupby("model_family"):
            v = cell(ga, bset, col)
            if np.isfinite(v):
                vals.append(v)
        return np.mean(vals) if vals else np.nan

    print("="*78)
    print("STRATIFIED Spearman(peak, -distance), pooled across archs in each regime")
    print("dTB = precision (T->B) | dBT = recall (B->T).  full / xy / flow sub-spaces")
    print("="*78)
    hdr = f"{'regime':>10s} {'stratum':>11s} | " + " ".join(
        f"{m+'_'+s:>9s}" for m in ["dTB","dBT"] for s in ["full","xy","flow"])
    for reg in ["FF", "TT"]:
        regname = "scratch" if reg == "FF" else "pretrained"
        print("\n"+hdr)
        print("-"*len(hdr))
        for sname, bset in [("real-motion", REAL), ("semantic", SEM)]:
            cells = [regime_cell(reg, bset, f"{m}_{s}")
                     for m in ["dTB","dBT"] for s in ["full","xy","flow"]]
            print(f"{regname:>10s} {sname:>11s} | " +
                  " ".join(f"{c:>+9.2f}" if np.isfinite(c) else f"{'--':>9s}" for c in cells))
    print("\nRead: if (semantic, dTB) is carried by xy but NOT flow -> the semantic")
    print("'precision' preference is a SPATIAL-SAMPLING artifact of sparse keypoints.")


if __name__ == "__main__":
    main()
