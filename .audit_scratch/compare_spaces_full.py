"""Full flow-only vs joint comparison for the Table-1 pair set.

For every (PURE source, benchmark) train_eval pair, compute mean k=1 NN (sqL2)
in BOTH spaces and BOTH directions, matching the pipeline protocol exactly
(normalize by 512, 16M seed-0 subsample, same faiss index policy as
compute_pairwise_self_distances.py).

  a = train (source), b = eval (benchmark)
  a_to_b  = train->eval  = dTB (off-target)
  b_to_a  = eval->train  = dBT (coverage)

joint columns should reproduce analysis_v3/pairwise_self_distances.csv (space=flow).
Output: .audit_scratch/space_compare_distances.csv
"""
import sys, gc, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path("/home/spencer/Projects/OnlineSyntheticCorrespondence")))
from scripts.coverage import faiss_ops, spaces

VEC = Path("/mnt/nvme_1tb_b/coverage_vectors")
OUT = Path("/home/spencer/Projects/OnlineSyntheticCorrespondence/.audit_scratch/space_compare_distances.csv")
IMG = 512; NMAX = 16_000_000

PURE = ["flyingthings","imagenet2dwarp","movi_f","pointodyssey","sintel","spair",
        "synthetic","synthetic_2d_warp","synthetic_large_zoom",
        "synthetic_random_flipping","synthetic_small_zoom"]
BENCH = [("flyingthings","test"),("kitti2012","val"),("kitti2015","val"),
         ("spair","test"),("pfpascal","test"),("pfwillow","test"),
         ("pointodyssey","test"),("tss","val"),("synthetic","val")]

def load_norm(name, split):
    p = VEC / f"{name}_{split}_flow.npy"
    if not p.exists(): return None
    v = np.load(p, mmap_mode="r")
    if len(v) > NMAX:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(v), NMAX, replace=False))
        v = v[idx]
    v = np.ascontiguousarray(v, dtype=np.float32)
    return spaces.normalize_flow_vectors(v, IMG, IMG)

def factory(n):
    return "Flat" if n < 50_000 else f"IVF{min(1024,max(64,n//100))},Flat"

def nn_mean(index_vecs, query_vecs):
    idx = faiss_ops.build_index(index_vecs, use_gpu=True, index_factory=factory(len(index_vecs)),
                                nprobe=64 if len(index_vecs)>=50_000 else None, verbose=False)
    try:
        d,_ = faiss_ops.compute_knn_distances(idx, query_vecs, k=1, verbose=False)
    finally:
        faiss_ops.release_index(idx)
    gc.collect()
    return float(np.mean(d[:,0]))   # sqL2

# Preload benchmark vectors (both spaces) once; reused across all sources.
print("preloading benchmarks...", flush=True)
B = {}
for bn, bs in BENCH:
    v = load_norm(bn, bs)
    if v is None:
        print(f"  MISS {bn}/{bs}"); continue
    B[(bn,bs)] = {"flow": spaces.to_flow_space(v), "joint": spaces.to_joint_space(v, 1.0)}
    print(f"  loaded {bn}/{bs} ({len(v):,})", flush=True)

rows = []
for src in PURE:
    t0 = time.time()
    v = load_norm(src, "train")
    if v is None:
        print(f"MISS source {src}"); continue
    A = {"flow": spaces.to_flow_space(v), "joint": spaces.to_joint_space(v, 1.0)}
    del v; gc.collect()
    print(f"\n=== source {src} ({len(A['flow']):,}) ===", flush=True)
    for (bn,bs), Bv in B.items():
        rec = {"source":src, "benchmark":bn}
        for sp in ("flow","joint"):
            a, b = A[sp], Bv[sp]
            rec[f"{sp}_a_to_b"] = nn_mean(b, a)   # train->eval (dTB)
            rec[f"{sp}_b_to_a"] = nn_mean(a, b)   # eval->train (dBT)
        rows.append(rec)
        print(f"  {bn:14s} flow dBT={rec['flow_b_to_a']:.6f} dTB={rec['flow_a_to_b']:.6f} | "
              f"joint dBT={rec['joint_b_to_a']:.6f} dTB={rec['joint_a_to_b']:.6f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)   # checkpoint each pair
    del A; gc.collect()
    print(f"  [{src} done in {time.time()-t0:.0f}s]", flush=True)

print(f"\nWROTE {OUT}  ({len(rows)} pairs)")
