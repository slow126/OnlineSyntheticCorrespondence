"""Decisive test: is pairwise_self_distances 'flow' actually flow-only [dx,dy]
or joint [x,y,dx,dy]? Reproduce movi_f<->kitti2015 mean k=1 NN from raw caches
in BOTH spaces and compare to the CSV value (eval->train = 0.000718).
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path("/home/spencer/Projects/OnlineSyntheticCorrespondence")))
from scripts.coverage import faiss_ops, spaces

VEC = Path("/mnt/nvme_1tb_b/coverage_vectors")
IMG = 512

def load(name, split, n_max=16_000_000):
    v = np.load(VEC / f"{name}_{split}_flow.npy", mmap_mode="r")
    if len(v) > n_max:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(v), n_max, replace=False))
        v = v[idx]
    v = np.ascontiguousarray(v, dtype=np.float32)
    return spaces.normalize_flow_vectors(v, IMG, IMG)

print("loading movi_f (train, a) ...", flush=True)
a = load("movi_f", "train")           # train  (16M)
print("loading kitti2015 (val, b) ...", flush=True)
b = load("kitti2015", "val")          # eval   (408k)
print(f"a={a.shape} b={b.shape}", flush=True)

def mean_nn_b_to_a(va, vb, label):
    """eval(b) -> train(a): for each b point, NN distance into a. squared L2."""
    n = len(va); dim = va.shape[1]
    factory = "Flat" if n < 50_000 else f"IVF{min(1024,max(64,n//100))},Flat"
    idx = faiss_ops.build_index(va, use_gpu=True, index_factory=factory,
                                nprobe=64 if "IVF" in factory else None, verbose=False)
    try:
        d, _ = faiss_ops.compute_knn_distances(idx, vb, k=5, verbose=False)
    finally:
        faiss_ops.release_index(idx)
    m = float(np.mean(d[:, 0]))   # squared L2 (faiss returns sqL2)
    print(f"  [{label}] dim={dim} mean_nn_b_to_a(k1, sqL2) = {m:.6f}")
    return m

print("\n=== JOINT [x,y,dx,dy] alpha=1.0 (what the generator code does) ===")
aj = spaces.to_joint_space(a, alpha=1.0)
bj = spaces.to_joint_space(b, alpha=1.0)
mj = mean_nn_b_to_a(aj, bj, "joint")

print("\n=== FLOW-ONLY [dx,dy] (what the handoff assumed) ===")
af = spaces.to_flow_space(a)
bf = spaces.to_flow_space(b)
mf = mean_nn_b_to_a(af, bf, "flow-only")

print("\n=== CSV says: mean_nn_b_to_a = 0.000718 (eval->train) ===")
print(f"  JOINT     = {mj:.6f}   ratio to CSV = {mj/0.000718:.2f}x")
print(f"  FLOW-ONLY = {mf:.6f}   ratio to CSV = {mf/0.000718:.2f}x")
print(f"\n  => CSV 'flow' space is: {'JOINT' if abs(mj-0.000718)<abs(mf-0.000718) else 'FLOW-ONLY'}")
