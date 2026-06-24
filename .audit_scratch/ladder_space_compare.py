"""Fig 8 magnitude ladder (0.25x->2x) in flow-only vs joint, target=kitti2015.
Recompute dBT (coverage, eval->train) and dTB (off-target, train->eval) in BOTH
spaces for the hq appearance rungs, and print alongside the joint values already
in analysis/coverage_v2_flow_ladder.csv."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/home/spencer/Projects/OnlineSyntheticCorrespondence")
from scripts.coverage import faiss_ops, spaces
VEC = Path("/mnt/nvme_1tb_b/coverage_vectors"); IMG=512
RUNGS=["m025","m050","m100","m150","m200"]; LAB=["0.25x","0.5x","1x","1.5x","2x"]
NMAX=2_600_000

def load(name):
    v=np.load(VEC/f"{name}_flow.npy", mmap_mode="r")
    if len(v)>NMAX:
        rng=np.random.default_rng(0); idx=np.sort(rng.choice(len(v),NMAX,replace=False)); v=v[idx]
    v=np.ascontiguousarray(v,dtype=np.float32)
    return spaces.normalize_flow_vectors(v,IMG,IMG)

def fac(n): return "Flat" if n<50_000 else f"IVF{min(1024,max(64,n//100))},Flat"
def nn(iv,qv):
    ix=faiss_ops.build_index(iv,use_gpu=True,index_factory=fac(len(iv)),nprobe=64 if len(iv)>=50_000 else None,verbose=False)
    try: d,_=faiss_ops.compute_knn_distances(ix,qv,k=1,verbose=False)
    finally: faiss_ops.release_index(ix)
    return float(np.mean(d[:,0]))

tgt=load("kitti2015_val")
T={"flow":spaces.to_flow_space(tgt),"joint":spaces.to_joint_space(tgt,1.0)}
# joint reference from the paper's ladder CSV
J=pd.read_csv("analysis/coverage_v2_flow_ladder.csv")
J=J[(J.eval_dataset=="kitti2015")&(J.train_dataset.str.contains("_hq"))]
print(f"{'rung':>6} | {'dBT_flow':>10} {'dBT_joint(recomp)':>17} {'dBT_joint(CSV)':>15} | "
      f"{'dTB_flow':>10} {'dTB_joint(recomp)':>17} {'dTB_joint(CSV)':>15}")
rows=[]
for r,lab in zip(RUNGS,LAB):
    s=load(f"kitti_{r}_hq_train")
    S={"flow":spaces.to_flow_space(s),"joint":spaces.to_joint_space(s,1.0)}
    rec={"rung":lab}
    for sp in ("flow","joint"):
        rec[f"dBT_{sp}"]=nn(S[sp],T[sp])   # eval->train coverage
        rec[f"dTB_{sp}"]=nn(T[sp],S[sp])   # train->eval off-target
    csv=J[J.train_dataset==f"kitti_{r}_hq"]
    dbt_csv=float(csv.mean_nn_eval_to_train_k1.iloc[0]) if len(csv) else float('nan')
    dtb_csv=float(csv.mean_nn_train_to_eval_k1.iloc[0]) if len(csv) else float('nan')
    rec["dBT_joint_csv"]=dbt_csv; rec["dTB_joint_csv"]=dtb_csv
    rows.append(rec)
    print(f"{lab:>6} | {rec['dBT_flow']:>10.6f} {rec['dBT_joint']:>17.6f} {dbt_csv:>15.6f} | "
          f"{rec['dTB_flow']:>10.6f} {rec['dTB_joint']:>17.6f} {dtb_csv:>15.6f}")
pd.DataFrame(rows).to_csv(".audit_scratch/ladder_space_compare.csv",index=False)
print("\nREAD: dBT smaller = better coverage. Inverted-U in coverage QUALITY = min dBT in the middle.")
