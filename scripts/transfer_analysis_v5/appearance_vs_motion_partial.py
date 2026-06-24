"""Disentangle appearance from motion as a RANKING signal (user, 2026-06-11).

The canonical analysis is tangled: sources differ in motion AND appearance at
once, so the DINO appearance-null could be a power artifact. This holds motion
fixed *statistically*: within each context (variant x benchmark, 11 sources) it
computes the PARTIAL Spearman of transfer vs appearance distance controlling for
motion distance, and vice-versa, then averages per regime.

If appearance carries no ranking signal, partial(transfer, appearance | motion)
~ 0 while partial(transfer, motion | appearance) stays ~ the raw motion rho.

Uses per-cell motion (flow) and appearance (dino) mean_nn distances from
analysis_v3/pairwise_self_distances.csv, joined to the (clean) transfer table.

    python scripts/transfer_analysis_v5/appearance_vs_motion_partial.py
"""
from __future__ import annotations
import argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

PURE = ["flyingthings","imagenet2dwarp","movi_f","pointodyssey","sintel","spair",
        "synthetic","synthetic_2d_warp","synthetic_large_zoom",
        "synthetic_random_flipping","synthetic_small_zoom"]
def regime_of(v):
    a,p,_=v.split("|"); return "scratch" if (p=="False" or a=="raft") else "pretrained"

def partial_sp(y, x, z):
    """partial Spearman of y,x controlling z (rank-residualize)."""
    if len(y) < 4: return np.nan
    ry,rx,rz=(pd.Series(v).rank().to_numpy(float) for v in (y,x,z))
    def resid(a,b):
        B=np.column_stack([np.ones_like(b),b]); return a-B@np.linalg.lstsq(B,a,rcond=None)[0]
    s=spearmanr(resid(ry,rz),resid(rx,rz)).statistic
    return s

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--table",default="scripts/transfer_analysis_v3/transfer_table.csv")
    ap.add_argument("--dist",default="analysis_v3/pairwise_self_distances.csv")
    ap.add_argument("--target",default="peak_pck")
    ap.add_argument("--out",default="scripts/transfer_analysis_v5/results/appearance_vs_motion_partial.csv")
    a=ap.parse_args()
    t=pd.read_csv(a.table); t=t[t.train_dataset.isin(PURE)].copy()
    t["variant"]=t.model_family.astype(str)+"|"+t.pretrained.astype(str)+"|"+t.freeze.astype(str)
    t=t[t.variant!="raft|False|False"]; t["regime"]=t.variant.map(regime_of)
    t["cv"]=t.benchmark+"|"+t.variant
    d=pd.read_csv(a.dist); te=d[d.pair_type=="train_eval"]
    def dist(space):
        x=te[te.space==space].set_index(["dataset_a","dataset_b"])
        return x[["mean_nn_a_to_b","mean_nn_b_to_a","mean_nn_sym"]]
    fl=dist("flow").rename(columns=lambda c:"mot_"+c.replace("mean_nn_",""))
    di=dist("dino").rename(columns=lambda c:"app_"+c.replace("mean_nn_",""))
    t=t.join(fl,on=["train_dataset","benchmark"]).join(di,on=["train_dataset","benchmark"])
    t=t.dropna(subset=[a.target,"mot_a_to_b","app_a_to_b"])

    # directions: scratch binds precision (a_to_b), pretrained binds recall (b_to_a); also sym
    DIRS=[("a_to_b","precision/off-target"),("b_to_a","recall/missing-support"),("sym","symmetric")]
    recs=[]
    for reg in ("scratch","pretrained"):
        rr=t[t.regime==reg]
        for key,lab in DIRS:
            mot=f"mot_{key}"; app=f"app_{key}"
            raw_m,raw_a,par_a,par_m=[],[],[],[]
            for _,c in rr.groupby("cv"):
                if c.train_dataset.nunique()<4: continue
                y=c[a.target].to_numpy(); m=-c[mot].to_numpy(); ap_=-c[app].to_numpy()
                raw_m.append(spearmanr(y,m).statistic)
                raw_a.append(spearmanr(y,ap_).statistic)
                par_a.append(partial_sp(y,ap_,-m))   # appearance | motion
                par_m.append(partial_sp(y,m,-ap_))    # motion | appearance
            f=lambda L:float(np.nanmean([x for x in L if np.isfinite(x)]))
            recs.append(dict(regime=reg,direction=lab,
                             raw_motion=f(raw_m),raw_appearance=f(raw_a),
                             partial_appearance_given_motion=f(par_a),
                             partial_motion_given_appearance=f(par_m),n_ctx=len(raw_m)))
    df=pd.DataFrame(recs)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.out,index=False)
    pd.set_option("display.width",200)
    print(df.to_string(index=False,float_format=lambda x:f"{x:+.3f}"))
    print("\nREAD: if appearance is not a ranking signal, partial_appearance_given_motion ~ 0")
    print("      while partial_motion_given_appearance stays ~ raw_motion.")
    print(f"\nwrote {a.out}")

if __name__=="__main__": main()
