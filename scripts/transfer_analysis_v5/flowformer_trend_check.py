"""Does the regime-direction asymmetry hold for FlowFormer (3rd architecture)?

FlowFormer 2x2 (pretrain x freeze) trains on the canonical sources on RC. Runs are
walltime-truncated at very different epoch budgets (imagenet ~1000ep, pointodyssey
~150ep), so a naive peak_pck ranking is confounded by training progress. We harvest
peak_pck at a MATCHED epoch cap so every source is compared at the same budget, then
compute the same directed within-context correlations as the canonical analysis:
  scratch (pretrainFalse): expect symmetric/both; pretrained-frozen (pretrainTrue,
  freezeTrue): expect recall (missing support) to lead -- the clean cross-arch claim.

Distances: reuse analysis_v3/pairwise_self_distances.csv (flow, train_eval) -- the
FlowFormer sources are the same canonical sources, so source->benchmark distances
are identical.

    python scripts/transfer_analysis_v5/flowformer_trend_check.py --cap 150
"""
from __future__ import annotations
import argparse, glob, os, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

PULL = "/home/spencer/Projects/OnlineSyntheticCorrespondence/.audit_scratch/flowformer_pull"
PURE = ["flyingthings","imagenet2dwarp","movi_f","pointodyssey","sintel","spair",
        "synthetic","synthetic_2d_warp","synthetic_large_zoom",
        "synthetic_random_flipping","synthetic_small_zoom"]

def parse(dirname):
    # <source>_flowformer_steps100_pretrain{T/F}_freeze{T/F}_<ts>
    base = dirname.split("_flowformer_steps100_")
    src = base[0]
    pre = "True" in base[1].split("freeze")[0]
    fz  = "freeze" + base[1].split("freeze")[1]
    fzv = fz.startswith("freezeTrue")
    return src, pre, fzv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=150, help="match all sources at <= this epoch")
    ap.add_argument("--dist", default="analysis_v3/pairwise_self_distances.csv")
    a = ap.parse_args()
    rows=[]
    for f in glob.glob(f"{PULL}/**/validation_results.csv", recursive=True):
        dn = os.path.basename(os.path.dirname(f))
        src, pre, fzv = parse(dn)
        if src not in PURE: continue
        v = pd.read_csv(f)
        vcap = v[v.epoch <= a.cap]
        if len(vcap)==0: continue
        for b,g in vcap.groupby("benchmark"):
            rows.append(dict(source=src, pretrained=pre, freeze=fzv, benchmark=b,
                             peak_pck=float(g.pck.max()), maxep=int(vcap.epoch.max())))
    pk = pd.DataFrame(rows)
    pk = pk[pk.benchmark!="middlebury"]
    pk["variant"] = "flowformer|"+pk.pretrained.astype(str)+"|"+pk.freeze.astype(str)
    d = pd.read_csv(a.dist); te=d[(d.pair_type=="train_eval")&(d.space=="flow")]
    fdist = te.set_index(["dataset_a","dataset_b"])[["mean_nn_a_to_b","mean_nn_b_to_a","mean_nn_sym"]]
    pk = pk.join(fdist, on=["source","benchmark"]).dropna(subset=["mean_nn_a_to_b"])
    print(f"=== matched-epoch cap <= {a.cap}; sources at cap: {sorted(pk.source.unique())} ===")
    print(f"benchmarks: {sorted(pk.benchmark.unique())}\n")
    recs=[]
    for var, sub in pk.groupby("variant"):
        def vctx(col):
            rs=[spearmanr(c.peak_pck, -c[col]).statistic for _,c in sub.groupby("benchmark")
                if c.source.nunique()>=3 and c[col].std()>1e-12]
            rs=[r for r in rs if np.isfinite(r)]
            return float(np.nanmean(rs)) if rs else np.nan, len(rs)
    # per-variant means across benchmarks
    for var, sub in pk.groupby("variant"):
        pr=[];rc=[];sy=[];nb=0
        for b,c in sub.groupby("benchmark"):
            if c.source.nunique()<3: continue
            nb+=1
            pr.append(spearmanr(c.peak_pck,-c.mean_nn_a_to_b).statistic)
            rc.append(spearmanr(c.peak_pck,-c.mean_nn_b_to_a).statistic)
            sy.append(spearmanr(c.peak_pck,-c.mean_nn_sym).statistic)
        f=lambda L:float(np.nanmean([x for x in L if np.isfinite(x)])) if L else np.nan
        _,pre,fz = var.split("|")
        regime = "scratch" if pre=="False" else ("pretrained-frozen" if fz=="True" else "pretrained-unfrozen")
        recs.append(dict(variant=var, regime=regime, n_bench=nb, n_src=sub.source.nunique(),
                         precision=f(pr), recall=f(rc), sym=f(sy)))
    df=pd.DataFrame(recs)
    out="scripts/transfer_analysis_v5/results/flowformer_trend_check.csv"
    df.to_csv(out,index=False)
    print(df.to_string(index=False,float_format=lambda x:f"{x:+.3f}"))
    print("\nPREDICTION: pretrained-frozen -> recall leads (like CATs++/GLU-Net frozen);")
    print("            scratch -> symmetric/both (precision lean is CATs++/RAFT-specific).")
    print(f"\nwrote {out}  (n_src per variant is small + walltime-truncated: TREND ONLY)")

if __name__=="__main__": main()
