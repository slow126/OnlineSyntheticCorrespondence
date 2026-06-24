"""Per-regime source/family jackknife of the directed correlations.

Question (user, 2026-06-11): is the scratch->precision / pretrained->recall
correlation induced by specific source DATASETS, or true across all of them?

For each regime (scratch / pretrained) we compute the regime-mean within-context
Spearman for BOTH directed distances — precision = d(T->B) (rank by -d_ab),
recall = d(B->T) (rank by -d_ba) — exactly as asym_vs_sym_table.py does, then:
  (A) baseline over all 11 canonical sources,
  (B) leave-one-source-out (11 drops),
  (C) leave-one-generator-family-out (5 families).
If the regime-relevant direction (precision for scratch, recall for pretrained)
keeps its sign/magnitude under every drop, the correlation is dataset-robust;
if a single drop collapses or flips it, it was dataset-induced.

Reads the (now Middlebury-free) transfer_table.csv. Writes nothing the pipeline
consumes — a standalone diagnostic.

    python scripts/transfer_analysis_v5/per_regime_source_jackknife.py \
        --out scripts/transfer_analysis_v5/results/per_regime_source_jackknife.csv
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
FAMILY = {"synthetic":"sdf3d","synthetic_large_zoom":"sdf3d",
          "synthetic_small_zoom":"sdf3d","synthetic_random_flipping":"sdf3d",
          "synthetic_2d_warp":"warp2d","imagenet2dwarp":"warp2d",
          "movi_f":"kubric","flyingthings":"realflow","pointodyssey":"realflow",
          "sintel":"realflow","spair":"semantic"}

def regime_of(v):
    a,p,_=v.split("|"); return "scratch" if (p=="False" or a=="raft") else "pretrained"

def regime_dir_means(t, target, keep_sources):
    """Return {regime: {'precision':.., 'recall':..}} over kept sources."""
    sub=t[t.train_dataset.isin(keep_sources)].copy()
    out={}
    for reg in ("scratch","pretrained"):
        rr=sub[sub.regime==reg]
        def vctx(col):
            rs=[spearmanr(c[target], -c[col]).statistic
                for _,c in rr.groupby("cv")
                if c.train_dataset.nunique()>=3 and c[col].std()>1e-12]
            rs=[r for r in rs if np.isfinite(r)]
            return float(np.nanmean(rs)) if rs else float("nan")
        out[reg]=dict(precision=vctx("mean_nn_a_to_b"), recall=vctx("mean_nn_b_to_a"))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--table", default="scripts/transfer_analysis_v3/transfer_table.csv")
    ap.add_argument("--dist", default="analysis_v3/pairwise_self_distances.csv")
    ap.add_argument("--target", default="peak_pck")
    ap.add_argument("--out", default="scripts/transfer_analysis_v5/results/per_regime_source_jackknife.csv")
    a=ap.parse_args()
    t=pd.read_csv(a.table); t=t[t.train_dataset.isin(PURE)].copy()
    t["variant"]=t.model_family.astype(str)+"|"+t.pretrained.astype(str)+"|"+t.freeze.astype(str)
    t=t[t.variant!="raft|False|False"]; t["regime"]=t.variant.map(regime_of)
    t["cv"]=t.benchmark+"|"+t.variant
    d=pd.read_csv(a.dist); te=d[(d.pair_type=="train_eval")&(d.space=="flow")]
    f=te.set_index(["dataset_a","dataset_b"])[["mean_nn_a_to_b","mean_nn_b_to_a"]]
    t=t.join(f,on=["train_dataset","benchmark"],how="left").dropna(subset=[a.target,"mean_nn_a_to_b"])

    rows=[]
    base=regime_dir_means(t,a.target,PURE)
    for reg in ("scratch","pretrained"):
        rows.append(dict(drop="(none/baseline)",kind="baseline",regime=reg,**base[reg]))
    # LOSO
    for s in PURE:
        keep=[x for x in PURE if x!=s]; m=regime_dir_means(t,a.target,keep)
        for reg in ("scratch","pretrained"):
            rows.append(dict(drop=f"-{s}",kind="LOSO",regime=reg,**m[reg]))
    # LOFO
    for fam in sorted(set(FAMILY.values())):
        keep=[x for x in PURE if FAMILY[x]!=fam]; m=regime_dir_means(t,a.target,keep)
        for reg in ("scratch","pretrained"):
            rows.append(dict(drop=f"-fam:{fam}",kind="LOFO",regime=reg,**m[reg]))
    df=pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.out,index=False)

    def band(reg, direction):
        sub=df[(df.regime==reg)&(df.kind.isin(["LOSO","LOFO"]))]
        b=df[(df.regime==reg)&(df.kind=="baseline")][direction].iloc[0]
        return b, sub[direction].min(), sub[direction].max()
    print(f"{'='*78}\nBASELINE regime means (all 11 sources):")
    for reg in ("scratch","pretrained"):
        print(f"  {reg:11s} precision={base[reg]['precision']:+.3f}  recall={base[reg]['recall']:+.3f}")
    print("\nJACKKNIFE RANGE under every single source/family drop (16 drops each):")
    for reg in ("scratch","pretrained"):
        for dirn in ("precision","recall"):
            b,lo,hi=band(reg,dirn)
            flag = "  <-- regime-relevant" if (reg=="scratch" and dirn=="precision") or (reg=="pretrained" and dirn=="recall") else ""
            print(f"  {reg:11s} {dirn:9s} base {b:+.3f}  range [{lo:+.3f}, {hi:+.3f}]  span {hi-lo:.3f}{flag}")
    print("\nMost influential single drops (largest move of the regime-relevant direction):")
    for reg,dirn in (("scratch","precision"),("pretrained","recall")):
        sub=df[(df.regime==reg)&(df.kind.isin(["LOSO","LOFO"]))].copy()
        b=df[(df.regime==reg)&(df.kind=="baseline")][dirn].iloc[0]
        sub["delta"]=sub[dirn]-b
        sub=sub.reindex(sub.delta.abs().sort_values(ascending=False).index)
        print(f"  {reg}/{dirn} (base {b:+.3f}):")
        for _,r in sub.head(4).iterrows():
            print(f"      {r['drop']:>22s}  {dirn}={r[dirn]:+.3f}  (Δ{r['delta']:+.3f})")
    print(f"\nwrote {a.out}")

if __name__=="__main__": main()
