"""Cross-architecture stratified predictability: does recall/coverage always work,
and is the scratch-precision lean just a semantic floor? Includes FlowFormer
(matched-epoch from the partial RC pull). Real-motion vs semantic benchmarks.
"""
import pandas as pd, numpy as np, glob, os, sys
from scipy.stats import spearmanr
PURE=["flyingthings","imagenet2dwarp","movi_f","pointodyssey","sintel","spair","synthetic",
      "synthetic_2d_warp","synthetic_large_zoom","synthetic_random_flipping","synthetic_small_zoom"]
REAL=['kitti2012','kitti2015','flyingthings','pointodyssey','synthetic']
SEM =['spair','pfpascal','pfwillow','tss']
CAP=int(sys.argv[1]) if len(sys.argv)>1 else 150
d=pd.read_csv('analysis_v3/pairwise_self_distances.csv')
te=d[(d.pair_type=='train_eval')&(d.space=='flow')]
FD=te.set_index(['dataset_a','dataset_b'])[['mean_nn_a_to_b','mean_nn_b_to_a','mean_nn_sym']]

rows=[]
# canonical-table models
t=pd.read_csv('scripts/transfer_analysis_v3/transfer_table.csv'); t=t[t.train_dataset.isin(PURE)]
for _,r in t.iterrows():
    arch=r.model_family
    # regime: FF=real scratch (pretrained False, unfrozen); TT=pretrained frozen; skip degenerate F|T
    reg=None
    if arch=='raft': reg='FF'  # raft is always scratch
    elif r.pretrained==False and r.freeze==False: reg='FF'
    elif r.pretrained==True and r.freeze==True: reg='TT'
    if reg: rows.append(dict(arch=arch,regime=reg,source=r.train_dataset,benchmark=r.benchmark,peak_pck=r.peak_pck))
# flowformer matched-epoch
for f in glob.glob('.audit_scratch/flowformer_pull/**/validation_results.csv',recursive=True):
    dn=os.path.basename(os.path.dirname(f)); src=dn.split('_flowformer')[0]
    if src not in PURE: continue
    pre='pretrainTrue' in dn; fz='freezeTrue' in dn
    reg='FF' if (not pre and not fz) else ('TT' if (pre and fz) else None)
    if not reg: continue
    v=pd.read_csv(f); v=v[v.epoch<=CAP]
    for b,g in v.groupby('benchmark'):
        rows.append(dict(arch='flowformer',regime=reg,source=src,benchmark=b,peak_pck=float(g.pck.max())))
df=pd.DataFrame(rows)
df=df.join(FD,on=['source','benchmark']).dropna(subset=['peak_pck','mean_nn_a_to_b'])
df=df[df.benchmark.isin(REAL+SEM)]

def rho(g,c): return spearmanr(g.peak_pck,-g[c]).statistic if g.source.nunique()>=3 and g[c].std()>1e-12 else np.nan
def block(sub):
    out={}
    for c,nm in [('mean_nn_a_to_b','prec'),('mean_nn_b_to_a','rec'),('mean_nn_sym','sym')]:
        rs=[rho(g,c) for _,g in sub.groupby('benchmark')]; out[nm]=np.nanmean([x for x in rs if np.isfinite(x)])
    out['spread']=sub.groupby('benchmark').apply(lambda g:g.peak_pck.max()-g.peak_pck.min()).mean()
    return out
print(f"=== matched-epoch cap<={CAP} (flowformer); regimes FF=scratch, TT=pretrained-frozen ===")
print(f"{'arch':11s} {'reg':3s} {'type':11s} | {'prec':>6} {'recall':>7} {'sym':>6} | {'pck_spread(floor?)':>18}")
for arch in ['catspp','glunet','raft','flowformer']:
    for reg in ['FF','TT']:
        for typ,bset in [('real-motion',REAL),('semantic',SEM)]:
            sub=df[(df.arch==arch)&(df.regime==reg)&(df.benchmark.isin(bset))]
            if sub.source.nunique()<3 or sub.benchmark.nunique()<1: continue
            o=block(sub)
            print(f"{arch:11s} {reg:3s} {typ:11s} | {o['prec']:>+6.2f} {o['rec']:>+7.2f} {o['sym']:>+6.2f} | {o['spread']:>18.1f}")
    print()
df.to_csv('scripts/transfer_analysis_v5/results/stratified_predictability_rows.csv',index=False)
