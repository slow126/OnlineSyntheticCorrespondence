"""Does a LEARNED linear combo of the 6 eps features beat a single scalar?

6 features = {1,4,16 px} x {coverage (b_covered_by_a), off-target (a_covered_by_b)}.
Held-out by leave-one-source-out (LOSO): the weights must generalize to unseen
sources, matching the paper's other learned-combiner checks. We report within-context
ranking rho (real/sem/all) for: single mean-NN coverage, single best eps coverage,
the LOSO 6-eps combo, a LOSO 6-eps+mean-NN combo, and the in-fold (overfit) ceiling.
Design-defined set. Read-only.
"""
import glob, os, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
PURE=['flyingthings','imagenet2dwarp','movi_f','pointodyssey','sintel','spair','synthetic',
      'synthetic_2d_warp','synthetic_large_zoom','synthetic_random_flipping','synthetic_small_zoom']
REAL=['kitti2012','kitti2015','flyingthings','pointodyssey','synthetic']; SEM=['spair','pfpascal','pfwillow','tss']
FEATS=['b_covered_by_a_eps1px','b_covered_by_a_eps4px','b_covered_by_a_eps16px',
       'a_covered_by_b_eps1px','a_covered_by_b_eps4px','a_covered_by_b_eps16px']
d=pd.read_csv('analysis_v3/pairwise_self_distances.csv')
fl=d[(d.pair_type=='train_eval')&(d.space=='flow')].set_index(['dataset_a','dataset_b'])
# design-defined peak table
t0=pd.read_csv('scripts/transfer_analysis_v3/transfer_table.csv'); t=t0[t0.train_dataset.isin(PURE)]
rows=[]
for _,r in t.iterrows():
    a=str(r.model_family).lower()
    if a=='raft': rows.append(dict(variant='raft|scr',source=r.train_dataset,benchmark=r.benchmark,peak=r.peak_pck))
    elif a in('catspp','glunet') and r.pretrained==True:
        rows.append(dict(variant=f"{a}|pre|{'f' if r.freeze else 't'}",source=r.train_dataset,benchmark=r.benchmark,peak=r.peak_pck))
ffb={}
for f in glob.glob('scripts/transfer_analysis_v5/flowformer_rc_results/*/validation_results.csv'):
    n=os.path.basename(os.path.dirname(f))
    if '_flowformer_steps100_' not in n or 'pretrainTrue' not in n: continue
    s=n.split('_flowformer_steps100_')[0]
    if s not in PURE: continue
    fr='f' if 'freezeTrue' in n else 't'; df=pd.read_csv(f); k=(s,fr)
    if k not in ffb or df.epoch.max()>ffb[k][0]: ffb[k]=(df.epoch.max(),df)
for (s,fr),(mx,df) in ffb.items():
    for b,g in df.groupby('benchmark'): rows.append(dict(variant=f'flowformer|pre|{fr}',source=s,benchmark=b,peak=float(g.pck.max())))
P=pd.DataFrame(rows)
# attach features + mean-NN coverage; drop RAFT-semantic (off-design)
for c in FEATS+['mean_nn_b_to_a']:
    P[c]=[fl[c].get((s,b),np.nan) if (s,b) in fl.index else np.nan for s,b in zip(P.source,P.benchmark)]
P=P[~(P.variant.str.startswith('raft') & P.benchmark.isin(SEM))].dropna(subset=['peak']+FEATS+['mean_nn_b_to_a']).copy()
P['ctx']=P.variant+'|'+P.benchmark
def z(s): sd=s.std(); return (s-s.mean())/sd if sd>1e-9 else s*0.0
P['zpeak']=P.groupby('ctx')['peak'].transform(z)
for c in FEATS+['mean_nn_b_to_a']: P['z_'+c]=P.groupby('ctx')[c].transform(z)

def ctx_rho(col):
    out={}
    for nm,bs in [('real',REAL),('sem',SEM),('all',REAL+SEM)]:
        rs=[spearmanr(g.zpeak,g[col]).statistic
            for _,g in P[P.benchmark.isin(bs)].groupby('ctx')
            if g.source.nunique()>=3 and g[col].std()>1e-9]
        out[nm]=np.mean(rs)
    return out
def loso(feat_cols):
    pred=pd.Series(np.nan,index=P.index)
    for src in P.source.unique():
        tr=P[P.source!=src]; te=P[P.source==src]
        m=LinearRegression().fit(tr[feat_cols], tr.zpeak)
        pred.loc[te.index]=m.predict(te[feat_cols])
    P['_pred']=pred
    return ctx_rho('_pred')

ZE=['z_'+c for c in FEATS]; ZM=['z_mean_nn_b_to_a']
def fmt(d): return f"real {d['real']:+.2f}  sem {d['sem']:+.2f}  all {d['all']:+.2f}"
print("single mean-NN coverage :", fmt(ctx_rho('z_mean_nn_b_to_a')))
print("single eps-4px coverage :", fmt(ctx_rho('z_b_covered_by_a_eps4px')))
print("LOSO 6-eps combo        :", fmt(loso(ZE)))
print("LOSO 6-eps + mean-NN    :", fmt(loso(ZE+ZM)))
# in-fold (overfit ceiling) for the 6-eps combo
m=LinearRegression().fit(P[ZE],P.zpeak); P['_inf']=m.predict(P[ZE])
print("6-eps in-fold (overfit) :", fmt(ctx_rho('_inf')))
print("\n6-eps weights (fit on all, standardized):")
for c,w in sorted(zip(FEATS,m.coef_),key=lambda x:-abs(x[1])): print(f"  {c:28s} {w:+.3f}")
