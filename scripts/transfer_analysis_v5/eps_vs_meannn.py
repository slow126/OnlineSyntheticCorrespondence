"""Head-to-head: eps-radius coverage vs mean-NN, on the DESIGN-DEFINED set.

For each estimator we report the within-context Spearman rho (transfer vs the
predictor), in the coverage (recall) direction, the off-target (precision)
direction, and symmetric -- stratified real / semantic / all -- plus selection
regret for the coverage direction. eps fractions: higher = better (sign +);
mean-NN distances: lower = better (sign -). Read-only.
"""
import glob, os, numpy as np, pandas as pd
from scipy.stats import spearmanr
PURE=['flyingthings','imagenet2dwarp','movi_f','pointodyssey','sintel','spair','synthetic',
      'synthetic_2d_warp','synthetic_large_zoom','synthetic_random_flipping','synthetic_small_zoom']
REAL=['kitti2012','kitti2015','flyingthings','pointodyssey','synthetic']; SEM=['spair','pfpascal','pfwillow','tss']

# distances keyed (source, benchmark), flow space, train_eval
d=pd.read_csv('analysis_v3/pairwise_self_distances.csv')
fl=d[(d.pair_type=='train_eval')&(d.space=='flow')].set_index(['dataset_a','dataset_b'])
# (column, higher_is_better) per direction
COV={'mean-NN':('mean_nn_b_to_a',False),'eps1':('b_covered_by_a_eps1px',True),
     'eps4':('b_covered_by_a_eps4px',True),'eps16':('b_covered_by_a_eps16px',True)}
PREC={'mean-NN':('mean_nn_a_to_b',False),'eps1':('a_covered_by_b_eps1px',True),
      'eps4':('a_covered_by_b_eps4px',True),'eps16':('a_covered_by_b_eps16px',True)}
SYM={'mean-NN':('mean_nn_sym',False),'eps1':('sym_eps1px',True),
     'eps4':('sym_eps4px',True),'eps16':('sym_eps16px',True)}

# ---- design-defined peak table (CATs/GLU pretrained + RAFT scratch + FlowFormer pretrained) ----
t0=pd.read_csv('scripts/transfer_analysis_v3/transfer_table.csv'); t=t0[t0.train_dataset.isin(PURE)]
rows=[]
for _,r in t.iterrows():
    a=str(r.model_family).lower()
    if a=='raft': rows.append(dict(variant='raft|scr',source=r.train_dataset,benchmark=r.benchmark,peak=r.peak_pck,raft=True))
    elif a in('catspp','glunet') and r.pretrained==True:
        rows.append(dict(variant=f"{a}|pre|{'f' if r.freeze else 't'}",source=r.train_dataset,benchmark=r.benchmark,peak=r.peak_pck,raft=False))
ffb={}
for f in glob.glob('scripts/transfer_analysis_v5/flowformer_rc_results/*/validation_results.csv'):
    n=os.path.basename(os.path.dirname(f))
    if '_flowformer_steps100_' not in n or 'pretrainTrue' not in n: continue
    s=n.split('_flowformer_steps100_')[0]
    if s not in PURE: continue
    fr='f' if 'freezeTrue' in n else 't'; df=pd.read_csv(f); k=(s,fr)
    if k not in ffb or df.epoch.max()>ffb[k][0]: ffb[k]=(df.epoch.max(),df)
for (s,fr),(mx,df) in ffb.items():
    for b,g in df.groupby('benchmark'): rows.append(dict(variant=f'flowformer|pre|{fr}',source=s,benchmark=b,peak=float(g.pck.max()),raft=False))
P=pd.DataFrame(rows)

def rho(col, higher, benches):
    rs=[]
    for v in P.variant.unique():
        sub=P[P.variant==v]
        for b in benches:
            if v.startswith('raft') and b in SEM: continue
            g=sub[sub.benchmark==b].copy()
            g['x']=[fl[col].get((s,b),np.nan) if (s,b) in fl.index else np.nan for s in g.source]
            g=g.dropna(subset=['peak','x'])
            if g.source.nunique()>=3 and g.x.std()>1e-12:
                r=spearmanr(g.peak,g.x).statistic
                rs.append(r if higher else -r)   # flip so + = predictive
    return np.array(rs)
def line(name,col,higher):
    rr=rho(col,higher,REAL); rs=rho(col,higher,SEM); al=np.concatenate([rr,rs])
    return f"  {name:9s} real {rr.mean():+.2f}   sem {rs.mean():+.2f}   all {al.mean():+.2f}"

for label,D in [('COVERAGE (recall, A->B)',COV),('OFF-TARGET (precision, B->A)',PREC),('SYMMETRIC',SYM)]:
    print(f"\n=== {label} ===")
    for name,(col,hi) in D.items(): print(line(name,col,hi))

# regret for the coverage direction (median peak-PCK given up)
def regret(col, higher):
    rr=[]
    for v in P.variant.unique():
        sub=P[P.variant==v]; bs=REAL if v.startswith('raft') else REAL+SEM
        for b in bs:
            g=sub[sub.benchmark==b].copy()
            g['x']=[fl[col].get((s,b),np.nan) if (s,b) in fl.index else np.nan for s in g.source]
            g=g.dropna(subset=['peak','x'])
            if g.source.nunique()<3 or g.x.std()<1e-12: continue
            pick=g.loc[g.x.idxmax() if higher else g.x.idxmin(),'peak']
            rr.append(g.peak.max()-pick)
    return np.median(rr)
print("\n=== SELECTION REGRET (coverage direction, median PCK given up, lower better) ===")
for name,(col,hi) in COV.items(): print(f"  {name:9s} regret = {regret(col,hi):.2f}")
