"""Recompute ALL paper numbers on the DESIGN-DEFINED context set.

Design-defined = each architecture in its intended training regime:
  - CATs++, GLU-Net, FlowFormer with a PRETRAINED backbone (both encoder states)
  - RAFT (the only from-scratch-designed arch) from scratch, REAL-MOTION targets only.
No outcome-based exclusion. Distances/peaks identical to make_stratified_law_tables.py.
Outputs the headline coverage rho, per-metric within-context rho (tab_predictors),
selection regret, and the reproducibility ceiling. Read-only.
"""
import glob, os, re
import numpy as np, pandas as pd
from scipy.stats import spearmanr

PURE = ['flyingthings','imagenet2dwarp','movi_f','pointodyssey','sintel','spair',
        'synthetic','synthetic_2d_warp','synthetic_large_zoom','synthetic_random_flipping','synthetic_small_zoom']
REAL = ['kitti2012','kitti2015','flyingthings','pointodyssey','synthetic']
SEM  = ['spair','pfpascal','pfwillow','tss']

# ---- distances keyed (source, benchmark) ----
d = pd.read_csv('analysis_v3/pairwise_self_distances.csv')
def mn(space):
    x = d[(d.pair_type=='train_eval')&(d.space==space)].set_index(['dataset_a','dataset_b'])
    return x[['mean_nn_a_to_b','mean_nn_b_to_a','mean_nn_sym']]
flow = mn('flow').rename(columns=lambda c:'mot_'+c); dino = mn('dino').rename(columns=lambda c:'dino_'+c)
t0 = pd.read_csv('scripts/transfer_analysis_v3/transfer_table.csv')
extra = t0[t0.train_dataset.isin(PURE)].groupby(['train_dataset','benchmark'])[['flow_sliced_w2','dino_sliced_w2','flow_fid','dino_fid']].first()
extra.index.names = ['dataset_a','dataset_b']
DIST = flow.join(dino).join(extra)
# metric -> distance column (motion + appearance)
MOT = {'coverage':'mot_mean_nn_b_to_a','precision':'mot_mean_nn_a_to_b','sym':'mot_mean_nn_sym','W2':'flow_sliced_w2','FID':'flow_fid'}
DIN = {'coverage':'dino_mean_nn_b_to_a','precision':'dino_mean_nn_a_to_b','sym':'dino_mean_nn_sym','W2':'dino_sliced_w2','FID':'dino_fid'}

# ---- peak_pck for design-defined variants ----
rows = []
t = t0[t0.train_dataset.isin(PURE)]
for _, r in t.iterrows():
    a = str(r.model_family).lower()
    if a == 'raft':
        rows.append(dict(variant='RAFT|scratch', source=r.train_dataset, benchmark=r.benchmark, peak=r.peak_pck, raft=True))
    elif a in ('catspp','glunet') and r.pretrained == True:
        fr = 'frz' if r.freeze else 'trn'
        rows.append(dict(variant=f'{a}|pre|{fr}', source=r.train_dataset, benchmark=r.benchmark, peak=r.peak_pck, raft=False))
# FlowFormer pretrained (harvest validation_results, dedup to max-epoch run per (source,regime))
ffroot = 'scripts/transfer_analysis_v5/flowformer_rc_results'
ffbest = {}
for f in glob.glob(ffroot+'/*/validation_results.csv'):
    name = os.path.basename(os.path.dirname(f))
    if '_flowformer_steps100_' not in name or 'pretrainTrue' not in name: continue
    src = name.split('_flowformer_steps100_')[0]
    if src not in PURE: continue
    fr = 'frz' if 'freezeTrue' in name else 'trn'
    try: df = pd.read_csv(f)
    except: continue
    key = (src, fr); mx = df.epoch.max()
    if key not in ffbest or mx > ffbest[key][0]:
        ffbest[key] = (mx, df)
for (src, fr), (mx, df) in ffbest.items():
    for b, g in df.groupby('benchmark'):
        rows.append(dict(variant=f'flowformer|pre|{fr}', source=src, benchmark=b, peak=float(g.pck.max()), raft=False))
P = pd.DataFrame(rows)
# attach distances
for col in set(list(MOT.values())+list(DIN.values())):
    P[col] = [DIST[col].get((s,b), np.nan) if (s,b) in DIST.index else np.nan for s,b in zip(P.source,P.benchmark)]

def context_rho(metric_col, benches, raft_ok=True):
    rs = []
    for v in P.variant.unique():
        if (not raft_ok) and v.startswith('RAFT'): continue
        sub = P[P.variant==v]
        for b in benches:
            if v.startswith('RAFT') and b in SEM:   # RAFT is real-motion only (off-design on semantic)
                continue
            g = sub[sub.benchmark==b].dropna(subset=['peak',metric_col])
            if g.source.nunique()>=3 and g[metric_col].std()>1e-12:
                rs.append(spearmanr(g.peak, -g[metric_col]).statistic)
    return np.array(rs)

print("="*64); print("DESIGN-DEFINED within-context rho (mean over contexts)"); print("="*64)
print(f"{'metric':12} {'real':>7} {'sem':>7} {'all':>7}   appearance real/sem/all")
for m in ['coverage','precision','sym','W2','FID']:
    rr=context_rho(MOT[m],REAL); rs=context_rho(MOT[m],SEM)
    dr=context_rho(DIN[m],REAL); ds=context_rho(DIN[m],SEM)
    allm=np.concatenate([rr,rs]); alld=np.concatenate([dr,ds])
    print(f"{m:12} {rr.mean():+7.2f} {rs.mean():+7.2f} {allm.mean():+7.2f}    {dr.mean():+.2f}/{ds.mean():+.2f}/{alld.mean():+.2f}")

# ---- selection regret (coverage-ranked pick vs oracle), median peak-PCK given up ----
def regret():
    cov, rnd = [], []
    for v in P.variant.unique():
        sub = P[P.variant==v]
        benches = REAL if v.startswith('RAFT') else REAL+SEM
        for b in benches:
            g = sub[sub.benchmark==b].dropna(subset=['peak',MOT['coverage']])
            if g.source.nunique()<3 or g[MOT['coverage']].std()<1e-12: continue
            best = g.peak.max()
            pick = g.loc[g[MOT['coverage']].idxmin(),'peak']   # min coverage-gap = best coverage
            cov.append(best-pick); rnd.append(best-g.peak.mean())
    return np.median(cov), np.median(rnd)
rc, rr_ = regret()
print(f"\nSelection regret (median peak-PCK given up): coverage={rc:.1f}   random={rr_:.1f}")

# ---- reproducibility ceiling: mean pairwise rank-agreement between variants ----
def ceiling():
    ags = []
    for b in REAL+SEM:
        vs = [v for v in P.variant.unique() if not (v.startswith('RAFT') and b in SEM)]
        piv = P[(P.benchmark==b)&(P.variant.isin(vs))].pivot_table(index='source',columns='variant',values='peak')
        cols = piv.columns
        for i in range(len(cols)):
            for j in range(i+1,len(cols)):
                pair = piv[[cols[i],cols[j]]].dropna()
                if len(pair)>=3 and pair.iloc[:,0].std()>0 and pair.iloc[:,1].std()>0:
                    ags.append(spearmanr(pair.iloc[:,0],pair.iloc[:,1]).statistic)
    return np.mean(ags)
cl = ceiling()
print(f"Reproducibility ceiling (mean pairwise rank-agreement): {cl:+.2f}")
print(f"\nVariants ({P.variant.nunique()}): {sorted(P.variant.unique())}")
