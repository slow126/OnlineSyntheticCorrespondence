#!/usr/bin/env python3
"""Pretrained-only k-means codebook estimator table (directional).
Within-context Spearman(peak_pck, -distance) using count-weighted k-means
codebook coverage AUC. Columns: coverage (dbt), off-target (dtb), symmetric."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr

REAL=['kitti2012','kitti2015','flyingthings','pointodyssey','synthetic']
SEM =['spair','pfpascal','pfwillow','tss']; ALLB=REAL+SEM
PURE=['flyingthings','imagenet2dwarp','movi_f','pointodyssey','sintel','spair','synthetic',
      'synthetic_2d_warp','synthetic_large_zoom','synthetic_random_flipping','synthetic_small_zoom']

km=pd.read_csv('analysis/coverage_v2_flow_only_raw_kmeans_curve_summary.csv').set_index(
    ['train_dataset','eval_dataset'])
t=pd.read_csv('scripts/transfer_analysis_v3/transfer_table_nomid.csv')
t=t[t.train_dataset.isin(PURE)].copy()
def variant(r):
    a=str(r.model_family).lower()
    return 'RAFT|scr|t' if a=='raft' else f"{a}|{'pre' if r.pretrained else 'scr'}|{'f' if r.freeze else 't'}"
t['variant']=t.apply(variant,axis=1)

def dist(s,b,which):
    if (s,b) not in km.index: return np.nan
    cov=km.loc[(s,b),'eval_to_train_auc_weighted']   # benchmark covered by source = coverage dbt
    off=km.loc[(s,b),'train_to_eval_auc_weighted']   # source covered by benchmark = off-target dtb
    return {'cov':cov,'off':off,'sym':0.5*(cov+off)}[which]  # higher AUC = better (sign already +)

def wc(sub,which):
    rs=[]
    for b in ALLB:
        g=sub[sub.benchmark==b]
        if g.train_dataset.nunique()<3: continue
        x=np.array([dist(s,b,which) for s in g.train_dataset]); y=g.peak_pck.to_numpy(float)
        m=np.isfinite(x)&np.isfinite(y)
        if m.sum()<3 or np.nanstd(x[m])<1e-12: continue
        r=spearmanr(y[m],x[m]).statistic
        if np.isfinite(r): rs.append(r)
    return np.mean(rs) if rs else np.nan

ROWS=[('CATs++','catspp|pre|t',r'\enctrained'),('CATs++','catspp|pre|f',r'\encfrozen'),
      ('GLU-Net','glunet|pre|t',r'\enctrained'),('GLU-Net','glunet|pre|f',r'\encfrozen')]

lines=[r'\begin{tabular}{l c rrr}',r'\toprule',
  r'Arch. & Enc. & coverage $\dbt$ & off-target $\dtb$ & symmetric \\',
  r'\midrule',
  r'\multicolumn{5}{@{}l}{\textit{Pretrained backbone}}\\']
print(f'{"cfg":14s}  cov     off     sym')
for arch,v,enc in ROWS:
    sub=t[t.variant==v]; c,o,sy=wc(sub,'cov'),wc(sub,'off'),wc(sub,'sym')
    print(f'{v:14s}  {c:+0.2f}   {o:+0.2f}   {sy:+0.2f}')
    lines.append(f'{arch:7s} & {enc} & \\bst{{{c:+.2f}}} & {o:+.2f} & {sy:+.2f} '+r'\\')
lines+=[r'\bottomrule',r'\end{tabular}']
open('ACCV_2026/tables/tab_supp_kmeans.tex','w').write('\n'.join(lines)+'\n')
print('\nwrote ACCV_2026/tables/tab_supp_kmeans.tex')
