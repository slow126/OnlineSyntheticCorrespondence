"""Combined 2-panel coverage-law figure (replaces the old 2-panel scatter + the
gap figure):
  (a) design-defined coverage scatter (within-context z transfer vs coverage),
  (b) zero-shot coverage ranking vs the retraining ceiling/floor by true-PCK gap,
      with contrast predictors (symmetric, off-target, appearance).
Design-defined scope throughout. Output: ACCV_2026/figures/results/F_coverage_law.png
"""
from __future__ import annotations
import glob, itertools, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

ROOT = "/home/spencer/Projects/OnlineSyntheticCorrespondence"
OUT  = f"{ROOT}/ACCV_2026/figures/results/F_coverage_law.png"
PURE = ["flyingthings","imagenet2dwarp","movi_f","pointodyssey","sintel","spair","synthetic",
        "synthetic_2d_warp","synthetic_large_zoom","synthetic_random_flipping","synthetic_small_zoom"]
REAL = ['kitti2012','kitti2015','flyingthings','pointodyssey','synthetic']
SEM  = ['spair','pfpascal','pfwillow','tss']
BENCH = REAL + SEM
COV  = 'flow_mean_nn_eval_to_train_k1'
PREDS = [("coverage ($-d_{T\\to S}$)",'flow_mean_nn_eval_to_train_k1',"#7c3aed",'D',2.6,8),
         ("symmetric $W_2$",          'flow_sliced_w2',                "#2563eb",'s',1.8,5),
         ("off-target ($-d_{S\\to T}$)",'flow_mean_nn_train_to_eval_k1',"#dc2626",'v',1.8,5),
         ("appearance (DINO)",        'dino_mean_nn_eval_to_train_k1', "#059669",'^',1.8,5)]
GAP_BINS=[0.0,1.0,2.0,5.0,10.0,np.inf]; GAP_LABELS=["0-1","1-2","2-5","5-10",">10"]
def binlab(g): return GAP_LABELS[min(np.searchsorted(GAP_BINS,g,side="right")-1,4)]

# ---- assemble peak + distances, design-defined ----
df = pd.read_csv(f"{ROOT}/scripts/transfer_analysis_v3/transfer_table_nomid.csv")
df = df[df.train_dataset.isin(PURE) & df.benchmark.isin(BENCH)].copy()
DCOLS = [c for _,c,*_ in PREDS]
luts = {c: df.dropna(subset=[c]).groupby(['train_dataset','benchmark'])[c].first() for c in DCOLS}
def cfg(r):
    if r.model_family=='raft': return 'FF'
    return {(False,False):'FF',(False,True):'FT',(True,False):'TF',(True,True):'TT'}[(r.pretrained,r.freeze)]
df['cfg']=df.apply(cfg,axis=1); df['arch']=df.model_family
rows=[df[['arch','cfg','train_dataset','benchmark','peak_pck']+DCOLS]
        .rename(columns={'train_dataset':'source','peak_pck':'peak'})]
ff=[]; CAP=150
for f in glob.glob(f"{ROOT}/scripts/transfer_analysis_v5/flowformer_rc_results/**/validation_results.csv",recursive=True):
    dn=os.path.basename(os.path.dirname(f)); src=dn.split('_flowformer')[0]
    if src not in PURE: continue
    pre='pretrainTrue' in dn; fz='freezeTrue' in dn
    c={(False,False):'FF',(False,True):'FT',(True,False):'TF',(True,True):'TT'}[(pre,fz)]
    try: v=pd.read_csv(f)
    except Exception: continue
    v=v[v.epoch<=CAP]
    for b,g in v.groupby('benchmark'):
        if b not in BENCH: continue
        rec=dict(arch='flowformer',cfg=c,source=src,benchmark=b,peak=float(g.pck.max()))
        for col in DCOLS: rec[col]=luts[col].get((src,b),np.nan)
        ff.append(rec)
if ff: rows.append(pd.DataFrame(ff))
P=pd.concat(rows,ignore_index=True).dropna(subset=['peak',COV])
P['variant']=P.arch+'|'+P.cfg
P['stratum']=np.where(P.benchmark.isin(REAL),'real','sem')
P['scratch']=P.cfg.isin(['FF','FT'])
P=P[(~P.scratch) | ((P.arch=='raft') & (P.stratum=='real'))].copy()   # design-defined
P['ctx']=P.arch+'|'+P.cfg+'|'+P.benchmark

# ---- panel (a) data: within-context z ----
def zt(s): sd=s.std(ddof=0); return (s-s.mean())/sd if sd>1e-9 else s*0.0
P['zpck']=P.groupby('ctx')['peak'].transform(zt)
P['zcov']=-P.groupby('ctx')[COV].transform(zt)
def regime(r):
    if r.arch=='raft': return 'raft'
    return 'pre_sem' if r.stratum=='sem' else 'pre_real'
P['regime']=P.apply(regime,axis=1)
RCOL={'pre_real':'#2563eb','pre_sem':'#059669','raft':'#e07b39'}
RLAB={'pre_real':'pretrained real-motion','pre_sem':'pretrained semantic','raft':'RAFT (scratch)'}
rA=pearsonr(P.zcov,P.zpck)[0]

# ---- panel (b) data: pairwise accuracy + oracle ----
def pairwise_acc(col):
    prec=[]
    for ctx,g in P.dropna(subset=[col]).groupby('ctx'):
        g=g.drop_duplicates('source'); pk=g['peak'].values; ds=g[col].values
        for i in range(len(g)):
            for j in range(i+1,len(g)):
                gap=abs(pk[i]-pk[j])
                if gap==0: continue
                prec.append((binlab(gap),(ds[i]<ds[j])==(pk[i]>pk[j])))
    return pd.DataFrame(prec,columns=['gap_bin','correct']).groupby('gap_bin').correct.mean().reindex(GAP_LABELS)
pacc={name:pairwise_acc(col) for name,col,*_ in PREDS}
emp=[]
for bench,bt in P.groupby('benchmark'):
    vs=sorted(bt.variant.unique()); perf={(r.variant,r.source):r.peak for r in bt.itertuples()}
    srcs={v:set(bt[bt.variant==v].source) for v in vs}
    for v1,v2 in itertools.combinations(vs,2):
        kind='same_arch' if v1.split('|')[0]==v2.split('|')[0] else 'cross_arch'
        for i,j in itertools.combinations(sorted(srcs[v1]&srcs[v2]),2):
            d1=perf[(v1,i)]-perf[(v1,j)]; d2=perf[(v2,i)]-perf[(v2,j)]
            if d1==0 or d2==0: continue
            for refgap in (abs(d1),abs(d2)): emp.append((kind,binlab(refgap),(d1>0)==(d2>0)))
E=pd.DataFrame(emp,columns=['kind','gap_bin','agree'])
oracle={k:E[E.kind==k].groupby('gap_bin').agree.mean().reindex(GAP_LABELS) for k in ['same_arch','cross_arch']}

# ---- render: TWO separate panels (no baked-in a/b titles; LaTeX subcaption adds them) ----
GRAY="#6b7280"
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                     "figure.dpi":200,"savefig.dpi":200})
OUTA=f"{ROOT}/ACCV_2026/figures/results/F_covlaw_a.png"
OUTB=f"{ROOT}/ACCV_2026/figures/results/F_covlaw_b.png"
# (a) scatter -- SQUARE axes, colored by TRAINING-SOURCE FAMILY (pooled for legibility)
# pool only the SDF variants into one group; keep every other source distinct
NAME={'flyingthings':'FlyingThings','pointodyssey':'PointOdyssey','sintel':'Sintel',
      'movi_f':'MOVi-F','imagenet2dwarp':'ImageNet-2D','spair':'SPair',
      'synthetic':'SDF (ours)','synthetic_2d_warp':'SDF (ours)','synthetic_large_zoom':'SDF (ours)',
      'synthetic_random_flipping':'SDF (ours)','synthetic_small_zoom':'SDF (ours)'}
ORDER=['SDF (ours)','FlyingThings','PointOdyssey','Sintel','MOVi-F','ImageNet-2D','SPair']
COL={'SDF (ours)':'#dc2626','FlyingThings':'#2563eb','PointOdyssey':'#0891b2',
     'Sintel':'#16a34a','MOVi-F':'#f59e0b','ImageNet-2D':'#7c3aed','SPair':'#db2777'}
P['grp']=P.source.map(NAME)
# pool points: average the per-context z over all model variants AND over source-variants
# that share a display group (the 5 SDF configs collapse to one SDF dot per target), so
# every source family contributes one dot per (family, target) -- equal visual weight.
agg=P.groupby(['grp','benchmark'],as_index=False).agg(
        zcov=('zcov','mean'),zpck=('zpck','mean'))
rAg=pearsonr(agg.zcov,agg.zpck)[0]
figA,axA=plt.subplots(figsize=(4.4,4.4))
for g in ORDER:
    d=agg[agg.grp==g]
    if not len(d): continue
    axA.scatter(d.zcov,d.zpck,s=26,alpha=0.85,color=COL[g],lw=0,label=g)
lim=float(np.nanmax(np.abs([agg.zcov.min(),agg.zcov.max(),agg.zpck.min(),agg.zpck.max()])))*1.1
b,a=np.polyfit(agg.zcov,agg.zpck,1)
axA.plot([-lim,lim],[b*-lim+a,b*lim+a],'k--',lw=1.3,zorder=1)  # least-squares fit (explained in caption; no slope label to avoid clashing with the n=685 stat)
axA.set_xlim(-lim,lim); axA.set_ylim(-lim,lim); axA.set_aspect('equal','box')
axA.set_xlabel("Motion coverage ($-\\,d_{T\\to S}$, within-context $z$)")
axA.set_ylabel("Transfer PCK (within-context $z$)")
axA.legend(fontsize=6.6,loc='lower right',framealpha=0.9,labelspacing=0.3)
figA.tight_layout(); figA.savefig(OUTA,bbox_inches="tight"); plt.close(figA)
# (b) gap curves
figB,axB=plt.subplots(figsize=(6.0,4.0))
axB.plot(GAP_LABELS,oracle['same_arch'],ls='-',marker='o',color=GRAY,ms=5,lw=1.5,label="retraining, same arch. (ceiling)")
axB.plot(GAP_LABELS,oracle['cross_arch'],ls='--',marker='o',color=GRAY,ms=4,lw=1.3,mfc='white',label="retraining, different arch.")
for name,col,c,mk,lw,ms in PREDS:
    axB.plot(GAP_LABELS,pacc[name],marker=mk,color=c,ms=ms,lw=lw,label=name)
axB.axhline(0.5,color="#9ca3af",lw=0.8,ls=':')
axB.annotate("coin flip (floor)",(0.02,0.505),xycoords=("axes fraction","data"),fontsize=8,color=GRAY,va="bottom")
axB.set_xlabel("True peak-PCK difference between the two candidate sources")
axB.set_ylabel("Chance of picking the better source")
axB.set_ylim(0.24,0.98); axB.legend(fontsize=7.5,loc='upper left',framealpha=0.92)
figB.tight_layout(); figB.savefig(OUTB,bbox_inches="tight"); plt.close(figB)
print(f"(a) scatter r={rA:.3f} n={len(P)} -> {OUTA}")
for name,col,*_ in PREDS: print(f"(b) {name} >10 = {pacc[name].loc['>10']:.3f}")
print(f"(b) oracle same={oracle['same_arch'].loc['>10']:.3f} cross={oracle['cross_arch'].loc['>10']:.3f} -> {OUTB}")
