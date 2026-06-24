"""Fig. 2 (headline scatter): within-context z-scored transfer PCK vs motion
coverage (-d_{B->T}), pooled across contexts. Two panels: LEFT all configs (mark
the from-scratch x semantic degenerate cells), RIGHT with that regime dropped.

Reads transfer_table.csv (the same table the rho-tables use), filters to the 9
non-Middlebury benchmarks, and folds FlowFormer in with run-dir dedup so the left
panel prints exactly n = 142 models x 9 = 1,278 plottable cells. Overwrites
ACCV_2026/figures/results/F_coverage_scatter.png and prints n + Pearson r.
"""
from __future__ import annotations
import glob, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

ROOT = "/home/spencer/Projects/OnlineSyntheticCorrespondence"
OUT  = f"{ROOT}/ACCV_2026/figures/results/F_coverage_scatter.png"
PURE = ["flyingthings","imagenet2dwarp","movi_f","pointodyssey","sintel","spair","synthetic",
        "synthetic_2d_warp","synthetic_large_zoom","synthetic_random_flipping","synthetic_small_zoom"]
REAL = ['kitti2012','kitti2015','flyingthings','pointodyssey','synthetic']
SEM  = ['spair','pfpascal','pfwillow','tss']
BENCH = REAL + SEM
COV  = 'flow_mean_nn_eval_to_train_k1'   # d_{B->T}; coverage axis = -COV
CAP  = 150

df = pd.read_csv(f"{ROOT}/scripts/transfer_analysis_v3/transfer_table_nomid.csv")
df = df[df.train_dataset.isin(PURE) & df.benchmark.isin(BENCH)].copy()
covlut = df.dropna(subset=[COV]).groupby(['train_dataset','benchmark'])[COV].first()
def cfg(r):
    if r.model_family=='raft': return 'FF'
    return {(False,False):'FF',(False,True):'FT',(True,False):'TF',(True,True):'TT'}[(r.pretrained,r.freeze)]
df['cfg']=df.apply(cfg,axis=1); df['arch']=df.model_family
rows=[df[['arch','cfg','train_dataset','benchmark','peak_pck',COV]]
        .rename(columns={'train_dataset':'source','peak_pck':'peak',COV:'cov'})]
# FlowFormer, deduplicated to one max-PCK row per (cfg,src,benchmark)
ff=[]
for f in glob.glob(f"{ROOT}/scripts/transfer_analysis_v5/flowformer_rc_results/**/validation_results.csv",recursive=True):
    dn=os.path.basename(os.path.dirname(f)); src=dn.split('_flowformer')[0]
    if src not in PURE: continue
    pre='pretrainTrue' in dn; fz='freezeTrue' in dn
    c={(False,False):'FF',(False,True):'FT',(True,False):'TF',(True,True):'TT'}[(pre,fz)]
    try: v=pd.read_csv(f)
    except Exception: continue
    v=v[v.epoch<=CAP]
    for b,g in v.groupby('benchmark'):
        if b in BENCH:
            ff.append(dict(arch='flowformer',cfg=c,source=src,benchmark=b,peak=float(g.pck.max()),cov=covlut.get((src,b),np.nan)))
if ff:
    ffdf=pd.DataFrame(ff).groupby(['arch','cfg','source','benchmark'],as_index=False).agg(peak=('peak','max'),cov=('cov','first'))
    rows.append(ffdf)
P=pd.concat(rows,ignore_index=True).dropna(subset=['peak','cov'])
P['scratch']=P.cfg.isin(['FF','FT']); P['issem']=P.benchmark.isin(SEM)
P['ctx']=P.arch+'|'+P.cfg+'|'+P.benchmark

# within-context z-scores (transform keeps row alignment exact)
def zt(s):
    sd=s.std(ddof=0); return (s-s.mean())/sd if sd>1e-9 else s*0.0
P['zpck']=P.groupby('ctx')['peak'].transform(zt)
P['zcov']=-P.groupby('ctx')['cov'].transform(zt)   # x = -coverage z

def regime(r):
    # Design-defined: each architecture in its intended regime.
    if r.arch=='raft':                       # RAFT is from-scratch by design (real-motion only)
        return 'raft_scratch' if not r.issem else 'offdesign'
    if r.scratch:                            # CATs++/GLU-Net/FlowFormer from scratch = off-design
        return 'offdesign'
    return 'pre_sem' if r.issem else 'pre_real'
P['regime']=P.apply(regime,axis=1)
COL={'pre_real':'#2563eb','pre_sem':'#059669','raft_scratch':'#e07b39','offdesign':'#dc2626'}
LAB={'pre_real':'pretrained real-motion','pre_sem':'pretrained semantic',
     'raft_scratch':'RAFT (from scratch)','offdesign':'off-design from-scratch'}

def rval(d): return pearsonr(d.zcov,d.zpck)[0]
left=P; right=P[P.regime!='offdesign']
print(f"LEFT  n={len(left)}  r={rval(left):+.3f}")
print(f"RIGHT n={len(right)} r={rval(right):+.3f}")
for rg in ['raft_scratch','pre_real','pre_sem']:
    d=P[P.regime==rg]
    if len(d)>=2: print(f"   {rg:14s} n={len(d)} r={rval(d):+.3f}")

plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                     "figure.dpi":200,"savefig.dpi":200})
fig,(axL,axR)=plt.subplots(1,2,figsize=(9.2,3.5),sharey=True)
def panel(ax,d,title,marker_deg=False):
    for rg in ['pre_real','pre_sem','raft_scratch','offdesign']:
        if rg=='offdesign' and not marker_deg: continue
        s=d[d.regime==rg]
        if not len(s): continue
        mk='x' if rg=='offdesign' else 'o'
        ax.scatter(s.zcov,s.zpck,s=8 if mk=='o' else 14,c=COL[rg],marker=mk,
                   alpha=0.45 if mk=='o' else 0.7,linewidths=0.6 if mk=='x' else 0,label=LAB[rg])
    xs=np.array([d.zcov.min(),d.zcov.max()]); b,a=np.polyfit(d.zcov,d.zpck,1)
    ax.plot(xs,b*xs+a,'k--',lw=1.3)
    ax.set_title(title,fontsize=9.5,loc='left'); ax.set_xlabel("motion coverage ($-\\,d_{B\\to T}$, within-context $z$)")
    ax.legend(fontsize=6.5,loc='lower right',framealpha=0.9)
panel(axL,left,f"All configurations  ($r{{=}}{rval(left):.2f}$, $n{{=}}{len(left):,}$)",marker_deg=True)
panel(axR,right,f"Design-defined configurations  ($r{{=}}{rval(right):.2f}$, $n{{=}}{len(right):,}$)")
axL.set_ylabel("transfer PCK (within-context $z$)")
fig.tight_layout(); fig.savefig(OUT,bbox_inches="tight"); plt.close(fig)
print("wrote",OUT)
