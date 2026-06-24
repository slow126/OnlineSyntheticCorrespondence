"""Fig. 3: zero-shot coverage ranking vs the retraining oracle, with contrast
predictors (symmetric, off-target, appearance) as the floor.

Per true-PCK-gap bin we plot the chance each fit-free score picks the truly-better
of two candidate sources, against the empirical agreement of independent
retrainings (same- and cross-architecture = the reproducibility ceiling/ floor for
a data-only rule). Design-defined scope (each architecture in its intended regime:
pretrained backbone-dependent matchers + RAFT from scratch, real-motion only).

Overwrites ACCV_2026/figures/results/F4_gap_stratified.png and prints the numbers.
"""
from __future__ import annotations
import glob, itertools, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/spencer/Projects/OnlineSyntheticCorrespondence"
OUT  = f"{ROOT}/ACCV_2026/figures/results/F4_gap_stratified.png"
PURE = ["flyingthings","imagenet2dwarp","movi_f","pointodyssey","sintel","spair","synthetic",
        "synthetic_2d_warp","synthetic_large_zoom","synthetic_random_flipping","synthetic_small_zoom"]
REAL = ['kitti2012','kitti2015','flyingthings','pointodyssey','synthetic']
SEM  = ['spair','pfpascal','pfwillow','tss']
BENCH = REAL + SEM
COV  = 'flow_mean_nn_eval_to_train_k1'                # d_{B->T}; coverage = -COV
# predictor distances (all ranked by -dist: smaller distance => predicted better)
PREDS = [
    ("coverage ($-d_{B\\to T}$)",      'flow_mean_nn_eval_to_train_k1', "#7c3aed", 'D', 2.6, 8),
    ("symmetric $W_2$",                'flow_sliced_w2',                "#2563eb", 's', 1.8, 5),
    ("off-target ($-d_{T\\to B}$)",    'flow_mean_nn_train_to_eval_k1', "#dc2626", 'v', 1.8, 5),
    ("appearance (DINO)",              'dino_mean_nn_eval_to_train_k1', "#059669", '^', 1.8, 5),
]
GAP_BINS=[0.0,1.0,2.0,5.0,10.0,np.inf]; GAP_LABELS=["0-1","1-2","2-5","5-10",">10"]
def binlab(g): return GAP_LABELS[min(np.searchsorted(GAP_BINS,g,side="right")-1, 4)]

# ---- assemble peak + distances for ALL configs ----
df = pd.read_csv(f"{ROOT}/scripts/transfer_analysis_v3/transfer_table_nomid.csv")
df = df[df.train_dataset.isin(PURE) & df.benchmark.isin(BENCH)].copy()
DCOLS = [c for _, c, *_ in PREDS]
luts = {c: df.dropna(subset=[c]).groupby(['train_dataset','benchmark'])[c].first() for c in DCOLS}
def cfg(r):
    if r.model_family=='raft': return 'FF'
    return {(False,False):'FF',(False,True):'FT',(True,False):'TF',(True,True):'TT'}[(r.pretrained,r.freeze)]
df['cfg']=df.apply(cfg,axis=1); df['arch']=df.model_family
rows=[df[['arch','cfg','train_dataset','benchmark','peak_pck']+DCOLS]
        .rename(columns={'train_dataset':'source','peak_pck':'peak'})]
# FlowFormer: peak from matched-epoch RC pull, distances from the (src,bench) lookups
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
# design-defined: pretrained backbone-dependent matchers (all targets) + RAFT scratch (real-motion)
P=P[(~P.scratch) | ((P.arch=='raft') & (P.stratum=='real'))].copy()
P['context']=P.variant+'|'+P.benchmark

# ---- pairwise accuracy by true gap, for each predictor ----
def pairwise_acc(col):
    prec=[]
    for ctx,g in P.dropna(subset=[col]).groupby('context'):
        g=g.drop_duplicates('source'); pk=g['peak'].values; ds=g[col].values
        for i in range(len(g)):
            for j in range(i+1,len(g)):
                gap=abs(pk[i]-pk[j])
                if gap==0: continue
                prec.append((binlab(gap), (ds[i]<ds[j])==(pk[i]>pk[j])))
    pdf=pd.DataFrame(prec,columns=['gap_bin','correct'])
    return pdf.groupby('gap_bin').correct.mean().reindex(GAP_LABELS)
pacc={name: pairwise_acc(col) for name,col,*_ in PREDS}

# ---- empirical reproducibility (retraining oracle) by true gap ----
emp=[]
for bench,bt in P.groupby('benchmark'):
    variants=sorted(bt.variant.unique())
    perf={(r.variant,r.source):r.peak for r in bt.itertuples()}
    srcs={v:set(bt[bt.variant==v].source) for v in variants}
    for v1,v2 in itertools.combinations(variants,2):
        kind='same_arch' if v1.split('|')[0]==v2.split('|')[0] else 'cross_arch'
        for i,j in itertools.combinations(sorted(srcs[v1]&srcs[v2]),2):
            d1=perf[(v1,i)]-perf[(v1,j)]; d2=perf[(v2,i)]-perf[(v2,j)]
            if d1==0 or d2==0: continue
            agree=(d1>0)==(d2>0)
            for refgap in (abs(d1),abs(d2)): emp.append((kind,binlab(refgap),bool(agree)))
E=pd.DataFrame(emp,columns=['kind','gap_bin','agree'])
oracle={k:E[E.kind==k].groupby('gap_bin').agree.mean().reindex(GAP_LABELS) for k in ['same_arch','cross_arch']}

for name,col,*_ in PREDS: print(f"[{name}] >10 acc = {pacc[name].loc['>10']:.3f}")
print(f"[oracle] same_arch >10 = {oracle['same_arch'].loc['>10']:.3f} | cross_arch >10 = {oracle['cross_arch'].loc['>10']:.3f}")

# ---- render ----
GRAY="#6b7280"
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,
                     "figure.dpi":200,"savefig.dpi":200})
fig,ax=plt.subplots(figsize=(7.2,4.4))
ax.plot(GAP_LABELS,oracle['same_arch'],ls='-',marker='o',color=GRAY,ms=6,lw=1.6,
        label="retraining, same architecture (ceiling)")
ax.plot(GAP_LABELS,oracle['cross_arch'],ls='--',marker='o',color=GRAY,ms=5,lw=1.4,mfc='white',
        label="retraining, different architecture")
for name,col,c,mk,lw,ms in PREDS:
    ax.plot(GAP_LABELS,pacc[name],marker=mk,color=c,ms=ms,lw=lw,label=name)
ax.axhline(0.5,color="#9ca3af",lw=0.8,ls=':')
ax.annotate("coin flip (floor)",(0.02,0.505),xycoords=("axes fraction","data"),fontsize=8.5,color=GRAY,va="bottom")
ax.set_xlabel("how different the two candidate training sets really are\n(true peak-PCK difference between them)")
ax.set_ylabel("chance of picking the truly\nbetter training set")
ax.set_title("Zero-shot coverage ranking tracks the reproducibility of retraining",loc="left",fontsize=11.5)
ax.legend(fontsize=8.3,loc="upper left",ncol=1,framealpha=0.92); ax.set_ylim(0.24,0.98)
fig.tight_layout(); fig.savefig(OUT,bbox_inches="tight"); plt.close(fig)
print("wrote",OUT)
