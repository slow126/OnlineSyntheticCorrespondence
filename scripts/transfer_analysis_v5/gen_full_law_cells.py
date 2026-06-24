"""Emit within-context rho cells for EVERY config (all regimes), stratified
real-motion / semantic, metrics dTB/dBT/sym/W2 -- validated method (reproduces
published tab:law). Used to build the supp full table + the main design-defined
subset. Read-only; prints LaTeX-ready rows."""
import glob, os, numpy as np, pandas as pd
from scipy.stats import spearmanr
PURE=['flyingthings','imagenet2dwarp','movi_f','pointodyssey','sintel','spair','synthetic','synthetic_2d_warp','synthetic_large_zoom','synthetic_random_flipping','synthetic_small_zoom']
REAL=['kitti2012','kitti2015','flyingthings','pointodyssey','synthetic']; SEM=['spair','pfpascal','pfwillow','tss']
d=pd.read_csv('analysis_v3/pairwise_self_distances.csv')
fl=d[(d.pair_type=='train_eval')&(d.space=='flow')].set_index(['dataset_a','dataset_b'])
DCOL={'dTB':'mean_nn_a_to_b','dBT':'mean_nn_b_to_a','sym':'mean_nn_sym'}
t0=pd.read_csv('scripts/transfer_analysis_v3/transfer_table.csv')
w2=t0[t0.train_dataset.isin(PURE)].groupby(['train_dataset','benchmark'])['flow_sliced_w2'].first()
def cells(df):  # df: source,benchmark,peak
    out={}
    for strat,bs in [('real',REAL),('sem',SEM)]:
        for m in ['dTB','dBT','sym','W2']:
            rs=[]
            for b in bs:
                g=df[df.benchmark==b].copy()
                if m=='W2': g['x']=[w2.get((s,b),np.nan) for s in g.source]
                else: g['x']=[fl[DCOL[m]].get((s,b),np.nan) if (s,b) in fl.index else np.nan for s in g.source]
                g=g.dropna(subset=['peak','x'])
                if g.source.nunique()>=3 and g.x.std()>1e-12: rs.append(spearmanr(g.peak,-g.x).statistic)
            out[(strat,m)]=np.mean(rs) if rs else np.nan
    return out
def fnum(x): return '--' if (x!=x) else f'{x:+.2f}'
def row(label,enc,c):
    r=[fnum(c[('real',m)]) for m in ['dTB','dBT','sym','W2']]+[fnum(c[('sem',m)]) for m in ['dTB','dBT','sym','W2']]
    return f'{label} & {enc} & '+' & '.join(r)+r' \\'
# CATs/GLU/RAFT from table
t=t0[t0.train_dataset.isin(PURE)].rename(columns={'train_dataset':'source','peak_pck':'peak'})
print('% --- canonical (CATs++, GLU-Net, RAFT) ---')
for arch,name in [('catspp','CATs++'),('glunet','GLU-Net')]:
    for pre in [False,True]:
        for frz in [False,True]:
            v=t[(t.model_family.str.lower()==arch)&(t.pretrained==pre)&(t.freeze==frz)]
            if len(v)==0: continue
            enc=r'\enctrained' if not frz else r'\encfrozen'
            reg='scratch' if not pre else 'pre'
            print(f'%{reg}:', row(name,enc,cells(v)))
vr=t[t.model_family.str.lower()=='raft']
print('%raft:', row('RAFT',r'\enctrained',cells(vr)))
# FlowFormer all regimes from harvest
ffbest={}
for f in glob.glob('scripts/transfer_analysis_v5/flowformer_rc_results/*/validation_results.csv'):
    n=os.path.basename(os.path.dirname(f))
    if '_flowformer_steps100_' not in n: continue
    s=n.split('_flowformer_steps100_')[0]
    if s not in PURE: continue
    pre='pretrainTrue' in n; frz='freezeTrue' in n
    try: df=pd.read_csv(f)
    except: continue
    k=(s,pre,frz); mx=df.epoch.max()
    if k not in ffbest or mx>ffbest[k][0]: ffbest[k]=(mx,df)
rows=[]
for (s,pre,frz),(mx,df) in ffbest.items():
    for b,g in df.groupby('benchmark'): rows.append(dict(source=s,pre=pre,frz=frz,benchmark=b,peak=float(g.pck.max())))
F=pd.DataFrame(rows)
print('% --- FlowFormer ---')
for pre in [False,True]:
    for frz in [False,True]:
        v=F[(F.pre==pre)&(F.frz==frz)]
        if len(v)==0: continue
        enc=r'\enctrained' if not frz else r'\encfrozen'
        reg='scratch' if not pre else 'pre'
        print(f'%{reg}:', row('FlowFormer',enc,cells(v)))
