"""Stratified Table 1 (motion) + Table 2 (DINO), symmetric. Rows: arch x regime
(scratch=FF, pretrained=TT). Cols: real-motion vs semantic x {dTB, dBT, sym, W2}.
Includes FlowFormer (matched-epoch from the partial RC pull)."""
import pandas as pd, numpy as np, glob, os, sys
from scipy.stats import spearmanr
PURE=["flyingthings","imagenet2dwarp","movi_f","pointodyssey","sintel","spair","synthetic",
      "synthetic_2d_warp","synthetic_large_zoom","synthetic_random_flipping","synthetic_small_zoom"]
REAL=['kitti2012','kitti2015','flyingthings','pointodyssey','synthetic']
SEM =['spair','pfpascal','pfwillow','tss']
CAP=150
# ---- distance lookups, all keyed (source,benchmark) ----
d=pd.read_csv('analysis_v3/pairwise_self_distances.csv')
def mn(space):
    x=d[(d.pair_type=='train_eval')&(d.space==space)].set_index(['dataset_a','dataset_b'])
    return x[['mean_nn_a_to_b','mean_nn_b_to_a','mean_nn_sym']]
flow=mn('flow').rename(columns=lambda c:'mot_'+c); dino=mn('dino').rename(columns=lambda c:'dino_'+c)
t0=pd.read_csv('scripts/transfer_analysis_v3/transfer_table.csv')
w2=t0[t0.train_dataset.isin(PURE)].groupby(['train_dataset','benchmark'])[['flow_sliced_w2','dino_sliced_w2']].first()
w2.index.names=['dataset_a','dataset_b']
DIST=flow.join(dino).join(w2)
# motion cols: prec=mot_mean_nn_a_to_b, rec=mot_mean_nn_b_to_a, sym=mot_mean_nn_sym, w2=flow_sliced_w2
MOT={'dTB':'mot_mean_nn_a_to_b','dBT':'mot_mean_nn_b_to_a','sym':'mot_mean_nn_sym','W2':'flow_sliced_w2'}
DIN={'dTB':'dino_mean_nn_a_to_b','dBT':'dino_mean_nn_b_to_a','sym':'dino_mean_nn_sym','W2':'dino_sliced_w2'}

# ---- unified peak_pck (canonical FF/TT + flowformer) ----
rows=[]
t=t0[t0.train_dataset.isin(PURE)]
for _,r in t.iterrows():
    a=r.model_family
    reg='FF' if (a=='raft' or (r.pretrained==False and r.freeze==False)) else ('TT' if (r.pretrained==True and r.freeze==True) else None)
    if reg: rows.append(dict(arch=a,regime=reg,source=r.train_dataset,benchmark=r.benchmark,peak=r.peak_pck))
# FlowFormer: full RC download (2026-06-13); dedup to the max-epoch run per (source,regime).
_ffbest={}
for f in glob.glob('scripts/transfer_analysis_v5/flowformer_rc_results/**/validation_results.csv',recursive=True):
    dn=os.path.basename(os.path.dirname(f)); src=dn.split('_flowformer')[0]
    if src not in PURE: continue
    pre='pretrainTrue' in dn; fz='freezeTrue' in dn
    reg='FF' if (not pre and not fz) else ('TT' if (pre and fz) else None)
    if not reg: continue
    try: mx=float(pd.read_csv(f).epoch.max())
    except Exception: continue
    k=(src,reg)
    if k not in _ffbest or mx>_ffbest[k][0]: _ffbest[k]=(mx,f)
for (src,reg),(mx,f) in _ffbest.items():
    v=pd.read_csv(f); v=v[v.epoch<=CAP]
    for b,g in v.groupby('benchmark'): rows.append(dict(arch='flowformer',regime=reg,source=src,benchmark=b,peak=float(g.pck.max())))
P=pd.DataFrame(rows).join(DIST,on=['source','benchmark'])

def cell(arch,reg,bset,col):
    sub=P[(P.arch==arch)&(P.regime==reg)&(P.benchmark.isin(bset))].dropna(subset=['peak',col])
    rs=[spearmanr(g.peak,-g[col]).statistic for _,g in sub.groupby('benchmark') if g.source.nunique()>=3 and g[col].std()>1e-12]
    rs=[x for x in rs if np.isfinite(x)]
    return np.mean(rs) if rs else np.nan

ROWS=[('catspp','FF'),('glunet','FF'),('raft','FF'),('flowformer','FF'),
      ('catspp','TT'),('glunet','TT'),('flowformer','TT')]
ANAME={'catspp':'CATs++','glunet':'GLU-Net','raft':'RAFT','flowformer':'FlowFormer'}

def render(space, M):
    # dBT (recall/coverage) is the column the narrative points at -> bold it in
    # both panels so the eye lands on "which part of motion". The one bolded
    # value that comes out negative (CATs++ scratch, semantic) is exactly the
    # floor-saturation exception, and standing it out serves the story.
    fmt=lambda x: '--' if np.isnan(x) else f"{x:+.2f}"
    fmtb=lambda x: '--' if np.isnan(x) else r"\textbf{"+f"{x:+.2f}"+"}"
    L=[r"\begin{tabular}{ll rrrr rrrr}", r"\toprule",
       r" & & \multicolumn{4}{c}{Real-motion targets} & \multicolumn{4}{c}{Semantic targets} \\",
       r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
       r"Arch. & Regime & $\dtb$ & $\dbt$ & sym & W2 & $\dtb$ & $\dbt$ & sym & W2 \\", r"\midrule"]
    prev=None
    for arch,reg in ROWS:
        if reg!=prev and prev is not None: L.append(r"\midrule")
        prev=reg
        regname='scratch' if reg=='FF' else 'pretrained'
        f=lambda k,bset: (fmtb if k=='dBT' else fmt)(cell(arch,reg,bset,M[k]))
        vals=[f(k,REAL) for k in ['dTB','dBT','sym','W2']]+\
             [f(k,SEM)  for k in ['dTB','dBT','sym','W2']]
        L.append(f"{ANAME[arch]} & {regname} & "+" & ".join(vals)+r" \\")
    L+= [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)

TAB=os.path.join(os.path.dirname(__file__),'..','..','ACCV_2026','tables')
print("################ MOTION (Table 1) ################")
print(render('flow',MOT))
print("\n################ DINO (Table 2) ################")
print(render('dino',DIN))
open('/tmp/tab_law_strat.tex','w').write(render('flow',MOT)+"\n")
open('/tmp/tab_law_dino_strat.tex','w').write(render('dino',DIN)+"\n")
open(os.path.join(TAB,'tab_law.tex'),'w').write(render('flow',MOT)+"\n")
open(os.path.join(TAB,'tab_law_dino.tex'),'w').write(render('dino',DIN)+"\n")
