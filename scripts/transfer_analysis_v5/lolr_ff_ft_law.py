"""LOLR partial CATs++ FF vs FT: per-regime real-motion/semantic split law table.
Mirrors make_stratified_law_tables.py machinery (REAL/SEM sets, distance joins,
cell() correlation) but peak_pck comes from the freshly-pulled lolr csvs.
Answers: have FF and FT diverged, and is the scratch camp still precision (dTB)
favored or has it tipped toward coverage/recall (dBT)?"""
import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
pd.set_option('display.width',200)

REAL=['kitti2012','kitti2015','flyingthings','pointodyssey','synthetic']
SEM =['spair','pfpascal','pfwillow','tss']
SNAP='cats_ff_ft_snapshtos'

# ---- distance lookups, keyed (source=dataset_a, benchmark=dataset_b) ----
d=pd.read_csv('analysis_v3/pairwise_self_distances.csv')
def mn(space):
    x=d[(d.pair_type=='train_eval')&(d.space==space)].set_index(['dataset_a','dataset_b'])
    return x[['mean_nn_a_to_b','mean_nn_b_to_a','mean_nn_sym']]
flow=mn('flow').rename(columns=lambda c:'mot_'+c); dino=mn('dino').rename(columns=lambda c:'dino_'+c)
t0=pd.read_csv('scripts/transfer_analysis_v3/transfer_table.csv')
PURE=sorted(flow.index.get_level_values(0).unique())
w2=t0[t0.train_dataset.isin(PURE)].groupby(['train_dataset','benchmark'])[['flow_sliced_w2','dino_sliced_w2']].first()
w2.index.names=['dataset_a','dataset_b']
DIST=flow.join(dino).join(w2)
MOT={'dTB':'mot_mean_nn_a_to_b','dBT':'mot_mean_nn_b_to_a','sym':'mot_mean_nn_sym','W2':'flow_sliced_w2'}
DIN={'dTB':'dino_mean_nn_a_to_b','dBT':'dino_mean_nn_b_to_a','sym':'dino_mean_nn_sym','W2':'dino_sliced_w2'}

# ---- load lolr peaks ----
def load(cap=None):
    rows=[]
    for f in sorted(glob.glob(f'{SNAP}/*/validation_results.csv')):
        dn=os.path.basename(os.path.dirname(f)); src=dn.split('_cats_lolr')[0]
        reg='FF' if 'freezeFalse' in dn else 'FT'
        v=pd.read_csv(f)
        if cap is not None: v=v[v.epoch<=cap]
        for b,g in v.groupby('benchmark'):
            rows.append(dict(regime=reg,source=src,benchmark=b,peak=float(g.pck.max()),maxep=int(g.epoch.max())))
    return pd.DataFrame(rows)

def cell(P,reg,bset,col):
    sub=P[(P.regime==reg)&(P.benchmark.isin(bset))].dropna(subset=['peak',col])
    rs=[spearmanr(g.peak,-g[col]).statistic for _,g in sub.groupby('benchmark')
        if g.source.nunique()>=3 and g[col].std()>1e-12]
    rs=[x for x in rs if np.isfinite(x)]
    return (np.mean(rs) if rs else np.nan, len(rs))

def law_table(P,M,title):
    print(f"\n################ {title} ################")
    print(f"{'reg':4s} {'set':9s} | {'dTB(prec)':>10s} {'dBT(rec)':>9s} {'sym':>7s} {'W2':>7s} | favored")
    Pj=P.join(DIST,on=['source','benchmark'])
    for reg in ['FF','FT']:
        for label,bset in [('real-mot',REAL),('semantic',SEM)]:
            vals={k:cell(Pj,reg,bset,M[k])[0] for k in ['dTB','dBT','sym','W2']}
            nb =cell(Pj,reg,bset,M['dTB'])[1]
            fav='precision(dTB)' if (np.nan_to_num(vals['dTB'])>np.nan_to_num(vals['dBT'])) else 'COVERAGE(dBT)'
            f=lambda x:'  --  ' if x is None or np.isnan(x) else f"{x:+.2f}"
            print(f"{reg:4s} {label:9s} | {f(vals['dTB']):>10s} {f(vals['dBT']):>9s} {f(vals['sym']):>7s} {f(vals['W2']):>7s} | {fav}  (n_bench={nb})")

# ===== divergence diagnostics =====
P=load()
print("=== epochs reached (FF / FT) per source ===")
ep=P.groupby(['source','regime']).maxep.max().unstack()
print(ep.to_string())

print("\n=== FF vs FT peak PCK divergence ===")
piv=P.pivot_table(index=['source','benchmark'],columns='regime',values='peak')
piv['dFT_minus_FF']=piv['FT']-piv['FF']
# overall agreement across all (source,benchmark) cells
ok=piv.dropna()
print(f"overall FF-FT cell corr (pearson) = {ok['FF'].corr(ok['FT']):.3f}   spearman = {ok['FF'].corr(ok['FT'],method='spearman'):.3f}   n={len(ok)}")
print(f"mean(FT-FF) over all cells = {ok['dFT_minus_FF'].mean():+.2f} pck   (FT>FF in {100*(ok['dFT_minus_FF']>0).mean():.0f}% of cells)")

print("\n=== per-benchmark FF-vs-FT source-ranking agreement (spearman) ===")
for b in REAL+SEM:
    s=piv.xs(b,level='benchmark').dropna()
    if len(s)>=3:
        rho=s['FF'].corr(s['FT'],method='spearman')
        print(f"  {b:14s} n={len(s)}  rho(FF,FT rank)={rho:+.2f}   meanFF={s['FF'].mean():5.1f} meanFT={s['FT'].mean():5.1f}")

print("\n=== source ranking on kitti2015 (real motion target) ===")
k=piv.xs('kitti2015',level='benchmark')[['FF','FT']].dropna().sort_values('FF',ascending=False)
print(k.round(2).to_string())

# ===== law tables (peak so far) =====
law_table(P,MOT,"MOTION  (Table 1)  peak-so-far  [epoch-heterogeneous]")
law_table(P,DIN,"DINO    (Table 2)  peak-so-far  [epoch-heterogeneous]")

# ===== matched-epoch robustness (cap at 30 so synthetic* @15 vs real @70 less skewed) =====
for cap in [15,30]:
    Pc=load(cap=cap)
    law_table(Pc,MOT,f"MOTION  matched epoch<= {cap}")
