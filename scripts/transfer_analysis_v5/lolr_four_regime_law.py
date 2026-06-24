"""All four CATs++ regimes side by side: scratch FF/FT (lolr local) + pretrained
TF/TT (transfer_table). Same real-motion/semantic x {dTB,dBT,sym,W2} law.
Key contrast: FT (frozen RANDOM enc) vs TT (frozen PRETRAINED enc) -- identical
freeze mechanic, only feature quality differs. Does the precision->coverage flip
track feature quality rather than the freeze?"""
import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
PURE=["flyingthings","imagenet2dwarp","movi_f","pointodyssey","sintel","spair","synthetic",
      "synthetic_2d_warp","synthetic_large_zoom","synthetic_random_flipping","synthetic_small_zoom"]
REAL=['kitti2012','kitti2015','flyingthings','pointodyssey','synthetic']
SEM =['spair','pfpascal','pfwillow','tss']
d=pd.read_csv('analysis_v3/pairwise_self_distances.csv')
def mn(s):
    x=d[(d.pair_type=='train_eval')&(d.space==s)].set_index(['dataset_a','dataset_b'])
    return x[['mean_nn_a_to_b','mean_nn_b_to_a','mean_nn_sym']]
flow=mn('flow').rename(columns=lambda c:'mot_'+c); dino=mn('dino').rename(columns=lambda c:'dino_'+c)
t0=pd.read_csv('scripts/transfer_analysis_v3/transfer_table.csv')
w2=t0[t0.train_dataset.isin(PURE)].groupby(['train_dataset','benchmark'])[['flow_sliced_w2','dino_sliced_w2']].first()
w2.index.names=['dataset_a','dataset_b']
DIST=flow.join(dino).join(w2)
MOT={'dTB':'mot_mean_nn_a_to_b','dBT':'mot_mean_nn_b_to_a','sym':'mot_mean_nn_sym','W2':'flow_sliced_w2'}
DIN={'dTB':'dino_mean_nn_a_to_b','dBT':'dino_mean_nn_b_to_a','sym':'dino_mean_nn_sym','W2':'dino_sliced_w2'}

rows=[]
# scratch FF/FT from local lolr
for f in glob.glob('cats_ff_ft_snapshtos/*/validation_results.csv'):
    dn=os.path.basename(os.path.dirname(f)); src=dn.split('_cats_lolr')[0]
    if src not in PURE: continue
    reg='FF' if 'freezeFalse' in dn else 'FT'
    v=pd.read_csv(f)
    for b,g in v.groupby('benchmark'):
        rows.append(dict(reg=reg,source=src,benchmark=b,peak=float(g.pck.max())))
# pretrained TF/TT from transfer_table
c=t0[(t0.model_family=='catspp')&(t0.pretrained==True)&(t0.train_dataset.isin(PURE))]
for _,r in c.iterrows():
    reg='TT' if r.freeze else 'TF'
    rows.append(dict(reg=reg,source=r.train_dataset,benchmark=r.benchmark,peak=r.peak_pck))
P=pd.DataFrame(rows).join(DIST,on=['source','benchmark'])

def cell(reg,bset,col):
    sub=P[(P.reg==reg)&(P.benchmark.isin(bset))].dropna(subset=['peak',col])
    rs=[spearmanr(g.peak,-g[col]).statistic for _,g in sub.groupby('benchmark')
        if g.source.nunique()>=3 and g[col].std()>1e-12]
    return np.mean([x for x in rs if np.isfinite(x)]) if rs else np.nan

def table(M,title):
    print(f"\n################ {title} ################")
    print(f"{'reg':4s} {'enc/feats':22s} {'set':9s} | {'dTB(prec)':>10s} {'dBT(rec)':>9s} {'sym':>7s} {'W2':>7s} | favored")
    desc={'FF':'scratch trained','FT':'scratch FROZEN-random','TF':'pretrained fine-tuned','TT':'pretrained FROZEN-imnet'}
    for reg in ['FF','FT','TF','TT']:
        for label,bset in [('real-mot',REAL),('semantic',SEM)]:
            v={k:cell(reg,bset,M[k]) for k in ['dTB','dBT','sym','W2']}
            fav='precision' if np.nan_to_num(v['dTB'])>np.nan_to_num(v['dBT']) else 'COVERAGE'
            f=lambda x:'  --  ' if x is None or np.isnan(x) else f"{x:+.2f}"
            print(f"{reg:4s} {desc[reg]:22s} {label:9s} | {f(v['dTB']):>10s} {f(v['dBT']):>9s} {f(v['sym']):>7s} {f(v['W2']):>7s} | {fav}")

table(MOT,"MOTION")
table(DIN,"DINO")
print("\n=== n sources per regime ===")
print(P.groupby('reg').source.nunique())
