import glob, os, numpy as np, pandas as pd
PURE=['flyingthings','imagenet2dwarp','movi_f','pointodyssey','sintel','spair',
      'synthetic','synthetic_2d_warp','synthetic_large_zoom','synthetic_random_flipping','synthetic_small_zoom']
REAL=['kitti2012','kitti2015','flyingthings','pointodyssey','synthetic']
SEM=['spair','pfpascal','pfwillow','tss']
d=pd.read_csv('analysis_v3/pairwise_self_distances.csv')
def mn(space):
    x=d[(d.pair_type=='train_eval')&(d.space==space)].set_index(['dataset_a','dataset_b'])
    return x[['mean_nn_a_to_b','mean_nn_b_to_a','mean_nn_sym']]
flow=mn('flow').rename(columns=lambda c:'mot_'+c); dino=mn('dino').rename(columns=lambda c:'dino_'+c)
t0=pd.read_csv('scripts/transfer_analysis_v3/transfer_table.csv')
extra=t0[t0.train_dataset.isin(PURE)].groupby(['train_dataset','benchmark'])[['flow_sliced_w2','dino_sliced_w2','flow_fid','dino_fid']].first()
extra.index.names=['dataset_a','dataset_b']
DIST=flow.join(dino).join(extra)
MOT={'Coverage':'mot_mean_nn_b_to_a','Off-target':'mot_mean_nn_a_to_b','Chamfer':'mot_mean_nn_sym','Sliced W2':'flow_sliced_w2','FID':'flow_fid'}
DIN={'Coverage':'dino_mean_nn_b_to_a','Off-target':'dino_mean_nn_a_to_b','Chamfer':'dino_mean_nn_sym','Sliced W2':'dino_sliced_w2','FID':'dino_fid'}
rows=[]; t=t0[t0.train_dataset.isin(PURE)]
for _,r in t.iterrows():
    a=str(r.model_family).lower()
    if a=='raft': rows.append(dict(variant='RAFT|scratch',source=r.train_dataset,benchmark=r.benchmark,peak=r.peak_pck))
    elif a in('catspp','glunet') and r.pretrained==True:
        fr='frz' if r.freeze else 'trn'; rows.append(dict(variant=f'{a}|pre|{fr}',source=r.train_dataset,benchmark=r.benchmark,peak=r.peak_pck))
ffroot='scripts/transfer_analysis_v5/flowformer_rc_results'; ffbest={}
for f in glob.glob(ffroot+'/*/validation_results.csv'):
    name=os.path.basename(os.path.dirname(f))
    if '_flowformer_steps100_' not in name or 'pretrainTrue' not in name: continue
    src=name.split('_flowformer_steps100_')[0]
    if src not in PURE: continue
    fr='frz' if 'freezeTrue' in name else 'trn'
    try: df=pd.read_csv(f)
    except: continue
    key=(src,fr); mx=df.epoch.max()
    if key not in ffbest or mx>ffbest[key][0]: ffbest[key]=(mx,df)
for (src,fr),(mx,df) in ffbest.items():
    for b,g in df.groupby('benchmark'): rows.append(dict(variant=f'flowformer|pre|{fr}',source=src,benchmark=b,peak=float(g.pck.max())))
P=pd.DataFrame(rows)
for col in set(list(MOT.values())+list(DIN.values())):
    P[col]=[DIST[col].get((s,b),np.nan) if (s,b) in DIST.index else np.nan for s,b in zip(P.source,P.benchmark)]

def regret(col):
    reg=[]
    for v in P.variant.unique():
        sub=P[P.variant==v]; benches=REAL if v.startswith('RAFT') else REAL+SEM
        for b in benches:
            g=sub[sub.benchmark==b].dropna(subset=['peak',col])
            if g.source.nunique()<3 or g[col].std()<1e-12: continue
            reg.append(g.peak.max()-g.loc[g[col].idxmin(),'peak'])
    return np.median(reg)
def rand_regret():
    rnd=[]
    for v in P.variant.unique():
        sub=P[P.variant==v]; benches=REAL if v.startswith('RAFT') else REAL+SEM
        for b in benches:
            g=sub[sub.benchmark==b].dropna(subset=['peak',MOT['Coverage']])
            if g.source.nunique()<3: continue
            rnd.append(g.peak.max()-g.peak.mean())
    return np.median(rnd)

print(f"{'distance':11s}  motion-regret  appearance-regret")
for m in ['Coverage','Off-target','Chamfer','Sliced W2','FID']:
    print(f"{m:11s}  {regret(MOT[m]):>8.1f}       {regret(DIN[m]):>8.1f}")
print(f"{'RANDOM':11s}  {rand_regret():>8.1f}       {rand_regret():>8.1f}")
