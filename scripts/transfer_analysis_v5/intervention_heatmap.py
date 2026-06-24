"""Controlled motion x appearance intervention heatmap (kitti2015 target).

Two panels: scratch (CATs++ FF) | pretrained (CATs++ TT). Points = canonical sources
+ kubric interventions (recovered/badmotion x hq/matte +-gso, trial19, lowtex_matte).
Axes = motion distance (BFV) and appearance distance (DINO), both eval->train NN to
kitti2015, recomputed uniformly from cached vectors. Color = transfer (kitti2015 peak
PCK): canonical from transfer_table, interventions from transfer_grid runs.

Vertical iso-transfer contours => motion drives transfer, appearance is a tax.
"""
import os, glob, sys
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "."); from scripts.coverage import spaces
VEC="/mnt/nvme_1tb_b/coverage_vectors"; N=50000; W=H=512; rng=np.random.default_rng(0); TGT="kitti2015"
BASES=["/mnt/nvme_1tb_a/snapshots","/mnt/nvme_1tb_b/snapshots","snapshots"]

def sub(p):
    a=np.load(p,mmap_mode="r"); n=min(N,len(a)); idx=np.sort(rng.choice(len(a),n,replace=False)); return np.asarray(a[idx],dtype=np.float32)
def load(name):
    sp="val" if name==TGT else "train"
    fp=f"{VEC}/{name}_{sp}_flow.npy"; dp=f"{VEC}/{name}_{sp}_dino_pca256_l2norm.npy"
    if not(os.path.exists(fp) and os.path.exists(dp)): return None
    return spaces.normalize_flow_vectors(sub(fp),W,H), sub(dp)
def mnn(q,d): return float(cKDTree(d).query(q,k=1,workers=4)[0].mean())
def grid_transfer(stem,suf):
    for b in BASES:
        g=glob.glob(f"{b}/transfer_grid/{stem}_{suf}*/validation_results.csv")
        if g:
            v=pd.read_csv(g[0]); k=v[v.benchmark==TGT]
            if len(k): return float(k.pck.max())
    return np.nan
# vector-name -> transfer_grid stem
IV={"kitti_recovered_hq":"kitti_recovered_hq","kitti_recovered_matte":"kitti_recovered_matte",
    "kitti_recovered_gso_hq":"kitti_recovered_gso_hq","kitti_recovered_gso_matte":"kitti_recovered_gso_matte",
    "kitti_badmotion_ft_gso_hq":"kitti_badmotion_ft_gso_hq","kitti_badmotion_ft_gso_matte":"kitti_badmotion_ft_gso_matte",
    "kitti2015_lowtex_matte":"lowtex_matte","kitti2015_hq_trial19":"trial19"}

t=pd.read_csv("scripts/transfer_analysis_v3/transfer_table.csv")
def canon_transfer(reg):
    p=(reg=="TT"); f=(reg=="TT")
    s=t[(t.model_family=="catspp")&(t.pretrained==p)&(t.freeze==f)&(t.benchmark==TGT)]
    return dict(zip(s.train_dataset,s.peak_pck))

# distances once
tgt=load(TGT); assert tgt; tF,tD=tgt
cands=sorted(set(os.path.basename(x)[:-len("_train_flow.npy")] for x in glob.glob(f"{VEC}/*_train_flow.npy")))
DIST={}
for s in cands:
    if s==TGT: continue
    L=load(s)
    if L: DIST[s]=(mnn(tF,L[0]), mnn(tD,L[1])); print(f"  dist {s:32s} dMot={DIST[s][0]:.4f} dApp={DIST[s][1]:.4f}",flush=True)

def points(reg):
    suf="pt1_fz1" if reg=="TT" else "pt0_fz0"
    trans=canon_transfer(reg)
    for vn,stem in IV.items(): trans[vn]=grid_transfer(stem,suf)
    rows=[]
    for s,(dm,da) in DIST.items():
        if s not in trans or not np.isfinite(trans.get(s,np.nan)): continue
        kind="intervention" if s in IV else ("mix" if ("_synthetic_" in s or "_2d_warp_" in s) else "canonical")
        rows.append(dict(ds=s,dMot=dm,dApp=da,transfer=trans[s],kind=kind))
    return pd.DataFrame(rows)

fig,axes=plt.subplots(1,2,figsize=(17,7))
mk={"canonical":"o","mix":"s","intervention":"*"}; sz={"canonical":120,"mix":90,"intervention":520}
for ax,reg,name in [(axes[0],"FF","scratch (CATs++ FF)"),(axes[1],"TT","pretrained (CATs++ TT)")]:
    g=points(reg); g.to_csv(f"scripts/transfer_analysis_v5/results/intervention_heatmap_{reg}.csv",index=False)
    for c in ["dMot","dApp","transfer"]: g[c+"z"]=(g[c]-g[c].mean())/g[c].std()
    X=np.column_stack([np.ones(len(g)),g.dMotz,g.dAppz]); b,*_=np.linalg.lstsq(X,g.transferz.values,rcond=None)
    def pc(a,bb,c): a=a-np.polyval(np.polyfit(c,a,1),c); bb=bb-np.polyval(np.polyfit(c,bb,1),c); return np.corrcoef(a,bb)[0,1]
    prM=pc(g.transferz.values,g.dMotz.values,g.dAppz.values); prA=pc(g.transferz.values,g.dAppz.values,g.dMotz.values)
    rsep=np.corrcoef(g.dMotz,g.dAppz)[0,1]
    n_iv=(g.kind=="intervention").sum()
    print(f"[{reg}] n={len(g)} iv={n_iv} beta_mot={b[1]:+.2f} beta_app={b[2]:+.2f} partial mot={prM:+.2f} app={prA:+.2f} sep={rsep:+.2f}")
    xs=np.linspace(g.dMotz.min()-.3,g.dMotz.max()+.3,200); ys=np.linspace(g.dAppz.min()-.3,g.dAppz.max()+.3,200)
    XX,YY=np.meshgrid(xs,ys); ZZ=b[0]+b[1]*XX+b[2]*YY
    im=ax.imshow(ZZ,origin="lower",extent=[xs.min(),xs.max(),ys.min(),ys.max()],aspect="auto",cmap="RdYlGn",alpha=.7)
    cs=ax.contour(XX,YY,ZZ,levels=7,colors="k",linewidths=.5,alpha=.5); ax.clabel(cs,inline=True,fontsize=6,fmt="%.1f")
    for kind,m in mk.items():
        s_=g[g.kind==kind]
        ax.scatter(s_.dMotz,s_.dAppz,c=s_.transferz,cmap="RdYlGn",marker=m,s=[sz[kind]]*len(s_),
                   edgecolors=("blue" if kind=="intervention" else "k"),linewidths=(2 if kind=="intervention" else 1),
                   zorder=(7 if kind=="intervention" else 5),vmin=ZZ.min(),vmax=ZZ.max(),label=kind)
    for _,r in g.iterrows():
        lab=r.ds.replace("synthetic","syn").replace("kitti_","").replace("kitti2015_","").replace("_ft_gso","").replace("_gso","")
        ax.annotate(lab,(r.dMotz,r.dAppz),fontsize=6,xytext=(3,3),textcoords="offset points",zorder=8)
    ax.axhline(0,color="gray",lw=.4,ls=":"); ax.axvline(0,color="gray",lw=.4,ls=":")
    ax.set_xlabel("motion dist d$_{mot}$ (BFV→kitti2015, z) → worse",fontsize=10)
    ax.set_ylabel("appearance dist d$_{app}$ (DINO→kitti2015, z) → worse",fontsize=10)
    ax.set_title(f"{name}\nβ_mot={b[1]:+.2f} β_app={b[2]:+.2f} | partial r mot={prM:+.2f} app={prA:+.2f} | n={len(g)} (★={n_iv} interv)",fontsize=9)
    ax.legend(loc="upper left",fontsize=8)
fig.suptitle("Controlled motion×appearance grid → transfer to kitti2015  (vertical contours ⇒ motion drives transfer)",fontsize=12)
plt.tight_layout(); out="scripts/transfer_analysis_v5/results/intervention_heatmap.png"; plt.savefig(out,dpi=130); print("WROTE",out)
