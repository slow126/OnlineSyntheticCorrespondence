"""When does peak PCK happen during lolr CATs++ training? Early (overfit/early-stop)
or late (still improving)? Validation every 5 epochs. Middlebury excluded."""
import pandas as pd, numpy as np, glob, os
REAL=['kitti2012','kitti2015','flyingthings','pointodyssey','synthetic']
SEM =['spair','pfpascal','pfwillow','tss']
rows=[]
for f in glob.glob('cats_ff_ft_snapshtos/*/validation_results.csv'):
    dn=os.path.basename(os.path.dirname(f)); src=dn.split('_cats_lolr')[0]
    reg='FF' if 'freezeFalse' in dn else 'FT'
    v=pd.read_csv(f); maxep=int(v.epoch.max())
    for b,g in v.groupby('benchmark'):
        if b=='middlebury': continue
        g=g.sort_values('epoch'); i=g.pck.idxmax()
        pe=int(g.loc[i,'epoch']); pp=float(g.loc[i,'pck']); fin=float(g.iloc[-1].pck)
        rows.append(dict(src=src,reg=reg,bench=b,maxep=maxep,peak_ep=pe,peak_pck=pp,
            final_pck=fin,frac=pe/maxep,pkloss=pp-fin,reldrop=(pp-fin)/pp if pp>1e-6 else 0,
            btype='real' if b in REAL else ('sem' if b in SEM else 'other'),conv=maxep>=200))
P=pd.DataFrame(rows); P=P[P.btype.isin(['real','sem'])]
C=P[P.conv]

print(f"=== converged(ep200) run-regimes={C[['src','reg']].drop_duplicates().shape[0]} ; "
      f"non-converged still climbing (pointodyssey ep110, synthetic* ep55-60 -> peak censored) ===\n")
print("=== CONVERGED runs (ep200): where does peak PCK land? ===")
for lab,sub in [('ALL',C),('real-motion',C[C.btype=='real']),('semantic',C[C.btype=='sem'])]:
    if not len(sub): continue
    print(f"  {lab:11s} n={len(sub):3d}  peak_ep median={sub.peak_ep.median():5.0f} mean={sub.peak_ep.mean():5.0f}"
          f"  frac median={sub.frac.median():.2f}  peak->final loss median={sub.pkloss.median():+.2f}pck "
          f"({100*sub.reldrop.median():+.0f}%)")
print("\n=== peak-epoch distribution (converged), share in each quarter ===")
bins=[0,50,100,150,201]; lbl=['ep1-50','ep50-100','ep100-150','ep150-200']
for lab,sub in [('real-motion',C[C.btype=='real']),('semantic',C[C.btype=='sem'])]:
    h=pd.cut(sub.peak_ep,bins=bins,labels=lbl,right=True).value_counts().reindex(lbl)
    print(f"  {lab:11s}: "+"  ".join(f"{l}={int(h[l]):2d}({100*h[l]/len(sub):3.0f}%)" for l in lbl))
print("\n=== by regime (converged, real-motion) ===")
for reg in ['FF','FT']:
    s=C[(C.reg==reg)&(C.btype=='real')]
    print(f"  {reg}: peak_ep median={s.peak_ep.median():.0f} frac={s.frac.median():.2f} peak->final loss={s.pkloss.median():+.2f}pck")
print("\n=== overfit: converged cells losing >2 pck from peak to final ===")
for lab,sub in [('real-motion',C[C.btype=='real']),('semantic',C[C.btype=='sem'])]:
    n=(sub.pkloss>2).sum(); print(f"  {lab:11s}: {n}/{len(sub)} ({100*n/len(sub):.0f}%)")
print("\n=== per real-benchmark median peak epoch (converged) ===")
for b in REAL:
    s=C[C.bench==b]
    if len(s): print(f"  {b:14s} n={len(s)} peak_ep median={s.peak_ep.median():5.0f} loss median={s.pkloss.median():+.2f}")
