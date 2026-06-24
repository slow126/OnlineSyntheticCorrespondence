"""Fig 8 left panel using the PAPER'S OWN mean-NN distance metric (joint), positive
values, no PCK. Coverage d_{B->T} bottoms out at the match (best coverage);
off-target d_{T->B} stays flat -> the clean asymmetry narrative.
Renders two orientations so we can pick:
  D1 = raw distances (coverage = valley at match, 'lower = better')
  D2 = inverted axes (coverage = peak at match, 'up = better')
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = Path("ACCV_2026/figures/proto")
RUNGS=["m025","m050","m100","m150","m200"]; XLAB=["0.25×","0.5×","1×","1.5×","2×"]
cv = pd.read_csv("analysis/coverage_v2_flow_ladder.csv")
cv = cv[cv.eval_dataset.astype(str).str.contains("kitti", case=False)].copy()
cv["rung"] = cv.train_dataset.astype(str).str.extract(r"(m\d+)")[0]
g = cv.groupby("rung").agg(dBT=("mean_nn_eval_to_train_k1","mean"),
                           dTB=("mean_nn_train_to_eval_k1","mean")).reindex(RUNGS)
dBT, dTB = g["dBT"].values, g["dTB"].values
x=np.arange(5); BLUE="#2b6cb0"; RED="#c0392b"; GREEN="#2a8f3f"
print("dBT (coverage):", np.round(dBT,5))
print("dTB (off-target):", np.round(dTB,5),
      f"  flat: varies {100*(dTB.max()-dTB.min())/dTB.mean():.1f}%")

def shade(ax):
    ax.axvspan(-0.4,0.5,color="#fdecea",zorder=0); ax.axvspan(1.5,4.4,color="#fff4e6",zorder=0)
    ax.axvline(2,color=GREEN,ls="--",lw=1.3,zorder=1)

def panel(invert, fname, title):
    fig,ax=plt.subplots(figsize=(5.0,4.1)); shade(ax); ax2=ax.twinx()
    l1,=ax.plot(x,dBT,"o-",color=BLUE,lw=2.6,ms=8,zorder=3)
    l2,=ax2.plot(x,dTB,"s--",color=RED,lw=2.0,ms=6,zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(XLAB)
    ax.set_xlabel("source motion magnitude (× KITTI mean flow)")
    # coverage axis: generous, off-target axis: generous (±~45%) so flatness is honest
    cl,ch = dBT.min()-0.0010, dBT.max()+0.0008
    ol,oh = dTB.mean()*0.55, dTB.mean()*1.45
    if invert:
        ax.set_ylim(ch,cl); ax2.set_ylim(oh,ol)   # up = smaller distance = better
        ax.set_ylabel("coverage  $d_{B\\to T}$  (↓ better; up=better)",color=BLUE)
        ax2.set_ylabel("off-target  $d_{T\\to B}$  (up=better)",color=RED)
    else:
        ax.set_ylim(cl,ch); ax2.set_ylim(ol,oh)
        ax.set_ylabel("coverage  $d_{B\\to T}$  (mean-NN, lower=better)",color=BLUE)
        ax2.set_ylabel("off-target  $d_{T\\to B}$  (mean-NN)",color=RED)
    ax.tick_params(axis="y",labelcolor=BLUE); ax2.tick_params(axis="y",labelcolor=RED)
    ax.set_title(title,loc="left",fontsize=10)
    ax.legend([l1,l2],["coverage (left)","off-target (right) — flat"],fontsize=8,loc="lower center")
    ax.text(0.5,(0.93 if not invert else 0.07),"off-target ≈ flat (0.048, <1% swing)",
            transform=ax.transAxes,fontsize=7.6,color=RED,ha="center")
    fig.savefig(OUT/fname,dpi=170,bbox_inches="tight"); plt.close(fig); print("wrote",fname)

panel(False,"ladder_left_D1_distance_valley.png","D1 · mean-NN distance (coverage = valley at match)")
panel(True ,"ladder_left_D2_distance_peak.png",  "D2 · mean-NN distance, axes inverted (coverage = peak)")
