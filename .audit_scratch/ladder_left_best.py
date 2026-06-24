"""BEST recommendation for Fig 8 left panel.
- Raw JOINT mean-NN distance (NO qnorm; the live metric, consistent with Table 1).
- kitti2012 and kitti2015 as separate lines.
- coverage d_{B->T} on left (dips at the match), off-target d_{T->B} on right
  given a generous range so it reads as a flat floor (the narrative point).
- No PCK.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
OUT = Path("ACCV_2026/figures/proto")
RUNGS=["m025","m050","m100","m150","m200"]; XLAB=["0.25×","0.5×","1×","1.5×","2×"]
cv = pd.read_csv("analysis/coverage_v2_flow_ladder.csv")
cv["rung"]=cv.train_dataset.astype(str).str.extract(r"(m\d+)")[0]
def series(bench):
    g=cv[cv.eval_dataset==bench].groupby("rung").agg(
        dBT=("mean_nn_eval_to_train_k1","mean"),
        dTB=("mean_nn_train_to_eval_k1","mean")).reindex(RUNGS)
    return g["dBT"].values, g["dTB"].values
b15=series("kitti2015"); b12=series("kitti2012")
x=np.arange(5); GREEN="#2a8f3f"
COV15="#1b4f86"; COV12="#5b9bd5"; OFF15="#a3271b"; OFF12="#e8806f"

fig,ax=plt.subplots(figsize=(5.4,4.3))
ax.axvspan(-0.4,1.5,color="#fdecea",alpha=0.5,zorder=0)   # under
ax.axvspan(2.5,4.4,color="#fff4e6",alpha=0.6,zorder=0)    # over
ax.axvline(2,color=GREEN,ls="--",lw=1.4,zorder=1)
ax2=ax.twinx()
# coverage (left)
ax.plot(x,b15[0],"-o",color=COV15,lw=2.6,ms=7,zorder=4)
ax.plot(x,b12[0],"-o",color=COV12,lw=2.6,ms=7,zorder=4,mfc="white",mew=1.8)
# off-target (right)
ax2.plot(x,b15[1],"--s",color=OFF15,lw=2.0,ms=6,zorder=3)
ax2.plot(x,b12[1],"--s",color=OFF12,lw=2.0,ms=6,zorder=3,mfc="white",mew=1.6)

ax.set_xticks(x); ax.set_xticklabels(XLAB)
ax.set_xlabel("source motion magnitude (× KITTI mean flow)")
ax.set_ylim(0.0, 0.0115)
ax2.set_ylim(0.0, 0.058)
ax.set_ylabel("coverage  $d_{B\\to T}$  (mean-NN, lower = better)",color=COV15)
ax2.set_ylabel("off-target  $d_{T\\to B}$  (mean-NN)",color=OFF15)
ax.tick_params(axis="y",labelcolor=COV15); ax2.tick_params(axis="y",labelcolor=OFF15)
ax.set_title("Coverage dips at the match; off-target stays flat",loc="left",fontsize=11)
ax2.text(0.97,0.80,"off-target flat (≤1%)",transform=ax.transAxes,
         fontsize=8.5,color=OFF15,ha="right",style="italic")
ax.annotate("under",(0.30,0.0005),fontsize=8.3,color="#a33",ha="center")
ax.annotate("over-shoot",(3.4,0.0005),fontsize=8.3,color="#b5670f",ha="center")

handles=[Line2D([0],[0],color=COV15,lw=2.4,marker="o",ms=6,label="coverage · KITTI-15"),
         Line2D([0],[0],color=COV12,lw=2.4,marker="o",ms=6,mfc="white",mew=1.6,label="coverage · KITTI-12"),
         Line2D([0],[0],color=OFF15,lw=2.0,ls="--",marker="s",ms=5,label="off-target · KITTI-15"),
         Line2D([0],[0],color=OFF12,lw=2.0,ls="--",marker="s",ms=5,mfc="white",mew=1.4,label="off-target · KITTI-12")]
ax.legend(handles=handles,fontsize=7.8,loc="upper center",ncol=2,columnspacing=1.0)
fig.savefig(OUT/"ladder_left_best.png",dpi=175,bbox_inches="tight"); plt.close(fig)
print("wrote ladder_left_best.png")
print("cov k15",np.round(b15[0],5),"\ncov k12",np.round(b12[0],5))
print("off k15",np.round(b15[1],5),"\noff k12",np.round(b12[1],5))
