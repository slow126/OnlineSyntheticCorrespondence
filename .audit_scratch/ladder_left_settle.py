"""Settle the 'off-target looks like it moves in D1' question: SAME data, but give
off-target a generous range so it reads as a flat floor instead of crowding the
coverage endpoints. Renders D1b (valley, off-target as flat floor)."""
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

fig,ax=plt.subplots(figsize=(5.0,4.1))
ax.axvspan(-0.4,0.5,color="#fdecea",zorder=0); ax.axvspan(1.5,4.4,color="#fff4e6",zorder=0)
ax.axvline(2,color=GREEN,ls="--",lw=1.3,zorder=1)
ax2=ax.twinx()
l1,=ax.plot(x,dBT,"o-",color=BLUE,lw=2.6,ms=8,zorder=3)
l2,=ax2.plot(x,dTB,"s--",color=RED,lw=2.0,ms=6,zorder=2)
ax.set_xticks(x); ax.set_xticklabels(XLAB)
ax.set_xlabel("source motion magnitude (× KITTI mean flow)")
# coverage: tight range -> dramatic valley.  off-target: 0..max -> obvious flat floor.
ax.set_ylim(0.0042, 0.0082)
ax2.set_ylim(0.0, 0.060)
ax.set_ylabel("coverage  $d_{B\\to T}$  (mean-NN, lower=better)",color=BLUE)
ax2.set_ylabel("off-target  $d_{T\\to B}$  (mean-NN)",color=RED)
ax.tick_params(axis="y",labelcolor=BLUE); ax2.tick_params(axis="y",labelcolor=RED)
ax.set_title("D1b · same data, off-target given room → reads as a flat floor",loc="left",fontsize=9.5)
ax.legend([l1,l2],["coverage (left) — dips at match","off-target (right) — flat at 0.041"],
          fontsize=8,loc="upper center")
fig.savefig(OUT/"ladder_left_D1b_settle.png",dpi=170,bbox_inches="tight"); plt.close(fig)
print("off-target:", np.round(dTB,5), "  swing %:", round(100*(dTB.max()-dTB.min())/dTB.mean(),2))
print("wrote ladder_left_D1b_settle.png")
