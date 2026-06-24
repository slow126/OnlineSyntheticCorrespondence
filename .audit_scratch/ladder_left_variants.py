"""Render 3 readability variants of Fig 8's LEFT panel (coverage / off-target vs
magnitude). No PCK. Pick one, then I patch make_ladder_fig.py."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("ACCV_2026/figures/proto"); OUT.mkdir(parents=True, exist_ok=True)
RUNGS=["m025","m050","m100","m150","m200"]; XLAB=["0.25×","0.5×","1×","1.5×","2×"]
cv = pd.read_csv("analysis/coverage_v2_flow_ladder.csv")
cv = cv[cv.eval_dataset.astype(str).str.contains("kitti", case=False)].copy()
cv["rung"] = cv.train_dataset.astype(str).str.extract(r"(m\d+)")[0]
gg = cv.groupby("rung").agg(coverage=("eval_covered_by_train_qnorm_k1","mean"),
                            offt=("train_outside_eval_qnorm_k1","mean")).reindex(RUNGS)
cov = gg["coverage"].values; off = gg["offt"].values
x = np.arange(5)
BLUE="#2b6cb0"; RED="#c0392b"; GREEN="#2a8f3f"

def shade(ax):
    ax.axvspan(-0.4,0.5,color="#fdecea",zorder=0)
    ax.axvspan(1.5,4.4,color="#fff4e6",zorder=0)
    ax.axvline(2,color=GREEN,ls="--",lw=1.3,zorder=1)

# ---------- Variant A: dual independent y-axes (twinx) ----------
fig,ax=plt.subplots(figsize=(4.6,4.0)); shade(ax)
ax2=ax.twinx()
l1,=ax.plot(x,cov,"o-",color=BLUE,lw=2.4,ms=7)
l2,=ax2.plot(x,off,"s--",color=RED,lw=2.0,ms=6)
ax.set_xticks(x); ax.set_xticklabels(XLAB)
ax.set_xlabel("source motion magnitude (× KITTI mean flow)")
ax.set_ylabel("KITTI motion covered  (coverage)",color=BLUE)
ax2.set_ylabel("source motion off-target",color=RED)
ax.tick_params(axis="y",colors=BLUE); ax2.tick_params(axis="y",colors=RED)
ax.set_ylim(0.10,0.34); ax2.set_ylim(0.90,1.0)
ax.set_title("A · dual independent axes",loc="left",fontsize=10)
ax.legend([l1,l2],["coverage (left)","off-target (right)"],fontsize=8,loc="lower center")
fig.savefig(OUT/"ladder_left_A_dualaxis.png",dpi=170,bbox_inches="tight"); plt.close(fig)

# ---------- Variant B: broken single axis (honest levels) ----------
fig,(axT,axB)=plt.subplots(2,1,figsize=(4.6,4.0),sharex=True,
                           gridspec_kw=dict(height_ratios=[1,1.25],hspace=0.08))
for a in (axT,axB): shade(a)
axT.plot(x,off,"s--",color=RED,lw=2.0,ms=6,label="off-target mass")
axB.plot(x,cov,"o-",color=BLUE,lw=2.4,ms=7,label="KITTI covered")
axT.set_ylim(0.93,0.995); axB.set_ylim(0.11,0.335)
axT.set_yticks([0.94,0.96,0.98]); axB.set_yticks([0.15,0.20,0.25,0.30])
# diagonal break marks
d=.012
for a,top in [(axT,True),(axB,False)]:
    kw=dict(transform=a.transAxes,color="k",clip_on=False,lw=0.9)
    if top: a.plot((-d,+d),(-d,+d),**kw); a.plot((1-d,1+d),(-d,+d),**kw); a.spines["bottom"].set_visible(False)
    else:   a.plot((-d,+d),(1-d,1+d),**kw); a.plot((1-d,1+d),(1-d,1+d),**kw); a.spines["top"].set_visible(False)
axT.tick_params(labelbottom=False)
axB.set_xticks(x); axB.set_xticklabels(XLAB)
axB.set_xlabel("source motion magnitude (× KITTI mean flow)")
axT.set_title("B · broken single axis (same scale, dead space removed)",loc="left",fontsize=9)
axT.text(0.02,0.94,"off-target (near ceiling, +0.04 on over-shoot)",transform=axT.transAxes,fontsize=7.5,color=RED,va="top")
axB.text(0.02,0.06,"coverage (peaks at match, −0.15 on over-shoot)",transform=axB.transAxes,fontsize=7.5,color=BLUE,va="bottom")
fig.savefig(OUT/"ladder_left_B_broken.png",dpi=170,bbox_inches="tight"); plt.close(fig)

# ---------- Variant C: current shared 0-1 axis (reference) ----------
fig,ax=plt.subplots(figsize=(4.6,4.0)); shade(ax)
ax.plot(x,cov,"o-",color=BLUE,lw=2.4,ms=7,label="KITTI covered by source")
ax.plot(x,off,"s--",color=RED,lw=2.0,ms=6,label="source off-target mass")
ax.set_xticks(x); ax.set_xticklabels(XLAB); ax.set_ylim(0.05,1.02)
ax.set_xlabel("source motion magnitude (× KITTI mean flow)")
ax.set_ylabel("support-overlap fraction")
ax.set_title("C · current shared axis (reference)",loc="left",fontsize=10)
ax.legend(fontsize=8,loc="center left")
fig.savefig(OUT/"ladder_left_C_shared.png",dpi=170,bbox_inches="tight"); plt.close(fig)
print("wrote A_dualaxis, B_broken, C_shared to", OUT)
