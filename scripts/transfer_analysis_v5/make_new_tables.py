"""Render the NEW (2026-06-11 audit) robustness/diagnostic analyses as LaTeX
booktabs tables -> ACCV_2026/tables/tab_new_*.tex. Nothing hand-transcribed.

Tables:
  tab_new_jackknife    per-regime source/family jackknife (dataset-robust?)
  tab_new_partial      appearance vs motion, partial correlations (disentangle)
  tab_new_fixedmotion  fixed-motion appearance: level vs selection (target-invariance)
  tab_new_regimelevel  regime-vs-level deconfound (flip is regime, not level)
  tab_new_ttarm        TT-arm adjudication (which OOS cells are interpretable)
  tab_new_seedaudit    OOS distance seed robustness (5-seed claim)
  tab_new_anticorr     anti-correlation of the two directions, per benchmark
  tab_new_diagonal     in-domain diagonal-cell sensitivity of the rule

    python scripts/transfer_analysis_v5/make_new_tables.py
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import spearmanr

ROOT = Path("/home/spencer/Projects/OnlineSyntheticCorrespondence")
RES = ROOT / "scripts/transfer_analysis_v5/results"
OUT = ROOT / "ACCV_2026/tables"
OUT.mkdir(parents=True, exist_ok=True)
def w(name, body): (OUT / name).write_text(body + "\n"); print("wrote", name)
def f2(x): return f"{x:+.2f}"
def f3(x): return f"{x:+.3f}"
PRETTY = {"a_to_b":"$\\dtb$","b_to_a":"$\\dbt$","sym":"sym"}
CM = "\\cmark"
def var(v):  # catspp|True|False -> CATs++ \cmark/--
    if "|" not in str(v): return esc(str(v))
    a,p,fz=v.split("|"); name={"catspp":"CATs++","glunet":"GLU-Net","raft":"RAFT"}[a]
    pp = CM if p=="True" else "--"; ff = CM if fz=="True" else "--"
    return f"{name} {pp}/{ff}"
def esc(s): return s.replace("_","\\_")

# ---------------------------------------------------------------- jackknife
jk = pd.read_csv(RES/"per_regime_source_jackknife.csv")
def jk_band(reg, col):
    base=jk[(jk.regime==reg)&(jk.kind=="baseline")][col].iloc[0]
    sub=jk[(jk.regime==reg)&(jk.kind.isin(["LOSO","LOFO"]))]
    return base, sub[col].min(), sub[col].max()
def jk_worst(reg,col):
    base=jk[(jk.regime==reg)&(jk.kind=="baseline")][col].iloc[0]
    sub=jk[(jk.regime==reg)&(jk.kind.isin(["LOSO","LOFO"]))].copy()
    sub["d"]=(sub[col]-base).abs(); r=sub.loc[sub.d.idxmax()]
    return r["drop"], r[col]
rows=[]
for reg,dirn,col in [("scratch","off-target $\\dtb$ (precision)","precision"),
                     ("pretrained","missing support $\\dbt$ (recall)","recall")]:
    b,lo,hi=jk_band(reg,col); drop,val=jk_worst(reg,col)
    dropl=esc(drop.replace("-fam:","fam "))
    rows.append(f"{reg} & {dirn} & {b:+.2f} & [{lo:+.2f},\\,{hi:+.2f}] & {hi-lo:.2f} & "
                f"\\texttt{{{dropl}}} ({val:+.2f}) \\\\")
w("tab_new_jackknife.tex", "\\begin{tabular}{llcccl}\n\\toprule\n"
  "regime & binding direction & baseline & jackknife range & span & most influential drop \\\\\n"
  "\\midrule\n"+"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}")

# ---------------------------------------------------------------- partial
pa=pd.read_csv(RES/"appearance_vs_motion_partial.csv")
rows=[]
for _,r in pa.iterrows():
    rows.append(f"{r.regime} & {r.direction} & {f2(r.raw_motion)} & {f2(r.raw_appearance)} & "
                f"{f2(r.partial_appearance_given_motion)} & {f2(r.partial_motion_given_appearance)} \\\\")
w("tab_new_partial.tex","\\begin{tabular}{llcccc}\n\\toprule\n"
  "& & \\multicolumn{2}{c}{raw $\\rho$} & \\multicolumn{2}{c}{partial $\\rho$} \\\\\n"
  "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
  "regime & direction & motion & appear. & appear.$\\mid$motion & motion$\\mid$appear. \\\\\n"
  "\\midrule\n"+"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}")

# ---------------------------------------------------------------- fixed-motion (compute here)
b=pd.read_csv(RES/"intervention_breakdown.csv")
kr=b[(b.arm=="FF") & b.source.str.contains("kitti_recovered") & ~b.source.str.contains("badmotion")].copy()
kr["app"]=kr.source.str.replace("kitti_recovered_","",regex=False)
benches=["kitti2015","kitti2012","flyingthings"]
piv=kr.pivot_table(index="app",columns="benchmark",values="peak_pck")
order=piv["kitti2015"].sort_values(ascending=False).index
rows=[]
for ap in order:
    cells=" & ".join(f"{piv.loc[ap,bn]:.1f}" for bn in benches)
    apl=esc(ap)
    rows.append(f"\\texttt{{{apl}}} & {cells} \\\\")
# within-motion appearance rho + cross-benchmark consistency
wr=[]
for bn in benches:
    g=kr[kr.benchmark==bn]; wr.append(spearmanr(g.peak_pck,-g["dino d(B->T)"]).statistic)
cons=[]
for i in range(len(benches)):
    for j in range(i+1,len(benches)):
        cons.append(spearmanr(piv[benches[i]],piv[benches[j]]).statistic)
# (within-motion / cross-benchmark rho are reported in the caption prose, not in-table)
w("tab_new_fixedmotion.tex","\\begin{tabular}{lccc}\n\\toprule\n"
  "appearance variant (fixed camera motion) & KITTI-15 & KITTI-12 & FlyingThings \\\\\n"
  "\\midrule\n"+"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}")
print(f"  [fixedmotion] within-motion rho {f2(wr[0])}/{f2(wr[1])}/{f2(wr[2])}; "
      f"cross-bench {f2(cons[0])}/{f2(cons[1])}/{f2(cons[2])}")

# ---------------------------------------------------------------- regime vs level
d=pd.read_csv(RES/"regime_vs_level_deconfound.csv")
def g(m,det=None):
    s=d[d.metric==m];
    if det is not None: s=s[s.detail==det]
    return float(s.value.iloc[0])
rows=[
 f"regime (pretrained vs scratch) & {g('ols_regime_pretrained_coef','cluster(benchmark)'):+.2f} & "
 f"{g('ols_regime_pretrained_p','cluster(benchmark)'):.1e} & {g('partial_spearman_gap_regime_given_level'):+.2f} \\\\",
 f"transfer level (per PCK) & {g('ols_level_coef','cluster(benchmark)'):+.4f} & "
 f"{g('ols_level_p','cluster(benchmark)'):.2f} & {g('partial_spearman_gap_level_given_regime'):+.2f} \\\\",
]
w("tab_new_regimelevel.tex","\\begin{tabular}{lccc}\n\\toprule\n"
  "predictor of the direction gap & OLS coef & $p$ (cluster) & partial $\\rho$ (other held) \\\\\n"
  "\\midrule\n"+"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}")
print(f"  [regimelevel] R2 regime {g('r2_regime_only'):.2f} vs level {g('r2_level_only'):.2f}; "
      f"within-regime gap-level pretr {g('within_pretrained_spearman_gap_level'):+.2f} "
      f"scratch {g('within_scratch_spearman_gap_level'):+.2f}")

# ---------------------------------------------------------------- TT-arm adjudication
tt=pd.read_csv(RES/"tt_arm_adjudication.csv")
sel=[("FF","flyingthings","precision"),
     ("TT","kitti2012","precision"),
     ("TT","kitti2015","precision"),
     ("TT","flyingthings","precision")]
rows=[]
for arm,bn,dr in sel:
    r=tt[(tt.arm==arm)&(tt.benchmark==bn)&(tt.direction==dr)].iloc[0]
    rows.append(f"{arm} & {bn} & {r.rho:+.2f} & {r.p_exact_2sided:.3f} & {r.rel_spread_pct:.0f}\\% & "
                f"{r.mc5_rank_selfcorr:.2f} \\\\")
w("tab_new_ttarm.tex","\\begin{tabular}{llccccl}\n\\toprule\n".replace("ccccl","cccc")+
  "arm & benchmark & $\\rho$ & exact $p$ ($n{=}9$) & dist.\\ spread & rank stab.\\ \\\\\n"
  "\\midrule\n"+"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}")

# ---------------------------------------------------------------- seed audit
sa=pd.read_csv(RES/"seed_audit_ff_summary.csv")
rows=[]
for _,r in sa.iterrows():
    if r.table=="canonical_paper": continue
    lab=r.table.replace("mean5_regen","5-seed mean")
    rows.append(f"{lab} & {r.flyingthings_precision:+.2f} & {r.ff_mean_precision:+.2f} \\\\")
w("tab_new_seedaudit.tex","\\begin{tabular}{lcc}\n\\toprule\n"
  "distance seed & FlyingThings precision $\\rho$ & FF-arm mean precision $\\rho$ \\\\\n"
  "\\midrule\n"+"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}")

# ---------------------------------------------------------------- anti-corr
ac=pd.read_csv(RES/"anticorr_by_benchmark.csv")
ac=ac[~ac.benchmark.str.startswith("MEAN")].sort_values("spearman_dtb_dbt")
rows=[f"{r.benchmark} & {r.spearman_dtb_dbt:+.2f} \\\\" for _,r in ac.iterrows()]
mean=ac.spearman_dtb_dbt.mean()
rows.append("\\midrule\nmean & "+f"{mean:+.2f} \\\\")
w("tab_new_anticorr.tex","\\begin{tabular}{lc}\n\\toprule\n"
  "benchmark & $\\rho(\\dtb,\\dbt)$ across sources \\\\\n\\midrule\n"+"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}")

# ---------------------------------------------------------------- diagonal sensitivity
dg=pd.read_csv(RES/"diagonal_sensitivity.csv")
dg=dg[~dg.variant.astype(str).str.upper().str.startswith("MEAN")]
rows=[f"{var(r.variant)} & {r.regime} & {r.rule_with_diagonal:+.2f} & {r.rule_without_diagonal:+.2f} & {r.delta:+.3f} \\\\"
      for _,r in dg.iterrows()]
mrow=f"\\midrule\nmean & & {dg.rule_with_diagonal.mean():+.3f} & {dg.rule_without_diagonal.mean():+.3f} & {dg.delta.mean():+.3f} \\\\"
w("tab_new_diagonal.tex","\\begin{tabular}{llccc}\n\\toprule\n"
  "variant & regime & rule (with diag.) & rule (no diag.) & $\\Delta$ \\\\\n\\midrule\n"+
  "\n".join(rows)+"\n"+mrow+"\n\\bottomrule\n\\end{tabular}")

print("\nall new tables written to", OUT)
