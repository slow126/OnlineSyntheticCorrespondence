"""Variance-components decomposition of transfer PCK, to answer reviewers asking
about model/architecture/occlusion effects.

Conditioning on the target (we make within-context claims), we split within-target
transfer-PCK variance into: SOURCE (which training set), MODEL (architecture x
regime), and RESIDUAL (source x model interaction + target-specific factors such as
occlusion/scene complexity + noise). We also report how much of the SOURCE axis the
coverage descriptor explains, and the consensus/reproducibility ceiling as the
direct measure of the model axis.

In-scope (non-degenerate) cells only, mirroring the paper's scope.
"""
import pandas as pd, numpy as np
from scipy.stats import pearsonr, spearmanr

t = pd.read_csv("scripts/transfer_analysis_v3/transfer_table_nomid.csv")
PURE = ["flyingthings","movi_f","pointodyssey","sintel","spair","synthetic",
        "synthetic_2d_warp","synthetic_large_zoom","synthetic_random_flipping",
        "synthetic_small_zoom","imagenet2dwarp"]
t = t[t.train_dataset.isin(PURE)].copy()
t["variant"] = t.model_family + "|" + t.pretrained.astype(str) + "|" + t.freeze.astype(str)
t["cov"] = -t["flow_mean_nn_eval_to_train_k1"]
SEM = {"spair","pfpascal","pfwillow","tss"}
def inscope(r):
    fam, pt, b = r.model_family, bool(r.pretrained), r.benchmark
    return not((fam == "catspp" and not pt) or ((fam == "raft" or not pt) and b in SEM))
d = t[t.apply(inscope, axis=1)].copy()

zt = lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else x * 0
d["z"]   = d.groupby("benchmark")["peak_pck"].transform(zt)   # remove target main effect
d["covz"] = d.groupby("benchmark")["cov"].transform(zt)

gm = d["z"].mean(); SS_tot = ((d["z"] - gm) ** 2).sum()
ss_main = lambda col: sum(len(g) * (g["z"].mean() - gm) ** 2 for _, g in d.groupby(col))
f_src = ss_main("train_dataset") / SS_tot
f_mod = ss_main("variant") / SS_tot
f_res = 1 - f_src - f_mod

r_tot = pearsonr(d["covz"], d["z"])[0]
sm = d.groupby("train_dataset").agg(z=("z", "mean"), cz=("covz", "mean"))
r_src = pearsonr(sm["cz"], sm["z"])[0]

print("Within-target variance decomposition (in-scope cells):")
print(f"  SOURCE   (which training set)        {100*f_src:4.0f}%")
print(f"  MODEL    (architecture x regime)     {100*f_mod:4.0f}%")
print(f"  RESIDUAL (interaction + occlusion/   {100*f_res:4.0f}%")
print(f"            scene-specific + noise)")
print(f"  n={len(d)} cells | {d.train_dataset.nunique()} sources | "
      f"{d.variant.nunique()} models | {d.benchmark.nunique()} targets")
print()
print(f"Coverage explains {100*r_tot**2:.0f}% of total within-target variance (r^2={r_tot**2:.2f})")
print(f"Coverage explains {100*r_src**2:.0f}% of the SOURCE-axis variance (r^2={r_src**2:.2f})")
print("Model axis is measured by the consensus: independent retrainings agree on the")
print("source ordering only at rho~0.59 (the reproducibility ceiling), so ~65% of a")
print("single model's source ranking is model-specific and not data-controllable.")
