"""Audit of the seed-averaging claim behind the pre-registered OOS result.

Question: does the paper's FlyingThings rho=+0.67 (FF arm, tab:dirtest) rest on
single-seed or 5-seed-averaged directional distances, and how much do the
FF-arm correlations move across individual subsample seeds?

Provenance established first (see intervention_distances_directional_5seed.py
docstring): the canonical le-wm CSV was written 2026-06-09 23:14:53 -0600 by a
session heredoc averaging NN-subsample seeds 0..4 (transcript 5764a6fe...jsonl,
command timestamp 2026-06-10T05:14:43Z); regeneration is bit-identical (27/27
non-middlebury rows). The on-disk intervention_distances_directional.py is the
OLD single-seed generator and does NOT reproduce the canonical CSV.

This script recomputes the OOS spearman table (same harvest + merge logic as
intervention_oos_test.py, middlebury dropped) for:
  - each per-seed distance table (seed0..seed4, NEW regenerated files)
  - the regenerated 5-seed mean
  - the canonical CSV the paper consumed (sanity: must equal the 5-seed mean)

Outputs (NEW files):
  results/seed_audit_oos_per_seed.csv   per (table, arm, benchmark) spearmans
  results/seed_audit_ff_summary.csv     FF-arm summary incl. flyingthings cell
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

GRID = Path("/mnt/nvme_1tb_a/snapshots/transfer_grid")
LEWM = Path("/home/spencer/Projects/le-wm/outputs")
RES = Path(__file__).parent / "results"

# --- grid peak PCKs, exactly as intervention_oos_test.py harvests them ---
rows = []
for d in sorted(GRID.iterdir()):
    f = d / "validation_results.csv"
    if not f.exists():
        continue
    v = pd.read_csv(f)
    if v["epoch"].nunique() < 50:  # still-training run (grid horizon=50); skip
        continue
    src = d.name.rsplit("_pt", 1)[0]
    arm = "FF" if "_pt0_fz0" in d.name else "TT"
    for b, g in v.groupby("benchmark"):
        rows.append((src, arm, b, float(g["pck"].max())))
pck = pd.DataFrame(rows, columns=["source", "arm", "benchmark", "peak_pck"])
pck = pck[pck.benchmark != "middlebury"]  # excluded everywhere (bugged eval)

TABLES = {f"seed{k}": LEWM / f"intervention_motion_distances_directional_seed{k}.csv"
          for k in range(5)}
TABLES["mean5_regen"] = LEWM / "intervention_motion_distances_directional_5seed.csv"
TABLES["canonical_paper"] = LEWM / "intervention_motion_distances_directional.csv"

recs = []
for name, path in TABLES.items():
    dist = pd.read_csv(path)
    m = pck.merge(dist, on=["source", "benchmark"], how="inner")
    for arm, sub in m.groupby("arm"):
        for b, g in sub.groupby("benchmark"):
            if g.source.nunique() < 3:
                continue
            recs.append(dict(
                table=name, arm=arm, benchmark=b, n=g.source.nunique(),
                precision=spearmanr(g.peak_pck, -g.flow_mean_nn_a_to_b).statistic,
                recall=spearmanr(g.peak_pck, -g.flow_mean_nn_b_to_a).statistic,
                sym=spearmanr(g.peak_pck, -g.flow_mean_nn_sym).statistic))
df = pd.DataFrame(recs)
out1 = RES / "seed_audit_oos_per_seed.csv"
df.to_csv(out1, index=False)

# --- FF-arm summary: per-table mean over the 3 live benchmarks + the
#     flyingthings cell (the paper's headline +0.67) ---
ff = df[df.arm == "FF"]
summ = []
for name in TABLES:
    s = ff[ff.table == name]
    fy = s[s.benchmark == "flyingthings"].iloc[0]
    summ.append(dict(
        table=name,
        ff_mean_precision=s.precision.mean(),
        ff_mean_recall=s.recall.mean(),
        ff_mean_sym=s.sym.mean(),
        flyingthings_precision=fy.precision,
        flyingthings_recall=fy.recall,
        flyingthings_sym=fy.sym))
summ = pd.DataFrame(summ)
out2 = RES / "seed_audit_ff_summary.csv"
summ.to_csv(out2, index=False)

pd.set_option("display.width", 200)
print("=== per (table, arm, benchmark) spearman(peak_pck, -distance), middlebury dropped ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
print("\n=== FF-arm summary (3 live benchmarks) ===")
print(summ.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

# sanity: canonical == regenerated 5-seed mean on every correlation
c = df[df.table == "canonical_paper"].drop(columns="table").reset_index(drop=True)
r = df[df.table == "mean5_regen"].drop(columns="table").reset_index(drop=True)
ident = bool(np.allclose(c[["precision", "recall", "sym"]],
                         r[["precision", "recall", "sym"]], atol=0, rtol=0))
print(f"\ncanonical_paper correlations identical to mean5_regen: {ident}")

# per-seed spread of the headline cell
fyp = summ[summ.table.str.startswith("seed")].flyingthings_precision
print(f"\nflyingthings FF precision across seeds 0..4: "
      f"{[f'{v:+.4f}' for v in fyp]}  (min {fyp.min():+.4f}, max {fyp.max():+.4f}, "
      f"mean {fyp.mean():+.4f})")
print(f"averaged-distance flyingthings FF precision (paper): "
      f"{summ[summ.table == 'canonical_paper'].flyingthings_precision.iloc[0]:+.4f}")
print(f"\nwrote {out1}\nwrote {out2}")
