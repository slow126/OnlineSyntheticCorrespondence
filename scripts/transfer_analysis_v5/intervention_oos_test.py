"""Pre-registered out-of-sample test of the Regime-Direction Law (v5, §7.3).

The law was discovered on the canonical-11 transfer table. The 13-cell kubric
intervention transfer grid (finished 2026-06-09) was never part of that
analysis. Prediction, stated in advance of looking at the grid:

  FF (from-scratch) cells: precision (a->b) ranks sources better than recall.
  TT (pretrained-frozen) cells: recall (b->a) ranks better than precision.

Inputs:
  - grid snapshots:  /mnt/nvme_1tb_a/snapshots/transfer_grid/*/validation_results.csv
  - directional distances: le-wm/outputs/intervention_motion_distances_directional.csv
    (same vectors/space as the original symmetric CSV; see
     le-wm/outputs/intervention_distances_directional.py)

Honest caveats baked into the output:
  - The KITTI-family sources were DESIGNED as matched-motion appearance ablations
    (hq vs matte), so their motion distances are near-tied on KITTI benchmarks by
    construction; appearance dominates those cells (the known from-scratch
    appearance cost). The discriminative cells are the cross-benchmark ones
    (flyingthings, middlebury).
  - TT arm has n=3 sources -> sign checks only, no significance.

    python scripts/transfer_analysis_v5/intervention_oos_test.py \
        --out scripts/transfer_analysis_v5/results/intervention_oos.csv
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

GRID = Path("/mnt/nvme_1tb_a/snapshots/transfer_grid")
DIST = Path("/home/spencer/Projects/le-wm/outputs/"
            "intervention_motion_distances_directional.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

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
    dist = pd.read_csv(DIST)
    import os
    # middlebury eval confirmed bugged (2026-06-10): excluded by default everywhere
    # (matches blocks.py) until models are re-evaluated with the fixed eval. The
    # distance table still carries middlebury rows, so drop here too or it leaks
    # back in via the inner merge. Override with DROP_BENCHMARKS if needed.
    drop = [b for b in os.environ.get("DROP_BENCHMARKS", "middlebury").split(",") if b]
    if drop:
        pck = pck[~pck.benchmark.isin(drop)]
    m = pck.merge(dist, on=["source", "benchmark"], how="inner")

    # the canonical-fitted scratch 2-coef model, applied to the grid (fully
    # OOS): a selection-layer combiner that charges both costs should
    # attenuate here in proportion to its recall weight (two-cost reading)
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from per_regime_linear import AB, BA, load as _load_canonical
    t, _ = _load_canonical()
    tr = t[t.regime == "scratch"]
    mu = tr.groupby("cv").peak_pck.transform("mean")
    Xw, scales = [], {}
    for c in [AB, BA]:
        dm = tr[c] - tr.groupby("cv")[c].transform("mean")
        scales[c] = dm.std()
        Xw.append((dm / scales[c]).values)
    w, *_ = np.linalg.lstsq(np.column_stack(Xw), (tr.peak_pck - mu).values,
                            rcond=None)

    def scratch_fit_rho(g):
        zp = (g.flow_mean_nn_a_to_b - g.flow_mean_nn_a_to_b.mean()) / scales[AB]
        zr = (g.flow_mean_nn_b_to_a - g.flow_mean_nn_b_to_a.mean()) / scales[BA]
        return spearmanr(g.peak_pck, w[0] * zp + w[1] * zr).statistic

    recs = []
    for arm, sub in m.groupby("arm"):
        for b, g in sub.groupby("benchmark"):
            if g.source.nunique() < 3:
                continue
            recs.append(dict(
                arm=arm, benchmark=b, n=g.source.nunique(),
                precision=spearmanr(g.peak_pck, -g.flow_mean_nn_a_to_b).statistic,
                recall=spearmanr(g.peak_pck, -g.flow_mean_nn_b_to_a).statistic,
                sym=spearmanr(g.peak_pck, -g.flow_mean_nn_sym).statistic,
                scratch_fit=scratch_fit_rho(g) if arm == "FF" else np.nan,
                # how much each distance actually VARIES across the sources
                # (max/min ratio): a correlation only carries causal weight
                # where its factor has real dynamic range
                prec_spread=float(g.flow_mean_nn_a_to_b.max()
                                  / g.flow_mean_nn_a_to_b.min()),
                rec_spread=float(g.flow_mean_nn_b_to_a.max()
                                 / g.flow_mean_nn_b_to_a.min()),
            ))
    df = pd.DataFrame(recs)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    for arm in ["FF", "TT"]:
        s = df[df.arm == arm]
        if len(s):
            print(f"\n{arm} MEAN: precision {s.precision.mean():+.3f}  "
                  f"recall {s.recall.mean():+.3f}  sym {s.sym.mean():+.3f}   "
                  f"(law predicts {'precision' if arm == 'FF' else 'recall'} wins)")
    ff = df[df.arm == "FF"]
    verdict = "PASS" if (len(ff) and ff.precision.mean() > ff.recall.mean()
                         and ff.precision.mean() > ff.sym.mean()) else "FAIL"
    print(f"\nFF-arm directional verdict: {verdict} "
          f"(precision > recall and > sym on the arm mean)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
