"""GLU-Net architecture-generality arm of the intervention OOS test.

The CATs++ intervention grid (intervention_oos_test.py) showed FF precision >
recall (off-target ranks from scratch). run_transfer_grid_glunet.py pre-registered:
on this GLU-Net FF grid the off-target cost should carry the FlyingThings ranking
(positive precision rho where d_TB varies 3.1x), the summed cost should attenuate,
and missing support should stay ~negative. If GLU-Net-FF came out recall-ranked on
FlyingThings, the two-cost model is wrong.

Mirrors intervention_oos_test.py exactly, except: GRID=transfer_grid_glunet,
all cells are FF, completion keyed on model_best.pth (the grid used eval-every-5
over 100 epochs, so the <50-unique-epoch skip does not apply), and raced
duplicate dirs are deduped to the latest per source.

    python scripts/transfer_analysis_v5/glunet_intervention_oos.py \
        --out scripts/transfer_analysis_v5/results/glunet_intervention_oos.csv
"""
from __future__ import annotations
import argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

GRID = Path("/mnt/nvme_1tb_a/snapshots/transfer_grid_glunet")
DIST = Path("/home/spencer/Projects/le-wm/outputs/intervention_motion_distances_directional.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scripts/transfer_analysis_v5/results/glunet_intervention_oos.csv")
    a = ap.parse_args()
    # dedup raced duplicates: keep the latest complete dir per source
    by_src = {}
    for d in sorted(GRID.iterdir()):
        if not (d / "model_best.pth").exists(): continue
        if "_pt0_fz0" not in d.name: continue
        src = d.name.rsplit("_pt", 1)[0]
        by_src[src] = d  # sorted() => last wins (latest timestamp)
    rows = []
    for src, d in by_src.items():
        v = pd.read_csv(d / "validation_results.csv")
        for b, g in v.groupby("benchmark"):
            rows.append((src, "FF", b, float(g["pck"].max())))
    pck = pd.DataFrame(rows, columns=["source","arm","benchmark","peak_pck"])
    pck = pck[pck.benchmark != "middlebury"]
    dist = pd.read_csv(DIST)
    m = pck.merge(dist, on=["source","benchmark"], how="inner")
    print(f"harvested {len(by_src)} GLU-Net FF cells: {sorted(by_src)}")
    print(f"merged rows: {len(m)} over benchmarks {sorted(m.benchmark.unique())}\n")
    recs = []
    for b, g in m.groupby("benchmark"):
        if g.source.nunique() < 3: continue
        recs.append(dict(arch="glunet", arm="FF", benchmark=b, n=g.source.nunique(),
            precision=spearmanr(g.peak_pck, -g.flow_mean_nn_a_to_b).statistic,
            recall=spearmanr(g.peak_pck, -g.flow_mean_nn_b_to_a).statistic,
            sym=spearmanr(g.peak_pck, -g.flow_mean_nn_sym).statistic,
            prec_spread=float(g.flow_mean_nn_a_to_b.max()/g.flow_mean_nn_a_to_b.min()),
            rec_spread=float(g.flow_mean_nn_b_to_a.max()/g.flow_mean_nn_b_to_a.min())))
    df = pd.DataFrame(recs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); df.to_csv(a.out, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print(f"\nGLU-Net FF MEAN: precision {df.precision.mean():+.3f}  recall {df.recall.mean():+.3f}  sym {df.sym.mean():+.3f}")
    fly = df[df.benchmark=="flyingthings"]
    if len(fly):
        r=fly.iloc[0]
        print(f"\nPRE-REGISTERED CELL (FlyingThings, where d_TB varies {r.prec_spread:.1f}x):")
        print(f"  precision={r.precision:+.3f}  recall={r.recall:+.3f}  sym={r.sym:+.3f}")
        print(f"  -> {'CONSISTENT with two-cost (precision>recall)' if r.precision>r.recall else 'BREAKS two-cost (recall>=precision)'}")
    print(f"\nwrote {a.out}")

if __name__=="__main__": main()
