"""Does directionality help the LEVEL (L), not just the ranking (g)? (v5)

Two borrowing problems, evaluated as absolute predictors (MAE + Pearson):

1. LOBO-L (new benchmark): predict perf(i,k,v) from the SAME source's perf on
   other benchmarks {perf(i,e,v): e != k}, weighted by benchmark-benchmark
   similarity. Weights compared: uniform | symmetric mean_nn | directional
   k->e ("target covered by the helper benchmark") | directional e->k.

2. LOTO-L / the TRIANGLE idea (new source): predict perf(i,k,v) from OTHER
   sources' perf on the same benchmark {perf(j,k,v): j != i}, weighted by
   train-train similarity. Weights: uniform (= cell mean, current pipeline L)
   | symmetric | directional i->j ("candidate covered by helper") | j->i.
   Also reports within-context Spearman of the prediction — the part the
   original triangle test failed (rank info beyond the cell mean).

Directional lookups respect stored orientation in pairwise_self_distances
(pair stored once; reversed key reads the mirrored column).

    python scripts/transfer_analysis_v5/l_directional_check.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

PURE = ["flyingthings", "imagenet2dwarp", "movi_f", "pointodyssey", "sintel",
        "spair", "synthetic", "synthetic_2d_warp", "synthetic_large_zoom",
        "synthetic_random_flipping", "synthetic_small_zoom"]
EPS = 1e-6


def directional_lut(df, col_ab="mean_nn_a_to_b", col_ba="mean_nn_b_to_a",
                    col_sym="mean_nn_sym"):
    """(x, y) -> dict(xy=dist x->y, yx=dist y->x, sym=...), orientation-exact."""
    lut = {}
    for r in df.itertuples():
        ab, ba, sym = getattr(r, col_ab), getattr(r, col_ba), getattr(r, col_sym)
        lut[(r.dataset_a, r.dataset_b)] = dict(xy=ab, yx=ba, sym=sym)
        lut[(r.dataset_b, r.dataset_a)] = dict(xy=ba, yx=ab, sym=sym)
    return lut


def idw(vals, dists):
    v = np.asarray(vals, float)
    d = np.asarray(dists, float)
    m = np.isfinite(v) & np.isfinite(d)
    if not m.any():
        return np.nan
    w = 1.0 / (d[m] + EPS)
    return float((w * v[m]).sum() / w.sum())


def main():
    t = pd.read_csv("scripts/transfer_analysis_v3/transfer_table.csv")
    t = t[t.train_dataset.isin(PURE)].copy()
    t["variant"] = (t.model_family.astype(str) + "|" + t.pretrained.astype(str)
                    + "|" + t.freeze.astype(str))
    t = t[t.variant != "raft|False|False"].dropna(subset=["peak_pck"])
    t["cv"] = t.benchmark + "|" + t.variant
    d = pd.read_csv("analysis_v3/pairwise_self_distances.csv")
    flow = d[d.space == "flow"]
    ee = directional_lut(flow[flow.pair_type == "eval_eval"])
    tt = directional_lut(flow[flow.pair_type == "train_train"])
    perf = {(r.train_dataset, r.benchmark, r.variant): r.peak_pck
            for r in t.itertuples()}
    benches = sorted(t.benchmark.unique())
    sources = sorted(t.train_dataset.unique())

    def evaluate(pred_fn, name, rank_check=False):
        preds, actuals, ctxs = [], [], []
        for r in t.itertuples():
            p = pred_fn(r)
            if np.isfinite(p):
                preds.append(p); actuals.append(r.peak_pck); ctxs.append(r.cv)
        preds, actuals = np.array(preds), np.array(actuals)
        mae = float(np.mean(np.abs(preds - actuals)))
        pr = float(pearsonr(preds, actuals)[0])
        line = f"  {name:<28} MAE={mae:6.2f}  pearson={pr:+.3f}"
        if rank_check:
            df = pd.DataFrame(dict(p=preds, a=actuals, c=ctxs))
            rs = [spearmanr(g.a, g.p).statistic for _, g in df.groupby("c")
                  if g.p.std() > 1e-12 and len(g) >= 3]
            rs = [x for x in rs if np.isfinite(x)]
            line += f"  ctx_spearman={np.nanmean(rs):+.3f}"
        print(line)

    print("=== 1. LOBO-L: borrow same source across benchmarks "
          "(weights = benchmark-benchmark) ===")
    def lobo(weight):
        def f(r):
            vals, dists = [], []
            for e in benches:
                if e == r.benchmark:
                    continue
                p = perf.get((r.train_dataset, e, r.variant))
                if p is None:
                    continue
                pair = ee.get((r.benchmark, e))
                if pair is None:
                    continue
                dd = (1.0 if weight == "uniform" else pair[weight])
                vals.append(p); dists.append(dd)
            return idw(vals, dists) if weight != "uniform" else \
                (float(np.mean(vals)) if vals else np.nan)
        return f
    for w, label in [("uniform", "uniform mean"),
                     ("sym", "IDW symmetric"),
                     ("xy", "IDW directional k->e"),
                     ("yx", "IDW directional e->k")]:
        evaluate(lobo(w), label)

    print("\n=== 2. LOTO-L / triangle: borrow other sources on same benchmark "
          "(weights = train-train) ===")
    def loto(weight):
        def f(r):
            vals, dists = [], []
            for j in sources:
                if j == r.train_dataset:
                    continue
                p = perf.get((j, r.benchmark, r.variant))
                if p is None:
                    continue
                pair = tt.get((r.train_dataset, j))
                if pair is None:
                    continue
                dd = (1.0 if weight == "uniform" else pair[weight])
                vals.append(p); dists.append(dd)
            return idw(vals, dists) if weight != "uniform" else \
                (float(np.mean(vals)) if vals else np.nan)
        return f
    for w, label in [("uniform", "uniform (cell mean = L)"),
                     ("sym", "IDW symmetric (triangle)"),
                     ("xy", "IDW directional i->j"),
                     ("yx", "IDW directional j->i")]:
        evaluate(loto(w), label, rank_check=True)


if __name__ == "__main__":
    main()
