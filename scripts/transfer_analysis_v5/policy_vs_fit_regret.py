"""Score the per-regime 2-coefficient fitted model as a SELECTION strategy.

Question (Spencer, 2026-06-10): should the recommended policy be the fitted
per-regime linear model (both directed mean-NN features, one OLS per regime)
instead of the fit-free regime-aware policy (symmetric from scratch,
d_B->T pretrained)?

This script answers it in the decision metrics of the utility table: held-out
top-1 regret and large-gap pairwise accuracy, with the same folds as
per_regime_linear.py. Verdict from the 2026-06-10 run: comparable, not
decisively better (fit regret 3.0/1.6/1.6 vs policy 1.2/2.1/2.8 across
LOTO/LOBO/JOINT; pairwise gap>10 0.75/0.80/0.76 vs 0.77/0.78/0.78), and the
fit is clearly worse exactly where the fit-free rule shines (pretrained LOTO
regret 2.97 vs 0.66). The paper therefore keeps the fit-free policy as the
recommendation.

    python scripts/transfer_analysis_v5/policy_vs_fit_regret.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from per_regime_linear import AB, BA, fit_predict, folds, load  # noqa: E402

RES = Path(__file__).parent / "results"


def regret_and_pairs(df, col):
    regs, pairs = [], []
    for _, c in df.groupby("cv"):
        c = c.drop_duplicates("train_dataset")
        if c.train_dataset.nunique() < 3:
            continue
        best = c.peak_pck.max()
        regs.append(best - c.loc[c[col].idxmax(), "peak_pck"])
        v = c[["peak_pck", col]].values
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                if abs(v[i, 0] - v[j, 0]) > 10:
                    pairs.append((v[i, 0] - v[j, 0]) * (v[i, 1] - v[j, 1]) > 0)
    return float(np.median(regs)), float(np.mean(pairs)), len(regs)


def main():
    t, _ = load()
    recs = []
    for sp in ["LOTO", "LOBO", "JOINT"]:
        df = t.copy()
        df["fitted"] = np.nan
        for regime in ["scratch", "pretrained"]:
            rmask = df.regime == regime
            for tr_mask, ts_mask in folds(df, sp):
                tr, ts = tr_mask & rmask, ts_mask & rmask
                if ts.sum() == 0 or tr.sum() < 20:
                    continue
                pred, _w = fit_predict(df, tr, ts, [AB, BA])
                df.loc[ts, "fitted"] = pred
        sub = df.dropna(subset=["fitted"])
        for scope, frame in [("pooled", sub),
                             ("scratch", sub[sub.regime == "scratch"]),
                             ("pretrained", sub[sub.regime == "pretrained"])]:
            mr, pa, n = regret_and_pairs(frame, "fitted")
            recs.append(dict(split=sp, scope=scope, median_regret=mr,
                             pairwise_acc_gap10=pa, n_contexts=n))
            print(f"{sp:<6} {scope:<11} median regret {mr:5.2f}  "
                  f"pairwise(gap>10) {pa:.3f}  (n={n})")
    out = pd.DataFrame(recs)
    out.to_csv(RES / "policy_vs_fit_regret.csv", index=False)
    print(f"\nwrote {RES / 'policy_vs_fit_regret.csv'}")


if __name__ == "__main__":
    main()
