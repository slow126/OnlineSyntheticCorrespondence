"""Top-k selection regret — the decision-centric alternative to Spearman.

For each context (benchmark|model|pretrained|freeze): the predictor picks its
top-ranked source; regret = (best source's actual transfer) - (picked source's
actual transfer), in PCK points. Near-tie rank flips between genuinely similar
sources cost ~0 regret by construction, so this metric is robust to exactly
the noise that tanks Spearman.

Reported per (split, family):
  - mean / median top-1 regret (PCK)
  - P(top-1 pick within 1 PCK of oracle)   "picked a co-optimal source"
  - P(oracle best in predictor's top-3)
  - random-pick baseline regret (expected regret of uniform choice, same contexts)

Usage:
    python scripts/transfer_analysis_v4/selection_regret.py \
        --rows-dir scripts/transfer_analysis_v4/results_perarch_merged/predictions/peak_pck \
        --out scripts/transfer_analysis_v4/results_perarch_merged/selection_regret.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def context_regret(g: pd.DataFrame, pred_col: str) -> dict | None:
    g = g.drop_duplicates("train_dataset")
    if g["train_dataset"].nunique() < 3:
        return None
    best = g["actual"].max()
    picked = g.loc[g[pred_col].idxmax(), "actual"]
    top3 = g.nlargest(min(3, len(g)), pred_col)["actual"]
    return dict(
        regret=best - picked,
        within1=(best - picked) <= 1.0,
        oracle_in_top3=bool((top3 >= best).any()),
        random_regret=float((best - g["actual"]).mean()),
        n_src=g["train_dataset"].nunique(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-dir", required=True)
    ap.add_argument("--families", nargs="+", default=["motion", "appearance"])
    ap.add_argument("--splits", nargs="+", default=["LOTO", "LOBO", "JOINT"])
    ap.add_argument("--pred-col", default="g")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows_dir = Path(args.rows_dir)
    recs = []
    for split in args.splits:
        for fam in args.families:
            f = rows_dir / f"rows_{split}_{fam}.csv"
            if not f.exists():
                continue
            rows = pd.read_csv(f)
            per_ctx = [r for _, g in rows.groupby("context_id")
                       if (r := context_regret(g, args.pred_col)) is not None]
            if not per_ctx:
                continue
            d = pd.DataFrame(per_ctx)
            recs.append(dict(
                split=split, family=fam, n_contexts=len(d),
                mean_regret=d["regret"].mean(),
                median_regret=d["regret"].median(),
                p_within1=d["within1"].mean(),
                p_oracle_top3=d["oracle_in_top3"].mean(),
                random_mean_regret=d["random_regret"].mean(),
            ))
    out = pd.DataFrame(recs)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}\n")
    for _, r in out.iterrows():
        print(f"  {r['split']:<6} {r['family']:<11} "
              f"mean regret={r['mean_regret']:5.2f} (median {r['median_regret']:4.2f}; "
              f"random {r['random_mean_regret']:5.2f})  "
              f"P(within 1 PCK)={r['p_within1']:.2f}  "
              f"P(best in top3)={r['p_oracle_top3']:.2f}")


if __name__ == "__main__":
    main()
