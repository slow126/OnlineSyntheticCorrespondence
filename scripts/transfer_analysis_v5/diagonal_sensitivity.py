"""Diagonal-cell sensitivity of the matched-direction regime rule.

Four of the canonical 11 sources (flyingthings, pointodyssey, spair,
synthetic) are also benchmarks, so train==benchmark "diagonal" cells sit
inside those transfer contexts. This script reproduces the fit-free
matched-direction rule (asym_vs_sym_table.py construction, exactly) with all
cells and again with diagonal cells dropped, per variant and on average.
Audit recompute being formalized: mean rule rho 0.507 -> 0.498; largest
per-variant move ~0.03 (catspp|True|True +0.502 -> +0.475).

    python scripts/transfer_analysis_v5/diagonal_sensitivity.py \
        --out scripts/transfer_analysis_v5/results/diagonal_sensitivity.csv
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

PURE = ["flyingthings", "imagenet2dwarp", "movi_f", "pointodyssey", "sintel",
        "spair", "synthetic", "synthetic_2d_warp", "synthetic_large_zoom",
        "synthetic_random_flipping", "synthetic_small_zoom"]


def regime_of(v):
    arch, pre, _ = v.split("|")
    return "scratch" if (pre == "False" or arch == "raft") else "pretrained"


def rule_rho(g, target, rule_col):
    """Mean within-context spearman(target, -rule_col) over benchmark contexts."""
    rs = []
    for _, c in g.groupby("benchmark"):
        if c.train_dataset.nunique() < 3 or c[rule_col].std() <= 1e-12:
            continue
        r = spearmanr(c[target], -c[rule_col]).statistic
        if np.isfinite(r):
            rs.append(r)
    return (float(np.mean(rs)), len(rs)) if rs else (float("nan"), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table",
                    default="scripts/transfer_analysis_v3/transfer_table_nomid.csv")
    ap.add_argument("--dist", default="analysis_v3/pairwise_self_distances.csv")
    ap.add_argument("--target", default="peak_pck")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    t = pd.read_csv(args.table)
    t = t[t.train_dataset.isin(PURE)].copy()
    t["variant"] = (t.model_family.astype(str) + "|" + t.pretrained.astype(str)
                    + "|" + t.freeze.astype(str))
    t = t[t.variant != "raft|False|False"]
    d = pd.read_csv(args.dist)
    te = d[(d.pair_type == "train_eval") & (d.space == "flow")]
    f = te.set_index(["dataset_a", "dataset_b"])[
        ["mean_nn_a_to_b", "mean_nn_b_to_a"]]
    t = t.join(f, on=["train_dataset", "benchmark"], how="left")
    t = t.dropna(subset=[args.target, "mean_nn_a_to_b"])

    diag = t.train_dataset == t.benchmark
    print(f"diagonal cells dropped: {int(diag.sum())} of {len(t)} "
          f"(sources also serving as benchmarks: "
          f"{sorted(t.loc[diag, 'train_dataset'].unique())})")

    rows = []
    for v, g in t.groupby("variant"):
        rule_col = ("mean_nn_a_to_b" if regime_of(v) == "scratch"
                    else "mean_nn_b_to_a")
        full, n_full = rule_rho(g, args.target, rule_col)
        nod, n_nod = rule_rho(g[g.train_dataset != g.benchmark],
                              args.target, rule_col)
        rows.append(dict(variant=v, regime=regime_of(v),
                         rule_with_diagonal=full,
                         rule_without_diagonal=nod,
                         delta=nod - full,
                         n_contexts_with=n_full, n_contexts_without=n_nod))
    df = pd.DataFrame(rows).sort_values("rule_with_diagonal", ascending=False)
    mean = dict(variant="MEAN", regime="",
                rule_with_diagonal=float(df.rule_with_diagonal.mean()),
                rule_without_diagonal=float(df.rule_without_diagonal.mean()),
                delta=float((df.rule_without_diagonal
                             - df.rule_with_diagonal).mean()),
                n_contexts_with=int(df.n_contexts_with.sum()),
                n_contexts_without=int(df.n_contexts_without.sum()))
    df = pd.concat([df, pd.DataFrame([mean])], ignore_index=True)
    df.to_csv(args.out, index=False)

    print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\nmean rule rho with diagonals:    {mean['rule_with_diagonal']:+.4f}"
          f"  (audit: 0.507)")
    print(f"mean rule rho without diagonals: {mean['rule_without_diagonal']:+.4f}"
          f"  (audit: 0.498)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
