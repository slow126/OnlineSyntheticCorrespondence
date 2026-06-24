"""Ceiling oracles for the recoverable-signal claim — explicit constructions (v5).

The user-facing question: "the rule scores +0.50; how much signal was there to
recover?" Three oracles, weakest to strongest, all built ONLY from actual
transfer outcomes (no features):

  O1 ALL-VARIANT (the v4 number, 0.528): for variant v, mean over benchmarks
     and ALL other variants v' of spearman(rank_v(sources), rank_v'(sources)).
     CONSERVATIVE/LOWER BOUND: cross-REGIME disagreement (which the law says is
     real signal, not noise) is counted as irreproducibility.
  O2 SAME-REGIME: same construction but v' restricted to v's regime group
     (scratch vs pretrained, raft=scratch). Cross-regime structure no longer
     deflates the ceiling. The fair comparison for a REGIME-AWARE predictor.
  O3 SAME-REGIME CONSENSUS: spearman(rank_v, mean rank of the OTHER same-regime
     variants) — averaging others' rankings before correlating (less noisy
     reference => higher ceiling; analogous to held-arch consensus).

For each oracle: per-variant value, the rule's per-variant rho, and fraction.

    python scripts/transfer_analysis_v5/ceiling_oracles.py \
        --out scripts/transfer_analysis_v5/results/ceiling_oracles.csv
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


def rule_col_for(v):
    return ("mean_nn_a_to_b" if regime_of(v) == "scratch" else "mean_nn_b_to_a")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="scripts/transfer_analysis_v3/transfer_table.csv")
    ap.add_argument("--dist", default="analysis_v3/pairwise_self_distances.csv")
    ap.add_argument("--target", default="peak_pck")
    ap.add_argument("--min-shared", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    t = pd.read_csv(args.table)
    t = t[t.train_dataset.isin(PURE)].copy()
    t["variant"] = (t.model_family.astype(str) + "|" + t.pretrained.astype(str)
                    + "|" + t.freeze.astype(str))
    t = t[t.variant != "raft|False|False"]
    t["cv"] = t.benchmark + "|" + t.variant
    d = pd.read_csv(args.dist)
    te = d[(d.pair_type == "train_eval") & (d.space == "flow")]
    f = te.set_index(["dataset_a", "dataset_b"])[
        ["mean_nn_a_to_b", "mean_nn_b_to_a"]]
    t = t.join(f, on=["train_dataset", "benchmark"], how="left")
    t = t.dropna(subset=[args.target, "mean_nn_a_to_b"])
    variants = sorted(t.variant.unique())

    # per-(variant, benchmark) source->pck series
    series = {(v, b): g.set_index("train_dataset")[args.target]
              for (v, b), g in t.groupby(["variant", "benchmark"])}

    def pair_rho(v, v2, b):
        a, c = series.get((v, b)), series.get((v2, b))
        if a is None or c is None:
            return np.nan
        j = pd.concat([a, c], axis=1, keys=["a", "b"]).dropna()
        if len(j) < args.min_shared:
            return np.nan
        r = spearmanr(j.a, j.b).statistic
        return r if np.isfinite(r) else np.nan

    def consensus_rho(v, group, b):
        a = series.get((v, b))
        if a is None:
            return np.nan
        ranks = []
        for v2 in group:
            c = series.get((v2, b))
            if c is not None:
                ranks.append(c.rank())
        if not ranks:
            return np.nan
        ref = pd.concat(ranks, axis=1).mean(axis=1)
        j = pd.concat([a, ref], axis=1, keys=["a", "r"]).dropna()
        if len(j) < args.min_shared:
            return np.nan
        r = spearmanr(j.a, j.r).statistic
        return r if np.isfinite(r) else np.nan

    rows = []
    for v in variants:
        gv = t[t.variant == v]
        rc = rule_col_for(v)
        rule = np.nanmean([
            spearmanr(c[args.target], -c[rc]).statistic
            for _, c in gv.groupby("cv")
            if c.train_dataset.nunique() >= 3 and c[rc].std() > 1e-15])
        others_all = [v2 for v2 in variants if v2 != v]
        others_reg = [v2 for v2 in others_all if regime_of(v2) == regime_of(v)]
        benches = sorted(gv.benchmark.unique())
        o1 = np.nanmean([pair_rho(v, v2, b) for v2 in others_all for b in benches])
        o2 = np.nanmean([pair_rho(v, v2, b) for v2 in others_reg for b in benches])
        o3 = np.nanmean([consensus_rho(v, others_reg, b) for b in benches])
        rows.append(dict(variant=v, regime=regime_of(v), rule_rho=rule,
                         O1_all_variant=o1, O2_same_regime=o2,
                         O3_same_regime_consensus=o3,
                         frac_O1=rule / o1 if o1 > 0 else np.nan,
                         frac_O2=rule / o2 if o2 > 0 else np.nan,
                         frac_O3=rule / o3 if o3 > 0 else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    m = df.mean(numeric_only=True)
    print(f"\nMEANS: rule {m.rule_rho:+.3f} | O1 {m.O1_all_variant:+.3f} "
          f"(frac {m.frac_O1:.2f}) | O2 {m.O2_same_regime:+.3f} "
          f"(frac {m.frac_O2:.2f}) | O3 {m.O3_same_regime_consensus:+.3f} "
          f"(frac {m.frac_O3:.2f})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
