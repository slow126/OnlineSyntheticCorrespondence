"""Anti-correlation heterogeneity of the two directed distances, by benchmark.

The paper (main.tex ~line 448) claims the two directed flow distances
d(T->B) = mean_nn_a_to_b and d(B->T) = mean_nn_b_to_a are anti-correlated
within contexts (mean Spearman ~ -0.35; audit recompute -0.374). No prior
script computed this, so we formalize it here at the (variant, benchmark)
context grain used everywhere else in v5 (asym_vs_sym_table.py construction):
canonical 11 sources, Middlebury-free table, raft|False|False dropped,
contexts with >= 3 sources and nonzero spread.

Because the distances depend only on (train_dataset, benchmark), the
per-context Spearman is identical for every variant sharing a benchmark
(verified: per-benchmark std across variants is reported); the context-grain
mean therefore equals the benchmark-grain mean on this complete 891-cell grid.
Both are printed and saved.

    python scripts/transfer_analysis_v5/anticorr_by_benchmark.py \
        --out scripts/transfer_analysis_v5/results/anticorr_by_benchmark.csv
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
    t = t.dropna(subset=[args.target, "mean_nn_a_to_b", "mean_nn_b_to_a"])

    # Per (variant, benchmark) context: spearman(d_tb, d_bt) across sources.
    ctx_rows = []
    for (bench, var), c in t.groupby(["benchmark", "variant"]):
        if c.train_dataset.nunique() < 3:
            continue
        if c.mean_nn_a_to_b.std() <= 1e-12 or c.mean_nn_b_to_a.std() <= 1e-12:
            continue
        r = spearmanr(c.mean_nn_a_to_b, c.mean_nn_b_to_a).statistic
        if np.isfinite(r):
            ctx_rows.append(dict(benchmark=bench, variant=var,
                                 n_sources=c.train_dataset.nunique(),
                                 spearman_dtb_dbt=float(r)))
    ctx = pd.DataFrame(ctx_rows)

    per_bench = (ctx.groupby("benchmark")
                 .agg(spearman_dtb_dbt=("spearman_dtb_dbt", "mean"),
                      std_across_variants=("spearman_dtb_dbt", "std"),
                      n_variant_contexts=("spearman_dtb_dbt", "size"),
                      n_sources=("n_sources", "max"))
                 .reset_index()
                 .sort_values("spearman_dtb_dbt"))

    mean_ctx = float(ctx.spearman_dtb_dbt.mean())
    mean_bench = float(per_bench.spearman_dtb_dbt.mean())

    out = per_bench.copy()
    out = pd.concat([out, pd.DataFrame([
        dict(benchmark="MEAN_over_benchmarks",
             spearman_dtb_dbt=mean_bench,
             std_across_variants=float("nan"),
             n_variant_contexts=int(per_bench.n_variant_contexts.sum()),
             n_sources=float("nan")),
        dict(benchmark="MEAN_over_contexts",
             spearman_dtb_dbt=mean_ctx,
             std_across_variants=float("nan"),
             n_variant_contexts=len(ctx),
             n_sources=float("nan")),
    ])], ignore_index=True)
    out.to_csv(args.out, index=False)

    print(out.to_string(index=False,
                        float_format=lambda x: f"{x:+.4f}"))
    print(f"\nmean within-context spearman(d_tb, d_bt), context grain "
          f"({len(ctx)} contexts): {mean_ctx:+.4f}")
    print(f"mean within-context spearman(d_tb, d_bt), benchmark grain "
          f"({len(per_bench)} benchmarks): {mean_bench:+.4f}")
    print(f"paper claim: ~ -0.35 (main.tex ~l.448); audit recompute: -0.374")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
