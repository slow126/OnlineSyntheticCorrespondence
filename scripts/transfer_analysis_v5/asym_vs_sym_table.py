"""Asymmetric vs symmetric motion metrics — why direction matters (v5, Table for §6.1).

Per-variant, fit-free within-context Spearman for:
  regime rule (direction matched to regime)  |  precision a->b  |  recall b->a
  vs the symmetric family: mean_nn_sym, FID, sliced-W2, MMD.

The point: symmetric metrics hedge the regime flip — decent everywhere, best
nowhere — and in flip-extreme cells (pretrained GLU-Net) they lose 0.3-0.4 rho
to the regime-matched direction. This is the failure mode that made the
original asymmetric-coverage hypothesis look like it "washed out".

    python scripts/transfer_analysis_v5/asym_vs_sym_table.py \
        --out scripts/transfer_analysis_v5/results/asym_vs_sym.csv
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="scripts/transfer_analysis_v3/transfer_table.csv")
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
    t["cv"] = t.benchmark + "|" + t.variant
    d = pd.read_csv(args.dist)
    te = d[(d.pair_type == "train_eval") & (d.space == "flow")]
    f = te.set_index(["dataset_a", "dataset_b"])[
        ["mean_nn_a_to_b", "mean_nn_b_to_a", "mean_nn_sym"]]
    t = t.join(f, on=["train_dataset", "benchmark"], how="left")
    t = t.dropna(subset=[args.target, "mean_nn_a_to_b"])

    SYM = [("sym", "mean_nn_sym"), ("fid", "flow_fid"),
           ("w2", "flow_sliced_w2"), ("mmd", "flow_mmd")]

    def vctx(g, col):
        rs = [spearmanr(c[args.target], -c[col]).statistic
              for _, c in g.groupby("cv")
              if c.train_dataset.nunique() >= 3 and c[col].std() > 1e-12]
        rs = [r for r in rs if np.isfinite(r)]
        return float(np.nanmean(rs)) if rs else float("nan")

    rows = []
    for v, g in t.groupby("variant"):
        rule_col = ("mean_nn_a_to_b" if regime_of(v) == "scratch"
                    else "mean_nn_b_to_a")
        rec = dict(variant=v, regime=regime_of(v),
                   rule=vctx(g, rule_col),
                   precision=vctx(g, "mean_nn_a_to_b"),
                   recall=vctx(g, "mean_nn_b_to_a"))
        for name, col in SYM:
            rec[name] = vctx(g, col) if col in g.columns else float("nan")
        rec["rule_minus_best_sym"] = rec["rule"] - np.nanmax(
            [rec[n] for n, _ in SYM])
        rows.append(rec)
    df = pd.DataFrame(rows).sort_values("rule_minus_best_sym", ascending=False)
    mean_row = df.drop(columns=["variant", "regime"]).mean(numeric_only=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print("\nMEANS:",
          "  ".join(f"{k}={mean_row[k]:+.3f}" for k in
                    ["rule", "precision", "recall", "sym", "fid", "w2", "mmd"]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
