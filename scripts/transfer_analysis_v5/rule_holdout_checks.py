"""Hold-out checks for the Regime-Direction rule (v5).

1. Leave-one-variant-out (LOVO) direction selection: for each held variant,
   choose the direction from the OTHER variants of its regime group only,
   then score the held variant. Tests that the direction *choice* (the only
   discrete decision in the rule) generalizes across variants.
2. Continuum check: per-variant direction preference d = rho(a->b) - rho(b->a)
   vs continuous regime proxies (mean transfer level, within-context range).

Fit-free everywhere; raw CSV inputs; deterministic.

    python scripts/transfer_analysis_v5/rule_holdout_checks.py \
        --out scripts/transfer_analysis_v5/results/rule_holdout_checks.csv
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


def regime_of(v: str) -> str:
    arch, pre, _ = v.split("|")
    return "scratch" if (pre == "False" or arch == "raft") else "pretrained"


def load(table_path, dist_path, target):
    t = pd.read_csv(table_path)
    t = t[t.train_dataset.isin(PURE)].copy()
    t["variant"] = (t.model_family.astype(str) + "|" + t.pretrained.astype(str)
                    + "|" + t.freeze.astype(str))
    t = t[t.variant != "raft|False|False"]
    t["cv"] = t.benchmark + "|" + t.variant
    d = pd.read_csv(dist_path)
    te = d[(d.pair_type == "train_eval") & (d.space == "flow")]
    f = te.set_index(["dataset_a", "dataset_b"])[
        ["mean_nn_a_to_b", "mean_nn_b_to_a", "mean_nn_sym"]]
    t = t.join(f, on=["train_dataset", "benchmark"], how="left")
    return t.dropna(subset=[target, "mean_nn_a_to_b"])


def vctx(g, col, target):
    rs = [spearmanr(c[target], -c[col]).statistic for _, c in g.groupby("cv")
          if c.train_dataset.nunique() >= 3 and c[col].std() > 1e-12]
    rs = [r for r in rs if np.isfinite(r)]
    return float(np.nanmean(rs)) if rs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="scripts/transfer_analysis_v3/transfer_table.csv")
    ap.add_argument("--dist", default="analysis_v3/pairwise_self_distances.csv")
    ap.add_argument("--target", default="peak_pck")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    t = load(args.table, args.dist, args.target)
    variants = sorted(t.variant.unique())
    d_v = {v: (vctx(t[t.variant == v], "mean_nn_a_to_b", args.target)
               - vctx(t[t.variant == v], "mean_nn_b_to_a", args.target))
           for v in variants}

    rows = []
    for held in variants:
        grp = [v for v in variants if v != held and regime_of(v) == regime_of(held)]
        mean_d = float(np.mean([d_v[v] for v in grp]))
        chosen = "mean_nn_a_to_b" if mean_d > 0 else "mean_nn_b_to_a"
        rule_dir = ("mean_nn_a_to_b" if regime_of(held) == "scratch"
                    else "mean_nn_b_to_a")
        rows.append(dict(
            variant=held, regime=regime_of(held),
            lovo_dir=chosen.replace("mean_nn_", ""),
            rule_dir=rule_dir.replace("mean_nn_", ""),
            agree=chosen == rule_dir,
            rho_lovo=vctx(t[t.variant == held], chosen, args.target),
            rho_rule=vctx(t[t.variant == held], rule_dir, args.target),
            rho_sym=vctx(t[t.variant == held], "mean_nn_sym", args.target),
            d=d_v[held],
            mean_level=float(t[t.variant == held][args.target].mean()),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print(df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print(f"\nLOVO mean rho: {df.rho_lovo.mean():+.3f}  "
          f"(rule {df.rho_rule.mean():+.3f}, sym {df.rho_sym.mean():+.3f}); "
          f"direction agreement {df.agree.mean():.0%}")
    rho_lvl = spearmanr(df.d, df.mean_level).statistic
    print(f"continuum: spearman(d, mean transfer level) = {rho_lvl:+.3f} "
          f"(n={len(df)})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
