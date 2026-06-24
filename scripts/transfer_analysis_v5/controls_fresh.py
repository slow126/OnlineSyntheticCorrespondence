"""Fresh v5 controls for the fit-free regime rule (no v4 inheritance).

All computed directly from transfer_table + pairwise_self_distances, deterministic:

1. SHUFFLE NULL — permute actuals within each context (200 perms), recompute the
   rule's mean within-context Spearman. Expect ~0, and an empirical p for the
   observed +0.50.
2. LEAVE-ONE-GENERATOR-FAMILY-OUT — drop each of the 5 source families (sdf3d,
   warp2d, kubric, realflow, semantic), recompute rule rho. No single family
   should be load-bearing.
3. SIZE CONTROL (fit-free) — rank sources by log dataset size / supervision
   density instead of motion: should predict ~nothing.
4. LEAVE-ONE-BENCHMARK-OUT — drop each benchmark, recompute rule rho (range).

    python scripts/transfer_analysis_v5/controls_fresh.py \
        --out scripts/transfer_analysis_v5/results/controls_fresh.csv
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
FAMILY_MAP = {
    "synthetic": "sdf3d", "synthetic_large_zoom": "sdf3d",
    "synthetic_small_zoom": "sdf3d", "synthetic_random_flipping": "sdf3d",
    "synthetic_2d_warp": "warp2d", "imagenet2dwarp": "warp2d",
    "movi_f": "kubric", "flyingthings": "realflow", "pointodyssey": "realflow",
    "sintel": "realflow", "spair": "semantic",
}
N_PERM = 200
SEED = 0


def rule_col_for(variant: str) -> str:
    arch, pre, _ = variant.split("|")
    return ("mean_nn_a_to_b" if (pre == "False" or arch == "raft")
            else "mean_nn_b_to_a")


def rule_rho(t: pd.DataFrame, target="peak_pck", actual_col=None) -> float:
    col = actual_col or target
    rs = []
    for _, c in t.groupby("cv"):
        rc = rule_col_for(c["variant"].iloc[0])
        if c.train_dataset.nunique() < 3 or c[rc].std() <= 1e-15:
            continue
        r = spearmanr(c[col], -c[rc]).statistic
        if np.isfinite(r):
            rs.append(r)
    return float(np.nanmean(rs)) if rs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="scripts/transfer_analysis_v3/transfer_table.csv")
    ap.add_argument("--dist", default="analysis_v3/pairwise_self_distances.csv")
    ap.add_argument("--target", default="peak_pck")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

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

    recs = []
    obs = rule_rho(t, args.target)
    recs.append(dict(control="observed_rule", value=obs, note="baseline"))
    print(f"observed rule rho: {obs:+.4f}")

    # 1. shuffle null
    nulls = []
    for _ in range(N_PERM):
        ts = t.copy()
        ts["shuf"] = ts.groupby("cv")[args.target].transform(
            lambda s: rng.permutation(s.values))
        nulls.append(rule_rho(ts, args.target, actual_col="shuf"))
    nulls = np.array(nulls)
    p_emp = float(np.mean(nulls >= obs))
    recs.append(dict(control="shuffle_null_mean", value=float(nulls.mean()),
                     note=f"sd={nulls.std():.4f}; p_emp(obs)={p_emp:.4f} ({N_PERM} perms)"))
    print(f"shuffle null: mean {nulls.mean():+.4f} sd {nulls.std():.4f}  "
          f"p_emp = {p_emp:.4f}")

    # 2. leave-one-generator-family-out
    for fam in sorted(set(FAMILY_MAP.values())):
        drop = [s for s, ff in FAMILY_MAP.items() if ff == fam]
        v = rule_rho(t[~t.train_dataset.isin(drop)], args.target)
        recs.append(dict(control=f"drop_family_{fam}", value=v,
                         note=f"dropped {len(drop)} sources"))
        print(f"drop family {fam:<9} rho = {v:+.4f}")

    # 3. fit-free size / supervision-density controls
    for col, name in [("log_train_n_samples", "size_n_samples"),
                      ("log_train_n_vectors", "size_n_vectors"),
                      ("log_train_valid_vectors_per_sample_capped", "supervision_density")]:
        if col not in t.columns:
            continue
        rs = [spearmanr(c[args.target], c[col]).statistic
              for _, c in t.groupby("cv")
              if c.train_dataset.nunique() >= 3 and c[col].std() > 1e-12]
        rs = [r for r in rs if np.isfinite(r)]
        v = float(np.nanmean(rs)) if rs else float("nan")
        recs.append(dict(control=f"control_{name}", value=v, note="fit-free rank"))
        print(f"size control {name:<22} rho = {v:+.4f}")

    # 4. leave-one-benchmark-out
    lobo = []
    for b in sorted(t.benchmark.unique()):
        v = rule_rho(t[t.benchmark != b], args.target)
        lobo.append(v)
        recs.append(dict(control=f"drop_benchmark_{b}", value=v, note=""))
    print(f"leave-one-benchmark-out range: [{min(lobo):+.4f}, {max(lobo):+.4f}]")

    pd.DataFrame(recs).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
