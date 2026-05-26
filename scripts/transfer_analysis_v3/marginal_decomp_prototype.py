"""Unified prior + interaction prototype for LOTO + LOBO.

Model (one structure, both splits):

    perf(i, k) = level(i, k) + f(features(i -> k))

  * level(i, k) = axis-aware IDW prior (the thing that already makes LOBO work):
        - eval-side: IDW over benchmarks e ~ k of perf(i, e)   [needs i seen -> LOBO]
        - train-side: IDW over trains  j ~ i of perf(j, k)     [needs k seen -> LOTO]
        "observed-or-back-off": use whichever side has data; average when both do.
        This is neighbourhood-LOCAL, so it keeps the eval-neighborhood ranking
        signal that a flat global mean throws away.

  * f(features(i -> k)) = ridge on symmetric flow pair features, fit to the
        in-fold residual (actual - level).  This is the shared, feature-only
        interaction term — it carries the within-context ranking signal that
        neighbour borrowing structurally cannot provide on LOTO (because the
        borrowed value is shared across candidates within a benchmark context).

The level provides absolute calibration + LOBO ranking; f provides the LOTO
within-context ranking.  The same f is fit and applied in both splits.

Run:
    python scripts/transfer_analysis_v3/marginal_decomp_prototype.py 2>/dev/null
"""
from __future__ import annotations

import argparse
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from triangle_prior_prototype import (
    build_lookup, idw_weights, SIMILARITY_METRICS,
    within_context_spearman, context_centered_spearman,
    context_mae, train_mean_auc_error,
)

warnings.filterwarnings("ignore")


@dataclass
class Config:
    name: str
    use_f: bool = True
    f_features: list[str] = field(default_factory=lambda: [
        "flow_mmd", "flow_fid", "flow_sliced_w2"])
    metric: str = "mean_nn_sym"          # IDW prior neighbourhood metric
    f_residual: str = "matched"          # 'matched' (held side) | 'pooled' (both sides)


def variant_key(row) -> str:
    return f"{row.get('model_family','')}|{row.get('pretrained','')}|{row.get('freeze','')}"


def idw_pred(dists: list[float], perfs: list[float], is_sim: bool) -> float | None:
    if not dists:
        return None
    w = idw_weights(np.asarray(dists, float), is_sim, "idw")
    p = np.asarray(perfs, float)
    tot = w.sum()
    return float((w * p).sum() / tot) if tot > 0 else float(p.mean())


class Meta:
    """Context metadata + distance lookups (from the FULL table — no perf leak)."""

    def __init__(self, table: pd.DataFrame, dist_df: pd.DataFrame, space: str, metric: str):
        self.ctx_bench, self.ctx_variant = {}, {}
        for cid, g in table.groupby("context_id"):
            self.ctx_bench[cid] = g["benchmark"].iloc[0]
            self.ctx_variant[cid] = g["variant"].iloc[0]
        self.var_bench_to_ctx = {(self.ctx_variant[c], self.ctx_bench[c]): c
                                 for c in self.ctx_bench}
        self.is_sim = metric in SIMILARITY_METRICS
        self.tt = build_lookup(dist_df, space, "train_train", metric)
        self.ee = build_lookup(dist_df, space, "eval_eval", metric)


def train_side(i, ctx, perf, fold_trains, meta) -> float | None:
    ds, ps = [], []
    for n in fold_trains:
        if n == i:
            continue
        p = perf.get((n, ctx))
        d = meta.tt.get((i, n))
        if p is not None and np.isfinite(p) and d is not None and np.isfinite(d):
            ds.append(d); ps.append(p)
    return idw_pred(ds, ps, meta.is_sim)


def eval_side(i, ctx, perf, fold_benchmarks, meta) -> float | None:
    k = meta.ctx_bench[ctx]; v = meta.ctx_variant[ctx]
    ds, ps = [], []
    for e in fold_benchmarks:
        if e == k:
            continue
        sib = meta.var_bench_to_ctx.get((v, e))
        if sib is None:
            continue
        p = perf.get((i, sib))
        d = meta.ee.get((k, e))
        if p is not None and np.isfinite(p) and d is not None and np.isfinite(d):
            ds.append(d); ps.append(p)
    return idw_pred(ds, ps, meta.is_sim)


def level(i, ctx, perf, fold_trains, fold_benchmarks, meta, side: str,
          eval_mean, train_mean, global_mean) -> float:
    """side: 'train' (LOTO held), 'eval' (LOBO held), 'both' (in-fold)."""
    t = train_side(i, ctx, perf, fold_trains, meta) if side in ("train", "both") else None
    e = eval_side(i, ctx, perf, fold_benchmarks, meta) if side in ("eval", "both") else None
    if t is not None and e is not None:
        return 0.5 * (t + e)
    if t is not None:
        return t
    if e is not None:
        return e
    if side == "eval":
        return train_mean.get(i, global_mean)
    return eval_mean.get(meta.ctx_bench[ctx], global_mean)


def run_split(table: pd.DataFrame, cfg: Config, meta: Meta, hold: str) -> pd.DataFrame:
    held_side = "train" if hold == "train_dataset" else "eval"
    f_cols = [c for c in cfg.f_features if c in table.columns]
    out = []
    for held_val in sorted(table[hold].unique()):
        infold = table[table[hold] != held_val]
        held = table[table[hold] == held_val]
        if infold.empty or held.empty:
            continue

        perf = {(r.train_dataset, r.context_id): float(r.auc_normalized)
                for r in infold.itertuples()}
        fold_trains = sorted(infold["train_dataset"].unique())
        fold_benchmarks = sorted(infold["benchmark"].unique())
        gmean = float(np.mean(list(perf.values())))
        eval_mean, train_mean = defaultdict(list), defaultdict(list)
        for (td, c), v in perf.items():
            eval_mean[meta.ctx_bench[c]].append(v); train_mean[td].append(v)
        eval_mean = {k: float(np.mean(v)) for k, v in eval_mean.items()}
        train_mean = {k: float(np.mean(v)) for k, v in train_mean.items()}

        # Fit f on in-fold residuals (LOO level via skip-self inside the sides).
        f_model = None
        if cfg.use_f and f_cols:
            # Fit f against the SAME one-sided prior the split uses at test time,
            # so f learns the candidate-specific (i->k) deviation that the
            # held-side prior structurally misses (the LOTO ranking signal).
            # 'pooled' trains one split-agnostic f on BOTH one-sided residuals.
            sides = [held_side] if cfg.f_residual == "matched" else ["train", "eval"]
            resid, Xr = [], []
            for r in infold.itertuples():
                feat = [getattr(r, c) for c in f_cols]
                for s in sides:
                    lvl = level(r.train_dataset, r.context_id, perf, fold_trains,
                                fold_benchmarks, meta, s, eval_mean, train_mean, gmean)
                    resid.append(float(r.auc_normalized) - lvl)
                    Xr.append(feat)
            X = np.asarray(Xr, float)
            imp = SimpleImputer(strategy="median").fit(X)
            scl = StandardScaler().fit(np.nan_to_num(imp.transform(X)))
            Xs = np.nan_to_num(scl.transform(np.nan_to_num(imp.transform(X))))
            ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0]).fit(Xs, np.asarray(resid))
            f_model = (imp, scl, ridge)

        for r in held.itertuples():
            lvl = level(r.train_dataset, r.context_id, perf, fold_trains,
                        fold_benchmarks, meta, held_side, eval_mean, train_mean, gmean)
            fval = 0.0
            if f_model is not None:
                imp, scl, ridge = f_model
                x = np.asarray([[getattr(r, c) for c in f_cols]], float)
                xs = np.nan_to_num(scl.transform(np.nan_to_num(imp.transform(x))))
                fval = float(ridge.predict(xs)[0])
            out.append((r.train_dataset, r.context_id, r.benchmark,
                        float(r.auc_normalized), lvl + fval))
    return pd.DataFrame(out, columns=["train_dataset", "context_id", "benchmark",
                                      "actual", "pred"])


def score(df: pd.DataFrame) -> dict:
    return {"MAE": context_mae(df), "ctx_rho": within_context_spearman(df),
            "cent_rho": context_centered_spearman(df), "tr_err": train_mean_auc_error(df)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="scripts/transfer_analysis_v3/transfer_table.csv")
    ap.add_argument("--dist", default="analysis_v3/pairwise_self_distances.csv")
    ap.add_argument("--space", default="flow")
    ap.add_argument("--metric", default="mean_nn_sym")
    args = ap.parse_args()

    root = Path(".").resolve()
    table = pd.read_csv(root / args.table).dropna(subset=["auc_normalized"]).copy()
    table["variant"] = table.apply(variant_key, axis=1)
    dist_df = pd.read_csv(root / args.dist)
    meta = Meta(table, dist_df, args.space, args.metric)

    sym = ["flow_mmd", "flow_fid", "flow_sliced_w2"]
    configs = [
        Config("prior only (no f)", use_f=False, metric=args.metric),
        Config("+ f matched (sym)", f_features=sym, metric=args.metric, f_residual="matched"),
        Config("+ f matched (FID)", f_features=["flow_fid"], metric=args.metric, f_residual="matched"),
        Config("+ f pooled (sym)", f_features=sym, metric=args.metric, f_residual="pooled"),
        Config("+ f pooled (FID)", f_features=["flow_fid"], metric=args.metric, f_residual="pooled"),
    ]

    print(f"space={args.space}  prior metric={args.metric}  rows={len(table)}")
    print("\nReference (current pipeline, flow-only diagnostic):")
    print("  LOTO idw_prior_two_way: MAE 9.16  ctx_rho 0.036")
    print("  LOBO idw_prior_two_way: MAE 8.68  ctx_rho 0.765\n")

    for split, hold in [("LOTO", "train_dataset"), ("LOBO", "benchmark")]:
        print(f"================ {split} ================")
        print(f"{'config':<24}{'MAE':>8}{'ctx_rho':>10}{'cent_rho':>10}{'tr_err':>9}")
        for cfg in configs:
            m = score(run_split(table, cfg, meta, hold))
            print(f"{cfg.name:<24}{m['MAE']:>8.3f}{m['ctx_rho']:>10.3f}"
                  f"{m['cent_rho']:>10.3f}{m['tr_err']:>9.3f}")
        print()


if __name__ == "__main__":
    main()
