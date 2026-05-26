"""Final transfer-predictor harness: shared interaction head + regime level.

Model:
    predicted(i, k) = L_regime(i, k)   +   g(features(i -> k))
                      └ nuisance level ┘     └ shared claim ┘

  * g  — the SHARED feature-interaction head (the scientific object).  Fit on
         within-context-demeaned features -> demeaned target (a fixed-effects /
         within estimator), so it learns ONLY the within-context relationship.
         This is where motion-vs-appearance is decided.
  * L  — regime-specific level (a nuisance), used ONLY for absolute prediction:
         - LOTO (benchmark seen): context_mean[(k, v)]
         - LOBO (benchmark unseen): eval-side IDW over similar benchmarks.

Two outputs per row:
  * rank_score = g(.)            -> within-context ranking (artifact-free; the
                                    leave-one-out level never enters the order).
  * abs_pred   = L + g(.)        -> calibrated value for predicted-vs-actual
                                    scatter / MAE.

Sweeps feature family (motion / appearance / both) x head (ridge / gbm) over
LOTO and LOBO.  Writes per-row predictions for plotting.

Run:
    python scripts/transfer_analysis_v3/transfer_predictor_prototype.py 2>/dev/null
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from triangle_prior_prototype import (
    build_lookup, idw_weights, SIMILARITY_METRICS,
    within_context_spearman, context_centered_spearman,
    context_mae, train_mean_auc_error,
)

warnings.filterwarnings("ignore")


def variant_key(row) -> str:
    return f"{row.get('model_family','')}|{row.get('pretrained','')}|{row.get('freeze','')}"


# Matched (i->k) metrics available in train_eval rows of BOTH flow and dino
# self-distances — apples-to-apples motion-vs-appearance feature set.
SELFDIST_METRICS = [
    "mean_nn_a_to_b", "mean_nn_b_to_a", "mean_nn_sym",
    "a_covered_by_b_eps1px", "b_covered_by_a_eps1px",
    "a_covered_by_b_eps4px", "b_covered_by_a_eps4px",
    "a_covered_by_b_eps16px", "b_covered_by_a_eps16px",
    "kl_a_to_b_k5", "kl_b_to_a_k5", "kl_a_to_b_k20", "kl_b_to_a_k20",
]


def add_selfdist_features(table: pd.DataFrame, dist_df: pd.DataFrame,
                          spaces=("flow", "dino")) -> pd.DataFrame:
    """Join train_eval self-distances onto each (train, benchmark) row as
    se_{space}_{metric} columns — complete, matched (i->k) features per space."""
    table = table.copy()
    for space in spaces:
        te = dist_df[(dist_df.space == space) & (dist_df.pair_type == "train_eval")]
        lut = {}  # (train, benchmark) -> {metric: value} (resolve both orientations)
        for r in te.itertuples():
            lut[(r.dataset_a, r.dataset_b)] = r
            lut[(r.dataset_b, r.dataset_a)] = r
        for m in SELFDIST_METRICS:
            col = f"se_{space}_{m}"
            table[col] = [getattr(lut.get((t, b)), m, np.nan)
                          for t, b in zip(table.train_dataset, table.benchmark)]
    return table


def feature_set(table: pd.DataFrame, family: str, source: str = "table") -> list[str]:
    if source == "self_dist":
        pre = {"motion": "se_flow_", "appearance": "se_dino_"}
        if family in pre:
            return [c for c in table.columns if c.startswith(pre[family])
                    and table[c].notna().mean() > 0.5]
        return [c for c in table.columns
                if (c.startswith("se_flow_") or c.startswith("se_dino_"))
                and table[c].notna().mean() > 0.5]
    drop = {"flow_train_isolation", "flow_eval_isolation",
            "dino_train_isolation", "dino_eval_isolation"}
    flow = [c for c in table.columns if c.startswith("flow_")
            and c not in drop and table[c].notna().mean() > 0.5]
    dino = [c for c in table.columns if c.startswith("dino_")
            and c not in drop and table[c].notna().mean() > 0.5]
    if family == "motion":
        return flow
    if family == "appearance":
        return dino
    return flow + dino


def make_head(kind: str):
    if kind == "ridge":
        return RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    if kind == "gbm":
        return GradientBoostingRegressor(n_estimators=120, max_depth=2,
                                         learning_rate=0.05, subsample=0.8,
                                         random_state=0)
    raise ValueError(kind)


def fit_plackett_luce(X: np.ndarray, groups: np.ndarray, y: np.ndarray,
                      l2: float = 1.0, epochs: int = 400, lr: float = 0.05):
    """Listwise Plackett-Luce: linear scores fit to reproduce the within-context
    ordering of actual performance.  No level / calibration — pure ordering.
    Returns a weight vector w; score = X @ w (shift-invariant within context)."""
    import torch
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = np.asarray(y, float)
    orders = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < 2:
            continue
        orders.append(torch.tensor(idx[np.argsort(-yt[idx])], dtype=torch.long))
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        s = Xt @ w
        nll = s.new_zeros(())
        for order in orders:
            so = s[order]
            # logsumexp over each suffix so[r:]  ->  PL log-likelihood
            lse = torch.logcumsumexp(so.flip(0), 0).flip(0)
            nll = nll + (lse - so).sum()
        (nll + l2 * (w * w).sum()).backward()
        opt.step()
    return w.detach().numpy()


def run(table: pd.DataFrame, family: str, head_kind: str, hold: str,
        ee: dict, ee_is_sim: bool, source: str = "table") -> pd.DataFrame:
    """hold = 'train_dataset' (LOTO) or 'benchmark' (LOBO)."""
    f_cols = feature_set(table, family, source)
    out = []
    for held_val in sorted(table[hold].unique()):
        infold = table[table[hold] != held_val]
        held = table[table[hold] == held_val]
        if infold.empty or held.empty:
            continue

        # Regime level pieces (from in-fold only).
        cm = infold.groupby("cv")["auc_normalized"].mean().to_dict()
        gmean = float(infold["auc_normalized"].mean())
        # perf[(train, benchmark, variant)] and per-variant benchmark list for eval-side IDW
        perf = {(r.train_dataset, r.benchmark, r.variant): float(r.auc_normalized)
                for r in infold.itertuples()}
        fold_benchmarks = sorted(infold["benchmark"].unique())

        # In-fold context feature means (for within-context demeaning).
        xmean = {cv: infold.loc[infold.cv == cv, f_cols].mean() for cv in infold.cv.unique()}
        # Held-out contexts (LOBO) have no in-fold rows: center using their own features.
        for cv in held.cv.unique():
            if cv not in xmean:
                xmean[cv] = held.loc[held.cv == cv, f_cols].mean()

        # Plackett-Luce: pure listwise ordering, no demeaning, no level.
        if head_kind == "pl":
            Xraw = infold[f_cols].values.astype(float)
            imp = SimpleImputer(strategy="median").fit(Xraw)
            scl = StandardScaler().fit(np.nan_to_num(imp.transform(Xraw)))
            Xs = np.nan_to_num(scl.transform(np.nan_to_num(imp.transform(Xraw))))
            w = fit_plackett_luce(Xs, infold["cv"].values,
                                  infold["auc_normalized"].values)
            for r in held.itertuples():
                x = np.asarray([[getattr(r, c) for c in f_cols]], float)
                xs = np.nan_to_num(scl.transform(np.nan_to_num(imp.transform(x))))
                s = float(xs @ w)
                out.append((r.train_dataset, r.cv, r.benchmark,
                            float(r.auc_normalized), s, np.nan))  # no abs_pred
            continue

        # Build demeaned training matrix (regressor heads).
        Xtr, ytr = [], []
        for r in infold.itertuples():
            xm = xmean[r.cv]
            Xtr.append([getattr(r, c) - xm[c] for c in f_cols])
            ytr.append(float(r.auc_normalized) - cm[r.cv])
        Xtr = np.asarray(Xtr, float)
        imp = SimpleImputer(strategy="median").fit(Xtr)
        scl = StandardScaler().fit(np.nan_to_num(imp.transform(Xtr)))
        Xs = np.nan_to_num(scl.transform(np.nan_to_num(imp.transform(Xtr))))
        head = make_head(head_kind).fit(Xs, np.asarray(ytr))

        def level(i, k, v, cv):
            if cv in cm:                       # LOTO: benchmark seen
                return cm[cv]
            ds, ps = [], []                    # LOBO: eval-side IDW over benchmarks
            for e in fold_benchmarks:
                if e == k:
                    continue
                p = perf.get((i, e, v))
                d = ee.get((k, e))
                if p is not None and np.isfinite(p) and d is not None and np.isfinite(d):
                    ds.append(d); ps.append(p)
            if ds:
                w = idw_weights(np.asarray(ds, float), ee_is_sim, "idw")
                return float((w * np.asarray(ps)).sum() / w.sum())
            return gmean

        for r in held.itertuples():
            xm = xmean[r.cv]
            x = np.asarray([[getattr(r, c) - xm[c] for c in f_cols]], float)
            xs = np.nan_to_num(scl.transform(np.nan_to_num(imp.transform(x))))
            dev = float(head.predict(xs)[0])
            L = level(r.train_dataset, r.benchmark, r.variant, r.cv)
            out.append((r.train_dataset, r.cv, r.benchmark, float(r.auc_normalized),
                        dev, L + dev))
    return pd.DataFrame(out, columns=["train_dataset", "context_id", "benchmark",
                                      "actual", "rank_score", "abs_pred"])


def report(df: pd.DataFrame) -> dict:
    rank = df.rename(columns={"rank_score": "pred"})[
        ["train_dataset", "context_id", "benchmark", "actual", "pred"]]
    m = {"ctx_rho": within_context_spearman(rank),       # ranking head, artifact-free
         "cent_rho": context_centered_spearman(rank)}
    if df["abs_pred"].notna().any():                     # PL has no absolute prediction
        absol = df.rename(columns={"abs_pred": "pred"})[
            ["train_dataset", "context_id", "benchmark", "actual", "pred"]]
        m["MAE"] = context_mae(absol)
        m["abs_r"] = float(pearsonr(absol["pred"], absol["actual"])[0])
        m["tr_err"] = train_mean_auc_error(absol)
    else:
        m["MAE"] = m["abs_r"] = m["tr_err"] = float("nan")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="scripts/transfer_analysis_v3/transfer_table.csv")
    ap.add_argument("--dist", default="analysis_v3/pairwise_self_distances.csv")
    ap.add_argument("--space", default="flow")
    ap.add_argument("--metric", default="mean_nn_sym")
    ap.add_argument("--families", nargs="+", default=["motion", "appearance"])
    ap.add_argument("--heads", nargs="+", default=["ridge", "gbm", "pl"])
    ap.add_argument("--feature-source", default="self_dist", choices=["table", "self_dist"],
                    help="'self_dist': matched (i->k) features from train_eval self-distances "
                         "(complete, apples-to-apples). 'table': transfer_table flow_/dino_ cols.")
    ap.add_argument("--out", default="scripts/transfer_analysis_v3/results/predictor_prototype")
    args = ap.parse_args()

    root = Path(".").resolve()
    table = pd.read_csv(root / args.table).dropna(subset=["auc_normalized"]).copy()
    table["variant"] = table.apply(variant_key, axis=1)
    table["cv"] = table["benchmark"] + "|" + table["variant"]
    dist_df = pd.read_csv(root / args.dist)
    ee = build_lookup(dist_df, args.space, "eval_eval", args.metric)
    ee_is_sim = args.metric in SIMILARITY_METRICS

    if args.feature_source == "self_dist":
        table = add_selfdist_features(table, dist_df)

    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"families={args.families}  heads={args.heads}  rows={len(table)}  "
          f"feature_source={args.feature_source}")
    for fam in args.families:
        print(f"  {fam}: {len(feature_set(table, fam, args.feature_source))} features")
    print("ranking head = artifact-free within-context order; abs = L + g (scatter/MAE)\n")
    for split, hold in [("LOTO", "train_dataset"), ("LOBO", "benchmark")]:
        print(f"================ {split} ================")
        print(f"{'family':<12}{'head':<8}{'ctx_rho':>9}{'cent_rho':>10}"
              f"{'MAE':>8}{'abs_r':>8}{'tr_err':>8}")
        for fam in args.families:
            for hk in args.heads:
                df = run(table, fam, hk, hold, ee, ee_is_sim, args.feature_source)
                df.to_csv(out_dir / f"pred_{split}_{fam}_{hk}.csv", index=False)
                m = report(df)
                print(f"{fam:<12}{hk:<8}{m['ctx_rho']:>9.3f}{m['cent_rho']:>10.3f}"
                      f"{m['MAE']:>8.2f}{m['abs_r']:>8.3f}{m['tr_err']:>8.2f}")
        print()
    print(f"per-row predictions written to {out_dir}/  (for scatter plots)")


if __name__ == "__main__":
    main()
