"""One small linear model per training regime — an actual predictor.

There are exactly two regimes, so we fit exactly two models. For each regime
(from-scratch, pretrained) separately:

    transfer  ~=  level anchor  +  w1 * d(T->B)  +  w2 * d(B->T)

where d(T->B) and d(B->T) are the two directed mean nearest-neighbor motion
distances, demeaned within each context (a context = benchmark x architecture
x training variant) and z-scored. The regime-direction law predicts the
coefficient pattern: the from-scratch model should load on d(T->B), the
pretrained model on d(B->T). A pooled (regime-blind) model is fit as the
contrast — pooling the two regimes averages opposite signs and is exactly the
"dilution" that plagued the earlier shared fits.

Evaluation uses the paper's three held-out settings, spelled out:
  - held-out training set:  fit on 10 training sets, predict the 11th
  - held-out benchmark:     fit on 9 benchmarks, predict the 10th
  - both held out (JOINT):  fit excludes the held training set AND benchmark

Reported per regime x setting:
  - ranking: mean within-context Spearman rho of the model's score vs actual
    transfer (compare: the one-direction fit-free rule, and the symmetric
    average of the two directions)
  - absolute: MAE and Pearson r of (level anchor + model score) vs actual
    peak PCK, using the same level anchors as the rest of the pipeline
  - the fitted coefficients (mean +/- sd across folds, standardized features)

Artifacts: results/per_regime_linear_summary.csv (one row per regime x setting)

    python scripts/transfer_analysis_v5/per_regime_linear.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/home/spencer/Projects/OnlineSyntheticCorrespondence")
V4 = ROOT / "scripts/transfer_analysis_v4"
RES = ROOT / "scripts/transfer_analysis_v5/results"

PURE = ["flyingthings", "imagenet2dwarp", "movi_f", "pointodyssey", "sintel",
        "spair", "synthetic", "synthetic_2d_warp", "synthetic_large_zoom",
        "synthetic_random_flipping", "synthetic_small_zoom"]
SCRATCH = {"catspp|False|False", "catspp|False|True", "glunet|False|False",
           "glunet|False|True", "raft|True|False"}  # RAFT: config-verified scratch
AB, BA = "mean_nn_a_to_b", "mean_nn_b_to_a"
EPS = ["a_covered_by_b_eps1px", "b_covered_by_a_eps1px",
       "a_covered_by_b_eps4px", "b_covered_by_a_eps4px",
       "a_covered_by_b_eps16px", "b_covered_by_a_eps16px"]
ALL_FEATS = [AB, BA] + EPS

DINO_AB, DINO_BA = "dino_mean_nn_a_to_b", "dino_mean_nn_b_to_a"

# candidate feature sets for the per-regime linear model; the DINO set is the
# appearance control — same model, same protocol, appearance features
FEATURE_SETS = {
    "motion mean-NN both dirs (2)": [AB, BA],
    "motion eps 4px both dirs (2)": ["a_covered_by_b_eps4px",
                                     "b_covered_by_a_eps4px"],
    "motion eps 1/4/16px both dirs (6)": EPS,
    "motion mean-NN + eps (8)": ALL_FEATS,
    "appearance (DINO) mean-NN both dirs (2)": [DINO_AB, DINO_BA],
}


def load():
    t = pd.read_csv(ROOT / "scripts/transfer_analysis_v3/transfer_table.csv")
    t = t[t.train_dataset.isin(PURE)].copy()
    t["variant"] = (t.model_family.astype(str) + "|" + t.pretrained.astype(str)
                    + "|" + t.freeze.astype(str))
    t = t[t.variant != "raft|False|False"]
    t["cv"] = t.benchmark + "|" + t.variant
    t = t.dropna(subset=["peak_pck"])
    t["regime"] = np.where(t.variant.isin(SCRATCH), "scratch", "pretrained")

    d = pd.read_csv(ROOT / "analysis_v3/pairwise_self_distances.csv")
    te = d[(d.pair_type == "train_eval") & (d.space == "flow")]
    f = te.set_index(["dataset_a", "dataset_b"])[ALL_FEATS]
    t = t.join(f, on=["train_dataset", "benchmark"], how="left")
    dino = d[(d.pair_type == "train_eval") & (d.space == "dino")]
    fd = (dino.set_index(["dataset_a", "dataset_b"])[[AB, BA]]
              .rename(columns={AB: DINO_AB, BA: DINO_BA}))
    t = t.join(fd, on=["train_dataset", "benchmark"], how="left")
    t = t.dropna(subset=ALL_FEATS + [DINO_AB, DINO_BA])

    # demean every feature within each context (feature means are
    # target-free, so this uses all sources of the context)
    for c in ALL_FEATS + [DINO_AB, DINO_BA]:
        t[c + "_dm"] = t[c] - t.groupby("cv")[c].transform("mean")

    # level anchors from the existing pipeline (per held-out setting)
    L = {}
    for split in ["LOTO", "LOBO", "JOINT"]:
        r = pd.read_csv(V4 / f"results_rule_v5core/predictions/peak_pck/"
                             f"rows_{split}_motion_rule.csv")
        L[split] = r.set_index(["train_dataset", "benchmark", "variant"])["L"]
    return t, L


def fit_predict(df, train_mask, test_mask, feats):
    """OLS of within-context-demeaned target on the demeaned, z-scored
    features; returns predictions for test rows + coefficients."""
    tr, ts = df[train_mask], df[test_mask]
    mu = tr.groupby("cv")["peak_pck"].transform("mean")
    y = (tr.peak_pck - mu).values
    X, Xt = [], []
    for c in [f + "_dm" for f in feats]:
        s = tr[c].std() or 1.0
        X.append(tr[c].values / s)
        Xt.append(ts[c].values / s)
    X = np.column_stack(X)
    Xt = np.column_stack(Xt)
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    return Xt @ w, w


def folds(df, split):
    if split == "LOTO":
        for src in PURE:
            yield df.train_dataset != src, df.train_dataset == src
    elif split == "LOBO":
        for b in df.benchmark.unique():
            yield df.benchmark != b, df.benchmark == b
    else:  # JOINT
        for src in PURE:
            for b in df.benchmark.unique():
                yield ((df.train_dataset != src) & (df.benchmark != b),
                       (df.train_dataset == src) & (df.benchmark == b))


def ctx_rho(df, col):
    out = []
    for _, c in df.groupby("cv"):
        if c.train_dataset.nunique() < 3 or c[col].std() <= 1e-15:
            continue
        r = spearmanr(c.peak_pck, c[col]).statistic
        if np.isfinite(r):
            out.append(r)
    return float(np.mean(out)) if out else np.nan


def run_featureset(t, L, feats):
    rows = []
    for split in ["LOTO", "LOBO", "JOINT"]:
        for regime in ["scratch", "pretrained", "pooled (regime-blind)"]:
            df = (t if regime.startswith("pooled")
                  else t[t.regime == regime]).copy()
            df["pred_g"] = np.nan
            ws = []
            for trm, tsm in folds(df, split):
                if tsm.sum() == 0:
                    continue
                p, w = fit_predict(df, trm, tsm, feats)
                df.loc[tsm, "pred_g"] = p
                ws.append(w)
            ws = np.array(ws)

            # baselines on the same rows: matched-direction rule + symmetric
            dir_col = AB if regime == "scratch" else BA
            df["rule_score"] = (-df[AB + "_dm"] if regime == "scratch"
                                else -df[BA + "_dm"])
            if regime.startswith("pooled"):
                df["rule_score"] = np.where(df.regime == "scratch",
                                            -df[AB + "_dm"], -df[BA + "_dm"])
            df["sym_score"] = -(df[AB + "_dm"] + df[BA + "_dm"]) / 2

            # absolute: pipeline level anchor + the model's score
            lv = df.join(L[split], on=["train_dataset", "benchmark", "variant"],
                         rsuffix="_anchor")
            pred_abs = lv["L"] + df.pred_g
            ok = np.isfinite(pred_abs)
            row = {
                "regime": regime, "setting": split,
                "ctx_rho_linear": ctx_rho(df, "pred_g"),
                "ctx_rho_rule": ctx_rho(df, "rule_score"),
                "ctx_rho_sym": ctx_rho(df, "sym_score"),
                "MAE(anchor+model)": float(np.abs(pred_abs[ok]
                                                  - df.peak_pck[ok]).mean()),
                "r(anchor+model)": float(pearsonr(pred_abs[ok],
                                                  df.peak_pck[ok])[0]),
            }
            if feats == [AB, BA]:  # headline model: report its 2 coefficients
                row.update({"w[d(T->B)]": ws[:, 0].mean(),
                            "w_ab_sd": ws[:, 0].std(),
                            "w[d(B->T)]": ws[:, 1].mean(),
                            "w_ba_sd": ws[:, 1].std()})
            rows.append(row)
            del dir_col
    return pd.DataFrame(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-benchmarks", default="",
                    help="comma-separated benchmarks to exclude (sensitivity "
                         "run; writes _drop_<names> suffixed artifacts)")
    args = ap.parse_args()
    drop = [b for b in args.drop_benchmarks.split(",") if b]
    suffix = ("_drop_" + "_".join(drop)) if drop else ""

    t, L = load()
    if drop:
        t = t[~t.benchmark.isin(drop)].copy()

    out = run_featureset(t, L, [AB, BA])
    out.to_csv(RES / f"per_regime_linear_summary{suffix}.csv", index=False)
    print(out.round(3).to_string(index=False))

    # feature-set comparison: does adding the eps-radius coverages (both
    # directions, 3 radii) beat the two mean-NN distances?
    comp = []
    for name, feats in FEATURE_SETS.items():
        r = run_featureset(t, L, feats)
        r.insert(0, "features", name)
        comp.append(r)
    comp = pd.concat(comp)[["features", "regime", "setting", "ctx_rho_linear",
                            "MAE(anchor+model)", "r(anchor+model)"]]
    comp.to_csv(RES / f"per_regime_featureset_comparison{suffix}.csv",
                index=False)
    print()
    print(comp.round(3).to_string(index=False))
    print(f"\nwrote per_regime_linear_summary{suffix}.csv + "
          f"per_regime_featureset_comparison{suffix}.csv")


if __name__ == "__main__":
    main()
