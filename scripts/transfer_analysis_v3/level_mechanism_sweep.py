"""Sweep the LEVEL mechanism (options A/B/C/D) on top of the shared within-estimator g.

Decomposition (one structure, both splits):

    perf(i, k) = L(i, k)  +  g(features(i -> k))
                 └ level ┘     └ the claim ┘

  g  — within estimator (fixed-effects): fit on WITHIN-CONTEXT-demeaned features
       -> demeaned target. Identical across every level mechanism. This is the
       scientific object (motion vs appearance), and it is what we RANK on.

  L  — the held-out entity's level. Four options:
       A  observed / borrow   : LOTO = in-fold context mean (benchmark seen);
                                LOBO = eval-side IDW over neighbor benchmarks.
                                (your current harness)
       B  feature-regressed    : three-way effects grand + alpha_i + beta_k +
                                gamma_v (variant is never held out, so its effect
                                is always observed and kept). LOTO: observed
                                (benchmark,variant) CELL band + feature-regressed
                                source effect. LOBO: observed source + variant
                                effects + feature-regressed benchmark effect
                                (RLFM-style cold-start).
       C  empirical-Bayes      : same effects, James-Stein/BLUP shrunk toward the
                                grand mean; an UNSEEN entity shrinks to the
                                population mean. LOTO C reduces to the observed
                                cell band (= A); LOBO keeps shrunk source+variant.
                                Avoids the leave-self-out anti-correlation.
       D  none (g-only)        : no level in the ranking score. This is the
                                estimand we report; A/B/C only add calibration.

Reports, per (split, family):
  [CLAIM]  rank by g only: ctx_rho / cent_rho, plus a label-shuffled control
           (must collapse to ~0 -> shows the ranking is not leaking).
  [LEVEL]  per mechanism: absolute MAE / abs_r from (L+g); the change in ranking
           rho from FOLDING L into the score (negative = level hurts the order);
           and the level-only ranking rho (g=0).

Leakage guards are explicit asserts inside the fold loop. Every level estimate
uses in-fold OUTCOMES only; held-entity features (i->k distances, not outcomes)
are allowed because they are inputs, not labels.

Run:
    python scripts/transfer_analysis_v3/level_mechanism_sweep.py 2>/dev/null
    python scripts/transfer_analysis_v3/level_mechanism_sweep.py --families motion appearance both
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from triangle_prior_prototype import (
    build_lookup, idw_weights, SIMILARITY_METRICS,
    within_context_spearman, context_centered_spearman, context_mae,
)
from transfer_predictor_prototype import (
    variant_key, add_selfdist_features, feature_set, make_head,
)

warnings.filterwarnings("ignore")

MECHANISMS = ["A", "B", "C"]   # D = g-only, reported separately as the claim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fit_within_g(infold: pd.DataFrame, held: pd.DataFrame, f_cols: list[str],
                  head_kind: str):
    """Fit the shared within-estimator g and return (predict_fn, cm, xmean, gmean).

    predict_fn(row) -> g deviation for a held row.  cm = in-fold context means,
    xmean = in-fold (or held-own) context feature means, gmean = in-fold grand.
    """
    cm = infold.groupby("cv")["auc_normalized"].mean().to_dict()
    gmean = float(infold["auc_normalized"].mean())
    xmean = {cv: infold.loc[infold.cv == cv, f_cols].mean() for cv in infold.cv.unique()}
    for cv in held.cv.unique():                       # held context: center on own feats
        if cv not in xmean:
            xmean[cv] = held.loc[held.cv == cv, f_cols].mean()

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

    def predict(r):
        xm = xmean[r.cv]
        x = np.asarray([[getattr(r, c) - xm[c] for c in f_cols]], float)
        xs = np.nan_to_num(scl.transform(np.nan_to_num(imp.transform(x))))
        return float(head.predict(xs)[0])

    return predict, cm, xmean, gmean


def _anova_effects(infold: pd.DataFrame):
    """Grand mean + main effects alpha_i (source), beta_k (benchmark), gamma_v
    (variant), plus the observed (benchmark,variant) cell mean."""
    grand = float(infold["auc_normalized"].mean())
    alpha = (infold.groupby("train_dataset")["auc_normalized"].mean() - grand).to_dict()
    beta = (infold.groupby("benchmark")["auc_normalized"].mean() - grand).to_dict()
    gamma = (infold.groupby("variant")["auc_normalized"].mean() - grand).to_dict()
    cell_mean = infold.groupby("cv")["auc_normalized"].mean().to_dict()
    return grand, alpha, beta, gamma, cell_mean


def _eb_shrunk(infold: pd.DataFrame, by: str, grand: float) -> dict:
    """James-Stein / BLUP shrunk effect per entity: factor = n*tau2 / (n*tau2 + sig2).
    Unseen entities are absent from the dict -> caller treats as 0 (grand mean)."""
    g = infold.groupby(by)["auc_normalized"]
    means, ns = g.mean(), g.size()
    raw = means - grand
    within = g.var(ddof=1)
    sig2 = float(np.nanmean(within.values)) if np.isfinite(np.nanmean(within.values)) else 1e-6
    sig2 = max(sig2, 1e-6)
    tau2 = max(float(raw.var(ddof=1)) - sig2 / float(ns.mean()), 1e-6)
    factor = (ns * tau2) / (ns * tau2 + sig2)
    return {e: float(factor[e] * raw[e]) for e in raw.index}


def _marginal_features(df: pd.DataFrame, by: str, f_cols: list[str]) -> dict:
    """Entity -> mean (i->k) feature vector over its partner axis (features only)."""
    return {e: grp[f_cols].mean() for e, grp in df.groupby(by)}


def _fit_marginal_regressor(effects: dict, marg: dict, f_cols: list[str]):
    """Ridge: entity marginal features -> in-fold entity effect (RLFM cold-start)."""
    ents = [e for e in effects if e in marg]
    if len(ents) < 3:
        return None
    X = np.asarray([marg[e].values for e in ents], float)
    y = np.asarray([effects[e] for e in ents], float)
    imp = SimpleImputer(strategy="median").fit(X)
    scl = StandardScaler().fit(np.nan_to_num(imp.transform(X)))
    Xs = np.nan_to_num(scl.transform(np.nan_to_num(imp.transform(X))))
    reg = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]).fit(Xs, y)

    def predict(feat_series):
        x = np.asarray([feat_series.values], float)
        xs = np.nan_to_num(scl.transform(np.nan_to_num(imp.transform(x))))
        return float(reg.predict(xs)[0])

    return predict


# ---------------------------------------------------------------------------
# One CV pass: emit per-row g and L_A / L_B / L_C (leakage-guarded)
# ---------------------------------------------------------------------------
def _build_ee(dist_df: pd.DataFrame, space: str, metric: str):
    """Return (ee lookup, is_similarity) for benchmark-benchmark IDW."""
    return build_lookup(dist_df, space, "eval_eval", metric), metric in SIMILARITY_METRICS


def add_random_features(table: pd.DataFrame, n_feats: int = 13,
                        seed: int = 42) -> pd.DataFrame:
    """Inject n_feats random gaussian columns as a dim-matched 'random' family.
    Used as the honest g-only control floor (no level memorization)."""
    rng = np.random.default_rng(seed)
    for j in range(n_feats):
        table[f"rnd_{j}"] = rng.standard_normal(len(table))
    return table


def feature_set_ext(table: pd.DataFrame, family: str, source: str) -> list[str]:
    """feature_set, extended to recognize 'random' (rnd_* columns)."""
    if family == "random":
        return [c for c in table.columns if c.startswith("rnd_")]
    return feature_set(table, family, source)


def cv_predict(table: pd.DataFrame, f_cols: list[str], hold: str,
               ee: dict, ee_is_sim: bool, head_kind: str = "ridge") -> pd.DataFrame:
    held_axis_is_source = (hold == "train_dataset")
    out = []
    for held_val in sorted(table[hold].unique()):
        infold = table[table[hold] != held_val]
        held = table[table[hold] == held_val]
        if infold.empty or held.empty:
            continue
        # ---- leakage guards ------------------------------------------------
        assert held_val not in set(infold[hold]), "held entity leaked into in-fold"
        assert set(held[hold]) == {held_val}, "held block contains other entities"

        predict_g, cm, xmean, gmean = _fit_within_g(infold, held, f_cols, head_kind)
        grand, alpha, beta, gamma, cell_mean = _anova_effects(infold)
        alpha_eb = _eb_shrunk(infold, "train_dataset", grand)
        gamma_eb = _eb_shrunk(infold, "variant", grand)

        # B: marginal-feature regressor for the HELD axis only (variant is never
        # held out, so gamma_v is always taken from the observed in-fold effect).
        if held_axis_is_source:
            src_marg = _marginal_features(infold, "train_dataset", f_cols)
            h_alpha = _fit_marginal_regressor(alpha, src_marg, f_cols)
            held_marg = held[f_cols].mean()            # held source's own (features only)
        else:
            bench_marg = _marginal_features(infold, "benchmark", f_cols)
            h_beta = _fit_marginal_regressor(beta, bench_marg, f_cols)
            held_marg = held[f_cols].mean()            # held benchmark's own (features only)

        # A (LOBO eval-side IDW) needs in-fold perf + benchmark list.
        perf = {(r.train_dataset, r.benchmark, r.variant): float(r.auc_normalized)
                for r in infold.itertuples()}
        fold_benchmarks = sorted(infold["benchmark"].unique())

        def level_A(i, k, v, cv):
            if cv in cm:                                    # LOTO: benchmark seen
                return cm[cv]
            ds, ps = [], []                                 # LOBO: eval-side IDW
            for e in fold_benchmarks:
                if e == k:
                    continue
                p = perf.get((i, e, v)); d = ee.get((k, e))
                if p is not None and np.isfinite(p) and d is not None and np.isfinite(d):
                    ds.append(d); ps.append(p)
            if ds:
                w = idw_weights(np.asarray(ds, float), ee_is_sim, "idw")
                return float((w * np.asarray(ps)).sum() / w.sum())
            return gmean

        for r in held.itertuples():
            g = predict_g(r)
            i, k, v, cv = r.train_dataset, r.benchmark, r.variant, r.cv
            L_A = level_A(i, k, v, cv)

            # B / C: variant-aware. LOTO holds the source (cell observable);
            # LOBO holds the benchmark (variant still observable).
            if held_axis_is_source:                    # LOTO
                band = cell_mean.get(cv, grand)        # observed (k,v) cell band (= A)
                a_hat = h_alpha(held_marg) if h_alpha is not None else 0.0
                L_B = band + a_hat                     # band + feature-regressed source
                L_C = band                             # unseen source shrinks to 0 -> band
            else:                                      # LOBO
                b_hat = h_beta(held_marg) if h_beta is not None else 0.0
                L_B = grand + alpha.get(i, 0.0) + b_hat + gamma.get(v, 0.0)
                L_C = grand + alpha_eb.get(i, 0.0) + gamma_eb.get(v, 0.0)

            out.append((i, cv, k, v, float(r.auc_normalized), g, L_A, L_B, L_C))
    return pd.DataFrame(out, columns=["train_dataset", "context_id", "benchmark",
                                      "variant", "actual", "g", "L_A", "L_B", "L_C"])


# ---------------------------------------------------------------------------
# Joint heldout (C3 / Pahikkala S4): hold BOTH source i AND benchmark k.
# Neither endpoint observable -> A's level falls back to grand + gamma_v.
# This is the harshest test of g (no observed-row borrowing at all).
# ---------------------------------------------------------------------------
def cv_predict_joint(table: pd.DataFrame, f_cols: list[str],
                     head_kind: str = "ridge") -> pd.DataFrame:
    rows = []
    for i_test in sorted(table["train_dataset"].unique()):
        for k_test in sorted(table["benchmark"].unique()):
            infold = table[(table["train_dataset"] != i_test) &
                           (table["benchmark"] != k_test)]
            held = table[(table["train_dataset"] == i_test) &
                         (table["benchmark"] == k_test)]
            if infold.empty or held.empty:
                continue
            # ---- leakage guards ------------------------------------------------
            assert i_test not in set(infold["train_dataset"])
            assert k_test not in set(infold["benchmark"])

            # Within-context g: held context has NO in-fold rows AND only one
            # source per cv, so within-context demeaning is undefined for the
            # held cell. Center by the in-fold grand feature mean — coarser
            # than LOTO/LOBO centering, but principled (no leakage, uses no
            # outcomes from held rows).
            cm = infold.groupby("cv")["auc_normalized"].mean().to_dict()
            gmean = float(infold["auc_normalized"].mean())
            xmean = {cv: infold.loc[infold.cv == cv, f_cols].mean()
                     for cv in infold.cv.unique()}
            grand_x = infold[f_cols].mean()
            for cv in held.cv.unique():
                xmean[cv] = grand_x

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

            grand, alpha, beta, gamma, cell_mean = _anova_effects(infold)

            for r in held.itertuples():
                xm = xmean[r.cv]
                x = np.asarray([[getattr(r, c) - xm[c] for c in f_cols]], float)
                xs = np.nan_to_num(scl.transform(np.nan_to_num(imp.transform(x))))
                g = float(head.predict(xs)[0])
                # Joint heldout level: only variant carries observed signal.
                L_A = grand + gamma.get(r.variant, 0.0)
                rows.append((r.train_dataset, r.cv, r.benchmark, r.variant,
                             float(r.auc_normalized), g, L_A, L_A, L_A))
    return pd.DataFrame(rows, columns=["train_dataset", "context_id", "benchmark",
                                       "variant", "actual", "g", "L_A", "L_B", "L_C"])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _rank_df(df, pred_col):
    return df.rename(columns={pred_col: "pred"})[
        ["train_dataset", "context_id", "benchmark", "actual", "pred"]]


def _abs_r(df, pred):
    a, p = df["actual"].values, df[pred].values
    m = np.isfinite(a) & np.isfinite(p)
    if m.sum() < 3 or np.std(p[m]) < 1e-9:
        return float("nan")
    return float(pearsonr(p[m], a[m])[0])


def report(df: pd.DataFrame, df_shuffled: pd.DataFrame, family: str):
    # CLAIM: g-only ranking (mechanism-independent).
    gd = _rank_df(df, "g")
    claim = {"ctx_rho": within_context_spearman(gd),
             "cent_rho": context_centered_spearman(gd),
             "shuffled_ctx": within_context_spearman(_rank_df(df_shuffled, "g"))}
    rank_g = claim["ctx_rho"]

    levels = []
    for mech in MECHANISMS:
        Lcol = f"L_{mech}"
        df = df.copy()
        df["abs_pred"] = df[Lcol] + df["g"]
        df["fold_pred"] = df[Lcol] + df["g"]      # fold L INTO the ranking score
        rank_fold = within_context_spearman(_rank_df(df, "fold_pred"))
        rank_lvl = within_context_spearman(_rank_df(df, Lcol))
        levels.append({
            "mech": mech,
            "MAE": context_mae(_rank_df(df, "abs_pred")),
            "abs_r": _abs_r(df, "abs_pred"),
            "rank_fold": rank_fold,
            "d_rank": rank_fold - rank_g,         # <0 => folding L hurts the order
            "level_only_rank": rank_lvl,
        })
    return claim, levels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def shuffle_within_context(table: pd.DataFrame, rng) -> pd.DataFrame:
    """Permute the target within each context -> destroys the (i->k) signal but
    preserves all marginals. The g-ranking on this must collapse to ~0."""
    t = table.copy()
    t["auc_normalized"] = (
        t.groupby("cv")["auc_normalized"]
         .transform(lambda s: rng.permutation(s.values)))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="scripts/transfer_analysis_v3/transfer_table.csv")
    ap.add_argument("--dist", default="analysis_v3/pairwise_self_distances.csv")
    ap.add_argument("--space", default="flow")
    ap.add_argument("--metric", default="mean_nn_sym")
    ap.add_argument("--families", nargs="+",
                    default=["motion", "appearance", "both", "random"],
                    help="'random' = dim-matched gaussian features, honest g-only "
                         "ranking floor (no level memorization).")
    ap.add_argument("--splits", nargs="+", default=["LOTO", "LOBO", "JOINT"],
                    choices=["LOTO", "LOBO", "JOINT"],
                    help="JOINT = held source AND held benchmark (C3/S4); the "
                         "harshest test of g (no observed-row borrowing).")
    ap.add_argument("--family-matched-prior", action="store_true",
                    help="Run appearance with dino-IDW (instead of flow-IDW) "
                         "as a level-prior ablation. Reported separately.")
    ap.add_argument("--head", default="ridge", choices=["ridge", "gbm"])
    ap.add_argument("--feature-source", default="self_dist", choices=["table", "self_dist"])
    ap.add_argument("--out", default="scripts/transfer_analysis_v3/results/level_sweep")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(".").resolve()
    table = pd.read_csv(root / args.table).dropna(subset=["auc_normalized"]).copy()
    table["variant"] = table.apply(variant_key, axis=1)
    table["cv"] = table["benchmark"] + "|" + table["variant"]
    dist_df = pd.read_csv(root / args.dist)
    ee = build_lookup(dist_df, args.space, "eval_eval", args.metric)
    ee_is_sim = args.metric in SIMILARITY_METRICS
    if args.feature_source == "self_dist":
        table = add_selfdist_features(table, dist_df)
    # Add dim-matched random columns so 'random' family runs alongside the others.
    table = add_random_features(table, n_feats=13, seed=42)
    table_sh = shuffle_within_context(table, rng)

    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"head={args.head}  feature_source={args.feature_source}  rows={len(table)}")
    print("CLAIM = rank by g only (within estimator); A/B/C add the level.")
    print("D = g-only is the CLAIM row.  d_rank<0 => folding that level into the "
          "order HURTS it.\n")
    SPLIT_HOLDS = {"LOTO": "train_dataset", "LOBO": "benchmark"}
    rows_csv = []
    for split in args.splits:
        print(f"================================ {split} ================================")
        for fam in args.families:
            f_cols = feature_set_ext(table, fam, args.feature_source)
            if split == "JOINT":
                df = cv_predict_joint(table, f_cols, args.head)
                df_sh = cv_predict_joint(table_sh, f_cols, args.head)
            else:
                hold = SPLIT_HOLDS[split]
                df = cv_predict(table, f_cols, hold, ee, ee_is_sim, args.head)
                df_sh = cv_predict(table_sh, f_cols, hold, ee, ee_is_sim, args.head)
            claim, levels = report(df, df_sh, fam)
            df.to_csv(out_dir / f"rows_{split}_{fam}.csv", index=False)

            # In JOINT, B/C collapse to A (no axis-specific level info available)
            # — show only A to avoid implying a difference.
            level_rows = [lv for lv in levels if lv["mech"] == "A"] if split == "JOINT" else levels

            print(f"\n  --- {fam}  ({len(f_cols)} feats) ---")
            print(f"  [CLAIM] g-only ranking:  ctx_rho={claim['ctx_rho']:+.3f}  "
                  f"cent_rho={claim['cent_rho']:+.3f}  "
                  f"(shuffled control={claim['shuffled_ctx']:+.3f})")
            print(f"  [LEVEL] {'mech':<5}{'MAE':>8}{'abs_r':>8}{'rank(g+L)':>11}"
                  f"{'d_rank':>9}{'level_only':>12}")
            for lv in level_rows:
                print(f"          {lv['mech']:<5}{lv['MAE']:>8.2f}{lv['abs_r']:>8.3f}"
                      f"{lv['rank_fold']:>11.3f}{lv['d_rank']:>+9.3f}"
                      f"{lv['level_only_rank']:>12.3f}")
                rows_csv.append({"split": split, "family": fam,
                                 "claim_ctx_rho": claim["ctx_rho"],
                                 "claim_cent_rho": claim["cent_rho"],
                                 "shuffled_ctx": claim["shuffled_ctx"], **lv})
        print()

    # -------------------------------------------------------------------
    # Family-matched-prior ablation: appearance with DINO-IDW vs flow-IDW.
    # Tests whether dino-similarity defines useful benchmark neighbors as
    # well as flow-similarity does. If flow gives the higher level_only,
    # that's independent evidence flow is the relevant axis.
    # -------------------------------------------------------------------
    if args.family_matched_prior:
        print("================ FAMILY-MATCHED PRIOR ABLATION (LOBO) ================")
        print("All numbers below are WITHIN-CONTEXT (within (benchmark, variant) cell)")
        print("except abs_r which is the pooled Pearson(L+g, actual).")
        print("g is identical across priors — only L differs.\n")
        ee_dino, ee_dino_is_sim = _build_ee(dist_df, "dino", args.metric)
        ablation_fams = [f for f in args.families if f not in {"random"}]
        df_sh_cache: dict[str, pd.DataFrame] = {}
        print(f"  {'family':<13}{'prior':<10}{'g-only ρ':>10}{'level_only':>12}"
              f"{'rank(g+L)':>11}{'d_rank':>9}{'MAE':>8}{'abs_r':>8}")
        for fam in ablation_fams:
            f_cols = feature_set_ext(table, fam, args.feature_source)
            df_flow = cv_predict(table, f_cols, "benchmark", ee, ee_is_sim, args.head)
            df_dino = cv_predict(table, f_cols, "benchmark", ee_dino, ee_dino_is_sim, args.head)
            df_sh_cache[fam] = df_sh_cache.get(fam) or cv_predict(
                table_sh, f_cols, "benchmark", ee, ee_is_sim, args.head)
            df_sh = df_sh_cache[fam]
            claim_flow, lv_flow = report(df_flow, df_sh, fam)
            claim_dino, lv_dino = report(df_dino, df_sh, fam)
            A_flow = next(lv for lv in lv_flow if lv["mech"] == "A")
            A_dino = next(lv for lv in lv_dino if lv["mech"] == "A")
            for tag, A in [("FLOW-IDW", A_flow), ("DINO-IDW", A_dino)]:
                claim_rho = claim_flow["ctx_rho"]    # same across priors
                print(f"  {fam:<13}{tag:<10}{claim_rho:>+10.3f}{A['level_only_rank']:>12.3f}"
                      f"{A['rank_fold']:>11.3f}{A['d_rank']:>+9.3f}"
                      f"{A['MAE']:>8.2f}{A['abs_r']:>8.3f}")
                rows_csv.append({"split": "LOBO_ABLATION", "family": fam,
                                 "claim_ctx_rho": claim_rho,
                                 "claim_cent_rho": claim_flow["cent_rho"],
                                 "shuffled_ctx": claim_flow["shuffled_ctx"],
                                 "mech": f"A_{tag.split('-')[0].lower()}_idw", **A})
            df_dino.to_csv(out_dir / f"rows_LOBO_{fam}_dinoIDW.csv", index=False)
        print()
    pd.DataFrame(rows_csv).to_csv(out_dir / "level_sweep_summary.csv", index=False)
    print(f"summary -> {out_dir}/level_sweep_summary.csv")
    print(f"per-row predictions -> {out_dir}/rows_<split>_<family>.csv")


if __name__ == "__main__":
    main()
