#!/usr/bin/env python3
"""
Few-shot mixed-effects transfer prediction.

Protocol
--------
Outer split: leave-one-benchmark-out (lobo).
  - Fit fixed effects β on all training benchmarks.
  - Estimate σ²_u (context random intercept variance) and σ²_ε (residual variance)
    from training contexts via method-of-moments.

Inner evaluation: for each held-out context c = (benchmark × model_family × pretrained × freeze):
  - k=0: predict absolute AUC using fixed effects only  (zero-shot).
  - k∈{1,2,5}: observe k (training_dataset, AUC) pairs → compute BLUP intercept → predict rest.
  - k=full: observe all-but-one (LOO upper bound).

BLUP formula:
    û_c = [k σ²_u / (k σ²_u + σ²_ε)] · (1/k) Σ_j (y_j − x_j β)

Outputs (in --output-dir):
  few_shot_context_metrics.csv   — per-context metrics for each (k, feature_group, benchmark)
  few_shot_aggregate.csv         — aggregate (mean ± 95% CI via bootstrap) across contexts
  few_shot_learning_curve.csv    — mean R²/MAE/Spearman as function of k, for plotting
"""

import argparse
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

K_VALUES = [0, 1, 2, 5]
N_SEEDS = 20          # random seeds for k-shot subsampling
MIN_CONTEXT_SIZE = 4  # skip contexts with fewer train datasets than this


# ---------------------------------------------------------------------------
# Feature groups (mirrors run_experiments.py)
# ---------------------------------------------------------------------------

def resolve_feature_groups(table_cols: list[str]) -> dict[str, list[str]]:
    def match(patterns):
        cols = []
        for pat in patterns:
            cols.extend(c for c in table_cols if c == pat or c.startswith(pat))
        return list(dict.fromkeys(cols))

    FLOW_EPS = ["eps1px", "eps4px", "eps16px"]
    flow_eps = match([f"flow_eval_covered_by_train_{t}" for t in FLOW_EPS]
                   + [f"flow_train_covered_by_eval_{t}" for t in FLOW_EPS])
    flow_nn  = match(["flow_mean_nn_eval_to_train_k1", "flow_mean_nn_train_to_eval_k1"])
    flow_km  = match([f"flow_km_eval_covered_by_train_{t}_weighted" for t in FLOW_EPS]
                   + [f"flow_km_train_covered_by_eval_{t}_weighted" for t in FLOW_EPS])
    dino     = match(["dino_mean_nn_eval_to_train_k1", "dino_mean_nn_train_to_eval_k1",
                      "dino_eval_covered_by_train_qnorm_k1", "dino_train_covered_by_eval_qnorm_k1"])
    density  = match(["log_train_n_vectors", "log_eval_n_vectors"])
    sym_mmd  = match(["flow_mmd", "dino_mmd"])
    sym_ot   = match(["flow_fid", "flow_sliced_w2", "dino_fid", "dino_sliced_w2"])

    groups = {
        "motion":            flow_nn + flow_eps,
        "motion_appearance": flow_nn + flow_eps + dino,
        "all":               flow_nn + flow_eps + dino + density + sym_mmd + sym_ot,
    }
    existing = set(table_cols)
    return {k: list(dict.fromkeys(c for c in v if c in existing))
            for k, v in groups.items()}


# ---------------------------------------------------------------------------
# Mixed-effects model with BLUP intercept estimation
# ---------------------------------------------------------------------------

class FewShotMixedEffectsModel:
    """
    Linear mixed-effects model:
        y_ic = x_ic @ β + u_c + ε_ic
        u_c ~ N(0, σ²_u),  ε_ic ~ N(0, σ²_ε)

    Fitting: ridge regression for fixed effects β; method-of-moments for σ²_u, σ²_ε.
    Prediction: BLUP û_c from k observed (x, y) pairs in context c.
    """

    def __init__(self):
        self.ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        self.var_u: float = 0.0
        self.var_e: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> None:
        self.ridge.fit(X, y)
        resid = y - self.ridge.predict(X)

        unique_groups = np.unique(groups)
        ctx_means = np.array([resid[groups == g].mean() for g in unique_groups])
        n_per_ctx = np.array([np.sum(groups == g) for g in unique_groups], dtype=float)

        # Within-context residual variance (pooled)
        within_vars = []
        for g in unique_groups:
            r = resid[groups == g]
            if len(r) > 1:
                within_vars.append(r.var(ddof=1))
        self.var_e = float(np.mean(within_vars)) if within_vars else float(np.var(resid))

        # Between-context variance → subtract sampling variance
        # Var(ȳ_c) ≈ σ²_u + σ²_ε / n_c → σ²_u ≈ Var(ȳ_c) − mean(σ²_ε / n_c)
        sampling_correction = float(np.mean(self.var_e / n_per_ctx))
        self.var_u = max(0.0, float(np.var(ctx_means, ddof=1)) - sampling_correction)

    def predict_fixed(self, X: np.ndarray) -> np.ndarray:
        """Zero-shot: fixed effects only."""
        return self.ridge.predict(X)

    def predict_blup(
        self,
        X_query: np.ndarray,
        X_shots: np.ndarray,
        y_shots: np.ndarray,
    ) -> np.ndarray:
        """
        Few-shot: fixed effects + BLUP context intercept.
        Shrinkage → 0 when k=0 or σ²_u=0 (degenerates to fixed effects).
        """
        fixed_query = self.ridge.predict(X_query)
        k = len(y_shots)
        if k == 0 or self.var_u == 0.0:
            return fixed_query
        fixed_shots = self.ridge.predict(X_shots)
        shot_resid = y_shots - fixed_shots
        shrinkage = (k * self.var_u) / (k * self.var_u + self.var_e)
        blup = shrinkage * float(shot_resid.mean())
        return fixed_query + blup


# ---------------------------------------------------------------------------
# Preprocessing (fit on training fold only)
# ---------------------------------------------------------------------------

def fit_preprocessor(df: pd.DataFrame, cols: list[str]):
    X = df[cols].values.astype(np.float64)
    imp = SimpleImputer(strategy="median").fit(X)
    scl = StandardScaler().fit(imp.transform(X))
    return imp, scl


def apply_preprocessor(df: pd.DataFrame, cols: list[str], preprocessor) -> np.ndarray:
    imp, scl = preprocessor
    return scl.transform(imp.transform(df[cols].values.astype(np.float64))).astype(np.float32)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def context_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    n = len(y_true)
    if n < 2:
        return {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp = spearmanr(y_true, y_pred).statistic
    return {
        "r2":       float(r2_score(y_true, y_pred)),
        "mae":      float(mean_absolute_error(y_true, y_pred)),
        "spearman": float(sp) if np.isfinite(sp) else float("nan"),
        "n":        n,
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = values[np.isfinite(values)]
    if len(vals) < 2:
        return float("nan"), float("nan")
    means = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------------------
# Leave-one-benchmark-out few-shot evaluation
# ---------------------------------------------------------------------------

def run_lobo_few_shot(
    df: pd.DataFrame,
    feature_cols: list[str],
    k_values: list[int],
    n_seeds: int = N_SEEDS,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Outer: hold out one benchmark.
    Inner: for each context_id within the held-out benchmark,
           evaluate k-shot prediction (average over n_seeds random shot selections).
    """
    rows = []
    benchmarks = sorted(df["benchmark"].unique())

    for benchmark in benchmarks:
        test_mask = df["benchmark"] == benchmark
        train_df  = df[~test_mask]
        test_df   = df[test_mask]

        if len(train_df) < 10 or len(test_df) < MIN_CONTEXT_SIZE:
            continue

        prep = fit_preprocessor(train_df, feature_cols)
        X_train = apply_preprocessor(train_df, feature_cols, prep)
        y_train = train_df["auc_normalized"].values.astype(np.float64)
        groups  = train_df["context_id"].values

        model = FewShotMixedEffectsModel()
        model.fit(X_train, y_train, groups)

        if verbose:
            print(f"  {benchmark:30s}  σ²_u={model.var_u:.4f}  σ²_ε={model.var_e:.4f}")

        for ctx_id, ctx_grp in test_df.groupby("context_id"):
            ctx_grp = ctx_grp.reset_index(drop=True)
            n_ctx = len(ctx_grp)
            if n_ctx < MIN_CONTEXT_SIZE:
                continue

            X_ctx = apply_preprocessor(ctx_grp, feature_cols, prep)
            y_ctx = ctx_grp["auc_normalized"].values.astype(np.float64)

            for k in k_values:
                k_eff = min(k, n_ctx - 2)  # must leave ≥2 for prediction
                if k_eff < 0:
                    continue

                if k_eff == 0:
                    # Zero-shot: fixed effects only, no shot sampling needed
                    pred = model.predict_fixed(X_ctx)
                    m = context_metrics(y_ctx, pred)
                    if m:
                        m.update({"benchmark": benchmark, "context_id": ctx_id,
                                  "k": 0, "n_ctx": n_ctx})
                        rows.append(m)
                    continue

                # Average over multiple random shot selections
                seed_rs, seed_sp, seed_mae = [], [], []
                rng = np.random.default_rng(42)
                for _ in range(n_seeds):
                    shot_idx = rng.choice(n_ctx, k_eff, replace=False)
                    pred_idx = np.array([i for i in range(n_ctx) if i not in set(shot_idx.tolist())])
                    if len(pred_idx) < 2:
                        continue
                    pred = model.predict_blup(
                        X_ctx[pred_idx],
                        X_ctx[shot_idx],
                        y_ctx[shot_idx],
                    )
                    m = context_metrics(y_ctx[pred_idx], pred)
                    if m:
                        seed_rs.append(m["r2"])
                        seed_sp.append(m["spearman"])
                        seed_mae.append(m["mae"])

                if not seed_rs:
                    continue
                rows.append({
                    "benchmark":  benchmark,
                    "context_id": ctx_id,
                    "k":          k_eff,
                    "n_ctx":      n_ctx,
                    "r2":         float(np.mean(seed_rs)),
                    "spearman":   float(np.nanmean(seed_sp)),
                    "mae":        float(np.mean(seed_mae)),
                    "n":          n_ctx - k_eff,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table",      default="scripts/transfer_analysis_v3/transfer_table.csv")
    parser.add_argument("--output-dir", default="scripts/transfer_analysis_v3/results/few_shot")
    parser.add_argument("--k-values",   nargs="+", type=int, default=K_VALUES)
    parser.add_argument("--n-seeds",    type=int, default=N_SEEDS)
    parser.add_argument("--feature-groups", nargs="+",
                        default=["motion", "motion_appearance", "all"])
    parser.add_argument("--debug",      action="store_true")
    args = parser.parse_args()

    table_path = Path(args.table)
    if not table_path.exists():
        raise SystemExit(f"Transfer table not found: {table_path}\nRun build_table.py first.")

    df = pd.read_csv(table_path)
    print(f"Loaded {len(df)} rows | "
          f"{df['train_dataset'].nunique()} train datasets | "
          f"{df['benchmark'].nunique()} benchmarks | "
          f"{df['context_id'].nunique()} contexts")

    feature_groups = resolve_feature_groups(df.columns.tolist())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_ctx_rows = []
    all_agg_rows = []

    for fg_name in args.feature_groups:
        feat_cols = feature_groups.get(fg_name, [])
        if not feat_cols:
            print(f"  Feature group '{fg_name}' — no columns found, skipping")
            continue
        print(f"\n=== Feature group: {fg_name} ({len(feat_cols)} features) ===")

        ctx_df = run_lobo_few_shot(
            df, feat_cols, args.k_values,
            n_seeds=args.n_seeds, verbose=not args.debug,
        )
        if ctx_df.empty:
            continue
        ctx_df["feature_group"] = fg_name
        all_ctx_rows.append(ctx_df)

        # Aggregate over contexts per (k, feature_group)
        for k, k_grp in ctx_df.groupby("k"):
            for metric in ["r2", "mae", "spearman"]:
                vals = k_grp[metric].dropna().values
                lo, hi = bootstrap_ci(vals)
                all_agg_rows.append({
                    "feature_group": fg_name,
                    "k":             k,
                    "metric":        metric,
                    "mean":          float(np.nanmean(vals)),
                    "median":        float(np.nanmedian(vals)),
                    "ci_lo":         lo,
                    "ci_hi":         hi,
                    "n_contexts":    int(len(vals)),
                })
            print(f"  k={k:2d}  "
                  f"R²={k_grp['r2'].mean():.3f}  "
                  f"MAE={k_grp['mae'].mean():.4f}  "
                  f"Spearman={k_grp['spearman'].mean():.3f}  "
                  f"({len(k_grp)} contexts)")

    if not all_ctx_rows:
        print("No results — check feature columns and table.")
        return

    ctx_out = pd.concat(all_ctx_rows, ignore_index=True)
    agg_out = pd.DataFrame(all_agg_rows)

    # Learning curve table: mean metric vs k, wide format
    curve_rows = []
    for (fg, metric), grp in agg_out.groupby(["feature_group", "metric"]):
        for _, r in grp.sort_values("k").iterrows():
            curve_rows.append({
                "feature_group": fg, "metric": metric,
                "k": int(r["k"]), "mean": r["mean"],
                "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
            })
    curve_out = pd.DataFrame(curve_rows)

    ctx_out.to_csv(out_dir / "few_shot_context_metrics.csv", index=False)
    agg_out.to_csv(out_dir / "few_shot_aggregate.csv", index=False)
    curve_out.to_csv(out_dir / "few_shot_learning_curve.csv", index=False)

    print(f"\n✓ Saved to {out_dir}/")
    print(f"  few_shot_context_metrics.csv  ({len(ctx_out)} rows)")
    print(f"  few_shot_aggregate.csv        ({len(agg_out)} rows)")
    print(f"  few_shot_learning_curve.csv   ({len(curve_out)} rows)")

    # Quick summary table
    try:
        pivot = curve_out[curve_out["metric"] == "r2"].pivot_table(
            values="mean", index="feature_group", columns="k", aggfunc="first")
        print("\nR² by k (learning curve, lobo split):")
        print(pivot.round(3).to_string())
    except Exception:
        pass


if __name__ == "__main__":
    main()
