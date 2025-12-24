#!/usr/bin/env python3
"""
Diagnostics for mixed-effects model stability and predictor reliability.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def zscore(series):
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mean) / std


def compute_vif(df, predictors):
    data = df[predictors].to_numpy(dtype=float)
    n, p = data.shape
    vifs = {}
    for i, pred in enumerate(predictors):
        y = data[:, i]
        X = np.delete(data, i, axis=1)
        if X.size == 0:
            vifs[pred] = 1.0
            continue
        X = np.column_stack([np.ones(n), X])
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X.dot(coef)
        denom = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - (np.sum((y - y_pred) ** 2) / denom if denom != 0 else np.nan)
        if np.isnan(r2) or r2 >= 1.0:
            vifs[pred] = np.inf
        else:
            vifs[pred] = float(1.0 / (1.0 - r2))
    return vifs


def compute_within_benchmark_slopes(df, predictors, target, min_rows=12):
    rows = []
    for benchmark, sub in df.groupby("benchmark"):
        sub = sub.dropna(subset=predictors + [target])
        if len(sub) < max(min_rows, len(predictors) + 2):
            continue
        z_df = sub.copy()
        z_df[target] = zscore(z_df[target])
        for col in predictors:
            z_df[col] = zscore(z_df[col])
        X = z_df[predictors].to_numpy(dtype=float)
        y = z_df[target].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(X)), X])
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X.dot(coef)
        denom = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - (np.sum((y - y_pred) ** 2) / denom if denom != 0 else np.nan)
        row = {"benchmark": benchmark, "n": int(len(sub)), "r2": float(r2)}
        for name, value in zip(predictors, coef[1:]):
            row[name] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_covariance(result):
    summary = {
        "converged": getattr(result, "converged", False),
        "aic": getattr(result, "aic", np.nan),
        "bic": getattr(result, "bic", np.nan),
        "llf": getattr(result, "llf", np.nan),
        "cov_rank": np.nan,
        "cov_min_eig": np.nan,
        "cov_max_eig": np.nan,
        "cov_det": np.nan,
    }

    try:
        cov = result.cov_re
    except Exception:
        return summary

    if cov is None:
        return summary

    try:
        cov = np.asarray(cov, dtype=float)
        if cov.size == 0:
            return summary
        summary["cov_rank"] = int(np.linalg.matrix_rank(cov))
        eigvals = np.linalg.eigvals(cov)
        eigvals = np.asarray(eigvals, dtype=float)
        summary["cov_min_eig"] = float(np.min(eigvals))
        summary["cov_max_eig"] = float(np.max(eigvals))
        summary["cov_det"] = float(np.linalg.det(cov))
    except Exception:
        return summary

    return summary


def fit_mixedlm(df, formula, group_col, re_formula):
    try:
        model = smf.mixedlm(formula, data=df, groups=df[group_col], re_formula=re_formula)
        result = model.fit(reml=False, method="lbfgs")
        return result, None
    except Exception as exc:
        return None, str(exc)


def main():
    parser = argparse.ArgumentParser(description="MixedLM diagnostics.")
    parser.add_argument(
        "--auc-table",
        default="analysis/leakage_free/auc_with_features.csv",
        help="AUC table with predictors.",
    )
    parser.add_argument(
        "--target",
        default="auc_normalized",
        help="Target column name.",
    )
    parser.add_argument(
        "--predictors",
        default=(
            "flow_train_to_eval_coverage_logit,flow_eval_to_train_coverage_logit,"
            "resnet_train_to_eval_coverage_logit,resnet_eval_to_train_coverage_logit,"
            "flow_mmd,feature_mmd"
        ),
        help="Comma-separated predictors.",
    )
    parser.add_argument(
        "--group-col",
        default="benchmark",
        help="Grouping column for random effects.",
    )
    parser.add_argument(
        "--random-slopes",
        default="flow_train_to_eval_coverage_logit,flow_mmd",
        help="Comma-separated predictors to use as random slopes.",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Standardize predictors and target before fitting.",
    )
    parser.add_argument(
        "--output-file",
        default="analysis/leakage_free/mixedlm_diagnostics.txt",
        help="Output summary file.",
    )
    args = parser.parse_args()

    auc_path = Path(args.auc_table)
    output_path = Path(args.output_file)
    if not auc_path.exists():
        output_path.write_text(f"Missing AUC table: {auc_path}")
        return

    df = pd.read_csv(auc_path)
    predictors = [p.strip() for p in args.predictors.split(",") if p.strip()]
    random_slopes = [p.strip() for p in args.random_slopes.split(",") if p.strip()]

    missing = [p for p in predictors if p not in df.columns]
    if missing:
        output_path.write_text(f"Missing predictors in table: {', '.join(missing)}")
        return

    if args.target not in df.columns:
        output_path.write_text(f"Missing target '{args.target}' in table")
        return

    df = df.dropna(subset=predictors + [args.target, args.group_col])
    if df.empty:
        output_path.write_text("No complete rows available for diagnostics.")
        return

    lines = []
    lines.append("MIXEDLM DIAGNOSTICS")
    lines.append("=" * 80)
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Groups ({args.group_col}): {df[args.group_col].nunique()}")
    group_sizes = df.groupby(args.group_col).size()
    lines.append(
        f"Group sizes: min={group_sizes.min()}, median={int(group_sizes.median())}, max={group_sizes.max()}"
    )

    if args.standardize:
        df = df.copy()
        df[args.target] = zscore(df[args.target])
        for col in predictors:
            df[col] = zscore(df[col])
        lines.append("Standardization: applied to target + predictors")

    lines.append("")
    lines.append("Collinearity checks:")
    corr = df[predictors].corr().fillna(0.0)
    high_corr = []
    for i in range(len(predictors)):
        for j in range(i + 1, len(predictors)):
            val = corr.iloc[i, j]
            if abs(val) >= 0.7:
                high_corr.append((predictors[i], predictors[j], val))
    if high_corr:
        lines.append("  High |corr| pairs (>=0.70):")
        for a, b, v in sorted(high_corr, key=lambda x: -abs(x[2])):
            lines.append(f"    {a} vs {b}: {v:+.3f}")
    else:
        lines.append("  No predictor pairs above |corr|>=0.70")

    vifs = compute_vif(df, predictors)
    lines.append("  VIF (>=5 flagged):")
    for name, value in sorted(vifs.items(), key=lambda x: -x[1]):
        flag = "*" if value >= 5 else ""
        if np.isinf(value):
            lines.append(f"    {name}: inf {flag}")
        else:
            lines.append(f"    {name}: {value:.2f} {flag}")

    lines.append("")
    lines.append("Within-benchmark slope variability:")
    slopes_df = compute_within_benchmark_slopes(df, predictors, args.target)
    if slopes_df.empty:
        lines.append("  Not enough data to compute within-benchmark slopes.")
    else:
        for pred in predictors:
            if pred not in slopes_df.columns:
                continue
            signs = slopes_df[pred].dropna()
            if signs.empty:
                continue
            pos = int((signs > 0).sum())
            neg = int((signs < 0).sum())
            lines.append(f"  {pred}: +{pos} / -{neg} across benchmarks (std={signs.std(ddof=0):.3f})")

    if not HAS_STATSMODELS:
        lines.append("")
        lines.append("Statsmodels not available; skipping MixedLM fits.")
        output_path.write_text("\n".join(lines))
        return

    lines.append("")
    lines.append("Model comparison:")
    formula = f"{args.target} ~ " + " + ".join(predictors)

    # OLS baseline
    try:
        ols = smf.ols(formula, data=df).fit()
        lines.append(f"  OLS: AIC={ols.aic:.2f}, BIC={ols.bic:.2f}, R2={ols.rsquared:.3f}")
    except Exception as exc:
        lines.append(f"  OLS failed: {exc}")

    # MixedLM random intercept
    ri_result, ri_err = fit_mixedlm(df, formula, args.group_col, re_formula="1")
    if ri_result is None:
        lines.append(f"  MixedLM (RI) failed: {ri_err}")
    else:
        ri_summary = summarize_covariance(ri_result)
        lines.append(
            "  MixedLM (RI): "
            f"AIC={ri_summary['aic']:.2f}, BIC={ri_summary['bic']:.2f}, "
            f"converged={ri_summary['converged']}"
        )
        lines.append(
            "    cov_re: rank={cov_rank}, min_eig={cov_min_eig:.3g}, max_eig={cov_max_eig:.3g}".format(
                **ri_summary
            )
        )

    # MixedLM random slopes
    random_slopes = [p for p in random_slopes if p in predictors]
    re_formula = "1"
    if random_slopes:
        re_formula = "1 + " + " + ".join(random_slopes)
    rs_result, rs_err = fit_mixedlm(df, formula, args.group_col, re_formula=re_formula)
    if rs_result is None:
        lines.append(f"  MixedLM (RS) failed: {rs_err}")
    else:
        rs_summary = summarize_covariance(rs_result)
        lines.append(
            "  MixedLM (RS): "
            f"AIC={rs_summary['aic']:.2f}, BIC={rs_summary['bic']:.2f}, "
            f"converged={rs_summary['converged']}"
        )
        lines.append(
            "    cov_re: rank={cov_rank}, min_eig={cov_min_eig:.3g}, max_eig={cov_max_eig:.3g}".format(
                **rs_summary
            )
        )

    lines.append("")
    lines.append("Actionable guidance:")
    if ri_result is None:
        lines.append("  - MixedLM random intercept failed; use OLS or simplify predictors.")
    else:
        ri_summary = summarize_covariance(ri_result)
        if ri_summary["cov_min_eig"] is not np.nan and ri_summary["cov_min_eig"] < 1e-6:
            lines.append("  - Random intercept variance ~0 (singular covariance). Random effects likely not supported.")
    if rs_result is None:
        lines.append("  - Random-slope model failed; reduce slopes or use random intercept only.")
    else:
        rs_summary = summarize_covariance(rs_result)
        if rs_summary["cov_min_eig"] is not np.nan and rs_summary["cov_min_eig"] < 1e-6:
            lines.append("  - Random slopes are singular; slope heterogeneity may be too weak for this data.")

    if high_corr:
        lines.append("  - High predictor collinearity detected; consider dropping/reducing correlated predictors.")
    if any(v >= 10 for v in vifs.values() if np.isfinite(v)):
        lines.append("  - VIF >= 10 suggests unstable coefficients; reduce predictors or use regularization.")

    output_path.write_text("\n".join(lines))
    print(f"Wrote diagnostics to {output_path}")


if __name__ == "__main__":
    main()
