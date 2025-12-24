#!/usr/bin/env python3
"""
Summarize leakage-free analysis outputs into a single text report.
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


def _to_bool(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False})


def _zscore(series):
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mean) / std


def _pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return np.nan
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return np.nan
    return float(np.dot(x, y) / denom)


def _spearman(x, y):
    if len(x) < 2:
        return np.nan
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return _pearson(rx, ry)


def _fit_standardized_model(df, predictors, target, model="ols", ridge_alpha=1.0):
    df_sub = df[predictors + [target]].dropna().copy()
    if df_sub.empty:
        return None
    df_sub[f"{target}_z"] = _zscore(df_sub[target])
    for col in predictors:
        df_sub[f"{col}_z"] = _zscore(df_sub[col])
    z_cols = [f"{c}_z" for c in predictors]

    if model == "ridge":
        X = df_sub[z_cols].to_numpy(dtype=float)
        y = df_sub[f"{target}_z"].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(X)), X])
        penalty = np.eye(X.shape[1])
        penalty[0, 0] = 0.0
        coef = np.linalg.solve(X.T @ X + float(ridge_alpha) * penalty, X.T @ y)
        params = {"Intercept": coef[0]}
        for name, value in zip(predictors, coef[1:]):
            params[name] = value
        y_pred = X.dot(coef)
        denom = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - (np.sum((y - y_pred) ** 2) / denom if denom != 0 else np.nan)
        return {"params": params, "pvalues": None, "n": len(df_sub), "r2": r2}

    if not HAS_STATSMODELS:
        X = df_sub[z_cols].to_numpy(dtype=float)
        y = df_sub[f"{target}_z"].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(X)), X])
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        params = {"Intercept": coef[0]}
        for name, value in zip(predictors, coef[1:]):
            params[name] = value
        return {"params": params, "pvalues": None, "n": len(df_sub)}

    formula = f"{target}_z ~ " + " + ".join(z_cols)
    result = smf.ols(formula, data=df_sub).fit()
    params = {name.replace("_z", ""): value for name, value in result.params.items()}
    pvalues = {name.replace("_z", ""): value for name, value in result.pvalues.items()}
    return {"params": params, "pvalues": pvalues, "n": len(df_sub), "r2": result.rsquared}


def _format_coef(name, coef, pval):
    if pval is None or np.isnan(pval):
        return f"{name}: {coef:+.3f}"
    return f"{name}: {coef:+.3f} (p={pval:.3g})"


def _filter_predictors(df, predictors):
    present = [p for p in predictors if p in df.columns]
    missing = [p for p in predictors if p not in df.columns]
    all_nan = [p for p in present if df[p].isna().all()]
    remaining = [p for p in present if p not in all_nan]
    constant = [p for p in remaining if df[p].nunique(dropna=True) < 2]
    filtered = [p for p in remaining if p not in constant]
    redundant = []
    for prefix in ("flow", "resnet", "dino"):
        cov = f"{prefix}_eval_to_train_coverage"
        outside = f"{prefix}_outside_mass"
        cov_logit = f"{cov}_logit"
        outside_logit = f"{outside}_logit"
        if cov in filtered and outside in filtered:
            redundant.append(outside)
        if cov_logit in filtered and outside_logit in filtered:
            redundant.append(outside_logit)
    if redundant:
        filtered = [p for p in filtered if p not in redundant]
    return filtered, missing, all_nan, constant, redundant


def summarize_predictions(summary_df, label_col, top_n=3):
    if summary_df.empty:
        return []
    overall = summary_df[summary_df[label_col] == "__overall__"]
    rows = []
    if not overall.empty:
        row = overall.iloc[0]
        rows.append(
            f"Overall: MAE={row['mae']:.2f}, RMSE={row['rmse']:.2f}, "
            f"Pearson={row['pearson']:.3f}, Spearman={row['spearman']:.3f}"
        )
    per_group = summary_df[summary_df[label_col] != "__overall__"].copy()
    if not per_group.empty:
        best = per_group.sort_values("spearman", ascending=False).head(top_n)
        worst = per_group.sort_values("spearman", ascending=True).head(top_n)
        rows.append("Best by Spearman: " + ", ".join(
            f"{r[label_col]} ({r['spearman']:.2f})" for _, r in best.iterrows()
        ))
        rows.append("Worst by Spearman: " + ", ".join(
            f"{r[label_col]} ({r['spearman']:.2f})" for _, r in worst.iterrows()
        ))
    return rows


def _format_rank_metrics(prefix, row):
    if row is None:
        return f"{prefix}: n/a"
    spearman = row.get("spearman")
    if pd.isna(spearman):
        spearman_str = "n/a"
    else:
        spearman_str = f"{spearman:.2f}"
    return (
        f"{prefix}: top1={row['top1']:.2f}, top3={row['top3']:.2f}, "
        f"regret={row['regret']:.2f}, spearman={spearman_str}"
    )


def _rank_family_summary(df, benchmarks):
    if df is None or df.empty:
        return None
    sub = df[df["benchmark"].isin(benchmarks)].copy()
    if sub.empty:
        return None
    return {
        "top1": float(sub["top1"].mean()),
        "top3": float(sub["top3"].mean()),
        "regret": float(sub["regret"].mean()),
        "spearman": float(sub["spearman"].mean()),
    }


def _select_baseline_selector(df, candidates):
    if df is None or df.empty:
        return None, None
    for name in candidates:
        sub = df[df["selector"] == name]
        if not sub.empty:
            return name, sub
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Summarize leakage-free outputs.")
    parser.add_argument(
        "--output-dir",
        default="analysis/leakage_free",
        help="Directory containing leakage-free outputs.",
    )
    parser.add_argument(
        "--auc-table",
        default="analysis/leakage_free/auc_with_features.csv",
        help="AUC with features CSV.",
    )
    parser.add_argument(
        "--lobo-summary",
        default="analysis/leakage_free/prediction_lobo_summary.csv",
        help="LOBO summary CSV.",
    )
    parser.add_argument(
        "--lobo-rank-summary",
        default="analysis/leakage_free/prediction_lobo_rank_summary.csv",
        help="LOBO ranking summary CSV.",
    )
    parser.add_argument(
        "--lobo-rank-baselines",
        default="analysis/leakage_free/prediction_lobo_rank_baselines.csv",
        help="Baseline selector ranking CSV.",
    )
    parser.add_argument(
        "--lobo-mixed-summary",
        default="analysis/leakage_free/prediction_lobo_mixed_summary.csv",
        help="LOBO MixedLM summary CSV.",
    )
    parser.add_argument(
        "--loto-summary",
        default="analysis/leakage_free/prediction_loto_summary.csv",
        help="LOTO summary CSV.",
    )
    parser.add_argument(
        "--loto-rank-summary",
        default="analysis/leakage_free/prediction_loto_rank_summary.csv",
        help="LOTO ranking summary CSV.",
    )
    parser.add_argument(
        "--loto-mixed-summary",
        default="analysis/leakage_free/prediction_loto_mixed_summary.csv",
        help="LOTO MixedLM summary CSV.",
    )
    parser.add_argument(
        "--target",
        default="auc_normalized",
        help="Target column name.",
    )
    parser.add_argument(
        "--prediction-target",
        default=None,
        help="Target used for LOBO/LOTO predictions (optional).",
    )
    parser.add_argument(
        "--linear-model",
        choices=["ols", "ridge"],
        default="ols",
        help="Linear model for summary (default: ols).",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge penalty strength when linear-model=ridge.",
    )
    parser.add_argument(
        "--predictors",
        default=(
            "flow_train_to_eval_coverage_logit,flow_eval_to_train_coverage_logit,"
            "resnet_train_to_eval_coverage_logit,resnet_eval_to_train_coverage_logit,"
            "flow_mmd,feature_mmd"
        ),
        help="Comma-separated predictor columns.",
    )
    parser.add_argument(
        "--output-file",
        default="analysis/leakage_free/summary_report.txt",
        help="Summary output file.",
    )
    parser.add_argument(
        "--within-benchmark-slopes",
        default="analysis/leakage_free/within_benchmark_slopes.csv",
        help="Within-benchmark slope CSV.",
    )
    args = parser.parse_args()

    out_lines = []
    out_lines.append("LEAKAGE-FREE SUMMARY")
    out_lines.append("=" * 80)

    auc_path = Path(args.auc_table)
    if not auc_path.exists():
        out_lines.append(f"Missing AUC table: {auc_path}")
        Path(args.output_file).write_text("\n".join(out_lines))
        return

    df = pd.read_csv(auc_path)
    predictors = [p.strip() for p in args.predictors.split(",") if p.strip()]
    predictors, missing, all_nan, constant, redundant = _filter_predictors(df, predictors)

    if args.target not in df.columns:
        out_lines.append(f"Target '{args.target}' not found in {auc_path}")
        Path(args.output_file).write_text("\n".join(out_lines))
        return
    if missing:
        out_lines.append(f"Dropped missing predictors: {', '.join(missing)}")
    if all_nan:
        out_lines.append(f"Dropped NaN-only predictors: {', '.join(all_nan)}")
    if constant:
        out_lines.append(f"Dropped constant predictors: {', '.join(constant)}")
    if redundant:
        out_lines.append(f"Dropped redundant predictors: {', '.join(redundant)}")
    if not predictors:
        out_lines.append("No valid predictors remain after filtering.")
        Path(args.output_file).write_text("\n".join(out_lines))
        return

    df["benchmark"] = df["benchmark"].astype(str)
    df["train_dataset"] = df["train_dataset"].astype(str)
    df["pretrained"] = _to_bool(df["pretrained"]) if "pretrained" in df.columns else np.nan
    df["freeze"] = _to_bool(df["freeze"]) if "freeze" in df.columns else np.nan

    n_rows = len(df)
    n_runs = df["run_id"].nunique() if "run_id" in df.columns else np.nan
    benchmarks = sorted(df["benchmark"].dropna().unique())
    train_datasets = sorted(df["train_dataset"].dropna().unique())

    out_lines.append(f"Rows: {n_rows}")
    out_lines.append(f"Runs: {n_runs}")
    out_lines.append(f"Benchmarks ({len(benchmarks)}): {', '.join(benchmarks)}")
    out_lines.append(f"Train datasets ({len(train_datasets)}): {', '.join(train_datasets)}")

    if "pretrained" in df.columns and "freeze" in df.columns:
        out_lines.append("")
        out_lines.append("Encoder config counts (pretrained, freeze):")
        for (pre, frz), group in df.groupby(["pretrained", "freeze"], dropna=False):
            out_lines.append(f"  {pre}/{frz}: {len(group)} rows")
    if "encoder_config" in df.columns:
        out_lines.append("")
        out_lines.append("Encoder config counts (FF/FT/TF/TT):")
        for name, group in df.groupby("encoder_config", dropna=False):
            out_lines.append(f"  {name}: {len(group)} rows")

    out_lines.append("")
    out_lines.append(f"Target: {args.target}")
    if args.prediction_target and args.prediction_target != args.target:
        out_lines.append(f"Prediction target: {args.prediction_target}")
    out_lines.append(f"Predictors: {', '.join(predictors)}")

    out_lines.append("")
    out_lines.append("Overall predictor signal (pairwise correlations):")
    for pred in predictors:
        if pred not in df.columns:
            out_lines.append(f"  {pred}: missing")
            continue
        sub = df[[pred, args.target]].dropna()
        pear = _pearson(sub[pred], sub[args.target]) if not sub.empty else np.nan
        spear = _spearman(sub[pred], sub[args.target]) if not sub.empty else np.nan
        out_lines.append(f"  {pred}: Pearson={pear:.3f}, Spearman={spear:.3f}")

    out_lines.append("")
    model_label = "OLS" if args.linear_model == "ols" else f"Ridge (alpha={args.ridge_alpha})"
    out_lines.append(f"Standardized {model_label} (all data):")
    model = _fit_standardized_model(
        df,
        predictors,
        args.target,
        model=args.linear_model,
        ridge_alpha=args.ridge_alpha,
    )
    if model is None:
        out_lines.append("  Not enough complete rows to fit model.")
    else:
        out_lines.append(f"  N={model['n']}")
        if "r2" in model:
            out_lines.append(f"  R2={model['r2']:.3f}")
        params = model["params"]
        pvals = model.get("pvalues")
        for pred in predictors:
            coef = params.get(pred, np.nan)
            pval = pvals.get(pred, np.nan) if pvals else np.nan
            out_lines.append("  " + _format_coef(pred, coef, pval))

    if "pretrained" in df.columns and "freeze" in df.columns:
        out_lines.append("")
        out_lines.append(f"Encoder-config-specific signal (standardized {model_label}):")
        config_models = {}
        for (pre, frz), group in df.groupby(["pretrained", "freeze"], dropna=False):
            label = f"pretrained={pre}, freeze={frz}"
            model = _fit_standardized_model(
                group,
                predictors,
                args.target,
                model=args.linear_model,
                ridge_alpha=args.ridge_alpha,
            )
            if model is None or model["n"] < len(predictors) + 5:
                out_lines.append(f"  {label}: insufficient rows (n={0 if model is None else model['n']})")
                continue
            config_models[(pre, frz)] = model
            out_lines.append(f"  {label}: n={model['n']}")
            params = model["params"]
            pvals = model.get("pvalues")
            for pred in predictors:
                coef = params.get(pred, np.nan)
                pval = pvals.get(pred, np.nan) if pvals else np.nan
                out_lines.append("    " + _format_coef(pred, coef, pval))

        if (False, False) in config_models and (True, True) in config_models:
            compare_candidates = [
                "resnet_train_to_eval_mean_dist",
                "resnet_eval_to_train_mean_dist",
                "resnet_train_to_eval_coverage_logit",
                "resnet_eval_to_train_coverage_logit",
                "resnet_train_to_eval_coverage",
                "resnet_eval_to_train_coverage",
            ]
            compare_name = None
            for name in compare_candidates:
                if (
                    name in config_models[(False, False)]["params"]
                    and name in config_models[(True, True)]["params"]
                ):
                    compare_name = name
                    break
            if compare_name:
                a = config_models[(False, False)]["params"].get(compare_name, np.nan)
                b = config_models[(True, True)]["params"].get(compare_name, np.nan)
                if not np.isnan(a) and not np.isnan(b):
                    out_lines.append("")
                    out_lines.append(
                        f"ResNet predictor comparison ({compare_name}): "
                        f"not-pretrained+unfrozen={a:+.3f}, pretrained+frozen={b:+.3f}"
                    )

    out_lines.append("")
    out_lines.append("Prediction validation (LOBO):")
    lobo_path = Path(args.lobo_summary)
    if lobo_path.exists():
        lobo_df = pd.read_csv(lobo_path)
        out_lines.extend("  " + line for line in summarize_predictions(lobo_df, "benchmark"))
    else:
        out_lines.append(f"  Missing: {lobo_path}")

    lobo_rank_path = Path(args.lobo_rank_summary)
    if lobo_rank_path.exists():
        lobo_rank_df = pd.read_csv(lobo_rank_path)
        flow_family = ["kitti2012", "kitti2015", "middlebury", "flyingthings", "pointodyssey"]
        semantic_family = ["spair", "pfpascal", "pfwillow", "tss"]
        overall = lobo_rank_df[lobo_rank_df["benchmark"] == "__overall__"]
        if not overall.empty:
            row = overall.iloc[0]
            out_lines.append(
                "  Rank@benchmark (mean): "
                f"top1={row['top1']:.2f}, top3={row['top3']:.2f}, "
                f"regret={row['regret']:.2f}, spearman={row['spearman']:.2f}"
            )
        per_benchmark = lobo_rank_df[lobo_rank_df["benchmark"] != "__overall__"]
        flow_summary = _rank_family_summary(per_benchmark, flow_family)
        semantic_summary = _rank_family_summary(per_benchmark, semantic_family)
        if flow_summary or semantic_summary:
            out_lines.append(
                "  Families: flow={kitti2012,kitti2015,middlebury,flyingthings,pointodyssey} "
                "semantic={spair,pfpascal,pfwillow,tss} (synthetic excluded)"
            )
        if flow_summary:
            out_lines.append("  " + _format_rank_metrics("Rank@benchmark (flow family)", flow_summary))
        if semantic_summary:
            out_lines.append("  " + _format_rank_metrics("Rank@benchmark (semantic family)", semantic_summary))

        baseline_path = Path(args.lobo_rank_baselines)
        if baseline_path.exists():
            baseline_df = pd.read_csv(baseline_path)
            baseline_overall = baseline_df[baseline_df["benchmark"] == "__overall__"]
            if not baseline_overall.empty:
                out_lines.append("  Baseline selectors (overall):")
                baseline_order = [
                    "flow_train_to_eval_mean_dist",
                    "flow_eval_to_train_mean_dist",
                    "resnet_train_to_eval_mean_dist",
                    "resnet_eval_to_train_mean_dist",
                    "dino_train_to_eval_mean_dist",
                    "dino_eval_to_train_mean_dist",
                    "flow_train_to_eval_coverage_logit",
                    "resnet_train_to_eval_coverage_logit",
                    "dino_train_to_eval_coverage_logit",
                    "flow_mmd",
                    "feature_mmd",
                    "dino_mmd",
                    "always_flyingthings",
                    "always_best_avg",
                ]
                for selector in baseline_order:
                    row = baseline_overall[baseline_overall["selector"] == selector]
                    if row.empty:
                        continue
                    out_lines.append(
                        "    " + _format_rank_metrics(selector, row.iloc[0].to_dict())
                    )

            baseline_per_benchmark = baseline_df[baseline_df["benchmark"] != "__overall__"]
            flow_name, flow_baseline = _select_baseline_selector(
                baseline_per_benchmark,
                ["flow_train_to_eval_mean_dist", "flow_mmd", "flow_eval_to_train_mean_dist"],
            )
            semantic_name, semantic_baseline = _select_baseline_selector(
                baseline_per_benchmark,
                [
                    "feature_mmd",
                    "dino_mmd",
                    "resnet_train_to_eval_mean_dist",
                    "dino_train_to_eval_mean_dist",
                    "dino_eval_to_train_mean_dist",
                    "resnet_eval_to_train_mean_dist",
                ],
            )
            flow_base_summary = _rank_family_summary(flow_baseline, flow_family)
            semantic_base_summary = _rank_family_summary(semantic_baseline, semantic_family)
            if flow_base_summary or semantic_base_summary:
                out_lines.append("  Baseline selectors (families):")
            if flow_base_summary:
                out_lines.append(
                    "    "
                    + _format_rank_metrics(f"{flow_name} (flow family)", flow_base_summary)
                )
        if semantic_base_summary:
            out_lines.append(
                "    " + _format_rank_metrics(f"{semantic_name} (semantic family)", semantic_base_summary)
            )


    lobo_mixed_path = Path(args.lobo_mixed_summary)
    if lobo_mixed_path.exists():
        out_lines.append("Prediction validation (LOBO, MixedLM):")
        lobo_mixed_df = pd.read_csv(lobo_mixed_path)
        out_lines.extend("  " + line for line in summarize_predictions(lobo_mixed_df, "benchmark"))

    out_lines.append("")
    out_lines.append("Prediction validation (LOTO):")
    loto_path = Path(args.loto_summary)
    if loto_path.exists():
        loto_df = pd.read_csv(loto_path)
        group_col = "train_dataset_group" if "train_dataset_group" in loto_df.columns else "train_dataset"
        out_lines.extend("  " + line for line in summarize_predictions(loto_df, group_col))
    else:
        out_lines.append(f"  Missing: {loto_path}")

    loto_rank_path = Path(args.loto_rank_summary)
    if loto_rank_path.exists():
        loto_rank_df = pd.read_csv(loto_rank_path)
        overall = loto_rank_df[loto_rank_df["benchmark"] == "__overall__"]
        if not overall.empty:
            row = overall.iloc[0]
            out_lines.append(
                "  Rank@benchmark (LOTO mean): "
                f"top1={row['top1']:.2f}, top3={row['top3']:.2f}, "
                f"regret={row['regret']:.2f}, spearman={row['spearman']:.2f}"
            )

    loto_mixed_path = Path(args.loto_mixed_summary)
    if loto_mixed_path.exists():
        out_lines.append("Prediction validation (LOTO, MixedLM):")
        loto_mixed_df = pd.read_csv(loto_mixed_path)
        group_col = "train_dataset_group" if "train_dataset_group" in loto_mixed_df.columns else "train_dataset"
        out_lines.extend("  " + line for line in summarize_predictions(loto_mixed_df, group_col))

    slopes_path = Path(args.within_benchmark_slopes)
    if slopes_path.exists():
        slopes_df = pd.read_csv(slopes_path)
        if not slopes_df.empty:
            out_lines.append("")
            out_lines.append("Within-benchmark slope consistency (standardized OLS):")
            for pred in [p for p in predictors if p in slopes_df.columns]:
                signs = slopes_df[pred].dropna()
                if signs.empty:
                    continue
                pos = int((signs > 0).sum())
                neg = int((signs < 0).sum())
                out_lines.append(f"  {pred}: +{pos} / -{neg} across benchmarks")

    out_lines.append("")
    out_lines.append("Takeaways (auto-generated):")
    if model is None:
        out_lines.append("  - Not enough data to identify strongest predictors.")
    else:
        params = model["params"]
        sorted_preds = sorted(
            [(pred, abs(params.get(pred, np.nan)), params.get(pred, np.nan)) for pred in predictors],
            key=lambda x: (np.nan_to_num(x[1]),),
            reverse=True,
        )
        top = [p for p in sorted_preds if not np.isnan(p[1])]
        if top:
            best = ", ".join(f"{p[0]} ({p[2]:+.2f} std)" for p in top[:2])
            out_lines.append(f"  - Strongest standardized predictors: {best}.")
        neg = [p for p in top if p[2] < 0]
        if neg:
            worst = ", ".join(f"{p[0]} ({p[2]:+.2f} std)" for p in neg[:2])
            out_lines.append(f"  - Negative predictors (higher -> lower target): {worst}.")

    output_path = Path(args.output_file)
    output_path.write_text("\n".join(out_lines))
    print(f"Wrote summary to {output_path}")


if __name__ == "__main__":
    main()
