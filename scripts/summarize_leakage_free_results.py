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


def _bootstrap_corr(x, y, method, n_boot=200, seed=17):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        xs = x[idx]
        ys = y[idx]
        if method == "pearson":
            val = _pearson(xs, ys)
        else:
            val = _spearman(xs, ys)
        if np.isnan(val):
            continue
        stats.append(val)
    if not stats:
        return (np.nan, np.nan)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return (float(lo), float(hi))


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


def _predictor_family(name):
    if name.startswith("flow_") or name == "flow_mmd":
        return "flow"
    if name.startswith("resnet_") or name.startswith("dino_"):
        return "semantic"
    if name in ("feature_mmd", "dino_mmd"):
        return "semantic"
    return "other"


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


def summarize_predictions_with_ci(summary_df, rows_df, label_col, top_n=3):
    if summary_df.empty:
        return []
    overall = summary_df[summary_df[label_col] == "__overall__"]
    rows = []
    if not overall.empty:
        row = overall.iloc[0]
        pearson = row.get("pearson", np.nan)
        spearman = row.get("spearman", np.nan)
        pearson_ci = (np.nan, np.nan)
        spearman_ci = (np.nan, np.nan)
        include_ci = False
        if rows_df is not None and not rows_df.empty:
            rows_df = rows_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["prediction", "target"])
            if not rows_df.empty:
                pred = rows_df["prediction"].to_numpy(dtype=float)
                target = rows_df["target"].to_numpy(dtype=float)
                pearson = _pearson(pred, target)
                spearman = _spearman(pred, target)
                pearson_ci = _bootstrap_corr(pred, target, "pearson")
                spearman_ci = _bootstrap_corr(pred, target, "spearman")
                include_ci = True
        if include_ci:
            rows.append(
                f"Overall: MAE={row['mae']:.2f}, RMSE={row['rmse']:.2f}, "
                f"Pearson={pearson:.3f} [{pearson_ci[0]:.3f},{pearson_ci[1]:.3f}], "
                f"Spearman={spearman:.3f} [{spearman_ci[0]:.3f},{spearman_ci[1]:.3f}]"
            )
        else:
            rows.append(
                f"Overall: MAE={row['mae']:.2f}, RMSE={row['rmse']:.2f}, "
                f"Pearson={pearson:.3f}, Spearman={spearman:.3f}"
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
    parts = [
        f"top1={row['top1']:.2f}",
        f"top3={row['top3']:.2f}",
    ]
    if "topk" in row and not pd.isna(row.get("topk")):
        topk_label = "topk"
        if "topk_frac" in row and not pd.isna(row.get("topk_frac")):
            topk_label = f"top{int(round(row['topk_frac'] * 100))}%"
        elif "topk_k" in row and not pd.isna(row.get("topk_k")):
            topk_label = f"top{int(round(row['topk_k']))}"
        parts.append(f"{topk_label}={row['topk']:.2f}")
    parts.append(f"regret={row['regret']:.2f}")
    if "mean_abs_rank_error" in row and not pd.isna(row.get("mean_abs_rank_error")):
        parts.append(f"rank_abs_err={row['mean_abs_rank_error']:.2f}")
    if "mean_abs_rank_pct_error" in row and not pd.isna(row.get("mean_abs_rank_pct_error")):
        parts.append(f"rank_pct_err={row['mean_abs_rank_pct_error']:.2f}")
    parts.append(f"spearman={spearman_str}")
    return f"{prefix}: " + ", ".join(parts)


def _rank_family_summary(df, benchmarks):
    if df is None or df.empty:
        return None
    sub = df[df["benchmark"].isin(benchmarks)].copy()
    if sub.empty:
        return None
    summary = {
        "top1": float(sub["top1"].mean()),
        "top3": float(sub["top3"].mean()),
        "regret": float(sub["regret"].mean()),
        "spearman": float(sub["spearman"].mean()),
    }
    if "topk" in sub.columns:
        summary["topk"] = float(sub["topk"].mean())
    if "topk_k" in sub.columns:
        summary["topk_k"] = float(sub["topk_k"].mean())
    if "topk_frac" in sub.columns:
        summary["topk_frac"] = float(sub["topk_frac"].mean())
    if "mean_abs_rank_error" in sub.columns:
        summary["mean_abs_rank_error"] = float(sub["mean_abs_rank_error"].mean())
    if "mean_abs_rank_pct_error" in sub.columns:
        summary["mean_abs_rank_pct_error"] = float(sub["mean_abs_rank_pct_error"].mean())
    return summary


def _bootstrap_rank_ci(df, metrics, n_boot=500, seed=17):
    if df is None or df.empty:
        return {}
    rng = np.random.default_rng(seed)
    rows = df[df["benchmark"] != "__overall__"]
    if rows.empty:
        return {}
    values = {m: [] for m in metrics}
    idx = np.arange(len(rows))
    for _ in range(n_boot):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        sample = rows.iloc[sample_idx]
        for metric in metrics:
            if metric not in sample.columns:
                continue
            values[metric].append(float(sample[metric].mean()))
    ci = {}
    for metric, vals in values.items():
        if not vals:
            continue
        lo, hi = np.percentile(vals, [2.5, 97.5])
        ci[metric] = (float(lo), float(hi))
    return ci


def _format_rank_ci(prefix, ci_map):
    if not ci_map:
        return f"{prefix}: n/a"
    parts = []
    for key in ("top1", "top3", "topk", "regret", "spearman"):
        if key not in ci_map:
            continue
        lo, hi = ci_map[key]
        parts.append(f"{key}=[{lo:.2f},{hi:.2f}]")
    if not parts:
        return f"{prefix}: n/a"
    return f"{prefix}: " + ", ".join(parts)


def _coverage_fraction(df, cols):
    if not cols:
        return np.nan
    return float(df[cols].notna().all(axis=1).mean())


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
        "--lobo-rows",
        default=None,
        help="LOBO prediction rows CSV (optional).",
    )
    parser.add_argument(
        "--lobo-rank-summary",
        default="analysis/leakage_free/prediction_lobo_rank_summary.csv",
        help="LOBO ranking summary CSV.",
    )
    parser.add_argument(
        "--lobo-rank-detail",
        default=None,
        help="LOBO ranking detail CSV (optional).",
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
        "--lobo-mixed-rows",
        default=None,
        help="LOBO MixedLM prediction rows CSV (optional).",
    )
    parser.add_argument(
        "--loto-summary",
        default="analysis/leakage_free/prediction_loto_summary.csv",
        help="LOTO summary CSV.",
    )
    parser.add_argument(
        "--loto-rows",
        default=None,
        help="LOTO prediction rows CSV (optional).",
    )
    parser.add_argument(
        "--loto-rank-summary",
        default="analysis/leakage_free/prediction_loto_rank_summary.csv",
        help="LOTO ranking summary CSV.",
    )
    parser.add_argument(
        "--loto-rank-detail",
        default=None,
        help="LOTO ranking detail CSV (optional).",
    )
    parser.add_argument(
        "--loto-mixed-summary",
        default="analysis/leakage_free/prediction_loto_mixed_summary.csv",
        help="LOTO MixedLM summary CSV.",
    )
    parser.add_argument(
        "--loto-mixed-rows",
        default=None,
        help="LOTO MixedLM prediction rows CSV (optional).",
    )
    parser.add_argument(
        "--lobo-permutation-summary",
        default=None,
        help="LOBO permutation summary CSV (optional).",
    )
    parser.add_argument(
        "--lobo-permutation-rank-summary",
        default=None,
        help="LOBO permutation rank summary CSV (optional).",
    )
    parser.add_argument(
        "--loto-permutation-summary",
        default=None,
        help="LOTO permutation summary CSV (optional).",
    )
    parser.add_argument(
        "--loto-permutation-rank-summary",
        default=None,
        help="LOTO permutation rank summary CSV (optional).",
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

    output_dir_path = Path(args.output_dir)
    inferred_dir = Path(args.output_file).parent if args.output_file else output_dir_path
    if output_dir_path == Path("analysis/leakage_free") and inferred_dir != output_dir_path:
        args.output_dir = str(inferred_dir)

    def _resolve_path(attr, filename):
        if getattr(args, attr) is None:
            setattr(args, attr, str(Path(args.output_dir) / filename))

    def _swap_default(attr, filename):
        current = Path(getattr(args, attr))
        default_path = Path("analysis/leakage_free") / filename
        if current == default_path and Path(args.output_dir) != default_path.parent:
            setattr(args, attr, str(Path(args.output_dir) / filename))

    _swap_default("auc_table", "auc_with_features.csv")
    _swap_default("lobo_summary", "prediction_lobo_summary.csv")
    _swap_default("lobo_rank_summary", "prediction_lobo_rank_summary.csv")
    _swap_default("lobo_rank_baselines", "prediction_lobo_rank_baselines.csv")
    _swap_default("lobo_mixed_summary", "prediction_lobo_mixed_summary.csv")
    _swap_default("loto_summary", "prediction_loto_summary.csv")
    _swap_default("loto_rank_summary", "prediction_loto_rank_summary.csv")
    _swap_default("loto_mixed_summary", "prediction_loto_mixed_summary.csv")
    _swap_default("within_benchmark_slopes", "within_benchmark_slopes.csv")

    _resolve_path("lobo_rows", "prediction_lobo_rows.csv")
    _resolve_path("lobo_rank_detail", "prediction_lobo_rank_detail.csv")
    _resolve_path("lobo_mixed_rows", "prediction_lobo_mixed_rows.csv")
    _resolve_path("loto_rows", "prediction_loto_rows.csv")
    _resolve_path("loto_rank_detail", "prediction_loto_rank_detail.csv")
    _resolve_path("loto_mixed_rows", "prediction_loto_mixed_rows.csv")
    _resolve_path("lobo_permutation_summary", "prediction_lobo_permutation_summary.csv")
    _resolve_path("lobo_permutation_rank_summary", "prediction_lobo_permutation_rank_summary.csv")
    _resolve_path("loto_permutation_summary", "prediction_loto_permutation_summary.csv")
    _resolve_path("loto_permutation_rank_summary", "prediction_loto_permutation_rank_summary.csv")

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

    flow_family = ["kitti2012", "kitti2015", "middlebury", "flyingthings", "pointodyssey"]
    semantic_family = ["spair", "pfpascal", "pfwillow", "tss"]
    flow_cols = [p for p in predictors if _predictor_family(p) == "flow"]
    semantic_cols = [p for p in predictors if _predictor_family(p) == "semantic"]
    if flow_cols or semantic_cols:
        out_lines.append("")
        out_lines.append("Predictor coverage by benchmark (fraction rows with all predictors present):")
        header = f"{'benchmark':<14} {'n':>4} {'flow':>7} {'semantic':>9}"
        out_lines.append("  " + header)
        out_lines.append("  " + "-" * len(header))
        for bench in benchmarks:
            sub = df[df["benchmark"] == bench]
            flow_frac = _coverage_fraction(sub, flow_cols) if flow_cols else np.nan
            sem_frac = _coverage_fraction(sub, semantic_cols) if semantic_cols else np.nan
            flow_str = f"{flow_frac:.2f}" if not np.isnan(flow_frac) else "n/a"
            sem_str = f"{sem_frac:.2f}" if not np.isnan(sem_frac) else "n/a"
            out_lines.append(f"  {bench:<14} {len(sub):>4} {flow_str:>7} {sem_str:>9}")

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

    headline_lines = []
    lobo_summary_path = Path(args.lobo_summary)
    lobo_rows_path = Path(args.lobo_rows)
    if lobo_summary_path.exists():
        lobo_df = pd.read_csv(lobo_summary_path)
        lobo_rows = pd.read_csv(lobo_rows_path) if lobo_rows_path.exists() else None
        lines = summarize_predictions_with_ci(lobo_df, lobo_rows, "benchmark")
        if lines:
            headline_lines.append("LOBO pred: " + lines[0].replace("Overall: ", ""))
    lobo_rank_path = Path(args.lobo_rank_summary)
    if lobo_rank_path.exists():
        lobo_rank_df = pd.read_csv(lobo_rank_path)
        overall = lobo_rank_df[lobo_rank_df["benchmark"] == "__overall__"]
        if not overall.empty:
            headline_lines.append(_format_rank_metrics("LOBO rank", overall.iloc[0].to_dict()))
            ci = _bootstrap_rank_ci(
                lobo_rank_df,
                ["top1", "top3", "topk", "regret", "spearman"],
            )
            headline_lines.append(_format_rank_ci("LOBO rank 95% CI", ci))
    loto_summary_path = Path(args.loto_summary)
    loto_rows_path = Path(args.loto_rows)
    if loto_summary_path.exists():
        loto_df = pd.read_csv(loto_summary_path)
        loto_rows = pd.read_csv(loto_rows_path) if loto_rows_path.exists() else None
        group_col = "train_dataset_group" if "train_dataset_group" in loto_df.columns else "train_dataset"
        lines = summarize_predictions_with_ci(loto_df, loto_rows, group_col)
        if lines:
            headline_lines.append("LOTO pred: " + lines[0].replace("Overall: ", ""))
    loto_rank_path = Path(args.loto_rank_summary)
    if loto_rank_path.exists():
        loto_rank_df = pd.read_csv(loto_rank_path)
        overall = loto_rank_df[loto_rank_df["benchmark"] == "__overall__"]
        if not overall.empty:
            headline_lines.append(_format_rank_metrics("LOTO rank", overall.iloc[0].to_dict()))
            ci = _bootstrap_rank_ci(
                loto_rank_df,
                ["top1", "top3", "topk", "regret", "spearman"],
            )
            headline_lines.append(_format_rank_ci("LOTO rank 95% CI", ci))

    if headline_lines:
        out_lines.append("")
        out_lines.append("Headline metrics:")
        out_lines.extend(f"  {line}" for line in headline_lines)

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

        if "r2" in model and not np.isnan(model["r2"]):
            base_r2 = model["r2"]
            drop_rows = []
            for pred in predictors:
                reduced = [p for p in predictors if p != pred]
                if not reduced:
                    continue
                reduced_model = _fit_standardized_model(
                    df,
                    reduced,
                    args.target,
                    model=args.linear_model,
                    ridge_alpha=args.ridge_alpha,
                )
                if reduced_model is None or "r2" not in reduced_model:
                    continue
                reduced_r2 = reduced_model["r2"]
                if np.isnan(reduced_r2):
                    continue
                drop_rows.append((pred, base_r2 - reduced_r2))
            if drop_rows:
                drop_rows.sort(key=lambda x: x[1], reverse=True)
                out_lines.append("")
                out_lines.append("Predictor drop-one sensitivity (delta R2):")
                for pred, delta in drop_rows[:5]:
                    out_lines.append(f"  {pred}: {delta:+.3f}")
        family_scores = {}
        for pred in predictors:
            coef = params.get(pred, np.nan)
            if np.isnan(coef):
                continue
            family = _predictor_family(pred)
            entry = family_scores.setdefault(family, {"abs_sum": 0.0, "count": 0, "top": (pred, abs(coef))})
            abs_coef = abs(coef)
            entry["abs_sum"] += abs_coef
            entry["count"] += 1
            if abs_coef > entry["top"][1]:
                entry["top"] = (pred, abs_coef)
        if family_scores:
            total_abs = sum(val["abs_sum"] for val in family_scores.values()) or 1.0
            out_lines.append("")
            out_lines.append(f"Predictor family importance (standardized {model_label}):")
            for family in sorted(family_scores.keys()):
                entry = family_scores[family]
                share = entry["abs_sum"] / total_abs
                top_pred, top_val = entry["top"]
                out_lines.append(
                    f"  {family}: abs_sum={entry['abs_sum']:.3f}, share={share:.2f}, top={top_pred} ({top_val:.3f})"
                )

    if "benchmark" in df.columns:
        out_lines.append("")
        out_lines.append(f"Family-specific {model_label} (standardized):")
        for name, family in (("flow family", flow_family), ("semantic family", semantic_family)):
            sub = df[df["benchmark"].isin(family)]
            model = _fit_standardized_model(
                sub,
                predictors,
                args.target,
                model=args.linear_model,
                ridge_alpha=args.ridge_alpha,
            )
            if model is None or model["n"] < len(predictors) + 5:
                out_lines.append(f"  {name}: insufficient rows (n={0 if model is None else model['n']})")
                continue
            out_lines.append(f"  {name}: n={model['n']}")
            if "r2" in model and not np.isnan(model["r2"]):
                out_lines.append(f"    R2={model['r2']:.3f}")
            params = model["params"]
            pvals = model.get("pvalues")
            for pred in predictors:
                coef = params.get(pred, np.nan)
                pval = pvals.get(pred, np.nan) if pvals else np.nan
                out_lines.append("    " + _format_coef(pred, coef, pval))

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
        lobo_rows_path = Path(args.lobo_rows)
        lobo_rows = pd.read_csv(lobo_rows_path) if lobo_rows_path.exists() else None
        out_lines.extend(
            "  " + line for line in summarize_predictions_with_ci(lobo_df, lobo_rows, "benchmark")
        )
    else:
        out_lines.append(f"  Missing: {lobo_path}")

    lobo_rank_path = Path(args.lobo_rank_summary)
    if lobo_rank_path.exists():
        lobo_rank_df = pd.read_csv(lobo_rank_path)
        overall = lobo_rank_df[lobo_rank_df["benchmark"] == "__overall__"]
        if not overall.empty:
            row = overall.iloc[0]
            out_lines.append(
                "  " + _format_rank_metrics("Rank@benchmark (mean)", row.to_dict())
            )
            ci = _bootstrap_rank_ci(
                lobo_rank_df,
                ["top1", "top3", "topk", "regret", "spearman"],
            )
            out_lines.append("  " + _format_rank_ci("Rank@benchmark 95% CI", ci))
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
        lobo_mixed_rows_path = Path(args.lobo_mixed_rows)
        lobo_mixed_rows = (
            pd.read_csv(lobo_mixed_rows_path) if lobo_mixed_rows_path.exists() else None
        )
        out_lines.extend(
            "  "
            + line
            for line in summarize_predictions_with_ci(lobo_mixed_df, lobo_mixed_rows, "benchmark")
        )

    out_lines.append("")
    out_lines.append("Prediction validation (LOTO):")
    loto_path = Path(args.loto_summary)
    if loto_path.exists():
        loto_df = pd.read_csv(loto_path)
        group_col = "train_dataset_group" if "train_dataset_group" in loto_df.columns else "train_dataset"
        loto_rows_path = Path(args.loto_rows)
        loto_rows = pd.read_csv(loto_rows_path) if loto_rows_path.exists() else None
        out_lines.extend(
            "  " + line for line in summarize_predictions_with_ci(loto_df, loto_rows, group_col)
        )
    else:
        out_lines.append(f"  Missing: {loto_path}")

    loto_rank_path = Path(args.loto_rank_summary)
    if loto_rank_path.exists():
        loto_rank_df = pd.read_csv(loto_rank_path)
        overall = loto_rank_df[loto_rank_df["benchmark"] == "__overall__"]
        if not overall.empty:
            row = overall.iloc[0]
            out_lines.append(
                "  " + _format_rank_metrics("Rank@benchmark (LOTO mean)", row.to_dict())
            )
            ci = _bootstrap_rank_ci(
                loto_rank_df,
                ["top1", "top3", "topk", "regret", "spearman"],
            )
            out_lines.append("  " + _format_rank_ci("Rank@benchmark 95% CI", ci))

    loto_mixed_path = Path(args.loto_mixed_summary)
    if loto_mixed_path.exists():
        out_lines.append("Prediction validation (LOTO, MixedLM):")
        loto_mixed_df = pd.read_csv(loto_mixed_path)
        group_col = "train_dataset_group" if "train_dataset_group" in loto_mixed_df.columns else "train_dataset"
        loto_mixed_rows_path = Path(args.loto_mixed_rows)
        loto_mixed_rows = (
            pd.read_csv(loto_mixed_rows_path) if loto_mixed_rows_path.exists() else None
        )
        out_lines.extend(
            "  "
            + line
            for line in summarize_predictions_with_ci(loto_mixed_df, loto_mixed_rows, group_col)
        )

    perm_lines = []
    perm_lobo_path = Path(args.lobo_permutation_summary)
    if perm_lobo_path.exists():
        perm_lobo_df = pd.read_csv(perm_lobo_path)
        perm_lines.append("LOBO permuted:")
        perm_lines.extend("  " + line for line in summarize_predictions_with_ci(perm_lobo_df, None, "benchmark"))
        perm_lobo_rank_path = Path(args.lobo_permutation_rank_summary)
        if perm_lobo_rank_path.exists():
            perm_rank_df = pd.read_csv(perm_lobo_rank_path)
            overall = perm_rank_df[perm_rank_df["benchmark"] == "__overall__"]
            if not overall.empty:
                perm_lines.append(
                    "  " + _format_rank_metrics("Rank@benchmark (permuted mean)", overall.iloc[0].to_dict())
                )
    perm_loto_path = Path(args.loto_permutation_summary)
    if perm_loto_path.exists():
        perm_loto_df = pd.read_csv(perm_loto_path)
        perm_lines.append("LOTO permuted:")
        group_col = "train_dataset_group" if "train_dataset_group" in perm_loto_df.columns else "train_dataset"
        perm_lines.extend("  " + line for line in summarize_predictions_with_ci(perm_loto_df, None, group_col))
        perm_loto_rank_path = Path(args.loto_permutation_rank_summary)
        if perm_loto_rank_path.exists():
            perm_rank_df = pd.read_csv(perm_loto_rank_path)
            overall = perm_rank_df[perm_rank_df["benchmark"] == "__overall__"]
            if not overall.empty:
                perm_lines.append(
                    "  " + _format_rank_metrics("Rank@benchmark (permuted mean)", overall.iloc[0].to_dict())
                )
    if perm_lines:
        out_lines.append("")
        out_lines.append("Permutation sanity check:")
        out_lines.extend("  " + line for line in perm_lines)

    slopes_path = Path(args.within_benchmark_slopes)
    if slopes_path.exists():
        slopes_df = pd.read_csv(slopes_path)
        if not slopes_df.empty:
            flow_family = ["kitti2012", "kitti2015", "middlebury", "flyingthings", "pointodyssey"]
            semantic_family = ["spair", "pfpascal", "pfwillow", "tss"]
            out_lines.append("")
            out_lines.append("Within-benchmark slope consistency (standardized OLS):")
            for pred in [p for p in predictors if p in slopes_df.columns]:
                signs = slopes_df[pred].dropna()
                if signs.empty:
                    continue
                pos = int((signs > 0).sum())
                neg = int((signs < 0).sum())
                out_lines.append(f"  {pred}: +{pos} / -{neg} across benchmarks")

            def _write_family_slope_summary(title, benchmarks):
                family_df = slopes_df[slopes_df["benchmark"].isin(benchmarks)]
                if family_df.empty:
                    return
                out_lines.append("")
                out_lines.append(f"Within-benchmark slope consistency ({title}):")
                for pred in [p for p in predictors if p in family_df.columns]:
                    signs = family_df[pred].dropna()
                    if signs.empty:
                        continue
                    pos = int((signs > 0).sum())
                    neg = int((signs < 0).sum())
                    out_lines.append(f"  {pred}: +{pos} / -{neg} across benchmarks")

            _write_family_slope_summary("flow family", flow_family)
            _write_family_slope_summary("semantic family", semantic_family)

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
