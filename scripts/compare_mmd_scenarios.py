#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _fmt(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "NA"
    try:
        return f"{float(val):.3f}"
    except Exception:
        return str(val)


def _rank_df(df: pd.DataFrame, metric: str, lower_better: bool) -> pd.DataFrame:
    if df is None or df.empty or metric not in df.columns:
        return pd.DataFrame()
    return df.sort_values(metric, ascending=lower_better)


def _filter_target(df: pd.DataFrame, targets: List[str]) -> pd.DataFrame:
    if not targets or df is None or df.empty or "target" not in df.columns:
        return df
    return df[df["target"].astype(str).isin(targets)].copy()


def _best_row(df: pd.DataFrame, metric: str, lower_better: bool) -> Optional[pd.Series]:
    ranked = _rank_df(df, metric, lower_better)
    if ranked.empty:
        return None
    return ranked.iloc[0]


def _load_summary(path: Path, fold: str) -> Optional[pd.DataFrame]:
    return _read_csv(path / f"prediction_{fold}_summary.csv")

def _load_rows(path: Path, fold: str) -> Optional[pd.DataFrame]:
    return _read_csv(path / f"prediction_{fold}_rows.csv")


def _combo_components(method: str) -> List[str]:
    if not method:
        return []
    name = str(method)
    if name.startswith("combo_"):
        return [c for c in name[len("combo_") :].split("__") if c]
    return [name]


def _is_strict_only(method: str, kind: str) -> bool:
    # kind: "train" or "eval"
    if not method:
        return False
    name = str(method)
    target = f"{kind}_only"
    other = "eval_only" if kind == "train" else "train_only"
    if other in name:
        return False
    comps = _combo_components(name)
    if not comps:
        return False
    # Strict: every component must be tagged with the target suffix.
    return all(target in c for c in comps)

def _pick_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            series = pd.to_numeric(df[c], errors="coerce")
            if series.notna().any():
                return c
    return None


def _valid_cols(df: Optional[pd.DataFrame], candidates: List[str], max_cols: Optional[int] = None) -> List[str]:
    if df is None or df.empty:
        return []
    out: List[str] = []
    for c in candidates:
        if c in df.columns:
            series = pd.to_numeric(df[c], errors="coerce")
            if series.notna().any():
                out.append(c)
                if max_cols is not None and len(out) >= max_cols:
                    break
    return out


def _directional_predictors(row: Optional[pd.Series], direction: str) -> List[str]:
    if row is None:
        return []
    preds = row.get("predictors", "")
    if preds is None or (isinstance(preds, float) and pd.isna(preds)):
        return []
    parts = [p.strip() for p in str(preds).split(",") if p.strip()]
    if direction == "eval":
        return [p for p in parts if "eval_to_train" in p]
    if direction == "train":
        return [p for p in parts if "train_to_eval" in p]
    return []


def _group_mean(df: pd.DataFrame, key_cols: Union[str, List[str]], cols: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    keys = [key_cols] if isinstance(key_cols, str) else list(key_cols)
    cols = [c for c in cols if c in df.columns]
    if not cols or any(k not in df.columns for k in keys):
        return pd.DataFrame()
    return df.groupby(keys)[cols].mean(numeric_only=True).reset_index()


def _value_with_stats(
    stats_df: pd.DataFrame,
    key_cols: Union[str, List[str]],
    group_vals: Dict[str, Any],
    col: str,
) -> Optional[tuple]:
    if stats_df is None or stats_df.empty or col not in stats_df.columns:
        return None
    keys = [key_cols] if isinstance(key_cols, str) else list(key_cols)
    sub = stats_df
    for k in keys:
        if k not in sub.columns:
            return None
        sub = sub[sub[k] == group_vals.get(k)]
    if sub.empty:
        return None
    val = float(sub[col].iloc[0])
    mean = float(stats_df[col].mean())
    std = float(stats_df[col].std(ddof=0))
    z = (val - mean) / std if std > 0 else float("nan")
    return val, mean, z


def _line_from_row(label: str, row: Optional[pd.Series], metric: str) -> str:
    if row is None:
        return f"- {label}: n/a"
    extras = []
    for col in ["jointood_mae", "loto_mae", "lobo_mae", "jointood_rank_spearman"]:
        if col in row.index and col != metric:
            extras.append(f"{col}={_fmt(row.get(col))}")
    extras_txt = ", " + ", ".join(extras) if extras else ""
    return (
        f"- {label}: `{row.get('method','NA')}` "
        f"({metric}={_fmt(row.get(metric))}, "
        f"n_predictors={row.get('n_predictors','NA')}{extras_txt})"
    )


def _format_group(row: pd.Series, key_cols: List[str]) -> str:
    parts = []
    for k in key_cols:
        label = "eval_dataset" if k == "benchmark" else k
        parts.append(f"{label}={row.get(k, 'NA')}")
    return ", ".join(parts)


def _group_mae_from_rows(rows_df: Optional[pd.DataFrame], key_cols: List[str]) -> pd.DataFrame:
    if rows_df is None or rows_df.empty or any(k not in rows_df.columns for k in key_cols):
        return pd.DataFrame()
    df = rows_df.copy()
    df["abs_err"] = (df["prediction"] - df["target"]).abs()
    grp = df.groupby(key_cols)["abs_err"].mean().reset_index()
    return grp.rename(columns={"abs_err": "mae"})


def _metric_slug(name: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in str(name))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_").lower() or "metric"


def _build_parameter_matched_table(
    mmd_row: Optional[pd.Series],
    parity_ranked: pd.DataFrame,
    rank_metric: str,
    rank_lower_better: bool,
    match_field: str,
) -> pd.DataFrame:
    cols = [
        "role",
        "method",
        "family",
        "symmetry",
        "model",
        "n_predictors",
        "n_predictors_base",
        match_field,
        "match_gap",
        rank_metric,
        "delta_mmd_minus_method",
        "improves_over_mmd",
        "jointood_mae",
        "jointood_spearman",
        "jointood_rank_spearman",
        "jointood_rank_pairwise_cindex",
        "jointood_rank_kendall_tau",
        "loto_rank_spearman",
        "loto_rank_pairwise_cindex",
        "loto_rank_kendall_tau",
        "lobo_rank_spearman",
        "lobo_rank_pairwise_cindex",
        "lobo_rank_kendall_tau",
        "loto_mae",
        "lobo_mae",
        "path",
    ]
    if mmd_row is None:
        return pd.DataFrame(columns=cols)

    mmd_metric = mmd_row.get(rank_metric)
    mmd_match_val = mmd_row.get(match_field)
    mmd_match_num = int(mmd_match_val) if pd.notna(mmd_match_val) else None

    rows: List[Dict[str, Any]] = []

    def _row_payload(row: pd.Series, role: str) -> Dict[str, Any]:
        method_metric = row.get(rank_metric)
        d_mmd_minus_method = float("nan")
        improves = float("nan")
        if pd.notna(mmd_metric) and pd.notna(method_metric):
            d_mmd_minus_method = float(mmd_metric) - float(method_metric)
            # Positive means candidate is better than MMD regardless of metric direction.
            improves = d_mmd_minus_method if rank_lower_better else -d_mmd_minus_method

        match_gap = float("nan")
        this_match = row.get(match_field)
        if mmd_match_num is not None and pd.notna(this_match):
            match_gap = abs(int(this_match) - mmd_match_num)

        out = {
            "role": role,
            "method": row.get("method"),
            "family": row.get("family"),
            "symmetry": row.get("symmetry"),
            "model": row.get("model"),
            "n_predictors": row.get("n_predictors"),
            "n_predictors_base": row.get("n_predictors_base"),
            match_field: row.get(match_field),
            "match_gap": match_gap,
            rank_metric: method_metric,
            "delta_mmd_minus_method": d_mmd_minus_method,
            "improves_over_mmd": improves,
            "jointood_mae": row.get("jointood_mae"),
            "jointood_spearman": row.get("jointood_spearman"),
            "jointood_rank_spearman": row.get("jointood_rank_spearman"),
            "jointood_rank_pairwise_cindex": row.get("jointood_rank_pairwise_cindex"),
            "jointood_rank_kendall_tau": row.get("jointood_rank_kendall_tau"),
            "loto_rank_spearman": row.get("loto_rank_spearman"),
            "loto_rank_pairwise_cindex": row.get("loto_rank_pairwise_cindex"),
            "loto_rank_kendall_tau": row.get("loto_rank_kendall_tau"),
            "lobo_rank_spearman": row.get("lobo_rank_spearman"),
            "lobo_rank_pairwise_cindex": row.get("lobo_rank_pairwise_cindex"),
            "lobo_rank_kendall_tau": row.get("lobo_rank_kendall_tau"),
            "loto_mae": row.get("loto_mae"),
            "lobo_mae": row.get("lobo_mae"),
            "path": row.get("path"),
        }
        return out

    rows.append(_row_payload(mmd_row, "mmd_baseline"))
    if parity_ranked is not None and not parity_ranked.empty:
        for _, r in parity_ranked.iterrows():
            rows.append(_row_payload(r, "matched_non_mmd"))

    out_df = pd.DataFrame(rows)
    # Keep only columns that exist/non-empty for readability, preserving order.
    seen = set()
    keep_cols = []
    for c in cols:
        if c in seen:
            continue
        seen.add(c)
        if c in out_df.columns:
            keep_cols.append(c)
    out_df = out_df[keep_cols]
    return out_df

def _add_aggregate_score(df: pd.DataFrame, metrics: List[str], lower_better: List[str]) -> List[str]:
    used = [m for m in metrics if m in df.columns]
    if not used:
        df["aggregate_score"] = float("nan")
        return []
    ranks = []
    for metric in used:
        series = df[metric]
        if metric in lower_better:
            rank = series.rank(ascending=False, pct=True)
        else:
            rank = series.rank(ascending=True, pct=True)
        ranks.append(rank)
    rank_df = pd.concat(ranks, axis=1)
    complete_mask = rank_df.notna().all(axis=1)
    df["aggregate_score"] = float("nan")
    df.loc[complete_mask, "aggregate_score"] = rank_df[complete_mask].mean(axis=1)
    return used


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MMD scenarios for symmetric vs asymmetric story.")
    parser.add_argument("--root", required=True, help="Output root (e.g., analysis_comprehensive_runs/hof_motion_v3)")
    parser.add_argument("--target-filter", default="", help="Comma-separated targets to include.")
    parser.add_argument("--metric", default="loto_mae", help="Primary metric (default: loto_mae).")
    parser.add_argument("--lower-better", action="store_true", help="Treat metric as lower-is-better.")
    parser.add_argument(
        "--model-filter",
        default="",
        help="Optional model filter (e.g., pairwise_rank, ols, ridge). Empty disables filtering.",
    )
    parser.add_argument(
        "--mmd-baseline-method",
        default="",
        help="Optional exact MMD baseline method to anchor parameter matching "
        "(e.g., mmd_flow_only_pairwise, mmd_dino_only_pairwise, mmd_only_pairwise).",
    )
    parser.add_argument(
        "--non-mmd-family-filter",
        default="",
        help="Optional non-MMD family filter for comparison pool "
        "(e.g., flow, appearance, motion). Empty disables filtering.",
    )
    parser.add_argument(
        "--include-pairwise",
        action="store_true",
        help="Include methods containing '_pairwise'.",
    )
    parser.add_argument(
        "--aggregate-metrics",
        default="loto_mae,lobo_mae",
        help="Comma-separated metrics for aggregate score (default: loto_mae,lobo_mae).",
    )
    parser.add_argument(
        "--aggregate-lower-better",
        default="loto_mae,lobo_mae",
        help="Comma-separated metrics where lower is better for aggregate scoring.",
    )
    parser.add_argument(
        "--use-aggregate",
        action="store_true",
        help="Use aggregate_score for ranking instead of --metric.",
    )
    parser.add_argument("--match-field", choices=["n_predictors", "n_predictors_base"], default="n_predictors")
    parser.add_argument("--match-window", type=int, default=0, help="Allowed absolute difference for parity match.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for parity match listing.")
    parser.add_argument("--fold", choices=["loto", "lobo"], default="loto", help="Fold for failure analysis (default: loto).")
    parser.add_argument(
        "--group-by",
        choices=["train_dataset", "eval_dataset", "train_eval_pair"],
        default="train_dataset",
        help="Grouping key for failure analysis and case studies.",
    )
    parser.add_argument("--failure-top-k", type=int, default=5, help="Top-k failure examples to show.")
    parser.add_argument("--failure-margin", type=float, default=0.5, help="Margin to label failure mode (default: 0.5 MAE).")
    parser.add_argument(
        "--strict-directional",
        action="store_true",
        help="Require all combo components to be eval_only/train_only for failure mode hints.",
    )
    parser.add_argument(
        "--predictor-eval-cols",
        default=(
            "flow_eval_to_train_mean_dist_over_radius_eval,"
            "flow_eval_to_train_mean_dist,"
            "flow_eval_to_train_eps1px,"
            "flow_eval_to_train_eps_at50,"
            "flow_eval_to_train_auc,"
            "flow_eval_to_train_coverage,"
            "dino_eval_to_train_mean_dist"
        ),
        help="Comma-separated priority list for eval->train predictor column.",
    )
    parser.add_argument(
        "--predictor-train-cols",
        default=(
            "flow_train_to_eval_mean_dist_over_radius_train,"
            "flow_train_to_eval_mean_dist,"
            "flow_train_to_eval_eps1px,"
            "flow_train_to_eval_eps_at50,"
            "flow_train_to_eval_auc,"
            "flow_train_to_eval_coverage,"
            "dino_train_to_eval_mean_dist"
        ),
        help="Comma-separated priority list for train->eval predictor column.",
    )
    parser.add_argument(
        "--predictor-margin",
        type=float,
        default=0.0,
        help="Margin for predictor-based mode decision (default: 0.0).",
    )
    parser.add_argument(
        "--case-top-k",
        type=int,
        default=3,
        help="Number of detailed case studies to print (default: 3).",
    )
    parser.add_argument("--output", default=None, help="Output text file (default: <root>/mmd_comparison.txt)")
    parser.add_argument(
        "--table-csv",
        default=None,
        help="Optional output CSV for parameter-matched MMD comparison table "
        "(default: <root>/mmd_parameter_matched_<metric>.csv).",
    )
    parser.add_argument(
        "--table-latex",
        default=None,
        help="Optional output LaTeX table for parameter-matched MMD comparison "
        "(default: <root>/mmd_parameter_matched_<metric>.tex).",
    )
    parser.add_argument(
        "--latex-caption",
        default="MMD vs parameter-matched non-MMD comparison.",
        help="Caption for --table-latex output.",
    )
    parser.add_argument(
        "--latex-label",
        default="tab:mmd_parameter_matched",
        help="Label for --table-latex output.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    method_summary = _read_csv(root / "method_summary.csv")
    if method_summary is None:
        raise SystemExit("Missing method_summary.csv")

    targets = [t.strip() for t in args.target_filter.split(",") if t.strip()]
    df = _filter_target(method_summary, targets)

    if args.model_filter.strip() and "model" in df.columns:
        df = df[df["model"].astype(str) == args.model_filter.strip()].copy()

    # Historical default was to exclude pairwise rows; keep that unless explicitly requested.
    if not args.include_pairwise:
        df = df[~df["method"].astype(str).str.contains("_pairwise", regex=False)]

    aggregate_metrics = [m.strip() for m in args.aggregate_metrics.split(",") if m.strip()]
    aggregate_lower = [m.strip() for m in args.aggregate_lower_better.split(",") if m.strip()]
    aggregate_used = _add_aggregate_score(df, aggregate_metrics, aggregate_lower)

    rank_metric = "aggregate_score" if args.use_aggregate else args.metric
    rank_lower_better = False if args.use_aggregate else args.lower_better

    # Define sets
    combos = df[df["method"].astype(str).str.startswith("combo_")]
    combos_mmd = combos[combos["method"].astype(str).str.contains("mmd", regex=False)]
    combos_no_mmd = combos[~combos["method"].astype(str).str.contains("mmd", regex=False)]

    method_series = df["method"].astype(str)
    if args.mmd_baseline_method.strip():
        mmd_only = df[method_series == args.mmd_baseline_method.strip()]
        if mmd_only.empty:
            raise SystemExit(
                f"No rows found for --mmd-baseline-method={args.mmd_baseline_method!r} "
                f"after current filters."
            )
    else:
        mmd_only = df[method_series.isin(["mmd_only", "mmd_only_pairwise"])]
        if mmd_only.empty:
            mmd_only = df[method_series.str.contains("mmd_only", regex=False)]
    non_mmd = df[~df["method"].astype(str).str.contains("mmd", regex=False)]
    if args.non_mmd_family_filter.strip() and "family" in non_mmd.columns:
        non_mmd = non_mmd[
            non_mmd["family"].astype(str) == args.non_mmd_family_filter.strip()
        ].copy()

    best_combo_no_mmd = _best_row(combos_no_mmd, rank_metric, rank_lower_better)
    best_combo_mmd = _best_row(combos_mmd, rank_metric, rank_lower_better)
    best_mmd_only = _best_row(mmd_only, rank_metric, rank_lower_better)

    # Parameter-matched (relative to mmd_only)
    match_vals = []
    if not mmd_only.empty and args.match_field in mmd_only.columns:
        val = mmd_only.iloc[0].get(args.match_field)
        if pd.notna(val):
            match_vals = [int(val)]
    parity = non_mmd
    if match_vals and args.match_field in parity.columns:
        parity = parity[parity[args.match_field].apply(lambda v: pd.notna(v) and any(abs(int(v) - mv) <= args.match_window for mv in match_vals))]

    parity_ranked = _rank_df(parity, rank_metric, rank_lower_better).head(args.top_k)

    lines: List[str] = []
    lines.append(f"# MMD Comparison Report for {root}")
    if targets:
        lines.append("Target filter: " + ", ".join(targets))
    if args.use_aggregate:
        lines.append("Metric: aggregate_score (mean percentile rank)")
        lines.append("Aggregate uses: " + ", ".join(aggregate_used) if aggregate_used else "Aggregate uses: n/a")
    else:
        lines.append(f"Metric: {args.metric} (lower_better={args.lower_better})")
    lines.append("")

    lines.append("## Best Combos (With vs Without MMD)")
    lines.append(_line_from_row("Best combo without MMD", best_combo_no_mmd, rank_metric))
    lines.append(_line_from_row("Best combo with MMD", best_combo_mmd, rank_metric))
    lines.append("")

    lines.append("## MMD Only vs Parameter-Matched Non-MMD")
    lines.append(_line_from_row("MMD only", best_mmd_only, rank_metric))
    if parity_ranked is None or parity_ranked.empty:
        lines.append("- Parameter-matched non-MMD: n/a")
    else:
        for _, row in parity_ranked.iterrows():
            lines.append(
                f"- Matched non-MMD: `{row.get('method','NA')}` "
                f"({rank_metric}={_fmt(row.get(rank_metric))}, "
                f"loto_mae={_fmt(row.get('loto_mae'))}, lobo_mae={_fmt(row.get('lobo_mae'))}, "
                f"{args.match_field}={row.get(args.match_field,'NA')})"
            )

    # Failure mode hints: compare eval_only vs train_only summaries against mmd_only
    lines.append("")
    lines.append("## Failure Mode Hints (Directional Asymmetry vs MMD)")
    lines.append(f"Fold: {args.fold}")
    if args.strict_directional:
        lines.append("Directional selection: strict (all combo components must be eval_only/train_only)")

    # Grouping configuration
    if args.group_by == "train_dataset":
        key_cols = ["train_dataset"]
    elif args.group_by == "eval_dataset":
        key_cols = ["benchmark"]
    else:
        key_cols = ["train_dataset", "benchmark"]

    methods_series = non_mmd["method"].astype(str)
    if args.strict_directional:
        eval_only_df = non_mmd[methods_series.map(lambda m: _is_strict_only(m, "eval"))]
        train_only_df = non_mmd[methods_series.map(lambda m: _is_strict_only(m, "train"))]
        # Fallback if strict filters remove everything.
        if eval_only_df.empty:
            eval_only_df = non_mmd[methods_series.str.contains("eval_only", regex=False)]
        if train_only_df.empty:
            train_only_df = non_mmd[methods_series.str.contains("train_only", regex=False)]
    else:
        eval_only_df = non_mmd[methods_series.str.contains("eval_only", regex=False)]
        train_only_df = non_mmd[methods_series.str.contains("train_only", regex=False)]
    best_eval_only = _best_row(eval_only_df, rank_metric, rank_lower_better)
    best_train_only = _best_row(train_only_df, rank_metric, rank_lower_better)

    lines.append(_line_from_row("Best eval_only", best_eval_only, rank_metric))
    lines.append(_line_from_row("Best train_only", best_train_only, rank_metric))
    lines.append(_line_from_row("MMD only", best_mmd_only, rank_metric))

    if best_mmd_only is None or best_eval_only is None or best_train_only is None:
        lines.append("- Not enough data for failure mode hints.")
    else:
        mmd_path = Path(best_mmd_only.get("path", ""))
        eval_path = Path(best_eval_only.get("path", ""))
        train_path = Path(best_train_only.get("path", ""))
        use_rows = args.group_by != "train_dataset"
        mmd_rows = _load_rows(mmd_path, args.fold)
        eval_rows = _load_rows(eval_path, args.fold)
        train_rows = _load_rows(train_path, args.fold)
        if use_rows:
            mmd_sum = _group_mae_from_rows(mmd_rows, key_cols)
            eval_sum = _group_mae_from_rows(eval_rows, key_cols)
            train_sum = _group_mae_from_rows(train_rows, key_cols)
        else:
            mmd_sum = _load_summary(mmd_path, args.fold)
            eval_sum = _load_summary(eval_path, args.fold)
            train_sum = _load_summary(train_path, args.fold)

        if mmd_sum is None or eval_sum is None or train_sum is None:
            lines.append("- Missing prediction summaries for failure mode hints.")
        else:
            if any(k not in mmd_sum.columns for k in key_cols) or any(k not in eval_sum.columns for k in key_cols) or any(k not in train_sum.columns for k in key_cols):
                lines.append("- Summary/rows do not share a common group key.")
            else:
                merged = (
                    mmd_sum[key_cols + ["mae"]].rename(columns={"mae": "mmd_mae"})
                    .merge(eval_sum[key_cols + ["mae"]].rename(columns={"mae": "eval_mae"}), on=key_cols, how="inner")
                    .merge(train_sum[key_cols + ["mae"]].rename(columns={"mae": "train_mae"}), on=key_cols, how="inner")
                )
                if merged.empty:
                    lines.append("- No overlapping groups for failure mode hints.")
                else:
                    margin = float(args.failure_margin)
                    def _mode(row):
                        if row.eval_mae + margin < row.train_mae:
                            return "eval_under_covered"
                        if row.train_mae + margin < row.eval_mae:
                            return "train_extra_mass"
                        return "ambiguous"
                    merged["mode"] = merged.apply(_mode, axis=1)
                    merged["eval_minus_train"] = merged["eval_mae"] - merged["train_mae"]
                    merged["delta_best"] = merged["mmd_mae"] - merged[["eval_mae", "train_mae"]].min(axis=1)
                    counts = merged["mode"].value_counts().to_dict()
                    lines.append(
                        f"- Mode counts: eval_under_covered={counts.get('eval_under_covered',0)}, "
                        f"train_extra_mass={counts.get('train_extra_mass',0)}, ambiguous={counts.get('ambiguous',0)}"
                    )
                    lines.append(
                        "- Interpretation: eval_under_covered => eval_only << train_only; "
                        "train_extra_mass => train_only << eval_only."
                    )
                    # Predictor-based evidence
                    use_rows_predictors = args.group_by != "train_dataset"
                    if use_rows_predictors:
                        mmd_feat = mmd_rows
                        eval_feat = eval_rows
                        train_feat = train_rows
                    else:
                        mmd_feat = _read_csv(mmd_path / "auc_with_features.csv")
                        eval_feat = _read_csv(eval_path / "auc_with_features.csv")
                        train_feat = _read_csv(train_path / "auc_with_features.csv")
                    eval_cols_fallback = [c.strip() for c in args.predictor_eval_cols.split(",") if c.strip()]
                    train_cols_fallback = [c.strip() for c in args.predictor_train_cols.split(",") if c.strip()]
                    # Prefer predictors from the best eval/train methods (directional), fallback to defaults.
                    eval_candidates = _directional_predictors(best_eval_only, "eval")
                    train_candidates = _directional_predictors(best_train_only, "train")
                    eval_cols_sel = _valid_cols(eval_feat, eval_candidates, max_cols=3)
                    train_cols_sel = _valid_cols(train_feat, train_candidates, max_cols=3)
                    if not eval_cols_sel:
                        eval_cols_sel = _valid_cols(eval_feat, eval_cols_fallback, max_cols=3)
                    if not train_cols_sel:
                        train_cols_sel = _valid_cols(train_feat, train_cols_fallback, max_cols=3)
                    eval_col = eval_cols_sel[0] if eval_cols_sel else None
                    train_col = train_cols_sel[0] if train_cols_sel else None
                    if eval_col is None or train_col is None:
                        eval_cols_sel = eval_cols_sel or _valid_cols(mmd_feat, eval_cols_fallback, max_cols=3)
                        train_cols_sel = train_cols_sel or _valid_cols(mmd_feat, train_cols_fallback, max_cols=3)
                        eval_col = eval_cols_sel[0] if eval_cols_sel else None
                        train_col = train_cols_sel[0] if train_cols_sel else None

                    if eval_col is None or train_col is None:
                        lines.append("- Predictor evidence: missing predictor columns in auc_with_features.csv.")
                    else:
                        # Try to build predictor group from eval/train features separately, then merge.
                        eval_group = _group_mean(eval_feat, key_cols, [eval_col]) if eval_feat is not None else pd.DataFrame()
                        train_group = _group_mean(train_feat, key_cols, [train_col]) if train_feat is not None else pd.DataFrame()
                        pred_group = pd.DataFrame()
                        if not eval_group.empty and not train_group.empty:
                            pred_group = eval_group.merge(train_group, on=key_cols, how="inner")
                        else:
                            # Fallback to a single source containing both columns.
                            pred_df = None
                            for candidate in [eval_feat, train_feat, mmd_feat]:
                                if candidate is None or any(k not in candidate.columns for k in key_cols):
                                    continue
                                if eval_col in candidate.columns and train_col in candidate.columns:
                                    pred_df = candidate
                                    break
                            if pred_df is not None:
                                pred_group = pred_df.groupby(key_cols)[[eval_col, train_col]].mean(numeric_only=True).reset_index()
                        if pred_group.empty:
                            lines.append("- Predictor evidence: missing or non-overlapping predictor values.")
                            pred_group = None
                        if pred_group is not None:
                            # Convert to mismatch scores: higher is worse. If coverage, invert.
                            def _score(col, val):
                                if col.endswith("coverage"):
                                    return 1.0 - val
                                return val
                            pred_group["eval_score"] = pred_group[eval_col].apply(lambda v: _score(eval_col, v))
                            pred_group["train_score"] = pred_group[train_col].apply(lambda v: _score(train_col, v))
                            pred_group["pred_delta"] = pred_group["eval_score"] - pred_group["train_score"]
                            pmargin = float(args.predictor_margin)
                            def _pred_mode(row):
                                if row.pred_delta > pmargin:
                                    return "eval_under_covered"
                                if row.pred_delta < -pmargin:
                                    return "train_extra_mass"
                                return "ambiguous"
                            pred_group["pred_mode"] = pred_group.apply(_pred_mode, axis=1)
                            top = merged.sort_values("delta_best", ascending=False).head(args.failure_top_k)
                            lines.append(f"- Predictor columns: eval={eval_col}, train={train_col}")
                            for _, row in top.iterrows():
                                match = pred_group
                                for k in key_cols:
                                    match = match[match[k] == row[k]]
                                if match.empty:
                                    lines.append(
                                        f"- {_format_group(row, key_cols)}: mmd_mae={_fmt(row.mmd_mae)}, "
                                        f"eval_mae={_fmt(row.eval_mae)}, train_mae={_fmt(row.train_mae)}, "
                                        f"eval-train={_fmt(row.eval_minus_train)}, best_delta={_fmt(row.delta_best)}, "
                                        f"mode={row['mode']}, pred_mode=n/a"
                                    )
                                    continue
                                pr = match.iloc[0]
                                lines.append(
                                    f"- {_format_group(row, key_cols)}: mmd_mae={_fmt(row.mmd_mae)}, "
                                    f"eval_mae={_fmt(row.eval_mae)}, train_mae={_fmt(row.train_mae)}, "
                                    f"eval-train={_fmt(row.eval_minus_train)}, best_delta={_fmt(row.delta_best)}, "
                                    f"mode={row['mode']}, pred_mode={pr['pred_mode']}, "
                                    f"eval_score={_fmt(pr['eval_score'])}, train_score={_fmt(pr['train_score'])}"
                                )

                    # Detailed case studies with model predictions + predictor values
                    lines.append("")
                    lines.append("## Failure Case Studies (Predictor Evidence)")
                    case_rows = merged.sort_values("delta_best", ascending=False).head(args.case_top_k)

                    # Load prediction rows for model predictions
                    if mmd_rows is None:
                        mmd_rows = _load_rows(mmd_path, args.fold)
                    if eval_rows is None:
                        eval_rows = _load_rows(eval_path, args.fold)
                    if train_rows is None:
                        train_rows = _load_rows(train_path, args.fold)

                    def _pred_stats(rows_df: Optional[pd.DataFrame]) -> pd.DataFrame:
                        if rows_df is None or any(k not in rows_df.columns for k in key_cols):
                            return pd.DataFrame()
                        if "prediction" not in rows_df.columns or "target" not in rows_df.columns:
                            return pd.DataFrame()
                        grp = rows_df.groupby(key_cols)[["prediction", "target"]].mean(numeric_only=True).reset_index()
                        grp = grp.rename(columns={"prediction": "pred_mean", "target": "target_mean"})
                        return grp

                    mmd_pred = _pred_stats(mmd_rows)
                    eval_pred = _pred_stats(eval_rows)
                    train_pred = _pred_stats(train_rows)

                    mmd_cols = [c for c in ["flow_mmd", "dino_mmd", "feature_mmd"] if mmd_feat is not None and c in mmd_feat.columns]
                    mmd_stats = _group_mean(mmd_feat, key_cols, mmd_cols) if mmd_feat is not None else pd.DataFrame()
                    eval_stats = _group_mean(eval_feat, key_cols, eval_cols_sel) if eval_feat is not None else pd.DataFrame()
                    train_stats = _group_mean(train_feat, key_cols, train_cols_sel) if train_feat is not None else pd.DataFrame()

                    for _, row in case_rows.iterrows():
                        lines.append(f"- Example: {_format_group(row, key_cols)}")
                        group_vals = {k: row.get(k) for k in key_cols}

                        def _emit_model(label: str, pred_df: pd.DataFrame, metric_row: pd.Series, feat_stats: pd.DataFrame, cols: List[str]):
                            if pred_df.empty:
                                lines.append(f"  - {label}: predictions n/a")
                                return
                            sub = pred_df
                            for k in key_cols:
                                sub = sub[sub[k] == group_vals.get(k)]
                            if sub.empty:
                                lines.append(f"  - {label}: predictions n/a")
                                return
                            pred_mean = _fmt(sub["pred_mean"].iloc[0])
                            target_mean = _fmt(sub["target_mean"].iloc[0])
                            mae_key = f"{label.lower()}_mae"
                            mae_val = _fmt(metric_row.get(mae_key, float("nan"))) if label.lower() in ["mmd", "eval", "train"] else "NA"
                            lines.append(f"  - {label} model: pred_mean={pred_mean}, target_mean={target_mean}, mae={mae_val}")
                            for col in cols:
                                vals = _value_with_stats(feat_stats, key_cols, group_vals, col)
                                if vals is None:
                                    lines.append(f"    - {col}: n/a")
                                else:
                                    val, mean, z = vals
                                    lines.append(f"    - {col}: {val:.4f} (mean={mean:.4f}, z={_fmt(z)})")

                        _emit_model("MMD", mmd_pred, row, mmd_stats, mmd_cols)
                        _emit_model("Eval", eval_pred, row, eval_stats, eval_cols_sel if eval_cols_sel else ([eval_col] if eval_col else []))
                        _emit_model("Train", train_pred, row, train_stats, train_cols_sel if train_cols_sel else ([train_col] if train_col else []))

    output_path = Path(args.output) if args.output else root / "mmd_comparison.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Wrote {output_path}")

    metric_slug = _metric_slug(rank_metric)
    table_csv_path = (
        Path(args.table_csv)
        if args.table_csv
        else root / f"mmd_parameter_matched_{metric_slug}.csv"
    )
    table_tex_path = (
        Path(args.table_latex)
        if args.table_latex
        else root / f"mmd_parameter_matched_{metric_slug}.tex"
    )
    table_df = _build_parameter_matched_table(
        mmd_row=best_mmd_only,
        parity_ranked=parity_ranked,
        rank_metric=rank_metric,
        rank_lower_better=rank_lower_better,
        match_field=args.match_field,
    )
    if not table_df.empty:
        table_csv_path.parent.mkdir(parents=True, exist_ok=True)
        table_df.to_csv(table_csv_path, index=False)
        print(f"Wrote {table_csv_path}")

        # Drop internal path from paper-facing LaTeX table by default.
        tex_df = table_df.copy()
        if "path" in tex_df.columns:
            tex_df = tex_df.drop(columns=["path"])
        tex_body = tex_df.to_latex(index=False, escape=True, na_rep="--")
        tex_text = "\n".join(
            [
                "\\begin{table}[t]",
                "\\centering",
                f"\\caption{{{args.latex_caption}}}",
                f"\\label{{{args.latex_label}}}",
                "\\resizebox{\\linewidth}{!}{%",
                tex_body.rstrip(),
                "}",
                "\\end{table}",
                "",
            ]
        )
        table_tex_path.parent.mkdir(parents=True, exist_ok=True)
        table_tex_path.write_text(tex_text)
        print(f"Wrote {table_tex_path}")
    else:
        print("No parameter-matched rows; skipped table CSV/LaTeX outputs.")


if __name__ == "__main__":
    main()
