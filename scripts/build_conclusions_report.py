#!/usr/bin/env python3
"""
Write a human-readable conclusions report from hypothesis/final tables.

Usage:
  python scripts/build_conclusions_report.py --root analysis_comprehensive_runs/hof_motion_v3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception:
        return None


def _fmt(val) -> str:
    try:
        if pd.isna(val):
            return "NA"
        if isinstance(val, (int, float)):
            return f"{val:.3f}"
    except Exception:
        pass
    return str(val)


def _top_n(df: pd.DataFrame, metric: str, n: int = 5, lower_better: Optional[List[str]] = None) -> pd.DataFrame:
    if metric not in df.columns:
        return pd.DataFrame()
    lower_better = lower_better or []
    return df.sort_values(metric, ascending=(metric in lower_better)).head(n)


def _add_aggregate_score(df: pd.DataFrame, metrics: List[str], lower_better: List[str]) -> List[str]:
    used = [m for m in metrics if m in df.columns]
    if not used:
        df["aggregate_score"] = float("nan")
        return []
    ranks = []
    for metric in used:
        series = df[metric]
        if metric in lower_better:
            # Lower is better: smaller values get higher percentiles.
            rank = series.rank(ascending=False, pct=True)
        else:
            # Higher is better: larger values get higher percentiles.
            rank = series.rank(ascending=True, pct=True)
        ranks.append(rank)
    if ranks:
        rank_df = pd.concat(ranks, axis=1)
        complete_mask = rank_df.notna().all(axis=1)
        df["aggregate_score"] = float("nan")
        df.loc[complete_mask, "aggregate_score"] = rank_df[complete_mask].mean(axis=1)
    else:
        df["aggregate_score"] = float("nan")
    return used


def _top_by_param_count(
    df: pd.DataFrame,
    metric: str,
    k: int = 3,
    lower_better: Optional[List[str]] = None,
    extra_metrics: Optional[List[str]] = None,
) -> List[str]:
    lines: List[str] = []
    if df is None or df.empty:
        return lines
    if "n_predictors" not in df.columns or metric not in df.columns:
        return lines
    grouped = df.dropna(subset=["n_predictors", metric]).copy()
    if grouped.empty:
        return lines
    grouped["n_predictors"] = grouped["n_predictors"].astype(int)
    lower_better = lower_better or []
    extra_metrics = extra_metrics or []
    for n_pred, sub in grouped.groupby("n_predictors"):
        sub = sub.sort_values(metric, ascending=(metric in lower_better)).head(k)
        entries = []
        for _, row in sub.iterrows():
            method = row.get("method", "NA")
            agg_val = _fmt(row.get(metric, float("nan")))
            metric_parts = [f"agg_score={agg_val}"]
            for m in extra_metrics:
                metric_parts.append(f"{m}={_fmt(row.get(m, float('nan')))}")
            entries.append(f"`{method}` ({', '.join(metric_parts)})")
        joined = ", ".join(entries) if entries else "NA"
        lines.append(f"- n_predictors={n_pred}: {joined}")
    return lines


def _find_row(df: pd.DataFrame, method: str) -> Optional[pd.Series]:
    if "method" not in df.columns:
        return None
    sub = df[df["method"] == method]
    if sub.empty:
        return None
    return sub.iloc[0]


def _family_tags(method: str) -> List[str]:
    if not method:
        return []
    name = str(method)
    families = []
    if "hof_density" in name:
        families.append("density")
    if "flow_" in name:
        families.append("flow")
    if "hof_" in name and "hof_density" not in name:
        families.append("hof")
    if "dino_" in name:
        families.append("dino")
    if "mmd" in name:
        families.append("mmd")
    return families


def _family_parity_by_param(
    df: pd.DataFrame,
    metric: str,
    families: List[str],
    lower_better: Optional[List[str]] = None,
    allow_combo: bool = False,
    counts: Optional[List[int]] = None,
) -> List[str]:
    lines: List[str] = []
    if df is None or df.empty:
        return lines
    if "n_predictors" not in df.columns or metric not in df.columns or "method" not in df.columns:
        return lines
    work = df.dropna(subset=["n_predictors", metric]).copy()
    if work.empty:
        return lines
    work["n_predictors"] = work["n_predictors"].astype(int)
    work["family_tags"] = work["method"].map(_family_tags)
    lower_better = lower_better or []
    grouped = {int(n_pred): sub for n_pred, sub in work.groupby("n_predictors")}
    if counts is None:
        counts = sorted(grouped.keys())
    for n_pred in counts:
        sub = grouped.get(int(n_pred))
        if sub is None or sub.empty:
            entries = [f"{fam}: n/a" for fam in families]
            lines.append(f"- n_predictors={n_pred}: " + ", ".join(entries))
            continue
        best_by_family = {}
        for fam in families:
            if allow_combo:
                fam_sub = sub[sub["family_tags"].apply(lambda tags: fam in tags)]
            else:
                fam_sub = sub[sub["family_tags"].apply(lambda tags: len(tags) == 1 and tags[0] == fam)]
                fam_sub = fam_sub[~fam_sub["method"].astype(str).str.startswith("combo_")]
            if fam_sub.empty:
                continue
            fam_sub = fam_sub.sort_values(metric, ascending=(metric in lower_better))
            best_by_family[fam] = fam_sub.iloc[0]
        if not best_by_family:
            entries = [f"{fam}: n/a" for fam in families]
            lines.append(f"- n_predictors={n_pred}: " + ", ".join(entries))
            continue
        # Determine best overall within this n_predictors for delta computation.
        if metric in lower_better:
            best_row = min(best_by_family.values(), key=lambda r: r.get(metric, float("inf")))
        else:
            best_row = max(best_by_family.values(), key=lambda r: r.get(metric, float("-inf")))
        best_val = float(best_row.get(metric, float("nan")))
        entries = []
        for fam in families:
            row = best_by_family.get(fam)
            if row is None:
                entries.append(f"{fam}: n/a")
                continue
            val = float(row.get(metric, float("nan")))
            delta = val - best_val if pd.notna(best_val) else float("nan")
            entries.append(
                f"{fam}: `{row.get('method','NA')}` ({metric}={_fmt(val)}, delta={_fmt(delta)})"
            )
        lines.append(f"- n_predictors={n_pred}: " + ", ".join(entries))
    return lines


def _parse_combo_components(method: str) -> List[str]:
    if not method or not str(method).startswith("combo_"):
        return []
    body = str(method)[len("combo_") :]
    return [c for c in body.split("__") if c]


def _combo_addon_deltas(
    df: pd.DataFrame,
    metric: str,
    lower_better: Optional[List[str]] = None,
    families: Optional[List[str]] = None,
) -> List[str]:
    lines: List[str] = []
    if df is None or df.empty or "method" not in df.columns:
        return lines
    families = families or ["hof", "dino", "mmd"]
    lower_better = lower_better or []
    df = df.copy()
    method_map = {str(row["method"]): row for _, row in df.iterrows()}

    addons_by_base: Dict[str, List[str]] = {}
    for method in df["method"].astype(str):
        comps = _parse_combo_components(method)
        if len(comps) != 2:
            continue
        flow_comp = next((c for c in comps if "flow_" in c), None)
        if not flow_comp:
            continue
        other_comp = comps[0] if comps[1] == flow_comp else comps[1]
        other_fams = _family_tags(other_comp)
        other_fams = [f for f in other_fams if f in families]
        if not other_fams:
            continue
        base_row = method_map.get(flow_comp)
        combo_row = method_map.get(method)
        if base_row is None or combo_row is None:
            continue
        addons_by_base.setdefault(flow_comp, []).append(method)

    for base_method, combo_methods in sorted(addons_by_base.items()):
        base_row = method_map.get(base_method)
        if base_row is None:
            continue
        base_n = base_row.get("n_predictors")
        base_agg = base_row.get(metric, float("nan"))
        base_loto = base_row.get("loto_mae", float("nan"))
        base_lobo = base_row.get("lobo_mae", float("nan"))
        line_parts = []
        for combo_method in sorted(combo_methods):
            combo_row = method_map.get(combo_method)
            if combo_row is None:
                continue
            combo_n = combo_row.get("n_predictors")
            combo_agg = combo_row.get(metric, float("nan"))
            combo_loto = combo_row.get("loto_mae", float("nan"))
            combo_lobo = combo_row.get("lobo_mae", float("nan"))
            delta_agg = combo_agg - base_agg if pd.notna(combo_agg) and pd.notna(base_agg) else float("nan")
            # For MAE, improvement is base - combo (positive is better).
            delta_loto = (
                base_loto - combo_loto if pd.notna(base_loto) and pd.notna(combo_loto) else float("nan")
            )
            delta_lobo = (
                base_lobo - combo_lobo if pd.notna(base_lobo) and pd.notna(combo_lobo) else float("nan")
            )
            comp_other = [c for c in _parse_combo_components(combo_method) if c != base_method][0]
            line_parts.append(
                f"+{comp_other} (n_pred {base_n}->{combo_n}, "
                f"delta_agg={_fmt(delta_agg)}, delta_loto_mae={_fmt(delta_loto)}, "
                f"delta_lobo_mae={_fmt(delta_lobo)})"
            )
        if line_parts:
            lines.append(f"- `{base_method}`: " + "; ".join(line_parts))
    return lines
def _dedupe_methods(df: pd.DataFrame, metric: str, lower_better: List[str]) -> pd.DataFrame:
    if df is None or df.empty or "method" not in df.columns:
        return df
    df = df.copy()
    if "aggregate_score" in df.columns and df["aggregate_score"].notna().any():
        df = df.sort_values("aggregate_score", ascending=False)
    elif metric in df.columns:
        df = df.sort_values(metric, ascending=(metric in lower_better))
    return df.drop_duplicates(subset=["method"], keep="first")


def _filter_by_target(df: Optional[pd.DataFrame], targets: List[str]) -> Optional[pd.DataFrame]:
    if df is None or df.empty or not targets:
        return df
    if "target" not in df.columns:
        return df
    return df[df["target"].astype(str).isin(targets)].copy()


def _best_method(
    df: pd.DataFrame,
    mask: pd.Series,
    metric: str,
    lower_better: List[str],
) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    sub = df[mask].copy()
    if sub.empty or metric not in sub.columns:
        return None
    sub = sub.dropna(subset=[metric])
    if sub.empty:
        return None
    ascending = metric in lower_better
    return sub.sort_values(metric, ascending=ascending).iloc[0]


def _asymmetry_block(df: pd.DataFrame, eval_name: str, train_name: str, both_name: str, metric: str):
    eval_row = _find_row(df, eval_name)
    train_row = _find_row(df, train_name)
    both_row = _find_row(df, both_name)
    eval_val = _fmt(eval_row[metric]) if eval_row is not None and metric in eval_row else "NA"
    train_val = _fmt(train_row[metric]) if train_row is not None and metric in train_row else "NA"
    both_val = _fmt(both_row[metric]) if both_row is not None and metric in both_row else "NA"
    return [
        f"- `{eval_name}`: {metric}={eval_val}",
        f"- `{train_name}`: {metric}={train_val}",
        f"- `{both_name}`: {metric}={both_val}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build conclusions report.")
    parser.add_argument("--root", required=True, help="Output root (e.g., analysis_comprehensive_runs/hof_motion_v3)")
    parser.add_argument("--metric", default="loto_mae", help="Primary metric for rankings")
    parser.add_argument(
        "--aggregate-metrics",
        default="loto_mae,lobo_mae",
        help="Comma-separated metrics for aggregate score (default: loto_mae,lobo_mae)",
    )
    parser.add_argument(
        "--aggregate-lower-better",
        default="loto_mae,lobo_mae,loto_rmse,lobo_rmse,loto_regret,lobo_regret",
        help="Comma-separated metrics where lower is better for aggregate scoring.",
    )
    parser.add_argument(
        "--target-filter",
        default="",
        help="Comma-separated targets to include (e.g., auc_normalized_observed).",
    )
    parser.add_argument("--output", default=None, help="Output text file (default: <root>/conclusions.txt)")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Missing root: {root}")

    output_path = Path(args.output) if args.output else root / "conclusions.txt"

    method_summary = _read_csv(root / "method_summary.csv")
    table_a = _read_csv(root / "final_tables" / "table_a_asymmetry.csv")
    table_b = _read_csv(root / "final_tables" / "table_b_motion_vs_appearance.csv")
    table_c = _read_csv(root / "final_tables" / "table_c_generalization.csv")
    table_d = _read_csv(root / "final_tables" / "table_d_parameter_fairness.csv")

    lines: List[str] = []
    lines.append(f"# Conclusions Report for {root}")
    lines.append(f"Metric used for rankings: {args.metric}")
    lines.append("Note: pairwise_rank models optimize ordering and may underperform on absolute-error metrics.")
    target_filter = [t.strip() for t in args.target_filter.split(",") if t.strip()]
    if target_filter:
        lines.append("Target filter: " + ", ".join(target_filter))
    lines.append("")

    if target_filter:
        method_summary = _filter_by_target(method_summary, target_filter)
        table_a = _filter_by_target(table_a, target_filter)
        table_b = _filter_by_target(table_b, target_filter)
        table_c = _filter_by_target(table_c, target_filter)
        table_d = _filter_by_target(table_d, target_filter)

    aggregate_metrics = [m.strip() for m in args.aggregate_metrics.split(",") if m.strip()]
    lower_better = [m.strip() for m in args.aggregate_lower_better.split(",") if m.strip()]
    aggregate_used: List[str] = []
    if method_summary is not None:
        aggregate_used = _add_aggregate_score(method_summary, aggregate_metrics, lower_better)
        method_summary = _dedupe_methods(method_summary, args.metric, lower_better)

    if method_summary is not None:
        if "target" in method_summary.columns:
            targets = sorted(set(method_summary["target"].dropna().astype(str)))
            lines.append("## Targets Used")
            lines.append("- target: " + ", ".join(targets) if targets else "- target: NA")
            if "prediction_target" in method_summary.columns:
                preds = sorted(set(method_summary["prediction_target"].dropna().astype(str)))
                lines.append("- prediction_target: " + ", ".join(preds) if preds else "- prediction_target: NA")
            lines.append("")

        lines.append("## Top Methods (Global)")
        top = _top_n(method_summary, args.metric, n=3, lower_better=lower_better)
        if not top.empty:
            for _, row in top.iterrows():
                method = row.get("method", "NA")
                val = _fmt(row.get(args.metric, float("nan")))
                n_pred = row.get("n_predictors", "NA")
                lines.append(f"- `{method}`: {args.metric}={val}, n_predictors={n_pred}")
        else:
            lines.append("- No rows found for top methods.")
        lines.append("")

    if method_summary is not None:
        lines.append("## Aggregate Score (Best Overall)")
        if aggregate_used:
            lines.append("Aggregate uses: " + ", ".join(aggregate_used))
            lines.append("Aggregate score = mean percentile rank across the metrics above.")
            top_agg = _top_n(method_summary, "aggregate_score", n=5)
            if not top_agg.empty:
                for _, row in top_agg.iterrows():
                    method = row.get("method", "NA")
                    score = _fmt(row.get("aggregate_score", float("nan")))
                    metric_parts = []
                    for metric in aggregate_used:
                        metric_parts.append(f"{metric}={_fmt(row.get(metric, float('nan')))}")
                    metrics_str = ", ".join(metric_parts)
                    lines.append(f"- `{method}`: aggregate={score}, {metrics_str}")
            else:
                lines.append("- No rows found for aggregate ranking.")
        else:
            lines.append("- Not enough metrics to compute aggregate score.")
        lines.append("")

        # Key takeaways: best single-family vs best combo (by aggregate score).
        lines.append("## Key Takeaways (Aggregate)")
        if "aggregate_score" in method_summary.columns:
            ms = method_summary.copy()
            methods = ms["method"].astype(str) if "method" in ms.columns else pd.Series("", index=ms.index)
            # Keep absolute (non-pairwise) methods for aggregate MAE takeaways.
            methods = methods.fillna("")
            is_pairwise = methods.str.contains("_pairwise", regex=False)
            ms = ms[~is_pairwise].copy()
            methods = ms["method"].astype(str)
            # Single-family means exactly one family tag (exclude mixed methods like flow+ dino).
            tags = methods.map(_family_tags)
            single_family = ~methods.str.startswith("combo_") & tags.apply(lambda t: len(t) == 1)
            flow = tags.apply(lambda t: t == ["flow"])
            hof = tags.apply(lambda t: t == ["hof"])
            dino = tags.apply(lambda t: t == ["dino"])
            mmd = tags.apply(lambda t: t == ["mmd"])
            best_flow = _best_method(ms, single_family & flow, "aggregate_score", [])
            best_hof = _best_method(ms, single_family & hof, "aggregate_score", [])
            best_dino = _best_method(ms, single_family & dino, "aggregate_score", [])
            best_mmd = _best_method(ms, single_family & mmd, "aggregate_score", [])
            best_combo = _best_method(ms, methods.str.startswith("combo_"), "aggregate_score", [])

            def _fmt_row(label: str, row: Optional[pd.Series]) -> None:
                if row is None:
                    lines.append(f"- {label}: n/a")
                    return
                lines.append(
                    f"- {label}: `{row.get('method','NA')}` "
                    f"(aggregate={_fmt(row.get('aggregate_score'))}, "
                    f"loto_mae={_fmt(row.get('loto_mae'))}, lobo_mae={_fmt(row.get('lobo_mae'))})"
                )

            _fmt_row("Best flow-only", best_flow)
            _fmt_row("Best hof-only", best_hof)
            _fmt_row("Best dino-only", best_dino)
            _fmt_row("Best mmd-only", best_mmd)
            _fmt_row("Best combo", best_combo)
        else:
            lines.append("- Aggregate score not available.")
        lines.append("")

    if table_a is not None:
        lines.append("## Asymmetry Checks (Table A)")
        lines.extend(_asymmetry_block(table_a, "hof_motion_k1_eval_only", "hof_motion_k1_train_only", "hof_motion_k1", args.metric))
        lines.append("")
        lines.extend(_asymmetry_block(table_a, "flow_eps_raw_joint_eval_only", "flow_eps_raw_joint_train_only", "flow_eps_raw_joint", args.metric))
        lines.append("")
        lines.extend(_asymmetry_block(table_a, "dino_rnorm_k5_eval_only", "dino_rnorm_k5_train_only", "dino_rnorm_k5", args.metric))
        lines.append("")

    if table_b is not None:
        lines.append("## Motion vs Appearance (Table B)")
        top_b = _top_n(table_b, args.metric, n=5, lower_better=lower_better)
        if not top_b.empty:
            for _, row in top_b.iterrows():
                method = row.get("method", "NA")
                val = _fmt(row.get(args.metric, float("nan")))
                lines.append(f"- `{method}`: {args.metric}={val}")
        else:
            lines.append("- No rows found in Table B.")
        lines.append("")

    if table_c is not None and "lobo_spearman" in table_c.columns:
        lines.append("## Generalization (Table C)")
        for _, row in table_c.iterrows():
            method = row.get("method", "NA")
            loto = _fmt(row.get("loto_spearman", float("nan")))
            lobo = _fmt(row.get("lobo_spearman", float("nan")))
            lines.append(f"- `{method}`: LOTO={loto}, LOBO={lobo}")
        lines.append("")

    if table_d is not None:
        lines.append("## Parameter Fairness (Table D)")
        for _, row in table_d.iterrows():
            method = row.get("method", "NA")
            n_pred = row.get("n_predictors", "NA")
            n_base = row.get("n_predictors_base", "NA")
            val = _fmt(row.get(args.metric, float("nan")))
            lines.append(f"- `{method}`: {args.metric}={val}, n_pred={n_pred}, n_base={n_base}")
        lines.append("")

    if method_summary is not None:
        method_summary = method_summary.copy()
        if "method" in method_summary.columns:
            is_pairwise = method_summary["method"].astype(str).str.contains("_pairwise", regex=False)
        else:
            is_pairwise = pd.Series(False, index=method_summary.index)

        absolute_df = method_summary[~is_pairwise].copy()
        pairwise_df = method_summary[is_pairwise].copy()

        # Parameter parity sections removed to keep report concise.

        lines.append("## Family Parity (Absolute Metrics, Best per Family by n_predictors)")
        lines.append(f"Aggregate uses: {', '.join(aggregate_used) if aggregate_used else 'n/a'}")
        lines.append("Single-family methods only (no combos). Missing families are shown as n/a.")
        all_counts = sorted(
            absolute_df["n_predictors"].dropna().astype(int).unique().tolist()
        )
        family_lines = _family_parity_by_param(
            absolute_df,
            "aggregate_score",
            families=["flow", "hof", "dino", "mmd"],
            lower_better=lower_better,
            allow_combo=False,
            counts=all_counts,
        )
        if family_lines:
            lines.extend(family_lines)
        else:
            lines.append("- Not enough data for family parity summary.")
        lines.append("")

        lines.append("## Combo Add-On Deltas (Absolute Metrics, Flow Base + One Add-On)")
        lines.append(f"Aggregate uses: {', '.join(aggregate_used) if aggregate_used else 'n/a'}")
        lines.append("Compares a flow-only base method to flow+{hof|dino|mmd} combos with exactly one add-on.")
        addon_lines = _combo_addon_deltas(
            absolute_df,
            "aggregate_score",
            lower_better=lower_better,
            families=["hof", "dino", "mmd"],
        )
        if addon_lines:
            lines.extend(addon_lines)
        else:
            lines.append("- Not enough data for combo add-on deltas.")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Wrote conclusions report to {output_path}")


if __name__ == "__main__":
    main()
