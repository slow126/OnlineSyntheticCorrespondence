#!/usr/bin/env python3
"""
Build a claim-focused zoom/flip summary table and LaTeX from measured numbers.

The script is fully data-driven:
1) Reads the per-run transfer table (auc_with_features.csv).
2) Loads the lane-wise best k=2 motion and appearance predictor sets from
   selected_exact_k_with_calibrated_diagnostics.csv.
3) Constructs a simple posthoc utility proxy from those 4 predictors.
4) Compares synthetic zoom/flip variants against base synthetic by context.
5) Writes CSV summaries and a paper-ready LaTeX table.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_PERF_CSV = (
    "analysis_comprehensive_runs/hof_motion_v3/density_joint/"
    "leakage_free_hof_motion_k1_plus_density_l2/auc_with_features.csv"
)
DEFAULT_UTILITY_SELECTED_CSV = (
    "analysis_comprehensive_runs/final_utility_sweep_ridge_ols_sixpack_rankobj/"
    "selected_exact_k_with_calibrated_diagnostics.csv"
)
DEFAULT_OUTPUT_DIR = "analysis/zoom_claim_tables"
DEFAULT_VARIANTS = [
    "synthetic_large_zoom",
    "synthetic_small_zoom",
    "synthetic_random_flipping",
]
DEFAULT_MATCHED_BENCHMARKS = ["kitti2012", "kitti2015"]

TOKEN_ALIASES = {
    # Utility-sweep canonical flow tokens to concrete columns in auc_with_features tables.
    "flow_eval_to_train_quantile": "flow_eval_to_train_eps1p5px",
    "flow_train_to_eval_quantile": "flow_train_to_eval_eps1px",
}


def _split_csv_arg(text: str) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def _as_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _direction_sign(name: str) -> int:
    """
    +1: higher is better, -1: lower is better.
    """
    lname = str(name).lower()
    negative_tokens = (
        "mean_dist",
        "median_dist",
        "p90_dist",
        "p95_dist",
        "kl_div",
        "_kl",
        "mmd",
        "outside",
        "asymmetry",
    )
    positive_tokens = (
        "coverage",
        "recall",
        "precision",
        "_eps",
        "eps",
        "quantile",
    )
    if any(tok in lname for tok in negative_tokens):
        return -1
    if any(tok in lname for tok in positive_tokens):
        return 1
    return -1


def _parse_signal_tokens(text: object) -> List[str]:
    if text is None:
        return []
    raw = str(text).strip()
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _pick_best_k2_tokens(selected_df: pd.DataFrame, lane: str) -> List[str]:
    sub = selected_df[selected_df["lane"].astype(str) == lane].copy()
    if sub.empty:
        raise SystemExit(f"No rows found for lane='{lane}' in selected utility CSV.")

    if "k_target" not in sub.columns:
        raise SystemExit("selected utility CSV missing required column: k_target")
    sub["k_target"] = pd.to_numeric(sub["k_target"], errors="coerce")
    sub["mean_jointood_rank_pairwise_cindex"] = pd.to_numeric(
        sub.get("mean_jointood_rank_pairwise_cindex"), errors="coerce"
    )
    sub["distance_to_k2"] = (sub["k_target"] - 2.0).abs()
    sub = sub.sort_values(
        ["distance_to_k2", "mean_jointood_rank_pairwise_cindex"],
        ascending=[True, False],
    )
    row = sub.iloc[0]
    tokens = _parse_signal_tokens(row.get("signal_tokens"))
    if len(tokens) < 2:
        raise SystemExit(
            f"Best row for lane='{lane}' did not provide >=2 tokens: {tokens}"
        )
    return tokens[:2]


def _resolve_predictor_tokens(tokens: Sequence[str], available_cols: Iterable[str]) -> Tuple[List[str], Dict[str, str], List[str]]:
    resolved: List[str] = []
    mapping: Dict[str, str] = {}
    missing: List[str] = []
    available = set(str(c) for c in available_cols)
    for tok in tokens:
        t = str(tok).strip()
        if t in available:
            resolved.append(t)
            mapping[t] = t
            continue
        alias = TOKEN_ALIASES.get(t, "")
        if alias and alias in available:
            resolved.append(alias)
            mapping[t] = alias
            continue
        missing.append(t)
    return resolved, mapping, missing


def _zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    mean = float(vals.mean())
    std = float(vals.std(ddof=0))
    if not np.isfinite(std) or std == 0.0:
        return pd.Series(np.zeros(len(vals), dtype=float), index=series.index)
    return (vals - mean) / std


def _bool_rate(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    vals = vals.dropna()
    if vals.empty:
        return float("nan")
    return float(vals.mean())


def _median_safe(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    return float(vals.median())


def _scope_filter(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "matched":
        return df[df["scope"].astype(str) == "matched"].copy()
    if scope == "nonmatched":
        return df[df["scope"].astype(str) == "nonmatched"].copy()
    if scope == "all_nonself":
        return df[df["scope"].astype(str).isin(["matched", "nonmatched"])].copy()
    return df.iloc[0:0].copy()


def _fmt_num(x: object, digits: int = 3) -> str:
    v = _as_float(x)
    if not np.isfinite(v):
        return "--"
    return f"{v:.{digits}f}"


def _build_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    if df.empty:
        body = "\\begin{tabular}{l}\\toprule\n(Empty) \\\\\n\\bottomrule\\end{tabular}"
    else:
        body = df.to_latex(
            index=False,
            escape=True,
            na_rep="--",
            column_format="llrrrrrrl",
        ).strip()
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\linewidth}{!}{%",
        body,
        "}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build zoom/flip claim table + LaTeX.")
    parser.add_argument("--perf-csv", default=DEFAULT_PERF_CSV, help="Input auc_with_features.csv")
    parser.add_argument(
        "--utility-selected-csv",
        default=DEFAULT_UTILITY_SELECTED_CSV,
        help="selected_exact_k_with_calibrated_diagnostics.csv used to pick best 2 flow + 2 appearance tokens.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--baseline", default="synthetic", help="Baseline train dataset.")
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated synthetic variants to compare against baseline.",
    )
    parser.add_argument(
        "--matched-benchmarks",
        default=",".join(DEFAULT_MATCHED_BENCHMARKS),
        help="Comma-separated benchmarks treated as matched target tasks.",
    )
    parser.add_argument(
        "--perf-col",
        default="auc_normalized_observed",
        help="Performance column in perf CSV.",
    )
    parser.add_argument(
        "--context-cols",
        default="model_family,pretrained,freeze,benchmark",
        help="Context columns used for within-context baseline deltas.",
    )
    parser.add_argument(
        "--latex-name",
        default="zoom_claim_table.tex",
        help="Output LaTeX file name.",
    )
    args = parser.parse_args()

    perf_path = Path(args.perf_csv)
    selected_path = Path(args.utility_selected_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = _split_csv_arg(args.variants)
    if not variants:
        raise SystemExit("No variants provided.")
    matched_benchmarks = set(_split_csv_arg(args.matched_benchmarks))
    context_cols = _split_csv_arg(args.context_cols)
    if not context_cols:
        raise SystemExit("No context columns provided.")

    perf_df = _read_csv(perf_path)
    sel_df = _read_csv(selected_path)

    for col in ["lane", "signal_tokens", "k_target"]:
        if col not in sel_df.columns:
            raise SystemExit(f"selected utility CSV missing required column: {col}")

    if args.perf_col not in perf_df.columns:
        raise SystemExit(f"Performance column not found in perf CSV: {args.perf_col}")
    required_base_cols = set(context_cols + ["train_dataset", args.perf_col])
    missing_base = [c for c in required_base_cols if c not in perf_df.columns]
    if missing_base:
        raise SystemExit(f"perf CSV missing required columns: {missing_base}")

    motion_tokens = _pick_best_k2_tokens(sel_df, "motion_only")
    appearance_tokens = _pick_best_k2_tokens(sel_df, "appearance_only")
    selected_tokens_raw = motion_tokens + appearance_tokens
    selected_predictors, token_mapping, missing_predictors = _resolve_predictor_tokens(
        selected_tokens_raw,
        perf_df.columns,
    )
    if missing_predictors:
        raise SystemExit(
            "Selected predictors are not all present in perf CSV. "
            f"Missing: {missing_predictors}"
        )
    if len(selected_predictors) != 4:
        raise SystemExit(
            "Expected exactly 4 resolved predictors (2 motion + 2 appearance), got: "
            f"{selected_predictors}"
        )

    work = perf_df.copy()
    work["train_dataset"] = work["train_dataset"].astype(str)
    keep_train = {args.baseline, *variants}
    work = work[work["train_dataset"].isin(keep_train)].copy()

    group_cols = context_cols + ["train_dataset"]
    agg_cols = [args.perf_col] + selected_predictors
    grouped = (
        work[group_cols + agg_cols]
        .groupby(group_cols, dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )

    rows: List[Dict[str, object]] = []
    per_context_cols = context_cols
    for _, ctx_df in grouped.groupby(per_context_cols, dropna=False):
        ctx = ctx_df.copy()
        present = set(ctx["train_dataset"].astype(str))
        if args.baseline not in present:
            continue

        for pred in selected_predictors:
            zcol = f"__z_{pred}"
            ctx[zcol] = _zscore(ctx[pred])
            sign = _direction_sign(pred)
            ctx[f"__u_{pred}"] = sign * pd.to_numeric(ctx[zcol], errors="coerce")
        u_cols = [f"__u_{p}" for p in selected_predictors]
        ctx["utility_u4"] = ctx[u_cols].mean(axis=1)

        base_row = ctx[ctx["train_dataset"] == args.baseline]
        if base_row.empty:
            continue
        base = base_row.iloc[0]

        for variant in variants:
            var_row = ctx[ctx["train_dataset"] == variant]
            if var_row.empty:
                continue
            var = var_row.iloc[0]
            bench = str(var.get("benchmark", ""))
            if bench == args.baseline:
                scope = "self"
            elif bench in matched_benchmarks:
                scope = "matched"
            else:
                scope = "nonmatched"

            delta_perf = _as_float(var.get(args.perf_col)) - _as_float(base.get(args.perf_col))
            delta_u = _as_float(var.get("utility_u4")) - _as_float(base.get("utility_u4"))
            sign_agree = float("nan")
            if np.isfinite(delta_perf) and np.isfinite(delta_u) and delta_perf != 0.0 and delta_u != 0.0:
                sign_agree = float(np.sign(delta_perf) == np.sign(delta_u))

            row: Dict[str, object] = {
                "variant": variant,
                "scope": scope,
                "delta_auc": delta_perf,
                "delta_utility_u4": delta_u,
                "perf_improved": float(delta_perf > 0.0) if np.isfinite(delta_perf) else float("nan"),
                "utility_improved": float(delta_u > 0.0) if np.isfinite(delta_u) else float("nan"),
                "sign_agreement": sign_agree,
            }
            for c in context_cols:
                row[c] = var.get(c)
            rows.append(row)

    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        raise SystemExit("No variant-vs-baseline rows were generated. Check inputs/filters.")

    summary_rows: List[Dict[str, object]] = []
    for variant in variants:
        sub_var = detail_df[detail_df["variant"].astype(str) == variant].copy()
        for scope in ["matched", "nonmatched", "all_nonself"]:
            block = _scope_filter(sub_var, scope)
            if block.empty:
                continue
            summary_rows.append(
                {
                    "variant": variant,
                    "scope": scope,
                    "n_contexts": int(len(block)),
                    "delta_auc_mean": float(pd.to_numeric(block["delta_auc"], errors="coerce").mean()),
                    "delta_auc_median": _median_safe(block["delta_auc"]),
                    "gain_rate": _bool_rate(block["perf_improved"]),
                    "delta_utility_u4_mean": float(
                        pd.to_numeric(block["delta_utility_u4"], errors="coerce").mean()
                    ),
                    "utility_up_rate": _bool_rate(block["utility_improved"]),
                    "sign_agreement_rate": _bool_rate(block["sign_agreement"]),
                }
            )
    summary_df = pd.DataFrame(summary_rows)

    def _pick_metric(v: str, scope: str, col: str) -> float:
        hit = summary_df[
            (summary_df["variant"].astype(str) == v)
            & (summary_df["scope"].astype(str) == scope)
        ]
        if hit.empty:
            return float("nan")
        return _as_float(hit.iloc[0].get(col))

    final_rows: List[Dict[str, object]] = []
    for variant in variants:
        expected = "improve on matched tasks" if "random_flipping" not in variant else "degrade on matched tasks"
        d_match = _pick_metric(variant, "matched", "delta_auc_mean")
        d_non = _pick_metric(variant, "nonmatched", "delta_auc_mean")
        g_match = _pick_metric(variant, "matched", "gain_rate")
        u_match = _pick_metric(variant, "matched", "delta_utility_u4_mean")
        u_non = _pick_metric(variant, "nonmatched", "delta_utility_u4_mean")
        sign_match = _pick_metric(variant, "matched", "sign_agreement_rate")

        if "random_flipping" in variant:
            supports = np.isfinite(d_match) and d_match < 0.0
        else:
            supports = np.isfinite(d_match) and d_match > 0.0
        verdict = "supports_claim" if supports else "does_not_support"

        final_rows.append(
            {
                "variant": variant,
                "expected_on_matched": expected,
                "delta_auc_matched_mean": d_match,
                "gain_rate_matched": g_match,
                "delta_auc_nonmatched_mean": d_non,
                "delta_u4_matched_mean": u_match,
                "delta_u4_nonmatched_mean": u_non,
                "u4_sign_agreement_matched": sign_match,
                "verdict": verdict,
            }
        )
    final_df = pd.DataFrame(final_rows)

    details_out = out_dir / "zoom_claim_pairwise_deltas.csv"
    summary_out = out_dir / "zoom_claim_summary_by_scope.csv"
    final_out = out_dir / "zoom_claim_final_table.csv"
    metadata_out = out_dir / "zoom_claim_metadata.csv"
    tex_out = out_dir / args.latex_name

    detail_df.to_csv(details_out, index=False)
    summary_df.to_csv(summary_out, index=False)
    final_df.to_csv(final_out, index=False)

    meta_rows = [
        {"key": "perf_csv", "value": str(perf_path)},
        {"key": "utility_selected_csv", "value": str(selected_path)},
        {"key": "baseline", "value": args.baseline},
        {"key": "variants", "value": ",".join(variants)},
        {"key": "matched_benchmarks", "value": ",".join(sorted(matched_benchmarks))},
        {"key": "perf_col", "value": args.perf_col},
        {"key": "motion_k2_tokens_raw", "value": ",".join(motion_tokens)},
        {"key": "appearance_k2_tokens_raw", "value": ",".join(appearance_tokens)},
        {"key": "resolved_predictors_used", "value": ",".join(selected_predictors)},
        {"key": "token_alias_mapping", "value": "; ".join(f"{k}->{v}" for k, v in token_mapping.items())},
    ]
    pd.DataFrame(meta_rows).to_csv(metadata_out, index=False)

    latex_df = final_df.copy()
    latex_df = latex_df.rename(
        columns={
            "variant": "Variant",
            "expected_on_matched": "Expected (Matched)",
            "delta_auc_matched_mean": "DeltaAUC Matched",
            "gain_rate_matched": "GainRate Matched",
            "delta_auc_nonmatched_mean": "DeltaAUC NonMatched",
            "delta_u4_matched_mean": "DeltaU4 Matched",
            "delta_u4_nonmatched_mean": "DeltaU4 NonMatched",
            "u4_sign_agreement_matched": "U4/AUC SignAgree Matched",
            "verdict": "Verdict",
        }
    )
    for c in [
        "DeltaAUC Matched",
        "GainRate Matched",
        "DeltaAUC NonMatched",
        "DeltaU4 Matched",
        "DeltaU4 NonMatched",
        "U4/AUC SignAgree Matched",
    ]:
        latex_df[c] = latex_df[c].map(lambda x: _fmt_num(x, digits=3))

    motion_label = ", ".join(motion_tokens)
    app_label = ", ".join(appearance_tokens)
    resolved_label = ", ".join(selected_predictors)
    caption = (
        "Coverage-guided synthetic zoom/flip intervention summary. "
        "Posthoc utility proxy U4 uses best 2 motion + best 2 appearance predictors "
        f"from utility sweep (raw motion tokens: {motion_label}; raw appearance tokens: {app_label}; "
        f"resolved columns: {resolved_label}). "
        "Numbers are variant-minus-base synthetic means across contexts."
    )
    latex = _build_latex_table(
        latex_df,
        caption=caption,
        label="tab:zoom_claim_u4_posthoc",
    )
    tex_out.write_text(latex)

    print(f"Wrote: {details_out}")
    print(f"Wrote: {summary_out}")
    print(f"Wrote: {final_out}")
    print(f"Wrote: {metadata_out}")
    print(f"Wrote: {tex_out}")


if __name__ == "__main__":
    main()
