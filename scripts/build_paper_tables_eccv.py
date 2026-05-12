#!/usr/bin/env python3
"""
Build two ECCV paper-facing result tables from compiled analysis outputs.

This script is intentionally deterministic and data-driven:
- Reads metrics from method_summary.csv
- Optionally pulls one highlighted run directly from raw summary files
- Writes:
  1) paired hypothesis table (CSV)
  2) compact method summary table (CSV)
  3) readable text rendering of both tables
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

CONTROL_PREDICTORS = {
    "log_n_samples_eval",
    "log_avg_flows_eval",
    "log_n_samples_train",
    "log_avg_flows_train",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def _as_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _is_finite(value: object) -> bool:
    x = _as_float(value)
    return pd.notna(x)


def _clean_summary(
    df: pd.DataFrame,
    target: Optional[str],
    model: Optional[str],
    metric: str,
) -> pd.DataFrame:
    out = df.copy()
    for col in ("n_predictors", metric, "loto_spearman", "loto_mae", "lobo_mae", "lobo_spearman"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if target and "target" in out.columns:
        out = out[out["target"].astype(str) == target]
    if model and "model" in out.columns:
        out = out[out["model"].astype(str) == model]
    if out.empty:
        raise SystemExit("No rows left after filtering method summary.")

    if metric not in out.columns:
        raise SystemExit(f"Metric column not found: {metric}")

    if "method" in out.columns:
        # If duplicate method rows exist, keep the best one on the selected metric.
        out = out.sort_values(metric, ascending=True).drop_duplicates(subset=["method"], keep="first")
    if "predictors" in out.columns:
        out["budget_predictors"] = out["predictors"].apply(_signal_predictor_count_from_text)
    elif "n_predictors_base" in out.columns:
        out["budget_predictors"] = pd.to_numeric(out["n_predictors_base"], errors="coerce")
    else:
        out["budget_predictors"] = pd.to_numeric(out.get("n_predictors"), errors="coerce")
    return out


def _best_row(df: pd.DataFrame, metric: str) -> pd.Series:
    return df.sort_values(metric, ascending=True).iloc[0]


def _best_by_family(df: pd.DataFrame, family: str, metric: str) -> pd.Series:
    fam = df[df["family"].astype(str) == family]
    if fam.empty:
        raise SystemExit(f"No methods found for family='{family}'.")
    return _best_row(fam, metric)


def _method_matches_base(method_name: str, base: str) -> bool:
    return method_name == base or method_name.startswith(f"{base}_")


def _signal_predictor_count_from_text(predictors_text: object) -> float:
    if predictors_text is None or not str(predictors_text).strip():
        return float("nan")
    tokens = [t.strip() for t in str(predictors_text).split(",") if t.strip()]
    keep = []
    for tok in tokens:
        if tok in CONTROL_PREDICTORS:
            continue
        if tok.startswith("enc_") or tok.startswith("mf_"):
            continue
        keep.append(tok)
    return float(len(keep))


def _budget_value(row: pd.Series) -> float:
    if "budget_predictors" in row.index and pd.notna(_as_float(row.get("budget_predictors"))):
        return _as_float(row.get("budget_predictors"))
    if "n_predictors_base" in row.index and pd.notna(_as_float(row.get("n_predictors_base"))):
        return _as_float(row.get("n_predictors_base"))
    return _as_float(row.get("n_predictors"))


def _budget_col(df: pd.DataFrame) -> str:
    if "budget_predictors" in df.columns:
        return "budget_predictors"
    if "n_predictors_base" in df.columns:
        return "n_predictors_base"
    return "n_predictors"


def _filter_methods_by_bases(df: pd.DataFrame, bases: Sequence[str]) -> pd.DataFrame:
    if "method" not in df.columns:
        return df.iloc[0:0].copy()
    cleaned = [b.strip() for b in bases if b.strip()]
    if not cleaned:
        return df.iloc[0:0].copy()
    mask = df["method"].astype(str).apply(lambda m: any(_method_matches_base(m, b) for b in cleaned))
    return df[mask].copy()


def _resolve_hof_core_df(summary_df: pd.DataFrame, hof_allow_bases: Sequence[str]) -> pd.DataFrame:
    # Preferred source is explicit motion-family rows.
    motion_df = summary_df[summary_df["family"].astype(str) == "motion"].copy()
    motion_hits = _filter_methods_by_bases(motion_df, hof_allow_bases)
    if not motion_hits.empty:
        return motion_hits

    # Safety fallback: allow hof_* names even if family label is wrong.
    all_hits = _filter_methods_by_bases(summary_df.copy(), hof_allow_bases)
    all_hits = all_hits[all_hits["method"].astype(str).str.startswith("hof_")]
    return all_hits


def _exclude_method_substring(df: pd.DataFrame, needle: str) -> pd.DataFrame:
    if "method" not in df.columns:
        return df.copy()
    return df[~df["method"].astype(str).str.contains(needle, case=False, regex=False)].copy()


def _single_family_subset(df: pd.DataFrame, family: str) -> pd.DataFrame:
    if "method" not in df.columns:
        return df.copy()
    names = df["method"].astype(str)
    if family == "flow":
        # Single-direction flow (train_only/eval_only) is the intended apples-to-apples with flow MMD.
        mask = names.str.contains("flow_eps_raw", case=False, regex=False)
        mask &= names.str.contains("_train_only|_eval_only|_single", case=False, regex=True)
        mask &= ~names.str.startswith("combo_")
        return df[mask].copy()
    if family == "appearance":
        # Appearance family methods are already mostly single; keep direct family rows.
        mask = ~names.str.startswith("combo_")
        return df[mask].copy()
    if family == "motion":
        # Prefer non-combined motion rows when available.
        mask = ~names.str.contains("plus_", case=False, regex=False)
        subset = df[mask].copy()
        return subset if not subset.empty else df.copy()
    return df.copy()


def _best_by_budget(df: pd.DataFrame, n_predictors: int, metric: str) -> Optional[pd.Series]:
    bcol = _budget_col(df)
    if bcol not in df.columns:
        return None
    work = df.copy()
    work[bcol] = work[bcol].apply(_as_float)
    matched = work[work[bcol] == float(n_predictors)]
    if matched.empty:
        return None
    return _best_row(matched, metric)


def _best_at_or_below_budget(df: pd.DataFrame, n_predictors: int, metric: str) -> Optional[pd.Series]:
    bcol = _budget_col(df)
    if bcol not in df.columns:
        return None
    work = df.copy()
    work[bcol] = work[bcol].apply(_as_float)
    candidates = work[work[bcol] <= float(n_predictors)]
    if candidates.empty:
        return None
    return _best_row(candidates, metric)


def _best_matched_or_nearest(df: pd.DataFrame, n_predictors: int, metric: str) -> Tuple[Optional[pd.Series], str, Optional[int]]:
    if df.empty:
        return None, "none", None
    bcol = _budget_col(df)
    if bcol not in df.columns:
        return _best_row(df, metric), "unconstrained", None
    work = df.copy()
    work[bcol] = work[bcol].apply(_as_float)
    exact = work[work[bcol] == float(n_predictors)]
    if not exact.empty:
        return _best_row(exact, metric), "exact", n_predictors
    work["n_diff"] = (work[bcol] - float(n_predictors)).abs()
    work = work.sort_values(["n_diff", metric], ascending=[True, True])
    near = work.iloc[0]
    return near, "nearest", int(_as_float(near.get(bcol)))


def _shared_budgets_between(df_a: pd.DataFrame, df_b: pd.DataFrame) -> List[int]:
    col_a = _budget_col(df_a)
    col_b = _budget_col(df_b)
    if col_a not in df_a.columns or col_b not in df_b.columns:
        return []
    a = {_as_float(x) for x in df_a[col_a].dropna()}
    b = {_as_float(x) for x in df_b[col_b].dropna()}
    out = sorted({int(x) for x in a.intersection(b) if pd.notna(x)})
    return out


def _best_by_family_and_budget(
    df: pd.DataFrame,
    family: str,
    n_predictors: int,
    metric: str,
) -> Optional[pd.Series]:
    bcol = _budget_col(df)
    if "family" not in df.columns or bcol not in df.columns:
        return None
    fam = df[df["family"].astype(str) == family].copy()
    if fam.empty:
        return None
    fam[bcol] = fam[bcol].apply(_as_float)
    matched = fam[fam[bcol] == float(n_predictors)]
    if matched.empty:
        return None
    return _best_row(matched, metric)


def _shared_budgets(df: pd.DataFrame, family_a: str, family_b: str) -> List[int]:
    bcol = _budget_col(df)
    if "family" not in df.columns or bcol not in df.columns:
        return []
    a = df[df["family"].astype(str) == family_a][bcol].dropna().apply(_as_float)
    b = df[df["family"].astype(str) == family_b][bcol].dropna().apply(_as_float)
    a_set = {int(x) for x in a if pd.notna(x)}
    b_set = {int(x) for x in b if pd.notna(x)}
    return sorted(a_set.intersection(b_set))


def _matched_asym_sym_pairs(df: pd.DataFrame, metric: str) -> List[Tuple[int, pd.Series, pd.Series]]:
    if "symmetry" not in df.columns or "n_predictors" not in df.columns:
        return []
    work = df.copy()
    work["n_predictors"] = work["n_predictors"].apply(_as_float)
    work = work[pd.notna(work["n_predictors"])]
    if work.empty:
        return []
    out: List[Tuple[int, pd.Series, pd.Series]] = []
    for n_val, block in work.groupby("n_predictors"):
        asym = block[block["symmetry"].astype(str) == "asym"]
        sym = block[block["symmetry"].astype(str) == "sym"]
        if asym.empty or sym.empty:
            continue
        out.append((int(n_val), _best_row(asym, metric), _best_row(sym, metric)))
    return sorted(out, key=lambda x: x[0])


def _directional_triplet_best(df: pd.DataFrame, family: str, metric: str) -> Optional[Tuple[pd.Series, pd.Series]]:
    """
    Find best base method in a family where:
      base
      base_eval_only
      base_train_only
    all exist. Return (base_row, best_single_direction_row).
    """
    if "method" not in df.columns or "family" not in df.columns:
        return None

    fam = df[df["family"].astype(str) == family].copy()
    if fam.empty:
        return None

    by_name = {str(r["method"]): r for _, r in fam.iterrows()}
    candidates: List[Tuple[pd.Series, pd.Series]] = []
    for name, row in by_name.items():
        if name.endswith("_eval_only") or name.endswith("_train_only"):
            continue
        eval_name = f"{name}_eval_only"
        train_name = f"{name}_train_only"
        if eval_name not in by_name or train_name not in by_name:
            continue
        eval_row = by_name[eval_name]
        train_row = by_name[train_name]
        best_single = eval_row if _as_float(eval_row[metric]) <= _as_float(train_row[metric]) else train_row
        candidates.append((row, best_single))

    if not candidates:
        return None
    candidates.sort(key=lambda pair: _as_float(pair[0][metric]))
    return candidates[0]


def _comparison_row(
    hypothesis: str,
    comparison: str,
    a: pd.Series,
    b: pd.Series,
    metric: str,
    corr_col: str,
    verdict: str,
) -> Dict[str, object]:
    a_metric = _as_float(a.get(metric))
    b_metric = _as_float(b.get(metric))
    a_sp = _as_float(a.get(corr_col))
    b_sp = _as_float(b.get(corr_col))
    return {
        "hypothesis": hypothesis,
        "comparison": comparison,
        "method_a": str(a.get("method", "")),
        "method_b": str(b.get("method", "")),
        "family_a": str(a.get("family", "")),
        "family_b": str(b.get("family", "")),
        "symmetry_a": str(a.get("symmetry", "")),
        "symmetry_b": str(b.get("symmetry", "")),
        "n_predictors_a": int(_budget_value(a)),
        "n_predictors_b": int(_budget_value(b)),
        f"{metric}_a": round(a_metric, 6),
        f"{metric}_b": round(b_metric, 6),
        f"delta_{metric}_b_minus_a": round(b_metric - a_metric, 6),
        f"{corr_col}_a": round(a_sp, 6),
        f"{corr_col}_b": round(b_sp, 6),
        f"delta_{corr_col}_a_minus_b": round(a_sp - b_sp, 6),
        "verdict": verdict,
    }


def _overall_row(df: pd.DataFrame) -> Optional[pd.Series]:
    for key in ("benchmark", "train_dataset", "train_dataset_group"):
        if key in df.columns:
            overall = df[df[key].astype(str) == "__overall__"]
            if not overall.empty:
                return overall.iloc[0]
    if not df.empty:
        return df.iloc[0]
    return None


def _extract_highlighted_run(run_dir: Path) -> Dict[str, object]:
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    loto = _read_csv(run_dir / "prediction_loto_summary.csv")
    lobo = _read_csv(run_dir / "prediction_lobo_summary.csv")
    jointood = _read_csv(run_dir / "prediction_jointood_summary.csv") if (run_dir / "prediction_jointood_summary.csv").exists() else pd.DataFrame()
    loto_rank = _read_csv(run_dir / "prediction_loto_rank_summary.csv") if (run_dir / "prediction_loto_rank_summary.csv").exists() else pd.DataFrame()
    lobo_rank = _read_csv(run_dir / "prediction_lobo_rank_summary.csv") if (run_dir / "prediction_lobo_rank_summary.csv").exists() else pd.DataFrame()
    jointood_rank = _read_csv(run_dir / "prediction_jointood_rank_summary.csv") if (run_dir / "prediction_jointood_rank_summary.csv").exists() else pd.DataFrame()
    loto_row = _overall_row(loto)
    lobo_row = _overall_row(lobo)
    jointood_row = _overall_row(jointood) if not jointood.empty else None
    loto_rank_row = _overall_row(loto_rank) if not loto_rank.empty else None
    lobo_rank_row = _overall_row(lobo_rank) if not lobo_rank.empty else None
    jointood_rank_row = _overall_row(jointood_rank) if not jointood_rank.empty else None
    if loto_row is None or lobo_row is None:
        raise SystemExit(f"Could not find overall rows in highlighted run summaries: {run_dir}")

    method_name = f"{run_dir.name.replace('leakage_free_', '')} (highlighted_run)"
    return {
        "method": method_name,
        "source_path": str(run_dir),
        "family": "mixed",
        "symmetry": "asym",
        "n_predictors": int(_as_float(meta.get("n_predictors"))),
        "jointood_mae": round(_as_float(jointood_row.get("mae")) if jointood_row is not None else float("nan"), 6),
        "loto_mae": round(_as_float(loto_row.get("mae")), 6),
        "lobo_mae": round(_as_float(lobo_row.get("mae")), 6),
        "loto_spearman": round(_as_float(loto_row.get("spearman")), 6),
        "lobo_spearman": round(_as_float(lobo_row.get("spearman")), 6),
        "jointood_rank_spearman": round(
            _as_float(jointood_rank_row.get("spearman")) if jointood_rank_row is not None else float("nan"), 6
        ),
        "jointood_rank_kendall_tau": round(
            _as_float(jointood_rank_row.get("kendall_tau_b"))
            if jointood_rank_row is not None and pd.notna(_as_float(jointood_rank_row.get("kendall_tau_b")))
            else (
                _as_float(jointood_rank_row.get("kendall_tau"))
                if jointood_rank_row is not None
                else float("nan")
            ),
            6,
        ),
        "jointood_rank_pairwise_cindex": round(
            _as_float(jointood_rank_row.get("pairwise_cindex"))
            if jointood_rank_row is not None and pd.notna(_as_float(jointood_rank_row.get("pairwise_cindex")))
            else (
                _as_float(jointood_rank_row.get("cindex"))
                if jointood_rank_row is not None
                else float("nan")
            ),
            6,
        ),
        "jointood_rank_pct_err": round(
            _as_float(jointood_rank_row.get("mean_abs_rank_pct_error"))
            if jointood_rank_row is not None
            else float("nan"),
            6,
        ),
        "loto_rank_spearman": round(
            _as_float(loto_rank_row.get("spearman")) if loto_rank_row is not None else float("nan"), 6
        ),
        "lobo_rank_spearman": round(
            _as_float(lobo_rank_row.get("spearman")) if lobo_rank_row is not None else float("nan"), 6
        ),
        "key_note": "User-highlighted leakage-free run",
    }


def _method_row(row: pd.Series, source_path: str, note: str) -> Dict[str, object]:
    return {
        "method": str(row.get("method", "")),
        "source_path": source_path,
        "family": str(row.get("family", "")),
        "symmetry": str(row.get("symmetry", "")),
        "n_predictors": int(_budget_value(row)),
        "jointood_mae": round(_as_float(row.get("jointood_mae")), 6),
        "loto_mae": round(_as_float(row.get("loto_mae")), 6),
        "lobo_mae": round(_as_float(row.get("lobo_mae")), 6),
        "loto_spearman": round(_as_float(row.get("loto_spearman")), 6),
        "lobo_spearman": round(_as_float(row.get("lobo_spearman")), 6),
        "jointood_rank_spearman": round(_as_float(row.get("jointood_rank_spearman")), 6),
        "jointood_rank_kendall_tau": round(_as_float(row.get("jointood_rank_kendall_tau")), 6),
        "jointood_rank_pairwise_cindex": round(_as_float(row.get("jointood_rank_pairwise_cindex")), 6),
        "jointood_rank_pct_err": round(_as_float(row.get("jointood_rank_pct_err")), 6),
        "loto_rank_spearman": round(_as_float(row.get("loto_rank_spearman")), 6),
        "lobo_rank_spearman": round(_as_float(row.get("lobo_rank_spearman")), 6),
        "key_note": note,
    }


def _format_value(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _render_ascii_table(rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> str:
    if not rows:
        return "(no rows)"
    widths: List[int] = []
    for col in columns:
        max_len = len(col)
        for row in rows:
            max_len = max(max_len, len(_format_value(row.get(col, ""))))
        widths.append(max_len)

    def fmt_line(values: Sequence[str]) -> str:
        parts = [values[i].ljust(widths[i]) for i in range(len(values))]
        return "| " + " | ".join(parts) + " |"

    header = fmt_line(list(columns))
    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    body = [fmt_line([_format_value(r.get(c, "")) for c in columns]) for r in rows]
    return "\n".join([header, sep] + body)


def _write_outputs(
    out_dir: Path,
    table1: List[Dict[str, object]],
    table2: List[Dict[str, object]],
    summary_path: Path,
    highlighted_run: Optional[Path],
    metric: str,
    corr_col: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    table1_path = out_dir / "table_1_hypothesis_validation.csv"
    table2_path = out_dir / "table_2_method_summary_compact.csv"
    readable_path = out_dir / "paper_tables_readable.txt"

    pd.DataFrame(table1).to_csv(table1_path, index=False)
    pd.DataFrame(table2).to_csv(table2_path, index=False)

    cols1 = [
        "hypothesis",
        "comparison",
        "method_a",
        "method_b",
        "n_predictors_a",
        "n_predictors_b",
        f"{metric}_a",
        f"{metric}_b",
        f"delta_{metric}_b_minus_a",
        f"{corr_col}_a",
        f"{corr_col}_b",
        f"delta_{corr_col}_a_minus_b",
        "verdict",
    ]
    cols2 = [
        "method",
        "family",
        "symmetry",
        "n_predictors",
        "jointood_mae",
        "loto_mae",
        "lobo_mae",
        "loto_spearman",
        "lobo_spearman",
        "jointood_rank_spearman",
        "jointood_rank_kendall_tau",
        "jointood_rank_pairwise_cindex",
        "jointood_rank_pct_err",
        "key_note",
        "source_path",
    ]

    lines: List[str] = []
    lines.append("Paper Tables (Auto-generated)")
    lines.append("=" * 100)
    lines.append(f"Date: {date.today().isoformat()}")
    lines.append(f"Summary source: {summary_path}")
    if highlighted_run:
        lines.append(f"Highlighted run source: {highlighted_run}")
    lines.append("")
    lines.append("TABLE 1: Hypothesis Validation (Paired Deltas)")
    lines.append("-" * 100)
    lines.append(f"Metric: {metric} (lower is better)")
    lines.append(f"Positive delta_{metric}_b_minus_a => method A better on the selected metric.")
    lines.append(f"Positive delta_{corr_col}_a_minus_b => method A better on correlation metric.")
    lines.append("")
    lines.append(_render_ascii_table(table1, cols1))
    lines.append("")
    lines.append("TABLE 2: Compact Method Summary")
    lines.append("-" * 100)
    lines.append(_render_ascii_table(table2, cols2))
    lines.append("")
    readable_path.write_text("\n".join(lines))

    print(f"Wrote {table1_path}")
    print(f"Wrote {table2_path}")
    print(f"Wrote {readable_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ECCV paper tables from compiled method summary.")
    parser.add_argument(
        "--summary",
        required=True,
        help="Path to method_summary.csv (will be created/overwritten if --manifest is provided).",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional method_summary manifest YAML; when set, compile_method_summary.py runs first.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write table outputs")
    parser.add_argument(
        "--target",
        default="auc_normalized_observed",
        help="Optional target filter. Use empty string to disable.",
    )
    parser.add_argument(
        "--model",
        default="ols",
        help="Optional model filter. Use empty string to disable.",
    )
    parser.add_argument(
        "--metric",
        default="loto_mae",
        help="Primary metric for selecting best methods in comparisons.",
    )
    parser.add_argument(
        "--highlight-run",
        default="",
        help="Optional run directory to include in compact table.",
    )
    parser.add_argument(
        "--hof-allow-methods",
        default="hof_motion_k5,hof_motion_k10,hof_motion_k20,hof_motion_k40,hof_kl_k5",
        help="Comma-separated HOF method bases to include in HOF-family comparisons.",
    )
    parser.add_argument(
        "--hof-primary-method",
        default="",
        help="Optional fixed HOF method for headline H2 comparison; default uses best-by-metric.",
    )
    parser.add_argument(
        "--h1-motion-baseline-method",
        default="mmd_flow_only",
        help="Baseline method used for H1 motion-family comparisons (flow/hof).",
    )
    parser.add_argument(
        "--h1-appearance-baseline-method",
        default="mmd_dino_only",
        help="Baseline method used for H1 appearance-family comparisons.",
    )
    parser.add_argument(
        "--h1-combined-baseline-method",
        default="mmd_only",
        help="Baseline method used for H1 asymmetric-combined (mixed family) comparisons.",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    out_dir = Path(args.output_dir)
    target = args.target.strip() or None
    model = args.model.strip() or None
    metric = args.metric.strip()
    highlight_run = Path(args.highlight_run).resolve() if args.highlight_run else None
    hof_allow_bases = [x.strip() for x in args.hof_allow_methods.split(",") if x.strip()]
    hof_primary_method = args.hof_primary_method.strip() or None
    h1_motion_baseline_method = args.h1_motion_baseline_method.strip()
    h1_appearance_baseline_method = args.h1_appearance_baseline_method.strip()
    h1_combined_baseline_method = args.h1_combined_baseline_method.strip()

    if args.manifest:
        manifest_path = Path(args.manifest)
        compile_script = Path(__file__).with_name("compile_method_summary.py")
        if not compile_script.exists():
            raise SystemExit(f"Missing compile script: {compile_script}")
        cmd = [
            sys.executable,
            str(compile_script),
            "--manifest",
            str(manifest_path),
            "--output",
            str(summary_path),
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    summary_all = _read_csv(summary_path)
    summary_df = _clean_summary(summary_all, target=target, model=model, metric=metric)
    corr_col = "jointood_spearman" if metric.startswith("jointood_") and "jointood_spearman" in summary_df.columns else "loto_spearman"

    method_to_path: Dict[str, str] = {}
    if "method" in summary_df.columns and "path" in summary_df.columns:
        for _, row in summary_df.iterrows():
            method_to_path[str(row["method"])] = str(row["path"])

    hof_core_df = _resolve_hof_core_df(summary_df, hof_allow_bases)
    if hof_core_df.empty:
        raise SystemExit(
            "No HOF rows matched the configured --hof-allow-methods "
            f"({','.join(hof_allow_bases)})."
        )

    table1_rows: List[Dict[str, object]] = []
    table1_constrained_rows: List[Dict[str, object]] = []

    # H1: directional-asymmetric vs explicit MMD scalar baselines (parameter-matched when possible).
    flow_single_best: Optional[pd.Series] = None
    app_single_best: Optional[pd.Series] = None
    h1_specs = [
        ("flow", "Flow single", h1_motion_baseline_method, True),
        ("appearance", "Appearance single", h1_appearance_baseline_method, True),
        ("motion", "HOF motion", h1_motion_baseline_method, False),
    ]
    for family, label, base_name, single_only in h1_specs:
        fam_source = hof_core_df if family == "motion" else summary_df[summary_df["family"].astype(str) == family].copy()
        fam_asym_all = fam_source[fam_source["symmetry"].astype(str) == "asym"].copy()
        if fam_asym_all.empty:
            continue
        fam_asym = _single_family_subset(fam_asym_all, family) if single_only else fam_asym_all.copy()
        if fam_asym.empty:
            fam_asym = fam_asym_all.copy()

        base_df = summary_df[summary_df["method"].astype(str) == base_name].copy()
        if base_df.empty:
            raise SystemExit(
                f"H1 baseline method '{base_name}' not found in filtered summary. "
                "Recompile summary or adjust --h1-*baseline-method arguments."
            )
        base_row = _best_row(base_df, metric)
        if not _is_finite(base_row.get(metric)):
            raise SystemExit(
                f"H1 baseline method '{base_name}' has missing metric '{metric}'. "
                "Please rerun that baseline with Joint-OOD enabled."
            )

        base_n = int(_budget_value(base_row))
        asym_row, match_kind, near_n = _best_matched_or_nearest(fam_asym, base_n, metric)
        if asym_row is None:
            continue
        if match_kind == "exact":
            desc = f"{label}: best asymmetric vs {base_name} (matched n_predictors={base_n})"
        elif match_kind == "nearest":
            desc = (
                f"{label}: best asymmetric vs {base_name} requested n_predictors={base_n} "
                f"(no exact match; nearest n_predictors={near_n})"
            )
        else:
            desc = f"{label}: best asymmetric vs {base_name}"

        verdict = (
            "Supports H1 (asymmetric better)"
            if _as_float(asym_row[metric]) <= _as_float(base_row[metric])
            else "Against H1"
        )
        table1_rows.append(
            _comparison_row(
                hypothesis="H1",
                comparison=desc,
                a=asym_row,
                b=base_row,
                metric=metric,
                corr_col=corr_col,
                verdict=verdict,
            )
        )
        at_or_below = _best_at_or_below_budget(fam_asym, base_n, metric)
        if at_or_below is not None:
            table1_constrained_rows.append(
                _comparison_row(
                    hypothesis="H1",
                    comparison=f"{label}: budget-constrained (<=) best asymmetric vs {base_name} (n_predictors<={base_n})",
                    a=at_or_below,
                    b=base_row,
                    metric=metric,
                    corr_col=corr_col,
                    verdict="Supports H1 (asymmetric better)"
                    if _as_float(at_or_below[metric]) <= _as_float(base_row[metric])
                    else "Against H1",
                )
            )
        if family == "flow":
            flow_single_best = asym_row
        if family == "appearance":
            app_single_best = asym_row

    # H1 combined: asymmetric mixed methods (e.g., flow/hof + dino) vs symmetric combined MMD.
    mixed_asym = summary_df[
        (summary_df["family"].astype(str) == "mixed")
        & (summary_df["symmetry"].astype(str) == "asym")
    ].copy()
    mixed_asym = _exclude_method_substring(mixed_asym, "hof_density")
    if not mixed_asym.empty:
        comb_base_df = summary_df[summary_df["method"].astype(str) == h1_combined_baseline_method].copy()
        if comb_base_df.empty:
            raise SystemExit(
                f"H1 combined baseline method '{h1_combined_baseline_method}' not found in filtered summary. "
                "Recompile summary or adjust --h1-combined-baseline-method."
            )
        comb_base = _best_row(comb_base_df, metric)
        if not _is_finite(comb_base.get(metric)):
            raise SystemExit(
                f"H1 combined baseline method '{h1_combined_baseline_method}' has missing metric '{metric}'. "
                "Please rerun that baseline with Joint-OOD enabled."
            )

        best_mixed_asym = _best_row(mixed_asym, metric)
        table1_rows.append(
            _comparison_row(
                hypothesis="H1",
                comparison=f"Combined asym (flow/hof + dino): best asymmetric vs {h1_combined_baseline_method}",
                a=best_mixed_asym,
                b=comb_base,
                metric=metric,
                corr_col=corr_col,
                verdict="Supports H1 (asymmetric better)"
                if _as_float(best_mixed_asym[metric]) <= _as_float(comb_base[metric])
                else "Against H1",
            )
        )

        comb_n = int(_budget_value(comb_base))
        mixed_bcol = _budget_col(mixed_asym)
        mixed_asym_matched = mixed_asym[mixed_asym[mixed_bcol].apply(_as_float) == float(comb_n)].copy()
        if not mixed_asym_matched.empty:
            best_mixed_asym_matched = _best_row(mixed_asym_matched, metric)
            table1_rows.append(
                _comparison_row(
                    hypothesis="H1",
                    comparison=(
                        "Combined asym (flow/hof + dino): parameter-matched "
                        f"vs {h1_combined_baseline_method} at n_predictors={comb_n}"
                    ),
                    a=best_mixed_asym_matched,
                    b=comb_base,
                    metric=metric,
                    corr_col=corr_col,
                    verdict="Supports H1 (asymmetric better)"
                    if _as_float(best_mixed_asym_matched[metric]) <= _as_float(comb_base[metric])
                    else "Against H1",
                )
            )
            mixed_asym_le = _best_at_or_below_budget(mixed_asym, comb_n, metric)
            constrained_cmp = (
                "Combined asym (flow/hof + dino): budget-constrained (<=) "
                f"vs {h1_combined_baseline_method} at n_predictors<={comb_n}"
            )
            if mixed_asym_le is None:
                mixed_asym_le = best_mixed_asym_matched
                constrained_cmp += " (no feasible <= candidate; using parameter-matched row)"
            table1_constrained_rows.append(
                _comparison_row(
                    hypothesis="H1",
                    comparison=constrained_cmp,
                    a=mixed_asym_le,
                    b=comb_base,
                    metric=metric,
                    corr_col=corr_col,
                    verdict="Supports H1 (asymmetric better)"
                    if _as_float(mixed_asym_le[metric]) <= _as_float(comb_base[metric])
                    else "Against H1",
                )
            )
        else:
            mixed_asym_nearest = mixed_asym.copy()
            mixed_asym_nearest["n_diff"] = (
                mixed_asym_nearest[mixed_bcol].apply(_as_float) - float(comb_n)
            ).abs()
            mixed_asym_nearest = mixed_asym_nearest.sort_values(["n_diff", metric], ascending=[True, True])
            best_mixed_asym_nearest = mixed_asym_nearest.iloc[0]
            near_n = int(_as_float(best_mixed_asym_nearest.get(mixed_bcol)))
            table1_rows.append(
                _comparison_row(
                    hypothesis="H1",
                    comparison=(
                        "Combined asym (flow/hof + dino): parameter-matched "
                        f"vs {h1_combined_baseline_method} requested n_predictors={comb_n} "
                        f"(no exact match; nearest n_predictors={near_n})"
                    ),
                    a=best_mixed_asym_nearest,
                    b=comb_base,
                    metric=metric,
                    corr_col=corr_col,
                    verdict="Supports H1 (asymmetric better)"
                    if _as_float(best_mixed_asym_nearest[metric]) <= _as_float(comb_base[metric])
                    else "Against H1",
                )
            )
            mixed_asym_le = _best_at_or_below_budget(mixed_asym, comb_n, metric)
            constrained_cmp = (
                "Combined asym (flow/hof + dino): budget-constrained (<=) "
                f"vs {h1_combined_baseline_method} at n_predictors<={comb_n}"
            )
            if mixed_asym_le is None:
                mixed_asym_le = best_mixed_asym_nearest
                constrained_cmp += " (no feasible <= candidate; using nearest row)"
            table1_constrained_rows.append(
                _comparison_row(
                    hypothesis="H1",
                    comparison=constrained_cmp,
                    a=mixed_asym_le,
                    b=comb_base,
                    metric=metric,
                    corr_col=corr_col,
                    verdict="Supports H1 (asymmetric better)"
                    if _as_float(mixed_asym_le[metric]) <= _as_float(comb_base[metric])
                    else "Against H1",
                )
            )

    # H2: family-level comparisons.
    best_flow = _best_by_family(summary_df, family="flow", metric=metric)
    best_app = _best_by_family(summary_df, family="appearance", metric=metric)
    mixed_df = summary_df[summary_df["family"].astype(str) == "mixed"].copy()
    mixed_no_density = _exclude_method_substring(mixed_df, "hof_density")
    if not mixed_no_density.empty:
        best_mixed = _best_row(mixed_no_density, metric)
    else:
        best_mixed = _best_row(mixed_df, metric)
    hof_primary_row = None
    if (
        hof_primary_method
        and "method" in hof_core_df.columns
        and (hof_core_df["method"].astype(str) == hof_primary_method).any()
    ):
        hof_primary_row = hof_core_df[hof_core_df["method"].astype(str) == hof_primary_method].iloc[0]
    best_hof = hof_primary_row if hof_primary_row is not None else _best_row(hof_core_df, metric)
    hof_h2_label = "Primary HOF method" if hof_primary_row is not None else "Best HOF method"

    app_budget = int(_budget_value(best_app))
    flow_budget_matched = _best_by_family_and_budget(
        summary_df, family="flow", n_predictors=app_budget, metric=metric
    )
    hof_budget_matched = _best_by_budget(hof_core_df, n_predictors=app_budget, metric=metric)
    if hof_primary_row is not None and int(_as_float(hof_primary_row.get("n_predictors"))) == app_budget:
        hof_budget_matched = hof_primary_row

    table1_rows.append(
        _comparison_row(
            hypothesis="H2",
            comparison="Best flow-family method vs best appearance method",
            a=best_flow,
            b=best_app,
            metric=metric,
            corr_col=corr_col,
            verdict="Supports H2 (motion-family better)"
            if _as_float(best_flow[metric]) <= _as_float(best_app[metric])
            else "Against H2",
        )
    )
    table1_rows.append(
        _comparison_row(
            hypothesis="H2",
            comparison=f"{hof_h2_label} vs best appearance method",
            a=best_hof,
            b=best_app,
            metric=metric,
            corr_col=corr_col,
            verdict="Supports H2 (HOF >= appearance)"
            if _as_float(best_hof[metric]) <= _as_float(best_app[metric])
            else "HOF below appearance on this metric",
        )
    )
    table1_rows.append(
        _comparison_row(
            hypothesis="H2",
            comparison="Best mixed method vs best appearance method",
            a=best_mixed,
            b=best_app,
            metric=metric,
            corr_col=corr_col,
            verdict="Supports H2 (combination helps)"
            if _as_float(best_mixed[metric]) <= _as_float(best_app[metric])
            else "Mixed below appearance on this metric",
        )
    )
    if flow_budget_matched is not None:
        table1_rows.append(
            _comparison_row(
                hypothesis="H2",
                comparison=f"Parameter-matched: best flow at n_predictors={app_budget} vs best appearance",
                a=flow_budget_matched,
                b=best_app,
                metric=metric,
                corr_col=corr_col,
                verdict="Supports H2 (flow better at matched budget)"
                if _as_float(flow_budget_matched[metric]) <= _as_float(best_app[metric])
                else "Flow worse at matched budget",
            )
        )
        flow_at_or_below = _best_at_or_below_budget(
            summary_df[summary_df["family"].astype(str) == "flow"].copy(),
            n_predictors=app_budget,
            metric=metric,
        )
        flow_constrained_cmp = f"Budget-constrained (<=): best flow vs best appearance at n_predictors<={app_budget}"
        if flow_at_or_below is None:
            flow_at_or_below = flow_budget_matched
            flow_constrained_cmp += " (no feasible <= candidate; using parameter-matched row)"
        table1_constrained_rows.append(
            _comparison_row(
                hypothesis="H2",
                comparison=flow_constrained_cmp,
                a=flow_at_or_below,
                b=best_app,
                metric=metric,
                corr_col=corr_col,
                verdict="Supports H2 (flow better at constrained budget)"
                if _as_float(flow_at_or_below[metric]) <= _as_float(best_app[metric])
                else "Flow worse at constrained budget",
            )
        )
    for n_budget in _shared_budgets(summary_df, family_a="flow", family_b="appearance"):
        if n_budget == app_budget:
            continue
        flow_n = _best_by_family_and_budget(summary_df, family="flow", n_predictors=n_budget, metric=metric)
        app_n = _best_by_family_and_budget(
            summary_df, family="appearance", n_predictors=n_budget, metric=metric
        )
        if flow_n is None or app_n is None:
            continue
        table1_rows.append(
            _comparison_row(
                hypothesis="H2",
                comparison=f"Parameter-matched: best flow vs best appearance at n_predictors={n_budget}",
                a=flow_n,
                b=app_n,
                metric=metric,
                corr_col=corr_col,
                verdict="Supports H2 (flow better at matched budget)"
                if _as_float(flow_n[metric]) <= _as_float(app_n[metric])
                else "Flow worse at matched budget",
            )
        )
        flow_le_n = _best_at_or_below_budget(
            summary_df[summary_df["family"].astype(str) == "flow"].copy(),
            n_predictors=n_budget,
            metric=metric,
        )
        flow_constrained_cmp = f"Budget-constrained (<=): best flow vs best appearance at n_predictors<={n_budget}"
        if flow_le_n is None:
            flow_le_n = flow_n
            flow_constrained_cmp += " (no feasible <= candidate; using parameter-matched row)"
        table1_constrained_rows.append(
            _comparison_row(
                hypothesis="H2",
                comparison=flow_constrained_cmp,
                a=flow_le_n,
                b=app_n,
                metric=metric,
                corr_col=corr_col,
                verdict="Supports H2 (flow better at constrained budget)"
                if _as_float(flow_le_n[metric]) <= _as_float(app_n[metric])
                else "Flow worse at constrained budget",
            )
        )
    if hof_budget_matched is not None:
        table1_rows.append(
            _comparison_row(
                hypothesis="H2",
                comparison=f"Parameter-matched: best HOF at n_predictors={app_budget} vs best appearance",
                a=hof_budget_matched,
                b=best_app,
                metric=metric,
                corr_col=corr_col,
                verdict="Supports H2 (HOF better at matched budget)"
                if _as_float(hof_budget_matched[metric]) <= _as_float(best_app[metric])
                else "HOF worse at matched budget",
            )
        )
        hof_at_or_below = _best_at_or_below_budget(
            hof_core_df,
            n_predictors=app_budget,
            metric=metric,
        )
        hof_constrained_cmp = f"Budget-constrained (<=): best HOF vs best appearance at n_predictors<={app_budget}"
        if hof_at_or_below is None:
            hof_at_or_below = hof_budget_matched
            hof_constrained_cmp += " (no feasible <= candidate; using parameter-matched row)"
        table1_constrained_rows.append(
            _comparison_row(
                hypothesis="H2",
                comparison=hof_constrained_cmp,
                a=hof_at_or_below,
                b=best_app,
                metric=metric,
                corr_col=corr_col,
                verdict="Supports H2 (HOF better at constrained budget)"
                if _as_float(hof_at_or_below[metric]) <= _as_float(best_app[metric])
                else "HOF worse at constrained budget",
            )
        )
    app_df = summary_df[summary_df["family"].astype(str) == "appearance"].copy()
    for n_budget in _shared_budgets_between(hof_core_df, app_df):
        if n_budget == app_budget:
            continue
        hof_n = _best_by_budget(hof_core_df, n_predictors=n_budget, metric=metric)
        app_n = _best_by_budget(app_df, n_predictors=n_budget, metric=metric)
        if hof_n is None or app_n is None:
            continue
        table1_rows.append(
            _comparison_row(
                hypothesis="H2",
                comparison=f"Parameter-matched: best HOF vs best appearance at n_predictors={n_budget}",
                a=hof_n,
                b=app_n,
                metric=metric,
                corr_col=corr_col,
                verdict="Supports H2 (HOF better at matched budget)"
                if _as_float(hof_n[metric]) <= _as_float(app_n[metric])
                else "HOF worse at matched budget",
            )
        )
        hof_le_n = _best_at_or_below_budget(
            hof_core_df,
            n_predictors=n_budget,
            metric=metric,
        )
        hof_constrained_cmp = f"Budget-constrained (<=): best HOF vs best appearance at n_predictors<={n_budget}"
        if hof_le_n is None:
            hof_le_n = hof_n
            hof_constrained_cmp += " (no feasible <= candidate; using parameter-matched row)"
        table1_constrained_rows.append(
            _comparison_row(
                hypothesis="H2",
                comparison=hof_constrained_cmp,
                a=hof_le_n,
                b=app_n,
                metric=metric,
                corr_col=corr_col,
                verdict="Supports H2 (HOF better at constrained budget)"
                if _as_float(hof_le_n[metric]) <= _as_float(app_n[metric])
                else "HOF worse at constrained budget",
            )
        )

    # Keep constrained comparisons together in a contiguous block.
    table1_rows.extend(table1_constrained_rows)

    # Table 2: compact representatives.
    table2_rows: List[Dict[str, object]] = []
    table2_rows.append(
        _method_row(
            best_flow,
            source_path=method_to_path.get(str(best_flow["method"]), ""),
            note="Best flow-family representative",
        )
    )

    flow_triplet = _directional_triplet_best(summary_df, family="flow", metric=metric)
    if flow_triplet is not None:
        flow_dir = flow_triplet[1]
        if str(flow_dir["method"]) != str(best_flow["method"]):
            table2_rows.append(
                _method_row(
                    flow_dir,
                    source_path=method_to_path.get(str(flow_dir["method"]), ""),
                    note="Best single-direction flow comparator",
                )
            )

    table2_rows.append(
        _method_row(
            best_hof,
            source_path=method_to_path.get(str(best_hof["method"]), ""),
            note="Primary HOF motion representative"
            if hof_primary_row is not None
            else "Best HOF motion representative",
        )
    )
    hof_alt = _best_row(hof_core_df, metric)
    if str(hof_alt["method"]) != str(best_hof["method"]):
        table2_rows.append(
            _method_row(
                hof_alt,
                source_path=method_to_path.get(str(hof_alt["method"]), ""),
                note="Best HOF under allowed method set",
            )
        )
    table2_rows.append(
        _method_row(
            best_app,
            source_path=method_to_path.get(str(best_app["method"]), ""),
            note="Best appearance representative",
        )
    )
    if flow_single_best is not None and str(flow_single_best["method"]) != str(best_flow["method"]):
        table2_rows.append(
            _method_row(
                flow_single_best,
                source_path=method_to_path.get(str(flow_single_best["method"]), ""),
                note="Best single-flow representative (MMD-matched comparison)",
            )
        )
    if app_single_best is not None and str(app_single_best["method"]) != str(best_app["method"]):
        table2_rows.append(
            _method_row(
                app_single_best,
                source_path=method_to_path.get(str(app_single_best["method"]), ""),
                note="Best single-appearance representative (MMD-matched comparison)",
            )
        )
    if flow_budget_matched is not None and str(flow_budget_matched["method"]) != str(best_flow["method"]):
        table2_rows.append(
            _method_row(
                flow_budget_matched,
                source_path=method_to_path.get(str(flow_budget_matched["method"]), ""),
                note=f"Best flow at matched appearance budget (n_predictors={app_budget})",
            )
        )
    table2_rows.append(
        _method_row(
            best_mixed,
            source_path=method_to_path.get(str(best_mixed["method"]), ""),
            note="Best mixed representative (hof_density excluded)",
        )
    )

    if "symmetry" in summary_df.columns:
        sym_df = summary_df[summary_df["symmetry"].astype(str) == "sym"]
        if not sym_df.empty:
            best_sym = _best_row(sym_df, metric)
            table2_rows.append(
                _method_row(
                    best_sym,
                    source_path=method_to_path.get(str(best_sym["method"]), ""),
                    note="Best symmetric baseline",
                )
            )
    for base_name, note in [
        (h1_motion_baseline_method, "Flow MMD baseline"),
        (h1_appearance_baseline_method, "Appearance MMD baseline"),
        (h1_combined_baseline_method, "Combined MMD baseline"),
    ]:
        base_df = summary_df[summary_df["method"].astype(str) == base_name].copy()
        if base_df.empty:
            continue
        base_row = _best_row(base_df, metric)
        if not _is_finite(base_row.get(metric)):
            continue
        table2_rows.append(
            _method_row(
                base_row,
                source_path=method_to_path.get(str(base_row["method"]), ""),
                note=note,
            )
        )

    if highlight_run is not None:
        table2_rows.append(_extract_highlighted_run(highlight_run))

    # Keep table2 method order stable and unique by method.
    seen = set()
    deduped: List[Dict[str, object]] = []
    for row in table2_rows:
        m = str(row["method"])
        if m in seen:
            continue
        seen.add(m)
        deduped.append(row)
    table2_rows = deduped

    _write_outputs(
        out_dir=out_dir,
        table1=table1_rows,
        table2=table2_rows,
        summary_path=summary_path,
        highlighted_run=highlight_run,
        metric=metric,
        corr_col=corr_col,
    )


if __name__ == "__main__":
    main()
