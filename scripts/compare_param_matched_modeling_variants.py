#!/usr/bin/env python3
"""
Compare three modeling variants on matched parameter settings.

For each variant, this script reads LOTO parameter-selection exports
(`heldout_protocol_fit_metrics.csv`), follows `selected_run_dir`, and extracts
overall LOTO/LOBO metrics from the underlying run summaries.

Outputs:
- param_matched_loto_comparison.csv
- param_matched_lobo_comparison.csv
- param_matched_protocol_summary.csv
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


PARAM_KEY_RE = re.compile(r"^(?:leakage_free_)?k\d+__")


@dataclass(frozen=True)
class VariantSpec:
    key: str
    root: Path
    loto_exports_subdir: str


def _normalize_param_key(name: str) -> str:
    key = name.strip()
    if key.startswith("leakage_free_"):
        key = key[len("leakage_free_") :]
    key = key.replace("__density_as_interactions", "")
    return key


def _safe_float(value: object) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isfinite(v):
        return v
    return float("nan")


def _read_overall_metric(
    run_dir: Path,
    filename: str,
    id_col: str,
    overall_token: str,
    metric_col: str,
) -> float:
    path = run_dir / filename
    if not path.exists():
        return float("nan")
    try:
        df = pd.read_csv(path)
    except Exception:
        return float("nan")
    if df.empty or metric_col not in df.columns:
        return float("nan")

    if id_col in df.columns:
        sub = df[df[id_col].astype(str) == overall_token]
        if sub.empty:
            return float("nan")
        return _safe_float(sub.iloc[0][metric_col])
    return _safe_float(df.iloc[0][metric_col])


def _resolve_run_dir(path_like: object, cwd: Path) -> Optional[Path]:
    if not isinstance(path_like, str):
        return None
    text = path_like.strip()
    if not text:
        return None
    p = Path(text)
    if p.exists():
        return p
    rel = (cwd / p).resolve()
    if rel.exists():
        return rel
    return None


def _load_variant_rows(
    spec: VariantSpec,
    *,
    space: str,
    loto_protocol: str,
    loto_summary_file: str,
    loto_metric_col: str,
    loto_id_col: str,
    loto_overall_token: str,
    lobo_summary_file: str,
    lobo_metric_col: str,
    lobo_id_col: str,
    lobo_overall_token: str,
    cwd: Path,
) -> pd.DataFrame:
    exports_root = spec.root / spec.loto_exports_subdir
    if not exports_root.exists():
        return pd.DataFrame()

    records: List[Dict[str, object]] = []
    for csv_path in sorted(exports_root.rglob("heldout_protocol_fit_metrics.csv")):
        parent_name = csv_path.parent.name

        try:
            fit_df = pd.read_csv(csv_path)
        except Exception:
            continue
        if fit_df.empty:
            continue

        rows = fit_df[(fit_df["protocol"] == loto_protocol) & (fit_df["space"] == space)]
        if rows.empty:
            continue
        row = rows.iloc[0]

        run_dir = _resolve_run_dir(row.get("selected_run_dir"), cwd=cwd)
        if run_dir is None:
            continue

        key_from_parent = (
            _normalize_param_key(parent_name) if PARAM_KEY_RE.match(parent_name) else None
        )
        key_from_run = (
            _normalize_param_key(run_dir.name) if PARAM_KEY_RE.match(run_dir.name) else None
        )
        param_key = key_from_parent or key_from_run
        if not param_key:
            continue

        loto_metric = _read_overall_metric(
            run_dir=run_dir,
            filename=loto_summary_file,
            id_col=loto_id_col,
            overall_token=loto_overall_token,
            metric_col=loto_metric_col,
        )
        lobo_metric = _read_overall_metric(
            run_dir=run_dir,
            filename=lobo_summary_file,
            id_col=lobo_id_col,
            overall_token=lobo_overall_token,
            metric_col=lobo_metric_col,
        )

        records.append(
            {
                "variant": spec.key,
                "param_key": param_key,
                "loto_metric": loto_metric,
                "lobo_metric": lobo_metric,
                "selected_run_dir": str(run_dir),
                "source_fit_csv": str(csv_path),
            }
        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)
    # De-duplicate key collisions by preferring rows with finite LOTO and LOBO.
    df["valid_count"] = df[["loto_metric", "lobo_metric"]].apply(
        lambda r: int(math.isfinite(_safe_float(r["loto_metric"])))
        + int(math.isfinite(_safe_float(r["lobo_metric"]))),
        axis=1,
    )
    df = df.sort_values(["param_key", "valid_count"], ascending=[True, False])
    df = df.drop_duplicates(subset=["param_key"], keep="first").drop(columns=["valid_count"])
    return df.reset_index(drop=True)


def _intersection_keys(frames: Sequence[pd.DataFrame], metric_col: str) -> List[str]:
    key_sets: List[set[str]] = []
    for df in frames:
        if df.empty:
            return []
        sub = df[pd.to_numeric(df[metric_col], errors="coerce").notna()]
        key_sets.append(set(sub["param_key"].astype(str)))
    if not key_sets:
        return []
    keys = set.intersection(*key_sets)
    return sorted(keys)


def _metric_dict(df: pd.DataFrame, metric_col: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for _, r in df.iterrows():
        out[str(r["param_key"])] = _safe_float(r[metric_col])
    return out


def _best_winners(
    per_variant_values: Dict[str, float],
    *,
    higher_is_better: bool,
    tie_tol: float,
) -> Tuple[List[str], float, float]:
    values = {k: v for k, v in per_variant_values.items() if math.isfinite(v)}
    if not values:
        return ([], float("nan"), float("nan"))

    best_value = max(values.values()) if higher_is_better else min(values.values())
    winners = [
        k
        for k, v in values.items()
        if abs(v - best_value) <= tie_tol
    ]

    if len(values) <= 1:
        margin = float("nan")
    else:
        sorted_vals = sorted(values.values(), reverse=higher_is_better)
        margin = sorted_vals[0] - sorted_vals[1] if higher_is_better else sorted_vals[1] - sorted_vals[0]
    return (sorted(winners), best_value, margin)


def _build_protocol_tables(
    *,
    protocol_name: str,
    variant_frames: Dict[str, pd.DataFrame],
    metric_col: str,
    higher_is_better: bool,
    tie_tol: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ordered_variants = list(variant_frames.keys())
    keys = _intersection_keys(list(variant_frames.values()), metric_col=metric_col)
    metric_maps = {
        v: _metric_dict(df, metric_col=metric_col)
        for v, df in variant_frames.items()
    }

    rows: List[Dict[str, object]] = []
    wins_raw = {v: 0 for v in ordered_variants}
    wins_fractional = {v: 0.0 for v in ordered_variants}
    rank_values: Dict[str, List[float]] = {v: [] for v in ordered_variants}
    mean_values: Dict[str, List[float]] = {v: [] for v in ordered_variants}

    for key in keys:
        values = {v: metric_maps[v].get(key, float("nan")) for v in ordered_variants}
        winners, best_value, margin = _best_winners(
            values, higher_is_better=higher_is_better, tie_tol=tie_tol
        )
        for w in winners:
            wins_raw[w] += 1
            wins_fractional[w] += 1.0 / float(len(winners))

        s = pd.Series(values, dtype=float)
        ranks = s.rank(ascending=not higher_is_better, method="average")
        for v in ordered_variants:
            val = _safe_float(values[v])
            if math.isfinite(val):
                mean_values[v].append(val)
            rv = _safe_float(ranks.get(v, float("nan")))
            if math.isfinite(rv):
                rank_values[v].append(rv)

        row: Dict[str, object] = {
            "protocol": protocol_name,
            "param_key": key,
            "winners": "|".join(winners),
            "best_value": best_value,
            "margin_to_second": margin,
        }
        for v in ordered_variants:
            row[f"{v}_metric"] = values[v]
        rows.append(row)

    detail_df = pd.DataFrame(rows)

    summary_rows: List[Dict[str, object]] = []
    n_cases = len(keys)
    for v in ordered_variants:
        mean_metric = float("nan") if not mean_values[v] else float(pd.Series(mean_values[v]).mean())
        median_metric = float("nan") if not mean_values[v] else float(pd.Series(mean_values[v]).median())
        mean_rank = float("nan") if not rank_values[v] else float(pd.Series(rank_values[v]).mean())
        frac = wins_fractional[v]
        summary_rows.append(
            {
                "protocol": protocol_name,
                "variant": v,
                "n_common_cases": n_cases,
                "wins_raw": wins_raw[v],
                "wins_fractional": frac,
                "win_rate_fractional": (frac / float(n_cases)) if n_cases else float("nan"),
                "mean_metric": mean_metric,
                "median_metric": median_metric,
                "mean_rank": mean_rank,
                "higher_is_better": higher_is_better,
                "metric_col": metric_col,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    return detail_df, summary_df


def _oriented_delta(first: float, second: float, *, higher_is_better: bool) -> float:
    if not (math.isfinite(first) and math.isfinite(second)):
        return float("nan")
    if higher_is_better:
        return first - second
    return second - first


def _build_pairwise_delta_tables(
    *,
    protocol_name: str,
    detail_df: pd.DataFrame,
    variants: Sequence[str],
    higher_is_better: bool,
    tie_tol: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pairwise_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for a, b in combinations(variants, 2):
        a_col = f"{a}_metric"
        b_col = f"{b}_metric"
        if a_col not in detail_df.columns or b_col not in detail_df.columns:
            continue

        sub = detail_df[["param_key", a_col, b_col]].copy()
        sub = sub.rename(columns={a_col: "metric_a", b_col: "metric_b"})
        sub["raw_delta_a_minus_b"] = sub["metric_a"] - sub["metric_b"]
        sub["oriented_delta_a_vs_b"] = sub.apply(
            lambda r: _oriented_delta(
                _safe_float(r["metric_a"]),
                _safe_float(r["metric_b"]),
                higher_is_better=higher_is_better,
            ),
            axis=1,
        )
        sub["protocol"] = protocol_name
        sub["variant_a"] = a
        sub["variant_b"] = b

        for _, r in sub.iterrows():
            pairwise_rows.append(
                {
                    "protocol": protocol_name,
                    "param_key": r["param_key"],
                    "variant_a": a,
                    "variant_b": b,
                    "metric_a": _safe_float(r["metric_a"]),
                    "metric_b": _safe_float(r["metric_b"]),
                    "raw_delta_a_minus_b": _safe_float(r["raw_delta_a_minus_b"]),
                    "oriented_delta_a_vs_b": _safe_float(r["oriented_delta_a_vs_b"]),
                }
            )

        valid = sub[pd.to_numeric(sub["oriented_delta_a_vs_b"], errors="coerce").notna()].copy()
        if valid.empty:
            continue

        oriented = valid["oriented_delta_a_vs_b"].astype(float)
        raw = valid["raw_delta_a_minus_b"].astype(float)
        a_better = int((oriented > tie_tol).sum())
        b_better = int((oriented < -tie_tol).sum())
        ties = int((oriented.abs() <= tie_tol).sum())
        n = int(len(oriented))

        summary_rows.append(
            {
                "protocol": protocol_name,
                "variant_a": a,
                "variant_b": b,
                "n_cases": n,
                "higher_is_better": higher_is_better,
                "mean_raw_delta_a_minus_b": float(raw.mean()),
                "median_raw_delta_a_minus_b": float(raw.median()),
                "mean_abs_raw_delta": float(raw.abs().mean()),
                "mean_oriented_delta_a_vs_b": float(oriented.mean()),
                "median_oriented_delta_a_vs_b": float(oriented.median()),
                "mean_abs_oriented_delta": float(oriented.abs().mean()),
                "a_better_count": a_better,
                "b_better_count": b_better,
                "tie_count": ties,
                "a_better_frac": float(a_better / n) if n else float("nan"),
                "b_better_frac": float(b_better / n) if n else float("nan"),
                "tie_frac": float(ties / n) if n else float("nan"),
            }
        )

    return pd.DataFrame(pairwise_rows), pd.DataFrame(summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare matched parameter cases across three modeling variants."
    )
    parser.add_argument(
        "--interaction-only-root",
        type=Path,
        default=Path(
            "analysis_comprehensive_runs/"
            "ridge_resid_weighted_ridge_a10_no_family_interaction_only_baseline_matched_v1"
        ),
    )
    parser.add_argument(
        "--offsets-interactions-root",
        type=Path,
        default=Path(
            "analysis_comprehensive_runs/"
            "ridge_resid_weighted_ridge_a10_no_family_interactions_baseline_matched_v1"
        ),
    )
    parser.add_argument(
        "--no-interactions-root",
        type=Path,
        default=Path(
            "analysis_comprehensive_runs/"
            "ridge_resid_weighted_ridge_a10_no_family_no_density_zscore_zeroshot_v3"
        ),
    )

    parser.add_argument(
        "--interaction-only-subdir",
        default="paper_plots_loto_collapsed_train_only_diag",
        help="Relative folder inside interaction-only root with per-parameter LOTO exports.",
    )
    parser.add_argument(
        "--offsets-interactions-subdir",
        default="paper_plots_param_matched_loto_pair_win_k32_interactions_baseline_matched_color_train_dataset_loto",
        help="Relative folder inside offsets+interactions root with per-parameter LOTO exports.",
    )
    parser.add_argument(
        "--no-interactions-subdir",
        default="paper_plots_param_matched_loto_pair_win_k32_color_train_dataset_loto",
        help="Relative folder inside no-interactions root with per-parameter LOTO exports.",
    )

    parser.add_argument("--space", default="model_space")
    parser.add_argument("--loto-protocol", default="loto")

    parser.add_argument("--loto-summary-file", default="prediction_loto_holdout_placement_summary.csv")
    parser.add_argument("--loto-id-col", default="fold")
    parser.add_argument("--loto-overall-token", default="__overall__")
    parser.add_argument("--loto-metric-col", default="rank_spearman_micro")
    parser.add_argument("--loto-lower-is-better", action="store_true")

    parser.add_argument("--lobo-summary-file", default="prediction_lobo_rank_summary.csv")
    parser.add_argument("--lobo-id-col", default="benchmark")
    parser.add_argument("--lobo-overall-token", default="__overall__")
    parser.add_argument("--lobo-metric-col", default="spearman")
    parser.add_argument("--lobo-lower-is-better", action="store_true")

    parser.add_argument(
        "--tie-tol",
        type=float,
        default=1e-12,
        help="Absolute tolerance used to consider values tied.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_comprehensive_runs/param_matched_modeling_variant_comparison"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()

    specs = [
        VariantSpec(
            key="interaction_only",
            root=args.interaction_only_root,
            loto_exports_subdir=args.interaction_only_subdir,
        ),
        VariantSpec(
            key="offsets_interactions",
            root=args.offsets_interactions_root,
            loto_exports_subdir=args.offsets_interactions_subdir,
        ),
        VariantSpec(
            key="no_interactions",
            root=args.no_interactions_root,
            loto_exports_subdir=args.no_interactions_subdir,
        ),
    ]

    variant_frames: Dict[str, pd.DataFrame] = {}
    for spec in specs:
        df = _load_variant_rows(
            spec,
            space=args.space,
            loto_protocol=args.loto_protocol,
            loto_summary_file=args.loto_summary_file,
            loto_metric_col=args.loto_metric_col,
            loto_id_col=args.loto_id_col,
            loto_overall_token=args.loto_overall_token,
            lobo_summary_file=args.lobo_summary_file,
            lobo_metric_col=args.lobo_metric_col,
            lobo_id_col=args.lobo_id_col,
            lobo_overall_token=args.lobo_overall_token,
            cwd=cwd,
        )
        variant_frames[spec.key] = df

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    loto_detail, loto_summary = _build_protocol_tables(
        protocol_name="loto",
        variant_frames=variant_frames,
        metric_col="loto_metric",
        higher_is_better=not args.loto_lower_is_better,
        tie_tol=args.tie_tol,
    )
    lobo_detail, lobo_summary = _build_protocol_tables(
        protocol_name="lobo",
        variant_frames=variant_frames,
        metric_col="lobo_metric",
        higher_is_better=not args.lobo_lower_is_better,
        tie_tol=args.tie_tol,
    )
    all_summary = pd.concat([loto_summary, lobo_summary], ignore_index=True)
    variants = list(variant_frames.keys())

    loto_pairwise, loto_pairwise_summary = _build_pairwise_delta_tables(
        protocol_name="loto",
        detail_df=loto_detail,
        variants=variants,
        higher_is_better=not args.loto_lower_is_better,
        tie_tol=args.tie_tol,
    )
    lobo_pairwise, lobo_pairwise_summary = _build_pairwise_delta_tables(
        protocol_name="lobo",
        detail_df=lobo_detail,
        variants=variants,
        higher_is_better=not args.lobo_lower_is_better,
        tie_tol=args.tie_tol,
    )
    pairwise_detail = pd.concat([loto_pairwise, lobo_pairwise], ignore_index=True)
    pairwise_summary = pd.concat(
        [loto_pairwise_summary, lobo_pairwise_summary], ignore_index=True
    )

    loto_detail_path = out_dir / "param_matched_loto_comparison.csv"
    lobo_detail_path = out_dir / "param_matched_lobo_comparison.csv"
    summary_path = out_dir / "param_matched_protocol_summary.csv"
    pairwise_detail_path = out_dir / "param_matched_pairwise_deltas.csv"
    pairwise_summary_path = out_dir / "param_matched_pairwise_delta_summary.csv"

    loto_detail.to_csv(loto_detail_path, index=False)
    lobo_detail.to_csv(lobo_detail_path, index=False)
    all_summary.to_csv(summary_path, index=False)
    pairwise_detail.to_csv(pairwise_detail_path, index=False)
    pairwise_summary.to_csv(pairwise_summary_path, index=False)

    print("Wrote:")
    print(f"  {loto_detail_path}")
    print(f"  {lobo_detail_path}")
    print(f"  {summary_path}")
    print(f"  {pairwise_detail_path}")
    print(f"  {pairwise_summary_path}")
    print("")
    print("Common-case counts:")
    print(f"  LOTO: {len(loto_detail)}")
    print(f"  LOBO: {len(lobo_detail)}")
    print("")
    print("Protocol summary:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(all_summary.to_string(index=False))
    print("")
    print("Pairwise delta summary:")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(pairwise_summary.to_string(index=False))


if __name__ == "__main__":
    main()
