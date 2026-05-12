#!/usr/bin/env python3
"""
Build a compact, camera-ready results CSV from existing final table outputs.

This script does not run new experiments. It only aggregates prior outputs:
  - predictor_group_tables/*
  - organized_param_grid/*
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import pandas as pd


DEFAULT_METRICS = [
    "loto_rank_pairwise_cindex",
    "loto_rank_spearman",
    "loto_mae",
    "loto_rmse",
]


def _split_csv_arg(text: str) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def _is_error_metric(name: str) -> bool:
    key = str(name).lower()
    return any(x in key for x in ("mae", "rmse", "regret", "abs_err"))


def _higher_is_better(name: str) -> bool:
    return not _is_error_metric(name)


def _oriented_delta(metric: str, first: float, second: float) -> float:
    if not (math.isfinite(first) and math.isfinite(second)):
        return float("nan")
    raw = float(first - second)
    return raw if _higher_is_better(metric) else -raw


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _append_directionality_rows(rows: List[Dict[str, object]], predictor_dir: Path, metrics: List[str]) -> None:
    path = predictor_dir / "method_summary_with_derived_groups.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if "symmetry" not in df.columns:
        return
    asym = df[df["symmetry"] == "asym"].copy()
    sym = df[df["symmetry"] == "sym"].copy()
    for metric in metrics:
        if metric not in df.columns:
            continue
        asym_vals = _to_num(asym[metric]).dropna()
        sym_vals = _to_num(sym[metric]).dropna()
        if asym_vals.empty or sym_vals.empty:
            continue
        asym_mean = float(asym_vals.mean())
        sym_mean = float(sym_vals.mean())
        asym_median = float(asym_vals.median())
        sym_median = float(sym_vals.median())
        rows.append(
            {
                "block": "A. Directional vs Symmetric",
                "comparison": "Asymmetric vs Symmetric",
                "metric": metric,
                "n_left": int(asym_vals.shape[0]),
                "n_right": int(sym_vals.shape[0]),
                "left_label": "asymmetric",
                "right_label": "symmetric",
                "left_value_mean": asym_mean,
                "right_value_mean": sym_mean,
                "delta_oriented_mean": _oriented_delta(metric, asym_mean, sym_mean),
                "left_value_median": asym_median,
                "right_value_median": sym_median,
                "delta_oriented_median": _oriented_delta(metric, asym_median, sym_median),
                "delta_oriented_pos_frac": float("nan"),
                "source": str(path),
            }
        )


def _append_pure_modality_rows(rows: List[Dict[str, object]], predictor_dir: Path, metrics: List[str]) -> None:
    path = predictor_dir / "pure_modalities_matched_k_deltas.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    pair_defs = [
        ("flow", "appearance"),
        ("flow", "hof"),
        ("hof", "appearance"),
    ]
    for metric in metrics:
        for first, second in pair_defs:
            v_first_col = f"{metric}__{first}"
            v_second_col = f"{metric}__{second}"
            delta_col = f"{metric}__{first}_minus_{second}_oriented"
            if v_first_col not in df.columns or v_second_col not in df.columns or delta_col not in df.columns:
                continue
            v_first = _to_num(df[v_first_col])
            v_second = _to_num(df[v_second_col])
            delta = _to_num(df[delta_col])
            valid = delta.notna()
            if not valid.any():
                continue
            rows.append(
                {
                    "block": "B. Pure Modality Matched-k",
                    "comparison": f"{first} vs {second}",
                    "metric": metric,
                    "n_left": int(v_first[valid].notna().sum()),
                    "n_right": int(v_second[valid].notna().sum()),
                    "left_label": first,
                    "right_label": second,
                    "left_value_mean": float(v_first[valid].mean()),
                    "right_value_mean": float(v_second[valid].mean()),
                    "delta_oriented_mean": float(delta[valid].mean()),
                    "left_value_median": float(v_first[valid].median()),
                    "right_value_median": float(v_second[valid].median()),
                    "delta_oriented_median": float(delta[valid].median()),
                    "delta_oriented_pos_frac": float((delta[valid] > 0).mean()),
                    "source": str(path),
                }
            )


def _append_incremental_rows(rows: List[Dict[str, object]], predictor_dir: Path, metrics: List[str]) -> None:
    path = predictor_dir / "incremental_domain_addition_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty or "added_domain" not in df.columns or "n_edges" not in df.columns:
        return
    for _, r in df.iterrows():
        domain = str(r["added_domain"])
        n_edges = int(r["n_edges"]) if pd.notna(r["n_edges"]) else 0
        for metric in metrics:
            mean_col = f"{metric}__oriented_delta_mean"
            med_col = f"{metric}__oriented_delta_median"
            pos_col = f"{metric}__oriented_delta_pos_frac"
            if mean_col not in df.columns:
                continue
            rows.append(
                {
                    "block": "C. Incremental Addition",
                    "comparison": f"+{domain} one-step",
                    "metric": metric,
                    "n_left": n_edges,
                    "n_right": 0,
                    "left_label": "edges",
                    "right_label": "",
                    "left_value_mean": float("nan"),
                    "right_value_mean": float("nan"),
                    "delta_oriented_mean": float(r[mean_col]) if pd.notna(r[mean_col]) else float("nan"),
                    "left_value_median": float("nan"),
                    "right_value_median": float("nan"),
                    "delta_oriented_median": float(r[med_col]) if med_col in df.columns and pd.notna(r[med_col]) else float("nan"),
                    "delta_oriented_pos_frac": float(r[pos_col]) if pos_col in df.columns and pd.notna(r[pos_col]) else float("nan"),
                    "source": str(path),
                }
            )


def _append_backbone_coverage_rows(rows: List[Dict[str, object]], organized_dir: Path) -> None:
    path = organized_dir / "organized_parameter_matched_rows.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty or "backbone_kind" not in df.columns:
        return
    grouped = (
        df.groupby("backbone_kind", dropna=False)
        .agg(n_rows=("backbone_kind", "size"), n_unique_backbones=("backbone_id", "nunique"))
        .reset_index()
        .sort_values(["n_unique_backbones", "n_rows"], ascending=False)
    )
    for _, r in grouped.iterrows():
        rows.append(
            {
                "block": "D. Parameter-Matched Coverage",
                "comparison": f"{r['backbone_kind']} coverage",
                "metric": "count",
                "n_left": int(r["n_unique_backbones"]),
                "n_right": int(r["n_rows"]),
                "left_label": "unique_backbones",
                "right_label": "rows",
                "left_value_mean": float(r["n_unique_backbones"]),
                "right_value_mean": float(r["n_rows"]),
                "delta_oriented_mean": float("nan"),
                "left_value_median": float("nan"),
                "right_value_median": float("nan"),
                "delta_oriented_median": float("nan"),
                "delta_oriented_pos_frac": float("nan"),
                "source": str(path),
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact camera-ready results CSV")
    parser.add_argument("--predictor-group-dir", required=True, help="Path to predictor_group_tables directory")
    parser.add_argument("--organized-grid-dir", required=True, help="Path to organized_param_grid directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metrics to include",
    )
    args = parser.parse_args()

    predictor_dir = Path(args.predictor_group_dir)
    organized_dir = Path(args.organized_grid_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = _split_csv_arg(args.metrics)

    rows: List[Dict[str, object]] = []
    _append_directionality_rows(rows, predictor_dir, metrics)
    _append_pure_modality_rows(rows, predictor_dir, metrics)
    _append_incremental_rows(rows, predictor_dir, metrics)
    _append_backbone_coverage_rows(rows, organized_dir)

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("No rows produced; check input directories and metric names.")
    full_csv = out_dir / "camera_ready_compact_results.csv"
    out.to_csv(full_csv, index=False)

    # Main camera-ready subset: keep concise rows for the paper table.
    ranking_metrics = {"loto_rank_pairwise_cindex", "loto_rank_spearman"}
    keep_a = out["block"] == "A. Directional vs Symmetric"
    keep_b = (out["block"] == "B. Pure Modality Matched-k") & (out["comparison"] == "flow vs appearance") & (
        out["metric"].isin(ranking_metrics)
    )
    keep_c = (out["block"] == "C. Incremental Addition") & (
        out["comparison"].isin({"+appearance one-step", "+hof one-step", "+flow one-step"})
    ) & (out["metric"].isin(ranking_metrics))
    keep_d = out["block"] == "D. Parameter-Matched Coverage"
    main = out[keep_a | keep_b | keep_c | keep_d].copy()
    metric_map = {
        "loto_rank_pairwise_cindex": "C-index",
        "loto_rank_spearman": "Rank Spearman",
        "loto_mae": "MAE",
        "loto_rmse": "RMSE",
        "count": "Count",
    }
    cmp_map = {
        "flow vs appearance": "Flow vs Appearance",
        "flow_backbone coverage": "Flow backbones (unique, rows)",
        "appearance_backbone coverage": "Appearance backbones (unique, rows)",
        "mmd_backbone coverage": "MMD backbones (unique, rows)",
        "+appearance one-step": "+Appearance (one-step)",
        "+flow one-step": "+Flow (one-step)",
        "+hof one-step": "+HOF (one-step)",
    }
    main["metric"] = main["metric"].map(metric_map).fillna(main["metric"])
    main["comparison"] = main["comparison"].map(cmp_map).fillna(main["comparison"])
    main = main[
        [
            "block",
            "comparison",
            "metric",
            "n_left",
            "n_right",
            "left_value_mean",
            "right_value_mean",
            "delta_oriented_mean",
            "delta_oriented_pos_frac",
        ]
    ]
    main_csv = out_dir / "camera_ready_main_table.csv"
    main.to_csv(main_csv, index=False)

    print(f"Wrote {full_csv} ({len(out)} rows)")
    print(f"Wrote {main_csv} ({len(main)} rows)")


if __name__ == "__main__":
    main()
