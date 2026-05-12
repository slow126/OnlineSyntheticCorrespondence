#!/usr/bin/env python3
"""
Compile a summary table across analysis methods.

Each method corresponds to an output directory from build_leakage_free_eval.py.
This script aggregates:
  - LOBO/LOTO prediction metrics (MAE, RMSE, Spearman)
  - LOBO/LOTO ranking metrics (Regret, Spearman)
  - Univariate sign stability (within_benchmark_slopes_univariate.csv)

Inputs are specified by a YAML manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml


FLOW_FAMILY = {"kitti2012", "kitti2015", "middlebury", "flyingthings", "pointodyssey"}
SEMANTIC_FAMILY = {"spair", "pfpascal", "pfwillow", "tss"}


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _overall_row(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    for col in ("benchmark", "train_dataset", "train_dataset_group", "fold"):
        if col in df.columns:
            overall = df[df[col] == "__overall__"]
            if not overall.empty:
                return overall.iloc[0]
    return None


def _pick_metric(row: Optional[pd.Series], key: str) -> float:
    if row is None:
        return float("nan")
    value = row.get(key, float("nan"))
    try:
        return float(value)
    except Exception:
        return float("nan")


def _pick_metric_alias(row: Optional[pd.Series], keys: Iterable[str]) -> float:
    if row is None:
        return float("nan")
    for key in keys:
        value = _pick_metric(row, key)
        if pd.notna(value):
            return value
    return float("nan")


def _sign_stability(
    df: Optional[pd.DataFrame],
    benchmarks: Optional[Iterable[str]] = None,
    threshold: float = 0.8,
) -> Tuple[float, int, int]:
    if df is None or df.empty:
        return float("nan"), 0, 0
    if benchmarks is not None and "benchmark" in df.columns:
        df = df[df["benchmark"].isin(benchmarks)]
    if df.empty:
        return float("nan"), 0, 0
    skip_cols = {"benchmark", "n", "r2", "mode"}
    predictors = [c for c in df.columns if c not in skip_cols]
    if not predictors:
        return float("nan"), 0, 0
    stable = 0
    total = 0
    for pred in predictors:
        signs = pd.Series(df[pred]).dropna()
        if signs.empty:
            continue
        signs = signs.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        if signs.empty:
            continue
        pos_frac = (signs > 0).mean()
        is_stable = (pos_frac >= threshold) or (pos_frac <= (1.0 - threshold))
        stable += 1 if is_stable else 0
        total += 1
    frac = float(stable) / float(total) if total else float("nan")
    return frac, stable, total


def _load_method_metrics(method: Dict, threshold: float) -> Dict[str, object]:
    base = Path(method["path"])

    lobo_summary = _read_csv(base / "prediction_lobo_summary.csv")
    loto_summary = _read_csv(base / "prediction_loto_summary.csv")
    jointood_summary = _read_csv(base / "prediction_jointood_summary.csv")
    lobo_rank = _read_csv(base / "prediction_lobo_rank_summary.csv")
    loto_rank = _read_csv(base / "prediction_loto_rank_summary.csv")
    if loto_rank is None or loto_rank.empty:
        # Newer LOTO outputs use holdout-placement summaries instead of rank_summary.
        loto_rank = _read_csv(base / "prediction_loto_holdout_placement_summary.csv")
    jointood_rank = _read_csv(base / "prediction_jointood_rank_summary.csv")
    uni_slopes = _read_csv(base / "within_benchmark_slopes_univariate.csv")
    run_meta = _read_json(base / "run_metadata.json") or {}

    lobo_row = _overall_row(lobo_summary)
    loto_row = _overall_row(loto_summary)
    jointood_row = _overall_row(jointood_summary)
    lobo_rank_row = _overall_row(lobo_rank)
    loto_rank_row = _overall_row(loto_rank)
    jointood_rank_row = _overall_row(jointood_rank)

    uni_frac, uni_stable, uni_total = _sign_stability(uni_slopes, None, threshold)
    uni_flow_frac, _, _ = _sign_stability(uni_slopes, FLOW_FAMILY, threshold)
    uni_sem_frac, _, _ = _sign_stability(uni_slopes, SEMANTIC_FAMILY, threshold)

    return {
        "lobo_mae": _pick_metric(lobo_row, "mae"),
        "lobo_rmse": _pick_metric(lobo_row, "rmse"),
        "lobo_spearman": _pick_metric(lobo_row, "spearman"),
        "loto_mae": _pick_metric(loto_row, "mae"),
        "loto_rmse": _pick_metric(loto_row, "rmse"),
        "loto_spearman": _pick_metric(loto_row, "spearman"),
        "jointood_mae": _pick_metric(jointood_row, "mae"),
        "jointood_rmse": _pick_metric(jointood_row, "rmse"),
        "jointood_spearman": _pick_metric(jointood_row, "spearman"),
        "lobo_regret": _pick_metric(lobo_rank_row, "regret"),
        "lobo_rank_spearman": _pick_metric(lobo_rank_row, "spearman"),
        "lobo_rank_kendall_tau": _pick_metric_alias(lobo_rank_row, ("kendall_tau_b", "kendall_tau")),
        "lobo_rank_pairwise_cindex": _pick_metric_alias(
            lobo_rank_row, ("pairwise_cindex", "cindex", "pairwise_accuracy", "pair_acc")
        ),
        "loto_regret": _pick_metric_alias(loto_rank_row, ("regret", "regret_micro")),
        "loto_rank_spearman": _pick_metric_alias(
            loto_rank_row, ("spearman", "rank_spearman_micro", "rank_spearman_fisher", "rank_spearman")
        ),
        "loto_rank_kendall_tau": _pick_metric_alias(
            loto_rank_row, ("kendall_tau_b", "kendall_tau", "rank_kendall_micro", "rank_kendall")
        ),
        "loto_rank_pairwise_cindex": _pick_metric_alias(
            loto_rank_row,
            (
                "pairwise_cindex",
                "cindex",
                "pairwise_accuracy",
                "pair_acc",
                "pairwise_win_rate_micro",
                "pairwise_win_rate",
            ),
        ),
        "jointood_regret": _pick_metric(jointood_rank_row, "regret"),
        "jointood_rank_spearman": _pick_metric(jointood_rank_row, "spearman"),
        "jointood_rank_kendall_tau": _pick_metric_alias(
            jointood_rank_row, ("kendall_tau_b", "kendall_tau")
        ),
        "jointood_rank_pairwise_cindex": _pick_metric_alias(
            jointood_rank_row, ("pairwise_cindex", "cindex", "pairwise_accuracy", "pair_acc")
        ),
        "jointood_top1": _pick_metric(jointood_rank_row, "top1"),
        "jointood_top3": _pick_metric(jointood_rank_row, "top3"),
        "jointood_topk": _pick_metric(jointood_rank_row, "topk"),
        "jointood_rank_abs_err": _pick_metric(jointood_rank_row, "mean_abs_rank_error"),
        "jointood_rank_pct_err": _pick_metric(jointood_rank_row, "mean_abs_rank_pct_error"),
        "uni_sign_stable_frac": uni_frac,
        "uni_sign_stable_count": uni_stable,
        "uni_sign_total": uni_total,
        "uni_sign_stable_frac_flow": uni_flow_frac,
        "uni_sign_stable_frac_semantic": uni_sem_frac,
        "target": run_meta.get("target"),
        "prediction_target": run_meta.get("prediction_target"),
        "predictors": ",".join(run_meta.get("predictors", [])) if run_meta.get("predictors") else None,
        "n_predictors": run_meta.get("n_predictors"),
        "n_predictors_base": run_meta.get("n_predictors_base"),
        "n_predictors_encoder_main_effects": run_meta.get("n_predictors_encoder_main_effects"),
        "n_predictors_model_family_main_effects": run_meta.get("n_predictors_model_family_main_effects"),
        "n_predictors_encoder_interactions": run_meta.get("n_predictors_encoder_interactions"),
        "n_predictors_model_family_interactions": run_meta.get("n_predictors_model_family_interactions"),
        "model": run_meta.get("model"),
        "rank_target": run_meta.get("rank_target"),
    }


def _load_manifest(path: Path) -> List[Dict]:
    data = yaml.safe_load(path.read_text())
    methods = data.get("methods", []) if isinstance(data, dict) else []
    cleaned = []
    for item in methods:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("label")
        path_val = item.get("path")
        if not name or not path_val:
            continue
        cleaned.append(
            {
                "name": str(name),
                "path": str(path_val),
                "family": item.get("family", ""),
                "symmetry": item.get("symmetry", ""),
                "notes": item.get("notes", ""),
                "order": item.get("order"),
            }
        )
    return cleaned


def _split_summary_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    abs_cols = [
        "method",
        "path",
        "family",
        "symmetry",
        "notes",
        "target",
        "prediction_target",
        "predictors",
        "n_predictors",
        "n_predictors_base",
        "n_predictors_encoder_main_effects",
        "n_predictors_model_family_main_effects",
        "n_predictors_encoder_interactions",
        "n_predictors_model_family_interactions",
        "model",
        "lobo_mae",
        "lobo_rmse",
        "lobo_spearman",
        "loto_mae",
        "loto_rmse",
        "loto_spearman",
        "jointood_mae",
        "jointood_rmse",
        "jointood_spearman",
        "uni_sign_stable_frac",
        "uni_sign_stable_count",
        "uni_sign_total",
        "uni_sign_stable_frac_flow",
        "uni_sign_stable_frac_semantic",
    ]
    rank_cols = [
        "method",
        "path",
        "family",
        "symmetry",
        "notes",
        "target",
        "prediction_target",
        "predictors",
        "n_predictors",
        "n_predictors_base",
        "n_predictors_encoder_main_effects",
        "n_predictors_model_family_main_effects",
        "n_predictors_encoder_interactions",
        "n_predictors_model_family_interactions",
        "model",
        "lobo_regret",
        "lobo_rank_spearman",
        "lobo_rank_kendall_tau",
        "lobo_rank_pairwise_cindex",
        "loto_regret",
        "loto_rank_spearman",
        "loto_rank_kendall_tau",
        "loto_rank_pairwise_cindex",
        "jointood_regret",
        "jointood_rank_spearman",
        "jointood_rank_kendall_tau",
        "jointood_rank_pairwise_cindex",
        "jointood_top1",
        "jointood_top3",
        "jointood_topk",
        "jointood_rank_abs_err",
        "jointood_rank_pct_err",
    ]
    abs_cols = [c for c in abs_cols if c in df.columns]
    rank_cols = [c for c in rank_cols if c in df.columns]
    return df[abs_cols].copy(), df[rank_cols].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile summary table across analysis methods.")
    parser.add_argument("--manifest", required=True, help="YAML manifest with methods.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--output-abs", default=None, help="Output absolute-metrics CSV path.")
    parser.add_argument("--output-rank", default=None, help="Output ranking-metrics CSV path.")
    parser.add_argument("--output-md", default=None, help="Optional Markdown table output.")
    parser.add_argument("--stability-threshold", type=float, default=0.8)
    parser.add_argument("--sort", default="family,symmetry,-loto_spearman")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    methods = _load_manifest(manifest_path)
    if not methods:
        raise SystemExit(f"No methods found in manifest: {manifest_path}")

    rows = []
    for method in methods:
        row = {
            "method": method["name"],
            "path": method["path"],
            "family": method["family"],
            "symmetry": method["symmetry"],
            "notes": method["notes"],
        }
        row.update(_load_method_metrics(method, args.stability_threshold))
        row["order"] = method.get("order")
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        sort_keys = [s.strip() for s in args.sort.split(",") if s.strip()]
        by = []
        ascending = []
        for key in sort_keys:
            if key.startswith("-"):
                by.append(key[1:])
                ascending.append(False)
            else:
                by.append(key)
                ascending.append(True)
        missing = [k for k in by if k not in df.columns]
        if missing:
            by = [k for k in by if k in df.columns]
            ascending = ascending[: len(by)]
        if by:
            df = df.sort_values(by=by, ascending=ascending)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote summary CSV to {output_path}")

    abs_path = Path(args.output_abs) if args.output_abs else None
    rank_path = Path(args.output_rank) if args.output_rank else None
    if abs_path is None or rank_path is None:
        stem = output_path.with_suffix("")
        if abs_path is None:
            abs_path = Path(f"{stem}_abs.csv")
        if rank_path is None:
            rank_path = Path(f"{stem}_rank.csv")

    abs_df, rank_df = _split_summary_tables(df)
    abs_df.to_csv(abs_path, index=False)
    rank_df.to_csv(rank_path, index=False)
    print(f"Wrote absolute-target summary CSV to {abs_path}")
    print(f"Wrote ranking-target summary CSV to {rank_path}")

    if args.output_md:
        md_path = Path(args.output_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(df.to_markdown(index=False) + "\n")
        print(f"Wrote summary Markdown to {md_path}")


if __name__ == "__main__":
    main()
