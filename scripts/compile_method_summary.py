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


def _overall_row(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    for col in ("benchmark", "train_dataset", "train_dataset_group"):
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
    lobo_rank = _read_csv(base / "prediction_lobo_rank_summary.csv")
    loto_rank = _read_csv(base / "prediction_loto_rank_summary.csv")
    uni_slopes = _read_csv(base / "within_benchmark_slopes_univariate.csv")

    lobo_row = _overall_row(lobo_summary)
    loto_row = _overall_row(loto_summary)
    lobo_rank_row = _overall_row(lobo_rank)
    loto_rank_row = _overall_row(loto_rank)

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
        "lobo_regret": _pick_metric(lobo_rank_row, "regret"),
        "lobo_rank_spearman": _pick_metric(lobo_rank_row, "spearman"),
        "loto_regret": _pick_metric(loto_rank_row, "regret"),
        "loto_rank_spearman": _pick_metric(loto_rank_row, "spearman"),
        "uni_sign_stable_frac": uni_frac,
        "uni_sign_stable_count": uni_stable,
        "uni_sign_total": uni_total,
        "uni_sign_stable_frac_flow": uni_flow_frac,
        "uni_sign_stable_frac_semantic": uni_sem_frac,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile summary table across analysis methods.")
    parser.add_argument("--manifest", required=True, help="YAML manifest with methods.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
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

    if args.output_md:
        md_path = Path(args.output_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(df.to_markdown(index=False) + "\n")
        print(f"Wrote summary Markdown to {md_path}")


if __name__ == "__main__":
    main()
