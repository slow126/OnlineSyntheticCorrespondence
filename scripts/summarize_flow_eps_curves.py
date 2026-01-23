#!/usr/bin/env python3
"""
Summarize flow epsilon curves into scalar metrics (AUC, eps@coverage).
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _parse_thresholds(raw: str) -> List[float]:
    values: List[float] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def _eps_at_coverage(eps: np.ndarray, cov: np.ndarray, target: float) -> float:
    if eps.size == 0:
        return float("nan")
    order = np.argsort(eps)
    eps = eps[order]
    cov = cov[order]
    cov = np.maximum.accumulate(cov)
    if cov[-1] < target:
        return float("nan")
    return float(np.interp(target, cov, eps))


def _auc(eps: np.ndarray, cov: np.ndarray) -> float:
    if eps.size == 0:
        return float("nan")
    order = np.argsort(eps)
    eps = eps[order]
    cov = cov[order]
    return float(np.trapz(cov, eps))


def _summarize_direction(
    sub: pd.DataFrame,
    eps_col: str,
    cov_col: str,
    thresholds: List[float],
) -> Dict[str, float]:
    eps = sub[eps_col].to_numpy(dtype=float)
    cov = sub[cov_col].to_numpy(dtype=float)
    result = {
        "auc": _auc(eps, cov),
    }
    for t in thresholds:
        key = f"eps_at{int(round(t * 100))}"
        result[key] = _eps_at_coverage(eps, cov, t)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize flow epsilon curves.")
    parser.add_argument("--input-csv", required=True, help="Path to epsilon curves CSV.")
    parser.add_argument("--output-csv", required=True, help="Path to output summary CSV.")
    parser.add_argument(
        "--coverage-thresholds",
        default="0.9,0.95",
        help="Comma-separated coverage thresholds for eps@coverage.",
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Expect weighted curve columns (k-means).",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    thresholds = _parse_thresholds(args.coverage_thresholds)

    df = pd.read_csv(input_path)
    if df.empty:
        raise SystemExit(f"No rows in {input_path}")

    rows: List[Dict[str, object]] = []

    if args.weighted:
        required = {"eval_covered_by_train_weighted", "train_covered_by_eval_weighted"}
        missing = required - set(df.columns)
        if missing:
            raise SystemExit(f"Missing weighted coverage columns: {sorted(missing)}")
        group_cols = ["train_dataset", "train_split", "eval_dataset", "eval_split"]
        for keys, sub in df.groupby(group_cols):
            row = dict(zip(group_cols, keys))
            eval_stats = _summarize_direction(
                sub, "epsilon_px", "eval_covered_by_train_weighted", thresholds
            )
            train_stats = _summarize_direction(
                sub, "epsilon_px", "train_covered_by_eval_weighted", thresholds
            )
            row["eval_to_train_auc_weighted"] = eval_stats["auc"]
            row["train_to_eval_auc_weighted"] = train_stats["auc"]
            for t in thresholds:
                suffix = int(round(t * 100))
                row[f"eval_to_train_eps_at{suffix}_weighted"] = eval_stats[f"eps_at{suffix}"]
                row[f"train_to_eval_eps_at{suffix}_weighted"] = train_stats[f"eps_at{suffix}"]
            rows.append(row)
    else:
        required = {"direction", "coverage"}
        missing = required - set(df.columns)
        if missing:
            raise SystemExit(f"Missing curve columns: {sorted(missing)}")
        group_cols = ["train_dataset", "train_split", "eval_dataset", "eval_split", "direction"]
        grouped = df.groupby(group_cols)
        summaries: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
        for keys, sub in grouped:
            train_dataset, train_split, eval_dataset, eval_split, direction = keys
            stats = _summarize_direction(sub, "epsilon_px", "coverage", thresholds)
            pair_key = (train_dataset, train_split, eval_dataset, eval_split)
            entry = summaries.setdefault(pair_key, {})
            if direction == "eval_to_train":
                entry["eval_to_train_auc"] = stats["auc"]
                for t in thresholds:
                    suffix = int(round(t * 100))
                    entry[f"eval_to_train_eps_at{suffix}"] = stats[f"eps_at{suffix}"]
            elif direction == "train_to_eval":
                entry["train_to_eval_auc"] = stats["auc"]
                for t in thresholds:
                    suffix = int(round(t * 100))
                    entry[f"train_to_eval_eps_at{suffix}"] = stats[f"eps_at{suffix}"]
        for (train_dataset, train_split, eval_dataset, eval_split), stats in summaries.items():
            row = {
                "train_dataset": train_dataset,
                "train_split": train_split,
                "eval_dataset": eval_dataset,
                "eval_split": eval_split,
            }
            row.update(stats)
            rows.append(row)

    out_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"✓ Wrote {len(out_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
