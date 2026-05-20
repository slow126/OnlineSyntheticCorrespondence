#!/usr/bin/env python3
"""Audit missing flow feature coverage for the active transfer table.

The transfer table can move faster than the cached feature CSVs.  This helper
checks the active (train_dataset, benchmark) pairs against the flow feature
sources and prints exactly which pairs are missing before a sweep.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _directed_pairs(path: Path, train_col: str, eval_col: str) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if train_col not in df.columns or eval_col not in df.columns:
        return set()
    return set(zip(df[train_col].astype(str), df[eval_col].astype(str)))


def _undirected_pairs(path: Path, a_col: str, b_col: str) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if a_col not in df.columns or b_col not in df.columns:
        return set()
    out: set[tuple[str, str]] = set()
    for a, b in zip(df[a_col].astype(str), df[b_col].astype(str)):
        out.add((a, b))
        out.add((b, a))
    return out


def _missing_frame(
    expected: set[tuple[str, str]],
    present: set[tuple[str, str]],
    feature: str,
) -> pd.DataFrame:
    rows = [
        {"feature_family": feature, "train_dataset": td, "benchmark": bm}
        for td, bm in sorted(expected - present)
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default="scripts/transfer_analysis_v3/transfer_table.csv")
    parser.add_argument("--flow-raw", default="analysis/coverage_v2_flow_only_raw_joint_full.csv")
    parser.add_argument("--flow-kmeans", default="analysis/coverage_v2_flow_only_raw_joint_kmeans_full.csv")
    parser.add_argument("--symmetric", default="analysis_v3/symmetric_distances.csv")
    parser.add_argument("--flow-mmd", default="flow_mmd_results_fast.csv")
    parser.add_argument("--pairwise-self", default="analysis_v3/pairwise_self_distances.csv")
    parser.add_argument("--output", default="analysis_v3/missing_flow_feature_pairs.csv")
    parser.add_argument("--require-clean-families", nargs="*", default=[],
        help="Feature families that must have zero missing pairs. "
             "Choices are the printed feature_family names.")
    args = parser.parse_args()

    table = pd.read_csv(args.table)
    expected = set(zip(table["train_dataset"].astype(str), table["benchmark"].astype(str)))
    print(f"Active transfer table: {len(table)} rows, {len(expected)} train/benchmark pairs")

    audits: list[pd.DataFrame] = []
    sources = [
        (
            "flow_raw_coverage",
            _directed_pairs(Path(args.flow_raw), "train_dataset", "eval_dataset"),
            Path(args.flow_raw),
        ),
        (
            "flow_kmeans_coverage",
            _directed_pairs(Path(args.flow_kmeans), "train_dataset", "eval_dataset"),
            Path(args.flow_kmeans),
        ),
        (
            "flow_fid_sw2",
            _directed_pairs(Path(args.symmetric), "train_dataset", "eval_dataset"),
            Path(args.symmetric),
        ),
        (
            "flow_mmd",
            _undirected_pairs(Path(args.flow_mmd), "dataset1", "dataset2"),
            Path(args.flow_mmd),
        ),
    ]

    pairwise_path = Path(args.pairwise_self)
    if pairwise_path.exists():
        sd = pd.read_csv(pairwise_path)
        flow_cross = sd[(sd.get("space") == "flow") & (sd.get("pair_type") == "train_eval")]
        pairwise_present = set(zip(flow_cross["dataset_a"].astype(str), flow_cross["dataset_b"].astype(str)))
    else:
        pairwise_present = set()
    sources.append(("pairwise_self_flow_train_eval", pairwise_present, pairwise_path))

    for feature, present, path in sources:
        missing = _missing_frame(expected, present, feature)
        audits.append(missing)
        status = "missing file" if not path.exists() else f"{len(present)} source pairs"
        print(f"{feature:32s}: {len(missing):3d}/{len(expected)} missing ({status})")
        if not missing.empty:
            print(missing[["train_dataset", "benchmark"]].to_string(index=False))
            print()

    out = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved audit rows to {out_path}")

    required = set(args.require_clean_families)
    if required:
        bad = out[out["feature_family"].isin(required)] if not out.empty else out
        if not bad.empty:
            counts = bad.groupby("feature_family").size().to_dict()
            print(f"ERROR: required feature families still have missing pairs: {counts}")
            sys.exit(1)


if __name__ == "__main__":
    main()
