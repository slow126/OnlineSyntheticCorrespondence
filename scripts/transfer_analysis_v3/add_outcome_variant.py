#!/usr/bin/env python3
"""Add a model-variant outcome block to an existing feature-complete table.

Dataset-pair features do not depend on model architecture. This utility joins
new outcomes to one validated feature row per (train_dataset, benchmark), which
avoids rebuilding feature files when only a new architecture has been trained.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_table import load_auc


OUTCOME_COLUMNS = {
    "model_family",
    "pretrained",
    "freeze",
    "context_id",
    "auc_normalized",
    "peak_pck",
}
KEY_COLUMNS = ["train_dataset", "benchmark"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-table", required=True)
    parser.add_argument("--auc-csv", required=True)
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(".").resolve()
    base = pd.read_csv(args.base_table)
    outcomes = load_auc(root, Path(args.auc_csv).resolve())
    outcomes = outcomes[outcomes["model_family"].eq(args.model_family)].copy()
    if outcomes.empty:
        raise SystemExit(f"No {args.model_family!r} rows found in {args.auc_csv}")

    feature_columns = [
        c for c in base.columns
        if c not in OUTCOME_COLUMNS and c not in KEY_COLUMNS
    ]
    grouped = base.groupby(KEY_COLUMNS, dropna=False)
    varying = []
    for col in feature_columns:
        if (grouped[col].nunique(dropna=False) > 1).any():
            varying.append(col)
    if varying:
        raise SystemExit(
            "Base table has variant-dependent values in feature columns: "
            + ", ".join(varying)
        )

    features = (
        base[KEY_COLUMNS + feature_columns]
        .drop_duplicates(subset=KEY_COLUMNS)
    )
    added = outcomes.merge(
        features,
        on=KEY_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    missing_features = added[feature_columns].isna().all(axis=1)
    if missing_features.any():
        missing = added.loc[missing_features, KEY_COLUMNS].drop_duplicates()
        raise SystemExit(
            "No feature row for new outcome pairs:\n"
            + missing.to_string(index=False)
        )

    added = added.reindex(columns=base.columns)
    combined = pd.concat([base, added], ignore_index=True)
    duplicate = combined.duplicated(subset=["train_dataset", "context_id"])
    if duplicate.any():
        raise SystemExit(
            f"{int(duplicate.sum())} duplicate (train_dataset, context_id) rows"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    print(
        f"Added {len(added)} {args.model_family} rows; "
        f"wrote {len(combined)} rows to {out}"
    )


if __name__ == "__main__":
    main()
