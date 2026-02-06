#!/usr/bin/env python3
"""
Convert MMD v2 CSV outputs (train/eval columns) to v1 format
expected by build_leakage_free_eval.py (dataset1/dataset2).
"""

import argparse
from pathlib import Path

import pandas as pd


def _coerce_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def convert_v2_to_v1(df: pd.DataFrame) -> pd.DataFrame:
    required = ["train_dataset", "train_split", "eval_dataset", "eval_split", "mmd2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing v2 columns: {missing}")
    out = pd.DataFrame(
        {
            "dataset1": df["train_dataset"].astype(str),
            "split1": df["train_split"].astype(str),
            "dataset2": df["eval_dataset"].astype(str),
            "split2": df["eval_split"].astype(str),
            "mmd2": _coerce_float(df["mmd2"]),
        }
    )
    if "mmd" in df.columns:
        out["mmd"] = _coerce_float(df["mmd"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MMD v2 CSV to v1 format.")
    parser.add_argument("--input", required=True, help="Path to v2 MMD CSV.")
    parser.add_argument("--output", required=True, help="Path to output v1 MMD CSV.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    df = pd.read_csv(input_path)
    if {"dataset1", "dataset2"}.issubset(df.columns):
        # Already v1 format; just normalize numeric columns and write.
        out = df.copy()
        if "mmd2" in out.columns:
            out["mmd2"] = _coerce_float(out["mmd2"])
        if "mmd" in out.columns:
            out["mmd"] = _coerce_float(out["mmd"])
    else:
        out = convert_v2_to_v1(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out)} rows to {output_path}")


if __name__ == "__main__":
    main()
