#!/usr/bin/env python3
"""
Quick sanity dump of HOF motion distance columns to a formatted .txt file.
"""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump HOF motion distance columns to text.")
    parser.add_argument(
        "--input",
        default="analysis/coverage_v2_hof_full_occ.csv",
        help="Input HOF coverage CSV.",
    )
    parser.add_argument(
        "--output",
        default="analysis/hof_motion_k1_preview.txt",
        help="Output text file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max rows to write (default: 50).",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Missing input CSV: {path}")

    df = pd.read_csv(path)
    cols = [
        "train_dataset",
        "train_split",
        "eval_dataset",
        "eval_split",
        "mean_nn_eval_to_train_k1",
        "mean_nn_train_to_eval_k1",
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in CSV: {missing}")

    out = df[cols].head(args.limit).copy()
    out["mean_nn_eval_to_train_k1"] = out["mean_nn_eval_to_train_k1"].map(lambda v: f"{v:.6f}")
    out["mean_nn_train_to_eval_k1"] = out["mean_nn_train_to_eval_k1"].map(lambda v: f"{v:.6f}")

    lines = []
    header = (
        f"{'train':<28} {'split':<6} {'eval':<28} {'split':<6} "
        f"{'eval->train_k1':>14} {'train->eval_k1':>14}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in out.iterrows():
        lines.append(
            f"{row['train_dataset']:<28} {row['train_split']:<6} "
            f"{row['eval_dataset']:<28} {row['eval_split']:<6} "
            f"{row['mean_nn_eval_to_train_k1']:>14} {row['mean_nn_train_to_eval_k1']:>14}"
        )

    Path(args.output).write_text("\n".join(lines))
    print(f"Wrote {args.output} ({len(out)} rows)")


if __name__ == "__main__":
    main()
