#!/usr/bin/env python3
"""
Merge per-rank CSVs from a parallel pairwise distance run, optionally seed from
a prior local run, drop duplicates, and symmetrize train-train / eval-eval pairs.

Usage:
    python scripts/transfer_analysis_v3/merge_pairwise_distances.py \
        --inputs /scratch/pairwise/rank_*.csv \
        --seed-csv analysis_v3/pairwise_self_distances.csv \
        --output analysis_v3/pairwise_self_distances.csv
"""
import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


EPS_PX = [1.0, 4.0, 16.0]
K_KL_VALUES = [5, 20]


def _eps_label(eps_px: float) -> str:
    if float(eps_px).is_integer():
        return f"eps{int(eps_px)}px"
    return f"eps{eps_px:g}px".replace(".", "p")


def _symmetrize(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with reversed rows added for non-self train-train and eval-eval pairs."""
    sym_mask = (df["pair_type"] != "train_eval") & (df["dataset_a"] != df["dataset_b"])
    non_self = df[sym_mask].copy()
    if non_self.empty:
        return df

    rev = non_self.copy()
    rev["dataset_a"], rev["dataset_b"] = non_self["dataset_b"].values, non_self["dataset_a"].values
    rev["split_a"],   rev["split_b"]   = non_self["split_b"].values,   non_self["split_a"].values
    rev["n_vecs_a"],  rev["n_vecs_b"]  = non_self["n_vecs_b"].values,  non_self["n_vecs_a"].values
    rev["mean_nn_a_to_b"] = non_self["mean_nn_b_to_a"].values
    rev["mean_nn_b_to_a"] = non_self["mean_nn_a_to_b"].values

    for eps_px in EPS_PX:
        lbl = _eps_label(eps_px)
        rev[f"a_covered_by_b_{lbl}"] = non_self[f"b_covered_by_a_{lbl}"].values
        rev[f"b_covered_by_a_{lbl}"] = non_self[f"a_covered_by_b_{lbl}"].values

    for k in K_KL_VALUES:
        col_ab = f"kl_a_to_b_k{k}"
        col_ba = f"kl_b_to_a_k{k}"
        if col_ab in non_self.columns:
            rev[col_ab] = non_self[col_ba].values
            rev[col_ba] = non_self[col_ab].values

    existing_keys = set(zip(df["space"], df["pair_type"], df["dataset_a"], df["dataset_b"]))
    rev = rev[~rev.apply(
        lambda r: (r["space"], r["pair_type"], r["dataset_a"], r["dataset_b"]) in existing_keys,
        axis=1,
    )]
    if rev.empty:
        return df
    combined = pd.concat([df, rev], ignore_index=True)
    print(f"Symmetrized: added {len(rev)} reversed rows → {len(combined)} total rows")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Rank output CSVs (globs OK, e.g. /scratch/rank_*.csv)")
    parser.add_argument("--seed-csv", default=None,
                        help="Optional prior-run CSV to include (e.g. the local run result)")
    parser.add_argument("--output", required=True,
                        help="Destination CSV (can be same as --seed-csv to update in place)")
    args = parser.parse_args()

    # Expand globs
    input_paths: list[Path] = []
    for pattern in args.inputs:
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"Warning: no files matched '{pattern}'")
        input_paths.extend(Path(p) for p in matches)

    if not input_paths:
        sys.exit("No input files found.")

    frames: list[pd.DataFrame] = []

    if args.seed_csv:
        seed = Path(args.seed_csv)
        if seed.exists() and seed.stat().st_size > 0:
            df_seed = pd.read_csv(seed)
            frames.append(df_seed)
            print(f"Seed: {len(df_seed)} rows from {seed.name}")

    for p in input_paths:
        if not p.exists() or p.stat().st_size == 0:
            print(f"Skipping empty/missing: {p}")
            continue
        df_rank = pd.read_csv(p)
        frames.append(df_rank)
        print(f"  {p.name}: {len(df_rank)} rows")

    if not frames:
        sys.exit("No data loaded.")

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["space", "pair_type", "dataset_a", "dataset_b"], keep="first"
    )
    if len(combined) < before:
        print(f"Dropped {before - len(combined)} duplicate rows")
    print(f"Combined: {len(combined)} unique rows")

    combined = _symmetrize(combined)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    print(f"Written: {out}  ({len(combined)} rows)")


if __name__ == "__main__":
    main()
