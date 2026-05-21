#!/usr/bin/env python3
"""Materialize the legacy raw flow coverage CSV from pairwise_self_distances.

The v3 modeling table only uses raw flow mean-NN and epsilon coverage columns
at 1/4/16px.  Those are already computed for all train-eval pairs by
pairwise_self_distances.csv, along with the KL columns used by flow_kl.
This helper writes a legacy-shaped coverage CSV so downstream audit and
FID/SW2 pair selection do not depend on rerunning the large FAISS coverage job.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairwise-self", default="analysis_v3/pairwise_self_distances.csv")
    parser.add_argument("--output", default="analysis/coverage_v2_flow_only_raw_joint_full.csv")
    args = parser.parse_args()

    src = Path(args.pairwise_self)
    if not src.exists():
        raise FileNotFoundError(src)

    df = pd.read_csv(src)
    cross = df[(df["space"] == "flow") & (df["pair_type"] == "train_eval")].copy()
    if cross.empty:
        raise ValueError(f"No flow train_eval rows found in {src}")

    out = pd.DataFrame({
        "space": "joint",
        "train_dataset": cross["dataset_a"],
        "train_split": cross["split_a"],
        "eval_dataset": cross["dataset_b"],
        "eval_split": cross["split_b"],
        "train_n_vectors": cross["n_vecs_a"].astype("int64"),
        "eval_n_vectors": cross["n_vecs_b"].astype("int64"),
        "distance_metric": "sqL2",
        "flow_normalized": True,
        "mean_nn_eval_to_train_k1": cross["mean_nn_b_to_a"],
        "mean_nn_train_to_eval_k1": cross["mean_nn_a_to_b"],
        "eval_covered_by_train_eps1px": cross["b_covered_by_a_eps1px"],
        "train_covered_by_eval_eps1px": cross["a_covered_by_b_eps1px"],
        "eval_covered_by_train_eps4px": cross["b_covered_by_a_eps4px"],
        "train_covered_by_eval_eps4px": cross["a_covered_by_b_eps4px"],
        "eval_covered_by_train_eps16px": cross["b_covered_by_a_eps16px"],
        "train_covered_by_eval_eps16px": cross["a_covered_by_b_eps16px"],
    })

    out = out.drop_duplicates(
        subset=["space", "train_dataset", "train_split", "eval_dataset", "eval_split"],
        keep="last",
    ).sort_values(["train_dataset", "eval_dataset"]).reset_index(drop=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} raw flow coverage rows to {out_path}")


if __name__ == "__main__":
    main()
