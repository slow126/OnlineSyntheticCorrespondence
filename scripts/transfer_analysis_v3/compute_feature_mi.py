#!/usr/bin/env python3
"""
Compute mutual information between each coverage feature and transfer AUC,
plus a pairwise feature redundancy matrix.

Two outputs:
  analysis_v3/feature_mi.csv          — MI(feature; AUC) with bootstrap 95% CIs
  analysis_v3/feature_redundancy.csv  — pairwise MI(feature_i; feature_j) matrix

MI is estimated with the KSG kNN estimator (sklearn mutual_info_regression).
Bootstrap resamples at the training-dataset level to respect the nested
structure of the data (11 training sets × 9 eval sets = 99 pairs).

Usage:
    python scripts/transfer_analysis_v3/compute_feature_mi.py
    python scripts/transfer_analysis_v3/compute_feature_mi.py \\
        --table scripts/transfer_analysis_v3/transfer_table.csv \\
        --n-boot 500 --k-neighbors 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


# Columns that are identifiers, not features
_ID_COLS = {
    "train_dataset", "benchmark", "model_family", "pretrained", "freeze",
    "context_id", "auc_normalized", "peak_pck",
}


def ksg_mi(X: np.ndarray, y: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """MI between each column of X and y using KSG estimator. Returns shape (n_features,)."""
    return mutual_info_regression(X, y, n_neighbors=k, random_state=seed)


def bootstrap_mi(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    k: int,
    n_boot: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Bootstrap MI(feature; AUC) with cluster resampling at the training-dataset level.
    Returns DataFrame with columns: feature, mi_point, ci_lo, ci_med, ci_hi.
    """
    train_sets = df["train_dataset"].unique()
    n_feat = len(feature_cols)

    point_df = df[feature_cols + [target]].dropna()
    point_scores = ksg_mi(point_df[feature_cols].values, point_df[target].values, k)

    boot_scores = np.zeros((n_boot, n_feat))
    for b in range(n_boot):
        sampled = rng.choice(train_sets, size=len(train_sets), replace=True)
        boot_rows = pd.concat(
            [df[df["train_dataset"] == t] for t in sampled], ignore_index=True
        )
        sub = boot_rows[feature_cols + [target]].dropna()
        if len(sub) < k + 1:
            boot_scores[b] = np.nan
            continue
        boot_scores[b] = ksg_mi(sub[feature_cols].values, sub[target].values, k)

    ci_lo  = np.nanpercentile(boot_scores, 2.5,  axis=0)
    ci_med = np.nanpercentile(boot_scores, 50.0, axis=0)
    ci_hi  = np.nanpercentile(boot_scores, 97.5, axis=0)

    return pd.DataFrame({
        "feature": feature_cols,
        "mi_point": point_scores,
        "ci_lo":    ci_lo,
        "ci_med":   ci_med,
        "ci_hi":    ci_hi,
    }).sort_values("mi_point", ascending=False).reset_index(drop=True)


def redundancy_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    k: int,
) -> pd.DataFrame:
    """
    Pairwise MI(feature_i; feature_j) matrix — point estimates only (no bootstrap).
    Returns a square DataFrame indexed and columned by feature name.
    """
    sub = df[feature_cols].dropna()
    X = sub.values  # (n_samples, n_features)
    n = len(feature_cols)
    mat = np.zeros((n, n))
    for i in range(n):
        scores = mutual_info_regression(X, X[:, i], n_neighbors=k, random_state=42)
        mat[i] = scores
    # Symmetrize (KSG is not exactly symmetric due to conditioning)
    mat = (mat + mat.T) / 2
    return pd.DataFrame(mat, index=feature_cols, columns=feature_cols)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table",
        default="scripts/transfer_analysis_v3/transfer_table.csv",
        help="Path to transfer_table.csv produced by build_table.py",
    )
    parser.add_argument("--target", default="auc_normalized")
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir", default="analysis_v3",
        help="Directory for output CSVs",
    )
    parser.add_argument(
        "--context", default=None,
        help="Restrict to a specific context_id (e.g. spair_test_catspp_False_False)",
    )
    parser.add_argument(
        "--exclude-train-datasets", nargs="+", default=[],
        help="Drop rows whose train_dataset is in this list before computing MI.",
    )
    args = parser.parse_args()

    table_path = Path(args.table)
    if not table_path.exists():
        print(f"ERROR: transfer table not found at {table_path}")
        print("  Run build_table.py first.")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(table_path)
    print(f"Loaded transfer table: {df.shape[0]} rows × {df.shape[1]} columns")

    if args.context:
        df = df[df["context_id"] == args.context]
        print(f"  Filtered to context '{args.context}': {len(df)} rows")

    if args.exclude_train_datasets:
        before = len(df)
        excluded = set(args.exclude_train_datasets)
        df = df[~df["train_dataset"].isin(excluded)]
        print(f"  Excluded train datasets {sorted(excluded)}: {before} -> {len(df)} rows")
        if df.empty:
            print("ERROR: no rows left after --exclude-train-datasets")
            sys.exit(1)

    # Feature columns: everything that isn't an identifier
    feature_cols = [c for c in df.columns if c not in _ID_COLS]

    # Drop features with too many NaNs (more than 30% missing)
    n = len(df)
    valid_feats = [
        c for c in feature_cols
        if df[c].notna().sum() >= max(10, 0.7 * n)
    ]
    dropped = set(feature_cols) - set(valid_feats)
    if dropped:
        print(f"  Dropped {len(dropped)} features with >30% missing: {sorted(dropped)}")
    feature_cols = valid_feats

    print(f"  {len(feature_cols)} features, target='{args.target}', "
          f"n_boot={args.n_boot}, k={args.k_neighbors}")

    rng = np.random.default_rng(args.seed)

    # --- Predictive MI: MI(feature; AUC) with bootstrap CIs ---
    print("\nComputing predictive MI (bootstrap)...")
    mi_df = bootstrap_mi(df, feature_cols, args.target, args.k_neighbors, args.n_boot, rng)

    mi_path = out_dir / "feature_mi.csv"
    mi_df.to_csv(mi_path, index=False)
    print(f"  Saved → {mi_path}")
    print(mi_df.head(15).to_string(index=False))

    # --- Redundancy matrix: pairwise MI(feature_i; feature_j) ---
    print("\nComputing redundancy matrix (point estimates)...")
    red_df = redundancy_matrix(df, feature_cols, args.k_neighbors)

    red_path = out_dir / "feature_redundancy.csv"
    red_df.to_csv(red_path)
    print(f"  Saved → {red_path}")

    # Print the top redundant pairs
    pairs = []
    for i, fi in enumerate(feature_cols):
        for j, fj in enumerate(feature_cols):
            if j <= i:
                continue
            pairs.append((fi, fj, red_df.loc[fi, fj]))
    pairs.sort(key=lambda x: -x[2])
    print("\nTop 10 most redundant feature pairs:")
    for fi, fj, mi in pairs[:10]:
        print(f"  {mi:.3f}  {fi}  ↔  {fj}")


if __name__ == "__main__":
    main()
