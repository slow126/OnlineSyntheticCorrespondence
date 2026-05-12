#!/usr/bin/env python3
"""
Assemble the transfer table: AUC results joined with all precomputed feature CSVs.

Inputs (all relative to project root):
  analysis/leakage_free_flow_kmeans_manifold/auc_results.csv
  analysis/coverage_v2_flow_only_raw_joint_full.csv    (flow raw)
  analysis/coverage_v2_flow_only_raw_joint_kmeans_full.csv  (flow k-means weighted)
  analysis_v3/coverage_dino_full.csv                   (DINO, after Step 0a)
  analysis_v3/symmetric_distances.csv                  (FID + sliced W2, after Step 0b)
  flow_mmd_results_fast.csv
  dino_mmd_results_fast.csv

Output:
  scripts/transfer_analysis_v3/transfer_table.csv

Usage:
    python scripts/transfer_analysis_v3/build_table.py
    python scripts/transfer_analysis_v3/build_table.py --root /path/to/project
    python scripts/transfer_analysis_v3/build_table.py \
        --train-datasets flyingthings imagenet2dwarp pointodyssey sintel spair \
                         synthetic synthetic_2d_warp synthetic_large_zoom \
                         synthetic_random_flipping synthetic_small_zoom
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_auc(root: Path) -> pd.DataFrame:
    path = root / "analysis/leakage_free_flow_kmeans_manifold/auc_results.csv"
    df = pd.read_csv(path)
    cols = ["train_dataset", "benchmark", "model_family", "pretrained", "freeze", "auc_normalized"]
    df = df[cols].copy()
    # Average across repeated runs of the same configuration
    group_cols = ["train_dataset", "benchmark", "model_family", "pretrained", "freeze"]
    n_before = len(df)
    df = df.groupby(group_cols, as_index=False)["auc_normalized"].mean()
    n_after = len(df)
    if n_before != n_after:
        print(f"  Averaged {n_before - n_after} duplicate rows "
              f"({n_before} → {n_after} unique configurations)")
    df["context_id"] = (
        df["benchmark"].astype(str) + "_"
        + df["model_family"].astype(str) + "_"
        + df["pretrained"].astype(str) + "_"
        + df["freeze"].astype(str)
    )
    assert not df.duplicated(subset=["train_dataset", "context_id"]).any()
    return df


# Columns kept from each coverage source (before prefix is applied).
# Only the columns used in feature groups + close neighbours worth keeping.
# Edit here to add back anything needed for new experiments.
_FLOW_KEEP = {
    "mean_nn_eval_to_train_k1", "mean_nn_train_to_eval_k1",
    "eval_covered_by_train_eps1px",  "train_covered_by_eval_eps1px",
    "eval_covered_by_train_eps4px",  "train_covered_by_eval_eps4px",
    "eval_covered_by_train_eps16px", "train_covered_by_eval_eps16px",
}
_FLOW_KM_KEEP = {
    "eval_covered_by_train_eps1px_weighted",  "train_covered_by_eval_eps1px_weighted",
    "eval_covered_by_train_eps4px_weighted",  "train_covered_by_eval_eps4px_weighted",
    "eval_covered_by_train_eps16px_weighted", "train_covered_by_eval_eps16px_weighted",
}
_DINO_KEEP = {
    "mean_nn_eval_to_train_k1", "mean_nn_train_to_eval_k1",
    # null-calibrated coverage (preferred); qnorm kept for backward compatibility
    "eval_covered_by_train_qnorm_k1",  "train_covered_by_eval_qnorm_k1",
}
_KEEP_BY_PREFIX: dict[str, set[str]] = {
    "flow_":    _FLOW_KEEP,
    "flow_km_": _FLOW_KM_KEEP,
    "dino_":    _DINO_KEEP,
}


def load_coverage(path: Path, prefix: str) -> pd.DataFrame:
    """Load a coverage CSV, keeping only the metric columns in the allowlist for this prefix."""
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    non_metric = {"space", "train_dataset", "train_split", "eval_dataset", "eval_split",
                  "train_n_vectors", "eval_n_vectors", "distance_metric", "flow_normalized",
                  "train_n_centroids", "eval_n_centroids", "train_total_weight", "eval_total_weight",
                  "train_kmeans_k", "eval_kmeans_k", "train_radius", "eval_radius"}
    all_metric_cols = [c for c in df.columns if c not in non_metric]

    keep = _KEEP_BY_PREFIX.get(prefix)
    if keep is not None:
        metric_cols = [c for c in all_metric_cols if c in keep]
        dropped = len(all_metric_cols) - len(metric_cols)
        if dropped:
            print(f"    Pruned {dropped} unused metric columns (keeping {len(metric_cols)})")
    else:
        metric_cols = all_metric_cols

    out = df[["train_dataset", "eval_dataset"] + metric_cols].rename(
        columns={"eval_dataset": "benchmark"})
    out = out.rename(columns={c: f"{prefix}{c}" for c in metric_cols})
    return out


def load_mmd(path: Path, col_name: str) -> pd.DataFrame:
    """Load a symmetric MMD CSV. Returns df with (train_dataset, benchmark, col_name)."""
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={"mmd": col_name})
    # Symmetric: emit both orderings so a simple left join on (train_dataset, benchmark) works
    fwd = df[["dataset1", "dataset2", col_name]].rename(
        columns={"dataset1": "train_dataset", "dataset2": "benchmark"})
    rev = df[["dataset2", "dataset1", col_name]].rename(
        columns={"dataset2": "train_dataset", "dataset1": "benchmark"})
    combined = pd.concat([fwd, rev], ignore_index=True).drop_duplicates(
        subset=["train_dataset", "benchmark"])
    return combined[["train_dataset", "benchmark", col_name]]


def load_symmetric_distances(path: Path) -> pd.DataFrame:
    """Load symmetric distances CSV. Returns df with (train_dataset, benchmark, metric_cols)."""
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping (run compute_symmetric_distances.py first).")
        return pd.DataFrame()
    df = pd.read_csv(path)
    metric_cols = [c for c in df.columns if c not in
                   {"train_dataset", "train_split", "eval_dataset", "eval_split"}]
    # Symmetric: emit both orderings, rename eval_dataset → benchmark
    fwd = df[["train_dataset", "eval_dataset"] + metric_cols].rename(
        columns={"eval_dataset": "benchmark"})
    rev = df[["eval_dataset", "train_dataset"] + metric_cols].rename(
        columns={"eval_dataset": "train_dataset", "train_dataset": "benchmark"})
    combined = pd.concat([fwd, rev], ignore_index=True).drop_duplicates(
        subset=["train_dataset", "benchmark"])
    return combined[["train_dataset", "benchmark"] + metric_cols]



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Project root directory.")
    parser.add_argument(
        "--train-datasets", nargs="+", default=None, metavar="DS",
        help="Restrict to these training datasets only (space-separated). "
             "Default: all datasets in the AUC file.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_path = root / "scripts/transfer_analysis_v3/transfer_table.csv"

    print("Loading AUC results...")
    auc = load_auc(root)
    if args.train_datasets is not None:
        allowed = set(args.train_datasets)
        before = len(auc)
        auc = auc[auc["train_dataset"].isin(allowed)].reset_index(drop=True)
        print(f"  Filtered to {len(allowed)} training datasets "
              f"({before} → {len(auc)} rows)")
        unknown = allowed - set(auc["train_dataset"].unique())
        if unknown:
            print(f"  WARNING: requested datasets not found in AUC: {sorted(unknown)}")
    print(f"  {len(auc)} rows, {auc['train_dataset'].nunique()} train datasets, "
          f"{auc['benchmark'].nunique()} benchmarks, {auc['context_id'].nunique()} contexts")

    def join_feature(base: pd.DataFrame, feat: pd.DataFrame, desc: str) -> pd.DataFrame:
        if feat.empty:
            return base
        new_cols = [c for c in feat.columns if c not in {"train_dataset", "benchmark"}]
        merged = base.merge(feat, on=["train_dataset", "benchmark"], how="left")
        n_matched = merged[new_cols[0]].notna().sum() if new_cols else 0
        print(f"  {desc}: {n_matched}/{len(merged)} rows matched, {len(new_cols)} new columns")
        return merged

    print("\nLoading and joining feature CSVs...")

    # --- Flow raw coverage (also extract density counts) ---
    flow_raw_path = root / "analysis/coverage_v2_flow_only_raw_joint_full.csv"
    if flow_raw_path.exists():
        _raw = pd.read_csv(flow_raw_path)
        density_df = _raw[["train_dataset", "eval_dataset", "train_n_vectors", "eval_n_vectors"]].rename(
            columns={"eval_dataset": "benchmark"})
        density_df["log_train_n_vectors"] = np.log1p(density_df["train_n_vectors"])
        density_df["log_eval_n_vectors"]  = np.log1p(density_df["eval_n_vectors"])
        density_df = density_df.drop(columns=["train_n_vectors", "eval_n_vectors"])
    else:
        density_df = pd.DataFrame()

    flow_cov = load_coverage(flow_raw_path, prefix="flow_")
    auc = join_feature(auc, flow_cov, "flow raw coverage")

    # --- Flow k-means coverage ---
    flow_km_cov = load_coverage(
        root / "analysis/coverage_v2_flow_only_raw_joint_kmeans_full.csv", prefix="flow_km_")
    auc = join_feature(auc, flow_km_cov, "flow k-means coverage")

    # --- DINO coverage (after Step 0a) ---
    dino_cov = load_coverage(root / "analysis_v3/coverage_dino_full.csv", prefix="dino_")
    auc = join_feature(auc, dino_cov, "DINO coverage")

    # --- DINO null-calibrated coverage (after Step 0c) ---
    dino_null_path = root / "analysis_v3/dino_null_coverage.csv"
    if dino_null_path.exists():
        null_df = pd.read_csv(dino_null_path)
        # Keep coverage columns (not the threshold diagnostics) and prefix
        null_cols = [c for c in null_df.columns
                     if c.startswith("eval_covered") or c.startswith("train_covered")]
        null_feat = null_df[["train_dataset", "eval_dataset"] + null_cols].rename(
            columns={"eval_dataset": "benchmark"})
        null_feat = null_feat.rename(columns={c: f"dino_{c}" for c in null_cols})
        auc = join_feature(auc, null_feat, "DINO null-calibrated coverage")
    else:
        print(f"  NOTE: {dino_null_path} not found — run compute_dino_null_coverage.py")

    # --- Symmetric distances (after Step 0b) ---
    sym_df = load_symmetric_distances(root / "analysis_v3/symmetric_distances.csv")
    auc = join_feature(auc, sym_df, "symmetric distances (FID/SW2)")

    # --- Flow MMD ---
    flow_mmd = load_mmd(root / "flow_mmd_results_fast.csv", "flow_mmd")
    auc = join_feature(auc, flow_mmd, "flow MMD")

    # --- DINO MMD ---
    dino_mmd = load_mmd(root / "dino_mmd_results_fast.csv", "dino_mmd")
    auc = join_feature(auc, dino_mmd, "DINO MMD")

    # --- Density features ---
    auc = join_feature(auc, density_df, "density (log n_vectors)")

    # Final validation
    dupes = auc.duplicated(subset=["train_dataset", "context_id"])
    assert not dupes.any(), f"{dupes.sum()} duplicate (train_dataset, context_id) rows in output"

    print(f"\nFinal table: {auc.shape[0]} rows × {auc.shape[1]} columns")

    # Missing value summary
    feature_cols = [c for c in auc.columns if c not in
                    {"train_dataset", "benchmark", "model_family", "pretrained",
                     "freeze", "context_id", "auc_normalized"}]
    group_prefixes = [("flow_", "Flow raw"), ("flow_km_", "Flow k-means"),
                      ("dino_", "DINO"), ("flow_mmd", "Flow MMD"), ("dino_mmd", "DINO MMD"),
                      ("flow_fid", "Flow FID"), ("flow_sliced_w2", "Flow SW2"),
                      ("dino_fid", "DINO FID"), ("dino_sliced_w2", "DINO SW2"),
                      ("log_", "Density")]
    print("\nMissing value summary:")
    for prefix, label in group_prefixes:
        cols = [c for c in feature_cols if c.startswith(prefix)]
        if cols:
            n_miss = auc[cols].isna().any(axis=1).sum()
            print(f"  {label:20s}: {n_miss:4d}/{len(auc)} rows with any NaN  ({len(cols)} columns)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    auc.to_csv(out_path, index=False)
    print(f"\n✓ Saved to {out_path}")


if __name__ == "__main__":
    main()
