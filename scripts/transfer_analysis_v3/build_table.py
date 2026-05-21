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
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

VECTOR_DENSITY_CAP = 10_000.0


def load_auc(root: Path) -> pd.DataFrame:
    path = root / "analysis/leakage_free_flow_kmeans_manifold/auc_results.csv"
    df = pd.read_csv(path)
    # Remap model_family to architecture (raft/catspp) from run_name.
    # The original model_family reflects snapshot directory names (e.g. raft_2d_mix,
    # synth_2d, 2d_warps, synthetic_long) which are the same RAFT/CATS++ architectures
    # trained on different data — not different model families.
    arch = df["run_name"].str.extract(r"_(raft_full|raft_baseline|cats)_", expand=False)
    df.loc[arch.isin(["raft_full", "raft_baseline"]), "model_family"] = "raft"
    df.loc[arch == "cats", "model_family"] = "catspp"
    # Drop early-terminated runs — fewer than 3 checkpoints in the 5000-step window
    # means the run was killed before it produced useful data.
    n_early = (df["auc_points"] < 3).sum()
    if n_early:
        print(f"  Dropping {n_early} early-terminated rows (auc_points < 3)")
        df = df[df["auc_points"] >= 3].copy()
    cols = ["train_dataset", "benchmark", "model_family", "pretrained", "freeze",
            "auc_normalized", "peak_pck"]
    df = df[cols].copy()
    # Average across repeated runs of the same configuration
    group_cols = ["train_dataset", "benchmark", "model_family", "pretrained", "freeze"]
    n_before = len(df)
    df = df.groupby(group_cols, as_index=False)[["auc_normalized", "peak_pck"]].mean()
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


def _parse_flow_count_stats_name(path: Path) -> tuple[str, str, str] | None:
    """Parse flow_counts_{dataset}_{split}_{space}.json."""
    stem = path.stem
    prefix = "flow_counts_"
    if not stem.startswith(prefix):
        return None
    parts = stem[len(prefix):].split("_")
    if len(parts) < 3:
        return None
    space = parts[-1]
    split = parts[-2]
    dataset = "_".join(parts[:-2])
    if space not in {"flow", "dino"} or split not in {"train", "test", "val"} or not dataset:
        return None
    return dataset, split, space


def _stats_nested(stats: dict, key: str, field: str) -> float:
    value = stats.get(key, {})
    if not isinstance(value, dict):
        return np.nan
    return value.get(field, np.nan)


def _safe_float(value: object, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_vector_profile_stats(vec_dir: Path) -> pd.DataFrame:
    """Load per-dataset sample count and vectors-per-sample profile stats."""
    stats_dir = vec_dir / "stats"
    if not stats_dir.exists():
        print(f"  NOTE: {stats_dir} not found — skipping vector profile stats.")
        return pd.DataFrame()

    rows_by_key: dict[tuple[str, str], tuple[int, dict]] = {}
    priority = {"dino": 0, "flow": 1}
    for path in sorted(stats_dir.glob("flow_counts_*_*.json")):
        parsed = _parse_flow_count_stats_name(path)
        if parsed is None:
            continue
        dataset, split, space = parsed
        key = (dataset, split)
        prio = priority[space]
        if key in rows_by_key and rows_by_key[key][0] > prio:
            continue
        try:
            with path.open("r") as f:
                stats = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  WARNING: could not read {path}: {exc}")
            continue

        images_seen = stats.get("images_seen", np.nan)
        if not pd.notna(images_seen) or float(images_seen) <= 0:
            continue
        images_seen = float(images_seen)
        row = {
            "dataset": dataset,
            "split": split,
            "vector_profile_source": space,
            "n_samples": images_seen,
            "zero_image_frac": _safe_float(stats.get("images_with_zero", 0.0), 0.0) / images_seen,
            "valid_vectors_per_sample": _safe_float(stats.get("total_valid_vectors")) / images_seen,
            "sampled_vectors_per_sample": _safe_float(stats.get("total_sampled_vectors")) / images_seen,
            "retained_vectors_per_sample": _safe_float(stats.get("total_vectors_retained")) / images_seen,
            "valid_vectors_mean": _stats_nested(stats, "valid_counts", "mean"),
            "valid_vectors_median": _stats_nested(stats, "valid_counts", "median"),
            "valid_vectors_p10": _stats_nested(stats, "valid_counts", "p10"),
            "valid_vectors_p90": _stats_nested(stats, "valid_counts", "p90"),
            "valid_vectors_p95": _stats_nested(stats, "valid_counts", "p95"),
            "sampled_vectors_mean": _stats_nested(stats, "sampled_counts", "mean"),
            "sampled_vectors_median": _stats_nested(stats, "sampled_counts", "median"),
        }
        rows_by_key[key] = (prio, row)

    if not rows_by_key:
        print(f"  NOTE: no flow_counts stats found in {stats_dir}.")
        return pd.DataFrame()
    rows = [row for _, row in rows_by_key.values()]
    out = pd.DataFrame(rows).drop_duplicates(subset=["dataset", "split"])
    print(f"  Loaded vector profile stats: {len(out)} dataset/split rows from {stats_dir}")
    return out


def make_vector_profile_features(
    auc: pd.DataFrame,
    profile_stats: pd.DataFrame,
    eval_splits: dict[str, str],
) -> pd.DataFrame:
    """Expand dataset/split vector profile stats to train/eval pair features."""
    if profile_stats.empty:
        return pd.DataFrame()

    index = profile_stats.set_index(["dataset", "split"])
    numeric_cols = [
        "n_samples",
        "zero_image_frac",
        "valid_vectors_per_sample",
        "sampled_vectors_per_sample",
        "retained_vectors_per_sample",
        "valid_vectors_mean",
        "valid_vectors_median",
        "valid_vectors_p10",
        "valid_vectors_p90",
        "valid_vectors_p95",
        "sampled_vectors_mean",
        "sampled_vectors_median",
    ]
    log_cols = [c for c in numeric_cols if c != "zero_image_frac"]

    rows = []
    for td, bm in sorted({(r.train_dataset, r.benchmark) for _, r in auc.iterrows()}):
        row = {"train_dataset": td, "benchmark": bm}
        for axis, key in [
            ("train", (td, "train")),
            ("eval", (bm, eval_splits.get(bm, "val"))),
        ]:
            if key not in index.index:
                continue
            stats = index.loc[key]
            for col in numeric_cols:
                value = stats.get(col, np.nan)
                if col in log_cols:
                    row[f"log_{axis}_{col}"] = np.log1p(value) if pd.notna(value) else np.nan
                    if col == "valid_vectors_per_sample":
                        # Treat very dense supervision as effectively dense enough.
                        # This keeps profile controls from extrapolating between
                        # SPair-like sparse labels and fully dense flow fields.
                        capped = min(float(value), VECTOR_DENSITY_CAP) if pd.notna(value) else np.nan
                        row[f"log_{axis}_{col}_capped"] = (
                            np.log1p(capped) if pd.notna(capped) else np.nan
                        )
                else:
                    row[f"{axis}_{col}"] = value
        rows.append(row)

    return pd.DataFrame(rows)


EVAL_SPLITS = {
    "flyingthings": "test", "kitti2012": "val", "kitti2015": "val",
    "spair": "test", "pfpascal": "test", "pfwillow": "test",
    "pointodyssey": "test", "tss": "val", "middlebury": "val", "synthetic": "val",
}



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Project root directory.")
    parser.add_argument(
        "--vec-dir", default="/mnt/nvme_1tb_b/coverage_vectors",
        help="Coverage vector directory containing stats/flow_counts_*.json. "
             "Used for sample-count and vectors-per-sample profile controls.",
    )
    parser.add_argument(
        "--train-datasets", nargs="+", default=None, metavar="DS",
        help="Restrict to these training datasets only (space-separated). "
             "Default: all datasets in the AUC file.",
    )
    parser.add_argument(
        "--min-context-size", type=int, default=None, metavar="N",
        help="Drop contexts where fewer than N training datasets have results. "
             "Prevents sparse contexts from inflating rankings of underrepresented "
             "datasets (e.g. imagenet2dwarp which only appears with 1-2 competitors). "
             "Recommended: 8 for a clean apples-to-apples comparison.",
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
    if args.min_context_size is not None:
        ctx_sizes = auc.groupby("context_id")["train_dataset"].nunique()
        valid_contexts = ctx_sizes[ctx_sizes >= args.min_context_size].index
        before = len(auc)
        auc = auc[auc["context_id"].isin(valid_contexts)].reset_index(drop=True)
        dropped_ds = set(auc["train_dataset"].unique())
        print(f"  Context-size filter (>= {args.min_context_size}): "
              f"{before} → {len(auc)} rows, "
              f"{len(valid_contexts)} contexts, "
              f"datasets: {sorted(auc['train_dataset'].unique())}")
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

    # Supplement density_df with n_vectors from pairwise_self_distances.csv.
    # Fills datasets missing from the coverage CSV (e.g. movi_f, synthetic/val).
    _sd_path = root / "analysis_v3/pairwise_self_distances.csv"
    if _sd_path.exists():
        _sd = pd.read_csv(_sd_path)
        _sd = _sd[_sd["space"] == "flow"]
        _pa = _sd[["dataset_a", "split_a", "n_vecs_a"]].rename(
            columns={"dataset_a": "dataset", "split_a": "split", "n_vecs_a": "n_vecs"})
        _pb = _sd[["dataset_b", "split_b", "n_vecs_b"]].rename(
            columns={"dataset_b": "dataset", "split_b": "split", "n_vecs_b": "n_vecs"})
        _nvec = (pd.concat([_pa, _pb])
                   .drop_duplicates(subset=["dataset", "split"])
                   .set_index(["dataset", "split"])["n_vecs"])
        _known = (set(zip(density_df["train_dataset"], density_df["benchmark"]))
                  if not density_df.empty else set())
        _all_pairs = {(r.train_dataset, r.benchmark) for _, r in auc.iterrows()}
        _supp = []
        for td, bm in _all_pairs - _known:
            tr_key = (td, "train")
            ev_key = (bm, EVAL_SPLITS.get(bm, "val"))
            if tr_key in _nvec.index and ev_key in _nvec.index:
                _supp.append({"train_dataset": td, "benchmark": bm,
                               "log_train_n_vectors": float(np.log1p(_nvec[tr_key])),
                               "log_eval_n_vectors":  float(np.log1p(_nvec[ev_key]))})
        if _supp:
            density_df = pd.concat([density_df, pd.DataFrame(_supp)], ignore_index=True)
            print(f"  Supplemented {len(_supp)} density rows from pairwise_self_distances.csv")

    flow_cov = load_coverage(flow_raw_path, prefix="flow_")
    auc = join_feature(auc, flow_cov, "flow raw coverage")

    # Backfill directed raw-flow NN/epsilon features from pairwise_self_distances.
    # This covers newer train/eval pairs when the legacy coverage CSV is stale.
    if _sd_path.exists():
        cross = _sd[
            (_sd["space"] == "flow") &
            (_sd["pair_type"] == "train_eval")
        ].copy()
        fill_map = {
            "mean_nn_b_to_a": "flow_mean_nn_eval_to_train_k1",
            "mean_nn_a_to_b": "flow_mean_nn_train_to_eval_k1",
            "b_covered_by_a_eps1px": "flow_eval_covered_by_train_eps1px",
            "a_covered_by_b_eps1px": "flow_train_covered_by_eval_eps1px",
            "b_covered_by_a_eps4px": "flow_eval_covered_by_train_eps4px",
            "a_covered_by_b_eps4px": "flow_train_covered_by_eval_eps4px",
            "b_covered_by_a_eps16px": "flow_eval_covered_by_train_eps16px",
            "a_covered_by_b_eps16px": "flow_train_covered_by_eval_eps16px",
        }
        available = [c for c in fill_map if c in cross.columns]
        if available:
            fill_feat = cross[["dataset_a", "dataset_b"] + available].rename(
                columns={"dataset_a": "train_dataset", "dataset_b": "benchmark", **fill_map}
            )
            fill_cols = [fill_map[c] for c in available]
            before_missing = {
                c: int(auc[c].isna().sum()) if c in auc.columns else len(auc)
                for c in fill_cols
            }
            auc = auc.merge(
                fill_feat,
                on=["train_dataset", "benchmark"],
                how="left",
                suffixes=("", "__self_fill"),
            )
            for col in fill_cols:
                fill_col = f"{col}__self_fill"
                if fill_col not in auc.columns:
                    continue
                if col in auc.columns:
                    auc[col] = auc[col].fillna(auc[fill_col])
                else:
                    auc[col] = auc[fill_col]
                auc = auc.drop(columns=[fill_col])
            filled = {
                c: before_missing[c] - int(auc[c].isna().sum())
                for c in fill_cols if c in auc.columns
            }
            n_filled = sum(filled.values())
            if n_filled:
                print(
                    "  Backfilled raw flow NN/epsilon features from "
                    f"pairwise_self_distances.csv ({n_filled} cells)"
                )

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

    # --- Pairwise self-distances: centrality features + KL divergence ---
    # Loaded from analysis_v3/pairwise_self_distances.csv (Step 0d), which now
    # contains train-train, eval-eval, and train-eval cross pairs.
    # Cross pairs (pair_type=="train_eval") supply KL divergence features that
    # replace the legacy kl_flow_features.csv / kl_dino_features.csv files.
    self_dist_path = root / "analysis_v3/pairwise_self_distances.csv"
    if self_dist_path.exists():
        sd = pd.read_csv(self_dist_path)

        # Centrality: per-dataset mean NN distance to all others in the same group
        for space in ["flow", "dino"]:
            space_df = sd[(sd["space"] == space) & (sd["dataset_a"] != sd["dataset_b"])]
            if space_df.empty or "mean_nn_sym" not in space_df.columns:
                continue
            train_iso = (space_df[space_df["pair_type"] == "train_train"]
                         .groupby("dataset_a")["mean_nn_sym"].mean())
            eval_iso  = (space_df[space_df["pair_type"] == "eval_eval"]
                         .groupby("dataset_a")["mean_nn_sym"].mean())
            auc[f"{space}_train_isolation"] = auc["train_dataset"].map(train_iso)
            auc[f"{space}_eval_isolation"]  = auc["benchmark"].map(eval_iso)

        # KL divergence from cross pairs (train × eval)
        for space, prefix, desc in [
            ("flow", "flow_", "flow KL divergence"),
            ("dino", "dino_", "DINO KL divergence"),
        ]:
            cross = sd[(sd["space"] == space) & (sd["pair_type"] == "train_eval")]
            kl_cols = [c for c in cross.columns if c.startswith("kl_")]
            if cross.empty or not kl_cols:
                continue
            kl_feat = cross[["dataset_a", "dataset_b"] + kl_cols].rename(
                columns={"dataset_a": "train_dataset", "dataset_b": "benchmark"})
            # kl_a_to_b → kl_train_to_eval, kl_b_to_a → kl_eval_to_train
            rename_map = {
                c: c.replace("kl_a_to_b", "kl_train_to_eval")
                    .replace("kl_b_to_a", "kl_eval_to_train")
                for c in kl_cols
            }
            kl_feat = kl_feat.rename(columns=rename_map)
            renamed = list(rename_map.values())
            kl_feat = kl_feat.rename(columns={c: f"{prefix}{c}" for c in renamed})
            auc = join_feature(auc, kl_feat, desc)

        print(f"  Loaded pairwise distances + KL from {self_dist_path.name}")
    else:
        # Legacy fallback: load KL from the separate per-space files if Step 0d
        # has not yet been run.
        for path, prefix, desc in [
            (root / "analysis_v3/kl_flow_features.csv", "flow_", "flow KL divergence (legacy)"),
            (root / "analysis_v3/kl_dino_features.csv", "dino_", "DINO KL divergence (legacy)"),
        ]:
            if path.exists():
                kl_df = pd.read_csv(path)
                kl_metric_cols = [c for c in kl_df.columns if c.startswith("kl_")]
                kl_feat = kl_df[["train_dataset", "eval_dataset"] + kl_metric_cols].rename(
                    columns={"eval_dataset": "benchmark"})
                kl_feat = kl_feat.rename(columns={c: f"{prefix}{c}" for c in kl_metric_cols})
                auc = join_feature(auc, kl_feat, desc)
        print(f"  NOTE: {self_dist_path} not found — run compute_pairwise_self_distances.py")

    # --- Density features ---
    auc = join_feature(auc, density_df, "density (log n_vectors)")

    # --- Vector profile controls: sample count and vectors per sample ---
    profile_stats = load_vector_profile_stats(Path(args.vec_dir))
    profile_df = make_vector_profile_features(auc, profile_stats, EVAL_SPLITS)
    auc = join_feature(auc, profile_df, "vector profile (samples + vectors/sample)")

    # --- Random scalar features (mechanism-only baseline) ---
    # One seeded random value per unique train dataset and per unique benchmark.
    # These carry zero real signal; paired with random_dist IDW they isolate the
    # IDW mechanism from both the neighborhood quality and the feature quality.
    rng = np.random.default_rng(42)
    train_datasets = auc["train_dataset"].unique()
    benchmarks     = auc["benchmark"].unique()
    random_train_map = dict(zip(train_datasets, rng.standard_normal(len(train_datasets))))
    random_eval_map  = dict(zip(benchmarks,     rng.standard_normal(len(benchmarks))))
    auc["random_train"] = auc["train_dataset"].map(random_train_map)
    auc["random_eval"]  = auc["benchmark"].map(random_eval_map)

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
                      ("log_train_n_vectors", "Total-vector density"),
                      ("log_eval_n_vectors", "Total-vector density"),
                      ("log_train_n_samples", "Sample count"),
                      ("log_eval_n_samples", "Sample count"),
                      ("log_train_valid_vectors", "Train vectors/sample"),
                      ("log_eval_valid_vectors", "Eval vectors/sample"),
                      ("train_zero_image_frac", "Train zero-image frac"),
                      ("eval_zero_image_frac", "Eval zero-image frac")]
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
