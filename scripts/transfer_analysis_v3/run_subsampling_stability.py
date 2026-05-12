#!/usr/bin/env python3
"""
Subsampling stability analysis for directed coverage.

Question: do training-dataset rankings change if we randomly subsample large
datasets down to match the smallest dataset?

Protocol
--------
For each (train_dataset, eval_dataset) pair in the full coverage CSV:
  1. Load train flow vectors; subsample to each cap in SUBSAMPLE_CAPS.
  2. Build a FAISS Flat index on the subsampled train vectors.
  3. Query with (subsampled) eval vectors → k=1 NN distances.
  4. Compute: mean NN distance + epsilon-coverage at {1, 4, 16} px.

Stability measured as Spearman ρ between full-data metric values and
subsampled metric values, pooling across all (train, eval) pairs.

Outputs (in --output-dir):
  pair_metrics.csv      — raw metric value for every (train, eval, cap)
  stability_table.csv   — Spearman ρ (full vs subsampled) per (metric, cap)
  ranking_stability.csv — per-eval-dataset Spearman ρ of train rankings

Usage:
    python scripts/transfer_analysis_v3/run_subsampling_stability.py \
        [--coverage-csv analysis/coverage_v2_flow_only_raw_joint_full.csv] \
        [--vec-dir /mnt/nvme_1tb_b/coverage_vectors] \
        [--output-dir scripts/transfer_analysis_v3/results/subsampling_stability] \
        [--gpu]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Subsample sizes to test. -1 = full dataset (no subsampling).
# Chosen to span the range from SPair/train (smallest, ~370K) down to tiny.
SUBSAMPLE_CAPS = [50_000, 200_000, 500_000, 2_000_000, -1]
SUBSAMPLE_LABELS = {50_000: "50K", 200_000: "200K", 500_000: "500K",
                    2_000_000: "2M", -1: "full"}

# Epsilon thresholds (in L2 normalized units, image 512×512, scale=2/512=1/256)
# eps_px / 256 → L2 threshold; FAISS returns sqL2, so compare dists**2
IMG_SCALE = 2.0 / 512.0  # = 1/256 normalized L2 per pixel
EPS_PX = [1, 4, 16]
EPS_SQ = {px: (px * IMG_SCALE) ** 2 for px in EPS_PX}

# Cap eval vectors loaded per dataset (for query speed; quantiles stable at 500K)
MAX_EVAL_VECS = 500_000

# Metrics to track
METRICS = (
    ["mean_nn_dist"]
    + [f"eval_covered_eps{px}px" for px in EPS_PX]
    + [f"train_covered_eps{px}px" for px in EPS_PX]
)


# ---------------------------------------------------------------------------
# FAISS helpers
# ---------------------------------------------------------------------------

_GPU_RES = None

def _gpu_res():
    global _GPU_RES
    if _GPU_RES is None:
        _GPU_RES = faiss.StandardGpuResources()
        _GPU_RES.setTempMemory(512 * 1024 * 1024)
    return _GPU_RES


def build_flat_index(vecs: np.ndarray, use_gpu: bool) -> "faiss.Index":
    dim = vecs.shape[1]
    idx = faiss.IndexFlatL2(dim)
    if use_gpu:
        try:
            idx = faiss.index_cpu_to_gpu(_gpu_res(), 0, idx)
        except Exception:
            pass
    idx.add(np.ascontiguousarray(vecs, dtype=np.float32))
    return idx


def release_index(idx) -> None:
    try:
        del idx
    except Exception:
        pass


def knn1_sq_dists(index, query: np.ndarray, batch: int = 100_000) -> np.ndarray:
    """Return (n_query,) squared L2 distances to 1-NN."""
    n = query.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(0, n, batch):
        q = np.ascontiguousarray(query[i : i + batch], dtype=np.float32)
        d, _ = index.search(q, 1)
        out[i : i + d.shape[0]] = d[:, 0]
    return out


# ---------------------------------------------------------------------------
# Vector loading
# ---------------------------------------------------------------------------

def load_flow_vecs(vec_dir: Path, dataset: str, split: str,
                   max_n: int = -1, rng=None) -> np.ndarray | None:
    p = vec_dir / f"{dataset}_{split}_flow.npy"
    if not p.exists():
        return None
    vecs = np.load(p, mmap_mode="r")
    n = len(vecs)
    if max_n > 0 and n > max_n:
        idx = rng.choice(n, max_n, replace=False)
        return np.array(vecs[idx], dtype=np.float32)
    return np.array(vecs[:n], dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-pair metric computation
# ---------------------------------------------------------------------------

def compute_pair_metrics(
    train_vecs: np.ndarray,   # (n_train, 4)
    eval_vecs:  np.ndarray,   # (n_eval, 4)
    use_gpu: bool,
) -> dict:
    """Directed coverage metrics for one (train, eval) pair."""
    metrics = {}

    # ---- eval → train ----
    idx_train = build_flat_index(train_vecs, use_gpu)
    try:
        sq_dists_e2t = knn1_sq_dists(idx_train, eval_vecs)
    finally:
        release_index(idx_train)

    metrics["mean_nn_dist"] = float(np.sqrt(np.maximum(sq_dists_e2t, 0)).mean())
    for px in EPS_PX:
        metrics[f"eval_covered_eps{px}px"] = float(np.mean(sq_dists_e2t <= EPS_SQ[px]))

    # ---- train → eval ----
    idx_eval = build_flat_index(eval_vecs, use_gpu)
    try:
        sq_dists_t2e = knn1_sq_dists(idx_eval, train_vecs)
    finally:
        release_index(idx_eval)

    for px in EPS_PX:
        metrics[f"train_covered_eps{px}px"] = float(np.mean(sq_dists_t2e <= EPS_SQ[px]))

    return metrics


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-csv",
        default="analysis/coverage_v2_flow_only_raw_joint_full.csv")
    parser.add_argument("--vec-dir",
        default="/mnt/nvme_1tb_b/coverage_vectors")
    parser.add_argument("--output-dir",
        default="scripts/transfer_analysis_v3/results/subsampling_stability")
    parser.add_argument("--gpu", action="store_true", default=True)
    parser.add_argument("--no-gpu", dest="gpu", action="store_false")
    parser.add_argument("--caps", nargs="+", type=int, default=SUBSAMPLE_CAPS,
        help="Subsample caps to test (-1 = full). Default: 50000 200000 500000 2000000 -1")
    args = parser.parse_args()

    if not _FAISS_AVAILABLE:
        raise SystemExit("faiss is required. Install faiss-gpu or faiss-cpu.")

    vec_dir  = Path(args.vec_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cov_df = pd.read_csv(args.coverage_csv)
    pairs = (cov_df[["train_dataset", "train_split", "eval_dataset", "eval_split"]]
             .drop_duplicates().reset_index(drop=True))
    print(f"Coverage CSV: {len(pairs)} unique (train, eval) pairs")
    print(f"Caps to test: {args.caps}")
    print(f"GPU: {args.gpu}")

    # Resume support
    pair_path = out_dir / "pair_metrics.csv"
    done_keys: set[tuple] = set()
    if pair_path.exists():
        prev = pd.read_csv(pair_path)
        for _, r in prev.iterrows():
            done_keys.add((r["train_dataset"], r["train_split"],
                           r["eval_dataset"],  r["eval_split"], int(r["cap"])))
        print(f"Resuming: {len(done_keys)} (pair, cap) combos already done")

    rng = np.random.default_rng(42)
    rows = []

    # Cache eval vectors (reused across training datasets)
    eval_cache: dict[tuple, np.ndarray | None] = {}

    # Group by eval dataset to minimize re-loading
    for (eval_ds, eval_sp), eval_group in pairs.groupby(["eval_dataset", "eval_split"]):
        eval_key = (eval_ds, eval_sp)
        if eval_key not in eval_cache:
            eval_cache[eval_key] = load_flow_vecs(
                vec_dir, eval_ds, eval_sp, max_n=MAX_EVAL_VECS, rng=rng)
        eval_vecs = eval_cache[eval_key]
        if eval_vecs is None:
            print(f"  MISSING eval: {eval_ds}/{eval_sp} — skipping")
            continue

        for _, row in eval_group.iterrows():
            train_ds, train_sp = row["train_dataset"], row["train_split"]
            t0 = time.time()

            # Load full train vectors once per training dataset
            train_full = load_flow_vecs(vec_dir, train_ds, train_sp, rng=rng)
            if train_full is None:
                print(f"  MISSING train: {train_ds}/{train_sp} — skipping")
                continue
            n_full = len(train_full)

            for cap in args.caps:
                key = (train_ds, train_sp, eval_ds, eval_sp, cap)
                if key in done_keys:
                    continue

                cap_actual = n_full if cap < 0 else min(cap, n_full)
                if cap_actual < n_full:
                    idx = rng.choice(n_full, cap_actual, replace=False)
                    train_vecs = train_full[idx]
                else:
                    train_vecs = train_full

                m = compute_pair_metrics(train_vecs, eval_vecs, args.gpu)
                rec = {
                    "train_dataset": train_ds, "train_split": train_sp,
                    "eval_dataset":  eval_ds,  "eval_split":  eval_sp,
                    "cap":           cap, "cap_label": SUBSAMPLE_LABELS.get(cap, str(cap)),
                    "n_train_full":  n_full, "n_train_used": cap_actual,
                    **m,
                }
                rows.append(rec)

            elapsed = time.time() - t0
            print(f"  {train_ds}/{train_sp} → {eval_ds}/{eval_sp}  "
                  f"n_full={n_full:,}  {elapsed:.1f}s")

            # Flush every 50 records
            if len(rows) >= 50:
                _flush(rows, pair_path)
                done_keys.update(
                    (r["train_dataset"], r["train_split"],
                     r["eval_dataset"],  r["eval_split"], r["cap"]) for r in rows)
                rows = []

        del eval_cache[eval_key]  # free memory

    if rows:
        _flush(rows, pair_path)

    # ---------------------------------------------------------------------------
    # Analysis: stability tables
    # ---------------------------------------------------------------------------
    if not pair_path.exists():
        print("No pair metrics to analyse.")
        return

    df = pd.read_csv(pair_path)
    print(f"\nAnalysing {len(df)} (pair, cap) records...")

    caps_present = sorted(c for c in df["cap"].unique() if c != -1)
    full_df = df[df["cap"] == -1]

    # Global stability: Spearman(full, subsampled) across all pairs
    stab_rows = []
    for metric in METRICS:
        if metric not in df.columns:
            continue
        full_vals = full_df.set_index(
            ["train_dataset", "train_split", "eval_dataset", "eval_split"])[metric]
        for cap in caps_present:
            sub_df = df[df["cap"] == cap]
            merged = sub_df.set_index(
                ["train_dataset", "train_split", "eval_dataset", "eval_split"]
            )[metric].rename("sub").to_frame().join(full_vals.rename("full"), how="inner")
            merged = merged.dropna()
            if len(merged) < 3:
                continue
            rho = spearmanr(merged["full"], merged["sub"]).statistic
            stab_rows.append({
                "metric":    metric,
                "cap":       cap,
                "cap_label": SUBSAMPLE_LABELS.get(cap, str(cap)),
                "spearman":  float(rho),
                "n_pairs":   len(merged),
            })
    stab_df = pd.DataFrame(stab_rows)
    stab_df.to_csv(out_dir / "stability_table.csv", index=False)
    print("\nGlobal stability (Spearman ρ, full vs subsampled):")
    if not stab_df.empty:
        pivot = stab_df.pivot_table(
            values="spearman", index="metric", columns="cap_label", aggfunc="first")
        print(pivot.round(3).to_string())

    # Per-eval ranking stability: for each eval dataset, Spearman of train rankings
    rank_rows = []
    for metric in METRICS:
        if metric not in df.columns:
            continue
        for eval_ds, eval_grp in df.groupby(["eval_dataset", "eval_split"]):
            full_g  = eval_grp[eval_grp["cap"] == -1].set_index("train_dataset")[metric]
            if len(full_g) < 3:
                continue
            for cap in caps_present:
                sub_g = eval_grp[eval_grp["cap"] == cap].set_index("train_dataset")[metric]
                joined = full_g.rename("full").to_frame().join(sub_g.rename("sub"), how="inner").dropna()
                if len(joined) < 3:
                    continue
                rho = spearmanr(joined["full"], joined["sub"]).statistic
                rank_rows.append({
                    "eval_dataset": eval_ds[0] if isinstance(eval_ds, tuple) else eval_ds,
                    "metric":       metric,
                    "cap":          cap,
                    "cap_label":    SUBSAMPLE_LABELS.get(cap, str(cap)),
                    "spearman":     float(rho),
                    "n_train":      len(joined),
                })
    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(out_dir / "ranking_stability.csv", index=False)

    print("\nPer-eval ranking stability (mean Spearman ρ across eval datasets):")
    if not rank_df.empty:
        summary = rank_df.groupby(["metric", "cap_label"])["spearman"].mean().unstack("cap_label")
        print(summary.round(3).to_string())

    print(f"\n✓ Results saved to {out_dir}/")


def _flush(rows: list, path: Path) -> None:
    batch = pd.DataFrame(rows)
    write_header = not path.exists() or path.stat().st_size == 0
    batch.to_csv(path, mode="a", header=write_header, index=False)


if __name__ == "__main__":
    main()
