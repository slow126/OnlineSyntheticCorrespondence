#!/usr/bin/env python3
"""
Null-distribution-calibrated cosine coverage for DINO features.

For each (train, eval) dataset pair:
  1. Sample N random cross-set pairs (train_i, eval_j) — independent, no
     nearest-neighbor structure — and compute their cosine similarities.
     This is the null distribution: "how similar are unrelated DINO features?"
  2. Find the cosine similarity at null percentiles {80, 90, 95, 99}.
  3. Coverage: fraction of eval points whose nearest-train cosine similarity
     exceeds the null threshold.

Because the DINO vectors are L2-normalised, cosine similarity = 1 - sqL2/2.
We convert FAISS sqL2 distances to cosine similarities in-place.

Output columns (one row per train×eval pair):
  dino_eval_covered_by_train_null{p}  — witness-existence coverage, eval→train
  dino_train_covered_by_eval_null{p}  — witness-existence coverage, train→eval
  dino_null_cos_threshold_{p}         — actual threshold used (diagnostic)

The paper argument: "Both directed-coverage operators share witness-existence
semantics; thresholds are calibrated to the natural scale of each space —
physical pixel units for BFV and null-distribution percentiles for DINO."

Usage:
    python scripts/transfer_analysis_v3/compute_dino_null_coverage.py \
        [--coverage-csv analysis_v3/coverage_dino_full.csv] \
        [--vec-dir /mnt/nvme_1tb_b/coverage_vectors] \
        [--output analysis_v3/dino_null_coverage.csv] \
        [--null-percentiles 80 90 95 99] \
        [--null-samples 100000] \
        [--max-vecs 500000] [--gpu]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import faiss
    _FAISS = True
except ImportError:
    _FAISS = False

NULL_PERCENTILES = [80, 90, 95, 99]
NULL_SAMPLES     = 100_000   # random pairs for null estimation
MAX_VECS         = 500_000   # cap per dataset (memory + speed)

_GPU_RES = None


def _gpu_resources():
    global _GPU_RES
    if _GPU_RES is None:
        _GPU_RES = faiss.StandardGpuResources()
        _GPU_RES.setTempMemory(512 * 1024 * 1024)
    return _GPU_RES


def _release(idx):
    try:
        del idx
    except Exception:
        pass


def _build_flat(vecs: np.ndarray, use_gpu: bool) -> "faiss.Index":
    idx = faiss.IndexFlatL2(vecs.shape[1])
    if use_gpu:
        try:
            idx = faiss.index_cpu_to_gpu(_gpu_resources(), 0, idx)
        except Exception:
            pass
    idx.add(np.ascontiguousarray(vecs, dtype=np.float32))
    return idx


def _knn1_cos(index, query: np.ndarray, batch: int = 50_000) -> np.ndarray:
    """Return (n,) cosine similarities to 1-NN for L2-normalised vectors."""
    n = query.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(0, n, batch):
        q = np.ascontiguousarray(query[i : i + batch], dtype=np.float32)
        sq_l2, _ = index.search(q, 1)
        # cos_sim = 1 - sqL2 / 2  (exact for unit-norm vectors)
        out[i : i + q.shape[0]] = 1.0 - sq_l2[:, 0] / 2.0
    return out


def load_vecs(vec_dir: Path, dataset: str, split: str,
              max_n: int, rng) -> np.ndarray | None:
    p = vec_dir / f"{dataset}_{split}_dino_pca256_l2norm.npy"
    if not p.exists():
        return None
    v = np.load(p, mmap_mode="r")
    n = len(v)
    if max_n > 0 and n > max_n:
        idx = rng.choice(n, max_n, replace=False)
        return np.array(v[idx], dtype=np.float32)
    return np.array(v, dtype=np.float32)


def compute_null_thresholds(
    train_vecs: np.ndarray,
    eval_vecs:  np.ndarray,
    n_samples:  int,
    percentiles: list[int],
    rng,
) -> dict[int, float]:
    """
    Sample random cross-set pairs and return cosine sim at each percentile.

    Null = distribution of cosine similarities between independently drawn
    train and eval DINO features (no structural matching).
    """
    n_tr = len(train_vecs)
    n_ev = len(eval_vecs)
    t_idx = rng.choice(n_tr, min(n_samples, n_tr * n_ev), replace=True)
    e_idx = rng.choice(n_ev, len(t_idx), replace=True)
    # cosine similarity for L2-normalised vectors = dot product
    cos_null = np.einsum("ij,ij->i",
                         train_vecs[t_idx].astype(np.float64),
                         eval_vecs[e_idx].astype(np.float64))
    return {p: float(np.percentile(cos_null, p)) for p in percentiles}


def compute_coverage_at_thresholds(
    cos_sims: np.ndarray,   # (n,) actual NN cosine similarities
    thresholds: dict[int, float],
) -> dict[str, float]:
    """Fraction of points whose NN cosine sim exceeds each null threshold."""
    return {f"null{p}": float(np.mean(cos_sims > t))
            for p, t in thresholds.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-csv",  default="analysis_v3/coverage_dino_full.csv")
    parser.add_argument("--vec-dir",       default="/mnt/nvme_1tb_b/coverage_vectors")
    parser.add_argument("--output",        default="analysis_v3/dino_null_coverage.csv")
    parser.add_argument("--null-percentiles", nargs="+", type=int,
                        default=NULL_PERCENTILES)
    parser.add_argument("--null-samples",  type=int, default=NULL_SAMPLES)
    parser.add_argument("--max-vecs",      type=int, default=MAX_VECS)
    parser.add_argument("--gpu",           action="store_true", default=True)
    parser.add_argument("--no-gpu",        dest="gpu", action="store_false")
    args = parser.parse_args()

    if not _FAISS:
        raise SystemExit("faiss is required.")

    vec_dir  = Path(args.vec_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cov_df = pd.read_csv(args.coverage_csv)
    pairs  = (cov_df[["train_dataset", "train_split", "eval_dataset", "eval_split"]]
              .drop_duplicates().reset_index(drop=True))
    print(f"Pairs to compute: {len(pairs)}")
    print(f"Null percentiles: {args.null_percentiles}")
    print(f"Null samples:     {args.null_samples:,}")
    print(f"Max vecs/dataset: {args.max_vecs:,}")
    print(f"GPU:              {args.gpu}")

    # Resume support
    done: set[tuple] = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        for _, r in prev.iterrows():
            done.add((r["train_dataset"], r["train_split"],
                      r["eval_dataset"],  r["eval_split"]))
        print(f"Already done: {len(done)}, remaining: {len(pairs) - len(done)}")
    todo = pairs[~pairs.apply(
        lambda r: (r["train_dataset"], r["train_split"],
                   r["eval_dataset"],  r["eval_split"]) in done, axis=1
    )].reset_index(drop=True)
    if todo.empty:
        print("All pairs already computed.")
        return

    rng  = np.random.default_rng(42)
    rows = []

    # Cache vectors — eval datasets are reused across many train datasets
    eval_cache: dict[tuple, np.ndarray | None] = {}

    for _, pair in todo.iterrows():
        td, ts = pair["train_dataset"], pair["train_split"]
        ed, es = pair["eval_dataset"],  pair["eval_split"]
        t0 = time.time()

        # Load vectors
        train_vecs = load_vecs(vec_dir, td, ts, args.max_vecs, rng)
        if train_vecs is None:
            print(f"  MISSING train: {td}/{ts}")
            continue

        ev_key = (ed, es)
        if ev_key not in eval_cache:
            eval_cache[ev_key] = load_vecs(vec_dir, ed, es, args.max_vecs, rng)
        eval_vecs = eval_cache[ev_key]
        if eval_vecs is None:
            print(f"  MISSING eval: {ed}/{es}")
            continue

        # ---- Null distribution ----
        null_thresholds = compute_null_thresholds(
            train_vecs, eval_vecs,
            n_samples=args.null_samples,
            percentiles=args.null_percentiles,
            rng=rng,
        )

        # ---- eval → train coverage ----
        idx_train = _build_flat(train_vecs, args.gpu)
        try:
            cos_e2t = _knn1_cos(idx_train, eval_vecs)
        finally:
            _release(idx_train)
        e2t_cov = compute_coverage_at_thresholds(cos_e2t, null_thresholds)

        # ---- train → eval coverage ----
        idx_eval = _build_flat(eval_vecs, args.gpu)
        try:
            cos_t2e = _knn1_cos(idx_eval, train_vecs)
        finally:
            _release(idx_eval)
        t2e_cov = compute_coverage_at_thresholds(cos_t2e, null_thresholds)

        elapsed = time.time() - t0
        row = {
            "train_dataset": td, "train_split": ts,
            "eval_dataset":  ed, "eval_split":  es,
        }
        for p in args.null_percentiles:
            row[f"eval_covered_by_train_null{p}"]  = e2t_cov[f"null{p}"]
            row[f"train_covered_by_eval_null{p}"]  = t2e_cov[f"null{p}"]
            row[f"null_cos_threshold_{p}"]         = null_thresholds[p]

        rows.append(row)
        print(f"  {td}/{ts} → {ed}/{es}  "
              f"n_tr={len(train_vecs):,}  n_ev={len(eval_vecs):,}  "
              f"null90={null_thresholds[90]:.3f}  "
              f"cov_e2t_null90={e2t_cov['null90']:.3f}  "
              f"cov_t2e_null90={t2e_cov['null90']:.3f}  "
              f"{elapsed:.1f}s")

        if len(rows) >= 10:
            _flush(rows, out_path)
            rows = []

    if rows:
        _flush(rows, out_path)

    if out_path.exists():
        final = pd.read_csv(out_path)
        print(f"\n✓ {len(final)} pairs saved to {out_path}")
        for p in args.null_percentiles:
            col = f"eval_covered_by_train_null{p}"
            if col in final.columns:
                v = final[col].dropna()
                print(f"  null{p:2d}: range [{v.min():.3f}, {v.max():.3f}]  "
                      f"median {v.median():.3f}")


def _flush(rows: list, path: Path) -> None:
    df = pd.DataFrame(rows)
    write_header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="a", header=write_header, index=False)


if __name__ == "__main__":
    main()
