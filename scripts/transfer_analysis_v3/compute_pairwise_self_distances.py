#!/usr/bin/env python3
"""
Compute pairwise distances between training datasets (train-train), between
evaluation benchmarks (eval-eval), and between training datasets and eval
benchmarks (train-eval cross pairs).

All metrics extracted from ONE FAISS search per direction per pair:
  - Mean k=1 NN distance (a→b and b→a)
  - ε-coverage at 1px, 4px, 16px (both directions)
  - KL divergence at k=5 and k=20 (both directions)

KL divergence is always computed. Self-kNN distances are loaded from the
knn_self file cache when available; otherwise computed on the fly from the
loaded vectors and memoized in memory for the rest of the run.

Both flow (joint 4D, normalized) and DINO (PCA-256, L2-norm) spaces.

The train-eval cross pairs replace the separate kl_flow_features.csv /
kl_dino_features.csv files — Step 0d now produces all pairwise distances
and KL divergences in one pass.

Output: analysis_v3/pairwise_self_distances.csv
Resumable: pairs already in CSV are skipped.

Usage:
    python scripts/transfer_analysis_v3/compute_pairwise_self_distances.py
    python scripts/transfer_analysis_v3/compute_pairwise_self_distances.py \
        --vec-dir /mnt/nvme_1tb_b/coverage_vectors \
        --output analysis_v3/pairwise_self_distances.csv \
        --max-flow 1000000 --max-dino 500000 --gpu \
        --pair-types train_eval eval_eval

Parallel / cluster usage (SLURM job array, one GPU per rank):

    # Each rank writes rank_{K}.csv independently; merge when all done.
    sbatch --array=0-7 run_pairwise.sh  # script passes --rank $SLURM_ARRAY_TASK_ID

    # Example rank invocation (dino only, since flow is already done locally):
    python scripts/transfer_analysis_v3/compute_pairwise_self_distances.py \
        --vec-dir /scratch/coverage_vectors \
        --output /scratch/pairwise/rank_${SLURM_ARRAY_TASK_ID}.csv \
        --seed-csv analysis_v3/pairwise_self_distances.csv \
        --spaces dino --stride 8 --rank ${SLURM_ARRAY_TASK_ID} --gpu

    # After all ranks finish, merge + symmetrize:
    python scripts/transfer_analysis_v3/merge_pairwise_distances.py \
        --inputs /scratch/pairwise/rank_*.csv \
        --seed-csv analysis_v3/pairwise_self_distances.csv \
        --output analysis_v3/pairwise_self_distances.csv
"""

import argparse
import gc
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.coverage import faiss_ops, spaces
from scripts.coverage import kl as kl_mod

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

TRAIN_DATASETS = [
    ("flyingthings",              "train"),
    ("imagenet2dwarp",            "train"),
    ("movi_f",                    "train"),
    ("pointodyssey",              "train"),
    ("sintel",                    "train"),
    ("spair",                     "train"),
    ("synthetic",                 "train"),
    ("synthetic_2d_warp",         "train"),
    ("synthetic_large_zoom",      "train"),
    ("synthetic_random_flipping", "train"),
    ("synthetic_small_zoom",      "train"),
]

EVAL_DATASETS = [
    ("flyingthings",  "test"),
    ("kitti2012",     "val"),
    ("kitti2015",     "val"),
    ("spair",         "test"),
    ("pfpascal",      "test"),
    ("pfwillow",      "test"),
    ("pointodyssey",  "test"),
    ("tss",           "val"),
    ("middlebury",    "val"),
    ("synthetic",     "val"),
]

IMG_W, IMG_H = 512, 512
EPS_PX = [1.0, 4.0, 16.0]
_SCALE = 2.0 / IMG_W
EPS_SQ = {e: (e * _SCALE) ** 2 for e in EPS_PX}

# k values for kNN KL estimator (Wang et al. 2009). Matches existing kl_flow/dino_features.csv.
K_KL_VALUES = [5, 20]
_K_MAX = max(K_KL_VALUES)


def _eps_label(eps_px: float) -> str:
    if float(eps_px).is_integer():
        return f"eps{int(eps_px)}px"
    return f"eps{eps_px:g}px".replace(".", "p")


# ---------------------------------------------------------------------------
# Vector loading
# ---------------------------------------------------------------------------

def _subsample_with_idx(
    vecs: np.ndarray, n_max: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray | None]:
    """Returns (vecs, idx) where idx is None if no subsampling was needed."""
    if len(vecs) <= n_max:
        return vecs, None
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(vecs), n_max, replace=False)
    idx.sort()
    return vecs[idx], idx


def load_flow_vectors(
    name: str, split: str, vec_dir: Path, n_max: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    path = vec_dir / f"{name}_{split}_flow.npy"
    if not path.exists():
        return None, None
    vecs = np.load(path, mmap_mode="r")
    vecs, idx = _subsample_with_idx(vecs, n_max)
    vecs = np.ascontiguousarray(vecs, dtype=np.float32)
    vecs = spaces.normalize_flow_vectors(vecs, IMG_W, IMG_H)
    vecs = spaces.to_joint_space(vecs, alpha=1.0)
    return vecs, idx


def load_dino_vectors(
    name: str, split: str, vec_dir: Path, n_max: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    path = vec_dir / f"{name}_{split}_dino_pca256_l2norm.npy"
    if not path.exists():
        return None, None
    vecs = np.load(path, mmap_mode="r")
    vecs, idx = _subsample_with_idx(vecs, n_max)
    return np.ascontiguousarray(vecs, dtype=np.float32), idx


# ---------------------------------------------------------------------------
# knn_self: load from file cache or compute on the fly
# ---------------------------------------------------------------------------

# Memoizes on-the-fly knn_self computations so each dataset pays the cost once.
# Key: (name, split, space) → (N_sub, _K_MAX) float32 L2 array  |  None on failure.
_KNNSELF_COMPUTED: dict = {}


def _knnself_path(name: str, split: str, space: str, vec_dir: Path) -> Path | None:
    if space == "flow":
        stem = f"knnself_{name}_{split}_flow_joint_norm2x1_sqL2_k40_dedup_a1"
    else:
        stem = f"knnself_{name}_{split}_dino_features_pca256_l2_sqL2_k40_dedup"
    for ext in [".npy", ".npz"]:
        p = vec_dir / "knn_self" / (stem + ext)
        if p.exists():
            return p
    return None


def _density_knnself_path(name: str, split: str, space: str, vec_dir: Path,
                          n_vecs: int) -> Path:
    """Level-specific persistent self-kNN cache.

    The canonical files under knn_self/ store squared-L2 distances for the full
    extraction. These density-level files store L2 distances for exactly the
    deterministic subsample used at this N, so keep them in a separate namespace.
    """
    if space == "flow":
        stem = f"knnself_{name}_{split}_flow_joint_norm2x1_L2_k{_K_MAX}_N{n_vecs}_seed0"
    else:
        stem = f"knnself_{name}_{split}_dino_pca256_L2_k{_K_MAX}_N{n_vecs}_seed0"
    return vec_dir / "knn_self" / "density_levels" / f"{stem}.npy"


def _load_density_knnself(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        arr = np.load(path, mmap_mode="r")
        if arr.ndim == 2 and arr.shape[1] >= _K_MAX:
            return np.ascontiguousarray(arr[:, :_K_MAX], dtype=np.float32)
    except Exception as e:
        print(f"  Warning: density knn_self cache load failed for {path.name}: {e}")
    return None


def _save_density_knnself(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    try:
        np.save(tmp, np.ascontiguousarray(arr, dtype=np.float32))
        os.replace(tmp, path)
        print(f"  [knn_self] Cached density-level self-kNN: {path}", flush=True)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _get_or_compute_knn_self(
    name: str,
    split: str,
    space: str,
    vec_dir: Path,
    vecs: np.ndarray,
    idx: np.ndarray | None,
    use_gpu: bool = True,
    batch_size: int | None = None,
) -> np.ndarray | None:
    """Return (N, _K_MAX) float32 L2 self-kNN distances, aligned with *vecs*.

    Priority:
      1. On-disk knn_self cache  — load, apply idx to align with subsampled vecs
      2. In-memory computed cache — already computed from the same subsampled vecs
      3. On-the-fly computation  — run FAISS self-kNN on *vecs*, memoize the result

    Because subsampling is deterministic (fixed seed=0), the same (name, split, space)
    always maps to the same vecs, so the memo is valid across all pairs.
    """
    # 1a. Density-level persistent cache for exactly this deterministic
    #     subsample size. This is what lets Slurm pair shards share work.
    density_path = _density_knnself_path(name, split, space, vec_dir, len(vecs))
    cached = _load_density_knnself(density_path)
    if cached is not None:
        return cached

    # 1b. Full extraction file cache. These canonical files store squared-L2;
    #     subset by idx and convert to L2.
    path = _knnself_path(name, split, space, vec_dir)
    if path is not None:
        try:
            if path.suffix == ".npz":
                data = np.load(path)
                arr = data[data.files[0]]
            else:
                arr = np.load(path, mmap_mode="r")
            if arr.shape[1] >= _K_MAX:
                if idx is not None:
                    arr = arr[idx]
                arr = np.sqrt(np.maximum(arr[:, :_K_MAX], 0.0))
                return np.ascontiguousarray(arr, dtype=np.float32)
        except Exception as e:
            print(f"  Warning: knn_self file load failed for {name}/{split}/{space}: {e}")

    # 2. In-memory cache (computed from subsampled vecs on a prior pair)
    mem_key = (name, split, space)
    if mem_key in _KNNSELF_COMPUTED:
        return _KNNSELF_COMPUTED[mem_key]  # None if prior computation failed

    # 3. Compute on the fly from the already-loaded (and possibly subsampled) vecs.
    #    For low-dimensional flow (4D) this is fast even at 1M vectors.
    cached = _load_density_knnself(density_path)
    if cached is not None:
        return cached

    n, dim = vecs.shape
    factory = "Flat" if n < 100_000 else f"IVF{min(1024, max(64, n // 100))},Flat"
    print(f"\n  [knn_self] Computing for {name}/{split}/{space} "
          f"({n:,} vecs, {dim}D, {factory})...", flush=True)
    try:
        dists = kl_mod.compute_self_knn_distances(
            vecs, k=_K_MAX,
            distance_metric="sq_l2",   # returns L2 (sqrt applied internally)
            use_gpu=use_gpu,
            index_factory=factory,
            batch_size=batch_size,
            verbose=True,
        )
        result: np.ndarray | None = np.ascontiguousarray(dists, dtype=np.float32)
        if result is not None and result.size:
            _save_density_knnself(density_path, result)
    except Exception as e:
        print(f"  [knn_self] Computation failed for {name}/{split}/{space}: {e}")
        result = None

    _KNNSELF_COMPUTED[mem_key] = result
    return result


# ---------------------------------------------------------------------------
# Pairwise metric computation (ONE search per direction)
# ---------------------------------------------------------------------------

def _index_factory_for(n_vecs: int, dim: int) -> str:
    if n_vecs < 50_000:
        return "Flat"
    nlist = min(1024, max(64, n_vecs // 100))
    return f"IVF{nlist},Flat"


def compute_pair_metrics(
    vecs_a: np.ndarray,
    vecs_b: np.ndarray,
    use_gpu: bool = True,
    nprobe: int = 64,
    knn_self_a: np.ndarray | None = None,
    knn_self_b: np.ndarray | None = None,
    batch_size: int | None = None,
) -> dict:
    """Single FAISS search per direction → NN distance + ε-coverage + KL for all thresholds.

    Searches with k=_K_MAX to extract k=1 NN distance (for coverage) and k=5/20
    (for KL) from the same result — no redundant searches.
    KL requires knn_self_a and knn_self_b (L2 self-distances, (N, _K_MAX) arrays).
    """
    dim = vecs_a.shape[1]

    def _search(index_vecs: np.ndarray, query_vecs: np.ndarray) -> np.ndarray:
        factory = _index_factory_for(len(index_vecs), dim)
        index = faiss_ops.build_index(
            index_vecs, use_gpu=use_gpu,
            index_factory=factory,
            nprobe=nprobe if "IVF" in factory else None,
            verbose=True,
        )
        try:
            dists, _ = faiss_ops.compute_knn_distances(
                index, query_vecs, k=_K_MAX, verbose=True, batch_size=batch_size)
        finally:
            faiss_ops.release_index(index)
        gc.collect()
        return dists  # (N, _K_MAX) squared L2

    a_to_b_sq = _search(vecs_b, vecs_a)
    b_to_a_sq = _search(vecs_a, vecs_b)

    # k=1 NN distances (squared L2) for coverage metrics
    a_to_b_k1 = a_to_b_sq[:, 0]
    b_to_a_k1 = b_to_a_sq[:, 0]

    mean_a_to_b = float(np.mean(a_to_b_k1))
    mean_b_to_a = float(np.mean(b_to_a_k1))
    row = {
        "mean_nn_a_to_b": mean_a_to_b,
        "mean_nn_b_to_a": mean_b_to_a,
        "mean_nn_sym":    (mean_a_to_b + mean_b_to_a) / 2.0,
    }

    for eps_px in EPS_PX:
        eps_sq = EPS_SQ[eps_px]
        lbl = _eps_label(eps_px)
        cov_a = float(np.mean(a_to_b_k1 <= eps_sq))
        cov_b = float(np.mean(b_to_a_k1 <= eps_sq))
        row[f"a_covered_by_b_{lbl}"] = cov_a
        row[f"b_covered_by_a_{lbl}"] = cov_b
        row[f"sym_{lbl}"]            = (cov_a + cov_b) / 2.0

    # KL divergence (kNN estimator, Wang et al. 2009)
    # Uses the same FAISS search result, no extra GPU work.
    if knn_self_a is not None and knn_self_b is not None:
        n_a_eff = min(len(knn_self_a), len(a_to_b_sq))
        n_b_eff = min(len(knn_self_b), len(b_to_a_sq))
        # Cross distances: sqrt squared L2 → L2 for the estimator
        a_to_b_l2 = np.sqrt(np.maximum(a_to_b_sq[:n_a_eff], 0.0))
        b_to_a_l2 = np.sqrt(np.maximum(b_to_a_sq[:n_b_eff], 0.0))

        kl_atob = kl_mod.compute_knn_kl_for_k_values(
            rho=knn_self_a[:n_a_eff], nu=a_to_b_l2,
            m=len(vecs_b), dim=dim, k_values=K_KL_VALUES, eps=1e-12,
        )
        kl_btoa = kl_mod.compute_knn_kl_for_k_values(
            rho=knn_self_b[:n_b_eff], nu=b_to_a_l2,
            m=len(vecs_a), dim=dim, k_values=K_KL_VALUES, eps=1e-12,
        )
        for k in K_KL_VALUES:
            row[f"kl_a_to_b_k{k}"] = kl_atob.get(k, float("nan"))
            row[f"kl_b_to_a_k{k}"] = kl_btoa.get(k, float("nan"))

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vec-dir",  default="/mnt/nvme_1tb_b/coverage_vectors")
    parser.add_argument("--output",   default="analysis_v3/pairwise_self_distances.csv")
    parser.add_argument("--max-flow", type=int, default=16_000_000,
                        help="Max flow vectors per dataset. Default 16M matches the "
                             "extraction cap (8000 flows/pair). Set higher to use all.")
    parser.add_argument("--max-dino", type=int, default=8_000_000,
                        help="Max DINO vectors per dataset. Default 8M matches the "
                             "extraction cap (1024 patches/image).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override GPU search batch size. Auto-selected by dim if not set "
                             "(500K for 4D flow, 50K for 256D DINO). "
                             "2000000 is a good push for 24GB: 8 batches for 16M flow vectors, "
                             "1 batch for 500K DINO vectors.")
    parser.add_argument("--gpu", action="store_true", default=True)
    parser.add_argument("--no-gpu", dest="gpu", action="store_false")
    parser.add_argument("--spaces", nargs="+", default=["flow", "dino"],
                        choices=["flow", "dino"])
    parser.add_argument("--pair-types", nargs="+",
                        default=["train_eval", "eval_eval"],
                        choices=["train_train", "eval_eval", "train_eval"],
                        help="Pair families to compute. Default is train_eval + "
                             "eval_eval, matching the final predictor: train_eval "
                             "feeds ranking features and eval_eval feeds benchmark "
                             "IDW/calibration. Add train_train for legacy/full runs.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Number of parallel workers. Worker K processes pairs where "
                             "global_index %% stride == rank.")
    parser.add_argument("--rank", type=int, default=0,
                        help="Index of this worker in [0, stride).")
    parser.add_argument("--seed-csv", default=None,
                        help="Read-only CSV of pairs already computed (e.g. from a prior local "
                             "run). These are added to the done set so they are skipped, but "
                             "this file is never written to.")
    args = parser.parse_args()

    if args.stride < 1 or args.rank < 0 or args.rank >= args.stride:
        raise ValueError(f"Invalid stride/rank: stride={args.stride}, rank={args.rank}")

    vec_dir  = Path(args.vec_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load already-completed pairs for resumption
    done: set[tuple] = set()
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            existing = pd.read_csv(out_path)
            for _, r in existing.iterrows():
                done.add((r["space"], r["pair_type"], r["dataset_a"], r["dataset_b"]))
            print(f"Resuming: {len(done)} pairs already in {out_path.name}")
        except Exception as e:
            print(f"Warning: could not read existing CSV ({e}), starting fresh")

    # Seed done set from a prior run on a different machine (read-only).
    if args.seed_csv:
        seed_path = Path(args.seed_csv)
        if seed_path.exists() and seed_path.stat().st_size > 0:
            try:
                seed_df = pd.read_csv(seed_path)
                before = len(done)
                for _, r in seed_df.iterrows():
                    done.add((r["space"], r["pair_type"], r["dataset_a"], r["dataset_b"]))
                print(f"Seeded {len(done) - before} additional pairs from {seed_path.name}")
            except Exception as e:
                print(f"Warning: could not read seed CSV ({e})")

    pair_types = set(args.pair_types)

    # Build pair lists
    #   self_pairs:  train-train and/or eval-eval (combinations + self-pairs)
    #   cross_pairs: train × eval (both KRR features and KL divergence)
    self_pairs: list[tuple] = []
    for datasets, ptype in [(TRAIN_DATASETS, "train_train"), (EVAL_DATASETS, "eval_eval")]:
        if ptype not in pair_types:
            continue
        for (na, sa), (nb, sb) in itertools.combinations(datasets, 2):
            self_pairs.append((ptype, na, sa, nb, sb))
        for (na, sa) in datasets:
            self_pairs.append((ptype, na, sa, na, sa))

    cross_pairs: list[tuple] = []
    if "train_eval" in pair_types:
        cross_pairs = [
            ("train_eval", na, sa, nb, sb)
            for (na, sa) in TRAIN_DATASETS
            for (nb, sb) in EVAL_DATASETS
        ]

    # Flatten to (space, ptype, na, sa, nb, sb) and apply stride/rank sharding.
    all_pairs: list[tuple] = [
        (space, ptype, na, sa, nb, sb)
        for space in args.spaces
        for (ptype, na, sa, nb, sb) in self_pairs + cross_pairs
    ]
    if args.stride > 1:
        all_pairs = [p for i, p in enumerate(all_pairs) if i % args.stride == args.rank]
        print(f"Rank {args.rank}/{args.stride}: {len(all_pairs)} pairs assigned "
              f"(of {(len(self_pairs) + len(cross_pairs)) * len(args.spaces)} total)")
    else:
        print(f"Total pairs: {len(all_pairs)}  "
              f"({len(self_pairs)} self + {len(cross_pairs)} cross) × {len(args.spaces)} spaces")
    print(f"Pair types: {', '.join(args.pair_types)}")

    n_computed = n_skipped = n_missing = 0
    current_space: str | None = None

    for space, ptype, na, sa, nb, sb in all_pairs:
        if space != current_space:
            current_space = space
            load_fn = load_flow_vectors if space == "flow" else load_dino_vectors
            n_max   = args.max_flow    if space == "flow" else args.max_dino
            print(f"\n{'='*60}")
            print(f"SPACE: {space.upper()}")
            print(f"{'='*60}")

        key = (space, ptype, na, nb)
        if key in done:
            n_skipped += 1
            continue

        self_pair = (na == nb and sa == sb)
        arrow = "↔" if ptype != "train_eval" else "→"
        print(f"\n  [{na}/{sa}] {arrow} [{nb}/{sb}] ({space})", end="", flush=True)

        vecs_a, idx_a = load_fn(na, sa, vec_dir, n_max)
        if vecs_a is None:
            print(f"  — missing vectors for {na}/{sa}, skipping")
            n_missing += 1
            continue

        if self_pair:
            row = {
                "space": space, "pair_type": ptype,
                "dataset_a": na, "split_a": sa,
                "dataset_b": nb, "split_b": sb,
                "n_vecs_a": len(vecs_a), "n_vecs_b": len(vecs_a),
                "mean_nn_a_to_b": 0.0, "mean_nn_b_to_a": 0.0, "mean_nn_sym": 0.0,
            }
            for eps_px in EPS_PX:
                lbl = _eps_label(eps_px)
                row[f"a_covered_by_b_{lbl}"] = 1.0
                row[f"b_covered_by_a_{lbl}"] = 1.0
                row[f"sym_{lbl}"]            = 1.0
            for k in K_KL_VALUES:
                row[f"kl_a_to_b_k{k}"] = 0.0
                row[f"kl_b_to_a_k{k}"] = 0.0
            del vecs_a
        else:
            vecs_b, idx_b = load_fn(nb, sb, vec_dir, n_max)
            if vecs_b is None:
                print(f"  — missing vectors for {nb}/{sb}, skipping")
                n_missing += 1
                del vecs_a
                continue

            knn_self_a = _get_or_compute_knn_self(na, sa, space, vec_dir, vecs_a, idx_a, args.gpu, args.batch_size)
            knn_self_b = _get_or_compute_knn_self(nb, sb, space, vec_dir, vecs_b, idx_b, args.gpu, args.batch_size)

            print(f"  ({len(vecs_a):,} × {len(vecs_b):,})", flush=True)
            metrics = compute_pair_metrics(
                vecs_a, vecs_b,
                use_gpu=args.gpu,
                knn_self_a=knn_self_a,
                knn_self_b=knn_self_b,
                batch_size=args.batch_size,
            )
            row = {
                "space": space, "pair_type": ptype,
                "dataset_a": na, "split_a": sa,
                "dataset_b": nb, "split_b": sb,
                "n_vecs_a": len(vecs_a), "n_vecs_b": len(vecs_b),
                **metrics,
            }
            del vecs_a, vecs_b

        gc.collect()

        _kl5 = row.get("kl_a_to_b_k5", float("nan"))
        print(f"  nn_sym={row['mean_nn_sym']:.6f}  kl_atob_k5={_kl5:.3f}")

        _flush_row(row, out_path)
        done.add(key)
        n_computed += 1

    print(f"\nDone. Computed={n_computed}, Skipped={n_skipped}, Missing={n_missing}")
    print(f"Output: {out_path}")

    if args.stride > 1:
        print("Skipping symmetrize (stride > 1). Run merge_pairwise_distances.py after all ranks finish.")
    else:
        # Symmetrize only train-train and eval-eval pairs (not cross pairs).
        _symmetrize(out_path)


def _flush_row(row: dict, out_path: Path) -> None:
    """Append a single row to CSV; writes header if file is empty/new."""
    df = pd.DataFrame([row])
    df.to_csv(out_path, mode="a",
              header=not out_path.exists() or out_path.stat().st_size == 0,
              index=False)


def _symmetrize(out_path: Path) -> None:
    """Add reversed rows for non-self train-train and eval-eval pairs."""
    if not out_path.exists():
        return
    df = pd.read_csv(out_path)

    # Only symmetrize self pairs (train-train and eval-eval), not cross pairs.
    sym_mask = (df["pair_type"] != "train_eval") & (df["dataset_a"] != df["dataset_b"])
    non_self = df[sym_mask].copy()
    if non_self.empty:
        return

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
    if not rev.empty:
        combined = pd.concat([df, rev], ignore_index=True)
        combined.to_csv(out_path, index=False)
        print(f"Symmetrized: added {len(rev)} reversed rows → {len(combined)} total rows")


if __name__ == "__main__":
    main()
