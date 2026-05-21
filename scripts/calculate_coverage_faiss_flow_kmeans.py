#!/usr/bin/env python3
"""
Coverage Pipeline v2.2 - Flow K-Means Epsilon Curves

Builds a k-means codebook per dataset from cached flow vectors,
then computes epsilon coverage curves between codebooks (weighted by counts).
"""

import argparse
import gc
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Dataset utilities
from calculate_coverage_faiss import (
    create_dataset_from_config,
    create_mixed_dataset_from_config,
    _is_synthetic_dataset,
)

# Vector extraction (reused from v2 pipeline)
from calculate_coverage_faiss_v2 import extract_vectors_from_dataset

# Modular coverage utilities
from coverage import cache, spaces, faiss_ops

try:
    import faiss
except ImportError as exc:
    raise SystemExit("faiss is required. Install faiss-cpu or faiss-gpu.") from exc


def _format_eps_label(eps_px: float) -> str:
    if float(eps_px).is_integer():
        return str(int(eps_px))
    return f"{eps_px:g}".replace(".", "p")


def _convert_epsilons(
    eps_px: List[float],
    img_w: int,
    img_h: int,
    use_normalized: bool,
) -> List[Dict[str, float]]:
    eps_info = []
    if not use_normalized:
        for e in eps_px:
            eps_info.append(
                {
                    "eps_px": float(e),
                    "eps_norm": float(e),
                    "eps_sq": float(e * e),
                }
            )
        return eps_info

    scale_x = 2.0 / float(img_w)
    scale_y = 2.0 / float(img_h)
    if img_w != img_h:
        print(
            f"  ⚠️  Non-square flow normalization ({img_w}x{img_h}). "
            "Using geometric-mean scaling for epsilon conversion."
        )
    scale = scale_x if img_w == img_h else float((scale_x * scale_y) ** 0.5)

    for e in eps_px:
        eps_norm = float(e) * scale
        eps_info.append(
            {
                "eps_px": float(e),
                "eps_norm": eps_norm,
                "eps_sq": float(eps_norm * eps_norm),
            }
        )
    return eps_info


def _distance_stats(dists: np.ndarray) -> Dict[str, float]:
    if dists.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
    return {
        "mean": float(np.mean(dists)),
        "median": float(np.median(dists)),
        "p90": float(np.quantile(dists, 0.90)),
        "p95": float(np.quantile(dists, 0.95)),
    }


def _kmeans_cache_path(
    cache_dir: Path,
    dataset: str,
    split: str,
    space: str,
    k: int,
) -> Path:
    safe_ds = cache.sanitize_name(dataset)
    safe_split = cache.sanitize_name(split)
    safe_space = cache.sanitize_name(space)
    return cache_dir / "kmeans" / f"kmeans_{safe_ds}_{safe_split}_{safe_space}_k{k}.npz"


def _load_kmeans(
    cache_dir: Path,
    dataset: str,
    split: str,
    space: str,
    k: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    path = _kmeans_cache_path(cache_dir, dataset, split, space, k)
    if not path.exists():
        return None
    data = np.load(path)
    return data["centroids"], data["weights"]


def _save_kmeans(
    cache_dir: Path,
    dataset: str,
    split: str,
    space: str,
    k: int,
    centroids: np.ndarray,
    weights: np.ndarray,
) -> Path:
    path = _kmeans_cache_path(cache_dir, dataset, split, space, k)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, centroids=centroids, weights=weights)
    return path


def _sample_vectors(
    vectors: np.ndarray,
    max_train: Optional[int],
    seed: int,
) -> np.ndarray:
    if max_train is None or vectors.shape[0] <= max_train:
        return vectors
    rng = np.random.default_rng(seed)
    indices = rng.choice(vectors.shape[0], size=max_train, replace=False)
    return vectors[indices]


def _train_kmeans(
    vectors: np.ndarray,
    k: int,
    niter: int,
    nredo: int,
    seed: int,
    use_gpu: bool,
    verbose: bool,
) -> np.ndarray:
    dim = vectors.shape[1]
    if vectors.shape[0] < k:
        raise ValueError(f"Not enough vectors for k-means: N={vectors.shape[0]:,}, k={k}")
    kmeans = faiss.Kmeans(
        dim,
        k,
        niter=niter,
        nredo=nredo,
        verbose=verbose,
        gpu=use_gpu,
        seed=seed,
    )
    kmeans.train(vectors)
    return kmeans.centroids


def _assign_weights(
    centroids: np.ndarray,
    vectors: np.ndarray,
    batch_size: int,
    use_gpu: bool,
    verbose: bool,
) -> np.ndarray:
    dim = centroids.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.ascontiguousarray(centroids, dtype=np.float32))
    if use_gpu:
        try:
            gpu_resources = faiss.StandardGpuResources()
            gpu_resources.setTempMemory(2 * 1024 * 1024 * 1024)
            index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
            if verbose:
                print("  Using GPU for centroid assignment")
        except Exception as exc:
            if verbose:
                print(f"  ⚠️  GPU assignment failed ({exc}); falling back to CPU")

    total = vectors.shape[0]
    counts = np.zeros(centroids.shape[0], dtype=np.int64)
    if verbose:
        print(f"  Assigning {total:,} vectors to {centroids.shape[0]:,} centroids...")
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = np.ascontiguousarray(vectors[start:end], dtype=np.float32)
        _, idx = index.search(batch, 1)
        counts += np.bincount(idx.ravel(), minlength=centroids.shape[0])
    return counts


def _weighted_coverage(distances: np.ndarray, weights: np.ndarray, eps_sq: float) -> float:
    if distances.size == 0:
        return float("nan")
    total = float(np.sum(weights))
    if total <= 0:
        return float("nan")
    covered = float(np.sum(weights[distances <= eps_sq]))
    return covered / total


def _load_or_extract_vectors(
    config: Dict,
    cache_dir: Path,
    device: str,
) -> Tuple[Dict[Tuple[str, str], np.ndarray], Dict[Tuple[str, str], np.ndarray]]:
    train_vectors = {}
    eval_vectors = {}

    require_cache = bool(config.get("kmeans", {}).get("require_cache", True))

    for ds_config in config["datasets"]:
        is_eval = ds_config.get("is_eval", False)
        dataset_name = ds_config.get("name")
        split = ds_config.get("split")

        print(f"\n[{dataset_name}/{split}] {'(eval)' if is_eval else '(train)'}")

        vectors = cache.load_cached_vectors(cache_dir, dataset_name, split, "flow")
        dataset = None
        dataloader = None

        if vectors is None:
            if require_cache:
                raise FileNotFoundError(
                    f"Missing cached vectors for {dataset_name}/{split} (flow). "
                    "Re-run extraction or set kmeans.require_cache=false."
                )

            if ds_config.get("mixed", False):
                print(
                    "  Mixed dataset: "
                    + " + ".join(
                        [f"{d}({p:.0%})" for d, p in zip(ds_config["datasets"], ds_config["percentages"])]
                    )
                )
                dataset = create_mixed_dataset_from_config(
                    ds_config["datasets"],
                    ds_config["percentages"],
                    split,
                    config["dataset_params"],
                    config["dataset_overrides"],
                    seed=config["sampling"]["seed"],
                )
                is_synthetic = any(_is_synthetic_dataset(name) for name in ds_config["datasets"])
            else:
                dataset = create_dataset_from_config(
                    dataset_name,
                    split,
                    config["dataset_params"],
                    config["dataset_overrides"],
                    entry_overrides=ds_config.get("overrides"),
                )
                is_synthetic = _is_synthetic_dataset(dataset_name)

            num_workers = 0 if is_synthetic else config["num_workers"]
            pin_memory = False if is_synthetic else True
            if is_synthetic and config["num_workers"] > 0:
                print("  ⚠️  Synthetic dataset detected - forcing num_workers=0 and pin_memory=False")

            dataloader = DataLoader(
                dataset,
                batch_size=config["batch_size"],
                shuffle=config["sampling"].get("shuffle", True),
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
            )

            extract_kwargs = {
                "max_vectors": config["sampling"]["max_vectors"],
                "vectors_per_image": config["sampling"].get(
                    "vectors_per_image", config["sampling"].get("flow_per_image_max", 2000)
                ),
                "seed": config["sampling"]["seed"],
                "device": device,
                "verbose": True,
            }
            img_size = config.get("flow_normalization", {}).get("image_size", [512, 512])
            extract_kwargs["image_size"] = img_size

            vectors = extract_vectors_from_dataset(
                dataset,
                dataloader,
                "flow",
                encoder=None,
                **extract_kwargs,
            )

            cache.save_cached_vectors(cache_dir, dataset_name, split, "flow", vectors)
        else:
            print(f"  ✓ Loaded {len(vectors):,} cached vectors")

        key = (dataset_name, split)
        if is_eval:
            eval_vectors[key] = vectors
        else:
            train_vectors[key] = vectors

        if dataset is not None:
            del dataset
        if dataloader is not None:
            del dataloader
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    return train_vectors, eval_vectors


def _load_or_extract_vectors_for_dataset(
    config: Dict,
    cache_dir: Path,
    device: str,
    ds_config: Dict,
) -> np.ndarray:
    """Load/extract one dataset's flow vectors, then let caller free it.

    The old k-means path loaded every dataset into memory before training
    codebooks. For large flow caches that is unnecessary: codebook training
    needs only one dataset at a time, and the pairwise stage needs only the
    small centroids/weights.
    """
    dataset_name = ds_config.get("name")
    split = ds_config.get("split")
    require_cache = bool(config.get("kmeans", {}).get("require_cache", True))

    vectors = cache.load_cached_vectors(
        cache_dir,
        dataset_name,
        split,
        "flow",
        mmap=bool(config.get("cache", {}).get("mmap", False)),
    )
    if vectors is not None:
        print(f"  ✓ Loaded {len(vectors):,} cached vectors")
        return vectors

    if require_cache:
        raise FileNotFoundError(
            f"Missing cached vectors for {dataset_name}/{split} (flow). "
            "Re-run extraction or set kmeans.require_cache=false."
        )

    dataset = None
    dataloader = None
    try:
        if ds_config.get("mixed", False):
            print(
                "  Mixed dataset: "
                + " + ".join(
                    [f"{d}({p:.0%})" for d, p in zip(ds_config["datasets"], ds_config["percentages"])]
                )
            )
            dataset = create_mixed_dataset_from_config(
                ds_config["datasets"],
                ds_config["percentages"],
                split,
                config["dataset_params"],
                config["dataset_overrides"],
                seed=config["sampling"]["seed"],
            )
            is_synthetic = any(_is_synthetic_dataset(name) for name in ds_config["datasets"])
        else:
            dataset = create_dataset_from_config(
                dataset_name,
                split,
                config["dataset_params"],
                config["dataset_overrides"],
                entry_overrides=ds_config.get("overrides"),
            )
            is_synthetic = _is_synthetic_dataset(dataset_name)

        num_workers = 0 if is_synthetic else config["num_workers"]
        pin_memory = False if is_synthetic else True
        if is_synthetic and config["num_workers"] > 0:
            print("  ⚠️  Synthetic dataset detected - forcing num_workers=0 and pin_memory=False")

        dataloader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=config["sampling"].get("shuffle", True),
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
        )

        img_size = config.get("flow_normalization", {}).get("image_size", [512, 512])
        vectors = extract_vectors_from_dataset(
            dataset,
            dataloader,
            "flow",
            encoder=None,
            max_vectors=config["sampling"]["max_vectors"],
            vectors_per_image=config["sampling"].get(
                "vectors_per_image", config["sampling"].get("flow_per_image_max", 2000)
            ),
            seed=config["sampling"]["seed"],
            device=device,
            verbose=True,
            image_size=img_size,
        )
        cache.save_cached_vectors(cache_dir, dataset_name, split, "flow", vectors)
        return vectors
    finally:
        if dataset is not None:
            del dataset
        if dataloader is not None:
            del dataloader
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


def run_pipeline(config_path: str):
    print(f"\n{'=' * 80}")
    print("COVERAGE PIPELINE v2.2 - FLOW K-MEANS EPSILON CURVES")
    print(f"{'=' * 80}\n")
    print(f"Config: {config_path}\n")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    representation = config["representation"]
    if representation != "flow":
        raise ValueError("This pipeline only supports representation: flow")

    cache_dir = Path(config["cache"]["dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() and config["faiss"]["use_gpu"] else "cpu"

    train_keys = [
        (d.get("name"), d.get("split"))
        for d in config["datasets"]
        if not d.get("is_eval", False)
    ]
    eval_keys = [
        (d.get("name"), d.get("split"))
        for d in config["datasets"]
        if d.get("is_eval", False)
    ]
    print(f"\nConfigured {len(train_keys)} train sets, {len(eval_keys)} eval sets")

    flow_norm = config.get("flow_normalization", {}).get("enabled", False)
    img_h, img_w = config.get("flow_normalization", {}).get("image_size", [512, 512])

    # ======================
    # STEP 1: Define Space
    # ======================
    space_name = config.get("flow_space", "flow")
    if space_name not in ("flow", "xy", "joint"):
        raise ValueError(f"Unsupported flow_space: {space_name}")
    joint_alpha = float(config.get("joint_alpha", 1.0))

    print(f"\n{'=' * 80}")
    if space_name == "flow":
        print("STEP 1: DEFINE FLOW SPACE (dx, dy)")
    elif space_name == "xy":
        print("STEP 1: DEFINE XY SPACE (x, y)")
    else:
        print(f"STEP 1: DEFINE JOINT SPACE (x, y, {joint_alpha:.3f}*dx, {joint_alpha:.3f}*dy)")
    print(f"{'=' * 80}\n")

    # ======================
    # STEP 2: K-Means Codebooks
    # ======================
    print(f"\n{'=' * 80}")
    print("STEP 2: K-MEANS CODEBOOKS")
    print(f"{'=' * 80}\n")

    kmeans_cfg = config.get("kmeans", {})
    kmeans_k = int(kmeans_cfg.get("k", 4096))
    kmeans_seed = int(kmeans_cfg.get("seed", 42))
    kmeans_niter = int(kmeans_cfg.get("niter", 20))
    kmeans_nredo = int(kmeans_cfg.get("nredo", 1))
    kmeans_train_max = kmeans_cfg.get("train_max", None)
    if kmeans_train_max is not None:
        kmeans_train_max = int(kmeans_train_max)
    kmeans_use_gpu = bool(kmeans_cfg.get("use_gpu", True))
    kmeans_force = bool(kmeans_cfg.get("force_recompute", False))
    kmeans_batch_size = int(kmeans_cfg.get("batch_size", 500000))

    codebooks: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}

    for ds_config in config["datasets"]:
        dataset_name = ds_config.get("name")
        split = ds_config.get("split")
        key = (dataset_name, split)
        print(f"\n[{dataset_name}/{split}] {'(eval)' if ds_config.get('is_eval', False) else '(train)'}")

        vectors = _load_or_extract_vectors_for_dataset(config, cache_dir, device, ds_config)
        if flow_norm:
            vectors = spaces.normalize_flow_vectors(vectors, img_w, img_h)
        vectors = spaces.transform_to_space(vectors, space_name, alpha=joint_alpha)

        n_vectors = vectors.shape[0]
        k_eff = min(kmeans_k, n_vectors)
        if k_eff < kmeans_k:
            print(f"  ⚠️  {dataset_name}/{split}: reducing k from {kmeans_k} to {k_eff} (N={n_vectors:,})")

        cached = None if kmeans_force else _load_kmeans(cache_dir, dataset_name, split, space_name, k_eff)
        if cached is not None:
            centroids, weights = cached
            print(f"  ✓ Loaded cached k-means for {dataset_name}/{split} (k={k_eff})")
        else:
            print(f"  Training k-means for {dataset_name}/{split} (k={k_eff})")
            kmeans_train_vectors = _sample_vectors(vectors, kmeans_train_max, kmeans_seed)
            kmeans_train_vectors = np.ascontiguousarray(kmeans_train_vectors, dtype=np.float32)
            if kmeans_train_vectors.shape[0] < k_eff:
                raise ValueError(
                    f"Not enough training vectors for k-means: N={kmeans_train_vectors.shape[0]:,}, k={k_eff}. "
                    "Increase kmeans.train_max or reduce kmeans.k."
                )
            centroids = _train_kmeans(
                kmeans_train_vectors,
                k_eff,
                niter=kmeans_niter,
                nredo=kmeans_nredo,
                seed=kmeans_seed,
                use_gpu=kmeans_use_gpu,
                verbose=True,
            )
            centroids = np.ascontiguousarray(centroids, dtype=np.float32)

            weights = _assign_weights(
                centroids,
                vectors,
                batch_size=kmeans_batch_size,
                use_gpu=kmeans_use_gpu,
                verbose=True,
            )
            _save_kmeans(cache_dir, dataset_name, split, space_name, k_eff, centroids, weights)
            print(f"  ✓ Saved k-means for {dataset_name}/{split}")
            del kmeans_train_vectors

        codebooks[key] = {"centroids": centroids, "weights": weights, "k": k_eff}
        del vectors
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # ======================
    # STEP 3: Epsilon Curves
    # ======================
    print(f"\n{'=' * 80}")
    print("STEP 3: EPSILON CURVES")
    print(f"{'=' * 80}\n")

    eps_cfg = config.get("epsilon_curves", {})
    eps_values_px = eps_cfg.get("values_px", [1, 2, 4, 8, 16, 32, 64])
    eps_info = _convert_epsilons(eps_values_px, img_w, img_h, use_normalized=flow_norm)

    print("Epsilon thresholds:")
    for info in eps_info:
        label = _format_eps_label(info["eps_px"])
        if flow_norm:
            print(f"  {label}px → {info['eps_norm']:.6f} (squared={info['eps_sq']:.6f})")
        else:
            print(f"  {label}px (squared={info['eps_sq']:.6f})")

    output_file = Path(config["output"]["kmeans_results_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    curves_file = Path(config["output"]["kmeans_curves_file"])
    curves_file.parent.mkdir(parents=True, exist_ok=True)

    done_pairs: set[tuple[str, str, str, str]] = set()
    if output_file.exists() and output_file.stat().st_size > 0:
        try:
            existing = pd.read_csv(output_file)
            for _, r in existing.iterrows():
                done_pairs.add(
                    (
                        str(r["train_dataset"]),
                        str(r["train_split"]),
                        str(r["eval_dataset"]),
                        str(r["eval_split"]),
                    )
                )
            print(f"  Resuming: {len(done_pairs)} k-means pairs already in {output_file.name}")
        except Exception as exc:
            print(f"  ⚠️  Could not read existing k-means results ({exc}); appending from scratch.")

    computed_pairs = 0
    skipped_pairs = 0

    for train_key in train_keys:
        for eval_key in eval_keys:
            train_name, train_split = train_key
            eval_name, eval_split = eval_key
            pair_key = (train_name, train_split, eval_name, eval_split)

            if pair_key in done_pairs:
                skipped_pairs += 1
                print(f"\n[{train_name}/{train_split} → {eval_name}/{eval_split}] already done — skipping")
                continue

            print(f"\n[{train_name}/{train_split} → {eval_name}/{eval_split}]")

            train_entry = codebooks[train_key]
            eval_entry = codebooks[eval_key]
            train_centroids = train_entry["centroids"]
            train_weights = train_entry["weights"]
            eval_centroids = eval_entry["centroids"]
            eval_weights = eval_entry["weights"]

            directed = faiss_ops.compute_directed_distances(
                train_centroids,
                eval_centroids,
                k=1,
                use_gpu=config["faiss"]["use_gpu"],
                index_factory=config["faiss"]["index_factory"],
                batch_size=config["faiss"].get("batch_size"),
                verbose=True,
            )

            eval_to_train = directed["eval_to_train"][:, 0]
            train_to_eval = directed["train_to_eval"][:, 0]

            row = {
                "space": space_name,
                "train_dataset": train_name,
                "train_split": train_split,
                "eval_dataset": eval_name,
                "eval_split": eval_split,
                "train_n_centroids": int(len(train_centroids)),
                "eval_n_centroids": int(len(eval_centroids)),
                "train_total_weight": int(np.sum(train_weights)),
                "eval_total_weight": int(np.sum(eval_weights)),
                "distance_metric": config["distance_metric"]["name"],
                "flow_normalized": bool(flow_norm),
                "train_kmeans_k": int(train_entry["k"]),
                "eval_kmeans_k": int(eval_entry["k"]),
            }

            eval_stats = _distance_stats(eval_to_train)
            train_stats = _distance_stats(train_to_eval)
            curve_rows = []

            row.update(
                {
                    "mean_nn_eval_to_train_k1": eval_stats["mean"],
                    "median_nn_eval_to_train_k1": eval_stats["median"],
                    "p90_nn_eval_to_train_k1": eval_stats["p90"],
                    "p95_nn_eval_to_train_k1": eval_stats["p95"],
                    "mean_nn_train_to_eval_k1": train_stats["mean"],
                    "median_nn_train_to_eval_k1": train_stats["median"],
                    "p90_nn_train_to_eval_k1": train_stats["p90"],
                    "p95_nn_train_to_eval_k1": train_stats["p95"],
                }
            )

            for info in eps_info:
                eps_px = info["eps_px"]
                eps_label = _format_eps_label(eps_px)
                eps_sq = info["eps_sq"]

                eval_cov = _weighted_coverage(eval_to_train, eval_weights, eps_sq)
                train_cov = _weighted_coverage(train_to_eval, train_weights, eps_sq)

                col_eval = f"eval_covered_by_train_eps{eps_label}px_weighted"
                col_train = f"train_covered_by_eval_eps{eps_label}px_weighted"

                row[col_eval] = eval_cov
                row[col_train] = train_cov

                curve_rows.append(
                    {
                        "space": space_name,
                        "train_dataset": train_name,
                        "train_split": train_split,
                        "eval_dataset": eval_name,
                        "eval_split": eval_split,
                        "epsilon_px": float(info["eps_px"]),
                        "epsilon_norm": float(info["eps_norm"]) if flow_norm else float("nan"),
                        "epsilon_sq": float(eps_sq),
                        "eval_covered_by_train_weighted": eval_cov,
                        "train_covered_by_eval_weighted": train_cov,
                        "train_kmeans_k": int(train_entry["k"]),
                        "eval_kmeans_k": int(eval_entry["k"]),
                    }
                )

            for info in eps_info:
                eps_label = _format_eps_label(info["eps_px"])
                col_eval = f"eval_covered_by_train_eps{eps_label}px_weighted"
                col_train = f"train_covered_by_eval_eps{eps_label}px_weighted"
                print(
                    f"  eps={eps_label}px: "
                    f"eval_covered={row[col_eval]:.3f}, "
                    f"train_covered={row[col_train]:.3f}"
                )

            write_header = not output_file.exists() or output_file.stat().st_size == 0
            pd.DataFrame([row]).to_csv(output_file, mode="a", header=write_header, index=False)
            curves_header = not curves_file.exists() or curves_file.stat().st_size == 0
            pd.DataFrame(curve_rows).to_csv(curves_file, mode="a", header=curves_header, index=False)
            done_pairs.add(pair_key)
            computed_pairs += 1
            print(f"  ✓ Checkpoint saved ({computed_pairs} new, {skipped_pairs} skipped)")

            del directed, eval_to_train, train_to_eval
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    print(f"✓ Results saved incrementally to: {output_file}")
    print(f"✓ Curves saved incrementally to: {curves_file}")
    if output_file.exists():
        print(f"  Total result rows: {len(pd.read_csv(output_file))}")
    if curves_file.exists():
        print(f"  Total curve rows: {len(pd.read_csv(curves_file))}")
    print(f"\n{'=' * 80}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description="Coverage Pipeline v2.2 (Flow K-Means)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    run_pipeline(args.config)


if __name__ == "__main__":
    main()
