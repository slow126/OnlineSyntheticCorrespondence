#!/usr/bin/env python3
"""
Coverage Pipeline v2.2 - Flow K-Means Manifold Distances

Builds a k-means codebook per dataset from cached flow vectors,
then computes directed NN distances between codebooks and normalizes
by per-dataset self-radius computed on centroids (weighted by counts).
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from coverage import cache, faiss_ops, spaces  # noqa: E402
from calculate_coverage_faiss_flow_kmeans import (  # noqa: E402
    _assign_weights,
    _load_kmeans,
    _load_or_extract_vectors,
    _sample_vectors,
    _save_kmeans,
    _train_kmeans,
)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return float("nan")
    mask = np.isfinite(values) & np.isfinite(weights)
    values = values[mask]
    weights = weights[mask]
    total = float(np.sum(weights))
    if total <= 0 or values.size == 0:
        return float("nan")
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cum = np.cumsum(weights)
    cutoff = float(quantile) * total
    idx = int(np.searchsorted(cum, cutoff, side="left"))
    idx = min(max(idx, 0), values.size - 1)
    return float(values[idx])


def _weighted_stats(distances: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
    if distances.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
    values = np.asarray(distances, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights)
    values = values[mask]
    weights = weights[mask]
    total = float(np.sum(weights))
    if total <= 0 or values.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
    mean = float(np.average(values, weights=weights))
    return {
        "mean": mean,
        "median": _weighted_quantile(values, weights, 0.50),
        "p90": _weighted_quantile(values, weights, 0.90),
        "p95": _weighted_quantile(values, weights, 0.95),
    }


def _kmeans_radius_cache_path(
    cache_dir: Path,
    dataset: str,
    split: str,
    space: str,
    k: int,
    self_k: int,
    quantile: float,
    neighbor_agg: str,
) -> Path:
    safe_ds = cache.sanitize_name(dataset)
    safe_split = cache.sanitize_name(split)
    safe_space = cache.sanitize_name(space)
    q_str = f"{quantile:.4f}".replace(".", "p")
    safe_agg = cache.sanitize_name(neighbor_agg)
    return (
        cache_dir
        / "kmeans_radius"
        / f"kmeans_radius_{safe_ds}_{safe_split}_{safe_space}_k{k}_self{self_k}_q{q_str}_{safe_agg}.npz"
    )


def _load_kmeans_radius(
    cache_dir: Path,
    dataset: str,
    split: str,
    space: str,
    k: int,
    self_k: int,
    quantile: float,
    neighbor_agg: str,
) -> Optional[Dict[str, float]]:
    path = _kmeans_radius_cache_path(
        cache_dir, dataset, split, space, k, self_k, quantile, neighbor_agg
    )
    if not path.exists():
        return None
    data = np.load(path)
    return {key: float(data[key]) for key in data.files}


def _save_kmeans_radius(
    cache_dir: Path,
    dataset: str,
    split: str,
    space: str,
    k: int,
    self_k: int,
    quantile: float,
    neighbor_agg: str,
    radius_data: Dict[str, float],
) -> Path:
    path = _kmeans_radius_cache_path(
        cache_dir, dataset, split, space, k, self_k, quantile, neighbor_agg
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.array(v) for k, v in radius_data.items()})
    return path


def _compute_weighted_self_radius(
    vectors: np.ndarray,
    weights: np.ndarray,
    k: int,
    radius_quantile: float,
    neighbor_agg: str,
    use_gpu: bool,
    index_factory: str,
    nprobe: Optional[int],
    batch_size: Optional[int],
    verbose: bool,
) -> Dict[str, float]:
    if vectors.shape[0] < 2:
        return {
            "radius": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "mean": float("nan"),
        }

    if verbose:
        print(
            f"  Computing weighted self-radius: {vectors.shape[0]:,} centroids, k={k}, q={radius_quantile:.2f}"
        )

    index = faiss_ops.build_index(
        vectors, use_gpu=use_gpu, index_factory=index_factory, nprobe=nprobe, verbose=verbose
    )
    distances, _ = faiss_ops.compute_knn_distances(
        index, vectors, k=k, exclude_self=True, batch_size=batch_size, verbose=verbose
    )
    if distances.size == 0:
        return {
            "radius": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "mean": float("nan"),
        }

    if neighbor_agg in ("first", "min"):
        sample = distances[:, 0]
    elif neighbor_agg in ("kth", "last", "max"):
        sample = distances[:, -1]
    elif neighbor_agg == "mean":
        sample = distances.mean(axis=1)
    elif neighbor_agg == "median":
        sample = np.median(distances, axis=1)
    else:
        raise ValueError(f"Unsupported neighbor_agg: {neighbor_agg}")

    stats = _weighted_stats(sample, weights)
    stats["radius"] = _weighted_quantile(sample, weights, radius_quantile)
    return stats


def run_pipeline(config_path: str):
    print(f"\n{'=' * 80}")
    print("COVERAGE PIPELINE v2.2 - FLOW K-MEANS MANIFOLD DISTANCES")
    print(f"{'=' * 80}\n")
    print(f"Config: {config_path}\n")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config.get("representation") != "flow":
        raise ValueError("This pipeline only supports representation: flow")

    cache_dir = Path(config["cache"]["dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() and config["faiss"]["use_gpu"] else "cpu"

    print(f"\n{'=' * 80}")
    print("STEP 0: LOAD/EXTRACT VECTORS")
    print(f"{'=' * 80}\n")

    train_vectors, eval_vectors = _load_or_extract_vectors(config, cache_dir, device)
    print(f"\nLoaded {len(train_vectors)} train sets, {len(eval_vectors)} eval sets")

    flow_norm = config.get("flow_normalization", {}).get("enabled", False)
    if flow_norm:
        print("\nNormalizing flow vectors to [-1, 1]...")
        img_h, img_w = config["flow_normalization"]["image_size"]
        for key, vectors in train_vectors.items():
            train_vectors[key] = spaces.normalize_flow_vectors(vectors, img_w, img_h)
        for key, vectors in eval_vectors.items():
            eval_vectors[key] = spaces.normalize_flow_vectors(vectors, img_w, img_h)
    else:
        img_h, img_w = config.get("flow_normalization", {}).get("image_size", [512, 512])

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

    space_train_vectors = {
        k: spaces.transform_to_space(v, space_name, alpha=joint_alpha) for k, v in train_vectors.items()
    }
    space_eval_vectors = {
        k: spaces.transform_to_space(v, space_name, alpha=joint_alpha) for k, v in eval_vectors.items()
    }

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
    all_vectors = {**space_train_vectors, **space_eval_vectors}
    for (dataset_name, split), vectors in all_vectors.items():
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
            train_subset = _sample_vectors(vectors, kmeans_train_max, kmeans_seed)
            train_subset = np.ascontiguousarray(train_subset, dtype=np.float32)
            if train_subset.shape[0] < k_eff:
                raise ValueError(
                    f"Not enough training vectors for k-means: N={train_subset.shape[0]:,}, k={k_eff}. "
                    "Increase kmeans.train_max or reduce kmeans.k."
                )
            centroids = _train_kmeans(
                train_subset,
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

        codebooks[(dataset_name, split)] = {"centroids": centroids, "weights": weights, "k": k_eff}

    print(f"\n{'=' * 80}")
    print("STEP 3: K-MEANS MANIFOLD DISTANCES")
    print(f"{'=' * 80}\n")

    manifold_cfg = config.get("kmeans_manifold", {})
    self_radius_k = int(manifold_cfg.get("self_radius_k", 1))
    radius_quantile = float(manifold_cfg.get("radius_quantile", 0.95))
    neighbor_agg = str(manifold_cfg.get("neighbor_agg", "kth"))
    use_weighted_radius = bool(manifold_cfg.get("weighted_radius", True))

    faiss_cfg = config.get("faiss", {})
    index_factory = str(faiss_cfg.get("index_factory", "Flat"))
    nprobe = faiss_cfg.get("nprobe", None)
    batch_size = faiss_cfg.get("batch_size", None)
    use_gpu = bool(faiss_cfg.get("use_gpu", True))

    radii: Dict[Tuple[str, str], Dict[str, float]] = {}
    for (dataset_name, split), entry in codebooks.items():
        k_eff = int(entry["k"])
        cached = _load_kmeans_radius(
            cache_dir, dataset_name, split, space_name, k_eff, self_radius_k, radius_quantile, neighbor_agg
        )
        if cached is not None:
            radii[(dataset_name, split)] = cached
            print(f"  ✓ Loaded cached radius for {dataset_name}/{split}")
            continue

        if use_weighted_radius:
            radius_data = _compute_weighted_self_radius(
                entry["centroids"],
                entry["weights"],
                k=self_radius_k,
                radius_quantile=radius_quantile,
                neighbor_agg=neighbor_agg,
                use_gpu=use_gpu,
                index_factory=index_factory,
                nprobe=nprobe,
                batch_size=batch_size,
                verbose=True,
            )
        else:
            radius_data = faiss_ops.compute_self_radius(
                entry["centroids"],
                k=self_radius_k,
                radius_quantile=radius_quantile,
                neighbor_agg=neighbor_agg,
                use_gpu=use_gpu,
                index_factory=index_factory,
                nprobe=nprobe,
                batch_size=batch_size,
                verbose=True,
            )
        _save_kmeans_radius(
            cache_dir,
            dataset_name,
            split,
            space_name,
            k_eff,
            self_radius_k,
            radius_quantile,
            neighbor_agg,
            radius_data,
        )
        radii[(dataset_name, split)] = radius_data

    results = []

    for train_key in space_train_vectors.keys():
        for eval_key in space_eval_vectors.keys():
            train_name, train_split = train_key
            eval_name, eval_split = eval_key

            print(f"\n[{train_name}/{train_split} → {eval_name}/{eval_split}]")

            train_entry = codebooks[train_key]
            eval_entry = codebooks[eval_key]
            train_centroids = train_entry["centroids"]
            eval_centroids = eval_entry["centroids"]
            train_weights = train_entry["weights"]
            eval_weights = eval_entry["weights"]

            directed = faiss_ops.compute_directed_distances(
                train_centroids,
                eval_centroids,
                k=1,
                use_gpu=use_gpu,
                index_factory=index_factory,
                batch_size=batch_size,
                verbose=True,
            )

            eval_to_train = directed["eval_to_train"][:, 0]
            train_to_eval = directed["train_to_eval"][:, 0]

            eval_stats = _weighted_stats(eval_to_train, eval_weights)
            train_stats = _weighted_stats(train_to_eval, train_weights)

            radius_train = radii[train_key]["radius"]
            radius_eval = radii[eval_key]["radius"]

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
                "self_radius_k": self_radius_k,
                "radius_quantile": radius_quantile,
                "radius_train": float(radius_train),
                "radius_eval": float(radius_eval),
                "mean_nn_eval_to_train": eval_stats["mean"],
                "median_nn_eval_to_train": eval_stats["median"],
                "p90_nn_eval_to_train": eval_stats["p90"],
                "p95_nn_eval_to_train": eval_stats["p95"],
                "mean_nn_train_to_eval": train_stats["mean"],
                "median_nn_train_to_eval": train_stats["median"],
                "p90_nn_train_to_eval": train_stats["p90"],
                "p95_nn_train_to_eval": train_stats["p95"],
                "weighted_distances": True,
                "weighted_radius": bool(use_weighted_radius),
                "radius_neighbor_agg": neighbor_agg,
                "flow_image_h": int(img_h),
                "flow_image_w": int(img_w),
            }
            results.append(row)

    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print(f"{'=' * 80}\n")

    results_df = pd.DataFrame(results)
    output_file = Path(config["output"]["kmeans_manifold_results_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)

    print(f"✓ Results saved to: {output_file}")
    print(f"  Total rows: {len(results_df)}")
    print(f"\n{'=' * 80}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description="Coverage Pipeline v2.2 (Flow K-Means Manifold)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
