#!/usr/bin/env python3
"""
Coverage + KL pipeline for HOF fingerprints.

- Loads cached per-image HOF fingerprints
- Builds vector caches for faster reuse
- Computes directed coverage (qnorm/rnorm) and KL divergence
- Reuses kNN distances wherever possible
"""

import argparse
import gc
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.coverage import cache, faiss_ops, metrics


# -----------------------------
# Helpers
# -----------------------------


def _as_l2(distances: np.ndarray, distance_metric: str) -> np.ndarray:
    metric = (distance_metric or "").lower()
    if metric in {"sql2", "sq_l2", "l2_sq", "sq-l2", "squared_l2"}:
        return np.sqrt(np.maximum(distances, 0.0))
    return distances


def _load_manifest(path: Path) -> Optional[Dict]:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return None
    import json

    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_hof_vectors_from_npz(
    hof_cache_dir: Path,
    dataset: str,
    split: str,
    max_samples: Optional[int] = None,
    verbose: bool = True,
) -> np.ndarray:
    ds_dir = hof_cache_dir / dataset / split
    if not ds_dir.exists():
        raise FileNotFoundError(f"HOF cache directory not found: {ds_dir}")

    manifest = _load_manifest(ds_dir)
    if manifest and "fingerprint_dim" in manifest:
        dim = int(manifest["fingerprint_dim"])
    else:
        dim = None

    files = sorted(ds_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No HOF fingerprints found in {ds_dir}")

    if max_samples is not None:
        files = files[: max_samples]

    if dim is None:
        with np.load(files[0]) as data:
            fp = data["fingerprint"]
            dim = int(fp.shape[0])

    n = len(files)
    if verbose:
        print(f"  Loading HOF fingerprints: {n:,} files, dim={dim}")

    vectors = np.zeros((n, dim), dtype=np.float32)
    for i, path in enumerate(files):
        with np.load(path) as data:
            fp = data["fingerprint"].astype(np.float32, copy=False)
        vectors[i] = fp

    return vectors


def _split_hof_components(
    vectors: np.ndarray,
    grid_hw: Tuple[int, int],
    hist_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    gh, gw = grid_hw
    per_cell = 1 + hist_bins
    expected_dim = gh * gw * per_cell
    if vectors.shape[1] != expected_dim:
        raise ValueError(
            f"Unexpected HOF dim={vectors.shape[1]} (expected {expected_dim}) "
            f"for grid={grid_hw}, hist_bins={hist_bins}."
        )
    cells = vectors.reshape(vectors.shape[0], gh * gw, per_cell)
    occ = cells[:, :, 0]
    hist = cells[:, :, 1:]
    hist = hist.reshape(vectors.shape[0], gh * gw * hist_bins)
    occ = occ.reshape(vectors.shape[0], gh * gw)
    return occ, hist


def _apply_occupancy_mode(
    vectors: np.ndarray,
    grid_hw: Tuple[int, int],
    hist_bins: int,
    mode: str,
    weight: float,
    l2_normalize: bool,
) -> np.ndarray:
    mode = (mode or "none").lower()
    if mode == "none":
        if l2_normalize:
            return cache.l2_normalize(vectors)
        return vectors

    occ, hist = _split_hof_components(vectors, grid_hw, hist_bins)

    if mode == "weighted_hist":
        gh, gw = grid_hw
        hist = hist.reshape(vectors.shape[0], gh * gw, hist_bins)
        occ_map = occ.reshape(vectors.shape[0], gh * gw, 1)
        hist = hist * occ_map
        hist = hist.reshape(vectors.shape[0], gh * gw * hist_bins)
        if l2_normalize:
            hist = cache.l2_normalize(hist)
        return hist

    if mode == "intersection":
        gh, gw = grid_hw
        hist = hist.reshape(vectors.shape[0], gh * gw, hist_bins)
        occ_map = occ.reshape(vectors.shape[0], gh * gw, 1)
        hist = hist * occ_map
        hist = hist.reshape(vectors.shape[0], gh * gw * hist_bins)
        occ_mass = np.maximum(occ.sum(axis=1, keepdims=True), 1e-6)
        hist = hist / np.sqrt(occ_mass)
        if l2_normalize:
            hist = cache.l2_normalize(hist)
        return hist

    if mode == "occ_only":
        if l2_normalize:
            return cache.l2_normalize(occ)
        return occ

    if mode == "concat":
        hist = cache.l2_normalize(hist)
        occ = cache.l2_normalize(occ)
        occ_weight = float(weight)
        hist_weight = 1.0 - occ_weight
        hist = hist * hist_weight
        occ = occ * occ_weight
        combined = np.concatenate([hist, occ], axis=1)
        if l2_normalize:
            combined = cache.l2_normalize(combined)
        return combined

    raise ValueError(f"Unknown occupancy_mode: {mode}")


def _load_or_build_vectors(
    cache_dir: Path,
    hof_cache_dir: Path,
    dataset: str,
    split: str,
    representation: str,
    max_samples: Optional[int],
    l2_normalize: bool,
    occupancy_mode: str,
    occupancy_weight: float,
    grid_hw: Tuple[int, int],
    hist_bins: int,
    occ_mean_cache: Optional[Dict[Tuple[str, str], np.ndarray]] = None,
    verbose: bool = True,
) -> np.ndarray:
    vectors = cache.load_cached_vectors(cache_dir, dataset, split, representation, mmap=False)
    occ_key = (dataset, split)
    occ_mean_dir = cache_dir / "occ_mean"
    occ_mean_dir.mkdir(parents=True, exist_ok=True)
    occ_mean_path = occ_mean_dir / f"occmean_{cache.sanitize_name(dataset)}_{cache.sanitize_name(split)}.npy"
    if vectors is not None:
        if verbose:
            print(f"  ✓ Loaded cached HOF vectors: {vectors.shape}")
        if occ_mean_cache is not None and occ_key not in occ_mean_cache:
            if occ_mean_path.exists():
                occ_mean_cache[occ_key] = np.load(occ_mean_path)
            else:
                raw = _load_hof_vectors_from_npz(
                    hof_cache_dir, dataset, split, max_samples=max_samples, verbose=False
                )
                occ, _ = _split_hof_components(raw, grid_hw, hist_bins)
                occ_mean = occ.mean(axis=0).astype(np.float32)
                np.save(occ_mean_path, occ_mean)
                occ_mean_cache[occ_key] = occ_mean
        return np.asarray(vectors, dtype=np.float32)

    raw_vectors = _load_hof_vectors_from_npz(
        hof_cache_dir, dataset, split, max_samples=max_samples, verbose=verbose
    )
    if occ_mean_cache is not None:
        occ, _ = _split_hof_components(raw_vectors, grid_hw, hist_bins)
        occ_mean = occ.mean(axis=0).astype(np.float32)
        np.save(occ_mean_path, occ_mean)
        occ_mean_cache[occ_key] = occ_mean

    vectors = _apply_occupancy_mode(
        raw_vectors,
        grid_hw=grid_hw,
        hist_bins=hist_bins,
        mode=occupancy_mode,
        weight=occupancy_weight,
        l2_normalize=l2_normalize,
    )

    cache.save_cached_vectors(cache_dir, dataset, split, representation, vectors, dtype="float32", compressed=False)
    if verbose:
        print(f"  ✓ Cached HOF vectors: {vectors.shape}")
    return vectors


def _compute_self_knn_distances(
    vectors: np.ndarray,
    k: int,
    use_gpu: bool,
    index_factory: str,
    nprobe: Optional[int],
    batch_size: Optional[int],
    filter_duplicates: bool,
    verbose: bool,
) -> np.ndarray:
    index = None
    fallback = None
    try:
        index = faiss_ops.build_index(
            vectors,
            use_gpu=use_gpu,
            index_factory=index_factory,
            nprobe=nprobe,
            verbose=verbose,
        )
        if index_factory.lower() != "flat":
            fallback = faiss_ops.build_index(vectors, use_gpu=True, index_factory="Flat", verbose=False)
        dists, _ = faiss_ops.compute_knn_distances(
            index,
            vectors,
            k=k,
            exclude_self=True,
            filter_duplicates=filter_duplicates,
            fallback_index=fallback,
            batch_size=batch_size,
            verbose=verbose,
        )
    finally:
        faiss_ops.release_index(index)
        faiss_ops.release_index(fallback)
    return dists


def _radius_from_self_distances(
    dists: np.ndarray,
    k: int,
    quantile: float,
    neighbor_agg: str,
) -> Dict[str, float]:
    if dists.size == 0:
        return {
            "radius": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "mean": float("nan"),
        }

    if neighbor_agg in ("first", "min"):
        sample = dists[:, 0]
    elif neighbor_agg in ("kth", "last", "max"):
        sample = dists[:, k - 1]
    elif neighbor_agg == "mean":
        sample = dists[:, :k].mean(axis=1)
    elif neighbor_agg == "median":
        sample = np.median(dists[:, :k], axis=1)
    else:
        raise ValueError(f"Unknown neighbor_agg: {neighbor_agg}")

    return {
        "radius": float(np.quantile(sample, quantile)),
        "median": float(np.median(sample)),
        "p90": float(np.quantile(sample, 0.90)),
        "p95": float(np.quantile(sample, 0.95)),
        "mean": float(np.mean(sample)),
    }


def _compute_kl_from_distances(
    self_dists: np.ndarray,
    cross_dists: np.ndarray,
    k_values: List[int],
    dim: int,
    eps: float,
) -> Dict[int, float]:
    n = self_dists.shape[0]
    m = cross_dists.shape[0]
    if n < 2 or m < 1:
        return {k: float("nan") for k in k_values}

    kl_vals: Dict[int, float] = {}
    for k in k_values:
        r = _as_l2(self_dists[:, k - 1], "sqL2")
        s = _as_l2(cross_dists[:, k - 1], "sqL2")
        r = np.maximum(r, eps)
        s = np.maximum(s, eps)
        mask = np.isfinite(r) & np.isfinite(s)
        if not np.any(mask):
            kl_vals[k] = float("nan")
            continue
        ratio = s[mask] / r[mask]
        term = np.log(ratio)
        n_eff = int(np.count_nonzero(mask))
        if n_eff < 2:
            kl_vals[k] = float("nan")
            continue
        kl = (dim / n_eff) * float(np.sum(term)) + float(np.log(m / max(n_eff - 1, 1)))
        kl_vals[k] = kl
    return kl_vals


def _get_dataset_key(ds_config: Dict) -> Tuple[str, str]:
    return ds_config.get("name"), ds_config.get("split")


# -----------------------------
# Main
# -----------------------------


def run(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    cache_dir = Path(config["cache"]["dir"]).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    hof_cache_dir = Path(config["cache"]["hof_dir"]).expanduser()

    representation = config.get("representation", "hof")
    hof_cfg = config.get("hof", {})
    l2_normalize = bool(hof_cfg.get("l2_normalize", False))
    occupancy_mode = str(hof_cfg.get("occupancy_mode", "none"))
    occupancy_weight = float(hof_cfg.get("occupancy_weight", 0.3))
    if occupancy_weight < 0.0 or occupancy_weight > 1.0:
        raise ValueError("hof.occupancy_weight must be in [0, 1].")

    grid_hw = tuple(hof_cfg.get("grid_hw", (32, 32)))
    angle_bins = int(hof_cfg.get("angle_bins", 8))
    mag_edges = hof_cfg.get("mag_edges", (0.0, 0.01, 0.03, 0.08, 0.25))
    mag_bins = max(1, len(mag_edges) - 1)
    hist_bins = angle_bins * mag_bins

    occ_tag = ""
    if occupancy_mode and occupancy_mode.lower() != "none":
        weight_tag = f"{occupancy_weight:.2f}".replace(".", "p")
        occ_tag = f"_occ{occupancy_mode.lower()}{weight_tag}"
    repr_name = f"{representation}{occ_tag}"
    if l2_normalize:
        repr_name = f"{repr_name}_l2"

    distance_metric = config.get("distance_metric", {}).get("name", "sqL2")

    faiss_cfg = config.get("faiss", {})
    use_gpu = bool(faiss_cfg.get("use_gpu", True))
    try:
        import torch

        if not torch.cuda.is_available():
            use_gpu = False
    except Exception:
        use_gpu = False

    index_factory = faiss_cfg.get("index_factory", "Flat")
    nprobe = faiss_cfg.get("nprobe")
    batch_size = faiss_cfg.get("batch_size")

    sampling_cfg = config.get("sampling", {})
    max_samples = sampling_cfg.get("max_samples")

    coverage_cfg = config.get("coverage", {})
    k_values = coverage_cfg.get("k_values", [1, 5])
    k_max_cov = int(coverage_cfg.get("k_max", max(k_values)))
    k_max_cov = max(k_max_cov, max(k_values))

    kl_cfg = config.get("kl", {})
    kl_enabled = bool(kl_cfg.get("enabled", True))
    kl_k_values = kl_cfg.get("k_values", [5, 10, 20, 40])
    kl_k_values = list(sorted(set(int(k) for k in kl_k_values)))
    kl_eps = float(kl_cfg.get("eps", 1e-12))

    k_max = max(k_max_cov, max(kl_k_values) if kl_enabled else 0)

    radius_k = int(coverage_cfg.get("self_radius_k", 5))
    radius_quantile = float(coverage_cfg.get("radius_quantile", 0.95))
    neighbor_agg = coverage_cfg.get("neighbor_agg", "kth")
    k_max = max(k_max, radius_k)

    compute_curves = bool(coverage_cfg.get("compute_curves", True))
    curve_quantiles = coverage_cfg.get("curve_quantiles", [0.80, 0.90, 0.95, 0.99])

    filter_duplicates = bool(coverage_cfg.get("filter_duplicates", True))

    normalization = "l2" if l2_normalize else "raw"
    if occupancy_mode and occupancy_mode.lower() != "none":
        weight_tag = f"{occupancy_weight:.2f}".replace(".", "p")
        normalization = f"{normalization}_occ{occupancy_mode.lower()}{weight_tag}"
    space = "hof"

    datasets = config.get("datasets", [])
    if not datasets:
        raise ValueError("No datasets in config")

    train_configs = [d for d in datasets if not d.get("is_eval", False)]
    eval_configs = [d for d in datasets if d.get("is_eval", False)]

    print(f"Loaded {len(train_configs)} train sets, {len(eval_configs)} eval sets")
    print(f"k_max (coverage+kl): {k_max}")
    if occupancy_mode and occupancy_mode.lower() != "none":
        print(f"Occupancy mode: {occupancy_mode} (weight={occupancy_weight:.2f})")

    # Precompute self distances + radii for all datasets
    self_dists_cache: Dict[Tuple[str, str], np.ndarray] = {}
    radii_cache: Dict[Tuple[str, str], Dict[str, float]] = {}
    dims_cache: Dict[Tuple[str, str], int] = {}

    occ_mean_cache: Dict[Tuple[str, str], np.ndarray] = {}

    for ds in datasets:
        ds_name, split = _get_dataset_key(ds)
        key = (ds_name, split)
        print(f"\n[{ds_name}/{split}] self-distance")
        vectors = _load_or_build_vectors(
            cache_dir,
            hof_cache_dir,
            ds_name,
            split,
            repr_name,
            max_samples=max_samples,
            l2_normalize=l2_normalize,
            occupancy_mode=occupancy_mode,
            occupancy_weight=occupancy_weight,
            grid_hw=grid_hw,
            hist_bins=hist_bins,
            occ_mean_cache=occ_mean_cache,
            verbose=True,
        )
        dims_cache[key] = int(vectors.shape[1])

        self_dists = cache.load_knn_self_distances(
            cache_dir,
            ds_name,
            split,
            representation=repr_name,
            space=space,
            k=k_max,
            normalization=normalization,
            distance_metric=distance_metric,
            filter_duplicates=filter_duplicates,
            mmap=False,
        )
        if self_dists is None:
            self_dists = _compute_self_knn_distances(
                vectors,
                k=k_max,
                use_gpu=use_gpu,
                index_factory=index_factory,
                nprobe=nprobe,
                batch_size=batch_size,
                filter_duplicates=filter_duplicates,
                verbose=True,
            )
            cache.save_knn_self_distances(
                cache_dir,
                ds_name,
                split,
                representation=repr_name,
                space=space,
                k=k_max,
                normalization=normalization,
                distance_metric=distance_metric,
                distances=self_dists,
                filter_duplicates=filter_duplicates,
            )
        else:
            print(f"  ✓ Loaded cached self-kNN distances: {self_dists.shape}")

        # Radius cache (reuse self distances)
        radius_data = cache.load_radius(
            cache_dir,
            ds_name,
            split,
            space=space,
            k=radius_k,
            quantile=radius_quantile,
            normalization=normalization,
            distance_metric=distance_metric,
        )
        if radius_data is None:
            radius_data = _radius_from_self_distances(
                self_dists,
                k=radius_k,
                quantile=radius_quantile,
                neighbor_agg=neighbor_agg,
            )
            cache.save_radius(
                cache_dir,
                ds_name,
                split,
                space,
                radius_data,
                k=radius_k,
                quantile=radius_quantile,
                normalization=normalization,
                distance_metric=distance_metric,
            )

        self_dists_cache[key] = np.asarray(self_dists)
        radii_cache[key] = radius_data

        del vectors
        gc.collect()

    # Pairwise coverage + KL
    coverage_rows: List[Dict] = []
    kl_rows: List[Dict] = []

    for train_cfg in train_configs:
        train_ds, train_split = _get_dataset_key(train_cfg)
        train_key = (train_ds, train_split)
        print(f"\n=== Train set: {train_ds}/{train_split} ===")
        train_vectors = _load_or_build_vectors(
            cache_dir,
            hof_cache_dir,
            train_ds,
            train_split,
            repr_name,
            max_samples=max_samples,
            l2_normalize=l2_normalize,
            occupancy_mode=occupancy_mode,
            occupancy_weight=occupancy_weight,
            grid_hw=grid_hw,
            hist_bins=hist_bins,
            occ_mean_cache=occ_mean_cache,
            verbose=True,
        )

        for eval_cfg in eval_configs:
            eval_ds, eval_split = _get_dataset_key(eval_cfg)
            eval_key = (eval_ds, eval_split)
            print(f"\n--- Eval set: {eval_ds}/{eval_split} ---")

            eval_vectors = _load_or_build_vectors(
                cache_dir,
                hof_cache_dir,
                eval_ds,
                eval_split,
                repr_name,
                max_samples=max_samples,
                l2_normalize=l2_normalize,
                occupancy_mode=occupancy_mode,
                occupancy_weight=occupancy_weight,
                grid_hw=grid_hw,
                hist_bins=hist_bins,
                occ_mean_cache=occ_mean_cache,
                verbose=True,
            )

            # Directed distances (cached)
            eval_to_train = cache.load_directed_distances(
                cache_dir,
                train_ds,
                train_split,
                eval_ds,
                eval_split,
                space,
                k_max,
                direction="eval_to_train",
                normalization=normalization,
                distance_metric=distance_metric,
            )
            train_to_eval = cache.load_directed_distances(
                cache_dir,
                train_ds,
                train_split,
                eval_ds,
                eval_split,
                space,
                k_max,
                direction="train_to_eval",
                normalization=normalization,
                distance_metric=distance_metric,
            )
            if eval_to_train is not None:
                print(f"  ✓ Loaded cached eval→train distances: {eval_to_train.shape}")
            if train_to_eval is not None:
                print(f"  ✓ Loaded cached train→eval distances: {train_to_eval.shape}")

            if eval_to_train is None:
                eval_to_train = faiss_ops.compute_eval_to_train(
                    train_vectors,
                    eval_vectors,
                    k=k_max,
                    use_gpu=use_gpu,
                    index_factory=index_factory,
                    nprobe=nprobe,
                    batch_size=batch_size,
                    verbose=True,
                )
                cache.save_directed_distances(
                    cache_dir,
                    train_ds,
                    train_split,
                    eval_ds,
                    eval_split,
                    space,
                    k_max,
                    direction="eval_to_train",
                    normalization=normalization,
                    distance_metric=distance_metric,
                    distances=eval_to_train,
                )

            if train_to_eval is None:
                train_to_eval = faiss_ops.compute_train_to_eval(
                    train_vectors,
                    eval_vectors,
                    k=k_max,
                    use_gpu=use_gpu,
                    index_factory=index_factory,
                    nprobe=nprobe,
                    batch_size=batch_size,
                    verbose=True,
                )
                cache.save_directed_distances(
                    cache_dir,
                    train_ds,
                    train_split,
                    eval_ds,
                    eval_split,
                    space,
                    k_max,
                    direction="train_to_eval",
                    normalization=normalization,
                    distance_metric=distance_metric,
                    distances=train_to_eval,
                )

            # Sanity check shapes
            if eval_to_train.shape[0] != eval_vectors.shape[0]:
                print(
                    f"  ⚠️  eval→train distances rows ({eval_to_train.shape[0]}) != n_eval ({eval_vectors.shape[0]})"
                )
            if train_to_eval.shape[0] != train_vectors.shape[0]:
                print(
                    f"  ⚠️  train→eval distances rows ({train_to_eval.shape[0]}) != n_train ({train_vectors.shape[0]})"
                )

            # Coverage metrics
            train_radius_data = radii_cache[train_key]
            eval_radius_data = radii_cache[eval_key]

            row = {
                "train_dataset": train_ds,
                "train_split": train_split,
                "eval_dataset": eval_ds,
                "eval_split": eval_split,
                "n_train": int(train_vectors.shape[0]),
                "n_eval": int(eval_vectors.shape[0]),
                "dim": int(train_vectors.shape[1]),
                "representation": repr_name,
                "distance_metric": distance_metric,
                "normalization": normalization,
            }

            occ_train = occ_mean_cache.get(train_key)
            occ_eval = occ_mean_cache.get(eval_key)
            if occ_train is not None and occ_eval is not None:
                diff = occ_train.astype(np.float32) - occ_eval.astype(np.float32)
                row["hof_density_l2"] = float(np.linalg.norm(diff))
                row["hof_density_l1"] = float(np.mean(np.abs(diff)))
                denom = np.linalg.norm(occ_train) * np.linalg.norm(occ_eval)
                if denom > 0:
                    row["hof_density_cosine"] = float(
                        np.dot(occ_train, occ_eval) / denom
                    )
                else:
                    row["hof_density_cosine"] = float("nan")

            cov_metrics = metrics.compute_coverage_metrics(
                train_radius_data["radius"],
                eval_radius_data["radius"],
                eval_to_train,
                train_to_eval,
                k_values=k_values,
            )
            row.update(cov_metrics)

            # Distance stats
            for k in k_values:
                if k > eval_to_train.shape[1]:
                    continue
                row[f"mean_nn_eval_to_train_k{k}"] = float(np.mean(eval_to_train[:, k - 1]))
                row[f"median_nn_eval_to_train_k{k}"] = float(np.median(eval_to_train[:, k - 1]))
                row[f"p90_nn_eval_to_train_k{k}"] = float(np.quantile(eval_to_train[:, k - 1], 0.90))
                row[f"p95_nn_eval_to_train_k{k}"] = float(np.quantile(eval_to_train[:, k - 1], 0.95))
                row[f"mean_nn_train_to_eval_k{k}"] = float(np.mean(train_to_eval[:, k - 1]))
                row[f"median_nn_train_to_eval_k{k}"] = float(np.median(train_to_eval[:, k - 1]))
                row[f"p90_nn_train_to_eval_k{k}"] = float(np.quantile(train_to_eval[:, k - 1], 0.90))
                row[f"p95_nn_train_to_eval_k{k}"] = float(np.quantile(train_to_eval[:, k - 1], 0.95))

            # Coverage curves
            if compute_curves:
                curves = metrics.compute_coverage_curves(
                    self_dists_cache[train_key],
                    self_dists_cache[eval_key],
                    eval_to_train,
                    train_to_eval,
                    quantiles=curve_quantiles,
                    k_values=k_values,
                )
                # Flatten curves into columns
                for metric_name, q_map in curves.items():
                    for q, val in q_map.items():
                        row[f"{metric_name}_q{q:.2f}"] = float(val)

            coverage_rows.append(row)

            # KL metrics
            if kl_enabled:
                kl_eval = _compute_kl_from_distances(
                    self_dists_cache[eval_key],
                    eval_to_train,
                    kl_k_values,
                    dims_cache[eval_key],
                    kl_eps,
                )
                kl_train = _compute_kl_from_distances(
                    self_dists_cache[train_key],
                    train_to_eval,
                    kl_k_values,
                    dims_cache[train_key],
                    kl_eps,
                )

                kl_row = {
                    "train_dataset": train_ds,
                    "train_split": train_split,
                    "eval_dataset": eval_ds,
                    "eval_split": eval_split,
                    "n_train": int(train_vectors.shape[0]),
                    "n_eval": int(eval_vectors.shape[0]),
                    "dim": int(train_vectors.shape[1]),
                    "representation": repr_name,
                    "distance_metric": distance_metric,
                    "normalization": normalization,
                }
                for k, val in kl_eval.items():
                    kl_row[f"kl_eval_to_train_k{k}"] = float(val)
                for k, val in kl_train.items():
                    kl_row[f"kl_train_to_eval_k{k}"] = float(val)
                kl_rows.append(kl_row)

            del eval_vectors
            gc.collect()

        del train_vectors
        gc.collect()

    # Save outputs
    out_cfg = config.get("output", {})
    results_file = Path(out_cfg.get("results_file", "analysis/coverage_v2_hof.csv"))
    results_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage_rows).to_csv(results_file, index=False)
    print(f"\nSaved coverage results to {results_file}")

    if kl_enabled:
        kl_file = Path(out_cfg.get("kl_results_file", "analysis/kl_v2_hof.csv"))
        kl_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(kl_rows).to_csv(kl_file, index=False)
        print(f"Saved KL results to {kl_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coverage + KL for HOF fingerprints")
    parser.add_argument("--config", required=True, help="Path to HOF coverage config")
    args = parser.parse_args()

    run(args.config)
