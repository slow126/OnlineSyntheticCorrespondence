#!/usr/bin/env python3
"""
Calculate coverage metrics using FAISS approximate nearest neighbors.

This produces directed coverage scores that can be used as drop-in predictors
for the existing leakage-free analysis (recall/precision/outside columns).
"""

import argparse
import gc
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.coreset.validation import extract_flow_vectors_from_batch
from src.data.synth.datasets.CorrespondenceDataset import CorrespondenceDataset
from src.data.synth.datasets.MixedCorrespondenceDataset import MixedCorrespondenceDataset
from src.mmd.encoders import BaseFeatureEncoder, ResNet101Encoder, DinoV3Encoder


def _is_synthetic_dataset(name: Optional[str]) -> bool:
    return isinstance(name, str) and name.startswith("synthetic")

try:
    import faiss  # type: ignore
except ImportError as exc:
    raise SystemExit(
        "faiss is required for calculate_coverage_faiss.py. "
        "Install faiss-cpu or faiss-gpu in your environment."
    ) from exc


def create_dataset_from_config(
    dataset_name: str,
    split: str,
    common_params: dict,
    dataset_overrides: dict,
    entry_overrides: Optional[dict] = None,
) -> CorrespondenceDataset:
    dataset_config = common_params.copy()
    if dataset_name in dataset_overrides:
        dataset_config.update(dataset_overrides[dataset_name])
    if entry_overrides:
        dataset_config.update(entry_overrides)
    adapter_name = dataset_name
    if entry_overrides and "dataset_name" in entry_overrides:
        adapter_name = str(entry_overrides["dataset_name"])
        dataset_config.pop("dataset_name", None)
    if adapter_name in dataset_overrides and dataset_name not in dataset_overrides:
        dataset_config.update(dataset_overrides[adapter_name])

    if "size" in dataset_config and isinstance(dataset_config["size"], list):
        dataset_config["size"] = tuple(dataset_config["size"])
    dataset_config["split"] = split
    if "max_kps" in dataset_config and dataset_config["max_kps"] is None:
        dataset_config["max_kps"] = None

    if adapter_name == "tss":
        dataset_config["thres"] = dataset_config.get("thres", "img")
        dataset_config["reverse_flow"] = dataset_config.get("reverse_flow", False)
    elif adapter_name == "middlebury":
        dataset_config["reverse_flow"] = dataset_config.get("reverse_flow", False)
    elif adapter_name == "pointodyssey":
        dataset_config["reverse_flow"] = dataset_config.get("reverse_flow", True)
        dataset_config["thres"] = dataset_config.get("thres", "img")
    elif adapter_name in ["kitti2012", "kitti2015"]:
        dataset_config["reverse_flow"] = dataset_config.get("reverse_flow", False)
        dataset_config["thres"] = dataset_config.get("thres", "img")
        if dataset_config.get("kitti_val_use_full_training", False) and split == "val":
            dataset_config["split"] = "training"
    elif adapter_name == "flyingthings":
        dataset_config["reverse_flow"] = dataset_config.get("reverse_flow", True)

    print(f"Creating dataset: {dataset_name} (split: {split})")
    return CorrespondenceDataset(adapter_name, **dataset_config)


def create_mixed_dataset_from_config(
    datasets_list: list,
    percentages: list,
    split: str,
    common_params: dict,
    dataset_overrides: dict,
    epoch_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> MixedCorrespondenceDataset:
    if len(datasets_list) != len(percentages):
        raise ValueError("Number of datasets must match number of percentages.")

    created_datasets = []
    for dataset_name in datasets_list:
        ds_config = common_params.copy()
        if dataset_name in dataset_overrides:
            ds_config.update(dataset_overrides[dataset_name])
        if "size" in ds_config and isinstance(ds_config["size"], list):
            ds_config["size"] = tuple(ds_config["size"])
        ds_config["split"] = split
        if "max_kps" in ds_config and ds_config["max_kps"] is None:
            ds_config["max_kps"] = None

        if dataset_name == "tss":
            ds_config["thres"] = ds_config.get("thres", "img")
            ds_config["reverse_flow"] = ds_config.get("reverse_flow", False)
        elif dataset_name == "middlebury":
            ds_config["reverse_flow"] = ds_config.get("reverse_flow", False)
        elif dataset_name == "pointodyssey":
            ds_config["reverse_flow"] = ds_config.get("reverse_flow", True)
            ds_config["thres"] = ds_config.get("thres", "img")
        elif dataset_name in ["kitti2012", "kitti2015"]:
            ds_config["reverse_flow"] = ds_config.get("reverse_flow", False)
            ds_config["thres"] = ds_config.get("thres", "img")
            if ds_config.get("kitti_val_use_full_training", False) and split == "val":
                ds_config["split"] = "training"
        elif dataset_name == "flyingthings":
            ds_config["reverse_flow"] = ds_config.get("reverse_flow", True)

        print(f"Creating sub-dataset: {dataset_name} (split: {split})")
        created_datasets.append(CorrespondenceDataset(dataset_name, **ds_config))

    print(f"Creating mixed dataset with {len(created_datasets)} datasets")
    return MixedCorrespondenceDataset(
        datasets=created_datasets,
        percentages=percentages,
        epoch_size=epoch_size,
        seed=seed,
    )


def create_encoder(encoder_name: str, device: torch.device) -> BaseFeatureEncoder:
    if encoder_name == "resnet101":
        return ResNet101Encoder(device=device)
    if encoder_name == "dino":
        return DinoV3Encoder(device=device)
    raise ValueError(f"Unknown encoder: {encoder_name}. Supported: 'resnet101', 'dino'")


def extract_features_from_batch(batch: dict, encoder: BaseFeatureEncoder) -> np.ndarray:
    if "src_img" in batch:
        img = batch["src_img"]
    elif "source" in batch:
        img = batch["source"]
    elif "image0" in batch:
        img = batch["image0"]
    else:
        raise ValueError(f"Could not find source image in batch. Keys: {batch.keys()}")

    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img)
    feats = encoder.extract_features(img)
    feats = feats.float().cpu().numpy().astype(np.float32, copy=False)
    return feats


def _extract_flow_vectors_per_image(
    batch: dict,
    per_image_max: Optional[int],
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    if per_image_max is None or per_image_max <= 0:
        return extract_flow_vectors_from_batch(batch)

    if "flow_full" in batch:
        flow_full = batch["flow_full"]
    elif "flow" in batch:
        flow_full = batch["flow"]
    else:
        return None

    if flow_full is None:
        return None

    if flow_full.dim() == 3:
        flow_full = flow_full.unsqueeze(0)

    batch_size, _, height, width = flow_full.shape
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x_flat = x_coords.reshape(-1)
    y_flat = y_coords.reshape(-1)

    all_vectors = []
    for sample_idx in range(batch_size):
        flow = flow_full[sample_idx].cpu().numpy()
        dx_flat = flow[0].reshape(-1)
        dy_flat = flow[1].reshape(-1)

        valid_mask = (
            np.isfinite(dx_flat)
            & np.isfinite(dy_flat)
            & ~((dx_flat == 0) & (dy_flat == 0))
        )
        if not np.any(valid_mask):
            continue

        vectors = np.stack(
            [x_flat[valid_mask], y_flat[valid_mask], dx_flat[valid_mask], dy_flat[valid_mask]],
            axis=1,
        ).astype(np.float32)

        if vectors.shape[0] > per_image_max:
            idx = rng.choice(vectors.shape[0], size=per_image_max, replace=False)
            vectors = vectors[idx]

        all_vectors.append(vectors)

    if not all_vectors:
        return None
    return np.vstack(all_vectors)


def _cache_key(label: str, split: str, representation: str, ext: str) -> str:
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    safe_split = re.sub(r"[^A-Za-z0-9_.-]+", "_", split)
    safe_repr = re.sub(r"[^A-Za-z0-9_.-]+", "_", representation)
    return f"{safe_label}_{safe_split}_{safe_repr}.{ext}"


def _radius_cache_key(
    label: str,
    split: str,
    representation: str,
    metric: str,
    support_mode: str,
    k: int,
    quantile: float,
    agg: str,
    norm_mode: str = "none",
    norm_by_label: Optional[str] = None,
) -> str:
    """Generate cache key for self-radius calculations.
    
    Args:
        norm_by_label: If using train_zscore, this is the train set that provided normalization stats.
                       For train sets, this is None (they normalize themselves).
                       For eval sets, this is the train label (eval normalized by that train's stats).
    """
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    safe_split = re.sub(r"[^A-Za-z0-9_.-]+", "_", split)
    safe_repr = re.sub(r"[^A-Za-z0-9_.-]+", "_", representation)
    safe_metric = re.sub(r"[^A-Za-z0-9_.-]+", "_", metric)
    safe_mode = re.sub(r"[^A-Za-z0-9_.-]+", "_", support_mode)
    safe_agg = re.sub(r"[^A-Za-z0-9_.-]+", "_", agg)
    safe_norm = re.sub(r"[^A-Za-z0-9_.-]+", "_", norm_mode)
    
    # Base key - ALWAYS include normalization mode to prevent cache collisions
    base = f"radius_{safe_label}_{safe_split}_{safe_repr}_{safe_metric}_{safe_mode}_{safe_norm}"
    
    # Add train-specific label if using train_zscore with a specific train normalizer
    if norm_mode == "train_zscore" and norm_by_label is not None:
        safe_norm_by = re.sub(r"[^A-Za-z0-9_.-]+", "_", norm_by_label)
        base = f"{base}_normby_{safe_norm_by}"
    
    if support_mode == "per_point_radius":
        # Per-point radius: depends on k and agg only
        return f"{base}_k{k}_{safe_agg}.npy"
    else:
        # Global radius: depends on k, quantile, and agg
        return f"{base}_k{k}_q{quantile:.3f}_{safe_agg}.npy"


def _load_cached_radius(
    cache_dir: Path,
    label: str,
    split: str,
    representation: str,
    metric: str,
    support_mode: str,
    k: int,
    quantile: float,
    agg: str,
    norm_mode: str = "none",
    norm_by_label: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Load cached self-radius from disk.
    
    Args:
        norm_by_label: If using train_zscore, the train set that provided normalization.
                       None for train sets (self-normalized) or when not using train_zscore.
    """
    if not cache_dir:
        return None
    key = _radius_cache_key(label, split, representation, metric, support_mode, 
                           k, quantile, agg, norm_mode, norm_by_label)
    path = cache_dir / key
    if path.exists():
        return np.load(path)
    return None


def _save_cached_radius(
    cache_dir: Path,
    label: str,
    split: str,
    representation: str,
    metric: str,
    support_mode: str,
    k: int,
    quantile: float,
    agg: str,
    radius_data: np.ndarray,
    norm_mode: str = "none",
    norm_by_label: Optional[str] = None,
) -> None:
    """Save self-radius to disk cache.
    
    Args:
        norm_by_label: If using train_zscore, the train set that provided normalization.
                       None for train sets (self-normalized) or when not using train_zscore.
    """
    if not cache_dir:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _radius_cache_key(label, split, representation, metric, support_mode, 
                           k, quantile, agg, norm_mode, norm_by_label)
    path = cache_dir / key
    np.save(path, radius_data)


def _cache_exists(cache_dir: Path, label: str, split: str, representation: str) -> bool:
    for ext in ("npy", "npz"):
        path = cache_dir / _cache_key(label, split, representation, ext)
        if path.exists():
            return True
    return False


def _load_cached_vectors(
    cache_dir: Path,
    label: str,
    split: str,
    representation: str,
    mmap: bool,
) -> Optional[np.ndarray]:
    for ext in ("npy", "npz"):
        path = cache_dir / _cache_key(label, split, representation, ext)
        if path.exists():
            if ext == "npy":
                return np.load(path, mmap_mode="r" if mmap else None)
            data = np.load(path, mmap_mode="r" if mmap else None)
            if "vectors" in data:
                return data["vectors"]
            raise ValueError(f"Missing 'vectors' in cache file: {path}")
    return None


def _save_cached_vectors(
    cache_dir: Path,
    label: str,
    split: str,
    representation: str,
    vectors: np.ndarray,
    fmt: str,
    dtype: Optional[str],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if dtype:
        vectors = vectors.astype(dtype, copy=False)
    fmt = fmt.lower()
    if fmt == "npz":
        path = cache_dir / _cache_key(label, split, representation, "npz")
        np.savez(path, vectors=vectors)
    else:
        path = cache_dir / _cache_key(label, split, representation, "npy")
        np.save(path, vectors)


def _mix_vectors_from_cache(
    datasets_list: list,
    percentages: list,
    split: str,
    representation: str,
    cache_dir: Path,
    rng: np.random.Generator,
    max_vectors: Optional[int],
    mmap: bool,
) -> Optional[np.ndarray]:
    vectors_list = []
    capacities = []
    for ds_name in datasets_list:
        cached = _load_cached_vectors(cache_dir, ds_name, split, representation, mmap)
        if cached is None:
            return None
        vectors_list.append(cached)
        capacities.append(int(cached.shape[0]))

    total_available = int(sum(capacities))
    if total_available == 0:
        return None

    total = int(max_vectors) if max_vectors else total_available
    total = min(total, total_available)

    weights = np.array(percentages, dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()

    counts = np.floor(weights * total).astype(int)
    counts = np.minimum(counts, capacities)
    leftover = total - int(counts.sum())

    if leftover > 0:
        order = np.argsort(weights)[::-1]
        while leftover > 0:
            updated = False
            for idx in order:
                if counts[idx] < capacities[idx]:
                    counts[idx] += 1
                    leftover -= 1
                    updated = True
                    if leftover == 0:
                        break
            if not updated:
                break

    mixed = []
    for vecs, n_keep in zip(vectors_list, counts):
        if n_keep <= 0:
            continue
        if n_keep >= vecs.shape[0]:
            mixed.append(np.asarray(vecs))
            continue
        idx = rng.choice(vecs.shape[0], size=int(n_keep), replace=False)
        mixed.append(np.asarray(vecs)[idx])

    if not mixed:
        return None
    return np.vstack(mixed).astype(np.float32, copy=False)


def _resolve_label(ds_config: dict) -> str:
    label = ds_config.get("name")
    is_mixed = ds_config.get("mixed", False) or "datasets" in ds_config
    if label or not is_mixed:
        return label
    datasets_list = ds_config.get("datasets", [])
    percentages = ds_config.get("percentages", [])
    if len(percentages) == 2 and len(datasets_list) == 2:
        pct1 = int(percentages[0] * 100)
        pct2 = int(percentages[1] * 100)
        return f"{datasets_list[0]}_{datasets_list[1]}_{pct1}_{pct2}"
    return "+".join(datasets_list)


def _apply_pairwise_pca(
    train_vecs: np.ndarray,
    eval_vecs: np.ndarray,
    pca_cfg: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if not pca_cfg or not pca_cfg.get("enabled", False):
        return train_vecs, eval_vecs

    fit_on = str(pca_cfg.get("fit_on", "train")).lower()
    max_train = pca_cfg.get("max_train_vectors", 200000)
    output_dim = int(pca_cfg.get("output_dim", 256))

    if fit_on == "eval":
        fit_vectors = eval_vecs
    elif fit_on == "all":
        fit_vectors = np.concatenate([train_vecs, eval_vecs], axis=0)
    else:
        fit_vectors = train_vecs

    if fit_vectors.size == 0:
        return train_vecs, eval_vecs

    if max_train and fit_vectors.shape[0] > max_train:
        idx = rng.choice(fit_vectors.shape[0], size=max_train, replace=False)
        fit_vectors = fit_vectors[idx]

    dim = fit_vectors.shape[1]
    if output_dim >= dim:
        return train_vecs, eval_vecs

    pca = faiss.PCAMatrix(dim, output_dim, eigen_power=-0.5 if pca_cfg.get("whiten") else 0.0)
    pca.train(fit_vectors.astype(np.float32, copy=False))

    train_vecs = pca.apply_py(train_vecs.astype(np.float32, copy=False))
    eval_vecs = pca.apply_py(eval_vecs.astype(np.float32, copy=False))
    return train_vecs, eval_vecs


def _subsample_dense(
    vectors: np.ndarray,
    threshold: Optional[int],
    fraction: float,
    min_keep: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if threshold is None or vectors.shape[0] <= threshold:
        return vectors
    keep = max(min_keep, int(vectors.shape[0] * fraction))
    keep = min(keep, vectors.shape[0])
    if keep == vectors.shape[0]:
        return vectors
    idx = rng.choice(vectors.shape[0], size=keep, replace=False)
    return vectors[idx]


class VectorCollector:
    def __init__(self, max_vectors: Optional[int], rng: np.random.Generator):
        self.max_vectors = max_vectors
        self.rng = rng
        self.buffers = []
        self.total = 0

    def add(self, vectors: Optional[np.ndarray]) -> None:
        if vectors is None or vectors.size == 0:
            return
        self.buffers.append(vectors)
        self.total += vectors.shape[0]
        if self.max_vectors and self.total > self.max_vectors * 2:
            data = np.concatenate(self.buffers, axis=0)
            idx = self.rng.choice(data.shape[0], size=self.max_vectors, replace=False)
            self.buffers = [data[idx]]
            self.total = self.max_vectors

    def finalize(self) -> np.ndarray:
        if not self.buffers:
            return np.empty((0, 0), dtype=np.float32)
        data = np.concatenate(self.buffers, axis=0)
        if self.max_vectors and data.shape[0] > self.max_vectors:
            idx = self.rng.choice(data.shape[0], size=self.max_vectors, replace=False)
            data = data[idx]
        return data.astype(np.float32, copy=False)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _compute_zscore_params(vectors: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    if vectors.size == 0:
        return np.zeros((vectors.shape[1],), dtype=np.float32), np.ones((vectors.shape[1],), dtype=np.float32)
    mean = vectors.mean(axis=0, dtype=np.float64)
    std = vectors.std(axis=0, dtype=np.float64)
    std = np.where(std < eps, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_zscore(vectors: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    return ((vectors - mean) / std).astype(np.float32, copy=False)


def _compute_mad_scale(vectors: np.ndarray) -> float:
    """Compute Median Absolute Deviation (MAD) as a robust scale estimator.
    
    MAD = median(|x - median(x)|) / 0.6745
    The 0.6745 factor makes MAD consistent with std for Gaussian data.
    """
    if vectors.size == 0:
        return 1.0
    median = np.median(vectors)
    mad = np.median(np.abs(vectors - median))
    # Convert MAD to std-equivalent scale (for Gaussian data)
    scale = mad / 0.6745
    return float(scale) if scale > 1e-10 else 1.0


def _compute_global_alpha(
    train_vectors_dict: Dict[str, np.ndarray],
    pos_dims: tuple = (0, 1),
    flow_dims: tuple = (2, 3),
    eps: float = 1e-6
) -> float:
    """Compute global alpha parameter for block balancing.
    
    α = mean_i(scale_i^pos) / (mean_i(scale_i^flow) + ε)
    
    Each dataset contributes equally regardless of size.
    Uses MAD for robust scale estimation.
    """
    if not train_vectors_dict:
        return 1.0
    
    pos_scales = []
    flow_scales = []
    
    for train_vecs in train_vectors_dict.values():
        if train_vecs.size == 0 or train_vecs.shape[1] < 4:
            continue
        
        # Position scale: MAD of [x, y]
        pos_data = train_vecs[:, list(pos_dims)].ravel()
        pos_scale = _compute_mad_scale(pos_data)
        pos_scales.append(pos_scale)
        
        # Flow scale: MAD of [dx, dy]
        flow_data = train_vecs[:, list(flow_dims)].ravel()
        flow_scale = _compute_mad_scale(flow_data)
        flow_scales.append(flow_scale)
    
    if not pos_scales or not flow_scales:
        return 1.0
    
    mean_pos = np.mean(pos_scales)
    mean_flow = np.mean(flow_scales)
    
    alpha = mean_pos / (mean_flow + eps)
    
    print(f"\n{'='*80}")
    print(f"Global α computation (equal-weighted across {len(pos_scales)} train sets):")
    print(f"  Position scales (MAD): {pos_scales}")
    print(f"  Flow scales (MAD): {flow_scales}")
    print(f"  Mean position scale: {mean_pos:.6f}")
    print(f"  Mean flow scale: {mean_flow:.6f}")
    print(f"  α = {alpha:.6f}")
    print(f"{'='*80}\n")
    
    return float(alpha)


def _apply_global_alpha(
    vectors: np.ndarray,
    alpha: float,
    pos_dims: tuple = (0, 1),
    flow_dims: tuple = (2, 3)
) -> np.ndarray:
    """Apply global alpha balancing: v' = [x, y, α·dx, α·dy]"""
    if vectors.size == 0:
        return vectors
    
    result = vectors.copy()
    for dim in flow_dims:
        if dim < vectors.shape[1]:
            result[:, dim] *= alpha
    
    return result.astype(np.float32, copy=False)


def _save_global_alpha(
    cache_dir: Path,
    representation: str,
    alpha: float,
    train_labels: list,
    pos_dims: tuple = (0, 1),
    flow_dims: tuple = (2, 3)
) -> None:
    """Save global alpha to disk cache with metadata.
    
    Cache includes:
    - alpha value
    - list of train set labels used to compute it
    - dimension configuration
    
    This allows validating the cache is still valid on subsequent runs.
    """
    if not cache_dir:
        return
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create cache filename
    safe_repr = re.sub(r"[^A-Za-z0-9_.-]+", "_", representation)
    cache_file = cache_dir / f"global_alpha_{safe_repr}.npz"
    
    # Save with metadata
    np.savez(
        cache_file,
        alpha=np.array([alpha], dtype=np.float32),
        train_labels=np.array(train_labels, dtype=object),
        pos_dims=np.array(pos_dims, dtype=np.int32),
        flow_dims=np.array(flow_dims, dtype=np.int32)
    )
    print(f"  Saved global α={alpha:.6f} to cache: {cache_file.name}")


def _load_global_alpha(
    cache_dir: Path,
    representation: str,
    train_labels: list,
    pos_dims: tuple = (0, 1),
    flow_dims: tuple = (2, 3)
) -> Optional[float]:
    """Load global alpha from disk cache.
    
    Returns None if:
    - Cache doesn't exist
    - Train set labels don't match (different datasets)
    - Dimension configuration changed
    """
    if not cache_dir:
        return None
    
    cache_dir = Path(cache_dir)
    safe_repr = re.sub(r"[^A-Za-z0-9_.-]+", "_", representation)
    cache_file = cache_dir / f"global_alpha_{safe_repr}.npz"
    
    if not cache_file.exists():
        return None
    
    try:
        data = np.load(cache_file, allow_pickle=True)
        
        # Validate metadata
        cached_labels = set(data['train_labels'].tolist())
        current_labels = set(train_labels)
        
        if cached_labels != current_labels:
            print(f"  ⚠️  Cached α invalid: train set labels changed")
            return None
        
        cached_pos = tuple(data['pos_dims'].tolist())
        cached_flow = tuple(data['flow_dims'].tolist())
        
        if cached_pos != pos_dims or cached_flow != flow_dims:
            print(f"  ⚠️  Cached α invalid: dimension configuration changed")
            return None
        
        alpha = float(data['alpha'].item())
        print(f"  ✓ Loaded cached global α={alpha:.6f} from {cache_file.name}")
        return alpha
        
    except Exception as e:
        print(f"  ⚠️  Error loading cached α: {e}")
        return None


def _nn_first(
    index: "faiss.Index",
    vectors: np.ndarray,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    if vectors.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
    dists, idxs = index.search(vectors, 1)
    if metric == "cosine":
        dists = 1.0 - dists
    else:
        dists = np.sqrt(np.maximum(dists, 0.0))
    return dists[:, 0], idxs[:, 0]


def _pointwise_radius(
    index: "faiss.Index",
    vectors: np.ndarray,
    metric: str,
    k: int,
    agg: str = "kth",
    batch_size: Optional[int] = None,
) -> np.ndarray:
    if vectors.size == 0:
        return np.array([], dtype=np.float32)
    if vectors.shape[0] < 2:
        return np.full((vectors.shape[0],), float("nan"), dtype=np.float32)
    k = max(int(k), 1)
    search_k = min(vectors.shape[0], k + 1)
    if search_k <= 1:
        return np.full((vectors.shape[0],), float("nan"), dtype=np.float32)
    
    # Auto-adjust batch size based on dimensionality
    if batch_size is None:
        dim = vectors.shape[1]
        if dim <= 4:
            batch_size = 500000  # Flow vectors (4D)
        elif dim <= 64:
            batch_size = 100000  # Medium dimensions
        elif dim <= 256:
            batch_size = 50000   # High dimensions (ResNet/DINO after PCA)
        else:
            batch_size = 20000   # Very high dimensions
    
    # Batch search for GPU indices to avoid OOM
    is_gpu = _is_gpu_index(index)
    if is_gpu and len(vectors) > batch_size:
        print(f"    Computing pointwise radius: {len(vectors):,} vectors in batches of {batch_size:,}")
        all_dists = []
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            batch_dists, _ = index.search(batch, search_k)
            all_dists.append(batch_dists)
            if (i // batch_size + 1) % 5 == 0:  # Print every 5 batches (was 10)
                print(f"      Processed {i+len(batch):,}/{len(vectors):,} vectors")
        dists = np.vstack(all_dists)
    else:
        dists, _ = index.search(vectors, search_k)
    
    if metric == "cosine":
        dists = 1.0 - dists
    else:
        dists = np.sqrt(np.maximum(dists, 0.0))
    neigh_dists = dists[:, 1:]
    if neigh_dists.size == 0:
        return np.full((vectors.shape[0],), float("nan"), dtype=np.float32)
    if agg in ("first", "min"):
        sample = neigh_dists[:, 0]
    elif agg in ("kth", "last", "max"):
        sample = neigh_dists[:, -1]
    elif agg == "mean":
        sample = neigh_dists.mean(axis=1)
    elif agg == "median":
        sample = np.median(neigh_dists, axis=1)
    else:
        raise ValueError(f"Unsupported neighbor aggregation: {agg}")
    return sample.astype(np.float32, copy=False)


def _filter_vectors(
    vectors: Optional[np.ndarray],
    drop_zero: bool,
    min_norm: float = 0.0,
) -> tuple[Optional[np.ndarray], int, int]:
    if vectors is None or vectors.size == 0:
        return vectors, 0, 0
    finite_mask = np.all(np.isfinite(vectors), axis=1)
    dropped_non_finite = int(vectors.shape[0] - finite_mask.sum())
    vectors = vectors[finite_mask]
    dropped_zero = 0
    if drop_zero and vectors.size:
        norms = np.linalg.norm(vectors, axis=1)
        zero_mask = norms > float(min_norm)
        dropped_zero = int(vectors.shape[0] - zero_mask.sum())
        vectors = vectors[zero_mask]
    return vectors, dropped_non_finite, dropped_zero


def _is_gpu_index(index: "faiss.Index") -> bool:
    """Check if an index is a GPU index."""
    # More robust GPU detection
    index_type = type(index).__name__
    return 'Gpu' in index_type or 'GPU' in index_type


def _build_index(
    vectors: np.ndarray,
    index_factory: str,
    metric: str,
    use_gpu: bool,
    nprobe: Optional[int],
    gpu_resources: Optional["faiss.StandardGpuResources"] = None,
) -> "faiss.Index":
    """
    Build a FAISS index, using GPU-friendly types when GPU is enabled.
    
    For GPU, HNSW indices are not natively supported. The function will:
    - Use Flat for exact search (very fast on GPU)
    - Use IVF indices for approximate search (GPU-friendly)
    - Convert HNSW to IVF when GPU is enabled
    
    Args:
        gpu_resources: Optional pre-created StandardGpuResources to reuse
    """
    dim = vectors.shape[1]
    if metric == "cosine":
        metric_type = faiss.METRIC_INNER_PRODUCT
    else:
        metric_type = faiss.METRIC_L2

    # Handle GPU-friendly index selection
    if use_gpu and faiss.get_num_gpus() > 0:
        index_factory_lower = index_factory.lower()
        n_vectors = vectors.shape[0]
        
        # Check if we have enough vectors for IVF training
        # IVF needs ~40x the number of centroids
        if "ivf" in index_factory_lower:
            import re
            match = re.search(r'ivf(\d+)', index_factory_lower)
            if match:
                nlist = int(match.group(1))
                min_vectors_needed = nlist * 40
                if n_vectors < min_vectors_needed:
                    print(f"  WARNING: Only {n_vectors:,} vectors, too few for IVF{nlist} (needs {min_vectors_needed:,})")
                    print(f"  Falling back to Flat index for exact search")
                    index_factory = "Flat"
        
        # HNSW is not GPU-native - convert to GPU-friendly alternatives
        if "hnsw" in index_factory_lower:
            # Extract HNSW parameter if present (e.g., HNSW32 -> 32)
            import re
            match = re.search(r'hnsw(\d+)', index_factory_lower)
            if match:
                # Use IVF with similar memory/accuracy tradeoff, but check vector count
                if n_vectors < 100000:
                    if n_vectors < 10000:
                        print(f"  GPU mode: Too few vectors ({n_vectors:,}), using Flat instead of HNSW")
                        index_factory = "Flat"
                    else:
                        nlist = 256
                        print(f"  GPU mode: Converting HNSW to GPU-friendly IVF{nlist},Flat")
                        index_factory = f"IVF{nlist},Flat"
                elif n_vectors < 1000000:
                    nlist = 1024
                    print(f"  GPU mode: Converting HNSW to GPU-friendly IVF{nlist},Flat")
                    index_factory = f"IVF{nlist},Flat"
                else:
                    nlist = 2048
                    print(f"  GPU mode: Converting HNSW to GPU-friendly IVF{nlist},Flat")
                    index_factory = f"IVF{nlist},Flat"
            else:
                # Default fallback
                if n_vectors < 10000:
                    print(f"  GPU mode: Too few vectors ({n_vectors:,}), using Flat instead of HNSW")
                    index_factory = "Flat"
                else:
                    print(f"  GPU mode: Converting HNSW to GPU-friendly IVF1024,Flat")
                    index_factory = "IVF1024,Flat"
        
        # Build index on CPU first (required for training)
        if index_factory.lower() == "flat":
            if metric_type == faiss.METRIC_INNER_PRODUCT:
                index = faiss.IndexFlatIP(dim)
            else:
                index = faiss.IndexFlatL2(dim)
        else:
            index = faiss.index_factory(dim, index_factory, metric_type)

        # Train and add vectors on CPU
        if index.is_trained is False:
            index.train(vectors)
        index.add(vectors)

        # Set nprobe before GPU transfer (if applicable)
        if nprobe is not None and hasattr(index, "nprobe"):
            index.nprobe = nprobe

        # Transfer to GPU using proper resource management
        gpu_id = 0  # Use first GPU
        
        # Reuse gpu_resources if provided, otherwise create new one
        if gpu_resources is None:
            gpu_resources = faiss.StandardGpuResources()
            # Optimized temp memory: 18GB for faster searches (6GB headroom)
            # FAISS will use this for search operations
            temp_memory_gb = 18
            gpu_resources.setTempMemory(temp_memory_gb * 1024 * 1024 * 1024)
            print(f"  Created new GPU resources with {temp_memory_gb}GB temp memory")
        
        # Convert to GPU index
        index = faiss.index_cpu_to_gpu(gpu_resources, gpu_id, index)
        
        # Set nprobe again after GPU transfer (in case it was reset)
        if nprobe is not None and hasattr(index, "nprobe"):
            index.nprobe = nprobe
        
        print(f"  Index transferred to GPU {gpu_id} ({vectors.shape[0]:,} vectors, dim={dim})")
            
    else:
        # CPU mode - check if we have enough vectors for IVF
        n_vectors = vectors.shape[0]
        if "ivf" in index_factory.lower():
            import re
            match = re.search(r'ivf(\d+)', index_factory.lower())
            if match:
                nlist = int(match.group(1))
                min_vectors_needed = nlist * 40
                if n_vectors < min_vectors_needed:
                    print(f"  WARNING: Only {n_vectors:,} vectors, too few for IVF{nlist} (needs {min_vectors_needed:,})")
                    print(f"  Falling back to Flat index for exact search")
                    index_factory = "Flat"
        
        print(f"  Building CPU index (type={index_factory}, vectors={vectors.shape[0]:,}, dim={dim})...")
        
        if index_factory.lower() == "flat":
            if metric_type == faiss.METRIC_INNER_PRODUCT:
                index = faiss.IndexFlatIP(dim)
            else:
                index = faiss.IndexFlatL2(dim)
        else:
            index = faiss.index_factory(dim, index_factory, metric_type)

        if index.is_trained is False:
            print(f"    Training index...")
            index.train(vectors)
            print(f"    Training complete.")
        
        print(f"    Adding {vectors.shape[0]:,} vectors to index...")
        index.add(vectors)
        print(f"    Index built successfully.")

        if nprobe is not None and hasattr(index, "nprobe"):
            index.nprobe = nprobe

    return index


def _nn_distances(
    index: "faiss.Index",
    vectors: np.ndarray,
    metric: str,
    k: int,
    agg: str = "first",
    batch_size: Optional[int] = None,
) -> np.ndarray:
    if vectors.size == 0:
        return np.array([], dtype=np.float32)
    
    # Auto-adjust batch size based on dimensionality
    if batch_size is None:
        dim = vectors.shape[1]
        if dim <= 4:
            batch_size = 500000  # Flow vectors (4D)
        elif dim <= 64:
            batch_size = 100000  # Medium dimensions
        elif dim <= 256:
            batch_size = 50000   # High dimensions (ResNet/DINO after PCA)
        else:
            batch_size = 20000   # Very high dimensions
    
    # Batch search for GPU indices to avoid OOM
    is_gpu = _is_gpu_index(index)
    if is_gpu and len(vectors) > batch_size:
        print(f"    Batched GPU search: {len(vectors):,} vectors in batches of {batch_size:,}")
        all_dists = []
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            batch_dists, _ = index.search(batch, k)
            all_dists.append(batch_dists)
            if (i // batch_size + 1) % 5 == 0:  # Print every 5 batches (was 10)
                print(f"      Processed {i+len(batch):,}/{len(vectors):,} vectors")
        dists = np.vstack(all_dists)
    else:
        dists, _ = index.search(vectors, k)
    if metric == "cosine":
        dists = 1.0 - dists
    else:
        dists = np.sqrt(np.maximum(dists, 0.0))
    if k <= 1 or agg in ("first", "min"):
        out = dists[:, 0]
    elif agg in ("kth", "last", "max"):
        out = dists[:, -1]
    elif agg == "mean":
        out = dists.mean(axis=1)
    elif agg == "median":
        out = np.median(dists, axis=1)
    else:
        raise ValueError(f"Unsupported neighbor aggregation: {agg}")
    return out.astype(np.float32, copy=False)


def _kl_divergence_from_samples(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    bins: int,
    eps: float,
) -> float:
    p_samples = p_samples[np.isfinite(p_samples)]
    q_samples = q_samples[np.isfinite(q_samples)]
    if p_samples.size == 0 or q_samples.size == 0:
        return float("nan")
    if bins < 2:
        return float("nan")
    combined = np.concatenate([p_samples, q_samples], axis=0)
    combined = combined[np.isfinite(combined)]
    if combined.size == 0:
        return float("nan")
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(combined, quantiles))
    if edges.size < 2:
        return 0.0
    p_counts, _ = np.histogram(p_samples, bins=edges)
    q_counts, _ = np.histogram(q_samples, bins=edges)
    p = p_counts.astype(np.float64) + eps
    q = q_counts.astype(np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def _kl_divergence_from_samples_linear(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    bins: int,
    eps: float,
) -> float:
    p_samples = p_samples[np.isfinite(p_samples)]
    q_samples = q_samples[np.isfinite(q_samples)]
    if p_samples.size == 0 or q_samples.size == 0:
        return float("nan")
    if bins < 2:
        return float("nan")
    combined = np.concatenate([p_samples, q_samples], axis=0)
    combined = combined[np.isfinite(combined)]
    if combined.size == 0:
        return float("nan")
    lo = float(np.min(combined))
    hi = float(np.max(combined))
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    p_counts, _ = np.histogram(p_samples, bins=edges)
    q_counts, _ = np.histogram(q_samples, bins=edges)
    p = p_counts.astype(np.float64) + eps
    q = q_counts.astype(np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def _log1p_samples(samples: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(samples, 0.0))


def _rank_percentiles(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    p = p_samples[np.isfinite(p_samples)]
    q = q_samples[np.isfinite(q_samples)]
    if p.size == 0 or q.size == 0:
        return None, None
    combined = np.concatenate([p, q], axis=0)
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    denom = max(order.size - 1, 1)
    percentiles = ranks / float(denom)
    return percentiles[: p.size], percentiles[p.size :]


def _convert_search_distances(dists: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        dists = 1.0 - dists
        dists = np.maximum(dists, 0.0)
        dists = np.sqrt(2.0 * dists)
    else:
        dists = np.sqrt(np.maximum(dists, 0.0))
    return dists


def _kth_neighbor_distances(
    index: "faiss.Index",
    vectors: np.ndarray,
    metric: str,
    k: int,
    exclude_self: bool,
    batch_size: Optional[int] = None,
) -> Optional[np.ndarray]:
    if vectors.size == 0 or k < 1:
        return None
    n = vectors.shape[0]
    if exclude_self and n <= k:
        return None
    if not exclude_self and n < k:
        return None
    search_k = k + 1 if exclude_self else k
    search_k = min(search_k, n)
    if exclude_self and search_k <= k:
        return None
    
    # Auto-adjust batch size based on dimensionality
    if batch_size is None:
        dim = vectors.shape[1]
        if dim <= 4:
            batch_size = 500000  # Flow vectors (4D)
        elif dim <= 64:
            batch_size = 100000  # Medium dimensions
        elif dim <= 256:
            batch_size = 50000   # High dimensions (ResNet/DINO after PCA)
        else:
            batch_size = 20000   # Very high dimensions
    
    # Batch search for GPU indices to avoid OOM
    is_gpu = _is_gpu_index(index)
    if is_gpu and len(vectors) > batch_size:
        print(f"    Computing KL k-NN distances: {len(vectors):,} vectors in batches of {batch_size:,}")
        all_dists = []
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            batch_dists, _ = index.search(batch, search_k)
            all_dists.append(batch_dists)
            if (i // batch_size + 1) % 5 == 0:  # Print every 5 batches (was 10)
                print(f"      Processed {i+len(batch):,}/{len(vectors):,} vectors")
        dists = np.vstack(all_dists)
    else:
        dists, _ = index.search(vectors, search_k)
    
    dists = _convert_search_distances(dists, metric)
    if exclude_self:
        tiny = 1e-12
        has_self = dists[:, 0] <= tiny
        idx = np.where(has_self, k, k - 1)
        return np.take_along_axis(dists, idx[:, None], axis=1).squeeze(1)
    return dists[:, k - 1]


def _knn_kl_divergence(
    p_vecs: np.ndarray,
    q_vecs: np.ndarray,
    p_index: "faiss.Index",
    q_index: "faiss.Index",
    metric: str,
    k: int,
    eps: float,
) -> float:
    if p_vecs.size == 0 or q_vecs.size == 0:
        return float("nan")
    n = p_vecs.shape[0]
    m = q_vecs.shape[0]
    if n <= 1 or m < 1 or k < 1:
        return float("nan")
    rho = _kth_neighbor_distances(p_index, p_vecs, metric, k, exclude_self=True)
    nu = _kth_neighbor_distances(q_index, p_vecs, metric, k, exclude_self=False)
    if rho is None or nu is None:
        return float("nan")
    mask = np.isfinite(rho) & np.isfinite(nu)
    if not np.any(mask):
        return float("nan")
    rho = np.maximum(rho[mask], eps)
    nu = np.maximum(nu[mask], eps)
    n_eff = rho.size
    if n_eff <= 1:
        return float("nan")
    dim = p_vecs.shape[1]
    return float((dim / n_eff) * np.sum(np.log(nu / rho)) + np.log(m / (n_eff - 1)))


def _self_radius(
    index: "faiss.Index",
    vectors: np.ndarray,
    metric: str,
    quantile: float,
    k: int = 1,
    agg: str = "first",
    batch_size: Optional[int] = None,
) -> float:
    if vectors.shape[0] < 2:
        return float("nan")
    k = max(int(k), 1)
    search_k = min(vectors.shape[0], k + 1)
    if search_k <= 1:
        return float("nan")
    
    # Auto-adjust batch size based on dimensionality
    if batch_size is None:
        dim = vectors.shape[1]
        if dim <= 4:
            batch_size = 500000  # Flow vectors (4D)
        elif dim <= 64:
            batch_size = 100000  # Medium dimensions
        elif dim <= 256:
            batch_size = 50000   # High dimensions (ResNet/DINO after PCA)
        else:
            batch_size = 20000   # Very high dimensions
    
    # Batch search for GPU indices to avoid OOM
    is_gpu = _is_gpu_index(index)
    if is_gpu and len(vectors) > batch_size:
        print(f"    Computing self-radius: {len(vectors):,} vectors in batches of {batch_size:,}")
        all_dists = []
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            batch_dists, _ = index.search(batch, search_k)
            all_dists.append(batch_dists)
            if (i // batch_size + 1) % 5 == 0:  # Print every 5 batches (was 10)
                print(f"      Processed {i+len(batch):,}/{len(vectors):,} vectors")
        dists = np.vstack(all_dists)
    else:
        dists, _ = index.search(vectors, search_k)
    
    if metric == "cosine":
        dists = 1.0 - dists
    else:
        dists = np.sqrt(np.maximum(dists, 0.0))
    neigh_dists = dists[:, 1:]
    if neigh_dists.size == 0:
        return float("nan")
    if agg in ("first", "min"):
        sample = neigh_dists[:, 0]
    elif agg in ("kth", "last", "max"):
        sample = neigh_dists[:, -1]
    elif agg == "mean":
        sample = neigh_dists.mean(axis=1)
    elif agg == "median":
        sample = np.median(neigh_dists, axis=1)
    else:
        raise ValueError(f"Unsupported neighbor aggregation: {agg}")
    return float(np.quantile(sample, quantile))


def _sample_dataset_vectors(
    ds_config: dict,
    common_params: dict,
    dataset_overrides: dict,
    representation: str,
    encoder: Optional[BaseFeatureEncoder],
    batch_size: int,
    num_workers: int,
    sampling_cfg: dict,
    rng: np.random.Generator,
    cache_cfg: Optional[dict] = None,
) -> Dict[str, object]:
    is_mixed = ds_config.get("mixed", False) or "datasets" in ds_config
    split = ds_config["split"]
    default_num_batches = sampling_cfg.get("batch_limit")
    num_batches = ds_config.get("num_batches", default_num_batches)
    entry_overrides = ds_config.get("overrides", None)

    cache_cfg = cache_cfg or {}
    cache_dir = cache_cfg.get("dir")
    cache_mode = str(cache_cfg.get("mode", "off")).lower()
    cache_format = str(cache_cfg.get("format", "npy")).lower()
    cache_dtype = cache_cfg.get("dtype")
    cache_mmap = bool(cache_cfg.get("mmap", False))
    mix_from_cache = bool(cache_cfg.get("mix_from_cache", False))

    if is_mixed:
        datasets_list = ds_config.get("datasets", [])
        percentages = ds_config.get("percentages", [])
        label = ds_config.get("name")
        if not label:
            if len(percentages) == 2 and len(datasets_list) == 2:
                pct1 = int(percentages[0] * 100)
                pct2 = int(percentages[1] * 100)
                label = f"{datasets_list[0]}_{datasets_list[1]}_{pct1}_{pct2}"
            else:
                label = "+".join(datasets_list)
        if cache_dir and mix_from_cache and cache_mode in ("read", "read_write"):
            mixed_vectors = _mix_vectors_from_cache(
                datasets_list,
                percentages,
                split,
                representation,
                Path(cache_dir),
                rng,
                sampling_cfg.get("max_vectors"),
                cache_mmap,
            )
            if mixed_vectors is not None:
                print(f"    [{label}] loaded mixed vectors from cache")
                return {
                    "label": label,
                    "split": split,
                    "is_eval": bool(ds_config.get("is_eval", False)),
                    "representation": representation,
                    "vectors": mixed_vectors,
                }
        dataset = create_mixed_dataset_from_config(
            datasets_list,
            percentages,
            split,
            common_params,
            dataset_overrides,
            epoch_size=ds_config.get("epoch_size"),
            seed=ds_config.get("seed"),
        )
        has_synthetic = any(_is_synthetic_dataset(ds_name) for ds_name in datasets_list)
        workers = 0 if has_synthetic else num_workers
    else:
        label = ds_config["name"]
        dataset_name = ds_config.get("dataset_name", label)
        if cache_dir and cache_mode in ("read", "read_write"):
            cached_vectors = _load_cached_vectors(
                Path(cache_dir), label, split, representation, cache_mmap
            )
            if cached_vectors is not None:
                print(f"    [{label}] loaded vectors from cache")
                return {
                    "label": label,
                    "split": split,
                    "is_eval": bool(ds_config.get("is_eval", False)),
                    "representation": representation,
                    "vectors": np.asarray(cached_vectors),
                }
        dataset = create_dataset_from_config(
            dataset_name, split, common_params, dataset_overrides, entry_overrides
        )
        workers = 0 if _is_synthetic_dataset(dataset_name) else num_workers

    default_shuffle = bool(sampling_cfg.get("shuffle", False))
    shuffle = bool(ds_config.get("shuffle", default_shuffle))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
        pin_memory=False,
    )

    dense_threshold = sampling_cfg.get("dense_threshold")
    dense_fraction = sampling_cfg.get("dense_fraction", 0.05)
    dense_min_keep = sampling_cfg.get("dense_min_keep", 2000)
    max_vectors = sampling_cfg.get("max_vectors")
    flow_per_image_max = sampling_cfg.get("flow_per_image_max")

    collector = VectorCollector(max_vectors, rng)
    total_batches = None if num_batches is None else int(num_batches)
    for batch_idx, batch in enumerate(dataloader):
        if num_batches is not None and batch_idx >= num_batches:
            break
        if representation == "flow":
            if flow_per_image_max is not None:
                vectors = _extract_flow_vectors_per_image(batch, int(flow_per_image_max), rng)
            else:
                vectors = extract_flow_vectors_from_batch(batch)
        else:
            if encoder is None:
                raise ValueError("Feature encoder is required for non-flow representations.")
            vectors = extract_features_from_batch(batch, encoder)

        if vectors is None or vectors.size == 0:
            continue

        vectors, dropped_non_finite, dropped_zero = _filter_vectors(
            vectors,
            drop_zero=(representation == "flow"),
            min_norm=0.0,
        )
        if dropped_non_finite or dropped_zero:
            print(
                f"    [{label}] filtered vectors:"
                f" non-finite={dropped_non_finite},"
                f" zero-norm={dropped_zero}"
            )
        if vectors is None or vectors.size == 0:
            continue

        vectors = _subsample_dense(
            vectors,
            dense_threshold,
            dense_fraction,
            dense_min_keep,
            rng,
        )
        collector.add(vectors)
        if (batch_idx + 1) % 5 == 0:
            print(
                f"    [{label}] batches={batch_idx + 1}"
                f"{'' if total_batches is None else f'/{total_batches}'}"
                f" vectors={collector.total}"
            )

    vectors = collector.finalize()
    if cache_dir and cache_mode in ("write", "read_write"):
        _save_cached_vectors(
            Path(cache_dir),
            label,
            split,
            representation,
            vectors,
            cache_format,
            cache_dtype,
        )
    print(f"    [{label}] done: vectors={vectors.shape[0]}")
    return {
        "label": label,
        "split": split,
        "is_eval": bool(ds_config.get("is_eval", False)),
        "representation": representation,
        "vectors": vectors,
    }


def _apply_pca(
    vectors_by_name: Dict[str, Dict[str, object]],
    pca_cfg: dict,
    metric: str,
) -> None:
    if not pca_cfg.get("enabled", False):
        return

    fit_on = pca_cfg.get("fit_on", "train")
    max_train = pca_cfg.get("max_train_vectors", 200000)
    output_dim = int(pca_cfg.get("output_dim", 256))

    fit_vectors = []
    for info in vectors_by_name.values():
        is_eval = info["is_eval"]
        if fit_on == "train" and is_eval:
            continue
        if fit_on == "eval" and not is_eval:
            continue
        fit_vectors.append(info["vectors"])

    if not fit_vectors:
        return

    train_vectors = np.concatenate(fit_vectors, axis=0)
    if max_train and train_vectors.shape[0] > max_train:
        rng = np.random.default_rng(0)
        idx = rng.choice(train_vectors.shape[0], size=max_train, replace=False)
        train_vectors = train_vectors[idx]

    dim = train_vectors.shape[1]
    if output_dim >= dim:
        return

    pca = faiss.PCAMatrix(dim, output_dim, eigen_power=-0.5 if pca_cfg.get("whiten") else 0.0)
    pca.train(train_vectors.astype(np.float32, copy=False))

    for info in vectors_by_name.values():
        vecs = info["vectors"]
        if vecs.size == 0:
            continue
        info["vectors"] = pca.apply_py(vecs.astype(np.float32, copy=False))


def _run_pairwise_cached(
    datasets_config: list,
    representation: str,
    encoder: Optional[BaseFeatureEncoder],
    batch_size: int,
    num_workers: int,
    sampling_cfg: dict,
    cache_cfg: dict,
    common_params: dict,
    dataset_overrides: dict,
    pca_cfg: dict,
    norm_cfg: dict,
    metric: str,
    index_factory: str,
    use_gpu: bool,
    nprobe: Optional[int],
    support_mode: str,
    radius_quantile: float,
    k: int,
    neighbor_agg: str,
    self_radius_k: int,
    kl_enabled: bool,
    kl_method: str,
    kl_knn_k: int,
    kl_bins: int,
    kl_eps: float,
    extra_kl_variants: bool,
    rng: np.random.Generator,
    output_file: str,
) -> dict:
    cache_dir = cache_cfg.get("dir")
    cache_mode = str(cache_cfg.get("mode", "off")).lower()
    cache_mmap = bool(cache_cfg.get("mmap", False))
    mix_from_cache = bool(cache_cfg.get("mix_from_cache", False))

    if not cache_dir:
        raise ValueError("cache.dir must be set when cache.pairwise is true.")

    cache_path = Path(cache_dir)

    specs = []
    for ds_config in datasets_config:
        label = _resolve_label(ds_config)
        split = ds_config["split"]
        ds_repr = ds_config.get("representation", representation)
        is_mixed = ds_config.get("mixed", False) or "datasets" in ds_config
        specs.append(
            {
                "config": ds_config,
                "label": label,
                "split": split,
                "representation": ds_repr,
                "is_eval": bool(ds_config.get("is_eval", False)),
                "is_mixed": is_mixed,
                "datasets_list": ds_config.get("datasets", []),
                "percentages": ds_config.get("percentages", []),
            }
        )

    # Cache all vectors first if in write mode
    if cache_mode in ("write", "read_write"):
        for spec in specs:
            if spec["is_mixed"]:
                continue
            if _cache_exists(cache_path, spec["label"], spec["split"], spec["representation"]):
                continue
            print(f"Caching {spec['label']} ({spec['split']})...")
            _sample_dataset_vectors(
                spec["config"],
                common_params,
                dataset_overrides,
                spec["representation"],
                encoder,
                batch_size,
                num_workers,
                sampling_cfg,
                rng,
                cache_cfg,
            )
            gc.collect()
    
    # Define load_vectors function for loading/caching vectors
    def load_vectors(spec: dict) -> Optional[np.ndarray]:
        if spec["is_mixed"]:
            if mix_from_cache:
                return _mix_vectors_from_cache(
                    spec["datasets_list"],
                    spec["percentages"],
                    spec["split"],
                    spec["representation"],
                    cache_path,
                    rng,
                    sampling_cfg.get("max_vectors"),
                    cache_mmap,
                )
            info = _sample_dataset_vectors(
                spec["config"],
                common_params,
                dataset_overrides,
                spec["representation"],
                encoder,
                batch_size,
                num_workers,
                sampling_cfg,
                rng,
                cache_cfg,
            )
            return info["vectors"]

        if cache_mode in ("read", "read_write"):
            cached = _load_cached_vectors(
                cache_path,
                spec["label"],
                spec["split"],
                spec["representation"],
                cache_mmap,
            )
            if cached is not None:
                return np.asarray(cached)
            if cache_mode == "read":
                raise FileNotFoundError(
                    f"Missing cache for {spec['label']} ({spec['split']}) in {cache_path}"
                )

        info = _sample_dataset_vectors(
            spec["config"],
            common_params,
            dataset_overrides,
            spec["representation"],
            encoder,
            batch_size,
            num_workers,
            sampling_cfg,
            rng,
            cache_cfg,
        )
        return info["vectors"]

    # Pre-load all vectors to ensure encoder is no longer needed
    # This allows us to delete the encoder before FAISS operations
    print("\n  Pre-loading all vectors...")
    all_vectors_loaded = {}
    for spec in specs:
        key = f"{spec['label']}_{spec['split']}"
        vectors = load_vectors(spec)
        if vectors is not None:
            all_vectors_loaded[key] = vectors
            print(f"    Loaded {vectors.shape[0]} vectors for {key}")
    
    # Delete encoder after all vectors are loaded to free GPU memory
    if encoder is not None:
        print("\n  Deleting encoder model to free GPU memory...")
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("  Encoder deleted and memory freed.")
    
    # Replace load_vectors to use pre-loaded vectors
    original_load_vectors = load_vectors
    def load_vectors(spec: dict) -> Optional[np.ndarray]:
        key = f"{spec['label']}_{spec['split']}"
        return all_vectors_loaded.get(key)

    train_specs = [s for s in specs if not s["is_eval"]]
    eval_specs = [s for s in specs if s["is_eval"]]

    norm_mode = str(norm_cfg.get("mode", "none")).strip().lower()
    norm_apply_to = norm_cfg.get("apply_to", None)
    if isinstance(norm_apply_to, str):
        norm_apply_to = [norm_apply_to]
    norm_eps = float(norm_cfg.get("eps", 1e-6))

    apply_l2 = metric == "cosine"
    if norm_mode == "train_zscore" and metric == "cosine":
        print("Warning: train_zscore with cosine metric disables L2 normalization.")
        apply_l2 = False
    
    # Compute global alpha if using global_block_alpha mode
    global_alpha = None
    if norm_mode == "global_block_alpha":
        if metric == "cosine":
            print("Warning: global_block_alpha with cosine metric disables L2 normalization.")
            apply_l2 = False
        
        print("\n  Setting up global α for block balancing...")
        
        # Collect train specs that need alpha
        train_labels_for_alpha = []
        representation_for_alpha = None
        for train_spec in train_specs:
            should_apply = not norm_apply_to or train_spec["representation"] in norm_apply_to
            if should_apply:
                train_labels_for_alpha.append(train_spec["label"])
                if representation_for_alpha is None:
                    representation_for_alpha = train_spec["representation"]
        
        if train_labels_for_alpha and representation_for_alpha:
            # Try to load cached alpha
            global_alpha = _load_global_alpha(
                cache_path if cache_dir else None,
                representation_for_alpha,
                train_labels_for_alpha,
                pos_dims=(0, 1),
                flow_dims=(2, 3)
            )
            
            # Compute if not cached
            if global_alpha is None:
                print("  Computing α from train set statistics...")
                train_vectors_for_alpha = {}
                for train_spec in train_specs:
                    if train_spec["label"] in train_labels_for_alpha:
                        train_raw = load_vectors(train_spec)
                        if train_raw is not None and train_raw.size > 0 and train_raw.shape[1] >= 4:
                            train_vectors_for_alpha[train_spec["label"]] = np.asarray(train_raw)
                
                if train_vectors_for_alpha:
                    global_alpha = _compute_global_alpha(
                        train_vectors_for_alpha,
                        pos_dims=(0, 1),
                        flow_dims=(2, 3),
                        eps=norm_eps
                    )
                    # Save to cache
                    if cache_dir:
                        _save_global_alpha(
                            cache_path,
                            representation_for_alpha,
                            global_alpha,
                            train_labels_for_alpha,
                            pos_dims=(0, 1),
                            flow_dims=(2, 3)
                        )
                else:
                    global_alpha = 1.0
                    print("  Warning: No suitable train vectors for alpha computation, using α=1.0")
        else:
            global_alpha = 1.0
            print("  Warning: No train sets found for alpha computation, using α=1.0")

    # Setup incremental CSV writing for checkpointing
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing results if resuming
    existing_results = {}
    if output_path.exists():
        print(f"\n⚠️  Found existing results file: {output_path}")
        print(f"   Loading to resume from checkpoint...")
        import csv
        with output_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["dataset1"], row["split1"], row["dataset2"], row["split2"])
                existing_results[key] = row
        print(f"   Loaded {len(existing_results)} existing results")
        print(f"   Will skip already-computed pairs\n")
    
    results = []
    total_pairs = len(train_specs) * len(eval_specs)
    computed_pairs = 0
    skipped_pairs = 0
    pair_idx = 0
    
    for train_spec in train_specs:
        train_raw = load_vectors(train_spec)
        if train_raw is None or train_raw.size == 0:
            continue

        train_vecs = np.asarray(train_raw)

        for eval_spec in eval_specs:
            eval_raw = load_vectors(eval_spec)
            if eval_raw is None or eval_raw.size == 0:
                continue

            eval_vecs = np.asarray(eval_raw)
            
            pair_idx += 1
            
            # Check if this pair was already computed
            pair_key = (train_spec["label"], train_spec["split"], 
                       eval_spec["label"], eval_spec["split"])
            if pair_key in existing_results:
                skipped_pairs += 1
                print(f"\n[Pair {pair_idx}/{total_pairs}] ✓ Skipping (already computed): "
                      f"{train_spec['label']} -> {eval_spec['label']}")
                continue
            
            print(f"\n{'='*80}")
            print(f"Processing pair {pair_idx}/{total_pairs}: {train_spec['label']} -> {eval_spec['label']}")
            print(f"  Train: {train_vecs.shape[0]:,} vectors")
            print(f"  Eval: {eval_vecs.shape[0]:,} vectors")
            print(f"  Progress: {computed_pairs} computed, {skipped_pairs} skipped")
            print(f"{'='*80}")

            train_proc, eval_proc = _apply_pairwise_pca(
                train_vecs, eval_vecs, pca_cfg, rng
            )

            # Apply normalization based on mode
            do_zscore = norm_mode == "train_zscore" and (
                not norm_apply_to or train_spec["representation"] in norm_apply_to
            )
            do_global_alpha = norm_mode == "global_block_alpha" and (
                not norm_apply_to or train_spec["representation"] in norm_apply_to
            )
            
            if do_zscore:
                mean, std = _compute_zscore_params(train_proc, norm_eps)
                train_proc = _apply_zscore(train_proc, mean, std)
                eval_proc = _apply_zscore(eval_proc, mean, std)
            elif do_global_alpha and global_alpha is not None:
                train_proc = _apply_global_alpha(train_proc, global_alpha)
                eval_proc = _apply_global_alpha(eval_proc, global_alpha)

            if apply_l2:
                train_proc = _l2_normalize(train_proc)
                eval_proc = _l2_normalize(eval_proc)

            # Create shared GPU resources if using GPU
            # This prevents memory fragmentation from creating multiple resources
            gpu_resources = None
            if use_gpu and faiss.get_num_gpus() > 0:
                print(f"  [1/5] Setting up shared GPU resources")
                gpu_resources = faiss.StandardGpuResources()
                # Optimized temp memory: 18GB for faster searches (6GB headroom for indices)
                temp_memory_gb = 18
                gpu_resources.setTempMemory(temp_memory_gb * 1024 * 1024 * 1024)
                # Batch size is auto-adjusted based on vector dimensionality
                dim = train_proc.shape[1]
                if dim <= 4:
                    batch_info = "500k (4D flow)"
                elif dim <= 64:
                    batch_info = "100k (64D)"
                elif dim <= 256:
                    batch_info = "50k (256D features)"
                else:
                    batch_info = f"20k ({dim}D)"
                print(f"        GPU temp memory: {temp_memory_gb}GB, batch size: {batch_info}")
            
            # Build indices directly with GPU (or CPU if not using GPU)
            print(f"  [2/5] Building FAISS indices...")
            train_idx = _build_index(
                train_proc.astype(np.float32, copy=False),
                index_factory,
                metric,
                use_gpu=use_gpu,
                nprobe=nprobe,
                gpu_resources=gpu_resources,
            )
            eval_idx = _build_index(
                eval_proc.astype(np.float32, copy=False),
                index_factory,
                metric,
                use_gpu=use_gpu,
                nprobe=nprobe,
                gpu_resources=gpu_resources,
            )

            print(f"  [3/5] Computing coverage metrics...")
            if support_mode == "per_point_radius":
                # Determine normalization mode for cache keys
                # train_zscore: eval depends on which train (train-specific cache)
                # global_block_alpha: same normalization for all (global cache)
                # none: no normalization
                cache_norm_mode = "none"
                cache_norm_by_label = None
                
                if do_zscore:
                    cache_norm_mode = "train_zscore"
                    cache_norm_by_label = None  # For train (self-normalized)
                elif do_global_alpha:
                    cache_norm_mode = "global_block_alpha"
                    cache_norm_by_label = None  # Global α same for everyone
                
                # Try to load cached train radii
                train_radii = None
                if cache_dir:
                    train_radii = _load_cached_radius(
                        cache_path, train_spec["label"], train_spec["split"],
                        train_spec["representation"], metric, support_mode,
                        self_radius_k, radius_quantile, neighbor_agg,
                        norm_mode=cache_norm_mode,
                        norm_by_label=None  # Train always self-normalized
                    )
                    if train_radii is not None:
                        print(f"    Loaded cached train radii")
                
                if train_radii is None:
                    train_radii = _pointwise_radius(
                        train_idx,
                        train_proc,
                        metric,
                        self_radius_k,
                        agg=neighbor_agg,
                    )
                    if cache_dir:
                        _save_cached_radius(
                            cache_path, train_spec["label"], train_spec["split"],
                            train_spec["representation"], metric, support_mode,
                            self_radius_k, radius_quantile, neighbor_agg, train_radii,
                            norm_mode=cache_norm_mode,
                            norm_by_label=None  # Train always self-normalized
                        )
                        print(f"    Cached train radii for future runs")
                
                dist_eval_to_train, idx_eval_to_train = _nn_first(train_idx, eval_proc, metric)
                
                # Try to load cached eval radii
                # For train_zscore: eval depends on train set (train-specific)
                # For global_block_alpha: eval has same α as everyone (globally cached)
                eval_radii = None
                eval_cache_norm_by = train_spec["label"] if do_zscore else None
                if cache_dir:
                    eval_radii = _load_cached_radius(
                        cache_path, eval_spec["label"], eval_spec["split"],
                        eval_spec["representation"], metric, support_mode,
                        self_radius_k, radius_quantile, neighbor_agg,
                        norm_mode=cache_norm_mode,
                        norm_by_label=eval_cache_norm_by
                    )
                    if eval_radii is not None:
                        if do_zscore:
                            print(f"    Loaded cached eval radii (normalized by {train_spec['label']})")
                        else:
                            print(f"    Loaded cached eval radii")
                
                if eval_radii is None:
                    eval_radii = _pointwise_radius(
                        eval_idx,
                        eval_proc,
                        metric,
                        self_radius_k,
                        agg=neighbor_agg,
                    )
                    if cache_dir:
                        _save_cached_radius(
                            cache_path, eval_spec["label"], eval_spec["split"],
                            eval_spec["representation"], metric, support_mode,
                            self_radius_k, radius_quantile, neighbor_agg, eval_radii,
                            norm_mode=cache_norm_mode,
                            norm_by_label=eval_cache_norm_by
                        )
                        if do_zscore:
                            print(f"    Cached eval radii for future runs (normalized by {train_spec['label']})")
                        else:
                            print(f"    Cached eval radii for future runs")
                
                dist_train_to_eval, idx_train_to_eval = _nn_first(eval_idx, train_proc, metric)

                mask_eval = (
                    np.isfinite(train_radii[idx_eval_to_train])
                    if idx_eval_to_train.size
                    else np.array([])
                )
                mask_train = (
                    np.isfinite(eval_radii[idx_train_to_eval])
                    if idx_train_to_eval.size
                    else np.array([])
                )

                recall = (
                    float(np.mean(dist_eval_to_train[mask_eval] <= train_radii[idx_eval_to_train][mask_eval]))
                    if mask_eval.size
                    else float("nan")
                )
                precision = (
                    float(np.mean(dist_train_to_eval[mask_train] <= eval_radii[idx_train_to_eval][mask_train]))
                    if mask_train.size
                    else float("nan")
                )

                radius_train = float(np.nanmedian(train_radii)) if train_radii.size else float("nan")
                radius_eval = float(np.nanmedian(eval_radii)) if eval_radii.size else float("nan")
            else:
                # Global radius mode
                # Determine normalization mode for cache keys
                cache_norm_mode = "none"
                if do_zscore:
                    cache_norm_mode = "train_zscore"
                elif do_global_alpha:
                    cache_norm_mode = "global_block_alpha"
                
                dist_eval_to_train = _nn_distances(train_idx, eval_proc, metric, k, agg=neighbor_agg)
                
                # Try to load cached train radius (train normalizes itself)
                cached_train_radius = None
                if cache_dir:
                    cached = _load_cached_radius(
                        cache_path, train_spec["label"], train_spec["split"],
                        train_spec["representation"], metric, support_mode,
                        self_radius_k, radius_quantile, neighbor_agg,
                        norm_mode=cache_norm_mode,
                        norm_by_label=None  # Train normalizes itself
                    )
                    if cached is not None:
                        cached_train_radius = float(cached.item() if hasattr(cached, 'item') else cached)
                        print(f"    Loaded cached train radius: {cached_train_radius:.6f}")
                
                if cached_train_radius is not None:
                    radius_train = cached_train_radius
                else:
                    radius_train = _self_radius(
                        train_idx,
                        train_proc,
                        metric,
                        radius_quantile,
                        k=self_radius_k,
                        agg=neighbor_agg,
                    )
                    if cache_dir:
                        _save_cached_radius(
                            cache_path, train_spec["label"], train_spec["split"],
                            train_spec["representation"], metric, support_mode,
                            self_radius_k, radius_quantile, neighbor_agg,
                            np.array([radius_train]),
                            norm_mode=cache_norm_mode,
                            norm_by_label=None  # Train normalizes itself
                        )
                        print(f"    Cached train radius for future runs")
                
                dist_train_to_eval = _nn_distances(eval_idx, train_proc, metric, k, agg=neighbor_agg)
                
                # Try to load cached eval radius
                # For train_zscore: eval depends on train set (train-specific)
                # For global_block_alpha: eval has same α as everyone (globally cached)
                cached_eval_radius = None
                eval_cache_norm_by = train_spec["label"] if do_zscore else None
                if cache_dir:
                    cached = _load_cached_radius(
                        cache_path, eval_spec["label"], eval_spec["split"],
                        eval_spec["representation"], metric, support_mode,
                        self_radius_k, radius_quantile, neighbor_agg,
                        norm_mode=cache_norm_mode,
                        norm_by_label=eval_cache_norm_by
                    )
                    if cached is not None:
                        cached_eval_radius = float(cached.item() if hasattr(cached, 'item') else cached)
                        if do_zscore:
                            print(f"    Loaded cached eval radius: {cached_eval_radius:.6f} (normalized by {train_spec['label']})")
                        else:
                            print(f"    Loaded cached eval radius: {cached_eval_radius:.6f}")
                
                if cached_eval_radius is not None:
                    radius_eval = cached_eval_radius
                else:
                    radius_eval = _self_radius(
                        eval_idx,
                        eval_proc,
                        metric,
                        radius_quantile,
                        k=self_radius_k,
                        agg=neighbor_agg,
                    )
                    if cache_dir:
                        _save_cached_radius(
                            cache_path, eval_spec["label"], eval_spec["split"],
                            eval_spec["representation"], metric, support_mode,
                            self_radius_k, radius_quantile, neighbor_agg,
                            np.array([radius_eval]),
                            norm_mode=cache_norm_mode,
                            norm_by_label=eval_cache_norm_by
                        )
                        if do_zscore:
                            print(f"    Cached eval radius for future runs (normalized by {train_spec['label']})")
                        else:
                            print(f"    Cached eval radius for future runs")
                recall = (
                    float(np.mean(dist_eval_to_train <= radius_train))
                    if np.isfinite(radius_train) and dist_eval_to_train.size
                    else float("nan")
                )
                precision = (
                    float(np.mean(dist_train_to_eval <= radius_eval))
                    if np.isfinite(radius_eval) and dist_train_to_eval.size
                    else float("nan")
                )

            print(f"  [4/5] Computing KL divergence...")
            kl_eval_to_train = float("nan")
            kl_train_to_eval = float("nan")
            if kl_enabled:
                if kl_method == "knn":
                    kl_eval_to_train = _knn_kl_divergence(
                        eval_proc,
                        train_proc,
                        eval_idx,
                        train_idx,
                        metric,
                        kl_knn_k,
                        max(kl_eps, 1e-12),
                    )
                    kl_train_to_eval = _knn_kl_divergence(
                        train_proc,
                        eval_proc,
                        train_idx,
                        eval_idx,
                        metric,
                        kl_knn_k,
                        max(kl_eps, 1e-12),
                    )
                elif kl_method == "hist" and kl_bins >= 2:
                    kl_eval_to_train = _kl_divergence_from_samples(
                        dist_eval_to_train,
                        dist_train_to_eval,
                        kl_bins,
                        max(kl_eps, 1e-12),
                    )
                    kl_train_to_eval = _kl_divergence_from_samples(
                        dist_train_to_eval,
                        dist_eval_to_train,
                        kl_bins,
                        max(kl_eps, 1e-12),
                    )
                else:
                    raise ValueError(f"Unsupported kl_method: {kl_method}")

            extra_kl = {}
            if extra_kl_variants:
                eps = max(kl_eps, 1e-12)
                extra_kl["kl_eval_to_train_hist"] = _kl_divergence_from_samples(
                    dist_eval_to_train,
                    dist_train_to_eval,
                    kl_bins,
                    eps,
                )
                extra_kl["kl_train_to_eval_hist"] = _kl_divergence_from_samples(
                    dist_train_to_eval,
                    dist_eval_to_train,
                    kl_bins,
                    eps,
                )

            result = {
                "dataset1": train_spec["label"],
                "split1": train_spec["split"],
                "dataset2": eval_spec["label"],
                "split2": eval_spec["split"],
                "representation": train_spec["representation"],
                "support_mode": support_mode,
                "normalization_mode": norm_mode,
                "k": k,
                "neighbor_agg": neighbor_agg,
                "self_radius_k": self_radius_k,
                "radius_quantile": radius_quantile,
                "radius_train": radius_train,
                "radius_eval": radius_eval,
                "recall": recall,
                "precision": precision,
                "outside": 1.0 - precision if np.isfinite(precision) else float("nan"),
                "train_to_eval_coverage": recall,
                "eval_to_train_coverage": precision,
                "mean_nn_eval_to_train": float(np.mean(dist_eval_to_train)) if dist_eval_to_train.size else float("nan"),
                "median_nn_eval_to_train": float(np.median(dist_eval_to_train)) if dist_eval_to_train.size else float("nan"),
                "p90_nn_eval_to_train": float(np.quantile(dist_eval_to_train, 0.9)) if dist_eval_to_train.size else float("nan"),
                "mean_nn_train_to_eval": float(np.mean(dist_train_to_eval)) if dist_train_to_eval.size else float("nan"),
                "median_nn_train_to_eval": float(np.median(dist_train_to_eval)) if dist_train_to_eval.size else float("nan"),
                "p90_nn_train_to_eval": float(np.quantile(dist_train_to_eval, 0.9)) if dist_train_to_eval.size else float("nan"),
                "kl_method": kl_method,
                "kl_knn_k": int(kl_knn_k),
                "kl_bins": int(kl_bins),
                "kl_eps": float(kl_eps),
                "kl_eval_to_train": kl_eval_to_train,
                "kl_train_to_eval": kl_train_to_eval,
                "n_train_vectors": int(train_proc.shape[0]),
                "n_eval_vectors": int(eval_proc.shape[0]),
            }
            if extra_kl:
                result.update(extra_kl)
            
            computed_pairs += 1
            
            print(f"  [5/5] Results computed!")
            print(
                f"  ✓ [{train_spec['label']} -> {eval_spec['label']}] "
                f"Recall={recall:.3f} Precision={precision:.3f}"
            )
            
            # Save incrementally after each pair (checkpoint)
            import csv
            write_header = not output_path.exists() or output_path.stat().st_size == 0
            with output_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(result.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(result)
            print(f"  💾 Checkpoint saved ({computed_pairs}/{total_pairs - skipped_pairs} new pairs)")

            # Clean up eval data and index
            del eval_raw, eval_vecs, eval_proc, eval_idx
            gc.collect()

        # Clean up train data, indices, and GPU resources
        del train_raw, train_vecs, train_proc, train_idx
        if gpu_resources is not None:
            del gpu_resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Return summary info (results already saved incrementally)
    return {
        "total_pairs": total_pairs,
        "computed": computed_pairs,
        "skipped": skipped_pairs,
        "output_file": str(output_path)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FAISS-based coverage metrics.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to FAISS coverage config YAML.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())

    representation = config.get("representation", "flow")
    encoder_name = config.get("encoder", "resnet101")
    batch_size = int(config.get("batch_size", 8))
    num_workers = int(config.get("num_workers", 4))

    sampling_cfg = config.get("sampling", {})
    rng = np.random.default_rng(int(sampling_cfg.get("seed", 42)))

    faiss_cfg = config.get("faiss", {})
    index_factory = faiss_cfg.get("index_factory", "HNSW32")
    metric = faiss_cfg.get("metric", "l2")
    use_gpu = bool(faiss_cfg.get("use_gpu", False))
    nprobe = faiss_cfg.get("nprobe")

    coverage_cfg = config.get("coverage", {})
    support_mode = str(coverage_cfg.get("support_mode", "global_radius")).strip().lower()
    radius_quantile = float(coverage_cfg.get("radius_quantile", 0.95))
    k = int(coverage_cfg.get("k", 1))
    neighbor_agg = str(coverage_cfg.get("neighbor_agg", "first"))
    self_radius_k = int(coverage_cfg.get("self_radius_k", 1))
    if support_mode == "per_point_radius" and neighbor_agg in ("first", "min"):
        neighbor_agg = "kth"
    kl_bins = int(coverage_cfg.get("kl_bins", 50))
    kl_eps = float(coverage_cfg.get("kl_eps", 1e-8))
    kl_knn_k = int(coverage_cfg.get("kl_knn_k", 5))
    kl_method = str(coverage_cfg.get("kl_method", "")).strip().lower()
    if not kl_method:
        kl_method = "knn" if "kl_knn_k" in coverage_cfg else "hist"
    kl_enabled = kl_method not in ("none", "off", "false")
    extra_kl_variants = bool(coverage_cfg.get("extra_kl_variants", True))

    output_cfg = config.get("output", {})
    output_file = output_cfg.get("results_file", "coverage_faiss_results.csv")
    cache_cfg = config.get("cache", {})
    norm_cfg = config.get("normalization", {})

    datasets_config = config.get("datasets", [])
    if not datasets_config:
        raise ValueError("No datasets specified in config.")

    if cache_cfg.get("mix_from_cache"):
        base_datasets = []
        mixed_datasets = []
        for ds in datasets_config:
            if ds.get("mixed", False) or "datasets" in ds:
                mixed_datasets.append(ds)
            else:
                base_datasets.append(ds)
        datasets_config = base_datasets + mixed_datasets

    common_params = config.get("dataset_params", {})
    dataset_overrides = config.get("dataset_overrides", {})

    encoder = None
    if representation != "flow":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = create_encoder(encoder_name, device)

    if cache_cfg.get("pairwise", False):
        summary = _run_pairwise_cached(
            datasets_config,
            representation,
            encoder,
            batch_size,
            num_workers,
            sampling_cfg,
            cache_cfg,
            common_params,
            dataset_overrides,
            config.get("pca", {}),
            norm_cfg,
            metric,
            index_factory,
            use_gpu,
            nprobe,
            support_mode,
            radius_quantile,
            k,
            neighbor_agg,
            self_radius_k,
            kl_enabled,
            kl_method,
            kl_knn_k,
            kl_bins,
            kl_eps,
            extra_kl_variants,
            rng,
            output_file,
        )
        
        # Results were saved incrementally during computation
        print(f"\n{'='*80}")
        print(f"✅ COVERAGE COMPUTATION COMPLETE!")
        print(f"{'='*80}")
        print(f"  Total pairs: {summary['total_pairs']}")
        print(f"  Computed: {summary['computed']}")
        print(f"  Skipped (resumed): {summary['skipped']}")
        print(f"  Output file: {summary['output_file']}")
        print(f"{'='*80}\n")
        return

    vectors_by_name: Dict[str, Dict[str, object]] = {}
    for ds_config in datasets_config:
        ds_repr = ds_config.get("representation", representation)
        info = _sample_dataset_vectors(
            ds_config,
            common_params,
            dataset_overrides,
            ds_repr,
            encoder,
            batch_size,
            num_workers,
            sampling_cfg,
            rng,
            cache_cfg,
        )
        label = info["label"]
        split = info["split"]
        key = f"{label}_{split}"
        vectors_by_name[key] = info
        print(f"  Collected {info['vectors'].shape[0]} vectors for {key}")

    # Delete encoder from memory after all vector extraction is complete
    # This frees GPU memory before FAISS operations
    if encoder is not None:
        print("\n  Deleting encoder model to free GPU memory...")
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("  Encoder deleted and memory freed.")

    _apply_pca(vectors_by_name, config.get("pca", {}), metric)

    norm_mode = str(norm_cfg.get("mode", "none")).strip().lower()
    norm_apply_to = norm_cfg.get("apply_to", None)
    if isinstance(norm_apply_to, str):
        norm_apply_to = [norm_apply_to]
    norm_eps = float(norm_cfg.get("eps", 1e-6))

    apply_l2 = metric == "cosine"
    if norm_mode == "train_zscore" and metric == "cosine":
        print("Warning: train_zscore with cosine metric disables L2 normalization.")
        apply_l2 = False

    if apply_l2:
        for info in vectors_by_name.values():
            vecs = info["vectors"]
            if vecs.size == 0:
                continue
            info["vectors"] = _l2_normalize(vecs)
    train_keys = [k for k, v in vectors_by_name.items() if not v["is_eval"]]
    eval_keys = [k for k, v in vectors_by_name.items() if v["is_eval"]]

    results = []
    if support_mode == "per_point_radius":
        # Create shared GPU resources for per_point_radius mode
        gpu_resources = None
        if use_gpu and faiss.get_num_gpus() > 0:
            print("\n  Setting up shared GPU resources for per_point_radius mode")
            gpu_resources = faiss.StandardGpuResources()
            temp_memory_gb = 18
            gpu_resources.setTempMemory(temp_memory_gb * 1024 * 1024 * 1024)
            print(f"  GPU temp memory: {temp_memory_gb}GB")
        
        for train_key in train_keys:
            train_info = vectors_by_name[train_key]
            train_vecs_raw = train_info["vectors"]
            if train_vecs_raw.size == 0:
                continue

            do_zscore = norm_mode == "train_zscore" and (
                not norm_apply_to or train_info["representation"] in norm_apply_to
            )
            if do_zscore:
                mean, std = _compute_zscore_params(train_vecs_raw, norm_eps)
                train_vecs = _apply_zscore(train_vecs_raw, mean, std)
            else:
                mean, std = None, None
                train_vecs = train_vecs_raw

            train_idx = _build_index(
                train_vecs.astype(np.float32, copy=False),
                index_factory,
                metric,
                use_gpu,
                nprobe,
                gpu_resources=gpu_resources,
            )
            train_radii = _pointwise_radius(
                train_idx,
                train_vecs,
                metric,
                self_radius_k,
                agg=neighbor_agg,
            )

            train_label = train_info["label"]
            train_split = train_info["split"]

            for eval_key in eval_keys:
                eval_info = vectors_by_name[eval_key]
                eval_vecs_raw = eval_info["vectors"]
                if eval_vecs_raw.size == 0:
                    continue

                if do_zscore and mean is not None and std is not None:
                    eval_vecs = _apply_zscore(eval_vecs_raw, mean, std)
                else:
                    eval_vecs = eval_vecs_raw

                eval_idx = _build_index(
                    eval_vecs.astype(np.float32, copy=False),
                    index_factory,
                    metric,
                    use_gpu,
                    nprobe,
                    gpu_resources=gpu_resources,
                )
                eval_radii = _pointwise_radius(
                    eval_idx,
                    eval_vecs,
                    metric,
                    self_radius_k,
                    agg=neighbor_agg,
                )

                dist_eval_to_train, idx_eval_to_train = _nn_first(train_idx, eval_vecs, metric)
                dist_train_to_eval, idx_train_to_eval = _nn_first(eval_idx, train_vecs, metric)

                mask_eval = np.isfinite(train_radii[idx_eval_to_train]) if idx_eval_to_train.size else np.array([])
                mask_train = np.isfinite(eval_radii[idx_train_to_eval]) if idx_train_to_eval.size else np.array([])

                recall = (
                    float(np.mean(dist_eval_to_train[mask_eval] <= train_radii[idx_eval_to_train][mask_eval]))
                    if mask_eval.size
                    else float("nan")
                )
                precision = (
                    float(np.mean(dist_train_to_eval[mask_train] <= eval_radii[idx_train_to_eval][mask_train]))
                    if mask_train.size
                    else float("nan")
                )

                kl_eval_to_train = float("nan")
                kl_train_to_eval = float("nan")
                if kl_enabled:
                    if kl_method == "knn":
                        kl_eval_to_train = _knn_kl_divergence(
                            eval_vecs,
                            train_vecs,
                            eval_idx,
                            train_idx,
                            metric,
                            kl_knn_k,
                            max(kl_eps, 1e-12),
                        )
                        kl_train_to_eval = _knn_kl_divergence(
                            train_vecs,
                            eval_vecs,
                            train_idx,
                            eval_idx,
                            metric,
                            kl_knn_k,
                            max(kl_eps, 1e-12),
                        )
                    elif kl_method == "hist" and kl_bins >= 2:
                        kl_eval_to_train = _kl_divergence_from_samples(
                            dist_eval_to_train,
                            dist_train_to_eval,
                            kl_bins,
                            max(kl_eps, 1e-12),
                        )
                        kl_train_to_eval = _kl_divergence_from_samples(
                            dist_train_to_eval,
                            dist_eval_to_train,
                            kl_bins,
                            max(kl_eps, 1e-12),
                        )
                    else:
                        raise ValueError(f"Unsupported kl_method: {kl_method}")

                extra_kl = {}
                radius_train_med = float(np.nanmedian(train_radii)) if train_radii.size else float("nan")
                radius_eval_med = float(np.nanmedian(eval_radii)) if eval_radii.size else float("nan")
                if extra_kl_variants:
                    eps = max(kl_eps, 1e-12)
                    extra_kl["kl_eval_to_train_hist"] = _kl_divergence_from_samples(
                        dist_eval_to_train,
                        dist_train_to_eval,
                        kl_bins,
                        eps,
                    )
                    extra_kl["kl_train_to_eval_hist"] = _kl_divergence_from_samples(
                        dist_train_to_eval,
                        dist_eval_to_train,
                        kl_bins,
                        eps,
                    )

                    extra_kl["kl_eval_to_train_hist_log1p_linear"] = _kl_divergence_from_samples_linear(
                        _log1p_samples(dist_eval_to_train),
                        _log1p_samples(dist_train_to_eval),
                        kl_bins,
                        eps,
                    )
                    extra_kl["kl_train_to_eval_hist_log1p_linear"] = _kl_divergence_from_samples_linear(
                        _log1p_samples(dist_train_to_eval),
                        _log1p_samples(dist_eval_to_train),
                        kl_bins,
                        eps,
                    )

                    if (
                        np.isfinite(radius_train_med)
                        and radius_train_med > 0
                        and np.isfinite(radius_eval_med)
                        and radius_eval_med > 0
                    ):
                        dist_eval_norm = dist_eval_to_train / float(radius_train_med)
                        dist_train_norm = dist_train_to_eval / float(radius_eval_med)
                        extra_kl["kl_eval_to_train_hist_radius"] = _kl_divergence_from_samples(
                            dist_eval_norm,
                            dist_train_norm,
                            kl_bins,
                            eps,
                        )
                        extra_kl["kl_train_to_eval_hist_radius"] = _kl_divergence_from_samples(
                            dist_train_norm,
                            dist_eval_norm,
                            kl_bins,
                            eps,
                        )
                    else:
                        extra_kl["kl_eval_to_train_hist_radius"] = float("nan")
                        extra_kl["kl_train_to_eval_hist_radius"] = float("nan")

                    med_eval = float(np.median(dist_eval_to_train)) if dist_eval_to_train.size else float("nan")
                    med_train = float(np.median(dist_train_to_eval)) if dist_train_to_eval.size else float("nan")
                    if np.isfinite(med_eval) and med_eval > 0 and np.isfinite(med_train) and med_train > 0:
                        dist_eval_med = dist_eval_to_train / med_eval
                        dist_train_med = dist_train_to_eval / med_train
                        extra_kl["kl_eval_to_train_hist_median"] = _kl_divergence_from_samples(
                            dist_eval_med,
                            dist_train_med,
                            kl_bins,
                            eps,
                        )
                        extra_kl["kl_train_to_eval_hist_median"] = _kl_divergence_from_samples(
                            dist_train_med,
                            dist_eval_med,
                            kl_bins,
                            eps,
                        )
                    else:
                        extra_kl["kl_eval_to_train_hist_median"] = float("nan")
                        extra_kl["kl_train_to_eval_hist_median"] = float("nan")

                    rank_eval, rank_train = _rank_percentiles(
                        dist_eval_to_train, dist_train_to_eval
                    )
                    if rank_eval is not None and rank_train is not None:
                        extra_kl["kl_eval_to_train_hist_rank"] = _kl_divergence_from_samples_linear(
                            rank_eval,
                            rank_train,
                            kl_bins,
                            eps,
                        )
                        extra_kl["kl_train_to_eval_hist_rank"] = _kl_divergence_from_samples_linear(
                            rank_train,
                            rank_eval,
                            kl_bins,
                            eps,
                        )
                    else:
                        extra_kl["kl_eval_to_train_hist_rank"] = float("nan")
                        extra_kl["kl_train_to_eval_hist_rank"] = float("nan")

                result = {
                    "dataset1": train_label,
                    "split1": train_split,
                    "dataset2": eval_info["label"],
                    "split2": eval_info["split"],
                    "representation": train_info["representation"],
                    "support_mode": support_mode,
                    "normalization_mode": norm_mode,
                    "k": k,
                    "neighbor_agg": neighbor_agg,
                    "self_radius_k": self_radius_k,
                    "radius_quantile": radius_quantile,
                    "radius_train": radius_train_med,
                    "radius_eval": radius_eval_med,
                    "recall": recall,
                    "precision": precision,
                    "outside": 1.0 - precision if np.isfinite(precision) else float("nan"),
                    "train_to_eval_coverage": recall,
                    "eval_to_train_coverage": precision,
                    "mean_nn_eval_to_train": float(np.mean(dist_eval_to_train)) if dist_eval_to_train.size else float("nan"),
                    "median_nn_eval_to_train": float(np.median(dist_eval_to_train)) if dist_eval_to_train.size else float("nan"),
                    "p90_nn_eval_to_train": float(np.quantile(dist_eval_to_train, 0.9)) if dist_eval_to_train.size else float("nan"),
                    "mean_nn_train_to_eval": float(np.mean(dist_train_to_eval)) if dist_train_to_eval.size else float("nan"),
                    "median_nn_train_to_eval": float(np.median(dist_train_to_eval)) if dist_train_to_eval.size else float("nan"),
                    "p90_nn_train_to_eval": float(np.quantile(dist_train_to_eval, 0.9)) if dist_train_to_eval.size else float("nan"),
                    "kl_method": kl_method,
                    "kl_knn_k": int(kl_knn_k),
                    "kl_bins": int(kl_bins),
                    "kl_eps": float(kl_eps),
                    "kl_eval_to_train": kl_eval_to_train,
                    "kl_train_to_eval": kl_train_to_eval,
                    "n_train_vectors": int(train_vecs.shape[0]),
                    "n_eval_vectors": int(eval_vecs.shape[0]),
                }
                if extra_kl:
                    result.update(extra_kl)
                results.append(result)
                print(
                    f"[{train_label} -> {eval_info['label']}] "
                    f"train->eval={recall:.3f} eval->train={precision:.3f}"
                )
    else:
        # Create shared GPU resources for all indices
        gpu_resources = None
        if use_gpu and faiss.get_num_gpus() > 0:
            print("\n  Setting up shared GPU resources for all indices")
            gpu_resources = faiss.StandardGpuResources()
            temp_memory_gb = 18
            gpu_resources.setTempMemory(temp_memory_gb * 1024 * 1024 * 1024)
            print(f"  GPU temp memory: {temp_memory_gb}GB")
        
        indices = {}
        radii = {}
        for key, info in vectors_by_name.items():
            vecs = info["vectors"]
            if vecs.size == 0:
                indices[key] = None
                radii[key] = float("nan")
                continue
            index = _build_index(
                vecs.astype(np.float32, copy=False),
                index_factory,
                metric,
                use_gpu,
                nprobe,
                gpu_resources=gpu_resources,
            )
            indices[key] = index
            radii[key] = _self_radius(
                index,
                vecs,
                metric,
                radius_quantile,
                k=self_radius_k,
                agg=neighbor_agg,
            )

        for train_key in train_keys:
            train_info = vectors_by_name[train_key]
            train_vecs = train_info["vectors"]
            train_idx = indices[train_key]
            if train_idx is None or train_vecs.size == 0:
                continue
            train_label = train_info["label"]
            train_split = train_info["split"]

            for eval_key in eval_keys:
                eval_info = vectors_by_name[eval_key]
                eval_vecs = eval_info["vectors"]
                eval_idx = indices[eval_key]
                if eval_idx is None or eval_vecs.size == 0:
                    continue

                dist_eval_to_train = _nn_distances(train_idx, eval_vecs, metric, k, agg=neighbor_agg)
                dist_train_to_eval = _nn_distances(eval_idx, train_vecs, metric, k, agg=neighbor_agg)

                radius_train = radii.get(train_key, float("nan"))
                radius_eval = radii.get(eval_key, float("nan"))

                recall = (
                    float(np.mean(dist_eval_to_train <= radius_train))
                    if np.isfinite(radius_train) and dist_eval_to_train.size
                    else float("nan")
                )
                precision = (
                    float(np.mean(dist_train_to_eval <= radius_eval))
                    if np.isfinite(radius_eval) and dist_train_to_eval.size
                    else float("nan")
                )

                kl_eval_to_train = float("nan")
                kl_train_to_eval = float("nan")
                if kl_enabled:
                    if kl_method == "knn":
                        kl_eval_to_train = _knn_kl_divergence(
                            eval_vecs,
                            train_vecs,
                            eval_idx,
                            train_idx,
                            metric,
                            kl_knn_k,
                            max(kl_eps, 1e-12),
                        )
                        kl_train_to_eval = _knn_kl_divergence(
                            train_vecs,
                            eval_vecs,
                            train_idx,
                            eval_idx,
                            metric,
                            kl_knn_k,
                            max(kl_eps, 1e-12),
                        )
                    elif kl_method == "hist" and kl_bins >= 2:
                        kl_eval_to_train = _kl_divergence_from_samples(
                            dist_eval_to_train,
                            dist_train_to_eval,
                            kl_bins,
                            max(kl_eps, 1e-12),
                        )
                        kl_train_to_eval = _kl_divergence_from_samples(
                            dist_train_to_eval,
                            dist_eval_to_train,
                            kl_bins,
                            max(kl_eps, 1e-12),
                        )
                    else:
                        raise ValueError(f"Unsupported kl_method: {kl_method}")

                extra_kl = {}
                if extra_kl_variants:
                    eps = max(kl_eps, 1e-12)
                    extra_kl["kl_eval_to_train_hist"] = _kl_divergence_from_samples(
                        dist_eval_to_train,
                        dist_train_to_eval,
                        kl_bins,
                        eps,
                    )
                    extra_kl["kl_train_to_eval_hist"] = _kl_divergence_from_samples(
                        dist_train_to_eval,
                        dist_eval_to_train,
                        kl_bins,
                        eps,
                    )

                    extra_kl["kl_eval_to_train_hist_log1p_linear"] = _kl_divergence_from_samples_linear(
                        _log1p_samples(dist_eval_to_train),
                        _log1p_samples(dist_train_to_eval),
                        kl_bins,
                        eps,
                    )
                    extra_kl["kl_train_to_eval_hist_log1p_linear"] = _kl_divergence_from_samples_linear(
                        _log1p_samples(dist_train_to_eval),
                        _log1p_samples(dist_eval_to_train),
                        kl_bins,
                        eps,
                    )

                    if np.isfinite(radius_train) and radius_train > 0 and np.isfinite(radius_eval) and radius_eval > 0:
                        dist_eval_norm = dist_eval_to_train / float(radius_train)
                        dist_train_norm = dist_train_to_eval / float(radius_eval)
                        extra_kl["kl_eval_to_train_hist_radius"] = _kl_divergence_from_samples(
                            dist_eval_norm,
                            dist_train_norm,
                            kl_bins,
                            eps,
                        )
                        extra_kl["kl_train_to_eval_hist_radius"] = _kl_divergence_from_samples(
                            dist_train_norm,
                            dist_eval_norm,
                            kl_bins,
                            eps,
                        )
                    else:
                        extra_kl["kl_eval_to_train_hist_radius"] = float("nan")
                        extra_kl["kl_train_to_eval_hist_radius"] = float("nan")

                    med_eval = float(np.median(dist_eval_to_train)) if dist_eval_to_train.size else float("nan")
                    med_train = float(np.median(dist_train_to_eval)) if dist_train_to_eval.size else float("nan")
                    if np.isfinite(med_eval) and med_eval > 0 and np.isfinite(med_train) and med_train > 0:
                        dist_eval_med = dist_eval_to_train / med_eval
                        dist_train_med = dist_train_to_eval / med_train
                        extra_kl["kl_eval_to_train_hist_median"] = _kl_divergence_from_samples(
                            dist_eval_med,
                            dist_train_med,
                            kl_bins,
                            eps,
                        )
                        extra_kl["kl_train_to_eval_hist_median"] = _kl_divergence_from_samples(
                            dist_train_med,
                            dist_eval_med,
                            kl_bins,
                            eps,
                        )
                    else:
                        extra_kl["kl_eval_to_train_hist_median"] = float("nan")
                        extra_kl["kl_train_to_eval_hist_median"] = float("nan")

                    rank_eval, rank_train = _rank_percentiles(
                        dist_eval_to_train, dist_train_to_eval
                    )
                    if rank_eval is not None and rank_train is not None:
                        extra_kl["kl_eval_to_train_hist_rank"] = _kl_divergence_from_samples_linear(
                            rank_eval,
                            rank_train,
                            kl_bins,
                            eps,
                        )
                        extra_kl["kl_train_to_eval_hist_rank"] = _kl_divergence_from_samples_linear(
                            rank_train,
                            rank_eval,
                            kl_bins,
                            eps,
                        )
                    else:
                        extra_kl["kl_eval_to_train_hist_rank"] = float("nan")
                        extra_kl["kl_train_to_eval_hist_rank"] = float("nan")

                result = {
                    "dataset1": train_label,
                    "split1": train_split,
                    "dataset2": eval_info["label"],
                    "split2": eval_info["split"],
                    "representation": train_info["representation"],
                    "support_mode": support_mode,
                    "normalization_mode": norm_mode,
                    "k": k,
                    "neighbor_agg": neighbor_agg,
                    "self_radius_k": self_radius_k,
                    "radius_quantile": radius_quantile,
                    "radius_train": radius_train,
                    "radius_eval": radius_eval,
                    "recall": recall,
                    "precision": precision,
                    "outside": 1.0 - precision if np.isfinite(precision) else float("nan"),
                    "train_to_eval_coverage": recall,
                    "eval_to_train_coverage": precision,
                    "mean_nn_eval_to_train": float(np.mean(dist_eval_to_train)) if dist_eval_to_train.size else float("nan"),
                    "median_nn_eval_to_train": float(np.median(dist_eval_to_train)) if dist_eval_to_train.size else float("nan"),
                    "p90_nn_eval_to_train": float(np.quantile(dist_eval_to_train, 0.9)) if dist_eval_to_train.size else float("nan"),
                    "mean_nn_train_to_eval": float(np.mean(dist_train_to_eval)) if dist_train_to_eval.size else float("nan"),
                    "median_nn_train_to_eval": float(np.median(dist_train_to_eval)) if dist_train_to_eval.size else float("nan"),
                    "p90_nn_train_to_eval": float(np.quantile(dist_train_to_eval, 0.9)) if dist_train_to_eval.size else float("nan"),
                    "kl_method": kl_method,
                    "kl_knn_k": int(kl_knn_k),
                    "kl_bins": int(kl_bins),
                    "kl_eps": float(kl_eps),
                    "kl_eval_to_train": kl_eval_to_train,
                    "kl_train_to_eval": kl_train_to_eval,
                    "n_train_vectors": int(train_vecs.shape[0]),
                    "n_eval_vectors": int(eval_vecs.shape[0]),
                }
                if extra_kl:
                    result.update(extra_kl)
                results.append(result)
                print(
                    f"[{train_label} -> {eval_info['label']}] "
                    f"train->eval={recall:.3f} eval->train={precision:.3f}"
                )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if results:
        import csv

        fieldnames = list(results[0].keys())
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved {len(results)} results to: {output_path}")
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
