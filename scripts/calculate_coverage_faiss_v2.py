#!/usr/bin/env python3
"""
Coverage Pipeline v2.0 - Modular Implementation

Implements the 5-step pipeline:
  0. Fixed sampling protocol (cached vectors)
  1. Alpha calibration (for flow joint space)
  2. Define spaces (xy, flow, joint for flow; features for dino/resnet)
  3. Per-dataset self-radius computation
  4. Cross-dataset directed NN distances
  5. Coverage metrics with dual normalization + curves
"""

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import json

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import dataset utilities from old script
from calculate_coverage_faiss import (
    create_dataset_from_config,
    create_mixed_dataset_from_config,
    _is_synthetic_dataset,
)

# Import new modular pipeline
from coverage import (
    cache,
    calibration,
    spaces,
    radius,
    metrics,
    kl,
    faiss_ops,
)

# Import feature extraction utilities
from src.coreset.validation import extract_flow_vectors_from_batch
from src.mmd.encoders import ResNet101Encoder, DinoV3Encoder

try:
    import faiss
except ImportError as exc:
    raise SystemExit(
        "faiss is required. Install faiss-cpu or faiss-gpu."
    ) from exc


def _pca_representation_name(
    representation: str,
    output_dim: int,
    l2_normalize: bool,
) -> str:
    suffix = f"_pca{output_dim}"
    if l2_normalize:
        suffix += "_l2norm"
    return f"{representation}{suffix}"


def _fit_pca_from_train_datasets(
    train_configs: List[Dict],
    config: Dict,
    encoder: object,
    cache_dir: Path,
    representation: str,
) -> object:
    max_train_vectors = config["pca"]["max_train_vectors"]
    vectors_per_image = config["sampling"].get(
        "vectors_per_image", config["sampling"].get("flow_per_image_max", 2000)
    )
    per_dataset_max = max(1, max_train_vectors // max(1, len(train_configs)))
    samples = []

    print(f"\nFitting PCA from train samples (target {max_train_vectors:,} vectors)...")
    for ds_config in train_configs:
        dataset_name = ds_config.get("name")
        split = ds_config.get("split")

        if ds_config.get("mixed", False):
            print(
                f"  Sampling mixed train dataset: "
                f"{' + '.join([f'{d}({p:.0%})' for d, p in zip(ds_config['datasets'], ds_config['percentages'])])}"
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
        dataloader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=config["sampling"].get("shuffle", True),
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
        )

        vectors = extract_vectors_from_dataset(
            dataset,
            dataloader,
            representation,
            encoder=encoder,
            max_vectors=per_dataset_max,
            vectors_per_image=vectors_per_image,
            seed=config["sampling"]["seed"],
            device="cuda" if torch.cuda.is_available() and config["faiss"]["use_gpu"] else "cpu",
            verbose=True,
        )
        if len(vectors) > 0:
            samples.append(vectors)

        del dataset, dataloader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not samples:
        raise ValueError("No PCA samples collected from train datasets.")

    sample_vecs = np.concatenate(samples, axis=0)
    if len(sample_vecs) > max_train_vectors:
        indices = np.random.choice(len(sample_vecs), size=max_train_vectors, replace=False)
        sample_vecs = sample_vecs[indices]

    pca = cache.fit_pca(
        sample_vecs,
        output_dim=config["pca"]["output_dim"],
        whiten=config["pca"]["whiten"],
    )
    cache.save_pca_model(pca, cache_dir, representation)
    return pca


def extract_vectors_from_dataset(
    dataset,
    dataloader,
    representation: str,
    encoder: Optional[object] = None,
    pca_model: Optional[object] = None,
    pca_l2_normalize: bool = False,
    max_vectors: int = 16_000_000,
    vectors_per_image: int = 2000,
    seed: int = 42,
    device: str = "cuda",
    verbose: bool = True,
    image_size: Optional[list] = None,
    stats_dir: Optional[Path] = None,
    dataset_label: Optional[str] = None,
    flow_count_logging: bool = False,
) -> np.ndarray:
    """
    Extract vectors from a dataset using fixed sampling protocol (Step 0).
    
    Sampling protocol (per image):
    - If n_valid ≤ vectors_per_image: keep all
    - Else: uniformly subsample to vectors_per_image
    - Stop when reaching max_vectors total
    
    Args:
        dataset: Dataset object
        dataloader: DataLoader
        representation: "flow", "resnet", or "dino"
        encoder: Feature encoder (for resnet/dino)
        max_vectors: Maximum total vectors to extract
        vectors_per_image: Target vectors per image (2000 for flow)
        seed: Random seed
        device: Device for computations
        verbose: Print progress
    
    Returns:
        (N, D) array of vectors
    """
    if verbose:
        print(f"  Extracting {representation} vectors...")
        print(f"  Target: {vectors_per_image:,} vectors per image, max {max_vectors:,} total")
    
    all_vectors = []
    total_vectors = 0
    count_logging_enabled = representation == "flow" or flow_count_logging
    per_image_valid_counts = []
    per_image_sampled_counts = []
    
    np.random.seed(seed)
    
    for batch_idx, batch in enumerate(tqdm(dataloader, disable=not verbose, desc="  Extracting")):
        if total_vectors >= max_vectors:
            break
        
        if representation == "flow":
            # Extract flow vectors [x, y, dx, dy] per image
            # Pass max_flow_magnitude based on image size
            max_flow = None
            if image_size is not None:
                # Max flow = image diagonal (conservative upper bound)
                import math
                max_flow = math.sqrt(image_size[0]**2 + image_size[1]**2)
            
            per_image_vectors = extract_flow_vectors_from_batch(batch, return_per_image=True, max_flow_magnitude=max_flow)
            
            if per_image_vectors is None or len(per_image_vectors) == 0:
                continue
            
            # Sanity check on first batch
            if batch_idx == 0 and verbose:
                total_in_batch = sum(len(img_vecs) for img_vecs in per_image_vectors)
                print(f"\n  First batch validation:")
                print(f"    Images in batch: {len(per_image_vectors)}")
                print(f"    Total flow vectors: {total_in_batch:,}")
                print(f"    Valid vectors per image: {[len(v) for v in per_image_vectors]}")
                
                # Check first image with valid flows
                for img_vecs in per_image_vectors:
                    if len(img_vecs) > 0:
                        print(f"    Sample image stats:")
                        print(f"      x: [{img_vecs[:, 0].min():.2f}, {img_vecs[:, 0].max():.2f}]")
                        print(f"      y: [{img_vecs[:, 1].min():.2f}, {img_vecs[:, 1].max():.2f}]")
                        print(f"      dx: [{img_vecs[:, 2].min():.2f}, {img_vecs[:, 2].max():.2f}]")
                        print(f"      dy: [{img_vecs[:, 3].min():.2f}, {img_vecs[:, 3].max():.2f}]")
                        
                        max_abs_flow = max(abs(img_vecs[:, 2].min()), abs(img_vecs[:, 2].max()),
                                         abs(img_vecs[:, 3].min()), abs(img_vecs[:, 3].max()))
                        if max_abs_flow > 1000:
                            print(f"\n  ⚠️⚠️⚠️  CRITICAL: Flow values are CORRUPTED! ⚠️⚠️⚠️")
                            print(f"  Max abs flow: {max_abs_flow:.0f} pixels")
                            raise ValueError(f"Flow values are corrupted at extraction time!")
                        break
            
            # Per-image sampling: Sample up to vectors_per_image from EACH image
            sampled_vectors = []
            for img_vectors in per_image_vectors:
                valid_count = len(img_vectors)
                per_image_valid_counts.append(valid_count)
                per_image_sampled_counts.append(
                    vectors_per_image if valid_count > vectors_per_image else valid_count
                )
                if len(img_vectors) == 0:
                    continue
                
                # Sample if image has more than target
                if len(img_vectors) > vectors_per_image:
                    indices = np.random.choice(len(img_vectors), size=vectors_per_image, replace=False)
                    img_vectors = img_vectors[indices]
                
                sampled_vectors.append(img_vectors)
            
            if len(sampled_vectors) == 0:
                continue
            
            # Debug: check before vstacking
            if batch_idx == 0 and verbose:
                print(f"    Pre-vstack check (first batch, {len(sampled_vectors)} images):")
                for i, img_v in enumerate(sampled_vectors[:3]):  # Check first 3 images
                    if len(img_v) > 0:
                        print(f"      Image {i}: {len(img_v)} vectors, dx:[{img_v[:, 2].min():.2f}, {img_v[:, 2].max():.2f}]")
            
            vectors = np.vstack(sampled_vectors)
            
            # Debug: check after vstacking
            if batch_idx == 0 and verbose:
                print(f"    Post-vstack check: {len(vectors)} total vectors, dx:[{vectors[:, 2].min():.2f}, {vectors[:, 2].max():.2f}]")
            
            # CRITICAL: Make a copy to avoid memory corruption during concatenation
            vectors = vectors.copy()
        
        else:  # resnet or dino
            if flow_count_logging:
                max_flow = None
                if image_size is not None:
                    import math

                    max_flow = math.sqrt(image_size[0]**2 + image_size[1]**2)
                per_image_vectors = extract_flow_vectors_from_batch(
                    batch,
                    return_per_image=True,
                    max_flow_magnitude=max_flow,
                )
                if per_image_vectors:
                    for img_vectors in per_image_vectors:
                        valid_count = len(img_vectors)
                        per_image_valid_counts.append(valid_count)
                        per_image_sampled_counts.append(
                            vectors_per_image if valid_count > vectors_per_image else valid_count
                        )
            # Extract spatial features from images
            # Flexible key handling for different dataset formats
            if "image0" in batch:
                images = batch["image0"]
            elif "src_img" in batch:
                images = batch["src_img"]
            elif "source" in batch:
                images = batch["source"]
            else:
                raise ValueError(f"Could not find image in batch. Keys: {batch.keys()}")
            
            if not isinstance(images, torch.Tensor):
                images = torch.tensor(images)
            
            images = images.to(device)  # (B, C, H, W)
            
            # extract_features returns flattened spatial features: (B*patches, D)
            # For standard ViT/ResNet: 32x32 = 1024 patches per image
            features = encoder.extract_features(images)  # (B*1024, D) for spatial patches
            
            vectors = features.cpu().numpy()
            del features
            if device == "cuda":
                torch.cuda.empty_cache()
            
            if not np.isfinite(vectors).all():
                bad_mask = ~np.isfinite(vectors)
                bad_count = int(np.count_nonzero(bad_mask))
                print(f"  ⚠️  Non-finite DINO/ResNet features detected in batch {batch_idx}: {bad_count} values")
                if np.isinf(vectors).any():
                    print("    Contains inf values")
                if np.isnan(vectors).any():
                    print("    Contains NaN values")
            
            B = images.shape[0]
            del images
            total_features = len(vectors)
            features_per_image_actual = total_features // B
            
            # Sanity check on first batch
            if batch_idx == 0 and verbose:
                print(f"\n  First batch validation:")
                print(f"    Batch size: {B}")
                print(f"    Total features: {total_features:,}")
                print(f"    Features per image: {features_per_image_actual:,} (should be 1024 for 32x32 patches)")
                print(f"    Feature dim: {vectors.shape[1]}")
                
                if features_per_image_actual != 1024:
                    print(f"    ⚠️  Warning: Expected 1024 patches per image, got {features_per_image_actual}")
            
            # Per-image sampling for features
            # Features are flattened as [img0_patches, img1_patches, img2_patches, ...]
            sampled_vectors = []
            for img_idx in range(B):
                start_idx = img_idx * features_per_image_actual
                end_idx = start_idx + features_per_image_actual
                img_vectors = vectors[start_idx:end_idx]
                
                # Sample if needed (usually keep all 1024 patches, unless vectors_per_image < 1024)
                if len(img_vectors) > vectors_per_image:
                    indices = np.random.choice(len(img_vectors), size=vectors_per_image, replace=False)
                    img_vectors = img_vectors[indices]
                
                sampled_vectors.append(img_vectors)
            
            if len(sampled_vectors) > 0:
                vectors = np.vstack(sampled_vectors)

            if pca_model is not None:
                vectors = cache.apply_pca(pca_model, vectors)
                if pca_l2_normalize:
                    vectors = cache.l2_normalize(vectors)
                _summarize_vector_stats(
                    vectors[: min(len(vectors), 50000)],
                    label=f"{dataset_label or 'dataset'} (post-pca)",
                    representation=representation,
                )
        
        all_vectors.append(vectors)
        total_vectors += len(vectors)
        
        if total_vectors >= max_vectors:
            break
    
    if len(all_vectors) == 0:
        if count_logging_enabled and stats_dir is not None:
            stats_dir = Path(stats_dir)
            stats_dir.mkdir(parents=True, exist_ok=True)
            label = (dataset_label or "dataset").replace("/", "_")
            stats_path = stats_dir / f"flow_counts_{label}.json"
            stats = {
                "dataset": dataset_label or "dataset",
                "images_seen": len(per_image_valid_counts),
                "images_with_zero": sum(1 for v in per_image_valid_counts if v == 0),
                "total_valid_vectors": int(np.sum(per_image_valid_counts)) if per_image_valid_counts else 0,
                "total_sampled_vectors": int(np.sum(per_image_sampled_counts)) if per_image_sampled_counts else 0,
                "total_vectors_retained": 0,
                "vectors_per_image_target": int(vectors_per_image),
                "max_vectors": int(max_vectors),
            }
            with stats_path.open("w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, sort_keys=True)
        return np.array([]).reshape(0, 4 if representation == "flow" else encoder.output_dim)
    
    # Debug: check before final concatenation
    if representation == "flow" and verbose:
        print(f"  Pre-final-concat check ({len(all_vectors)} batches):")
        for i, batch_v in enumerate(all_vectors[:5]):  # Check first 5 batches
            if len(batch_v) > 0:
                print(f"    Batch {i}: shape={batch_v.shape}, dtype={batch_v.dtype}, dx:[{batch_v[:, 2].min():.2f}, {batch_v[:, 2].max():.2f}]")
    
    # Try concatenating in a safer way
    if representation == "flow" and verbose:
        print(f"  Attempting concatenation of {len(all_vectors)} arrays...")
        # Check last batch too
        if len(all_vectors) > 0:
            last_batch = all_vectors[-1]
            print(f"    Last batch: shape={last_batch.shape}, dtype={last_batch.dtype}, dx:[{last_batch[:, 2].min():.2f}, {last_batch[:, 2].max():.2f}]")
    
    all_vectors = np.concatenate(all_vectors, axis=0)
    
    # Debug: check after final concatenation
    if representation == "flow" and verbose:
        print(f"  Post-final-concat check: {len(all_vectors)} vectors, dx:[{all_vectors[:, 2].min():.2f}, {all_vectors[:, 2].max():.2f}]")
        print(f"    Final dtype: {all_vectors.dtype}")
        print(f"    Memory location: {all_vectors.__array_interface__['data']}")
        # Check if first few vectors still look good
        print(f"    First 5 dx values: {all_vectors[:5, 2]}")
        print(f"    Last 5 dx values: {all_vectors[-5:, 2]}")
        
        # CRITICAL: Find where the corruption is
        dx_col = all_vectors[:, 2]
        dy_col = all_vectors[:, 3]
        max_dx = dx_col.max()
        min_dx = dx_col.min()
        print(f"    Searching for corrupted values...")
        
        # Find the WORST corrupted values (not just > 1000, but > 100000)
        very_bad_mask = (np.abs(dx_col) > 100000) | (np.abs(dy_col) > 100000)
        very_bad_indices = np.where(very_bad_mask)[0]
        
        if very_bad_indices.size > 0:
            print(f"    Found {very_bad_indices.size:,} SEVERELY corrupted values (>100k)!")
            # Find the one with min value
            min_idx = np.argmin(dx_col)
            max_idx = np.argmax(dx_col)
            print(f"    Min dx at index {min_idx:,} (~batch {min_idx//16384}): {dx_col[min_idx]:.2f}")
            print(f"    Max dx at index {max_idx:,} (~batch {max_idx//16384}): {dx_col[max_idx]:.2f}")
            print(f"    Vector with min dx: [{all_vectors[min_idx, 0]:.2f}, {all_vectors[min_idx, 1]:.2f}, {all_vectors[min_idx, 2]:.2f}, {all_vectors[min_idx, 3]:.2f}]")
            print(f"    Vector with max dx: [{all_vectors[max_idx, 0]:.2f}, {all_vectors[max_idx, 1]:.2f}, {all_vectors[max_idx, 2]:.2f}, {all_vectors[max_idx, 3]:.2f}]")
        elif max_dx > 1000 or min_dx < -1000:
            # Find indices of moderately corrupted values
            bad_mask = (dx_col > 1000) | (dx_col < -1000) | (dy_col > 1000) | (dy_col < -1000)
            bad_indices = np.where(bad_mask)[0]
            print(f"    Found {bad_indices.size:,} moderately corrupted values (>1000)!")
            if bad_indices.size > 0:
                first_bad = bad_indices[0]
                print(f"    First corrupted at index: {first_bad:,} (~{first_bad//16384} batches in)")
                print(f"    Value at that index: [{all_vectors[first_bad, 0]:.2f}, {all_vectors[first_bad, 1]:.2f}, {all_vectors[first_bad, 2]:.2f}, {all_vectors[first_bad, 3]:.2f}]")
        else:
            print(f"    No corruption found in final array!")
    
    # Trim to max_vectors
    if len(all_vectors) > max_vectors:
        all_vectors = all_vectors[:max_vectors]
    
    if verbose:
        print(f"  Extracted {len(all_vectors):,} vectors")

    if count_logging_enabled and stats_dir is not None:
        def _count_stats(values: List[int]) -> Dict[str, float]:
            if not values:
                return {
                    "mean": float("nan"),
                    "median": float("nan"),
                    "p10": float("nan"),
                    "p90": float("nan"),
                    "p95": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                }
            arr = np.asarray(values, dtype=np.float64)
            return {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "p10": float(np.quantile(arr, 0.10)),
                "p90": float(np.quantile(arr, 0.90)),
                "p95": float(np.quantile(arr, 0.95)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

        stats_dir = Path(stats_dir)
        stats_dir.mkdir(parents=True, exist_ok=True)
        label = (dataset_label or "dataset").replace("/", "_")
        stats_path = stats_dir / f"flow_counts_{label}.json"
        stats = {
            "dataset": dataset_label or "dataset",
            "images_seen": len(per_image_valid_counts),
            "images_with_zero": sum(1 for v in per_image_valid_counts if v == 0),
            "total_valid_vectors": int(np.sum(per_image_valid_counts)) if per_image_valid_counts else 0,
            "total_sampled_vectors": int(np.sum(per_image_sampled_counts)) if per_image_sampled_counts else 0,
            "total_vectors_retained": int(len(all_vectors)),
            "vectors_per_image_target": int(vectors_per_image),
            "max_vectors": int(max_vectors),
            "valid_counts": _count_stats(per_image_valid_counts),
            "sampled_counts": _count_stats(per_image_sampled_counts),
        }
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, sort_keys=True)
    
    return all_vectors


def _validate_vectors(
    vectors: np.ndarray,
    representation: str,
    expect_l2_normalized: bool,
    dataset_label: str,
    sample_size: int = 50000,
) -> None:
    """Sanity-check cached/extracted vectors for finiteness and normalization."""
    if vectors is None or vectors.size == 0:
        return
    n = vectors.shape[0]
    take = min(sample_size, n)
    idx = np.random.choice(n, size=take, replace=False) if n > take else slice(None)
    sample = vectors[idx]
    finite_mask = np.isfinite(sample)
    if not np.all(finite_mask):
        bad = np.count_nonzero(~finite_mask)
        raise ValueError(
            f"[{dataset_label}] Found {bad} non-finite values in {representation} vectors. "
            "Delete cached vectors and re-extract."
        )
    stats = _summarize_vector_stats(sample, label=dataset_label, representation=representation)
    if stats["zero_frac"] > 0.001:
        raise ValueError(
            f"[{dataset_label}] Found {stats['zero_frac']:.3%} near-zero vectors in {representation} "
            "sample; cached vectors may be invalid."
        )
    if stats["dup_rate"] > 0.05:
        print(
            f"  ⚠️  [{dataset_label}] High duplicate rate in {representation} sample "
            f"({stats['dup_rate']:.2%}); continuing."
        )
    if expect_l2_normalized and representation in ("dino", "resnet"):
        mean_norm = stats["mean_norm"]
        if not (0.90 <= mean_norm <= 1.10):
            raise ValueError(
                f"[{dataset_label}] Expected L2-normalized vectors (mean norm≈1), got {mean_norm:.3f}. "
                "Cached vectors may be stale; delete cache and re-extract."
            )
        # Stricter check: ensure most vectors are near unit norm.
        frac_outside = stats["frac_outside_norm"]
        if (mean_norm < 0.95 or mean_norm > 1.05) or (frac_outside > 0.01):
            raise ValueError(
                f"[{dataset_label}] L2-norm check failed (mean={mean_norm:.3f}, "
                f"frac_outside_[0.8,1.2]={frac_outside:.3%}). "
                "Cached vectors may be from a different PCA/L2 setting; delete cache and re-extract."
            )


def _summarize_vector_stats(
    sample: np.ndarray,
    label: str,
    representation: str,
) -> Dict[str, float]:
    norms = np.linalg.norm(sample, axis=1)
    mean_norm = float(np.mean(norms))
    min_norm = float(np.min(norms))
    max_norm = float(np.max(norms))
    zero_frac = float(np.mean(norms <= 1e-12))
    frac_outside_norm = float(np.mean((norms < 0.80) | (norms > 1.20)))

    rounded = np.round(sample.astype(np.float32), decimals=4)
    contig = np.ascontiguousarray(rounded)
    view = contig.view(np.dtype((np.void, contig.dtype.itemsize * contig.shape[1])))
    unique_vals, counts = np.unique(view, return_counts=True)
    unique_count = int(unique_vals.size)
    dup_rate = 1.0 - (unique_count / float(len(sample)))
    top_idx = int(np.argmax(counts)) if counts.size else 0
    top_count = int(counts[top_idx]) if counts.size else 0
    top_vec = rounded[top_idx] if counts.size else None
    top_norm = float(np.linalg.norm(top_vec)) if top_vec is not None else float("nan")

    print(
        f"  [{label}] {representation} stats: "
        f"mean_norm={mean_norm:.6f}, min_norm={min_norm:.6f}, max_norm={max_norm:.6f}, "
        f"zero_frac={zero_frac:.3%}, dup_rate={dup_rate:.2%}"
    )
    if top_vec is not None:
        print(
            f"    top_dup_count={top_count}, top_dup_norm={top_norm:.6f}, "
            f"top_dup_is_zero={'yes' if top_norm <= 1e-6 else 'no'}"
        )

    return {
        "mean_norm": mean_norm,
        "min_norm": min_norm,
        "max_norm": max_norm,
        "zero_frac": zero_frac,
        "dup_rate": dup_rate,
        "frac_outside_norm": frac_outside_norm,
    }


def run_pipeline(
    config_path: str,
    direction_override: Optional[str] = None,
    kl_only: bool = False,
):
    """Run the full coverage pipeline."""
    
    print(f"\n{'='*80}")
    print(f"COVERAGE PIPELINE V2.0")
    print(f"{'='*80}\n")
    print(f"Config: {config_path}\n")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if direction_override:
        config.setdefault("coverage", {})
        config["coverage"]["direction"] = direction_override
    if kl_only:
        config.setdefault("cache", {})
        config["cache"]["lazy_load"] = True

    representation = config['representation']
    cache_dir = Path(config['cache']['dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() and config['faiss']['use_gpu'] else "cpu"

    # Determine cache representation (may include PCA suffix for streaming)
    stream_pca = False
    cache_repr = representation
    if representation in ["resnet", "dino"] and config.get("pca", {}).get("enabled", False):
        stream_pca = config["pca"].get("streaming", True)
        if stream_pca:
            cache_repr = _pca_representation_name(
                representation,
                config["pca"]["output_dim"],
                config["pca"].get("l2_normalize", False),
            )

    # Only initialize encoder if we need to extract vectors (i.e., cache miss)
    need_encoder = False
    if representation in ["resnet", "dino"]:
        for ds_config in config['datasets']:
            dataset_name = ds_config.get('name')
            split = ds_config.get('split')
            if not cache.cache_exists(cache_dir, dataset_name, split, cache_repr):
                need_encoder = True
                break

    # Initialize encoder if needed
    encoder = None
    if need_encoder:
        if representation == "resnet":
            encoder = ResNet101Encoder(device=device)
        elif representation == "dino":
            encoder = DinoV3Encoder(device=device)
    
    # ======================
    # STEP 0: Load/Extract Vectors
    # ======================
    print(f"\n{'='*80}")
    print(f"STEP 0: LOAD/EXTRACT VECTORS")
    print(f"{'='*80}\n")

    lazy_load = bool(config.get("cache", {}).get("lazy_load", False))
    train_vectors = {}
    eval_vectors = {}
    train_keys = []
    eval_keys = []

    pca_model = None
    if representation in ["resnet", "dino"] and config.get("pca", {}).get("enabled", False):
        if stream_pca:
            pca_model = cache.load_pca_model(cache_dir, representation)

    if stream_pca and pca_model is None and need_encoder:
        train_configs = [d for d in config["datasets"] if not d.get("is_eval", False)]
        pca_model = _fit_pca_from_train_datasets(
            train_configs,
            config,
            encoder,
            cache_dir,
            representation,
        )
    
    for ds_config in config['datasets']:
        is_eval = ds_config.get('is_eval', False)
        dataset_name = ds_config.get('name')
        split = ds_config.get('split')
        
        print(f"\n[{dataset_name}/{split}] {'(eval)' if is_eval else '(train)'}")
        
        vectors = None
        cache_present = False
        if not lazy_load:
            # Try to load from cache first
            vectors = cache.load_cached_vectors(
                cache_dir,
                dataset_name,
                split,
                cache_repr,
                mmap=config.get("cache", {}).get("mmap", False),
            )
        else:
            cache_present = cache.cache_exists(cache_dir, dataset_name, split, cache_repr)
            if cache_present:
                print("  ✓ Cached vectors present")

        dataset = None
        dataloader = None
        
        if lazy_load and cache_present:
            key = (dataset_name, split)
            if is_eval:
                eval_keys.append(key)
            else:
                train_keys.append(key)
            continue

        if vectors is None:
            # Create dataset (mixed or regular)
            if ds_config.get('mixed', False):
                print(f"  Mixed dataset: {' + '.join([f'{d}({p:.0%})' for d, p in zip(ds_config['datasets'], ds_config['percentages'])])}")
                dataset = create_mixed_dataset_from_config(
                    ds_config['datasets'],
                    ds_config['percentages'],
                    split,
                    config['dataset_params'],
                    config['dataset_overrides'],
                    seed=config['sampling']['seed'],
                )
                # Mixed datasets with synthetic need special handling
                is_synthetic = any(_is_synthetic_dataset(name) for name in ds_config['datasets'])
            else:
                dataset = create_dataset_from_config(
                    dataset_name,
                    split,
                    config['dataset_params'],
                    config['dataset_overrides'],
                    entry_overrides=ds_config.get('overrides'),
                )
                is_synthetic = _is_synthetic_dataset(dataset_name)
            
            # Synthetic datasets must use num_workers=0 to avoid segfaults
            # Also disable pin_memory since synthetic returns CUDA tensors
            num_workers = 0 if is_synthetic else config['num_workers']
            pin_memory = False if is_synthetic else True
            
            if is_synthetic and config['num_workers'] > 0:
                print(f"  ⚠️  Synthetic dataset detected - forcing num_workers=0 and pin_memory=False")
            
            dataloader = DataLoader(
                dataset,
                batch_size=config['batch_size'],
                shuffle=config['sampling'].get('shuffle', True),
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None,
            )
            
            # Extract vectors
            # For flow, pass image size for magnitude filtering
            extract_kwargs = {
                'max_vectors': config['sampling']['max_vectors'],
                'vectors_per_image': config['sampling'].get('vectors_per_image', config['sampling'].get('flow_per_image_max', 2000)),
                'seed': config['sampling']['seed'],
                'device': device,
                'verbose': True,
            }
            img_size = config.get("flow_normalization", {}).get("image_size", [512, 512])
            if representation == "flow":
                extract_kwargs["image_size"] = img_size
                extract_kwargs["stats_dir"] = cache_dir / "stats"
                extract_kwargs["dataset_label"] = f"{dataset_name}_{split}_{representation}"
            elif config.get("flow_count_logging", False):
                extract_kwargs["image_size"] = img_size
                extract_kwargs["stats_dir"] = cache_dir / "stats"
                extract_kwargs["dataset_label"] = f"{dataset_name}_{split}_{representation}"
                extract_kwargs["flow_count_logging"] = True
            
            vectors = extract_vectors_from_dataset(
                dataset,
                dataloader,
                representation,
                encoder=encoder,
                pca_model=pca_model if stream_pca else None,
                pca_l2_normalize=config.get("pca", {}).get("l2_normalize", False),
                **extract_kwargs
            )
            
            # Debug: check vectors before caching
            if representation == "flow":
                print(f"  Pre-cache check:")
                print(f"    Shape: {vectors.shape}")
                print(f"    x: [{vectors[:, 0].min():.2f}, {vectors[:, 0].max():.2f}]")
                print(f"    y: [{vectors[:, 1].min():.2f}, {vectors[:, 1].max():.2f}]")
                print(f"    dx: [{vectors[:, 2].min():.2f}, {vectors[:, 2].max():.2f}]")
                print(f"    dy: [{vectors[:, 3].min():.2f}, {vectors[:, 3].max():.2f}]")
            
            # Cache for future runs
            cache.save_cached_vectors(cache_dir, dataset_name, split, cache_repr, vectors)
        else:
            print(f"  ✓ Loaded {len(vectors):,} cached vectors")

        key = (dataset_name, split)
        if is_eval:
            eval_keys.append(key)
        else:
            train_keys.append(key)

        if not lazy_load:
            # Validate vectors for non-finite values and expected normalization
            expect_l2 = (
                representation in ("dino", "resnet")
                and config.get("pca", {}).get("enabled", False)
                and config.get("pca", {}).get("l2_normalize", False)
            )
            _validate_vectors(
                vectors,
                representation,
                expect_l2_normalized=expect_l2,
                dataset_label=f"{dataset_name}/{split}",
            )
            # Store vectors
            if is_eval:
                eval_vectors[key] = vectors
            else:
                train_vectors[key] = vectors
        else:
            if vectors is not None:
                del vectors

        # Clean up (only delete if they were created)
        if dataset is not None:
            del dataset
        if dataloader is not None:
            del dataloader
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    
    if lazy_load:
        print(f"\nFound {len(train_keys)} train sets, {len(eval_keys)} eval sets")
    else:
        print(f"\nLoaded {len(train_vectors)} train sets, {len(eval_vectors)} eval sets")

    # Release encoder after extraction to free GPU memory.
    if encoder is not None:
        try:
            encoder_device = getattr(encoder, "device", None)
            if encoder_device is not None and str(encoder_device).startswith("cuda"):
                try:
                    encoder.dino.model.to("cpu")
                except Exception:
                    pass
                try:
                    encoder.backbone.to("cpu")
                except Exception:
                    pass
        finally:
            del encoder
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    
    # ======================
    # PREPROCESSING
    # ======================
    if lazy_load and representation in ["resnet", "dino"] and not stream_pca:
        raise ValueError("Lazy-load requires streaming PCA for resnet/dino.")

    if representation == "flow":
        # Normalize flow vectors to [-1, 1]
        if config['flow_normalization']['enabled']:
            print(f"\nNormalizing flow vectors to [-1, 1]...")
            img_h, img_w = config['flow_normalization']['image_size']
            
            if lazy_load:
                print("  Lazy-load enabled; normalization will be applied on load per dataset.")
            else:
                # Debug: check vector ranges before normalization
                sample_key = list(train_vectors.keys())[0]
                sample_vec = train_vectors[sample_key]
                print(f"  Before normalization (sample from {sample_key}):")
                print(f"    x range: [{sample_vec[:, 0].min():.2f}, {sample_vec[:, 0].max():.2f}]")
                print(f"    y range: [{sample_vec[:, 1].min():.2f}, {sample_vec[:, 1].max():.2f}]")
                print(f"    dx range: [{sample_vec[:, 2].min():.2f}, {sample_vec[:, 2].max():.2f}]")
                print(f"    dy range: [{sample_vec[:, 3].min():.2f}, {sample_vec[:, 3].max():.2f}]")
                
                # SANITY CHECK: Flow values should be reasonable
                max_abs_flow = max(abs(sample_vec[:, 2].min()), abs(sample_vec[:, 2].max()),
                                  abs(sample_vec[:, 3].min()), abs(sample_vec[:, 3].max()))
                if max_abs_flow > 1000:
                    print(f"\n  ⚠️⚠️⚠️  CRITICAL ERROR ⚠️⚠️⚠️")
                    print(f"  Flow values are HUGE (max: {max_abs_flow:.0f} pixels)!")
                    print(f"  Normal optical flow is typically < 200 pixels.")
                    print(f"  Your cached vectors are likely CORRUPTED.")
                    print(f"\n  To fix:")
                    print(f"    1. Delete corrupted cache: rm {cache_dir}/*_flow.npy")
                    print(f"    2. Re-run pipeline to re-extract vectors")
                    print(f"\n  Aborting to prevent invalid results...")
                    raise ValueError(
                        f"Flow vectors are corrupted! Max abs flow: {max_abs_flow:.0f} pixels. "
                        f"Expected < 1000 pixels. Delete cache and re-extract."
                    )
                
                for key, vectors in train_vectors.items():
                    train_vectors[key] = spaces.normalize_flow_vectors(vectors, img_w, img_h)
                for key, vectors in eval_vectors.items():
                    eval_vectors[key] = spaces.normalize_flow_vectors(vectors, img_w, img_h)
                
                # Debug: check after normalization
                sample_vec_norm = train_vectors[sample_key]
                print(f"  After normalization:")
                print(f"    x range: [{sample_vec_norm[:, 0].min():.4f}, {sample_vec_norm[:, 0].max():.4f}]")
                print(f"    y range: [{sample_vec_norm[:, 1].min():.4f}, {sample_vec_norm[:, 1].max():.4f}]")
                print(f"    dx range: [{sample_vec_norm[:, 2].min():.4f}, {sample_vec_norm[:, 2].max():.4f}]")
                print(f"    dy range: [{sample_vec_norm[:, 3].min():.4f}, {sample_vec_norm[:, 3].max():.4f}]")
    
    elif representation in ["resnet", "dino"] and not stream_pca:
        # Apply PCA + L2 normalization
        if config['pca']['enabled']:
            print(f"\nApplying PCA + L2 normalization...")
            
            # Fit PCA on training data
            train_vectors, eval_vectors = cache.apply_pca_preprocessing(
                cache_dir,
                train_vectors,
                eval_vectors,
                representation=representation,
                output_dim=config['pca']['output_dim'],
                whiten=config['pca']['whiten'],
                do_l2_normalize=config['pca']['l2_normalize'],
                max_train_vectors=config['pca']['max_train_vectors'],
                verbose=True,
            )
    
    # ======================
    # STEP 1: Alpha Calibration (Flow Only)
    # ======================
    global_alpha = None
    
    disable_alpha = os.getenv("COVERAGE_DISABLE_ALPHA", "").strip().lower() in {"1", "true", "yes", "y"}

    if representation == "flow" and config['calibration']['enabled'] and not disable_alpha:
        print(f"\n{'='*80}")
        print(f"STEP 1: ALPHA CALIBRATION")
        print(f"{'='*80}\n")
        
        dedup_alpha = config.get("calibration", {}).get("dedup", False)
        env_alpha = os.getenv("COVERAGE_ALPHA_DEDUP", "").strip().lower()
        if env_alpha in {"1", "true", "yes", "y"}:
            dedup_alpha = True

        global_alpha, per_dataset_alphas = calibration.load_or_compute_alpha(
            cache_dir,
            train_vectors,
            k=config['calibration']['k'],
            aggregation=config['calibration']['aggregation'],
            use_gpu=config['faiss']['use_gpu'],
            force_recompute=config['calibration']['force_recompute'],
            dedup=dedup_alpha,
            verbose=True,
        )
    elif representation == "flow" and disable_alpha:
        global_alpha = 1.0
        print(f"\n{'='*80}")
        print(f"STEP 1: ALPHA CALIBRATION (DISABLED)")
        print(f"{'='*80}\n")
        print("  ⚠️  Alpha calibration disabled via COVERAGE_DISABLE_ALPHA=1. Using alpha=1.0")
    elif representation == "flow" and not config['calibration']['enabled']:
        global_alpha = 1.0
        print(f"\n{'='*80}")
        print(f"STEP 1: ALPHA CALIBRATION (DISABLED IN CONFIG)")
        print(f"{'='*80}\n")
        print("  ⚠️  Alpha calibration disabled in config. Using alpha=1.0")
    
    # ======================
    # STEP 2: Define Spaces
    # ======================
    print(f"\n{'='*80}")
    print(f"STEP 2: DEFINE SPACES")
    print(f"{'='*80}\n")
    
    enabled_spaces = config['spaces']['enabled']
    print(f"Enabled spaces: {enabled_spaces}")

    def _load_vectors_for_key(key: tuple) -> np.ndarray:
        dataset_name, split = key
        vectors = cache.load_cached_vectors(
            cache_dir,
            dataset_name,
            split,
            cache_repr,
            mmap=config.get("cache", {}).get("mmap", False),
        )
        if vectors is None:
            raise ValueError(f"Missing cached vectors for {dataset_name}/{split}")
        expect_l2 = (
            representation in ("dino", "resnet")
            and config.get("pca", {}).get("enabled", False)
            and config.get("pca", {}).get("l2_normalize", False)
        )
        _validate_vectors(
            vectors,
            representation,
            expect_l2_normalized=expect_l2,
            dataset_label=f"{dataset_name}/{split}",
        )
        if representation == "flow" and config['flow_normalization']['enabled']:
            img_h, img_w = config['flow_normalization']['image_size']
            vectors = spaces.normalize_flow_vectors(vectors, img_w, img_h)
        return vectors

    def _to_space(vectors: np.ndarray, space_name: str) -> np.ndarray:
        if representation != "flow":
            return vectors
        if space_name == "xy":
            return spaces.to_xy_space(vectors)
        if space_name == "flow":
            return spaces.to_flow_space(vectors)
        if space_name == "joint":
            return spaces.to_joint_space(vectors, global_alpha)
        raise ValueError(f"Unknown space: {space_name}")

    space_train_vectors = {}
    space_eval_vectors = {}
    if not lazy_load:
        # Transform vectors into each space
        for space_name in enabled_spaces:
            print(f"\nTransforming to {space_name} space...")
            space_train_vectors[space_name] = {}
            space_eval_vectors[space_name] = {}
            for key, vectors in train_vectors.items():
                space_train_vectors[space_name][key] = _to_space(vectors, space_name)
            for key, vectors in eval_vectors.items():
                space_eval_vectors[space_name][key] = _to_space(vectors, space_name)
            print(f"  {space_name}: {len(space_train_vectors[space_name])} train, {len(space_eval_vectors[space_name])} eval")

    # ======================
    # OPTIONAL: KL DIVERGENCE (kNN)
    # ======================
    kl_cfg = config.get("kl", {}) or {}
    kl_active = bool(kl_cfg.get("enabled", False) or kl_only)
    if kl_only and not kl_cfg:
        raise ValueError("KL-only run requires a 'kl' section in the config.")

    if kl_active:
        print(f"\n{'='*80}")
        print(f"KL DIVERGENCE (kNN)")
        print(f"{'='*80}\n")

        kl_spaces = kl_cfg.get("spaces", enabled_spaces)
        k_values = [int(k) for k in kl_cfg.get("k_values", [])]
        if not k_values:
            raise ValueError("KL config must define k_values (e.g., [5, 10, 20, 40]).")
        k_values = sorted(set(k_values))
        k_max = max(k_values)
        kl_eps = float(kl_cfg.get("eps", 1e-12))
        kl_index_factory = kl_cfg.get("index_factory", config["faiss"]["index_factory"])
        kl_nprobe = kl_cfg.get("nprobe", config["faiss"].get("nprobe"))
        kl_batch_size = kl_cfg.get("batch_size", config["faiss"].get("batch_size"))
        kl_filter_duplicates = bool(kl_cfg.get("filter_duplicates", True))
        cache_self_dists = bool(kl_cfg.get("cache_self_dists", True))
        cache_self_dists_disk = bool(kl_cfg.get("cache_self_dists_disk", False))
        resume_pairs = bool(kl_cfg.get("resume", False))
        flush_each_pair = bool(kl_cfg.get("flush_each_pair", False))
        kl_streaming = bool(kl_cfg.get("streaming", False))

        def _kl_norm_tag() -> str:
            if representation == "flow":
                return config.get("flow_normalization", {}).get("scheme", "norm2x1")
            if representation in ("dino", "resnet") and config.get("pca", {}).get("enabled", False):
                tag = f"pca{config['pca']['output_dim']}"
                if config["pca"].get("l2_normalize", False):
                    tag += "_l2"
                return tag
            return "none"

        kl_norm_tag = _kl_norm_tag()

        kl_results = []
        self_knn_cache = {space: {} for space in kl_spaces}

        output_kl = kl_cfg.get("results_file") or config.get("output", {}).get("kl_results_file")
        if not output_kl:
            raise ValueError("KL results_file not specified (set kl.results_file or output.kl_results_file).")
        output_kl_path = Path(output_kl)
        output_kl_path.parent.mkdir(parents=True, exist_ok=True)

        done_pairs = set()
        if resume_pairs and output_kl_path.exists():
            try:
                existing = pd.read_csv(output_kl_path)
                for _, row in existing.iterrows():
                    key = (
                        str(row.get("space")),
                        str(row.get("train_dataset")),
                        str(row.get("train_split")),
                        str(row.get("eval_dataset")),
                        str(row.get("eval_split")),
                    )
                    done_pairs.add(key)
                print(f"  ✓ Resuming from {output_kl_path} ({len(done_pairs)} pairs cached)")
            except Exception as exc:
                print(f"  ⚠️  Failed to read existing KL results ({exc}); starting fresh.")
                done_pairs = set()

        def _write_row(row: dict, header: List[str]) -> None:
            import csv
            write_header = (not output_kl_path.exists()) or output_kl_path.stat().st_size == 0
            mode = "a"
            with open(output_kl_path, mode, newline="") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

        def _load_space_vectors(key: tuple, space_name: str) -> np.ndarray:
            if lazy_load:
                vecs = _load_vectors_for_key(key)
                vecs = _to_space(vecs, space_name)
            else:
                if key in space_train_vectors[space_name]:
                    vecs = space_train_vectors[space_name][key]
                else:
                    vecs = space_eval_vectors[space_name][key]
            return vecs

        def _get_self_knn(
            key: tuple,
            space_name: str,
            vectors: Optional[np.ndarray] = None,
        ) -> np.ndarray:
            if cache_self_dists and key in self_knn_cache[space_name]:
                return self_knn_cache[space_name][key]
            if cache_self_dists_disk:
                ds_name, ds_split = key
                alpha = global_alpha if space_name == "joint" else None
                cached = cache.load_knn_self_distances(
                    cache_dir,
                    ds_name,
                    ds_split,
                    representation,
                    space_name,
                    k_max,
                    kl_norm_tag,
                    config["distance_metric"]["name"],
                    filter_duplicates=kl_filter_duplicates,
                    alpha=alpha,
                )
                if cached is not None:
                    if cache_self_dists:
                        self_knn_cache[space_name][key] = cached
                    return cached
            vecs = vectors if vectors is not None else _load_space_vectors(key, space_name)
            if vecs.shape[0] < 2 or k_max < 1:
                dists = np.array([], dtype=np.float32).reshape(0, k_max)
            else:
                if kl_streaming and cache_self_dists_disk:
                    ds_name, ds_split = key
                    alpha = global_alpha if space_name == "joint" else None
                    knn_dir = cache_dir / "knn_self"
                    knn_dir.mkdir(parents=True, exist_ok=True)
                    knn_path = knn_dir / cache.knn_self_cache_key(
                        ds_name,
                        ds_split,
                        representation,
                        space_name,
                        k_max,
                        kl_norm_tag,
                        config["distance_metric"]["name"],
                        filter_duplicates=kl_filter_duplicates,
                        alpha=alpha,
                        ext="npy",
                    )
                    dists = kl.compute_self_knn_distances_streaming(
                        vecs,
                        k=k_max,
                        distance_metric=config["distance_metric"]["name"],
                        out_path=knn_path,
                        filter_duplicates=kl_filter_duplicates,
                        use_gpu=config["faiss"]["use_gpu"],
                        index_factory=kl_index_factory,
                        nprobe=kl_nprobe,
                        batch_size=kl_batch_size,
                        verbose=True,
                    )
                else:
                    dists = kl.compute_self_knn_distances(
                        vecs,
                        k=k_max,
                        distance_metric=config["distance_metric"]["name"],
                        filter_duplicates=kl_filter_duplicates,
                        use_gpu=config["faiss"]["use_gpu"],
                        index_factory=kl_index_factory,
                        nprobe=kl_nprobe,
                        batch_size=kl_batch_size,
                        verbose=True,
                    )
            dists = dists.astype(np.float32, copy=False)
            if cache_self_dists_disk:
                ds_name, ds_split = key
                alpha = global_alpha if space_name == "joint" else None
                if not (kl_streaming and cache_self_dists_disk):
                    cache.save_knn_self_distances(
                        cache_dir,
                        ds_name,
                        ds_split,
                        representation,
                        space_name,
                        k_max,
                        kl_norm_tag,
                        config["distance_metric"]["name"],
                        dists,
                        filter_duplicates=kl_filter_duplicates,
                        alpha=alpha,
                    )
            if cache_self_dists:
                self_knn_cache[space_name][key] = dists
            if lazy_load and vectors is None:
                del vecs
            return dists

        for space_name in kl_spaces:
            print(f"\n{'='*60}")
            print(f"SPACE (KL): {space_name.upper()}")
            print(f"{'='*60}\n")

            train_iter = train_keys if lazy_load else space_train_vectors[space_name].keys()
            eval_iter = eval_keys if lazy_load else space_eval_vectors[space_name].keys()
            eval_keys_list = list(eval_iter) if resume_pairs else eval_iter

            header = [
                "space",
                "train_dataset",
                "train_split",
                "eval_dataset",
                "eval_split",
                "train_n_vectors",
                "eval_n_vectors",
                "dim",
            ]
            for k in k_values:
                header.append(f"kl_train_to_eval_k{k}")
                header.append(f"kl_eval_to_train_k{k}")

            for train_key in train_iter:
                train_name, train_split = train_key
                if resume_pairs and eval_keys_list:
                    any_missing = False
                    for eval_key in eval_keys_list:
                        eval_name, eval_split = eval_key
                        pair_key = (space_name, train_name, train_split, eval_name, eval_split)
                        if pair_key not in done_pairs:
                            any_missing = True
                            break
                    if not any_missing:
                        print(f"\n[{train_name}/{train_split}] all eval pairs cached; skipping")
                        continue
                if lazy_load:
                    train_vecs = _load_vectors_for_key(train_key)
                    train_vecs = _to_space(train_vecs, space_name)
                else:
                    train_vecs = space_train_vectors[space_name][train_key]

                train_self = _get_self_knn(train_key, space_name, vectors=train_vecs)

                for eval_key in eval_keys_list:
                    eval_name, eval_split = eval_key
                    pair_key = (space_name, train_name, train_split, eval_name, eval_split)
                    if pair_key in done_pairs:
                        print("  ✓ Pair already cached; skipping")
                        continue
                    print(f"\n[{train_name}/{train_split} → {eval_name}/{eval_split}]")

                    if lazy_load:
                        eval_vecs = _load_vectors_for_key(eval_key)
                        eval_vecs = _to_space(eval_vecs, space_name)
                    else:
                        eval_vecs = space_eval_vectors[space_name][eval_key]

                    eval_self = _get_self_knn(eval_key, space_name, vectors=eval_vecs)

                    if kl_streaming:
                        kl_train_to_eval = kl.compute_knn_kl_streaming(
                            train_vecs,
                            eval_vecs,
                            train_self,
                            k_values=k_values,
                            distance_metric=config["distance_metric"]["name"],
                            eps=kl_eps,
                            filter_duplicates=kl_filter_duplicates,
                            use_gpu=config["faiss"]["use_gpu"],
                            index_factory=kl_index_factory,
                            nprobe=kl_nprobe,
                            batch_size=kl_batch_size,
                            verbose=True,
                        )
                        kl_eval_to_train = kl.compute_knn_kl_streaming(
                            eval_vecs,
                            train_vecs,
                            eval_self,
                            k_values=k_values,
                            distance_metric=config["distance_metric"]["name"],
                            eps=kl_eps,
                            filter_duplicates=kl_filter_duplicates,
                            use_gpu=config["faiss"]["use_gpu"],
                            index_factory=kl_index_factory,
                            nprobe=kl_nprobe,
                            batch_size=kl_batch_size,
                            verbose=True,
                        )
                    else:
                        # Directed cross distances (exclude self + duplicates)
                        train_to_eval = kl.compute_cross_knn_distances(
                            train_vecs,
                            eval_vecs,
                            k=k_max,
                            distance_metric=config["distance_metric"]["name"],
                            filter_duplicates=kl_filter_duplicates,
                            use_gpu=config["faiss"]["use_gpu"],
                            index_factory=kl_index_factory,
                            nprobe=kl_nprobe,
                            batch_size=kl_batch_size,
                            verbose=True,
                        )
                        eval_to_train = kl.compute_cross_knn_distances(
                            eval_vecs,
                            train_vecs,
                            k=k_max,
                            distance_metric=config["distance_metric"]["name"],
                            filter_duplicates=kl_filter_duplicates,
                            use_gpu=config["faiss"]["use_gpu"],
                            index_factory=kl_index_factory,
                            nprobe=kl_nprobe,
                            batch_size=kl_batch_size,
                            verbose=True,
                        )

                        # KL(P||Q) and KL(Q||P)
                        kl_train_to_eval = kl.compute_knn_kl_for_k_values(
                            train_self,
                            train_to_eval,
                            m=int(eval_vecs.shape[0]),
                            dim=int(train_vecs.shape[1]),
                            k_values=k_values,
                            eps=kl_eps,
                        )
                        kl_eval_to_train = kl.compute_knn_kl_for_k_values(
                            eval_self,
                            eval_to_train,
                            m=int(train_vecs.shape[0]),
                            dim=int(eval_vecs.shape[1]),
                            k_values=k_values,
                            eps=kl_eps,
                        )

                    row = {
                        "space": space_name,
                        "train_dataset": train_name,
                        "train_split": train_split,
                        "eval_dataset": eval_name,
                        "eval_split": eval_split,
                        "train_n_vectors": int(train_vecs.shape[0]),
                        "eval_n_vectors": int(eval_vecs.shape[0]),
                        "dim": int(train_vecs.shape[1]),
                    }
                    for k in k_values:
                        row[f"kl_train_to_eval_k{k}"] = kl_train_to_eval.get(k, float("nan"))
                        row[f"kl_eval_to_train_k{k}"] = kl_eval_to_train.get(k, float("nan"))
                    if flush_each_pair:
                        _write_row(row, header)
                        done_pairs.add(pair_key)
                    else:
                        kl_results.append(row)

                    if not kl_streaming:
                        del train_to_eval, eval_to_train
                    del eval_self
                    if lazy_load:
                        del eval_vecs
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()

                if lazy_load:
                    del train_vecs
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

        # Save KL results (if not flushed incrementally)
        if not flush_each_pair:
            if kl_results:
                kl_df = pd.DataFrame(kl_results)
                if resume_pairs and output_kl_path.exists():
                    kl_df.to_csv(output_kl_path, index=False, mode="a", header=False)
                else:
                    kl_df.to_csv(output_kl_path, index=False)
            print(f"\n✓ KL results saved to: {output_kl_path}")
            print(f"  Total rows: {len(kl_results)}")
        else:
            print(f"\n✓ KL results saved incrementally to: {output_kl_path}")

        if kl_only:
            print(f"\n{'='*80}")
            print(f"KL-ONLY PIPELINE COMPLETE")
            print(f"{'='*80}\n")
            return
    
    # ======================
    # STEP 3: Self-Radius Computation
    # ======================
    print(f"\n{'='*80}")
    print(f"STEP 3: SELF-RADIUS COMPUTATION")
    print(f"{'='*80}\n")
    
    all_radii = {}  # space_name -> {(dataset_name, split): radius_data}
    
    for space_name in enabled_spaces:
        dedup_self_radius = config.get("coverage", {}).get("self_radius_dedup", False)
        env_dedup = os.getenv("COVERAGE_SELF_RADIUS_DEDUP", "").strip().lower()
        if env_dedup in {"1", "true", "yes", "y"}:
            dedup_self_radius = True

        radii_for_space = {}
        all_radii[space_name] = radii_for_space

        if not lazy_load:
            all_datasets_in_space = {**space_train_vectors[space_name], **space_eval_vectors[space_name]}
            radii_for_space.update(
                radius.compute_all_radii(
                    cache_dir,
                    all_datasets_in_space,
                    space=space_name,
                    k=config['coverage']['self_radius_k'],
                    quantile=config['coverage']['radius_quantile'],
                    neighbor_agg=config['coverage']['neighbor_agg'],
                    filter_duplicates=True,
                    dedup=dedup_self_radius,
                    batch_size=config['faiss'].get('batch_size'),
                    alpha=global_alpha if space_name == "joint" else None,
                    normalization=config.get('flow_normalization', {}).get('scheme', 'norm2x1'),
                    distance_metric=config['distance_metric']['name'],
                    use_gpu=config['faiss']['use_gpu'],
                    index_factory=config['faiss']['index_factory'],
                    nprobe=config['faiss'].get('nprobe'),
                    force_recompute=False,
                    verbose=True,
                )
            )
        else:
            for key in train_keys + eval_keys:
                ds_name, ds_split = key
                print(f"\n[{ds_name}/{ds_split}]")
                vectors = _load_vectors_for_key(key)
                vectors = _to_space(vectors, space_name)
                radii_for_space[key] = radius.load_or_compute_radius(
                    cache_dir,
                    ds_name,
                    ds_split,
                    space_name,
                    vectors,
                    k=config['coverage']['self_radius_k'],
                    quantile=config['coverage']['radius_quantile'],
                    neighbor_agg=config['coverage']['neighbor_agg'],
                    filter_duplicates=True,
                    dedup=dedup_self_radius,
                    batch_size=config['faiss'].get('batch_size'),
                    alpha=global_alpha if space_name == "joint" else None,
                    normalization=config.get('flow_normalization', {}).get('scheme', 'norm2x1'),
                    distance_metric=config['distance_metric']['name'],
                    use_gpu=config['faiss']['use_gpu'],
                    index_factory=config['faiss']['index_factory'],
                    nprobe=config['faiss'].get('nprobe'),
                    force_recompute=False,
                    verbose=True,
                )
                del vectors
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
    
    # ======================
    # STEPS 4-5: Coverage Metrics
    # ======================
    print(f"\n{'='*80}")
    print(f"STEPS 4-5: COVERAGE METRICS")
    print(f"{'='*80}\n")
    
    results = []
    output_file = Path(config['output']['results_file'])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # Resume: load already-completed pairs so we can skip them
    _done_pairs: set[tuple] = set()
    if output_file.exists() and output_file.stat().st_size > 0:
        try:
            _existing = pd.read_csv(output_file)
            for _, _r in _existing.iterrows():
                _done_pairs.add((_r['space'], _r['train_dataset'], _r['train_split'],
                                 _r['eval_dataset'], _r['eval_split']))
            print(f"  Resuming: {len(_done_pairs)} pairs already in {output_file.name}")
        except Exception:
            pass

    for space_name in enabled_spaces:
        print(f"\n{'='*60}")
        print(f"SPACE: {space_name.upper()}")
        print(f"{'='*60}\n")

        train_iter = train_keys if lazy_load else space_train_vectors[space_name].keys()
        eval_iter = eval_keys if lazy_load else space_eval_vectors[space_name].keys()
        for train_key in train_iter:
            for eval_key in eval_iter:
                train_name, train_split = train_key
                eval_name, eval_split = eval_key

                if (space_name, train_name, train_split, eval_name, eval_split) in _done_pairs:
                    print(f"\n[{train_name}/{train_split} → {eval_name}/{eval_split}] already done — skipping")
                    continue

                print(f"\n[{train_name}/{train_split} → {eval_name}/{eval_split}]")

                if lazy_load:
                    train_vecs = _load_vectors_for_key(train_key)
                    eval_vecs = _load_vectors_for_key(eval_key)
                    train_vecs = _to_space(train_vecs, space_name)
                    eval_vecs = _to_space(eval_vecs, space_name)
                else:
                    train_vecs = space_train_vectors[space_name][train_key]
                    eval_vecs = space_eval_vectors[space_name][eval_key]
                train_radius_data = all_radii[space_name][train_key]
                eval_radius_data = all_radii[space_name][eval_key]
                
                # Compute coverage metrics
                coverage_result = metrics.compute_pairwise_coverage(
                    train_vecs,
                    eval_vecs,
                    train_radius_data,
                    eval_radius_data,
                    cache_dir=cache_dir,
                    train_label=train_key,
                    eval_label=eval_key,
                    space=space_name,
                    normalization=config.get('flow_normalization', {}).get('scheme', 'norm2x1'),
                    distance_metric=config['distance_metric']['name'],
                    alpha=global_alpha if space_name == "joint" else None,
                    direction=config.get("coverage", {}).get("direction", "both"),
                    k_max=config['coverage']['k_max'],
                    k_values=config['coverage']['k_values'],
                    use_gpu=config['faiss']['use_gpu'],
                    index_factory=config['faiss']['index_factory'],
                    nprobe=config['faiss'].get('nprobe'),
                    batch_size=config['faiss'].get('batch_size'),
                    compute_curves=config['coverage'].get('compute_curves', False),
                    curve_quantiles=config['coverage'].get('curve_quantiles', [0.80, 0.90, 0.95, 0.99]),
                    curve_max_vectors=config['coverage'].get('curve_max_vectors', 500_000),
                    filter_duplicates=(representation == "flow"),
                    verbose=True,
                )
                
                # Store results
                row = {
                    'space': space_name,
                    'train_dataset': train_name,
                    'train_split': train_split,
                    'eval_dataset': eval_name,
                    'eval_split': eval_split,
                    'train_n_vectors': len(train_vecs),
                    'eval_n_vectors': len(eval_vecs),
                    'train_radius': train_radius_data['radius'],
                    'eval_radius': eval_radius_data['radius'],
                }
                
                # Add all metrics
                row.update(coverage_result['metrics'])
                
                results.append(row)

                # Flush this pair to disk immediately so progress survives OOM/kill
                _flush_row = pd.DataFrame([row])
                _write_header = not output_file.exists() or output_file.stat().st_size == 0
                _flush_row.to_csv(output_file, mode="a", header=_write_header, index=False)
                _done_pairs.add((space_name, train_name, train_split, eval_name, eval_split))

                # Print key metrics
                for k in config['coverage']['k_values']:
                    if f'eval_covered_by_train_qnorm_k{k}' in coverage_result['metrics']:
                        eval_cov = coverage_result['metrics'][f'eval_covered_by_train_qnorm_k{k}']
                        train_cov = coverage_result['metrics'][f'train_covered_by_eval_qnorm_k{k}']
                        print(f"  k={k}: eval_covered={eval_cov:.3f}, train_covered={train_cov:.3f}")

                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                if lazy_load:
                    del train_vecs
                    del eval_vecs
    
    # ======================
    # Save Results
    # ======================
    print(f"\n{'='*80}")
    print(f"SAVING RESULTS")
    print(f"{'='*80}\n")
    
    # Results were flushed pair-by-pair above; just report final count.
    results_df = pd.read_csv(output_file) if output_file.exists() else pd.DataFrame(results)

    print(f"✓ Results saved to: {output_file}")
    print(f"  Total rows: {len(results_df)}")
    print(f"\n{'='*80}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Coverage Pipeline v2.0")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--direction",
        type=str,
        choices=["both", "eval_to_train", "train_to_eval"],
        default=None,
        help="Compute only one direction of cross-dataset distances.",
    )
    parser.add_argument(
        "--kl-only",
        action="store_true",
        help="Compute KL divergence only (skip coverage metrics).",
    )
    args = parser.parse_args()
    
    run_pipeline(args.config, direction_override=args.direction, kl_only=args.kl_only)


if __name__ == "__main__":
    main()
