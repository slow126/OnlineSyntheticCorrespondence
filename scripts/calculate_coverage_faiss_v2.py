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
            
            B = images.shape[0]
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


def run_pipeline(config_path: str):
    """Run the full coverage pipeline."""
    
    print(f"\n{'='*80}")
    print(f"COVERAGE PIPELINE V2.0")
    print(f"{'='*80}\n")
    print(f"Config: {config_path}\n")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    representation = config['representation']
    cache_dir = Path(config['cache']['dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() and config['faiss']['use_gpu'] else "cpu"
    
    # Initialize encoder if needed
    encoder = None
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

    train_vectors = {}
    eval_vectors = {}

    stream_pca = False
    pca_model = None
    cache_repr = representation
    if representation in ["resnet", "dino"] and config.get("pca", {}).get("enabled", False):
        stream_pca = config["pca"].get("streaming", True)
        if stream_pca:
            cache_repr = _pca_representation_name(
                representation,
                config["pca"]["output_dim"],
                config["pca"].get("l2_normalize", False),
            )
            pca_model = cache.load_pca_model(cache_dir, representation)

    if stream_pca and pca_model is None:
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
        
        # Try to load from cache first
        vectors = cache.load_cached_vectors(
            cache_dir,
            dataset_name,
            split,
            cache_repr,
            mmap=config.get("cache", {}).get("mmap", False),
        )
        
        dataset = None
        dataloader = None
        
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
            # Debug: check vectors after loading
            if representation == "flow":
                print(f"  Post-load check:")
                print(f"    Shape: {vectors.shape}")
                print(f"    x: [{vectors[:, 0].min():.2f}, {vectors[:, 0].max():.2f}]")
                print(f"    y: [{vectors[:, 1].min():.2f}, {vectors[:, 1].max():.2f}]")
                print(f"    dx: [{vectors[:, 2].min():.2f}, {vectors[:, 2].max():.2f}]")
                print(f"    dy: [{vectors[:, 3].min():.2f}, {vectors[:, 3].max():.2f}]")
        
        # Store vectors
        key = (dataset_name, split)
        if is_eval:
            eval_vectors[key] = vectors
        else:
            train_vectors[key] = vectors
        
        # Clean up (only delete if they were created)
        if dataset is not None:
            del dataset
        if dataloader is not None:
            del dataloader
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    
    print(f"\nLoaded {len(train_vectors)} train sets, {len(eval_vectors)} eval sets")
    
    # ======================
    # PREPROCESSING
    # ======================
    if representation == "flow":
        # Normalize flow vectors to [-1, 1]
        if config['flow_normalization']['enabled']:
            print(f"\nNormalizing flow vectors to [-1, 1]...")
            img_h, img_w = config['flow_normalization']['image_size']
            
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
    
    if representation == "flow" and config['calibration']['enabled']:
        print(f"\n{'='*80}")
        print(f"STEP 1: ALPHA CALIBRATION")
        print(f"{'='*80}\n")
        
        global_alpha, per_dataset_alphas = calibration.load_or_compute_alpha(
            cache_dir,
            train_vectors,
            k=config['calibration']['k'],
            aggregation=config['calibration']['aggregation'],
            use_gpu=config['faiss']['use_gpu'],
            force_recompute=config['calibration']['force_recompute'],
            verbose=True,
        )
    
    # ======================
    # STEP 2: Define Spaces
    # ======================
    print(f"\n{'='*80}")
    print(f"STEP 2: DEFINE SPACES")
    print(f"{'='*80}\n")
    
    enabled_spaces = config['spaces']['enabled']
    print(f"Enabled spaces: {enabled_spaces}")
    
    # Transform vectors into each space
    space_train_vectors = {}
    space_eval_vectors = {}
    
    for space_name in enabled_spaces:
        print(f"\nTransforming to {space_name} space...")
        
        space_train_vectors[space_name] = {}
        space_eval_vectors[space_name] = {}
        
        for key, vectors in train_vectors.items():
            if representation == "flow":
                if space_name == "xy":
                    space_train_vectors[space_name][key] = spaces.to_xy_space(vectors)
                elif space_name == "flow":
                    space_train_vectors[space_name][key] = spaces.to_flow_space(vectors)
                elif space_name == "joint":
                    space_train_vectors[space_name][key] = spaces.to_joint_space(vectors, global_alpha)
            else:  # features
                space_train_vectors[space_name][key] = vectors
        
        for key, vectors in eval_vectors.items():
            if representation == "flow":
                if space_name == "xy":
                    space_eval_vectors[space_name][key] = spaces.to_xy_space(vectors)
                elif space_name == "flow":
                    space_eval_vectors[space_name][key] = spaces.to_flow_space(vectors)
                elif space_name == "joint":
                    space_eval_vectors[space_name][key] = spaces.to_joint_space(vectors, global_alpha)
            else:  # features
                space_eval_vectors[space_name][key] = vectors
        
        print(f"  {space_name}: {len(space_train_vectors[space_name])} train, {len(space_eval_vectors[space_name])} eval")
    
    # ======================
    # STEP 3: Self-Radius Computation
    # ======================
    print(f"\n{'='*80}")
    print(f"STEP 3: SELF-RADIUS COMPUTATION")
    print(f"{'='*80}\n")
    
    all_radii = {}  # space_name -> {(dataset_name, split): radius_data}
    
    for space_name in enabled_spaces:
        all_datasets_in_space = {**space_train_vectors[space_name], **space_eval_vectors[space_name]}
        
        radii_for_space = radius.compute_all_radii(
            cache_dir,
            all_datasets_in_space,
            space=space_name,
            k=config['coverage']['self_radius_k'],
            quantile=config['coverage']['radius_quantile'],
            neighbor_agg=config['coverage']['neighbor_agg'],
            alpha=global_alpha if space_name == "joint" else None,
            normalization=config.get('flow_normalization', {}).get('scheme', 'norm2x1'),
            distance_metric=config['distance_metric']['name'],
            use_gpu=config['faiss']['use_gpu'],
            index_factory=config['faiss']['index_factory'],
            force_recompute=False,
            verbose=True,
        )
        
        all_radii[space_name] = radii_for_space
    
    # ======================
    # STEPS 4-5: Coverage Metrics
    # ======================
    print(f"\n{'='*80}")
    print(f"STEPS 4-5: COVERAGE METRICS")
    print(f"{'='*80}\n")
    
    results = []
    
    for space_name in enabled_spaces:
        print(f"\n{'='*60}")
        print(f"SPACE: {space_name.upper()}")
        print(f"{'='*60}\n")
        
        for train_key in space_train_vectors[space_name].keys():
            for eval_key in space_eval_vectors[space_name].keys():
                train_name, train_split = train_key
                eval_name, eval_split = eval_key
                
                print(f"\n[{train_name}/{train_split} → {eval_name}/{eval_split}]")
                
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
                    k_max=config['coverage']['k_max'],
                    k_values=config['coverage']['k_values'],
                    use_gpu=config['faiss']['use_gpu'],
                    index_factory=config['faiss']['index_factory'],
                    batch_size=config['faiss'].get('batch_size'),
                    compute_curves=config['coverage']['compute_curves'],
                    curve_quantiles=config['coverage']['curve_quantiles'],
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
                
                # Print key metrics
                for k in config['coverage']['k_values']:
                    if f'eval_covered_by_train_qnorm_k{k}' in coverage_result['metrics']:
                        eval_cov = coverage_result['metrics'][f'eval_covered_by_train_qnorm_k{k}']
                        train_cov = coverage_result['metrics'][f'train_covered_by_eval_qnorm_k{k}']
                        print(f"  k={k}: eval_covered={eval_cov:.3f}, train_covered={train_cov:.3f}")
    
    # ======================
    # Save Results
    # ======================
    print(f"\n{'='*80}")
    print(f"SAVING RESULTS")
    print(f"{'='*80}\n")
    
    results_df = pd.DataFrame(results)
    output_file = Path(config['output']['results_file'])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)
    
    print(f"✓ Results saved to: {output_file}")
    print(f"  Total rows: {len(results_df)}")
    print(f"\n{'='*80}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Coverage Pipeline v2.0")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
