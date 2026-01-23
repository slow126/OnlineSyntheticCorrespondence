#!/usr/bin/env python3
"""
MMD Pipeline v2.0 - Aligned with Coverage Pipeline v2.0

Implements MMD calculation consistent with FAISS coverage metrics:
  - Same normalization (flow: alpha-scaling, features: PCA + L2)
  - Same vector caching
  - Same train/eval split
  - Efficient: only compute train→eval MMD (not train-to-train)
"""

import argparse
import gc
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import dataset utilities
from scripts.calculate_coverage_faiss import (
    create_dataset_from_config,
    create_mixed_dataset_from_config,
    _is_synthetic_dataset,
)

# Import coverage pipeline modules (reuse for consistency!)
from scripts.coverage import (
    cache,
    calibration,
    spaces,
)

# Import feature extraction
from src.coreset.validation import extract_flow_vectors_from_batch
from src.mmd.encoders import ResNet101Encoder, DinoV3Encoder

# Import MMD
from src.mmd import load_config_from_yaml, StreamingMMD, StreamingMMDTorch


def extract_vectors_from_dataset(
    dataset,
    dataloader,
    representation: str,
    encoder: Optional[object] = None,
    max_vectors: int = 16_000_000,
    flow_per_image_max: int = 2000,
    vectors_per_image: Optional[int] = None,
    seed: int = 42,
    device: str = "cuda",
    verbose: bool = True,
) -> np.ndarray:
    """
    Extract vectors (reuses same logic as coverage pipeline).
    
    Returns:
        (N, D) array of vectors
    """
    if verbose:
        print(f"  Extracting {representation} vectors...")
    
    all_vectors = []
    total_vectors = 0
    
    np.random.seed(seed)
    
    for batch_idx, batch in enumerate(tqdm(dataloader, disable=not verbose, desc="  Extracting")):
        if total_vectors >= max_vectors:
            break
        
        if representation == "flow":
            # Extract flow vectors [x, y, dx, dy]
            vectors = extract_flow_vectors_from_batch(batch, device=device)
            
            if vectors is None or len(vectors) == 0:
                continue
            
            # Sample up to flow_per_image_max per image
            if len(vectors) > flow_per_image_max:
                indices = np.random.choice(len(vectors), size=flow_per_image_max, replace=False)
                vectors = vectors[indices]
        
        else:  # resnet or dino
            # Extract features from images
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

            with torch.no_grad():
                features = encoder.extract_features(images)  # (B*patches, D)

            vectors = features.cpu().numpy()

            if vectors_per_image is not None and vectors_per_image > 0:
                B = images.shape[0]
                total_features = len(vectors)
                features_per_image_actual = total_features // B if B else 0
                if features_per_image_actual > 0:
                    sampled_vectors = []
                    for img_idx in range(B):
                        start_idx = img_idx * features_per_image_actual
                        end_idx = start_idx + features_per_image_actual
                        img_vectors = vectors[start_idx:end_idx]
                        if len(img_vectors) > vectors_per_image:
                            indices = np.random.choice(
                                len(img_vectors),
                                size=vectors_per_image,
                                replace=False,
                            )
                            img_vectors = img_vectors[indices]
                        sampled_vectors.append(img_vectors)
                    if sampled_vectors:
                        vectors = np.vstack(sampled_vectors)
        
        all_vectors.append(vectors)
        total_vectors += len(vectors)
        
        if total_vectors >= max_vectors:
            break
    
    if len(all_vectors) == 0:
        output_dim = 4 if representation == "flow" else encoder.output_dim
        return np.array([]).reshape(0, output_dim)
    
    all_vectors = np.concatenate(all_vectors, axis=0)
    
    # Trim to max_vectors
    if len(all_vectors) > max_vectors:
        all_vectors = all_vectors[:max_vectors]
    
    if verbose:
        print(f"  Extracted {len(all_vectors):,} vectors")
    
    return all_vectors


def _pca_representation_name(representation: str, output_dim: int, l2_normalize: bool) -> str:
    suffix = f"_pca{output_dim}"
    if l2_normalize:
        suffix += "_l2norm"
    return f"{representation}{suffix}"


def _stream_vectors(
    streaming_mmd,
    vectors: np.ndarray,
    dataset_id: str,
    backend: str,
    mmd_device: Optional[torch.device],
    batch_size: int,
) -> None:
    """Stream vectors to MMD in manageable batches to avoid GPU OOM."""
    if len(vectors) == 0:
        return
    if batch_size <= 0 or batch_size >= len(vectors):
        if backend == "torch":
            vectors_tensor = torch.from_numpy(vectors).float().to(mmd_device)
            streaming_mmd.update(dataset_id, vectors_tensor)
        else:
            streaming_mmd.update(dataset_id, vectors)
        return
    for start in range(0, len(vectors), batch_size):
        batch = vectors[start:start + batch_size]
        if backend == "torch":
            vectors_tensor = torch.from_numpy(batch).float().to(mmd_device)
            streaming_mmd.update(dataset_id, vectors_tensor)
        else:
            streaming_mmd.update(dataset_id, batch)


def _normalize_flow_vectors(vectors: np.ndarray, config: dict) -> np.ndarray:
    if config.get('flow_normalization', {}).get('enabled', False):
        img_h, img_w = config['flow_normalization']['image_size']
        return spaces.normalize_flow_vectors(vectors, img_w, img_h)
    return vectors


def _compute_global_alpha(
    train_configs: list,
    config: dict,
    load_vectors_fn,
) -> float:
    alphas = []
    per_dataset = {}
    for ds_config in train_configs:
        dataset_name = ds_config.get('name')
        split = ds_config.get('split')
        vectors, _ = load_vectors_fn(ds_config)
        vectors = _normalize_flow_vectors(vectors, config)
        alpha = calibration.compute_per_dataset_alpha(
            vectors,
            k=config['calibration']['k'],
            use_gpu=True,
            verbose=True,
        )
        per_dataset[f"{dataset_name}_{split}"] = alpha
        if np.isfinite(alpha) and alpha > 0:
            alphas.append(alpha)
    if not alphas:
        raise ValueError("All per-dataset alphas are invalid; check flow vectors or disable calibration.")
    aggregation = config['calibration']['aggregation']
    if aggregation == "geometric_mean":
        global_alpha = float(np.exp(np.mean(np.log(np.array(alphas)))))
    elif aggregation == "median":
        global_alpha = float(np.median(np.array(alphas)))
    else:
        raise ValueError(f"Unknown alpha aggregation: {aggregation}")
    print(f"\nGlobal alpha ({aggregation}): {global_alpha:.6f}")
    return global_alpha


def _load_vectors_for_dataset(
    ds_config: dict,
    cache_dir: Path,
    representation: str,
    config: dict,
    encoder: Optional[object],
    device: str,
    pca_cache_repr: str,
    use_cached_pca: bool,
) -> Tuple[np.ndarray, str]:
    dataset_name = ds_config.get('name')
    split = ds_config.get('split')
    cache_mmap = config.get("cache", {}).get("mmap", False)

    if use_cached_pca:
        vectors = cache.load_cached_vectors(
            cache_dir,
            dataset_name,
            split,
            pca_cache_repr,
            mmap=cache_mmap,
        )
        if vectors is not None:
            return vectors, "cached_pca"

    vectors = cache.load_cached_vectors(
        cache_dir,
        dataset_name,
        split,
        representation,
        mmap=cache_mmap,
    )
    if vectors is not None:
        return vectors, "cached_raw"

    # Cache miss: build dataset and extract
    if ds_config.get('mixed', False):
        dataset = create_mixed_dataset_from_config(
            ds_config['datasets'],
            ds_config['percentages'],
            split,
            config['dataset_params'],
            config['dataset_overrides'],
            seed=config['sampling']['seed'],
        )
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

    num_workers = 0 if is_synthetic else config['num_workers']
    pin_memory = False if is_synthetic else True
    if is_synthetic and config['num_workers'] > 0:
        print(f"  ⚠️  Synthetic dataset detected - forcing num_workers=0 and pin_memory=False")

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None,
    )

    vectors = extract_vectors_from_dataset(
        dataset,
        dataloader,
        representation,
        encoder=encoder,
        max_vectors=config['sampling']['max_vectors'],
        flow_per_image_max=config['sampling']['flow_per_image_max'],
        vectors_per_image=config['sampling'].get('vectors_per_image'),
        seed=config['sampling']['seed'],
        device=device,
        verbose=True,
    )
    cache.save_cached_vectors(cache_dir, dataset_name, split, representation, vectors)

    del dataset, dataloader
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return vectors, "extracted_raw"


def _preprocess_vectors(
    vectors: np.ndarray,
    source: str,
    representation: str,
    config: dict,
    pca_model: Optional[object],
    global_alpha: Optional[float],
) -> np.ndarray:
    if representation == "flow":
        vectors = _normalize_flow_vectors(vectors, config)
        mmd_space = config.get('mmd_space', 'joint')
        if mmd_space == "joint":
            alpha = 1.0 if global_alpha is None else global_alpha
            if global_alpha is None:
                print("  ⚠️  Alpha not computed; defaulting to α=1.0 for joint space.")
            vectors = spaces.to_joint_space(vectors, alpha)
        elif mmd_space == "xy":
            vectors = spaces.to_xy_space(vectors)
        elif mmd_space == "flow":
            vectors = spaces.to_flow_space(vectors)
        else:
            raise ValueError(f"Unknown mmd_space: {mmd_space}")
        return vectors

    if config.get('pca', {}).get('enabled', False):
        if source == "cached_pca":
            return vectors
        if pca_model is None:
            raise ValueError(
                "PCA model is missing; cannot transform raw vectors. "
                "Run the coverage pipeline to generate PCA cache or disable pca.use_cached."
            )
        vectors = cache.apply_pca(pca_model, vectors)
        if config['pca'].get('l2_normalize', False):
            vectors = cache.l2_normalize(vectors)
    return vectors


def run_mmd_pipeline(config_path: str):
    """Run MMD pipeline aligned with coverage pipeline v2."""
    
    print(f"\n{'='*80}")
    print(f"MMD PIPELINE V2.0")
    print(f"{'='*80}\n")
    print(f"Config: {config_path}\n")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    representation = config['representation']
    cache_dir = Path(config['cache']['dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize encoder if needed
    encoder = None
    if representation == "resnet":
        encoder = ResNet101Encoder(device=device)
    elif representation == "dino":
        encoder = DinoV3Encoder(device=device)
    
    # ======================
    # STEP 0: Load/Extract Vectors (streaming)
    # ======================
    print(f"\n{'='*80}")
    print(f"STEP 0: LOAD/EXTRACT VECTORS")
    print(f"{'='*80}\n")

    train_configs = [d for d in config['datasets'] if not d.get('is_eval', False)]
    eval_configs = [d for d in config['datasets'] if d.get('is_eval', False)]

    if not train_configs and not eval_configs:
        raise ValueError("No datasets configured for MMD.")

    pca_cfg = config.get("pca", {})
    pca_enabled = representation in ["resnet", "dino"] and pca_cfg.get("enabled", False)
    use_cached_pca = pca_enabled and bool(pca_cfg.get("use_cached", True))
    pca_cache_repr = representation
    pca_model = None
    if pca_enabled and use_cached_pca:
        pca_cache_repr = _pca_representation_name(
            representation,
            pca_cfg["output_dim"],
            pca_cfg.get("l2_normalize", False),
        )
        pca_model = cache.load_pca_model(cache_dir, representation)

    def load_vectors(ds_config: dict) -> Tuple[np.ndarray, str]:
        return _load_vectors_for_dataset(
            ds_config,
            cache_dir,
            representation,
            config,
            encoder,
            device,
            pca_cache_repr,
            use_cached_pca,
        )
    
    # ======================
    # STEP 1: Alpha Calibration (Flow Only)
    # ======================
    global_alpha = None

    if representation == "flow" and config['calibration']['enabled']:
        print(f"\n{'='*80}")
        print(f"STEP 1: ALPHA CALIBRATION")
        print(f"{'='*80}\n")
        global_alpha = _compute_global_alpha(train_configs, config, load_vectors)

    # ======================
    # STEP 2: Transform to Space for MMD
    # ======================
    print(f"\n{'='*80}")
    print(f"STEP 2: TRANSFORM TO SPACE")
    print(f"{'='*80}\n")

    mmd_space = config.get('mmd_space', 'joint')  # Default to joint for flow
    print(f"MMD space: {mmd_space}")
    
    # ======================
    # STEP 3: Initialize MMD
    # ======================
    print(f"\n{'='*80}")
    print(f"STEP 3: INITIALIZE MMD")
    print(f"{'='*80}\n")
    
    # Load MMD config
    mmd_config_path = config.get('mmd_config_path', 'src/configs/mmd_configs/mmd_config.yaml')
    mmd_preset = config.get('mmd_preset', 'flow_vectors')
    mmd_config = load_config_from_yaml(mmd_config_path, preset=mmd_preset)
    mmd_device_override = config.get("mmd_device")
    if mmd_device_override:
        mmd_config.device = mmd_device_override
    
    # Initialize with first dataset to determine input dim
    first_config = train_configs[0] if train_configs else eval_configs[0]
    first_name = first_config.get('name')
    first_split = first_config.get('split')
    first_id = f"{first_name}_{first_split}"
    print(f"\nLoading first dataset for dimension check: {first_id}")

    first_vectors_raw, first_source = load_vectors(first_config)
    first_vectors = _preprocess_vectors(
        first_vectors_raw,
        first_source,
        representation,
        config,
        pca_model,
        global_alpha,
    )

    actual_dim = first_vectors.shape[1]
    if mmd_config.input_dim != actual_dim:
        print(f"Updating MMD input_dim from {mmd_config.input_dim} to {actual_dim}")
        mmd_config.input_dim = actual_dim

    # Create RFF map
    rff_map = mmd_config.create_rff_map()

    # Create streaming MMD instance
    if mmd_config.backend == 'torch':
        streaming_mmd = StreamingMMDTorch(rff_map)
        mmd_device = rff_map.device
    else:
        streaming_mmd = StreamingMMD(config=mmd_config)
        mmd_device = None

    print(f"StreamingMMD backend: {mmd_config.backend}")
    print(f"Input dimension: {actual_dim}")
    print(f"RFF features: {rff_map.total_features}")
    if mmd_device:
        print(f"Device: {mmd_device}")

    # ======================
    # STEP 4: Stream Vectors to MMD
    # ======================
    print(f"\n{'='*80}")
    print(f"STEP 4: STREAM VECTORS TO MMD")
    print(f"{'='*80}\n")

    mmd_batch_size = int(config.get("mmd_batch_size", 20000))
    dataset_counts: Dict[str, int] = {}
    train_ids: list = []
    eval_ids: list = []

    def _stream_dataset(
        ds_config: dict,
        is_eval: bool,
        vectors_override: Optional[np.ndarray] = None,
        source_override: Optional[str] = None,
    ) -> None:
        dataset_name = ds_config.get('name')
        split = ds_config.get('split')
        dataset_id = f"{dataset_name}_{split}"
        vectors, source = (vectors_override, source_override) if vectors_override is not None else load_vectors(ds_config)
        vectors = _preprocess_vectors(
            vectors,
            source,
            representation,
            config,
            pca_model,
            global_alpha,
        )
        print(f"  [{dataset_id}]: {len(vectors):,} vectors")
        _stream_vectors(
            streaming_mmd,
            vectors,
            dataset_id,
            mmd_config.backend,
            mmd_device,
            mmd_batch_size,
        )
        dataset_counts[dataset_id] = int(len(vectors))
        if is_eval:
            eval_ids.append(dataset_id)
        else:
            train_ids.append(dataset_id)

    print("Streaming train vectors...")
    _stream_dataset(
        first_config,
        is_eval=first_config.get('is_eval', False),
        vectors_override=first_vectors_raw,
        source_override=first_source,
    )

    for ds_config in train_configs:
        if ds_config is first_config:
            continue
        _stream_dataset(ds_config, is_eval=False)

    print("\nStreaming eval vectors...")
    for ds_config in eval_configs:
        if ds_config is first_config:
            continue
        _stream_dataset(ds_config, is_eval=True)
    
    # ======================
    # STEP 5: Compute MMD (train→eval only!)
    # ======================
    print(f"\n{'='*80}")
    print(f"STEP 5: COMPUTE MMD (TRAIN→EVAL ONLY)")
    print(f"{'='*80}\n")
    
    results = []
    
    for train_id in train_ids:
        for eval_id in eval_ids:
            train_name, train_split = train_id.rsplit("_", 1)
            eval_name, eval_split = eval_id.rsplit("_", 1)

            print(f"[{train_id} → {eval_id}]")

            mmd2_val = streaming_mmd.mmd2(train_id, eval_id)
            mmd_val = streaming_mmd.mmd(train_id, eval_id)

            print(f"  MMD² = {mmd2_val:.6f}, MMD = {mmd_val:.6f}")

            results.append({
                'space': mmd_space if representation == 'flow' else 'features',
                'representation': representation,
                'train_dataset': train_name,
                'train_split': train_split,
                'eval_dataset': eval_name,
                'eval_split': eval_split,
                'train_n_vectors': dataset_counts.get(train_id, 0),
                'eval_n_vectors': dataset_counts.get(eval_id, 0),
                'mmd2': mmd2_val,
                'mmd': mmd_val,
            })
    
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
    parser = argparse.ArgumentParser(description="MMD Pipeline v2.0")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    run_mmd_pipeline(args.config)


if __name__ == "__main__":
    main()
