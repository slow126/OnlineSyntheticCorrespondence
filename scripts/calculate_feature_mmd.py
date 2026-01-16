"""
Calculate MMD between deep features from different datasets.

This script:
1. Loads datasets using CorrespondenceDataset
2. Extracts deep features from images using encoder backbones (ResNet101, Dino, etc.)
3. Flattens features to [N, C] format where N = B*H*W and C is feature dimension
4. Calculates MMD between datasets using the MMD library iteratively
"""

import argparse
import sys
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synth.datasets.CorrespondenceDataset import CorrespondenceDataset
from src.data.synth.datasets.MixedCorrespondenceDataset import MixedCorrespondenceDataset
from src.mmd import (
    load_config_from_yaml, 
    StreamingMMD, 
    StreamingMMDTorch,
    BaseFeatureEncoder,
    ResNet101Encoder,
    DinoV3Encoder,
)


def _is_synthetic_dataset(name: Optional[str]) -> bool:
    return isinstance(name, str) and name.startswith("synthetic")


def extract_features_from_batch(
    batch: dict,
    encoder: BaseFeatureEncoder,
    device: torch.device
) -> torch.Tensor:
    """
    Extract features from a batch.
    
    Args:
        batch: Batch dictionary from dataloader
        encoder: Feature encoder instance
        device: Device for tensors
    
    Returns:
        Features tensor [B*H*W, C] ready for MMD streaming
    """
    # Extract source image from batch
    if 'src_img' in batch:
        img = batch['src_img']
    elif 'source' in batch:
        img = batch['source']
    elif 'image0' in batch:
        img = batch['image0']
    else:
        raise ValueError(f"Could not find source image in batch. Available keys: {batch.keys()}")
    
    # Ensure image is a tensor
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img)
    
    # Extract features using encoder
    features = encoder.extract_features(img)  # [B*H*W, C]
    
    return features


def stream_features_to_mmd(
    dataloader: DataLoader,
    num_batches: Optional[int],
    dataset_name: str,
    encoder: BaseFeatureEncoder,
    streaming_mmd,
    backend: str,
    device: torch.device = None
) -> int:
    """
    Stream features directly to StreamingMMD without accumulating in memory.
    
    Args:
        dataloader: DataLoader for the dataset
        num_batches: Number of batches to process (None = full dataset)
        dataset_name: Name of dataset (for logging)
        encoder: Feature encoder instance
        streaming_mmd: StreamingMMD or StreamingMMDTorch instance to update
        backend: 'numpy' or 'torch'
        device: Device for PyTorch tensors (if backend is 'torch')
    
    Returns:
        Total number of feature vectors processed
    """
    batches_processed = 0
    total_vectors = 0
    
    print(f"  Streaming features from {dataset_name} to MMD...")
    
    for batch_idx, batch in enumerate(dataloader):
        if num_batches is not None and batches_processed >= num_batches:
            break
        
        try:
            # Extract features from batch
            features = extract_features_from_batch(batch, encoder, device)  # [B*H*W, C]
            
            if features.shape[0] > 0:
                if features.dtype != torch.float32:
                    features = features.float()
                # Update StreamingMMD immediately - don't accumulate!
                if backend == 'torch':
                    # Features are already tensors, ensure they're on correct device
                    features = features.to(device)
                    streaming_mmd.update(dataset_name, features)
                else:
                    # Convert to numpy for numpy backend
                    features_np = features.cpu().numpy().astype(np.float32, copy=False)
                    streaming_mmd.update(dataset_name, features_np)
                
                total_vectors += features.shape[0]
        except Exception as e:
            print(f"    Warning: Error processing batch {batch_idx}: {e}")
            continue
        
        batches_processed += 1
        
        if (batch_idx + 1) % 5 == 0:
            print(f"    Processed {batch_idx + 1} batches, {total_vectors} feature vectors streamed...")
    
    print(f"    Streamed {total_vectors} feature vectors from {dataset_name}")
    return total_vectors


def create_dataset_from_config(
    dataset_name: str,
    split: str,
    common_params: dict,
    dataset_overrides: dict,
    entry_overrides: dict = None
) -> CorrespondenceDataset:
    """
    Create a CorrespondenceDataset from config parameters.
    
    Args:
        dataset_name: Name of dataset
        split: Split to use ('train', 'val', 'test')
        common_params: Common parameters for all datasets
        dataset_overrides: Dataset-specific overrides
    
    Returns:
        CorrespondenceDataset instance
    """
    # Start with common parameters
    dataset_config = common_params.copy()
    
    # Apply dataset-specific overrides
    if dataset_name in dataset_overrides:
        dataset_config.update(dataset_overrides[dataset_name])
    if entry_overrides:
        dataset_config.update(entry_overrides)
    
    # Handle size tuple
    if 'size' in dataset_config and isinstance(dataset_config['size'], list):
        dataset_config['size'] = tuple(dataset_config['size'])
    
    # Set split
    dataset_config['split'] = split
    
    # Handle max_kps: null -> None
    if 'max_kps' in dataset_config and dataset_config['max_kps'] is None:
        dataset_config['max_kps'] = None
    
    # Add dataset-specific parameters based on dataset type
    if dataset_name == 'synthetic':
        # Synthetic-specific params are already in overrides
        pass
    elif dataset_name == 'tss':
        dataset_config['thres'] = dataset_config.get('thres', 'img')
        dataset_config['reverse_flow'] = dataset_config.get('reverse_flow', False)
    elif dataset_name == 'middlebury':
        dataset_config['reverse_flow'] = dataset_config.get('reverse_flow', False)
    elif dataset_name == 'pointodyssey':
        dataset_config['reverse_flow'] = dataset_config.get('reverse_flow', True)
        dataset_config['thres'] = dataset_config.get('thres', 'img')
    elif dataset_name in ['kitti2012', 'kitti2015']:
        dataset_config['reverse_flow'] = dataset_config.get('reverse_flow', False)
        dataset_config['thres'] = dataset_config.get('thres', 'img')
        # Handle special case for kitti_val_use_full_training
        kitti_val_use_full_training = dataset_config.get('kitti_val_use_full_training', False)
        if kitti_val_use_full_training and dataset_config.get('split') == 'val':
            dataset_config['split'] = 'training'  # Use full training set for validation
    elif dataset_name == 'flyingthings':
        dataset_config['reverse_flow'] = dataset_config.get('reverse_flow', True)
    # For spair, pfpascal, pfwillow: datapath is already set in overrides
    
    print(f"Creating dataset: {dataset_name} (split: {split})")
    dataset = CorrespondenceDataset(dataset_name, **dataset_config)
    return dataset


def create_mixed_dataset_from_config(
    datasets_list: list,
    percentages: list,
    split: str,
    common_params: dict,
    dataset_overrides: dict,
    epoch_size: Optional[int] = None,
    seed: Optional[int] = None
) -> MixedCorrespondenceDataset:
    if len(datasets_list) != len(percentages):
        raise ValueError(f"Number of datasets ({len(datasets_list)}) must match number of percentages ({len(percentages)})")
    created_datasets = []
    for dataset_name in datasets_list:
        ds_config = common_params.copy()
        if dataset_name in dataset_overrides:
            ds_config.update(dataset_overrides[dataset_name])
        if 'size' in ds_config and isinstance(ds_config['size'], list):
            ds_config['size'] = tuple(ds_config['size'])
        ds_config['split'] = split
        if 'max_kps' in ds_config and ds_config['max_kps'] is None:
            ds_config['max_kps'] = None
        if dataset_name == 'tss':
            ds_config['thres'] = ds_config.get('thres', 'img')
            ds_config['reverse_flow'] = ds_config.get('reverse_flow', False)
        elif dataset_name == 'middlebury':
            ds_config['reverse_flow'] = ds_config.get('reverse_flow', False)
        elif dataset_name == 'pointodyssey':
            ds_config['reverse_flow'] = ds_config.get('reverse_flow', True)
            ds_config['thres'] = ds_config.get('thres', 'img')
        elif dataset_name in ['kitti2012', 'kitti2015']:
            ds_config['reverse_flow'] = ds_config.get('reverse_flow', False)
            ds_config['thres'] = ds_config.get('thres', 'img')
            kitti_val_use_full_training = ds_config.get('kitti_val_use_full_training', False)
            if kitti_val_use_full_training and ds_config.get('split') == 'val':
                ds_config['split'] = 'training'
        elif dataset_name == 'flyingthings':
            ds_config['reverse_flow'] = ds_config.get('reverse_flow', True)
        print(f"Creating sub-dataset: {dataset_name} (split: {split})")
        sub_dataset = CorrespondenceDataset(dataset_name, **ds_config)
        created_datasets.append(sub_dataset)
    print(f"Creating mixed dataset with {len(created_datasets)} datasets")
    mixed_dataset = MixedCorrespondenceDataset(
        datasets=created_datasets,
        percentages=percentages,
        epoch_size=epoch_size,
        seed=seed,
    )
    return mixed_dataset


def create_encoder(encoder_name: str, device: torch.device) -> BaseFeatureEncoder:
    """
    Create encoder instance based on name.
    
    Args:
        encoder_name: Name of encoder ('resnet101', 'dino', etc.)
        device: Device to run encoder on
    
    Returns:
        Encoder instance
    """
    if encoder_name == 'resnet101':
        return ResNet101Encoder(device=device)
    if encoder_name == 'dino':
        return DinoV3Encoder(device=device)
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}. Supported: 'resnet101', 'dino'")


def main():
    parser = argparse.ArgumentParser(description='Calculate MMD between deep features from datasets')
    parser.add_argument('--config', type=str, 
                       default='src/configs/mmd_configs/feature_mmd_config.yaml',
                       help='Path to feature MMD config YAML file')
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract config sections
    datasets_config = config['datasets']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    mmd_preset = config['mmd_preset']
    encoder_name = config.get('encoder', 'resnet101')
    common_params = config['dataset_params']
    dataset_overrides = config.get('dataset_overrides', {})
    sampling_cfg = config.get('sampling', {})
    batch_limit_default = sampling_cfg.get('batch_limit', 500)
    shuffle_default = bool(sampling_cfg.get('shuffle', True))
    output_config = config.get('output', {})
    
    # Load MMD config
    mmd_config_path = 'src/configs/mmd_configs/mmd_config.yaml'
    print(f"Loading MMD config preset: {mmd_preset}")
    mmd_config = load_config_from_yaml(mmd_config_path, preset=mmd_preset)
    
    # Create encoder to get feature dimension
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = create_encoder(encoder_name, device)
    feature_dim = encoder.feature_dim
    
    # Verify input_dim matches feature dimension
    if mmd_config.input_dim != feature_dim:
        print(f"Warning: MMD config input_dim={mmd_config.input_dim}, but {encoder_name} features are {feature_dim}D")
        print(f"Updating MMD config input_dim to {feature_dim}")
        mmd_config.input_dim = feature_dim
    
    # Create RFF map and StreamingMMD BEFORE processing datasets
    print("\n" + "="*60)
    print("INITIALIZING MMD COMPUTATION")
    print("="*60)
    
    rff_map = mmd_config.create_rff_map()
    
    # Create streaming MMD instance
    if mmd_config.backend == 'torch':
        from src.mmd import StreamingMMDTorch
        streaming_mmd = StreamingMMDTorch(rff_map)
        device = rff_map.device
    else:
        streaming_mmd = StreamingMMD(config=mmd_config)
        device = None
    
    print(f"Created StreamingMMD with backend: {mmd_config.backend}")
    print(f"Feature encoder: {encoder_name} (dim={feature_dim})")
    if device is not None:
        print(f"Using device: {device}")
    
    # Process datasets and stream features directly to MMD (no accumulation!)
    print("\n" + "="*60)
    print("STREAMING FEATURES TO MMD (NO MEMORY ACCUMULATION)")
    print("="*60)
    
    dataset_vector_counts = {}
    
    for ds_config in datasets_config:
        is_mixed = ds_config.get('mixed', False) or 'datasets' in ds_config
        split = ds_config['split']
        num_batches = ds_config.get('num_batches', batch_limit_default)
        entry_overrides = ds_config.get('overrides', None)

        if is_mixed:
            datasets_list = ds_config.get('datasets', [])
            percentages = ds_config.get('percentages', [])
            label = ds_config.get('name')
            if not label:
                if len(percentages) == 2 and len(datasets_list) == 2:
                    pct1 = int(percentages[0] * 100)
                    pct2 = int(percentages[1] * 100)
                    label = f"{datasets_list[0]}_{datasets_list[1]}_{pct1}_{pct2}"
                else:
                    label = "+".join(datasets_list)
            dataset_id = f"{label}_{split}"
            dataset = create_mixed_dataset_from_config(
                datasets_list,
                percentages,
                split,
                common_params,
                dataset_overrides,
                epoch_size=ds_config.get('epoch_size', None),
                seed=ds_config.get('seed', None),
            )
            has_synthetic = any(_is_synthetic_dataset(ds_name) for ds_name in datasets_list)
            workers = 0 if has_synthetic else num_workers
        else:
            label = ds_config['name']
            dataset_name = ds_config.get('dataset_name', label)
            dataset_id = f"{label}_{split}"
            dataset = create_dataset_from_config(
                dataset_name, split, common_params, dataset_overrides, entry_overrides
            )
            workers = 0 if _is_synthetic_dataset(dataset_name) else num_workers
        
        shuffle = bool(ds_config.get('shuffle', shuffle_default))
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn,
            pin_memory=False
        )
        
        # Stream features directly to StreamingMMD (no accumulation!)
        # Use dataset_id instead of dataset_name to treat splits as unique
        vector_count = stream_features_to_mmd(
            dataloader, num_batches, dataset_id, encoder,
            streaming_mmd, mmd_config.backend, device
        )
        dataset_vector_counts[dataset_id] = vector_count
    
    # Now calculate pairwise MMD (all data already streamed)
    print("\n" + "="*60)
    print("CALCULATING MMD BETWEEN DATASETS")
    print("="*60)
    
    # Calculate pairwise MMD
    dataset_names = list(dataset_vector_counts.keys())
    mmd_results = []
    
    print("\nPairwise MMD² results:")
    print("-" * 60)
    
    for i, name1 in enumerate(dataset_names):
        for name2 in dataset_names[i+1:]:
            if dataset_vector_counts[name1] == 0 or dataset_vector_counts[name2] == 0:
                print(f"  {name1} vs {name2}: SKIPPED (no vectors)")
                continue
            
            mmd2_val = streaming_mmd.mmd2(name1, name2)
            mmd_val = streaming_mmd.mmd(name1, name2)
            
            print(f"  {name1:15} vs {name2:15}: MMD² = {mmd2_val:.6f}, MMD = {mmd_val:.6f}")
            
            mmd_results.append({
                'dataset1': name1,
                'dataset2': name2,
                'mmd2': mmd2_val,
                'mmd': mmd_val,
                'num_vectors1': dataset_vector_counts[name1],
                'num_vectors2': dataset_vector_counts[name2]
            })
    
    # Save results if requested
    if output_config.get('save_results', False):
        results_file = output_config.get('results_file', 'feature_mmd_results.csv')
        import csv
        
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset1', 'split1', 'dataset2', 'split2', 'mmd2', 'mmd', 'num_vectors1', 'num_vectors2'])
            for result in mmd_results:
                # Parse dataset_id format: "dataset_name_split"
                dataset1_id = result['dataset1']
                dataset2_id = result['dataset2']
                
                # Split the ID into name and split (handle edge cases)
                if '_' in dataset1_id:
                    dataset1_name, split1 = dataset1_id.rsplit('_', 1)
                else:
                    dataset1_name, split1 = dataset1_id, 'unknown'
                
                if '_' in dataset2_id:
                    dataset2_name, split2 = dataset2_id.rsplit('_', 1)
                else:
                    dataset2_name, split2 = dataset2_id, 'unknown'
                
                writer.writerow([
                    dataset1_name,
                    split1,
                    dataset2_name,
                    split2,
                    result['mmd2'],
                    result['mmd'],
                    result['num_vectors1'],
                    result['num_vectors2']
                ])
        
        print(f"\nResults saved to: {results_file}")
    
    print("="*60)
    print("Done!")


if __name__ == "__main__":
    main()
