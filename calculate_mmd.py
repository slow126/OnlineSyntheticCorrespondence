"""
Calculate MMD between flow features from different datasets.

This script:
1. Loads datasets using CorrespondenceDataset
2. Extracts flow_full from batches (dense grid format)
3. Converts flows to [x, y, dx, dy] format
4. Filters out invalid flows (inf/nan and zero flows (0,0))
5. Calculates MMD between datasets using the MMD library
"""

import argparse
import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Tuple

from src.data.synth.datasets.CorrespondenceDataset import CorrespondenceDataset
from src.mmd import load_config_from_yaml, StreamingMMD, StreamingMMDTorch, mmd2_rff


def extract_flow_vectors(flow_full: torch.Tensor) -> np.ndarray:
    """
    Extract flow vectors as [x, y, dx, dy] from flow_full dense grid tensor.
    
    Args:
        flow_full: [2, H, W] tensor where:
            - flow_full[0] = dx (horizontal flow)
            - flow_full[1] = dy (vertical flow)
            - Invalid flows are marked with inf/nan or (0,0) vectors
    
    Returns:
        [N, 4] array where N is number of valid flows, columns are [x, y, dx, dy]
        - x, y: pixel coordinates (0-indexed)
        - dx, dy: flow displacement values
        - Filters out: inf/nan values and (0,0) flow vectors (invalid/no-flow regions)
    """
    if flow_full is None:
        return np.empty((0, 4), dtype=np.float32)
    
    # flow_full is [2, H, W]
    _, H, W = flow_full.shape
    dx = flow_full[0].cpu().numpy()  # [H, W]
    dy = flow_full[1].cpu().numpy()  # [H, W]
    
    # Create coordinate grid
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    # y_coords: [H, W], x_coords: [H, W]
    
    # Flatten to vectors
    x_flat = x_coords.flatten()  # [H*W]
    y_flat = y_coords.flatten()  # [H*W]
    dx_flat = dx.flatten()  # [H*W]
    dy_flat = dy.flatten()  # [H*W]
    
    # Filter invalid flows (inf/nan and zero flows)
    # Zero flows (0,0) often represent invalid/no-flow regions
    valid_mask = (
        np.isfinite(dx_flat) & 
        np.isfinite(dy_flat) & 
        ~((dx_flat == 0) & (dy_flat == 0))
    )
    
    # Stack to [N, 4] format: [x, y, dx, dy]
    flow_vectors = np.stack([
        x_flat[valid_mask],
        y_flat[valid_mask],
        dx_flat[valid_mask],
        dy_flat[valid_mask]
    ], axis=1).astype(np.float32)
    
    return flow_vectors


def stream_flows_to_mmd(
    dataloader: DataLoader,
    num_batches: int,
    dataset_name: str,
    streaming_mmd,
    backend: str,
    device: torch.device = None
) -> int:
    """
    Stream flow vectors directly to StreamingMMD without accumulating in memory.
    
    Args:
        dataloader: DataLoader for the dataset
        num_batches: Number of batches to process
        dataset_name: Name of dataset (for logging)
        streaming_mmd: StreamingMMD or StreamingMMDTorch instance to update
        backend: 'numpy' or 'torch'
        device: Device for PyTorch tensors (if backend is 'torch')
    
    Returns:
        Total number of flow vectors processed
    """
    batches_processed = 0
    total_vectors = 0
    
    print(f"  Streaming flows from {dataset_name} to MMD...")
    
    for batch_idx, batch in enumerate(dataloader):
        if batches_processed >= num_batches:
            break
        
        # Get flow_full from batch
        if 'flow_full' in batch:
            flow_full_batch = batch['flow_full']
        elif 'flow' in batch:
            flow_full_batch = batch['flow']
        else:
            print(f"    Warning: No flow_full or flow in batch {batch_idx}")
            continue
        
        if flow_full_batch is None:
            print(f"    Warning: flow_full is None in batch {batch_idx}")
            continue
        
        # flow_full_batch is [B, 2, H, W]
        batch_size = flow_full_batch.shape[0]
        
        # Process each sample in batch and update StreamingMMD immediately
        for sample_idx in range(batch_size):
            flow_full = flow_full_batch[sample_idx]  # [2, H, W]
            flow_vectors = extract_flow_vectors(flow_full)  # [N, 4]
            
            if len(flow_vectors) > 0:
                # Update StreamingMMD immediately - don't accumulate!
                if backend == 'torch':
                    flows_tensor = torch.from_numpy(flow_vectors).float().to(device)
                    streaming_mmd.update(dataset_name, flows_tensor)
                else:
                    streaming_mmd.update(dataset_name, flow_vectors)
                
                total_vectors += len(flow_vectors)
        
        batches_processed += 1
        
        if (batch_idx + 1) % 5 == 0:
            print(f"    Processed {batch_idx + 1} batches, {total_vectors} flow vectors streamed...")
    
    print(f"    Streamed {total_vectors} flow vectors from {dataset_name}")
    return total_vectors


def create_dataset_from_config(
    dataset_name: str,
    split: str,
    common_params: dict,
    dataset_overrides: dict
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


def main():
    parser = argparse.ArgumentParser(description='Calculate MMD between flow features from datasets')
    parser.add_argument('--config', type=str, 
                       default='src/configs/mmd_configs/flow_mmd_config.yaml',
                       help='Path to flow MMD config YAML file')
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
    common_params = config['dataset_params']
    dataset_overrides = config.get('dataset_overrides', {})
    output_config = config.get('output', {})
    
    # Load MMD config
    mmd_config_path = 'src/configs/mmd_configs/mmd_config.yaml'
    print(f"Loading MMD config preset: {mmd_preset}")
    mmd_config = load_config_from_yaml(mmd_config_path, preset=mmd_preset)
    
    # Verify input_dim matches flow vector dimension (4: x, y, dx, dy)
    # Only warn if using a preset that's not designed for flow vectors
    if mmd_config.input_dim != 4:
        if mmd_preset != 'flow_vectors':
            print(f"Warning: MMD config input_dim={mmd_config.input_dim}, but flow vectors are 4D [x, y, dx, dy]")
            print(f"Updating MMD config input_dim to 4")
        mmd_config.input_dim = 4
    
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
    if device is not None:
        print(f"Using device: {device}")
    
    # Process datasets and stream flows directly to MMD (no accumulation!)
    print("\n" + "="*60)
    print("STREAMING FLOWS TO MMD (NO MEMORY ACCUMULATION)")
    print("="*60)
    
    dataset_vector_counts = {}
    
    for ds_config in datasets_config:
        dataset_name = ds_config['name']
        split = ds_config['split']
        num_batches = ds_config['num_batches']
        
        # Create unique identifier combining dataset name and split
        dataset_id = f"{dataset_name}_{split}"
        
        # Create dataset
        dataset = create_dataset_from_config(
            dataset_name, split, common_params, dataset_overrides
        )
        
        # Create dataloader
        # Use num_workers=0 for synthetic (GPU-bound rendering)
        workers = 0 if dataset_name == 'synthetic' else num_workers
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            pin_memory=False
        )
        
        # Stream flows directly to StreamingMMD (no accumulation!)
        # Use dataset_id instead of dataset_name to treat splits as unique
        vector_count = stream_flows_to_mmd(
            dataloader, num_batches, dataset_id, 
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
        results_file = output_config.get('results_file', 'flow_mmd_results.csv')
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

