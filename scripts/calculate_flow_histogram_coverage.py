"""
Calculate stratified histogram coverage between flow distributions from different datasets.

This script:
1. Loads datasets using CorrespondenceDataset
2. Extracts flow_full from batches (dense grid format)
3. Builds spatially-stratified polar histograms of flow vectors
4. Calculates pairwise coverage metrics between datasets

Coverage metric: proportion of val histogram mass that falls in bins where train has mass
    coverage = H_val[H_train > 0].sum() / H_val.sum()
"""

import argparse
import csv
import sys
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synth.datasets.CorrespondenceDataset import CorrespondenceDataset


def build_spatial_histogram(
    flow_full: torch.Tensor,
    spatial_bins: int = 32,
    mag_bins: int = 8,
    dir_bins: int = 16,
    max_magnitude: float = 50.0,
) -> np.ndarray:
    """
    Build a spatially-stratified polar histogram from a dense flow field.
    
    Args:
        flow_full: [2, H, W] tensor (dx, dy)
        spatial_bins: Number of spatial bins per dimension (creates spatial_bins x spatial_bins grid)
        mag_bins: Number of magnitude bins (log-spaced)
        dir_bins: Number of direction bins (uniform over [-pi, pi))
        max_magnitude: Maximum magnitude for binning (values above are clipped to last bin)
    
    Returns:
        H: [spatial_bins, spatial_bins, mag_bins, dir_bins] histogram
    """
    if flow_full is None:
        return np.zeros((spatial_bins, spatial_bins, mag_bins, dir_bins), dtype=np.float64)
    
    _, H, W = flow_full.shape
    dx = flow_full[0].cpu().numpy()  # [H, W]
    dy = flow_full[1].cpu().numpy()  # [H, W]
    
    # Create coordinate grid
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Flatten
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    dx_flat = dx.flatten()
    dy_flat = dy.flatten()
    
    # Filter invalid flows (inf/nan and zero flows)
    valid_mask = (
        np.isfinite(dx_flat) & 
        np.isfinite(dy_flat) & 
        ~((dx_flat == 0) & (dy_flat == 0))
    )
    
    if not valid_mask.any():
        return np.zeros((spatial_bins, spatial_bins, mag_bins, dir_bins), dtype=np.float64)
    
    x_valid = x_flat[valid_mask]
    y_valid = y_flat[valid_mask]
    dx_valid = dx_flat[valid_mask]
    dy_valid = dy_flat[valid_mask]
    
    # Compute spatial bins
    sx = np.clip((x_valid / W * spatial_bins).astype(int), 0, spatial_bins - 1)
    sy = np.clip((y_valid / H * spatial_bins).astype(int), 0, spatial_bins - 1)
    
    # Compute polar coordinates
    mag = np.sqrt(dx_valid**2 + dy_valid**2)
    direction = np.arctan2(dy_valid, dx_valid)  # [-pi, pi)
    
    # Bin magnitude (log-spaced)
    # log1p(mag) / log1p(max_mag) maps [0, max_mag] to [0, 1]
    mag_normalized = np.log1p(mag) / np.log1p(max_magnitude)
    mag_bin = np.clip((mag_normalized * mag_bins).astype(int), 0, mag_bins - 1)
    
    # Bin direction (uniform over [-pi, pi))
    dir_normalized = (direction + np.pi) / (2 * np.pi)  # [0, 1)
    dir_bin = np.clip((dir_normalized * dir_bins).astype(int), 0, dir_bins - 1)
    
    # Build histogram
    hist = np.zeros((spatial_bins, spatial_bins, mag_bins, dir_bins), dtype=np.float64)
    np.add.at(hist, (sy, sx, mag_bin, dir_bin), 1)
    
    return hist


def stream_flows_to_histogram(
    dataloader: DataLoader,
    num_batches: int,
    dataset_name: str,
    spatial_bins: int,
    mag_bins: int,
    dir_bins: int,
    max_magnitude: float,
) -> Tuple[np.ndarray, int]:
    """
    Stream flow data and accumulate into a histogram.
    
    Returns:
        (histogram, total_vectors)
    """
    hist = np.zeros((spatial_bins, spatial_bins, mag_bins, dir_bins), dtype=np.float64)
    total_vectors = 0
    batches_processed = 0
    
    pbar = tqdm(
        dataloader,
        total=num_batches if num_batches else len(dataloader),
        desc=f"  {dataset_name}",
        unit="batch",
    )
    
    for batch_idx, batch in enumerate(pbar):
        if num_batches is not None and batches_processed >= num_batches:
            break
        
        # Get flow_full from batch
        if 'flow_full' in batch:
            flow_full_batch = batch['flow_full']
        elif 'flow' in batch:
            flow_full_batch = batch['flow']
        else:
            continue
        
        if flow_full_batch is None:
            continue
        
        # flow_full_batch is [B, 2, H, W]
        batch_size = flow_full_batch.shape[0]
        
        for sample_idx in range(batch_size):
            flow_full = flow_full_batch[sample_idx]
            sample_hist = build_spatial_histogram(
                flow_full, spatial_bins, mag_bins, dir_bins, max_magnitude
            )
            hist += sample_hist
            total_vectors += int(sample_hist.sum())
        
        batches_processed += 1
        pbar.set_postfix(vectors=total_vectors)
    
    return hist, total_vectors


def compute_histogram_coverage(H_train: np.ndarray, H_val: np.ndarray) -> float:
    """
    Compute coverage: proportion of val mass in bins where train has mass.
    
    coverage = H_val[H_train > 0].sum() / H_val.sum()
    """
    val_total = H_val.sum()
    if val_total == 0:
        return 0.0
    
    covered_mass = H_val[H_train > 0].sum()
    return covered_mass / val_total


def create_dataset_from_config(
    dataset_name: str,
    split: str,
    common_params: dict,
    dataset_overrides: dict
) -> CorrespondenceDataset:
    """Create a CorrespondenceDataset from config parameters."""
    dataset_config = common_params.copy()
    
    if dataset_name in dataset_overrides:
        dataset_config.update(dataset_overrides[dataset_name])
    
    if 'size' in dataset_config and isinstance(dataset_config['size'], list):
        dataset_config['size'] = tuple(dataset_config['size'])
    
    dataset_config['split'] = split
    
    if dataset_name == 'tss':
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
        kitti_val_use_full_training = dataset_config.get('kitti_val_use_full_training', False)
        if kitti_val_use_full_training and dataset_config.get('split') == 'val':
            dataset_config['split'] = 'training'
    elif dataset_name == 'flyingthings':
        dataset_config['reverse_flow'] = dataset_config.get('reverse_flow', True)
    
    print(f"Creating dataset: {dataset_name} (split: {split})")
    return CorrespondenceDataset(dataset_name, **dataset_config)


def main():
    parser = argparse.ArgumentParser(
        description='Calculate stratified histogram coverage between flow distributions'
    )
    parser.add_argument(
        '--config', type=str,
        default='src/configs/mmd_configs/flow_histogram_config.yaml',
        help='Path to config YAML file'
    )
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract config sections
    datasets_config = config['datasets']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    common_params = config['dataset_params']
    dataset_overrides = config.get('dataset_overrides', {})
    output_config = config.get('output', {})
    
    # Histogram parameters
    histogram_params = config.get('histogram_params', {})
    spatial_bins = histogram_params.get('spatial_bins', 32)
    mag_bins = histogram_params.get('mag_bins', 8)
    dir_bins = histogram_params.get('dir_bins', 16)
    max_magnitude = histogram_params.get('max_magnitude', 50.0)
    
    print(f"\nHistogram params: spatial={spatial_bins}x{spatial_bins}, "
          f"mag_bins={mag_bins}, dir_bins={dir_bins}, max_mag={max_magnitude}")
    
    # Build histograms for each dataset
    print("\n" + "="*60)
    print("BUILDING FLOW HISTOGRAMS")
    print("="*60)
    
    histograms: Dict[str, np.ndarray] = {}
    vector_counts: Dict[str, int] = {}
    
    for ds_config in datasets_config:
        dataset_name = ds_config['name']
        split = ds_config['split']
        num_batches = ds_config.get('num_batches', None)
        
        dataset_id = f"{dataset_name}_{split}"
        
        dataset = create_dataset_from_config(
            dataset_name, split, common_params, dataset_overrides
        )
        
        workers = 0 if dataset_name == 'synthetic' else num_workers
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            pin_memory=False
        )
        
        hist, vec_count = stream_flows_to_histogram(
            dataloader, num_batches, dataset_id,
            spatial_bins, mag_bins, dir_bins, max_magnitude
        )
        
        histograms[dataset_id] = hist
        vector_counts[dataset_id] = vec_count
        print(f"    {dataset_id}: {vec_count} vectors, "
              f"{(hist > 0).sum()} non-empty bins")
    
    # Compute pairwise coverage
    print("\n" + "="*60)
    print("COMPUTING PAIRWISE COVERAGE")
    print("="*60)
    
    train_splits = {"train", "training"}
    eval_splits = {"val", "test", "validation"}
    
    results = []
    
    for train_id, H_train in histograms.items():
        train_name, train_split = train_id.rsplit('_', 1)
        if train_split not in train_splits:
            continue
        
        for eval_id, H_val in histograms.items():
            eval_name, eval_split = eval_id.rsplit('_', 1)
            if eval_split not in eval_splits:
                continue
            
            coverage = compute_histogram_coverage(H_train, H_val)
            
            print(f"  {train_id} -> {eval_id}: coverage = {coverage:.4f}")
            
            results.append({
                'dataset1': train_name,
                'split1': train_split,
                'dataset2': eval_name,
                'split2': eval_split,
                'histogram_coverage': coverage,
                'num_vectors1': vector_counts[train_id],
                'num_vectors2': vector_counts[eval_id],
            })
    
    # Save results
    if output_config.get('save_results', True):
        results_file = output_config.get('results_file', 'histogram_coverage_results.csv')
        
        with open(results_file, 'w', newline='') as f:
            fieldnames = ['dataset1', 'split1', 'dataset2', 'split2', 
                         'histogram_coverage', 'num_vectors1', 'num_vectors2']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nResults saved to: {results_file}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
