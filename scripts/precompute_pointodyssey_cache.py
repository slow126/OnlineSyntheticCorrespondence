#!/usr/bin/env python3
"""
Precompute PointOdyssey cache to avoid conflicts during training.

This script uses PyTorch DataLoader for efficient parallel processing
with all its optimizations (prefetching, batching, etc.).

Usage:
    python scripts/precompute_pointodyssey_cache.py \
        --pointodyssey_root /path/to/PointOdyssey \
        --dset train \
        --S 8 \
        --N 32 \
        --strides 1 2 4 \
        --size 512 \
        --feature_size 32 \
        --num_workers 16
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, SequentialSampler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.synth.datasets.PointOdysseyCorrespondence import PointOdysseyFlowDataset


class PrecomputeDataset:
    """Wrapper dataset that calls __getitem_precompute__ for DataLoader."""
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        index = self.indices[idx]
        return self.base_dataset.__getitem_precompute__(index)


def precompute_collate_fn(batch):
    """
    Custom collate function for precomputation.
    Returns a dict mapping index -> gotit for all items in batch.
    """
    result = {}
    for item in batch:
        result[item['index']] = item['gotit']
    return result


def main():
    parser = argparse.ArgumentParser(description='Precompute PointOdyssey cache')
    parser.add_argument('--pointodyssey_root', type=str, required=True,
                        help='Root directory of PointOdyssey dataset')
    parser.add_argument('--dset', type=str, default='train', choices=['train', 'val'],
                        help='Dataset split to precompute')
    parser.add_argument('--S', type=int, default=8,
                        help='Sequence length')
    parser.add_argument('--N', type=int, default=32,
                        help='Number of points to track')
    parser.add_argument('--strides', type=int, nargs='+', default=[1, 2, 4],
                        help='Strides for dataset')
    parser.add_argument('--size', type=int, default=512,
                        help='Image size')
    parser.add_argument('--feature_size', type=int, default=32,
                        help='Feature size for CATs')
    parser.add_argument('--max_pts', type=int, default=200,
                        help='Maximum number of keypoints')
    parser.add_argument('--max_sequences', type=int, default=None,
                        help='Maximum number of sequences (None = all)')
    parser.add_argument('--all_points', action='store_true',
                        help='Use all points')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='Number of DataLoader workers (use lots for speed)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for DataLoader (1 is fine for precompute)')
    parser.add_argument('--prefetch_factor', type=int, default=4,
                        help='Prefetch factor for DataLoader')
    
    args = parser.parse_args()
    
    print("="*60)
    print("PointOdyssey Cache Precomputation")
    print("="*60)
    print(f"  Dataset: {args.dset}")
    print(f"  Root: {args.pointodyssey_root}")
    print(f"  S={args.S}, N={args.N}, strides={args.strides}")
    print(f"  size={args.size}, feature_size={args.feature_size}")
    print(f"  num_workers={args.num_workers}, batch_size={args.batch_size}")
    print("="*60)
    
    # Create dataset (same parameters as training)
    print("\nCreating dataset...")
    dataset = PointOdysseyFlowDataset(
        dataset_location=args.pointodyssey_root,
        dset=args.dset,
        use_augs=False,
        S=args.S,
        N=args.N,
        strides=args.strides,
        quick=False,
        verbose=True,  # Enable verbose to see progress
        resize_size=(args.size+64, args.size+64),
        crop_size=(args.size, args.size),
        filter_instances=True,
        downsample_for_cats=True,
        cats_feat_size=args.feature_size,
        all_points=args.all_points,
        max_sequences=args.max_sequences,
        max_pts=args.max_pts,
    )
    
    print(f"\nDataset length: {len(dataset)}")
    print(f"Cache file: {dataset.cache_file}")
    
    # Check if cache already exists
    if os.path.exists(dataset.cache_file):
        print(f"\n⚠️  Cache file already exists: {dataset.cache_file}")
        response = input("Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborting.")
            return
    
    # Get base dataset length (all indices to process)
    base_dataset_len = len(dataset.base_dataset)
    
    # Create wrapper dataset that uses __getitem_precompute__
    precompute_dataset = PrecomputeDataset(dataset, list(range(base_dataset_len)))
    
    # Create DataLoader with SequentialSampler
    dataloader = DataLoader(
        precompute_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(precompute_dataset),
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=False,  # Not needed for precompute (CPU only)
        collate_fn=precompute_collate_fn,
    )
    
    # Collect all results in a dictionary
    print(f"\nProcessing {base_dataset_len} indices with DataLoader ({args.num_workers} workers)...")
    print("(Results will be collected in memory and written once at the end)")
    
    all_results = {}  # index -> gotit
    
    try:
        with tqdm(total=base_dataset_len, desc="Processing indices") as pbar:
            for batch_results in dataloader:
                # batch_results is a dict: {index: gotit, ...}
                all_results.update(batch_results)
                pbar.update(len(batch_results))
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user!")
        print(f"Collected {len(all_results)} results so far...")
    
    # Write results to cache file
    print("\nWriting cache file...")
    valid_indices = sorted([idx for idx, gotit in all_results.items() if gotit])
    invalid_indices = sorted([idx for idx, gotit in all_results.items() if not gotit])
    
    cache_data = {
        'valid': valid_indices,
        'invalid': invalid_indices,
        'timestamp': time.time(),
        'total_samples': base_dataset_len
    }
    
    # Write to final cache file
    final_temp = dataset.cache_file + '.final_merge.tmp'
    with open(final_temp, 'w') as f:
        json.dump(cache_data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    
    # Atomic rename
    os.replace(final_temp, dataset.cache_file)
    
    # Print summary
    total_cached = len(all_results)
    coverage = (total_cached / base_dataset_len) * 100 if base_dataset_len > 0 else 0
    
    print("\n" + "="*60)
    print("Cache Precomputation Complete!")
    print("="*60)
    print(f"  Valid indices: {len(valid_indices):,}")
    print(f"  Invalid indices: {len(invalid_indices):,}")
    print(f"  Total cached: {total_cached:,} / {base_dataset_len:,}")
    print(f"  Coverage: {coverage:.1f}%")
    print(f"  Cache file: {dataset.cache_file}")
    print("="*60)
    print("\n✅ You can now run training jobs - they will use this cache in read-only mode.")


if __name__ == '__main__':
    main()

