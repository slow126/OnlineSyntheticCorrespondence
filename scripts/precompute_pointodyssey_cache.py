#!/usr/bin/env python3
"""
Precompute PointOdyssey cache to avoid conflicts during training.

This script uses a threaded DataLoader to efficiently precompute the cache
without needing GPU. Can use lots of system RAM for fast parallel processing.

Usage:
    python scripts/precompute_pointodyssey_cache.py \
        --pointodyssey_root /path/to/PointOdyssey \
        --dset train \
        --S 8 \
        --N 32 \
        --strides 1 2 4 \
        --size 512 \
        --feature_size 32 \
        --num_workers 16 \
        --batch_size 32
"""

import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.synth.datasets.PointOdysseyCorrespondence import PointOdysseyFlowDataset


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
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of worker threads for DataLoader (use lots for speed)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for DataLoader (does not affect cache, just speed)')
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
    
    # Create DataLoader with many workers (no GPU needed, uses system RAM)
    print(f"\nCreating DataLoader with {args.num_workers} workers...")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,  # Sequential is fine for precomputation
        prefetch_factor=args.prefetch_factor,
        pin_memory=False,  # No GPU, so no pin_memory needed
        persistent_workers=True if args.num_workers > 0 else False,
    )
    
    # Iterate through all batches to build cache
    print(f"\nBuilding cache by processing {len(dataloader)} batches...")
    print("(This may take a while - dataset will discover valid/invalid indices)")
    
    valid_count = 0
    invalid_count = 0
    error_count = 0
    
    try:
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            # Just accessing the batch builds the cache
            # The dataset's __getitem__ method handles cache updates
            try:
                if batch is not None:
                    valid_count += args.batch_size
                else:
                    invalid_count += args.batch_size
            except Exception as e:
                error_count += 1
                if batch_idx % 100 == 0:  # Print errors occasionally
                    print(f"\nError in batch {batch_idx}: {e}")
            
            # Periodic cache save (every 100 batches)
            if (batch_idx + 1) % 100 == 0:
                # Force a save by triggering the save mechanism
                # The dataset will save automatically based on _cache_save_interval
                pass
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user!")
        print("Saving partial cache...")
    
    # Force final save
    print("\nSaving final cache...")
    dataset.save_cache_final()
    
    # Print summary
    with dataset._cache_lock:
        final_valid = len(dataset._valid_indices)
        final_invalid = len(dataset._invalid_indices)
        total_cached = final_valid + final_invalid
        coverage = (total_cached / len(dataset)) * 100 if len(dataset) > 0 else 0
    
    print("\n" + "="*60)
    print("Cache Precomputation Complete!")
    print("="*60)
    print(f"  Valid indices: {final_valid:,}")
    print(f"  Invalid indices: {final_invalid:,}")
    print(f"  Total cached: {total_cached:,} / {len(dataset):,}")
    print(f"  Coverage: {coverage:.1f}%")
    print(f"  Cache file: {dataset.cache_file}")
    print("="*60)
    print("\n✅ You can now run training jobs - they will use this cache in read-only mode.")


if __name__ == '__main__':
    main()

