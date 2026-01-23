#!/usr/bin/env python3
"""
Create a subsampled version of FlyingThings dataset.

This script:
1. Enumerates all pairs in both TRAIN and TEST splits
2. Randomly samples up to N pairs from each split (default: 10,000 per split)
3. Copies the corresponding image and flow files to a new directory
4. Maintains the same directory structure so it works with existing loaders

Usage:
    python subsample_flyingthings.py \
        --source /path/to/FlyingThings3D \
        --output /path/to/FlyingThings3D_subsampled \
        --num_pairs 10000

Default paths (from remote.yaml):
    source: /home/slow1/Data/FlyingThings3D_Pytorch/FlyingThings3D
    output: /home/slow1/Data/FlyingThings3D_subsampled_10k
"""

import argparse
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import glob


def enumerate_pairs(source_dir, split='train'):
    """
    Enumerate all image pairs and their corresponding flow files.
    
    Returns:
        List of tuples: (img1_path, img2_path, flow_path, relative_path_structure)
    """
    pairs = []
    
    # Map to uppercase for source (FlyingThings3D uses TRAIN/TEST)
    # but we'll output lowercase for torchvision compatibility
    split_dir = 'TRAIN' if split == 'train' or split == 'training' else 'TEST'
    
    # Find all image directories
    frames_dir = os.path.join(source_dir, 'frames_cleanpass', split_dir)
    flow_dir = os.path.join(source_dir, 'optical_flow', split_dir)
    
    if not os.path.exists(frames_dir):
        raise ValueError(f"Frames directory not found: {frames_dir}")
    if not os.path.exists(flow_dir):
        raise ValueError(f"Flow directory not found: {flow_dir}")
    
    # Get all scene directories
    scene_dirs = sorted(glob.glob(os.path.join(frames_dir, '*')))
    
    print(f"Found {len(scene_dirs)} scenes")
    
    for scene_dir in tqdm(scene_dirs, desc="Scanning scenes"):
        scene_name = os.path.basename(scene_dir)
        
        # Get all sequence directories in this scene
        seq_dirs = sorted(glob.glob(os.path.join(scene_dir, '*')))
        
        for seq_dir in seq_dirs:
            seq_name = os.path.basename(seq_dir)
            
            # Get images from left camera
            img_dir = os.path.join(seq_dir, 'left')
            if not os.path.exists(img_dir):
                continue
                
            images = sorted(glob.glob(os.path.join(img_dir, '*.png')))
            if len(images) < 2:
                continue
            
            # Get corresponding flow directory
            flow_seq_dir = os.path.join(flow_dir, scene_name, seq_name)
            
            # Check both into_future and into_past directions
            for direction in ['into_future', 'into_past']:
                flow_cam_dir = os.path.join(flow_seq_dir, direction, 'left')
                if not os.path.exists(flow_cam_dir):
                    continue
                
                flows = sorted(glob.glob(os.path.join(flow_cam_dir, '*.pfm')))
                
                # Create pairs: for into_future, flow[i] goes from image[i] to image[i+1]
                # for into_past, flow[i+1] goes from image[i+1] to image[i]
                if direction == 'into_future':
                    for i in range(len(flows)):
                        if i + 1 < len(images):
                            img1 = images[i]
                            img2 = images[i + 1]
                            flow = flows[i]
                            
                            # Store relative paths for reconstruction
                            rel_img1 = os.path.relpath(img1, source_dir)
                            rel_img2 = os.path.relpath(img2, source_dir)
                            rel_flow = os.path.relpath(flow, source_dir)
                            
                            pairs.append((img1, img2, flow, rel_img1, rel_img2, rel_flow))
                else:  # into_past
                    for i in range(len(flows) - 1):
                        if i + 1 < len(images):
                            img1 = images[i + 1]
                            img2 = images[i]
                            flow = flows[i + 1]
                            
                            rel_img1 = os.path.relpath(img1, source_dir)
                            rel_img2 = os.path.relpath(img2, source_dir)
                            rel_flow = os.path.relpath(flow, source_dir)
                            
                            pairs.append((img1, img2, flow, rel_img1, rel_img2, rel_flow))
    
    return pairs


def copy_pair(src_root, dst_root, img1_path, img2_path, flow_path, rel_img1, rel_img2, rel_flow):
    """Copy a pair of images and flow file, maintaining directory structure."""
    # Keep TRAIN/TEST uppercase as expected by torchvision FlyingThings3D loader
    # (torchvision automatically appends /FlyingThings3D to root and expects uppercase splits)
    
    # Create destination paths
    dst_img1 = os.path.join(dst_root, rel_img1)
    dst_img2 = os.path.join(dst_root, rel_img2)
    dst_flow = os.path.join(dst_root, rel_flow)
    
    # Create directories
    os.makedirs(os.path.dirname(dst_img1), exist_ok=True)
    os.makedirs(os.path.dirname(dst_img2), exist_ok=True)
    os.makedirs(os.path.dirname(dst_flow), exist_ok=True)
    
    # Copy files
    shutil.copy2(img1_path, dst_img1)
    shutil.copy2(img2_path, dst_img2)
    shutil.copy2(flow_path, dst_flow)


def main():
    # Default paths
    # Note: torchvision's FlyingThings3D loader automatically appends /FlyingThings3D to the root
    # So the source should point to the directory containing FlyingThings3D/
    DEFAULT_SOURCE = "/home/slow1/Data/FlyingThings3D_Pytorch/FlyingThings3D"
    DEFAULT_OUTPUT = "/home/slow1/Data/FlyingThings3D_subsampled_10k/FlyingThings3D"
    
    parser = argparse.ArgumentParser(description='Subsample FlyingThings dataset')
    parser.add_argument('--source', type=str, default=DEFAULT_SOURCE,
                       help=f'Path to full FlyingThings3D dataset (default: {DEFAULT_SOURCE})')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                       help=f'Path to output subsampled dataset (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--num_pairs', type=int, default=10000,
                       help='Number of pairs to sample per split (default: 10000)')
    parser.add_argument('--splits', type=str, nargs='+', default=['test'],
                       choices=['train', 'test'],
                       help='Which splits to process (default: test only). Use --splits train test for both.')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print("="*60)
    print("FlyingThings Dataset Subsampling")
    print("="*60)
    print(f"Source: {args.source}")
    print(f"Output: {args.output}")
    print(f"Number of pairs per split: {args.num_pairs}")
    print(f"Processing splits: {', '.join([s.upper() for s in args.splits])}")
    print(f"Seed: {args.seed}")
    print("="*60)
    
    # Check source exists
    if not os.path.exists(args.source):
        raise ValueError(f"Source directory does not exist: {args.source}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    total_copied = 0
    split_counts = {}
    
    # Process specified splits
    for split_name in args.splits:
        print(f"\n{'='*60}")
        print(f"Processing {split_name.upper()} split")
        print(f"{'='*60}")
        
        # Enumerate all pairs for this split
        print(f"\nEnumerating all pairs in {split_name} split...")
        all_pairs = enumerate_pairs(args.source, split_name)
        
        print(f"\nFound {len(all_pairs):,} total pairs in {split_name} split")
        
        if len(all_pairs) == 0:
            print(f"Warning: No pairs found in {split_name} split, skipping...")
            continue
        
        if len(all_pairs) < args.num_pairs:
            print(f"Warning: Only {len(all_pairs)} pairs available in {split_name}, but {args.num_pairs} requested.")
            print(f"Using all {len(all_pairs)} pairs.")
            selected_pairs = all_pairs
        else:
            # Randomly sample pairs
            print(f"\nRandomly sampling {args.num_pairs:,} pairs from {split_name} split...")
            selected_pairs = random.sample(all_pairs, args.num_pairs)
        
        # Copy selected pairs
        print(f"\nCopying {len(selected_pairs):,} pairs from {split_name} split to output directory...")
        for img1, img2, flow, rel_img1, rel_img2, rel_flow in tqdm(selected_pairs, desc=f"Copying {split_name}"):
            copy_pair(args.source, args.output, img1, img2, flow, rel_img1, rel_img2, rel_flow)
        
        split_counts[split_name] = len(selected_pairs)
        total_copied += len(selected_pairs)
        print(f"✓ Copied {len(selected_pairs):,} pairs from {split_name} split")
    
    print("\n" + "="*60)
    print("Done!")
    print(f"Subsampled dataset created at: {args.output}")
    print(f"Total pairs copied: {total_copied:,}")
    print(f"  - TRAIN: {split_counts.get('train', 0):,} pairs")
    print(f"  - TEST: {split_counts.get('test', 0):,} pairs")
    print("="*60)
    
    # Print usage instructions
    print("\nTo use this dataset, point your datapath to the parent directory:")
    print(f"  {os.path.dirname(args.output)}")
    print("\n(torchvision's FlyingThings3D automatically appends /FlyingThings3D to the root)")
    print("The directory structure is preserved with uppercase TRAIN/TEST splits.")


if __name__ == '__main__':
    main()
