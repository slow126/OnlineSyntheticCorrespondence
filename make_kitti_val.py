#!/usr/bin/env python3
"""
Create a fixed KITTI validation split file for consistent validation across experiments.

This script generates a split file that can be used to ensure the same train/val split
is used across all KITTI experiments, following standard practice.
"""

import os
import sys
import argparse
import glob
import numpy as np
from pathlib import Path

# Add the project root to path to import kitti utilities
sys.path.insert(0, str(Path(__file__).parent))


def create_fixed_split(
    kitti_2012_root: str,
    kitti_2015_root: str,
    output_root: str,
    split_ratio: float = 0.8,
    seed: int = 42,
    occ_type: str = 'occ',
):
    """
    Create a new KITTI root directory with fixed train/val splits.
    
    Creates kitti-split directory structure:
    kitti-split/
      kitti-2012/
        train/  (copied training data files)
        val/    (copied validation data files)
      kitti-2015/
        train/  (copied training data files)
        val/    (copied validation data files)
    
    Files are split based on a fixed random seed to ensure consistency.
    
    Args:
        kitti_2012_root: Path to original kitti-2012 directory
        kitti_2015_root: Path to original kitti-2015 directory
        output_root: Path to new kitti-split root directory
        split_ratio: Ratio of training samples (default 0.8 = 80% train, 20% val)
        seed: Random seed for reproducibility (default 42)
        occ_type: 'occ', 'noc', or 'only_occ' (default 'occ')
    """
    kitti_2012_root = Path(kitti_2012_root)
    kitti_2015_root = Path(kitti_2015_root)
    output_root = Path(output_root)
    
    # Create kitti-split subdirectory in output_root
    kitti_split_dir = output_root / 'kitti-split'
    
    # Check that original directories exist
    if not (kitti_2012_root / 'training').exists():
        raise ValueError(f"KITTI-2012 training directory not found: {kitti_2012_root / 'training'}")
    if not (kitti_2015_root / 'training').exists():
        raise ValueError(f"KITTI-2015 training directory not found: {kitti_2015_root / 'training'}")
    
    print(f"Creating new KITTI split structure at: {kitti_split_dir}")
    print(f"KITTI-2012 source: {kitti_2012_root}")
    print(f"KITTI-2015 source: {kitti_2015_root}")
    print(f"Files will be copied (not symlinked)\n")
    
    # Process both versions
    for version, source_root in [('2012', kitti_2012_root), ('2015', kitti_2015_root)]:
        print(f"\n{'='*60}")
        print(f"Processing KITTI-{version}")
        print(f"{'='*60}")
        
        training_dir = source_root / 'training'
        output_version_dir = kitti_split_dir / f'kitti-{version}'
        output_train_dir = output_version_dir / 'train'
        output_val_dir = output_version_dir / 'val'
        
        # Create output directories
        output_train_dir.mkdir(parents=True, exist_ok=True)
        output_val_dir.mkdir(parents=True, exist_ok=True)
    
        # Get all samples from training directory
        # Replicate make_dataset logic to get samples in the exact same order
        occ = (occ_type == 'occ')
        only_occ = (occ_type == 'only_occ')
        
        if only_occ:
            flow_dir = 'flow_occ'
            flow_dir_noc = 'flow_noc'
        else:
            flow_dir = 'flow_occ' if occ else 'flow_noc'
        
        flow_dir_path = training_dir / flow_dir
        if not flow_dir_path.exists():
            print(f"Warning: Flow directory not found: {flow_dir_path}, skipping...")
            continue
        
        img_dir = 'colored_0'
        if not (training_dir / img_dir).exists():
            img_dir = 'image_2'
        if not (training_dir / img_dir).exists():
            raise ValueError(f"Image directory not found: {training_dir / img_dir}")
        
        # Get all samples in the same order as make_dataset
        flow_map_paths = list(glob.iglob(str(flow_dir_path / '*.png')))
        flow_map_paths.sort()  # Sort to ensure consistent ordering
        
        all_samples = []
        for flow_map_path in flow_map_paths:
            flow_map = os.path.basename(flow_map_path)
            root_filename = flow_map[:-7]  # name of image
            img1 = os.path.join(img_dir, root_filename + '_11.png')
            img2 = os.path.join(img_dir, root_filename + '_10.png')
            
            # Check if images exist
            if not ((training_dir / img1).exists() or (training_dir / img2).exists()):
                continue
            
            all_samples.append({
                'flow_map': flow_map,
                'flow_path': os.path.join(flow_dir, flow_map),
                'img1': img1,
                'img2': img2,
                'root_filename': root_filename
            })
        
        print(f"Found {len(all_samples)} total samples")
        
        # Create fixed split using seed
        np.random.seed(seed)
        split_values = np.random.uniform(0, 1, len(all_samples)) < split_ratio
        
        train_count = split_values.sum()
        val_count = len(all_samples) - train_count
        
        print(f"Split: {train_count} train ({train_count/len(all_samples)*100:.1f}%), {val_count} val ({val_count/len(all_samples)*100:.1f}%)")
        
        # Create directory structure and copy files into train/val
        import shutil
        
        # Create subdirectories in train/val
        for subdir in [img_dir, flow_dir]:
            if (training_dir / subdir).exists():
                (output_train_dir / subdir).mkdir(parents=True, exist_ok=True)
                (output_val_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        if only_occ and (training_dir / flow_dir_noc).exists():
            (output_train_dir / flow_dir_noc).mkdir(parents=True, exist_ok=True)
            (output_val_dir / flow_dir_noc).mkdir(parents=True, exist_ok=True)
        
        # Copy files based on split
        train_files_copied = 0
        val_files_copied = 0
        
        for sample, is_train in zip(all_samples, split_values):
            target_dir = output_train_dir if is_train else output_val_dir
            
            # Copy flow file
            source_flow = training_dir / sample['flow_path']
            target_flow = target_dir / sample['flow_path']
            if source_flow.exists():
                shutil.copy2(source_flow, target_flow)
                if is_train:
                    train_files_copied += 1
                else:
                    val_files_copied += 1
            
            # Copy image files
            for img_path in [sample['img1'], sample['img2']]:
                source_img = training_dir / img_path
                target_img = target_dir / img_path
                if source_img.exists() and not target_img.exists():
                    shutil.copy2(source_img, target_img)
            
            # Copy flow_noc if needed
            if only_occ:
                flow_noc_path = flow_dir_noc / sample['flow_map']
                source_flow_noc = training_dir / flow_noc_path
                target_flow_noc = target_dir / flow_noc_path
                if source_flow_noc.exists():
                    shutil.copy2(source_flow_noc, target_flow_noc)
        
        print(f"Created train/val directories at:")
        print(f"  Train: {output_train_dir} ({train_files_copied} flow files)")
        print(f"  Val: {output_val_dir} ({val_files_copied} flow files)")
        print(f"  (Files copied, not symlinked)")
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully created KITTI split structure at: {kitti_split_dir}")
    print(f"{'='*60}")
    print(f"\nDirectory structure:")
    print(f"  {kitti_split_dir}/")
    print(f"    kitti-2012/")
    print(f"      train/  (copied training data files)")
    print(f"      val/    (copied validation data files)")
    print(f"    kitti-2015/")
    print(f"      train/  (copied training data files)")
    print(f"      val/    (copied validation data files)")
    print(f"\nTo use this split:")
    print(f"  Set kitti_root to: {kitti_split_dir}/kitti-2012 or {kitti_split_dir}/kitti-2015")
    print(f"  Use split='train' or split='val' in KittiDataset")
    print(f"  The dataset will automatically use the train/val directories")
    
    return kitti_split_dir


def main():
    parser = argparse.ArgumentParser(
        description='Create a new KITTI root directory with fixed train/val splits'
    )
    parser.add_argument(
        '--kitti_2012_root',
        type=str,
        required=True,
        help='Path to original kitti-2012 directory (should contain training subdirectory)'
    )
    parser.add_argument(
        '--kitti_2015_root',
        type=str,
        required=True,
        help='Path to original kitti-2015 directory (should contain training subdirectory)'
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default='kitti-split',
        help='Path to new kitti-split root directory (default: kitti-split)'
    )
    parser.add_argument(
        '--split_ratio',
        type=float,
        default=0.8,
        help='Ratio of training samples (default: 0.8 = 80%% train, 20%% val)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--occ_type',
        type=str,
        default='occ',
        choices=['occ', 'noc', 'only_occ'],
        help='Type of flow to use (default: occ)'
    )
    
    args = parser.parse_args()
    
    try:
        create_fixed_split(
            kitti_2012_root=args.kitti_2012_root,
            kitti_2015_root=args.kitti_2015_root,
            output_root=args.output_root,
            split_ratio=args.split_ratio,
            seed=args.seed,
            occ_type=args.occ_type,
        )
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

