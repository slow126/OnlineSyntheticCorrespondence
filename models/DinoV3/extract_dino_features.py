#!/usr/bin/env python3
"""
Extract DinoV3 spatial features from synthetic dataset batches.
Uses the new YAML-based datamodule wrapper and DinoV3 model.
"""

import os
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import pickle
import json

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(project_root))

# Import our modules
from models.DinoV3.DinoV3 import DinoV3
from src.data.synth.datasets.OnlineCorrespondenceDataset import OnlineCorrespondenceDataset
from torch.utils.data import DataLoader
import yaml

import models.CATs_PlusPlus.data.download as download



def extract_features_from_batch(dino_model, batch, device='cuda'):
    """
    Extract spatial features from a batch of images using DinoV3.
    
    Args:
        dino_model: DinoV3 model instance
        batch: Batch dictionary with 'src_img' and 'trg_img' keys (matching train_cats.py format)
        device: Device to run inference on
        
    Returns:
        Dictionary with extracted features for source and target images
    """
    features = {}
    
    # Handle the correct batch format from train_cats.py
    for key in ['src_img', 'trg_img']:
        if key in batch:
            images = batch[key]  # Shape: (batch_size, 3, H, W)
            
            # Move images to device if not already there
            if images.device != device:
                images = images.to(device)
            
            # Extract spatial features for entire batch at once - much more efficient!
            spatial_features = dino_model.get_spatial_features(images)
            features[key] = spatial_features.cpu()  # (batch_size, num_patches, dim)
    
    return features


def save_features(features_dict, save_path, batch_idx, metadata=None):
    """
    Save extracted features to disk.
    
    Args:
        features_dict: Dictionary containing extracted features
        save_path: Path to save the features
        batch_idx: Batch index for naming
        metadata: Additional metadata to save
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save features as pickle
    features_file = save_path / f"features_batch_{batch_idx:04d}.pkl"
    with open(features_file, 'wb') as f:
        pickle.dump(features_dict, f)
    
    # Save metadata as JSON
    if metadata is not None:
        metadata_file = save_path / f"metadata_batch_{batch_idx:04d}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"Saved features for batch {batch_idx} to {features_file}")


def main():
    parser = argparse.ArgumentParser(description='Extract DinoV3 features from a complete dataset pass')
    
    # Dataset selection - single parameter for clarity
    parser.add_argument('--dataset', type=str, default='synthetic', 
                       choices=['synthetic', 'spair', 'pfpascal', 'pfwillow', 'caltech'],
                       help='Dataset to process (one complete pass)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                       help='Dataset split to process')
    
    # Synthetic dataset config
    parser.add_argument('--geometry_config', type=str, 
                       default='src/configs/online_synth_configs/OnlineGeometryConfig.yaml',
                       help='Path to geometry config file for synthetic dataset')
    parser.add_argument('--processor_config', type=str, 
                       default='src/configs/online_synth_configs/OnlineProcessorConfig.yaml',
                       help='Path to processor config file for synthetic dataset')
    
    # Real dataset config
    parser.add_argument('--datapath', type=str, default='./models/Datasets_CATs',
                       help='Path to real datasets')
    parser.add_argument('--thres', type=str, default='img', choices=['auto', 'img', 'bbox', 'bbox-kp'])
    parser.add_argument('--feature_size', type=int, default=32,
                       help='Feature size for the model')
    
    # Processing parameters
    parser.add_argument('--output_dir', type=str, default='extracted_features',
                       help='Directory to save extracted features')
    parser.add_argument('--num_batches', type=int, default=None,
                       help='Maximum number of batches to process (default: process entire dataset)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for processing')
    parser.add_argument('--n_threads', type=int, default=0,
                       help='Number of parallel threads for dataloaders (0 recommended for OpenGL compatibility)')
    
    # Model parameters
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on')
    parser.add_argument('--model_name', type=str, 
                       default='facebook/dinov3-vit7b16-pretrain-lvd1689m',
                       help='DinoV3 model name')
    
    args = parser.parse_args()
    
    print("=== DinoV3 Feature Extraction ===")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Model: {args.model_name}")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Warning about OpenGL multiprocessing
    if args.n_threads > 0:
        print("⚠️  WARNING: Using multiple workers with OpenGL rendering may cause segmentation faults.")
        print("   Consider using --n_threads 0 for stable processing.")
    
    # Create output directory with dataset-specific subdirectory
    base_output_dir = Path(args.output_dir)
    dataset_output_dir = base_output_dir / args.dataset
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Base output directory: {base_output_dir}")
    print(f"Dataset output directory: {dataset_output_dir}")
    
    # Initialize DinoV3 model
    print("\nInitializing DinoV3 model...")
    dino_model = DinoV3(pretrained_model_name=args.model_name)
    print("DinoV3 model loaded successfully!")

    # Create dataset and dataloader (matching train_cats.py implementation)
    if args.dataset == 'synthetic':
        print("Creating synthetic dataset...")
        dataset = OnlineCorrespondenceDataset(
            geometry_config_path=args.geometry_config,
            processor_config_path=args.processor_config,
            split=args.split
        )
        dataset.cuda()
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            num_workers=args.n_threads, 
            shuffle=(args.split == 'train'), 
            collate_fn=dataset.collate_fn
        )
        print(f"Dataset size: {len(dataloader)} batches")
    else:
        # For real datasets, download if needed
        download.download_dataset(args.datapath, args.dataset)
        
        # Map split names to match download.py expectations
        split_map = {'train': 'trn', 'val': 'val', 'test': 'test'}
        split_name = split_map.get(args.split, 'trn')
 
        dataset = download.load_dataset(
            args.dataset, args.datapath, args.thres, device, 
            split_name, False, args.feature_size
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.n_threads,
            persistent_workers=True if args.n_threads > 0 else False,
            prefetch_factor=8 if args.n_threads > 0 else None,
            shuffle=(args.split == 'train')
        )
        print(f"Dataset size: {len(dataloader)} batches")
    
    # Process batches (either limited or complete dataset)
    max_batches = args.num_batches if args.num_batches is not None else len(dataloader)
    if args.num_batches is not None:
        print(f"\nProcessing {max_batches} batches (debugging mode)...")
    else:
        print(f"\nProcessing complete dataset: {len(dataloader)} batches...")
    
    all_features = []
    batch_metadata = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
        # Stop if we've reached the batch limit
        if batch_idx >= max_batches:
            break
            
        try:
            # Debug: Print batch keys for first batch
            if batch_idx == 0:
                print(f"Batch keys: {list(batch.keys())}")
                if 'src_img' in batch:
                    print(f"Source image shape: {batch['src_img'].shape}")
                if 'trg_img' in batch:
                    print(f"Target image shape: {batch['trg_img'].shape}")
            
            # Extract features from the batch
            features = extract_features_from_batch(dino_model, batch, device)
            
            # Create metadata for this batch
            metadata = {
                'batch_idx': batch_idx,
                'batch_size': batch['src_img'].shape[0] if 'src_img' in batch else 0,
                'image_shape': batch['src_img'].shape[1:] if 'src_img' in batch else None,
                'feature_shapes': {k: v.shape for k, v in features.items()},
                'device': str(device),
                'model_name': args.model_name,
                'dataset': args.dataset,
                'split': args.split
            }
            
            # Add category information if it exists (for real datasets)
            if 'category' in batch:
                metadata['category'] = batch['category'].tolist() if hasattr(batch['category'], 'tolist') else batch['category']
                print(f"Batch {batch_idx}: Found categories: {metadata['category']}")
            else:
                if args.dataset != 'synthetic':
                    metadata['category'] = None
                    if batch_idx == 0:  # Only print this once
                        print(f"Batch {batch_idx}: No category information found")
                else:
                    # Set category as a list with 'synthetic' for each sample in the batch
                    batch_size = batch['src_img'].shape[0]
                    metadata['category'] = ['synthetic'] * batch_size
                    if batch_idx == 0:  # Only print this once
                        print(f"Batch {batch_idx}: Setting category to synthetic for {batch_size} samples")
            
            # Save features for this batch
            save_features(features, dataset_output_dir, batch_idx, metadata)
            
            # Visualize features for first batch
            if batch_idx == 0:
                try:
                    dino_model.visualize_features_grid(features)
                except Exception as viz_error:
                    print(f"Could not visualize features: {viz_error}")
            
            # Store for summary
            all_features.append(features)
            batch_metadata.append(metadata)
            
            # Print progress info
            if 'src_img' in features:
                print(f"Batch {batch_idx}: src features shape {features['src_img'].shape}")
            if 'trg_img' in features:
                print(f"Batch {batch_idx}: trg features shape {features['trg_img'].shape}")
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary
    summary = {
        'total_batches_processed': len(all_features),
        'model_name': args.model_name,
        'dataset': args.dataset,
        'split': args.split,
        'geometry_config': args.geometry_config if args.dataset == 'synthetic' else None,
        'processor_config': args.processor_config if args.dataset == 'synthetic' else None,
        'datapath': args.datapath if args.dataset != 'synthetic' else None,
        'batch_metadata': batch_metadata,
        'feature_statistics': {},
        'category_statistics': {}
    }
    
    # Calculate category statistics if categories exist
    all_categories = []
    for metadata in batch_metadata:
        if metadata.get('category') is not None:
            if isinstance(metadata['category'], list):
                all_categories.extend(metadata['category'])
            else:
                all_categories.append(metadata['category'])
    
    if all_categories:
        from collections import Counter
        category_counts = Counter(all_categories)
        summary['category_statistics'] = {
            'total_samples_with_categories': len(all_categories),
            'unique_categories': len(category_counts),
            'category_distribution': dict(category_counts),
            'most_common_categories': category_counts.most_common(10)
        }
        print(f"\nCategory Statistics:")
        print(f"  Total samples with categories: {len(all_categories)}")
        print(f"  Unique categories: {len(category_counts)}")
        print(f"  Most common categories: {category_counts.most_common(5)}")
    else:
        summary['category_statistics'] = {
            'total_samples_with_categories': 0,
            'unique_categories': 0,
            'category_distribution': {},
            'most_common_categories': []
        }
        print(f"\nNo category information found in any batches (synthetic dataset)")
    
    # Calculate feature statistics
    if all_features:
        for key in ['src_img', 'trg_img']:
            if key in all_features[0]:
                all_key_features = [f[key] for f in all_features]
                stacked_features = torch.cat(all_key_features, dim=0)
                
                summary['feature_statistics'][key] = {
                    'total_samples': stacked_features.shape[0],
                    'feature_shape': stacked_features.shape[1:],
                    'mean': stacked_features.mean().item(),
                    'std': stacked_features.std().item(),
                    'min': stacked_features.min().item(),
                    'max': stacked_features.max().item()
                }
    
    # Save summary
    summary_file = dataset_output_dir / 'extraction_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Extraction Complete ===")
    print(f"Processed {len(all_features)} batches")
    print(f"Features saved to: {dataset_output_dir}")
    print(f"Summary saved to: {summary_file}")
    
    # Print feature statistics
    if summary['feature_statistics']:
        print("\nFeature Statistics:")
        for key, stats in summary['feature_statistics'].items():
            print(f"  {key}: {stats['total_samples']} samples, shape {stats['feature_shape']}")
            print(f"    mean: {stats['mean']:.4f}, std: {stats['std']:.4f}")
    else:
        print("No features were successfully extracted.")
    
    # Print final summary
    print(f"\nFinal Summary:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Split: {args.split}")
    print(f"  Model: {args.model_name}")
    print(f"  Batches processed: {len(all_features)}/{max_batches}")
    if args.num_batches is not None:
        print(f"  Mode: Debugging (limited to {args.num_batches} batches)")
    else:
        print(f"  Mode: Complete dataset processing")
    print(f"  Base output directory: {base_output_dir}")
    print(f"  Dataset output directory: {dataset_output_dir}")


if __name__ == '__main__':
    main()
