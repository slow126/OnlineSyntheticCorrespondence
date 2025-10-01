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
from src.data.synth.online_synth_datamodule import create_datamodule_from_config
from src.data.real.real_datamodule import create_real_datamodule_from_config
from models.DinoV3.DinoV3 import DinoV3



def extract_features_from_batch(dino_model, batch, device='cuda'):
    """
    Extract spatial features from a batch of images using DinoV3.
    
    Args:
        dino_model: DinoV3 model instance
        batch: Batch dictionary with 'src' and 'trg' keys
        device: Device to run inference on
        
    Returns:
        Dictionary with extracted features for source and target images
    """
    features = {}
    
    for key in ['src', 'trg']:
        if key in batch:
            images = batch[key]  # Shape: (batch_size, 3, H, W)
            
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
    parser = argparse.ArgumentParser(description='Extract DinoV3 features from synthetic dataset')
    parser.add_argument('--dataset', type=str, default='synthetic',
                       help='Dataset to process')
    parser.add_argument('--config', type=str, 
                       default='src/configs/online_synth_configs/config.yaml',
                       help='Path to YAML config file')
    parser.add_argument('--output_dir', type=str, default='extracted_features',
                       help='Directory to save extracted features')
    parser.add_argument('--num_batches', type=int, default=10,
                       help='Number of batches to process')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size from config')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on')
    parser.add_argument('--model_name', type=str, 
                       default='facebook/dinov3-vit7b16-pretrain-lvd1689m',
                       help='DinoV3 model name')
    
    args = parser.parse_args()
    
    print("=== DinoV3 Feature Extraction ===")
    print(f"Config: {args.config}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of batches: {args.num_batches}")
    print(f"Device: {args.device}")
    print(f"Model: {args.model_name}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize DinoV3 model
    print("\nInitializing DinoV3 model...")
    dino_model = DinoV3(pretrained_model_name=args.model_name)
    print("DinoV3 model loaded successfully!")

    # Create datamodule from config
    print("\nCreating datamodule from config...")
    override_kwargs = {}
    
    if args.batch_size is not None:
        override_kwargs['batch_size'] = args.batch_size
    
    if args.dataset == 'synthetic':
        dm = create_datamodule_from_config(args.config, **override_kwargs)

        # Create dataloader
        train_loader = dm.train_dataloader()
        
        print(f"Number of batches in dataloader: {len(train_loader)}")
    
    elif args.dataset == 'spair':
        dm = create_real_datamodule_from_config(args.config, **override_kwargs)
        train_loader = dm.train_dataloader()
        print(f"Number of batches in dataloader: {len(train_loader)}")
    
    # Process batches
    print(f"\nProcessing {args.num_batches} batches...")
    
    all_features = []
    batch_metadata = []
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Extracting features")):
        batch = [{k: v.to(args.device) for k, v in x.items()} for x in batch]
        processed_batch = dm.on_after_batch_transfer(batch, batch_idx)
        if batch_idx >= args.num_batches:
            break
        
        try:
            # Extract features from the batch
            features = extract_features_from_batch(dino_model, processed_batch, args.device)
            
            # Create metadata for this batch
            metadata = {
                'batch_idx': batch_idx,
                'batch_size': batch['src'].shape[0] if 'src' in batch else 0,
                'image_shape': batch['src'].shape[1:] if 'src' in batch else None,
                'feature_shapes': {k: v.shape for k, v in features.items()},
                'device': args.device,
                'model_name': args.model_name
            }
            
            # Save features for this batch
            save_features(features, output_dir, batch_idx, metadata)
            if batch_idx == 0:
                dino_model.visualize_features_grid(features)
            
            # Store for summary
            all_features.append(features)
            batch_metadata.append(metadata)
            
            # Print progress info
            if 'src' in features:
                print(f"Batch {batch_idx}: src features shape {features['src'].shape}")
            if 'trg' in features:
                print(f"Batch {batch_idx}: trg features shape {features['trg'].shape}")
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            continue
    
    # Save summary
    summary = {
        'total_batches_processed': len(all_features),
        'model_name': args.model_name,
        'config_path': args.config,
        'batch_metadata': batch_metadata,
        'feature_statistics': {}
    }
    
    # Calculate feature statistics
    if all_features:
        for key in ['src', 'trg']:
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
    summary_file = output_dir / 'extraction_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Extraction Complete ===")
    print(f"Processed {len(all_features)} batches")
    print(f"Features saved to: {output_dir}")
    print(f"Summary saved to: {summary_file}")
    
    # Print feature statistics
    if summary['feature_statistics']:
        print("\nFeature Statistics:")
        for key, stats in summary['feature_statistics'].items():
            print(f"  {key}: {stats['total_samples']} samples, shape {stats['feature_shape']}")
            print(f"    mean: {stats['mean']:.4f}, std: {stats['std']:.4f}")


if __name__ == '__main__':
    main()
