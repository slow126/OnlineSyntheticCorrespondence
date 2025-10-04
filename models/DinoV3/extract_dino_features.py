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
from src.data.synth.datasets.OnlineCorrespondenceDataset import OnlineCorrespondenceDataset
from torch.utils.data import DataLoader
import yaml

sys.path.append('models/CATs-PlusPlus/data')
import download



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
    parser.add_argument('--correspondence_config', type=str, 
                       default='src/configs/online_synth_configs/OnlineDatasetConfig_UMAP.yaml',
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
    parser.add_argument('--benchmark', type=str, default='spair', choices=['pfpascal', 'spair'])
    parser.add_argument('--datapath', type=str, default='../Datasets_CATs',
                       help='Path to dataset')
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox', 'bbox-kp'])
    parser.add_argument('--feature_size', type=int, default=32,
                       help='Feature size for the model')
    parser.add_argument('--n_threads', type=int, default=4,
                       help='Number of threads to use')
    
    args = parser.parse_args()
    
    print("=== DinoV3 Feature Extraction ===")
    print(f"Config: {args.correspondence_config}")
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

    with open(args.correspondence_config, 'r') as f:
        correspondence_config = yaml.load(f, Loader=yaml.FullLoader)

    
    if args.dataset == 'synthetic':
        # Create datamodule from config
        print(f"Creating Synthetic Correspondence Dataset")
        if args.batch_size is None:
            args.batch_size = correspondence_config['dataset_configs']['train_dataset_config']['init_args']['batch_size']


        dataset = OnlineCorrespondenceDataset(
            geometry_config_path=correspondence_config['dataset_configs']['train_dataset_config']['init_args']['geometry_config_path'],
            processor_config_path=correspondence_config['dataset_configs']['train_dataset_config']['init_args']['processor_config_path']
        )
        dataset.cuda()
        train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.n_threads, shuffle=True, collate_fn=dataset.collate_fn)
        print(f"Number of batches in dataloader: {len(train_loader)}")
    
    elif args.dataset == 'spair':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = download.load_dataset(args.benchmark, args.datapath, args.thres, device, 'train', False, args.feature_size)
        train_loader = DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      num_workers=args.n_threads,
                                      persistent_workers=True,
                                      prefetch_factor=8,
                                      shuffle=False)
        print(f"Number of batches in dataloader: {len(train_loader)}")
    
    # Process batches
    print(f"\nProcessing {args.num_batches} batches...")
    
    all_features = []
    batch_metadata = []
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Extracting features")):
        
        
        try:
            # Extract features from the batch
            features = extract_features_from_batch(dino_model, batch, args.device)
            
            # Create metadata for this batch
            metadata = {
                'batch_idx': batch_idx,
                'batch_size': batch['src_img'].shape[0] if 'src_img' in batch else 0,
                'image_shape': batch['src_img'].shape[1:] if 'src_img' in batch else None,
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
            if 'src_img' in features:
                print(f"Batch {batch_idx}: src features shape {features['src_img'].shape}")
            if 'trg_img' in features:
                print(f"Batch {batch_idx}: trg features shape {features['trg_img'].shape}")
                
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
