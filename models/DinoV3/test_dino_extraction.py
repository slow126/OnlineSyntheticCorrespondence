#!/usr/bin/env python3
"""
Quick test script to extract DinoV3 features from a few synthetic samples
and visualize the results. Updated to work with the new extract_dino_features.py approach.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(project_root))

# Import our modules (matching extract_dino_features.py)
from models.DinoV3.DinoV3 import DinoV3
from src.data.synth.datasets.OnlineCorrespondenceDataset import OnlineCorrespondenceDataset


def extract_features_from_batch(dino_model, batch, device='cuda'):
    """
    Extract spatial features from a batch of images using DinoV3.
    (Copied from extract_dino_features.py for consistency)
    
    Args:
        dino_model: DinoV3 model instance
        batch: Batch dictionary with 'src_img' and 'trg_img' keys
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


def test_single_batch():
    """Test feature extraction on a single batch using the new approach."""
    
    print("=== Testing DinoV3 Feature Extraction ===")
    
    # Initialize DinoV3 model
    print("Loading DinoV3 model...")
    dino_model = DinoV3()
    print("DinoV3 model loaded!")
    
    # Create dataset directly (matching extract_dino_features.py)
    print("Creating synthetic dataset...")
    dataset = OnlineCorrespondenceDataset(
        geometry_config_path='src/configs/online_synth_configs/OnlineGeometryConfig.yaml',
        processor_config_path='src/configs/online_synth_configs/OnlineProcessorConfig.yaml',
        split='train'
    )
    dataset.cuda()
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=4,  # Small batch for testing
        num_workers=0, 
        shuffle=True, 
        collate_fn=dataset.collate_fn
    )
    
    print(f"Dataset size: {len(dataloader)} batches")
    
    # Get a single batch
    print("Getting a batch...")
    batch = next(iter(dataloader))
    
    print(f"Batch keys: {list(batch.keys())}")
    if 'src_img' in batch:
        print(f"Source image shape: {batch['src_img'].shape}")
    if 'trg_img' in batch:
        print(f"Target image shape: {batch['trg_img'].shape}")
    if 'flow' in batch:
        print(f"Flow shape: {batch['flow'].shape}")
    if 'category' in batch:
        print(f"Categories: {batch['category']}")
    else:
        print("No category information found (synthetic dataset)")
    
    # Extract features using the same function as extract_dino_features.py
    print("\nExtracting DinoV3 features...")
    features = extract_features_from_batch(dino_model, batch, 'cuda')
    
    print(f"Extracted features keys: {list(features.keys())}")
    if 'src_img' in features:
        print(f"Source features shape: {features['src_img'].shape}")
    if 'trg_img' in features:
        print(f"Target features shape: {features['trg_img'].shape}")
    
    # Visualize features for the first batch (matching extract_dino_features.py)
    print("Creating feature visualization...")
    try:
        dino_model.visualize_features_grid(features)
        print("Feature visualization saved!")
    except Exception as viz_error:
        print(f"Could not visualize features: {viz_error}")
    
    # Also save individual images for inspection
    if 'src_img' in batch:
        src_img_tensor = batch['src_img'][0].cpu()
        # Normalize for visualization
        src_img_vis = (src_img_tensor - src_img_tensor.min()) / (src_img_tensor.max() - src_img_tensor.min() + 1e-8)
        # Save as numpy for inspection
        np.save("test_src_img.npy", src_img_vis.numpy())
        print("Saved source image as test_src_img.npy")
    
    if 'trg_img' in batch:
        trg_img_tensor = batch['trg_img'][0].cpu()
        # Normalize for visualization
        trg_img_vis = (trg_img_tensor - trg_img_tensor.min()) / (trg_img_tensor.max() - trg_img_tensor.min() + 1e-8)
        # Save as numpy for inspection
        np.save("test_trg_img.npy", trg_img_vis.numpy())
        print("Saved target image as test_trg_img.npy")
    
    print("\n=== Test Complete ===")
    print("Generated files:")
    print("  - test_src_img.npy (source image)")
    print("  - test_trg_img.npy (target image)")
    print("  - Feature visualization (if successful)")
    
    return features


def test_batch_processing():
    """Test processing multiple batches using the new approach."""
    
    print("\n=== Testing Batch Processing ===")
    
    # Initialize DinoV3 model
    dino_model = DinoV3()
    
    # Create dataset directly (matching extract_dino_features.py)
    print("Creating synthetic dataset...")
    dataset = OnlineCorrespondenceDataset(
        geometry_config_path='src/configs/online_synth_configs/OnlineGeometryConfig.yaml',
        processor_config_path='src/configs/online_synth_configs/OnlineProcessorConfig.yaml',
        split='train'
    )
    dataset.cuda()
    
    # Create dataloader with small batch
    dataloader = DataLoader(
        dataset, 
        batch_size=2,  # Small batch for testing
        num_workers=0, 
        shuffle=True, 
        collate_fn=dataset.collate_fn
    )
    
    print(f"Processing {min(3, len(dataloader))} batches...")
    
    all_features = []
    
    # Process a few batches
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 3:  # Limit to 3 batches for testing
            break
            
        print(f"\nProcessing batch {batch_idx}...")
        print(f"  Batch keys: {list(batch.keys())}")
        if 'category' in batch:
            print(f"  Categories: {batch['category']}")
        
        # Extract features using the same function as extract_dino_features.py
        features = extract_features_from_batch(dino_model, batch, 'cuda')
        
        print(f"  Extracted features keys: {list(features.keys())}")
        if 'src_img' in features:
            print(f"  Source features shape: {features['src_img'].shape}")
        if 'trg_img' in features:
            print(f"  Target features shape: {features['trg_img'].shape}")
        
        all_features.append(features)
    
    print(f"\nProcessed {len(all_features)} batches successfully!")
    
    # Calculate some statistics
    if all_features and 'src_img' in all_features[0]:
        src_features = [f['src_img'] for f in all_features]
        stacked_src_features = torch.cat(src_features, dim=0)
        print(f"Total source samples processed: {stacked_src_features.shape[0]}")
        print(f"Feature shape: {stacked_src_features.shape[1:]}")
        print(f"Feature statistics:")
        print(f"  Mean: {stacked_src_features.mean().item():.4f}")
        print(f"  Std: {stacked_src_features.std().item():.4f}")
    
    return all_features


if __name__ == '__main__':
    print("Testing DinoV3 feature extraction with new extract_dino_features.py approach...")
    
    # Test single batch
    print("\n" + "="*60)
    features = test_single_batch()
    
    # Test batch processing
    print("\n" + "="*60)
    batch_features = test_batch_processing()
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("The test script now uses the same approach as extract_dino_features.py:")
    print("  - Direct OnlineCorrespondenceDataset creation")
    print("  - Correct batch format (src_img, trg_img)")
    print("  - Same feature extraction function")
    print("  - Compatible with train_cats.py dataset handling")
