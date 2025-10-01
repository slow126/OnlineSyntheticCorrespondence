#!/usr/bin/env python3
"""
Quick test script to extract DinoV3 features from a few synthetic samples
and visualize the results.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(project_root))

# Import our modules
from src.data.synth.online_synth_datamodule import create_datamodule_from_config
from models.DinoV3.DinoV3 import DinoV3


def test_single_batch():
    """Test feature extraction on a single batch."""
    
    print("=== Testing DinoV3 Feature Extraction ===")
    
    # Initialize DinoV3 model
    print("Loading DinoV3 model...")
    dino_model = DinoV3()
    print("DinoV3 model loaded!")
    
    # Create datamodule
    print("Creating datamodule...")
    config_path = "src/configs/online_synth_configs/config.yaml"
    dm = create_datamodule_from_config(config_path, batch_size=4)  # Small batch for testing
    dm.post_init()
    dm.setup('fit')
    
    print(f"Dataset length: {len(dm.train_data)}")
    
    # Get a single batch
    print("Getting a batch...")
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    # Process the batch through the datamodule's batch transfer method
    print("Processing batch through on_after_batch_transfer...")
    processed_batch = dm.on_after_batch_transfer(batch, 0)
    
    print(f"Processed batch keys: {processed_batch.keys()}")
    print(f"Source shape: {processed_batch['src'].shape}")
    print(f"Target shape: {processed_batch['trg'].shape}")
    print(f"Flow shape: {processed_batch['flow'].shape}")
    
    # Extract features for source images
    print("\nExtracting DinoV3 features...")
    src_images = processed_batch['src']  # (batch_size, 3, H, W)
    trg_images = processed_batch['trg']  # (batch_size, 3, H, W)
    
    # Process first image from source
    img_tensor = src_images[0]  # (3, H, W)
    
    # Convert tensor to PIL Image
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    pil_image = Image.fromarray(img_np)
    
    # Extract spatial features
    print("Extracting spatial features...")
    spatial_features = dino_model.get_spatial_features(pil_image)
    print(f"Spatial features shape: {spatial_features.shape}")
    
    # Visualize the features
    print("Creating feature visualization...")
    dino_model.visualize_spatial_features(
        pil_image, 
        save_path="test_spatial_features.png",
        input_image_save_path="test_input_image.png"
    )
    
    # Also visualize the target image
    trg_img_tensor = trg_images[0]
    trg_img_np = trg_img_tensor.permute(1, 2, 0).cpu().numpy()
    trg_img_np = np.clip(trg_img_np, 0, 1)
    trg_img_np = (trg_img_np * 255).astype(np.uint8)
    trg_pil_image = Image.fromarray(trg_img_np)
    
    dino_model.visualize_spatial_features(
        trg_pil_image,
        save_path="test_target_spatial_features.png",
        input_image_save_path="test_target_image.png"
    )
    
    print("\n=== Test Complete ===")
    print("Generated files:")
    print("  - test_input_image.png (source image)")
    print("  - test_spatial_features.png (source features)")
    print("  - test_target_image.png (target image)")
    print("  - test_target_spatial_features.png (target features)")
    
    return spatial_features


def test_batch_processing():
    """Test processing multiple images in a batch."""
    
    print("\n=== Testing Batch Processing ===")
    
    # Initialize DinoV3 model
    dino_model = DinoV3()
    
    # Create datamodule with small batch
    config_path = "src/configs/online_synth_configs/config.yaml"
    dm = create_datamodule_from_config(config_path, batch_size=2)
    dm.post_init()
    dm.setup('fit')
    
    # Get a batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    # Process the batch through the datamodule's batch transfer method
    processed_batch = dm.on_after_batch_transfer(batch, 0)
    
    print(f"Processing batch of {processed_batch['src'].shape[0]} images...")
    
    # Process all images in the batch
    src_images = processed_batch['src']
    batch_features = []
    
    for i in range(src_images.shape[0]):
        # Convert tensor to PIL
        img_tensor = src_images[i]
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)
        
        # Extract features
        features = dino_model.get_spatial_features(pil_image)
        batch_features.append(features.cpu())
        
        print(f"  Image {i}: features shape {features.shape}")
    
    # Stack features
    stacked_features = torch.stack(batch_features, dim=0)
    print(f"Stacked features shape: {stacked_features.shape}")
    
    return stacked_features


if __name__ == '__main__':
    # Test single batch
    features = test_single_batch()
    
    # Test batch processing
    batch_features = test_batch_processing()
    
    print("\nAll tests completed successfully!")
