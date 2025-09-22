"""
Example usage of the CATs++ wrapper with synthetic correspondence dataset.
This demonstrates how to use the wrapper for training CATs++ models.
"""

import torch
from torch.utils.data import DataLoader
from src.data.synth.datasets.cats_wrapper import CATsSyntheticDataset


def example_basic_usage():
    """Basic example of using the CATs++ wrapper."""
    
    print("=== Basic Usage Example ===")
    
    # Create dataset
    dataset = CATsSyntheticDataset(
        num_samples=100,  # Number of synthetic samples
        size=256,         # Image size
        split='trn',      # Training split
        augmentation=True, # Enable augmentation
        seed=42,
    )
    
    print(f"Created dataset with {len(dataset)} samples")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=2,
        shuffle=True
    )
    
    # Get a batch
    batch = next(iter(dataloader))
    
    print("Batch structure:")
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"  {key}: {value.shape} ({value.dtype})")
        else:
            print(f"  {key}: {type(value)}")
    
    return batch


def example_training_setup():
    """Example of setting up for training."""
    
    print("\n=== Training Setup Example ===")
    
    # Create train and validation datasets
    train_dataset = CATsSyntheticDataset(
        num_samples=1000,
        size=256,
        split='trn',
        augmentation=True,
        seed=42,
    )
    
    val_dataset = CATsSyntheticDataset(
        num_samples=200,
        size=256,
        split='val',
        augmentation=False,  # No augmentation for validation
        seed=43,  # Different seed for validation
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        num_workers=4,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        num_workers=4,
        shuffle=False
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


def example_custom_shaders():
    """Example with custom shader configuration."""
    
    print("\n=== Custom Shaders Example ===")
    
    # Custom shader configuration
    custom_shaders = {
        'program0': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/terrain_sdf.glsl',
        'program1': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/quatJulia.c',
        'merge': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/merge.glsl',
    }
    
    dataset = CATsSyntheticDataset(
        shaders=custom_shaders,
        num_samples=50,
        size=128,
        split='trn',
        augmentation=False,
        seed=123,
    )
    
    print(f"Created dataset with custom shaders: {len(dataset)} samples")
    
    # Test getting a sample
    sample = dataset[0]
    print("Sample keys:", list(sample.keys()))
    print("Source image shape:", sample['src_img'].shape)
    print("Target image shape:", sample['trg_img'].shape)
    
    return dataset


def example_keypoint_analysis():
    """Example of analyzing generated keypoints."""
    
    print("\n=== Keypoint Analysis Example ===")
    
    dataset = CATsSyntheticDataset(
        num_samples=10,
        size=128,
        split='trn',
        augmentation=False,
        seed=456,
    )
    
    # Analyze keypoints from multiple samples
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        src_kps = sample['src_kps']
        trg_kps = sample['trg_kps']
        n_pts = sample['n_pts']
        
        print(f"Sample {i+1}:")
        print(f"  Number of valid points: {n_pts}")
        print(f"  Source keypoints range: [{src_kps.min():.1f}, {src_kps.max():.1f}]")
        print(f"  Target keypoints range: [{trg_kps.min():.1f}, {trg_kps.max():.1f}]")
        
        # Check for invalid keypoints (marked with -1)
        src_valid = (src_kps >= 0).all(dim=0).sum()
        trg_valid = (trg_kps >= 0).all(dim=0).sum()
        print(f"  Valid source keypoints: {src_valid}")
        print(f"  Valid target keypoints: {trg_valid}")


def main():
    """Run all examples."""
    
    print("CATs++ Synthetic Dataset Wrapper Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_basic_usage()
        example_training_setup()
        example_custom_shaders()
        example_keypoint_analysis()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        print("\nTo start training with CATs++:")
        print("python train_cats.py --num_samples 1000 --epochs 50 --batch-size 16")
        
    except Exception as e:
        print(f"❌ Error in examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
