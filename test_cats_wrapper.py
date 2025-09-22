"""
Test script to verify the CATs++ wrapper works correctly.
"""

import torch
from torch.utils.data import DataLoader
from src.data.synth.datasets.cats_wrapper import CATsSyntheticDataset


def test_cats_wrapper():
    """Test the CATs++ wrapper with a small dataset."""
    
    print("Testing CATs++ wrapper...")
    
    # Create a small test dataset
    dataset = CATsSyntheticDataset(
        shader_code_path='/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/quatJulia.c',
        num_samples=10,  # Small number for testing
        size=128,  # Smaller size for faster testing
        split='trn',
        augmentation=False,
        seed=42,
    )
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # Create a dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=0,  # Use 0 workers for testing
        shuffle=False
    )
    
    # Test getting a batch
    print("Testing batch retrieval...")
    try:
        batch = next(iter(dataloader))
        
        print("Batch keys:", list(batch.keys()))
        print("Source image shape:", batch['src_img'].shape)
        print("Target image shape:", batch['trg_img'].shape)
        print("Source keypoints shape:", batch['src_kps'].shape)
        print("Target keypoints shape:", batch['trg_kps'].shape)
        print("Number of points:", batch['n_pts'])
        
        # Verify tensor types and ranges
        assert batch['src_img'].dtype == torch.float32
        assert batch['trg_img'].dtype == torch.float32
        assert batch['src_kps'].dtype == torch.float32
        assert batch['trg_kps'].dtype == torch.float32
        
        # Verify image normalization (should be in [-1, 1] range approximately)
        print("Source image range:", batch['src_img'].min().item(), "to", batch['src_img'].max().item())
        print("Target image range:", batch['trg_img'].min().item(), "to", batch['trg_img'].max().item())
        
        # Verify keypoint ranges
        print("Source keypoints range:", batch['src_kps'].min().item(), "to", batch['src_kps'].max().item())
        print("Target keypoints range:", batch['trg_kps'].min().item(), "to", batch['trg_kps'].max().item())
        
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_cats_wrapper()
    if success:
        print("\n🎉 CATs++ wrapper is working correctly!")
    else:
        print("\n💥 CATs++ wrapper has issues that need to be fixed.")
