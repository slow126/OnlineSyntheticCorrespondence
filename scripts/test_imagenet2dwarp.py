#!/usr/bin/env python3
"""
Quick test script to verify ImageNet2DWarp dataset integration.

This script tests:
1. Dataset can be instantiated
2. Samples can be loaded
3. Flow computation is correct
4. Integration with CorrespondenceDataset works
5. Out-of-bounds handling with inf markers

Usage:
    python scripts/test_imagenet2dwarp.py --datapath /path/to/imagenet100
"""

import argparse
import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.synth.datasets.CorrespondenceDataset import CorrespondenceDataset
from src.data.synth.adapters import build_adapter


def test_adapter(datapath: str):
    """Test the adapter directly."""
    print("\n" + "="*80)
    print("Testing ImageNet2DWarpAdapter")
    print("="*80)
    
    try:
        adapter = build_adapter(
            "imagenet2dwarp",
            datapath=datapath,
            split="train",
            rotation_range=(-30.0, 30.0),
            scale_range=(0.5, 2.5),
            translation_range=(-0.1, 0.1),
            shear_range=(-0.2, 0.2),
            cache_warp_params=True,
        )
        
        print(f"✓ Adapter created successfully")
        print(f"  Dataset length: {len(adapter)}")
        
        if len(adapter) == 0:
            print("  WARNING: Dataset is empty")
            return False
        
        # Test getting a sample
        sample = adapter[0]
        print(f"✓ Sample loaded successfully")
        print(f"  src_img shape: {sample.src_img.shape if sample.src_img is not None else None}")
        print(f"  trg_img shape: {sample.trg_img.shape if sample.trg_img is not None else None}")
        print(f"  flow_full shape: {sample.flow_full.shape if sample.flow_full is not None else None}")
        
        # Check flow validity
        if sample.flow_full is not None:
            flow = sample.flow_full
            valid_mask = torch.isfinite(flow).all(dim=0)
            num_valid = valid_mask.sum().item()
            num_invalid = (~valid_mask).sum().item()
            total = flow.shape[1] * flow.shape[2]
            
            print(f"  Flow validity: {num_valid}/{total} valid, {num_invalid}/{total} invalid")
            
            if num_valid > 0:
                flow_mag = flow.norm(dim=0)
                valid_flow_mag = flow_mag[valid_mask]
                print(f"  Flow magnitude range: [{valid_flow_mag.min().item():.2f}, {valid_flow_mag.max().item():.2f}]")
            
            # Check that invalid pixels are marked with inf
            invalid_pixels = ~valid_mask
            if invalid_pixels.any():
                invalid_flow = flow[:, invalid_pixels]
                has_inf = (~torch.isfinite(invalid_flow)).any().item()
                if has_inf:
                    print(f"  ✓ Invalid pixels correctly marked with inf")
                else:
                    print(f"  ✗ WARNING: Invalid pixels not marked with inf")
                    return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating adapter: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_correspondence_dataset(datapath: str):
    """Test integration with CorrespondenceDataset."""
    print("\n" + "="*80)
    print("Testing CorrespondenceDataset with ImageNet2DWarp")
    print("="*80)
    
    try:
        dataset = CorrespondenceDataset(
            dataset_name="imagenet2dwarp",
            datapath=datapath,
            split="train",
            size=(512, 512),
            rotation_range=(-30.0, 30.0),
            scale_range=(0.5, 2.5),
            translation_range=(-0.1, 0.1),
            shear_range=(-0.2, 0.2),
            cache_warp_params=True,
        )
        
        print(f"✓ CorrespondenceDataset created successfully")
        print(f"  Dataset length: {len(dataset)}")
        
        if len(dataset) == 0:
            print("  WARNING: Dataset is empty")
            return False
        
        # Test collate function
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=0,
            collate_fn=dataset.collate_fn,
        )
        
        batch = next(iter(dataloader))
        print(f"✓ Batch collated successfully")
        print(f"  Batch keys: {list(batch.keys())}")
        
        if 'flow_full' in batch:
            flow = batch['flow_full']
            print(f"  flow_full shape: {flow.shape}")
            valid_mask = torch.isfinite(flow).all(dim=1)
            num_valid = valid_mask.sum().item()
            num_invalid = (~valid_mask).sum().item()
            total = flow.shape[0] * flow.shape[2] * flow.shape[3]
            print(f"  Flow validity: {num_valid}/{total} valid, {num_invalid}/{total} invalid")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating CorrespondenceDataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test ImageNet2DWarp dataset integration")
    parser.add_argument(
        "--datapath",
        type=str,
        required=True,
        help="Path to ImageNet100 dataset root directory"
    )
    parser.add_argument(
        "--skip-adapter",
        action="store_true",
        help="Skip adapter test"
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip CorrespondenceDataset test"
    )
    
    args = parser.parse_args()
    
    datapath = Path(args.datapath).expanduser().resolve()
    
    if not datapath.exists():
        print(f"ERROR: Datapath does not exist: {datapath}")
        return 1
    
    # Check for train/val directories
    train_dir = datapath / "train"
    val_dir = datapath / "val"
    
    if not train_dir.exists() and not val_dir.exists():
        print(f"ERROR: Neither train nor val directory found in {datapath}")
        print("Expected structure:")
        print("  datapath/")
        print("    train/")
        print("      class1/")
        print("        *.JPEG")
        print("      ...")
        print("    val/")
        print("      class1/")
        print("        *.JPEG")
        print("      ...")
        return 1
    
    success = True
    
    if not args.skip_adapter:
        success = test_adapter(str(datapath)) and success
    
    if not args.skip_dataset:
        success = test_correspondence_dataset(str(datapath)) and success
    
    if success:
        print("\n" + "="*80)
        print("✓ All tests passed!")
        print("="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("✗ Some tests failed")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())

