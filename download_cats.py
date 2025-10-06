#!/usr/bin/env python3
"""
Minimal script to download necessary files for CATs++ training.
Downloads model weights and datasets without CUDA dependencies.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the models path for imports
sys.path.append('models/CATs_PlusPlus')
sys.path.append('models/CATs_PlusPlus/data')

# Import only the download module
import models.CATs_PlusPlus.data.download as download


def main():
    parser = argparse.ArgumentParser(description='Download CATs++ datasets and model weights')
    
    # Dataset parameters
    parser.add_argument('--datapath', type=str, default='./models/Datasets_CATs',
                        help='Path to download datasets')
    parser.add_argument('--benchmark', type=str, default='spair', 
                        choices=['spair', 'pfpascal', 'pfwillow', 'caltech'],
                        help='Dataset to download')
    parser.add_argument('--download_all', action='store_true',
                        help='Download all available datasets')
    
    # Model weights
    parser.add_argument('--download_weights', action='store_true',
                        help='Download pretrained model weights')
    parser.add_argument('--weights_path', type=str, default='./models/CATs_PlusPlus/pretrained',
                        help='Path to download model weights')
    
    args = parser.parse_args()
    
    print("🚀 Starting CATs++ download process...")
    
    # Create directories
    os.makedirs(args.datapath, exist_ok=True)
    if args.download_weights:
        os.makedirs(args.weights_path, exist_ok=True)
    
    # Download datasets
    if args.download_all:
        datasets = ['spair', 'pfpascal', 'pfwillow', 'caltech']
        print(f"📦 Downloading all datasets to {args.datapath}")
        for dataset in datasets:
            print(f"  Downloading {dataset}...")
            try:
                download.download_dataset(args.datapath, dataset)
                print(f"  ✅ {dataset} downloaded successfully")
            except Exception as e:
                print(f"  ❌ Failed to download {dataset}: {e}")
    else:
        print(f"📦 Downloading {args.benchmark} dataset to {args.datapath}")
        try:
            download.download_dataset(args.datapath, args.benchmark)
            print(f"✅ {args.benchmark} dataset downloaded successfully")
        except Exception as e:
            print(f"❌ Failed to download {args.benchmark}: {e}")
    
    # Download model weights (if requested)
    if args.download_weights:
        print(f"🏋️ Downloading pretrained model weights...")
        try:
            # Download backbone weights (ResNet101) - these are automatically cached by PyTorch
            print("  Downloading ResNet101 backbone weights...")
            import torch
            from torchvision.models import resnet
            
            # This will download and cache the ResNet101 weights
            backbone = resnet.resnet101(pretrained=True)
            print("  ✅ ResNet101 backbone weights downloaded and cached")
            
            # Download full CATs++ model weights if available
            print("  Downloading CATs++ pretrained model weights...")
            try:
                # Check if pretrained weights already exist
                pretrained_path = os.path.join(args.weights_path, 'spair')
                if not os.path.exists(pretrained_path):
                    os.makedirs(pretrained_path, exist_ok=True)
                
                # Download pretrained model from Google Drive (if available)
                # Note: You may need to add the actual Google Drive ID for CATs++ pretrained weights
                print("  ⚠️  CATs++ pretrained weights download not implemented yet")
                print("     You can manually download from the CATs++ repository if needed")
                
            except Exception as e:
                print(f"  ⚠️  Could not download CATs++ pretrained weights: {e}")
                print("     Training will start from scratch with pretrained backbone")
            
        except Exception as e:
            print(f"  ❌ Failed to download model weights: {e}")
            print("   Backbone weights will be downloaded automatically when training starts")
    
    print("🎉 Download process completed!")
    print(f"📁 Datasets location: {args.datapath}")
    if args.download_weights:
        print(f"🏋️ Weights location: {args.weights_path}")


if __name__ == "__main__":
    main()
