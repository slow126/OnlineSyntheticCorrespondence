#!/usr/bin/env python3
"""
Quick script to check SPair train dataloader length.
Creates the dataset and dataloader exactly as training would.
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from train_cats_unified import load_config, create_training_dataset

def main():
    config_path = 'test_spair_minimal.yaml'
    print(f"Loading config from: {config_path}")
    
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    dataset_config = config['dataset']
    training_config = config['training']
    
    print("\n" + "="*60)
    print("SPair Train Dataloader Length Check")
    print("="*60)
    
    print(f"\nDataset config:")
    print(f"  dataset_name: {dataset_config.get('dataset_name')}")
    print(f"  split: {dataset_config.get('split')}")
    print(f"  size: {dataset_config.get('size')}")
    
    print(f"\nTraining config:")
    print(f"  batch_size: {training_config.get('batch_size')}")
    print(f"  n_threads: {training_config.get('n_threads')}")
    
    print("\nCreating training dataset...")
    try:
        train_dataset = create_training_dataset(config, device=None)
        
        print(f"✅ Dataset created successfully!")
        print(f"📊 Dataset length: {len(train_dataset):,}")
        
        # Create dataloader exactly as training would
        batch_size = training_config['batch_size']
        n_threads = training_config.get('n_threads', 0)
        
        # Check if synthetic (would use num_workers=0)
        train_dataset_name = dataset_config.get('dataset_name', '')
        is_synthetic = isinstance(train_dataset_name, str) and train_dataset_name.startswith("synthetic")
        train_num_workers = 0 if is_synthetic else n_threads
        
        print(f"\nCreating train dataloader...")
        print(f"  batch_size: {batch_size}")
        print(f"  num_workers: {train_num_workers}")
        print(f"  shuffle: True")
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=train_num_workers,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
            prefetch_factor=batch_size if train_num_workers > 0 else None,
            pin_memory=True if train_num_workers > 0 else False
        )
        
        print(f"\n✅ Train dataloader created successfully!")
        print(f"📊 Train dataloader length: {len(train_dataloader):,} batches")
        print(f"📊 Total samples per epoch: {len(train_dataloader) * batch_size:,}")
        
        # Try to get a batch to verify it works
        if len(train_dataloader) > 0:
            print(f"\nTesting first batch...")
            batch = next(iter(train_dataloader))
            print(f"✅ Batch loaded successfully!")
            print(f"   Batch keys: {list(batch.keys())}")
            if 'src_img' in batch:
                print(f"   src_img shape: {batch['src_img'].shape}")
            if 'trg_img' in batch:
                print(f"   trg_img shape: {batch['trg_img'].shape}")
        else:
            print("⚠️  Dataloader is empty!")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
    return 0

if __name__ == '__main__':
    sys.exit(main())
