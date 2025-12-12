"""
Build and save coresets for datasets.

This script streams through datasets and builds weighted coresets,
saving them to disk for later use in coverage analysis or validation.

The coreset construction uses optimized incremental MiniBatchKMeans:
- Uses partial_fit() for true streaming updates (no need to hold all data in memory)
- Warm-starts with existing centers for efficient incremental learning
- Handles weighted centers properly by reassigning all points after updates

Usage:
    python scripts/build_coresets.py --config configs/coreset_configs/build_datasets.yaml

To process complete datasets, set num_batches: null in the config (or omit it).
"""

import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.coreset import CoresetConfig, WeightedCoreset, load_config_from_yaml
from src.coreset.validation import extract_flow_vectors_from_batch
from src.data.synth.datasets.CorrespondenceDataset import CorrespondenceDataset
from src.mmd.encoders import BaseFeatureEncoder, ResNet101Encoder


def create_dataset_from_config(
    dataset_name: str,
    split: str,
    common_params: dict,
    dataset_overrides: dict
) -> CorrespondenceDataset:
    """
    Create a CorrespondenceDataset from config parameters.
    
    Similar to calculate_mmd.py implementation.
    """
    # Start with common parameters
    dataset_config = common_params.copy()
    
    # Apply dataset-specific overrides
    if dataset_name in dataset_overrides:
        dataset_config.update(dataset_overrides[dataset_name])
    
    # Handle size tuple
    if 'size' in dataset_config and isinstance(dataset_config['size'], list):
        dataset_config['size'] = tuple(dataset_config['size'])
    
    # Set split
    dataset_config['split'] = split
    
    # Handle max_kps: null -> None
    if 'max_kps' in dataset_config and dataset_config['max_kps'] is None:
        dataset_config['max_kps'] = None
    
    # Add dataset-specific parameters based on dataset type
    if dataset_name == 'synthetic':
        pass  # Synthetic-specific params already in overrides
    elif dataset_name == 'tss':
        dataset_config['thres'] = dataset_config.get('thres', 'img')
        dataset_config['reverse_flow'] = dataset_config.get('reverse_flow', False)
    elif dataset_name == 'middlebury':
        dataset_config['reverse_flow'] = dataset_config.get('reverse_flow', False)
    elif dataset_name == 'pointodyssey':
        dataset_config['reverse_flow'] = dataset_config.get('reverse_flow', True)
        dataset_config['thres'] = dataset_config.get('thres', 'img')
    elif dataset_name in ['kitti2012', 'kitti2015']:
        dataset_config['reverse_flow'] = dataset_config.get('reverse_flow', False)
        dataset_config['thres'] = dataset_config.get('thres', 'img')
        # Handle special case for kitti_val_use_full_training
        kitti_val_use_full_training = dataset_config.get('kitti_val_use_full_training', False)
        if kitti_val_use_full_training and dataset_config.get('split') == 'val':
            dataset_config['split'] = 'training'  # Use full training set for validation
    elif dataset_name == 'flyingthings':
        dataset_config['reverse_flow'] = dataset_config.get('reverse_flow', True)
    elif dataset_name in ['spair', 'pfpascal', 'pfwillow']:
        pass  # datapath already set in overrides
    
    print(f"Creating dataset: {dataset_name} (split: {split})")
    dataset = CorrespondenceDataset(dataset_name, **dataset_config)
    return dataset


def create_encoder(encoder_name: str, device: torch.device) -> BaseFeatureEncoder:
    """Create feature encoder by name."""
    if encoder_name == 'resnet101':
        return ResNet101Encoder(device=device)
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}. Supported: 'resnet101'")


def extract_features_from_batch(
    batch: dict,
    encoder: BaseFeatureEncoder,
) -> np.ndarray:
    """
    Extract flattened features [N, C] from a batch using the provided encoder.
    
    Looks for common image keys in the batch.
    """
    if encoder is None:
        raise ValueError("Encoder must be provided for feature extraction.")

    # Find image tensor
    if 'src_img' in batch:
        img = batch['src_img']
    elif 'source' in batch:
        img = batch['source']
    elif 'image0' in batch:
        img = batch['image0']
    else:
        raise ValueError(f"Could not find source image in batch. Available keys: {batch.keys()}")

    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img)

    feats = encoder.extract_features(img)  # [N, C] (flattened)
    return feats.cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description='Build and save weighted coresets for datasets'
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--encoder', type=str, default=None,
        help='Feature encoder to use for representation!=flow (overrides config, default: resnet101)'
    )
    parser.add_argument(
        '--subsample-fraction', type=float, default=None,
        help='Uniform subsample fraction for very dense batches (overrides config)'
    )
    parser.add_argument(
        '--subsample-threshold', type=int, default=None,
        help='If a batch has more vectors than this, subsample (overrides config)'
    )
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract config sections
    datasets_config = config['datasets']
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4)
    encoder_name = args.encoder if args.encoder is not None else config.get('encoder', 'resnet101')
    common_params = config.get('dataset_params', {})
    dataset_overrides = config.get('dataset_overrides', {})
    
    # Load coreset config
    if 'coreset' in config:
        coreset_cfg = config['coreset']
        if 'preset' in coreset_cfg:
            # Load from preset file
            preset_file = config.get(
                'coreset_config_file',
                'src/configs/coreset_configs/coreset_config.yaml'
            )
            coreset_config = load_config_from_yaml(
                preset_file,
                preset=coreset_cfg['preset']
            )
        else:
            coreset_config = CoresetConfig(**coreset_cfg)
    else:
        # Default config
        coreset_config = CoresetConfig(
            K_max=10000,
            K_overflow=5000,
            distance='euclidean',
            device='cpu'
        )
    
    print("\n" + "="*60)
    print("CORESET CONFIGURATION")
    print("="*60)
    print(f"K_max: {coreset_config.K_max}")
    print(f"K_overflow: {coreset_config.K_overflow}")
    print(f"Distance: {coreset_config.distance}")
    print(f"Device: {coreset_config.device}")
    
    # Subsample settings: from CLI if provided, else from config
    subsample_cfg = config.get('subsample', {})
    subsample_fraction = args.subsample_fraction if args.subsample_fraction is not None else subsample_cfg.get('fraction', 1.0)
    subsample_threshold = args.subsample_threshold if args.subsample_threshold is not None else subsample_cfg.get('threshold', 0)

    # Determine if we need a feature encoder (for representation != flow)
    needs_encoder = any(ds.get('representation', 'flow') != 'flow' for ds in datasets_config)
    encoder = None
    encoder_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if needs_encoder:
        encoder = create_encoder(encoder_name, encoder_device)

    # Process each dataset
    total_datasets = len(datasets_config)
    
    for ds_idx, ds_config in enumerate(datasets_config, 1):
        dataset_name = ds_config['name']
        split = ds_config['split']
        representation = ds_config.get('representation', 'flow')
        # num_batches: None = process all batches (complete dataset)
        # Set to a number to limit processing for testing
        num_batches = ds_config.get('num_batches', None)
        is_eval = ds_config.get('is_eval', False)
        output_path = ds_config['output']
        
        print("\n" + "="*60)
        print(f"DATASET {ds_idx}/{total_datasets}: {dataset_name} ({split})")
        print("="*60)
        print(f"Is eval: {is_eval}")
        if num_batches is None:
            print(f"Num batches: ALL (complete dataset)")
        else:
            print(f"Num batches: {num_batches} (limited for testing)")
        print(f"Representation: {representation}")
        print(f"Output: {output_path}")
        
        start_time = time.time()
        
        # Create dataset
        print(f"  Creating dataset object...")
        dataset_start = time.time()
        dataset = create_dataset_from_config(
            dataset_name, split, common_params, dataset_overrides
        )
        print(f"  ✓ Dataset created in {time.time() - dataset_start:.1f}s")
        
        # Create dataloader
        # Use num_workers=0 for synthetic (GPU-bound rendering)
        workers = 0 if dataset_name == 'synthetic' else num_workers
        
        print(f"  Creating dataloader (workers={workers})...")
        dataloader_start = time.time()
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            pin_memory=False
        )
        print(f"  ✓ DataLoader created in {time.time() - dataloader_start:.1f}s")
        print(f"  Waiting for first batch...")
        
        # Build coreset
        coreset = WeightedCoreset(
            K_max=coreset_config.K_max,
            K_overflow=coreset_config.K_overflow,
            distance=coreset_config.distance,
            device=coreset_config.device,
            is_eval=is_eval,
            epsilon_quantile=coreset_config.epsilon_quantile,
            max_epsilon_samples=coreset_config.max_epsilon_samples,
        )
        
        batches_processed = 0
        total_vectors = 0
        
        # Create progress bar
        pbar = tqdm(
            enumerate(dataloader),
            total=num_batches if num_batches else len(dataloader),
            desc=f"Processing {dataset_name}",
            unit="batch",
            ncols=100
        )
        
        for batch_idx, batch in pbar:
            if num_batches is not None and batches_processed >= num_batches:
                break
            
            # Extract vectors based on representation
            if representation == 'flow':
                vectors = extract_flow_vectors_from_batch(batch)
            elif representation == 'resnet':
                vectors = extract_features_from_batch(batch, encoder)
            else:
                raise ValueError(f"Unknown representation '{representation}'. Supported: flow, resnet.")
            
            if vectors is not None and len(vectors) > 0:
                # Uniform subsample if batch is very dense
                if len(vectors) > subsample_threshold and subsample_fraction < 1.0:
                    n_keep = max(
                        subsample_threshold,
                        int(len(vectors) * subsample_fraction)
                    )
                    if n_keep < len(vectors):
                        idx = np.random.choice(len(vectors), size=n_keep, replace=False)
                        vectors = vectors[idx]

                coreset.update(vectors)
                total_vectors += len(vectors)
            
            batches_processed += 1
            
            # Update progress bar with current stats
            pbar.set_postfix({
                'vectors': total_vectors,
                'centers': len(coreset.centers) if coreset.centers is not None else 0
            })
        
        pbar.close()
        
        # Finalize
        print(f"\nFinalizing coreset...")
        coreset.finalize()
        
        elapsed_time = time.time() - start_time
        
        print(f"\nCoreset statistics:")
        print(f"  Centers: {len(coreset.get_centers())}")
        print(f"  Total samples: {coreset.total_samples}")
        print(f"  Dimension: {coreset.dimension}")
        print(f"  Time taken: {elapsed_time:.1f}s ({elapsed_time/60:.1f}min)")
        print(f"  Throughput: {total_vectors/elapsed_time:.0f} vectors/sec")
        
        # Save
        coreset.save(output_path)
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
