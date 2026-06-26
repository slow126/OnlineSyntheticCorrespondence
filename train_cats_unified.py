"""
Unified training script for CATs++ model using config-based setup.
This script loads training configuration from YAML and uses CorrespondenceDataset for all datasets.
"""

import argparse
import csv
import os
import pickle
import random
import time
import yaml
from os import path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from termcolor import colored
from torch.utils.data import DataLoader
from pathlib import Path

# Import CATs++ model and utilities
from models.CATs_PlusPlus.models.cats_improved import CATsImproved
import models.CATs_PlusPlus.utils_training.optimize as optimize
from models.CATs_PlusPlus.utils_training.utils import parse_list, load_checkpoint, save_checkpoint, boolean_string
import models.CATs_PlusPlus.data.download as download
from models.CATs_PlusPlus.utils_training.eval_instance import MultiBenchmarkEvaluator
from models.CATs_PlusPlus.utils_training.optimize_multi import validate_epoch_multi_benchmark
from src.data.synth.datasets.CorrespondenceDataset import CorrespondenceDataset
from src.data.synth.datasets.MixedCorrespondenceDataset import MixedCorrespondenceDataset
from src.data.synth.datasets.visualizers import CorrespondenceVisualizer
try:
    from src.data.synth.datasets.cats_flow_visualizers import CATSFlowVisualizer
except Exception:
    CATSFlowVisualizer = None


def visualize_batch_flow(model, batch, device, train_dataset_name, val_dataset_name, split_name, flow_source='gt', 
                         feature_size=32, epoch=None):
    """
    Visualize batch flow (ground truth or predicted) for debugging.
    
    Args:
        model: Model instance (can be None if flow_source='gt' or if 'pred_flow' already in batch)
        batch: Batch dictionary containing images and flow
        device: Device to run model on
        train_dataset_name: Name of training dataset (for grouping experiments)
        val_dataset_name: Name of validation dataset (only used when split_name='val')
        split_name: 'train' or 'val' (for directory naming)
        flow_source: 'gt' (ground truth from dataset) or 'pred' (model prediction)
        feature_size: Feature size for downsampled flow visualization
        epoch: Optional epoch number (for directory naming)
    """
    debug_dir = Path("debug")
    debug_dir.mkdir(exist_ok=True, parents=True)
    
    # Create train dataset-specific subdirectory (groups all experiments by training dataset)
    train_dataset_debug_dir = debug_dir / train_dataset_name
    train_dataset_debug_dir.mkdir(exist_ok=True, parents=True)
    
    # Create split-specific subdirectory
    if split_name == 'train':
        # For training: debug/{train_dataset_name}/train/
        split_debug_dir = train_dataset_debug_dir / 'train'
    elif split_name == 'val':
        # For validation: debug/{train_dataset_name}/val/{val_dataset_name}/
        if val_dataset_name is None:
            raise ValueError("val_dataset_name must be provided when split_name='val'")
        val_dir = train_dataset_debug_dir / 'val'
        val_dir.mkdir(exist_ok=True, parents=True)
        split_debug_dir = val_dir / val_dataset_name
    else:
        raise ValueError(f"split_name must be 'train' or 'val', got '{split_name}'")
    
    split_debug_dir.mkdir(exist_ok=True, parents=True)
    
    # Add epoch suffix if provided
    # For pre-training (epoch=-1), use "_pretrain", otherwise use epoch number
    if epoch is not None:
        if epoch == -1:
            epoch_suffix = "_pretrain"
        else:
            epoch_suffix = f"_epoch_{epoch + 1}"
    else:
        epoch_suffix = ""
    
    try:
        # Get flow - either from dataset or model prediction
        if flow_source == 'pred':
            # Check if pred_flow already exists in batch
            if 'pred_flow' in batch:
                print(f"Using existing 'pred_flow' from batch")
                pred_flow = batch['pred_flow']
                flow_tensor = pred_flow[0].cpu() if isinstance(pred_flow, torch.Tensor) else pred_flow[0].cpu()
            else:
                # Need to run forward pass
                if model is None or batch is None:
                    print(f"Warning: model/batch missing and 'pred_flow' not in batch. Skipping visualization.")
                    return
                print(f"Running model forward pass to get predictions...")
                model.eval()
                with torch.no_grad():
                    pred_flow = model(
                        batch['trg_img'].to(device, non_blocking=True),
                        batch['src_img'].to(device, non_blocking=True)
                    )
                flow_tensor = pred_flow[0].cpu()
            flow_key = 'pred_flow'
        else:  # flow_source == 'gt'
            # Prefer feature-grid flow if present, otherwise fall back to full-res
            flow_candidates = ['flow_downsampled', 'flow', 'flow_full']
            flow_key = next((k for k in flow_candidates if k in batch), None)
            if flow_key is None:
                raise KeyError("No flow tensor found in batch (looked for flow_downsampled/flow/flow_full)")
            flow_tensor = batch[flow_key]
            flow_tensor = flow_tensor[0].cpu() if isinstance(flow_tensor, torch.Tensor) else flow_tensor[0]
        
        # Visualize downsampled flow using CATSFlowVisualizer (raw batch, no normalization)
        try:
            from src.data.synth.datasets.cats_flow_visualizers import CATSFlowVisualizer
            
            # Check if flow is downsampled (feat_size x feat_size) or full resolution
            flow_shape = flow_tensor.shape
            
            if len(flow_shape) == 3 and flow_shape[1] == flow_shape[2] and flow_shape[1] == feature_size:
                # Flow is downsampled
                dataset_display_name = train_dataset_name if split_name == 'train' else val_dataset_name
                print(f"\nFlow is downsampled: shape={flow_shape}, feat_size={feature_size}, source={flow_source}")
                non_zero_count = ((flow_tensor[0] != 0) | (flow_tensor[1] != 0)).sum()
                print(f"Non-zero flow count: {non_zero_count} for dataset {dataset_display_name} ({split_name}, {flow_source})")
                flow_norms = flow_tensor.norm(dim=0)
                non_zero_mask = flow_norms > 0
                if non_zero_mask.any():
                    avg_length = flow_norms[non_zero_mask].mean().item()
                else:
                    avg_length = 0.0
                print(f"Average flow length: {avg_length} for dataset {dataset_display_name} ({split_name}, {flow_source})")

                # Create batch dict with raw images (not normalized - visualizer will handle display)
                # Use the flow tensor we selected above
                flow_to_visualize = pred_flow if flow_source == 'pred' else batch[flow_key]
                    
                batch_dict_raw = {
                    'src_img': batch['src_img'].cpu(),
                    'trg_img': batch['trg_img'].cpu(),
                    'flow_downsampled': flow_to_visualize.cpu() if isinstance(flow_to_visualize, torch.Tensor) else flow_to_visualize
                }
                
                # Create visualizer with normalization disabled to see actual batch values
                cats_visualizer = CATSFlowVisualizer(
                    feat_size=feature_size,
                    figsize=(20, 15),
                    dpi=150,
                    show_patch_boundaries=True,
                    normalize_images=False  # Don't normalize to see actual batch values
                )
                
                # Visualize side-by-side
                cats_visualizer.visualize_downsampled_flow_batch(
                    batch_dict_raw,
                    save_path=str(split_debug_dir / f"batch_downsampled_flow_{flow_key}{epoch_suffix}_side_by_side.png"),
                    max_samples=4,
                    visualization_mode='side_by_side'
                )
                
                # Visualize overlay
                cats_visualizer.visualize_downsampled_flow_batch(
                    batch_dict_raw,
                    save_path=str(split_debug_dir / f"batch_downsampled_flow_{flow_key}{epoch_suffix}_overlay.png"),
                    max_samples=4,
                    visualization_mode='overlay'
                )
                
                print(f"Saved CATS flow visualizations to {split_debug_dir} (raw batch, no normalization, {flow_source})")

            else:
                print(f"\nFlow is full resolution: shape={flow_shape}, skipping downsampled flow visualization")
                # Visualize full resolution flow
                from src.data.synth.datasets.visualizers import CorrespondenceVisualizer
                visualizer = CorrespondenceVisualizer()
                
                # Create batch dict with appropriate flow
                if flow_source == 'pred':
                    batch_vis = batch.copy()
                    batch_vis['flow'] = pred_flow.cpu()
                else:
                    batch_vis = batch.copy()
                    batch_vis['flow'] = batch[flow_key]
                
                visualizer.visualize_rendered_batch(
                    batch_vis, 
                    save_path=str(split_debug_dir / f"batch_full_resolution_flow_{flow_key}{epoch_suffix}_overlay.png"), 
                    visualization_mode="overlay"
                )
                visualizer.visualize_rendered_batch(
                    batch_vis, 
                    save_path=str(split_debug_dir / f"batch_full_resolution_flow_{flow_key}{epoch_suffix}_side_by_side.png"), 
                    visualization_mode="side_by_side"
                )
                print(f"Saved full resolution flow visualizations to {split_debug_dir} ({flow_source})")
                
        except ImportError as e:
            print(f"Could not import CATSFlowVisualizer: {e}")
        except Exception as e:
            print(f"Error creating CATS flow visualization: {e}")
            import traceback
            traceback.print_exc()

        print(f"Saved sample batch visualizations to {split_debug_dir} ({flow_source})")
    except Exception as e:
        print(f"Could not save sample batch for debug ({flow_source}): {e}")


def load_config(config_path):
    """Load and validate YAML config file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ['model', 'training', 'dataset', 'evaluation', 'paths']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in config file")
    
    # Validate evaluation benchmarks and alphas match
    eval_benchmarks = config['evaluation']['eval_benchmarks']
    eval_alphas = config['evaluation']['eval_alphas']
    if len(eval_benchmarks) != len(eval_alphas):
        raise ValueError(f"Number of eval_benchmarks ({len(eval_benchmarks)}) must match number of eval_alphas ({len(eval_alphas)})")
    
    return config


def create_training_dataset(config, device=None):
    """Create training dataset using CorrespondenceDataset or MixedCorrespondenceDataset"""
    dataset_config = config['dataset'].copy()
    
    # Check if this is a mixed dataset configuration
    is_mixed = dataset_config.get('mixed', False) or 'datasets' in dataset_config
    
    if is_mixed:
        # Mixed dataset configuration
        datasets_list = dataset_config.pop('datasets', [])
        percentages = dataset_config.pop('percentages', [])
        dataset_overrides = dataset_config.pop('dataset_overrides', {})
        epoch_size = dataset_config.pop('epoch_size', None)
        seed = dataset_config.pop('seed', None)
        
        if len(datasets_list) != len(percentages):
            raise ValueError(f"Number of datasets ({len(datasets_list)}) must match number of percentages ({len(percentages)})")
        
        # Common parameters (applied to all datasets unless overridden)
        common_params = dataset_config.copy()
        
        # Handle common parameters
        if 'max_kps' in common_params and common_params['max_kps'] is None:
            common_params['max_kps'] = None
        if 'size' in common_params and isinstance(common_params['size'], list):
            common_params['size'] = tuple(common_params['size'])
        
        # Handle verbose/debug from training config if not in dataset config
        if 'verbose' not in common_params:
            common_params['verbose'] = config['training'].get('enable_debug', False)
        if 'debug' not in common_params:
            common_params['debug'] = config['training'].get('enable_debug', False)
        
        # Create individual datasets
        created_datasets = []
        for dataset_name in datasets_list:
            # Start with common parameters
            ds_config = common_params.copy()
            
            # Apply dataset-specific overrides
            if dataset_name in dataset_overrides:
                ds_config.update(dataset_overrides[dataset_name])
            
            # Handle max_kps: null -> None
            if 'max_kps' in ds_config and ds_config['max_kps'] is None:
                ds_config['max_kps'] = None
            
            # Handle size tuple
            if 'size' in ds_config and isinstance(ds_config['size'], list):
                ds_config['size'] = tuple(ds_config['size'])
            
            print(f"Creating sub-dataset: {dataset_name}")
            sub_dataset = CorrespondenceDataset(dataset_name, **ds_config)
            created_datasets.append(sub_dataset)
        
        # Create mixed dataset
        print(f"Creating mixed dataset with {len(created_datasets)} datasets")
        mixed_dataset = MixedCorrespondenceDataset(
            datasets=created_datasets,
            percentages=percentages,
            epoch_size=epoch_size,
            seed=seed,
        )
        return mixed_dataset
    else:
        # Single dataset configuration (backward compatible)
        dataset_name = dataset_config.pop('dataset_name')
        
        # Handle max_kps: null -> None
        if 'max_kps' in dataset_config and dataset_config['max_kps'] is None:
            dataset_config['max_kps'] = None
        
        # Handle size tuple
        if 'size' in dataset_config and isinstance(dataset_config['size'], list):
            dataset_config['size'] = tuple(dataset_config['size'])
        
        # Handle verbose/debug from training config if not in dataset config
        if 'verbose' not in dataset_config:
            dataset_config['verbose'] = config['training'].get('enable_debug', False)
        if 'debug' not in dataset_config:
            dataset_config['debug'] = config['training'].get('enable_debug', False)
        
        # NOTE: Don't set device in dataset config - collate_fn keeps tensors on CPU
        # to avoid CUDA re-initialization errors with multiprocessing. Training loop handles GPU.
        
        print(f"Creating training dataset: {dataset_name}")
        dataset = CorrespondenceDataset(dataset_name, **dataset_config)
        return dataset


def create_validation_datasets(config, device=None):
    """Create validation datasets for all benchmarks using CorrespondenceDataset"""
    eval_config = config['evaluation']
    val_datasets = {}
    val_dataloaders = {}
    
    # Set device (GPU by default if available)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get common parameters
    common_size = tuple(config['dataset']['size']) if isinstance(config['dataset']['size'], list) else config['dataset']['size']
    common_downsample_flow = config['dataset']['downsample_flow']
    
    for benchmark in eval_config['eval_benchmarks']:
        print(f"Creating validation dataset: {benchmark}")
        
        # Start with base config - let CorrespondenceDataset handle all the details
        val_dataset_config = {
            'split': 'val',
            'size': common_size,
            'downsample_flow': common_downsample_flow,
            'max_kps': None,  # No truncation for validation
            'verbose': False,
            'debug': False
        }
        
        # Get benchmark-specific overrides from config
        val_datasets_config = eval_config.get('val_datasets', {}).get(benchmark, {})
        # Allow per-benchmark normalization override (falls back to adapter defaults if None)
        # Handle mixed datasets - check if benchmark matches any sub-dataset
        dataset_config = config['dataset']
        is_mixed = dataset_config.get('mixed', False) or 'datasets' in dataset_config
        if is_mixed:
            # For mixed datasets, check if benchmark is in the datasets list
            datasets_list = dataset_config.get('datasets', [])
            train_dataset_matches = benchmark in datasets_list
            # Get normalization from dataset_overrides if benchmark is a training dataset
            if train_dataset_matches and benchmark in dataset_config.get('dataset_overrides', {}):
                val_dataset_config['normalize_images'] = val_datasets_config.get(
                    'normalize_images',
                    dataset_config['dataset_overrides'][benchmark].get('normalize_images', None)
                )
            else:
                val_dataset_config['normalize_images'] = val_datasets_config.get('normalize_images', None)
        else:
            # Single dataset - use existing logic
            train_dataset_name = dataset_config.get('dataset_name', '')
            val_dataset_config['normalize_images'] = val_datasets_config.get(
                'normalize_images',
                None if benchmark != train_dataset_name else dataset_config.get('normalize_images', None)
            )
        
        # Add benchmark-specific parameters based on dataset type
        if benchmark == 'synthetic':
            val_dataset_config['geometry_config_path'] = val_datasets_config.get(
                'geometry_config_path',
                'src/configs/online_synth_configs/OnlineGeometryConfig_Val.yaml'
            )
            val_dataset_config['processor_config_path'] = val_datasets_config.get(
                'processor_config_path',
                config['dataset'].get('processor_config_path', 'src/configs/online_synth_configs/OnlineProcessorConfig.yaml')
            )
            val_dataset_config['opengl_device_index'] = config['dataset'].get('opengl_device_index', None)
            val_dataset_config['geometry_config_overrides'] = config['dataset'].get('geometry_config_overrides', None)
            # If set, serve a pre-rendered synthetic val set from disk (no live
            # renderer): see scripts/precompute_synthetic_val.py + CachedSyntheticAdapter.
            val_dataset_config['synthetic_val_cache'] = val_datasets_config.get('synthetic_val_cache', None)
        
        elif benchmark.startswith('tss'):
            # 'tss' or per-alpha entries 'tss_a10'/'tss_a05'/'tss_a03' (same data,
            # different PCK alpha via eval_alphas zip) all use the TSS adapter.
            val_dataset_config['datapath'] = eval_config['tss_root']
            val_dataset_config['reverse_flow'] = val_datasets_config.get('reverse_flow', False)
            val_dataset_config['thres'] = eval_config['thres']
        
        elif benchmark == 'middlebury':
            val_dataset_config['datapath'] = eval_config['middlebury_root']
            val_dataset_config['reverse_flow'] = val_datasets_config.get('reverse_flow', False)
            val_dataset_config['split'] = 'val'  # Ignored by dataset but kept for API consistency
        
        elif benchmark == 'pointodyssey':
            val_dataset_config['dataset_location'] = eval_config['pointodyssey_root']
            val_dataset_config['pointodyssey_use_augs'] = val_datasets_config.get('pointodyssey_use_augs', False)
            # Support both new (prefixed) and old (unprefixed) parameter names for backward compatibility
            val_dataset_config['pointodyssey_sequence_length'] = val_datasets_config.get('pointodyssey_sequence_length', val_datasets_config.get('sequence_length', 4))
            val_dataset_config['pointodyssey_num_pts_to_track'] = val_datasets_config.get('pointodyssey_num_pts_to_track', val_datasets_config.get('num_pts_to_track', 32))
            val_dataset_config['pointodyssey_strides'] = val_datasets_config.get('pointodyssey_strides', val_datasets_config.get('strides', [4]))
            val_dataset_config['pointodyssey_quick'] = val_datasets_config.get('pointodyssey_quick', val_datasets_config.get('quick', False))
            val_dataset_config['reverse_flow'] = val_datasets_config.get('reverse_flow', True)
            val_dataset_config['thres'] = eval_config['thres']
            val_dataset_config['use_all_valid'] = val_datasets_config.get('use_all_valid', True)
            val_dataset_config['pointodyssey_disable_motion_filter'] = val_datasets_config.get('pointodyssey_disable_motion_filter', val_datasets_config.get('disable_motion_filter', False))
            val_dataset_config['val_sequence_fraction'] = val_datasets_config.get('val_sequence_fraction', None)
            # Note: max_pts is deprecated - use max_kps instead (or set to null for no truncation)
            # For backward compatibility, still check max_pts but prefer max_kps
            max_kps_val = val_datasets_config.get('max_kps', val_datasets_config.get('max_pts', None))
            val_dataset_config['max_kps'] = max_kps_val  # CorrespondenceDataset will handle None correctly
        
        elif benchmark in ['kitti2012', 'kitti2015']:
            val_dataset_config['datapath'] = eval_config['kitti_root']
            # Handle special case for kitti_val_use_full_training
            if eval_config.get('kitti_val_use_full_training', False):
                val_dataset_config['split'] = 'training'  # Special case - not mapped by CorrespondenceDataset
            else:
                val_dataset_config['split'] = val_datasets_config.get('split', 'val')
            val_dataset_config['kitti_occ_type'] = val_datasets_config.get('kitti_occ_type', 'occ')
            val_dataset_config['reverse_flow'] = val_datasets_config.get('reverse_flow', False)
            val_dataset_config['thres'] = eval_config['thres']
            # Handle max_kps if specified in config
            max_pts = val_datasets_config.get('max_pts', None)
            if max_pts is not None:
                val_dataset_config['max_kps'] = max_pts
        
        elif benchmark == 'flyingthings':
            val_dataset_config['datapath'] = eval_config['flyingthings_root']
            val_dataset_config['split'] = val_datasets_config.get('split', 'test')  # CorrespondenceDataset maps 'val' -> 'test' for FlyingThings
            val_dataset_config['reverse_flow'] = val_datasets_config.get('reverse_flow', True)
        
        elif benchmark == 'sintel':
            val_dataset_config['sintel_root'] = eval_config['sintel_root']
            val_dataset_config['split'] = val_datasets_config.get('split', 'train')  # Sintel test split has no ground truth
            val_dataset_config['pass_name'] = val_datasets_config.get('pass_name', 'clean')
            val_dataset_config['reverse_flow'] = val_datasets_config.get('reverse_flow', True)

        elif benchmark.startswith('tapvid_davis'):
            # tap_vid_probe (quarantined): sparse real-motion benchmark, mirrors pointodyssey.
            # benchmark may be 'tapvid_davis' or a per-stride entry 'tapvid_davis_s{N}'
            # (all route to the tapvid_davis adapter; stride comes from this entry's config).
            # `tapvid_davis_root` points at the preprocess_davis.py mmap cache dir.
            val_dataset_config['cache_dir'] = eval_config['tapvid_davis_root']
            val_dataset_config['tapvid_stride'] = val_datasets_config.get('tapvid_stride', val_datasets_config.get('stride', 5))
            val_dataset_config['tapvid_frame_step'] = val_datasets_config.get('tapvid_frame_step', 5)
            val_dataset_config['tapvid_min_pts'] = val_datasets_config.get('tapvid_min_pts', 1)
            val_dataset_config['tapvid_max_pairs_per_video'] = val_datasets_config.get('tapvid_max_pairs_per_video', None)
            val_dataset_config['reverse_flow'] = val_datasets_config.get('reverse_flow', True)
            val_dataset_config['thres'] = eval_config['thres']

        else:  # spair, pfpascal, pfwillow
            val_dataset_config['datapath'] = eval_config['datapath']
            val_dataset_config['thres'] = eval_config['thres']
            val_dataset_config['augmentation'] = False  # No augmentation for validation
            split_to_use = eval_config.get('split_to_use_for_validation', 'val')
            # PFWillow only has 'test' split (no separate val), so use 'test' for consistency
            if benchmark == 'pfwillow':
                val_dataset_config['split'] = 'test'
            else:
                val_dataset_config['split'] = split_to_use
        
        # Create dataset using CorrespondenceDataset - it handles all the parameter mapping.
        # Per-stride tapvid_davis_s{N} entries all map to the 'tapvid_davis' adapter;
        # per-alpha tss_a{NN} entries all map to the 'tss' adapter.
        adapter_name = ('tapvid_davis' if benchmark.startswith('tapvid_davis')
                        else 'tss' if benchmark.startswith('tss')
                        else benchmark)
        dataset = CorrespondenceDataset(adapter_name, **val_dataset_config)
        
        # Subsample FlyingThings dataset if val_dataset_fraction is specified
        original_collate_fn = dataset.collate_fn  # Save collate_fn before wrapping
        if benchmark == 'flyingthings':
            val_dataset_fraction = val_datasets_config.get('val_dataset_fraction', None)
            if val_dataset_fraction is not None and val_dataset_fraction < 1.0:
                from torch.utils.data import Subset
                dataset_size = len(dataset)
                subset_size = int(dataset_size * val_dataset_fraction)
                # Use first N samples for consistent validation
                indices = list(range(subset_size))
                subset_dataset = Subset(dataset, indices)
                # Preserve collate_fn on the Subset wrapper
                subset_dataset.collate_fn = original_collate_fn
                dataset = subset_dataset
                print(f"  Subsampled FlyingThings: {dataset_size} -> {subset_size} samples (fraction: {val_dataset_fraction})")
        
        # Create dataloader
        # Use num_workers=0 for synthetic dataset (GPU-bound rendering, can't use multiprocessing)
        if benchmark == 'synthetic':
            num_workers = 0
        else:
            num_workers = eval_config.get('val_num_workers', 0)
        batch_size = eval_config.get('val_batch_size', 8)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=False,
            prefetch_factor=eval_config.get('prefetch_factor', 2) if num_workers > 0 else None,
            shuffle=False,
            pin_memory=True if num_workers > 0 else False,
            collate_fn=dataset.collate_fn
        )
        
        val_datasets[benchmark] = dataset
        val_dataloaders[benchmark] = dataloader
        print(f"  Val dataloader for benchmark '{benchmark}' size: {len(dataloader)}")
    
    return val_datasets, val_dataloaders


def inspect_datasets(
    config,
    output_dir: str = "debug_collate",
    batch_size: int = 2,
    datasets_to_check=None,
    save_visuals: bool = True,
):
    """
    Run a quick sanity check on datasets: load one batch, print shapes, and save visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    checks = []

    # Training dataset
    train_ds = create_training_dataset(config)
    checks.append(("train", config["dataset"]["dataset_name"], train_ds))

    # Validation datasets
    val_datasets, _ = create_validation_datasets(config)
    for name, ds in val_datasets.items():
        checks.append(("val", name, ds))

    if datasets_to_check:
        wanted = set(datasets_to_check)
        checks = [c for c in checks if c[1] in wanted]

    for split, name, ds in checks:
        loader = DataLoader(
            ds,
            batch_size=min(batch_size, max(1, len(ds))),
            num_workers=0,
            shuffle=False,
            collate_fn=ds.collate_fn,
        )
        batch = next(iter(loader))
        print(f"[INSPECT] {split}:{name} keys -> {list(batch.keys())}")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {list(v.shape)} {v.dtype} {v.device}")

        # Full-res visualization
        if save_visuals and ("flow_full" in batch or "flow" in batch):
            flow = batch.get("flow_full", batch.get("flow"))
            # Subsample hard for speed: regular sampling, fewer arrows, one sample
            vis = CorrespondenceVisualizer(
                sampling_mode="regular",
                arrow_density=8,
            )
            save_overlay = os.path.join(output_dir, f"{split}_{name}_flow_overlay.png")
            vis.visualize_rendered_batch(
                {"src_img": batch["src_img"].cpu(), "trg_img": batch["trg_img"].cpu(), "flow": flow.cpu()},
                save_path=save_overlay,
                max_samples=1,
                visualization_mode="overlay",
                sampling_mode="all_valid" if name in {"spair", "pfpascal", "pfwillow", "pointodyssey"} else "regular",
            )
            print(f"  saved {save_overlay}")

        # Downsampled flow visualization (if available)
        if save_visuals and CATSFlowVisualizer and "flow_downsampled" in batch:
            cats_vis = CATSFlowVisualizer(
                feat_size=batch["flow_downsampled"].shape[-1],
                figsize=(10, 7),
                dpi=120,
                show_patch_boundaries=True,
                normalize_images=False,
            )
            save_ds = os.path.join(output_dir, f"{split}_{name}_flow_downsampled.png")
            cats_vis.visualize_downsampled_flow_batch(
                {
                    "src_img": batch["src_img"].cpu(),
                    "trg_img": batch["trg_img"].cpu(),
                    "flow_downsampled": batch["flow_downsampled"].cpu(),
                },
                save_path=save_ds,
                max_samples=1,
                visualization_mode="overlay",
            )
            print(f"  saved {save_ds}")


def main():
    # Argument parsing - only config path
    parser = argparse.ArgumentParser(description='CATs++ Unified Training Script with Config File')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--inspect-data', action='store_true',
                        help='Run a quick data sanity check with visualizations and exit')
    parser.add_argument('--inspect-output-dir', type=str, default='debug_collate',
                        help='Output directory for data inspection visualizations')
    parser.add_argument('--inspect-visualize', action='store_true',
                        help='When using --inspect-data, actually save visualizations (skip if false)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)

    if args.inspect_data:
        inspect_datasets(config, output_dir=args.inspect_output_dir, save_visuals=args.inspect_visualize)
        return
    
    # Extract config sections
    model_config = config['model']
    training_config = config['training']
    dataset_config = config['dataset']
    eval_config = config['evaluation']
    paths_config = config['paths']
    
    # Set random seeds
    seed = training_config.get('seed', 2021)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Ensure device always has explicit index for proper comparison (handles "cuda" vs "cuda:0")
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Create experiment name from config filename
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    name_exp = time.strftime(f'{config_name}_%Y_%m_%d_%H_%M')
    
    # Initialize multi-benchmark evaluator
    eval_benchmarks_config = dict(zip(eval_config['eval_benchmarks'], eval_config['eval_alphas']))
    multi_evaluator = MultiBenchmarkEvaluator(eval_benchmarks_config)
    print(f"Initialized evaluator for benchmarks: {multi_evaluator.get_available_benchmarks()}")
    
    # Download evaluation datasets (only for standard benchmarks that need downloading)
    standard_benchmarks = ['spair', 'pfpascal', 'pfwillow']
    for benchmark in eval_config['eval_benchmarks']:
        if benchmark in standard_benchmarks:
            download.download_dataset(eval_config['datapath'], benchmark)
    
    # Download training dataset if it's a standard benchmark dataset
    # Handle mixed datasets
    is_mixed = dataset_config.get('mixed', False) or 'datasets' in dataset_config
    if is_mixed:
        # For mixed datasets, download each sub-dataset if needed
        datasets_list = dataset_config.get('datasets', [])
        for ds_name in datasets_list:
            if ds_name in standard_benchmarks:
                download.download_dataset(eval_config['datapath'], ds_name)
        train_dataset_name = '+'.join(datasets_list) if datasets_list else 'mixed'
        has_synthetic = 'synthetic' in datasets_list
    else:
        train_dataset_name = dataset_config.get('dataset_name', 'unknown')
        if train_dataset_name in standard_benchmarks:
            download.download_dataset(eval_config['datapath'], train_dataset_name)
        has_synthetic = train_dataset_name == 'synthetic'
    
    # Create training dataset
    train_dataset = create_training_dataset(config, device=device)
    
    # Create training dataloader
    batch_size = training_config['batch_size']
    n_threads = training_config.get('n_threads', 0)
    
    # Use num_workers=0 for synthetic dataset (GPU-bound rendering)
    train_num_workers = 0 if has_synthetic else n_threads
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=train_num_workers,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        prefetch_factor=batch_size if train_num_workers > 0 else None,
        pin_memory=True if train_num_workers > 0 else False
    )
    
    print(f"Train dataset size: {len(train_dataloader)}")
    
    # Create validation datasets
    val_datasets, val_dataloaders = create_validation_datasets(config, device=device)
    
    # Initialize model
    print("Initializing CATs++ model...")
    if model_config.get('freeze', True):
        print('Backbone frozen!')
    
    model = CATsImproved(backbone=model_config.get('backbone', 'resnet101'), freeze=model_config.get('freeze', True), pretrained_backbone=model_config.get('pretrained_backbone', True))
    
    # Count parameters
    def count_parameters(model):
        return sum(p.numel() for name, p in model.named_parameters() 
                  if p.requires_grad and 'backbone' not in name)
    
    print(f'The number of trainable parameters: {count_parameters(model)}')
    
    # Setup optimizer
    param_model = [param for name, param in model.named_parameters() if 'backbone' not in name]
    param_backbone = [param for name, param in model.named_parameters() if 'backbone' in name]
    
    def _to_float(val, name):
        # Accept scalars, strings, or 1-element sequences from YAML/CLI and coerce to float
        if isinstance(val, (list, tuple)):
            if len(val) == 0:
                raise ValueError(f"{name} cannot be empty")
            val = val[0]
        try:
            return float(val)
        except (TypeError, ValueError):
            raise ValueError(f"Expected {name} to be numeric, got {val!r}")

    lr = _to_float(training_config.get('lr', 3e-4), 'lr')
    lr_backbone = _to_float(training_config.get('lr_backbone', 3e-6), 'lr_backbone')
    weight_decay = _to_float(training_config.get('weight_decay', 0.05), 'weight_decay')
    
    optimizer = optim.AdamW([
        {'params': param_model, 'lr': lr}, 
        {'params': param_backbone, 'lr': lr_backbone}
    ], weight_decay=weight_decay)
    
    # Setup scheduler
    scheduler_type = training_config.get('scheduler', 'step')
    epochs = training_config.get('epochs', 50)
    
    if scheduler_type == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    else:
        step_raw = training_config.get('step', '[70, 80, 90]')
        if isinstance(step_raw, (list, tuple)):
            milestones = [int(s) for s in step_raw]
        else:
            milestones = parse_list(str(step_raw))
        step_gamma = _to_float(training_config.get('step_gamma', 0.5), 'step_gamma')
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=step_gamma
        )
    
    # Load pretrained model if specified
    pretrained_path = paths_config.get('pretrained', None)
    start_epoch = paths_config.get('start_epoch', -1)
    
    if pretrained_path:
        # If pointing to a directory, automatically use model_best.pth
        if os.path.isdir(pretrained_path):
            pretrained_path_full = os.path.join(pretrained_path, 'model_best.pth')
            if not os.path.exists(pretrained_path_full):
                raise FileNotFoundError(f"model_best.pth not found in directory: {pretrained_path}")
            print(f"Loading pretrained model from directory: {pretrained_path}")
            print(f"Using checkpoint: {pretrained_path_full}")
            pretrained_path = pretrained_path_full
        else:
            print(f"Loading pretrained model from: {pretrained_path}")
        
        model, optimizer, scheduler, start_epoch_loaded, best_val = load_checkpoint(
            model, optimizer, scheduler, filename=pretrained_path
        )
        
        # Override start_epoch if loaded from checkpoint
        if start_epoch == -1:
            start_epoch = start_epoch_loaded - 1  # -1 because we'll increment in the loop
        
        # Load additional checkpoint data if available
        if os.path.isfile(pretrained_path):
            checkpoint = torch.load(pretrained_path)
            if 'best_val_per_benchmark' in checkpoint:
                best_val_per_benchmark = checkpoint['best_val_per_benchmark']
                print(f"Loaded best performance tracking: {best_val_per_benchmark}")
            else:
                best_val_per_benchmark = {}
                for benchmark in eval_config['eval_benchmarks']:
                    best_val_per_benchmark[benchmark] = 0.0
            
            if 'best_epoch_per_benchmark' in checkpoint:
                best_epoch_per_benchmark = checkpoint['best_epoch_per_benchmark']
                print(f"Loaded best epoch tracking: {best_epoch_per_benchmark}")
            else:
                best_epoch_per_benchmark = {}
                for benchmark in eval_config['eval_benchmarks']:
                    best_epoch_per_benchmark[benchmark] = 0
            
            if 'best_avg_pck' in checkpoint:
                best_avg_pck = checkpoint['best_avg_pck']
                best_avg_epoch = checkpoint.get('best_avg_epoch', 0)
                print(f"Loaded best average PCK: {best_avg_pck:.2f}% (epoch {best_avg_epoch})")
            else:
                best_avg_pck = 0.0
                best_avg_epoch = 0
        
        # Transfer optimizer states to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
        # For finetuning, create a new snapshot directory to avoid overwriting
        pretrained_name = os.path.basename(os.path.dirname(pretrained_path))
        cur_snapshot = f"{pretrained_name}_finetune_{name_exp}"
        print(f"Finetuning: Creating new snapshot directory: {cur_snapshot}")
    else:
        # Create snapshot directory for training from scratch
        cur_snapshot = name_exp
        print(f"Training from scratch: Using snapshot directory: {cur_snapshot}")
    
    # Create snapshot directory
    snapshots_dir = paths_config.get('snapshots', './snapshots')
    if not os.path.isdir(snapshots_dir):
        os.mkdir(snapshots_dir)
    
    if not osp.isdir(osp.join(snapshots_dir, cur_snapshot)):
        os.makedirs(osp.join(snapshots_dir, cur_snapshot))
    
    save_path = osp.join(snapshots_dir, cur_snapshot)
    
    # Save config file to snapshot directory
    with open(osp.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Save arguments (only if not loading from checkpoint)
    if not pretrained_path:
        with open(osp.join(save_path, 'args.pkl'), 'wb') as f:
            pickle.dump(config, f)
    else:
        # For finetuning, save the finetuning config
        with open(osp.join(save_path, 'finetune_config.yaml'), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        # Also save reference to original pretrained model
        with open(osp.join(save_path, 'pretrained_source.txt'), 'w') as f:
            f.write(f"Finetuned from: {pretrained_path}\n")
            f.write(f"Original model: {pretrained_name}\n")
    
    # Initialize best_val and start_epoch if not loading from checkpoint
    if not pretrained_path:
        best_val = 0
        start_epoch = 0 if start_epoch == -1 else start_epoch
    
    # Initialize best performance tracking for each benchmark (if not loaded from checkpoint)
    if not pretrained_path:
        best_val_per_benchmark = {}
        best_epoch_per_benchmark = {}
        best_avg_pck = 0.0
        best_avg_epoch = 0
        for benchmark in eval_config['eval_benchmarks']:
            best_val_per_benchmark[benchmark] = 0.0
            best_epoch_per_benchmark[benchmark] = 0
        print(f"Initialized best performance tracking for benchmarks: {list(best_val_per_benchmark.keys())}")
    
    # Setup logging
    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    test_writer = SummaryWriter(os.path.join(save_path, 'test'))
    
    def write_training_summary(epoch, is_final=False):
        """Write training summary to text file"""
        summary_file = os.path.join(save_path, 'training_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("TRAINING SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Current epoch: {epoch + 1}\n")
            f.write(f"Training time so far: {time.time() - train_started:.2f} seconds\n")
            f.write(f"Total epochs planned: {epochs}\n")
            f.write(f"Best primary benchmark PCK: {best_val:.4f}%\n")
            f.write(f"Best average PCK: {best_avg_pck:.4f}% (epoch {best_avg_epoch})\n")
            f.write(f"Primary benchmark: {eval_config['eval_benchmarks'][0]}\n\n")
            
            f.write("BEST PERFORMANCE PER BENCHMARK:\n")
            f.write("-" * 50 + "\n")
            for benchmark, best_pck in best_val_per_benchmark.items():
                best_epoch = best_epoch_per_benchmark.get(benchmark, 0)
                checkpoint_file = f"epoch_{best_epoch}.pth" if best_epoch > 0 else "N/A"
                f.write(f"{benchmark:12}: {best_pck:.2f}% PCK (epoch {best_epoch}, {checkpoint_file})\n")
            
            f.write("\nMOTION-AWARE METRICS (from latest epoch):\n")
            f.write("-" * 50 + "\n")
            f.write("Motion-aware PCK and static bias metrics are logged in validation_results.csv\n")
            f.write("Metrics include: PCK (motion-aware), PCK by motion bins, zero-flow precision/recall/F1, static bias ratio\n")
            
            f.write("\nTRAINING CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Train dataset: {train_dataset_name}\n")
            f.write(f"Learning rate: {lr}\n")
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"Feature size: {dataset_config['downsample_flow']}\n")
            f.write(f"Evaluation benchmarks: {', '.join(eval_config['eval_benchmarks'])}\n")
            f.write(f"Evaluation alphas: {', '.join(map(str, eval_config['eval_alphas']))}\n")
            f.write(f"Backbone: {model_config.get('backbone', 'resnet101')}\n")
            f.write(f"Freeze backbone: {model_config.get('freeze', True)}\n")
            f.write(f"Pretrained backbone: {model_config.get('pretrained_backbone', True)}\n")
            f.write(f"Augmentation: {training_config.get('augmentation', False)}\n")
            
            if is_final:
                f.write(f"\nTraining completed in: {time.time() - train_started:.2f} seconds\n")
                f.write("STATUS: Training completed successfully\n")
            else:
                f.write(f"\nSTATUS: Training in progress (epoch {epoch + 1}/{epochs})\n")
        
        if is_final:
            print(f"Final training summary saved to: {summary_file}")
        else:
            print(f"Training summary updated: {summary_file}")
    
    def save_benchmark_model(benchmark, epoch, pck_score, model_state, optimizer_state, scheduler_state, val_results):
        """Save individual benchmark best model"""
        checkpoint_data = {
            'epoch': epoch + 1,
            'state_dict': model_state,
            'optimizer': optimizer_state,
            'scheduler': scheduler_state,
            'best_pck': pck_score,
            'benchmark': benchmark,
            'val_results': val_results,
        }
        filename = f"{benchmark}_best.pth"
        torch.save(checkpoint_data, os.path.join(save_path, filename))
        print(f"Saved best {benchmark} model: {filename} (PCK: {pck_score:.2f}%)")
    
    def save_overall_best_model(epoch, avg_pck, model_state, optimizer_state, scheduler_state, val_results):
        """Save overall best model (best average across benchmarks)"""
        checkpoint_data = {
            'epoch': epoch + 1,
            'state_dict': model_state,
            'optimizer': optimizer_state,
            'scheduler': scheduler_state,
            'best_avg_pck': avg_pck,
            'val_results': val_results,
            'best_val_per_benchmark': best_val_per_benchmark,
            'best_epoch_per_benchmark': best_epoch_per_benchmark,
        }
        filename = "model_best.pth"
        torch.save(checkpoint_data, os.path.join(save_path, filename))
        print(f"Saved overall best model: {filename} (Avg PCK: {avg_pck:.2f}%)")
    
    model = model.to(device)
    
    print("Model initialized successfully!")
    print(f"Starting training from epoch {start_epoch}")
    print(f"Total epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Backbone learning rate: {lr_backbone}")
    
    # Create CSV file for logging validation results vs training steps
    validation_log_file = os.path.join(save_path, 'validation_results.csv')
    validation_log_initialized = False
    print(f"Validation results will be logged to: {validation_log_file}")
    
    def log_validation_results(epoch, cumulative_steps, val_results):
        """Log validation results to CSV file with immediate flushing"""
        nonlocal validation_log_initialized
        
        # Write header if first time
        if not validation_log_initialized:
            with open(validation_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'training_steps', 'benchmark', 'pck', 'loss',
                                'pck_motion_aware', 'pck_motion_small', 'pck_motion_medium', 'pck_motion_large',
                                'zero_flow_precision', 'zero_flow_recall', 'zero_flow_f1', 'static_bias_ratio',
                                'mmd2_pred_corr_vs_pred_miss', 'mmd2_pred_corr_vs_gt', 'mmd2_pred_miss_vs_gt'])
                f.flush()
                os.fsync(f.fileno())
            validation_log_initialized = True
            print(f"Created validation results CSV: {validation_log_file}")
        
        # Append results for each benchmark with immediate flushing
        with open(validation_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for benchmark, results in val_results.items():
                pck_motion_aware = results.get('pck_motion_aware', '')
                motion_binned = results.get('motion_binned', {})
                pck_motion_small = motion_binned.get('small', {}).get('mean_pck', '') if motion_binned else ''
                pck_motion_medium = motion_binned.get('medium', {}).get('mean_pck', '') if motion_binned else ''
                pck_motion_large = motion_binned.get('large', {}).get('mean_pck', '') if motion_binned else ''
                
                zero_flow_metrics = results.get('zero_flow_metrics', {})
                zero_precision = zero_flow_metrics.get('zero_precision', '') if zero_flow_metrics else ''
                zero_recall = zero_flow_metrics.get('zero_recall', '') if zero_flow_metrics else ''
                zero_f1 = zero_flow_metrics.get('zero_f1', '') if zero_flow_metrics else ''
                static_bias = zero_flow_metrics.get('static_bias_ratio', '') if zero_flow_metrics else ''
                
                # Get MMD values
                mmd_pred_corr_vs_pred_miss = results.get('mmd2_pred_corr_vs_pred_miss', '')
                mmd_pred_corr_vs_gt = results.get('mmd2_pred_corr_vs_gt', '')
                mmd_pred_miss_vs_gt = results.get('mmd2_pred_miss_vs_gt', '')
                
                writer.writerow([
                    epoch + 1,
                    cumulative_steps,
                    benchmark,
                    f"{results['pck']:.4f}",
                    f"{results['loss']:.6f}",
                    f"{pck_motion_aware:.4f}" if isinstance(pck_motion_aware, (int, float)) else '',
                    f"{pck_motion_small:.4f}" if isinstance(pck_motion_small, (int, float)) else '',
                    f"{pck_motion_medium:.4f}" if isinstance(pck_motion_medium, (int, float)) else '',
                    f"{pck_motion_large:.4f}" if isinstance(pck_motion_large, (int, float)) else '',
                    f"{zero_precision:.4f}" if isinstance(zero_precision, (int, float)) else '',
                    f"{zero_recall:.4f}" if isinstance(zero_recall, (int, float)) else '',
                    f"{zero_f1:.4f}" if isinstance(zero_f1, (int, float)) else '',
                    f"{static_bias:.4f}" if isinstance(static_bias, (int, float)) else '',
                    f"{mmd_pred_corr_vs_pred_miss:.6f}" if isinstance(mmd_pred_corr_vs_pred_miss, (int, float)) else '',
                    f"{mmd_pred_corr_vs_gt:.6f}" if isinstance(mmd_pred_corr_vs_gt, (int, float)) else '',
                    f"{mmd_pred_miss_vs_gt:.6f}" if isinstance(mmd_pred_miss_vs_gt, (int, float)) else ''
                ])
            f.flush()
            os.fsync(f.fileno())
    
    # ============================================================
    # INITIAL EVALUATION (Before Training)
    # ============================================================
    if training_config.get('eval_initial', False):
        print("Initial evaluation enabled")
        print("\n" + "="*60)
        print("INITIAL EVALUATION (Before Training)")
        print("="*60)
        
        # Get MMD config - for initial eval, calculate MMD if enabled (always calculate for epoch=-1)
        mmd_every_n_epochs = training_config.get('mmd_every_n_epochs', 0)
        if mmd_every_n_epochs > 0:
            print(f"MMD calculation enabled for initial evaluation")
        
        initial_val_results = validate_epoch_multi_benchmark(
            model, val_dataloaders, device, -1, multi_evaluator,
            primary_benchmark=eval_config['eval_benchmarks'][0],
            mmd_every_n_epochs=mmd_every_n_epochs
        )
        
        # Log initial results to TensorBoard (epoch=-1)
        print("\nInitial evaluation results:")
        for benchmark, results in initial_val_results.items():
            print(f"  {benchmark}: PCK={results['pck']:.2f}%, Loss={results['loss']:.4f}")
            test_writer.add_scalar(f'val/{benchmark}/PCK', results['pck'], -1)
            test_writer.add_scalar(f'val/{benchmark}/loss', results['loss'], -1)
            
            # Print MMD results if present
            if 'mmd2_pred_corr_vs_pred_miss' in results:
                mmd_val = results['mmd2_pred_corr_vs_pred_miss']
                if isinstance(mmd_val, (int, float)) and mmd_val == mmd_val:  # Check for NaN
                    print(f"  {benchmark} - MMD^2 (pred_corr vs pred_miss): {mmd_val:.6f}")
            if 'mmd2_pred_corr_vs_gt' in results:
                mmd_val = results['mmd2_pred_corr_vs_gt']
                if isinstance(mmd_val, (int, float)) and mmd_val == mmd_val:
                    print(f"  {benchmark} - MMD^2 (pred_corr vs gt): {mmd_val:.6f}")
            if 'mmd2_pred_miss_vs_gt' in results:
                mmd_val = results['mmd2_pred_miss_vs_gt']
                if isinstance(mmd_val, (int, float)) and mmd_val == mmd_val:
                    print(f"  {benchmark} - MMD^2 (pred_miss vs gt): {mmd_val:.6f}")
            
            # Log motion-aware metrics
            if 'pck_motion_aware' in results:
                test_writer.add_scalar(f'val/{benchmark}/PCK_motion_aware', results['pck_motion_aware'], -1)
            
            if 'motion_binned' in results:
                for bin_name, bin_data in results['motion_binned'].items():
                    if bin_data.get('count', 0) > 0:
                        test_writer.add_scalar(f'val/{benchmark}/PCK_motion_{bin_name}', bin_data['mean_pck'], -1)
            
            if 'zero_flow_metrics' in results:
                zfm = results['zero_flow_metrics']
                test_writer.add_scalar(f'val/{benchmark}/zero_flow_precision', zfm.get('zero_precision', 0), -1)
                test_writer.add_scalar(f'val/{benchmark}/zero_flow_recall', zfm.get('zero_recall', 0), -1)
                test_writer.add_scalar(f'val/{benchmark}/zero_flow_f1', zfm.get('zero_f1', 0), -1)
                test_writer.add_scalar(f'val/{benchmark}/static_bias_ratio', zfm.get('static_bias_ratio', 0), -1)
            
            # Log MMD metrics if present
            if 'mmd2_pred_corr_vs_pred_miss' in results:
                test_writer.add_scalar(f'val/{benchmark}/MMD2_pred_corr_vs_pred_miss', 
                                      results['mmd2_pred_corr_vs_pred_miss'], -1)
            if 'mmd2_pred_corr_vs_gt' in results:
                test_writer.add_scalar(f'val/{benchmark}/MMD2_pred_corr_vs_gt', 
                                      results['mmd2_pred_corr_vs_gt'], -1)
            if 'mmd2_pred_miss_vs_gt' in results:
                test_writer.add_scalar(f'val/{benchmark}/MMD2_pred_miss_vs_gt', 
                                      results['mmd2_pred_miss_vs_gt'], -1)
        
        # Log initial results to CSV (epoch=0, training_steps=0)
        log_validation_results(-1, 0, initial_val_results)
    
        # Calculate initial average PCK
        initial_avg_pck = sum(r['pck'] for r in initial_val_results.values()) / len(initial_val_results)
        test_writer.add_scalar('val/average/PCK', initial_avg_pck, -1)
        print(f"\nInitial average PCK across benchmarks: {initial_avg_pck:.2f}%")
        print("="*60 + "\n")
        
    # Pre-training visualizations (if enabled)
    reference_train_batch = None
    reference_val_batches = {}
    enable_debug = training_config.get('enable_debug', False)
    persist_debug_batches = training_config.get('debug_visualization_persist', False)
    feature_size = dataset_config['downsample_flow']
    
    if enable_debug:
        print("\n" + "="*60)
        print("PRE-TRAINING VISUALIZATIONS")
        print("="*60)
        
        # Limit which benchmarks to visualize to avoid holding large batches for every dataset
        debug_viz_benchmarks = training_config.get('debug_visualization_benchmarks', None)
        if debug_viz_benchmarks is None:
            # Default: only primary eval benchmark to keep memory in check
            debug_viz_benchmarks = [eval_config['eval_benchmarks'][0]]
        elif isinstance(debug_viz_benchmarks, str):
            if debug_viz_benchmarks.lower() == 'all':
                debug_viz_benchmarks = list(val_dataloaders.keys())
            else:
                debug_viz_benchmarks = [debug_viz_benchmarks]
        else:
            debug_viz_benchmarks = list(debug_viz_benchmarks)
        
        # Sample and save reference train batch
        print("Sampling reference train batch...")
        reference_train_batch = next(iter(train_dataloader))
        # Move to CPU to ensure persistence across epochs
        if isinstance(reference_train_batch, dict):
            reference_train_batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in reference_train_batch.items()}
        
        # Visualize training data with ground truth flow
        print("\nVisualizing train GT flow...")
        visualize_batch_flow(
            model=None,
            batch=reference_train_batch,
            device=device,
            train_dataset_name=train_dataset_name,
            val_dataset_name=None,
            split_name='train',
            flow_source='gt',
            feature_size=feature_size,
            epoch=-1
        )
        
        # Visualize training data with predicted flow (untrained model)
        print("\nVisualizing train pred flow (untrained model)...")
        if reference_train_batch is not None:
            visualize_batch_flow(
                model=model,
                batch=reference_train_batch,
                device=device,
                train_dataset_name=train_dataset_name,
                val_dataset_name=None,
                split_name='train',
                flow_source='pred',
                feature_size=feature_size,
                epoch=-1
            )
        else:
            print("Skipping train pred flow visualization because reference_train_batch is None")

        if not persist_debug_batches:
            # Release train batch if we are not persisting for epoch-by-epoch visualizations
            reference_train_batch = None
        
        # Sample and save reference val batches for selected benchmarks
        print("\nSampling reference val batches for selected benchmarks...")
        for benchmark, val_dataloader in val_dataloaders.items():
            if benchmark not in debug_viz_benchmarks:
                print(f"  Skipping {benchmark} visualizations (not in debug_visualization_benchmarks)")
                continue

            print(f"  Sampling batch for {benchmark}...")
            val_batch = next(iter(val_dataloader))
            # Move to CPU to ensure persistence across epochs
            if isinstance(val_batch, dict):
                val_batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in val_batch.items()}
            reference_val_batches[benchmark] = val_batch
            
            # Visualize validation data with ground truth flow
            print(f"\nVisualizing {benchmark} val GT flow...")
            visualize_batch_flow(
                model=None,
                batch=reference_val_batches[benchmark],
                device=device,
                train_dataset_name=train_dataset_name,
                val_dataset_name=benchmark,
                split_name='val',
                flow_source='gt',
                feature_size=feature_size,
                epoch=-1
            )
            
            # Visualize validation data with predicted flow (untrained model)
            print(f"\nVisualizing {benchmark} val pred flow (untrained model)...")
            visualize_batch_flow(
                model=model,
                batch=reference_val_batches[benchmark],
                device=device,
                train_dataset_name=train_dataset_name,
                val_dataset_name=benchmark,
                split_name='val',
                flow_source='pred',
                feature_size=feature_size,
                epoch=-1
            )
            
            if not persist_debug_batches:
                # Do not keep batches in memory unless explicitly requested
                reference_val_batches.pop(benchmark, None)
        
        print("="*60 + "\n")
    
    # Initialize cumulative training steps counter
    cumulative_training_steps = 0
    
    def get_steps_per_epoch(epoch):
        """Calculate steps per epoch based on config setting"""
        steps_per_epoch_config = training_config.get('steps_per_epoch', None)
        if steps_per_epoch_config is None:
            return len(train_dataloader)
        elif steps_per_epoch_config == 'logarithmic':
            # Logarithmic progression: 2^epoch, capped at 2048
            steps = min(2 ** epoch, 2048)
            return steps
        else:
            # Integer value
            return steps_per_epoch_config
    
    # Training loop
    train_started = time.time()
    
    for epoch in range(start_epoch, epochs):
        scheduler.step(epoch)
        
        # Training
        # Calculate steps per epoch for this epoch (may vary if using logarithmic mode)
        steps_per_epoch = get_steps_per_epoch(epoch)
        steps_per_epoch_config = training_config.get('steps_per_epoch', None)
        if steps_per_epoch_config == 'logarithmic':
            print(f"Epoch {epoch + 1}: Using {steps_per_epoch} steps (logarithmic mode)")
        
        # Create flow filter if parameters are provided (only for training)
        flow_filter = None
        min_flow_length = training_config.get('min_flow_length', None)
        max_flow_length = training_config.get('max_flow_length', None)
        if min_flow_length is not None or max_flow_length is not None:
            from src.data.synth.datasets.flow_filter import FlowLengthFilter
            flow_filter = FlowLengthFilter(min_flow_length=min_flow_length, max_flow_length=max_flow_length)
            if epoch == start_epoch:  # Only print once at the start
                print(f"Flow filtering enabled: min={min_flow_length}, max={max_flow_length}")
        
        train_loss = optimize.train_epoch(
            model, optimizer, train_dataloader, device, epoch, train_writer, 
            steps_per_epoch=steps_per_epoch,
            flow_filter=flow_filter
        )
        
        # Update cumulative training steps
        cumulative_training_steps += steps_per_epoch
        
        train_writer.add_scalar('train loss', train_loss, epoch)
        train_writer.add_scalar('learning_rate', scheduler.get_lr()[0], epoch)
        train_writer.add_scalar('learning_rate_backbone', scheduler.get_lr()[1], epoch)
        train_writer.add_scalar('cumulative_training_steps', cumulative_training_steps, epoch)
        print(colored('==> ', 'green') + 'Train average loss:', train_loss)
        print(f"  Cumulative training steps: {cumulative_training_steps}")
        
        # Validation
        mmd_every_n_epochs = training_config.get('mmd_every_n_epochs', 0)
        if mmd_every_n_epochs > 0:
            print(f"MMD calculation enabled: every {mmd_every_n_epochs} epochs (current epoch: {epoch})")
        val_results = validate_epoch_multi_benchmark(
            model, val_dataloaders, device, epoch, multi_evaluator,
            primary_benchmark=eval_config['eval_benchmarks'][0],
            mmd_every_n_epochs=mmd_every_n_epochs
        )
        
        # Log results for each benchmark
        print(colored('==> ', 'blue') + 'epoch :', epoch + 1)
        pck_scores = []
        for benchmark, results in val_results.items():
            print(f"{benchmark} - Val Loss: {results['loss']:.4f}, PCK: {results['pck']:.2f}%")
            test_writer.add_scalar(f'val/{benchmark}/PCK', results['pck'], epoch)
            test_writer.add_scalar(f'val/{benchmark}/loss', results['loss'], epoch)
            
            # Print MMD results if present (they should already be printed from validation, but ensure visibility)
            if 'mmd2_pred_corr_vs_pred_miss' in results:
                mmd_val = results['mmd2_pred_corr_vs_pred_miss']
                if isinstance(mmd_val, (int, float)) and mmd_val == mmd_val:  # Check for NaN
                    print(f"{benchmark} - MMD^2 (pred_corr vs pred_miss): {mmd_val:.6f}")
            if 'mmd2_pred_corr_vs_gt' in results:
                mmd_val = results['mmd2_pred_corr_vs_gt']
                if isinstance(mmd_val, (int, float)) and mmd_val == mmd_val:
                    print(f"{benchmark} - MMD^2 (pred_corr vs gt): {mmd_val:.6f}")
            if 'mmd2_pred_miss_vs_gt' in results:
                mmd_val = results['mmd2_pred_miss_vs_gt']
                if isinstance(mmd_val, (int, float)) and mmd_val == mmd_val:
                    print(f"{benchmark} - MMD^2 (pred_miss vs gt): {mmd_val:.6f}")
            
            # Log motion-aware metrics
            if 'pck_motion_aware' in results:
                test_writer.add_scalar(f'val/{benchmark}/PCK_motion_aware', results['pck_motion_aware'], epoch)
            
            if 'motion_binned' in results:
                for bin_name, bin_data in results['motion_binned'].items():
                    if bin_data.get('count', 0) > 0:
                        test_writer.add_scalar(f'val/{benchmark}/PCK_motion_{bin_name}', bin_data['mean_pck'], epoch)
                        test_writer.add_scalar(f'val/{benchmark}/motion_{bin_name}_count', bin_data['count'], epoch)
            
            if 'zero_flow_metrics' in results:
                zfm = results['zero_flow_metrics']
                test_writer.add_scalar(f'val/{benchmark}/zero_flow_precision', zfm.get('zero_precision', 0), epoch)
                test_writer.add_scalar(f'val/{benchmark}/zero_flow_recall', zfm.get('zero_recall', 0), epoch)
                test_writer.add_scalar(f'val/{benchmark}/zero_flow_f1', zfm.get('zero_f1', 0), epoch)
                test_writer.add_scalar(f'val/{benchmark}/static_bias_ratio', zfm.get('static_bias_ratio', 0), epoch)
            
            # Log MMD metrics if present
            if 'mmd2_pred_corr_vs_pred_miss' in results:
                test_writer.add_scalar(f'val/{benchmark}/MMD2_pred_corr_vs_pred_miss', 
                                      results['mmd2_pred_corr_vs_pred_miss'], epoch)
            if 'mmd2_pred_corr_vs_gt' in results:
                test_writer.add_scalar(f'val/{benchmark}/MMD2_pred_corr_vs_gt', 
                                      results['mmd2_pred_corr_vs_gt'], epoch)
            if 'mmd2_pred_miss_vs_gt' in results:
                test_writer.add_scalar(f'val/{benchmark}/MMD2_pred_miss_vs_gt', 
                                      results['mmd2_pred_miss_vs_gt'], epoch)
            
            # Log per-category results for TSS
            if benchmark == 'tss' and 'pck_by_category' in results:
                for cat, pck in results['pck_by_category'].items():
                    print(f"  {cat}: {pck:.2f}%")
                    test_writer.add_scalar(f'val/{benchmark}/{cat}/PCK', pck, epoch)
            
            pck_scores.append(results['pck'])
            
            # Track best performance for each benchmark and save individual models
            if results['pck'] > best_val_per_benchmark[benchmark]:
                best_val_per_benchmark[benchmark] = results['pck']
                best_epoch_per_benchmark[benchmark] = epoch + 1
                print(f"New best {benchmark} PCK: {results['pck']:.2f}% (epoch {epoch + 1})")
                
                # Save individual benchmark best model
                save_benchmark_model(
                    benchmark, epoch, results['pck'], 
                    model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), 
                    optimizer.state_dict(), 
                    scheduler.state_dict(), val_results
                )
        
        # Calculate average PCK across all benchmarks
        avg_pck = sum(pck_scores) / len(pck_scores)
        test_writer.add_scalar('val/average/PCK', avg_pck, epoch)
        print(f"Average PCK across benchmarks: {avg_pck:.2f}%")
        
        # Log validation results to CSV (vs training steps)
        log_validation_results(epoch, cumulative_training_steps, val_results)
        
        # In-training visualizations (if enabled and batches were persisted)
        if enable_debug and persist_debug_batches and reference_train_batch is not None:
            print("\nGenerating epoch visualizations...")
            
            # Visualize train GT flow (same batch as pre-training)
            visualize_batch_flow(
                model=None,
                batch=reference_train_batch,
                device=device,
                train_dataset_name=train_dataset_name,
                val_dataset_name=None,
                split_name='train',
                flow_source='gt',
                feature_size=feature_size,
                epoch=epoch
            )
            
            # Visualize train pred flow (same batch as pre-training)
            visualize_batch_flow(
                model=model,
                batch=reference_train_batch,
                device=device,
                train_dataset_name=train_dataset_name,
                val_dataset_name=None,
                split_name='train',
                flow_source='pred',
                feature_size=feature_size,
                epoch=epoch
            )
            
            # Visualize val GT and pred flow for each benchmark (same batches as pre-training)
            for benchmark, val_batch in reference_val_batches.items():
                # Visualize val GT flow
                visualize_batch_flow(
                    model=None,
                    batch=val_batch,
                    device=device,
                    train_dataset_name=train_dataset_name,
                    val_dataset_name=benchmark,
                    split_name='val',
                    flow_source='gt',
                    feature_size=feature_size,
                    epoch=epoch
                )
                
                # Visualize val pred flow
                visualize_batch_flow(
                    model=model,
                    batch=val_batch,
                    device=device,
                    train_dataset_name=train_dataset_name,
                    val_dataset_name=benchmark,
                    split_name='val',
                    flow_source='pred',
                    feature_size=feature_size,
                    epoch=epoch
                )
            
            print("Epoch visualizations complete.\n")
        
        # Track best average performance and save overall best model
        if avg_pck > best_avg_pck:
            best_avg_pck = avg_pck
            best_avg_epoch = epoch + 1
            print(f"New best average PCK: {avg_pck:.2f}% (epoch {epoch + 1})")
            
            # Save overall best model
            save_overall_best_model(
                epoch, avg_pck, model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), 
                optimizer.state_dict(), scheduler.state_dict(), val_results
            )
        
        # Use primary benchmark for best_val tracking
        primary_benchmark = eval_config['eval_benchmarks'][0]
        primary_results = val_results[primary_benchmark]
        is_best = primary_results['pck'] > best_val
        best_val = max(primary_results['pck'], best_val)
        
        # Save regular epoch checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_loss': best_val,
            'val_results': val_results,
            'best_val_per_benchmark': best_val_per_benchmark,
            'best_epoch_per_benchmark': best_epoch_per_benchmark,
            'best_avg_pck': best_avg_pck,
            'best_avg_epoch': best_avg_epoch,
        }, is_best=is_best, save_path=save_path, filename='epoch_{}.pth'.format(epoch + 1))
        
        if is_best:
            print(f"New best primary benchmark ({primary_benchmark}) PCK: {best_val:.2f}%")
        
        # Write updated summary after each epoch
        write_training_summary(epoch, is_final=False)
    
    print(f'Training took: {time.time() - train_started:.2f} seconds')
    print(f'Best validation PCK: {best_val:.4f}')
    
    # Print and log best performance for each benchmark
    print("\n" + "="*60)
    print("BEST PERFORMANCE PER BENCHMARK:")
    print("="*60)
    
    # Log final best performances to TensorBoard
    for benchmark, best_pck in best_val_per_benchmark.items():
        best_epoch = best_epoch_per_benchmark.get(benchmark, 0)
        print(f"{benchmark:12}: {best_pck:.2f}% PCK (epoch {best_epoch})")
        test_writer.add_scalar(f'final_best/{benchmark}/PCK', best_pck, 0)
        test_writer.add_scalar(f'final_best/{benchmark}/epoch', best_epoch, 0)
    
    print("-" * 60)
    print(f"{'AVERAGE':12}: {best_avg_pck:.2f}% PCK (epoch {best_avg_epoch})")
    test_writer.add_scalar('final_best/average/PCK', best_avg_pck, 0)
    test_writer.add_scalar('final_best/average/epoch', best_avg_epoch, 0)
    print("="*60)
    
    # Write final summary
    write_training_summary(epochs - 1, is_final=True)
    
    # Close TensorBoard writers
    train_writer.close()
    test_writer.close()


if __name__ == "__main__":
    main()
