#!/usr/bin/env python3
"""
Calculate smoothness metrics (Total Variation and Laplacian Smoothness) on predicted flow fields.

This script loads model snapshots, runs inference on benchmarks (especially SPair),
and calculates smoothness metrics to compare SPair-only vs SPair+synthetic models.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from train_lightning import create_model, load_config
from train_cats_unified import create_validation_datasets
from src.data.base import CorrespondenceDataset


def calculate_total_variation(flow: torch.Tensor) -> float:
    """
    Calculate Total Variation (TV) of flow field.
    
    TV measures the sum of absolute differences between neighboring pixels.
    Lower TV = smoother flow field.
    
    Args:
        flow: Flow tensor of shape (B, 2, H, W) or (2, H, W)
        
    Returns:
        Mean TV across batch and spatial dimensions
    """
    if flow.dim() == 3:
        flow = flow.unsqueeze(0)  # Add batch dimension
    
    # Calculate gradients in x and y directions
    # flow is (B, 2, H, W) where channel 0 is x-flow, channel 1 is y-flow
    flow_x = flow[:, 0:1, :, :]  # (B, 1, H, W)
    flow_y = flow[:, 1:2, :, :]  # (B, 1, H, W)
    
    # Compute gradients using finite differences
    # Horizontal gradient (difference in x-direction)
    grad_x_h = torch.abs(flow_x[:, :, :, 1:] - flow_x[:, :, :, :-1])  # (B, 1, H, W-1)
    grad_y_h = torch.abs(flow_y[:, :, :, 1:] - flow_y[:, :, :, :-1])  # (B, 1, H, W-1)
    
    # Vertical gradient (difference in y-direction)
    grad_x_v = torch.abs(flow_x[:, :, 1:, :] - flow_x[:, :, :-1, :])  # (B, 1, H-1, W)
    grad_y_v = torch.abs(flow_y[:, :, 1:, :] - flow_y[:, :, :-1, :])  # (B, 1, H-1, W)
    
    # Total variation is the sum of all gradients
    tv = grad_x_h.sum() + grad_y_h.sum() + grad_x_v.sum() + grad_y_v.sum()
    
    # Normalize by number of pixels
    B, _, H, W = flow.shape
    num_pixels = B * H * W
    tv_normalized = tv.item() / num_pixels
    
    return tv_normalized


def calculate_laplacian_smoothness(flow: torch.Tensor) -> float:
    """
    Calculate Laplacian smoothness of flow field.
    
    Laplacian measures the second-order derivatives (curvature).
    Lower Laplacian = smoother flow field.
    
    Args:
        flow: Flow tensor of shape (B, 2, H, W) or (2, H, W)
        
    Returns:
        Mean Laplacian magnitude across batch and spatial dimensions
    """
    if flow.dim() == 3:
        flow = flow.unsqueeze(0)  # Add batch dimension
    
    # Laplacian kernel (approximation using second-order finite differences)
    # This is a 3x3 kernel that computes the Laplacian
    laplacian_kernel = torch.tensor([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=flow.dtype, device=flow.device).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
    
    # Apply Laplacian to each flow component
    flow_x = flow[:, 0:1, :, :]  # (B, 1, H, W)
    flow_y = flow[:, 1:2, :, :]  # (B, 1, H, W)
    
    # Pad for convolution
    flow_x_padded = F.pad(flow_x, (1, 1, 1, 1), mode='reflect')
    flow_y_padded = F.pad(flow_y, (1, 1, 1, 1), mode='reflect')
    
    # Compute Laplacian
    laplacian_x = F.conv2d(flow_x_padded, laplacian_kernel)  # (B, 1, H, W)
    laplacian_y = F.conv2d(flow_y_padded, laplacian_kernel)  # (B, 1, H, W)
    
    # Compute magnitude of Laplacian
    laplacian_mag = torch.sqrt(laplacian_x**2 + laplacian_y**2)  # (B, 1, H, W)
    
    # Return mean magnitude
    return laplacian_mag.mean().item()


def load_model_from_checkpoint(checkpoint_path: str, config_path: Optional[str] = None) -> torch.nn.Module:
    """
    Load model from checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint .pth file
        config_path: Optional path to config.yaml (for model creation)
        
    Returns:
        Loaded model in eval mode
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Try to find config.yaml in the same directory as checkpoint
    if config_path is None:
        config_path = checkpoint_path.parent / "config.yaml"
        if not config_path.exists():
            # Try parent directory
            config_path = checkpoint_path.parent.parent / "config.yaml"
    
    if config_path and Path(config_path).exists():
        config = load_config(str(config_path))
        model_config = config.get('model', {})
        paths_config = config.get('paths', {})
        model = create_model(model_config, paths_config)
    else:
        # Default model config if no config found
        print(f"Warning: No config.yaml found, using default GLUNet config")
        from src.model.glunet.glunet_lightning import GLUNet
        model = GLUNet()
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'model.' prefix if present (for GLUNet Lightning modules)
    # Also handle DataParallel wrapping (remove 'module.' prefix)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        # Remove DataParallel prefix
        if new_key.startswith('module.'):
            new_key = new_key[7:]
        # Remove GLUNet Lightning module prefix
        if new_key.startswith('model.'):
            new_key = new_key[6:]
        new_state_dict[new_key] = v
    
    # Try loading with strict=False to handle missing keys
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError as e:
        print(f"Warning: Some keys didn't match, loading with strict=False")
        model.load_state_dict(new_state_dict, strict=False)
    
    model.eval()
    
    return model


def run_inference_on_benchmark(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    benchmark_name: str
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Run inference on a benchmark dataset and collect flow predictions.
    
    Args:
        model: Model to run inference with
        dataloader: DataLoader for the benchmark
        device: Device to run on
        benchmark_name: Name of benchmark (for logging)
        
    Returns:
        Tuple of (predicted_flows, ground_truth_flows) lists
    """
    model = model.to(device)
    model.eval()
    
    pred_flows = []
    gt_flows = []
    
    print(f"Running inference on {benchmark_name} ({len(dataloader)} batches)...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Get source and target images
            src_key = 'src_img' if 'src_img' in batch else 'src'
            trg_key = 'trg_img' if 'trg_img' in batch else 'trg'
            flow_key = 'flow'
            
            src = batch[src_key].to(device)
            trg = batch[trg_key].to(device)
            
            # Run model forward pass
            # Handle different model types
            if hasattr(model, 'model'):  # GLUNet wrapper (Lightning module)
                preds = model.model(src, trg)
                # Get full resolution prediction
                if isinstance(preds, dict):
                    if 'full' in preds:
                        pred_flow = preds['full']
                    elif 'level3' in preds:
                        # Upsample to full resolution
                        pred_flow = F.interpolate(
                            preds['level3'],
                            size=src.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    else:
                        # Take last level
                        pred_flow = list(preds.values())[-1]
                        pred_flow = F.interpolate(
                            pred_flow,
                            size=src.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )
                else:
                    pred_flow = preds
            elif hasattr(model, 'hpn_learner'):  # CATs model
                # CATs expects (trg_img, src_img) order
                pred_flow = model(trg, src)
            elif hasattr(model, 'raft'):  # RAFT wrapper
                # RAFT wrapper handles its own normalization
                pred_flow = model(trg, src)
            else:
                # Direct model call (try both orders)
                try:
                    pred_flow = model(src, trg)
                except:
                    pred_flow = model(trg, src)
                
                if isinstance(pred_flow, (list, tuple)):
                    pred_flow = pred_flow[-1]  # Take last prediction
            
            # Get ground truth flow if available
            if flow_key in batch:
                gt_flow = batch[flow_key].to(device)
                # Handle invalid flow (inf values)
                if torch.isinf(gt_flow).any():
                    # Create mask for valid pixels
                    valid_mask = torch.isfinite(gt_flow).all(dim=1, keepdim=True)
                    # Only keep valid regions
                    pred_flow = pred_flow * valid_mask.float()
                    gt_flow = torch.where(torch.isfinite(gt_flow), gt_flow, torch.zeros_like(gt_flow))
            else:
                gt_flow = None
            
            # Store predictions
            pred_flows.append(pred_flow.cpu())
            if gt_flow is not None:
                gt_flows.append(gt_flow.cpu())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    print(f"  Completed inference on {benchmark_name}")
    return pred_flows, gt_flows


def calculate_smoothness_metrics(flows: List[torch.Tensor]) -> Dict[str, float]:
    """
    Calculate smoothness metrics for a list of flow tensors.
    
    Args:
        flows: List of flow tensors, each of shape (B, 2, H, W)
        
    Returns:
        Dictionary with mean TV and Laplacian smoothness
    """
    all_tv = []
    all_laplacian = []
    
    for flow_batch in flows:
        # Process each sample in batch
        B = flow_batch.shape[0]
        for b in range(B):
            flow = flow_batch[b]  # (2, H, W)
            
            # Skip if flow is all zeros or invalid
            if flow.abs().max() < 1e-6:
                continue
            
            tv = calculate_total_variation(flow)
            laplacian = calculate_laplacian_smoothness(flow)
            
            all_tv.append(tv)
            all_laplacian.append(laplacian)
    
    return {
        'mean_tv': np.mean(all_tv) if all_tv else 0.0,
        'std_tv': np.std(all_tv) if all_tv else 0.0,
        'mean_laplacian': np.mean(all_laplacian) if all_laplacian else 0.0,
        'std_laplacian': np.std(all_laplacian) if all_laplacian else 0.0,
        'num_samples': len(all_tv)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate smoothness metrics on flow predictions"
    )
    parser.add_argument(
        '--checkpoints',
        type=str,
        nargs='+',
        required=True,
        help='Paths to checkpoint files (.pth) or CSV file with checkpoint_path column'
    )
    parser.add_argument(
        '--benchmarks',
        type=str,
        nargs='+',
        default=['spair'],
        help='Benchmarks to evaluate on (default: spair)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config.yaml (if not found near checkpoints)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='analysis/flow_smoothness_results.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of dataloader workers'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    # Parse checkpoint paths
    checkpoint_paths = []
    if len(args.checkpoints) == 1 and args.checkpoints[0].endswith('.csv'):
        # Load from CSV
        df = pd.read_csv(args.checkpoints[0])
        if 'checkpoint_path' in df.columns:
            checkpoint_paths = df['checkpoint_path'].dropna().tolist()
        else:
            raise ValueError("CSV must have 'checkpoint_path' column")
    else:
        checkpoint_paths = args.checkpoints
    
    # Filter out non-existent paths
    valid_paths = [p for p in checkpoint_paths if Path(p).exists()]
    if len(valid_paths) < len(checkpoint_paths):
        print(f"Warning: {len(checkpoint_paths) - len(valid_paths)} checkpoint paths not found")
    
    if not valid_paths:
        raise ValueError("No valid checkpoint paths found")
    
    print(f"Found {len(valid_paths)} checkpoint(s) to evaluate")
    
    # Create base config for dataloaders
    # Try to load config from first checkpoint directory
    base_config_path = Path(valid_paths[0]).parent / "config.yaml"
    if not base_config_path.exists():
        base_config_path = Path(valid_paths[0]).parent.parent / "config.yaml"
    
    if base_config_path.exists():
        base_config = load_config(str(base_config_path))
    else:
        # Create minimal config
        print("Warning: No config found, using default config")
        base_config = {
            'dataset': {
                'size': (256, 256),
                'downsample_flow': False
            },
            'evaluation': {
                'eval_benchmarks': args.benchmarks,
                'eval_alphas': [0.05] * len(args.benchmarks),
                'thres': 'img',
                'split_to_use_for_validation': 'test',
                'val_batch_size': args.batch_size,
                'val_num_workers': args.num_workers,
                'datapath': './models/Datasets_CATs',
                'tss_root': '/home/spencer/Data/correspondence/TSS_CVPR2016',
                'kitti_root': '/home/spencer/Data/correspondence/kitti',
                'pointodyssey_root': '/home/spencer/Data/PointOdyssey',
                'flyingthings_root': '/home/spencer/Data/FlyingThings3D_tiny',
                'middlebury_root': '/home/spencer/Data/middlebury/all',
            }
        }
    
    # Override benchmarks in config
    base_config['evaluation']['eval_benchmarks'] = args.benchmarks
    
    # Create validation dataloaders
    device = torch.device(args.device)
    print(f"Creating validation dataloaders for benchmarks: {args.benchmarks}")
    val_datasets, val_dataloaders = create_validation_datasets(base_config, device=device)
    
    # Results storage
    all_results = []
    
    # Process each checkpoint
    for checkpoint_path in valid_paths:
        print(f"\n{'='*60}")
        print(f"Processing checkpoint: {checkpoint_path}")
        print(f"{'='*60}")
        
        try:
            # Load model
            model = load_model_from_checkpoint(checkpoint_path, args.config)
            model = model.to(device)
            
            # Extract checkpoint info
            checkpoint_dir = Path(checkpoint_path).parent
            checkpoint_name = checkpoint_dir.name
            
            # Try to get training dataset info from config or directory name
            train_dataset = "unknown"
            if base_config_path.exists():
                dataset_cfg = base_config.get('dataset', {})
                if dataset_cfg.get('mixed', False) or 'datasets' in dataset_cfg:
                    datasets_list = dataset_cfg.get('datasets', [])
                    train_dataset = '+'.join(datasets_list) if datasets_list else 'mixed'
                else:
                    train_dataset = dataset_cfg.get('dataset_name', 'unknown')
            
            # Evaluate on each benchmark
            for benchmark in args.benchmarks:
                if benchmark not in val_dataloaders:
                    print(f"Warning: Benchmark '{benchmark}' not available, skipping")
                    continue
                
                dataloader = val_dataloaders[benchmark]
                
                # Run inference
                pred_flows, gt_flows = run_inference_on_benchmark(
                    model, dataloader, device, benchmark
                )
                
                # Calculate smoothness metrics
                print(f"Calculating smoothness metrics for {benchmark}...")
                metrics = calculate_smoothness_metrics(pred_flows)
                
                # Store results
                result = {
                    'checkpoint_path': str(checkpoint_path),
                    'checkpoint_name': checkpoint_name,
                    'train_dataset': train_dataset,
                    'benchmark': benchmark,
                    'mean_tv': metrics['mean_tv'],
                    'std_tv': metrics['std_tv'],
                    'mean_laplacian': metrics['mean_laplacian'],
                    'std_laplacian': metrics['std_laplacian'],
                    'num_samples': metrics['num_samples']
                }
                all_results.append(result)
                
                print(f"  {benchmark} - Mean TV: {metrics['mean_tv']:.6f}, "
                      f"Mean Laplacian: {metrics['mean_laplacian']:.6f}")
        
        except Exception as e:
            print(f"Error processing {checkpoint_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"Total results: {len(all_results)}")
    print(f"{'='*60}")
    
    # Print summary
    if len(all_results) > 0:
        print("\nSummary:")
        print(df_results.groupby(['train_dataset', 'benchmark'])[
            ['mean_tv', 'mean_laplacian']
        ].mean())


if __name__ == '__main__':
    main()

