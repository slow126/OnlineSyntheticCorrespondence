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
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from train_lightning import create_model, load_config
from train_cats_unified import create_validation_datasets


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
    benchmark_name: str,
    include_gt: bool = False,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """
    Run inference on a benchmark dataset and compute smoothness metrics.
    
    Args:
        model: Model to run inference with
        dataloader: DataLoader for the benchmark
        device: Device to run on
        benchmark_name: Name of benchmark (for logging)
        
    Returns:
        List of smoothness metrics per sample
    """
    model = model.to(device)
    model.eval()
    
    smoothness_metrics = []
    gt_smoothness_metrics = []
    
    print(f"\nRunning inference on {benchmark_name}...", flush=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"  {benchmark_name}", unit="batch", file=sys.stdout)):
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
                # CATs models return dict or tensor - handle both
                if isinstance(pred_flow, dict):
                    # Try to get the finest resolution prediction
                    if 'flow' in pred_flow:
                        pred_flow = pred_flow['flow']
                    elif 'full' in pred_flow:
                        pred_flow = pred_flow['full']
                    else:
                        # Take the last/finest prediction
                        pred_flow = list(pred_flow.values())[-1]
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
            
            def resize_flow(flow, target_h, target_w):
                orig_h, orig_w = flow.shape[-2:]
                if (orig_h, orig_w) == (target_h, target_w):
                    return flow
                flow = F.interpolate(
                    flow,
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                )
                scale_h = target_h / orig_h
                scale_w = target_w / orig_w
                flow = flow * torch.tensor([scale_w, scale_h], device=flow.device).view(1, 2, 1, 1)
                return flow

            # Always ensure prediction matches source image resolution
            if pred_flow.shape[-2:] != src.shape[-2:]:
                pred_flow = resize_flow(pred_flow, src.shape[-2], src.shape[-1])
            
            # Get ground truth flow if available
            if flow_key in batch:
                gt_flow = batch[flow_key].to(device)
                
                # Resize prediction to match ground truth if needed
                if pred_flow.shape[-2:] != gt_flow.shape[-2:]:
                    pred_flow = resize_flow(pred_flow, gt_flow.shape[-2], gt_flow.shape[-1])
                
                # Handle invalid flow (inf values)
                if torch.isinf(gt_flow).any():
                    # Create mask for valid pixels
                    valid_mask = torch.isfinite(gt_flow).all(dim=1, keepdim=True)
                    # Only keep valid regions
                    pred_flow = pred_flow * valid_mask.float()
                    gt_flow = torch.where(torch.isfinite(gt_flow), gt_flow, torch.zeros_like(gt_flow))
            else:
                gt_flow = None
            
            # Compute smoothness immediately (streaming - avoid memory accumulation)
            # Move to CPU and process each sample in batch
            pred_flow_cpu = pred_flow.cpu()
            gt_flow_cpu = gt_flow.cpu() if include_gt and gt_flow is not None else None
            B = pred_flow_cpu.shape[0]

            for b in range(B):
                flow = pred_flow_cpu[b]  # (2, H, W)
                
                # Skip if flow is all zeros or invalid
                if flow.abs().max() < 1e-6:
                    continue
                
                # Compute smoothness for this flow
                tv = calculate_total_variation(flow)
                laplacian = calculate_laplacian_smoothness(flow)
                
                smoothness_metrics.append({'tv': tv, 'laplacian': laplacian})

                if gt_flow_cpu is not None:
                    gt_flow_sample = gt_flow_cpu[b]
                    gt_tv = calculate_total_variation(gt_flow_sample)
                    gt_laplacian = calculate_laplacian_smoothness(gt_flow_sample)
                    gt_smoothness_metrics.append({'tv': gt_tv, 'laplacian': gt_laplacian})
            
            # Free memory immediately
            del pred_flow_cpu, pred_flow
            if gt_flow_cpu is not None:
                del gt_flow_cpu
            if gt_flow is not None:
                del gt_flow
    
    print(f"  ✓ Completed {len(smoothness_metrics)} predictions on {benchmark_name}", flush=True)
    return smoothness_metrics, gt_smoothness_metrics


def calculate_smoothness_metrics(flow_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate statistics from pre-computed smoothness metrics.
    
    Args:
        flow_metrics: List of dicts with 'tv' and 'laplacian' keys
        
    Returns:
        Dictionary with mean TV and Laplacian smoothness
    """
    if len(flow_metrics) == 0:
        return {
            'mean_tv': 0.0,
            'std_tv': 0.0,
            'mean_laplacian': 0.0,
            'std_laplacian': 0.0,
            'num_samples': 0
        }
    
    # Extract values
    all_tv = [m['tv'] for m in flow_metrics]
    all_laplacian = [m['laplacian'] for m in flow_metrics]
    
    return {
        'mean_tv': np.mean(all_tv),
        'std_tv': np.std(all_tv),
        'mean_laplacian': np.mean(all_laplacian),
        'std_laplacian': np.std(all_laplacian),
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
    parser.add_argument(
        '--include-gt',
        action='store_true',
        help='Also compute smoothness on ground-truth flow when available.'
    )
    
    args = parser.parse_args()
    
    # Parse snapshot directories
    snapshot_dirs = []
    if len(args.checkpoints) == 1 and args.checkpoints[0].endswith('.csv'):
        # Load from CSV
        df = pd.read_csv(args.checkpoints[0])
        if 'snapshot_dir' in df.columns:
            snapshot_dirs = df['snapshot_dir'].dropna().unique().tolist()
        else:
            raise ValueError("CSV must have 'snapshot_dir' column")
    else:
        # Assume args are checkpoint paths, extract parent directories
        snapshot_dirs = list(set([str(Path(p).parent) for p in args.checkpoints]))
    
    # Filter out non-existent directories
    valid_snapshot_dirs = [d for d in snapshot_dirs if Path(d).exists()]
    if len(valid_snapshot_dirs) < len(snapshot_dirs):
        print(f"Warning: {len(snapshot_dirs) - len(valid_snapshot_dirs)} snapshot directories not found")
    
    if not valid_snapshot_dirs:
        raise ValueError("No valid snapshot directories found")
    
    print(f"\nFound {len(valid_snapshot_dirs)} snapshot(s) to evaluate", flush=True)
    
    # Create base config for dataloaders
    # Try to load config from first snapshot directory
    base_config_path = Path(valid_snapshot_dirs[0]) / "config.yaml"
    if not base_config_path.exists():
        base_config_path = Path(valid_snapshot_dirs[0]).parent / "config.yaml"
    
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
    
    # Override evaluation settings from command-line args
    base_config['evaluation']['eval_benchmarks'] = args.benchmarks
    base_config['evaluation']['val_batch_size'] = args.batch_size
    base_config['evaluation']['val_num_workers'] = args.num_workers
    
    # Override dataset paths to use local paths (in case config has remote paths)
    base_config['evaluation']['kitti_root'] = '/home/spencer/Data/correspondence/kitti'
    base_config['evaluation']['middlebury_root'] = '/home/spencer/Data/middlebury/all'
    base_config['evaluation']['datapath'] = './models/Datasets_CATs'
    
    # Create validation dataloaders
    device = torch.device(args.device)
    print(f"Creating validation dataloaders for benchmarks: {args.benchmarks}", flush=True)
    val_datasets, val_dataloaders = create_validation_datasets(base_config, device=device)
    print(f"✓ Dataloaders created\n", flush=True)
    
    # Results storage
    all_results = []
    
    # Process each snapshot
    print(f"\n{'='*80}", flush=True)
    print(f"Processing {len(valid_snapshot_dirs)} snapshot(s) x {len(args.benchmarks)} benchmark(s)", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    for idx, checkpoint_path in enumerate(valid_snapshot_dirs, 1):
        snapshot_dir = Path(checkpoint_path)
        print(f"\n[{idx}/{len(valid_snapshot_dirs)}] Processing: {snapshot_dir.name}", flush=True)
        print(f"-" * 80, flush=True)
        
        # Try to get training dataset info from config
        train_dataset = "unknown"
        config_path = snapshot_dir / 'config.yaml'
        if config_path.exists():
            try:
                snapshot_config = load_config(str(config_path))
                dataset_cfg = snapshot_config.get('dataset', {})
                if dataset_cfg.get('mixed', False) or 'datasets' in dataset_cfg:
                    datasets_list = dataset_cfg.get('datasets', [])
                    train_dataset = '+'.join(datasets_list) if datasets_list else 'mixed'
                else:
                    train_dataset = dataset_cfg.get('dataset_name', 'unknown')
            except:
                pass
        
        # Evaluate on each benchmark with benchmark-specific checkpoints
        for benchmark in args.benchmarks:
            if benchmark not in val_dataloaders:
                print(f"  Warning: Benchmark '{benchmark}' not available, skipping", flush=True)
                continue
            
            print(f"\n  Benchmark: {benchmark}", flush=True)
            
            try:
                # Find benchmark-specific checkpoint
                from scripts.run_smoothness_comparison import find_best_checkpoint
                benchmark_checkpoint = find_best_checkpoint(snapshot_dir, benchmark)
                
                if benchmark_checkpoint is None:
                    print(f"    ✗ No valid checkpoint found for {benchmark}", flush=True)
                    continue
                
                benchmark_checkpoint = Path(benchmark_checkpoint)
                
                # Load model
                print(f"    Loading model from checkpoint: {benchmark_checkpoint.name}...", flush=True)
                model = load_model_from_checkpoint(benchmark_checkpoint, args.config)
                model = model.to(device)
                print(f"    ✓ Model loaded successfully", flush=True)
                
                dataloader = val_dataloaders[benchmark]
                
                # Run inference
                smoothness_metrics, gt_smoothness_metrics = run_inference_on_benchmark(
                    model, dataloader, device, benchmark, include_gt=args.include_gt
                )
                
                # Calculate smoothness statistics
                metrics = calculate_smoothness_metrics(smoothness_metrics)
                gt_metrics = calculate_smoothness_metrics(gt_smoothness_metrics)
                
                # Store results
                result = {
                    'checkpoint_path': str(benchmark_checkpoint),
                    'checkpoint_name': snapshot_dir.name,
                    'train_dataset': train_dataset,
                    'benchmark': benchmark,
                    'mean_tv': metrics['mean_tv'],
                    'std_tv': metrics['std_tv'],
                    'mean_laplacian': metrics['mean_laplacian'],
                    'std_laplacian': metrics['std_laplacian'],
                    'num_samples': metrics['num_samples']
                }
                if args.include_gt:
                    result.update({
                        'mean_tv_gt': gt_metrics['mean_tv'],
                        'std_tv_gt': gt_metrics['std_tv'],
                        'mean_laplacian_gt': gt_metrics['mean_laplacian'],
                        'std_laplacian_gt': gt_metrics['std_laplacian'],
                        'num_samples_gt': gt_metrics['num_samples'],
                    })
                all_results.append(result)
                
                print(f"    ✓ {benchmark}: TV={metrics['mean_tv']:.6f}, Laplacian={metrics['mean_laplacian']:.6f}", flush=True)
                
                # Clean up model to free GPU memory
                del model
                torch.cuda.empty_cache()
            
            except Exception as e:
                print(f"\n    ✗ Error processing {benchmark} on {snapshot_dir.name}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                # Clean up even on error
                try:
                    del model
                    torch.cuda.empty_cache()
                except:
                    pass
                continue
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(output_path, index=False)
    print(f"\n{'='*60}", flush=True)
    print(f"Results saved to: {output_path}", flush=True)
    print(f"Total results: {len(all_results)}", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Print summary
    if len(all_results) > 0:
        print("\nSummary:", flush=True)
        print(df_results.groupby(['train_dataset', 'benchmark'])[
            ['mean_tv', 'mean_laplacian']
        ].mean(), flush=True)


if __name__ == '__main__':
    main()
