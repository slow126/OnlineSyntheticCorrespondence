"""
Training script for CATs++ model using synthetic correspondence dataset.
This script initializes the CATs++ model and sets up training with the online synthetic dataset.
"""

import argparse
import csv
import os
import pickle
import random
import time
from os import path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from termcolor import colored
from torch.utils.data import DataLoader

# Import CATs++ model and utilities
import sys
from models.CATs_PlusPlus.models.cats_improved import CATsImproved
import models.CATs_PlusPlus.utils_training.optimize as optimize
from models.CATs_PlusPlus.utils_training.utils import parse_list, load_checkpoint, save_checkpoint, boolean_string
from src.data.synth.datasets.OnlineCorrespondenceDataset import OnlineCorrespondenceDataset
import models.CATs_PlusPlus.data.download as download
from models.CATs_PlusPlus.utils_training.eval_instance import MultiBenchmarkEvaluator
from models.CATs_PlusPlus.utils_training.optimize_multi import validate_epoch_multi_benchmark
from src.data.synth.datasets.FlyingThingsDataset import FlyingThingsDataset
from src.data.synth.datasets.PointOdysseyCorrespondence import PointOdysseyFlowDataset

# Import our synthetic dataset wrapper
import torchvision
from pathlib import Path


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
                if model is None:
                    print(f"Warning: model is None and 'pred_flow' not in batch. Skipping visualization.")
                    return
                print(f"Running model forward pass to get predictions...")
                model.eval()
                with torch.no_grad():
                    pred_flow = model(
                        batch['trg_img'].to(device),
                        batch['src_img'].to(device)
                    )
                flow_tensor = pred_flow[0].cpu()
            flow_key = 'pred_flow'
        else:  # flow_source == 'gt'
            flow_tensor = batch['flow'][0].cpu()
            flow_key = 'gt_flow'
        
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
                # Use pred_flow if available, otherwise use batch['flow']
                if flow_source == 'pred':
                    flow_to_visualize = pred_flow
                else:
                    flow_to_visualize = batch['flow']
                    
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
                    batch_vis = batch
                
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


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='CATs++ Training Script with Synthetic Data')
    
    # Paths
    parser.add_argument('--name_exp', type=str,
                        default=time.strftime('%Y_%m_%d_%H_%M'),
                        help='name of the experiment to save')
    parser.add_argument('--snapshots', type=str, default='./snapshots')
    parser.add_argument('--pretrained', dest='pretrained', default=None,
                       help='path to pre-trained model')
    parser.add_argument('--start_epoch', type=int, default=-1,
                        help='start epoch')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='training batch size')
    parser.add_argument('--n_threads', type=int, default=0,
                        help='number of parallel threads for dataloaders (0 recommended for OpenGL compatibility)')
    parser.add_argument('--seed', type=int, default=2021,
                        help='Pseudo-RNG seed')
    parser.add_argument('--backbone', type=str, default='resnet101')
    parser.add_argument('--feature_size', type=lambda x: None if str(x).lower() == 'none' else int(x), default=32,
                        help='feature size for downsampled flow. Size of patches that are averaged together for flow vectors. [default: 32]')
    parser.add_argument('--size', type=int, default=512,
                        help='size of the images. [default: 512]')
    
    # Synthetic dataset parameters
    parser.add_argument('--train_dataset', type=str, default='synthetic', choices=['synthetic', 'spair', 'pfpascal', 'pfwillow', 'caltech', 'flyingthings', 'pointodyssey', 'kitti2012', 'kitti2015'])
    parser.add_argument('--config', type=str, default='src/configs/online_synth_configs/OnlineDatasetConfig.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--geometry_config_path', type=str, 
                        default='src/configs/online_synth_configs/OnlineGeometryConfig.yaml',
                        help='Path to geometry config YAML file')
    parser.add_argument('--processor_config_path', type=str,
                        default='src/configs/online_synth_configs/OnlineProcessorConfig.yaml',
                        help='Path to processor config YAML file')
    
    # Sampler override arguments (optional - override config values)
    parser.add_argument('--angle_sampler_x_loc', type=float, default=None,
                        help='Override angle_sampler.x_components.loc')
    parser.add_argument('--angle_sampler_x_scale', type=float, default=None,
                        help='Override angle_sampler.x_components.scale')
    parser.add_argument('--angle_sampler_x_distribution', type=str, default=None,
                        help='Override angle_sampler.x_components.distribution')
    parser.add_argument('--angle_sampler_y_loc', type=float, default=None,
                        help='Override angle_sampler.y_components.loc')
    parser.add_argument('--angle_sampler_y_scale', type=float, default=None,
                        help='Override angle_sampler.y_components.scale')
    parser.add_argument('--angle_sampler_y_distribution', type=str, default=None,
                        help='Override angle_sampler.y_components.distribution')
    
    parser.add_argument('--scale_sampler_abs_loc', type=float, default=None,
                        help='Override scale_sampler.abs_components.loc')
    parser.add_argument('--scale_sampler_abs_scale', type=float, default=None,
                        help='Override scale_sampler.abs_components.scale')
    parser.add_argument('--scale_sampler_abs_distribution', type=str, default=None,
                        help='Override scale_sampler.abs_components.distribution')
    parser.add_argument('--scale_sampler_rel_loc', type=float, default=None,
                        help='Override scale_sampler.rel_components.loc')
    parser.add_argument('--scale_sampler_rel_scale', type=float, default=None,
                        help='Override scale_sampler.rel_components.scale')
    parser.add_argument('--scale_sampler_rel_distribution', type=str, default=None,
                        help='Override scale_sampler.rel_components.distribution')
                    
    # FlyingThings dataset parameters
    parser.add_argument('--flyingthings_root', type=str, default='/home/spencer/Data/FlyingThings3D_tiny/',
                        help='root directory of the FlyingThings3D dataset')
    parser.add_argument('--subsample_flow', type=lambda x: None if str(x).lower() == 'none' else float(x), default=0.01,
                        help='subsample ratio for flow vectors (fraction to keep, e.g., 0.1 for 10%%). Set to "none" to disable. [default: 0.01]')
    parser.add_argument('--subsample_flow_seed', type=lambda x: None if str(x).lower() == 'none' else int(x), default=None,
                        help='random seed for flow subsampling (for reproducibility). Set to "none" for random. [default: None]')
    
    # KITTI dataset parameters
    parser.add_argument('--kitti_root', type=str, default='/home/spencer/Data/correspondence/kitti-split',
                        help='root directory containing kitti-2012 and kitti-2015 folders')


    # PointOdyssey dataset parameters
    parser.add_argument('--pointodyssey_root', type=str, default='/home/spencer/Data/PointOdyssey',
                        help='root directory of the PointOdyssey dataset')
    parser.add_argument('--verbose_pointodyssey', type=boolean_string, nargs='?', const=True, default=False,
                        help='verbose mode')
    parser.add_argument('--all_points_pointodyssey', type=boolean_string, nargs='?', const=True, default=False,
                        help='use all points in the PointOdyssey dataset')
    parser.add_argument('--num_pts_to_track_pointodyssey', type=int, default=32,
                        help='number of points to track in the PointOdyssey dataset')
    parser.add_argument('--strides_pointodyssey', type=int, nargs='+', default=[4],
                        help='strides for the PointOdyssey dataset')
    parser.add_argument('--sequence_length_pointodyssey', type=int, default=4,
                        help='sequence length for the PointOdyssey dataset')
    parser.add_argument('--pointodyssey_val_max_sequences', type=int, default=None,
                        help='Maximum number of sequences to use for PointOdyssey validation (None = all, deterministic sampling)')
    parser.add_argument('--pointodyssey_max_pts', type=int, default=200,
                        help='Maximum number of keypoints for PointOdyssey training (default: 200, use as many as possible)')

    # Flow filtering parameters (applied during training only)
    parser.add_argument('--min_flow_length', type=lambda x: None if str(x).lower() == 'none' else float(x), default=None,
                        help='Minimum flow vector length for flow filtering during training. Set to "none" to disable.')
    parser.add_argument('--max_flow_length', type=lambda x: None if str(x).lower() == 'none' else float(x), default=None,
                        help='Maximum flow vector length for flow filtering during training. Set to "none" to disable.')

    # Training parameters
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 3e-4)')
    parser.add_argument('--lr-backbone', type=float, default=3e-6, metavar='LR',
                        help='learning rate for backbone (default: 3e-6)')
    parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine'])
    parser.add_argument('--step', type=str, default='[70, 80, 90]')
    parser.add_argument('--step_gamma', type=float, default=0.5)
    parser.add_argument('--freeze', type=boolean_string, nargs='?', const=True, default=True)
    parser.add_argument('--augmentation', type=boolean_string, nargs='?', const=True, default=True)
    parser.add_argument('--steps_per_epoch', type=lambda x: None if str(x).lower() == 'none' else ('logarithmic' if str(x).lower() == 'logarithmic' else int(x)), default=None,
                        help='number of steps per epoch. Can be an integer, "none" (all steps), or "logarithmic" (base 2 progression: 1, 2, 4, 8, ..., 1024). [default: None, meaning all steps in the dataset]')
    parser.add_argument('--enable_debug', type=boolean_string, nargs='?', const=True, default=False,
                        help='Enable debug visualizations and PointOdyssey verbose mode. [default: False]')
    
    # Evaluation parameters
    parser.add_argument('--eval_benchmarks', type=str, nargs='+', default=['spair'],
                        choices=['synthetic', 'spair', 'pfpascal', 'pfwillow', 'caltech', 'tss', 'pointodyssey', 'kitti2012', 'kitti2015'],
                        help='list of benchmarks for evaluation during training')
    parser.add_argument('--eval_alphas', type=float, nargs='+', default=[0.1],
                        help='list of alpha values for each evaluation benchmark (must match eval_benchmarks length)')
    parser.add_argument('--thres', type=str, default='img', choices=['auto', 'img', 'bbox', 'bbox-kp'])
    parser.add_argument('--datapath', type=str, default='./models/Datasets_CATs')
    parser.add_argument('--split_to_use_for_validation', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--val_batch_size', type=int, default=8,
                        help='batch size for validation. [default: 8]')
    parser.add_argument('--val_num_workers', type=int, default=16,
                        help='number of workers for validation. [default: 16]')
    parser.add_argument('--tss_root', type=str, default='/home/spencer/Data/correspondence/TSS_CVPR2016',
                        help='root directory of the TSS dataset')



    
    args = parser.parse_args()
    
    # Validate multi-benchmark arguments
    if len(args.eval_benchmarks) == 0:
        raise ValueError("At least one evaluation benchmark must be specified via --eval_benchmarks")
    if len(args.eval_benchmarks) != len(args.eval_alphas):
        raise ValueError(f"Number of eval_benchmarks ({len(args.eval_benchmarks)}) must match number of eval_alphas ({len(args.eval_alphas)})")
    
    # Create benchmark-alpha mapping
    eval_benchmarks_config = dict(zip(args.eval_benchmarks, args.eval_alphas))
    print(f"Multi-benchmark evaluation config: {eval_benchmarks_config}")
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Initialize multi-benchmark evaluator
    multi_evaluator = MultiBenchmarkEvaluator(eval_benchmarks_config)
    print(f"Initialized evaluator for benchmarks: {multi_evaluator.get_available_benchmarks()}")
    
    # Download evaluation datasets (only for standard benchmarks that need downloading)
    standard_benchmarks = ['spair', 'pfpascal', 'pfwillow', 'caltech']
    for benchmark in args.eval_benchmarks:
        if benchmark in standard_benchmarks:
            download.download_dataset(args.datapath, benchmark)
    
    # Download training dataset if it's a standard benchmark dataset
    if args.train_dataset in ['spair', 'pfpascal', 'pfwillow', 'caltech']:
        download.download_dataset(args.datapath, args.train_dataset)
    
    # Build geometry config overrides from command-line args (for synthetic dataset)
    geometry_config_overrides = {}
    if args.train_dataset == 'synthetic':
        # Angle sampler overrides
        if args.angle_sampler_x_loc is not None or args.angle_sampler_x_scale is not None or args.angle_sampler_x_distribution is not None:
            geometry_config_overrides.setdefault('angle_sampler', {}).setdefault('x_components', {})
            if args.angle_sampler_x_loc is not None:
                geometry_config_overrides['angle_sampler']['x_components']['loc'] = args.angle_sampler_x_loc
            if args.angle_sampler_x_scale is not None:
                geometry_config_overrides['angle_sampler']['x_components']['scale'] = args.angle_sampler_x_scale
            if args.angle_sampler_x_distribution is not None:
                geometry_config_overrides['angle_sampler']['x_components']['distribution'] = args.angle_sampler_x_distribution
        
        if args.angle_sampler_y_loc is not None or args.angle_sampler_y_scale is not None or args.angle_sampler_y_distribution is not None:
            geometry_config_overrides.setdefault('angle_sampler', {}).setdefault('y_components', {})
            if args.angle_sampler_y_loc is not None:
                geometry_config_overrides['angle_sampler']['y_components']['loc'] = args.angle_sampler_y_loc
            if args.angle_sampler_y_scale is not None:
                geometry_config_overrides['angle_sampler']['y_components']['scale'] = args.angle_sampler_y_scale
            if args.angle_sampler_y_distribution is not None:
                geometry_config_overrides['angle_sampler']['y_components']['distribution'] = args.angle_sampler_y_distribution
        
        # Scale sampler overrides
        if args.scale_sampler_abs_loc is not None or args.scale_sampler_abs_scale is not None or args.scale_sampler_abs_distribution is not None:
            geometry_config_overrides.setdefault('scale_sampler', {}).setdefault('abs_components', {})
            if args.scale_sampler_abs_loc is not None:
                geometry_config_overrides['scale_sampler']['abs_components']['loc'] = args.scale_sampler_abs_loc
            if args.scale_sampler_abs_scale is not None:
                geometry_config_overrides['scale_sampler']['abs_components']['scale'] = args.scale_sampler_abs_scale
            if args.scale_sampler_abs_distribution is not None:
                geometry_config_overrides['scale_sampler']['abs_components']['distribution'] = args.scale_sampler_abs_distribution
        
        if args.scale_sampler_rel_loc is not None or args.scale_sampler_rel_scale is not None or args.scale_sampler_rel_distribution is not None:
            geometry_config_overrides.setdefault('scale_sampler', {}).setdefault('rel_components', {})
            if args.scale_sampler_rel_loc is not None:
                geometry_config_overrides['scale_sampler']['rel_components']['loc'] = args.scale_sampler_rel_loc
            if args.scale_sampler_rel_scale is not None:
                geometry_config_overrides['scale_sampler']['rel_components']['scale'] = args.scale_sampler_rel_scale
            if args.scale_sampler_rel_distribution is not None:
                geometry_config_overrides['scale_sampler']['rel_components']['distribution'] = args.scale_sampler_rel_distribution
        
        # Use None if no overrides provided
        if not geometry_config_overrides:
            geometry_config_overrides = None
    
    # Create training dataset
    if args.train_dataset == 'synthetic':
        print("Creating synthetic dataset...")
        train_dataset = OnlineCorrespondenceDataset(
            geometry_config_path=args.geometry_config_path,
            processor_config_path=args.processor_config_path,
            split='train',
            opengl_device_index=None,  # Auto-detect from torch.cuda.current_device() (works with Lightning DDP)
            geometry_config_overrides=geometry_config_overrides
        )
        
        # Use num_workers=0 for synthetic dataset (GPU-bound rendering, multiprocessing adds overhead)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            num_workers=0,  # Single process per DataLoader (multi-GPU handled by Lightning DDP)
            shuffle=True, 
            collate_fn=train_dataset.collate_fn
        )
    elif args.train_dataset == 'flyingthings':
        train_dataset = FlyingThingsDataset(root=args.flyingthings_root, split="train", transforms=None, size=(args.size, args.size), downsample_flow=args.feature_size, 
                                            subsample_flow=args.subsample_flow, subsample_flow_seed=args.subsample_flow_seed, use_valid_mask=True, reverse_flow=True, filter_out_of_bounds=True)
        # Note: Dataset returns CPU tensors - DataLoader handles GPU transfer
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_threads, shuffle=True, prefetch_factor=args.batch_size if args.n_threads > 0 else None, pin_memory=True)
    elif args.train_dataset in ['kitti2012', 'kitti2015']:
        from src.data.synth.datasets.KittiDataset import KittiDataset
        version = '2012' if '2012' in args.train_dataset else '2015'
        train_dataset = KittiDataset(
            root=os.path.join(args.kitti_root, f'kitti-{version}'),
            split='train',
            version=version,
            occ_type='occ',
            size=(args.size, args.size),
            downsample_flow=args.feature_size,
            normalize=True,
            normalize_images=False
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.n_threads,
            shuffle=True,
            prefetch_factor=args.batch_size if args.n_threads > 0 else None,
            pin_memory=True
        )
    elif args.train_dataset == 'pointodyssey':
        train_dataset = PointOdysseyFlowDataset(
            dataset_location=args.pointodyssey_root, 
            dset='train', 
            use_augs=False, 
            S=args.sequence_length_pointodyssey, 
            N=args.num_pts_to_track_pointodyssey, 
            strides=args.strides_pointodyssey, 
            quick=False, 
            verbose=args.enable_debug and args.verbose_pointodyssey, 
            resize_size=(args.size+64, args.size+64), 
            crop_size=(args.size, args.size), 
            filter_instances=True, 
            downsample_for_cats=True, 
            cats_feat_size=args.feature_size, 
            all_points=True,
            max_pts=args.pointodyssey_max_pts  # Use more keypoints for training
        )
        # Note: Dataset returns CPU tensors - DataLoader handles GPU transfer
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_threads, shuffle=True, prefetch_factor=args.batch_size if args.n_threads > 0 else None, pin_memory=True)
    elif args.train_dataset in ['spair', 'pfpascal', 'pfwillow', 'caltech']:
        # Load standard benchmark dataset for training
        train_dataset = download.load_dataset(args.train_dataset, args.datapath, args.thres, device, 'trn', args.augmentation, args.feature_size)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.n_threads,
            persistent_workers=args.n_threads > 0,
            prefetch_factor=args.batch_size if args.n_threads > 0 else None,
            shuffle=True,
            pin_memory=True
        )
    else:
        raise ValueError(f"Unknown train_dataset: {args.train_dataset}. Must be one of: synthetic, flyingthings, pointodyssey, spair, pfpascal, pfwillow, caltech, kitti2012, kitti2015")

    print(f"Train dataset size: {len(train_dataloader)}")

    # Setup validation dataloaders for all benchmarks
    val_loaders = {}
    val_dataloaders = {}
    
    for benchmark in args.eval_benchmarks:
        if benchmark == 'synthetic':
            # Use val geometry config for validation, but apply same overrides
            val_geometry_config_path = 'src/configs/online_synth_configs/OnlineGeometryConfig_Val.yaml'
            val_dataset = OnlineCorrespondenceDataset(
                geometry_config_path=val_geometry_config_path,
                processor_config_path=args.processor_config_path,
                split='val',
                geometry_config_overrides=geometry_config_overrides
            )
            val_dataset.cuda()
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.n_threads, shuffle=False, collate_fn=val_dataset.collate_fn)
        elif benchmark == 'tss':
            from src.data.synth.datasets.TSSDataset import TSSDataset
            val_dataset = TSSDataset(
                root=args.tss_root, 
                device=device,
                size=args.size,
                feature_size=args.feature_size,
                thres=args.thres
            )
            val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.val_num_workers, persistent_workers=True, prefetch_factor=8, shuffle=False)
        elif benchmark == 'pointodyssey':
            val_dataset = PointOdysseyFlowDataset(
                dataset_location=args.pointodyssey_root,
                dset='val',
                use_augs=False,
                S=args.sequence_length_pointodyssey,
                N=args.num_pts_to_track_pointodyssey,
                quick=False,
                max_sequences=args.pointodyssey_val_max_sequences,
                verbose=args.enable_debug and args.verbose_pointodyssey,
                resize_size=(args.size+64, args.size+64),
                crop_size=(args.size, args.size),
                filter_instances=True,
                downsample_for_cats=True,  
                cats_feat_size=args.feature_size,
                all_points=False,
                max_pts=40,
                thres=args.thres,
                normalize_images=True, 
            )
            val_dataloader = DataLoader(
                val_dataset, 
                batch_size=args.val_batch_size, 
                num_workers=args.val_num_workers, 
                persistent_workers=True, 
                prefetch_factor=8, 
                shuffle=False, 
                pin_memory=True
            )
        elif benchmark in ['kitti2012', 'kitti2015']:
            from src.data.synth.datasets.KittiDataset import KittiDataset
            version = '2012' if '2012' in benchmark else '2015'
            val_dataset = KittiDataset(
                root=os.path.join(args.kitti_root, f'kitti-{version}'),
                split='val',
                version=version,
                occ_type='occ',
                size=(args.size, args.size),
                downsample_flow=args.feature_size,
                normalize=True,
                normalize_images=True,  # Enable keypoint format
                thres=args.thres,
                max_pts=200
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=args.val_batch_size,
                num_workers=args.val_num_workers,
                persistent_workers=True,
                prefetch_factor=8,
                shuffle=False,
                pin_memory=True
            )
        else:
            val_dataset = download.load_dataset(benchmark, args.datapath, args.thres, device, args.split_to_use_for_validation, False, args.feature_size)
            val_dataloader = DataLoader(val_dataset,
                batch_size=args.val_batch_size,
                num_workers=args.val_num_workers,
                persistent_workers=True,
                prefetch_factor=8,
                shuffle=False,
                pin_memory=True)
        
        val_loaders[benchmark] = val_dataset
        val_dataloaders[benchmark] = val_dataloader
        print(f"Val dataloader for benchmark '{benchmark}' size: {len(val_dataloader)}")
    

    # Initialize model
    print("Initializing CATs++ model...")
    if args.freeze:
        print('Backbone frozen!')
    
    model = CATsImproved(backbone=args.backbone, freeze=args.freeze)
    
    # Count parameters
    def count_parameters(model):
        return sum(p.numel() for name, p in model.named_parameters() 
                  if p.requires_grad and 'backbone' not in name)
    
    print(f'The number of trainable parameters: {count_parameters(model)}')
    
    # Setup optimizer
    param_model = [param for name, param in model.named_parameters() if 'backbone' not in name]
    param_backbone = [param for name, param in model.named_parameters() if 'backbone' in name]
    
    optimizer = optim.AdamW([
        {'params': param_model, 'lr': args.lr}, 
        {'params': param_backbone, 'lr': args.lr_backbone}
    ], weight_decay=args.weight_decay)
    
    # Setup scheduler
    if args.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    else:
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=parse_list(args.step), gamma=args.step_gamma
        )
    
    # Load pretrained model if specified
    if args.pretrained:
        # If pointing to a directory, automatically use model_best.pth
        if os.path.isdir(args.pretrained):
            pretrained_path = os.path.join(args.pretrained, 'model_best.pth')
            if not os.path.exists(pretrained_path):
                raise FileNotFoundError(f"model_best.pth not found in directory: {args.pretrained}")
            print(f"Loading pretrained model from directory: {args.pretrained}")
            print(f"Using checkpoint: {pretrained_path}")
        else:
            pretrained_path = args.pretrained
            print(f"Loading pretrained model from: {pretrained_path}")
        
        model, optimizer, scheduler, start_epoch, best_val = load_checkpoint(
            model, optimizer, scheduler, filename=pretrained_path
        )
        
        # Load additional checkpoint data if available
        if os.path.isfile(pretrained_path):
            checkpoint = torch.load(pretrained_path)
            if 'best_val_per_benchmark' in checkpoint:
                best_val_per_benchmark = checkpoint['best_val_per_benchmark']
                print(f"Loaded best performance tracking: {best_val_per_benchmark}")
            else:
                # Initialize if not found in checkpoint
                best_val_per_benchmark = {}
                for benchmark in args.eval_benchmarks:
                    best_val_per_benchmark[benchmark] = 0.0
            
            if 'best_epoch_per_benchmark' in checkpoint:
                best_epoch_per_benchmark = checkpoint['best_epoch_per_benchmark']
                print(f"Loaded best epoch tracking: {best_epoch_per_benchmark}")
            else:
                # Initialize if not found in checkpoint
                best_epoch_per_benchmark = {}
                for benchmark in args.eval_benchmarks:
                    best_epoch_per_benchmark[benchmark] = 0
            
            if 'best_avg_pck' in checkpoint:
                best_avg_pck = checkpoint['best_avg_pck']
                best_avg_epoch = checkpoint.get('best_avg_epoch', 0)
                print(f"Loaded best average PCK: {best_avg_pck:.2f}% (epoch {best_avg_epoch})")
            else:
                # Initialize if not found in checkpoint
                best_avg_pck = 0.0
                best_avg_epoch = 0
        # Transfer optimizer states to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
        # For finetuning, create a new snapshot directory to avoid overwriting
        pretrained_name = os.path.basename(os.path.dirname(args.pretrained))
        cur_snapshot = f"{pretrained_name}_finetune_{args.name_exp}"
        print(f"Finetuning: Creating new snapshot directory: {cur_snapshot}")
    else:
        # Create snapshot directory for training from scratch
        cur_snapshot = args.name_exp
        print(f"Training from scratch: Using snapshot directory: {cur_snapshot}")
    
    # Create snapshot directory
    if not os.path.isdir(args.snapshots):
        os.mkdir(args.snapshots)
    
    if not osp.isdir(osp.join(args.snapshots, cur_snapshot)):
        os.makedirs(osp.join(args.snapshots, cur_snapshot))
    
    # Save arguments (only if not loading from checkpoint)
    if not args.pretrained:
        with open(osp.join(args.snapshots, cur_snapshot, 'args.pkl'), 'wb') as f:
            pickle.dump(args, f)
    else:
        # For finetuning, save the finetuning arguments
        with open(osp.join(args.snapshots, cur_snapshot, 'finetune_args.pkl'), 'wb') as f:
            pickle.dump(args, f)
        # Also save reference to original pretrained model
        with open(osp.join(args.snapshots, cur_snapshot, 'pretrained_source.txt'), 'w') as f:
            f.write(f"Finetuned from: {args.pretrained}\n")
            f.write(f"Original model: {pretrained_name}\n")
    
    # Initialize best_val and start_epoch if not loading from checkpoint
    if not args.pretrained:
        best_val = 0
        start_epoch = 0
    
    # Initialize best performance tracking for each benchmark (if not loaded from checkpoint)
    if not args.pretrained:
        best_val_per_benchmark = {}
        best_epoch_per_benchmark = {}
        best_avg_pck = 0.0  # Track best average PCK across all benchmarks
        best_avg_epoch = 0  # Track epoch with best average PCK
        for benchmark in args.eval_benchmarks:
            best_val_per_benchmark[benchmark] = 0.0
            best_epoch_per_benchmark[benchmark] = 0
        print(f"Initialized best performance tracking for benchmarks: {list(best_val_per_benchmark.keys())}")
    
    # Setup logging
    save_path = osp.join(args.snapshots, cur_snapshot)
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
            f.write(f"Total epochs planned: {args.epochs}\n")
            f.write(f"Best primary benchmark PCK: {best_val:.4f}%\n")
            f.write(f"Best average PCK: {best_avg_pck:.4f}% (epoch {best_avg_epoch})\n")
            f.write(f"Primary benchmark: {args.eval_benchmarks[0]}\n\n")
            
            f.write("BEST PERFORMANCE PER BENCHMARK:\n")
            f.write("-" * 50 + "\n")
            for benchmark, best_pck in best_val_per_benchmark.items():
                best_epoch = best_epoch_per_benchmark.get(benchmark, 0)
                checkpoint_file = f"epoch_{best_epoch}.pth" if best_epoch > 0 else "N/A"
                f.write(f"{benchmark:12}: {best_pck:.2f}% PCK (epoch {best_epoch}, {checkpoint_file})\n")
            
            ############# Motion Aware Section ########
            f.write("\nMOTION-AWARE METRICS (from latest epoch):\n")
            f.write("-" * 50 + "\n")
            # Get latest validation results (would need to be passed in or stored)
            # For now, just note that motion-aware metrics are available
            f.write("Motion-aware PCK and static bias metrics are logged in validation_results.csv\n")
            f.write("Metrics include: PCK (motion-aware), PCK by motion bins, zero-flow precision/recall/F1, static bias ratio\n")
            ############# End Motion Aware Section ########
            
            f.write("\nTRAINING CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Train dataset: {args.train_dataset}\n")
            f.write(f"Learning rate: {args.lr}\n")
            f.write(f"Batch size: {args.batch_size}\n")
            f.write(f"Feature size: {args.feature_size}\n")
            f.write(f"Evaluation benchmarks: {', '.join(args.eval_benchmarks)}\n")
            f.write(f"Evaluation alphas: {', '.join(map(str, args.eval_alphas))}\n")
            f.write(f"Backbone: resnet101\n")
            f.write(f"Freeze backbone: {args.freeze}\n")
            f.write(f"Augmentation: {args.augmentation}\n")
            
            if is_final:
                f.write(f"\nTraining completed in: {time.time() - train_started:.2f} seconds\n")
                f.write("STATUS: Training completed successfully\n")
            else:
                f.write(f"\nSTATUS: Training in progress (epoch {epoch + 1}/{args.epochs})\n")
        
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
    print(f"Total epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Backbone learning rate: {args.lr_backbone}")
    
    # Pre-training visualizations (if enabled)
    reference_train_batch = None
    reference_val_batches = {}
    if args.enable_debug:
        print("\n" + "="*60)
        print("PRE-TRAINING VISUALIZATIONS")
        print("="*60)
        
        # Sample and save reference train batch
        print("Sampling reference train batch...")
        reference_train_batch = next(iter(train_dataloader))
        # Move to CPU to ensure persistence across epochs
        if isinstance(reference_train_batch, dict):
            reference_train_batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in reference_train_batch.items()}
        
        # Visualize training data with ground truth flow
        print("\nVisualizing train GT flow...")
        visualize_batch_flow(
            model=None,  # No model needed for GT
            batch=reference_train_batch,
            device=device,
            train_dataset_name=args.train_dataset,
            val_dataset_name=None,
            split_name='train',
            flow_source='gt',
            feature_size=args.feature_size,
            epoch=-1  # -1 indicates pre-training
        )
        
        # Visualize training data with predicted flow (untrained model)
        print("\nVisualizing train pred flow (untrained model)...")
        visualize_batch_flow(
            model=model,
            batch=reference_train_batch,
            device=device,
            train_dataset_name=args.train_dataset,
            val_dataset_name=None,
            split_name='train',
            flow_source='pred',
            feature_size=args.feature_size,
            epoch=-1  # -1 indicates pre-training
        )
        
        # Sample and save reference val batches for each benchmark
        print("\nSampling reference val batches for all benchmarks...")
        for benchmark, val_dataloader in val_dataloaders.items():
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
                train_dataset_name=args.train_dataset,
                val_dataset_name=benchmark,
                split_name='val',
                flow_source='gt',
                feature_size=args.feature_size,
                epoch=-1  # -1 indicates pre-training
            )
            
            # Visualize validation data with predicted flow (untrained model)
            print(f"\nVisualizing {benchmark} val pred flow (untrained model)...")
            visualize_batch_flow(
                model=model,
                batch=reference_val_batches[benchmark],
                device=device,
                train_dataset_name=args.train_dataset,
                val_dataset_name=benchmark,
                split_name='val',
                flow_source='pred',
                feature_size=args.feature_size,
                epoch=-1  # -1 indicates pre-training
            )
        
        print("="*60 + "\n")
    
    # Initialize cumulative training steps counter
    cumulative_training_steps = 0
    
    def get_steps_per_epoch(epoch):
        """Calculate steps per epoch based on args.steps_per_epoch setting"""
        if args.steps_per_epoch is None:
            return len(train_dataloader)
        elif args.steps_per_epoch == 'logarithmic':
            # Logarithmic progression: 2^epoch, capped at 1024
            steps = min(2 ** epoch, 2048)
            return steps
        else:
            # Integer value
            return args.steps_per_epoch
    
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
                                'zero_flow_precision', 'zero_flow_recall', 'zero_flow_f1', 'static_bias_ratio'])
                f.flush()  # Ensure header is written immediately
                os.fsync(f.fileno())  # Force OS to write to disk
            validation_log_initialized = True
            print(f"Created validation results CSV: {validation_log_file}")
        
        # Append results for each benchmark with immediate flushing
        with open(validation_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for benchmark, results in val_results.items():
                ############# Motion Aware Section ########
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
                    f"{static_bias:.4f}" if isinstance(static_bias, (int, float)) else ''
                ])
                ############# End Motion Aware Section ########
            f.flush()  # Ensure data is written to buffer immediately
            os.fsync(f.fileno())  # Force OS to write to disk
    
    # Training loop
    train_started = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        scheduler.step(epoch)
        
        # Training
        # Calculate steps per epoch for this epoch (may vary if using logarithmic mode)
        steps_per_epoch = get_steps_per_epoch(epoch)
        if args.steps_per_epoch == 'logarithmic':
            print(f"Epoch {epoch + 1}: Using {steps_per_epoch} steps (logarithmic mode)")
        
        # Create flow filter if parameters are provided (only for training)
        flow_filter = None
        if args.min_flow_length is not None or args.max_flow_length is not None:
            from src.data.synth.datasets.flow_filter import FlowLengthFilter
            flow_filter = FlowLengthFilter(min_flow_length=args.min_flow_length, max_flow_length=args.max_flow_length)
            if epoch == start_epoch:  # Only print once at the start
                print(f"Flow filtering enabled: min={args.min_flow_length}, max={args.max_flow_length}")
        
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
        val_results = validate_epoch_multi_benchmark(
            model, val_dataloaders, device, epoch, multi_evaluator,
            primary_benchmark=args.eval_benchmarks[0]
        )
        
        # Log results for each benchmark
        print(colored('==> ', 'blue') + 'epoch :', epoch + 1)
        pck_scores = []
        for benchmark, results in val_results.items():
            print(f"{benchmark} - Val Loss: {results['loss']:.4f}, PCK: {results['pck']:.2f}%")
            test_writer.add_scalar(f'val/{benchmark}/PCK', results['pck'], epoch)
            test_writer.add_scalar(f'val/{benchmark}/loss', results['loss'], epoch)
            
            ############# Motion Aware Section ########
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
            ############# End Motion Aware Section ########
            
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
        
        # In-training visualizations (if enabled)
        if args.enable_debug and reference_train_batch is not None:
            print("\nGenerating epoch visualizations...")
            
            # Visualize train GT flow (same batch as pre-training)
            visualize_batch_flow(
                model=None,
                batch=reference_train_batch,
                device=device,
                train_dataset_name=args.train_dataset,
                val_dataset_name=None,
                split_name='train',
                flow_source='gt',
                feature_size=args.feature_size,
                epoch=epoch
            )
            
            # Visualize train pred flow (same batch as pre-training)
            visualize_batch_flow(
                model=model,
                batch=reference_train_batch,
                device=device,
                train_dataset_name=args.train_dataset,
                val_dataset_name=None,
                split_name='train',
                flow_source='pred',
                feature_size=args.feature_size,
                epoch=epoch
            )
            
            # Visualize val GT and pred flow for each benchmark (same batches as pre-training)
            for benchmark, val_batch in reference_val_batches.items():
                # Visualize val GT flow
                visualize_batch_flow(
                    model=None,
                    batch=val_batch,
                    device=device,
                    train_dataset_name=args.train_dataset,
                    val_dataset_name=benchmark,
                    split_name='val',
                    flow_source='gt',
                    feature_size=args.feature_size,
                    epoch=epoch
                )
                
                # Visualize val pred flow
                visualize_batch_flow(
                    model=model,
                    batch=val_batch,
                    device=device,
                    train_dataset_name=args.train_dataset,
                    val_dataset_name=benchmark,
                    split_name='val',
                    flow_source='pred',
                    feature_size=args.feature_size,
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
        primary_benchmark = args.eval_benchmarks[0]
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
        }, is_best, save_path, 'epoch_{}.pth'.format(epoch + 1))
        
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
    write_training_summary(args.epochs - 1, is_final=True)
    
    # Close TensorBoard writers
    train_writer.close()
    test_writer.close()

if __name__ == "__main__":
    main()
