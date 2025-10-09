"""
Training script for CATs++ model using synthetic correspondence dataset.
This script initializes the CATs++ model and sets up training with the online synthetic dataset.
"""

import argparse
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
from models.CATs_PlusPlus.utils_training.evaluation import Evaluator
from models.CATs_PlusPlus.utils_training.utils import parse_list, load_checkpoint, save_checkpoint, boolean_string
from src.data.synth.datasets.OnlineCorrespondenceDataset import OnlineCorrespondenceDataset
import models.CATs_PlusPlus.data.download as download
from models.CATs_PlusPlus.utils_training.eval_instance import MultiBenchmarkEvaluator
from models.CATs_PlusPlus.utils_training.optimize_multi import validate_epoch_multi_benchmark

# Import our synthetic dataset wrapper
import torchvision
from pathlib import Path

# TODO: Evaluate on multiple datasets at the same time. Will need to modify the validate_epoch function to support this.

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
    
    # Synthetic dataset parameters
    parser.add_argument('--train_dataset', type=str, default='synthetic', choices=['synthetic', 'spair', 'pfpascal', 'pfwillow', 'caltech'])
    parser.add_argument('--config', type=str, default='src/configs/online_synth_configs/OnlineDatasetConfig.yaml',
                        help='Path to YAML config file')
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
    
    # Evaluation parameters
    parser.add_argument('--benchmark', type=str, default='spair', choices=['synthetic', 'spair', 'pfpascal', 'pfwillow', 'caltech'],
                        help='single benchmark for training (legacy support)')
    parser.add_argument('--eval_benchmarks', type=str, nargs='+', default=['spair'],
                        choices=['synthetic', 'spair', 'pfpascal', 'pfwillow', 'caltech'],
                        help='list of benchmarks for evaluation during training')
    parser.add_argument('--eval_alphas', type=float, nargs='+', default=[0.1],
                        help='list of alpha values for each evaluation benchmark (must match eval_benchmarks length)')
    parser.add_argument('--thres', type=str, default='img', choices=['auto', 'img', 'bbox', 'bbox-kp'])
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='alpha for single benchmark mode (legacy support)')
    parser.add_argument('--datapath', type=str, default='./models/Datasets_CATs')
    parser.add_argument('--split_to_use_for_validation', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--feature_size', type=int, default=32,
                        help='feature size for downsampled flow. [default: 32]')
    parser.add_argument('--val_batch_size', type=int, default=8,
                        help='batch size for validation. [default: 8]')
    parser.add_argument('--val_num_workers', type=int, default=16,
                        help='number of workers for validation. [default: 16]')
    
    args = parser.parse_args()
    
    # Validate multi-benchmark arguments
    if len(args.eval_benchmarks) != len(args.eval_alphas):
        raise ValueError(f"Number of eval_benchmarks ({len(args.eval_benchmarks)}) must match number of eval_alphas ({len(args.eval_alphas)})")
    
    # Create benchmark-alpha mapping
    if len(args.eval_benchmarks) > 0:
        eval_benchmarks_config = dict(zip(args.eval_benchmarks, args.eval_alphas))
        print(f"Multi-benchmark evaluation config: {eval_benchmarks_config}")
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Warning about OpenGL multiprocessing
    if args.n_threads > 0:
        print("⚠️  WARNING: Using multiple workers with OpenGL rendering may cause segmentation faults.")
        print("   Consider using --n_threads 0 for stable training.")
    
    # Initialize Evaluator (for synthetic data, we'll use a simple version)
    # Note: You may need to modify the Evaluator class to work with synthetic data
    if len(args.eval_benchmarks) > 0:
        eval_benchmarks_config = dict(zip(args.eval_benchmarks, args.eval_alphas))
        print(f"Multi-benchmark evaluation config: {eval_benchmarks_config}")
        multi_evaluator = MultiBenchmarkEvaluator(eval_benchmarks_config)
        print(f"Initialized evaluator for benchmarks: {multi_evaluator.get_available_benchmarks()}")
    else:
        try:
            Evaluator.initialize(args.benchmark, args.alpha)
        except:
            print("Warning: Could not initialize Evaluator for synthetic data")
    
    # Check if the dataset is already downloaded
    download.download_dataset(args.datapath, args.benchmark)
    # Create synthetic dataset
    if args.train_dataset == 'synthetic':
        # Create dataloaders
        # Note: Using num_workers=0 to avoid OpenGL context issues with multiprocessing
        print("Creating synthetic dataset...")
        train_dataset = OnlineCorrespondenceDataset(
            geometry_config_path='src/configs/online_synth_configs/OnlineGeometryConfig.yaml',
            processor_config_path='src/configs/online_synth_configs/OnlineProcessorConfig.yaml',
            split='train'
        )
        train_dataset.cuda()
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_threads, shuffle=True, collate_fn=train_dataset.collate_fn)
    else:
        train_dataset = download.load_dataset(args.benchmark, args.datapath, args.thres, device, 'trn', False, args.feature_size)
        train_dataloader = DataLoader(train_dataset,
        batch_size=args.val_batch_size,
        num_workers=args.val_num_workers,
        persistent_workers=True,
        prefetch_factor=8,
        shuffle=True)

    print(f"Train dataset size: {len(train_dataloader)}")


    if len(args.eval_benchmarks) > 0:
        val_loaders = {benchmark: download.load_dataset(benchmark, args.datapath, args.thres, device, 'val', False, args.feature_size) for benchmark in args.eval_benchmarks}
        val_dataloaders = {benchmark: DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.val_num_workers, persistent_workers=True, prefetch_factor=8, shuffle=False) for benchmark, val_dataset in val_loaders.items()}
        # Loop through the validation dataloaders and print their name and size
        for benchmark, dataloader in val_dataloaders.items():
            print(f"Val dataloader for benchmark '{benchmark}' size: {len(dataloader)}")
    else:
        if args.benchmark == 'synthetic':
            val_dataset = OnlineCorrespondenceDataset(
                geometry_config_path='src/configs/online_synth_configs/OnlineGeometryConfig_Val.yaml',
                processor_config_path='src/configs/online_synth_configs/OnlineProcessorConfig.yaml',
                split='val'
            )
            val_dataset.cuda()
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.n_threads, shuffle=False, collate_fn=val_dataset.collate_fn)
        else:
            val_dataset = download.load_dataset(args.benchmark, args.datapath, args.thres, device, 'val', False, args.feature_size)
            val_dataloader = DataLoader(val_dataset,
            batch_size=args.val_batch_size,
            num_workers=args.val_num_workers,
            persistent_workers=True,
            prefetch_factor=8,
            shuffle=False)
        
        print(f"Val dataset size: {len(val_dataloader)}")
    

    
    
    
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
                if len(args.eval_benchmarks) > 0:
                    for benchmark in args.eval_benchmarks:
                        best_val_per_benchmark[benchmark] = 0.0
            
            if 'best_epoch_per_benchmark' in checkpoint:
                best_epoch_per_benchmark = checkpoint['best_epoch_per_benchmark']
                print(f"Loaded best epoch tracking: {best_epoch_per_benchmark}")
            else:
                # Initialize if not found in checkpoint
                best_epoch_per_benchmark = {}
                if len(args.eval_benchmarks) > 0:
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
        if len(args.eval_benchmarks) > 0:
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
        if len(args.eval_benchmarks) == 0:
            return
            
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
    
    # Move model to device and wrap with DataParallel
    model = nn.DataParallel(model)
    model = model.to(device)
    
    print("Model initialized successfully!")
    print(f"Starting training from epoch {start_epoch}")
    print(f"Total epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Backbone learning rate: {args.lr_backbone}")
    
    # Training loop
    train_started = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        scheduler.step(epoch)
        
        # Grab a sample batch from the training dataloader and save it to the debug folder

        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True, parents=True)
        try:
            sample_batch = next(iter(train_dataloader))
            # The new batch format is a dict with keys like 'src_img', 'trg_img', 'flow', etc.
            batch = sample_batch

            # Take the first sample in the batch for visualization
            src_img_tensor = batch['src_img'][0].cpu()
            trg_img_tensor = batch['trg_img'][0].cpu()
            flow_tensor = batch['flow'][0].cpu()

            # Normalize images for visualization ([0,1])
            src_img_vis = (src_img_tensor - src_img_tensor.min()) / (src_img_tensor.max() - src_img_tensor.min() + 1e-8)
            trg_img_vis = (trg_img_tensor - trg_img_tensor.min()) / (trg_img_tensor.max() - trg_img_tensor.min() + 1e-8)

            # Save source and target images
            torchvision.utils.save_image(src_img_vis, debug_dir / "sample_src.png")
            torchvision.utils.save_image(trg_img_vis, debug_dir / "sample_trg.png")
            # Save flow as numpy for inspection
            np.save(debug_dir / "sample_flow.npy", flow_tensor.numpy())

            # # Optionally save keypoints for inspection
            # src_kps = batch['src_kps'][0].cpu().numpy()
            # trg_kps = batch['trg_kps'][0].cpu().numpy()
            # np.save(debug_dir / "sample_src_kps.npy", src_kps)
            # np.save(debug_dir / "sample_trg_kps.npy", trg_kps)

            print(f"Saved a sample batch to {debug_dir}")
        except Exception as e:
            print(f"Could not save sample batch for debug: {e}")

        # Training
        train_loss = optimize.train_epoch(
            model, optimizer, train_dataloader, device, epoch, train_writer
        )
        train_writer.add_scalar('train loss', train_loss, epoch)
        train_writer.add_scalar('learning_rate', scheduler.get_lr()[0], epoch)
        train_writer.add_scalar('learning_rate_backbone', scheduler.get_lr()[1], epoch)
        print(colored('==> ', 'green') + 'Train average loss:', train_loss)
        
        # Validation
        if len(args.eval_benchmarks) > 0:
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
                pck_scores.append(results['pck'])
                
                # Track best performance for each benchmark and save individual models
                if results['pck'] > best_val_per_benchmark[benchmark]:
                    best_val_per_benchmark[benchmark] = results['pck']
                    best_epoch_per_benchmark[benchmark] = epoch + 1
                    print(f"New best {benchmark} PCK: {results['pck']:.2f}% (epoch {epoch + 1})")
                    
                    # Save individual benchmark best model
                    save_benchmark_model(
                        benchmark, epoch, results['pck'], 
                        model.module.state_dict(), optimizer.state_dict(), 
                        scheduler.state_dict(), val_results
                    )
            
            # Calculate average PCK across all benchmarks
            avg_pck = sum(pck_scores) / len(pck_scores)
            test_writer.add_scalar('val/average/PCK', avg_pck, epoch)
            print(f"Average PCK across benchmarks: {avg_pck:.2f}%")
            
            # Track best average performance and save overall best model
            if avg_pck > best_avg_pck:
                best_avg_pck = avg_pck
                best_avg_epoch = epoch + 1
                print(f"New best average PCK: {avg_pck:.2f}% (epoch {epoch + 1})")
                
                # Save overall best model
                save_overall_best_model(
                    epoch, avg_pck, model.module.state_dict(), 
                    optimizer.state_dict(), scheduler.state_dict(), val_results
                )
            
            # Use primary benchmark for legacy best_val tracking
            primary_benchmark = args.eval_benchmarks[0]
            primary_results = val_results[primary_benchmark]
            is_best = primary_results['pck'] > best_val
            best_val = max(primary_results['pck'], best_val)
            
            # Save regular epoch checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
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
        else:
            val_loss_grid, val_mean_pck = optimize.validate_epoch(
                model, val_dataloader, device, epoch=epoch
            )
            print(colored('==> ', 'blue') + 'Val average grid loss:', val_loss_grid)
            print('mean PCK is {}'.format(val_mean_pck))
            print(colored('==> ', 'blue') + 'epoch :', epoch + 1)
            test_writer.add_scalar('mean PCK', val_mean_pck, epoch)
            test_writer.add_scalar('val loss', val_loss_grid, epoch)
            
            # Save checkpoint
            is_best = val_mean_pck > best_val
            best_val = max(val_mean_pck, best_val)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_loss': best_val,
            }, is_best, save_path, 'epoch_{}.pth'.format(epoch + 1))
    
    print(f'Training took: {time.time() - train_started:.2f} seconds')
    print(f'Best validation PCK: {best_val:.4f}')
    
    # Print and log best performance for each benchmark
    if len(args.eval_benchmarks) > 0:
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
