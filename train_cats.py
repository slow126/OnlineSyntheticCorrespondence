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
sys.path.append('models/CATs-PlusPlus')
from models.cats_improved import CATsImproved
import utils_training.optimize as optimize
from utils_training.evaluation import Evaluator
from utils_training.utils import parse_list, load_checkpoint, save_checkpoint, boolean_string

# Import our synthetic dataset wrapper
from src.data.synth.online_synth_datamodule import create_datamodule_from_config
import torchvision
from pathlib import Path



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
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size')
    parser.add_argument('--n_threads', type=int, default=0,
                        help='number of parallel threads for dataloaders (0 recommended for OpenGL compatibility)')
    parser.add_argument('--seed', type=int, default=2021,
                        help='Pseudo-RNG seed')
    parser.add_argument('--backbone', type=str, default='resnet101')
    
    # Synthetic dataset parameters
    parser.add_argument('--train_config', type=str, default='src/configs/online_synth_configs/train_config.yaml', 
                        help='Path to YAML config file')
    parser.add_argument('--val_config', type=str, default='src/configs/online_synth_configs/val_config.yaml', 
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
    
    parser.add_argument('--freeze', type=boolean_string, nargs='?', const=True, default=False)
    parser.add_argument('--augmentation', type=boolean_string, nargs='?', const=True, default=True)
    
    # Evaluation parameters
    parser.add_argument('--benchmark', type=str, default='synthetic', choices=['synthetic'])
    parser.add_argument('--thres', type=str, default='img', choices=['auto', 'img', 'bbox', 'bbox-kp'])
    parser.add_argument('--alpha', type=float, default=0.1)
    
    args = parser.parse_args()
    
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
    try:
        Evaluator.initialize(args.benchmark, args.alpha)
    except:
        print("Warning: Could not initialize Evaluator for synthetic data")
    
    # Create synthetic dataset
    print("Creating synthetic dataset...")
    train_dataset = create_datamodule_from_config(args.train_config)
    val_dataset = create_datamodule_from_config(args.val_config)
    
    
    # Create dataloaders
    # Note: Using num_workers=0 to avoid OpenGL context issues with multiprocessing
    train_dataloader = train_dataset.train_dataloader()
    val_dataloader = val_dataset.train_dataloader()
    
    print(f"Train dataset size: {len(train_dataloader)}")
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
        print(f"Loading pretrained model from {args.pretrained}")
        model, optimizer, scheduler, start_epoch, best_val = load_checkpoint(
            model, optimizer, scheduler, filename=args.pretrained
        )
        # Transfer optimizer states to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        cur_snapshot = os.path.basename(os.path.dirname(args.pretrained))
    else:
        # Create snapshot directory
        if not os.path.isdir(args.snapshots):
            os.mkdir(args.snapshots)
        
        cur_snapshot = args.name_exp
        if not osp.isdir(osp.join(args.snapshots, cur_snapshot)):
            os.makedirs(osp.join(args.snapshots, cur_snapshot))
        
        # Save arguments
        with open(osp.join(args.snapshots, cur_snapshot, 'args.pkl'), 'wb') as f:
            pickle.dump(args, f)
        
        best_val = 0
        start_epoch = 0
    
    # Setup logging
    save_path = osp.join(args.snapshots, cur_snapshot)
    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    test_writer = SummaryWriter(os.path.join(save_path, 'test'))
    
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

            # Optionally save keypoints for inspection
            src_kps = batch['src_kps'][0].cpu().numpy()
            trg_kps = batch['trg_kps'][0].cpu().numpy()
            np.save(debug_dir / "sample_src_kps.npy", src_kps)
            np.save(debug_dir / "sample_trg_kps.npy", trg_kps)

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


if __name__ == "__main__":
    main()
