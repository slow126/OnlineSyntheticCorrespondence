"""
Example script showing how to use multi-benchmark evaluation during training
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

from models.cats_improved import CATsImproved
import utils_training.optimize_multi as optimize_multi
from utils_training.eval_instance import MultiBenchmarkEvaluator
from utils_training.utils import parse_list, log_args, load_checkpoint, save_checkpoint, boolean_string
from data import download


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='CATs Multi-Benchmark Training Script')
    
    # Paths
    parser.add_argument('--name_exp', type=str,
                        default=time.strftime('%Y_%m_%d_%H_%M'),
                        help='name of the experiment to save')
    parser.add_argument('--snapshots', type=str, default='./snapshots')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='training batch size')
    parser.add_argument('--n_threads', type=int, default=1,
                        help='number of parallel threads for dataloaders')
    parser.add_argument('--seed', type=int, default=2021,
                        help='Pseudo-RNG seed')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='weight decay')
    
    # Data parameters
    parser.add_argument('--datapath', type=str, default='../Datasets_CATs')
    parser.add_argument('--train_benchmark', type=str, default='pfpascal',
                        choices=['pfpascal', 'spair', 'pfwillow'],
                        help='benchmark for training')
    parser.add_argument('--eval_benchmarks', type=str, nargs='+', 
                        default=['pfpascal', 'spair'],
                        choices=['pfpascal', 'spair', 'pfwillow'],
                        help='benchmarks for evaluation during training')
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--feature_size', type=int, default=16)

    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create snapshots directory
    os.makedirs(args.snapshots, exist_ok=True)
    
    # Initialize multi-benchmark evaluator
    eval_benchmarks_config = {benchmark: args.alpha for benchmark in args.eval_benchmarks}
    multi_evaluator = MultiBenchmarkEvaluator(eval_benchmarks_config)
    print(f"Initialized evaluator for benchmarks: {multi_evaluator.get_available_benchmarks()}")
    
    # Download datasets
    for benchmark in [args.train_benchmark] + args.eval_benchmarks:
        download.download_dataset(args.datapath, benchmark)
    
    # Create training dataloader
    train_dataset = download.load_dataset(args.train_benchmark, args.datapath, args.thres, 
                                        device, 'trn', True, args.feature_size)
    train_dataloader = DataLoader(train_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.n_threads,
                                 shuffle=True)
    
    # Create validation dataloaders for each benchmark
    val_loaders = {}
    for benchmark in args.eval_benchmarks:
        val_dataset = download.load_dataset(benchmark, args.datapath, args.thres, 
                                          device, 'val', False, args.feature_size)
        val_loaders[benchmark] = DataLoader(val_dataset,
                                           batch_size=args.batch_size,
                                           num_workers=args.n_threads,
                                           shuffle=False)
        print(f"Created validation dataloader for {benchmark}: {len(val_dataset)} samples")

    # Model
    model = CATsImproved(backbone='resnet101')
    model = nn.DataParallel(model)
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # Tensorboard writer
    writer = SummaryWriter(log_dir=osp.join(args.snapshots, args.name_exp))

    # Training loop
    best_pck = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Training
        train_loss = optimize_multi.train_epoch(model, optimizer, train_dataloader, 
                                              device, epoch, writer)
        
        # Validation on multiple benchmarks
        val_results = optimize_multi.validate_epoch_multi_benchmark(
            model, val_loaders, device, epoch, multi_evaluator, 
            primary_benchmark=args.eval_benchmarks[0]
        )
        
        # Log results
        writer.add_scalar('train/loss', train_loss, epoch)
        for benchmark, results in val_results.items():
            writer.add_scalar(f'val/{benchmark}/loss', results['loss'], epoch)
            writer.add_scalar(f'val/{benchmark}/pck', results['pck'], epoch)
        
        # Print results
        print(f"Train Loss: {train_loss:.4f}")
        for benchmark, results in val_results.items():
            print(f"{benchmark} - Val Loss: {results['loss']:.4f}, PCK: {results['pck']:.2f}%")
        
        # Save best model (using first evaluation benchmark as primary)
        primary_benchmark = args.eval_benchmarks[0]
        primary_pck = val_results[primary_benchmark]['pck']
        
        if primary_pck > best_pck:
            best_pck = primary_pck
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_pck': best_pck,
                'val_results': val_results
            }, True, args.snapshots, args.name_exp)
            print(f"New best model saved! PCK: {best_pck:.2f}%")
        
        scheduler.step()
    
    writer.close()
    print(f"Training completed! Best PCK: {best_pck:.2f}%")


if __name__ == "__main__":
    main()
