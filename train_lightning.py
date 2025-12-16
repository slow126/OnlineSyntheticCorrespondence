"""
PyTorch Lightning training script for correspondence models.

This script uses PyTorch Lightning to train correspondence models while
preserving all functionality from train_cats_unified.py including:
- MMD calculations alongside PCK metrics
- Multi-benchmark evaluation
- Debug visualizations
- CSV logging
- Checkpoint management
"""

import argparse
import os
import random
import time
import yaml
import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path
from tensorboardX import SummaryWriter

# Import existing functions
from train_cats_unified import (
    load_config, create_training_dataset, create_validation_datasets,
    inspect_datasets
)

# Import Lightning components
from src.training.correspondence_lightning import CorrespondenceLightningModule
from src.training.correspondence_datamodule import CorrespondenceDataModule
from src.training.callbacks.mmd_validation import MMDValidationCallback
from src.training.callbacks.csv_logging import CSVLoggingCallback
from src.training.callbacks.visualization import VisualizationCallback
from src.training.callbacks.checkpoint import CheckpointCallback
from src.training.callbacks.summary import SummaryCallback

# Import model and utilities
from models.CATs_PlusPlus.models.cats_improved import CATsImproved
from models.CATs_PlusPlus.utils_training.eval_instance import MultiBenchmarkEvaluator
import models.CATs_PlusPlus.data.download as download
from models.CATs_PlusPlus.utils_training.utils import load_checkpoint

# Import model wrappers - import lazily to avoid path conflicts
# We'll import them in create_model() when needed
RAFTWrapper = None
FlowFormerWrapper = None


def create_model(model_config, paths_config):
    """
    Create model based on config.
    
    Args:
        model_config: Model configuration dictionary
        paths_config: Paths configuration dictionary (for pretrained paths)
        
    Returns:
        Model instance
    """
    import sys
    from pathlib import Path
    
    model_type = model_config.get('type', 'cats').lower()
    
    if model_type == 'cats':
        # CATs++ model (existing)
        pretrained_backbone = model_config.get('pretrained_backbone', True)
        if not pretrained_backbone:
            print('='*60)
            print('TRAINING FROM SCRATCH (pretrained_backbone=False)')
            print('='*60)
        else:
            print(f'Using pretrained backbone: {pretrained_backbone}')
        
        model = CATsImproved(
            backbone=model_config.get('backbone', 'resnet101'),
            freeze=model_config.get('freeze', True),
            pretrained_backbone=pretrained_backbone
        )
        
        # Count parameters (excluding backbone for CATs)
        def count_parameters(model):
            return sum(p.numel() for name, p in model.named_parameters() 
                      if p.requires_grad and 'backbone' not in name)
        
    elif model_type == 'raft':
        # RAFT model
        print("Initializing RAFT model...")
        # Import RAFT wrapper only when needed to avoid path conflicts
        models_path = Path(__file__).parent / "models"
        raft_path = models_path / "RAFT"
        if str(raft_path) not in sys.path:
            sys.path.insert(0, str(raft_path))
        from raft_wrapper import RAFTWrapper
        
        pretrained_path = paths_config.get('pretrained', model_config.get('pretrained_path', None))
        
        model = RAFTWrapper(
            small=model_config.get('small', False),
            iters=model_config.get('iters', 12),
            alternate_corr=model_config.get('alternate_corr', False),
            mixed_precision=model_config.get('mixed_precision', False),
            dropout=model_config.get('dropout', 0.0),
            pretrained_path=pretrained_path,
        )
        
        # Count all trainable parameters for RAFT
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    elif model_type == 'flowformer':
        # FlowFormer model
        print("Initializing FlowFormer model...")
        # Import FlowFormer wrapper only when needed to avoid path conflicts
        models_path = Path(__file__).parent / "models"
        flowformer_path = models_path / "FlowFormer-Official"
        if str(flowformer_path) not in sys.path:
            sys.path.insert(0, str(flowformer_path))
        from flowformer_wrapper import FlowFormerWrapper
        
        pretrained_path = paths_config.get('pretrained', model_config.get('pretrained_path', None))
        
        model = FlowFormerWrapper(
            pretrain=model_config.get('pretrain', True),
            iters=model_config.get('iters', 12),
            pretrained_path=pretrained_path,
        )
        
        # Count all trainable parameters for FlowFormer
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: 'cats', 'raft', 'flowformer'")
    
    print(f'The number of trainable parameters: {count_parameters(model)}')
    
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='PyTorch Lightning Training Script')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML config file')
    parser.add_argument('--inspect-data', action='store_true',
                       help='Run a quick data sanity check with visualizations and exit')
    parser.add_argument('--inspect-output-dir', type=str, default='debug_collate',
                       help='Output directory for data inspection visualizations')
    parser.add_argument('--inspect-visualize', action='store_true',
                       help='When using --inspect-data, actually save visualizations')
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
    
    # Set device
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
    
    # Download evaluation datasets
    standard_benchmarks = ['spair', 'pfpascal', 'pfwillow']
    for benchmark in eval_config['eval_benchmarks']:
        if benchmark in standard_benchmarks:
            download.download_dataset(eval_config['datapath'], benchmark)
    
    # Download training dataset if it's a standard benchmark dataset
    # Handle mixed datasets
    is_mixed = dataset_config.get('mixed', False) or 'datasets' in dataset_config
    if is_mixed:
        # For mixed datasets, check each sub-dataset
        datasets_list = dataset_config.get('datasets', [])
        for dataset_name in datasets_list:
            if dataset_name in standard_benchmarks:
                download.download_dataset(eval_config['datapath'], dataset_name)
    else:
        train_dataset_name = dataset_config.get('dataset_name')
        if train_dataset_name and train_dataset_name in standard_benchmarks:
            download.download_dataset(eval_config['datapath'], train_dataset_name)
    
    # Initialize model
    print("Initializing model...")
    model = create_model(model_config, paths_config)
    
    # Handle pretrained checkpoint loading for finetuning
    pretrained_path = paths_config.get('pretrained', None)
    start_epoch = paths_config.get('start_epoch', -1)
    
    # Create snapshot directory
    snapshots_dir = paths_config.get('snapshots', './snapshots')
    if not os.path.isdir(snapshots_dir):
        os.mkdir(snapshots_dir)
    
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
        
        # For finetuning, create a new snapshot directory
        pretrained_name = os.path.basename(os.path.dirname(pretrained_path))
        cur_snapshot = f"{pretrained_name}_finetune_{name_exp}"
        print(f"Finetuning: Creating new snapshot directory: {cur_snapshot}")
    else:
        # Create snapshot directory for training from scratch
        cur_snapshot = name_exp
        print(f"Training from scratch: Using snapshot directory: {cur_snapshot}")
    
    if not os.path.isdir(os.path.join(snapshots_dir, cur_snapshot)):
        os.makedirs(os.path.join(snapshots_dir, cur_snapshot))
    
    save_path = os.path.join(snapshots_dir, cur_snapshot)
    
    # Save config file to snapshot directory
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Save reference to pretrained model if finetuning
    if pretrained_path:
        with open(os.path.join(save_path, 'pretrained_source.txt'), 'w') as f:
            f.write(f"Finetuned from: {pretrained_path}\n")
            f.write(f"Original model: {pretrained_name}\n")
    
    # Create Lightning module
    lightning_module = CorrespondenceLightningModule(
        model=model,
        config=config,
        multi_evaluator=multi_evaluator
    )
    
    # Load pretrained checkpoint if specified (for finetuning)
    # Note: For CATs, we use load_checkpoint to preserve optimizer/scheduler state
    # For RAFT/FlowFormer, pretrained weights are already loaded in create_model
    model_type = model_config.get('type', 'cats').lower()
    pretrained_checkpoint_data = None
    
    if pretrained_path and model_type == 'cats':
        print(f"\n{'='*60}")
        print("FINETUNING MODE (CATs)")
        print(f"{'='*60}")
        print(f"Loading checkpoint from: {pretrained_path}")
        
        # Load checkpoint manually to get optimizer/scheduler states
        # Create temporary optimizer/scheduler to load states
        temp_optimizer = lightning_module.configure_optimizers()['optimizer']
        temp_scheduler = lightning_module.configure_optimizers()['lr_scheduler']['scheduler']
        
        model, temp_optimizer, temp_scheduler, start_epoch_loaded, best_val = load_checkpoint(
            model, temp_optimizer, temp_scheduler, filename=pretrained_path
        )
        
        # Update Lightning module's model
        lightning_module.model = model
        
        # Override start_epoch if loaded from checkpoint
        if start_epoch == -1:
            start_epoch = start_epoch_loaded - 1  # -1 because Lightning will increment
        
        # Load additional checkpoint data if available (best performance tracking)
        if os.path.isfile(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Pass best performance tracking to checkpoint callback
            pretrained_checkpoint_data = {}
            if 'best_val_per_benchmark' in checkpoint:
                pretrained_checkpoint_data['best_val_per_benchmark'] = checkpoint['best_val_per_benchmark']
                print(f"Loaded best performance tracking: {checkpoint['best_val_per_benchmark']}")
            if 'best_epoch_per_benchmark' in checkpoint:
                pretrained_checkpoint_data['best_epoch_per_benchmark'] = checkpoint['best_epoch_per_benchmark']
                print(f"Loaded best epoch tracking: {checkpoint['best_epoch_per_benchmark']}")
            if 'best_avg_pck' in checkpoint:
                pretrained_checkpoint_data['best_avg_pck'] = checkpoint['best_avg_pck']
                pretrained_checkpoint_data['best_avg_epoch'] = checkpoint.get('best_avg_epoch', 0)
                print(f"Loaded best average PCK: {checkpoint['best_avg_pck']:.2f}% (epoch {pretrained_checkpoint_data['best_avg_epoch']})")
        
        print(f"{'='*60}\n")
    elif pretrained_path and model_type in ['raft', 'flowformer']:
        print(f"\n{'='*60}")
        print(f"FINETUNING MODE ({model_type.upper()})")
        print(f"{'='*60}")
        print(f"Pretrained weights already loaded in model initialization")
        print(f"{'='*60}\n")
    
    # Create data module
    datamodule = CorrespondenceDataModule(config, device=device)
    
    # Setup callbacks
    callbacks = []
    
    # MMD validation callback (performs validation with MMD calculations)
    callbacks.append(MMDValidationCallback(config, multi_evaluator))
    
    # CSV logging callback
    callbacks.append(CSVLoggingCallback(save_path))
    
    # Visualization callback (if enabled)
    if training_config.get('enable_debug', False):
        callbacks.append(VisualizationCallback(config))
    
    # Checkpoint callback (with pretrained checkpoint data if finetuning)
    # pretrained_checkpoint_data is set above for CATs models
    callbacks.append(CheckpointCallback(save_path, config, pretrained_checkpoint_data))
    
    # Summary callback
    train_started = time.time()
    callbacks.append(SummaryCallback(save_path, config, train_started))
    
    # Setup TensorBoard logger
    logger = pl.loggers.TensorBoardLogger(
        save_dir=save_path,
        name='',
        version='',
        log_graph=False
    )
    
    # Configure trainer
    # Handle steps_per_epoch limit if specified
    steps_per_epoch_config = training_config.get('steps_per_epoch', None)
    
    # Check for logarithmic mode - not supported with Lightning
    if steps_per_epoch_config == 'logarithmic':
        print("\n" + "="*80)
        print("WARNING: 'logarithmic' steps_per_epoch is not supported with train_lightning.py")
        print("PyTorch Lightning does not support dynamically changing limit_train_batches")
        print("between epochs. Defaulting to 100 steps per epoch instead.")
        print("Use train_cats_unified.py if you need logarithmic step progression.")
        print("="*80 + "\n")
        limit_train_batches = 100
    elif steps_per_epoch_config is not None:
        # Fixed number of steps per epoch
        limit_train_batches = steps_per_epoch_config
    else:
        limit_train_batches = 1.0  # Use all batches
    
    # Validation frequency options
    check_val_every_n_epoch = training_config.get('check_val_every_n_epoch', 1)
    val_check_interval = training_config.get('val_check_interval', 1.0)
    
    trainer = pl.Trainer(
        max_epochs=training_config.get('epochs', 50),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=False,
        num_sanity_val_steps=0,  # Skip validation sanity check
        check_val_every_n_epoch=check_val_every_n_epoch,  # Check validation every N epochs
        val_check_interval=val_check_interval,  # Check validation at end of epoch (1.0) or after N steps (int)
        limit_train_batches=limit_train_batches,  # Limit training batches if specified
    )
    
    # Initial evaluation is handled by MMDValidationCallback's on_train_start
    # if training_config.get('eval_initial', False) is True
    
    # Train
    print(f"Starting training from epoch {start_epoch + 1}")
    print(f"Total epochs: {training_config.get('epochs', 50)}")
    print(f"Batch size: {training_config['batch_size']}")
    print(f"Learning rate: {training_config.get('lr', 3e-4)}")
    print(f"Backbone learning rate: {training_config.get('lr_backbone', 3e-6)}")
    
    trainer.fit(lightning_module, datamodule, ckpt_path=None)
    
    print(f'\nTraining took: {time.time() - train_started:.2f} seconds')
    print(f'Training completed successfully!')


if __name__ == "__main__":
    main()
