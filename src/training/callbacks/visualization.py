"""
Callback for debug flow visualizations.

Uses existing visualize_batch_flow function from train_cats_unified.py.
"""

import torch
import pytorch_lightning as pl
from typing import Dict, Any, Optional
from train_cats_unified import visualize_batch_flow


class VisualizationCallback(pl.Callback):
    """
    Callback for debug flow visualizations.
    
    Handles pre-training visualizations and per-epoch visualizations
    if debug_visualization_persist is enabled.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize callback.
        
        Args:
            config: Full training config dict
        """
        super().__init__()
        self.config = config
        self.training_config = config['training']
        self.dataset_config = config['dataset']
        self.eval_config = config['evaluation']
        
        self.enable_debug = self.training_config.get('enable_debug', False)
        self.persist_debug_batches = self.training_config.get('debug_visualization_persist', False)
        self.feature_size = self.dataset_config['downsample_flow']
        
        # Store reference batches for per-epoch visualization
        self.reference_train_batch = None
        self.reference_val_batches = {}
    
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called at the start of training - do pre-training visualizations."""
        if not self.enable_debug:
            return
        
        print("\n" + "="*60)
        print("PRE-TRAINING VISUALIZATIONS")
        print("="*60)
        
        # Get which benchmarks to visualize
        debug_viz_benchmarks = self.training_config.get('debug_visualization_benchmarks', None)
        if debug_viz_benchmarks is None:
            debug_viz_benchmarks = [self.eval_config['eval_benchmarks'][0]]
        elif isinstance(debug_viz_benchmarks, str):
            if debug_viz_benchmarks.lower() == 'all':
                datamodule = trainer.datamodule
                debug_viz_benchmarks = list(datamodule.get_val_dataloaders().keys())
            else:
                debug_viz_benchmarks = [debug_viz_benchmarks]
        else:
            debug_viz_benchmarks = list(debug_viz_benchmarks)
        
        # Sample and save reference train batch
        print("Sampling reference train batch...")
        train_loader = trainer.datamodule.train_dataloader()
        self.reference_train_batch = next(iter(train_loader))
        # Move to CPU to ensure persistence across epochs
        if isinstance(self.reference_train_batch, dict):
            self.reference_train_batch = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v 
                for k, v in self.reference_train_batch.items()
            }
        
        train_dataset_name = trainer.datamodule.get_train_dataset_name()
        
        # Visualize training data with ground truth flow
        print("\nVisualizing train GT flow...")
        visualize_batch_flow(
            model=None,
            batch=self.reference_train_batch,
            device=pl_module.device,
            train_dataset_name=train_dataset_name,
            val_dataset_name=None,
            split_name='train',
            flow_source='gt',
            feature_size=self.feature_size,
            epoch=-1
        )
        
        # Visualize training data with predicted flow (untrained model)
        print("\nVisualizing train pred flow (untrained model)...")
        if self.reference_train_batch is not None:
            visualize_batch_flow(
                model=pl_module.model,
                batch=self.reference_train_batch,
                device=pl_module.device,
                train_dataset_name=train_dataset_name,
                val_dataset_name=None,
                split_name='train',
                flow_source='pred',
                feature_size=self.feature_size,
                epoch=-1
            )
        
        if not self.persist_debug_batches:
            self.reference_train_batch = None
        
        # Sample and save reference val batches for selected benchmarks
        print("\nSampling reference val batches for selected benchmarks...")
        val_dataloaders = trainer.datamodule.get_val_dataloaders()
        for benchmark, val_dataloader in val_dataloaders.items():
            if benchmark not in debug_viz_benchmarks:
                print(f"  Skipping {benchmark} visualizations (not in debug_visualization_benchmarks)")
                continue
            
            print(f"  Sampling batch for {benchmark}...")
            val_batch = next(iter(val_dataloader))
            # Move to CPU to ensure persistence across epochs
            if isinstance(val_batch, dict):
                val_batch = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v 
                    for k, v in val_batch.items()
                }
            self.reference_val_batches[benchmark] = val_batch
            
            # Visualize validation data with ground truth flow
            print(f"\nVisualizing {benchmark} val GT flow...")
            visualize_batch_flow(
                model=None,
                batch=self.reference_val_batches[benchmark],
                device=pl_module.device,
                train_dataset_name=train_dataset_name,
                val_dataset_name=benchmark,
                split_name='val',
                flow_source='gt',
                feature_size=self.feature_size,
                epoch=-1
            )
            
            # Visualize validation data with predicted flow (untrained model)
            print(f"\nVisualizing {benchmark} val pred flow (untrained model)...")
            visualize_batch_flow(
                model=pl_module.model,
                batch=self.reference_val_batches[benchmark],
                device=pl_module.device,
                train_dataset_name=train_dataset_name,
                val_dataset_name=benchmark,
                split_name='val',
                flow_source='pred',
                feature_size=self.feature_size,
                epoch=-1
            )
            
            if not self.persist_debug_batches:
                self.reference_val_batches.pop(benchmark, None)
        
        print("="*60 + "\n")
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called at the end of validation epoch - do per-epoch visualizations if enabled."""
        if not (self.enable_debug and self.persist_debug_batches):
            return
        
        if self.reference_train_batch is None:
            return
        
        print("\nGenerating epoch visualizations...")
        
        train_dataset_name = trainer.datamodule.get_train_dataset_name()
        epoch = trainer.current_epoch
        
        # Visualize train GT flow (same batch as pre-training)
        visualize_batch_flow(
            model=None,
            batch=self.reference_train_batch,
            device=pl_module.device,
            train_dataset_name=train_dataset_name,
            val_dataset_name=None,
            split_name='train',
            flow_source='gt',
            feature_size=self.feature_size,
            epoch=epoch
        )
        
        # Visualize train pred flow (same batch as pre-training)
        visualize_batch_flow(
            model=pl_module.model,
            batch=self.reference_train_batch,
            device=pl_module.device,
            train_dataset_name=train_dataset_name,
            val_dataset_name=None,
            split_name='train',
            flow_source='pred',
            feature_size=self.feature_size,
            epoch=epoch
        )
        
        # Visualize val GT and pred flow for each benchmark (same batches as pre-training)
        for benchmark, val_batch in self.reference_val_batches.items():
            # Visualize val GT flow
            visualize_batch_flow(
                model=None,
                batch=val_batch,
                device=pl_module.device,
                train_dataset_name=train_dataset_name,
                val_dataset_name=benchmark,
                split_name='val',
                flow_source='gt',
                feature_size=self.feature_size,
                epoch=epoch
            )
            
            # Visualize val pred flow
            visualize_batch_flow(
                model=pl_module.model,
                batch=val_batch,
                device=pl_module.device,
                train_dataset_name=train_dataset_name,
                val_dataset_name=benchmark,
                split_name='val',
                flow_source='pred',
                feature_size=self.feature_size,
                epoch=epoch
            )
        
        print("Epoch visualizations complete.\n")
