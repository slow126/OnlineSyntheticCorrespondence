"""
Callback for writing training summary text file.

Updates training_summary.txt after each epoch with best PCK per benchmark
and configuration details.
"""

import os
import time
import pytorch_lightning as pl
from typing import Dict, Any


class SummaryCallback(pl.Callback):
    """
    Callback that writes training summary to text file.
    
    Updates training_summary.txt after each epoch with best performance
    and configuration details.
    """
    
    def __init__(self, save_path: str, config: Dict[str, Any], train_started: float):
        """
        Initialize callback.
        
        Args:
            save_path: Directory where training_summary.txt will be saved
            config: Full training config dict
            train_started: Timestamp when training started
        """
        super().__init__()
        self.save_path = save_path
        self.config = config
        self.train_started = train_started
        
        self.model_config = config['model']
        self.training_config = config['training']
        self.dataset_config = config['dataset']
        self.eval_config = config['evaluation']
        
        self.summary_file = os.path.join(save_path, 'training_summary.txt')
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Write training summary after validation epoch."""
        # Get best performance from checkpoint callback
        checkpoint_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, type(pl_module).__module__ + '.CheckpointCallback'):
                checkpoint_callback = callback
                break
        
        # Try to find checkpoint callback by checking if it has get_best_performance method
        best_perf = None
        for callback in trainer.callbacks:
            if hasattr(callback, 'get_best_performance'):
                best_perf = callback.get_best_performance()
                break
        
        if best_perf is None:
            # Initialize with defaults if not found
            best_perf = {
                'best_val_per_benchmark': {},
                'best_epoch_per_benchmark': {},
                'best_avg_pck': 0.0,
                'best_avg_epoch': 0,
            }
            for benchmark in self.eval_config['eval_benchmarks']:
                best_perf['best_val_per_benchmark'][benchmark] = 0.0
                best_perf['best_epoch_per_benchmark'][benchmark] = 0
        
        best_val_per_benchmark = best_perf['best_val_per_benchmark']
        best_epoch_per_benchmark = best_perf['best_epoch_per_benchmark']
        best_avg_pck = best_perf['best_avg_pck']
        best_avg_epoch = best_perf['best_avg_epoch']
        
        # Get primary benchmark best PCK
        primary_benchmark = self.eval_config['eval_benchmarks'][0]
        best_val = best_val_per_benchmark.get(primary_benchmark, 0.0)
        
        epoch = trainer.current_epoch
        epochs = self.training_config.get('epochs', 50)
        train_dataset_name = self.dataset_config['dataset_name']
        batch_size = self.training_config['batch_size']
        lr = self.training_config.get('lr', 3e-4)
        
        with open(self.summary_file, 'w') as f:
            f.write("TRAINING SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Current epoch: {epoch + 1}\n")
            f.write(f"Training time so far: {time.time() - self.train_started:.2f} seconds\n")
            f.write(f"Total epochs planned: {epochs}\n")
            f.write(f"Best primary benchmark PCK: {best_val:.4f}%\n")
            f.write(f"Best average PCK: {best_avg_pck:.4f}% (epoch {best_avg_epoch})\n")
            f.write(f"Primary benchmark: {primary_benchmark}\n\n")
            
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
            f.write(f"Feature size: {self.dataset_config['downsample_flow']}\n")
            f.write(f"Evaluation benchmarks: {', '.join(self.eval_config['eval_benchmarks'])}\n")
            f.write(f"Evaluation alphas: {', '.join(map(str, self.eval_config['eval_alphas']))}\n")
            f.write(f"Backbone: {self.model_config.get('backbone', 'resnet101')}\n")
            f.write(f"Freeze backbone: {self.model_config.get('freeze', True)}\n")
            f.write(f"Pretrained backbone: {self.model_config.get('pretrained_backbone', True)}\n")
            f.write(f"Augmentation: {self.training_config.get('augmentation', False)}\n")
            
            is_final = (epoch + 1) >= epochs
            if is_final:
                f.write(f"\nTraining completed in: {time.time() - self.train_started:.2f} seconds\n")
                f.write("STATUS: Training completed successfully\n")
            else:
                f.write(f"\nSTATUS: Training in progress (epoch {epoch + 1}/{epochs})\n")
        
        if (epoch + 1) >= epochs:
            print(f"Final training summary saved to: {self.summary_file}")
        else:
            print(f"Training summary updated: {self.summary_file}")
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Write final summary at end of training."""
        # Get best performance from checkpoint callback
        best_perf = None
        for callback in trainer.callbacks:
            if hasattr(callback, 'get_best_performance'):
                best_perf = callback.get_best_performance()
                break
        
        if best_perf is None:
            return
        
        best_val_per_benchmark = best_perf['best_val_per_benchmark']
        best_epoch_per_benchmark = best_perf['best_epoch_per_benchmark']
        best_avg_pck = best_perf['best_avg_pck']
        best_avg_epoch = best_perf['best_avg_epoch']
        
        # Print final best performance
        print("\n" + "="*60)
        print("BEST PERFORMANCE PER BENCHMARK:")
        print("="*60)
        
        for benchmark, best_pck in best_val_per_benchmark.items():
            best_epoch = best_epoch_per_benchmark.get(benchmark, 0)
            print(f"{benchmark:12}: {best_pck:.2f}% PCK (epoch {best_epoch})")
        
        print("-" * 60)
        print(f"{'AVERAGE':12}: {best_avg_pck:.2f}% PCK (epoch {best_avg_epoch})")
        print("="*60)
