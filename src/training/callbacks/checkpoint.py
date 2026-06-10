"""
Callback for saving best models per benchmark and overall best.

Handles checkpoint saving including finetuning snapshot directory naming.
"""

import os
import torch
import pytorch_lightning as pl
from typing import Dict, Any
from models.CATs_PlusPlus.utils_training.utils import save_checkpoint


class CheckpointCallback(pl.Callback):
    """
    Callback for saving best models per benchmark and overall best.
    
    Saves:
    - Individual benchmark best models: {benchmark}_best.pth
    - Overall best model: model_best.pth
    - Regular epoch checkpoints: epoch_{epoch}.pth
    """
    
    def __init__(self, save_path: str, config: Dict[str, Any], pretrained_checkpoint_data: Dict[str, Any] = None):
        """
        Initialize callback.
        
        Args:
            save_path: Directory to save checkpoints
            config: Full training config dict
            pretrained_checkpoint_data: Optional dict with best performance tracking from pretrained checkpoint
        """
        super().__init__()
        self.save_path = save_path
        self.config = config
        self.eval_config = config['evaluation']

        # Optimizer + scheduler state (AdamW keeps two momentum buffers per
        # parameter) roughly triples checkpoint size and is ONLY needed to
        # resume training, which is not wired here. Eval/inference loads the
        # weights identically without it. Default off; flip on by setting
        # paths.save_optimizer_state: true if you ever wire resume.
        self.save_optimizer_state = config.get('paths', {}).get('save_optimizer_state', False)
        
        # Track best performance per benchmark
        # Initialize from pretrained checkpoint if provided (for finetuning)
        if pretrained_checkpoint_data:
            self.best_val_per_benchmark = pretrained_checkpoint_data.get('best_val_per_benchmark', {})
            self.best_epoch_per_benchmark = pretrained_checkpoint_data.get('best_epoch_per_benchmark', {})
            self.best_avg_pck = pretrained_checkpoint_data.get('best_avg_pck', 0.0)
            self.best_avg_epoch = pretrained_checkpoint_data.get('best_avg_epoch', 0)
            
            # Ensure all benchmarks are initialized
            for benchmark in self.eval_config['eval_benchmarks']:
                if benchmark not in self.best_val_per_benchmark:
                    self.best_val_per_benchmark[benchmark] = 0.0
                if benchmark not in self.best_epoch_per_benchmark:
                    self.best_epoch_per_benchmark[benchmark] = 0
        else:
            self.best_val_per_benchmark = {}
            self.best_epoch_per_benchmark = {}
            self.best_avg_pck = 0.0
            self.best_avg_epoch = 0
            
            # Initialize best performance tracking
            for benchmark in self.eval_config['eval_benchmarks']:
                self.best_val_per_benchmark[benchmark] = 0.0
                self.best_epoch_per_benchmark[benchmark] = 0
    
    def _optim_states(self, trainer: pl.Trainer):
        """Return (optimizer_state, scheduler_state).

        Both are None unless paths.save_optimizer_state is enabled, so the
        checkpoints carry only the model weights needed for eval/inference.
        """
        if not self.save_optimizer_state:
            return None, None
        optimizer_state = trainer.optimizers[0].state_dict() if trainer.optimizers else None
        scheduler_state = (trainer.lr_scheduler_configs[0].scheduler.state_dict()
                           if trainer.lr_scheduler_configs else None)
        return optimizer_state, scheduler_state

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Save best models after validation epoch."""
        val_results = pl_module.get_val_results()
        if not val_results:
            return
        
        epoch = trainer.current_epoch
        
        # Track best performance for each benchmark and save individual models
        for benchmark, results in val_results.items():
            if results['pck'] > self.best_val_per_benchmark[benchmark]:
                self.best_val_per_benchmark[benchmark] = results['pck']
                self.best_epoch_per_benchmark[benchmark] = epoch + 1
                print(f"New best {benchmark} PCK: {results['pck']:.2f}% (epoch {epoch + 1})")
                
                # Optimizer/scheduler states (None unless save_optimizer_state).
                optimizer_state, scheduler_state = self._optim_states(trainer)

                # Save individual benchmark best model
                self._save_benchmark_model(
                    benchmark, epoch, results['pck'],
                    pl_module.model.module.state_dict() if hasattr(pl_module.model, 'module') else pl_module.model.state_dict(),
                    optimizer_state,
                    scheduler_state,
                    val_results
                )
        
        # Calculate average PCK across all benchmarks
        pck_scores = [r['pck'] for r in val_results.values()]
        avg_pck = sum(pck_scores) / len(pck_scores) if pck_scores else 0.0
        
        # Track best average performance and save overall best model
        if avg_pck > self.best_avg_pck:
            self.best_avg_pck = avg_pck
            self.best_avg_epoch = epoch + 1
            print(f"New best average PCK: {avg_pck:.2f}% (epoch {epoch + 1})")
            
            # Optimizer/scheduler states (None unless save_optimizer_state).
            optimizer_state, scheduler_state = self._optim_states(trainer)
            
            # Save overall best model
            self._save_overall_best_model(
                epoch, avg_pck,
                pl_module.model.module.state_dict() if hasattr(pl_module.model, 'module') else pl_module.model.state_dict(),
                optimizer_state,
                scheduler_state,
                val_results
            )
        
        # Use primary benchmark for best_val tracking
        primary_benchmark = self.eval_config['eval_benchmarks'][0]
        primary_results = val_results[primary_benchmark]
        is_best = primary_results['pck'] > max(self.best_val_per_benchmark.values())
        
        # Get optimizer and scheduler states from Lightning
        optimizer_state = None
        scheduler_state = None
        if trainer.optimizers:
            optimizer_state = trainer.optimizers[0].state_dict()
        if trainer.lr_scheduler_configs:
            scheduler_state = trainer.lr_scheduler_configs[0].scheduler.state_dict()
        
        # Save regular epoch checkpoint
        checkpoint_data = {
            'epoch': epoch + 1,
            'state_dict': pl_module.model.module.state_dict() if hasattr(pl_module.model, 'module') else pl_module.model.state_dict(),
            'optimizer': optimizer_state,
            'scheduler': scheduler_state,
            'best_loss': self.best_val_per_benchmark[primary_benchmark],
            'val_results': val_results,
            'best_val_per_benchmark': self.best_val_per_benchmark,
            'best_epoch_per_benchmark': self.best_epoch_per_benchmark,
            'best_avg_pck': self.best_avg_pck,
            'best_avg_epoch': self.best_avg_epoch,
        }
        
        # When paths.save_epoch_checkpoints is False, reuse a single rolling
        # 'last.pth' instead of one file per epoch. model_best.pth and
        # {benchmark}_best.pth are still written independently (is_best copy
        # below + _save_benchmark_model), so only the best models persist and
        # disk stays bounded regardless of epoch count.
        save_epoch_checkpoints = self.config.get('paths', {}).get('save_epoch_checkpoints', True)
        epoch_filename = f'epoch_{epoch + 1}.pth' if save_epoch_checkpoints else 'last.pth'

        save_checkpoint(
            checkpoint_data,
            is_best=is_best,
            save_path=self.save_path,
            filename=epoch_filename
        )
        
        if is_best:
            print(f"New best primary benchmark ({primary_benchmark}) PCK: {self.best_val_per_benchmark[primary_benchmark]:.2f}%")
    
    def _save_benchmark_model(self, benchmark: str, epoch: int, pck_score: float,
                             model_state: Dict, optimizer_state: Dict, scheduler_state: Dict,
                             val_results: Dict[str, Any]):
        """Save individual benchmark best model."""
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
        torch.save(checkpoint_data, os.path.join(self.save_path, filename))
        print(f"Saved best {benchmark} model: {filename} (PCK: {pck_score:.2f}%)")
    
    def _save_overall_best_model(self, epoch: int, avg_pck: float,
                                 model_state: Dict, optimizer_state: Dict, scheduler_state: Dict,
                                 val_results: Dict[str, Any]):
        """Save overall best model (best average across benchmarks)."""
        checkpoint_data = {
            'epoch': epoch + 1,
            'state_dict': model_state,
            'optimizer': optimizer_state,
            'scheduler': scheduler_state,
            'best_avg_pck': avg_pck,
            'val_results': val_results,
            'best_val_per_benchmark': self.best_val_per_benchmark,
            'best_epoch_per_benchmark': self.best_epoch_per_benchmark,
        }
        filename = "model_best.pth"
        torch.save(checkpoint_data, os.path.join(self.save_path, filename))
        print(f"Saved overall best model: {filename} (Avg PCK: {avg_pck:.2f}%)")
    
    def get_best_performance(self) -> Dict[str, Any]:
        """Get best performance tracking dicts."""
        return {
            'best_val_per_benchmark': self.best_val_per_benchmark,
            'best_epoch_per_benchmark': self.best_epoch_per_benchmark,
            'best_avg_pck': self.best_avg_pck,
            'best_avg_epoch': self.best_avg_epoch,
        }
