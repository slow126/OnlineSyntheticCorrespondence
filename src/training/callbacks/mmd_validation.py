"""
Callback for multi-benchmark validation with MMD calculations.

This callback calls the existing validate_epoch_multi_benchmark function
to preserve exact MMD calculation logic.
"""

import pytorch_lightning as pl
import torch
import gc
from typing import Dict, Any
from models.CATs_PlusPlus.utils_training.optimize_multi import validate_epoch_multi_benchmark


class MMDValidationCallback(pl.Callback):
    """
    Callback that performs multi-benchmark validation with MMD calculations.
    
    Uses the existing validate_epoch_multi_benchmark function to ensure
    MMD calculations are preserved exactly as in the original implementation.
    """
    
    def __init__(self, config: Dict[str, Any], multi_evaluator: Any):
        """
        Initialize callback.
        
        Args:
            config: Full training config dict
            multi_evaluator: MultiBenchmarkEvaluator instance
        """
        super().__init__()
        self.config = config
        self.multi_evaluator = multi_evaluator
        self.training_config = config['training']
        self.eval_config = config['evaluation']
    
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called at the start of training - do initial evaluation if enabled."""
        if not self.training_config.get('eval_initial', False):
            return
        
        print("\n" + "="*60)
        print("INITIAL EVALUATION (Before Training)")
        print("="*60)
        
        # Get validation dataloaders from datamodule
        datamodule = trainer.datamodule
        if not hasattr(datamodule, 'get_val_dataloaders'):
            raise RuntimeError("DataModule must have get_val_dataloaders() method")
        
        val_dataloaders = datamodule.get_val_dataloaders()
        
        # Get MMD config - for initial eval, calculate MMD if enabled
        mmd_every_n_epochs = self.training_config.get('mmd_every_n_epochs', 0)
        if mmd_every_n_epochs > 0:
            print(f"MMD calculation enabled for initial evaluation")
        
        # Get motion-aware config (default True for backward compatibility)
        use_motion_aware = self.eval_config.get('use_motion_aware', True)
        min_motion_pixels = self.eval_config.get('min_motion_pixels', 5.0)
        zero_threshold = self.eval_config.get('zero_threshold', 0.5)
        
        # Debug: Print motion-aware setting
        config_value = self.eval_config.get('use_motion_aware', 'not set')
        print(f"Motion-aware evaluation (initial): {use_motion_aware} (type: {type(use_motion_aware).__name__}, from config: {config_value} (type: {type(config_value).__name__}))")
        
        # Perform validation using existing function (epoch=-1 for initial)
        val_results = validate_epoch_multi_benchmark(
            net=pl_module.model,
            val_loaders=val_dataloaders,
            device=pl_module.device,
            epoch=-1,  # Initial evaluation before training
            multi_evaluator=self.multi_evaluator,
            primary_benchmark=self.eval_config['eval_benchmarks'][0],
            use_motion_aware=use_motion_aware,
            min_motion_pixels=min_motion_pixels,
            zero_threshold=zero_threshold,
            mmd_every_n_epochs=mmd_every_n_epochs
        )
        
        # Store results
        pl_module.set_val_results(val_results)
        
        # Log to TensorBoard (epoch=-1)
        for benchmark, results in val_results.items():
            trainer.logger.experiment.add_scalar(f'val/{benchmark}/PCK', results['pck'], -1)
            trainer.logger.experiment.add_scalar(f'val/{benchmark}/loss', results['loss'], -1)
            
            # Log MMD results if present
            if 'mmd2_pred_corr_vs_pred_miss' in results:
                mmd_val = results['mmd2_pred_corr_vs_pred_miss']
                if isinstance(mmd_val, (int, float)) and mmd_val == mmd_val:
                    trainer.logger.experiment.add_scalar(f'val/{benchmark}/MMD2_pred_corr_vs_pred_miss', mmd_val, -1)
            if 'mmd2_pred_corr_vs_gt' in results:
                mmd_val = results['mmd2_pred_corr_vs_gt']
                if isinstance(mmd_val, (int, float)) and mmd_val == mmd_val:
                    trainer.logger.experiment.add_scalar(f'val/{benchmark}/MMD2_pred_corr_vs_gt', mmd_val, -1)
            if 'mmd2_pred_miss_vs_gt' in results:
                mmd_val = results['mmd2_pred_miss_vs_gt']
                if isinstance(mmd_val, (int, float)) and mmd_val == mmd_val:
                    trainer.logger.experiment.add_scalar(f'val/{benchmark}/MMD2_pred_miss_vs_gt', mmd_val, -1)
        
        # Calculate and log average PCK
        pck_scores = [r['pck'] for r in val_results.values()]
        avg_pck = sum(pck_scores) / len(pck_scores) if pck_scores else 0.0
        trainer.logger.experiment.add_scalar('val/average/PCK', avg_pck, -1)
        
        print(f"\nInitial average PCK across benchmarks: {avg_pck:.2f}%")
        print("="*60 + "\n")
    
    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called at the start of validation epoch."""
        # Get validation dataloaders from datamodule
        datamodule = trainer.datamodule
        if not hasattr(datamodule, 'get_val_dataloaders'):
            raise RuntimeError("DataModule must have get_val_dataloaders() method")
        
        val_dataloaders = datamodule.get_val_dataloaders()
        
        # Get MMD config
        mmd_every_n_epochs = self.training_config.get('mmd_every_n_epochs', 0)
        
        # Get motion-aware config (default True for backward compatibility)
        use_motion_aware = self.eval_config.get('use_motion_aware', True)
        min_motion_pixels = self.eval_config.get('min_motion_pixels', 5.0)
        zero_threshold = self.eval_config.get('zero_threshold', 0.5)
        
        # Debug: Print motion-aware setting
        config_value = self.eval_config.get('use_motion_aware', 'not set')
        print(f"Motion-aware evaluation: {use_motion_aware} (type: {type(use_motion_aware).__name__}, from config: {config_value} (type: {type(config_value).__name__}))")
        
        # Perform validation using existing function
        # This preserves exact MMD calculation logic
        val_results = validate_epoch_multi_benchmark(
            net=pl_module.model,
            val_loaders=val_dataloaders,
            device=pl_module.device,
            epoch=trainer.current_epoch,
            multi_evaluator=self.multi_evaluator,
            primary_benchmark=self.eval_config['eval_benchmarks'][0],
            use_motion_aware=use_motion_aware,
            min_motion_pixels=min_motion_pixels,
            zero_threshold=zero_threshold,
            mmd_every_n_epochs=mmd_every_n_epochs
        )
        
        # Store results in Lightning module for other callbacks to access
        pl_module.set_val_results(val_results)
        
        # Log results to TensorBoard
        for benchmark, results in val_results.items():
            # Log PCK and loss
            trainer.logger.experiment.add_scalar(f'val/{benchmark}/PCK', results['pck'], trainer.current_epoch)
            trainer.logger.experiment.add_scalar(f'val/{benchmark}/loss', results['loss'], trainer.current_epoch)
            
            # Log motion-aware metrics
            if 'pck_motion_aware' in results:
                trainer.logger.experiment.add_scalar(f'val/{benchmark}/PCK_motion_aware', results['pck_motion_aware'], trainer.current_epoch)
            
            if 'motion_binned' in results:
                for bin_name, bin_data in results['motion_binned'].items():
                    if bin_data.get('count', 0) > 0:
                        trainer.logger.experiment.add_scalar(f'val/{benchmark}/PCK_motion_{bin_name}', bin_data['mean_pck'], trainer.current_epoch)
                        trainer.logger.experiment.add_scalar(f'val/{benchmark}/motion_{bin_name}_count', bin_data['count'], trainer.current_epoch)
            
            if 'zero_flow_metrics' in results:
                zfm = results['zero_flow_metrics']
                trainer.logger.experiment.add_scalar(f'val/{benchmark}/zero_flow_precision', zfm.get('zero_precision', 0), trainer.current_epoch)
                trainer.logger.experiment.add_scalar(f'val/{benchmark}/zero_flow_recall', zfm.get('zero_recall', 0), trainer.current_epoch)
                trainer.logger.experiment.add_scalar(f'val/{benchmark}/zero_flow_f1', zfm.get('zero_f1', 0), trainer.current_epoch)
                trainer.logger.experiment.add_scalar(f'val/{benchmark}/static_bias_ratio', zfm.get('static_bias_ratio', 0), trainer.current_epoch)
            
            # Log MMD metrics if present
            if 'mmd2_pred_corr_vs_pred_miss' in results:
                mmd_val = results['mmd2_pred_corr_vs_pred_miss']
                if isinstance(mmd_val, (int, float)) and mmd_val == mmd_val:  # Check for NaN
                    trainer.logger.experiment.add_scalar(f'val/{benchmark}/MMD2_pred_corr_vs_pred_miss', mmd_val, trainer.current_epoch)
            
            if 'mmd2_pred_corr_vs_gt' in results:
                mmd_val = results['mmd2_pred_corr_vs_gt']
                if isinstance(mmd_val, (int, float)) and mmd_val == mmd_val:
                    trainer.logger.experiment.add_scalar(f'val/{benchmark}/MMD2_pred_corr_vs_gt', mmd_val, trainer.current_epoch)
            
            if 'mmd2_pred_miss_vs_gt' in results:
                mmd_val = results['mmd2_pred_miss_vs_gt']
                if isinstance(mmd_val, (int, float)) and mmd_val == mmd_val:
                    trainer.logger.experiment.add_scalar(f'val/{benchmark}/MMD2_pred_miss_vs_gt', mmd_val, trainer.current_epoch)
            
            # Log per-category results for TSS
            if benchmark == 'tss' and 'pck_by_category' in results:
                for cat, pck in results['pck_by_category'].items():
                    trainer.logger.experiment.add_scalar(f'val/{benchmark}/{cat}/PCK', pck, trainer.current_epoch)
        
        # Calculate and log average PCK
        pck_scores = [r['pck'] for r in val_results.values()]
        avg_pck = sum(pck_scores) / len(pck_scores) if pck_scores else 0.0
        trainer.logger.experiment.add_scalar('val/average/PCK', avg_pck, trainer.current_epoch)
        
        # Print results
        print(f"\nValidation Results (Epoch {trainer.current_epoch + 1}):")
        for benchmark, results in val_results.items():
            print(f"  {benchmark}: PCK={results['pck']:.2f}%, Loss={results['loss']:.4f}")
            
            # Print MMD results if present
            if 'mmd2_pred_corr_vs_pred_miss' in results:
                mmd_val = results['mmd2_pred_corr_vs_pred_miss']
                if isinstance(mmd_val, (int, float)) and mmd_val == mmd_val:
                    print(f"    MMD^2 (pred_corr vs pred_miss): {mmd_val:.6f}")
            if 'mmd2_pred_corr_vs_gt' in results:
                mmd_val = results['mmd2_pred_corr_vs_gt']
                if isinstance(mmd_val, (int, float)) and mmd_val == mmd_val:
                    print(f"    MMD^2 (pred_corr vs gt): {mmd_val:.6f}")
            if 'mmd2_pred_miss_vs_gt' in results:
                mmd_val = results['mmd2_pred_miss_vs_gt']
                if isinstance(mmd_val, (int, float)) and mmd_val == mmd_val:
                    print(f"    MMD^2 (pred_miss vs gt): {mmd_val:.6f}")
        
        print(f"  Average PCK: {avg_pck:.2f}%")
