"""
Callback for logging validation results to CSV.

Preserves exact CSV format from train_cats_unified.py.
"""

import csv
import os
from pathlib import Path
import pytorch_lightning as pl
from typing import Dict, Any


class CSVLoggingCallback(pl.Callback):
    """
    Callback that logs validation results to CSV file.
    
    Uses the same format as train_cats_unified.py's log_validation_results function.
    """
    
    def __init__(self, save_path: str):
        """
        Initialize callback.
        
        Args:
            save_path: Directory where validation_results.csv will be saved
        """
        super().__init__()
        self.save_path = Path(save_path)
        self.validation_log_file = self.save_path / 'validation_results.csv'
        self.validation_log_initialized = False
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Log validation results to CSV after validation epoch."""
        val_results = pl_module.get_val_results()
        if not val_results:
            return
        
        epoch = trainer.current_epoch
        cumulative_steps = pl_module.get_cumulative_training_steps()
        
        # Write header if first time
        if not self.validation_log_initialized:
            with open(self.validation_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'training_steps', 'benchmark', 'pck', 'loss',
                                'pck_motion_aware', 'pck_motion_small', 'pck_motion_medium', 'pck_motion_large',
                                'zero_flow_precision', 'zero_flow_recall', 'zero_flow_f1', 'static_bias_ratio',
                                'mmd2_pred_corr_vs_pred_miss', 'mmd2_pred_corr_vs_gt', 'mmd2_pred_miss_vs_gt'])
                f.flush()
                os.fsync(f.fileno())
            self.validation_log_initialized = True
            print(f"Created validation results CSV: {self.validation_log_file}")
        
        # Append results for each benchmark with immediate flushing
        with open(self.validation_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for benchmark, results in val_results.items():
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
                
                # Get MMD values
                mmd_pred_corr_vs_pred_miss = results.get('mmd2_pred_corr_vs_pred_miss', '')
                mmd_pred_corr_vs_gt = results.get('mmd2_pred_corr_vs_gt', '')
                mmd_pred_miss_vs_gt = results.get('mmd2_pred_miss_vs_gt', '')
                
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
                    f"{static_bias:.4f}" if isinstance(static_bias, (int, float)) else '',
                    f"{mmd_pred_corr_vs_pred_miss:.6f}" if isinstance(mmd_pred_corr_vs_pred_miss, (int, float)) else '',
                    f"{mmd_pred_corr_vs_gt:.6f}" if isinstance(mmd_pred_corr_vs_gt, (int, float)) else '',
                    f"{mmd_pred_miss_vs_gt:.6f}" if isinstance(mmd_pred_miss_vs_gt, (int, float)) else ''
                ])
            f.flush()
            os.fsync(f.fileno())
