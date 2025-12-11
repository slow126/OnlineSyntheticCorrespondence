"""
Callback for coverage validation using weighted coresets.

This callback computes coverage metrics between model predictions and ground truth
using precomputed eval coresets, similar to MMDValidationCallback.
"""

import pytorch_lightning as pl
import torch
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

# Import coreset utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.coreset import WeightedCoreset, coverage_by_train, extraneous_mass_fraction
from src.coreset.validation import extract_flow_vectors_from_batch


class CoresetValidationCallback(pl.Callback):
    """
    Callback that performs coverage validation using weighted coresets.
    
    At validation time (every N epochs):
    - Builds small coreset from model predictions
    - Loads precomputed eval label coreset
    - Computes bidirectional coverage metrics:
        * Labels → Predictions: "Does model cover the label space?"
        * Predictions → Labels: "Does model generate extraneous predictions?"
    
    Config options:
        coreset_every_n_epochs: Compute coverage every N epochs (0 = disabled)
        coreset_k_max: Size of prediction coreset (smaller for online use)
        coreset_min_count: Minimum count for absolute coverage
        coreset_precomputed: Dict mapping benchmark names to coreset file paths
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        coreset_every_n_epochs: int = 5,
        coreset_k_max: int = 5000,
        coreset_min_count: int = 100,
        precomputed_coresets: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize callback.
        
        Args:
            config: Full training config dict
            coreset_every_n_epochs: Compute coverage every N epochs (0 = disabled)
            coreset_k_max: Size of prediction coreset
            coreset_min_count: Minimum count for absolute coverage
            precomputed_coresets: Dict of {benchmark: coreset_file_path}
        """
        super().__init__()
        self.config = config
        self.coreset_every_n_epochs = coreset_every_n_epochs
        self.coreset_k_max = coreset_k_max
        self.coreset_min_count = coreset_min_count
        self.precomputed_coresets = precomputed_coresets or {}
        
        # Load precomputed coresets
        self.eval_coresets = {}
        for benchmark, path in self.precomputed_coresets.items():
            if Path(path).exists():
                print(f"Loading eval coreset for {benchmark}: {path}")
                self.eval_coresets[benchmark] = WeightedCoreset.load(path)
            else:
                print(f"Warning: Coreset file not found for {benchmark}: {path}")
    
    def _should_compute_coverage(self, epoch: int) -> bool:
        """Check if coverage should be computed this epoch."""
        if self.coreset_every_n_epochs <= 0:
            return False
        if epoch < 0:  # Initial eval
            return True
        return (epoch + 1) % self.coreset_every_n_epochs == 0
    
    def _build_prediction_coreset(
        self,
        pl_module: pl.LightningModule,
        dataloader,
        benchmark: str
    ) -> Optional[WeightedCoreset]:
        """
        Build a coreset from model predictions on the validation set.
        
        Args:
            pl_module: Lightning module (contains model)
            dataloader: Validation dataloader
            benchmark: Benchmark name
        
        Returns:
            WeightedCoreset built from predictions, or None if failed
        """
        # Create coreset
        coreset = WeightedCoreset(
            K_max=self.coreset_k_max,
            K_overflow=min(2000, self.coreset_k_max // 2),
            distance='euclidean',
            device='cpu',
            is_eval=False,  # Predictions don't need epsilon
        )
        
        # Set model to eval mode
        pl_module.eval()
        
        total_vectors = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(pl_module.device)
                
                # Get model predictions
                # This depends on your model's forward signature
                # Assuming model outputs flow predictions
                try:
                    outputs = pl_module.model(batch)
                    
                    # Extract predicted flow
                    if isinstance(outputs, dict):
                        pred_flow = outputs.get('flow', outputs.get('flow_full'))
                    else:
                        pred_flow = outputs
                    
                    if pred_flow is None:
                        continue
                    
                    # Create batch dict for extraction
                    pred_batch = {'flow_full': pred_flow}
                    
                    # Extract flow vectors
                    vectors = extract_flow_vectors_from_batch(pred_batch)
                    
                    if vectors is not None and len(vectors) > 0:
                        coreset.update(vectors)
                        total_vectors += len(vectors)
                
                except Exception as e:
                    print(f"Warning: Failed to process batch {batch_idx} for {benchmark}: {e}")
                    continue
                
                # Limit batches for speed
                if batch_idx >= 20:  # Process first 20 batches
                    break
        
        if total_vectors == 0:
            print(f"Warning: No prediction vectors extracted for {benchmark}")
            return None
        
        coreset.finalize()
        print(f"Built prediction coreset for {benchmark}: {len(coreset.get_centers())} centers from {total_vectors} vectors")
        
        return coreset
    
    def _compute_coverage_metrics(
        self,
        pred_coreset: WeightedCoreset,
        eval_coreset: WeightedCoreset,
        benchmark: str
    ) -> Dict[str, float]:
        """
        Compute bidirectional coverage metrics.
        
        Args:
            pred_coreset: Coreset built from predictions
            eval_coreset: Precomputed eval label coreset
            benchmark: Benchmark name
        
        Returns:
            Dict of metric names to values
        """
        pred_centers = pred_coreset.get_centers()
        pred_counts = pred_coreset.get_counts()
        eval_centers = eval_coreset.get_centers()
        eval_counts = eval_coreset.get_counts()
        
        # Get epsilon scales from eval coreset
        epsilon_scales = eval_coreset.get_epsilon_scales()
        if epsilon_scales is None:
            print(f"Warning: Eval coreset for {benchmark} has no epsilon scales")
            return {}
        
        metrics = {}
        
        # For each epsilon scale
        for eps_name, eps_value in epsilon_scales.items():
            if not eps_name.startswith('eps_'):
                continue
            
            # Labels → Predictions: How well do predictions cover labels?
            coverage_labels_by_preds = coverage_by_train(
                pred_centers, pred_counts, eval_centers,
                epsilon=eps_value, min_count=self.coreset_min_count
            )
            
            # Predictions → Labels: Extraneous predictions
            extran_preds = extraneous_mass_fraction(
                pred_centers, pred_counts, eval_centers,
                epsilon=eps_value
            )
            
            # Store metrics with descriptive names
            prefix = eps_name  # e.g., 'eps_base', 'eps_2x'
            metrics[f'coverage_labels_by_preds_{prefix}_rel'] = coverage_labels_by_preds['coverage_rel']
            metrics[f'coverage_labels_by_preds_{prefix}_abs'] = coverage_labels_by_preds['coverage_abs']
            metrics[f'rho_95_labels_to_preds_{prefix}'] = coverage_labels_by_preds['rho_95']
            metrics[f'extraneous_pred_mass_{prefix}'] = extran_preds['extraneous_mass_frac']
        
        return metrics
    
    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called at the start of validation epoch."""
        epoch = trainer.current_epoch
        
        if not self._should_compute_coverage(epoch):
            return
        
        print("\n" + "="*60)
        print(f"COVERAGE VALIDATION (Epoch {epoch + 1})")
        print("="*60)
        
        # Get validation dataloaders
        datamodule = trainer.datamodule
        if not hasattr(datamodule, 'get_val_dataloaders'):
            print("Warning: DataModule does not have get_val_dataloaders() method")
            return
        
        val_dataloaders = datamodule.get_val_dataloaders()
        
        # Compute coverage for each benchmark
        for benchmark, eval_coreset in self.eval_coresets.items():
            if benchmark not in val_dataloaders:
                print(f"Skipping {benchmark}: no validation dataloader")
                continue
            
            print(f"\nComputing coverage for {benchmark}...")
            
            # Build prediction coreset
            pred_coreset = self._build_prediction_coreset(
                pl_module,
                val_dataloaders[benchmark],
                benchmark
            )
            
            if pred_coreset is None:
                print(f"Skipping {benchmark}: failed to build prediction coreset")
                continue
            
            # Compute metrics
            metrics = self._compute_coverage_metrics(
                pred_coreset, eval_coreset, benchmark
            )
            
            # Log to TensorBoard
            for metric_name, metric_value in metrics.items():
                trainer.logger.experiment.add_scalar(
                    f'val/{benchmark}/{metric_name}',
                    metric_value,
                    epoch
                )
            
            # Print summary
            print(f"\n  {benchmark} coverage metrics:")
            for metric_name, metric_value in metrics.items():
                if 'coverage' in metric_name or 'extraneous' in metric_name:
                    print(f"    {metric_name}: {metric_value:.2%}")
                else:
                    print(f"    {metric_name}: {metric_value:.4f}")
        
        print("="*60 + "\n")
