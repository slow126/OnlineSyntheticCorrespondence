"""
Callback for coverage validation using weighted coresets.

This callback computes coverage metrics between model predictions and ground truth
using precomputed eval coresets, similar to MMDValidationCallback.
"""

import pytorch_lightning as pl
import torch
from typing import Dict, Any, Optional
from pathlib import Path

# Import coreset utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.coreset import (
    WeightedCoreset,
    codebook_from_coreset,
    recall_train_covers_eval_soft,
    precision_train_wrt_eval_soft,
)
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
        coreset_k_nn: int = 5,
        coreset_bandwidth: Optional[float] = None,
        coreset_bandwidth_scale: float = 1.0,
        coreset_M_train: float = 100.0,
        coreset_M_eval: float = 20.0,
        coreset_kernel: str = "gaussian",
        precomputed_coresets: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize callback.
        
        Args:
            config: Full training config dict
            coreset_every_n_epochs: Compute coverage every N epochs (0 = disabled)
            coreset_k_max: Size of prediction coreset
            coreset_k_nn: k for soft k-NN metrics
            coreset_bandwidth: Optional bandwidth (None = inferred)
            coreset_bandwidth_scale: Scale factor applied to inferred bandwidth
            coreset_M_train: Saturation threshold for recall
            coreset_M_eval: Saturation threshold for precision
            coreset_kernel: Kernel type ('gaussian' or 'inverse')
            precomputed_coresets: Dict of {benchmark: coreset_file_path}
        """
        super().__init__()
        self.config = config
        self.coreset_every_n_epochs = coreset_every_n_epochs
        self.coreset_k_max = coreset_k_max
        self.coreset_k_nn = coreset_k_nn
        self.coreset_bandwidth = coreset_bandwidth
        self.coreset_bandwidth_scale = coreset_bandwidth_scale
        self.coreset_M_train = coreset_M_train
        self.coreset_M_eval = coreset_M_eval
        self.coreset_kernel = coreset_kernel
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
        Compute bidirectional soft k-NN coverage metrics.
        
        Args:
            pred_coreset: Coreset built from predictions
            eval_coreset: Precomputed eval label coreset
            benchmark: Benchmark name
        
        Returns:
            Dict of metric names to values
        """
        pred_cb = codebook_from_coreset(pred_coreset)
        eval_cb = codebook_from_coreset(eval_coreset)

        recall = recall_train_covers_eval_soft(
            pred_cb,
            eval_cb,
            k=self.coreset_k_nn,
            bandwidth=self.coreset_bandwidth,
            bandwidth_scale=self.coreset_bandwidth_scale,
            M_train=self.coreset_M_train,
            kernel=self.coreset_kernel,
        )
        recall_labels_cover_pred = recall_train_covers_eval_soft(
            eval_cb,
            pred_cb,
            k=self.coreset_k_nn,
            bandwidth=self.coreset_bandwidth,
            bandwidth_scale=self.coreset_bandwidth_scale,
            M_train=self.coreset_M_eval,
            kernel=self.coreset_kernel,
        )
        pred_precision = precision_train_wrt_eval_soft(
            pred_cb,
            eval_cb,
            k=self.coreset_k_nn,
            bandwidth=self.coreset_bandwidth,
            bandwidth_scale=self.coreset_bandwidth_scale,
            M_eval=self.coreset_M_eval,
            kernel=self.coreset_kernel,
        )
        outside = 1.0 - pred_precision

        return {
            'recall_pred_covers_labels': recall,
            'precision_pred_wrt_labels': pred_precision,
            'outside_pred_mass': outside,
            # symmetric direction for reference
            'recall_labels_cover_pred': recall_labels_cover_pred,
        }
    
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
                if metric_name.startswith(('recall', 'precision', 'outside')):
                    print(f"    {metric_name}: {metric_value:.2%}")
                else:
                    print(f"    {metric_name}: {metric_value:.4f}")
        
        print("="*60 + "\n")
