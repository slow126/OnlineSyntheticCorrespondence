"""
PyTorch Lightning module for correspondence models.

This module wraps any model with forward(trg_img, src_img) -> flow interface
and handles training/validation with MMD calculations and PCK evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
from typing import Dict, Any, Optional
from models.CATs_PlusPlus.utils_training.optimize import EPE
from models.CATs_PlusPlus.utils_training.utils import parse_list


class CorrespondenceLightningModule(pl.LightningModule):
    """
    Lightning module for correspondence models.
    
    Supports models with forward(trg_img, src_img) -> flow interface.
    Handles training with EPE loss and validation with multi-benchmark evaluation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        multi_evaluator: Any,  # MultiBenchmarkEvaluator
    ):
        """
        Initialize Lightning module.
        
        Args:
            model: Model instance with forward(trg_img, src_img) -> flow
            config: Full training config dict (with model, training, dataset, evaluation, paths sections)
            multi_evaluator: MultiBenchmarkEvaluator instance for validation
        """
        super().__init__()
        self.model = model
        self.config = config
        self.multi_evaluator = multi_evaluator
        
        # Extract config sections
        self.model_config = config['model']
        self.training_config = config['training']
        self.eval_config = config['evaluation']
        
        # Training state
        self.cumulative_training_steps = 0
        self.current_epoch_val_results = {}
        
        # Flow filter for training (optional)
        self.flow_filter = None
        min_flow_length = self.training_config.get('min_flow_length', None)
        max_flow_length = self.training_config.get('max_flow_length', None)
        if min_flow_length is not None or max_flow_length is not None:
            from src.data.synth.datasets.flow_filter import FlowLengthFilter
            self.flow_filter = FlowLengthFilter(
                min_flow_length=min_flow_length,
                max_flow_length=max_flow_length
            )
            print(f"Flow filtering enabled: min={min_flow_length}, max={max_flow_length}")
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=['model', 'multi_evaluator'])
    
    def forward(self, trg_img: torch.Tensor, src_img: torch.Tensor) -> torch.Tensor:
        """Forward pass through model."""
        return self.model(trg_img, src_img)
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step.
        
        Note: Actual validation is handled by MMDValidationCallback which calls
        validate_epoch_multi_benchmark. This is a dummy step to satisfy Lightning's
        validation loop requirements.
        """
        # Return empty dict - validation is handled by callback
        return {}
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Training step with EPE loss.
        
        Args:
            batch: Batch dictionary with 'src_img', 'trg_img', 'flow' or 'flow_downsampled'
            batch_idx: Batch index
            
        Returns:
            Dictionary with 'loss' key
        """
        # Move batch to device if needed
        device = self.device
        gpu_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                value_device = value.device
                needs_transfer = (
                    value_device.type != device.type or
                    (value_device.index if value_device.index is not None else 0) != 
                    (device.index if device.index is not None else 0)
                )
                if needs_transfer:
                    gpu_batch[key] = value.to(device, non_blocking=True)
                else:
                    gpu_batch[key] = value
            else:
                gpu_batch[key] = value
        
        # Ensure async transfers complete
        if device.type == 'cuda' and any(isinstance(v, torch.Tensor) and v.device.type == 'cuda' for v in gpu_batch.values()):
            torch.cuda.synchronize(device)
        
        # Get ground truth flow
        if 'flow_downsampled' in gpu_batch:
            flow_gt_key = 'flow_downsampled'
        else:
            flow_gt_key = 'flow'
        
        # Apply flow filtering if specified (only during training)
        if self.flow_filter is not None and flow_gt_key in gpu_batch:
            gpu_batch[flow_gt_key] = self.flow_filter.filter_batch_flow(gpu_batch[flow_gt_key])
        
        flow_gt = gpu_batch[flow_gt_key]
        
        # Forward pass
        pred_flow = self.model(gpu_batch['trg_img'], gpu_batch['src_img'])
        
        # Compute loss
        loss = EPE(pred_flow, flow_gt)
        
        # Log loss
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return {'loss': loss}
    
    def on_train_epoch_start(self):
        """Called at the start of each training epoch."""
        # Log steps per epoch if using logarithmic mode
        steps_per_epoch_config = self.training_config.get('steps_per_epoch', None)
        if steps_per_epoch_config == 'logarithmic':
            steps_per_epoch = min(2 ** self.current_epoch, 2048)
            print(f"Epoch {self.current_epoch + 1}: Using {steps_per_epoch} steps (logarithmic mode)")
    
    def on_train_batch_end(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any], batch_idx: int):
        """Update cumulative training steps."""
        self.cumulative_training_steps += 1
    
    def configure_optimizers(self):
        """
        Configure optimizer and scheduler.
        
        Supports separate learning rates for model vs backbone.
        """
        # Get learning rates
        def _to_float(val, name):
            if isinstance(val, (list, tuple)):
                if len(val) == 0:
                    raise ValueError(f"{name} cannot be empty")
                val = val[0]
            try:
                return float(val)
            except (TypeError, ValueError):
                raise ValueError(f"Expected {name} to be numeric, got {val!r}")
        
        lr = _to_float(self.training_config.get('lr', 3e-4), 'lr')
        lr_backbone = _to_float(self.training_config.get('lr_backbone', 3e-6), 'lr_backbone')
        weight_decay = _to_float(self.training_config.get('weight_decay', 0.05), 'weight_decay')
        
        # Separate parameters for model vs backbone
        param_model = [param for name, param in self.model.named_parameters() if 'backbone' not in name]
        param_backbone = [param for name, param in self.model.named_parameters() if 'backbone' in name]
        
        optimizer = optim.AdamW([
            {'params': param_model, 'lr': lr},
            {'params': param_backbone, 'lr': lr_backbone}
        ], weight_decay=weight_decay)
        
        # Setup scheduler
        scheduler_type = self.training_config.get('scheduler', 'step')
        epochs = self.training_config.get('epochs', 50)
        
        if scheduler_type == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )
        else:
            step_raw = self.training_config.get('step', '[70, 80, 90]')
            if isinstance(step_raw, (list, tuple)):
                milestones = [int(s) for s in step_raw]
            else:
                milestones = parse_list(str(step_raw))
            step_gamma = _to_float(self.training_config.get('step_gamma', 0.5), 'step_gamma')
            scheduler = lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=step_gamma
            )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }
    
    def get_cumulative_training_steps(self) -> int:
        """
        Get cumulative training steps (for CSV logging).
        
        Note: This counts actual training steps taken, accounting for
        steps_per_epoch limits (including logarithmic mode).
        """
        return self.cumulative_training_steps
    
    def get_val_results(self) -> Dict[str, Any]:
        """Get validation results from last validation epoch."""
        return self.current_epoch_val_results
    
    def set_val_results(self, results: Dict[str, Any]):
        """Set validation results (called by validation callback)."""
        self.current_epoch_val_results = results
