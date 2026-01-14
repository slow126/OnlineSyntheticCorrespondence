"""
PyTorch Lightning DataModule for correspondence datasets.

Manages train/val datasets using CorrespondenceDataset.
"""

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Dict, Any, Optional
from train_cats_unified import create_training_dataset, create_validation_datasets


class CorrespondenceDataModule(pl.LightningDataModule):
    """
    DataModule for correspondence training.
    
    Handles training dataset and multiple validation datasets (one per benchmark).
    """
    
    def __init__(self, config: Dict[str, Any], device: Optional[torch.device] = None):
        """
        Initialize DataModule.
        
        Args:
            config: Full training config dict (with dataset, training, evaluation sections)
            device: Optional device (for dataset creation)
        """
        super().__init__()
        self.config = config
        self.device = device
        
        # Extract config sections
        self.dataset_config = config['dataset']
        self.training_config = config['training']
        self.eval_config = config['evaluation']
        
        # Will be set in setup()
        self.train_dataset = None
        self.val_datasets = {}
        self.val_dataloaders = {}
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets and dataloaders.
        
        Called by Lightning before training starts.
        """
        if stage == 'fit' or stage is None:
            # Create training dataset
            self.train_dataset = create_training_dataset(self.config, device=self.device)
            
            # Create validation datasets
            self.val_datasets, self.val_dataloaders = create_validation_datasets(
                self.config, device=self.device
            )
            
            print(f"Train dataset size: {len(self.train_dataset)}")
            for benchmark, dataloader in self.val_dataloaders.items():
                print(f"  Val dataloader for benchmark '{benchmark}' size: {len(dataloader)}")
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("train_dataset not initialized. Call setup() first.")
        
        batch_size = self.training_config['batch_size']
        n_threads = self.training_config.get('n_threads', 0)
        def is_synthetic_name(name: str) -> bool:
            return isinstance(name, str) and name.startswith("synthetic")
        
        # Check if dataset is mixed or single
        is_mixed = self.dataset_config.get('mixed', False) or 'datasets' in self.dataset_config
        if is_mixed:
            # For mixed datasets, check if any sub-dataset is synthetic
            datasets_list = self.dataset_config.get('datasets', [])
            has_synthetic = any(is_synthetic_name(name) for name in datasets_list)
            train_num_workers = 0 if has_synthetic else n_threads
        else:
            train_dataset_name = self.dataset_config.get('dataset_name', '')
            # Use num_workers=0 for synthetic dataset (GPU-bound rendering)
            train_num_workers = 0 if is_synthetic_name(train_dataset_name) else n_threads
        
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=train_num_workers,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
            prefetch_factor=batch_size if train_num_workers > 0 else None,
            pin_memory=True if train_num_workers > 0 else False
        )
    
    def val_dataloader(self) -> DataLoader:
        """
        Return a single validation dataloader for Lightning's validation loop.
        
        Note: Actual multi-benchmark validation is handled by MMDValidationCallback
        which calls validate_epoch_multi_benchmark with all validation dataloaders.
        This returns the primary benchmark's dataloader as a placeholder.
        """
        if not self.val_dataloaders:
            raise RuntimeError("val_dataloaders not initialized. Call setup() first.")
        
        # Return primary benchmark dataloader (Lightning needs a single dataloader)
        # The callback will handle multi-benchmark validation
        primary_benchmark = self.eval_config['eval_benchmarks'][0]
        return self.val_dataloaders[primary_benchmark]
    
    def get_val_dataloaders(self) -> Dict[str, DataLoader]:
        """Get validation dataloaders dict."""
        return self.val_dataloaders
    
    def get_train_dataset_name(self) -> str:
        """Get training dataset name."""
        # Handle mixed datasets
        is_mixed = self.dataset_config.get('mixed', False) or 'datasets' in self.dataset_config
        if is_mixed:
            # For mixed datasets, return a string representation
            datasets_list = self.dataset_config.get('datasets', [])
            if datasets_list:
                return '+'.join(datasets_list)  # e.g., "spair+synthetic"
            return "mixed"
        else:
            # Single dataset
            return self.dataset_config.get('dataset_name', 'unknown')
