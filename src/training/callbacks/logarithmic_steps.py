"""
Callback to handle logarithmic steps per epoch.

Updates trainer.limit_train_batches at the start of each epoch
to implement logarithmic progression: 2^epoch, capped at 2048.
"""

import pytorch_lightning as pl
from typing import Optional


class LogarithmicStepsCallback(pl.Callback):
    """
    Callback to update limit_train_batches at the start of each epoch
    for logarithmic steps per epoch progression.
    """
    
    def __init__(self):
        super().__init__()
        self.enabled = False
    
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Update limit_train_batches at the start of each training epoch."""
        if not self.enabled:
            return
        
        epoch = trainer.current_epoch
        steps = min(2 ** epoch, 2048)
        trainer.limit_train_batches = steps
        
        if epoch == 0 or steps != min(2 ** (epoch - 1), 2048):
            print(f"Epoch {epoch + 1}: Using {steps} steps (logarithmic mode: 2^{epoch} capped at 2048)")
