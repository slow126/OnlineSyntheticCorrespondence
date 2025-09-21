import os
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
import torch


__all__ = [
    'BaseDatamodule',
    'make2',
]


def make2(val):
    if isinstance(val, (list, tuple)):
        return val
    return [val, val]


class BaseDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        root: Union[str, Path],
        batch_size: int = 128,
        num_workers: int = 8,
        shuffle: bool = True,
        copy_data_local: Optional[str] = None,
    ):
        super().__init__()
        self.prepare_data_per_node = True
        self.root = Path(root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.copy_data_local = copy_data_local
        self.post_init()

    def post_init(self):
        # Override to do dataset specific setup
        pass

    def prepare_data(self):
        if self.copy_data_local is not None:
            from .copy_data import copy_data
            self.original_root = self.root
            self.root = copy_data(
                copy_from = getattr(self, 'copy_from', self.root),
                copy_to = self.copy_data_local,
                datadir = os.environ['DATADIR'],
                file_list = getattr(self, 'file_list', None),
                pattern = getattr(self, 'pattern', None)
            )
            print(f'Setting root for dataloading to {str(self.root)}')

    def setup(self, stage: str):
        raise NotImplementedError()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=self.shuffle, pin_memory=True, drop_last=True, collate_fn=self.collate,
            worker_init_fn=self.get_worker_init(),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False, pin_memory=True, drop_last=False, collate_fn=self.collate,
            worker_init_fn=self.get_worker_init(),
        )

    def collate(self, batch):
        # Can override to do custom collate
        return torch.utils.data.default_collate(batch)

    def get_worker_init(self):
        # Override to return a worker_init_fn for the dataloader
        return None