"""
HD1K Dataset for CATs++ training.

Uses torchvision.datasets.HD1K.
"""

from typing import Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets


class HD1KSimpleDataset(Dataset, nn.Module):
    """
    HD1K dataset for optical flow estimation.
    
    Note: HD1K is a small dataset typically used for fine-tuning or validation.
    
    Returns:
        - src_img: Source image (first frame)
        - trg_img: Target image (second frame) 
        - flow: Optical flow from src to trg (dx, dy)
    """
    
    def __init__(
        self, 
        root: str, 
        split: str = 'train',
        transforms: Optional[Callable] = None, 
        reverse_flow: bool = False
    ):
        """
        Initialize HD1K dataset.
        
        Args:
            root: Path to HD1K dataset root
            split: 'train' or 'test' (HD1K only has training data with flow annotations)
            transforms: Optional transforms to apply
            reverse_flow: If True, swap source and target images
        """
        Dataset.__init__(self)
        nn.Module.__init__(self)
        
        self.dataset = datasets.HD1K(
            root=root,
            split=split,
            transforms=transforms
        )
        self.reverse_flow = reverse_flow
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        The HD1K dataset returns: (img1, img2, flow, valid_flow_mask)
        """
        item = self.dataset[idx]
        
        if self.reverse_flow:
            src_index = 1  # img2
            trg_index = 0  # img1
        else:
            src_index = 0  # img1
            trg_index = 1  # img2
        
        # Convert PIL Images to tensors (keep on CPU - DataLoader will handle GPU transfer)
        src_img = torch.from_numpy(np.array(item[src_index])).permute(2, 0, 1).float() / 255.0
        trg_img = torch.from_numpy(np.array(item[trg_index])).permute(2, 0, 1).float() / 255.0
        
        # Flow is item[2], already a numpy array (H, W, 2)
        # Convert to torch tensor and permute to (2, H, W)
        flow = torch.from_numpy(np.array(item[2])).permute(2, 0, 1).float()

        return {
            "src_img": src_img,
            "trg_img": trg_img,
            "flow": flow,
        }
