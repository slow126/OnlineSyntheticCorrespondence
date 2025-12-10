"""
Monkaa Dataset (part of SceneFlow) for CATs++ training.

Uses torchvision.datasets.SceneFlowStereo with variant='Monkaa'.
"""

from typing import Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets


class MonkaaSimpleDataset(Dataset, nn.Module):
    """
    Monkaa dataset from the SceneFlow dataset collection.
    
    Returns:
        - src_img: Source image (first frame)
        - trg_img: Target image (second frame) 
        - flow: Optical flow from trg to src (dx, dy)
    """
    
    def __init__(self, root: str, split: str, transforms: Optional[Callable] = None, reverse_flow: bool = False):
        """
        Initialize Monkaa dataset.
        
        Args:
            root: Path to SceneFlow dataset root (should contain 'Monkaa' subdirectory)
            split: 'train' or 'test' (SceneFlow uses 'train' split)
            transforms: Optional transforms to apply
            reverse_flow: If True, swap source and target images
        """
        Dataset.__init__(self)
        nn.Module.__init__(self)
        self.dataset = datasets.SceneFlowStereo(root=root, variant='Monkaa', split=split, transforms=transforms)
        self.reverse_flow = reverse_flow
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        The SceneFlowStereo dataset returns: (img1_left, img1_right, img2_left, img2_right, disparity, flow)
        We use img1_left and img2_left for optical flow training.
        """
        item = self.dataset[idx]
        
        if self.reverse_flow:
            src_index = 2  # img2_left
            trg_index = 0  # img1_left
        else:
            src_index = 0  # img1_left
            trg_index = 2  # img2_left
        
        # Convert PIL Images to tensors (keep on CPU - DataLoader will handle GPU transfer)
        src_img = torch.from_numpy(np.array(item[src_index])).permute(2, 0, 1).float() / 255.0
        trg_img = torch.from_numpy(np.array(item[trg_index])).permute(2, 0, 1).float() / 255.0
        
        # Flow is item[5], already a numpy array (H, W, 2)
        # Convert to torch tensor and permute to (2, H, W)
        flow = torch.from_numpy(np.array(item[5])).permute(2, 0, 1).float()

        return {
            "src_img": src_img,
            "trg_img": trg_img,
            "flow": flow,
        }
