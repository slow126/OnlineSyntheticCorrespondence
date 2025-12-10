"""
MPI Sintel Dataset for CATs++ training.

Uses torchvision.datasets.Sintel with support for 'clean' and 'final' passes.
"""

from typing import Optional, Callable, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets


class SintelSimpleDataset(Dataset, nn.Module):
    """
    MPI Sintel dataset for optical flow estimation.
    
    Returns:
        - src_img: Source image (first frame)
        - trg_img: Target image (second frame) 
        - flow: Optical flow from src to trg (dx, dy)
    """
    
    def __init__(
        self, 
        root: str, 
        split: str = 'train',
        pass_name: str = 'clean',
        size: Optional[Tuple[int, int]] = None,
        transforms: Optional[Callable] = None, 
        reverse_flow: bool = False
    ):
        """
        Initialize Sintel dataset.
        
        Args:
            root: Path to Sintel dataset root (should contain 'training' and 'test' subdirectories)
            split: 'train' or 'test'
            pass_name: 'clean', 'final', or 'both' (rendering pass to use)
            size: Optional (H, W) tuple to resize images and flow
            transforms: Optional transforms to apply
            reverse_flow: If True, swap source and target images
        """
        Dataset.__init__(self)
        nn.Module.__init__(self)
        
        # Sintel uses 'training' and 'test' as split names
        split_map = {'train': 'train', 'val': 'test', 'test': 'test'}
        sintel_split = split_map.get(split, split)
        
        self.dataset = datasets.Sintel(
            root=root, 
            split=sintel_split,
            pass_name=pass_name,
            transforms=transforms
        )
        self.reverse_flow = reverse_flow
        self.size = size
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        The Sintel dataset returns: (img1, img2, flow, valid_flow_mask)
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
        
        # Flow is item[2], already a numpy array in (2, H, W) format from torchvision
        # No need to permute - just convert to torch tensor
        flow = torch.from_numpy(item[2]).float()
        
        # Resize if size is specified
        if self.size is not None:
            H_orig, W_orig = src_img.shape[1], src_img.shape[2]
            H_new, W_new = self.size
            
            # Resize images
            src_img = F.interpolate(src_img.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False).squeeze(0)
            trg_img = F.interpolate(trg_img.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False).squeeze(0)
            
            # Resize and scale flow
            # Flow needs to be scaled by the resize ratio
            scale_h = H_new / H_orig
            scale_w = W_new / W_orig
            
            flow_resized = F.interpolate(flow.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False).squeeze(0)
            
            # Scale flow values by resize ratio
            flow_resized[0] *= scale_w  # x component
            flow_resized[1] *= scale_h  # y component
            
            flow = flow_resized

        return {
            "src_img": src_img,
            "trg_img": trg_img,
            "flow": flow,
        }
