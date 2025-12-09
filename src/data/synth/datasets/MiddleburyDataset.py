"""
Middlebury Stereo Dataset for CATs++ training and evaluation.

Loads stereo image pairs (im0.png, im1.png) and disparity maps (disp0.pfm, disp1.pfm)
and converts disparities to optical flow. Creates pairs in both directions to double
the evaluation set size.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict
import random

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

# PFM reading utility (copied from pips2.utils.basic to avoid import issues)
import re

def readPFM(file):
    """Read PFM (Portable FloatMap) file format."""
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


class MiddleburySimpleDataset(Dataset):
    """
    Middlebury Stereo Dataset for CATs++. 
    
    Creates pairs in both directions (left-to-right and right-to-left) to double
    the evaluation set size. This is an eval-only dataset (no train/val split).
    
    Returns:
        - src_img: Source image [3, H, W]
        - trg_img: Target image [3, H, W] 
        - flow: Optical flow from trg to src (converted from disparity)
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'val',
        reverse_flow: bool = False,
    ):
        """
        Initialize Middlebury dataset.
        
        Args:
            root: Path to middlebury dataset folder (should contain scene directories)
            split: 'train' or 'val' (accepted for API consistency but ignored - eval-only dataset)
            reverse_flow: If True, reverse flow direction
        """
        self.root = Path(root)
        self.split = split  # Accepted but ignored - this is eval-only
        self.reverse_flow = reverse_flow
        
        # Find all scene directories
        self.scenes = []
        for item in sorted(self.root.iterdir()):
            if item.is_dir():
                # Check if it has the required files
                im0_path = item / 'im0.png'
                im1_path = item / 'im1.png'
                disp0_path = item / 'disp0.pfm'
                disp1_path = item / 'disp1.pfm'
                
                if im0_path.exists() and im1_path.exists() and disp0_path.exists() and disp1_path.exists():
                    self.scenes.append(item)
        
        if len(self.scenes) == 0:
            raise ValueError(f"No valid Middlebury scenes found in {self.root}")
        
        # No train/val split - all scenes are used for evaluation
        # Dataset length is 2 * num_scenes (pairwise approach)
        print(f"Middlebury dataset initialized: {len(self.scenes)} scenes, {len(self.scenes) * 2} samples (pairwise)")
    
    def __len__(self):
        return len(self.scenes) * 2  # Pairwise: each scene produces 2 samples
    
    def _load_image(self, img_path: Path) -> torch.Tensor:
        """Load and preprocess image."""
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor: (H, W, C) -> (C, H, W) and normalize to [0, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        return img_tensor
    
    def _load_disparity(self, disp_path: Path) -> np.ndarray:
        """Load disparity from PFM file."""
        disp = readPFM(str(disp_path))
        # Handle invalid disparities (typically inf or very large values)
        disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
        return disp
    
    def _disparity_to_flow(self, disp: np.ndarray, reverse: bool = False) -> torch.Tensor:
        """
        Convert disparity to optical flow from target to source.
        
        In Middlebury stereo:
        - im0 is left image (source)
        - im1 is right image (target)
        - disp0 is disparity for left image: pixel (x,y) in im0 corresponds to (x-disp0[x,y], y) in im1
        - disp1 is disparity for right image: pixel (x,y) in im1 corresponds to (x+disp1[x,y], y) in im0
        
        Flow convention: flow from target to source
        - Flow from im1 to im0: use disp1, flow = (disp1, 0) at each pixel in im1
        - Flow from im0 to im1: use disp0, flow = (-disp0, 0) at each pixel in im0
        
        Args:
            disp: Disparity map [H, W]
            reverse: If True, reverse flow direction
            
        Returns:
            Flow tensor [2, H, W]
        """
        H, W = disp.shape
        
        # Create flow: (dx, dy) = (disparity, 0) for horizontal flow
        # For flow from right (im1) to left (im0), use disp1: flow = (disp1, 0)
        # This means: pixel in im1 at (x,y) came from (x+disp1[x,y], y) in im0
        flow_x = disp.copy()
        flow_y = np.zeros_like(disp)
        
        if reverse:
            flow_x = -flow_x
        
        # Convert to tensor: (H, W) -> (2, H, W)
        flow = np.stack([flow_x, flow_y], axis=0)
        flow_tensor = torch.from_numpy(flow).float()
        
        # Mark invalid disparities (zero or very small, or very large) as invalid flow (inf)
        invalid_mask = (disp <= 0) | (disp > 1e6)
        flow_tensor[:, invalid_mask] = float('inf')
        
        return flow_tensor
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Pairwise approach:
        - Even indices (0, 2, 4, ...): left-to-right (src=im0, trg=im1)
        - Odd indices (1, 3, 5, ...): right-to-left (src=im1, trg=im0)
        
        Returns:
            Dictionary containing:
                - 'src_img': Source image [3, H, W]
                - 'trg_img': Target image [3, H, W]
                - 'flow': Flow [2, H, W] from target to source
        """
        scene_idx = idx // 2
        direction = idx % 2  # 0 = left-to-right, 1 = right-to-left
        
        scene_dir = self.scenes[scene_idx]
        
        if direction == 0:
            # Left-to-right: src=im0 (left), trg=im1 (right)
            # Flow from im1 to im0 using disp1
            src_img = self._load_image(scene_dir / 'im0.png')
            trg_img = self._load_image(scene_dir / 'im1.png')
            disp1 = self._load_disparity(scene_dir / 'disp1.pfm')
            flow = self._disparity_to_flow(disp1, reverse=self.reverse_flow)
        else:
            # Right-to-left: src=im1 (right), trg=im0 (left)
            # Flow from im0 to im1 using -disp0
            src_img = self._load_image(scene_dir / 'im1.png')
            trg_img = self._load_image(scene_dir / 'im0.png')
            disp0 = self._load_disparity(scene_dir / 'disp0.pfm')
            # For flow from im0 to im1, we need -disp0 (negative direction)
            flow = self._disparity_to_flow(disp0, reverse=not self.reverse_flow)
            # Since we're reversing the direction, we need to negate the flow
            flow[0] = -flow[0]
        
        # Ensure flow matches image dimensions (in case of size mismatch)
        _, img_h, img_w = src_img.shape
        flow_h, flow_w = flow.shape[1], flow.shape[2]
        
        if flow_h != img_h or flow_w != img_w:
            # Resize flow to match image size
            flow = torch.nn.functional.interpolate(
                flow.unsqueeze(0),
                size=(img_h, img_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            # Scale flow values proportionally
            flow[0] *= (img_w / flow_w)
            flow[1] *= (img_h / flow_h)
        
        # Create sample dict
        sample = {
            'src_img': src_img,
            'trg_img': trg_img,
            'flow': flow,
        }
        
        return sample


if __name__ == "__main__":
    # Test dataset
    import sys
    
    root = "/home/spencer/Data/middlebury/all"
    if len(sys.argv) > 1:
        root = sys.argv[1]
    
    print(f"Testing Middlebury dataset with root: {root}")
    
    # Test dataset
    print("\n=== Testing Dataset ===")
    dataset = MiddleburySimpleDataset(
        root=root,
        split='val',
        reverse_flow=False,
    )
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Number of scenes: {len(dataset.scenes)}")
    
    # Test first sample (left-to-right)
    sample0 = dataset[0]
    print(f"\nSample 0 (left-to-right) keys: {sample0.keys()}")
    print(f"src_img shape: {sample0['src_img'].shape}")
    print(f"trg_img shape: {sample0['trg_img'].shape}")
    print(f"flow shape: {sample0['flow'].shape}")
    print(f"flow range: [{sample0['flow'].min():.2f}, {sample0['flow'].max():.2f}]")
    print(f"Valid flow pixels: {(~torch.isinf(sample0['flow'])).sum().item()}")
    
    # Test second sample (right-to-left)
    sample1 = dataset[1]
    print(f"\nSample 1 (right-to-left) keys: {sample1.keys()}")
    print(f"src_img shape: {sample1['src_img'].shape}")
    print(f"trg_img shape: {sample1['trg_img'].shape}")
    print(f"flow shape: {sample1['flow'].shape}")
    print(f"flow range: [{sample1['flow'].min():.2f}, {sample1['flow'].max():.2f}]")
    print(f"Valid flow pixels: {(~torch.isinf(sample1['flow'])).sum().item()}")
    
    print("\nDataset test completed successfully!")
