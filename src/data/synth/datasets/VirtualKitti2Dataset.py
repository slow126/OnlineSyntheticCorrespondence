"""
Virtual KITTI 2 Dataset for CATs++ training.

Custom implementation for Virtual KITTI 2 dataset (no torchvision support).
"""

import os
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class VirtualKitti2SimpleDataset(Dataset):
    """
    Virtual KITTI 2 dataset for optical flow estimation.
    
    Virtual KITTI 2 is a synthetic dataset with photo-realistic rendering.
    Dataset structure:
        vkitti_2.0.3/
            Scene01/
                clone/
                    frames/
                        rgb/
                            Camera_0/
                                rgb_00000.jpg
                                rgb_00001.jpg
                                ...
                        forwardFlow/
                            Camera_0/
                                flow_00000.png
                                flow_00001.png
                                ...
                    ...
                15-deg-left/
                ...
            Scene02/
            ...
    
    Returns:
        - src_img: Source image (first frame)
        - trg_img: Target image (second frame) 
        - flow: Optical flow from src to trg (dx, dy)
    """
    
    def __init__(
        self, 
        root: str, 
        split: str = 'train',
        camera: str = 'Camera_0',
        reverse_flow: bool = False
    ):
        """
        Initialize Virtual KITTI 2 dataset.
        
        Args:
            root: Path to Virtual KITTI 2 dataset root (e.g., vkitti_2.0.3/)
            split: 'train' or 'test' (we'll use scenes for train/test split)
            camera: Camera to use (default: 'Camera_0')
            reverse_flow: If True, swap source and target images
        """
        self.root = Path(root)
        self.split = split
        self.camera = camera
        self.reverse_flow = reverse_flow
        
        # Define train/test splits (Scene01-Scene06 for train, Scene18-Scene20 for test)
        if split == 'train':
            scenes = ['Scene01', 'Scene02', 'Scene06']
        elif split in ['val', 'test']:
            scenes = ['Scene18', 'Scene20']
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
        
        # Collect all image pairs and flow files
        self.samples = []
        
        for scene in scenes:
            scene_path = self.root / scene
            if not scene_path.exists():
                print(f"Warning: Scene {scene} not found at {scene_path}")
                continue
            
            # Iterate through all variants (clone, 15-deg-left, etc.)
            for variant in scene_path.iterdir():
                if not variant.is_dir():
                    continue
                
                rgb_dir = variant / 'frames' / 'rgb' / camera
                flow_dir = variant / 'frames' / 'forwardFlow' / camera
                
                if not rgb_dir.exists() or not flow_dir.exists():
                    continue
                
                # Get all RGB images and sort them
                rgb_files = sorted(rgb_dir.glob('rgb_*.jpg'))
                
                # Create pairs (img[i], img[i+1]) with corresponding flow
                for i in range(len(rgb_files) - 1):
                    img1_path = rgb_files[i]
                    img2_path = rgb_files[i + 1]
                    
                    # Flow file naming: flow_00000.png corresponds to flow from img 0 to img 1
                    frame_num = int(img1_path.stem.split('_')[1])
                    flow_file = flow_dir / f'flow_{frame_num:05d}.png'
                    
                    if flow_file.exists():
                        self.samples.append({
                            'img1': str(img1_path),
                            'img2': str(img2_path),
                            'flow': str(flow_file)
                        })
        
        print(f"Virtual KITTI 2 dataset initialized: {len(self.samples)} samples in '{split}' split")
        
    def __len__(self):
        return len(self.samples)
    
    def _load_image(self, img_path: str) -> torch.Tensor:
        """Load and preprocess image."""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor: (H, W, C) -> (C, H, W) and normalize to [0, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        return img_tensor
    
    def _load_flow(self, flow_path: str) -> torch.Tensor:
        """
        Load flow from PNG file.
        
        Virtual KITTI 2 stores flow in a similar format to KITTI:
        - 16-bit PNG with 3 channels
        - First two channels encode flow (u, v)
        - Third channel is validity mask
        """
        flow_img = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED)
        if flow_img is None:
            raise ValueError(f"Failed to load flow: {flow_path}")
        
        # Flow is encoded as: (flow_value * 64) + 2^15
        # Extract u and v components
        h, w = flow_img.shape[:2]
        
        if flow_img.ndim == 3:
            # 3-channel flow
            u = flow_img[:, :, 2].astype(np.float32)
            v = flow_img[:, :, 1].astype(np.float32)
            valid = flow_img[:, :, 0].astype(np.float32)
        else:
            raise ValueError(f"Unexpected flow format: {flow_img.shape}")
        
        # Decode flow values: (value - 2^15) / 64
        u = (u - 2**15) / 64.0
        v = (v - 2**15) / 64.0
        
        # Create flow tensor (2, H, W)
        flow = np.stack([u, v], axis=0).astype(np.float32)
        flow = torch.from_numpy(flow)
        
        # Mark invalid pixels as inf (following KITTI convention)
        invalid_mask = (valid == 0)
        flow[:, invalid_mask] = float('inf')
        
        return flow
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            - 'src_img': Source image [3, H, W]
            - 'trg_img': Target image [3, H, W]
            - 'flow': Flow [2, H, W]
        """
        sample = self.samples[idx]
        
        if self.reverse_flow:
            src_img_path = sample['img2']
            trg_img_path = sample['img1']
        else:
            src_img_path = sample['img1']
            trg_img_path = sample['img2']
        
        src_img = self._load_image(src_img_path)
        trg_img = self._load_image(trg_img_path)
        flow = self._load_flow(sample['flow'])
        
        return {
            'src_img': src_img,
            'trg_img': trg_img,
            'flow': flow,
        }
