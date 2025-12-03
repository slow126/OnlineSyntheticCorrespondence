"""
KITTI Dataset for CATs++ training and evaluation.

Supports both KITTI-2012 and KITTI-2015 datasets with auto-detection.
Extracts keypoints from flow for validation (KITTI doesn't provide keypoints natively).
"""

import os
import os.path
from pathlib import Path
from typing import Optional, Tuple, Dict
import random

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize, resize
from torchvision.transforms import InterpolationMode

# Import KITTI flow loading utilities
from src.data.real.datasets.optical_flow.kitti import load_flow_from_png, make_dataset
from src.data.synth.datasets.FlyingThingsDataset import FlowDownsampler, FlowAwareResize


class KittiSimpleDataset(Dataset):
    """
    KITTI-2012 and KITTI-2015 dataset for CATs++. This should replace the Bespoke Flow Dataset.
    
    Returns:
        - src_img: Source image (first frame)
        - trg_img: Target image (second frame) 
        - flow: Optical flow from trg to src (dx, dy)
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        version: str = 'auto',
        occ_type: str = 'occ',
        reverse_flow: bool = False,
    ):
        """
        Initialize KITTI dataset.
        
        Args:
            root: Path to kitti-2012 or kitti-2015 folder (should contain 'training' subdirectory)
            split: 'train' or 'val'
            version: 'auto', '2012', or '2015' (auto-detects by checking for colored_0 vs image_2)
            occ_type: 'noc', 'occ', or 'only_occ' (type of flow to use)
            size: Optional (H, W) tuple to resize images
            downsample_flow: Optional feat_size for CATS downsampling (e.g., 32)
            normalize: If True, applies ImageNet normalization to images
            normalize_images: If True, returns validation format with keypoints
            thres: PCK threshold type ('img' or 'bbox')
            max_pts: Maximum number of keypoints for validation
            split_ratio: Train/val split ratio for old structure with 'training' directory (default 0.8)
                         Not used if 'train'/'val' directories exist (files are already split)
        """
        self.root = Path(root)
        self.split = split
        self.occ_type = occ_type
        self.reverse_flow = reverse_flow
        
        # Build dataset file list
        # Support both old structure (training directory with split) and new structure (train/val directories)
        # Check for new structure first (train/val directories)
        train_dir = self.root / 'train'
        val_dir = self.root / 'val'
        training_dir = self.root / 'training'
        
        # Auto-detect version if needed
        if version == 'auto':
            # Try new structure first (train/val)
            if train_dir.exists():
                if (train_dir / 'colored_0').exists():
                    version = '2012'
                elif (train_dir / 'image_2').exists():
                    version = '2015'
            # Fall back to old structure (training)
            elif training_dir.exists():
                if (training_dir / 'colored_0').exists():
                    version = '2012'
                elif (training_dir / 'image_2').exists():
                    version = '2015'
            
            if version == 'auto':
                raise ValueError(f"Cannot auto-detect KITTI version. Checked for 'colored_0' (2012) or 'image_2' (2015) in train/val or training directories")
        
        self.version = version
        
        if train_dir.exists() and val_dir.exists():
            # New structure: separate train/val directories with files already split
            # Files are physically separated, no split logic needed
            print(f"Using new structure: train/val directories")
            if split == 'train':
                data_dir = train_dir
                self.data_dir_name = 'train'
            elif split == 'val':
                data_dir = val_dir
                self.data_dir_name = 'val'
            else:
                raise ValueError(f"split must be 'train' or 'val', got '{split}'")
            
            # Use make_dataset to get file list (all files in directory, split=1.0)
            occ = (occ_type == 'occ')
            only_occ = (occ_type == 'only_occ')
            
            file_list, _ = make_dataset(str(data_dir), split=1.0, occ=occ, only_occ=only_occ)
            self.file_list = file_list
        elif training_dir.exists():
            # Old structure: training directory with split ratio
            # Use make_dataset to get file list with split ratio
            occ = (occ_type == 'occ')
            only_occ = (occ_type == 'only_occ')
            if split == 'train':
                split = 'training'
            
            # Special case: split='training' means use all data from training directory
            if split == 'training':
                # Use split=0.0 to get all samples in val_list (we'll use it for validation)
                _, all_list = make_dataset(str(training_dir), split=0.0, occ=occ, only_occ=only_occ)
                self.file_list = all_list
                self.data_dir_name = 'training'
                print(f"Using full training set: {len(self.file_list)} samples")
            else:
                train_list, val_list = make_dataset(str(training_dir), split=split_ratio, occ=occ, only_occ=only_occ)
                
                # Select appropriate split
                if split == 'train':
                    self.file_list = train_list
                    self.data_dir_name = 'training'
                elif split == 'val':
                    self.file_list = val_list
                    self.data_dir_name = 'training'
                else:
                    raise ValueError(f"split must be 'train', 'val', or 'training', got '{split}'")
        else:
            raise ValueError(f"Neither 'train'/'val' directories nor 'training' directory found in {self.root}")
        
        print(f"KITTI-{version} dataset initialized: {len(self.file_list)} samples in '{split}' split (occ_type={occ_type})")
    
    def __len__(self):
        return len(self.file_list)
    
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
    
    def _load_flow(self, flow_path) -> torch.Tensor:
        """Load flow from PNG file."""
        if isinstance(flow_path, (list, tuple)):
            # For only_occ mode, we get both flow_occ and flow_noc
            flow_path = flow_path[0]  # Use flow_occ
        
        full_path = self.root / self.data_dir_name / flow_path
        flow_np, valid_mask = load_flow_from_png(str(full_path))
        
        # Convert to torch tensor: (H, W, 2) -> (2, H, W)
        flow = torch.from_numpy(flow_np).permute(2, 0, 1).float()
        
        # Invalid pixels are already inf, keep them
        return flow

    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Training format (normalize_images=False):
                - 'src_img': Source image [3, H, W]
                - 'trg_img': Target image [3, H, W]
                - 'flow': Flow [2, H, W] or [2, feat_size, feat_size] if downsampled
            
            Validation format (normalize_images=True):
                - All training fields plus:
                - 'src_kps': Source keypoints [2, max_pts]
                - 'trg_kps': Target keypoints [2, max_pts]
                - 'n_pts': Number of valid keypoints
                - 'pckthres': PCK threshold
                - 'src_imsize': Source image size (H, W)
                - 'trg_imsize': Target image size (H, W)
                - 'datalen': Dataset length
        """
        # Get file paths
        img_paths, flow_path = self.file_list[idx]
        
        # img_paths[0] is source (img1), img_paths[1] is target (img2)
        # Flow is from target to source (matches our convention)
        # Load images
        if self.reverse_flow:
            src_img_index = 1
            trg_img_index = 0
        else:
            src_img_index = 0
            trg_img_index = 1

        src_img_path = str(self.root / self.data_dir_name / img_paths[src_img_index])
        trg_img_path = str(self.root / self.data_dir_name / img_paths[trg_img_index])
        

        
        src_img = self._load_image(src_img_path)
        trg_img = self._load_image(trg_img_path)
        
        # Load flow
        flow = self._load_flow(flow_path)
        
        # Create sample dict
        sample = {
            'src_img': src_img,
            'trg_img': trg_img,
            'flow': flow,
        }
        
        return sample


class KittiDataset(Dataset):
    """
    KITTI-2012 and KITTI-2015 dataset for CATs++.
    
    Supports both training and validation modes:
    - Training: Returns {'src_img', 'trg_img', 'flow'}
    - Validation: Returns {'src_img', 'trg_img', 'flow', 'src_kps', 'trg_kps', 'n_pts', 'pckthres', 'src_imsize', 'trg_imsize', 'datalen'}
    
    Keypoints are extracted from flow for validation (KITTI doesn't provide keypoints natively).
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        version: str = 'auto',
        occ_type: str = 'occ',
        size: Optional[Tuple[int, int]] = None,
        downsample_flow: Optional[int] = None,
        normalize: bool = True,
        normalize_images: bool = False,
        thres: str = 'img',
        max_pts: int = 200,
        split_ratio: float = 0.8,
    ):
        """
        Initialize KITTI dataset.
        
        Args:
            root: Path to kitti-2012 or kitti-2015 folder (should contain 'training' subdirectory)
            split: 'train' or 'val'
            version: 'auto', '2012', or '2015' (auto-detects by checking for colored_0 vs image_2)
            occ_type: 'noc', 'occ', or 'only_occ' (type of flow to use)
            size: Optional (H, W) tuple to resize images
            downsample_flow: Optional feat_size for CATS downsampling (e.g., 32)
            normalize: If True, applies ImageNet normalization to images
            normalize_images: If True, returns validation format with keypoints
            thres: PCK threshold type ('img' or 'bbox')
            max_pts: Maximum number of keypoints for validation
            split_ratio: Train/val split ratio for old structure with 'training' directory (default 0.8)
                         Not used if 'train'/'val' directories exist (files are already split)
        """
        self.root = Path(root)
        self.split = split
        self.occ_type = occ_type
        self.size = size
        self.downsample_flow = downsample_flow
        self.normalize = normalize
        self.normalize_images = normalize_images
        self.thres = thres
        self.max_pts = max_pts
        
        # Build dataset file list
        # Support both old structure (training directory with split) and new structure (train/val directories)
        # Check for new structure first (train/val directories)
        train_dir = self.root / 'train'
        val_dir = self.root / 'val'
        training_dir = self.root / 'training'
        
        # Auto-detect version if needed
        if version == 'auto':
            # Try new structure first (train/val)
            if train_dir.exists():
                if (train_dir / 'colored_0').exists():
                    version = '2012'
                elif (train_dir / 'image_2').exists():
                    version = '2015'
            # Fall back to old structure (training)
            elif training_dir.exists():
                if (training_dir / 'colored_0').exists():
                    version = '2012'
                elif (training_dir / 'image_2').exists():
                    version = '2015'
            
            if version == 'auto':
                raise ValueError(f"Cannot auto-detect KITTI version. Checked for 'colored_0' (2012) or 'image_2' (2015) in train/val or training directories")
        
        self.version = version
        
        if train_dir.exists() and val_dir.exists():
            # New structure: separate train/val directories with files already split
            # Files are physically separated, no split logic needed
            print(f"Using new structure: train/val directories")
            if split == 'train':
                data_dir = train_dir
                self.data_dir_name = 'train'
            elif split == 'val':
                data_dir = val_dir
                self.data_dir_name = 'val'
            else:
                raise ValueError(f"split must be 'train' or 'val', got '{split}'")
            
            # Use make_dataset to get file list (all files in directory, split=1.0)
            occ = (occ_type == 'occ')
            only_occ = (occ_type == 'only_occ')
            
            file_list, _ = make_dataset(str(data_dir), split=1.0, occ=occ, only_occ=only_occ)
            self.file_list = file_list
        elif training_dir.exists():
            # Old structure: training directory with split ratio
            # Use make_dataset to get file list with split ratio
            occ = (occ_type == 'occ')
            only_occ = (occ_type == 'only_occ')
            
            # Special case: split='training' means use all data from training directory
            if split == 'training':
                # Use split=0.0 to get all samples in val_list (we'll use it for validation)
                _, all_list = make_dataset(str(training_dir), split=0.0, occ=occ, only_occ=only_occ)
                self.file_list = all_list
                self.data_dir_name = 'training'
                print(f"Using full training set: {len(self.file_list)} samples")
            else:
                train_list, val_list = make_dataset(str(training_dir), split=split_ratio, occ=occ, only_occ=only_occ)
                
                # Select appropriate split
                if split == 'train':
                    self.file_list = train_list
                    self.data_dir_name = 'training'
                elif split == 'val':
                    self.file_list = val_list
                    self.data_dir_name = 'training'
                else:
                    raise ValueError(f"split must be 'train', 'val', or 'training', got '{split}'")
        else:
            raise ValueError(f"Neither 'train'/'val' directories nor 'training' directory found in {self.root}")
        
        # Create resize transform if size is specified
        if size is not None:
            self.resize_transform = FlowAwareResize(size)
        else:
            self.resize_transform = None
        
        # Create flow downsampler if specified
        if downsample_flow is not None:
            self.flow_downsampler = FlowDownsampler(downsample_flow)
        else:
            self.flow_downsampler = None
        
        print(f"KITTI-{version} dataset initialized: {len(self.file_list)} samples in '{split}' split (occ_type={occ_type})")
    
    def __len__(self):
        return len(self.file_list)
    
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
    
    def _load_flow(self, flow_path) -> torch.Tensor:
        """Load flow from PNG file."""
        if isinstance(flow_path, (list, tuple)):
            # For only_occ mode, we get both flow_occ and flow_noc
            flow_path = flow_path[0]  # Use flow_occ
        
        full_path = self.root / self.data_dir_name / flow_path
        flow_np, valid_mask = load_flow_from_png(str(full_path))
        
        # Convert to torch tensor: (H, W, 2) -> (2, H, W)
        flow = torch.from_numpy(flow_np).permute(2, 0, 1).float()
        
        # Invalid pixels are already inf, keep them
        return flow
    
    def _sample_keypoints_from_flow(
        self,
        flow: torch.Tensor,
        num_kps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample keypoints from valid flow regions.
        
        Args:
            flow: Flow tensor [2, H, W]
            num_kps: Number of keypoints to sample
            
        Returns:
            trg_kps: Target keypoints [2, num_kps] (x, y format)
            src_kps: Source keypoints [2, num_kps] (computed from flow)
        """
        _, h, w = flow.shape
        
        # Find valid flow regions (not inf and non-zero magnitude)
        flow_mag = flow.norm(dim=0)
        valid_mask = flow_mag.isfinite() & (flow_mag > 0)
        
        if not valid_mask.any():
            # Fallback: sample uniformly (shouldn't happen with KITTI, but handle gracefully)
            y_coords = torch.randint(0, h, (num_kps,))
            x_coords = torch.randint(0, w, (num_kps,))
            trg_kps = torch.stack([x_coords.float(), y_coords.float()])
            src_kps = trg_kps.clone()  # No flow, so source = target
            return trg_kps, src_kps
        
        # Sample from valid regions
        valid_y, valid_x = torch.where(valid_mask)
        num_valid = len(valid_y)
        
        if num_valid <= num_kps:
            # Use all valid points
            indices = torch.arange(num_valid)
        else:
            # Randomly sample
            indices = torch.randperm(num_valid)[:num_kps]
        
        sampled_y = valid_y[indices]
        sampled_x = valid_x[indices]
        
        trg_kps = torch.stack([sampled_x.float(), sampled_y.float()])  # [2, num_kps] (x, y)
        
        # Compute source keypoints using flow
        # Flow goes from target to source, so: src_kp = trg_kp + flow(trg_kp)
        src_kps = torch.zeros_like(trg_kps)
        for i in range(len(indices)):
            y, x = int(sampled_y[i]), int(sampled_x[i])
            if y < flow.shape[1] and x < flow.shape[2]:
                src_kps[0, i] = trg_kps[0, i] + flow[0, y, x]
                src_kps[1, i] = trg_kps[1, i] + flow[1, y, x]
            else:
                src_kps[:, i] = trg_kps[:, i]  # Fallback if out of bounds
        
        return trg_kps, src_kps
    
    def _get_pckthres(self, imsize: Tuple[int, int]) -> torch.Tensor:
        """Get PCK threshold based on image size."""
        if self.thres == 'img':
            return torch.tensor(max(imsize), dtype=torch.float32)
        else:
            # Default to image size
            return torch.tensor(max(imsize), dtype=torch.float32)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Training format (normalize_images=False):
                - 'src_img': Source image [3, H, W]
                - 'trg_img': Target image [3, H, W]
                - 'flow': Flow [2, H, W] or [2, feat_size, feat_size] if downsampled
            
            Validation format (normalize_images=True):
                - All training fields plus:
                - 'src_kps': Source keypoints [2, max_pts]
                - 'trg_kps': Target keypoints [2, max_pts]
                - 'n_pts': Number of valid keypoints
                - 'pckthres': PCK threshold
                - 'src_imsize': Source image size (H, W)
                - 'trg_imsize': Target image size (H, W)
                - 'datalen': Dataset length
        """
        # Get file paths
        img_paths, flow_path = self.file_list[idx]
        
        # img_paths[0] is source (img1), img_paths[1] is target (img2)
        # Flow is from target to source (matches our convention)
        src_img_path = str(self.root / self.data_dir_name / img_paths[0])
        trg_img_path = str(self.root / self.data_dir_name / img_paths[1])
        
        # Load images
        src_img = self._load_image(src_img_path)
        trg_img = self._load_image(trg_img_path)
        
        # Load flow
        flow = self._load_flow(flow_path)
        full_flow = flow.clone()
        
        # Get original image size before any transforms
        _, orig_h, orig_w = src_img.shape
        orig_imsize = (orig_h, orig_w)
        
        # Create sample dict
        sample = {
            'src_img': src_img,
            'trg_img': trg_img,
            'flow': flow,
        }
        
        # Apply resize transform if specified
        if self.resize_transform is not None:
            sample = self.resize_transform(sample)
        
        # Apply ImageNet normalization if requested
        if self.normalize:
            src_img = sample['src_img']
            trg_img = sample['trg_img']
            src_img = normalize(src_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            trg_img = normalize(trg_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            sample['src_img'] = src_img
            sample['trg_img'] = trg_img
        
        # Extract keypoints for validation mode (before downsampling)
        if self.normalize_images:
            # Sample keypoints from flow after resize but before downsampling
            # This ensures keypoints are in the resized image coordinate space
            # Use the current flow (already resized if resize was applied, but not yet downsampled)
            flow_for_kps = sample['flow']
            trg_kps, src_kps = self._sample_keypoints_from_flow(flow_for_kps, self.max_pts)
            
            # Pad/truncate to max_pts
            n_valid = min(trg_kps.shape[1], self.max_pts)
            if trg_kps.shape[1] < self.max_pts:
                pad_size = self.max_pts - trg_kps.shape[1]
                trg_kps = torch.cat([trg_kps, torch.zeros(2, pad_size, dtype=torch.float32)], dim=1)
                src_kps = torch.cat([src_kps, torch.zeros(2, pad_size, dtype=torch.float32)], dim=1)
            elif trg_kps.shape[1] > self.max_pts:
                trg_kps = trg_kps[:, :self.max_pts]
                src_kps = src_kps[:, :self.max_pts]
                n_valid = self.max_pts
        
        # Apply flow downsampling if specified (after keypoint extraction)
        if self.flow_downsampler is not None:
            sample['flow'] = self.flow_downsampler(sample['flow'])
        
        # Add validation fields if in validation mode
        if self.normalize_images:
            # Get image size after transforms
            src_img_final = sample['src_img']
            if src_img_final.ndim == 3:
                C, H, W = src_img_final.shape
                img_size_tuple = (H, W)
            else:
                H, W = src_img_final.shape[-2:]
                img_size_tuple = (H, W)
            
            # Get PCK threshold
            pckthres = self._get_pckthres(img_size_tuple)
            
            # Add validation fields
            sample['src_kps'] = src_kps
            sample['trg_kps'] = trg_kps
            sample['n_pts'] = torch.tensor(n_valid, dtype=torch.int32)
            sample['pckthres'] = pckthres
            sample['src_imsize'] = img_size_tuple
            sample['trg_imsize'] = img_size_tuple
            sample['datalen'] = torch.tensor(len(self), dtype=torch.int32)
        
        return sample


if __name__ == "__main__":
    # Test dataset
    import sys
    
    root = "/home/spencer/Data/correspondence/kitti-2012"
    if len(sys.argv) > 1:
        root = sys.argv[1]
    
    print(f"Testing KITTI dataset with root: {root}")
    
    # Test training mode
    print("\n=== Training Mode ===")
    train_dataset = KittiDataset(
        root=root,
        split='train',
        version='auto',
        occ_type='occ',
        size=(512, 512),
        downsample_flow=32,
        normalize=True,
        normalize_images=False
    )
    
    sample = train_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"src_img shape: {sample['src_img'].shape}")
    print(f"trg_img shape: {sample['trg_img'].shape}")
    print(f"flow shape: {sample['flow'].shape}")
    
    # Test validation mode
    print("\n=== Validation Mode ===")
    val_dataset = KittiDataset(
        root=root,
        split='val',
        version='auto',
        occ_type='occ',
        size=(512, 512),
        downsample_flow=32,
        normalize=True,
        normalize_images=True,
        max_pts=200
    )
    
    val_sample = val_dataset[0]
    print(f"Sample keys: {val_sample.keys()}")
    print(f"src_img shape: {val_sample['src_img'].shape}")
    print(f"trg_img shape: {val_sample['trg_img'].shape}")
    print(f"flow shape: {val_sample['flow'].shape}")
    print(f"src_kps shape: {val_sample['src_kps'].shape}")
    print(f"trg_kps shape: {val_sample['trg_kps'].shape}")
    print(f"n_pts: {val_sample['n_pts'].item()}")
    print(f"pckthres: {val_sample['pckthres'].item()}")
    
    print("\nDataset test completed successfully!")

