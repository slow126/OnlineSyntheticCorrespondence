"""
TSS (Transferring Semantic Segmentation) Dataset for evaluation with CATs++ model.
This dataset reads TSS dataset format and converts it to the format expected by CATs validation.
"""

from pathlib import Path
from typing import Union, Optional, Dict
import random

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.functional import interpolate
from torchvision import transforms
from torchvision.transforms.functional import normalize

from src.io import read_flo_file


class TSSSimpleDataset(Dataset):
    """
    TSS Dataset for training. Returns full resolution flow in pixel space.
    
    Returns:
        - src_img: Source image [3, H, W]
        - trg_img: Target image [3, H, W]
        - flow: Flow [2, H, W] in pixel space (full resolution)
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        reverse_flow: bool = False,
    ):
        """
        Initialize TSS dataset.
        
        Args:
            root: Root directory of TSS dataset
            reverse_flow: If True, reverse flow direction
        """
        self.root = Path(root)
        self.reverse_flow = reverse_flow
        
        # Load dataset pairs
        self.labels = {}
        self.pairs = []
        
        idx = 0
        for sub in sorted(self.root.iterdir()):
            if not sub.is_dir():
                continue
            self.labels[sub.name] = idx
            idx += 1
            pair_dirs = sorted(sub.iterdir())
            self.pairs.extend(pair_dirs)
    
    def __len__(self):
        return len(self.pairs)
    
    def _read_image(self, path: Path, name: str) -> torch.Tensor:
        """Read image and convert to tensor [3, H, W] in [0, 1] range"""
        img = Image.open(path.joinpath(name)).convert('RGB')
        # Convert to tensor: (H, W, C) -> (C, H, W) and normalize to [0, 1]
        # Use .contiguous() to ensure tensor can be resized during collation
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().contiguous() / 255.0
        return img_tensor
    
    def _read_flow(self, path: Path, name: str) -> torch.Tensor:
        """Read flow file and return in pixel space [2, H, W]"""
        flow = read_flo_file(path.joinpath(name))
        h, w = flow.shape[:2]
        
        # Use .contiguous() to ensure tensor can be resized during collation
        flow = torch.from_numpy(flow).moveaxis(-1, 0).contiguous()  # [2, H, W]
        
        # Mark invalid flow (typically > 1e9)
        flow[flow > 1e9] = torch.inf
        
        return flow
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary containing:
                - 'src_img': Source image [3, H, W] in [0, 1] range
                - 'trg_img': Target image [3, H, W] in [0, 1] range
                - 'flow': Flow [2, H, W] in pixel space (full resolution)
        """
        pair_dir = self.pairs[idx]
        
        # Read images
        src_img = self._read_image(pair_dir, 'image1.png')
        trg_img = self._read_image(pair_dir, 'image2.png')
        
        # Read flow (flow2 goes from image2 to image1, which is target to source)
        flow = self._read_flow(pair_dir, 'flow2.flo')
        
        # Reverse flow if requested
        if self.reverse_flow:
            flow = -flow
        
        sample = {
            'src_img': src_img,
            'trg_img': trg_img,
            'flow': flow,  # Full resolution flow in pixel space
        }
        
        return sample


class TSSDataset(Dataset):
    """
    TSS Dataset for CATs++ evaluation.
    
    This dataset reads TSS format (images and flow files) and converts to the format
    expected by CATs validation pipeline:
    - Extracts keypoints from real TSS flow files (more accurate than keypoint annotations)
    - Creates downsampled flow from keypoints (matching other datasets like SPair, PFPascal)
    - Provides all fields needed for validation:
      * src_img, trg_img: normalized image tensors
      * flow: downsampled flow [2, feature_size, feature_size] for EPE loss
      * src_kps, trg_kps: keypoints [2, max_pts] extracted from real TSS flow
      * n_pts: number of valid keypoints
      * pckthres: PCK threshold for evaluation
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        device: str = 'cuda',
        size: int = 512,
        feature_size: int = 32,
        max_pts: int = 40,
        thres: str = 'img',
        augmentation: bool = False,
        sample_keypoints: bool = True,
        num_keypoints: Optional[int] = None,
    ):
        """
        Args:
            root: Root directory of TSS dataset
            device: Device to put tensors on ('cuda' or 'cpu')
            size: Image size (default: 512)
            feature_size: Feature size for downsampled flow (default: 32)
            max_pts: Maximum number of keypoints (default: 40)
            thres: PCK threshold type ('img' or 'bbox')
            augmentation: Whether to apply augmentation (should be False for validation)
            sample_keypoints: Whether to sample keypoints from valid flow regions
            num_keypoints: Number of keypoints to sample (if None, uses max_pts)
        """
        self.root = Path(root)
        self.device = device
        self.size = size
        self.feature_size = feature_size
        self.max_pts = max_pts
        self.thres = thres
        self.augmentation = augmentation
        self.sample_keypoints = sample_keypoints
        self.num_keypoints = num_keypoints if num_keypoints is not None else max_pts
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Load dataset pairs
        self.labels = {}
        self.pairs = []
        self.flipped = []
        
        idx = 0
        for sub in sorted(self.root.iterdir()):
            if not sub.is_dir():
                continue
            self.labels[sub.name] = idx
            idx += 1
            pair_dirs = sorted(sub.iterdir())
            self.pairs.extend(pair_dirs)
        
        # Read flipped flags
        for p in self.pairs:
            flip_file = p.joinpath('flip_gt.txt')
            if flip_file.exists():
                self.flipped.append(int(flip_file.open().read()))
            else:
                self.flipped.append(0)
        
        # Initialize KeypointToFlow converter to create downsampled flow from keypoints
        # (This matches how other datasets like SPair, PFPascal work)
        try:
            from models.CATs_PlusPlus.data.keypoint_to_flow import KeypointToFlow
            self.kps_to_flow = KeypointToFlow(
                receptive_field_size=35,
                jsz=self.size // self.feature_size,
                feat_size=self.feature_size,
                img_size=self.size
            )
        except ImportError:
            self.kps_to_flow = None
    
    def __len__(self):
        return len(self.pairs)
    
    def _read_image(self, path: Path, name: str) -> torch.Tensor:
        """Read and normalize image"""
        img = Image.open(path.joinpath(name)).convert('RGB')
        img = self.transform(img)
        return img
    
    def _read_flow(self, path: Path, name: str) -> torch.Tensor:
        """Read flow file and resize to target size"""
        flow = read_flo_file(path.joinpath(name))
        h, w = flow.shape[:2]
        
        flow = torch.from_numpy(flow).moveaxis(-1, 0)  # [2, H, W]
        
        # Resize flow
        if self.size is not None:
            flow = interpolate(flow.unsqueeze(0), (self.size, self.size), mode='nearest-exact').squeeze(0)
            # Scale flow values to match new size
            flow[0] *= (self.size / w)
            flow[1] *= (self.size / h)
        
        # Mark invalid flow (typically > 1e9)
        flow[flow > 1e9] = torch.inf
        
        return flow
    
    def _sample_keypoints_from_flow(
        self,
        flow: torch.Tensor,
        num_kps: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample keypoints from valid flow regions.
        
        Args:
            flow: Flow tensor [2, H, W]
            num_kps: Number of keypoints to sample
            
        Returns:
            trg_kps: Target keypoints [2, num_kps]
            src_kps: Source keypoints [2, num_kps] (computed from flow)
        """
        _, h, w = flow.shape
        
        # Find valid flow regions (not inf)
        flow_mag = flow.norm(dim=0)
        valid_mask = flow_mag.isfinite() & (flow_mag > 0)
        
        if not valid_mask.any():
            # Fallback: sample uniformly
            y_coords = torch.randint(0, h, (num_kps,))
            x_coords = torch.randint(0, w, (num_kps,))
            trg_kps = torch.stack([x_coords.float(), y_coords.float()])
            # Use flow to compute source keypoints
            src_kps = trg_kps.clone()
            for i in range(num_kps):
                x, y = int(x_coords[i]), int(y_coords[i])
                if valid_mask[y, x]:
                    src_kps[0, i] = trg_kps[0, i] - flow[0, y, x]
                    src_kps[1, i] = trg_kps[1, i] - flow[1, y, x]
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
        
        trg_kps = torch.stack([sampled_x.float(), sampled_y.float()])  # [2, num_kps]
        
        # Compute source keypoints using flow
        # Flow goes from target to source, so: src_kp = trg_kp + flow(trg_kp)
        # (flow2[x, y] = displacement from image2 to image1)
        src_kps = torch.zeros_like(trg_kps)
        for i in range(len(indices)):
            y, x = int(sampled_y[i]), int(sampled_x[i])
            if y < flow.shape[1] and x < flow.shape[2]:
                src_kps[0, i] = trg_kps[0, i] + flow[0, y, x]
                src_kps[1, i] = trg_kps[1, i] + flow[1, y, x]
            else:
                src_kps[:, i] = trg_kps[:, i]  # Fallback if out of bounds
        
        return trg_kps, src_kps
    
    def _get_pckthres(self, imsize: tuple) -> torch.Tensor:
        """Compute PCK threshold (returns single value tensor, will be batched by DataLoader)"""
        if self.thres == 'img':
            return torch.tensor(max(imsize[0], imsize[1]), dtype=torch.float32)
        elif self.thres == 'bbox':
            # For TSS, we don't have bboxes, so use image size
            return torch.tensor(max(imsize[0], imsize[1]), dtype=torch.float32)
        else:
            raise ValueError(f"Unknown threshold type: {self.thres}")
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        pair_dir = self.pairs[idx]
        
        # Read images
        src_img = self._read_image(pair_dir, 'image1.png')
        trg_img = self._read_image(pair_dir, 'image2.png')
        
        # Read flow (flow2 goes from image2 to image1, which is target to source)
        # For CATs evaluation, we need flow from target (image2) to source (image1)
        flow_full = self._read_flow(pair_dir, 'flow2.flo')
        
        # Sample keypoints from valid flow regions
        if self.sample_keypoints:
            trg_kps, src_kps = self._sample_keypoints_from_flow(flow_full, self.num_keypoints)
            n_pts = min(self.num_keypoints, len(trg_kps[0]))
        else:
            # Fallback: uniform grid
            grid_size = int(np.sqrt(self.num_keypoints))
            y_coords = torch.linspace(0, self.size - 1, grid_size)
            x_coords = torch.linspace(0, self.size - 1, grid_size)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            trg_kps = torch.stack([xx.flatten(), yy.flatten()])[:, :self.num_keypoints]
            
            # Compute src_kps from flow
            # Flow goes from target to source, so: src_kp = trg_kp + flow(trg_kp)
            src_kps = torch.zeros_like(trg_kps)
            for i in range(self.num_keypoints):
                x, y = int(trg_kps[0, i]), int(trg_kps[1, i])
                if y < self.size and x < self.size and y >= 0 and x >= 0:
                    src_kps[0, i] = trg_kps[0, i] + flow_full[0, y, x]
                    src_kps[1, i] = trg_kps[1, i] + flow_full[1, y, x]
                else:
                    src_kps[:, i] = trg_kps[:, i]  # Fallback if out of bounds
            n_pts = self.num_keypoints
        
        # Pad keypoints to max_pts
        if n_pts < self.max_pts:
            pad_size = self.max_pts - n_pts
            trg_kps = torch.cat([trg_kps, torch.ones(2, pad_size) * -1], dim=1)
            src_kps = torch.cat([src_kps, torch.ones(2, pad_size) * -1], dim=1)
        
        # Create downsampled flow from keypoints (matching other datasets)
        # This is needed for the EPE loss computation in validation
        if self.kps_to_flow is not None:
            # KeypointToFlow expects [2, N] format (not batched), and n_pts as a single tensor
            batch_for_flow = {
                'src_kps': src_kps,  # [2, max_pts]
                'trg_kps': trg_kps,  # [2, max_pts]
                'n_pts': torch.tensor(n_pts)
            }
            flow_downsampled = self.kps_to_flow(batch_for_flow)  # [2, feature_size, feature_size]
        else:
            # Fallback: simple downsampling of full flow
            flow_downsampled = interpolate(
                flow_full.unsqueeze(0), 
                (self.feature_size, self.feature_size), 
                mode='bilinear'
            ).squeeze(0)
            # Scale flow values
            flow_downsampled[0] *= (self.feature_size / self.size)
            flow_downsampled[1] *= (self.feature_size / self.size)
        
        # Get PCK threshold (should be a tensor for batch indexing)
        pckthres = self._get_pckthres((self.size, self.size))
        
        # Get label
        label = self.labels[pair_dir.parent.name]
        flipped = self.flipped[idx]
        
        batch = {
            'src_img': src_img,
            'trg_img': trg_img,
            'flow': flow_downsampled,  # Downsampled flow [2, feature_size, feature_size] for EPE loss
            'src_kps': src_kps,  # [2, max_pts] - extracted from real TSS flow
            'trg_kps': trg_kps,  # [2, max_pts] - extracted from real TSS flow
            'n_pts': torch.tensor(n_pts),
            'pckthres': pckthres,
            'label': torch.tensor(label),
            'flipped': torch.tensor(flipped),
            'src_imname': pair_dir.joinpath('image1.png').name,
            'trg_imname': pair_dir.joinpath('image2.png').name,
            'src_imsize': (self.size, self.size),
            'trg_imsize': (self.size, self.size),
            'category_id': torch.tensor(label),
            'category': pair_dir.parent.name,
            'datalen': len(self.pairs),
            'flow_full': flow_full,
        }
        
        return batch

