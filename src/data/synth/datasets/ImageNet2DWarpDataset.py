"""
ImageNet 2D Warp Dataset for correspondence learning.

This dataset loads ImageNet100 images and applies 2D affine transformations
to generate correspondence pairs. Uses bilinear interpolation for warping.
"""

from pathlib import Path
from typing import Union, Optional, Dict, Tuple
import random
import pickle

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import kornia


class ImageNet2DWarpDataset(Dataset):
    """
    ImageNet 2D Warp Dataset for training.
    
    Applies affine transformations to ImageNet images to generate correspondence pairs.
    Uses bilinear interpolation for warping (no fancy methods).
    
    Returns:
        - src_img: Source image [3, H, W] in [0, 1] range
        - trg_img: Target (warped) image [3, H, W] in [0, 1] range
        - flow: Flow [2, H, W] in pixel space (full resolution)
            Flow convention: flow from trg to src, so flow = src_location - trg_location
            Invalid pixels marked with float('inf')
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        rotation_range: Tuple[float, float] = (-30.0, 30.0),  # degrees
        scale_range: Tuple[float, float] = (0.5, 2.5),  # as specified
        translation_range: Tuple[float, float] = (-0.1, 0.1),  # fraction of image size
        shear_range: Tuple[float, float] = (-0.2, 0.2),
        cache_warp_params: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize ImageNet 2D Warp dataset.
        
        Args:
            root: Root directory of ImageNet100 dataset
            split: 'train' or 'val'
            rotation_range: (min, max) rotation angle in degrees
            scale_range: (min, max) scale factor (default: 0.5 to 2.5)
            translation_range: (min, max) translation as fraction of image size
            shear_range: (min, max) shear factor
            cache_warp_params: If True, cache warp parameters for reproducibility
            cache_dir: Directory to cache warp parameters (default: root/cache)
            seed: Random seed for reproducibility
        """
        self.root = Path(root)
        self.split = split
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.shear_range = shear_range
        self.cache_warp_params = cache_warp_params
        
        # Set random seed
        self.rng = random.Random(seed) if seed is not None else random.Random()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Cache directory
        if cache_dir is None:
            cache_dir = self.root / "cache"
        self.cache_dir = Path(cache_dir)
        if self.cache_warp_params:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load image paths
        split_dir = self.root / split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        self.image_paths = []
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            # Find all images in this class
            for ext in ['*.JPEG', '*.jpg', '*.png']:
                self.image_paths.extend(class_dir.glob(ext))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {split_dir}")
        
        print(f"ImageNet2DWarpDataset: Found {len(self.image_paths)} images in {split} split")
        
        # Load or create warp parameter cache
        self.warp_params_cache = {}
        if self.cache_warp_params:
            cache_file = self.cache_dir / f"warp_params_{split}.pkl"
            if cache_file.exists():
                print(f"Loading warp parameters from {cache_file}")
                with open(cache_file, 'rb') as f:
                    self.warp_params_cache = pickle.load(f)
                print(f"Loaded {len(self.warp_params_cache)} cached warp parameters")
    
    def _sample_affine_params(self) -> Dict[str, float]:
        """Sample affine transformation parameters."""
        return {
            'rotation': self.rng.uniform(*self.rotation_range),
            'scale_x': self.rng.uniform(*self.scale_range),
            'scale_y': self.rng.uniform(*self.scale_range),
            'translation_x': self.rng.uniform(*self.translation_range),
            'translation_y': self.rng.uniform(*self.translation_range),
            'shear_x': self.rng.uniform(*self.shear_range),
            'shear_y': self.rng.uniform(*self.shear_range),
        }
    
    def _get_affine_matrix(
        self,
        params: Dict[str, float],
        img_h: int,
        img_w: int,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """
        Create affine transformation matrix from parameters.
        
        Args:
            params: Dictionary with rotation, scale_x, scale_y, translation_x, translation_y, shear_x, shear_y
            img_h: Image height
            img_w: Image width
            device: Device for tensor
        
        Returns:
            Affine matrix [3, 3]
        """
        # Convert rotation to radians
        angle_rad = np.deg2rad(params['rotation'])
        
        # Center of image
        center = torch.tensor([img_w / 2, img_h / 2], device=device)
        
        # Translation in pixels (from fraction of image size)
        translation = torch.tensor([
            params['translation_x'] * img_w,
            params['translation_y'] * img_h
        ], device=device)
        
        # Scale
        scale = torch.tensor([params['scale_x'], params['scale_y']], device=device)
        
        # Rotation angle
        angle = torch.tensor([angle_rad], device=device)
        
        # Shear
        shear = torch.tensor([params['shear_x'], params['shear_y']], device=device)
        
        # Create affine matrix using kornia
        affine_matrix = kornia.geometry.get_affine_matrix2d(
            translations=translation.unsqueeze(0),
            center=center.unsqueeze(0),
            scale=scale.unsqueeze(0),
            angle=angle,
            sx=shear[0:1],
            sy=shear[1:2],
        )
        
        return affine_matrix[0]  # Remove batch dimension
    
    def _compute_flow_from_affine(
        self,
        affine_matrix: torch.Tensor,
        img_h: int,
        img_w: int,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """
        Compute dense flow field from affine transformation using inverse warp.
        
        Flow convention: flow from trg to src, so flow = src_location - trg_location
        For each target pixel (x_t, y_t), we find the source location (x_s, y_s) using inverse transform.
        Then flow = (x_s - x_t, y_s - y_t).
        
        Args:
            affine_matrix: Affine transformation matrix [3, 3] (transforms source to target)
            img_h: Image height
            img_w: Image width
            device: Device for tensors
        
        Returns:
            Flow tensor [2, H, W] where flow[0] = dx, flow[1] = dy
            Invalid pixels marked with float('inf')
        """
        # Create coordinate grid for target image (normalized to [-1, 1])
        # Target pixels are at integer coordinates
        y_coords, x_coords = torch.meshgrid(
            torch.arange(img_h, dtype=torch.float32, device=device),
            torch.arange(img_w, dtype=torch.float32, device=device),
            indexing='ij'
        )
        
        # Stack to [H, W, 2] format (x, y)
        target_coords = torch.stack([x_coords, y_coords], dim=-1)  # [H, W, 2]
        
        # Convert to homogeneous coordinates [H, W, 3]
        target_coords_hom = torch.cat([
            target_coords,
            torch.ones(img_h, img_w, 1, device=device)
        ], dim=-1)
        
        # Compute inverse affine transform (from target to source)
        inv_affine = torch.linalg.inv(affine_matrix)  # [3, 3]
        
        # Apply inverse transform to get source coordinates
        # Reshape for matrix multiplication: [H*W, 3] @ [3, 3] -> [H*W, 3]
        target_flat = target_coords_hom.reshape(-1, 3)  # [H*W, 3]
        source_flat = (inv_affine @ target_flat.T).T  # [H*W, 3]
        source_coords = source_flat[:, :2]  # [H*W, 2]
        source_coords = source_coords.reshape(img_h, img_w, 2)  # [H, W, 2]
        
        # Compute flow: flow = src_location - trg_location
        flow = source_coords - target_coords  # [H, W, 2]
        
        # Mark out-of-bounds pixels as invalid
        # A pixel is invalid if its source location is outside the image bounds
        source_x = source_coords[..., 0]
        source_y = source_coords[..., 1]
        
        valid_mask = (
            (source_x >= 0) & (source_x < img_w) &
            (source_y >= 0) & (source_y < img_h)
        )
        
        # Convert to [2, H, W] format (dx, dy)
        flow = flow.permute(2, 0, 1)  # [2, H, W]
        
        # Mark invalid pixels with inf
        flow[:, ~valid_mask] = float('inf')
        
        return flow
    
    def _warp_image(
        self,
        img: torch.Tensor,
        affine_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Warp image using affine transformation with bilinear interpolation.
        
        Args:
            img: Image tensor [3, H, W] in [0, 1] range
            affine_matrix: Affine transformation matrix [3, 3]
        
        Returns:
            Warped image [3, H, W] in [0, 1] range
        """
        _, img_h, img_w = img.shape
        device = img.device
        
        # Create coordinate grid for target image in pixel coordinates
        # We'll work in pixel space first, then normalize for grid_sample
        y_coords, x_coords = torch.meshgrid(
            torch.arange(img_h, dtype=torch.float32, device=device),
            torch.arange(img_w, dtype=torch.float32, device=device),
            indexing='ij'
        )
        
        # Stack to [H, W, 2] format (x, y) in pixel coordinates
        target_coords = torch.stack([x_coords, y_coords], dim=-1)  # [H, W, 2]
        
        # Convert to homogeneous coordinates [H, W, 3]
        target_coords_hom = torch.cat([
            target_coords,
            torch.ones(img_h, img_w, 1, device=device)
        ], dim=-1)
        
        # Apply inverse affine transform to get source coordinates in pixel space
        inv_affine = torch.linalg.inv(affine_matrix)
        target_flat = target_coords_hom.reshape(-1, 3)  # [H*W, 3]
        source_flat = (inv_affine @ target_flat.T).T  # [H*W, 3]
        source_coords = source_flat[:, :2].reshape(img_h, img_w, 2)  # [H, W, 2] in pixel space
        
        # Normalize to [-1, 1] for grid_sample
        # grid_sample expects (x, y) where x is horizontal, y is vertical
        # (-1, -1) is top-left, (1, 1) is bottom-right
        source_grid_norm = source_coords.clone()
        source_grid_norm[..., 0] = 2.0 * source_coords[..., 0] / (img_w - 1) - 1.0  # x coordinate
        source_grid_norm[..., 1] = 2.0 * source_coords[..., 1] / (img_h - 1) - 1.0  # y coordinate
        
        # Warp image using bilinear interpolation
        img_batch = img.unsqueeze(0)  # [1, 3, H, W]
        grid_batch = source_grid_norm.unsqueeze(0)  # [1, H, W, 2]
        
        warped = F.grid_sample(
            img_batch,
            grid_batch,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        
        return warped.squeeze(0)  # [3, H, W]
    
    def _read_image(self, path: Path) -> torch.Tensor:
        """Read image and convert to tensor [3, H, W] in [0, 1] range."""
        img = Image.open(path).convert('RGB')
        # Convert to tensor: (H, W, C) -> (C, H, W) and normalize to [0, 1]
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().contiguous() / 255.0
        return img_tensor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary containing:
                - 'src_img': Source image [3, H, W] in [0, 1] range
                - 'trg_img': Target (warped) image [3, H, W] in [0, 1] range
                - 'flow': Flow [2, H, W] in pixel space (full resolution)
                    Flow convention: flow from trg to src
                    Invalid pixels marked with float('inf')
        """
        # Get image path
        img_path = self.image_paths[idx]
        
        # Load source image
        src_img = self._read_image(img_path)
        _, img_h, img_w = src_img.shape
        
        # Get or generate warp parameters
        if self.cache_warp_params and idx in self.warp_params_cache:
            params = self.warp_params_cache[idx]
        else:
            params = self._sample_affine_params()
            if self.cache_warp_params:
                self.warp_params_cache[idx] = params
        
        # Create affine matrix
        device = src_img.device
        affine_matrix = self._get_affine_matrix(params, img_h, img_w, device=device)
        
        # Compute flow from affine transform
        flow = self._compute_flow_from_affine(affine_matrix, img_h, img_w, device=device)
        
        # Warp image using bilinear interpolation
        trg_img = self._warp_image(src_img, affine_matrix)
        
        sample = {
            'src_img': src_img,
            'trg_img': trg_img,
            'flow': flow,  # Full resolution flow in pixel space
        }
        
        return sample
    
    def save_cache(self):
        """Save warp parameters cache to disk."""
        if self.cache_warp_params and len(self.warp_params_cache) > 0:
            cache_file = self.cache_dir / f"warp_params_{self.split}.pkl"
            print(f"Saving {len(self.warp_params_cache)} warp parameters to {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(self.warp_params_cache, f)

