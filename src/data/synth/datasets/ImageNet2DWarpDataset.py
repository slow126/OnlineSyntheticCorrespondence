"""
ImageNet 2D Warp Dataset for correspondence learning.

This dataset loads ImageNet100 images and applies 2D affine transformations
to generate correspondence pairs. Uses bilinear interpolation for warping.
"""

from pathlib import Path
from typing import Union, Optional, Dict, Tuple
import itertools
import importlib.util
import site
import sys
from importlib.machinery import PathFinder
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
        hf_dataset: Optional[str] = None,
        hf_split: Optional[str] = None,
        hf_cache_dir: Optional[Union[str, Path]] = None,
        hf_streaming: bool = False,
        hf_max_samples: Optional[int] = None,
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
        self.hf_dataset_name = None
        self.hf_dataset = None
        self.hf_samples = None
        self.hf_dataset_len = None
        root_str = str(root)
        if hf_dataset:
            self.hf_dataset_name = hf_dataset
        elif root_str.startswith("hf://"):
            self.hf_dataset_name = root_str[len("hf://"):]
        elif root_str.startswith("hf:"):
            self.hf_dataset_name = root_str[len("hf:"):]

        self.root = None if self.hf_dataset_name else Path(root)
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
            if self.hf_dataset_name:
                cache_dir = Path("./cache/imagenet2dwarp")
            else:
                cache_dir = self.root / "cache"
        self.cache_dir = Path(cache_dir)
        if self.cache_warp_params:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load image paths or HF dataset
        self.image_paths = []
        if self.hf_dataset_name:
            datasets_module = self._import_hf_datasets()
            load_dataset = datasets_module.load_dataset
            hf_split_name = hf_split or ("validation" if split in ("val", "validation") else split)
            streaming = bool(hf_streaming)
            try:
                self.hf_dataset = load_dataset(
                    self.hf_dataset_name,
                    split=hf_split_name,
                    cache_dir=str(hf_cache_dir) if hf_cache_dir else None,
                    streaming=streaming,
                )
                if streaming:
                    if hf_max_samples is None:
                        raise ValueError(
                            "HF streaming requires hf_max_samples to materialize examples "
                            "for random access."
                        )
                    self.hf_samples = list(itertools.islice(self.hf_dataset, int(hf_max_samples)))
                    self.hf_dataset = None
                    self.hf_dataset_len = len(self.hf_samples)
                    print(
                        f"ImageNet2DWarpDataset: Loaded HF dataset {self.hf_dataset_name} "
                        f"({hf_split_name}) streaming, cached {self.hf_dataset_len} samples"
                    )
                else:
                    if hf_max_samples is not None:
                        limit = min(int(hf_max_samples), len(self.hf_dataset))
                        self.hf_dataset = self.hf_dataset.select(range(limit))
                    self.hf_dataset_len = len(self.hf_dataset)
                    print(
                        f"ImageNet2DWarpDataset: Loaded HF dataset {self.hf_dataset_name} "
                        f"({hf_split_name}) with {self.hf_dataset_len} images"
                    )
            except (NotImplementedError, ConnectionError):
                self.hf_dataset = self._load_hf_from_cache(hf_split_name, hf_cache_dir, hf_max_samples)
                self.hf_dataset_len = len(self.hf_dataset)
                print(
                    f"ImageNet2DWarpDataset: Loaded HF dataset {self.hf_dataset_name} "
                    f"({hf_split_name}) from cached Arrow files with {self.hf_dataset_len} images"
                )
        else:
            split_dir = self.root / split
            if not split_dir.exists():
                raise ValueError(f"Split directory not found: {split_dir}")

            for class_dir in sorted(split_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
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
        center = torch.tensor([img_w / 2, img_h / 2], device=device, dtype=torch.float32)
        
        # Translation in pixels (from fraction of image size)
        translation = torch.tensor(
            [
                params['translation_x'] * img_w,
                params['translation_y'] * img_h,
            ],
            device=device,
            dtype=torch.float32,
        )
        
        # Scale
        scale = torch.tensor(
            [params['scale_x'], params['scale_y']],
            device=device,
            dtype=torch.float32,
        )
        
        # Rotation angle
        angle = torch.tensor([angle_rad], device=device, dtype=torch.float32)
        
        # Shear
        shear = torch.tensor(
            [params['shear_x'], params['shear_y']],
            device=device,
            dtype=torch.float32,
        )
        
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
    
    def _read_image(self, source) -> torch.Tensor:
        """Read image and convert to tensor [3, H, W] in [0, 1] range."""
        if isinstance(source, (str, Path)):
            img = Image.open(source).convert('RGB')
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            return img_tensor.contiguous()
        if isinstance(source, Image.Image):
            img = source.convert('RGB')
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            return img_tensor.contiguous()
        if isinstance(source, np.ndarray):
            if source.ndim == 3 and source.shape[-1] in (1, 3):
                img_tensor = torch.from_numpy(source).permute(2, 0, 1).float()
                if img_tensor.max() > 1.0:
                    img_tensor = img_tensor / 255.0
                return img_tensor.contiguous()
            raise ValueError(f"Unsupported numpy image shape: {source.shape}")
        if isinstance(source, torch.Tensor):
            img_tensor = source.float()
            if img_tensor.ndim == 3 and img_tensor.shape[-1] in (1, 3):
                img_tensor = img_tensor.permute(2, 0, 1)
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor / 255.0
            return img_tensor.contiguous()
        raise ValueError(f"Unsupported image type: {type(source).__name__}")

    def _import_hf_datasets(self):
        try:
            import datasets as ds  # type: ignore
            if hasattr(ds, "load_dataset"):
                return ds
        except Exception:
            ds = None

        if "datasets" in sys.modules:
            mod = sys.modules.get("datasets")
            if mod is not None and not hasattr(mod, "load_dataset"):
                del sys.modules["datasets"]

        search_paths = []
        try:
            search_paths.extend(site.getsitepackages())
        except Exception:
            pass
        try:
            user_site = site.getusersitepackages()
            if user_site:
                search_paths.append(user_site)
        except Exception:
            pass

        for base in search_paths:
            if not base:
                continue
            spec = PathFinder.find_spec("datasets", [base])
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules["datasets"] = module
            spec.loader.exec_module(module)
            if hasattr(module, "load_dataset"):
                return module

        raise ImportError(
            "Hugging Face datasets is required for hf:// sources. "
            "Install with: pip install datasets"
        )

    def _resolve_hf_cache_root(self, hf_cache_dir: Optional[Union[str, Path]]) -> Path:
        if hf_cache_dir:
            base = Path(hf_cache_dir).expanduser()
        else:
            try:
                from datasets import config as ds_config  # type: ignore
                base = Path(ds_config.HF_DATASETS_CACHE)
            except Exception:
                base = Path.home() / ".cache" / "huggingface" / "datasets"
        dataset_dir = self.hf_dataset_name.replace("/", "___")
        if base.name != dataset_dir and (base / dataset_dir).exists():
            return base / dataset_dir
        return base

    def _load_hf_from_cache(
        self,
        split_name: str,
        hf_cache_dir: Optional[Union[str, Path]],
        hf_max_samples: Optional[int],
    ):
        from datasets import Dataset, concatenate_datasets  # type: ignore
        cache_root = self._resolve_hf_cache_root(hf_cache_dir)
        if not cache_root.exists():
            raise ValueError(f"HF cache directory not found: {cache_root}")
        split_token = "validation" if split_name in ("val", "validation") else split_name
        arrow_files = sorted(cache_root.rglob(f"*{split_token}*.arrow"))
        if not arrow_files:
            raise ValueError(
                f"No cached Arrow files found for split '{split_token}' under {cache_root}"
            )
        datasets = [Dataset.from_file(str(path)) for path in arrow_files]
        dataset = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
        if hf_max_samples is not None:
            limit = min(int(hf_max_samples), len(dataset))
            dataset = dataset.select(range(limit))
        return dataset
    
    def __len__(self):
        if self.hf_dataset_len is not None:
            return self.hf_dataset_len
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
        if self.hf_samples is not None:
            item = self.hf_samples[idx]
            img_item = item.get("image") or item.get("img") or item.get("pixel_values")
            if img_item is None:
                raise ValueError("HF dataset sample missing image field")
            src_img = self._read_image(img_item)
        elif self.hf_dataset is not None:
            item = self.hf_dataset[idx]
            img_item = item.get("image") or item.get("img") or item.get("pixel_values")
            if img_item is None:
                raise ValueError("HF dataset sample missing image field")
            src_img = self._read_image(img_item)
        else:
            img_path = self.image_paths[idx]
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
