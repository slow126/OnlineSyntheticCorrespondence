import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from typing import Optional, Callable, Tuple
from torch.utils.data import Dataset
from src.data.synth.datasets.visualizers import CorrespondenceVisualizer
from torch.utils.data import DataLoader
import numpy as np


class FlowAwareResize:
    """
    Custom transform that resizes both images and flow vectors properly.
    Flow vectors need to be scaled by the resize factor.
    """
    
    def __init__(self, size: Tuple[int, int]):
        self.size = size
        self.resize_transform = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
    
    def __call__(self, sample):
        src_img, trg_img, flow = sample['src_img'], sample['trg_img'], sample['flow']
        
        # Get original dimensions
        orig_h, orig_w = src_img.shape[-2:]
        new_h, new_w = self.size
        
        # Calculate scale factors
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        
        # Resize images
        src_img_resized = self.resize_transform(src_img)
        trg_img_resized = self.resize_transform(trg_img)
        
        # Resize flow and scale the flow values
        flow_resized = self.resize_transform(flow)
        # Scale flow vectors by the resize factor
        flow_resized[0] *= scale_x  # x-component
        flow_resized[1] *= scale_y  # y-component
        
        return {
            'src_img': src_img_resized,
            'trg_img': trg_img_resized,
            'flow': flow_resized
        }


class FlowSubsampler:
    """
    Subsample flow by decimating flow vectors.
    Keeps only a percentage of flow vectors and sets the rest to None.
    """
    
    def __init__(self, subsample_ratio: float = 0.1, random_seed: Optional[int] = None,
                 filter_out_of_bounds: bool = True, use_valid_mask: bool = True):
        """
        Initialize flow subsampler.
        
        Args:
            subsample_ratio: Fraction of flow vectors to keep (e.g., 0.1 for 10%)
            random_seed: Random seed for reproducible subsampling
            filter_out_of_bounds: If True, filter out flow vectors that point outside frame boundaries
            use_valid_mask: If True, use valid_flow_mask to filter out occluded pixels
        """
        self.subsample_ratio = subsample_ratio
        self.random_seed = random_seed
        self.filter_out_of_bounds = filter_out_of_bounds
        self.use_valid_mask = use_valid_mask
        if random_seed is not None:
            torch.manual_seed(random_seed)
    
    def __call__(self, flow: torch.Tensor, valid_flow_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Subsample flow by keeping only a percentage of flow vectors.
        
        Args:
            flow: Input flow tensor of shape (B, 2, H, W) or (2, H, W)
            valid_flow_mask: Optional mask indicating valid flow vectors (not occluded)
            
        Returns:
            Subsampled flow tensor with same shape, but most vectors set to None/invalid
        """
        if flow is None:
            return flow
            
        # Handle both batched and single flow tensors
        is_batched = flow.dim() == 4
        if not is_batched:
            flow = flow.unsqueeze(0)  # Add batch dimension
            if valid_flow_mask is not None:
                valid_flow_mask = valid_flow_mask.unsqueeze(0)
        
        # Get flow dimensions
        B, C, H, W = flow.shape
        
        # Create a mask for subsampling using uniform random sampling
        # Start with all positions as candidates
        candidate_mask = torch.ones(H, W, dtype=torch.bool, device=flow.device)
        
        # Apply valid flow mask if available and enabled
        if self.use_valid_mask and valid_flow_mask is not None:
            # valid_flow_mask should be (B, H, W) or (H, W)
            if valid_flow_mask.dim() == 3:  # (B, H, W)
                candidate_mask = valid_flow_mask[0]  # Use first batch
            else:  # (H, W)
                candidate_mask = valid_flow_mask
        
        # Calculate number of valid candidates
        num_valid_candidates = candidate_mask.sum().item()
        num_keep = min(int(num_valid_candidates * self.subsample_ratio), num_valid_candidates)
        
        # Create subsampling mask
        subsample_mask = torch.zeros(H, W, dtype=torch.bool, device=flow.device)
        
        if num_keep > 0:
            # Generate random positions for uniform sampling from valid candidates
            if self.random_seed is not None:
                torch.manual_seed(self.random_seed)
            
            # Get valid indices directly (much faster)
            valid_indices = torch.nonzero(candidate_mask.flatten(), as_tuple=False).squeeze(-1)
            
            # Randomly select from valid indices
            random_indices = torch.randperm(len(valid_indices))[:num_keep]
            selected_indices = valid_indices[random_indices]
            
            # Convert flat indices to 2D coordinates and set mask
            selected_y = selected_indices // W
            selected_x = selected_indices % W
            subsample_mask[selected_y, selected_x] = True
        
        # Expand mask to match flow dimensions
        subsample_mask = subsample_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        subsample_mask = subsample_mask.expand(B, C, H, W)  # (B, C, H, W)
        
        # Create subsampled flow
        flow_subsampled = flow.clone()
        
        # Note: Flow transformations (reverse_flow, swap_xy, flip_x, flip_y) are NOT applied here
        # They are applied once at the end in FlyingThingsDataset.__getitem__ to ensure consistency
        
        # Filter out flow vectors that point outside frame boundaries
        if self.filter_out_of_bounds:
            # Only check selected positions (much faster)
            selected_mask = subsample_mask[0, 0]  # (H, W)
            if selected_mask.any():
                # Create coordinate grids only for selected positions
                y_coords, x_coords = torch.meshgrid(
                    torch.arange(H, device=flow.device, dtype=flow.dtype),
                    torch.arange(W, device=flow.device, dtype=flow.dtype),
                    indexing='ij'
                )
                
                # Calculate target positions only for selected flow vectors
                target_x = x_coords + flow_subsampled[0, 0]  # (H, W)
                target_y = y_coords + flow_subsampled[0, 1]  # (H, W)
                
                # Create mask for in-bounds flow vectors
                in_bounds_mask = (
                    (target_x >= 0) & (target_x < W) & 
                    (target_y >= 0) & (target_y < H) &
                    torch.isfinite(flow_subsampled[0]).all(dim=0)
                )
                
                # Update subsample mask to also filter out-of-bounds vectors
                subsample_mask[0, 0] = selected_mask & in_bounds_mask
                subsample_mask = subsample_mask.expand(B, C, H, W)
        
        # Set non-selected flow vectors to invalid values (inf)
        flow_subsampled[~subsample_mask] = float('inf')
        
        # Remove batch dimension if input was single flow
        if not is_batched:
            flow_subsampled = flow_subsampled.squeeze(0)
        
        return flow_subsampled


class FlowDownsampler:
    """
    Downsample flow to be compatible with CATS model.
    Converts flow from (B, 2, H, W) to (B, 2, feat_size, feat_size) format expected by CATS.
    """
    
    def __init__(self, feat_size: int):
        """
        Initialize flow downsampler.
        
        Args:
            feat_size: Target feature size (e.g., 32 for 32x32 downsampled flow)
        """
        self.feat_size = feat_size
    
    def __call__(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Downsample flow to be compatible with CATS model.
        Flow is normalized to feature grid units to match CATS convention.
        
        Args:
            flow: Input flow tensor of shape (B, 2, H, W) or (2, H, W) in pixel space
            
        Returns:
            Downsampled flow tensor of shape (B, 2, feat_size, feat_size) or (2, feat_size, feat_size)
            in feature grid units (normalized by downsampling factor)
        """
        if flow is None:
            return flow
            
        # Handle both batched and single flow tensors
        is_batched = flow.dim() == 4
        if not is_batched:
            flow = flow.unsqueeze(0)  # Add batch dimension
        
        # Get flow dimensions
        B, C, H, W = flow.shape
        
        # Calculate the scale factor for both dimensions
        scale_factor_h = H / self.feat_size
        scale_factor_w = W / self.feat_size
        
        # Downsample the flow using masked average pooling
        # We need to handle the case where flow might contain inf values
        # Only average over valid pixels to avoid skewing the result
        
        # Create a mask for valid flow values
        valid_mask = torch.isfinite(flow).all(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Set invalid values to 0 for pooling (temporary, won't affect masked average)
        flow_clean = flow.clone()
        flow_clean[~valid_mask.expand_as(flow_clean)] = 0
        
        # Sum of valid flow values in each pooling region
        flow_sum = torch.nn.functional.adaptive_avg_pool2d(
            flow_clean, (self.feat_size, self.feat_size)
        ) * (scale_factor_h * scale_factor_w)  # Multiply back to get sum
        
        # Count of valid pixels in each pooling region
        valid_count = torch.nn.functional.adaptive_avg_pool2d(
            valid_mask.float(), (self.feat_size, self.feat_size)
        ) * (scale_factor_h * scale_factor_w)  # Multiply back to get count
        
        # Compute masked average: divide sum by count of valid pixels (not total pixels)
        # Avoid division by zero for regions with no valid pixels
        valid_count_safe = torch.clamp(valid_count, min=1e-8)
        flow_downsampled = flow_sum / valid_count_safe

        # Normalize flow to feature grid units to match CATS convention
        # CATS expects flow in feature grid units, not pixel space
        # A flow of 1.0 = one feature grid cell = (H // feat_size) pixels
        downsampling_factor = H // self.feat_size
        flow_downsampled = flow_downsampled / downsampling_factor

        # Mark regions with no valid pixels as invalid (set to 0 or inf)
        # Create downsampled mask for invalid regions
        valid_mask_downsampled = valid_count > 0.5  # At least 0.5 valid pixels

        # Set invalid regions back to [0, 0]
        flow_downsampled[~valid_mask_downsampled.expand_as(flow_downsampled)] = 0
        
        # Note: Flow transformations (reverse_flow, swap_xy, flip_x, flip_y) are NOT applied here
        # They are applied once at the end in FlyingThingsDataset.__getitem__ to ensure consistency
        
        # Remove batch dimension if input was single flow
        if not is_batched:
            flow_downsampled = flow_downsampled.squeeze(0)
        
        return flow_downsampled


class FlyingThingsSimpleDataset(Dataset, nn.Module):
    def __init__(self, root: str, split: str, transforms: Optional[Callable] = None, reverse_flow: bool = False):
        Dataset.__init__(self)
        nn.Module.__init__(self)
        self.dataset = datasets.FlyingThings3D(root=root, split=split, transforms=transforms)
        self.reverse_flow = reverse_flow
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.reverse_flow:
            src_index = 1
            trg_index = 0
        else:
            src_index = 0
            trg_index = 1
        
        # Convert PIL Images to tensors (keep on CPU - DataLoader will handle GPU transfer)
        src_img = torch.from_numpy(np.array(item[src_index])).permute(2, 0, 1).float() / 255.0
        trg_img = torch.from_numpy(np.array(item[trg_index])).permute(2, 0, 1).float() / 255.0
        flow = torch.from_numpy(np.array(item[2])).float()

        return {
            "src_img": src_img,
            "trg_img": trg_img,
            "flow": flow,
        }

class FlyingThingsDataset(Dataset, nn.Module):
    def __init__(self, root: str, split: str, transforms: Optional[Callable] = None, 
                 size: Optional[Tuple[int, int]] = None, 
                 downsample_flow: Optional[int] = None,
                 subsample_flow: Optional[float] = None,
                 subsample_flow_seed: Optional[int] = None,
                 reverse_flow: bool = False,
                 filter_out_of_bounds: bool = True, use_valid_mask: bool = True,
                 normalize: bool = True,
                 max_pts: int = 200,
                 thres: str = 'img'):
        Dataset.__init__(self)
        nn.Module.__init__(self)
        self.dataset = datasets.FlyingThings3D(root=root, split=split, transforms=transforms)
        self.size = size
        self.downsample_flow = downsample_flow
        self.subsample_flow = subsample_flow
        
        # Store flow transformation parameters (applied once at the end, not in processors)
        self.reverse_flow = reverse_flow
        self.normalize = normalize
        self.max_pts = max_pts
        self.thres = thres
        
        # Cache device detection for faster tensor operations
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create resize transform if size is specified
        if size is not None:
            self.resize_transform = FlowAwareResize(size)
        else:
            self.resize_transform = None
        
        # Create flow subsampler if subsample_flow is specified
        # Note: flow transformations are NOT applied here, they're applied at the end in __getitem__
        if subsample_flow is not None:
            self.flow_subsampler = FlowSubsampler(subsample_flow, subsample_flow_seed, filter_out_of_bounds, use_valid_mask)
        else:
            self.flow_subsampler = FlowSubsampler(1.0, subsample_flow_seed, filter_out_of_bounds, use_valid_mask)
        
        # Create flow downsampler if downsample_flow is specified
        # Note: flow transformations are NOT applied here, they're applied at the end in __getitem__
        if downsample_flow is not None:
            self.flow_downsampler = FlowDownsampler(downsample_flow)
        else:
            self.flow_downsampler = None
        
    def __len__(self):
        return len(self.dataset)
    
    def _sample_keypoints_from_flow(
        self,
        flow: torch.Tensor,
        num_kps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample keypoints from valid flow regions.
        
        Args:
            flow: Flow tensor [2, H, W] (after all transforms)
            num_kps: Number of keypoints to sample
            
        Returns:
            trg_kps: Target keypoints [2, num_kps] (x, y format)
            src_kps: Source keypoints [2, num_kps] (computed from flow)
        """
        _, h, w = flow.shape
        
        # Find valid flow regions (not inf and non-zero magnitude)
        flow_mag = flow.norm(dim=0)
        valid_mask = flow_mag.isfinite() & (flow_mag > 0)
        
        # Sample from valid regions
        valid_y, valid_x = torch.where(valid_mask)
        num_valid = len(valid_y)
        
        if num_valid == 0:
            # No valid points - return zeros
            trg_kps = torch.zeros((2, num_kps), dtype=torch.float32)
            src_kps = torch.zeros((2, num_kps), dtype=torch.float32)
            return trg_kps, src_kps
        
        if num_valid <= num_kps:
            # Use all valid points, then pad with zeros
            indices = torch.arange(num_valid)
            n_to_pad = num_kps - num_valid
        else:
            # Randomly sample exactly num_kps points
            indices = torch.randperm(num_valid)[:num_kps]
            n_to_pad = 0
        
        sampled_y = valid_y[indices]
        sampled_x = valid_x[indices]
        
        trg_kps = torch.stack([sampled_x.float(), sampled_y.float()])  # [2, n_sampled] (x, y)
        
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
        
        # Pad to num_kps if needed
        if n_to_pad > 0:
            trg_kps = torch.cat([trg_kps, torch.zeros((2, n_to_pad), dtype=torch.float32)], dim=1)
            src_kps = torch.cat([src_kps, torch.zeros((2, n_to_pad), dtype=torch.float32)], dim=1)
        
        n_pts = min(num_kps, num_valid)
        return trg_kps, src_kps, n_pts
    
    def _get_pckthres(self, imsize: Tuple[int, int]) -> torch.Tensor:
        """Get PCK threshold based on image size."""
        if self.thres == 'img':
            return torch.tensor(max(imsize), dtype=torch.float32)
        else:
            # Default to image size
            return torch.tensor(max(imsize), dtype=torch.float32)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.reverse_flow:
            src_index = 1
            trg_index = 0
        else:
            src_index = 0
            trg_index = 1
        
        # Convert PIL Images to tensors (keep on CPU - DataLoader will handle GPU transfer)
        src_img = torch.from_numpy(np.array(item[src_index])).permute(2, 0, 1).float() / 255.0
        trg_img = torch.from_numpy(np.array(item[trg_index])).permute(2, 0, 1).float() / 255.0
        flow = torch.from_numpy(np.array(item[2])).float()

        return {
            "src_img": src_img,
            "trg_img": trg_img,
            "flow": flow,
        }

    def __getitem__old(self, idx):
        item = self.dataset[idx]
        if self.reverse_flow:
            src_index = 1
            trg_index = 0
        else:
            src_index = 0
            trg_index = 1
        
        # Convert PIL Images to tensors (keep on CPU - DataLoader will handle GPU transfer)
        src_img = torch.from_numpy(np.array(item[src_index])).permute(2, 0, 1).float() / 255.0
        trg_img = torch.from_numpy(np.array(item[trg_index])).permute(2, 0, 1).float() / 255.0
        flow = torch.from_numpy(np.array(item[2])).float()
        
        # Normalize images if requested (model expects ImageNet normalization)
        # ImageNet normalization: (img - mean) / std produces "dark and crunchy" appearance
        if self.normalize:
            from torchvision.transforms.functional import normalize
            src_img = normalize(src_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            trg_img = normalize(trg_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Check if valid flow mask is available (4-tuple vs 3-tuple)
        valid_flow_mask = None
        if len(item) == 4:
            valid_flow_mask = torch.from_numpy(np.array(item[3])).bool()

        # Create sample dict
        sample = {
            "src_img": src_img,
            "trg_img": trg_img,
            "flow": flow,
        }
        full_flow = sample['flow'].clone()
        # Add valid flow mask if available
        if valid_flow_mask is not None:
            sample["valid_flow_mask"] = valid_flow_mask

        # Apply resize transform if specified
        if self.resize_transform is not None:
            sample = self.resize_transform(sample)
        
        # Sample keypoints from flow before subsampling and downsampling
        # This gives better spatial coverage from the full-resolution flow
        trg_kps, src_kps, n_pts = self._sample_keypoints_from_flow(sample['flow'], self.max_pts)
        sample["trg_kps"] = trg_kps
        sample["src_kps"] = src_kps
        sample["n_pts"] = torch.tensor(n_pts, dtype=torch.int32)
        
        # Apply flow subsampling if specified (before downsampling)
        if self.flow_subsampler is not None:
            valid_mask = sample.get('valid_flow_mask', None)
            sample['flow'] = self.flow_subsampler(sample['flow'], valid_mask)

        
        # Apply flow downsampling if specified (after subsampling)
        if self.flow_downsampler is not None:
            sample['flow'] = self.flow_downsampler(sample['flow'])
        
        # Add validation fields (pckthres, imsize, datalen)
        # Get image size after all transforms
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
        sample['pckthres'] = pckthres
        sample['src_imsize'] = img_size_tuple
        sample['trg_imsize'] = img_size_tuple
        sample['datalen'] = torch.tensor(len(self), dtype=torch.int32)
        sample['flow_full'] = full_flow
        return sample
    
    

if __name__ == "__main__":
    # Test with reversed flow (the working configuration)
    print("Testing with reversed flow:")
    dataset_reversed = FlyingThingsDataset(
        root="/home/spencer/Data/FlyingThings3D_tiny/", 
        split="train", 
        transforms=None, 
        subsample_flow=0.1, 
        downsample_flow=None, 
        reverse_flow=True, 
        filter_out_of_bounds=True,
        use_valid_mask=True,
        size=(512, 512)
    )

    dataset_forward = FlyingThingsDataset(
        root="/home/spencer/Data/FlyingThings3D_tiny/", 
        split="train", 
        transforms=None, 
        subsample_flow=0.1, 
        downsample_flow=None,  # Keep full resolution
        reverse_flow=False, 
        filter_out_of_bounds=True,
        use_valid_mask=True,
        size=(512, 512)
    )

    sample_reversed = dataset_reversed[0]
    print(f"Reversed flow sample values: {sample_reversed['flow'][:, 16, 16]}")
    
    # Test with DataLoader
    print("\nTesting DataLoader with reversed flow:")
    visualizer = CorrespondenceVisualizer()
    dataloader = DataLoader(dataset_reversed, batch_size=4, shuffle=False)
    batch = next(iter(dataloader))
    print(f"Batch shapes: src={batch['src_img'].shape}, trg={batch['trg_img'].shape}, flow={batch['flow'].shape}")

    # Save visualizations
    visualizer.visualize_rendered_batch(batch, save_path="debug/flyingthings_dataset_reversed_overlay.png", visualization_mode="overlay")
    visualizer.visualize_rendered_batch(batch, save_path="debug/flyingthings_dataset_reversed_side_by_side.png", visualization_mode="side_by_side")
    print("Saved reversed flow visualizations to debug/flyingthings_dataset_reversed_overlay.png and debug/flyingthings_dataset_reversed_side_by_side.png")
    
    print("Testing with forward flow:")
    sample_forward = dataset_forward[0]
    print(f"Forward flow sample values: {sample_forward['flow'][:, 16, 16]}")
    print("\nTesting DataLoader with forward flow:")
    dataloader_forward = DataLoader(dataset_forward, batch_size=4, shuffle=False)
    batch_forward = next(iter(dataloader_forward))
    print(f"Forward batch shapes: src={batch_forward['src_img'].shape}, trg={batch_forward['trg_img'].shape}, flow={batch_forward['flow'].shape}")

    visualizer.visualize_rendered_batch(batch_forward, save_path="debug/flyingthings_dataset_forward_overlay.png", visualization_mode="overlay")
    visualizer.visualize_rendered_batch(batch_forward, save_path="debug/flyingthings_dataset_forward_side_by_side.png", visualization_mode="side_by_side")
    print("Saved forward flow visualizations to debug/flyingthings_dataset_forward_overlay.png and debug/flyingthings_dataset_forward_side_by_side.png")


    # Test with downsampled flow for CATS
    print("\n" + "="*60)
    print("Testing with downsampled flow for CATS:")
    print("="*60)
    
    # Create dataset with full-resolution flow (no downsampling)
    dataset_full = FlyingThingsDataset(
        root="/home/spencer/Data/FlyingThings3D_tiny/", 
        split="train", 
        transforms=None, 
        subsample_flow=0.1, 
        downsample_flow=None,  # Keep full resolution
        reverse_flow=True, 
        filter_out_of_bounds=True,
        use_valid_mask=True,
        size=(512, 512)
    )


    
    # Create dataset with downsampled flow
    dataset_downsampled = FlyingThingsDataset(
        root="/home/spencer/Data/FlyingThings3D_tiny/", 
        split="train", 
        transforms=None, 
        subsample_flow=1.0, 
        downsample_flow=32,  # feat_size=32
        reverse_flow=True, 
        filter_out_of_bounds=True,
        use_valid_mask=True,
        size=(512, 512)
    )
    
    sample_downsampled = dataset_downsampled[0]
    print(f"Downsampled flow shape: {sample_downsampled['flow'].shape}")
    print(f"Downsampled flow sample values: {sample_downsampled['flow'][:, 10, 10]}")
    print(f"Flow value ranges - X: [{sample_downsampled['flow'][0].min():.2f}, {sample_downsampled['flow'][0].max():.2f}]")
    print(f"Flow value ranges - Y: [{sample_downsampled['flow'][1].min():.2f}, {sample_downsampled['flow'][1].max():.2f}]")
    
    # Test with DataLoaders for comparison (shuffle=False to get same samples)
    print("\nTesting DataLoader with downsampled flow (shuffle=False for comparison):")
    try:
        from src.data.synth.datasets.cats_flow_visualizers import CATSFlowVisualizer
        
        cats_visualizer = CATSFlowVisualizer(
            feat_size=32,
            figsize=(20, 15),
            dpi=150,
            show_patch_boundaries=True
        )
        
        # Use shuffle=False to ensure same samples for comparison
        dataloader_full = DataLoader(dataset_full, batch_size=4, shuffle=False)
        dataloader_downsampled = DataLoader(dataset_downsampled, batch_size=4, shuffle=False)
        
        batch_full = next(iter(dataloader_full))
        batch_downsampled = next(iter(dataloader_downsampled))
        
        print(f"Full-res batch shapes: src={batch_full['src_img'].shape}, trg={batch_full['trg_img'].shape}, flow={batch_full['flow'].shape}")
        print(f"Downsampled batch shapes: src={batch_downsampled['src_img'].shape}, trg={batch_downsampled['trg_img'].shape}, flow={batch_downsampled['flow'].shape}")
        
        # Prepare batch dict for downsampled-only visualization
        batch_dict_downsampled = {
            'src_img': batch_downsampled['src_img'],
            'trg_img': batch_downsampled['trg_img'],
            'flow_downsampled': batch_downsampled['flow']
        }
        
        # Visualize downsampled flow (side-by-side)
        print("\nCreating downsampled flow visualization (side-by-side)...")
        cats_visualizer.visualize_downsampled_flow_batch(
            batch_dict_downsampled,
            save_path="debug/flyingthings_dataset_downsampled_flow.png",
            max_samples=4,
            visualization_mode='side_by_side'
        )
        print("Saved downsampled flow visualization to debug/flyingthings_dataset_downsampled_flow.png")
        
        # Visualize downsampled flow (overlay)
        print("\nCreating downsampled flow visualization (overlay)...")
        cats_visualizer.visualize_downsampled_flow_batch(
            batch_dict_downsampled,
            save_path="debug/flyingthings_dataset_downsampled_flow_overlay.png",
            max_samples=4,
            visualization_mode='overlay'
        )
        print("Saved downsampled flow overlay visualization to debug/flyingthings_dataset_downsampled_flow_overlay.png")
        
        # Prepare batch dict for comparison (both full-res and downsampled)
        batch_dict_comparison = {
            'src_img': batch_full['src_img'],  # Use same images from full-res dataset
            'trg_img': batch_full['trg_img'],
            'flow_full': batch_full['flow'],
            'flow_downsampled': batch_downsampled['flow']
        }
        
        # Visualize comparison (full-res vs downsampled)
        print("\nCreating comparison visualization (full-res vs downsampled)...")
        cats_visualizer.visualize_comparison_batch(
            batch_dict_comparison,
            save_path="debug/flyingthings_dataset_flow_comparison.png",
            max_samples=4
        )
        print("Saved comparison visualization to debug/flyingthings_dataset_flow_comparison.png")
        
    except ImportError as e:
        print(f"Could not import CATSFlowVisualizer: {e}")
        print("Skipping downsampled flow visualization")