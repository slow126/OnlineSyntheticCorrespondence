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
                 reverse_flow: bool = False, swap_xy: bool = False, flip_x: bool = False, flip_y: bool = False,
                 filter_out_of_bounds: bool = True, use_valid_mask: bool = True):
        """
        Initialize flow subsampler.
        
        Args:
            subsample_ratio: Fraction of flow vectors to keep (e.g., 0.1 for 10%)
            random_seed: Random seed for reproducible subsampling
            reverse_flow: If True, reverse the flow direction (multiply by -1)
            swap_xy: If True, swap x and y components of flow
            flip_x: If True, flip the sign of x component
            flip_y: If True, flip the sign of y component
            filter_out_of_bounds: If True, filter out flow vectors that point outside frame boundaries
            use_valid_mask: If True, use valid_flow_mask to filter out occluded pixels
        """
        self.subsample_ratio = subsample_ratio
        self.random_seed = random_seed
        self.reverse_flow = reverse_flow
        self.swap_xy = swap_xy
        self.flip_x = flip_x
        self.flip_y = flip_y
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
            
            # Get valid positions
            valid_positions = torch.stack(torch.meshgrid(
                torch.arange(H, device=flow.device),
                torch.arange(W, device=flow.device),
                indexing='ij'
            ), dim=-1).reshape(-1, 2)  # (H*W, 2)
            
            # Filter to only valid positions
            valid_indices = torch.nonzero(candidate_mask.flatten(), as_tuple=False).squeeze(-1)
            valid_positions = valid_positions[valid_indices]
            
            # Randomly select from valid positions
            random_indices = torch.randperm(len(valid_positions))[:num_keep]
            selected_positions = valid_positions[random_indices]
            
            # Set selected positions to True
            for pos in selected_positions:
                subsample_mask[pos[0], pos[1]] = True
        
        # Expand mask to match flow dimensions
        subsample_mask = subsample_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        subsample_mask = subsample_mask.expand(B, C, H, W)  # (B, C, H, W)
        
        # Create subsampled flow
        flow_subsampled = flow.clone()
        
        # Apply flow transformations
        if self.reverse_flow:
            flow_subsampled = -flow_subsampled
        
        if self.swap_xy:
            # Swap x and y components: [x, y] -> [y, x]
            flow_subsampled = flow_subsampled.flip(1)  # Flip along channel dimension
        
        if self.flip_x:
            if is_batched:
                flow_subsampled[:, 0] = -flow_subsampled[:, 0]  # Flip x component
            else:
                flow_subsampled[0, 0] = -flow_subsampled[0, 0]  # Flip x component
        
        if self.flip_y:
            if is_batched:
                flow_subsampled[:, 1] = -flow_subsampled[:, 1]  # Flip y component
            else:
                flow_subsampled[0, 1] = -flow_subsampled[0, 1]  # Flip y component
        
        # Filter out flow vectors that point outside frame boundaries
        if self.filter_out_of_bounds:
            # Create coordinate grids for source positions
            y_coords, x_coords = torch.meshgrid(
                torch.arange(H, device=flow.device, dtype=flow.dtype),
                torch.arange(W, device=flow.device, dtype=flow.dtype),
                indexing='ij'
            )
            
            # Expand grids to match batch dimension
            x_coords = x_coords.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
            y_coords = y_coords.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
            
            # Calculate target positions (where flow vectors point to)
            target_x = x_coords + flow_subsampled[:, 0]  # (B, H, W)
            target_y = y_coords + flow_subsampled[:, 1]  # (B, H, W)
            
            # Create mask for in-bounds flow vectors
            in_bounds_mask = (
                (target_x >= 0) & (target_x < W) & 
                (target_y >= 0) & (target_y < H) &
                torch.isfinite(flow_subsampled).all(dim=1, keepdim=True)
            )
            
            # Update subsample mask to also filter out-of-bounds vectors
            subsample_mask = subsample_mask & in_bounds_mask
        
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
    
    def __init__(self, feat_size: int, reverse_flow: bool = False, swap_xy: bool = False, flip_x: bool = False, flip_y: bool = False):
        self.feat_size = feat_size
        self.reverse_flow = reverse_flow
        self.swap_xy = swap_xy
        self.flip_x = flip_x
        self.flip_y = flip_y
    
    def __call__(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Downsample flow to be compatible with CATS model.
        
        Args:
            flow: Input flow tensor of shape (B, 2, H, W) or (2, H, W)
            
        Returns:
            Downsampled flow tensor of shape (B, 2, feat_size, feat_size) or (2, feat_size, feat_size)
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
        
        # Downsample the flow using average pooling
        # We need to handle the case where flow might contain inf values
        flow_clean = flow.clone()
        
        # Create a mask for valid flow values
        valid_mask = torch.isfinite(flow).all(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Set invalid values to 0 for pooling
        flow_clean[~valid_mask.expand_as(flow_clean)] = 0
        
        # Apply adaptive average pooling to downsample
        flow_downsampled = torch.nn.functional.adaptive_avg_pool2d(
            flow_clean, (self.feat_size, self.feat_size)
        )
        
        # Scale the flow values by the average scale factor to maintain proper magnitude
        # Use the average of both scale factors for consistent scaling
        avg_scale_factor = (scale_factor_h + scale_factor_w) / 2
        flow_downsampled = flow_downsampled / avg_scale_factor
        
        # Restore invalid values as inf where appropriate
        # Create downsampled mask for invalid regions
        valid_mask_downsampled = torch.nn.functional.adaptive_avg_pool2d(
            valid_mask.float(), (self.feat_size, self.feat_size)
        ) > 0.5  # Keep as valid if majority of pixels in the region are valid
        
        # Set invalid regions back to [0, 0]
        flow_downsampled[~valid_mask_downsampled.expand_as(flow_downsampled)] = 0
        
        # Apply flow transformations
        if self.reverse_flow:
            flow_downsampled = -flow_downsampled
        
        if self.swap_xy:
            # Swap x and y components: [x, y] -> [y, x]
            flow_downsampled = flow_downsampled.flip(1)  # Flip along channel dimension
        
        if self.flip_x:
            if is_batched:
                flow_downsampled[:, 0] = -flow_downsampled[:, 0]  # Flip x component
            else:
                flow_downsampled[0, 0] = -flow_downsampled[0, 0]  # Flip x component
        
        if self.flip_y:
            if is_batched:
                flow_downsampled[:, 1] = -flow_downsampled[:, 1]  # Flip y component
            else:
                flow_downsampled[0, 1] = -flow_downsampled[0, 1]  # Flip y component
        
        # Remove batch dimension if input was single flow
        if not is_batched:
            flow_downsampled = flow_downsampled.squeeze(0)
        
        return flow_downsampled

class FlyingThingsDataset(Dataset, nn.Module):
    def __init__(self, root: str, split: str, transforms: Optional[Callable] = None, 
                 size: Optional[Tuple[int, int]] = None, 
                 downsample_flow: Optional[int] = None,
                 subsample_flow: Optional[float] = None,
                 subsample_flow_seed: Optional[int] = None,
                 reverse_flow: bool = False, swap_xy: bool = False, 
                 flip_x: bool = False, flip_y: bool = False,
                 filter_out_of_bounds: bool = True, use_valid_mask: bool = True):
        Dataset.__init__(self)
        nn.Module.__init__(self)
        self.dataset = datasets.FlyingThings3D(root=root, split=split, transforms=transforms)
        self.size = size
        self.downsample_flow = downsample_flow
        self.subsample_flow = subsample_flow
        
        # Create resize transform if size is specified
        if size is not None:
            self.resize_transform = FlowAwareResize(size)
        else:
            self.resize_transform = None
        
        # Create flow subsampler if subsample_flow is specified
        if subsample_flow is not None:
            self.flow_subsampler = FlowSubsampler(subsample_flow, subsample_flow_seed, reverse_flow, swap_xy, flip_x, flip_y, filter_out_of_bounds, use_valid_mask)
        else:
            self.flow_subsampler = None
        
        # Create flow downsampler if downsample_flow is specified
        if downsample_flow is not None:
            self.flow_downsampler = FlowDownsampler(downsample_flow, reverse_flow, swap_xy, flip_x, flip_y)
        else:
            self.flow_downsampler = None
        
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        src_img = torch.tensor(np.array(item[0])).permute(2, 0, 1).float() / 255.0
        trg_img = torch.tensor(np.array(item[1])).permute(2, 0, 1).float() / 255.0
        flow = torch.tensor(np.array(item[2])).float()
        
        # Check if valid flow mask is available (4-tuple vs 3-tuple)
        valid_flow_mask = None
        if len(item) == 4:
            valid_flow_mask = torch.tensor(np.array(item[3])).bool()

        # Move tensors to the same device as the module
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        src_img = src_img.to(device)
        trg_img = trg_img.to(device)
        flow = flow.to(device)
        if valid_flow_mask is not None:
            valid_flow_mask = valid_flow_mask.to(device)

        # Create sample dict
        sample = {
            "src_img": src_img,
            "trg_img": trg_img,
            "flow": flow,
        }
        
        # Add valid flow mask if available
        if valid_flow_mask is not None:
            sample["valid_flow_mask"] = valid_flow_mask
        
        # Apply resize transform if specified
        if self.resize_transform is not None:
            sample = self.resize_transform(sample)
        
        # Apply flow subsampling if specified (before downsampling)
        if self.flow_subsampler is not None:
            valid_mask = sample.get('valid_flow_mask', None)
            sample['flow'] = self.flow_subsampler(sample['flow'], valid_mask)
        
        # Apply flow downsampling if specified (after subsampling)
        if self.flow_downsampler is not None:
            sample['flow'] = self.flow_downsampler(sample['flow'])
        
        return sample
    
    
    

if __name__ == "__main__":
    # Test with reversed flow (the working configuration)
    print("Testing with reversed flow:")
    dataset_reversed = FlyingThingsDataset(
        root="/home/spencer/Data/FlyingThings3D_tiny/", 
        split="train", 
        transforms=None, 
        subsample_flow=0.02, 
        downsample_flow=None, 
        reverse_flow=True, 
        filter_out_of_bounds=True,
        use_valid_mask=True
    )
    sample_reversed = dataset_reversed[0]
    print(f"Reversed flow sample values: {sample_reversed['flow'][:, 16, 16]}")
    
    # Test with DataLoader
    print("\nTesting DataLoader with reversed flow:")
    visualizer = CorrespondenceVisualizer()
    dataloader = DataLoader(dataset_reversed, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))
    print(f"Batch shapes: src={batch['src_img'].shape}, trg={batch['trg_img'].shape}, flow={batch['flow'].shape}")

    # Save visualizations
    visualizer.visualize_rendered_batch(batch, save_path="debug/flyingthings_dataset_reversed_overlay.png", visualization_mode="overlay")
    visualizer.visualize_rendered_batch(batch, save_path="debug/flyingthings_dataset_reversed_side_by_side.png", visualization_mode="side_by_side")
    print("Saved reversed flow visualizations to debug/flyingthings_dataset_reversed_overlay.png and debug/flyingthings_dataset_reversed_side_by_side.png")