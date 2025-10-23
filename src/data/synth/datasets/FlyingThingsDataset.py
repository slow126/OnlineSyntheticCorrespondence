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


class FlowDownsampler:
    """
    Downsample flow to be compatible with CATS model.
    Converts flow from (B, 2, H, W) to (B, 2, feat_size, feat_size) format expected by CATS.
    """
    
    def __init__(self, feat_size: int):
        self.feat_size = feat_size
    
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
        
        # Remove batch dimension if input was single flow
        if not is_batched:
            flow_downsampled = flow_downsampled.squeeze(0)
        
        return flow_downsampled

class FlyingThingsDataset(Dataset, nn.Module):
    def __init__(self, root: str, split: str, transforms: Optional[Callable] = None, 
                 size: Optional[Tuple[int, int]] = None, 
                 downsample_flow: Optional[int] = None):
        Dataset.__init__(self)
        nn.Module.__init__(self)
        self.dataset = datasets.FlyingThings3D(root=root, split=split, transforms=transforms)
        self.size = size
        self.downsample_flow = downsample_flow
        
        # Create resize transform if size is specified
        if size is not None:
            self.resize_transform = FlowAwareResize(size)
        else:
            self.resize_transform = None
        
        # Create flow downsampler if downsample_flow is specified
        if downsample_flow is not None:
            self.flow_downsampler = FlowDownsampler(downsample_flow)
        else:
            self.flow_downsampler = None
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        src_img = torch.tensor(np.array(item[0])).permute(2, 0, 1).float() / 255.0
        trg_img = torch.tensor(np.array(item[1])).permute(2, 0, 1).float() / 255.0
        flow = torch.tensor(np.array(item[2])).float()

        # Move tensors to the same device as the module
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        src_img = src_img.to(device)
        trg_img = trg_img.to(device)
        flow = flow.to(device)

        # Create sample dict
        sample = {
            "src_img": src_img,
            "trg_img": trg_img,
            "flow": flow,
        }
        
        # Apply resize transform if specified
        if self.resize_transform is not None:
            sample = self.resize_transform(sample)
        
        # Apply flow downsampling if specified
        if self.flow_downsampler is not None:
            sample['flow'] = self.flow_downsampler(sample['flow'])
        
        return sample
    
    
    

if __name__ == "__main__":
    # Test without resizing
    print("Testing without resizing:")
    dataset_orig = FlyingThingsDataset(root="/home/spencer/Data/FlyingThings3D_tiny/", split="train", transforms=None)
    sample_orig = dataset_orig[0]
    print(f"Original shapes: src={sample_orig['src_img'].shape}, trg={sample_orig['trg_img'].shape}, flow={sample_orig['flow'].shape}")
    
    # Test with resizing
    print("\nTesting with resizing to (256, 256):")
    dataset_resized = FlyingThingsDataset(root="/home/spencer/Data/FlyingThings3D_tiny/", split="train", transforms=None, size=(256, 256))
    sample_resized = dataset_resized[0]
    print(f"Resized shapes: src={sample_resized['src_img'].shape}, trg={sample_resized['trg_img'].shape}, flow={sample_resized['flow'].shape}")
    
    # Test with flow downsampling
    print("\nTesting with flow downsampling to 32x32:")
    dataset_downsampled = FlyingThingsDataset(root="/home/spencer/Data/FlyingThings3D_tiny/", split="train", transforms=None, downsample_flow=32)
    sample_downsampled = dataset_downsampled[0]
    print(f"Downsampled flow shape: {sample_downsampled['flow'].shape}")
    
    # Test with both resizing and flow downsampling
    print("\nTesting with both resizing (256, 256) and flow downsampling (32x32):")
    dataset_both = FlyingThingsDataset(root="/home/spencer/Data/FlyingThings3D_tiny/", split="train", transforms=None, size=(256, 256), downsample_flow=32)
    sample_both = dataset_both[0]
    print(f"Combined shapes: src={sample_both['src_img'].shape}, trg={sample_both['trg_img'].shape}, flow={sample_both['flow'].shape}")
    
    # Test with DataLoader
    print("\nTesting DataLoader with combined transforms:")
    visualizer = CorrespondenceVisualizer()
    dataloader = DataLoader(dataset_both, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))
    print(f"Batch shapes: src={batch['src_img'].shape}, trg={batch['trg_img'].shape}, flow={batch['flow'].shape}")

    # Save visualizations
    visualizer.visualize_rendered_batch(batch, save_path="debug/flyingthings_dataset_combined.png", visualization_mode="overlay")
    visualizer.visualize_rendered_batch(batch, save_path="debug/flyingthings_dataset_combined_side_by_side.png", visualization_mode="side_by_side")
    print("Saved combined visualizations to debug/flyingthings_dataset_combined.png and debug/flyingthings_dataset_combined_side_by_side.png")