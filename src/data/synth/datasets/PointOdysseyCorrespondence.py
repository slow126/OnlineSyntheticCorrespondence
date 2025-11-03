"""
PointOdyssey dataset wrapper for correspondence and optical flow.

This module provides a wrapper around the PointOdyssey dataset that returns
data in a format suitable for correspondence learning: src, trg, and flow.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random as _random

# Add the pips2 path to sys.path so the utils can be found
pips2_path = Path(__file__).parent / "pips2"
sys.path.insert(0, str(pips2_path))

# Import the base dataset and utils
from src.data.synth.datasets.pips2.datasets.pointodysseydataset import PointOdysseyDataset as BasePointOdysseyDataset


class PointOdysseyFlowDataset(torch.utils.data.Dataset):
    """
    Wrapper for PointOdyssey dataset that returns correspondence data.
    
    Returns:
        - src: Source image (first frame)
        - trg: Target image (second frame) 
        - flow: Optical flow from trg to src (dx, dy)
    """
    
    def __init__(self, 
                 dataset_location: str = '/home/spencer/Data/sample',
                 dset: str = 'train',
                 use_augs: bool = False,
                 S: int = 8,
                 N: int = 32,
                 strides: list = [1, 2, 4],
                 clip_step: int = 2,
                 resize_size: tuple = (368+64, 496+64),
                 crop_size: tuple = (368, 496),
                 req_full: bool = False,
                 quick: bool = False,
                 verbose: bool = False,
                 filter_instances: bool = False,
                 reverse_flow: bool = False,
                 downsample_for_cats: bool = False,
                 cats_feat_size: int = 32,
                 all_points: bool = False):
        """
        Initialize the PointOdyssey flow dataset.
        
        Args:
            dataset_location: Path to PointOdyssey dataset
            dset: Dataset split ('train', 'val', 'test')
            use_augs: Whether to use data augmentations
            S: Number of frames per sequence
            N: Number of points to track
            strides: Frame strides for sampling
            clip_step: Step size for clip sampling
            resize_size: Size to resize images to
            crop_size: Size to crop images to
            req_full: Whether to require full sequences
            quick: Whether to use quick mode (fewer samples)
            verbose: Whether to print verbose information
            target_size: Optional target size for resizing (H, W)
        """
        # Check if the dataset has the expected structure (with train/val/test subdirs)
        expected_dset_path = os.path.join(dataset_location, dset)
        if not os.path.exists(expected_dset_path):
            # If no train/val/test subdirs, assume sequences are directly in dataset_location
            print(f"Warning: No '{dset}' subdirectory found in {dataset_location}")
            print("Assuming sequences are directly in the dataset location")
            # Create a temporary structure by pointing to the parent directory
            actual_dataset_location = os.path.dirname(dataset_location)
            actual_dset = os.path.basename(dataset_location)
        else:
            actual_dataset_location = dataset_location
            actual_dset = dset
            
        self.base_dataset = BasePointOdysseyDataset(
            dataset_location=actual_dataset_location,
            dset=actual_dset,
            use_augs=use_augs,
            S=S,
            N=N,
            strides=strides,
            clip_step=clip_step,
            resize_size=resize_size,
            crop_size=crop_size,
            req_full=req_full,
            quick=quick,
            verbose=verbose,
            all_points=all_points
        )
        
        self.S = S
        self.N = N
        self.filter_instances = filter_instances
        self.downsample_for_cats = downsample_for_cats
        self.cats_feat_size = cats_feat_size
        self.verbose = verbose
        self.reverse_flow = reverse_flow
        # Device management - defaults to CPU
        self._device = torch.device('cpu')
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.base_dataset)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary containing:
                - 'src_img': Source image tensor (C, H, W)
                - 'trg_img': Target image tensor (C, H, W) 
                - 'flow': Flow tensor (2, H, W) from trg to src
        """

        sample, gotit = self.base_dataset[index]
        
        while not gotit:
            # Resampling because index failed to get valid samples.
            new_index = _random.randint(0, len(self.base_dataset) - 1)
            if self.verbose:
                print(f"Resampling because index {index} failed to get valid samples. New index: {new_index}")
            sample, gotit = self.base_dataset[new_index]
        
        # Move all base dataset tensors to GPU immediately
        # Extract the data we need and move to device
        rgbs = sample['rgbs'].to(self._device, non_blocking=True)  # (S, C, H, W)
        trajs = sample['trajs'].to(self._device, non_blocking=True)  # (S, N, 2)
        visibs = sample['visibs'].to(self._device, non_blocking=True)  # (S, N)
        valids = sample['valids'].to(self._device, non_blocking=True)  # (S, N)
        masks = sample['masks'].to(self._device, non_blocking=True)   # (S, 1, H, W) instance ids

        # Sample two non-repeating frames (order doesn't matter)
        
        # i, j = _random.sample(range(self.S), 2)
        i, j = 0, self.S - 1

        src_img = rgbs[i]
        trg_img = rgbs[j]
        
        # Convert images to float32 in [0, 1] range using PyTorch operations on GPU
        # Using .to(torch.float32) and division - all operations stay on GPU
        src_img = src_img.to(torch.float32) / 255.0
        trg_img = trg_img.to(torch.float32) / 255.0

        # Trajectories and flags for those two frames
        src_trajs = trajs[i]
        trg_trajs = trajs[j]
        src_vis = visibs[i]
        trg_vis = visibs[j]
        src_valid = valids[i]
        trg_valid = valids[j]
        src_mask = masks[i]
        trg_mask = masks[j]

        # Create sparse flow field from trg->src
        flow = self._create_flow_field(
            src_trajs, trg_trajs, src_vis, trg_vis, src_valid, trg_valid,
            src_img.shape, src_mask, trg_mask, self.filter_instances, self._device
        )

        # Optionally downsample flow for CATS compatibility
        if self.downsample_for_cats:
            flow = self._downsample_flow_for_cats(flow, self.cats_feat_size)
        else:
            # If not downsampling, convert invalid (inf) pixels to 0 to avoid passing infs to model
            # This matches the behavior of the downsampler which also sets invalid regions to 0
            invalid_mask = ~torch.isfinite(flow).all(dim=0, keepdim=False)  # (H, W)
            flow[:, invalid_mask] = 0.0
        
        out = {
            'src_img': src_img,
            'trg_img': trg_img,
            'flow': flow,
            'masks': masks
        }

        # Move to configured device
        if self._device is not None:
            for k, v in out.items():
                if isinstance(v, torch.Tensor) and v.device != self._device:
                    out[k] = v.to(self._device, non_blocking=True)

        return out
    
    def _create_flow_field(self, 
                          src_trajs: torch.Tensor, 
                          trg_trajs: torch.Tensor,
                          src_vis: torch.Tensor,
                          trg_vis: torch.Tensor, 
                          src_valid: torch.Tensor,
                          trg_valid: torch.Tensor,
                          img_shape: Tuple[int, int, int],
                          src_mask: torch.Tensor,
                          trg_mask: torch.Tensor,
                          filter_instances: bool,
                          device: torch.device) -> torch.Tensor:
        """
        Create a flow field from target to source.
        
        Args:
            src_trajs: Source frame trajectories (N, 2)
            trg_trajs: Target frame trajectories (N, 2)
            src_vis: Source frame visibility (N,)
            trg_vis: Target frame visibility (N,)
            src_valid: Source frame validity (N,)
            trg_valid: Target frame validity (N,)
            img_shape: Image shape (C, H, W)
            src_mask: Source instance mask (C, H, W)
            trg_mask: Target instance mask (C, H, W)
        Returns:
            Flow field tensor (2, H, W) where flow[0] = dx, flow[1] = dy from trg to src
        """
        C, H, W = img_shape
        
        # Initialize flow field with inf (invalid) on the specified device
        # Flow format: (2, H, W) where flow[0] = dx, flow[1] = dy
        # Invalid pixels start as inf, which will be converted to 0 by downsampler if needed
        # This allows proper handling of sparse flow fields
        flow = torch.full((2, H, W), float('inf'), dtype=torch.float32, device=device)
        
        # Ensure masks are (H, W) instance id maps
        if src_mask.ndim == 3:
            src_mask = src_mask.squeeze(0)
        if trg_mask.ndim == 3:
            trg_mask = trg_mask.squeeze(0)
        
        # Find points that are visible and valid in both frames
        valid_points = (src_vis > 0) & (trg_vis > 0) & (src_valid > 0) & (trg_valid > 0)
        
        if not valid_points.any():
            return flow
        
        # Get valid trajectories
        valid_src_trajs = src_trajs[valid_points]  # (M, 2) in pixel coords
        valid_trg_trajs = trg_trajs[valid_points]  # (M, 2)

        # Displacement from trg to src: [dx, dy]
        flow_vectors = valid_src_trajs - valid_trg_trajs  # (M, 2)

        # Round to nearest integer pixel positions for placement
        x_t = torch.round(valid_trg_trajs[:, 0]).long()
        y_t = torch.round(valid_trg_trajs[:, 1]).long()

        # In-bounds mask
        in_bounds = (x_t >= 0) & (x_t < W) & (y_t >= 0) & (y_t < H)

        if in_bounds.any():
            x_ib = x_t[in_bounds]
            y_ib = y_t[in_bounds]
            flow_ib = flow_vectors[in_bounds]  # (M, 2)

            if filter_instances:
                # Compute background (0) and floor/max id filtering
                max_id = torch.max(torch.max(src_mask), torch.max(trg_mask)).item()
                src_ok = (src_mask[y_ib, x_ib] != 0) & (src_mask[y_ib, x_ib] != max_id)
                trg_ok = (trg_mask[y_ib, x_ib] != 0) & (trg_mask[y_ib, x_ib] != max_id)
                keep = src_ok & trg_ok
                if keep.any():
                    # Assign flow vectors: flow[:, y, x] = [dx, dy] for (2, H, W) format
                    flow[:, y_ib[keep], x_ib[keep]] = flow_ib[keep].T  # flow_ib[keep] is (K, 2), .T makes (2, K)
            else:
                # Assign flow vectors: flow[:, y, x] = [dx, dy] for (2, H, W) format
                flow[:, y_ib, x_ib] = flow_ib.T  # flow_ib is (M, 2), .T makes (2, M)
        
        return flow

    def _downsample_flow_for_cats(self, flow: torch.Tensor, feat_size: int) -> torch.Tensor:
        """Downsample (2, H, W) flow to (2, feat_size, feat_size) normalized to feature grid units.
        Only invalid (inf) values are excluded from averaging. Zero vectors are valid and included.
        Flow values are normalized to feature grid units to match CATS convention:
        - A flow of 1.0 = one feature grid cell = (H // feat_size) pixels
        - This matches how CATS stores flow from keypoint annotations.
        """
        if flow is None:
            return flow
        _, H, W = flow.shape  # flow is (2, H, W)
        flow_batch = flow.unsqueeze(0)  # (1, 2, H, W)

        # Valid mask: only exclude infs, zeros are valid (represent zero motion)
        # Check for finite values (includes zeros, excludes infs)
        valid_mask = torch.isfinite(flow_batch).all(dim=1, keepdim=True)  # (1, 1, H, W)

        # Set invalid values to zero for pooling (temporary, won't affect masked average)
        flow_for_pool = flow_batch.clone()
        flow_for_pool[~valid_mask.expand_as(flow_for_pool)] = 0
        
        # Calculate scale factors for converting averages to sums
        scale_factor_h = H / feat_size
        scale_factor_w = W / feat_size
        
        # Sum of valid flow values in each pooling region
        flow_sum = torch.nn.functional.adaptive_avg_pool2d(
            flow_for_pool, (feat_size, feat_size)
        ) * (scale_factor_h * scale_factor_w)  # Multiply back to get sum
        
        # Count of valid pixels in each pooling region
        valid_count = torch.nn.functional.adaptive_avg_pool2d(
            valid_mask.float(), (feat_size, feat_size)
        ) * (scale_factor_h * scale_factor_w)  # Multiply back to get count
        
        # Compute masked average: divide sum by count of valid pixels (not total pixels)
        # Avoid division by zero for regions with no valid pixels
        valid_count_safe = torch.clamp(valid_count, min=1e-8)
        flow_ds = flow_sum / valid_count_safe

        # Normalize flow to feature grid units to match CATS convention
        # CATS expects flow in feature grid units, not pixel space
        # A flow of 1.0 = one feature grid cell = (H // feat_size) pixels
        # Use the same dimension for both x and y to match other datasets (FlyingThings, SPair, etc.)
        # For square images (which PointOdyssey uses), H == W, so this is consistent
        downsampling_factor = H // feat_size
        flow_ds = flow_ds / downsampling_factor

        # Mark regions with no valid pixels as invalid (set to 0)
        # For sparse flow (like PointOdyssey keypoints), use a very low threshold
        # to preserve patches that have any valid flow, even if sparse
        # Threshold of > 0 means we keep any patch with at least some valid pixels
        valid_mask_downsampled = valid_count > 1e-6  # Keep patches with any valid pixels
        flow_ds[~valid_mask_downsampled.expand_as(flow_ds)] = 0

        return flow_ds.squeeze(0)  # (2, feat_size, feat_size)


    # ---- PyTorch-friendly device API ----
    @property
    def device(self):
        return self._device

    def to(self, device):
        new_device = torch.device(device)
        if self._device != new_device:
            self._device = new_device
        return self

    def cuda(self, device=None):
        if device is None:
            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                new_device = torch.device(f'cuda:{idx}')
            else:
                new_device = torch.device('cuda')  # will error on use if unavailable
        else:
            if isinstance(device, int):
                new_device = torch.device(f'cuda:{device}')
            else:
                new_device = torch.device(device)
        if self._device != new_device:
            self._device = new_device
        return self

    def cpu(self):
        new_device = torch.device('cpu')
        if self._device != new_device:
            self._device = new_device
        return self
    
    def _resize_sample(self, src_img: torch.Tensor, trg_img: torch.Tensor, 
                      flow: torch.Tensor, size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Resize images and flow to target size.
        
        Args:
            src_img: Source image (C, H, W)
            trg_img: Target image (C, H, W)
            flow: Flow field (H, W, 2)
            target_size: Target size (H, W)
            
        Returns:
            Resized src_img, trg_img, flow
        """
        target_h, target_w = size, size
        
        # Resize images
        src_img_resized = torch.nn.functional.interpolate(
            src_img.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False
        ).squeeze(0)
        
        trg_img_resized = torch.nn.functional.interpolate(
            trg_img.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False
        ).squeeze(0)
        
        # Resize flow
        # Flow needs special handling - we need to scale the flow vectors by the resize factor
        orig_h, orig_w = flow.shape[:2]
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        # Resize flow field
        flow_resized = torch.nn.functional.interpolate(
            flow.permute(2, 0, 1).unsqueeze(0),  # (1, 2, H, W)
            size=(target_h, target_w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)  # (H, W, 2)
        
        # Scale flow vectors by resize factors
        flow_resized[:, :, 0] *= scale_x  # x component
        flow_resized[:, :, 1] *= scale_y  # y component
        
        return src_img_resized, trg_img_resized, flow_resized
    
    def _create_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Create a dummy sample when the base dataset fails."""
        print("Creating dummy sample")
        if self.size is not None:
            H, W = self.size, self.size
        else:
            H, W = 368, 496
            
        return {
            'src_img': torch.zeros((3, H, W), dtype=torch.float32, device=self._device),
            'trg_img': torch.zeros((3, H, W), dtype=torch.float32, device=self._device),
            'flow': torch.zeros((2, H, W), dtype=torch.float32, device=self._device),
            'masks': torch.zeros((self.S, 1, H, W), dtype=torch.int64, device=self._device)
        }
    
    def visualize_masks(self, masks: torch.Tensor, save_path: str = "./debug/class_masks_visualization.png"):
        """
        Visualize instance masks with distinct colors for each instance ID.
        
        Args:
            masks: Instance masks tensor (S, 1, H, W) where values are instance IDs 0-k
            save_path: Path to save the visualization
        """
        S, C, H, W = masks.shape
        
        # Create a figure with subplots for each frame
        fig, axes = plt.subplots(2, (S + 1) // 2, figsize=(4 * ((S + 1) // 2), 8))
        if S == 1:
            axes = [axes]
        elif S <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Get unique instance IDs across all frames
        all_instance_ids = torch.unique(masks).cpu().numpy()
        num_instances = len(all_instance_ids)
        
        # Create a colormap with distinct colors
        # Class 0 (background) gets bright red
        colors = ['red']  # Class 0 = bright red
        if num_instances > 1:
            # Generate distinct colors for other classes
            other_colors = plt.cm.tab20(np.linspace(0, 1, max(1, num_instances - 1)))
            colors.extend([other_colors[i] for i in range(num_instances - 1)])
        
        # Create a custom colormap
        cmap = mcolors.ListedColormap(colors)
        
        # Normalize instance IDs to [0, num_instances-1] for colormap
        id_to_index = {instance_id: i for i, instance_id in enumerate(all_instance_ids)}
        
        for s in range(S):
            mask_frame = masks[s, 0].cpu().numpy()  # (H, W)
            
            # Convert instance IDs to colormap indices
            mask_colored = np.zeros_like(mask_frame, dtype=float)
            for instance_id in all_instance_ids:
                mask_colored[mask_frame == instance_id] = id_to_index[instance_id]
            
            # Plot the mask
            im = axes[s].imshow(mask_colored, cmap=cmap, vmin=0, vmax=num_instances-1)
            axes[s].set_title(f'Frame {s} - Instance Masks')
            axes[s].axis('off')
        
        # Hide unused subplots
        for s in range(S, len(axes)):
            axes[s].axis('off')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=axes, shrink=0.8, aspect=20)
        cbar.set_ticks(range(num_instances))
        cbar.set_ticklabels([f'ID {int(id)}' for id in all_instance_ids])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Mask visualization saved to: {save_path}")
        print(f"Found {num_instances} unique instance IDs: {all_instance_ids}")
        print(f"Class 0 (background) is colored bright red")
    
    def visualize_masks_batch(self, batch_masks: torch.Tensor, save_path: str = "./debug/class_masks_batch_visualization.png"):
        """
        Visualize instance masks for a batch of samples.
        
        Args:
            batch_masks: Batch of instance masks tensor (B, S, 1, H, W)
            save_path: Path to save the visualization
        """
        B, S, C, H, W = batch_masks.shape
        
        # Create a figure with subplots for each sample and frame
        fig, axes = plt.subplots(B, S, figsize=(4 * S, 4 * B))
        if B == 1:
            axes = axes.reshape(1, -1)
        if S == 1:
            axes = axes.reshape(-1, 1)
        
        # Get unique instance IDs across all samples and frames
        all_instance_ids = torch.unique(batch_masks).cpu().numpy()
        num_instances = len(all_instance_ids)
        
        # Create a colormap with distinct colors
        # Class 0 (background) gets bright red
        colors = ['red']  # Class 0 = bright red
        if num_instances > 1:
            # Generate distinct colors for other classes
            other_colors = plt.cm.tab20(np.linspace(0, 1, max(1, num_instances - 1)))
            colors.extend([other_colors[i] for i in range(num_instances - 1)])
        
        # Create a custom colormap
        cmap = mcolors.ListedColormap(colors)
        
        # Normalize instance IDs to [0, num_instances-1] for colormap
        id_to_index = {instance_id: i for i, instance_id in enumerate(all_instance_ids)}
        
        for b in range(B):
            for s in range(S):
                mask_frame = batch_masks[b, s, 0].cpu().numpy()  # (H, W)
                
                # Convert instance IDs to colormap indices
                mask_colored = np.zeros_like(mask_frame, dtype=float)
                for instance_id in all_instance_ids:
                    mask_colored[mask_frame == instance_id] = id_to_index[instance_id]
                
                # Plot the mask
                im = axes[b, s].imshow(mask_colored, cmap=cmap, vmin=0, vmax=num_instances-1)
                axes[b, s].set_title(f'Sample {b}, Frame {s}')
                axes[b, s].axis('off')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=axes, shrink=0.8, aspect=20)
        cbar.set_ticks(range(num_instances))
        cbar.set_ticklabels([f'ID {int(id)}' for id in all_instance_ids])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Batch mask visualization saved to: {save_path}")
        print(f"Found {num_instances} unique instance IDs: {all_instance_ids}")
        print(f"Class 0 (background) is colored bright red")


def test_dataset_with_visualization(dataset_path: str = None, size: Optional[int] = None, downsample_for_cats: bool = False):
    """Test the dataset wrapper with visualization."""
    print("Testing PointOdyssey Flow Dataset with Visualization...")
    
    # Default dataset path if not provided
    if dataset_path is None:
        dataset_path = '/home/spencer/Data/sample'
        print(f"Using default dataset path: {dataset_path}")
        print("To specify a different path, use: python PointOdysseyCorrespondence.py --dataset_path /path/to/pointodyssey")
    else:
        print(f"Using dataset path: {dataset_path}")
    
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        print("Please provide a valid path to the PointOdyssey dataset using --dataset_path")
        return
    
    # The PointOdyssey dataset expects a 'train' subdirectory
    # If the dataset_path points directly to sequences, we need to adjust
    train_path = os.path.join(dataset_path, 'train')
    if not os.path.exists(train_path):
        print(f"Note: No 'train' subdirectory found. The dataset expects sequences to be in {train_path}")
        print("Creating a temporary train directory structure...")
        # We'll modify the dataset to look directly in the provided path
        actual_dataset_path = dataset_path
    else:
        actual_dataset_path = dataset_path
    
    # Create dataset
    dataset = PointOdysseyFlowDataset(
        dataset_location=dataset_path,
        dset='train',
        use_augs=False,
        S=4,
        N=64,
        quick=False,  # Use quick mode for testing
        verbose=True,
        filter_instances=True,
        resize_size=(size+64, size+64),
        crop_size=(size, size),
        all_points=downsample_for_cats,
        downsample_for_cats=downsample_for_cats,
        cats_feat_size=32,
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) == 0:
        print("No samples in dataset")
        return
    
    # Collect samples for batch visualization
    batch_data = []
    batch_masks = []
    max_samples = 4
    
    for i in range(min(len(dataset), max_samples)):
        try:
            sample = dataset[i]
            
            # Images are already float32 in [0, 1] range from dataset
            src_img = sample['src_img']
            trg_img = sample['trg_img']
            
            batch_data.append({
                'src_img': src_img.unsqueeze(0),  # Add batch dimension
                'trg_img': trg_img.unsqueeze(0),  # Add batch dimension
                'flow': sample['flow'].unsqueeze(0)     # Add batch dimension
            })
            
            # Collect sequence masks for visualization (expects B x S x 1 x H x W)
            batch_masks.append(sample['masks'].unsqueeze(0))  # Add batch dimension
            
            print(f"Sample {i}:")
            print(f"  Source shape: {src_img.shape}")
            print(f"  Target shape: {trg_img.shape}")
            print(f"  Flow shape: {sample['flow'].shape}")
            
            # Debug image format
            print(f"  Source dtype: {src_img.dtype}")
            print(f"  Source range: [{src_img.min():.2f}, {src_img.max():.2f}]")
            
            # Check flow statistics
            flow = sample['flow']  # (2, H, W)
            # Valid flow is where both components are finite
            valid_mask = torch.isfinite(flow).all(dim=0)  # (H, W)
            if valid_mask.any():
                print(f"  Valid flow points: {valid_mask.sum().item()}")
                print(f"  Flow range: x=[{flow[0][valid_mask].min():.2f}, {flow[0][valid_mask].max():.2f}], y=[{flow[1][valid_mask].min():.2f}, {flow[1][valid_mask].max():.2f}]")
                # Sample a few valid flow vectors
                valid_indices = torch.nonzero(valid_mask, as_tuple=False)
                if len(valid_indices) > 0:
                    sample_indices = valid_indices[:min(5, len(valid_indices))]
                    sample_flows = flow[:, sample_indices[:, 0], sample_indices[:, 1]].T  # (N, 2)
                    print(f"  Sample flow vectors: {sample_flows[:5]}")
            else:
                print("  No valid flow found")
                
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            continue
    
    if not batch_data:
        print("No valid samples loaded")
        return
    
    # Combine all samples into a single batch
    src_batch = torch.cat([item['src_img'] for item in batch_data], dim=0)
    trg_batch = torch.cat([item['trg_img'] for item in batch_data], dim=0)
    flow_batch = torch.cat([item['flow'] for item in batch_data], dim=0)
    masks_batch = torch.cat(batch_masks, dim=0)
    
    batch_dict = {
        'src_img': src_batch,
        'trg_img': trg_batch,
        'flow': flow_batch
    }
    
    # Visualize masks
    print("\nVisualizing instance masks...")
    dataset_instance = PointOdysseyFlowDataset(
        dataset_location=dataset_path,
        dset='train',
        use_augs=False,
        S=8,
        N=32,
        quick=False,
        verbose=False,
        resize_size=(size+64, size+64),
        crop_size=(size, size),
        all_points=False,
    )
    dataset_instance.visualize_masks_batch(masks_batch, "./debug/class_masks_batch_visualization.png")
    
    print(f"\nBatch shapes:")
    print(f"  src_img: {src_batch.shape}")
    print(f"  trg_img: {trg_batch.shape}")
    print(f"  flow: {flow_batch.shape}")
    
    # Import visualizer
    try:
        from src.data.synth.datasets.visualizers import CorrespondenceVisualizer
        from src.data.synth.datasets.cats_flow_visualizers import CATSFlowVisualizer
        
        # Create visualizer
        visualizer = CorrespondenceVisualizer(
            figsize=(20, 15),
            dpi=150,
            arrow_scale=1.0,
            arrow_density=20
        )

        cats_visualizer = CATSFlowVisualizer(
            feat_size=32,
            figsize=(20, 15),
            dpi=150,
            show_patch_boundaries=True
        )

        # Replace (0,0) flow vectors with (inf, inf) for visualization
        # The visualizer looks for infs to identify invalid values
        # but we need zeros for the model during training
        flow_for_viz = batch_dict['flow'].clone()  # (B, 2, H, W)
        # Check if both components are exactly 0: (B, 2, H, W) -> (B, 1, H, W)
        zero_mask = (flow_for_viz.abs().sum(dim=1, keepdim=True) == 0)  # True where flow is exactly (0,0)
        flow_for_viz[zero_mask.expand_as(flow_for_viz)] = torch.inf  # Replace with inf for visualization
        batch_dict_viz = {
            'src_img': batch_dict['src_img'],
            'trg_img': batch_dict['trg_img'],
            'flow': flow_for_viz
        }
        
        # Visualize with side-by-side layout
        print("\nCreating side-by-side visualization...")
        visualizer.visualize_rendered_batch(
            batch_dict_viz,
            save_path="./debug/pointodyssey_flow_side_by_side.png",
            max_samples=len(batch_data),
            visualization_mode='side_by_side',
            sampling_mode='all_valid'
        )

        batch_dict_downsampled = {
            'src_img': batch_dict['src_img'],
            'trg_img': batch_dict['trg_img'],
            'flow_downsampled': batch_dict['flow']
        }
        
        if downsample_for_cats:
            cats_visualizer.visualize_downsampled_flow_batch(
                batch_dict_downsampled,
                save_path="./debug/pointodyssey_flow_downsampled.png",
                max_samples=len(batch_data)
            )
        
        # Visualize with overlay layout
        print("Creating overlay visualization...")
        visualizer.visualize_rendered_batch(
            batch_dict_viz,
            save_path="./debug/pointodyssey_flow_overlay.png",
            max_samples=len(batch_data),
            visualization_mode='overlay',
            sampling_mode='all_valid'
        )
        
        print("Visualization complete! Check the generated PNG files.")
        
    except ImportError as e:
        print(f"Could not import visualizer: {e}")
        print("Skipping visualization, but dataset test completed successfully.")
    
    return batch_dict


def test_dataset():
    """Test the dataset wrapper without visualization."""
    print("Testing PointOdyssey Flow Dataset...")
    
    # Create dataset
    dataset = PointOdysseyFlowDataset(
        dataset_location='/home/spencer/Data/sample',
        dset='train',
        use_augs=False,
        S=8,
        N=32,
        quick=False,  # Use quick mode for testing
        verbose=True,
        size=256  # Resize to 256x256 (square)
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        # Get a sample
        sample = dataset[0]
        
        print(f"Sample keys: {sample.keys()}")
        print(f"Source image shape: {sample['src_img'].shape}")
        print(f"Target image shape: {sample['trg_img'].shape}")
        print(f"Flow shape: {sample['flow'].shape}")
        
        # Check flow statistics
        flow = sample['flow']  # (2, H, W)
        # Valid flow is where both components are finite
        valid_mask = torch.isfinite(flow).all(dim=0)  # (H, W)
        if valid_mask.any():
            print(f"Valid flow points: {valid_mask.sum().item()}")
            print(f"Flow range: x=[{flow[0][valid_mask].min():.2f}, {flow[0][valid_mask].max():.2f}], y=[{flow[1][valid_mask].min():.2f}, {flow[1][valid_mask].max():.2f}]")
        else:
            print("No valid flow found")
        
        return sample
    else:
        print("No samples in dataset")
        return None


def test_mask_visualization():
    """Test mask visualization specifically."""
    print("Testing Mask Visualization...")
    
    # Create dataset
    dataset = PointOdysseyFlowDataset(
        dataset_location='/home/spencer/Data/sample',
        dset='train',
        use_augs=False,
        S=8,
        N=32,
        quick=False,
        verbose=True
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        # Get a sample
        sample = dataset[0]
        masks = sample['masks']  # Shape: (S, 1, H, W)
        
        print(f"Masks shape: {masks.shape}")
        print(f"Unique instance IDs: {torch.unique(masks)}")
        
        # Visualize masks
        dataset.visualize_masks(masks, "./debug/class_masks_visualization.png")
        
        return sample
    else:
        print("No samples in dataset")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test PointOdyssey flow dataset')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to PointOdyssey dataset directory')
    parser.add_argument('--size', type=int, default=None,
                        help='Target square size for resizing (size x size)')
    parser.add_argument('--dset', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to use')
    parser.add_argument('--visualize', action='store_true',
                        help='Run with visualization')
    parser.add_argument('--masks', action='store_true',
                        help='Test mask visualization only')
    
    args = parser.parse_args()
    
    # Use size directly if provided
    size = args.size if args.size else None
    
    if args.masks:
        # Test mask visualization only
        sample = test_mask_visualization()
    elif args.visualize:
        # Test with visualization
        batch_dict = test_dataset_with_visualization(args.dataset_path, size)
    else:
        # Test without visualization
        sample = test_dataset()

