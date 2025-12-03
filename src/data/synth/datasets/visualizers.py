"""
Simplified visualization utilities for synthetic correspondence datasets.

This module provides a simple visualizer for geometry and normals data.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from pathlib import Path


class GeometryVisualizer:
    """
    A simple visualizer for geometry data that displays:
    - Source and target geometry/normals in a grid
    - Batch data where [0] is src and [1] is trg
    """
    
    def __init__(self, figsize: tuple = (20, 15), dpi: int = 150):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Figure size (width, height) in inches
            dpi: Dots per inch for saved images
        """
        self.figsize = figsize
        self.dpi = dpi
        
    def visualize_batch(self, batch_data: List, save_path: Optional[str] = None, max_samples: int = 4) -> None:
        """
        Visualize a batch of data where each item is a pair [src, trg].
        Displays both geometry and normals for each pair.
        
        Args:
            batch_data: List where each item is [src_data, trg_data]
            save_path: Path to save the visualization
            max_samples: Maximum number of samples to display
        """
        if not batch_data:
            raise ValueError("batch_data cannot be empty")
        
        # Limit batch size
        batch_size = min(len(batch_data), max_samples)
        
        # Create figure with 4 columns: src_geometry, src_normals, trg_geometry, trg_normals
        fig, axes = plt.subplots(batch_size, 4, figsize=self.figsize, dpi=self.dpi)
        if batch_size == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each sample
        for i in range(batch_size):
            pair = batch_data[i]
            if len(pair) != 2:
                raise ValueError(f"Each item in batch_data must be a pair [src, trg], but item {i} has {len(pair)} elements")
            
            src_data = pair[0]
            trg_data = pair[1]
            
            # Source geometry
            src_geom = self._prepare_data(src_data['geometry'])
            axes[i, 0].imshow(src_geom)
            axes[i, 0].set_title(f'Src Geometry {i+1}')
            axes[i, 0].axis('off')
            
            # Source normals
            src_norm = self._prepare_data(src_data['normals'])
            axes[i, 1].imshow(src_norm)
            axes[i, 1].set_title(f'Src Normals {i+1}')
            axes[i, 1].axis('off')
            
            # Target geometry
            trg_geom = self._prepare_data(trg_data['geometry'])
            axes[i, 2].imshow(trg_geom)
            axes[i, 2].set_title(f'Trg Geometry {i+1}')
            axes[i, 2].axis('off')
            
            # Target normals
            trg_norm = self._prepare_data(trg_data['normals'])
            axes[i, 3].imshow(trg_norm)
            axes[i, 3].set_title(f'Trg Normals {i+1}')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            self._save_figure(fig, save_path)
        else:
            plt.show()
        
        plt.close(fig)
    
    
    def _prepare_data(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array with proper channel ordering."""
        if torch.is_tensor(tensor):
            data = tensor.detach().cpu().numpy()
        else:
            data = tensor
        
        # Handle different channel orders
        if len(data.shape) == 3:  # (C, H, W) or (H, W, C)
            if data.shape[0] == 3:  # (C, H, W)
                data = np.transpose(data, (1, 2, 0))  # Convert to (H, W, C)
        elif len(data.shape) == 2:  # (H, W) - grayscale
            data = np.expand_dims(data, axis=-1)  # Add channel dimension
            data = np.repeat(data, 3, axis=-1)  # Repeat to make RGB
        
        # Normalize data to [0, 1] range for display
        if data.dtype in [np.float32, np.float64]:
            data_min = data.min()
            data_max = data.max()
            if data_max > data_min:
                data = (data - data_min) / (data_max - data_min)
        
        # Ensure data is in [0, 1] range
        data = np.clip(data, 0, 1)
        
        return data
    
    def _save_figure(self, fig: plt.Figure, save_path: str) -> None:
        """Save figure to the specified path."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', pad_inches=0.1)
        print(f"Visualization saved to: {save_path}")

class CorrespondenceVisualizer:
    """
    A visualizer for rendered correspondence data that displays:
    - Source and target images stacked vertically
    - Flow visualization with arrows showing correspondence
    
    Example usage:
        # For regular grid sampling (default)
        visualizer = CorrespondenceVisualizer(sampling_mode='regular')
        
        # For sparse flow data (like PointOdyssey) - plots all valid flow vectors
        visualizer = CorrespondenceVisualizer(sampling_mode='all_valid')
        
        # Visualize with side-by-side layout
        visualizer.visualize_rendered_batch(batch_dict, visualization_mode='side_by_side')
    """
    
    def __init__(self, figsize: tuple = (20, 15), dpi: int = 150, arrow_scale: float = 1.0, arrow_density: int = 30, 
                 sampling_mode: str = 'regular'):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Figure size (width, height) in inches
            dpi: Dots per inch for saved images
            arrow_scale: Scale factor for flow arrows (larger = longer arrows)
            arrow_density: Number of arrows per dimension (higher = more arrows)
            sampling_mode: 'regular' for regular grid sampling, 'all_valid' for all valid flow vectors
        """
        self.figsize = figsize
        self.dpi = dpi
        self.arrow_scale = arrow_scale
        self.arrow_density = arrow_density
        self.sampling_mode = sampling_mode
    
    def set_sampling_mode(self, mode: str) -> None:
        """
        Set the sampling mode for flow visualization.
        
        Args:
            mode: 'regular' for regular grid sampling, 'all_valid' for all valid flow vectors
        """
        if mode not in ['regular', 'all_valid']:
            raise ValueError("sampling_mode must be 'regular' or 'all_valid'")
        self.sampling_mode = mode
        
    def visualize_rendered_batch(self, batch_dict: dict, save_path: Optional[str] = None, max_samples: int = 4, 
                                visualization_mode: str = 'overlay', sampling_mode: str = 'regular') -> None:
        """
        Visualize a batch of rendered correspondence data.
        
        Args:
            batch_dict: Dictionary with keys 'src_img', 'trg_img', 'flow', optionally 'masks'
                       Each value is a tensor of shape [batch_size, channels, height, width]
                       Masks should be shape [batch_size, S, 1, H, W] where S is sequence length
            save_path: Path to save the visualization
            max_samples: Maximum number of samples to display
            visualization_mode: 'side_by_side', 'overlay', or 'overlay_background_aware'
            sampling_mode: 'regular' for regular grid sampling, 'all_valid' for all valid flow vectors
        """
        if not batch_dict or 'src_img' not in batch_dict or 'trg_img' not in batch_dict or 'flow' not in batch_dict:
            raise ValueError("batch_dict must contain 'src_img', 'trg_img', and 'flow' keys")
        
        src_batch = batch_dict['src_img']
        trg_batch = batch_dict['trg_img']
        flow_batch = batch_dict['flow']
        masks_batch = batch_dict.get('masks', None)  # Optional: (batch_size, S, 1, H, W)
        
        batch_size = min(src_batch.shape[0], max_samples)
        
        self.sampling_mode = sampling_mode
        
        if visualization_mode == 'side_by_side':
            self._visualize_side_by_side(src_batch, trg_batch, flow_batch, batch_size, save_path)
        elif visualization_mode == 'overlay':
            self._visualize_overlay(src_batch, trg_batch, flow_batch, batch_size, save_path)
        elif visualization_mode == 'overlay_background_aware':
            if masks_batch is None:
                raise ValueError("masks_batch is required for 'overlay_background_aware' mode")
            self._visualize_overlay_background_aware(src_batch, trg_batch, flow_batch, masks_batch, batch_size, save_path)
        else:
            raise ValueError("visualization_mode must be 'side_by_side', 'overlay', or 'overlay_background_aware'")
    
    def _visualize_side_by_side(self, src_batch, trg_batch, flow_batch, batch_size, save_path):
        """Visualize src and trg side by side with correspondence arrows between them."""
        # Create figure with 1 column for each sample
        fig, axes = plt.subplots(batch_size, 1, figsize=(self.figsize[0], self.figsize[1] * batch_size), dpi=self.dpi)
        if batch_size == 1:
            axes = [axes]
        
        for i in range(batch_size):
            src_img = self._prepare_image(src_batch[i])
            trg_img = self._prepare_image(trg_batch[i])
            flow = self._prepare_flow(flow_batch[i])
            
            # Create side-by-side layout
            h, w = src_img.shape[:2]
            combined_img = np.zeros((h, w * 2, 3))
            combined_img[:, :w] = src_img
            combined_img[:, w:] = trg_img
            
            axes[i].imshow(combined_img)
            axes[i].set_title(f'Sample {i+1}: Src (left) + Trg (right) with Correspondence')
            axes[i].axis('off')
            
            # Plot correspondence arrows between the images
            self._plot_correspondence_arrows(axes[i], flow, w, h)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        else:
            plt.show()
        
        plt.close(fig)
    
    def _visualize_overlay(self, src_batch, trg_batch, flow_batch, batch_size, save_path):
        """Visualize src and trg overlaid with flow arrows on top."""
        # Create figure with 2 columns: src, overlay
        fig, axes = plt.subplots(batch_size, 2, figsize=self.figsize, dpi=self.dpi)
        if batch_size == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(batch_size):
            src_img = self._prepare_image(src_batch[i])
            trg_img = self._prepare_image(trg_batch[i])
            flow = self._prepare_flow(flow_batch[i])
            
            # Show source image
            axes[i, 0].imshow(src_img)
            axes[i, 0].set_title(f'Sample {i+1}: Source Image')
            axes[i, 0].axis('off')
            
            # Create overlay: direct channel assignment with stronger tinting
            overlay = np.zeros_like(src_img)

            # Convert to grayscale for each image
            src_gray = np.mean(src_img, axis=2)  # Shape: (H, W)
            trg_gray = np.mean(trg_img, axis=2)  # Shape: (H, W)

            # Use max of both for base to preserve brightness
            base_gray = np.maximum(src_gray, trg_gray)  # Shape: (H, W)

            # Strong tinting: directly use src/trg grayscale values in respective channels
            # This makes dark objects appear as dark red/green, not hidden
            overlay[:, :, 0] = src_gray  # Red channel = src (dark objects will be dark red)
            overlay[:, :, 1] = trg_gray  # Green channel = trg (white areas will be bright green)
            overlay[:, :, 2] = base_gray  # Blue = max to preserve brightness
            
            axes[i, 1].imshow(overlay)
            axes[i, 1].set_title(f'Sample {i+1}: Overlay (Red=Src, Green=Trg) + Flow')
            axes[i, 1].axis('off')
            
            # Plot flow arrows on the overlay
            self._plot_flow_on_image(axes[i, 1], flow, src_img.shape[:2])
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        else:
            plt.show()
        
        plt.close(fig)
    
    def _visualize_overlay_background_aware(self, src_batch, trg_batch, flow_batch, masks_batch, batch_size, save_path):
        """Visualize src and trg overlaid with flow arrows, using masks to handle background areas specially."""
        # Create figure with 2 columns: src, overlay
        fig, axes = plt.subplots(batch_size, 2, figsize=self.figsize, dpi=self.dpi)
        if batch_size == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(batch_size):
            src_img = self._prepare_image(src_batch[i])
            trg_img = self._prepare_image(trg_batch[i])
            flow = self._prepare_flow(flow_batch[i])
            
            # Extract masks for this sample: (S, 1, H, W) -> get first and last frame masks
            masks = masks_batch[i]  # (S, 1, H, W)
            if torch.is_tensor(masks):
                masks = masks.cpu().numpy()
            
            # Get src mask (first frame, index 0) and trg mask (last frame, index S-1)
            S = masks.shape[0]
            src_mask = masks[0, 0, :, :]  # (H, W) - first frame mask
            trg_mask = masks[S-1, 0, :, :]  # (H, W) - last frame mask
            
            # Identify background pixels: mask[-1] = landscape/background, mask[0] = sky/horizon
            # Background is where mask value is 0 (sky) or max (landscape/background)
            max_mask_value = np.max(masks)
            src_background = (src_mask == 0) | (src_mask == max_mask_value)  # Sky or landscape
            trg_background = (trg_mask == 0) | (trg_mask == max_mask_value)  # Sky or landscape
            background_mask = src_background | trg_background  # Union of both backgrounds
            
            # Show source image
            axes[i, 0].imshow(src_img)
            axes[i, 0].set_title(f'Sample {i+1}: Source Image')
            axes[i, 0].axis('off')
            
            # Create overlay: visualize masks directly - src mask as red, trg mask as green
            overlay = np.zeros_like(src_img)
            
            # Create binary object masks: pixels that are NOT background (sky or landscape)
            src_object_mask = ~src_background  # (H, W) - True where src has objects
            trg_object_mask = ~trg_background  # (H, W) - True where trg has objects
            
            # Red channel: src mask objects (1.0 where objects, 0.0 where background)
            overlay[:, :, 0] = src_object_mask.astype(np.float32)
            
            # Green channel: trg mask objects (1.0 where objects, 0.0 where background)
            overlay[:, :, 1] = trg_object_mask.astype(np.float32)
            
            # Blue channel: 0 (keep it black)
            overlay[:, :, 2] = 0.0
            
            axes[i, 1].imshow(overlay)
            axes[i, 1].set_title(f'Sample {i+1}: Background-Aware Overlay (Red=Src, Green=Trg) + Flow')
            axes[i, 1].axis('off')
            
            # Plot flow arrows on the overlay
            self._plot_flow_on_image(axes[i, 1], flow, src_img.shape[:2])
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        else:
            plt.show()
        
        plt.close(fig)
    
    def _prepare_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert image tensor to numpy array for display."""
        if torch.is_tensor(tensor):
            img = tensor.detach().cpu().numpy()
        else:
            img = tensor
        
        # Handle channel ordering: (C, H, W) -> (H, W, C)
        if len(img.shape) == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        
        # Normalize to [0, 1] range
        if img.dtype in [np.float32, np.float64]:
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
        
        img = np.clip(img, 0, 1)
        return img
    
    def _prepare_flow(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert flow tensor to numpy array."""
        if torch.is_tensor(tensor):
            flow = tensor.detach().cpu().numpy()
        else:
            flow = tensor
        
        # Handle channel ordering: (C, H, W) -> (H, W, C)
        if len(flow.shape) == 3 and flow.shape[0] == 2:
            flow = np.transpose(flow, (1, 2, 0))  # (H, W, 2)
        
        return flow
    
    def _plot_correspondence_arrows(self, ax, flow: np.ndarray, w: int, h: int) -> None:
        """Plot correspondence arrows between side-by-side images using exact pixel correspondences."""
        flow_h, flow_w = flow.shape[:2]
        
        if self.sampling_mode == 'all_valid':
            # Plot all valid flow vectors (useful for sparse flow data)
            x_coords, y_coords, flow_x, flow_y = self._get_all_valid_flow_vectors(flow)
        else:
            # Regular grid sampling (original behavior)
            x_coords, y_coords, flow_x, flow_y = self._get_regular_sampled_flow_vectors(flow)
        
        # Filter out invalid flow (infinite or NaN values)
        valid_mask = np.isfinite(flow_x) & np.isfinite(flow_y)
        
        if not np.any(valid_mask):
            ax.text(0.5, 0.5, 'No valid flow', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            return
        
        # Flow is defined at target pixel coordinates
        # flow[x, y] = [dx, dy] means target pixel [x, y] corresponds to source pixel [x + dx, y + dy]
        # For side-by-side: start from target pixels and draw arrows to their source correspondences
        
        # Start points are target pixel coordinates in right image
        start_x = x_coords[valid_mask] + w  # Move to right image
        start_y = y_coords[valid_mask]
        
        # End points are corresponding source pixels in left image
        # source_pixel = target_pixel + flow
        end_x = x_coords[valid_mask] + flow_x[valid_mask]  # Source pixel x-coordinate
        end_y = y_coords[valid_mask] + flow_y[valid_mask]  # Source pixel y-coordinate
        
        # Generate random rainbow colors for each arrow
        num_arrows = len(start_x)
        colors = self._generate_rainbow_colors(num_arrows)
        
        # Plot arrows with random rainbow colors
        for i in range(num_arrows):
            ax.annotate('', xy=(end_x[i], end_y[i]), xytext=(start_x[i], start_y[i]),
                       arrowprops=dict(arrowstyle='->', color=colors[i], alpha=0.8, lw=1.5))
        
        # Plot keypoint markers: green dots for target (start), red dots for source (end)
        # Target keypoints (green) - at start of arrow in right image
        ax.scatter(start_x, start_y, c='green', s=30, marker='o', alpha=0.9, 
                  edgecolors='darkgreen', linewidths=1, zorder=10, label='Target (trg)')
        # Source keypoints (red) - at end of arrow in left image
        ax.scatter(end_x, end_y, c='red', s=30, marker='o', alpha=0.9, 
                  edgecolors='darkred', linewidths=1, zorder=10, label='Source (src)')
        
        # Set axis limits to match combined image
        ax.set_xlim(0, w * 2)
        ax.set_ylim(h, 0)  # Flip y-axis to match image coordinates
    
    def _get_regular_sampled_flow_vectors(self, flow: np.ndarray) -> tuple:
        """Get flow vectors sampled on a regular grid (original behavior)."""
        flow_h, flow_w = flow.shape[:2]
        
        # Adaptive arrow density based on image size
        adaptive_density = max(self.arrow_density, min(flow_h, flow_w) // 20)
        step_y = max(1, flow_h // adaptive_density)
        step_x = max(1, flow_w // adaptive_density)
        
        # Get exact pixel coordinates
        y_indices = np.arange(0, flow_h, step_y)
        x_indices = np.arange(0, flow_w, step_x)
        
        # Create coordinate grids using exact indices
        y_coords, x_coords = np.meshgrid(y_indices, x_indices, indexing='ij')
        
        # Sample flow at these exact coordinates
        # Note: flow[0] = dx (x-offset), flow[1] = dy (y-offset) based on flow_by_coordinate_matching
        flow_x = flow[y_coords, x_coords, 0]  # dx values
        flow_y = flow[y_coords, x_coords, 1]  # dy values
        
        return x_coords, y_coords, flow_x, flow_y
    
    def _get_all_valid_flow_vectors(self, flow: np.ndarray) -> tuple:
        """Get all valid flow vectors from sparse flow data."""
        flow_h, flow_w = flow.shape[:2]
        
        # Extract flow components
        flow_x = flow[:, :, 0]  # dx values
        flow_y = flow[:, :, 1]  # dy values
        
        # Create coordinate grids for all pixels
        y_coords, x_coords = np.meshgrid(np.arange(flow_h), np.arange(flow_w), indexing='ij')
        
        # Flatten all coordinates and flow values
        x_coords_flat = x_coords.flatten()
        y_coords_flat = y_coords.flatten()
        flow_x_flat = flow_x.flatten()
        flow_y_flat = flow_y.flatten()
        
        return x_coords_flat, y_coords_flat, flow_x_flat, flow_y_flat
    
    def _plot_flow_on_image(self, ax, flow: np.ndarray, img_shape: tuple) -> None:
        """Plot flow arrows on top of an image using exact pixel correspondences."""
        h, w = img_shape
        flow_h, flow_w = flow.shape[:2]
        
        if self.sampling_mode == 'all_valid':
            # Plot all valid flow vectors (useful for sparse flow data)
            x_coords, y_coords, flow_x, flow_y = self._get_all_valid_flow_vectors(flow)
        else:
            # Regular grid sampling (original behavior)
            x_coords, y_coords, flow_x, flow_y = self._get_regular_sampled_flow_vectors(flow)
        
        # Filter out invalid flow (infinite or NaN values)
        valid_mask = np.isfinite(flow_x) & np.isfinite(flow_y)
        
        if not np.any(valid_mask):
            ax.text(0.5, 0.5, 'No valid flow', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            return
        
        # Use exact pixel coordinates and flow values
        valid_x = x_coords[valid_mask]
        valid_y = y_coords[valid_mask]
        valid_flow_x = flow_x[valid_mask]  # Exact dx values
        valid_flow_y = flow_y[valid_mask]  # Exact dy values
        
        # Flow definition: trg + flow = src (flow points from target to source)
        # For overlay: Red = src, Green = trg
        # We want arrows from Green (target) to Red (source)
        # Start at target pixel positions (valid_x, valid_y) and point with flow direction
        # Generate random rainbow colors for each arrow
        num_arrows = len(valid_x)
        colors = self._generate_rainbow_colors(num_arrows)
        
        # Plot arrows from target pixels (green) pointing to source pixels (red)
        for i in range(num_arrows):
            ax.quiver(valid_x[i], valid_y[i], valid_flow_x[i], valid_flow_y[i],
                     angles='xy', scale_units='xy', scale=1,
                     color=colors[i], alpha=0.8, width=0.003)
        
        # Compute source keypoint positions: target + flow
        src_x = valid_x + valid_flow_x
        src_y = valid_y + valid_flow_y
        
        # Plot keypoint markers: green dots for target (start), red dots for source (end)
        # Target keypoints (green) - at start of arrow
        ax.scatter(valid_x, valid_y, c='green', s=30, marker='o', alpha=0.9, 
                  edgecolors='darkgreen', linewidths=1, zorder=10, label='Target (trg)')
        # Source keypoints (red) - at end of arrow
        ax.scatter(src_x, src_y, c='red', s=30, marker='o', alpha=0.9, 
                  edgecolors='darkred', linewidths=1, zorder=10, label='Source (src)')
        
        # Set axis limits to match image
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)  # Flip y-axis to match image coordinates
    
    def _generate_rainbow_colors(self, num_colors: int) -> list:
        """Generate random rainbow colors for arrows."""
        import random
        
        colors = []
        for _ in range(num_colors):
            # Generate random hue (0-360 degrees)
            hue = random.uniform(0, 360)
            # Convert HSV to RGB
            import colorsys
            rgb = colorsys.hsv_to_rgb(hue/360, 0.8, 0.9)  # High saturation and value
            colors.append(rgb)
        
        return colors
    
    def _save_figure(self, fig: plt.Figure, save_path: str) -> None:
        """Save figure to the specified path."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', pad_inches=0.1)
        print(f"Visualization saved to: {save_path}")
        