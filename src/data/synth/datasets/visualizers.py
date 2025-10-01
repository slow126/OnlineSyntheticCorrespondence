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
    
    def __init__(self, figsize: tuple = (15, 10), dpi: int = 100):
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
    """
    
    def __init__(self, figsize: tuple = (15, 10), dpi: int = 100, arrow_scale: float = 1.0, arrow_density: int = 20):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Figure size (width, height) in inches
            dpi: Dots per inch for saved images
            arrow_scale: Scale factor for flow arrows (larger = longer arrows)
            arrow_density: Number of arrows per dimension (higher = more arrows)
        """
        self.figsize = figsize
        self.dpi = dpi
        self.arrow_scale = arrow_scale
        self.arrow_density = arrow_density
        
    def visualize_rendered_batch(self, batch_dict: dict, save_path: Optional[str] = None, max_samples: int = 4, 
                                visualization_mode: str = 'overlay') -> None:
        """
        Visualize a batch of rendered correspondence data.
        
        Args:
            batch_dict: Dictionary with keys 'src', 'trg', 'flow'
                       Each value is a tensor of shape [batch_size, channels, height, width]
            save_path: Path to save the visualization
            max_samples: Maximum number of samples to display
            visualization_mode: 'side_by_side' or 'overlay'
        """
        if not batch_dict or 'src' not in batch_dict or 'trg' not in batch_dict or 'flow' not in batch_dict:
            raise ValueError("batch_dict must contain 'src', 'trg', and 'flow' keys")
        
        src_batch = batch_dict['src']
        trg_batch = batch_dict['trg']
        flow_batch = batch_dict['flow']
        
        batch_size = min(src_batch.shape[0], max_samples)
        
        if visualization_mode == 'side_by_side':
            self._visualize_side_by_side(src_batch, trg_batch, flow_batch, batch_size, save_path)
        elif visualization_mode == 'overlay':
            self._visualize_overlay(src_batch, trg_batch, flow_batch, batch_size, save_path)
        else:
            raise ValueError("visualization_mode must be 'side_by_side' or 'overlay'")
    
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
            
            # Create overlay: src (red channel) + trg (green channel) + flow arrows
            overlay = np.zeros_like(src_img)
            overlay[:, :, 0] = src_img[:, :, 0]  # Red channel = src
            overlay[:, :, 1] = trg_img[:, :, 1]  # Green channel = trg
            overlay[:, :, 2] = (src_img[:, :, 2] + trg_img[:, :, 2]) / 2  # Blue channel = average
            
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
        
        # Create exact pixel coordinate grids (no interpolation)
        step_y = max(1, flow_h // self.arrow_density)
        step_x = max(1, flow_w // self.arrow_density)
        
        # Get exact pixel coordinates
        y_indices = np.arange(0, flow_h, step_y)
        x_indices = np.arange(0, flow_w, step_x)
        
        # Create coordinate grids using exact indices
        y_coords, x_coords = np.meshgrid(y_indices, x_indices, indexing='ij')
        
        # Sample flow at these exact coordinates
        # Note: flow[0] = dx (x-offset), flow[1] = dy (y-offset) based on flow_by_coordinate_matching
        flow_x = flow[y_coords, x_coords, 0]  # dx values
        flow_y = flow[y_coords, x_coords, 1]  # dy values
        
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
        
        # Set axis limits to match combined image
        ax.set_xlim(0, w * 2)
        ax.set_ylim(h, 0)  # Flip y-axis to match image coordinates
    
    def _plot_flow_on_image(self, ax, flow: np.ndarray, img_shape: tuple) -> None:
        """Plot flow arrows on top of an image using exact pixel correspondences."""
        h, w = img_shape
        flow_h, flow_w = flow.shape[:2]
        
        # Create exact pixel coordinate grids (no interpolation)
        step_y = max(1, flow_h // self.arrow_density)
        step_x = max(1, flow_w // self.arrow_density)
        
        # Get exact pixel coordinates
        y_indices = np.arange(0, flow_h, step_y)
        x_indices = np.arange(0, flow_w, step_x)
        
        # Create coordinate grids using exact indices
        y_coords, x_coords = np.meshgrid(y_indices, x_indices, indexing='ij')
        
        # Sample flow at these exact coordinates
        # Note: flow[0] = dx (x-offset), flow[1] = dy (y-offset) based on flow_by_coordinate_matching
        flow_x = flow[y_coords, x_coords, 0]  # dx values
        flow_y = flow[y_coords, x_coords, 1]  # dy values
        
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
        
        # Generate random rainbow colors for each arrow
        num_arrows = len(valid_x)
        colors = self._generate_rainbow_colors(num_arrows)
        
        # Plot arrows with individual colors using exact flow values
        for i in range(num_arrows):
            ax.quiver(valid_x[i], valid_y[i], valid_flow_x[i], valid_flow_y[i],
                     angles='xy', scale_units='xy', scale=1,
                     color=colors[i], alpha=0.8, width=0.003)
        
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
        