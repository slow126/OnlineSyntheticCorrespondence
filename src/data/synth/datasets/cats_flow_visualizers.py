"""
Visualization utilities for debugging downsampled flow vectors used by CATS models.

This module provides a visualizer for displaying downsampled flow vectors (feat_size x feat_size)
on full-resolution images, helping debug poor performance on datasets like PointOdyssey and FlyingThings.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from pathlib import Path


class CATSFlowVisualizer:
    """
    A visualizer for downsampled flow vectors used by CATS models.
    
    Displays full-resolution images side-by-side with downsampled flow vectors
    plotted at patch centers, helping debug flow downsampling issues.
    
    Example usage:
        visualizer = CATSFlowVisualizer(feat_size=32)
        visualizer.visualize_downsampled_flow_batch(
            batch_dict, 
            save_path='debug/downsampled_flow.png'
        )
    """
    
    def __init__(self, feat_size: int, figsize: tuple = (20, 15), dpi: int = 150, 
                 show_patch_boundaries: bool = False, normalize_images: bool = True):
        """
        Initialize the visualizer.
        
        Args:
            feat_size: The downsampled feature size (e.g., 32 for 32x32 downsampled flow)
            figsize: Figure size (width, height) in inches
            dpi: Dots per inch for saved images
            show_patch_boundaries: If True, display patch boundaries as grid lines
            normalize_images: If True, normalize images to [0,1]. If False, only clip to [0,1] to see actual values.
        """
        self.feat_size = feat_size
        self.figsize = figsize
        self.dpi = dpi
        self.show_patch_boundaries = show_patch_boundaries
        self.normalize_images = normalize_images
    
    def visualize_downsampled_flow_batch(self, batch_dict: dict, save_path: Optional[str] = None, 
                                         max_samples: int = 4, visualization_mode: str = 'side_by_side') -> None:
        """
        Visualize a batch of downsampled flow vectors on full-resolution images.
        
        Flow is expected to be in feature grid units (normalized by downsampling factor),
        as per CATS convention. It will be converted to pixel space for visualization.
        
        Args:
            batch_dict: Dictionary with keys:
                       - 'src_img': Source images (B, 3, H, W)
                       - 'trg_img': Target images (B, 3, H, W)
                       - 'flow_downsampled': Downsampled flow (B, 2, feat_size, feat_size) in feature grid units
                       - 'flow_full' (optional): Full-resolution flow (B, 2, H, W) for comparison
            save_path: Path to save the visualization
            max_samples: Maximum number of samples to display
            visualization_mode: 'side_by_side' or 'overlay'
        """
        if not batch_dict or 'src_img' not in batch_dict or 'trg_img' not in batch_dict:
            raise ValueError("batch_dict must contain 'src_img' and 'trg_img' keys")
        
        if 'flow_downsampled' not in batch_dict:
            raise ValueError("batch_dict must contain 'flow_downsampled' key")
        
        src_batch = batch_dict['src_img']
        trg_batch = batch_dict['trg_img']
        flow_ds_batch = batch_dict['flow_downsampled']
        
        batch_size = min(src_batch.shape[0], max_samples)
        
        if visualization_mode == 'side_by_side':
            self._visualize_side_by_side_downsampled(src_batch, trg_batch, flow_ds_batch, batch_size, save_path)
        elif visualization_mode == 'overlay':
            self._visualize_overlay_downsampled(src_batch, trg_batch, flow_ds_batch, batch_size, save_path)
        else:
            raise ValueError("visualization_mode must be 'side_by_side' or 'overlay'")
    
    def visualize_comparison_batch(self, batch_dict: dict, save_path: Optional[str] = None, 
                                   max_samples: int = 4) -> None:
        """
        Visualize both full-resolution and downsampled flow on the same samples for comparison.
        
        Args:
            batch_dict: Dictionary with keys:
                       - 'src_img': Source images (B, 3, H, W)
                       - 'trg_img': Target images (B, 3, H, W)
                       - 'flow_downsampled': Downsampled flow (B, 2, feat_size, feat_size)
                       - 'flow_full': Full-resolution flow (B, 2, H, W)
            save_path: Path to save the visualization
            max_samples: Maximum number of samples to display
        """
        if not batch_dict or 'src_img' not in batch_dict or 'trg_img' not in batch_dict:
            raise ValueError("batch_dict must contain 'src_img' and 'trg_img' keys")
        
        if 'flow_downsampled' not in batch_dict:
            raise ValueError("batch_dict must contain 'flow_downsampled' key")
        
        if 'flow_full' not in batch_dict:
            raise ValueError("batch_dict must contain 'flow_full' key for comparison")
        
        src_batch = batch_dict['src_img']
        trg_batch = batch_dict['trg_img']
        flow_ds_batch = batch_dict['flow_downsampled']
        flow_full_batch = batch_dict['flow_full']
        
        batch_size = min(src_batch.shape[0], max_samples)
        
        # Create figure with 3 columns: full-res overlay, downsampled side-by-side, downsampled overlay
        fig, axes = plt.subplots(batch_size, 3, figsize=(self.figsize[0] * 1.5, self.figsize[1] * batch_size), dpi=self.dpi)
        if batch_size == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(batch_size):
            src_img = self._prepare_image(src_batch[i], normalize=self.normalize_images)
            trg_img = self._prepare_image(trg_batch[i], normalize=self.normalize_images)
            flow_ds = self._prepare_flow(flow_ds_batch[i])
            flow_full = self._prepare_flow(flow_full_batch[i])
            
            # Left column: Full-resolution flow overlay
            overlay_full = np.zeros_like(src_img)
            overlay_full[:, :, 0] = src_img[:, :, 0]  # Red channel = src
            overlay_full[:, :, 1] = trg_img[:, :, 1]  # Green channel = trg
            overlay_full[:, :, 2] = (src_img[:, :, 2] + trg_img[:, :, 2]) / 2  # Blue channel = average
            
            axes[i, 0].imshow(overlay_full)
            axes[i, 0].set_title(f'Sample {i+1}: Full-Res Flow Overlay')
            axes[i, 0].axis('off')
            self._plot_flow_on_image(axes[i, 0], flow_full, src_img.shape[:2])
            
            # Middle column: Downsampled flow side-by-side
            h, w = src_img.shape[:2]
            combined_img = np.zeros((h, w * 2, 3))
            combined_img[:, :w] = src_img
            combined_img[:, w:] = trg_img
            
            axes[i, 1].imshow(combined_img)
            axes[i, 1].set_title(f'Sample {i+1}: Downsampled Flow Side-by-Side')
            axes[i, 1].axis('off')
            self._plot_downsampled_flow_arrows(axes[i, 1], flow_ds, w, h, src_img.shape[:2])
            
            # Right column: Downsampled flow overlay
            overlay_ds = np.zeros_like(src_img)
            overlay_ds[:, :, 0] = src_img[:, :, 0]  # Red channel = src
            overlay_ds[:, :, 1] = trg_img[:, :, 1]  # Green channel = trg
            overlay_ds[:, :, 2] = (src_img[:, :, 2] + trg_img[:, :, 2]) / 2  # Blue channel = average
            
            axes[i, 2].imshow(overlay_ds)
            axes[i, 2].set_title(f'Sample {i+1}: Downsampled Flow Overlay')
            axes[i, 2].axis('off')
            self._plot_downsampled_flow_on_image(axes[i, 2], flow_ds, src_img.shape[:2])
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        else:
            plt.show()
        
        plt.close(fig)
    
    def _visualize_side_by_side_downsampled(self, src_batch, trg_batch, flow_ds_batch, batch_size, save_path):
        """Visualize downsampled flow with side-by-side layout."""
        # Create figure with 1 row for each sample
        fig, axes = plt.subplots(batch_size, 1, figsize=(self.figsize[0], self.figsize[1] * batch_size), dpi=self.dpi)
        if batch_size == 1:
            axes = [axes]
        
        for i in range(batch_size):
            src_img = self._prepare_image(src_batch[i], normalize=self.normalize_images)
            trg_img = self._prepare_image(trg_batch[i], normalize=self.normalize_images)
            flow_ds = self._prepare_flow(flow_ds_batch[i])
            
            # Get image dimensions
            h, w = src_img.shape[:2]
            
            # Create side-by-side layout
            combined_img = np.zeros((h, w * 2, 3))
            combined_img[:, :w] = src_img
            combined_img[:, w:] = trg_img
            
            axes[i].imshow(combined_img)
            axes[i].set_title(f'Sample {i+1}: Downsampled Flow (feat_size={self.feat_size}) - Src (left) + Trg (right)')
            axes[i].axis('off')
            
            # Plot downsampled flow arrows at patch centers
            self._plot_downsampled_flow_arrows(axes[i], flow_ds, w, h, src_img.shape[:2])
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        else:
            plt.show()
        
        plt.close(fig)
    
    def _visualize_overlay_downsampled(self, src_batch, trg_batch, flow_ds_batch, batch_size, save_path):
        """Visualize downsampled flow with overlay layout."""
        # Create figure with 2 columns: src, overlay
        fig, axes = plt.subplots(batch_size, 2, figsize=self.figsize, dpi=self.dpi)
        if batch_size == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(batch_size):
            src_img = self._prepare_image(src_batch[i], normalize=self.normalize_images)
            trg_img = self._prepare_image(trg_batch[i], normalize=self.normalize_images)
            flow_ds = self._prepare_flow(flow_ds_batch[i])
            
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
            axes[i, 1].set_title(f'Sample {i+1}: Overlay (Red=Src, Green=Trg) + Downsampled Flow')
            axes[i, 1].axis('off')
            
            # Plot downsampled flow arrows on the overlay
            self._plot_downsampled_flow_on_image(axes[i, 1], flow_ds, src_img.shape[:2])
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        else:
            plt.show()
        
        plt.close(fig)
    
    def _plot_downsampled_flow_on_image(self, ax, flow_ds: np.ndarray, img_shape: tuple) -> None:
        """Plot downsampled flow vectors at patch centers on a single image (for overlay mode).
        
        Flow is expected to be in feature grid units (normalized by downsampling factor).
        It will be converted back to pixel space for visualization.
        """
        img_h, img_w = img_shape
        
        # Calculate scale factors for patch centers
        scale_h = img_h / self.feat_size
        scale_w = img_w / self.feat_size
        
        # Convert flow from feature grid units to pixel space
        # Flow is in feature grid units: 1.0 = one patch = (img_h // feat_size) pixels
        downsampling_factor = img_h // self.feat_size
        
        # Verify flow dimensions match feat_size
        flow_ds_h, flow_ds_w = flow_ds.shape[:2]
        if flow_ds_h != self.feat_size or flow_ds_w != self.feat_size:
            raise ValueError(f"Flow shape {flow_ds.shape[:2]} does not match feat_size {self.feat_size}")
        
        # Create coordinate grids for patch centers
        y_indices = np.arange(self.feat_size)
        x_indices = np.arange(self.feat_size)
        y_coords, x_coords = np.meshgrid(y_indices, x_indices, indexing='ij')
        
        # Calculate patch centers in full-resolution coordinates
        patch_center_x = (x_coords + 0.5) * scale_w
        patch_center_y = (y_coords + 0.5) * scale_h
        
        # Extract flow components (in feature grid units)
        flow_x = flow_ds[:, :, 0]  # dx values in feature grid units
        flow_y = flow_ds[:, :, 1]  # dy values in feature grid units
        
        # Flatten arrays
        patch_center_x_flat = patch_center_x.flatten()
        patch_center_y_flat = patch_center_y.flatten()
        # Convert from feature grid units to pixel space
        flow_x_flat = flow_x.flatten() * downsampling_factor
        flow_y_flat = flow_y.flatten() * downsampling_factor
        
        # Filter out invalid flow (infinite or NaN values)
        valid_mask = np.isfinite(flow_x_flat) & np.isfinite(flow_y_flat)
        
        if not np.any(valid_mask):
            ax.text(0.5, 0.5, 'No valid flow', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            return
        
        # Get valid coordinates and flow values
        valid_x_centers = patch_center_x_flat[valid_mask]
        valid_y_centers = patch_center_y_flat[valid_mask]
        valid_flow_x = flow_x_flat[valid_mask]
        valid_flow_y = flow_y_flat[valid_mask]
        
        # Filter out zero flow vectors (0, 0)
        non_zero_mask = (valid_flow_x != 0) | (valid_flow_y != 0)
        
        if not np.any(non_zero_mask):
            ax.text(0.5, 0.5, 'No non-zero flow', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            return
        
        # Get non-zero flow vectors only
        valid_x_centers = valid_x_centers[non_zero_mask]
        valid_y_centers = valid_y_centers[non_zero_mask]
        valid_flow_x = valid_flow_x[non_zero_mask]
        valid_flow_y = valid_flow_y[non_zero_mask]
        
        # For overlay mode: plot arrows on the image
        # Start from patch centers, end at center + flow
        start_x = valid_x_centers
        start_y = valid_y_centers
        end_x = valid_x_centers + valid_flow_x
        end_y = valid_y_centers + valid_flow_y
        
        # Generate rainbow colors for each arrow
        num_arrows = len(start_x)
        colors = self._generate_rainbow_colors(num_arrows)
        
        # Plot arrows with colors (exact pixel correspondences, no scaling)
        for i in range(num_arrows):
            ax.annotate('', xy=(end_x[i], end_y[i]), xytext=(start_x[i], start_y[i]),
                       arrowprops=dict(arrowstyle='->', color=colors[i], alpha=0.8, lw=0.8))
        
        # Optionally draw patch boundaries
        if self.show_patch_boundaries:
            # Draw vertical lines (patch boundaries)
            for j in range(self.feat_size + 1):
                x_line = j * scale_w
                ax.axvline(x_line, color='yellow', alpha=0.3, linewidth=0.5)
            
            # Draw horizontal lines (patch boundaries)
            for i in range(self.feat_size + 1):
                y_line = i * scale_h
                ax.axhline(y_line, color='yellow', alpha=0.3, linewidth=0.5)
        
        # Set axis limits to match image
        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)  # Flip y-axis to match image coordinates
    
    def _plot_flow_on_image(self, ax, flow: np.ndarray, img_shape: tuple) -> None:
        """Plot full-resolution flow vectors on an image (helper for comparison)."""
        h, w = img_shape
        flow_h, flow_w = flow.shape[:2]
        
        # Extract flow components
        flow_x = flow[:, :, 0]  # dx values
        flow_y = flow[:, :, 1]  # dy values
        
        # Create coordinate grids for all pixels
        y_coords, x_coords = np.meshgrid(np.arange(flow_h), np.arange(flow_w), indexing='ij')
        
        # Flatten arrays
        x_coords_flat = x_coords.flatten()
        y_coords_flat = y_coords.flatten()
        flow_x_flat = flow_x.flatten()
        flow_y_flat = flow_y.flatten()
        
        # Filter out invalid flow (infinite or NaN values)
        valid_mask = np.isfinite(flow_x_flat) & np.isfinite(flow_y_flat)
        
        if not np.any(valid_mask):
            ax.text(0.5, 0.5, 'No valid flow', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            return
        
        # Filter out zero flow vectors (0, 0)
        non_zero_mask = (flow_x_flat[valid_mask] != 0) | (flow_y_flat[valid_mask] != 0)
        final_mask = valid_mask.copy()
        final_mask[valid_mask] = non_zero_mask
        
        if not np.any(final_mask):
            ax.text(0.5, 0.5, 'No non-zero flow', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            return
        
        # Sample flow vectors (every Nth pixel to avoid clutter)
        # Use adaptive sampling based on image size
        stride = max(1, min(flow_h, flow_w) // 20)
        
        # Get sampled coordinates
        valid_x = x_coords_flat[final_mask][::stride]
        valid_y = y_coords_flat[final_mask][::stride]
        valid_flow_x = flow_x_flat[final_mask][::stride]
        valid_flow_y = flow_y_flat[final_mask][::stride]
        
        # Generate rainbow colors
        num_arrows = len(valid_x)
        colors = self._generate_rainbow_colors(num_arrows)
        
        # Plot arrows
        for i in range(num_arrows):
            ax.quiver(valid_x[i], valid_y[i], valid_flow_x[i], valid_flow_y[i],
                     angles='xy', scale_units='xy', scale=1,
                     color=colors[i], alpha=0.8, width=0.003)
        
        # Set axis limits to match image
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)  # Flip y-axis to match image coordinates
    
    def _plot_downsampled_flow_arrows(self, ax, flow_ds: np.ndarray, w: int, h: int, 
                                     img_shape: tuple) -> None:
        """
        Plot downsampled flow vectors at patch centers on side-by-side images.
        
        Args:
            ax: Matplotlib axis to plot on
            flow_ds: Downsampled flow array of shape (feat_size, feat_size, 2) in feature grid units
            w: Width of each image (before combining)
            h: Height of images
            img_shape: Tuple (H, W) of full-resolution image
        """
        flow_ds_h, flow_ds_w = flow_ds.shape[:2]
        img_h, img_w = img_shape
        
        # Calculate scale factors for patch centers
        scale_h = img_h / self.feat_size
        scale_w = img_w / self.feat_size
        
        # Convert flow from feature grid units to pixel space
        # Flow is in feature grid units: 1.0 = one patch = (img_h // feat_size) pixels
        downsampling_factor = img_h // self.feat_size
        
        # Verify flow dimensions match feat_size
        if flow_ds_h != self.feat_size or flow_ds_w != self.feat_size:
            raise ValueError(f"Flow shape {flow_ds.shape[:2]} does not match feat_size {self.feat_size}")
        
        # Create coordinate grids for patch centers
        # For each downsampled location (i, j), patch center is at:
        # x_center = (j + 0.5) * img_w / feat_size
        # y_center = (i + 0.5) * img_h / feat_size
        y_indices = np.arange(self.feat_size)
        x_indices = np.arange(self.feat_size)
        y_coords, x_coords = np.meshgrid(y_indices, x_indices, indexing='ij')
        
        # Calculate patch centers in full-resolution coordinates
        patch_center_x = (x_coords + 0.5) * scale_w
        patch_center_y = (y_coords + 0.5) * scale_h
        
        # Extract flow components (in feature grid units)
        flow_x = flow_ds[:, :, 0]  # dx values in feature grid units
        flow_y = flow_ds[:, :, 1]  # dy values in feature grid units
        
        # Flatten arrays
        patch_center_x_flat = patch_center_x.flatten()
        patch_center_y_flat = patch_center_y.flatten()
        # Convert from feature grid units to pixel space
        flow_x_flat = flow_x.flatten() * downsampling_factor
        flow_y_flat = flow_y.flatten() * downsampling_factor
        
        # Filter out invalid flow (infinite or NaN values)
        valid_mask = np.isfinite(flow_x_flat) & np.isfinite(flow_y_flat)
        
        if not np.any(valid_mask):
            ax.text(0.5, 0.5, 'No valid flow', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            return
        
        # Get valid coordinates and flow values
        valid_x_centers = patch_center_x_flat[valid_mask]
        valid_y_centers = patch_center_y_flat[valid_mask]
        valid_flow_x = flow_x_flat[valid_mask]
        valid_flow_y = flow_y_flat[valid_mask]
        
        # Filter out zero flow vectors (0, 0)
        non_zero_mask = (valid_flow_x != 0) | (valid_flow_y != 0)
        
        if not np.any(non_zero_mask):
            ax.text(0.5, 0.5, 'No non-zero flow', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            return
        
        # Get non-zero flow vectors only
        valid_x_centers = valid_x_centers[non_zero_mask]
        valid_y_centers = valid_y_centers[non_zero_mask]
        valid_flow_x = valid_flow_x[non_zero_mask]
        valid_flow_y = valid_flow_y[non_zero_mask]
        
        # For side-by-side visualization:
        # Start points are target pixel coordinates in right image
        # End points are corresponding source pixels in left image
        start_x = valid_x_centers + w  # Move to right image
        start_y = valid_y_centers
        
        # End points: source pixel = target pixel + flow
        # flow[x, y] = [dx, dy] means target pixel corresponds to source pixel at [x + dx, y + dy]
        end_x = valid_x_centers + valid_flow_x
        end_y = valid_y_centers + valid_flow_y
        
        # Generate rainbow colors for each arrow
        num_arrows = len(start_x)
        colors = self._generate_rainbow_colors(num_arrows)
        
        # Plot arrows with colors (exact pixel correspondences, no scaling)
        for i in range(num_arrows):
            ax.annotate('', xy=(end_x[i], end_y[i]), xytext=(start_x[i], start_y[i]),
                       arrowprops=dict(arrowstyle='->', color=colors[i], alpha=0.8, lw=0.8))
        
        # Optionally draw patch boundaries
        if self.show_patch_boundaries:
            # Draw vertical lines (patch boundaries)
            for j in range(self.feat_size + 1):
                x_line = j * scale_w
                ax.axvline(x_line, color='yellow', alpha=0.3, linewidth=0.5)
                ax.axvline(x_line + w, color='yellow', alpha=0.3, linewidth=0.5)
            
            # Draw horizontal lines (patch boundaries)
            for i in range(self.feat_size + 1):
                y_line = i * scale_h
                ax.axhline(y_line, color='yellow', alpha=0.3, linewidth=0.5)
        
        # Set axis limits to match combined image
        ax.set_xlim(0, w * 2)
        ax.set_ylim(h, 0)  # Flip y-axis to match image coordinates
    
    def _prepare_image(self, tensor: torch.Tensor, normalize: bool = True) -> np.ndarray:
        """Convert image tensor to numpy array for display.
        
        Args:
            tensor: Image tensor
            normalize: If True, normalize to [0, 1] range. If False, clip to [0, 1] without scaling.
        """
        if torch.is_tensor(tensor):
            img = tensor.detach().cpu().numpy()
        else:
            img = tensor
        
        # Handle channel ordering: (C, H, W) -> (H, W, C)
        if len(img.shape) == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        
        # Normalize to [0, 1] range if requested
        if normalize:
            # Convert uint8 (0-255) to float (0-1)
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            # If float and max > 1, normalize from [min, max] to [0, 1]
            elif img.dtype in [np.float32, np.float64]:
                img_max = img.max()
                if img_max > 1.0:
                    img_min = img.min()
                    if img_max > img_min:
                        img = (img - img_min) / (img_max - img_min)
                # If max <= 1, assume already in [0, 1] range
        
        # Clip to valid range for display
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

