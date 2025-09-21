#!/usr/bin/env python3
"""
Test script for OnlineComponentsDatamodule that samples data and saves visualizations.
Based on the datamodule.py template.

This script:
1. Creates an OnlineComponentsDatamodule with various shader configurations
2. Samples data from the datamodule
3. Generates source, target, and flow visualizations
4. Saves the results as image grids

Usage:
    python test_datamodule.py
"""

import os
import sys
import random
import copy
from pathlib import Path
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add project root to path
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))

# Load environment variables
import dotenv
dotenv.load_dotenv(project_root.joinpath('.env'))

# Import required modules
from src.data.synth.datamodule import OnlineComponentsDatamodule
from src.data.synth.datasets.online_dataset import debug_show_image


def visualize_flow_overlay(src_img, trg_img, flow_tensor, subsample=8, scale=1.0):
    """
    Create an overlay visualization showing source and target images with flow vectors.
    
    Args:
        src_img: Source image tensor (C, H, W)
        trg_img: Target image tensor (C, H, W)
        flow_tensor: Flow tensor (2, H, W) containing dx, dy for each pixel
        subsample: Subsample factor for flow vectors (show every Nth vector)
        scale: Scale factor for flow vector visualization
    
    Returns:
        numpy array of the overlay visualization
    """
    # Convert images to numpy and normalize
    src_np = src_img.permute(1, 2, 0).cpu().numpy()
    trg_np = trg_img.permute(1, 2, 0).cpu().numpy()
    flow_np = flow_tensor.cpu().numpy()
    
    # Normalize images to [0, 1]
    src_np = (src_np - src_np.min()) / (src_np.max() - src_np.min())
    trg_np = (trg_np - trg_np.min()) / (trg_np.max() - trg_np.min())
    
    # Create overlay: blend source and target with transparency
    overlay = 0.5 * src_np + 0.5 * trg_np
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(overlay)
    
    # Extract flow components
    dx = flow_np[0]  # horizontal flow
    dy = flow_np[1]  # vertical flow
    
    # Create grid for flow vectors
    h, w = dx.shape
    y_coords, x_coords = np.meshgrid(
        np.arange(0, h, subsample),
        np.arange(0, w, subsample),
        indexing='ij'
    )
    
    # Get flow values at subsampled points
    dx_subsampled = dx[::subsample, ::subsample]
    dy_subsampled = dy[::subsample, ::subsample]
    
    # Filter out invalid flow (inf, nan, or very small magnitudes)
    valid_mask = np.isfinite(dx_subsampled) & np.isfinite(dy_subsampled)
    magnitude = np.sqrt(dx_subsampled**2 + dy_subsampled**2)
    valid_mask &= (magnitude > 0.5)  # Only show vectors with meaningful magnitude
    
    # Debug: Print flow visualization stats
    total_vectors = dx_subsampled.size
    valid_vectors = valid_mask.sum()
    if valid_vectors > 0:
        max_mag = magnitude[valid_mask].max()
        print(f"    Flow viz: {valid_vectors}/{total_vectors} vectors shown, max magnitude: {max_mag:.2f}")
    else:
        print(f"    Flow viz: No valid vectors to display")
    
    if valid_mask.any():
        # Cycle through ROYGBIV colors for each arrow
        roygbiv = [
            (1.0, 0.0, 0.0),    # Red
            (1.0, 0.5, 0.0),    # Orange
            (1.0, 1.0, 0.0),    # Yellow
            (0.0, 1.0, 0.0),    # Green
            (0.0, 0.0, 1.0),    # Blue
            (0.29, 0.0, 0.51),  # Indigo
            (0.56, 0.0, 1.0),   # Violet
        ]
        num_colors = len(roygbiv)
        # Get the indices of valid vectors in flattened order
        valid_indices = np.argwhere(valid_mask)
        # Assign a color to each vector by cycling through ROYGBIV
        arrow_colors = np.array([roygbiv[i % num_colors] for i in range(valid_indices.shape[0])])
        # Draw flow vectors with per-arrow color
        ax.quiver(
            x_coords[valid_mask],
            y_coords[valid_mask],
            dx_subsampled[valid_mask],
            dy_subsampled[valid_mask],
            angles='xy',
            scale_units='xy',
            scale=1,
            color=arrow_colors,
            alpha=0.8,
            width=0.002,
            headwidth=3,
            headlength=3
        )
    
    ax.axis('off')
    
    # Convert plot to numpy array
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    buf = buf[:, :, :3]  # Remove alpha channel
    plt.close(fig)
    
    return buf


def create_test_datamodule():
    """Create and configure the test datamodule."""
    
    # Set up paths
    root = Path('/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/quatJulia.c')
    val_root = Path(os.environ.get('DATADIR', '/tmp')).joinpath('synthetic/')
    
    # Shader configurations
    shaders = {
        'program0': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/terrain_sdf.glsl',
        'program1': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/quatJulia.c',
        'program2': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/mandelbulb.glsl',
        'program3': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/deformed_sphere.glsl',
        'merge': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/merge.glsl',
    }
    
    # Worley sampler configuration
    worley_sampler_dict = {
        'matching_prob': 0.50,
        'terrain_matching_prob': 0.50,
        'texture_scale0': {
            'alpha': {
                'offset': 0.5,
                'scale': 7.5,
            },
        },
        'texture_scale1': {
            'alpha': {
                'offset': 20.0,
                'scale': 80.0,
            },
        },
        'mixing_param0': {
            'alpha': {
                'offset': 0.0,
                'scale': 0.0,
            },
        },
        'mixing_param1': {
            'alpha': {
                'offset': 0.0,
                'scale': 0.0,
            },
        },
        'terrain_texture_scale0': {
            'alpha': {
                'offset': 0.0,
                'scale': 0.5,
            },
        },
        'terrain_texture_scale1': {
            'alpha': {
                'offset': 10.0,
                'scale': 40.0,
            },
        },
        'terrain_mixing_param0': {
            'alpha': {
                'offset': 0.0,
                'scale': 0.0,
            },
        },
        'terrain_mixing_param1': {
            'alpha': {
                'offset': 0.0,
                'scale': 0.0,
            },
        },
    }
    
    # Mandelbulb sampler configuration
    mandelbulb_sampler_dict = {
        'power_range': {
            'min': 1.5,
            'offset': 2.5,
        },
    }
    
    # Background color sampler configuration
    bg_color_sampler_dict = {
        'matching_prob': 0.5,
        'hue_offset': 0.0,
        'hue_scale': 1.0,
        'saturation_offset': 0.0,
        'saturation_scale': 0.5,
        'value_offset': 0.25,
        'value_scale': 0.75,
    }
    
    # Create the datamodule
    dm = OnlineComponentsDatamodule(
        root=root,
        val_root=val_root,
        batch_size=32,
        num_workers=0,  # Use 0 for debugging
        shuffle=False,
        shaders=shaders,
        use_worley_sampler=True,
        worley_sampler_dict=worley_sampler_dict,
        mandelbulb_sampler=mandelbulb_sampler_dict,
        bg_color_sampler_dict=bg_color_sampler_dict,
        use_grf_sampler=False,
        num_samples=100,  # Small number for testing
        seed=random.randint(0, 10000)
    )
    
    return dm


def sample_and_visualize(dm, num_samples=16, save_dir="debug/test_outputs"):
    """
    Sample from the datamodule and create visualizations.
    
    Args:
        dm: The datamodule instance
        num_samples: Number of samples to generate
        save_dir: Directory to save outputs
    """
    
    # Create output directory
    output_dir = Path(save_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize the datamodule
    dm.post_init()
    dm.setup('fit')
    
    print(f"Dataset length: {len(dm.train_data)}")
    print(f"Generating {num_samples} samples...")
    
    # Lists to store images
    src_images = []
    trg_images = []
    flow_overlays = []
    
    # Sample data
    for i in range(min(num_samples, len(dm.train_data))):
        print(f"Processing sample {i+1}/{num_samples}")
        
        # Get a sample from the dataset
        batch = dm.train_data[i]
        
        # Move to GPU and add batch dimension
        batch = [{k: v.unsqueeze(0).cuda() for k, v in x.items()} for x in batch]
        
        # Process through the datamodule
        processed_batch = dm.on_after_batch_transfer(batch, 0)
        
        # Extract images and flow
        src_img = processed_batch['src'].squeeze(0).cpu()
        trg_img = processed_batch['trg'].squeeze(0).cpu()
        flow = processed_batch['flow'].squeeze(0).cpu()
        
        # Normalize images to [0, 1] for visualization
        src_img = (src_img - src_img.min()) / (src_img.max() - src_img.min())
        trg_img = (trg_img - trg_img.min()) / (trg_img.max() - trg_img.min())
        
        # Create flow overlay visualization
        flow_overlay = visualize_flow_overlay(src_img, trg_img, flow, subsample=8, scale=2.0)
        
        # Debug: Print flow statistics
        valid_flow = flow.isfinite().all(0)
        num_valid = valid_flow.sum().item()
        flow_magnitude = torch.sqrt(flow[0]**2 + flow[1]**2)
        max_magnitude = flow_magnitude[valid_flow].max().item() if num_valid > 0 else 0
        print(f"  Sample {i+1}: {num_valid} valid flow vectors, max magnitude: {max_magnitude:.2f}")
        
        # Store for grid creation
        src_images.append(src_img)
        trg_images.append(trg_img)
        flow_overlays.append(torch.from_numpy(flow_overlay.copy()).permute(2, 0, 1).float() / 255.0)  # Convert to CHW format and normalize
    
    # Create and save grids
    print("Creating grids...")
    
    # Source images grid
    src_grid = torchvision.utils.make_grid(src_images, nrow=4, padding=2, pad_value=0.5)
    torchvision.utils.save_image(src_grid, output_dir / 'src_images_grid.png')
    
    # Target images grid
    trg_grid = torchvision.utils.make_grid(trg_images, nrow=4, padding=2, pad_value=0.5)
    torchvision.utils.save_image(trg_grid, output_dir / 'trg_images_grid.png')
    
    # Flow overlay grid
    flow_grid = torchvision.utils.make_grid(flow_overlays, nrow=4, padding=2, pad_value=0.5)
    torchvision.utils.save_image(flow_grid, output_dir / 'flow_overlays_grid.png')
    
    # Create a combined matplotlib figure with three columns
    create_combined_visualization(src_images, trg_images, flow_overlays, output_dir)
    
    print(f"All visualizations saved to {output_dir}")
    
    return src_images, trg_images, flow_overlays


def create_combined_visualization(src_images, trg_images, flow_overlays, output_dir):
    """Create a combined matplotlib visualization with three columns: src, target, flow overlay."""
    
    num_samples = len(src_images)
    nrows = min(4, num_samples)
    
    # Create figure with proper spacing to avoid text overlap
    fig, axes = plt.subplots(nrows, 3, figsize=(15, nrows * 3.5))
    
    # Handle case where we have only one row
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    # Add column labels at the top with more space
    fig.text(0.17, 0.96, 'Source', fontsize=18, ha='center', weight='bold')
    fig.text(0.5, 0.96, 'Target', fontsize=18, ha='center', weight='bold')
    fig.text(0.83, 0.96, 'Flow Overlay', fontsize=18, ha='center', weight='bold')
    
    for i in range(num_samples):
        row = i % nrows
        
        # Source image (column 0)
        src_img = src_images[i].permute(1, 2, 0).cpu().numpy()
        axes[row, 0].imshow(src_img)
        axes[row, 0].axis('off')
        
        # Target image (column 1)
        trg_img = trg_images[i].permute(1, 2, 0).cpu().numpy()
        axes[row, 1].imshow(trg_img)
        axes[row, 1].axis('off')
        
        # Flow overlay (column 2)
        flow_img = flow_overlays[i].permute(1, 2, 0).cpu().numpy()
        axes[row, 2].imshow(flow_img)
        axes[row, 2].axis('off')
        
        # # Add sample label to the left of the source image with more space
        # axes[row, 0].text(-0.15, 0.5, f'Sample {i+1}', transform=axes[row, 0].transAxes, 
        #                  fontsize=14, ha='right', va='center', weight='bold')
    
    # Hide empty subplots if needed
    for i in range(num_samples, nrows * 3):
        row = i // 3
        col = i % 3
        if row < nrows:
            axes[row, col].axis('off')
    
    # Add main title with more space
    plt.suptitle('Online Datamodule Test Results', fontsize=22, weight='bold', y=0.99)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, left=0.15, hspace=0.3, wspace=0.1)
    
    # Save the combined visualization
    plt.savefig(output_dir / 'combined_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Combined visualization saved to {output_dir / 'combined_visualization.png'}")


def main():
    """Main function to run the test."""
    print("Creating test datamodule...")
    
    try:
        # Create the datamodule
        dm = create_test_datamodule()
        
        # Sample and visualize
        src_images, trg_images, flow_visualizations = sample_and_visualize(dm, num_samples=16)
        
        print("Test completed successfully!")
        print(f"Generated {len(src_images)} samples")
        print("Check the debug/test_outputs directory for results")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
