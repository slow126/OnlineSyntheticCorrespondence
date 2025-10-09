import torch
import torch.nn.functional as F
import math
from typing import Dict, Optional, Union, Tuple


def generate_spherical_uv(geometry):
    """Generate UV coordinates using spherical projection"""
    x, y, z = geometry[..., 0], geometry[..., 1], geometry[..., 2]
    
    # Convert to spherical coordinates
    u = 0.5 + torch.atan2(z, x) / (2 * torch.pi)  # Azimuth: [0, 1]
    v = 0.5 - torch.asin(y) / torch.pi            # Elevation: [0, 1]
    
    # Convert to [-1, 1] range for PyTorch grid_sample
    return torch.stack([u, v], dim=-1) * 2 - 1


def generate_rounded_cube_uv(geometry):
    """Generate UV coordinates using rounded cube projection
    
    This creates a more uniform UV mapping by projecting onto the faces of a rounded cube,
    which reduces distortion compared to spherical projection.
    """
    x, y, z = geometry[..., 0], geometry[..., 1], geometry[..., 2]
    
    # Normalize coordinates to unit sphere
    norm = torch.sqrt(x**2 + y**2 + z**2 + 1e-8)
    x_norm = x / norm
    y_norm = y / norm
    z_norm = z / norm
    
    # Find the dominant axis (face of the cube)
    abs_x = torch.abs(x_norm)
    abs_y = torch.abs(y_norm)
    abs_z = torch.abs(z_norm)
    
    # Determine which face to project onto
    max_abs = torch.max(torch.max(abs_x, abs_y), abs_z)
    
    # Initialize UV coordinates
    u = torch.zeros_like(x)
    v = torch.zeros_like(y)
    
    # Project onto each face based on dominant axis
    # +X face
    mask_x_pos = (abs_x == max_abs) & (x_norm > 0)
    u[mask_x_pos] = 0.5 + z_norm[mask_x_pos] / (2 * abs_x[mask_x_pos])
    v[mask_x_pos] = 0.5 + y_norm[mask_x_pos] / (2 * abs_x[mask_x_pos])
    
    # -X face
    mask_x_neg = (abs_x == max_abs) & (x_norm <= 0)
    u[mask_x_neg] = 0.5 - z_norm[mask_x_neg] / (2 * abs_x[mask_x_neg])
    v[mask_x_neg] = 0.5 + y_norm[mask_x_neg] / (2 * abs_x[mask_x_neg])
    
    # +Y face
    mask_y_pos = (abs_y == max_abs) & (y_norm > 0)
    u[mask_y_pos] = 0.5 + x_norm[mask_y_pos] / (2 * abs_y[mask_y_pos])
    v[mask_y_pos] = 0.5 - z_norm[mask_y_pos] / (2 * abs_y[mask_y_pos])
    
    # -Y face
    mask_y_neg = (abs_y == max_abs) & (y_norm <= 0)
    u[mask_y_neg] = 0.5 + x_norm[mask_y_neg] / (2 * abs_y[mask_y_neg])
    v[mask_y_neg] = 0.5 + z_norm[mask_y_neg] / (2 * abs_y[mask_y_neg])
    
    # +Z face
    mask_z_pos = (abs_z == max_abs) & (z_norm > 0)
    u[mask_z_pos] = 0.5 + x_norm[mask_z_pos] / (2 * abs_z[mask_z_pos])
    v[mask_z_pos] = 0.5 + y_norm[mask_z_pos] / (2 * abs_z[mask_z_pos])
    
    # -Z face
    mask_z_neg = (abs_z == max_abs) & (z_norm <= 0)
    u[mask_z_neg] = 0.5 - x_norm[mask_z_neg] / (2 * abs_z[mask_z_neg])
    v[mask_z_neg] = 0.5 + y_norm[mask_z_neg] / (2 * abs_z[mask_z_neg])
    
    # Clamp to [0, 1] range
    u = torch.clamp(u, 0, 1)
    v = torch.clamp(v, 0, 1)
    
    # Convert to [-1, 1] range for PyTorch grid_sample
    return torch.stack([u, v], dim=-1) * 2 - 1


def apply_phong_shading_pytorch(colors, normals, camera, light, ambient=0.4, diffuse=0.6, specular=0.45, specular_exp=10.0):
    """Apply Phong shading to colors using PyTorch operations"""
    # Ensure inputs are on the same device
    device = colors.device
    normals = normals.to(device)
    camera = camera.to(device)
    light = light.to(device)
    
    # Calculate lighting vectors
    # Light direction
    light_dir = light.unsqueeze(1).unsqueeze(1) - camera.unsqueeze(1).unsqueeze(1)
    light_dir = F.normalize(light_dir, dim=-1)
    
    # View direction
    view_dir = camera.unsqueeze(1).unsqueeze(1) - camera.unsqueeze(1).unsqueeze(1)
    view_dir = F.normalize(view_dir, dim=-1)
    
    # Half vector
    half_dir = F.normalize(light_dir + view_dir, dim=-1)
    
    # Calculate lighting components
    # Diffuse
    diffuse_term = torch.clamp(torch.sum(normals * light_dir, dim=-1, keepdim=True), 0, 1)
    
    # Specular
    specular_term = torch.clamp(torch.sum(normals * half_dir, dim=-1, keepdim=True), 0, 1)
    specular_term = torch.pow(specular_term, specular_exp)
    
    # Combine lighting
    lighting = ambient + diffuse * diffuse_term + specular * specular_term
    
    # Apply lighting to colors
    shaded_colors = colors * lighting
    
    # Apply gamma correction
    gamma = torch.tensor(1.0 / 2.2, dtype=shaded_colors.dtype, device=shaded_colors.device)
    shaded_colors = torch.pow(torch.clamp(shaded_colors, 0, 1), gamma)
    
    return shaded_colors


def apply_uv_texture_batched(geometry, normals, object_ids, texture_dict, camera, light, 
                            projection='rounded_cube', ambient=0.4, diffuse=0.6, specular=0.45, specular_exp=10.0):
    """Apply UV texturing with efficient batch processing for GPU operations
    
    Args:
        geometry: (B, H, W, 3) 3D coordinates
        normals: (B, H, W, 3) surface normals
        object_ids: (B, H, W) object ID for each pixel
        texture_dict: Dict mapping object_id -> texture tensor (C, H, W)
        camera: (B, 3) camera positions
        light: (B, 3) light positions
        projection: UV projection method ('spherical' or 'rounded_cube')
        ambient, diffuse, specular, specular_exp: Phong shading parameters
    
    Returns:
        textured_colors: (B, H, W, 3) final textured and shaded colors
    """
    batch_size, height, width = geometry.shape[:3]
    device = geometry.device
    
    # Initialize output
    final_colors = torch.zeros(batch_size, height, width, 3, device=device)
    
    # Get unique object IDs across all batches
    unique_objects = torch.unique(object_ids)
        
    
    for obj_id in unique_objects:
        if obj_id.item() == -1:  # Skip background
            continue
            
        # Create mask for this object across all batches
        obj_mask = (object_ids == obj_id)  # (B, H, W)
        
        if not obj_mask.any():
            continue
        
        # Get assigned texture for this object from the provided dictionary
        if obj_id.item() not in texture_dict:
            print(f"Warning: No texture assigned for object ID {obj_id.item()}, skipping...")
            continue
            
        texture_2d = texture_dict[obj_id.item()]  # (C, H, W)
        
        # Process each batch item separately for this object
        for b in range(batch_size):
            batch_obj_mask = obj_mask[b]  # (H, W)
            
            if not batch_obj_mask.any():
                continue
            
            # Get object pixels for this batch
            obj_geometry = geometry[b][batch_obj_mask]  # (N, 3)
            obj_normals = normals[b][batch_obj_mask]    # (N, 3)
            
            # Generate UV coordinates for this object using the specified projection
            # Calculate object's local coordinate system for consistent UV mapping
            obj_center = obj_geometry.mean(dim=0)
            obj_scale = obj_geometry.std(dim=0) + 1e-6  # Avoid division by zero
            
            # Transform to local space
            local_geometry = (obj_geometry - obj_center) / obj_scale
            
            # Generate UV coordinates using the specified projection method
            if projection == 'spherical':
                obj_uv = generate_spherical_uv(local_geometry)  # (N, 2)
            elif projection == 'rounded_cube':
                obj_uv = generate_rounded_cube_uv(local_geometry)  # (N, 2)
            else:
                raise ValueError(f"Unsupported projection method: {projection}. Use 'spherical' or 'rounded_cube'.")
            
            # Sample from texture
            # Reshape for grid_sample: (1, C, H, W) and (1, 1, N, 2)
            texture_batch = texture_2d.unsqueeze(0)  # (1, C, H, W)
            uv_batch = obj_uv.unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)
            
            # Sample colors
            sampled_colors = F.grid_sample(
                texture_batch, 
                uv_batch, 
                mode='bilinear', 
                padding_mode='border',
                align_corners=False
            )  # (1, C, 1, N)
            
            # Reshape back
            obj_colors = sampled_colors.squeeze(0).squeeze(1).T  # (N, C)
            
            # Apply Phong shading with batch-specific camera/light
            batch_camera = camera[b:b+1]  # (1, 3)
            batch_light = light[b:b+1]    # (1, 3)
            
            obj_shaded = apply_phong_shading_pytorch(
                obj_colors.unsqueeze(0),  # (1, N, 3)
                obj_normals.unsqueeze(0),  # (1, N, 3)
                batch_camera, batch_light, ambient, diffuse, specular, specular_exp
            ).squeeze(0)  # (N, 3)
            
            # Store back in final result
            final_colors[b][batch_obj_mask] = obj_shaded.to(dtype=torch.float32, device=final_colors.device)
    
    return final_colors


def create_random_texture(size=(3, 256, 256), device='cpu'):
    """Create a random texture for testing
    
    Args:
        size: (C, H, W) size of texture
        device: Device to create tensor on
    
    Returns:
        texture: Random texture tensor
    """
    return torch.rand(size, device=device)




# Example usage and testing
if __name__ == "__main__":
    # Test with dummy data
    batch_size, height, width = 2, 64, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create dummy geometry and normals
    geometry = torch.randn(batch_size, height, width, 3, device=device)
    normals = F.normalize(torch.randn(batch_size, height, width, 3, device=device), dim=-1)
    
    # Create dummy object IDs (0=background, 1,2,3=objects)
    object_ids = torch.randint(0, 4, (batch_size, height, width), device=device)
    
    # Create dummy textures as a dictionary mapping object_id -> texture
    texture_dict = {
        1: create_random_texture(device=device),
        2: create_random_texture(device=device),
        3: create_random_texture(device=device),
    }
    
    # Create dummy camera and light
    camera = torch.randn(batch_size, 3, device=device)
    light = torch.randn(batch_size, 3, device=device)
    
    # Apply UV texturing with spherical projection
    result_spherical = apply_uv_texture_batched(
        geometry, normals, object_ids, texture_dict, camera, light, projection='spherical'
    )
    
    # Apply UV texturing with rounded cube projection
    result_cube = apply_uv_texture_batched(
        geometry, normals, object_ids, texture_dict, camera, light, projection='rounded_cube'
    )
    
    print(f"Spherical result shape: {result_spherical.shape}")
    print(f"Spherical result range: [{result_spherical.min():.3f}, {result_spherical.max():.3f}]")
    print(f"Rounded cube result shape: {result_cube.shape}")
    print(f"Rounded cube result range: [{result_cube.min():.3f}, {result_cube.max():.3f}]")