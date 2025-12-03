"""
Unified flow processing utilities for correspondence datasets.
These functions handle conversion between flow, keypoints, and trajectories.
"""

import torch 
from typing import Tuple, Optional, Union
import torch.nn.functional as F


def prepare_invalids(flow: torch.Tensor, dataset_name: Optional[str] = None, verbose: bool = False) -> torch.Tensor:
    """
    Unify invalid flow representations across datasets.
    
    Different datasets use different conventions for invalid flow:
    - Sparse flow datasets (benchmark): Use (0, 0) for regions without keypoints
    - Dense flow datasets (FlyingThings, KITTI): Use inf for invalid regions
    
    This function standardizes to using inf for invalid flow.
    
    Args:
        flow: Flow tensor [2, H, W] or [B, 2, H, W]
        dataset_name: Optional dataset name to determine handling strategy
            - 'pfpascal', 'pfwillow', 'caltech', 'spair': Sparse flow (convert (0,0) to inf)
            - 'flyingthings', 'kitti', 'pointodyssey': Dense flow (keep inf as inf)
            - None: Auto-detect based on flow sparsity
        verbose: If True, print diagnostic information
    
    Returns:
        Flow tensor with unified invalid representation (inf for invalid)
    """
    if flow is None:
        return flow
    
    is_batched = flow.dim() == 4
    if not is_batched:
        flow = flow.unsqueeze(0)
    
    B, C, H, W = flow.shape
    
    # Determine if this is a sparse flow dataset
    sparse_datasets = ['pfpascal', 'pfwillow', 'caltech', 'spair']
    is_sparse = dataset_name in sparse_datasets if dataset_name else None
    
    # If dataset_name not provided, try to auto-detect sparsity
    if is_sparse is None:
        # Check if flow has many (0, 0) vectors and few inf values
        # This suggests sparse flow from keypoints
        flow_mag = flow.norm(dim=1)  # [B, H, W]
        zero_vectors = (flow_mag < 1e-6).sum().item()
        inf_vectors = (~flow.isfinite()).any(dim=1).sum().item()
        total_pixels = B * H * W
        
        # If more than 50% are zero and less than 10% are inf, likely sparse
        is_sparse = (zero_vectors > 0.5 * total_pixels) and (inf_vectors < 0.1 * total_pixels)
        
        if verbose:
            print(f"  [prepare_invalids] Auto-detected sparsity: is_sparse={is_sparse}, "
                  f"zero_vectors={zero_vectors}/{total_pixels} ({100*zero_vectors/total_pixels:.1f}%), "
                  f"inf_vectors={inf_vectors}/{total_pixels} ({100*inf_vectors/total_pixels:.1f}%)")
    elif verbose:
        print(f"  [prepare_invalids] Using dataset_name={dataset_name}, is_sparse={is_sparse}")
    
    # Create mask for invalid regions
    if is_sparse:
        # For sparse flow: (0, 0) vectors are invalid (regions without keypoints)
        flow_mag = flow.norm(dim=1, keepdim=True)  # [B, 1, H, W]
        invalid_mask = flow_mag < 1e-6  # Very small magnitude = invalid
        if verbose:
            num_invalid = invalid_mask.sum().item()
            print(f"  [prepare_invalids] Sparse mode: converting {num_invalid} zero vectors to inf")
    else:
        # For dense flow: inf values are invalid
        invalid_mask = ~flow.isfinite().all(dim=1, keepdim=True)  # [B, 1, H, W]
        if verbose:
            num_invalid = invalid_mask.sum().item()
            print(f"  [prepare_invalids] Dense mode: {num_invalid} invalid (inf) regions preserved")
    
    # Set invalid regions to inf
    flow_unified = flow.clone()
    flow_unified[invalid_mask.expand_as(flow_unified)] = float('inf')
    
    if not is_batched:
        flow_unified = flow_unified.squeeze(0)
    
    if verbose:
        final_invalid = (~flow_unified.isfinite()).any(dim=1 if is_batched else 0).sum().item()
        print(f"  [prepare_invalids] Final invalid count: {final_invalid}")
    
    return flow_unified


def flow_from_kps(
    src_kps: torch.Tensor,
    trg_kps: torch.Tensor,
    img_size: Tuple[int, int],
    feat_size: Optional[int] = None,
    receptive_field_size: int = 35,  # Not used anymore, kept for compatibility
    verbose: bool = False
) -> torch.Tensor:
    """
    Convert keypoint correspondences to dense flow field.
    
    Simple version that directly computes flow from correspondences without box searching.
    Creates full-resolution flow field - use downsample_flow() separately if needed.
    
    Flow convention: flow from trg to src, so flow = src_kps - trg_kps, and src = trg + flow.
    Flow vectors are placed at target keypoint locations.
    
    Args:
        src_kps: Source keypoints [2, N] (x, y format)
        trg_kps: Target keypoints [2, N] (x, y format) - already correspondences with src_kps
        img_size: Image size (H, W)
        feat_size: Not used (kept for compatibility). Use downsample_flow() separately.
        receptive_field_size: Not used (kept for compatibility)
        verbose: Whether to print debug info
    
    Returns:
        Flow tensor [2, H, W] in pixel space
        Invalid regions are marked with inf
    """
    H, W = img_size
    
    if verbose:
        print(f"  [flow_from_kps] Input: {src_kps.shape[1]} keypoints, img_size=({H}, {W})")
    
    device = src_kps.device
    dtype = src_kps.dtype
    N = src_kps.shape[1]
    
    # Initialize flow map with inf (invalid)
    flow_map = torch.full((2, H, W), float('inf'), dtype=dtype, device=device)
    
    if N == 0:
        if verbose:
            print(f"  [flow_from_kps] WARNING: No keypoints provided")
        return flow_map
    
    # Compute flow vectors directly from correspondences
    # Flow convention: flow from trg to src, so flow = src_kps - trg_kps
    flow_vectors = src_kps - trg_kps  # [2, N]
    
    # Convert target keypoints to integer pixel coordinates (flow is placed at target locations)
    trg_kps_int = trg_kps.round().long()  # [2, N]
    
    # Clamp to valid image bounds
    trg_kps_int[0] = torch.clamp(trg_kps_int[0], 0, W - 1)  # x coordinates
    trg_kps_int[1] = torch.clamp(trg_kps_int[1], 0, H - 1)  # y coordinates
    
    # Place flow vectors at target keypoint locations
    # flow_map[0, y, x] = dx, flow_map[1, y, x] = dy
    flow_map[0, trg_kps_int[1], trg_kps_int[0]] = flow_vectors[0]  # dx
    flow_map[1, trg_kps_int[1], trg_kps_int[0]] = flow_vectors[1]  # dy
    
    if verbose:
        valid_flow = torch.isfinite(flow_map).all(dim=0).sum().item()
        total_pixels = H * W
        print(f"  [flow_from_kps] Output: {valid_flow}/{total_pixels} valid flow pixels "
              f"({100*valid_flow/total_pixels:.1f}%) - sparse flow field")
    
    return flow_map


def kps_from_flow(
    flow: torch.Tensor,
    num_kps: Optional[int],
    min_flow_magnitude: float = 0.0,
    random_seed: Optional[int] = None,
    verbose: bool = False,
    use_fast_sampling: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Sample keypoints from valid flow regions.
    
    Flow convention: flow from trg to src, so flow = src - trg and src = trg + flow.
    This function samples target keypoints from valid flow regions and computes
    corresponding source keypoints using the flow.
    
    For dense flows, uses grid-based sampling to avoid expensive torch.where operations.
    
    Args:
        flow: Flow tensor [2, H, W] or [B, 2, H, W] (flow from trg to src)
        num_kps: Number of keypoints to sample (None = use all valid keypoints)
        min_flow_magnitude: Minimum flow magnitude to consider valid (default: 0.0)
        random_seed: Optional random seed for reproducible sampling
        use_fast_sampling: If True, use grid-based sampling for dense flows (default: True)
        verbose: If True, print diagnostic information
    
    Returns:
        trg_kps: Target keypoints [2, num_kps] or [B, 2, num_kps] (x, y format)
        src_kps: Source keypoints [2, num_kps] or [B, 2, num_kps] (computed as src = trg + flow)
        n_valid: Number of valid keypoints (actual, not padded)
    """
    is_batched = flow.dim() == 4
    if not is_batched:
        flow = flow.unsqueeze(0)
    
    B, _, H, W = flow.shape
    
    if verbose:
        print(f"  [kps_from_flow] Input: flow shape {list(flow.shape)}, num_kps={num_kps}, "
              f"min_flow_magnitude={min_flow_magnitude}, use_fast_sampling={use_fast_sampling}")
    
    # Find valid flow regions (not inf and magnitude >= min_flow_magnitude)
    flow_mag = flow.norm(dim=1)  # [B, H, W]
    valid_mask = flow_mag.isfinite() & (flow_mag > min_flow_magnitude)
    
    if verbose:
        num_valid = valid_mask.sum().item()
        total_pixels = B * H * W
        print(f"  [kps_from_flow] Found {num_valid}/{total_pixels} valid flow pixels "
              f"({100*num_valid/total_pixels:.1f}%)")
    
    trg_kps_list = []
    src_kps_list = []
    n_valid_list = []
    
    for b in range(B):
        batch_valid_mask = valid_mask[b]  # [H, W]
        
        if not batch_valid_mask.any():
            # No valid points - return zeros
            if verbose:
                print(f"  [kps_from_flow] Batch {b}: No valid flow pixels, returning zeros")
            kps_size = num_kps if num_kps is not None else 0
            trg_kps = torch.zeros((2, kps_size), dtype=torch.float32, device=flow.device)
            src_kps = torch.zeros((2, kps_size), dtype=torch.float32, device=flow.device)
            trg_kps_list.append(trg_kps)
            src_kps_list.append(src_kps)
            n_valid_list.append(0)
            continue
        
        # Fast path: grid-based extraction for dense flows (avoids expensive torch.where)
        # Use this for both sampling (num_kps set) and dense extraction (num_kps is None)
        if use_fast_sampling and (H * W) > 10000:
            # For large dense flows, use grid-based approach
            if num_kps is not None:
                # Sampling mode: Sample ~4x more points than needed to account for invalid ones
                grid_stride = max(1, int((H * W / (num_kps * 4)) ** 0.5))
            else:
                # Dense mode: Use stride=1 to get all pixels (still faster than torch.where)
                grid_stride = 1
            
            # Create grid coordinates
            y_coords = torch.arange(0, H, grid_stride, device=flow.device)
            x_coords = torch.arange(0, W, grid_stride, device=flow.device)
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
            y_flat = y_grid.flatten()
            x_flat = x_grid.flatten()
            
            # Filter by validity (vectorized - much faster than torch.where)
            valid_indices = batch_valid_mask[y_flat, x_flat]
            valid_y = y_flat[valid_indices]
            valid_x = x_flat[valid_indices]
            num_valid_from_grid = len(valid_y)
            
            if num_kps is None:
                # Dense mode: Use all valid points from grid
                sampled_y = valid_y
                sampled_x = valid_x
                n_valid = num_valid_from_grid
            elif num_valid_from_grid >= num_kps:
                # Randomly sample from grid points
                if random_seed is not None:
                    generator = torch.Generator(device=flow.device)
                    generator.manual_seed(random_seed + b)
                    perm = torch.randperm(num_valid_from_grid, generator=generator, device=flow.device)
                else:
                    perm = torch.randperm(num_valid_from_grid, device=flow.device)
                indices = perm[:num_kps]
                sampled_y = valid_y[indices]
                sampled_x = valid_x[indices]
                n_valid = num_kps
            else:
                # Not enough from grid - fallback to torch.where for remaining points
                if verbose:
                    print(f"  [kps_from_flow] Batch {b}: Grid sampling found {num_valid_from_grid} valid points, "
                          f"need {num_kps}, falling back to full sampling")
                # Use all grid points, then sample more
                remaining = num_kps - num_valid_from_grid
                all_valid_y, all_valid_x = torch.where(batch_valid_mask)
                num_all_valid = len(all_valid_y)
                
                if num_all_valid <= num_kps:
                    # Use all valid points
                    indices = torch.arange(num_all_valid, device=flow.device)
                    sampled_y = all_valid_y[indices]
                    sampled_x = all_valid_x[indices]
                    n_valid = num_all_valid
                else:
                    # Combine grid points with additional random samples
                    if random_seed is not None:
                        generator = torch.Generator(device=flow.device)
                        generator.manual_seed(random_seed + b + 1000)
                        perm = torch.randperm(num_all_valid, generator=generator, device=flow.device)
                    else:
                        perm = torch.randperm(num_all_valid, device=flow.device)
                    additional_indices = perm[:remaining]
                    
                    # Combine grid and additional samples
                    sampled_y = torch.cat([valid_y, all_valid_y[additional_indices]])
                    sampled_x = torch.cat([valid_x, all_valid_x[additional_indices]])
                    n_valid = num_kps
        else:
            # Standard path: use torch.where (fine for small flows or when num_kps is None)
            valid_y, valid_x = torch.where(batch_valid_mask)
            num_valid = len(valid_y)
            
            if num_kps is None:
                indices = torch.arange(num_valid, device=flow.device)
                sampled_y = valid_y[indices]
                sampled_x = valid_x[indices]
                n_valid = num_valid
            elif num_valid <= num_kps:
                indices = torch.arange(num_valid, device=flow.device)
                sampled_y = valid_y[indices]
                sampled_x = valid_x[indices]
                n_valid = num_valid
            else:
                if random_seed is not None:
                    generator = torch.Generator(device=flow.device)
                    generator.manual_seed(random_seed + b)
                    indices = torch.randperm(num_valid, generator=generator, device=flow.device)[:num_kps]
                else:
                    indices = torch.randperm(num_valid, device=flow.device)[:num_kps]
                sampled_y = valid_y[indices]
                sampled_x = valid_x[indices]
                n_valid = num_kps
        
        trg_kps = torch.stack([sampled_x.float(), sampled_y.float()])  # [2, n_valid]
        
        # Vectorized flow lookup (much faster than loop)
        y_int = sampled_y.long()
        x_int = sampled_x.long()
        # Clamp to valid range
        y_int = torch.clamp(y_int, 0, H - 1)
        x_int = torch.clamp(x_int, 0, W - 1)
        
        # Advanced indexing: [B, 2, H, W] -> [2, n_valid]
        flow_x = flow[b, 0, y_int, x_int]  # [n_valid]
        flow_y = flow[b, 1, y_int, x_int]  # [n_valid]
        
        src_kps = trg_kps.clone()
        src_kps[0] += flow_x  # x coordinates
        src_kps[1] += flow_y  # y coordinates
        
        # Pad to num_kps if needed (only if num_kps is specified)
        if num_kps is not None and n_valid < num_kps:
            padding = torch.zeros((2, num_kps - n_valid), dtype=torch.float32, device=flow.device)
            trg_kps = torch.cat([trg_kps, padding], dim=1)
            src_kps = torch.cat([src_kps, padding], dim=1)
        
        trg_kps_list.append(trg_kps)
        src_kps_list.append(src_kps)
        n_valid_list.append(n_valid)
        
        if verbose and b == 0:  # Only print for first batch item
            print(f"  [kps_from_flow] Batch {b}: Sampled {n_valid} keypoints")
    
    if is_batched:
        # When using all keypoints (num_kps is None), each sample may have different counts
        # Need to pad to max size before stacking
        if num_kps is None:
            max_kps = max(n_valid_list)
            # Pad all keypoints to max_kps
            for b in range(B):
                if trg_kps_list[b].shape[1] < max_kps:
                    pad_size = max_kps - trg_kps_list[b].shape[1]
                    padding = torch.zeros((2, pad_size), dtype=trg_kps_list[b].dtype, device=trg_kps_list[b].device)
                    trg_kps_list[b] = torch.cat([trg_kps_list[b], padding], dim=1)
                    src_kps_list[b] = torch.cat([src_kps_list[b], padding], dim=1)
        
        trg_kps = torch.stack(trg_kps_list, dim=0)  # [B, 2, num_kps or max_kps]
        src_kps = torch.stack(src_kps_list, dim=0)  # [B, 2, num_kps or max_kps]
        return trg_kps, src_kps, n_valid_list[0] if len(n_valid_list) == 1 else n_valid_list
    else:
        return trg_kps_list[0], src_kps_list[0], n_valid_list[0]


def downsample_flow(flow: torch.Tensor, feat_size: int, verbose: bool = False) -> torch.Tensor:
    """
    Downsample flow using masked average pooling.
    
    Uses the same approach as FlyingThingsDataset.FlowDownsampler:
    - Only averages over valid (finite) pixels
    - Normalizes to feature grid units
    - Sets regions with no valid pixels to inf
    
    Args:
        flow: Flow tensor [2, H, W] or [B, 2, H, W] in pixel space
        feat_size: Target feature size (e.g., 32 for 32x32 output)
    
    Returns:
        Downsampled flow [2, feat_size, feat_size] or [B, 2, feat_size, feat_size]
        in feature grid units. Invalid regions are marked with inf.
    """
    if flow is None:
        return flow
    
    is_batched = flow.dim() == 4
    if not is_batched:
        flow = flow.unsqueeze(0)
    
    B, C, H, W = flow.shape
    
    if verbose:
        print(f"  [downsample_flow] Input: {list(flow.shape)} -> target feat_size={feat_size}")
    
    # Calculate scale factors
    scale_factor_h = H / feat_size
    scale_factor_w = W / feat_size
    
    # Create mask for valid flow values (finite)
    valid_mask = torch.isfinite(flow).all(dim=1, keepdim=True)  # [B, 1, H, W]
    
    # Set invalid values to 0 for pooling (temporary, won't affect masked average)
    flow_clean = flow.clone()
    flow_clean[~valid_mask.expand_as(flow_clean)] = 0
    
    # Sum of valid flow values in each pooling region
    flow_sum = F.adaptive_avg_pool2d(
        flow_clean, (feat_size, feat_size)
    ) * (scale_factor_h * scale_factor_w)  # Multiply back to get sum
    
    # Count of valid pixels in each pooling region
    valid_count = F.adaptive_avg_pool2d(
        valid_mask.float(), (feat_size, feat_size)
    ) * (scale_factor_h * scale_factor_w)  # Multiply back to get count
    
    # Compute masked average: divide sum by count of valid pixels
    valid_count_safe = torch.clamp(valid_count, min=1e-8)
    flow_downsampled = flow_sum / valid_count_safe
    
    # Normalize flow to feature grid units to match CATS convention
    # A flow of 1.0 = one feature grid cell = (H // feat_size) pixels
    downsampling_factor = H // feat_size
    flow_downsampled = flow_downsampled / downsampling_factor
    
    # Mark regions with no valid pixels as invalid (set to inf)
    valid_mask_downsampled = valid_count > 0.5  # At least 0.5 valid pixels
    flow_downsampled[~valid_mask_downsampled.expand_as(flow_downsampled)] = float('inf')
    
    if verbose:
        num_valid = torch.isfinite(flow_downsampled).all(dim=1).sum().item()
        total_pixels = B * feat_size * feat_size
        print(f"  [downsample_flow] Output: {list(flow_downsampled.shape)}, "
              f"{num_valid}/{total_pixels} valid pixels ({100*num_valid/total_pixels:.1f}%)")
    
    if not is_batched:
        flow_downsampled = flow_downsampled.squeeze(0)
    
    return flow_downsampled


def upsample_flow(
    flow: torch.Tensor,
    target_size: Tuple[int, int],
    method: str = 'bilinear',
    verbose: bool = False
) -> torch.Tensor:
    """
    Upsample flow to target resolution.
    
    Args:
        flow: Flow tensor [2, H, W] or [B, 2, H, W] in feature grid units
        target_size: Target size (H, W)
        method: Interpolation method ('bilinear' or 'nearest')
    
    Returns:
        Upsampled flow with scaled flow vectors. Invalid regions (inf) are preserved.
    """
    if flow is None:
        return flow
    
    is_batched = flow.dim() == 4
    if not is_batched:
        flow = flow.unsqueeze(0)
    
    B, C, H, W = flow.shape
    target_h, target_w = target_size
    
    if verbose:
        print(f"  [upsample_flow] Input: {list(flow.shape)} -> target_size=({target_h}, {target_w}), method={method}")
    
    # Create mask for invalid regions
    valid_mask = torch.isfinite(flow).all(dim=1, keepdim=True)  # [B, 1, H, W]
    
    # Temporarily set invalid values to 0 for interpolation
    flow_clean = flow.clone()
    flow_clean[~valid_mask.expand_as(flow_clean)] = 0
    
    # Upsample flow
    mode = 'bilinear' if method == 'bilinear' else 'nearest'
    flow_upsampled = F.interpolate(
        flow_clean, size=(target_h, target_w), mode=mode, align_corners=False if method == 'bilinear' else None
    )
    
    # Scale flow vectors by upsampling factor
    # At this point, flow_upsampled is always [B, 2, target_h, target_w] (we added batch dim if needed)
    scale_h = target_h / H
    scale_w = target_w / W
    flow_upsampled[:, 0, :, :] *= scale_w  # x component (channel 0)
    flow_upsampled[:, 1, :, :] *= scale_h  # y component (channel 1)
    
    # Upsample valid mask and restore invalid regions
    valid_mask_upsampled = F.interpolate(
        valid_mask.float(), size=(target_h, target_w), mode='nearest'
    ).bool()
    
    # Set invalid regions back to inf
    flow_upsampled[~valid_mask_upsampled.expand_as(flow_upsampled)] = float('inf')
    
    if verbose:
        num_valid = torch.isfinite(flow_upsampled).all(dim=1).sum().item()
        total_pixels = B * target_h * target_w
        print(f"  [upsample_flow] Output: {list(flow_upsampled.shape)}, "
              f"{num_valid}/{total_pixels} valid pixels ({100*num_valid/total_pixels:.1f}%)")
    
    if not is_batched:
        flow_upsampled = flow_upsampled.squeeze(0)
    
    return flow_upsampled
