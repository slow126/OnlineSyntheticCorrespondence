"""
MMD validation utilities for computing Maximum Mean Discrepancy between
correct/incorrect predictions and ground truth during validation.

Uses streaming MMD to avoid accumulating all data in memory.
"""

import torch
from typing import List, Tuple, Optional, Dict
from src.mmd.mmd_torch import StreamingMMDTorch
from src.mmd.config import MMDConfig, load_config_from_yaml
from src.data.synth.datasets.flow_utils import upsample_flow


def extract_flow_vectors_at_keypoints_batch(
    pred_flow: torch.Tensor,
    gt_flow: torch.Tensor,
    trg_kps: torch.Tensor,
    correct_ids: List[torch.Tensor],
    n_pts: torch.Tensor,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract flow vectors at keypoint locations for a single batch, separated into correct/incorrect.
    
    This upsamples predicted flow (feature grid) into pixel space before sampling so it aligns
    with the ground-truth flow and the PCK evaluation coordinates.
    """
    batch_size = pred_flow.shape[0]
    pred_corr_list = []
    pred_miss_list = []
    gt_list = []

    # Ensure flows are on the right device
    pred_flow = pred_flow.to(device)
    gt_flow = gt_flow.to(device)

    # Infer pixel resolution from ground-truth flow (already in pixel space)
    _, _, pixel_h, pixel_w = gt_flow.shape

    # Upsample predicted flow to pixel resolution once per batch
    pred_flow_pixel = upsample_flow(pred_flow, (pixel_h, pixel_w), method='bilinear')  # [B, 2, H, W]
    
    for b in range(batch_size):
        npt = n_pts[b].item()
        if npt == 0:
            continue
        
        # Get keypoints for this sample (only valid ones, pixel coordinates)
        kps = trg_kps[b, :, :npt]  # [2, npt]
        kps_x = kps[0].long()
        kps_y = kps[1].long()
        
        # Clamp to valid pixel bounds
        kps_x = torch.clamp(kps_x, 0, pixel_w - 1)
        kps_y = torch.clamp(kps_y, 0, pixel_h - 1)
        
        # Extract flows at keypoint locations (all in pixel space)
        pred_dx = pred_flow_pixel[b, 0, kps_y, kps_x]  # [npt]
        pred_dy = pred_flow_pixel[b, 1, kps_y, kps_x]  # [npt]
        gt_dx = gt_flow[b, 0, kps_y, kps_x]  # [npt]
        gt_dy = gt_flow[b, 1, kps_y, kps_x]  # [npt]

        # Drop keypoints with non-finite flows on either side to avoid NaNs
        finite_mask = torch.isfinite(pred_dx) & torch.isfinite(pred_dy) & torch.isfinite(gt_dx) & torch.isfinite(gt_dy)
        if finite_mask.sum() == 0:
            continue
        pred_dx = pred_dx[finite_mask]
        pred_dy = pred_dy[finite_mask]
        gt_dx = gt_dx[finite_mask]
        gt_dy = gt_dy[finite_mask]
        kps_x = kps_x[finite_mask]
        kps_y = kps_y[finite_mask]
        npt = pred_dx.shape[0]

        # Convert to [x, y, dx, dy] format
        kps_x_float = kps_x.float()
        kps_y_float = kps_y.float()
        
        # Stack all flows: [npt, 4]
        pred_flows = torch.stack([kps_x_float, kps_y_float, pred_dx, pred_dy], dim=1)
        gt_flows = torch.stack([kps_x_float, kps_y_float, gt_dx, gt_dy], dim=1)
        
        # Separate correct vs incorrect (map original correct_ids to filtered set)
        correct_mask = torch.zeros(npt, dtype=torch.bool, device=device)
        if len(correct_ids[b]) > 0:
            orig_correct = correct_ids[b].to(device)
            # Keep only those that survived finite_mask
            valid_indices = torch.arange(n_pts[b].item(), device=device)[finite_mask]
            if valid_indices.numel() > 0:
                correct_mask = torch.isin(valid_indices, orig_correct)
        
        pred_corr_list.append(pred_flows[correct_mask])
        pred_miss_list.append(pred_flows[~correct_mask])
        gt_list.append(gt_flows)
    
    # Concatenate all batches
    pred_corr = torch.cat(pred_corr_list, dim=0) if pred_corr_list else torch.empty((0, 4), device=device, dtype=torch.float32)
    pred_miss = torch.cat(pred_miss_list, dim=0) if pred_miss_list else torch.empty((0, 4), device=device, dtype=torch.float32)
    gt_all = torch.cat(gt_list, dim=0) if gt_list else torch.empty((0, 4), device=device, dtype=torch.float32)
    
    return pred_corr, pred_miss, gt_all

def create_mmd_streaming(mmd_config: Optional[MMDConfig] = None, device: Optional[torch.device] = None) -> StreamingMMDTorch:
    """
    Create a StreamingMMDTorch instance for incremental MMD computation.
    
    Args:
        mmd_config: Optional MMDConfig. If None, loads from mmd_config.yaml (preset: flow_vectors)
        device: Optional device. If None, uses CUDA if available, else CPU
    
    Returns:
        StreamingMMDTorch instance ready for incremental updates
    """
    # Load or create MMD config (use flow_vectors preset for 4D flow vectors)
    if mmd_config is None:
        try:
            mmd_config = load_config_from_yaml('src/configs/mmd_configs/mmd_config.yaml', preset='flow_vectors')
            # Override device from config if device parameter is provided
            if device is not None:
                device_str = str(device) if isinstance(device, str) else str(device.type)
                if device_str.startswith('cuda'):
                    device_str = 'cuda'  # Use 'cuda' instead of 'cuda:0' for config
                mmd_config.device = device_str
        except Exception as e:
            # Fallback to default config
            if device is None:
                device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device_str = str(device) if isinstance(device, str) else str(device.type)
                if device_str.startswith('cuda'):
                    device_str = 'cuda'  # Use 'cuda' instead of 'cuda:0' for config
            
            mmd_config = MMDConfig(
                input_dim=4,
                sigmas=[0.5, 1.0, 2.0],
                features_per_sigma=256,
                seed=42,
                backend='torch',
                device=device_str,
                unbiased=True
            )
    
    # Create RFF map and streaming MMD
    rff_map = mmd_config.create_rff_map()
    streaming_mmd = StreamingMMDTorch(rff_map)
    
    return streaming_mmd


def update_mmd_streaming(
    streaming_mmd: StreamingMMDTorch,
    pred_flow: torch.Tensor,
    gt_flow: torch.Tensor,
    trg_kps: torch.Tensor,
    correct_ids: List[torch.Tensor],
    n_pts: torch.Tensor,
    device: torch.device
):
    """
    Update streaming MMD with flow vectors from current batch.
    
    Args:
        streaming_mmd: StreamingMMDTorch instance to update
        pred_flow: [B, 2, H, W] predicted flow
        gt_flow: [B, 2, H, W] ground truth flow
        trg_kps: [B, 2, max_kps] target keypoints
        correct_ids: List of tensors, each containing indices of correct keypoints per sample
        n_pts: [B] number of valid keypoints per sample
        device: Device to use
    """
    # Extract flow vectors for current batch
    pred_corr, pred_miss, gt_all = extract_flow_vectors_at_keypoints_batch(
        pred_flow, gt_flow, trg_kps, correct_ids, n_pts, device
    )
    
    # Update streaming MMD with each group
    if pred_corr.shape[0] > 0:
        streaming_mmd.update('pred_corr', pred_corr)
    if pred_miss.shape[0] > 0:
        streaming_mmd.update('pred_miss', pred_miss)
    if gt_all.shape[0] > 0:
        streaming_mmd.update('gt', gt_all)


def compute_mmd_from_streaming(streaming_mmd: StreamingMMDTorch) -> Dict[str, float]:
    """
    Compute final MMD^2 values between the three groups after all batches processed.
    
    Args:
        streaming_mmd: StreamingMMDTorch instance that has been updated with all batches
    
    Returns:
        Dictionary with MMD^2 values:
        - 'mmd2_pred_corr_vs_pred_miss': MMD between correct and incorrect predictions
        - 'mmd2_pred_corr_vs_gt': MMD between correct predictions and ground truth
        - 'mmd2_pred_miss_vs_gt': MMD between incorrect predictions and ground truth
        Values are NaN if insufficient samples
    """
    results = {}
    
    # Check if we have enough samples
    has_pred_corr = 'pred_corr' in streaming_mmd.state and streaming_mmd.state['pred_corr']['count'].item() > 0
    has_pred_miss = 'pred_miss' in streaming_mmd.state and streaming_mmd.state['pred_miss']['count'].item() > 0
    has_gt = 'gt' in streaming_mmd.state and streaming_mmd.state['gt']['count'].item() > 0
    
    # Compute MMD^2 between groups
    if has_pred_corr and has_pred_miss:
        results['mmd2_pred_corr_vs_pred_miss'] = streaming_mmd.mmd2('pred_corr', 'pred_miss')
    else:
        results['mmd2_pred_corr_vs_pred_miss'] = float('nan')
    
    if has_pred_corr and has_gt:
        results['mmd2_pred_corr_vs_gt'] = streaming_mmd.mmd2('pred_corr', 'gt')
    else:
        results['mmd2_pred_corr_vs_gt'] = float('nan')
    
    if has_pred_miss and has_gt:
        results['mmd2_pred_miss_vs_gt'] = streaming_mmd.mmd2('pred_miss', 'gt')
    else:
        results['mmd2_pred_miss_vs_gt'] = float('nan')
    
    return results
