"""
Helper functions for integrating coresets with validation pipeline.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Any
from .weighted_coreset import WeightedCoreset
from .metrics import (
    codebook_from_coreset,
    recall_train_covers_eval_soft,
    precision_train_wrt_eval_soft,
    outside_mass_fraction_soft,
)


def extract_flow_vectors_from_batch(batch: Dict[str, Any], return_per_image: bool = False, max_flow_magnitude: Optional[float] = None):
    """
    Extract flow vectors as [x, y, dx, dy] from batch.
    
    Args:
        batch: Batch dict with 'flow_full' or 'flow' key
        return_per_image: If True, return list of per-image arrays instead of stacked
    
    Returns:
        If return_per_image=False: (N, 4) array of flow vectors, or None if no valid flows
        If return_per_image=True: List of (N_i, 4) arrays (one per image), or None if no valid flows
    """
    # Get flow_full from batch
    if 'flow_full' in batch:
        flow_full = batch['flow_full']
    elif 'flow' in batch:
        flow_full = batch['flow']
    else:
        return None
    
    if flow_full is None:
        return None
    
    # flow_full is [B, 2, H, W] or [2, H, W]
    if flow_full.dim() == 3:
        flow_full = flow_full.unsqueeze(0)
    
    batch_size, _, H, W = flow_full.shape
    
    all_vectors = []
    
    for b in range(batch_size):
        flow = flow_full[b].cpu().numpy()  # [2, H, W]
        dx = flow[0]  # [H, W]
        dy = flow[1]  # [H, W]
        
        # Create coordinate grid
        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        
        # Flatten
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()
        dx_flat = dx.flatten()
        dy_flat = dy.flatten()
        
        # Filter invalid flows (inf/nan, zero flows, and extreme values)
        # Optical flow magnitude should be <= image diagonal (pixels can't move farther than that)
        if max_flow_magnitude is None:
            # Default: no magnitude filtering (only inf/nan/zero)
            valid_mask = (
                np.isfinite(dx_flat) & 
                np.isfinite(dy_flat) & 
                ~((dx_flat == 0) & (dy_flat == 0))
            )
        else:
            # Filter by magnitude (reject flow > image diagonal)
            valid_mask = (
                np.isfinite(dx_flat) & 
                np.isfinite(dy_flat) & 
                ~((dx_flat == 0) & (dy_flat == 0)) &
                (np.abs(dx_flat) <= max_flow_magnitude) &
                (np.abs(dy_flat) <= max_flow_magnitude)
            )
        
        # Stack to [N, 4] format: [x, y, dx, dy]
        if valid_mask.any():
            flow_vectors = np.stack([
                x_flat[valid_mask],
                y_flat[valid_mask],
                dx_flat[valid_mask],
                dy_flat[valid_mask]
            ], axis=1).astype(np.float32)
            all_vectors.append(flow_vectors)
        else:
            # Add empty array to maintain batch structure if return_per_image=True
            if return_per_image:
                all_vectors.append(np.empty((0, 4), dtype=np.float32))
    
    if len(all_vectors) == 0:
        return None
    
    # Return per-image or stacked
    if return_per_image:
        return all_vectors
    else:
        return np.vstack(all_vectors) if all_vectors else None


def build_coreset_from_dataloader(
    dataloader,
    coreset_config: Dict[str, Any],
    num_batches: Optional[int] = None,
    is_eval: bool = False,
    extract_fn=None
) -> WeightedCoreset:
    """
    Build a coreset by streaming through a dataloader.
    
    Args:
        dataloader: PyTorch DataLoader
        coreset_config: Config dict with K_max, K_overflow, etc.
        num_batches: Limit to this many batches (None = all)
        is_eval: Whether this is an eval dataset (computes epsilon)
        extract_fn: Optional function to extract vectors from batch
                   Default: extract_flow_vectors_from_batch
    
    Returns:
        WeightedCoreset instance
    """
    if extract_fn is None:
        extract_fn = extract_flow_vectors_from_batch
    
    coreset = WeightedCoreset(
        K_max=coreset_config.get('K_max', 10000),
        K_overflow=coreset_config.get('K_overflow', 5000),
        distance=coreset_config.get('distance', 'euclidean'),
        device=coreset_config.get('device', 'cpu'),
        is_eval=is_eval,
    )
    
    batches_processed = 0
    total_vectors = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if num_batches is not None and batches_processed >= num_batches:
            break
        
        vectors = extract_fn(batch)
        
        if vectors is not None and len(vectors) > 0:
            coreset.update(vectors)
            total_vectors += len(vectors)
        
        batches_processed += 1
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1} batches, {total_vectors} vectors...")
    
    coreset.finalize()
    print(f"  Finalized coreset: {len(coreset.get_centers())} centers from {total_vectors} vectors")
    
    return coreset


def compute_bidirectional_coverage(
    train_coreset: WeightedCoreset,
    eval_coreset: WeightedCoreset,
    k: int = 5,
    bandwidth: Optional[float] = None,
    bandwidth_scale: float = 1.0,
    M_train: float = 100.0,
    M_eval: float = 20.0,
    kernel: str = "gaussian",
    batch_size: int = 1024,
    eps: float = 1e-12,
) -> Dict[str, Dict[str, float]]:
    """
    Compute bidirectional soft k-NN coverage metrics between train and eval.
    
    Args:
        train_coreset: Training dataset coreset
        eval_coreset: Evaluation dataset coreset
        k: Number of neighbors for soft k-NN
        bandwidth: Optional bandwidth; if None, inferred from distances
        bandwidth_scale: Multiplier applied to inferred bandwidth
        M_train: Saturation threshold for recall (train mass)
        M_eval: Saturation threshold for precision (eval mass)
        kernel: Kernel type ('gaussian' or 'inverse')
        batch_size: Batch size for distance computations
        eps: Numerical stability epsilon
    
    Returns:
        Dict with two sub-dicts:
            'train_to_eval': Recall/precision/outside for train covering eval
            'eval_to_train': Recall/precision/outside for eval covering train
    """
    train_cb = codebook_from_coreset(train_coreset)
    eval_cb = codebook_from_coreset(eval_coreset)

    def _metrics(a_cb, b_cb):
        recall = recall_train_covers_eval_soft(
            a_cb, b_cb,
            k=k,
            bandwidth=bandwidth,
            bandwidth_scale=bandwidth_scale,
            M_train=M_train,
            eps=eps,
            kernel=kernel,
            batch_size=batch_size,
        )
        precision = precision_train_wrt_eval_soft(
            a_cb, b_cb,
            k=k,
            bandwidth=bandwidth,
            bandwidth_scale=bandwidth_scale,
            M_eval=M_eval,
            eps=eps,
            kernel=kernel,
            batch_size=batch_size,
        )
        outside = 1.0 - precision
        return {
            'recall': recall,
            'precision': precision,
            'outside': outside,
        }

    return {
        'train_to_eval': _metrics(train_cb, eval_cb),
        'eval_to_train': _metrics(eval_cb, train_cb),
    }
