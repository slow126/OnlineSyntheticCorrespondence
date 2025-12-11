"""
Helper functions for integrating coresets with validation pipeline.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Any
from .weighted_coreset import WeightedCoreset
from .metrics import coverage_by_train, extraneous_mass_fraction


def extract_flow_vectors_from_batch(batch: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Extract flow vectors as [x, y, dx, dy] from batch.
    
    Args:
        batch: Batch dict with 'flow_full' or 'flow' key
    
    Returns:
        (N, 4) array of flow vectors, or None if no valid flows
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
        
        # Filter invalid flows (inf/nan and zero flows)
        valid_mask = (
            np.isfinite(dx_flat) & 
            np.isfinite(dy_flat) & 
            ~((dx_flat == 0) & (dy_flat == 0))
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
    
    if len(all_vectors) == 0:
        return None
    
    return np.vstack(all_vectors)


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
    min_count: int = 0
) -> Dict[str, Dict[str, float]]:
    """
    Compute bidirectional coverage metrics between train and eval.
    
    Args:
        train_coreset: Training dataset coreset
        eval_coreset: Evaluation dataset coreset (should have epsilon_scales)
        min_count: Minimum count for absolute coverage
    
    Returns:
        Dict with two sub-dicts:
            'train_to_eval': How well train covers eval
            'eval_to_train': How well eval covers train (extraneous mass)
    """
    train_centers = train_coreset.get_centers()
    train_counts = train_coreset.get_counts()
    eval_centers = eval_coreset.get_centers()
    eval_counts = eval_coreset.get_counts()
    
    # Get epsilon from eval coreset
    epsilon_scales = eval_coreset.get_epsilon_scales()
    if epsilon_scales is None:
        raise ValueError("Eval coreset must have epsilon_scales. Set is_eval=True when building.")
    
    results = {}
    
    # For each epsilon scale, compute metrics
    for eps_name, eps_value in epsilon_scales.items():
        if not eps_name.startswith('eps_'):
            continue
        
        # Train → Eval coverage
        coverage = coverage_by_train(
            train_centers, train_counts, eval_centers,
            epsilon=eps_value, min_count=min_count
        )
        
        # Eval → Train extraneous mass (train centers far from eval)
        extran = extraneous_mass_fraction(
            train_centers, train_counts, eval_centers,
            epsilon=eps_value
        )
        
        results[eps_name] = {
            'epsilon': eps_value,
            **{f'coverage_{k}': v for k, v in coverage.items() if k != 'epsilon'},
            **{f'extraneous_{k}': v for k, v in extran.items() if k != 'epsilon'},
        }
    
    return results
