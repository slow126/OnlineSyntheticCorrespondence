"""
Coverage metrics computation (Steps 4-5 of pipeline).

Computes:
- Cross-dataset directed NN distances
- Coverage metrics with dual normalization (qnorm and rnorm)
- Optional coverage curves over multiple quantiles
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np

from . import faiss_ops
from . import cache


def compute_coverage_metrics(
    train_radius: float,
    eval_radius: float,
    eval_to_train_distances: np.ndarray,
    train_to_eval_distances: np.ndarray,
    k_values: List[int] = [1, 5],
) -> Dict[str, float]:
    """
    Compute coverage metrics with dual normalization.
    
    Args:
        train_radius: Train self-radius (from Step 3)
        eval_radius: Eval self-radius (from Step 3)
        eval_to_train_distances: (N_eval, K) distances from eval to train
        train_to_eval_distances: (N_train, K) distances from train to eval
        k_values: Which k values to extract (e.g., [1, 5])
        
    Returns:
        Dictionary with all coverage metrics
    """
    metrics = {}
    
    for k_idx, k in enumerate(k_values):
        if k > eval_to_train_distances.shape[1]:
            continue  # Skip if k is larger than searched
        
        # Extract distances to k-th neighbor (0-indexed, so k-1)
        d_eval_to_train = eval_to_train_distances[:, k-1]
        d_train_to_eval = train_to_eval_distances[:, k-1]
        
        # Query-normalized (qnorm) - HEADLINE METRICS
        # Uses query set's radius (eval uses R_E, train uses R_T)
        eval_covered_qnorm = np.mean(d_eval_to_train <= eval_radius)
        train_covered_qnorm = np.mean(d_train_to_eval <= train_radius)
        train_outside_qnorm = 1.0 - train_covered_qnorm
        
        # Reference-normalized (rnorm) - DIAGNOSTIC METRICS
        # Uses reference set's radius (eval uses R_T, train uses R_E)
        eval_covered_rnorm = np.mean(d_eval_to_train <= train_radius)
        train_covered_rnorm = np.mean(d_train_to_eval <= eval_radius)
        train_outside_rnorm = 1.0 - train_covered_rnorm
        
        # Store metrics
        metrics[f'eval_covered_by_train_qnorm_k{k}'] = float(eval_covered_qnorm)
        metrics[f'train_covered_by_eval_qnorm_k{k}'] = float(train_covered_qnorm)
        metrics[f'train_outside_eval_qnorm_k{k}'] = float(train_outside_qnorm)
        
        metrics[f'eval_covered_by_train_rnorm_k{k}'] = float(eval_covered_rnorm)
        metrics[f'train_covered_by_eval_rnorm_k{k}'] = float(train_covered_rnorm)
        metrics[f'train_outside_eval_rnorm_k{k}'] = float(train_outside_rnorm)
    
    return metrics


def compute_coverage_curves(
    train_self_distances: np.ndarray,
    eval_self_distances: np.ndarray,
    eval_to_train_distances: np.ndarray,
    train_to_eval_distances: np.ndarray,
    quantiles: List[float] = [0.80, 0.90, 0.95, 0.99],
    k_values: List[int] = [1, 5],
) -> Dict[str, Dict[float, float]]:
    """
    Compute coverage curves over multiple radius quantiles.
    
    This is essentially free since we reuse the same distance arrays,
    just with different threshold values.
    
    Args:
        train_self_distances: (N_train, K) self-distances for train
        eval_self_distances: (N_eval, K) self-distances for eval
        eval_to_train_distances: (N_eval, K) distances from eval to train
        train_to_eval_distances: (N_train, K) distances from train to eval
        quantiles: Quantiles to test
        k_values: Which k values to use
        
    Returns:
        Dictionary mapping metric names to quantile curves
    """
    curves = {}
    
    for k_idx, k in enumerate(k_values):
        if k > eval_to_train_distances.shape[1]:
            continue
        
        # Extract distances to k-th neighbor
        d_eval_to_train = eval_to_train_distances[:, k-1]
        d_train_to_eval = train_to_eval_distances[:, k-1]
        d_train_self = train_self_distances[:, k-1]
        d_eval_self = eval_self_distances[:, k-1]
        
        # For each quantile, compute coverage
        for q in quantiles:
            R_train_q = np.quantile(d_train_self, q)
            R_eval_q = np.quantile(d_eval_self, q)
            
            # Query-normalized
            eval_covered_qnorm = float(np.mean(d_eval_to_train <= R_eval_q))
            train_covered_qnorm = float(np.mean(d_train_to_eval <= R_train_q))
            
            # Reference-normalized
            eval_covered_rnorm = float(np.mean(d_eval_to_train <= R_train_q))
            train_covered_rnorm = float(np.mean(d_train_to_eval <= R_eval_q))
            
            # Store in curves
            metric_base_qnorm_eval = f'eval_covered_by_train_qnorm_k{k}'
            metric_base_qnorm_train = f'train_covered_by_eval_qnorm_k{k}'
            metric_base_rnorm_eval = f'eval_covered_by_train_rnorm_k{k}'
            metric_base_rnorm_train = f'train_covered_by_eval_rnorm_k{k}'
            
            if metric_base_qnorm_eval not in curves:
                curves[metric_base_qnorm_eval] = {}
                curves[metric_base_qnorm_train] = {}
                curves[metric_base_rnorm_eval] = {}
                curves[metric_base_rnorm_train] = {}
            
            curves[metric_base_qnorm_eval][q] = eval_covered_qnorm
            curves[metric_base_qnorm_train][q] = train_covered_qnorm
            curves[metric_base_rnorm_eval][q] = eval_covered_rnorm
            curves[metric_base_rnorm_train][q] = train_covered_rnorm
    
    return curves


def compute_pairwise_coverage(
    train_vectors: np.ndarray,
    eval_vectors: np.ndarray,
    train_radius_data: Dict[str, float],
    eval_radius_data: Dict[str, float],
    cache_dir: Optional[Path] = None,
    train_label: Optional[Tuple[str, str]] = None,
    eval_label: Optional[Tuple[str, str]] = None,
    space: str = "features",
    normalization: str = "norm2x1",
    distance_metric: str = "sqL2",
    alpha: Optional[float] = None,
    direction: str = "both",
    k_max: int = 5,
    k_values: List[int] = [1, 5],
    use_gpu: bool = True,
    index_factory: str = "Flat",
    nprobe: Optional[int] = None,
    batch_size: Optional[int] = None,
    compute_curves: bool = False,
    curve_quantiles: List[float] = [0.80, 0.90, 0.95, 0.99],
    filter_duplicates: bool = True,
    verbose: bool = True,
) -> Dict[str, any]:
    """
    Compute full coverage analysis for a train/eval pair.
    
    Args:
        train_vectors: (N_train, D) training vectors in transformed space
        eval_vectors: (N_eval, D) eval vectors in transformed space
        train_radius_data: Train self-radius data (from Step 3)
        eval_radius_data: Eval self-radius data (from Step 3)
        k_max: Maximum k to search (must be >= max(k_values))
        k_values: Which k values to compute metrics for
        use_gpu: Use GPU acceleration
        index_factory: FAISS index type
        batch_size: Batch size for GPU searches
        compute_curves: Whether to compute coverage curves
        curve_quantiles: Quantiles for coverage curves
        verbose: Print progress
        
    Returns:
        Dictionary with:
            - 'metrics': Coverage metrics dict
            - 'distance_stats': Distance statistics
            - 'curves': Coverage curves (if compute_curves=True)
    """
    if verbose:
        print(f"  Computing pairwise coverage:")
        print(f"    Train: {train_vectors.shape[0]:,} vectors")
        print(f"    Eval: {eval_vectors.shape[0]:,} vectors")
        print(f"    Dim: {train_vectors.shape[1]}")
        print(f"    k_max: {k_max}, k_values: {k_values}")
    
    # Compute directed distances (optionally cached)
    eval_to_train = None
    train_to_eval = None
    if cache_dir is not None and train_label and eval_label:
        train_ds, train_split = train_label
        eval_ds, eval_split = eval_label
        if direction in ("both", "eval_to_train"):
            eval_to_train = cache.load_directed_distances(
                cache_dir,
                train_ds,
                train_split,
                eval_ds,
                eval_split,
                space,
                k_max,
                direction="eval_to_train",
                normalization=normalization,
                distance_metric=distance_metric,
                alpha=alpha,
            )
        if direction in ("both", "train_to_eval"):
            train_to_eval = cache.load_directed_distances(
                cache_dir,
                train_ds,
                train_split,
                eval_ds,
                eval_split,
                space,
                k_max,
                direction="train_to_eval",
                normalization=normalization,
                distance_metric=distance_metric,
                alpha=alpha,
            )

    if direction in ("both", "eval_to_train") and eval_to_train is None:
        eval_to_train = faiss_ops.compute_eval_to_train(
            train_vectors,
            eval_vectors,
            k=k_max,
            use_gpu=use_gpu,
            index_factory=index_factory,
            nprobe=nprobe,
            batch_size=batch_size,
            verbose=verbose,
        )
        if cache_dir is not None and train_label and eval_label:
            cache.save_directed_distances(
                cache_dir,
                train_label[0],
                train_label[1],
                eval_label[0],
                eval_label[1],
                space,
                k_max,
                direction="eval_to_train",
                normalization=normalization,
                distance_metric=distance_metric,
                distances=eval_to_train,
                alpha=alpha,
            )

    if direction in ("both", "train_to_eval") and train_to_eval is None:
        train_to_eval = faiss_ops.compute_train_to_eval(
            train_vectors,
            eval_vectors,
            k=k_max,
            use_gpu=use_gpu,
            index_factory=index_factory,
            nprobe=nprobe,
            batch_size=batch_size,
            verbose=verbose,
        )
        if cache_dir is not None and train_label and eval_label:
            cache.save_directed_distances(
                cache_dir,
                train_label[0],
                train_label[1],
                eval_label[0],
                eval_label[1],
                space,
                k_max,
                direction="train_to_eval",
                normalization=normalization,
                distance_metric=distance_metric,
                distances=train_to_eval,
                alpha=alpha,
            )

    if direction != "both":
        return {
            "metrics": {},
            "distance_stats": {},
            "curves": {},
        }
    
    # Extract radii from radius_data
    train_radius = train_radius_data['radius']
    eval_radius = eval_radius_data['radius']
    
    # Compute coverage metrics
    metrics = compute_coverage_metrics(
        train_radius,
        eval_radius,
        eval_to_train,
        train_to_eval,
        k_values=k_values,
    )
    
    # Add raw distance statistics
    for k_idx, k in enumerate(k_values):
        if k > eval_to_train.shape[1]:
            continue
        
        eval_to_train_k = eval_to_train[:, k-1]
        train_to_eval_k = train_to_eval[:, k-1]
        
        metrics[f'mean_nn_eval_to_train_k{k}'] = float(np.mean(eval_to_train_k))
        metrics[f'median_nn_eval_to_train_k{k}'] = float(np.median(eval_to_train_k))
        metrics[f'p90_nn_eval_to_train_k{k}'] = float(np.quantile(eval_to_train_k, 0.90))
        metrics[f'p95_nn_eval_to_train_k{k}'] = float(np.quantile(eval_to_train_k, 0.95))
        
        metrics[f'mean_nn_train_to_eval_k{k}'] = float(np.mean(train_to_eval_k))
        metrics[f'median_nn_train_to_eval_k{k}'] = float(np.median(train_to_eval_k))
        metrics[f'p90_nn_train_to_eval_k{k}'] = float(np.quantile(train_to_eval_k, 0.90))
        metrics[f'p95_nn_train_to_eval_k{k}'] = float(np.quantile(train_to_eval_k, 0.95))
    
    result = {
        'metrics': metrics,
    }
    
    # Compute coverage curves if requested
    if compute_curves:
        if verbose:
            print(f"  Computing coverage curves over quantiles: {curve_quantiles}")
        
        # Need self-distances for curves - recompute k_max NN for both sets
        if verbose:
            print(f"    Computing train self-distances for curves...")
        train_index = faiss_ops.build_index(train_vectors, use_gpu=use_gpu, index_factory=index_factory, verbose=False)
        try:
            train_self_dists, _ = faiss_ops.compute_knn_distances(
                train_index,
                train_vectors,
                k=k_max,
                exclude_self=True,
                filter_duplicates=filter_duplicates,
                batch_size=batch_size,
                verbose=False,
            )
        finally:
            faiss_ops.release_index(train_index)
        
        if verbose:
            print(f"    Computing eval self-distances for curves...")
        eval_index = faiss_ops.build_index(eval_vectors, use_gpu=use_gpu, index_factory=index_factory, verbose=False)
        try:
            eval_self_dists, _ = faiss_ops.compute_knn_distances(
                eval_index,
                eval_vectors,
                k=k_max,
                exclude_self=True,
                filter_duplicates=filter_duplicates,
                batch_size=batch_size,
                verbose=False,
            )
        finally:
            faiss_ops.release_index(eval_index)
        
        curves = compute_coverage_curves(
            train_self_dists,
            eval_self_dists,
            eval_to_train,
            train_to_eval,
            quantiles=curve_quantiles,
            k_values=k_values,
        )
        
        result['curves'] = curves
        
        if verbose:
            print(f"    Computed {len(curves)} coverage curves")
    
    if verbose:
        print(f"  Coverage computation complete")
    
    return result
