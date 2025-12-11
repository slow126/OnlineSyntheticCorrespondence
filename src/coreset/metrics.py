"""
Distance-based coverage metrics for weighted coresets.

All metrics use only distances and counts—no kernels or RFF.
"""

import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict, Optional, Tuple


def compute_nn_distances(
    centers: np.ndarray,
    queries: np.ndarray,
    metric: str = 'euclidean',
    batch_size: int = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute nearest neighbor distances from queries to centers.
    
    Args:
        centers: (N_centers, D) array of center points
        queries: (N_queries, D) array of query points
        metric: Distance metric for scipy.cdist
        batch_size: Process queries in batches to manage memory
    
    Returns:
        nn_distances: (N_queries,) array of distances to nearest center
        nn_indices: (N_queries,) array of indices of nearest centers
    """
    n_queries = len(queries)
    nn_distances = np.zeros(n_queries)
    nn_indices = np.zeros(n_queries, dtype=int)
    
    for i in range(0, n_queries, batch_size):
        batch = queries[i:i+batch_size]
        # cdist: (batch_size, n_centers)
        dists = cdist(batch, centers, metric=metric)
        nn_distances[i:i+len(batch)] = dists.min(axis=1)
        nn_indices[i:i+len(batch)] = dists.argmin(axis=1)
    
    return nn_distances, nn_indices


def estimate_epsilon_from_eval(
    eval_vectors: np.ndarray,
    quantile: float = 0.5,
    max_samples: int = 50000
) -> Dict[str, float]:
    """
    Estimate epsilon scale from eval distribution geometry.
    
    Computes distance from each eval point to its nearest OTHER eval point,
    then returns the specified quantile as epsilon.
    
    This provides a dataset-specific distance scale that reflects the
    natural spacing of points in the eval distribution.
    
    Args:
        eval_vectors: (N, D) array of evaluation vectors
        quantile: Quantile to use for base epsilon (default: 0.5 = median)
        max_samples: Subsample to this many points if dataset is larger
    
    Returns:
        Dict with multiple epsilon scales:
            'eps_base': quantile of intra-eval NN distances
            'eps_2x': 2 * eps_base
            'eps_4x': 4 * eps_base
            'nn_dists_stats': dict of statistics about intra-eval distances
    
    Example:
        >>> eval_data = np.random.randn(1000, 4)
        >>> epsilon_scales = estimate_epsilon_from_eval(eval_data)
        >>> print(epsilon_scales['eps_base'])
    """
    # Subsample if too large
    if len(eval_vectors) > max_samples:
        indices = np.random.choice(len(eval_vectors), max_samples, replace=False)
        eval_vectors = eval_vectors[indices]
    
    # Compute pairwise distances
    dists = cdist(eval_vectors, eval_vectors, metric='euclidean')
    
    # For each point, find distance to nearest OTHER point
    # Set diagonal to inf to exclude self-distances
    np.fill_diagonal(dists, np.inf)
    nn_dists = dists.min(axis=1)
    
    eps_base = float(np.quantile(nn_dists, quantile))
    
    return {
        'eps_base': eps_base,
        'eps_2x': 2.0 * eps_base,
        'eps_4x': 4.0 * eps_base,
        'nn_dists_stats': {
            'mean': float(nn_dists.mean()),
            'median': float(np.median(nn_dists)),
            'p25': float(np.quantile(nn_dists, 0.25)),
            'p75': float(np.quantile(nn_dists, 0.75)),
            'p95': float(np.quantile(nn_dists, 0.95)),
        }
    }


def coverage_by_train(
    train_centers: np.ndarray,
    train_counts: np.ndarray,
    eval_vectors: np.ndarray,
    epsilon: float,
    min_count: int = 0,
    batch_size: int = 10000
) -> Dict[str, float]:
    """
    Compute how well train covers eval.
    
    For each eval point, finds the nearest train center and computes:
    - coverage_rel: fraction of eval points within epsilon (ignores counts)
    - coverage_abs: fraction within epsilon AND nearest center has count >= min_count
    - quantile radii: distribution of distances from eval to nearest train
    
    Args:
        train_centers: (K, D) array of train coreset centers
        train_counts: (K,) array of counts for each center
        eval_vectors: (N_eval, D) array of eval points
        epsilon: Distance threshold for coverage
        min_count: Minimum count required for absolute coverage
        batch_size: Process eval points in batches
    
    Returns:
        Dict with:
            'coverage_rel': relative coverage (0 to 1)
            'coverage_abs': absolute coverage with min_count threshold (0 to 1)
            'rho_95': 95th percentile of eval→train distances
            'rho_median': median of eval→train distances
            'rho_mean': mean of eval→train distances
            'epsilon': epsilon value used
            'min_count': min_count value used
    
    Example:
        >>> coverage = coverage_by_train(
        ...     train_centers, train_counts, eval_data,
        ...     epsilon=5.0, min_count=100
        ... )
        >>> print(f"Coverage: {coverage['coverage_rel']:.2%}")
    """
    n_eval = len(eval_vectors)
    nn_distances = np.zeros(n_eval)
    nn_indices = np.zeros(n_eval, dtype=int)
    
    # Compute nearest train center for each eval point
    for i in range(0, n_eval, batch_size):
        batch = eval_vectors[i:i+batch_size]
        dists = cdist(batch, train_centers, metric='euclidean')
        nn_distances[i:i+len(batch)] = dists.min(axis=1)
        nn_indices[i:i+len(batch)] = dists.argmin(axis=1)
    
    # Coverage metrics
    coverage_rel = (nn_distances <= epsilon).mean()
    
    # Absolute coverage: requires sufficient count
    if min_count > 0:
        has_support = train_counts[nn_indices] >= min_count
        coverage_abs = ((nn_distances <= epsilon) & has_support).mean()
    else:
        coverage_abs = coverage_rel
    
    return {
        'coverage_rel': float(coverage_rel),
        'coverage_abs': float(coverage_abs),
        'rho_95': float(np.percentile(nn_distances, 95)),
        'rho_median': float(np.median(nn_distances)),
        'rho_mean': float(nn_distances.mean()),
        'epsilon': epsilon,
        'min_count': min_count,
    }


def extraneous_mass_fraction(
    train_centers: np.ndarray,
    train_counts: np.ndarray,
    eval_vectors: np.ndarray,
    epsilon: float,
    batch_size: int = 10000
) -> Dict[str, float]:
    """
    Compute fraction of train mass that is far from any eval point.
    
    For each train center, finds the nearest eval point. Returns the fraction
    of total training count that lies beyond epsilon from any eval point.
    
    This measures how much training mass is "out-of-distribution" relative
    to the eval data.
    
    Args:
        train_centers: (K, D) array of train coreset centers
        train_counts: (K,) array of counts for each center
        eval_vectors: (N_eval, D) array of eval points
        epsilon: Distance threshold
        batch_size: Process centers in batches
    
    Returns:
        Dict with:
            'extraneous_mass_frac': fraction of count beyond epsilon (0 to 1)
            'extraneous_centers_frac': fraction of centers beyond epsilon (0 to 1)
            'epsilon': epsilon value used
            'extraneous_count': absolute count beyond epsilon
            'total_count': total count across all centers
    
    Example:
        >>> extran = extraneous_mass_fraction(
        ...     train_centers, train_counts, eval_data, epsilon=5.0
        ... )
        >>> print(f"Extraneous mass: {extran['extraneous_mass_frac']:.2%}")
    """
    n_centers = len(train_centers)
    nn_distances = np.zeros(n_centers)
    
    # Compute nearest eval point for each train center
    for i in range(0, n_centers, batch_size):
        batch = train_centers[i:i+batch_size]
        dists = cdist(batch, eval_vectors, metric='euclidean')
        nn_distances[i:i+len(batch)] = dists.min(axis=1)
    
    # Fraction of count that is extraneous
    extraneous_mask = nn_distances > epsilon
    extraneous_count = train_counts[extraneous_mask].sum()
    total_count = train_counts.sum()
    
    return {
        'extraneous_mass_frac': float(extraneous_count / total_count),
        'extraneous_centers_frac': float(extraneous_mask.mean()),
        'epsilon': epsilon,
        'extraneous_count': float(extraneous_count),
        'total_count': float(total_count),
    }
