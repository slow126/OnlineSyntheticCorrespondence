"""
Distance-based metrics for weighted coresets and codebooks.

Provides soft k-NN precision/recall-style metrics over per-dataset codebooks,
plus utilities for estimating distance scales from eval data.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import numpy as np
import torch
from scipy.spatial.distance import cdist


@dataclass
class DatasetCodebook:
    """Lightweight container for a dataset codebook."""
    centroids: torch.Tensor  # [K, d], float32/float64
    counts: torch.Tensor     # [K], int/float

    @property
    def weights(self) -> torch.Tensor:
        """Return normalized weights (double)."""
        total = torch.clamp(self.counts.sum().double(), min=1.0)
        return (self.counts.double() / total)


def codebook_from_coreset(coreset: "WeightedCoreset") -> DatasetCodebook:
    """
    Convert a WeightedCoreset into a DatasetCodebook.

    Imported lazily to avoid circular imports.
    """
    centroids = torch.from_numpy(coreset.get_centers())
    counts = torch.from_numpy(coreset.get_counts())
    return DatasetCodebook(centroids=centroids, counts=counts)


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

# ---------------------------------------------------------------------------
# Soft k-NN codebook precision/recall metrics
# ---------------------------------------------------------------------------

def _batched_cdist_topk(
    queries: torch.Tensor,
    centers: torch.Tensor,
    k: int,
    batch_size: int = 1024
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute top-k distances/indices from queries to centers using torch.cdist,
    processing queries in batches to limit memory.
    """
    n_queries = queries.shape[0]
    topk_dists = []
    topk_idx = []

    for start in range(0, n_queries, batch_size):
        q_batch = queries[start:start + batch_size]
        dists = torch.cdist(q_batch, centers)
        vals, idx = torch.topk(dists, k=min(k, centers.shape[0]), dim=1, largest=False)
        topk_dists.append(vals)
        topk_idx.append(idx)

    return torch.cat(topk_dists, dim=0), torch.cat(topk_idx, dim=0)


def _infer_bandwidth(
    topk_dists: torch.Tensor,
    bandwidth: Optional[float],
    bandwidth_scale: float,
    eps: float,
    min_bandwidth: Optional[float] = None
) -> float:
    """
    Choose a bandwidth if none provided. Uses median of k-th NN distances with
    an optional adaptive floor.
    """
    if bandwidth is not None:
        bw = float(bandwidth)
    else:
        # Use last column (k-th NN for each query); fall back to overall median.
        kth = topk_dists[:, -1]
        median = torch.median(kth).item()
        bw = median * bandwidth_scale

    bw = max(bw, eps)
    if min_bandwidth is not None and math.isfinite(min_bandwidth):
        bw = max(bw, float(min_bandwidth))
    return bw


def _pairwise_min_bandwidth(topk_dists: torch.Tensor, quantile: float, eps: float) -> float:
    """
    Compute a pair-specific minimum bandwidth from neighbor distances.
    Uses a quantile of the k-th NN distances to avoid collapsing on sparse sets.
    """
    try:
        q = torch.quantile(topk_dists[:, -1], quantile).item()
        if not math.isfinite(q):
            return eps
        return max(q, eps)
    except Exception:
        return eps


def _kernel_weights(
    dists: torch.Tensor,
    bandwidth: float,
    kernel: str = "gaussian",
    eps: float = 1e-12,
    normalize: bool = False,
) -> torch.Tensor:
    """Compute normalized kernel weights over neighbors."""
    if kernel == "gaussian":
        weights = torch.exp(-0.5 * (dists / bandwidth) ** 2)
    elif kernel == "inverse":
        weights = 1.0 / torch.clamp(dists, min=bandwidth)
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    weights = weights.clamp(min=eps)
    if normalize:
        weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=eps)
    return weights


def recall_train_covers_eval_simple(
    train_cb: DatasetCodebook,
    eval_cb: DatasetCodebook,
    k: int = 1,
    batch_size: int = 1024,
    percentile: float = 0.5
) -> float:
    """
    Simple recall metric: fraction of eval points that have nearby train neighbors.
    Uses mean distance to k nearest neighbors, normalized by dataset scale.
    More robust to dataset size and hyperparameters.
    
    Returns a score in [0, 1] where:
    - 1.0 = all eval points are very close to train points (perfect coverage)
    - 0.0 = all eval points are far from train points (poor coverage)
    """
    import torch
    
    train_centroids = train_cb.centroids.double()
    eval_centroids = eval_cb.centroids.double()
    eval_counts = eval_cb.counts.double()
    
    # Check if this is self-coverage (same codebook)
    is_self_coverage = (train_cb is eval_cb) or (
        train_centroids.shape == eval_centroids.shape and
        torch.allclose(train_centroids, eval_centroids, atol=1e-8)
    )
    
    # Compute distances from each eval point to k nearest train points
    # For self-coverage, we only need k=1 (the self-match)
    k_query = 1 if is_self_coverage else k
    topk_dists, _ = _batched_cdist_topk(
        queries=eval_centroids,
        centers=train_centroids,
        k=min(k_query, train_centroids.shape[0]),
        batch_size=batch_size
    )
    
    # For self-coverage, use self-matches (distance 0) directly
    # This gives perfect coverage since every point finds itself
    if is_self_coverage:
        # Use self-match (first column, distance 0) for perfect self-coverage
        mean_dists = topk_dists[:, 0]  # Should be all zeros
    else:
        # Mean distance per eval point (average over k neighbors)
        mean_dists = topk_dists.mean(dim=1)  # Shape: (n_eval,)
    
    # Normalize by the scale of the train dataset (median distance between train points)
    # This makes the metric scale-invariant
    train_pairwise_dists = _batched_cdist_topk(
        queries=train_centroids[:min(1000, len(train_centroids))],  # Sample for efficiency
        centers=train_centroids,
        k=2,  # k=2 to get nearest neighbor (excluding self)
        batch_size=batch_size
    )[0][:, 1]  # Take 2nd nearest (1st is self)
    
    train_scale = torch.quantile(train_pairwise_dists, percentile).item()
    train_scale = max(train_scale, 1e-6)  # Avoid division by zero
    
    # Convert distances to coverage scores using exponential decay
    # Closer points get higher scores
    coverage_scores = torch.exp(-mean_dists / train_scale)
    
    # Weight by eval point importance (counts)
    total_eval_count = eval_counts.sum()
    if total_eval_count > 0:
        weights = eval_counts / total_eval_count
        recall = float((weights * coverage_scores).sum().item())
    else:
        recall = 0.0
    
    return recall


def recall_train_covers_eval_soft(
    train_cb: DatasetCodebook,
    eval_cb: DatasetCodebook,
    k: int = 5,
    bandwidth: Optional[float] = None,
    bandwidth_scale: float = 1.0,
    M_train: float = 100.0,
    eps: float = 1e-12,
    kernel: str = "gaussian",
    batch_size: int = 1024,
    adaptive_bandwidth: bool = False,
    min_bandwidth_quantile: float = 0.3,
    adaptive_mass: bool = False,
    mass_quantile: float = 0.75,
    mass_floor: float = 1.0,
    use_simple: bool = False
) -> float:
    """
    Soft k-NN recall: fraction of eval mass supported by train mass.

    For each eval centroid, gather k nearest train centroids, apply a kernel
    over distances to get weights, compute local train mass = sum(weights * counts),
    then score = min(local_mass / M_train, 1). Aggregate with eval weights.
    
    If use_simple=True, uses a simpler distance-based metric instead.
    """
    if use_simple:
        return recall_train_covers_eval_simple(train_cb, eval_cb, k=k, batch_size=batch_size)
    
    e_centroids = train_cb.centroids.double()
    t_centroids = eval_cb.centroids.double()
    e_counts = train_cb.counts.double()
    t_counts = eval_cb.counts.double()

    N_T = torch.clamp(t_counts.sum(), min=eps)
    w_T = t_counts / N_T

    topk_dists, topk_idx = _batched_cdist_topk(
        queries=t_centroids,
        centers=e_centroids,
        k=k,
        batch_size=batch_size
    )
    min_bw = _pairwise_min_bandwidth(topk_dists, min_bandwidth_quantile, eps) if adaptive_bandwidth else None
    bw = _infer_bandwidth(topk_dists, bandwidth, bandwidth_scale, eps, min_bandwidth=min_bw)
    neigh_weights = _kernel_weights(topk_dists, bandwidth=bw, kernel=kernel, eps=eps, normalize=False)

    local_mass = (neigh_weights * e_counts[topk_idx]).sum(dim=1)
    if adaptive_mass:
        # Use a dataset-specific mass scale based on the count distribution.
        # Scale M_train proportionally to the dataset's typical count.
        mass_stat = torch.quantile(e_counts, mass_quantile).item()
        # Use the quantile as a base, scaled by k (number of neighbors) to account
        # for the fact that local_mass sums over k neighbors
        # This makes M_train_eff scale naturally with the dataset's count distribution
        M_train_eff = max(mass_stat * k, mass_floor)
    else:
        M_train_eff = M_train

    c_rec = torch.clamp(local_mass / M_train_eff, max=1.0)

    return float((w_T * c_rec).sum().item())


def precision_train_wrt_eval_simple(
    train_cb: DatasetCodebook,
    eval_cb: DatasetCodebook,
    k: int = 1,
    batch_size: int = 1024,
    percentile: float = 0.5
) -> float:
    """
    Simple precision metric: fraction of train points that have nearby eval neighbors.
    Uses mean distance to k nearest neighbors, normalized by dataset scale.
    More robust to dataset size and hyperparameters.
    
    Returns a score in [0, 1] where:
    - 1.0 = all train points are very close to eval points (train is well-supported by eval)
    - 0.0 = all train points are far from eval points (train has "outside" mass)
    """
    import torch
    
    train_centroids = train_cb.centroids.double()
    eval_centroids = eval_cb.centroids.double()
    train_counts = train_cb.counts.double()
    
    # Check if this is self-coverage (same codebook)
    is_self_coverage = (train_cb is eval_cb) or (
        train_centroids.shape == eval_centroids.shape and
        torch.allclose(train_centroids, eval_centroids, atol=1e-8)
    )
    
    # Compute distances from each train point to k nearest eval points
    # For self-coverage, we only need k=1 (the self-match)
    k_query = 1 if is_self_coverage else k
    topk_dists, _ = _batched_cdist_topk(
        queries=train_centroids,
        centers=eval_centroids,
        k=min(k_query, eval_centroids.shape[0]),
        batch_size=batch_size
    )
    
    # For self-coverage, use self-matches (distance 0) directly
    # This gives perfect coverage since every point finds itself
    if is_self_coverage:
        # Use self-match (first column, distance 0) for perfect self-coverage
        mean_dists = topk_dists[:, 0]  # Should be all zeros
    else:
        # Mean distance per train point (average over k neighbors)
        mean_dists = topk_dists.mean(dim=1)  # Shape: (n_train,)
    
    # Normalize by the scale of the eval dataset
    eval_pairwise_dists = _batched_cdist_topk(
        queries=eval_centroids[:min(1000, len(eval_centroids))],  # Sample for efficiency
        centers=eval_centroids,
        k=2,  # k=2 to get nearest neighbor (excluding self)
        batch_size=batch_size
    )[0][:, 1]  # Take 2nd nearest (1st is self)
    
    eval_scale = torch.quantile(eval_pairwise_dists, percentile).item()
    eval_scale = max(eval_scale, 1e-6)  # Avoid division by zero
    
    # Convert distances to precision scores using exponential decay
    precision_scores = torch.exp(-mean_dists / eval_scale)
    
    # Weight by train point importance (counts)
    total_train_count = train_counts.sum()
    if total_train_count > 0:
        weights = train_counts / total_train_count
        precision = float((weights * precision_scores).sum().item())
    else:
        precision = 0.0
    
    return precision


def precision_train_wrt_eval_soft(
    train_cb: DatasetCodebook,
    eval_cb: DatasetCodebook,
    k: int = 5,
    bandwidth: Optional[float] = None,
    bandwidth_scale: float = 1.0,
    M_eval: float = 20.0,
    eps: float = 1e-12,
    kernel: str = "gaussian",
    batch_size: int = 1024,
    adaptive_bandwidth: bool = False,
    min_bandwidth_quantile: float = 0.3,
    adaptive_mass: bool = False,
    mass_quantile: float = 0.75,
    mass_floor: float = 1.0,
    use_simple: bool = False
) -> float:
    """
    Soft k-NN precision: fraction of train mass supported by eval mass.

    Mirror of recall, swapping train/eval roles.
    
    If use_simple=True, uses a simpler distance-based metric instead.
    """
    if use_simple:
        return precision_train_wrt_eval_simple(train_cb, eval_cb, k=k, batch_size=batch_size)
    e_centroids = train_cb.centroids.double()
    t_centroids = eval_cb.centroids.double()
    e_counts = train_cb.counts.double()
    t_counts = eval_cb.counts.double()

    N_E = torch.clamp(e_counts.sum(), min=eps)
    w_E = e_counts / N_E

    topk_dists, topk_idx = _batched_cdist_topk(
        queries=e_centroids,
        centers=t_centroids,
        k=k,
        batch_size=batch_size
    )
    min_bw = _pairwise_min_bandwidth(topk_dists, min_bandwidth_quantile, eps) if adaptive_bandwidth else None
    bw = _infer_bandwidth(topk_dists, bandwidth, bandwidth_scale, eps, min_bandwidth=min_bw)
    neigh_weights = _kernel_weights(topk_dists, bandwidth=bw, kernel=kernel, eps=eps, normalize=False)

    local_mass = (neigh_weights * t_counts[topk_idx]).sum(dim=1)
    if adaptive_mass:
        # Use a dataset-specific mass scale based on the count distribution.
        # Scale M_eval proportionally to the dataset's typical count.
        mass_stat = torch.quantile(t_counts, mass_quantile).item()
        # Use the quantile as a base, scaled by k (number of neighbors) to account
        # for the fact that local_mass sums over k neighbors
        # This makes M_eval_eff scale naturally with the dataset's count distribution
        M_eval_eff = max(mass_stat * k, mass_floor)
    else:
        M_eval_eff = M_eval

    c_prec = torch.clamp(local_mass / M_eval_eff, max=1.0)

    return float((w_E * c_prec).sum().item())


def outside_mass_fraction_soft(
    train_cb: DatasetCodebook,
    eval_cb: DatasetCodebook,
    **kwargs
) -> float:
    """Convenience: 1 - precision for soft k-NN metric."""
    P = precision_train_wrt_eval_soft(train_cb, eval_cb, **kwargs)
    return 1.0 - P
