"""
Per-dataset self-radius computation (Step 3 of pipeline).

Computes robust self-radius R_D^S for each dataset D in each space S,
using k-NN distances within the dataset.
"""

from pathlib import Path
from typing import Dict, Optional
import numpy as np

from . import faiss_ops
from . import cache


def _dedup_vectors_exact(vectors: np.ndarray, verbose: bool = True) -> np.ndarray:
    """Exact row-wise deduplication for float vectors."""
    if vectors.size == 0:
        return vectors
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D vectors array, got shape {vectors.shape}")
    n, d = vectors.shape
    if verbose:
        print(f"  Deduplicating vectors (exact): {n:,}x{d}")
    contig = np.ascontiguousarray(vectors)
    view = contig.view(np.dtype((np.void, contig.dtype.itemsize * d)))
    _, idx = np.unique(view, return_index=True)
    idx = np.sort(idx)
    unique = contig[idx]
    if verbose:
        print(f"  Deduped: {n:,} -> {len(unique):,} ({(len(unique)/n)*100:.2f}%)")
    return unique


def compute_per_dataset_radii(
    vectors: np.ndarray,
    k: int = 5,
    quantile: float = 0.95,
    neighbor_agg: str = "kth",
    filter_duplicates: bool = True,
    dedup: bool = False,
    batch_size: Optional[int] = None,
    use_gpu: bool = True,
    index_factory: str = "Flat",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Compute self-radius for a single dataset.
    
    Args:
        vectors: (N, D) vectors in the target space
        k: Number of neighbors for kNN
        quantile: Quantile for radius (e.g., 0.95 for p95)
        neighbor_agg: 'kth' for k-th neighbor, 'mean' for mean of k neighbors
        use_gpu: Use GPU acceleration
        index_factory: FAISS index type
        verbose: Print progress
        
    Returns:
        Dictionary with 'radius', 'median', 'p90', 'p95', 'mean'
    """
    if verbose:
        print(f"  Computing self-radius: {vectors.shape[0]:,} vectors, k={k}, q={quantile:.2f}")
    
    if dedup:
        vectors = _dedup_vectors_exact(vectors, verbose=verbose)

    radius_data = faiss_ops.compute_self_radius(
        vectors,
        k=k,
        radius_quantile=quantile,
        neighbor_agg=neighbor_agg,
        filter_duplicates=filter_duplicates,
        batch_size=batch_size,
        use_gpu=use_gpu,
        index_factory=index_factory,
        verbose=verbose,
    )
    
    if verbose:
        print(f"    Radius (p{int(quantile*100)}): {radius_data['radius']:.6f}")
        print(f"    Median: {radius_data['median']:.6f}")
        print(f"    Mean: {radius_data['mean']:.6f}")
    
    return radius_data


def load_or_compute_radius(
    cache_dir: Path,
    dataset_name: str,
    split: str,
    space: str,
    vectors: np.ndarray,
    k: int = 5,
    quantile: float = 0.95,
    neighbor_agg: str = "kth",
    filter_duplicates: bool = True,
    dedup: bool = False,
    batch_size: Optional[int] = None,
    alpha: Optional[float] = None,
    normalization: str = "norm2x1",
    distance_metric: str = "sqL2",
    use_gpu: bool = True,
    index_factory: str = "Flat",
    force_recompute: bool = False,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Load cached radius or compute if not found.
    
    Args:
        cache_dir: Cache directory
        dataset_name: Dataset name
        split: Split name
        space: Space name (xy, flow, joint, features)
        vectors: (N, D) vectors in the target space
        k: Number of neighbors
        quantile: Quantile for radius
        neighbor_agg: Neighbor aggregation method
        alpha: Alpha value (for joint space cache key)
        normalization: Normalization scheme
        distance_metric: Distance metric name
        use_gpu: Use GPU
        index_factory: FAISS index type
        force_recompute: Force recomputation even if cached
        verbose: Print progress
        
    Returns:
        Dictionary with radius statistics
    """
    def _radius_is_valid(data: Dict[str, float]) -> bool:
        # Reject non-finite values and obviously corrupted magnitudes.
        for key in ("radius", "median", "p90", "p95", "mean"):
            val = float(data.get(key, float("nan")))
            if not np.isfinite(val):
                return False
            if val > 1e6:
                return False
        return True

    # Try to load from cache
    if not force_recompute:
        cached = cache.load_radius(
            cache_dir,
            dataset_name,
            split,
            space,
            k=k,
            quantile=quantile,
            dedup=dedup,
            alpha=alpha,
            normalization=normalization,
            distance_metric=distance_metric,
        )
        
        if cached is not None and _radius_is_valid(cached):
            if verbose:
                print(f"  ✓ Using cached radius: {cached['radius']:.6f}")
            return cached
        if cached is not None and verbose:
            print("  ⚠️  Cached radius looks invalid (non-finite or huge); recomputing.")
    
    # Compute radius
    if verbose:
        print(f"  Computing radius for {dataset_name}/{split}/{space}")
    
    radius_data = compute_per_dataset_radii(
        vectors,
        k=k,
        quantile=quantile,
        neighbor_agg=neighbor_agg,
        filter_duplicates=filter_duplicates,
        dedup=dedup,
        batch_size=batch_size,
        use_gpu=use_gpu,
        index_factory=index_factory,
        verbose=verbose,
    )
    
    # Save to cache
    cache.save_radius(
        cache_dir,
        dataset_name,
        split,
        space,
        radius_data,
        k=k,
        quantile=quantile,
        dedup=dedup,
        alpha=alpha,
        normalization=normalization,
        distance_metric=distance_metric,
    )
    
    return radius_data


def compute_all_radii(
    cache_dir: Path,
    dataset_vectors: Dict[str, np.ndarray],
    space: str,
    k: int = 5,
    quantile: float = 0.95,
    neighbor_agg: str = "kth",
    filter_duplicates: bool = True,
    dedup: bool = False,
    batch_size: Optional[int] = None,
    alpha: Optional[float] = None,
    normalization: str = "norm2x1",
    distance_metric: str = "sqL2",
    use_gpu: bool = True,
    index_factory: str = "Flat",
    force_recompute: bool = False,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Compute radii for all datasets in a given space.
    
    Args:
        cache_dir: Cache directory
        dataset_vectors: Dict mapping (dataset_name, split) tuples to vectors
        space: Space name
        k: Number of neighbors
        quantile: Quantile for radius
        neighbor_agg: Neighbor aggregation
        alpha: Alpha value (for joint space)
        normalization: Normalization scheme
        distance_metric: Distance metric
        use_gpu: Use GPU
        index_factory: FAISS index type
        force_recompute: Force recomputation
        verbose: Print progress
        
    Returns:
        Dict mapping (dataset_name, split) to radius data
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"SELF-RADIUS COMPUTATION - {space.upper()} SPACE")
        print(f"{'='*60}")
        print(f"Datasets: {len(dataset_vectors)}")
        print(f"k: {k}, quantile: {quantile}, neighbor_agg: {neighbor_agg}")
        if alpha is not None:
            print(f"Alpha (for joint space): {alpha:.6f}")
    
    all_radii = {}
    
    for (dataset_name, split), vectors in dataset_vectors.items():
        if verbose:
            print(f"\n[{dataset_name}/{split}]")
        
        radius_data = load_or_compute_radius(
            cache_dir,
            dataset_name,
            split,
            space,
            vectors,
            k=k,
            quantile=quantile,
            neighbor_agg=neighbor_agg,
            filter_duplicates=filter_duplicates,
            dedup=dedup,
            batch_size=batch_size,
            alpha=alpha,
            normalization=normalization,
            distance_metric=distance_metric,
            use_gpu=use_gpu,
            index_factory=index_factory,
            force_recompute=force_recompute,
            verbose=verbose,
        )
        
        all_radii[(dataset_name, split)] = radius_data
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"RADIUS SUMMARY - {space.upper()}")
        print(f"{'='*60}")
        for (dataset_name, split), radius_data in sorted(all_radii.items()):
            r = radius_data['radius']
            med = radius_data['median']
            print(f"  {dataset_name:30s} {split:8s}: R={r:.6f}, median={med:.6f}")
        print(f"{'='*60}\n")
    
    return all_radii
