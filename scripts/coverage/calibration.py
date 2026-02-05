"""
Alpha calibration for flow scaling (Step 1 of pipeline).

Computes global alpha so [x,y] and [dx,dy] contribute comparably in distance.
Uses per-dataset 2D kNN self-radius ratio, then aggregates with geometric mean.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from . import faiss_ops
from . import cache
from . import spaces


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


def compute_per_dataset_alpha(
    vectors: np.ndarray,
    k: int = 5,
    use_gpu: bool = True,
    verbose: bool = True,
    max_vectors_for_alpha: int = 1_000_000,
    dedup: bool = False,
) -> float:
    """
    Compute per-dataset alpha from 2D kNN self-radius ratio.
    
    Alpha = median(d_k_xy) / (median(d_k_flow) + epsilon)
    
    Args:
        vectors: (N, 4) normalized flow vectors [x, y, dx, dy]
        k: Number of neighbors for kNN
        use_gpu: Use GPU acceleration
        verbose: Print progress
        max_vectors_for_alpha: Maximum vectors to use (subsample if needed to avoid XY collisions)
        
    Returns:
        Alpha value for this dataset
    """
    if vectors.shape[1] != 4:
        raise ValueError(f"Expected shape (N, 4), got {vectors.shape}")
    
    # Optional: Subsample if we have too many vectors
    # Disabled by default - we now search for 128 neighbors to skip duplicates instead
    if max_vectors_for_alpha is not None and max_vectors_for_alpha > 0 and len(vectors) > max_vectors_for_alpha:
        if verbose:
            print(f"  Subsampling {len(vectors):,} → {max_vectors_for_alpha:,} vectors for alpha")
        indices = np.random.choice(len(vectors), size=max_vectors_for_alpha, replace=False)
        vectors = vectors[indices]
    
    if verbose:
        print(f"  Computing per-dataset alpha: {vectors.shape[0]:,} vectors, k={k}")

    if dedup:
        vectors = _dedup_vectors_exact(vectors, verbose=verbose)
    
    # Extract xy and flow spaces
    xy_vectors = spaces.to_xy_space(vectors)
    flow_vectors = spaces.to_flow_space(vectors)

    if dedup:
        if verbose:
            print("  Deduplicating XY/flow spaces for alpha...")
        xy_vectors = _dedup_vectors_exact(xy_vectors, verbose=verbose)
        flow_vectors = _dedup_vectors_exact(flow_vectors, verbose=verbose)
    
    # Compute kNN distances in both spaces
    if verbose:
        print(f"    Computing kNN distances in XY space...")
    
    # For alpha calibration, use IVF for speed (approximate is fine here)
    # IVF is MUCH faster than Flat for large datasets (10-100x speedup)
    n_vectors = len(xy_vectors)
    if n_vectors > 100_000:
        # Use IVF for large datasets
        nlist = min(4096, max(256, int(np.sqrt(n_vectors))))
        index_factory = f"IVF{nlist},Flat"
        nprobe = min(64, nlist // 4)
        if verbose:
            print(f"    Using IVF index (nlist={nlist}, nprobe={nprobe}) for speed")
    else:
        index_factory = "Flat"
        nprobe = None
    
    xy_radius_data = faiss_ops.compute_self_radius(
        xy_vectors,
        k=k,
        radius_quantile=0.50,  # Use median for alpha (not p95)
        neighbor_agg="kth",
        filter_duplicates=False,
        use_gpu=use_gpu,
        index_factory=index_factory,
        nprobe=nprobe,
        batch_size=100000,  # Smaller batch for alpha (searching k+32 neighbors)
        verbose=verbose,
    )
    
    if verbose:
        print(f"    Computing kNN distances in Flow space...")
    
    flow_radius_data = faiss_ops.compute_self_radius(
        flow_vectors,
        k=k,
        radius_quantile=0.50,  # Use median for alpha
        neighbor_agg="kth",
        filter_duplicates=False,
        use_gpu=use_gpu,
        index_factory=index_factory,
        nprobe=nprobe,
        batch_size=100000,  # Smaller batch for alpha (searching k+32 neighbors)
        verbose=verbose,
    )
    
    # Extract median distances (quantile=0.50 → stored in 'median' field)
    s_xy = xy_radius_data['median']
    s_flow = flow_radius_data['median']
    
    # Check for invalid values
    if not np.isfinite(s_xy) or not np.isfinite(s_flow):
        if verbose:
            print(f"    ⚠️  WARNING: Invalid median distances (s_xy={s_xy}, s_flow={s_flow})")
            print(f"    Returning alpha=nan")
        return float('nan')
    
    if s_xy <= 0:
        if verbose:
            print(f"    ⚠️  WARNING: XY median distance is {s_xy} (all points at same location?)")
            print(f"    Returning alpha=0")
        return 0.0
    
    if s_flow <= 0:
        if verbose:
            print(f"    ⚠️  WARNING: Flow median distance is {s_flow} (no motion in dataset)")
            print(f"    This dataset has no flow - cannot compute meaningful alpha")
            print(f"    Returning alpha=nan")
        return float('nan')
    
    # Compute alpha
    epsilon = 1e-12
    alpha = s_xy / (s_flow + epsilon)
    
    # Sanity check result
    if not np.isfinite(alpha) or alpha <= 0:
        if verbose:
            print(f"    ⚠️  WARNING: Computed invalid alpha={alpha}")
            print(f"    (s_xy={s_xy}, s_flow={s_flow})")
        return float('nan')
    
    if verbose:
        print(f"    Median XY distance: {s_xy:.6f}")
        print(f"    Median Flow distance: {s_flow:.6f}")
        print(f"    Alpha for this dataset: {alpha:.6f}")
    
    return float(alpha)


def compute_global_alpha(
    dataset_vectors: Dict[str, np.ndarray],
    k: int = 5,
    aggregation: str = "geometric_mean",
    use_gpu: bool = True,
    verbose: bool = True,
    max_vectors_for_alpha: int = 1_000_000,
    dedup: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute global alpha from multiple training datasets.
    
    Args:
        dataset_vectors: Dict mapping dataset names to (N, 4) normalized flow vectors
        k: Number of neighbors for kNN
        aggregation: "geometric_mean" or "median"
        use_gpu: Use GPU acceleration
        verbose: Print progress
        
    Returns:
        (global_alpha, per_dataset_alphas)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"ALPHA CALIBRATION")
        print(f"{'='*60}")
        print(f"Datasets: {len(dataset_vectors)}")
        print(f"Aggregation: {aggregation}")
        print(f"k: {k}")
        print(f"Using all available vectors per dataset")
    
    per_dataset_alphas = {}
    
    for dataset_name, vectors in dataset_vectors.items():
        if verbose:
            print(f"\n[{dataset_name}]")
        
        alpha_i = compute_per_dataset_alpha(
            vectors, 
            k=k, 
            use_gpu=use_gpu, 
            verbose=verbose,
            max_vectors_for_alpha=max_vectors_for_alpha,
            dedup=dedup,
        )
        per_dataset_alphas[dataset_name] = alpha_i
    
    # Aggregate across datasets
    if len(per_dataset_alphas) == 0:
        raise ValueError("No datasets provided for alpha calibration!")
    
    alpha_values = np.array(list(per_dataset_alphas.values()))
    
    # Filter out invalid alphas (0, negative, nan, inf)
    valid_mask = np.isfinite(alpha_values) & (alpha_values > 0)
    if not np.any(valid_mask):
        raise ValueError(
            f"All per-dataset alphas are invalid! Values: {alpha_values}. "
            "This usually means datasets have no motion (all flow vectors are zero)."
        )
    
    if np.sum(~valid_mask) > 0:
        invalid_names = [name for name, alpha in per_dataset_alphas.items() 
                        if not (np.isfinite(alpha) and alpha > 0)]
        if verbose:
            print(f"\n⚠️  WARNING: {np.sum(~valid_mask)} datasets have invalid alphas, skipping:")
            for name in invalid_names:
                print(f"    {name}: alpha={per_dataset_alphas[name]}")
        alpha_values = alpha_values[valid_mask]
    
    if aggregation == "geometric_mean":
        # Geometric mean: exp(mean(log(alpha_i)))
        # Only works with positive values (already filtered above)
        global_alpha = float(np.exp(np.mean(np.log(alpha_values))))
        if verbose:
            print(f"\n{'='*60}")
            print(f"Global alpha (geometric mean): {global_alpha:.6f}")
    elif aggregation == "median":
        global_alpha = float(np.median(alpha_values))
        if verbose:
            print(f"\n{'='*60}")
            print(f"Global alpha (median): {global_alpha:.6f}")
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    # Sanity check
    if not np.isfinite(global_alpha) or global_alpha <= 0:
        raise ValueError(
            f"Computed global alpha is invalid: {global_alpha}. "
            f"Per-dataset alphas: {per_dataset_alphas}"
        )
    
    if verbose:
        print(f"Per-dataset alphas:")
        for name, alpha_i in sorted(per_dataset_alphas.items()):
            # Convert tuple name to string if needed
            name_str = str(name) if isinstance(name, tuple) else name
            if np.isfinite(alpha_i) and alpha_i > 0:
                ratio = alpha_i / global_alpha
                print(f"  {name_str:40s}: {alpha_i:.6f} (ratio to global: {ratio:.3f})")
            else:
                print(f"  {name_str:40s}: {alpha_i:.6f} (INVALID - skipped)")
        print(f"{'='*60}\n")
    
    return global_alpha, per_dataset_alphas


def load_or_compute_alpha(
    cache_dir: Path,
    dataset_vectors: Dict[str, np.ndarray],
    k: int = 5,
    aggregation: str = "geometric_mean",
    use_gpu: bool = True,
    force_recompute: bool = False,
    verbose: bool = True,
    dedup: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """
    Load cached alpha or compute if not found.
    
    Args:
        cache_dir: Cache directory
        dataset_vectors: Dict mapping dataset names to vectors
        k: Number of neighbors
        aggregation: Aggregation method
        use_gpu: Use GPU
        force_recompute: Force recomputation even if cached
        verbose: Print progress
        
    Returns:
        (global_alpha, per_dataset_alphas)
    """
    # Try to load from cache
    if not force_recompute:
        cached = cache.load_alpha(cache_dir, representation="flow", dedup=dedup)
        if cached is not None:
            global_alpha, per_dataset_alphas = cached
            
            # Verify cached alpha is for the same datasets
            cached_datasets = set(per_dataset_alphas.keys())
            current_datasets = set(dataset_vectors.keys())
            
            if cached_datasets == current_datasets:
                if verbose:
                    print(f"✓ Using cached global alpha: {global_alpha:.6f}")
                return global_alpha, per_dataset_alphas
            else:
                if verbose:
                    print(f"⚠️  Cached alpha has different datasets, recomputing")
                    print(f"    Cached: {cached_datasets}")
                    print(f"    Current: {current_datasets}")
    
    # Compute alpha
    global_alpha, per_dataset_alphas = compute_global_alpha(
        dataset_vectors,
        k=k,
        aggregation=aggregation,
        use_gpu=use_gpu,
        verbose=verbose,
        max_vectors_for_alpha=None,  # No subsampling - search for 128 neighbors instead
        dedup=dedup,
    )
    
    # Save to cache
    cache.save_alpha(
        cache_dir,
        global_alpha,
        per_dataset_alphas,
        representation="flow",
        dedup=dedup,
        extra_metadata={
            'k': np.array(k),
            'aggregation': aggregation,
            'dedup': bool(dedup),
        },
    )
    
    return global_alpha, per_dataset_alphas
