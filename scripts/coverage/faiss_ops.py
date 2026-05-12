"""
Unified FAISS operations for coverage metrics.

All representations (flow, dino, resnet) use L2 metric:
- Flow: L2 on raw/α-scaled coordinates
- Dino/ResNet: L2 on unit-normalized vectors (equivalent to cosine)

This module provides reusable building blocks:
- build_index: Create FAISS index with GPU support
- compute_knn_distances: Generic kNN distance computation
- compute_self_radius: Self-radius using kNN
- compute_directed_distances: Cross-dataset NN distances
"""

from typing import Dict, Optional, Tuple
import os
import gc
import numpy as np

try:
    import faiss
except ImportError as exc:
    raise SystemExit(
        "faiss is required. Install faiss-cpu or faiss-gpu."
    ) from exc

_GPU_RESOURCES = None


def _get_gpu_resources(temp_memory_bytes: int = 512 * 1024 * 1024) -> "faiss.StandardGpuResources":
    """Reuse a single GPU resource pool to avoid per-index GPU memory growth.
    512 MB scratch is sufficient for batched IVF train/add/search; the old 12 GB
    default left no room for two large indices to coexist on a 24 GB card."""
    global _GPU_RESOURCES
    if _GPU_RESOURCES is None:
        import os
        env_gb = os.getenv("FAISS_GPU_TEMP_GB")
        if env_gb:
            try:
                temp_memory_bytes = int(float(env_gb) * 1024 * 1024 * 1024)
            except ValueError:
                pass
        _GPU_RESOURCES = faiss.StandardGpuResources()
        _GPU_RESOURCES.setTempMemory(temp_memory_bytes)
    return _GPU_RESOURCES


def release_index(index: Optional["faiss.Index"]) -> None:
    """Best-effort cleanup for FAISS indices to free GPU memory promptly."""
    if index is None:
        return
    try:
        # Force deletion of the index object.
        del index
    finally:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _has_invalid_distances(distances: np.ndarray) -> bool:
    if distances.size == 0:
        return False
    return (~np.isfinite(distances)).any() or (distances >= 1e30).any()


def _has_excessive_sentinel(
    distances: np.ndarray,
    sentinel: float = 1e9,
    frac_threshold: float = 1e-3,
) -> bool:
    if distances.size == 0:
        return False
    frac = float(np.mean(distances >= sentinel))
    return frac > frac_threshold


def _invalid_fraction(distances: np.ndarray) -> float:
    """Fraction of distances that are invalid (non-finite or sentinel-max)."""
    if distances.size == 0:
        return 0.0
    invalid = (~np.isfinite(distances)) | (distances >= 1e30)
    return float(np.mean(invalid))


def _is_gpu_index(index: faiss.Index) -> bool:
    """Check if an index is a GPU index."""
    index_type = type(index).__name__
    return 'Gpu' in index_type or 'GPU' in index_type


def _auto_batch_size(dim: int) -> int:
    """Automatically determine batch size based on dimensionality."""
    if dim <= 4:
        return 500000  # Flow vectors (4D)
    elif dim <= 64:
        return 100000  # Medium dimensions
    elif dim <= 256:
        return 50000   # High dimensions (ResNet/DINO after PCA)
    else:
        return 20000   # Very high dimensions


def build_index(
    vectors: np.ndarray,
    use_gpu: bool = True,
    index_factory: str = "Flat",
    nprobe: Optional[int] = None,
    verbose: bool = True,
) -> faiss.Index:
    """
    Build FAISS index (always L2 metric).
    
    Args:
        vectors: (N, D) array of vectors
        use_gpu: Whether to use GPU acceleration
        index_factory: FAISS index type ("Flat", "IVF1024,Flat", "HNSW32", etc.)
        nprobe: Number of clusters to probe for IVF indices
        verbose: Print progress messages
        
    Returns:
        FAISS index
    """
    if vectors.size == 0 or vectors.shape[0] == 0:
        raise ValueError("Cannot build index from empty vectors")
        
    n_vectors, dim = vectors.shape
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    
    if verbose:
        print(f"  Building index: {n_vectors:,} vectors, {dim} dims, type={index_factory}, gpu={use_gpu}")
    
    # Convert HNSW to IVF for GPU (HNSW not GPU-compatible)
    if use_gpu and "hnsw" in index_factory.lower():
        if n_vectors < 10000:
            index_factory = "Flat"
            if verbose:
                print(f"    GPU mode: Too few vectors for IVF, using Flat")
        else:
            nlist = min(2048, max(256, n_vectors // 100))
            index_factory = f"IVF{nlist},Flat"
            if verbose:
                print(f"    GPU mode: Converting HNSW to IVF{nlist},Flat")
    
    # Check IVF has enough vectors
    if "ivf" in index_factory.lower():
        import re
        match = re.search(r'ivf(\d+)', index_factory.lower())
        if match:
            nlist = int(match.group(1))
            min_needed = nlist * 40
            if n_vectors < min_needed:
                if verbose:
                    print(f"    WARNING: Only {n_vectors:,} vectors, too few for IVF{nlist} (needs {min_needed:,})")
                    print(f"    Falling back to Flat")
                index_factory = "Flat"
    
    # Build index (always L2 metric)
    if index_factory.lower() == "flat":
        index = faiss.IndexFlatL2(dim)
    else:
        index = faiss.index_factory(dim, index_factory, faiss.METRIC_L2)

    # Transfer to GPU before training so k-means and add happen on GPU.
    # (Transferring after training means all the slow work already ran on CPU.)
    if use_gpu:
        try:
            gpu_resources = _get_gpu_resources()
            index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
            if verbose:
                print(f"    Transferred to GPU (pre-train)")
        except Exception as e:
            if verbose:
                print(f"    WARNING: GPU transfer failed ({e}), using CPU")

    # Train if needed (runs on GPU if transfer above succeeded)
    if not index.is_trained:
        if verbose:
            print(f"    Training index...")
        index.train(vectors)
        if verbose:
            print(f"    Training complete")

    # Add vectors (runs on GPU)
    if verbose:
        print(f"    Adding {n_vectors:,} vectors...")
    index.add(vectors)

    # IVF diagnostics — need CPU index for invlists inspection
    if verbose and "ivf" in index_factory.lower():
        try:
            cpu_idx = faiss.index_gpu_to_cpu(index) if use_gpu else index
            nlist = cpu_idx.nlist
            sizes = np.fromiter((cpu_idx.invlists.list_size(i) for i in range(nlist)), dtype=np.int64)
            if sizes.size:
                print(
                    f"    IVF list sizes: min={int(sizes.min())}, "
                    f"median={int(np.median(sizes))}, max={int(sizes.max())}, "
                    f"p90={int(np.quantile(sizes, 0.90))}"
                )
            del cpu_idx
        except Exception:
            pass

    # Set nprobe
    if nprobe is not None and hasattr(index, "nprobe"):
        index.nprobe = nprobe
    
    if verbose:
        print(f"    Index ready: {index.ntotal:,} vectors")
    
    return index


def compute_knn_distances(
    index: faiss.Index,
    query_vectors: np.ndarray,
    k: int,
    exclude_self: bool = False,
    filter_duplicates: bool = True,
    fallback_index: Optional[faiss.Index] = None,
    batch_size: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute k-nearest neighbor distances (L2 metric).
    
    Args:
        index: FAISS index to search
        query_vectors: (N, D) query vectors
        k: Number of neighbors
        exclude_self: If True, exclude the first neighbor (assumed to be self with dist~0)
        batch_size: Batch size for GPU searches (auto if None)
        verbose: Print progress
        
    Returns:
        distances: (N, K) L2 distances to k nearest neighbors
        indices: (N, K) indices of k nearest neighbors
    """
    if query_vectors.size == 0:
        return np.array([], dtype=np.float32).reshape(0, k), np.array([], dtype=np.int64).reshape(0, k)
    
    n_query = query_vectors.shape[0]
    query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)
    
    # Adjust k if excluding self/duplicates
    # Search for extra neighbors to account for duplicates (optional)
    if exclude_self:
        if filter_duplicates:
            # Use a moderate buffer (k + 32) to balance memory usage and duplicate handling
            search_k = min(k + 32, index.ntotal)
        else:
            # Only need one extra neighbor to drop self
            search_k = min(k + 1, index.ntotal)
    else:
        search_k = min(k, index.ntotal)
    
    if search_k < 1:
        raise ValueError(f"Not enough vectors in index for k={k} (index has {index.ntotal})")
    
    # Auto batch size
    if batch_size is None:
        batch_size = _auto_batch_size(query_vectors.shape[1])
    
    # For large datasets with exclude_self, reduce batch size to avoid OOM
    # (searching k+32 neighbors uses more memory, especially with sorting)
    if exclude_self and n_query > 5_000_000:
        batch_size = min(batch_size, 100_000)  # Very conservative batch size for huge datasets with duplicate filtering
    
    # Batched search for GPU
    is_gpu = _is_gpu_index(index)
    if is_gpu and n_query > batch_size:
        if verbose:
            print(f"    Batched GPU search: {n_query:,} queries, batch_size={batch_size:,}, k={search_k}")
        
        all_dists = []
        all_indices = []
        
        # Use tqdm for progress bar
        from tqdm import tqdm
        batch_iter = range(0, n_query, batch_size)
        if verbose:
            batch_iter = tqdm(batch_iter, desc="      Batches", unit="batch", 
                            total=(n_query + batch_size - 1) // batch_size)
        
        for i in batch_iter:
            batch = query_vectors[i:i+batch_size]
            if verbose and (not np.isfinite(batch).all()):
                bad = np.count_nonzero(~np.isfinite(batch))
                print(f"    ⚠️  Non-finite query vectors in batch [{i}:{i+len(batch)}]: {bad} values")
            # Optional query list diagnostics for IVF
            if verbose and os.getenv("FAISS_LOG_QUERY_LISTS") and i == 0:
                try:
                    sample = batch[: min(len(batch), 10000)]
                    cpu_index = faiss.index_gpu_to_cpu(index) if is_gpu else index
                    if hasattr(cpu_index, "quantizer"):
                        _, list_ids = cpu_index.quantizer.search(sample, 1)
                        list_ids = list_ids.reshape(-1)
                        counts = np.bincount(list_ids, minlength=cpu_index.nlist)
                        top5 = np.argsort(counts)[-5:][::-1]
                        top1_frac = float(counts[top5[0]]) / float(len(list_ids))
                        top5_frac = float(counts[top5].sum()) / float(len(list_ids))
                        print(
                            f"    IVF query list usage (sample={len(list_ids)}): "
                            f"top1={top1_frac:.2%}, top5={top5_frac:.2%}"
                        )
                except Exception:
                    pass
            batch_dists, batch_indices = index.search(batch, search_k)
            if verbose and (not np.isfinite(batch_dists).all()):
                bad = np.count_nonzero(~np.isfinite(batch_dists))
                print(f"    ⚠️  Non-finite distances in batch [{i}:{i+len(batch)}]: {bad} values")
                print(f"       dist min/max: {np.nanmin(batch_dists):.6g} / {np.nanmax(batch_dists):.6g}")
            all_dists.append(batch_dists)
            all_indices.append(batch_indices)
        
        raw_dists = np.vstack(all_dists)
        raw_indices = np.vstack(all_indices)
    else:
        raw_dists, raw_indices = index.search(query_vectors, search_k)

    # Treat extreme values or invalid indices as invalid distances.
    invalid_mask = (~np.isfinite(raw_dists)) | (raw_dists >= 1e30) | (raw_indices < 0)
    if verbose:
        n_invalid_idx = int(np.count_nonzero(raw_indices < 0))
        n_invalid_dist = int(np.count_nonzero((~np.isfinite(raw_dists)) | (raw_dists >= 1e30)))
        if n_invalid_idx or n_invalid_dist:
            print(
                f"    ⚠️  Invalids breakdown: invalid_index={n_invalid_idx}, "
                f"invalid_distance={n_invalid_dist}, total={raw_dists.size}"
            )
    if verbose and invalid_mask.any():
        bad = np.count_nonzero(invalid_mask)
        print(f"    ⚠️  Invalid distances detected: {bad} values out of {raw_dists.size}")
        print(f"       dist min/max: {np.nanmin(raw_dists):.6g} / {np.nanmax(raw_dists):.6g}")
    if invalid_mask.any():
        raw_dists = raw_dists.copy()
        raw_dists[invalid_mask] = np.inf
        if fallback_index is not None:
            bad_rows = np.any(invalid_mask, axis=1)
            n_bad = int(np.count_nonzero(bad_rows))
            if verbose:
                print(f"    ⚠️  Retrying {n_bad:,} invalid queries with Flat fallback...")
            if n_bad > 0:
                fallback_queries = query_vectors[bad_rows]
                fb_dists, fb_indices = fallback_index.search(fallback_queries, search_k)
                raw_dists[bad_rows] = fb_dists
                raw_indices[bad_rows] = fb_indices
    
    # Keep squared L2 distances (do NOT take sqrt for consistency)
    # FAISS returns squared L2 by default - this is what we want throughout
    distances = raw_dists
    
    # Handle exclude_self (robust to exact duplicates if requested)
    if exclude_self:
        if filter_duplicates:
            # Exclude ALL neighbors with distance ~ 0 (self + exact duplicates)
            # This is critical when pooling vectors from multiple images
            tiny = 1e-12
            
            # Don't modify FAISS arrays in-place - work with a view instead
            # Create a copy for sorting to avoid corrupting GPU memory
            distances_copy = distances.copy()
            
            # Mark duplicates and invalids to inf so they sort to end
            is_duplicate = distances_copy <= tiny
            distances_copy[is_duplicate] = np.inf
            distances_copy[~np.isfinite(distances_copy)] = np.inf
            
            # Sort by distance to get non-duplicates first
            sort_idx = np.argsort(distances_copy, axis=1)
            
            # Take first k (non-duplicates will be first after filtering)
            row_idx = np.arange(n_query)[:, None]
            result_indices = raw_indices[row_idx, sort_idx[:, :k]]
            result_dists = distances_copy[row_idx, sort_idx[:, :k]]

            # If duplicate filtering yields non-finite distances, retry those rows with fallback (Flat) index.
            if fallback_index is not None and (not np.isfinite(result_dists).all()):
                bad_rows = ~np.isfinite(result_dists).all(axis=1)
                n_bad = int(np.count_nonzero(bad_rows))
                if verbose:
                    print(f"    ⚠️  Non-finite distances after duplicate filtering: {n_bad} rows")
                if n_bad > 0:
                    fb_queries = query_vectors[bad_rows]
                    fb_dists, fb_indices = fallback_index.search(fb_queries, search_k)
                    fb_copy = fb_dists.copy()
                    fb_copy[fb_copy <= tiny] = np.inf
                    fb_copy[~np.isfinite(fb_copy)] = np.inf
                    fb_sort = np.argsort(fb_copy, axis=1)
                    fb_row_idx = np.arange(fb_queries.shape[0])[:, None]
                    fb_result_indices = fb_indices[fb_row_idx, fb_sort[:, :k]]
                    fb_result_dists = fb_copy[fb_row_idx, fb_sort[:, :k]]
                    result_indices[bad_rows] = fb_result_indices
                    result_dists[bad_rows] = fb_result_dists

            if verbose and (not np.isfinite(result_dists).all()):
                bad = np.count_nonzero(~np.isfinite(result_dists))
                print(f"    ⚠️  Non-finite distances after duplicate filtering: {bad} values")
                print(f"       dist min/max: {np.nanmin(result_dists):.6g} / {np.nanmax(result_dists):.6g}")
            
            return result_dists, result_indices
        else:
            # Only drop the closest neighbor (assumed self); keep exact duplicates.
            if raw_dists.shape[1] < k + 1:
                raise ValueError(f"Not enough neighbors to exclude self for k={k}")
            result_dists = raw_dists[:, 1:k+1]
            result_indices = raw_indices[:, 1:k+1]
            return result_dists, result_indices
    
    return distances, raw_indices


def compute_self_radius(
    vectors: np.ndarray,
    k: int = 5,
    radius_quantile: float = 0.95,
    neighbor_agg: str = "kth",
    filter_duplicates: bool = True,
    use_gpu: bool = True,
    index_factory: str = "Flat",
    nprobe: Optional[int] = None,
    batch_size: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Compute self-radius: typical distance to k-th nearest neighbor within a dataset.
    
    Args:
        vectors: (N, D) vectors
        k: Number of neighbors to consider
        radius_quantile: Quantile for radius (0.95 = p95)
        neighbor_agg: How to aggregate k neighbors ("kth", "first", "mean", "median")
        use_gpu: Use GPU acceleration
        index_factory: FAISS index type
        batch_size: Batch size for searches
        verbose: Print progress
        
    Returns:
        Dictionary with:
            - radius: self-radius (quantile of distances)
            - median: median distance
            - p90: 90th percentile
            - p95: 95th percentile
            - mean: mean distance
    """
    if vectors.shape[0] < 2:
        return {
            'radius': float('nan'),
            'median': float('nan'),
            'p90': float('nan'),
            'p95': float('nan'),
            'mean': float('nan'),
        }
    
    if verbose:
        print(f"  Computing self-radius: {vectors.shape[0]:,} vectors, k={k}, quantile={radius_quantile}")
    
    # Build index
    index = build_index(vectors, use_gpu=use_gpu, index_factory=index_factory, nprobe=nprobe, verbose=verbose)
    fallback_index = None
    if index_factory.lower() != "flat":
        fallback_index = build_index(vectors, use_gpu=True, index_factory="Flat", verbose=False)
    
    # Search for k+1 neighbors (including self)
    try:
        distances, _ = compute_knn_distances(
            index,
            vectors,
            k=k,
            exclude_self=True,
            filter_duplicates=filter_duplicates,
            fallback_index=fallback_index,
            batch_size=batch_size,
            verbose=verbose,
        )
    finally:
        release_index(index)
        release_index(fallback_index)

    invalid_frac = _invalid_fraction(distances)
    if (invalid_frac > 1e-4) or _has_excessive_sentinel(distances):
        if fallback_index is None and (use_gpu or index_factory.lower() != "flat"):
            if verbose:
                print(
                    "  ⚠️  Invalid distances detected; retrying with Flat index on "
                    f"{'GPU' if use_gpu else 'CPU'}... (invalid_frac={invalid_frac:.6f})"
                )
            index = build_index(vectors, use_gpu=use_gpu, index_factory="Flat", nprobe=None, verbose=verbose)
            try:
                distances, _ = compute_knn_distances(
                    index, vectors, k=k, exclude_self=True, batch_size=batch_size, verbose=verbose
                )
            finally:
                release_index(index)
        elif verbose:
            print(
                f"  ⚠️  Invalid distances remain after fallback requery "
                f"(invalid_frac={invalid_frac:.6f}); continuing without full Flat fallback."
            )
    
    if distances.size == 0:
        return {
            'radius': float('nan'),
            'median': float('nan'),
            'p90': float('nan'),
            'p95': float('nan'),
            'mean': float('nan'),
        }
    
    # Aggregate across k neighbors
    if neighbor_agg in ("first", "min"):
        sample = distances[:, 0]
    elif neighbor_agg in ("kth", "last", "max"):
        sample = distances[:, -1]
    elif neighbor_agg == "mean":
        sample = distances.mean(axis=1)
    elif neighbor_agg == "median":
        sample = np.median(distances, axis=1)
    else:
        raise ValueError(f"Unsupported neighbor_agg: {neighbor_agg}")
    
    # Compute statistics
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return {
            'radius': float('nan'),
            'median': float('nan'),
            'p90': float('nan'),
            'p95': float('nan'),
            'mean': float('nan'),
        }
    
    result = {
        'radius': float(np.quantile(sample, radius_quantile)),
        'median': float(np.median(sample)),
        'p90': float(np.quantile(sample, 0.90)),
        'p95': float(np.quantile(sample, 0.95)),
        'mean': float(np.mean(sample)),
    }
    
    if verbose:
        print(f"    Self-radius: {result['radius']:.6f} (median={result['median']:.6f}, p90={result['p90']:.6f})")
    
    return result


def compute_directed_distances(
    train_vectors: np.ndarray,
    eval_vectors: np.ndarray,
    k: int = 5,
    use_gpu: bool = True,
    index_factory: str = "Flat",
    nprobe: Optional[int] = None,
    batch_size: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute directed nearest-neighbor distances between train and eval.
    
    Args:
        train_vectors: (N_train, D) training vectors
        eval_vectors: (N_eval, D) eval vectors
        k: Number of neighbors to search
        use_gpu: Use GPU acceleration
        index_factory: FAISS index type
        batch_size: Batch size for searches
        verbose: Print progress
        
    Returns:
        Dictionary with:
            - eval_to_train: (N_eval, k) distances from eval to nearest train
            - train_to_eval: (N_train, k) distances from train to nearest eval
    """
    if verbose:
        print(f"  Computing directed distances: train={train_vectors.shape[0]:,}, eval={eval_vectors.shape[0]:,}, k={k}")
    
    # Build indices
    eval_to_train = None
    train_to_eval = None
    train_index = None
    eval_index = None
    train_fallback = None
    eval_fallback = None

    # Build train index and compute eval → train distances first
    if verbose:
        print(f"  Building train index...")
    try:
        train_index = build_index(
            train_vectors, use_gpu=use_gpu, index_factory=index_factory, nprobe=nprobe, verbose=verbose
        )
        if index_factory.lower() != "flat":
            train_fallback = build_index(train_vectors, use_gpu=True, index_factory="Flat", verbose=False)
        if verbose:
            print(f"  Computing eval→train distances...")
        eval_to_train, _ = compute_knn_distances(
            train_index,
            eval_vectors,
            k=k,
            exclude_self=False,
            fallback_index=train_fallback,
            batch_size=batch_size,
            verbose=verbose,
        )
    finally:
        release_index(train_index)
        release_index(train_fallback)

    # Build eval index and compute train → eval distances second
    if verbose:
        print(f"  Building eval index...")
    try:
        eval_index = build_index(
            eval_vectors, use_gpu=use_gpu, index_factory=index_factory, nprobe=nprobe, verbose=verbose
        )
        if index_factory.lower() != "flat":
            eval_fallback = build_index(eval_vectors, use_gpu=True, index_factory="Flat", verbose=False)
        if verbose:
            print(f"  Computing train→eval distances...")
        train_to_eval, _ = compute_knn_distances(
            eval_index,
            train_vectors,
            k=k,
            exclude_self=False,
            fallback_index=eval_fallback,
            batch_size=batch_size,
            verbose=verbose,
        )
    finally:
        release_index(eval_index)
        release_index(eval_fallback)

    invalid_frac_eval = _invalid_fraction(eval_to_train)
    invalid_frac_train = _invalid_fraction(train_to_eval)
    if (
        (invalid_frac_eval > 1e-4)
        or (invalid_frac_train > 1e-4)
        or _has_excessive_sentinel(eval_to_train)
        or _has_excessive_sentinel(train_to_eval)
    ):
        if verbose:
            print(
                "  ⚠️  Invalid distances remain after fallback requery; "
                f"continuing without full Flat fallback. "
                f"(invalid_frac_eval={invalid_frac_eval:.6f}, invalid_frac_train={invalid_frac_train:.6f})"
            )
    
    if verbose:
        print(f"  Directed distances complete")
    
    return {
        'eval_to_train': eval_to_train,
        'train_to_eval': train_to_eval,
    }


def compute_eval_to_train(
    train_vectors: np.ndarray,
    eval_vectors: np.ndarray,
    k: int = 5,
    use_gpu: bool = True,
    index_factory: str = "Flat",
    nprobe: Optional[int] = None,
    batch_size: Optional[int] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Compute eval → train distances only."""
    if verbose:
        print(f"  Computing eval→train distances...")
        print(f"  Building train index...")
    train_index = None
    try:
        train_index = build_index(
            train_vectors, use_gpu=use_gpu, index_factory=index_factory, nprobe=nprobe, verbose=verbose
        )
        eval_to_train, _ = compute_knn_distances(
            train_index,
            eval_vectors,
            k=k,
            exclude_self=False,
            batch_size=batch_size,
            verbose=verbose,
        )
    finally:
        release_index(train_index)
    return eval_to_train


def compute_train_to_eval(
    train_vectors: np.ndarray,
    eval_vectors: np.ndarray,
    k: int = 5,
    use_gpu: bool = True,
    index_factory: str = "Flat",
    nprobe: Optional[int] = None,
    batch_size: Optional[int] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Compute train → eval distances only."""
    if verbose:
        print(f"  Computing train→eval distances...")
        print(f"  Building eval index...")
    eval_index = None
    try:
        eval_index = build_index(
            eval_vectors, use_gpu=use_gpu, index_factory=index_factory, nprobe=nprobe, verbose=verbose
        )
        train_to_eval, _ = compute_knn_distances(
            eval_index,
            train_vectors,
            k=k,
            exclude_self=False,
            batch_size=batch_size,
            verbose=verbose,
        )
    finally:
        release_index(eval_index)
    return train_to_eval


def distance_statistics(distances: np.ndarray) -> Dict[str, float]:
    """
    Compute summary statistics for distance array.
    
    Args:
        distances: (N,) or (N, k) distance array
        
    Returns:
        Dictionary with mean, median, p90, p95, min, max
    """
    if distances.size == 0:
        return {
            'mean': float('nan'),
            'median': float('nan'),
            'p90': float('nan'),
            'p95': float('nan'),
            'min': float('nan'),
            'max': float('nan'),
        }
    
    # Flatten if needed
    dists_flat = distances.flatten()
    dists_flat = dists_flat[np.isfinite(dists_flat)]
    
    if dists_flat.size == 0:
        return {
            'mean': float('nan'),
            'median': float('nan'),
            'p90': float('nan'),
            'p95': float('nan'),
            'min': float('nan'),
            'max': float('nan'),
        }
    
    return {
        'mean': float(np.mean(dists_flat)),
        'median': float(np.median(dists_flat)),
        'p90': float(np.quantile(dists_flat, 0.90)),
        'p95': float(np.quantile(dists_flat, 0.95)),
        'min': float(np.min(dists_flat)),
        'max': float(np.max(dists_flat)),
    }
