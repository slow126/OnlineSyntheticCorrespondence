"""
KL divergence utilities for coverage pipeline.

Two estimators:
  kNN (Wang et al. 2009):
    KL(P||Q) ≈ (d / n) * sum_i log( s_k(x_i) / r_k(x_i) ) + log( m / (n - 1) )
    where r_k / s_k are the k-th self / cross NN distances.

  KDE (Silverman rule-of-thumb diagonal bandwidth):
    KL(P||Q) ≈ E_P[ log p̂(x) - log q̂(x) ]
    Evaluated via Monte Carlo on a subsample of P.
"""

from typing import Dict, List, Optional
import numpy as np

from . import faiss_ops


def _as_l2(distances: np.ndarray, distance_metric: str) -> np.ndarray:
    """Convert squared L2 distances to L2 if needed."""
    metric = (distance_metric or "").lower()
    if metric in {"sql2", "sq_l2", "l2_sq", "sq-l2", "squared_l2"}:
        return np.sqrt(np.maximum(distances, 0.0))
    return distances


def _auto_batch_size(dim: int) -> int:
    if dim <= 4:
        return 500000
    if dim <= 64:
        return 100000
    if dim <= 256:
        return 50000
    return 20000


def _filter_knn_batch(
    raw_dists: np.ndarray,
    raw_indices: np.ndarray,
    k: int,
    exclude_self: bool,
    filter_duplicates: bool,
) -> np.ndarray:
    if raw_dists.size == 0:
        return raw_dists[:, :k]
    if exclude_self:
        if filter_duplicates:
            tiny = 1e-12
            distances_copy = raw_dists.copy()
            is_duplicate = distances_copy <= tiny
            distances_copy[is_duplicate] = np.inf
            distances_copy[~np.isfinite(distances_copy)] = np.inf
            sort_idx = np.argsort(distances_copy, axis=1)
            row_idx = np.arange(raw_dists.shape[0])[:, None]
            result_dists = distances_copy[row_idx, sort_idx[:, :k]]
            return result_dists
        # Drop closest neighbor only
        if raw_dists.shape[1] < k + 1:
            raise ValueError(f"Not enough neighbors to exclude self for k={k}")
        return raw_dists[:, 1 : k + 1]
    return raw_dists[:, :k]


def compute_self_knn_distances_streaming(
    vectors: np.ndarray,
    k: int,
    distance_metric: str,
    out_path,
    filter_duplicates: bool = True,
    use_gpu: bool = True,
    index_factory: str = "Flat",
    nprobe: Optional[int] = None,
    batch_size: Optional[int] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Streaming self-kNN computation that writes directly to a .npy memmap.
    Returns the memmap array.
    """
    if vectors.size == 0 or vectors.shape[0] < 2 or k < 1:
        return np.array([], dtype=np.float32).reshape(0, k)

    n = vectors.shape[0]
    dim = vectors.shape[1]
    if batch_size is None:
        batch_size = _auto_batch_size(dim)

    index = None
    fallback_index = None
    try:
        index = faiss_ops.build_index(
            vectors,
            use_gpu=use_gpu,
            index_factory=index_factory,
            nprobe=nprobe,
            verbose=verbose,
        )
        if index_factory.lower() != "flat":
            fallback_index = faiss_ops.build_index(
                vectors,
                use_gpu=True,
                index_factory="Flat",
                nprobe=None,
                verbose=False,
            )

        # Create memmap output
        from numpy.lib.format import open_memmap

        mm = open_memmap(out_path, mode="w+", dtype="float32", shape=(n, k))

        search_k = min(n, k + 32) if filter_duplicates else min(n, k + 1)
        if search_k < 1:
            return np.array([], dtype=np.float32).reshape(0, k)

        if verbose:
            print(f"    Streaming self-kNN: n={n:,}, k={k}, search_k={search_k}, batch_size={batch_size:,}")

        for i in range(0, n, batch_size):
            batch = np.ascontiguousarray(vectors[i : i + batch_size], dtype=np.float32)
            raw_dists, raw_indices = index.search(batch, search_k)

            # Handle invalids with fallback if needed
            invalid_mask = (~np.isfinite(raw_dists)) | (raw_dists >= 1e30) | (raw_indices < 0)
            if invalid_mask.any() and fallback_index is not None:
                bad_rows = np.any(invalid_mask, axis=1)
                if np.any(bad_rows):
                    fb_queries = batch[bad_rows]
                    fb_dists, fb_indices = fallback_index.search(fb_queries, search_k)
                    raw_dists[bad_rows] = fb_dists
                    raw_indices[bad_rows] = fb_indices

            result_dists = _filter_knn_batch(
                raw_dists, raw_indices, k=k, exclude_self=True, filter_duplicates=filter_duplicates
            )
            result_dists = _as_l2(result_dists, distance_metric).astype(np.float32, copy=False)
            mm[i : i + result_dists.shape[0], :] = result_dists

        mm.flush()
        return mm
    finally:
        faiss_ops.release_index(index)
        faiss_ops.release_index(fallback_index)


def compute_knn_kl_streaming(
    query_vectors: np.ndarray,
    ref_vectors: np.ndarray,
    self_knn: np.ndarray,
    k_values: List[int],
    distance_metric: str,
    eps: float,
    filter_duplicates: bool = True,
    use_gpu: bool = True,
    index_factory: str = "Flat",
    nprobe: Optional[int] = None,
    batch_size: Optional[int] = None,
    verbose: bool = True,
) -> Dict[int, float]:
    """
    Streaming KL computation. Processes query vectors in batches and accumulates
    sum(log(nu/rho)) for each k without materializing full cross distance arrays.
    """
    if query_vectors.size == 0 or ref_vectors.size == 0:
        return {k: float("nan") for k in k_values}

    n = query_vectors.shape[0]
    dim = query_vectors.shape[1]
    max_k = max(k_values)

    if batch_size is None:
        batch_size = _auto_batch_size(dim)

    sum_log = {k: 0.0 for k in k_values}
    count = {k: 0 for k in k_values}

    index = None
    fallback_index = None
    try:
        index = faiss_ops.build_index(
            ref_vectors,
            use_gpu=use_gpu,
            index_factory=index_factory,
            nprobe=nprobe,
            verbose=verbose,
        )
        if index_factory.lower() != "flat":
            fallback_index = faiss_ops.build_index(
                ref_vectors,
                use_gpu=True,
                index_factory="Flat",
                nprobe=None,
                verbose=False,
            )

        search_k = min(ref_vectors.shape[0], max_k + 32) if filter_duplicates else min(ref_vectors.shape[0], max_k + 1)
        if search_k < 1:
            return {k: float("nan") for k in k_values}

        if verbose:
            print(f"    Streaming cross-kNN: n={n:,}, k_max={max_k}, search_k={search_k}, batch_size={batch_size:,}")

        for i in range(0, n, batch_size):
            batch = np.ascontiguousarray(query_vectors[i : i + batch_size], dtype=np.float32)
            raw_dists, raw_indices = index.search(batch, search_k)

            invalid_mask = (~np.isfinite(raw_dists)) | (raw_dists >= 1e30) | (raw_indices < 0)
            if invalid_mask.any() and fallback_index is not None:
                bad_rows = np.any(invalid_mask, axis=1)
                if np.any(bad_rows):
                    fb_queries = batch[bad_rows]
                    fb_dists, fb_indices = fallback_index.search(fb_queries, search_k)
                    raw_dists[bad_rows] = fb_dists
                    raw_indices[bad_rows] = fb_indices

            nu = _filter_knn_batch(raw_dists, raw_indices, k=max_k, exclude_self=True, filter_duplicates=filter_duplicates)
            nu = _as_l2(nu, distance_metric)
            rho = self_knn[i : i + nu.shape[0], :]

            for k in k_values:
                rho_k = rho[:, k - 1]
                nu_k = nu[:, k - 1]
                mask = np.isfinite(rho_k) & np.isfinite(nu_k)
                if not np.any(mask):
                    continue
                rho_k = np.maximum(rho_k[mask], eps)
                nu_k = np.maximum(nu_k[mask], eps)
                sum_log[k] += float(np.sum(np.log(nu_k / rho_k)))
                count[k] += int(mask.sum())

        out = {}
        m = int(ref_vectors.shape[0])
        for k in k_values:
            n_eff = count[k]
            if n_eff <= 1:
                out[k] = float("nan")
                continue
            out[k] = float((dim / n_eff) * sum_log[k] + np.log(m / (n_eff - 1)))
        return out
    finally:
        faiss_ops.release_index(index)
        faiss_ops.release_index(fallback_index)


def compute_self_knn_distances(
    vectors: np.ndarray,
    k: int,
    distance_metric: str,
    filter_duplicates: bool = True,
    use_gpu: bool = True,
    index_factory: str = "Flat",
    nprobe: Optional[int] = None,
    batch_size: Optional[int] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Compute self kNN distances (excluding self) for a dataset."""
    if vectors.size == 0 or vectors.shape[0] < 2 or k < 1:
        return np.array([], dtype=np.float32).reshape(0, k)

    index = None
    fallback_index = None
    try:
        index = faiss_ops.build_index(
            vectors,
            use_gpu=use_gpu,
            index_factory=index_factory,
            nprobe=nprobe,
            verbose=verbose,
        )
        if index_factory.lower() != "flat":
            fallback_index = faiss_ops.build_index(
                vectors,
                use_gpu=True,
                index_factory="Flat",
                nprobe=None,
                verbose=False,
            )
        distances, _ = faiss_ops.compute_knn_distances(
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
        faiss_ops.release_index(index)
        faiss_ops.release_index(fallback_index)

    return _as_l2(distances, distance_metric)


def compute_cross_knn_distances(
    query_vectors: np.ndarray,
    ref_vectors: np.ndarray,
    k: int,
    distance_metric: str,
    filter_duplicates: bool = True,
    use_gpu: bool = True,
    index_factory: str = "Flat",
    nprobe: Optional[int] = None,
    batch_size: Optional[int] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Compute cross kNN distances from query to reference (excluding duplicates)."""
    if query_vectors.size == 0 or ref_vectors.size == 0 or k < 1:
        return np.array([], dtype=np.float32).reshape(0, k)

    index = None
    fallback_index = None
    try:
        index = faiss_ops.build_index(
            ref_vectors,
            use_gpu=use_gpu,
            index_factory=index_factory,
            nprobe=nprobe,
            verbose=verbose,
        )
        if index_factory.lower() != "flat":
            fallback_index = faiss_ops.build_index(
                ref_vectors,
                use_gpu=True,
                index_factory="Flat",
                nprobe=None,
                verbose=False,
            )
        distances, _ = faiss_ops.compute_knn_distances(
            index,
            query_vectors,
            k=k,
            exclude_self=True,
            filter_duplicates=filter_duplicates,
            fallback_index=fallback_index,
            batch_size=batch_size,
            verbose=verbose,
        )
    finally:
        faiss_ops.release_index(index)
        faiss_ops.release_index(fallback_index)

    return _as_l2(distances, distance_metric)


def compute_knn_kl_for_k_values(
    rho: np.ndarray,
    nu: np.ndarray,
    m: int,
    dim: int,
    k_values: List[int],
    eps: float,
    k_column_map: Optional[Dict[int, int]] = None,
) -> Dict[int, float]:
    """
    Compute kNN KL divergence for specified k values.

    rho: (N, K) self distances for P (exclude self)
    nu:  (N, K) cross distances from P to Q
    """
    out: Dict[int, float] = {}
    if rho.size == 0 or nu.size == 0:
        for k in k_values:
            out[k] = float("nan")
        return out

    max_k = min(rho.shape[1], nu.shape[1])
    for k in k_values:
        col_idx = k - 1 if k_column_map is None else k_column_map.get(k, -1)
        if col_idx < 0 or col_idx >= max_k:
            out[k] = float("nan")
            continue
        rho_k = rho[:, col_idx]
        nu_k = nu[:, col_idx]
        mask = np.isfinite(rho_k) & np.isfinite(nu_k)
        if not np.any(mask):
            out[k] = float("nan")
            continue
        rho_k = np.maximum(rho_k[mask], eps)
        nu_k = np.maximum(nu_k[mask], eps)
        n_eff = int(rho_k.size)
        if n_eff <= 1:
            out[k] = float("nan")
            continue
        out[k] = float((dim / n_eff) * np.sum(np.log(nu_k / rho_k)) + np.log(m / (n_eff - 1)))

    return out


# ---------------------------------------------------------------------------
# KDE-based KL divergence (Silverman rule-of-thumb diagonal bandwidth)
# ---------------------------------------------------------------------------

def silverman_bandwidth(n: int, d: int, std_per_dim: np.ndarray) -> np.ndarray:
    """
    Diagonal Silverman rule-of-thumb bandwidth for a d-dimensional Gaussian KDE.
      h_j = sigma_j * (4 / (n * (d + 2)))^(1 / (d + 4))
    """
    factor = (4.0 / (n * (d + 2))) ** (1.0 / (d + 4))
    return factor * np.maximum(std_per_dim, 1e-30)


def _auto_kde_batch_size(n_ref: int, d: int, target_mb: int = 256) -> int:
    """Batch size so (batch, n_ref) float64 matrix stays under target_mb."""
    bytes_per_row = n_ref * 8
    return max(1, min(2000, (target_mb * 1024 * 1024) // bytes_per_row))


def _log_kde_density(
    query: np.ndarray,  # (n_q, d) float32
    ref: np.ndarray,    # (n_r, d) float32
    bw: np.ndarray,     # (d,)     float64 bandwidth per dimension
    batch_size: int,
) -> np.ndarray:
    """Log KDE density at each query point using diagonal Gaussian kernel."""
    n_q, d = query.shape
    n_r = ref.shape[0]
    bw = np.maximum(bw, 1e-30).astype(np.float64)

    # Whiten once, work in normalized space
    q_n = (query.astype(np.float64)) / bw[None, :]  # (n_q, d)
    r_n = (ref.astype(np.float64)) / bw[None, :]    # (n_r, d)

    # Constant: -d/2 * log(2pi) - sum(log bw_j)
    log_norm = -0.5 * d * np.log(2.0 * np.pi) - np.sum(np.log(bw))
    log_n_r = np.log(float(n_r))

    sq_r = np.sum(r_n ** 2, axis=1)  # (n_r,) — precompute, reuse across batches

    log_densities = np.empty(n_q, dtype=np.float64)
    for i in range(0, n_q, batch_size):
        q_b = q_n[i : i + batch_size]  # (bs, d)
        bs = q_b.shape[0]
        sq_q = np.sum(q_b ** 2, axis=1)  # (bs,)
        # ||q - r||^2 via expansion: sq_q + sq_r - 2 q r^T
        dists_sq = sq_q[:, None] + sq_r[None, :] - 2.0 * (q_b @ r_n.T)  # (bs, n_r)
        log_kern = log_norm - 0.5 * dists_sq  # (bs, n_r)
        # Numerically stable logsumexp over reference axis
        lk_max = log_kern.max(axis=1)
        log_sum = lk_max + np.log(np.sum(np.exp(log_kern - lk_max[:, None]), axis=1))
        log_densities[i : i + bs] = log_sum - log_n_r

    return log_densities


def compute_kde_kl(
    p_vectors: np.ndarray,
    q_vectors: np.ndarray,
    max_ref_p: int = 50_000,
    max_ref_q: int = 50_000,
    max_query: int = 20_000,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
) -> float:
    """
    KL(P||Q) via KDE with Silverman's rule-of-thumb diagonal bandwidth.

    Both reference sets are subsampled to max_ref_{p,q} for tractability.
    Each bandwidth is calibrated to its own reference set size, so estimates
    are not biased by the raw dataset size ratio — only by the capped sizes.

    Evaluation points are drawn from P (Monte Carlo estimate of E_P[log p - log q]).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n_p, d = p_vectors.shape
    n_q = q_vectors.shape[0]

    if n_p < 2 or n_q < 2:
        return float("nan")

    # Subsample reference sets
    p_ref_idx = rng.choice(n_p, min(n_p, max_ref_p), replace=False)
    ref_p = np.ascontiguousarray(p_vectors[p_ref_idx], dtype=np.float32)

    q_ref_idx = rng.choice(n_q, min(n_q, max_ref_q), replace=False)
    ref_q = np.ascontiguousarray(q_vectors[q_ref_idx], dtype=np.float32)

    # Query points from P
    query_idx = rng.choice(n_p, min(n_p, max_query), replace=False)
    query = np.ascontiguousarray(p_vectors[query_idx], dtype=np.float32)

    # Bandwidth: calibrated to each reference set
    std_p = ref_p.astype(np.float64).std(axis=0)
    std_q = ref_q.astype(np.float64).std(axis=0)
    bw_p = silverman_bandwidth(len(ref_p), d, std_p)
    bw_q = silverman_bandwidth(len(ref_q), d, std_q)

    bs_p = _auto_kde_batch_size(len(ref_p), d)
    bs_q = _auto_kde_batch_size(len(ref_q), d)

    if verbose:
        print(f"    KDE KL: n_ref_p={len(ref_p):,}, n_ref_q={len(ref_q):,}, "
              f"n_query={len(query):,}, d={d}, "
              f"bw_p_mean={bw_p.mean():.4f}, bw_q_mean={bw_q.mean():.4f}")

    log_p = _log_kde_density(query, ref_p, bw_p, bs_p)
    log_q = _log_kde_density(query, ref_q, bw_q, bs_q)

    valid = np.isfinite(log_p) & np.isfinite(log_q)
    if not np.any(valid):
        return float("nan")

    return float(np.mean((log_p - log_q)[valid]))
