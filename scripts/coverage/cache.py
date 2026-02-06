"""
Caching and preprocessing utilities for coverage analysis.

Handles:
- Raw vector caching (load/save)
- PCA fitting and transformation
- L2 normalization (for dino/resnet)
- Flow normalization (to [-1, 1])
- Alpha caching (for flow scaling)
- Radius caching (per dataset/space)
"""

from pathlib import Path
from typing import Optional, Dict, Tuple
import re
import pickle
import numpy as np
from sklearn.decomposition import PCA


def sanitize_name(name: str) -> str:
    """Sanitize dataset/split name for use in filenames."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


# ============================================================================
# Raw Vector Caching
# ============================================================================

def vector_cache_key(dataset: str, split: str, representation: str, ext: str = "npy") -> str:
    """
    Generate cache key for raw vectors.
    
    Format: {dataset}_{split}_{representation}.{ext}
    """
    safe_ds = sanitize_name(dataset)
    safe_split = sanitize_name(split)
    safe_repr = sanitize_name(representation)
    return f"{safe_ds}_{safe_split}_{safe_repr}.{ext}"


def load_cached_vectors(
    cache_dir: Path,
    dataset: str,
    split: str,
    representation: str,
    mmap: bool = False,
) -> Optional[np.ndarray]:
    """
    Load cached vectors from disk.
    
    Args:
        cache_dir: Cache directory
        dataset: Dataset name
        split: Split name
        representation: "flow", "dino", or "resnet"
        mmap: Use memory mapping
        
    Returns:
        (N, D) array of vectors, or None if not found
    """
    for ext in ("npy", "npz"):
        path = cache_dir / vector_cache_key(dataset, split, representation, ext)
        if path.exists():
            if ext == "npy":
                return np.load(path, mmap_mode="r" if mmap else None)
            else:  # npz
                data = np.load(path, mmap_mode="r" if mmap else None)
                if "vectors" in data:
                    return data["vectors"]
                else:
                    raise ValueError(f"Missing 'vectors' key in {path}")
    return None


# ============================================================================
# Directed distance caching (cross-dataset)
# ============================================================================

def directed_cache_key(
    train_dataset: str,
    train_split: str,
    eval_dataset: str,
    eval_split: str,
    space: str,
    k: int,
    direction: str,
    normalization: str,
    distance_metric: str,
    alpha: Optional[float] = None,
    ext: str = "npz",
) -> str:
    safe_train = sanitize_name(train_dataset)
    safe_train_split = sanitize_name(train_split)
    safe_eval = sanitize_name(eval_dataset)
    safe_eval_split = sanitize_name(eval_split)
    safe_space = sanitize_name(space)
    safe_norm = sanitize_name(normalization)
    safe_metric = sanitize_name(distance_metric)
    safe_dir = sanitize_name(direction)
    alpha_str = f"_a{alpha:.6g}" if alpha is not None else ""
    return (
        f"directed_{safe_train}_{safe_train_split}_to_{safe_eval}_{safe_eval_split}_"
        f"{safe_space}_{safe_norm}_{safe_metric}_k{k}_{safe_dir}{alpha_str}.{ext}"
    )


def load_directed_distances(
    cache_dir: Path,
    train_dataset: str,
    train_split: str,
    eval_dataset: str,
    eval_split: str,
    space: str,
    k: int,
    direction: str,
    normalization: str,
    distance_metric: str,
    alpha: Optional[float] = None,
) -> Optional[np.ndarray]:
    directed_dir = cache_dir / "directed"
    path = directed_dir / directed_cache_key(
        train_dataset,
        train_split,
        eval_dataset,
        eval_split,
        space,
        k,
        direction,
        normalization,
        distance_metric,
        alpha=alpha,
        ext="npz",
    )
    if not path.exists():
        return None
    data = np.load(path)
    return data["distances"] if "distances" in data else None


def save_directed_distances(
    cache_dir: Path,
    train_dataset: str,
    train_split: str,
    eval_dataset: str,
    eval_split: str,
    space: str,
    k: int,
    direction: str,
    normalization: str,
    distance_metric: str,
    distances: np.ndarray,
    alpha: Optional[float] = None,
) -> Path:
    directed_dir = cache_dir / "directed"
    directed_dir.mkdir(parents=True, exist_ok=True)
    path = directed_dir / directed_cache_key(
        train_dataset,
        train_split,
        eval_dataset,
        eval_split,
        space,
        k,
        direction,
        normalization,
        distance_metric,
        alpha=alpha,
        ext="npz",
    )
    np.savez(path, distances=distances)
    return path


def save_cached_vectors(
    cache_dir: Path,
    dataset: str,
    split: str,
    representation: str,
    vectors: np.ndarray,
    dtype: Optional[str] = "float32",
    compressed: bool = False,
) -> Path:
    """
    Save vectors to cache.
    
    Args:
        cache_dir: Cache directory
        dataset: Dataset name
        split: Split name
        representation: "flow", "dino", or "resnet"
        vectors: (N, D) array to save
        dtype: Data type to save as
        compressed: Use npz compression
        
    Returns:
        Path to saved file
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    if dtype:
        vectors = vectors.astype(dtype, copy=False)
    
    ext = "npz" if compressed else "npy"
    path = cache_dir / vector_cache_key(dataset, split, representation, ext)
    
    if compressed:
        np.savez_compressed(path, vectors=vectors)
    else:
        np.save(path, vectors)
    
    return path


def cache_exists(
    cache_dir: Path,
    dataset: str,
    split: str,
    representation: str,
) -> bool:
    """Check if cached vectors exist."""
    for ext in ("npy", "npz"):
        path = cache_dir / vector_cache_key(dataset, split, representation, ext)
        if path.exists():
            return True
    return False


# ============================================================================
# PCA + L2 Normalization (for Dino/ResNet)
# ============================================================================

def fit_pca(
    vectors: np.ndarray,
    output_dim: int,
    whiten: bool = False,
    random_state: int = 42,
) -> PCA:
    """
    Fit PCA model on vectors.
    
    Args:
        vectors: (N, D) training vectors
        output_dim: Target dimensionality
        whiten: Whether to whiten
        random_state: Random seed
        
    Returns:
        Fitted PCA model
    """
    print(f"  Fitting PCA: {vectors.shape[0]:,} vectors, {vectors.shape[1]}→{output_dim} dims")
    
    pca = PCA(n_components=output_dim, whiten=whiten, random_state=random_state)
    pca.fit(vectors)
    
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"    Explained variance: {explained_var:.4f}")
    
    return pca


def apply_pca(pca: PCA, vectors: np.ndarray) -> np.ndarray:
    """
    Apply PCA transformation.
    
    Args:
        pca: Fitted PCA model
        vectors: (N, D_in) vectors
        
    Returns:
        (N, D_out) transformed vectors
    """
    return pca.transform(vectors).astype(np.float32)


def l2_normalize(vectors: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """
    L2 normalize vectors to unit length.
    
    Args:
        vectors: (N, D) vectors
        epsilon: Small constant for numerical stability
        
    Returns:
        (N, D) unit-normalized vectors
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, epsilon)  # Avoid division by zero
    return (vectors / norms).astype(np.float32)


def save_pca_model(pca: PCA, cache_dir: Path, representation: str) -> Path:
    """
    Save PCA model to cache.
    
    Args:
        pca: PCA model
        cache_dir: Cache directory
        representation: "dino" or "resnet"
        
    Returns:
        Path to saved model
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"pca_model_{sanitize_name(representation)}.pkl"
    
    with open(path, 'wb') as f:
        pickle.dump(pca, f)
    
    print(f"  Saved PCA model to {path}")
    return path


def load_pca_model(cache_dir: Path, representation: str) -> Optional[PCA]:
    """
    Load PCA model from cache.
    
    Args:
        cache_dir: Cache directory
        representation: "dino" or "resnet"
        
    Returns:
        PCA model or None if not found
    """
    path = cache_dir / f"pca_model_{sanitize_name(representation)}.pkl"
    
    if not path.exists():
        return None
    
    with open(path, 'rb') as f:
        pca = pickle.load(f)
    
    print(f"  Loaded PCA model from {path}")
    return pca


def apply_pca_preprocessing(
    cache_dir: Path,
    train_vectors: Dict[Tuple[str, str], np.ndarray],
    eval_vectors: Dict[Tuple[str, str], np.ndarray],
    representation: str,
    output_dim: int = 256,
    whiten: bool = False,
    do_l2_normalize: bool = True,
    max_train_vectors: int = 500_000,
    verbose: bool = True,
) -> Tuple[Dict[Tuple[str, str], np.ndarray], Dict[Tuple[str, str], np.ndarray]]:
    """
    Apply PCA + optional L2 normalization to all vectors.
    
    This is a convenience function that:
    1. Loads or fits PCA model on training data
    2. Transforms all vectors
    3. Optionally L2-normalizes to unit length
    
    Args:
        cache_dir: Cache directory
        train_vectors: Dict mapping (dataset, split) to raw vectors
        eval_vectors: Dict mapping (dataset, split) to raw vectors
        representation: "dino" or "resnet"
        output_dim: Target dimensionality
        whiten: Whether to whiten during PCA
        do_l2_normalize: Whether to L2 normalize after PCA
        max_train_vectors: Max vectors for PCA fitting
        verbose: Print progress
        
    Returns:
        (transformed_train_vectors, transformed_eval_vectors)
    """
    if verbose:
        print(f"\nPCA Preprocessing:")
        print(f"  Input dim: {next(iter(train_vectors.values())).shape[1]}")
        print(f"  Output dim: {output_dim}")
        print(f"  Whiten: {whiten}")
        print(f"  L2 normalize: {do_l2_normalize}")
    
    # Try to load existing PCA model
    pca = load_pca_model(cache_dir, representation)
    
    if pca is None:
        if verbose:
            print(f"  Fitting PCA on training data...")
        
        # Collect training vectors for PCA fitting
        all_train_vecs = []
        for vectors in train_vectors.values():
            all_train_vecs.append(vectors)
        all_train_vecs = np.concatenate(all_train_vecs, axis=0)
        
        # Sample if too large
        if len(all_train_vecs) > max_train_vectors:
            indices = np.random.choice(len(all_train_vecs), size=max_train_vectors, replace=False)
            all_train_vecs = all_train_vecs[indices]
        
        if verbose:
            print(f"  Fitting on {len(all_train_vecs):,} vectors...")
        
        pca = fit_pca(all_train_vecs, output_dim=output_dim, whiten=whiten)
        save_pca_model(pca, cache_dir, representation)
    
    # Transform all vectors
    transformed_train = {}
    transformed_eval = {}
    
    if verbose:
        print(f"  Transforming vectors...")
    
    for key, vectors in train_vectors.items():
        transformed = apply_pca(pca, vectors)
        if do_l2_normalize:
            transformed = l2_normalize(transformed)
        transformed_train[key] = transformed
    
    for key, vectors in eval_vectors.items():
        transformed = apply_pca(pca, vectors)
        if do_l2_normalize:
            transformed = l2_normalize(transformed)
        transformed_eval[key] = transformed
    
    if verbose:
        sample_dim = next(iter(transformed_train.values())).shape[1]
        print(f"  ✓ Transformed to {sample_dim}-D")
        if do_l2_normalize:
            sample_norm = np.linalg.norm(next(iter(transformed_train.values()))[0])
            print(f"  ✓ L2 normalized (sample norm: {sample_norm:.6f})")
    
    return transformed_train, transformed_eval


# ============================================================================
# Alpha Caching (Flow Calibration)
# ============================================================================

def alpha_cache_key(representation: str = "flow", dedup: bool = False) -> str:
    """Generate cache key for global alpha."""
    key = f"global_alpha_{sanitize_name(representation)}"
    if dedup:
        key += "_dedup"
    return f"{key}.npz"


def save_alpha(
    cache_dir: Path,
    alpha: float,
    per_dataset_alphas: Dict[str, float],
    representation: str = "flow",
    dedup: bool = False,
    extra_metadata: Optional[Dict] = None,
) -> Path:
    """
    Save global alpha and per-dataset alphas.
    
    Args:
        cache_dir: Cache directory
        alpha: Global alpha value
        per_dataset_alphas: Dict mapping dataset names to per-dataset alphas
        representation: "flow"
        extra_metadata: Additional metadata to save
        
    Returns:
        Path to saved file
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / alpha_cache_key(representation, dedup=dedup)
    
    # Prepare data
    save_dict = {
        'alpha': np.array(alpha, dtype=np.float32),
        'per_dataset_alphas': per_dataset_alphas,
    }
    
    if extra_metadata:
        save_dict.update(extra_metadata)
    
    np.savez(path, **save_dict)
    
    print(f"  Saved global α={alpha:.6f} to {path}")
    return path


def load_alpha(
    cache_dir: Path,
    representation: str = "flow",
    dedup: bool = False,
) -> Optional[Tuple[float, Dict[str, float]]]:
    """
    Load global alpha and per-dataset alphas.
    
    Args:
        cache_dir: Cache directory
        representation: "flow"
        
    Returns:
        (global_alpha, per_dataset_alphas) or None if not found
    """
    path = cache_dir / alpha_cache_key(representation, dedup=dedup)
    
    if not path.exists():
        return None
    
    try:
        data = np.load(path, allow_pickle=True)
        alpha = float(data['alpha'].item())
        per_dataset_alphas = data['per_dataset_alphas'].item()
        
        print(f"  Loaded cached global α={alpha:.6f} from {path}")
        print(f"    Per-dataset alphas: {len(per_dataset_alphas)} datasets")
        
        return alpha, per_dataset_alphas
    
    except Exception as e:
        print(f"  ⚠️  Error loading alpha cache: {e}")
        return None


# ============================================================================
# Radius Caching (Per Dataset/Space)
# ============================================================================

def radius_cache_key(
    dataset: str,
    split: str,
    space: str,
    k: int = 5,
    quantile: float = 0.95,
    dedup: bool = False,
    alpha: Optional[float] = None,
    normalization: str = "norm2x1",
    distance_metric: str = "sqL2",
) -> str:
    """
    Generate cache key for self-radius.
    
    CRITICAL: Cache keys must include ALL factors that affect radius value:
    - Normalization mapping (norm2x1 = 2*x/W-1)
    - Distance metric (sqL2 = squared L2, no sqrt)
    - k value and quantile
    - For joint space: alpha value
    
    Format: radius_{dataset}_{split}_{space}_{norm}_{metric}_k{k}_q{quantile}[_a{alpha}].npz
    """
    safe_ds = sanitize_name(dataset)
    safe_split = sanitize_name(split)
    safe_space = sanitize_name(space)
    q_str = f"{quantile:.2f}".replace(".", "p")
    
    # Base key
    key = f"radius_{safe_ds}_{safe_split}_{safe_space}_{normalization}_{distance_metric}_k{k}_q{q_str}"
    
    # Add alpha for joint space
    if alpha is not None and space == "joint":
        alpha_str = f"{alpha:.4f}".replace(".", "p").replace("-", "m")
        key += f"_a{alpha_str}"
    
    if dedup:
        key += "_dedup"
    
    return f"{key}.npz"


def save_radius(
    cache_dir: Path,
    dataset: str,
    split: str,
    space: str,
    radius_data: Dict[str, float],
    k: int = 5,
    quantile: float = 0.95,
    dedup: bool = False,
    alpha: Optional[float] = None,
    normalization: str = "norm2x1",
    distance_metric: str = "sqL2",
) -> Path:
    """
    Save self-radius to cache with full metadata.
    
    Args:
        cache_dir: Cache directory
        dataset: Dataset name
        split: Split name
        space: Space name ("xy", "flow", "joint", "features")
        radius_data: Dict with 'radius', 'median', 'p90', 'p95', 'mean'
        k: k value used
        quantile: Quantile used
        alpha: Alpha value (for joint space)
        normalization: Normalization scheme
        distance_metric: Distance metric used
        
    Returns:
        Path to saved file
    """
    radii_dir = cache_dir / "radii"
    radii_dir.mkdir(parents=True, exist_ok=True)
    
    path = radii_dir / radius_cache_key(dataset, split, space, k, quantile, dedup, alpha, normalization, distance_metric)
    
    # Save with full metadata
    save_dict = {key: np.array(val) for key, val in radius_data.items()}
    save_dict.update({
        'k': np.array(k),
        'quantile': np.array(quantile),
        'alpha': np.array(alpha if alpha is not None else np.nan),
        'distance_metric': distance_metric,
        'normalization': normalization,
    })
    
    np.savez(path, **save_dict)
    
    return path


def load_radius(
    cache_dir: Path,
    dataset: str,
    split: str,
    space: str,
    k: int = 5,
    quantile: float = 0.95,
    dedup: bool = False,
    alpha: Optional[float] = None,
    normalization: str = "norm2x1",
    distance_metric: str = "sqL2",
) -> Optional[Dict[str, float]]:
    """
    Load self-radius from cache.
    
    Args:
        cache_dir: Cache directory
        dataset: Dataset name
        split: Split name
        space: Space name
        k: k value
        quantile: Quantile value
        alpha: Alpha value (for joint space)
        normalization: Normalization scheme
        distance_metric: Distance metric
        
    Returns:
        Dict with radius data or None if not found
    """
    radii_dir = cache_dir / "radii"
    path = radii_dir / radius_cache_key(dataset, split, space, k, quantile, dedup, alpha, normalization, distance_metric)
    
    if not path.exists():
        return None
    
    try:
        data = np.load(path, allow_pickle=True)
        result = {}
        for key in data.files:
            val = data[key]
            if isinstance(val, np.ndarray) and val.shape == ():
                result[key] = float(val.item()) if np.issubdtype(val.dtype, np.number) else str(val.item())
            else:
                result[key] = val
        return result
    except Exception as e:
        print(f"  ⚠️  Error loading radius cache: {e}")
        return None


def radius_cache_exists(
    cache_dir: Path,
    dataset: str,
    split: str,
    space: str,
    k: int = 5,
    quantile: float = 0.95,
    alpha: Optional[float] = None,
    normalization: str = "norm2x1",
    distance_metric: str = "sqL2",
) -> bool:
    """Check if radius cache exists."""
    radii_dir = cache_dir / "radii"
    path = radii_dir / radius_cache_key(dataset, split, space, k, quantile, alpha, normalization, distance_metric)
    return path.exists()
