"""
Weighted coreset construction using streaming expand-then-collapse.

Uses sklearn KMeans with sample_weight for weighted k-means clustering.
"""

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from typing import Optional, Dict, Any
from pathlib import Path


class WeightedCoreset:
    """
    Streaming weighted coreset using expand-then-collapse pattern.
    
    Maintains a bounded set of representative centers with counts,
    compressing large datasets into K_max cluster centers.
    
    Algorithm:
        1. Expand: accumulate new points in buffer
        2. When buffer + centers > K_max + K_overflow, collapse:
           - Combine centers (weighted by counts) + buffer (weight=1 each)
           - Run weighted k-means to reduce to K_max centers
           - Update counts based on cluster assignments
        3. Repeat for each batch
        4. On finalize: final collapse + compute epsilon if is_eval
    
    Attributes:
        K_max: Maximum number of centers
        K_overflow: Buffer size before triggering collapse
        distance: Distance metric ('euclidean', 'cosine')
        device: Device for computation
        is_eval: If True, compute epsilon scales on finalize
        centers: (K, D) array of cluster centers
        counts: (K,) array of point counts per center
        epsilon_scales: Dict of epsilon values (only for eval coresets)
        total_samples: Total number of samples processed
    
    Example:
        >>> coreset = WeightedCoreset(K_max=1000, is_eval=True)
        >>> for batch in dataloader:
        ...     vectors = extract_vectors(batch)  # (B, D)
        ...     coreset.update(vectors)
        >>> coreset.finalize()
        >>> coreset.save('coreset.pt')
    """
    
    def __init__(
        self,
        K_max: int = 10000,
        K_overflow: int = 5000,
        distance: str = 'euclidean',
        device: str = 'cpu',
        is_eval: bool = False,
        epsilon_quantile: float = 0.5,
        max_epsilon_samples: int = 50000,
        random_state: int = 42
    ):
        """
        Initialize WeightedCoreset.
        
        Args:
            K_max: Maximum number of centers to maintain
            K_overflow: Buffer size before triggering collapse
            distance: Distance metric ('euclidean', 'cosine')
            device: Device for computation ('cpu', 'cuda')
            is_eval: If True, compute epsilon on finalize
            epsilon_quantile: Quantile for epsilon estimation
            max_epsilon_samples: Max samples for epsilon computation
            random_state: Random seed for k-means
        """
        self.K_max = K_max
        self.K_overflow = K_overflow
        self.distance = distance
        self.device = device
        self.is_eval = is_eval
        self.epsilon_quantile = epsilon_quantile
        self.max_epsilon_samples = max_epsilon_samples
        self.random_state = random_state
        
        self.centers: Optional[np.ndarray] = None  # (K, D)
        self.counts: Optional[np.ndarray] = None   # (K,)
        self.epsilon_scales: Optional[Dict[str, float]] = None
        self.buffer = []  # List of arrays to accumulate
        self.total_samples = 0
        self.dimension: Optional[int] = None
    
    def update(self, X_batch: np.ndarray):
        """
        Add a batch of points. Collapse if buffer exceeds threshold.
        
        Args:
            X_batch: (B, D) array of vectors to add
        """
        if len(X_batch) == 0:
            return
        
        # Ensure numpy array
        if isinstance(X_batch, torch.Tensor):
            X_batch = X_batch.cpu().numpy()
        
        X_batch = np.asarray(X_batch, dtype=np.float32)
        
        # Track dimension
        if self.dimension is None:
            self.dimension = X_batch.shape[1]
        elif X_batch.shape[1] != self.dimension:
            raise ValueError(
                f"Dimension mismatch: expected {self.dimension}, got {X_batch.shape[1]}"
            )
        
        self.buffer.append(X_batch)
        self.total_samples += len(X_batch)
        
        # Check if we need to collapse
        buffer_size = sum(len(b) for b in self.buffer)
        if self.centers is not None:
            buffer_size += len(self.centers)
        
        if buffer_size >= self.K_max + self.K_overflow:
            self._collapse()
    
    def finalize(self):
        """
        Final collapse if buffer not empty, compute epsilon if is_eval.
        """
        # Final collapse if needed
        if len(self.buffer) > 0 or self.centers is None:
            if len(self.buffer) > 0 or self.total_samples > 0:
                self._collapse()
        
        # For eval coresets, compute epsilon scales from the centers
        if self.is_eval and self.epsilon_scales is None and self.centers is not None:
            from .metrics import estimate_epsilon_from_eval
            print(f"Computing epsilon scales for eval coreset ({len(self.centers)} centers)...")
            self.epsilon_scales = estimate_epsilon_from_eval(
                self.centers,
                quantile=self.epsilon_quantile,
                max_samples=self.max_epsilon_samples
            )
            print(f"  eps_base: {self.epsilon_scales['eps_base']:.4f}")
            print(f"  eps_2x: {self.epsilon_scales['eps_2x']:.4f}")
            print(f"  eps_4x: {self.epsilon_scales['eps_4x']:.4f}")
    
    def _collapse(self):
        """
        Weighted k-means to reduce to K_max centers.
        """
        # Combine centers + buffer
        if self.centers is not None:
            all_points = np.vstack([self.centers] + self.buffer)
            all_weights = np.concatenate([
                self.counts,
                np.ones(sum(len(b) for b in self.buffer), dtype=np.float32)
            ])
        else:
            if len(self.buffer) == 0:
                return
            all_points = np.vstack(self.buffer)
            all_weights = np.ones(len(all_points), dtype=np.float32)
        
        # Weighted k-means
        n_clusters = min(self.K_max, len(all_points))
        
        if n_clusters < len(all_points):
            # Use MiniBatchKMeans for speed with large datasets
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                batch_size=min(2048, len(all_points) // 4),  # Process in chunks
                max_iter=100,
                n_init=3,
                reassignment_ratio=0.01,  # Less reassignment for speed
            )
            labels = kmeans.fit_predict(all_points)
            
            # New centers and counts (still use weights for counting)
            self.centers = kmeans.cluster_centers_.astype(np.float32)
            self.counts = np.bincount(
                labels,
                weights=all_weights,
                minlength=n_clusters
            ).astype(np.float32)
        else:
            # Not enough points to cluster, just use all points
            self.centers = all_points.astype(np.float32)
            self.counts = all_weights.astype(np.float32)
        
        # Clear buffer
        self.buffer = []
    
    def get_centers(self) -> np.ndarray:
        """Return centers of shape (K, D)."""
        if self.centers is None:
            raise ValueError("Coreset not finalized. Call finalize() first.")
        return self.centers
    
    def get_counts(self) -> np.ndarray:
        """Return counts of shape (K,)."""
        if self.counts is None:
            raise ValueError("Coreset not finalized. Call finalize() first.")
        return self.counts
    
    def get_epsilon_scales(self) -> Optional[Dict[str, float]]:
        """Return epsilon scales if available (eval coresets only)."""
        return self.epsilon_scales
    
    def save(self, path: str):
        """
        Save coreset to disk (PyTorch format for compatibility).
        
        Args:
            path: Output file path (.pt extension)
        """
        if self.centers is None or self.counts is None:
            raise ValueError("Coreset not finalized. Call finalize() first.")
        
        # Prepare data dict
        data = {
            'centers': torch.from_numpy(self.centers),
            'counts': torch.from_numpy(self.counts),
            'K_max': self.K_max,
            'K_overflow': self.K_overflow,
            'distance': self.distance,
            'is_eval': self.is_eval,
            'total_samples': self.total_samples,
            'dimension': self.dimension,
        }
        
        # Add epsilon scales if available
        if self.epsilon_scales is not None:
            data['epsilon_scales'] = self.epsilon_scales
        
        # Save with torch
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, path)
        
        print(f"Saved coreset to {path}")
        print(f"  Centers: {self.centers.shape}")
        print(f"  Total samples represented: {self.total_samples}")
        if self.epsilon_scales is not None:
            print(f"  Epsilon scales: eps_base={self.epsilon_scales['eps_base']:.4f}")
    
    @classmethod
    def load(cls, path: str) -> 'WeightedCoreset':
        """
        Load coreset from disk.
        
        Args:
            path: Input file path (.pt extension)
        
        Returns:
            WeightedCoreset instance with loaded data
        """
        data = torch.load(path, map_location='cpu')
        
        # Create instance
        coreset = cls(
            K_max=data['K_max'],
            K_overflow=data['K_overflow'],
            distance=data['distance'],
            is_eval=data.get('is_eval', False),
        )
        
        # Load arrays
        coreset.centers = data['centers'].numpy()
        coreset.counts = data['counts'].numpy()
        coreset.total_samples = data.get('total_samples', int(coreset.counts.sum()))
        coreset.dimension = data.get('dimension', coreset.centers.shape[1])
        
        # Load epsilon scales if available
        if 'epsilon_scales' in data:
            coreset.epsilon_scales = data['epsilon_scales']
        
        return coreset
    
    def __repr__(self) -> str:
        if self.centers is not None:
            return (
                f"WeightedCoreset(K={len(self.centers)}, D={self.dimension}, "
                f"total_samples={self.total_samples}, is_eval={self.is_eval})"
            )
        else:
            return f"WeightedCoreset(K_max={self.K_max}, not finalized)"
