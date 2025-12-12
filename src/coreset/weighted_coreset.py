"""
Weighted coreset construction using streaming incremental k-means.

Uses sklearn MiniBatchKMeans with incremental partial_fit() for efficient
streaming updates. Handles weighted centers by warm-starting with existing
centers and using incremental updates.
"""

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from typing import Optional, Dict, Any
from pathlib import Path


class WeightedCoreset:
    """
    Streaming weighted coreset using incremental MiniBatchKMeans.
    
    Maintains a bounded set of representative centers with counts,
    compressing large datasets into K_max cluster centers.
    
    Algorithm (optimized):
        1. Accumulate new points in buffer
        2. When buffer reaches threshold or we have existing centers:
           - Use existing centers to warm-start MiniBatchKMeans
           - Use partial_fit() to incrementally update with buffer points
           - Handle weighted centers by replicating them proportionally
        3. Repeat for each batch (incremental updates)
        4. On finalize: final incremental update + compute epsilon if is_eval
    
    Key optimizations:
        - Uses partial_fit() instead of fit_predict() for true streaming
        - Warm-starts with existing centers (no need to re-cluster from scratch)
        - Handles weighted points by replication or weighted initialization
        - More memory efficient (doesn't need to hold all points in memory)
    
    Attributes:
        K_max: Maximum number of centers
        K_overflow: Buffer size before triggering incremental update
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
        
        # MiniBatchKMeans instance for incremental updates
        self._kmeans: Optional[MiniBatchKMeans] = None
        self._kmeans_initialized = False
    
    def update(self, X_batch: np.ndarray):
        """
        Add a batch of points. Incrementally update k-means if needed.
        
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
        
        # Check if we need to do an incremental update
        buffer_size = sum(len(b) for b in self.buffer)
        current_centers_size = len(self.centers) if self.centers is not None else 0
        
        # Trigger incremental update if:
        # 1. Buffer is large enough (K_overflow threshold)
        # 2. We have centers and buffer combined exceeds threshold
        # 3. We don't have centers yet but buffer is substantial
        should_update = False
        if current_centers_size > 0:
            # If we have centers, update when buffer + centers exceeds threshold
            if buffer_size + current_centers_size >= self.K_max + self.K_overflow:
                should_update = True
        else:
            # If no centers yet, initialize when buffer reaches K_max
            if buffer_size >= self.K_max:
                should_update = True
        
        if should_update:
            self._incremental_update()
    
    def finalize(self):
        """
        Final incremental update if buffer not empty, compute epsilon if is_eval.
        """
        # Final incremental update if needed
        if len(self.buffer) > 0:
            self._incremental_update()
        elif self.centers is None and self.total_samples > 0:
            # Edge case: we have samples but no centers yet
            self._incremental_update()
        
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
    
    def _incremental_update(self):
        """
        Incrementally update k-means using partial_fit().
        
        Optimized approach:
        1. Warm-start with existing centers if available
        2. Use partial_fit() to incrementally update with buffer points only
        3. Recompute counts by assigning all weighted points to new centers
        """
        if len(self.buffer) == 0:
            return
        
        # Combine all buffer batches
        buffer_points = np.vstack(self.buffer) if len(self.buffer) > 1 else self.buffer[0]
        buffer_size = len(buffer_points)
        
        # Determine number of clusters
        current_centers_size = len(self.centers) if self.centers is not None else 0
        n_clusters = min(self.K_max, max(current_centers_size, min(self.K_max, buffer_size)))
        
        if n_clusters == 0:
            self.buffer = []
            return
        
        # Initialize or update MiniBatchKMeans
        needs_reinit = (
            not self._kmeans_initialized or 
            self._kmeans is None or
            self._kmeans.n_clusters != n_clusters
        )
        
        if needs_reinit:
            batch_size = min(2048, max(100, buffer_size // 10))
            
            self._kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                batch_size=batch_size,
                max_iter=100,
                n_init=1,  # Faster initialization
                reassignment_ratio=0.01,
                max_no_improvement=10,
                verbose=0,
            )
            
            # Warm-start with existing centers if available
            if self.centers is not None and len(self.centers) > 0:
                if len(self.centers) <= n_clusters:
                    # Use all existing centers, pad if needed
                    if len(self.centers) < n_clusters:
                        n_pad = n_clusters - len(self.centers)
                        pad_indices = np.random.choice(
                            buffer_size, size=min(n_pad, buffer_size), replace=False
                        )
                        init_centers = np.vstack([self.centers, buffer_points[pad_indices]])
                    else:
                        init_centers = self.centers
                else:
                    # Select top centers by count
                    top_indices = np.argsort(self.counts)[-n_clusters:]
                    init_centers = self.centers[top_indices]
                
                self._kmeans.cluster_centers_ = init_centers.astype(np.float32)
            else:
                # Initialize with k-means++ on sample of buffer
                # CRITICAL: n_clusters cannot exceed available samples
                # Use at least n_clusters samples, but cap at reasonable limit for efficiency
                # If buffer_size is large, use more samples (up to 2*n_clusters or buffer_size)
                max_sample_size = min(buffer_size, max(n_clusters, min(2000, 2 * n_clusters)))
                sample_size = max_sample_size
                
                # Ensure n_clusters doesn't exceed sample_size (safety check)
                if n_clusters > sample_size:
                    # Adjust n_clusters to not exceed sample_size
                    n_clusters = sample_size
                    # Recreate MiniBatchKMeans with corrected n_clusters
                    self._kmeans = MiniBatchKMeans(
                        n_clusters=n_clusters,
                        random_state=self.random_state,
                        batch_size=batch_size,
                        max_iter=100,
                        n_init=1,
                        reassignment_ratio=0.01,
                        max_no_improvement=10,
                        verbose=0,
                    )
                
                # Use all buffer points if sample_size >= buffer_size, otherwise sample
                if sample_size >= buffer_size:
                    sample_points = buffer_points
                else:
                    sample_indices = np.random.choice(buffer_size, size=sample_size, replace=False)
                    sample_points = buffer_points[sample_indices]
                
                self._kmeans.fit(sample_points)
            
            self._kmeans_initialized = True
        
        # Incremental update: use partial_fit with both existing centers and buffer points
        # CRITICAL: Include existing centers (replicated by counts) to prevent large batches
        # from shifting centers too much when there's already significant existing mass.
        # This ensures the existing mass is represented during the incremental update.
        
        # Prepare points for partial_fit: include existing centers replicated by counts
        points_for_update = []
        
        if self.centers is not None and len(self.centers) > 0:
            # Replicate existing centers proportionally to their counts
            # This ensures existing mass is represented during partial_fit, preventing
            # large batches from shifting centers too much.
            # 
            # Strategy: Replicate each center based on its count, but scale total replication
            # to be roughly comparable to buffer_size. This balances existing mass with new data.
            max_reps_per_center = 200  # Cap per center to avoid memory issues
            total_existing_mass = self.counts.sum()
            
            # Calculate replication: scale so total replicated points is similar to buffer_size
            # This ensures existing mass is represented but doesn't completely dominate
            if total_existing_mass > 0 and buffer_size > 0:
                # Target total replication: match buffer size (or cap at reasonable limit)
                target_total_reps = min(buffer_size, 50000)  # Cap at 50k to avoid memory issues
                # Scale factor to achieve target total replication
                scale_factor = target_total_reps / total_existing_mass
            else:
                scale_factor = 1.0
            
            for center, count in zip(self.centers, self.counts):
                # Replicate based on count, scaled to balance with buffer
                n_reps = min(
                    max_reps_per_center,
                    max(1, int(count * scale_factor))
                )
                points_for_update.append(np.tile(center[None, :], (n_reps, 1)))
        
        # Add buffer points
        points_for_update.append(buffer_points)
        
        # Combine all points for incremental update
        if len(points_for_update) > 1:
            all_update_points = np.vstack(points_for_update)
        else:
            all_update_points = points_for_update[0]
        
        # Shuffle to mix existing centers and new points
        n_points = len(all_update_points)
        shuffle_idx = np.random.permutation(n_points)
        all_update_points = all_update_points[shuffle_idx]
        
        # Use partial_fit on combined points (existing centers + buffer)
        batch_size = self._kmeans.batch_size
        for i in range(0, n_points, batch_size):
            batch = all_update_points[i:i+batch_size]
            self._kmeans.partial_fit(batch)
        
        # Get updated centers
        new_centers = self._kmeans.cluster_centers_.astype(np.float32)
        
        # Recompute counts by assigning all weighted points to new centers
        # We need to account for:
        # 1. Existing centers with their counts (weighted)
        # 2. Buffer points (weight=1 each)
        
        # Build list of all points with their weights
        all_points_list = []
        all_weights_list = []
        
        # Add existing centers with their counts as weights
        if self.centers is not None and len(self.centers) > 0:
            all_points_list.append(self.centers)
            all_weights_list.append(self.counts)
        
        # Add buffer points (weight=1 each)
        all_points_list.append(buffer_points)
        all_weights_list.append(np.ones(buffer_size, dtype=np.float32))
        
        # Combine
        if len(all_points_list) > 1:
            all_points = np.vstack(all_points_list)
            all_weights = np.concatenate(all_weights_list)
        else:
            all_points = all_points_list[0]
            all_weights = all_weights_list[0]
        
        # Assign all points to new centers and compute weighted counts
        labels = self._kmeans.predict(all_points)
        self.counts = np.bincount(
            labels,
            weights=all_weights,
            minlength=len(new_centers)
        ).astype(np.float32)
        
        # Update centers
        self.centers = new_centers
        
        # If we have more centers than K_max, collapse further
        if len(self.centers) > self.K_max:
            self._collapse_to_kmax()
        
        # Clear buffer
        self.buffer = []
    
    def _collapse_to_kmax(self):
        """
        Collapse centers to exactly K_max using a final k-means step.
        Called when we have more than K_max centers.
        """
        if len(self.centers) <= self.K_max:
            return
        
        # Use existing centers weighted by counts to reduce to K_max
        # Replicate centers based on counts for weighted k-means
        max_reps = 50
        replicated_points = []
        replicated_weights = []
        
        for center, count in zip(self.centers, self.counts):
            n_reps = min(max_reps, max(1, int(count)))
            replicated_points.append(np.tile(center[None, :], (n_reps, 1)))
            replicated_weights.extend([count / n_reps] * n_reps)
        
        all_points = np.vstack(replicated_points)
        all_weights = np.array(replicated_weights, dtype=np.float32)
        
        # Final k-means to reduce to K_max
        kmeans_final = MiniBatchKMeans(
            n_clusters=self.K_max,
            random_state=self.random_state,
            batch_size=min(2048, len(all_points) // 4),
            max_iter=100,
            n_init=3,
            reassignment_ratio=0.01,
            verbose=0,
        )
        
        # Warm-start with top K_max centers by count
        top_indices = np.argsort(self.counts)[-self.K_max:]
        kmeans_final.cluster_centers_ = self.centers[top_indices].astype(np.float32)
        
        # Fit on all replicated points
        labels = kmeans_final.fit_predict(all_points)
        
        # Update centers and counts
        self.centers = kmeans_final.cluster_centers_.astype(np.float32)
        self.counts = np.bincount(
            labels,
            weights=all_weights,
            minlength=self.K_max
        ).astype(np.float32)
        
        # Reset kmeans for next incremental update
        self._kmeans = None
        self._kmeans_initialized = False
    
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
