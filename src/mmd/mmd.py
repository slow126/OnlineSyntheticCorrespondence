# mmd.py
import numpy as np
from typing import Dict, Union, Optional, TYPE_CHECKING
from .rff import RFFMap

if TYPE_CHECKING:
    from .config import MMDConfig

class StreamingMMD:
    """
    Maintains streaming mean RFF embeddings per dataset and
    provides (approximate) MMD^2 distances.

    You:
    - construct with an RFFMap (shared across ALL datasets) or MMDConfig
    - call update(dataset_id, batch_embeddings)
    - later call mmd2(dataset_a, dataset_b)
    """

    def __init__(self, rff_map: Optional[RFFMap] = None, config: Optional['MMDConfig'] = None):
        if rff_map is None and config is None:
            raise ValueError("Must provide either rff_map or config")
        if rff_map is not None and config is not None:
            raise ValueError("Cannot provide both rff_map and config")
        
        if config is not None:
            from .config import MMDConfig as MMDConfigType
            if config.backend != 'numpy':
                raise ValueError(f"StreamingMMD only supports numpy backend, got {config.backend}")
            rff_map = config.create_rff_map()
        
        self.rff = rff_map
        self.state: Dict[str, Dict[str, np.ndarray]] = {}
        # state[dataset_id] = {"mu": [M], "count": scalar}

    def _ensure_dataset(self, dataset_id: str):
        if dataset_id not in self.state:
            self.state[dataset_id] = {
                "mu": np.zeros(self.rff.total_features, dtype=np.float64),
                "count": 0.0,
            }

    def update(self, dataset_id: str, X_batch: np.ndarray):
        """
        X_batch: [B, D] raw embeddings (e.g., per-patch DINO features or PCA-reduced).
        This can be called in a streaming fashion over 100s of millions of embeddings.
        """
        self._ensure_dataset(dataset_id)
        s = self.state[dataset_id]
        mu = s["mu"]
        count = s["count"]

        # Transform to RFF space
        phi = self.rff.transform(X_batch)   # [B, M]
        batch_size = phi.shape[0]

        # Batch mean in RFF space
        batch_mean = phi.mean(axis=0)       # [M]

        # Update global mean (online)
        # New global count
        new_count = count + batch_size
        # Weighted average of old mean and batch mean
        mu *= count / new_count
        mu += batch_mean * (batch_size / new_count)

        # Store back
        s["mu"] = mu
        s["count"] = new_count

    def get_mu(self, dataset_id: str) -> np.ndarray:
        self._ensure_dataset(dataset_id)
        return self.state[dataset_id]["mu"]

    def mmd2(self, dataset_a: str, dataset_b: str) -> float:
        """
        Approximate MMD^2 via ||mu_A - mu_B||^2 in RFF space.
        """
        mu_a = self.get_mu(dataset_a)
        mu_b = self.get_mu(dataset_b)
        diff = mu_a - mu_b
        return float(diff @ diff)

    def mmd(self, dataset_a: str, dataset_b: str) -> float:
        return self.mmd2(dataset_a, dataset_b) ** 0.5

    def save_state(self, path: str):
        """
        Save per-dataset means & counts so you can resume or
        compute more pairwise distances later.
        """
        np.savez_compressed(
            path,
            dataset_ids=np.array(list(self.state.keys())),
            mus=np.stack([self.state[k]["mu"] for k in self.state.keys()], axis=0),
            counts=np.array([self.state[k]["count"] for k in self.state.keys()]),
        )

    def load_state(self, path: str):
        data = np.load(path, allow_pickle=True)
        ids = [str(i) for i in data["dataset_ids"]]
        mus = data["mus"]
        counts = data["counts"]
        self.state = {}
        for i, ds_id in enumerate(ids):
            self.state[ds_id] = {"mu": mus[i], "count": float(counts[i])}
