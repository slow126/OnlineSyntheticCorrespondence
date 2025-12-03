# mmd_torch.py
import torch
from typing import Dict
from .rff_torch import RFFMapTorch

class StreamingMMDTorch:
    """
    Maintains streaming mean RFF embeddings per dataset and
    provides (approximate) MMD^2 distances (PyTorch version with GPU support).

    You:
    - construct with an RFFMapTorch (shared across ALL datasets)
    - call update(dataset_id, batch_embeddings)
    - later call mmd2(dataset_a, dataset_b)
    """

    def __init__(self, rff_map: RFFMapTorch):
        self.rff = rff_map
        self.device = rff_map.device
        self.state: Dict[str, Dict[str, torch.Tensor]] = {}
        # state[dataset_id] = {"mu": [M], "count": scalar}

    def _ensure_dataset(self, dataset_id: str):
        if dataset_id not in self.state:
            self.state[dataset_id] = {
                "mu": torch.zeros(self.rff.total_features, dtype=torch.float32, device=self.device),
                "count": torch.tensor(0.0, dtype=torch.float32, device=self.device),
            }

    def update(self, dataset_id: str, X_batch: torch.Tensor):
        """
        X_batch: [B, D] raw embeddings (e.g., per-patch DINO features or PCA-reduced).
        This can be called in a streaming fashion over 100s of millions of embeddings.
        """
        self._ensure_dataset(dataset_id)
        s = self.state[dataset_id]
        mu = s["mu"]
        count = s["count"]

        # Ensure X_batch is on the correct device
        X_batch = X_batch.to(self.device)

        # Transform to RFF space
        phi = self.rff.transform(X_batch)   # [B, M]
        batch_size = phi.shape[0]

        # Batch mean in RFF space
        batch_mean = phi.mean(dim=0)       # [M]

        # Update global mean (online)
        # New global count
        new_count = count + batch_size
        # Weighted average of old mean and batch mean
        mu = mu * (count / new_count) + batch_mean * (batch_size / new_count)

        # Store back
        s["mu"] = mu
        s["count"] = new_count

    def get_mu(self, dataset_id: str) -> torch.Tensor:
        self._ensure_dataset(dataset_id)
        return self.state[dataset_id]["mu"]

    def mmd2(self, dataset_a: str, dataset_b: str) -> float:
        """
        Approximate MMD^2 via ||mu_A - mu_B||^2 in RFF space.
        """
        mu_a = self.get_mu(dataset_a)
        mu_b = self.get_mu(dataset_b)
        diff = mu_a - mu_b
        return float((diff ** 2).sum().item())

    def mmd(self, dataset_a: str, dataset_b: str) -> float:
        return self.mmd2(dataset_a, dataset_b) ** 0.5

    def save_state(self, path: str):
        """
        Save per-dataset means & counts so you can resume or
        compute more pairwise distances later.
        """
        save_dict = {
            'dataset_ids': list(self.state.keys()),
            'mus': torch.stack([self.state[k]["mu"].cpu() for k in self.state.keys()], dim=0),
            'counts': torch.tensor([self.state[k]["count"].item() for k in self.state.keys()]),
        }
        torch.save(save_dict, path)

    def load_state(self, path: str):
        data = torch.load(path, map_location=self.device)
        ids = [str(i) for i in data["dataset_ids"]]
        mus = data["mus"].to(self.device)
        counts = data["counts"].to(self.device)
        self.state = {}
        for i, ds_id in enumerate(ids):
            self.state[ds_id] = {
                "mu": mus[i],
                "count": float(counts[i].item())
            }

