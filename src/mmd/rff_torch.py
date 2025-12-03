# rff_torch.py
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class RFFConfigTorch:
    input_dim: int                # D
    sigmas: List[float]           # one or more bandwidths
    features_per_sigma: int       # M per sigma
    seed: int = 0                 # for reproducibility
    device: Optional[str] = None  # 'cuda', 'cpu', etc.


class RFFMapTorch(nn.Module):
    """
    Random Fourier Features for (possibly multi-scale) RBF kernels (PyTorch version).
    You initialize this ONCE, then reuse across all datasets.
    Supports GPU acceleration.
    """

    def __init__(self, config: RFFConfigTorch):
        super().__init__()
        self.config = config
        
        # Set device
        if config.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)
        
        # Use generator for seed control
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(config.seed)

        self.params: List[Tuple[torch.Tensor, torch.Tensor, float]] = []
        self.total_features = 0

        for sigma in config.sigmas:
            omega, b, scale = self._init_single_rff(
                D=config.input_dim,
                M=config.features_per_sigma,
                sigma=sigma
            )
            self.params.append((omega, b, scale))
            self.total_features += config.features_per_sigma

    def _init_single_rff(self, D: int, M: int, sigma: float):
        # omega_j ~ N(0, 1/sigma^2 I_D)
        omega = torch.randn(
            D, M,
            generator=self.generator,
            dtype=torch.float32,
            device=self.device
        ) / sigma
        
        # b_j ~ Uniform[0, 2π]
        b = torch.rand(
            M,
            generator=self.generator,
            dtype=torch.float32,
            device=self.device
        ) * 2 * np.pi
        
        scale = np.sqrt(2.0 / M)
        
        # Register as buffers so they're moved with the model
        omega = nn.Parameter(omega, requires_grad=False)
        b = nn.Parameter(b, requires_grad=False)
        
        return omega, b, scale

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [B, D]
        Returns phi: [B, total_features]
        """
        B, D = X.shape
        assert D == self.config.input_dim

        # Ensure X is on the correct device
        X = X.to(self.device)

        feats = []
        for (omega, b, scale) in self.params:
            proj = X @ omega          # [B, M]
            proj = proj + b           # broadcast [M]
            phi = scale * torch.cos(proj)
            feats.append(phi)

        return torch.cat(feats, dim=-1)   # [B, total_features]

    def save(self, path: str):
        """Save ω, b, and config so you can reload exactly the same map."""
        save_dict = {
            'sigmas': np.array(self.config.sigmas),
            'features_per_sigma': self.config.features_per_sigma,
            'input_dim': self.config.input_dim,
            'seed': self.config.seed,
            'device': str(self.device),
            'params_omega': [p[0].detach().cpu().numpy() for p in self.params],
            'params_b': [p[1].detach().cpu().numpy() for p in self.params],
            'params_scale': [p[2] for p in self.params],
        }
        torch.save(save_dict, path)

    @staticmethod
    def load(path: str) -> "RFFMapTorch":
        data = torch.load(path, map_location='cpu')
        cfg = RFFConfigTorch(
            input_dim=int(data["input_dim"]),
            sigmas=list(data["sigmas"]),
            features_per_sigma=int(data["features_per_sigma"]),
            seed=int(data["seed"]),
            device=data.get("device", None),
        )
        obj = RFFMapTorch(cfg)
        # Overwrite with stored params (in case you want exact reproducibility)
        omegas = [torch.from_numpy(o).to(obj.device) for o in data["params_omega"]]
        bs = [torch.from_numpy(b).to(obj.device) for b in data["params_b"]]
        scales = data["params_scale"]
        
        obj.params = [(nn.Parameter(o, requires_grad=False), 
                       nn.Parameter(b, requires_grad=False), 
                       s) for o, b, s in zip(omegas, bs, scales)]
        obj.total_features = sum(o.shape[1] for o in omegas)
        return obj

