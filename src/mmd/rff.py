# rff.py
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import MMDConfig

@dataclass
class RFFConfig:
    input_dim: int                # D
    sigmas: List[float]           # one or more bandwidths
    features_per_sigma: int       # M per sigma
    seed: int = 0                 # for reproducibility


class RFFMap:
    """
    Random Fourier Features for (possibly multi-scale) RBF kernels.
    You initialize this ONCE, then reuse across all datasets.
    """

    def __init__(self, config: Union[RFFConfig, 'MMDConfig']):
        # Handle MMDConfig by converting to RFFConfig
        if hasattr(config, 'backend') and config.backend == 'numpy':
            # It's an MMDConfig, convert it
            from .config import MMDConfig as MMDConfigType
            if isinstance(config, MMDConfigType):
                config = config.to_rff_config()
        
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        self.params: List[Tuple[np.ndarray, np.ndarray, float]] = []
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
        omega = self.rng.normal(
            loc=0.0,
            scale=1.0 / sigma,
            size=(D, M)
        )
        # b_j ~ Uniform[0, 2π]
        b = self.rng.uniform(
            low=0.0,
            high=2 * np.pi,
            size=(M,)
        )
        scale = np.sqrt(2.0 / M)
        return omega, b, scale

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        X: [B, D]
        Returns phi: [B, total_features]
        """
        B, D = X.shape
        assert D == self.config.input_dim

        feats = []
        for (omega, b, scale) in self.params:
            proj = X @ omega          # [B, M]
            proj += b                 # broadcast [M]
            phi = scale * np.cos(proj)
            feats.append(phi)

        return np.concatenate(feats, axis=-1)   # [B, total_features]

    def save(self, path: str):
        """Save ω, b, and config so you can reload exactly the same map."""
        np.savez_compressed(
            path,
            sigmas=np.array(self.config.sigmas),
            features_per_sigma=self.config.features_per_sigma,
            input_dim=self.config.input_dim,
            seed=self.config.seed,
            params_omega=[p[0] for p in self.params],
            params_b=[p[1] for p in self.params],
            params_scale=[p[2] for p in self.params],
        )

    @staticmethod
    def load(path: str) -> "RFFMap":
        data = np.load(path, allow_pickle=True)
        cfg = RFFConfig(
            input_dim=int(data["input_dim"]),
            sigmas=list(data["sigmas"]),
            features_per_sigma=int(data["features_per_sigma"]),
            seed=int(data["seed"]),
        )
        obj = RFFMap(cfg)
        # overwrite with stored params (in case you want exact reproducibility)
        omegas = list(data["params_omega"])
        bs = list(data["params_b"])
        scales = list(data["params_scale"])

        obj.params = [(o, b, s) for o, b, s in zip(omegas, bs, scales)]
        obj.total_features = sum(o.shape[1] for o in omegas)
        return obj
