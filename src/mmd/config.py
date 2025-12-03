# config.py
import yaml
from dataclasses import dataclass, asdict
from typing import List, Optional
from .rff import RFFConfig, RFFMap
from .rff_torch import RFFConfigTorch, RFFMapTorch

@dataclass
class MMDConfig:
    """
    Unified configuration for MMD computation with RBF kernels and RFF.
    """
    input_dim: int
    sigmas: List[float]
    features_per_sigma: int
    seed: int
    backend: str = 'numpy'  # 'numpy' or 'torch'
    device: Optional[str] = None  # For PyTorch: 'cuda', 'cpu', etc.
    unbiased: bool = False

    def to_rff_config(self):
        """Convert to RFFConfig for NumPy backend."""
        if self.backend != 'numpy':
            raise ValueError("to_rff_config() only works for numpy backend")
        return RFFConfig(
            input_dim=self.input_dim,
            sigmas=self.sigmas,
            features_per_sigma=self.features_per_sigma,
            seed=self.seed,
        )

    def to_rff_config_torch(self):
        """Convert to RFFConfigTorch for PyTorch backend."""
        if self.backend != 'torch':
            raise ValueError("to_rff_config_torch() only works for torch backend")
        return RFFConfigTorch(
            input_dim=self.input_dim,
            sigmas=self.sigmas,
            features_per_sigma=self.features_per_sigma,
            seed=self.seed,
            device=self.device,
        )

    def create_rff_map(self):
        """Factory function to create appropriate RFFMap from config."""
        if self.backend == 'numpy':
            return RFFMap(self.to_rff_config())
        elif self.backend == 'torch':
            return RFFMapTorch(self.to_rff_config_torch())
        else:
            raise ValueError(f"Unknown backend: {self.backend}")


def load_config_from_yaml(path: str, preset: str = None) -> MMDConfig:
    """
    Load MMDConfig from YAML file.
    
    Args:
        path: Path to YAML config file
        preset: Name of preset to load (e.g., 'flow_features', 'dino_features').
                If None, looks for 'mmd' key or uses root-level config.
    
    Returns:
        MMDConfig instance
    
    Example:
        # Load default config
        config = load_config_from_yaml('configs/mmd_config.yaml')
        
        # Load specific preset
        config = load_config_from_yaml('configs/mmd_config.yaml', preset='flow_features')
    """
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Handle preset selection
    if preset is not None:
        if preset not in data:
            available = list(data.keys())
            raise ValueError(
                f"Preset '{preset}' not found in config file. "
                f"Available presets: {available}"
            )
        data = data[preset]
    # Handle both direct config and nested 'mmd' key
    elif 'mmd' in data:
        data = data['mmd']
    
    return MMDConfig(**data)


def save_config_to_yaml(config: MMDConfig, path: str):
    """Save MMDConfig to YAML file."""
    data = asdict(config)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

