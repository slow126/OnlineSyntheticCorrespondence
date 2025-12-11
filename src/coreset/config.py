"""
Configuration dataclass for weighted coresets.

Similar to src/mmd/config.py but for coreset building and metrics.
"""

import yaml
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class CoresetConfig:
    """
    Configuration for weighted coreset construction.
    
    Attributes:
        K_max: Maximum number of centers to maintain
        K_overflow: Buffer size before triggering collapse
        distance: Distance metric ('euclidean', 'cosine', etc.)
        device: Device for computation ('cpu', 'cuda')
        is_eval: If True, compute epsilon scales on finalization
        epsilon_quantile: Quantile to use for epsilon estimation (default: 0.5 = median)
        max_epsilon_samples: Max samples for epsilon estimation
    """
    K_max: int
    K_overflow: int
    distance: str = 'euclidean'
    device: str = 'cpu'
    is_eval: bool = False
    epsilon_quantile: float = 0.5
    max_epsilon_samples: int = 50000


def load_config_from_yaml(path: str, preset: Optional[str] = None) -> CoresetConfig:
    """
    Load CoresetConfig from YAML file.
    
    Args:
        path: Path to YAML config file
        preset: Name of preset to load (e.g., 'flow_vectors', 'dino_features').
                If None, looks for 'coreset' key or uses root-level config.
    
    Returns:
        CoresetConfig instance
    
    Example:
        # Load default config
        config = load_config_from_yaml('configs/coreset_config.yaml')
        
        # Load specific preset
        config = load_config_from_yaml('configs/coreset_config.yaml', preset='flow_vectors')
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
    # Handle both direct config and nested 'coreset' key
    elif 'coreset' in data:
        data = data['coreset']
    
    return CoresetConfig(**data)


def save_config_to_yaml(config: CoresetConfig, path: str):
    """Save CoresetConfig to YAML file."""
    data = asdict(config)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
