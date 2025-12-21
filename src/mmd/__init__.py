# __init__.py
"""
MMD (Maximum Mean Discrepancy) library with RBF kernels and Random Fourier Features.

Supports both NumPy and PyTorch backends with GPU acceleration.
"""

# NumPy backend
from .rbf import RBFKernel
from .rff import RFFConfig, RFFMap
from .mmd import StreamingMMD

# PyTorch backend
from .rbf_torch import RBFKernelTorch
from .rff_torch import RFFConfigTorch, RFFMapTorch
from .mmd_torch import StreamingMMDTorch

# Config system
from .config import MMDConfig, load_config_from_yaml, save_config_to_yaml

# Functional APIs
from .functional import (
    rbf_kernel,
    mmd2_rbf,
    mmd2_rff,
    mmd2_rff_from_features,
)

# Validation utilities
from .validation import (
    validate_mmd_zero,
    compare_exact_vs_rff,
    test_permutation_invariance,
    test_sigma_sensitivity,
)

# Feature encoders
from .encoders import BaseFeatureEncoder, ResNet101Encoder, DinoV3Encoder

__all__ = [
    # NumPy classes
    'RBFKernel',
    'RFFConfig',
    'RFFMap',
    'StreamingMMD',
    # PyTorch classes
    'RBFKernelTorch',
    'RFFConfigTorch',
    'RFFMapTorch',
    'StreamingMMDTorch',
    # Config
    'MMDConfig',
    'load_config_from_yaml',
    'save_config_to_yaml',
    # Functional APIs
    'rbf_kernel',
    'mmd2_rbf',
    'mmd2_rff',
    'mmd2_rff_from_features',
    # Validation
    'validate_mmd_zero',
    'compare_exact_vs_rff',
    'test_permutation_invariance',
    'test_sigma_sensitivity',
    # Feature encoders
    'BaseFeatureEncoder',
    'ResNet101Encoder',
    'DinoV3Encoder',
]
