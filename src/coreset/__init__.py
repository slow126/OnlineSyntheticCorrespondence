"""
Weighted Coreset module for coverage and density metrics.

This module provides streaming weighted coresets for compressing large datasets
into representative clusters (centers + counts) and computing geometric coverage metrics.
"""

from .config import CoresetConfig, load_config_from_yaml, save_config_to_yaml
from .weighted_coreset import WeightedCoreset
from .metrics import (
    estimate_epsilon_from_eval,
    coverage_by_train,
    extraneous_mass_fraction,
    compute_nn_distances,
)

__all__ = [
    'CoresetConfig',
    'load_config_from_yaml',
    'save_config_to_yaml',
    'WeightedCoreset',
    'estimate_epsilon_from_eval',
    'coverage_by_train',
    'extraneous_mass_fraction',
    'compute_nn_distances',
]
