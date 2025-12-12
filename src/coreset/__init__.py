"""
Weighted Coreset module for coverage and density metrics.

This module provides streaming weighted coresets for compressing large datasets
into representative clusters (centers + counts) and computing geometric coverage metrics.
"""

from .config import CoresetConfig, load_config_from_yaml, save_config_to_yaml
from .weighted_coreset import WeightedCoreset
from .metrics import (
    estimate_epsilon_from_eval,
    compute_nn_distances,
    DatasetCodebook,
    codebook_from_coreset,
    recall_train_covers_eval_soft,
    precision_train_wrt_eval_soft,
    outside_mass_fraction_soft,
)

__all__ = [
    'CoresetConfig',
    'load_config_from_yaml',
    'save_config_to_yaml',
    'WeightedCoreset',
    'estimate_epsilon_from_eval',
    'compute_nn_distances',
    'DatasetCodebook',
    'codebook_from_coreset',
    'recall_train_covers_eval_soft',
    'precision_train_wrt_eval_soft',
    'outside_mass_fraction_soft',
]
