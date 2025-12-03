# validation.py
import numpy as np
import torch
from typing import Union, List, Tuple
from .rbf import RBFKernel
from .rbf_torch import RBFKernelTorch
from .rff import RFFMap
from .rff_torch import RFFMapTorch
from .functional import mmd2_rbf, mmd2_rff

def validate_mmd_zero(
    X: Union[np.ndarray, torch.Tensor],
    sigmas: Union[float, List[float]] = 1.0,
    backend: str = 'numpy',
    tolerance: float = 1e-6
) -> Tuple[bool, float]:
    """
    Check that MMD(X, X) ≈ 0 (should be very small for identical distributions).
    
    Args:
        X: [n, d] samples
        sigmas: Single sigma or list of sigmas
        backend: 'numpy' or 'torch'
        tolerance: Maximum allowed MMD value
    
    Returns:
        (is_valid, mmd_value) tuple
    """
    mmd_val = mmd2_rbf(X, X, sigmas, backend=backend)
    is_valid = mmd_val < tolerance
    return is_valid, mmd_val


def compare_exact_vs_rff(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    sigmas: Union[float, List[float]] = 1.0,
    n_features: int = 1024,
    backend: str = 'numpy',
    tolerance: float = 0.1
) -> Tuple[bool, float, float]:
    """
    Consistency test: compare exact RBF MMD vs RFF approximation.
    
    Args:
        X: [m, d] samples from first distribution
        Y: [n, d] samples from second distribution
        sigmas: Single sigma or list of sigmas
        n_features: Number of RFF features to use
        backend: 'numpy' or 'torch'
        tolerance: Maximum allowed relative difference
    
    Returns:
        (is_valid, exact_mmd, rff_mmd) tuple
    """
    # Compute exact MMD
    exact_mmd = mmd2_rbf(X, Y, sigmas, backend=backend)
    
    # Compute RFF approximation
    if backend == 'numpy':
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.detach().cpu().numpy()
        d = X.shape[1]
        sigmas_list = [sigmas] if isinstance(sigmas, (int, float)) else sigmas
        from .rff import RFFConfig
        rff_config = RFFConfig(
            input_dim=d,
            sigmas=sigmas_list,
            features_per_sigma=n_features // len(sigmas_list),
            seed=0
        )
        rff_map = RFFMap(rff_config)
    else:
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y).float()
        device = X.device if isinstance(X, torch.Tensor) else 'cpu'
        d = X.shape[1]
        sigmas_list = [sigmas] if isinstance(sigmas, (int, float)) else sigmas
        from .rff_torch import RFFConfigTorch
        rff_config = RFFConfigTorch(
            input_dim=d,
            sigmas=sigmas_list,
            features_per_sigma=n_features // len(sigmas_list),
            seed=0,
            device=device
        )
        rff_map = RFFMapTorch(rff_config)
    
    rff_mmd = mmd2_rff(X, Y, rff_map)
    
    # Check relative difference
    if exact_mmd > 0:
        rel_diff = abs(exact_mmd - rff_mmd) / exact_mmd
    else:
        rel_diff = abs(exact_mmd - rff_mmd)
    
    is_valid = rel_diff < tolerance
    return is_valid, exact_mmd, rff_mmd


def test_permutation_invariance(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    sigmas: Union[float, List[float]] = 1.0,
    backend: str = 'numpy',
    tolerance: float = 1e-10
) -> Tuple[bool, float, float]:
    """
    Test that MMD is invariant to row permutations.
    
    Args:
        X: [m, d] samples from first distribution
        Y: [n, d] samples from second distribution
        sigmas: Single sigma or list of sigmas
        backend: 'numpy' or 'torch'
        tolerance: Maximum allowed difference between permuted and original
    
    Returns:
        (is_valid, original_mmd, permuted_mmd) tuple
    """
    original_mmd = mmd2_rbf(X, Y, sigmas, backend=backend)
    
    # Permute rows
    if backend == 'numpy':
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.detach().cpu().numpy()
        perm_X = np.random.permutation(X)
        perm_Y = np.random.permutation(Y)
    else:
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y).float()
        perm_X = X[torch.randperm(X.shape[0])]
        perm_Y = Y[torch.randperm(Y.shape[0])]
    
    permuted_mmd = mmd2_rbf(perm_X, perm_Y, sigmas, backend=backend)
    
    diff = abs(original_mmd - permuted_mmd)
    is_valid = diff < tolerance
    return is_valid, original_mmd, permuted_mmd


def test_sigma_sensitivity(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    sigma_range: List[float],
    backend: str = 'numpy'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Test how MMD behaves as sigma varies.
    
    Args:
        X: [m, d] samples from first distribution
        Y: [n, d] samples from second distribution
        sigma_range: List of sigma values to test
        backend: 'numpy' or 'torch'
    
    Returns:
        (sigmas, mmd_values) tuple
    """
    mmd_values = []
    for sigma in sigma_range:
        mmd_val = mmd2_rbf(X, Y, sigma, backend=backend)
        mmd_values.append(mmd_val)
    
    if backend == 'torch' and isinstance(mmd_values[0], torch.Tensor):
        mmd_values = [float(v.item()) if isinstance(v, torch.Tensor) else v for v in mmd_values]
    
    return np.array(sigma_range), np.array(mmd_values)

