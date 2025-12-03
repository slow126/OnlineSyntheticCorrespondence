# functional.py
import numpy as np
import torch
from typing import Union, List
from .rbf import RBFKernel
from .rbf_torch import RBFKernelTorch
from .rff import RFFMap
from .rff_torch import RFFMapTorch

def rbf_kernel(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    sigmas: Union[float, List[float]],
    backend: str = 'numpy'
) -> Union[np.ndarray, torch.Tensor]:
    """
    Unified interface for RBF kernel computation.
    
    Args:
        X: [m, d] first set of samples
        Y: [n, d] second set of samples
        sigmas: Single sigma or list of sigmas for multi-scale kernel
        backend: 'numpy' or 'torch'
    
    Returns:
        Kernel matrix K: [m, n]
    """
    if backend == 'numpy':
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.detach().cpu().numpy()
        kernel = RBFKernel(sigmas)
        return kernel(X, Y)
    elif backend == 'torch':
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y).float()
        # Determine device from input tensors
        device = X.device if isinstance(X, torch.Tensor) else 'cpu'
        kernel = RBFKernelTorch(sigmas, device=device)
        return kernel(X, Y)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def mmd2_rbf(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    sigmas: Union[float, List[float]],
    unbiased: bool = False,
    backend: str = 'numpy'
) -> float:
    """
    Exact MMD^2 using RBF kernel.
    
    Args:
        X: [m, d] samples from first distribution
        Y: [n, d] samples from second distribution
        sigmas: Single sigma or list of sigmas for multi-scale kernel
        unbiased: If True, use unbiased estimator (exclude diagonal terms)
        backend: 'numpy' or 'torch'
    
    Returns:
        MMD^2 value (scalar)
    """
    if backend == 'numpy':
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.detach().cpu().numpy()
        kernel = RBFKernel(sigmas)
        return kernel.mmd2_naive(X, Y, unbiased=unbiased)
    elif backend == 'torch':
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y).float()
        # Determine device from input tensors
        device = X.device if isinstance(X, torch.Tensor) else 'cpu'
        kernel = RBFKernelTorch(sigmas, device=device)
        return kernel.mmd2_naive(X, Y, unbiased=unbiased)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def mmd2_rff_from_features(
    Z_X: Union[np.ndarray, torch.Tensor],
    Z_Y: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute MMD^2 from pre-computed RFF features.
    
    Args:
        Z_X: [m, d_rff] RFF features for first distribution
        Z_Y: [n, d_rff] RFF features for second distribution
    
    Returns:
        MMD^2 value (scalar)
    """
    if isinstance(Z_X, torch.Tensor) and isinstance(Z_Y, torch.Tensor):
        mu_X = Z_X.mean(dim=0)
        mu_Y = Z_Y.mean(dim=0)
        diff = mu_X - mu_Y
        return float((diff ** 2).sum().item())
    else:
        # NumPy
        if isinstance(Z_X, torch.Tensor):
            Z_X = Z_X.detach().cpu().numpy()
        if isinstance(Z_Y, torch.Tensor):
            Z_Y = Z_Y.detach().cpu().numpy()
        mu_X = Z_X.mean(axis=0)
        mu_Y = Z_Y.mean(axis=0)
        diff = mu_X - mu_Y
        return float(np.sum(diff ** 2))


def mmd2_rff(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    rff_map: Union[RFFMap, RFFMapTorch]
) -> float:
    """
    One-shot RFF-based MMD^2 for two batches.
    
    Args:
        X: [m, d] samples from first distribution
        Y: [n, d] samples from second distribution
        rff_map: RFFMap or RFFMapTorch instance
    
    Returns:
        MMD^2 value (scalar)
    """
    if isinstance(rff_map, RFFMapTorch):
        # PyTorch path
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(rff_map.device)
        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y).float().to(rff_map.device)
        phi_X = rff_map.transform(X)
        phi_Y = rff_map.transform(Y)
    else:
        # NumPy path
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.detach().cpu().numpy()
        phi_X = rff_map.transform(X)
        phi_Y = rff_map.transform(Y)
    
    return mmd2_rff_from_features(phi_X, phi_Y)

