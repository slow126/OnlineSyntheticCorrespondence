# rbf_torch.py
import torch
from typing import Union, List

class RBFKernelTorch:
    """
    PyTorch RBF kernel with GPU support.
    Supports single or multiple sigmas (multi-scale kernel).
    For multiple sigmas, kernels are summed with equal weights.
    """

    def __init__(self, sigmas: Union[float, List[float]], device: str = None):
        """
        Args:
            sigmas: Single sigma (float) or list of sigmas for multi-scale kernel
            device: Device to use ('cuda', 'cpu', etc.). If None, uses device of input tensors.
        """
        if isinstance(sigmas, (int, float)):
            sigmas = [float(sigmas)]
        else:
            sigmas = [float(s) for s in sigmas]
        self.sigmas = sigmas
        self.device = device

    def _get_device(self, tensor: torch.Tensor) -> torch.device:
        """Get device from tensor or use self.device"""
        if self.device is not None:
            return torch.device(self.device)
        return tensor.device

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        X: [m, d], Y: [n, d]
        returns K: [m, n] with K[i,j] = k(X[i], Y[j])
        
        For multi-sigma, returns sum of kernels: k(x,y) = sum_m k_sigma_m(x,y)
        """
        m, d = X.shape
        n, d2 = Y.shape
        assert d == d2

        device = self._get_device(X)
        
        # Ensure X and Y are on the same device
        X = X.to(device)
        Y = Y.to(device)

        # pairwise squared distances
        X_sq = torch.sum(X**2, dim=1, keepdim=True)    # [m, 1]
        Y_sq = torch.sum(Y**2, dim=1, keepdim=True)    # [n, 1]
        dist2 = X_sq + Y_sq.T - 2 * X @ Y.T           # [m, n]

        # Sum kernels for all sigmas (equal weights)
        K = torch.zeros((m, n), dtype=torch.float32, device=device)
        for sigma in self.sigmas:
            K += torch.exp(-dist2 / (2 * sigma**2))
        
        return K

    def mmd2_naive(self, X: torch.Tensor, Y: torch.Tensor, unbiased: bool = False) -> float:
        """
        Exact MMD^2 using this RBF kernel.
        ONLY for small subsets (O(n^2)).
        
        Args:
            X: [m, d] samples from first distribution
            Y: [n, d] samples from second distribution
            unbiased: If True, use unbiased estimator (exclude diagonal terms)
        
        Returns:
            MMD^2 value (scalar)
        """
        m, _ = X.shape
        n, _ = Y.shape

        K_xx = self.__call__(X, X)    # [m, m]
        K_yy = self.__call__(Y, Y)    # [n, n]
        K_xy = self.__call__(X, Y)    # [m, n]

        if unbiased:
            # Unbiased estimator: exclude diagonal terms
            if m > 1:
                K_xx_no_diag = (K_xx.sum() - torch.trace(K_xx)) / (m * (m - 1))
            else:
                K_xx_no_diag = torch.tensor(0.0, device=K_xx.device)
            
            if n > 1:
                K_yy_no_diag = (K_yy.sum() - torch.trace(K_yy)) / (n * (n - 1))
            else:
                K_yy_no_diag = torch.tensor(0.0, device=K_yy.device)
            
            term_xy = K_xy.mean()
            result = K_xx_no_diag + K_yy_no_diag - 2 * term_xy
            return float(result.item())
        else:
            # Biased estimator (original)
            term_xx = K_xx.sum() / (m * m)
            term_yy = K_yy.sum() / (n * n)
            term_xy = K_xy.sum() / (m * n)
            result = term_xx + term_yy - 2 * term_xy
            return float(result.item())

