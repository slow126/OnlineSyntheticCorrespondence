# rbf.py
import numpy as np
from typing import Union, List

class RBFKernel:
    """
    Plain RBF kernel. Mainly useful for:
    - sanity checks on small subsets (exact MMD),
    - experimenting with sigma choices.
    
    Supports single or multiple sigmas (multi-scale kernel).
    For multiple sigmas, kernels are summed with equal weights.
    """

    def __init__(self, sigmas: Union[float, List[float]]):
        """
        Args:
            sigmas: Single sigma (float) or list of sigmas for multi-scale kernel
        """
        if isinstance(sigmas, (int, float)):
            sigmas = [float(sigmas)]
        else:
            sigmas = [float(s) for s in sigmas]
        self.sigmas = sigmas

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        X: [m, d], Y: [n, d]
        returns K: [m, n] with K[i,j] = k(X[i], Y[j])
        
        For multi-sigma, returns sum of kernels: k(x,y) = sum_m k_sigma_m(x,y)
        """
        m, d = X.shape
        n, d2 = Y.shape
        assert d == d2

        # pairwise squared distances
        X_sq = np.sum(X**2, axis=1, keepdims=True)    # [m, 1]
        Y_sq = np.sum(Y**2, axis=1, keepdims=True)    # [n, 1]
        dist2 = X_sq + Y_sq.T - 2 * X @ Y.T           # [m, n]

        # Sum kernels for all sigmas (equal weights)
        K = np.zeros((m, n), dtype=np.float64)
        for sigma in self.sigmas:
            K += np.exp(-dist2 / (2 * sigma**2))
        
        return K

    def mmd2_naive(self, X: np.ndarray, Y: np.ndarray, unbiased: bool = False) -> float:
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
                K_xx_no_diag = (K_xx.sum() - np.trace(K_xx)) / (m * (m - 1))
            else:
                K_xx_no_diag = 0.0
            
            if n > 1:
                K_yy_no_diag = (K_yy.sum() - np.trace(K_yy)) / (n * (n - 1))
            else:
                K_yy_no_diag = 0.0
            
            term_xy = K_xy.mean()
            return float(K_xx_no_diag + K_yy_no_diag - 2 * term_xy)
        else:
            # Biased estimator (original)
            term_xx = K_xx.sum() / (m * m)
            term_yy = K_yy.sum() / (n * n)
            term_xy = K_xy.sum() / (m * n)
            return float(term_xx + term_yy - 2 * term_xy)
