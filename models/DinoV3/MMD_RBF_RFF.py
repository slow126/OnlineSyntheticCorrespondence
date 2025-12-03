"""
Maximum Mean Discrepancy (MMD) with RBF kernel using Random Fourier Features (RFF).

MMD is a metric for comparing two probability distributions. This implementation
uses Random Fourier Features to approximate the RBF kernel, making it computationally
efficient for large-scale comparisons.

References:
    - Gretton, A., et al. "A kernel two-sample test." JMLR 2012.
    - Rahimi, A., & Recht, B. "Random features for large-scale kernel machines." NIPS 2007.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class MMD_RBF_RFF(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) with RBF kernel using Random Fourier Features.
    
    This class computes MMD between two samples using an RBF kernel approximated
    via Random Fourier Features, which provides O(n) complexity instead of O(n²).
    
    Args:
        sigma (float): Bandwidth parameter for the RBF kernel. Default: 1.0
        n_features (int): Number of random Fourier features. More features = better
            approximation but slower computation. Default: 1000
        device (str, optional): Device to use ('cuda' or 'cpu'). If None, uses
            the device of the input tensors.
        seed (int, optional): Random seed for generating RFF projections. If None,
            uses random initialization.
    """
    
    def __init__(
        self,
        sigma: float = 1.0,
        n_features: int = 1000,
        device: Optional[str] = None,
        seed: Optional[int] = None
    ):
        super().__init__()
        self.sigma = sigma
        self.n_features = n_features
        self.device = device
        self.seed = seed
        
        # Will be initialized on first forward pass when we know the input dimension
        self.register_buffer('W', None)  # Random projection matrix
        self.register_buffer('b', None)  # Random bias terms
        self._initialized = False
    
    def _initialize_rff(self, input_dim: int):
        """
        Initialize Random Fourier Features for RBF kernel approximation.
        
        For RBF kernel k(x, y) = exp(-||x-y||² / (2σ²)), we use the RFF approximation:
        k(x, y) ≈ φ(x)ᵀ φ(y) where φ(x) = sqrt(2/n_features) * cos(Wx + b)
        
        Args:
            input_dim: Dimensionality of input features
        """
        if self._initialized and self.W is not None and self.W.shape[0] == input_dim:
            return
        
        # Set random seed if provided
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        
        # Sample from the Fourier transform of the RBF kernel
        # For RBF kernel, the spectral density is N(0, 1/σ² I)
        # So we sample W ~ N(0, 1/σ² I)
        W = torch.randn(self.n_features, input_dim, dtype=torch.float32)
        W = W / self.sigma  # Scale by 1/sigma
        
        # Sample uniform biases b ~ Uniform(0, 2π)
        b = torch.rand(self.n_features, dtype=torch.float32) * 2 * np.pi
        
        # Move to device if specified
        if self.device is not None:
            W = W.to(self.device)
            b = b.to(self.device)
        
        # Register as buffers so they're moved with the model
        self.register_buffer('W', W)
        self.register_buffer('b', b)
        self._initialized = True
    
    def _rff_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply Random Fourier Features transformation.
        
        Args:
            X: Input tensor of shape (n_samples, n_features)
            
        Returns:
            Transformed features of shape (n_samples, n_rff_features)
        """
        if not self._initialized:
            self._initialize_rff(X.shape[-1])
        
        # Ensure W and X are on the same device
        if X.device != self.W.device:
            X = X.to(self.W.device)
        
        # Compute W @ X^T and add bias: (n_rff, n_samples)
        projections = torch.matmul(self.W, X.t()) + self.b.unsqueeze(1)
        
        # Apply cosine: (n_rff, n_samples)
        cos_proj = torch.cos(projections)
        
        # Scale by sqrt(2/n_features) and transpose: (n_samples, n_rff)
        scale = np.sqrt(2.0 / self.n_features)
        phi_X = scale * cos_proj.t()
        
        return phi_X
    
    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute MMD between two samples X and Y.
        
        MMD²(X, Y) = ||μ_X - μ_Y||²
                   = <μ_X, μ_X> + <μ_Y, μ_Y> - 2<μ_X, μ_Y>
        
        where μ_X and μ_Y are the mean embeddings in the RFF space.
        
        Args:
            X: First sample tensor of shape (n_samples_X, n_features)
            Y: Second sample tensor of shape (n_samples_Y, n_features)
            return_components: If True, return individual components of MMD.
                Returns (mmd_squared, XX_term, YY_term, XY_term)
                
        Returns:
            MMD² value (scalar tensor). If return_components=True, returns tuple.
        """
        # Ensure inputs are 2D
        if X.dim() != 2:
            raise ValueError(f"X must be 2D tensor, got shape {X.shape}")
        if Y.dim() != 2:
            raise ValueError(f"Y must be 2D tensor, got shape {Y.shape}")
        
        # Ensure same feature dimension
        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                f"X and Y must have same feature dimension. "
                f"Got X: {X.shape[1]}, Y: {Y.shape[1]}"
            )
        
        # Transform to RFF space
        phi_X = self._rff_transform(X)  # (n_X, n_rff)
        phi_Y = self._rff_transform(Y)  # (n_Y, n_rff)
        
        # Compute mean embeddings
        mu_X = phi_X.mean(dim=0)  # (n_rff,)
        mu_Y = phi_Y.mean(dim=0)  # (n_rff,)
        
        # Compute MMD² = ||μ_X - μ_Y||²
        mmd_squared = torch.sum((mu_X - mu_Y) ** 2)
        
        if return_components:
            # Compute individual terms for analysis
            XX_term = torch.sum(mu_X ** 2)
            YY_term = torch.sum(mu_Y ** 2)
            XY_term = -2 * torch.sum(mu_X * mu_Y)
            return mmd_squared, XX_term, YY_term, XY_term
        
        return mmd_squared
    
    def compute_mmd_unbiased(
        self,
        X: torch.Tensor,
        Y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute unbiased MMD estimator using U-statistics.
        
        This version uses all pairs within and between samples, providing
        an unbiased estimate of MMD².
        
        MMD²_unbiased = (1/(n(n-1))) Σᵢ≠ⱼ k(xᵢ, xⱼ) 
                      + (1/(m(m-1))) Σᵢ≠ⱼ k(yᵢ, yⱼ)
                      - (2/(nm)) Σᵢ,ⱼ k(xᵢ, yⱼ)
        
        Args:
            X: First sample tensor of shape (n_samples_X, n_features)
            Y: Second sample tensor of shape (n_samples_Y, n_features)
            
        Returns:
            Unbiased MMD² estimate (scalar tensor)
        """
        # Ensure inputs are 2D
        if X.dim() != 2:
            raise ValueError(f"X must be 2D tensor, got shape {X.shape}")
        if Y.dim() != 2:
            raise ValueError(f"Y must be 2D tensor, got shape {Y.shape}")
        
        # Ensure same feature dimension
        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                f"X and Y must have same feature dimension. "
                f"Got X: {X.shape[1]}, Y: {Y.shape[1]}"
            )
        
        # Transform to RFF space
        phi_X = self._rff_transform(X)  # (n_X, n_rff)
        phi_Y = self._rff_transform(Y)  # (n_Y, n_rff)
        
        n_X = X.shape[0]
        n_Y = Y.shape[0]
        
        # Compute kernel matrices using RFF approximation
        # k(x_i, x_j) ≈ φ(x_i)^T φ(x_j)
        K_XX = torch.matmul(phi_X, phi_X.t())  # (n_X, n_X)
        K_YY = torch.matmul(phi_Y, phi_Y.t())  # (n_Y, n_Y)
        K_XY = torch.matmul(phi_X, phi_Y.t())  # (n_X, n_Y)
        
        # Remove diagonal (self-similarity) for within-sample terms
        # For XX: sum all pairs except diagonal
        XX_sum = K_XX.sum() - torch.trace(K_XX)
        XX_term = XX_sum / (n_X * (n_X - 1)) if n_X > 1 else torch.tensor(0.0, device=X.device)
        
        # For YY: sum all pairs except diagonal
        YY_sum = K_YY.sum() - torch.trace(K_YY)
        YY_term = YY_sum / (n_Y * (n_Y - 1)) if n_Y > 1 else torch.tensor(0.0, device=Y.device)
        
        # Cross term: all pairs between X and Y
        XY_term = -2 * K_XY.mean()
        
        mmd_squared = XX_term + YY_term + XY_term
        return mmd_squared
    
    def compute_mmd_linear(
        self,
        X: torch.Tensor,
        Y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute linear-time MMD estimator (same as forward method).
        
        This is the most efficient version, using O(n) computation.
        
        Args:
            X: First sample tensor of shape (n_samples_X, n_features)
            Y: Second sample tensor of shape (n_samples_Y, n_features)
            
        Returns:
            MMD² value (scalar tensor)
        """
        return self.forward(X, Y)


# Example usage and testing
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create MMD instance
    mmd = MMD_RBF_RFF(sigma=1.0, n_features=1000, seed=42)
    
    # Example 1: Compare two similar distributions
    print("Example 1: Similar distributions")
    X1 = torch.randn(100, 64)  # 100 samples, 64 features
    Y1 = torch.randn(100, 64) + 0.1  # Slightly shifted
    mmd_value1 = mmd(X1, Y1)
    print(f"MMD² between similar distributions: {mmd_value1.item():.4f}")
    
    # Example 2: Compare two different distributions
    print("\nExample 2: Different distributions")
    X2 = torch.randn(100, 64)
    Y2 = torch.randn(100, 64) + 5.0  # Very different
    mmd_value2 = mmd(X2, Y2)
    print(f"MMD² between different distributions: {mmd_value2.item():.4f}")
    
    # Example 3: Unbiased estimator
    print("\nExample 3: Unbiased MMD estimator")
    mmd_unbiased = mmd.compute_mmd_unbiased(X1, Y1)
    print(f"Unbiased MMD²: {mmd_unbiased.item():.4f}")
    
    # Example 4: With components
    print("\nExample 4: MMD with components")
    mmd_val, XX, YY, XY = mmd(X1, Y1, return_components=True)
    print(f"MMD²: {mmd_val.item():.4f}")
    print(f"  XX term: {XX.item():.4f}")
    print(f"  YY term: {YY.item():.4f}")
    print(f"  XY term: {XY.item():.4f}")
    print(f"  Sum check: {(XX + YY + XY).item():.4f}")

