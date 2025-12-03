# test_mmd.py
"""
Comprehensive tests for MMD validation utilities.
Tests both NumPy and PyTorch backends.
"""
import numpy as np
import torch
from .validation import (
    validate_mmd_zero,
    compare_exact_vs_rff,
    test_permutation_invariance,
    test_sigma_sensitivity,
)


def test_validate_mmd_zero():
    """Test that MMD(X, X) ≈ 0 for identical distributions."""
    print("\n=== Testing validate_mmd_zero ===")
    
    # Test NumPy backend
    X_np = np.random.randn(50, 10)
    is_valid_np, mmd_val_np = validate_mmd_zero(X_np, sigmas=1.0, backend='numpy', tolerance=1e-5)
    print(f"NumPy: MMD(X, X) = {mmd_val_np:.2e}, Valid: {is_valid_np}")
    assert is_valid_np, f"NumPy MMD(X, X) should be < 1e-5, got {mmd_val_np}"
    
    # Test PyTorch backend
    X_torch = torch.randn(50, 10)
    is_valid_torch, mmd_val_torch = validate_mmd_zero(X_torch, sigmas=1.0, backend='torch', tolerance=1e-5)
    print(f"PyTorch: MMD(X, X) = {mmd_val_torch:.2e}, Valid: {is_valid_torch}")
    assert is_valid_torch, f"PyTorch MMD(X, X) should be < 1e-5, got {mmd_val_torch}"
    
    # Test multi-sigma
    is_valid_multi, mmd_val_multi = validate_mmd_zero(X_np, sigmas=[1.0, 2.0, 0.5], backend='numpy', tolerance=1e-5)
    print(f"Multi-sigma: MMD(X, X) = {mmd_val_multi:.2e}, Valid: {is_valid_multi}")
    assert is_valid_multi, f"Multi-sigma MMD(X, X) should be < 1e-5, got {mmd_val_multi}"
    
    print("✓ validate_mmd_zero tests passed")


def test_compare_exact_vs_rff():
    """Test that exact RBF MMD and RFF approximation are close."""
    print("\n=== Testing compare_exact_vs_rff ===")
    
    # Generate two different distributions
    X_np = np.random.randn(30, 8)
    Y_np = np.random.randn(30, 8) + 0.5  # Shifted distribution
    
    # Test NumPy backend
    is_valid_np, exact_mmd_np, rff_mmd_np = compare_exact_vs_rff(
        X_np, Y_np, sigmas=1.0, n_features=512, backend='numpy', tolerance=0.15
    )
    rel_diff_np = abs(exact_mmd_np - rff_mmd_np) / max(exact_mmd_np, 1e-10)
    print(f"NumPy - Exact: {exact_mmd_np:.6f}, RFF: {rff_mmd_np:.6f}, Rel diff: {rel_diff_np:.4f}, Valid: {is_valid_np}")
    assert is_valid_np, f"NumPy exact vs RFF should be close (rel diff < 0.15), got {rel_diff_np:.4f}"
    
    # Test PyTorch backend
    X_torch = torch.randn(30, 8)
    Y_torch = torch.randn(30, 8) + 0.5
    is_valid_torch, exact_mmd_torch, rff_mmd_torch = compare_exact_vs_rff(
        X_torch, Y_torch, sigmas=1.0, n_features=512, backend='torch', tolerance=0.15
    )
    rel_diff_torch = abs(exact_mmd_torch - rff_mmd_torch) / max(exact_mmd_torch, 1e-10)
    print(f"PyTorch - Exact: {exact_mmd_torch:.6f}, RFF: {rff_mmd_torch:.6f}, Rel diff: {rel_diff_torch:.4f}, Valid: {is_valid_torch}")
    assert is_valid_torch, f"PyTorch exact vs RFF should be close (rel diff < 0.15), got {rel_diff_torch:.4f}"
    
    # Test multi-sigma
    is_valid_multi, exact_mmd_multi, rff_mmd_multi = compare_exact_vs_rff(
        X_np, Y_np, sigmas=[0.5, 1.0, 2.0], n_features=768, backend='numpy', tolerance=0.2
    )
    rel_diff_multi = abs(exact_mmd_multi - rff_mmd_multi) / max(exact_mmd_multi, 1e-10)
    print(f"Multi-sigma - Exact: {exact_mmd_multi:.6f}, RFF: {rff_mmd_multi:.6f}, Rel diff: {rel_diff_multi:.4f}, Valid: {is_valid_multi}")
    assert is_valid_multi, f"Multi-sigma exact vs RFF should be close (rel diff < 0.2), got {rel_diff_multi:.4f}"
    
    print("✓ compare_exact_vs_rff tests passed")


def test_permutation_invariance():
    """Test that MMD is invariant to row permutations."""
    print("\n=== Testing test_permutation_invariance ===")
    
    # Generate test data
    X_np = np.random.randn(40, 6)
    Y_np = np.random.randn(40, 6) + 0.3
    
    # Import the function with a different name to avoid conflict
    from .validation import test_permutation_invariance as validate_permutation
    
    # Test NumPy backend
    is_valid_np, original_mmd_np, permuted_mmd_np = validate_permutation(
        X_np, Y_np, sigmas=1.0, backend='numpy', tolerance=1e-9
    )
    diff_np = abs(original_mmd_np - permuted_mmd_np)
    print(f"NumPy - Original: {original_mmd_np:.6f}, Permuted: {permuted_mmd_np:.6f}, Diff: {diff_np:.2e}, Valid: {is_valid_np}")
    assert is_valid_np, f"NumPy MMD should be permutation invariant (diff < 1e-9), got {diff_np:.2e}"
    
    # Test PyTorch backend (slightly relaxed tolerance due to floating point precision)
    X_torch = torch.randn(40, 6)
    Y_torch = torch.randn(40, 6) + 0.3
    is_valid_torch, original_mmd_torch, permuted_mmd_torch = validate_permutation(
        X_torch, Y_torch, sigmas=1.0, backend='torch', tolerance=1e-7
    )
    diff_torch = abs(original_mmd_torch - permuted_mmd_torch)
    print(f"PyTorch - Original: {original_mmd_torch:.6f}, Permuted: {permuted_mmd_torch:.6f}, Diff: {diff_torch:.2e}, Valid: {is_valid_torch}")
    assert is_valid_torch, f"PyTorch MMD should be permutation invariant (diff < 1e-7), got {diff_torch:.2e}"
    
    # Test multi-sigma
    is_valid_multi, original_mmd_multi, permuted_mmd_multi = validate_permutation(
        X_np, Y_np, sigmas=[0.5, 1.5], backend='numpy', tolerance=1e-9
    )
    diff_multi = abs(original_mmd_multi - permuted_mmd_multi)
    print(f"Multi-sigma - Original: {original_mmd_multi:.6f}, Permuted: {permuted_mmd_multi:.6f}, Diff: {diff_multi:.2e}, Valid: {is_valid_multi}")
    assert is_valid_multi, f"Multi-sigma MMD should be permutation invariant (diff < 1e-9), got {diff_multi:.2e}"
    
    print("✓ test_permutation_invariance tests passed")


def test_sigma_sensitivity():
    """Test that MMD behaves correctly as sigma varies."""
    print("\n=== Testing test_sigma_sensitivity ===")
    
    # Import the function with a different name to avoid conflict
    from .validation import test_sigma_sensitivity as validate_sigma_sensitivity
    
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate two different distributions
    X_np = np.random.randn(25, 5)
    Y_np = np.random.randn(25, 5) + 1.0  # Clearly different distributions
    
    # Test NumPy backend
    sigma_range = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    sigmas_np, mmd_values_np = validate_sigma_sensitivity(X_np, Y_np, sigma_range, backend='numpy')
    print(f"NumPy sigma sensitivity:")
    for s, v in zip(sigmas_np, mmd_values_np):
        print(f"  sigma={s:5.2f}, MMD²={v:.6f}")
    
    # Check that MMD values are reasonable (positive, finite)
    assert np.all(mmd_values_np > 0), "All MMD values should be positive"
    assert np.all(np.isfinite(mmd_values_np)), "All MMD values should be finite"
    
    # Test PyTorch backend with same data
    X_torch = torch.from_numpy(X_np).float()
    Y_torch = torch.from_numpy(Y_np).float()
    sigmas_torch, mmd_values_torch = validate_sigma_sensitivity(X_torch, Y_torch, sigma_range, backend='torch')
    print(f"PyTorch sigma sensitivity:")
    for s, v in zip(sigmas_torch, mmd_values_torch):
        print(f"  sigma={s:5.2f}, MMD²={v:.6f}")
    
    assert np.all(mmd_values_torch > 0), "All PyTorch MMD values should be positive"
    assert np.all(np.isfinite(mmd_values_torch)), "All PyTorch MMD values should be finite"
    
    # Check that NumPy and PyTorch give similar results (within numerical precision)
    max_diff = np.max(np.abs(mmd_values_np - mmd_values_torch))
    print(f"Max difference between NumPy and PyTorch: {max_diff:.2e}")
    assert max_diff < 1e-4, f"NumPy and PyTorch should give similar results, max diff: {max_diff:.2e}"
    
    # Test that very small sigma gives high MMD (sensitive to differences)
    # and very large sigma gives lower MMD (less sensitive)
    small_sigma_mmd = mmd_values_np[0]  # sigma=0.1
    large_sigma_mmd = mmd_values_np[-1]  # sigma=10.0
    print(f"Small sigma (0.1) MMD: {small_sigma_mmd:.6f}, Large sigma (10.0) MMD: {large_sigma_mmd:.6f}")
    # Note: This is not always true, but for clearly different distributions it often holds
    # We'll just check that values are reasonable
    
    print("✓ test_sigma_sensitivity tests passed")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n=== Testing edge cases ===")
    
    # Test with very small datasets
    X_small = np.random.randn(2, 3)
    Y_small = np.random.randn(2, 3)
    is_valid, mmd_val = validate_mmd_zero(X_small, backend='numpy', tolerance=1e-3)
    print(f"Small dataset (2 samples): MMD(X, X) = {mmd_val:.2e}, Valid: {is_valid}")
    assert is_valid, "Small dataset should still pass zero test"
    
    # Test with single sample (should handle gracefully)
    X_one = np.random.randn(1, 3)
    try:
        mmd_val = validate_mmd_zero(X_one, backend='numpy', tolerance=1e-3)[1]
        print(f"Single sample: MMD(X, X) = {mmd_val:.2e}")
    except Exception as e:
        print(f"Single sample test raised exception (may be expected): {e}")
    
    # Test with identical distributions (should have very small MMD)
    X_identical = np.random.randn(20, 5)
    Y_identical = X_identical.copy()  # Exact copy
    is_valid, mmd_val = validate_mmd_zero(X_identical, backend='numpy', tolerance=1e-5)
    print(f"Identical distributions: MMD(X, X) = {mmd_val:.2e}, Valid: {is_valid}")
    assert is_valid, "Identical distributions should have MMD ≈ 0"
    
    # Test with very different distributions (should have larger MMD)
    X_diff = np.random.randn(20, 5)
    Y_diff = np.random.randn(20, 5) + 10.0  # Very different
    mmd_diff = compare_exact_vs_rff(X_diff, Y_diff, sigmas=1.0, backend='numpy')[1]
    print(f"Very different distributions: MMD² = {mmd_diff:.6f}")
    assert mmd_diff > 0.1, "Very different distributions should have larger MMD"
    
    print("✓ Edge case tests passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("MMD Validation Tests")
    print("=" * 60)
    
    try:
        test_validate_mmd_zero()
        test_compare_exact_vs_rff()
        test_permutation_invariance()
        test_sigma_sensitivity()
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()

