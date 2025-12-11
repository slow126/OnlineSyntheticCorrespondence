"""
Unit tests for weighted coreset module.

Run with: pytest src/coreset/test_coreset.py -v
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from .weighted_coreset import WeightedCoreset
from .metrics import (
    estimate_epsilon_from_eval,
    coverage_by_train,
    extraneous_mass_fraction,
    compute_nn_distances,
)
from .config import CoresetConfig, load_config_from_yaml, save_config_to_yaml


class TestCoresetConfig:
    """Tests for CoresetConfig."""
    
    def test_config_creation(self):
        """Test creating a config."""
        config = CoresetConfig(
            K_max=1000,
            K_overflow=500,
            distance='euclidean',
            device='cpu',
        )
        assert config.K_max == 1000
        assert config.K_overflow == 500
        assert config.distance == 'euclidean'
    
    def test_config_save_load(self):
        """Test saving and loading config."""
        config = CoresetConfig(
            K_max=1000,
            K_overflow=500,
            is_eval=True,
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            save_config_to_yaml(config, f.name)
            loaded = load_config_from_yaml(f.name)
        
        assert loaded.K_max == config.K_max
        assert loaded.K_overflow == config.K_overflow
        assert loaded.is_eval == config.is_eval


class TestWeightedCoreset:
    """Tests for WeightedCoreset."""
    
    def test_basic_coreset(self):
        """Test basic coreset construction."""
        coreset = WeightedCoreset(K_max=10, K_overflow=5)
        
        # Add some data
        data = np.random.randn(100, 4).astype(np.float32)
        coreset.update(data)
        coreset.finalize()
        
        centers = coreset.get_centers()
        counts = coreset.get_counts()
        
        assert centers.shape[0] <= 10  # Should be at most K_max
        assert centers.shape[1] == 4   # Same dimension
        assert len(counts) == len(centers)
        assert counts.sum() == pytest.approx(100, rel=0.01)  # Total count
    
    def test_streaming_coreset(self):
        """Test streaming with multiple batches."""
        coreset = WeightedCoreset(K_max=20, K_overflow=10)
        
        # Add multiple batches
        for _ in range(5):
            data = np.random.randn(50, 3).astype(np.float32)
            coreset.update(data)
        
        coreset.finalize()
        
        centers = coreset.get_centers()
        counts = coreset.get_counts()
        
        assert centers.shape[0] <= 20
        assert centers.shape[1] == 3
        assert coreset.total_samples == 250
    
    def test_save_load(self):
        """Test saving and loading coreset."""
        coreset = WeightedCoreset(K_max=10, is_eval=False)
        
        data = np.random.randn(100, 4).astype(np.float32)
        coreset.update(data)
        coreset.finalize()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'coreset.pt'
            coreset.save(str(path))
            
            loaded = WeightedCoreset.load(str(path))
        
        np.testing.assert_array_almost_equal(
            coreset.get_centers(),
            loaded.get_centers()
        )
        np.testing.assert_array_almost_equal(
            coreset.get_counts(),
            loaded.get_counts()
        )
        assert loaded.total_samples == coreset.total_samples
    
    def test_epsilon_computation(self):
        """Test epsilon computation for eval coresets."""
        coreset = WeightedCoreset(K_max=20, is_eval=True)
        
        # Create data with known spacing
        data = np.random.randn(100, 2).astype(np.float32)
        coreset.update(data)
        coreset.finalize()
        
        epsilon_scales = coreset.get_epsilon_scales()
        assert epsilon_scales is not None
        assert 'eps_base' in epsilon_scales
        assert 'eps_2x' in epsilon_scales
        assert epsilon_scales['eps_2x'] == pytest.approx(2 * epsilon_scales['eps_base'])


class TestMetrics:
    """Tests for metric functions."""
    
    def test_compute_nn_distances(self):
        """Test nearest neighbor distance computation."""
        centers = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
        queries = np.array([[0, 0], [1, 0], [2, 2]], dtype=np.float32)
        
        distances, indices = compute_nn_distances(centers, queries)
        
        assert len(distances) == 3
        assert len(indices) == 3
        assert distances[0] == pytest.approx(0.0)  # Exact match
        assert distances[2] == pytest.approx(0.0)  # Exact match
        assert distances[1] < 2.0  # Close to [1,1] or [0,0]
    
    def test_epsilon_estimation(self):
        """Test epsilon estimation from eval data."""
        # Create 2D grid with known spacing
        x = np.linspace(0, 10, 11)
        y = np.linspace(0, 10, 11)
        xx, yy = np.meshgrid(x, y)
        data = np.stack([xx.flatten(), yy.flatten()], axis=1).astype(np.float32)
        
        epsilon_scales = estimate_epsilon_from_eval(data, quantile=0.5)
        
        assert 'eps_base' in epsilon_scales
        assert epsilon_scales['eps_base'] > 0
        # For a regular grid, median NN distance should be ~1.0
        assert epsilon_scales['eps_base'] == pytest.approx(1.0, rel=0.1)
    
    def test_coverage_by_train(self):
        """Test coverage metric."""
        # Train covers a square [0, 10] x [0, 10]
        train_centers = np.random.rand(100, 2).astype(np.float32) * 10
        train_counts = np.ones(100, dtype=np.float32)
        
        # Eval is inside the square
        eval_centers = np.random.rand(50, 2).astype(np.float32) * 8 + 1  # [1, 9] x [1, 9]
        
        coverage = coverage_by_train(
            train_centers, train_counts, eval_centers,
            epsilon=2.0, min_count=0
        )
        
        assert 'coverage_rel' in coverage
        assert 0 <= coverage['coverage_rel'] <= 1
        # Most eval points should be covered with epsilon=2.0
        assert coverage['coverage_rel'] > 0.5
    
    def test_coverage_perfect(self):
        """Test perfect coverage case."""
        # Train and eval are the same
        centers = np.random.randn(50, 3).astype(np.float32)
        counts = np.ones(50, dtype=np.float32)
        
        coverage = coverage_by_train(
            centers, counts, centers,
            epsilon=0.01, min_count=0
        )
        
        assert coverage['coverage_rel'] == pytest.approx(1.0)
        assert coverage['rho_mean'] == pytest.approx(0.0, abs=1e-5)
    
    def test_extraneous_mass(self):
        """Test extraneous mass metric."""
        # Train has some points far from eval
        train_centers = np.vstack([
            np.random.randn(50, 2),  # Near origin
            np.random.randn(50, 2) + 100,  # Far away
        ]).astype(np.float32)
        train_counts = np.ones(100, dtype=np.float32)
        
        # Eval is near origin
        eval_centers = np.random.randn(50, 2).astype(np.float32)
        
        extran = extraneous_mass_fraction(
            train_centers, train_counts, eval_centers,
            epsilon=10.0
        )
        
        assert 'extraneous_mass_frac' in extran
        assert 0 <= extran['extraneous_mass_frac'] <= 1
        # About half of train mass should be extraneous
        assert extran['extraneous_mass_frac'] > 0.3


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_pipeline(self):
        """Test full pipeline: build coresets, compute metrics."""
        # Build train coreset
        train_data = np.random.randn(500, 4).astype(np.float32)
        train_coreset = WeightedCoreset(K_max=50, is_eval=False)
        train_coreset.update(train_data)
        train_coreset.finalize()
        
        # Build eval coreset with epsilon
        eval_data = np.random.randn(300, 4).astype(np.float32)
        eval_coreset = WeightedCoreset(K_max=30, is_eval=True)
        eval_coreset.update(eval_data)
        eval_coreset.finalize()
        
        # Get components
        train_centers = train_coreset.get_centers()
        train_counts = train_coreset.get_counts()
        eval_centers = eval_coreset.get_centers()
        epsilon_scales = eval_coreset.get_epsilon_scales()
        
        assert epsilon_scales is not None
        eps = epsilon_scales['eps_base']
        
        # Compute coverage
        coverage = coverage_by_train(
            train_centers, train_counts, eval_centers,
            epsilon=eps, min_count=0
        )
        
        # Compute extraneous mass
        extran = extraneous_mass_fraction(
            train_centers, train_counts, eval_centers,
            epsilon=eps
        )
        
        # Sanity checks
        assert 0 <= coverage['coverage_rel'] <= 1
        assert 0 <= extran['extraneous_mass_frac'] <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
