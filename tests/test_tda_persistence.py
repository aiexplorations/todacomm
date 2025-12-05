"""
Tests for TDA persistence computation.
"""

import pytest
import numpy as np
from tda_perturbation.tda.persistence import (
    TDAConfig,
    compute_persistence,
    summarize_diagrams,
    uniform_sample,
    multiscale_sample,
    adaptive_sample
)


def test_tda_config():
    """Test TDA configuration."""
    config = TDAConfig(
        maxdim=1,
        metric="euclidean",
        pca_components=20,
        sampling_strategy="uniform",
        max_points=1000
    )
    
    assert config.maxdim == 1
    assert config.metric == "euclidean"
    assert config.sampling_strategy == "uniform"


def test_uniform_sample():
    """Test uniform sampling."""
    X = np.random.randn(1000, 10)
    sampled = uniform_sample(X, n_samples=100, seed=42)
    
    assert sampled.shape[0] == 100
    assert sampled.shape[1] == 10


def test_multiscale_sample():
    """Test multiscale sampling."""
    X = np.random.randn(1000, 10)
    config = TDAConfig(max_points=100, sampling_strategy="multiscale")
    
    result = multiscale_sample(X, config, seed=42)
    
    assert "points" in result
    assert "method" in result
    assert result["points"].shape[0] <= 100


def test_adaptive_sample():
    """Test adaptive sampling."""
    X = np.random.randn(1000, 10)
    config = TDAConfig(max_points=100, sampling_strategy="adaptive")
    
    result = adaptive_sample(X, config, seed=42)
    
    assert "points" in result
    assert "method" in result
    assert result["points"].shape[0] == 100


def test_compute_persistence():
    """Test persistence computation."""
    # Create simple point cloud
    X = np.random.randn(100, 10)
    config = TDAConfig(maxdim=1, pca_components=5, max_points=50)
    
    result = compute_persistence(X, config, seed=42)
    
    assert "dgms" in result
    assert "X_sampled" in result
    assert "sampling_info" in result
    assert len(result["dgms"]) == 2  # H0 and H1


def test_summarize_diagrams():
    """Test diagram summarization."""
    # Create mock persistence diagrams
    dgm_h0 = np.array([[0, 1], [0, 2], [0, np.inf]])
    dgm_h1 = np.array([[1, 3], [2, 4]])
    dgms = [dgm_h0, dgm_h1]
    
    summaries = summarize_diagrams(dgms)
    
    assert "H0_count" in summaries
    assert "H0_total_persistence" in summaries
    assert "H0_max_lifetime" in summaries
    assert "H1_count" in summaries
    assert "H1_total_persistence" in summaries
    assert "H1_max_lifetime" in summaries
    
    assert summaries["H0_count"] == 3
    assert summaries["H1_count"] == 2


def test_summarize_empty_diagram():
    """Test summarization of empty diagrams."""
    dgms = [np.array([]), np.array([])]
    
    summaries = summarize_diagrams(dgms)
    
    assert summaries["H0_count"] == 0
    assert summaries["H0_total_persistence"] == 0
    assert summaries["H1_count"] == 0


def test_persistence_with_different_metrics():
    """Test persistence with different distance metrics."""
    X = np.random.randn(50, 10)
    
    for metric in ["euclidean", "cosine"]:
        config = TDAConfig(maxdim=1, metric=metric, max_points=30)
        result = compute_persistence(X, config, seed=42)
        
        assert result is not None
        assert len(result["dgms"]) == 2


def test_persistence_high_dimensional():
    """Test persistence on high-dimensional data (like transformer outputs)."""
    # Simulate transformer hidden states
    X = np.random.randn(200, 768)  # 768 = GPT-2 hidden size
    
    config = TDAConfig(
        maxdim=1,
        pca_components=50,  # Reduce to 50 dims
        max_points=100,
        sampling_strategy="uniform"
    )
    
    result = compute_persistence(X, config, seed=42)
    
    assert result is not None
    assert result["X_sampled"].shape[0] == 100
    assert result["X_proc"].shape[1] == 50  # PCA reduced


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
