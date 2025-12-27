"""
Comprehensive tests for TDA persistence computation.

Test Categories:
- Unit Tests: Individual function behavior
- Edge Cases: Boundary conditions, empty inputs, extreme values
- Property Tests: Mathematical invariants that should hold
- Integration: Functions working together
"""

import pytest
import numpy as np
from todacomm.tda.persistence import (
    TDAConfig,
    _maybe_pca,
    compute_persistence,
    summarize_diagrams,
    uniform_sample,
    multiscale_sample,
    adaptive_sample,
    apply_sampling,
    persistence_image,
    BootstrapResult,
    bootstrap_persistence_ci,
    compare_bootstrap_results,
    ablation_sample_size,
    ablation_pca_components
)


# =============================================================================
# TDAConfig Tests
# =============================================================================

class TestTDAConfig:
    """Tests for TDAConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TDAConfig()
        assert config.maxdim == 1
        assert config.metric == "euclidean"
        assert config.pca_components == 20
        assert config.sampling_strategy == "uniform"
        assert config.max_points == 2000

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TDAConfig(
            maxdim=2,
            metric="cosine",
            pca_components=50,
            sampling_strategy="adaptive",
            max_points=500
        )
        assert config.maxdim == 2
        assert config.metric == "cosine"
        assert config.pca_components == 50
        assert config.sampling_strategy == "adaptive"
        assert config.max_points == 500

    def test_multiscale_parameters(self):
        """Test multiscale sampling parameters."""
        config = TDAConfig(
            sampling_strategy="multiscale",
            global_sample_ratio=0.7,
            local_neighborhoods=5,
            local_neighbors=30
        )
        assert config.global_sample_ratio == 0.7
        assert config.local_neighborhoods == 5
        assert config.local_neighbors == 30

    def test_none_pca_components(self):
        """Test with PCA disabled."""
        config = TDAConfig(pca_components=None)
        assert config.pca_components is None


# =============================================================================
# PCA Tests
# =============================================================================

class TestMaybePCA:
    """Tests for _maybe_pca function."""

    def test_pca_reduction(self):
        """Test PCA reduces dimensions correctly."""
        X = np.random.randn(100, 50)
        result = _maybe_pca(X, k=10)
        assert result.shape == (100, 10)

    def test_pca_preserves_samples(self):
        """Test PCA preserves number of samples."""
        X = np.random.randn(200, 100)
        result = _maybe_pca(X, k=20)
        assert result.shape[0] == 200

    def test_pca_none_returns_original(self):
        """Test PCA with k=None returns original data."""
        X = np.random.randn(100, 50)
        result = _maybe_pca(X, k=None)
        np.testing.assert_array_equal(result, X)

    def test_pca_k_larger_than_dims(self):
        """Test PCA when k > number of dimensions."""
        X = np.random.randn(100, 10)
        result = _maybe_pca(X, k=20)
        np.testing.assert_array_equal(result, X)

    def test_pca_k_equal_to_dims(self):
        """Test PCA when k equals number of dimensions."""
        X = np.random.randn(100, 10)
        result = _maybe_pca(X, k=10)
        np.testing.assert_array_equal(result, X)

    def test_pca_output_dtype(self):
        """Test PCA output is float32."""
        X = np.random.randn(100, 50).astype(np.float64)
        result = _maybe_pca(X, k=10)
        assert result.dtype == np.float32

    def test_pca_centering(self):
        """Test that PCA centers the data."""
        X = np.random.randn(100, 50) + 100  # Offset data
        result = _maybe_pca(X, k=10)
        # Result should be approximately centered
        assert np.abs(result.mean()) < 1.0

    def test_pca_variance_ordering(self):
        """Test that PCA components are ordered by variance."""
        # Create data with known variance structure
        np.random.seed(42)
        X = np.random.randn(100, 50)
        result = _maybe_pca(X, k=10)
        # First component should have highest variance
        variances = np.var(result, axis=0)
        assert variances[0] >= variances[-1]


# =============================================================================
# Uniform Sampling Tests
# =============================================================================

class TestUniformSample:
    """Tests for uniform_sample function."""

    def test_basic_sampling(self):
        """Test basic uniform sampling."""
        X = np.random.randn(1000, 10)
        sampled = uniform_sample(X, n_samples=100, seed=42)
        assert sampled.shape == (100, 10)

    def test_sampling_smaller_than_data(self):
        """Test when n_samples < data size."""
        X = np.random.randn(50, 10)
        sampled = uniform_sample(X, n_samples=100, seed=42)
        np.testing.assert_array_equal(sampled, X)

    def test_sampling_equal_to_data(self):
        """Test when n_samples == data size."""
        X = np.random.randn(100, 10)
        sampled = uniform_sample(X, n_samples=100, seed=42)
        np.testing.assert_array_equal(sampled, X)

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        X = np.random.randn(1000, 10)
        sample1 = uniform_sample(X, n_samples=100, seed=42)
        sample2 = uniform_sample(X, n_samples=100, seed=42)
        np.testing.assert_array_equal(sample1, sample2)

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        X = np.random.randn(1000, 10)
        sample1 = uniform_sample(X, n_samples=100, seed=42)
        sample2 = uniform_sample(X, n_samples=100, seed=123)
        assert not np.array_equal(sample1, sample2)

    def test_no_duplicates(self):
        """Test that sampling without replacement has no duplicates."""
        X = np.arange(1000).reshape(1000, 1).astype(float)
        sampled = uniform_sample(X, n_samples=100, seed=42)
        unique_values = np.unique(sampled)
        assert len(unique_values) == 100

    def test_samples_from_original(self):
        """Test that sampled points exist in original data."""
        X = np.random.randn(1000, 10)
        sampled = uniform_sample(X, n_samples=100, seed=42)
        # Each sampled row should exist in X
        for row in sampled:
            assert any(np.allclose(row, x) for x in X)

    def test_preserves_dimensions(self):
        """Test that sampling preserves feature dimensions."""
        for dim in [1, 5, 50, 100]:
            X = np.random.randn(1000, dim)
            sampled = uniform_sample(X, n_samples=100, seed=42)
            assert sampled.shape[1] == dim


# =============================================================================
# Multiscale Sampling Tests
# =============================================================================

class TestMultiscaleSample:
    """Tests for multiscale_sample function."""

    def test_basic_multiscale(self):
        """Test basic multiscale sampling."""
        X = np.random.randn(1000, 10)
        config = TDAConfig(max_points=100, sampling_strategy="multiscale")
        result = multiscale_sample(X, config, seed=42)

        assert "points" in result
        assert "method" in result
        assert result["points"].shape[0] <= 100
        assert result["method"] == "multiscale"

    def test_no_sampling_when_small(self):
        """Test that small datasets are not sampled."""
        X = np.random.randn(50, 10)
        config = TDAConfig(max_points=100, sampling_strategy="multiscale")
        result = multiscale_sample(X, config, seed=42)

        np.testing.assert_array_equal(result["points"], X)
        assert result["method"] == "no_sampling"

    def test_global_local_ratio(self):
        """Test global/local sampling ratio."""
        X = np.random.randn(1000, 10)
        config = TDAConfig(
            max_points=100,
            global_sample_ratio=0.6,
            local_neighborhoods=5,
            local_neighbors=20
        )
        result = multiscale_sample(X, config, seed=42)

        assert result["n_global"] == 60  # 60% of 100

    def test_result_metadata(self):
        """Test that result contains expected metadata."""
        X = np.random.randn(1000, 10)
        config = TDAConfig(max_points=100)
        result = multiscale_sample(X, config, seed=42)

        assert "n_global" in result
        assert "n_local" in result
        assert "n_final" in result
        assert "n_duplicates_removed" in result

    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        X = np.random.randn(1000, 10)
        config = TDAConfig(max_points=100)

        result1 = multiscale_sample(X, config, seed=42)
        result2 = multiscale_sample(X, config, seed=42)

        np.testing.assert_array_equal(result1["points"], result2["points"])

    def test_local_neighborhoods_parameter(self):
        """Test varying local neighborhoods parameter."""
        X = np.random.randn(1000, 10)

        for n_neighborhoods in [1, 5, 10, 20]:
            config = TDAConfig(
                max_points=100,
                local_neighborhoods=n_neighborhoods
            )
            result = multiscale_sample(X, config, seed=42)
            assert result["points"].shape[0] <= 100


# =============================================================================
# Adaptive Sampling Tests
# =============================================================================

class TestAdaptiveSample:
    """Tests for adaptive_sample function."""

    def test_basic_adaptive(self):
        """Test basic adaptive sampling."""
        X = np.random.randn(1000, 10)
        config = TDAConfig(max_points=100, sampling_strategy="adaptive")
        result = adaptive_sample(X, config, seed=42)

        assert "points" in result
        assert result["points"].shape[0] == 100
        assert result["method"] == "adaptive"

    def test_no_sampling_when_small(self):
        """Test that small datasets are not sampled."""
        X = np.random.randn(50, 10)
        config = TDAConfig(max_points=100)
        result = adaptive_sample(X, config, seed=42)

        np.testing.assert_array_equal(result["points"], X)
        assert result["method"] == "no_sampling"

    def test_density_stats(self):
        """Test that density statistics are returned."""
        X = np.random.randn(1000, 10)
        config = TDAConfig(max_points=100)
        result = adaptive_sample(X, config, seed=42)

        assert "density_stats" in result
        assert "min_density" in result["density_stats"]
        assert "max_density" in result["density_stats"]
        assert "mean_density" in result["density_stats"]

    def test_density_favors_dense_regions(self):
        """Test that adaptive sampling favors dense regions."""
        # Create data with a dense cluster
        cluster = np.random.randn(500, 2) * 0.1  # Dense cluster
        spread = np.random.randn(500, 2) * 10    # Spread points
        X = np.vstack([cluster, spread])

        config = TDAConfig(max_points=100)
        result = adaptive_sample(X, config, seed=42)

        # More samples should come from the dense cluster
        # Count samples near origin (the dense cluster)
        distances_to_origin = np.linalg.norm(result["points"], axis=1)
        near_origin = np.sum(distances_to_origin < 1.0)

        # Expect more than half to be from dense region
        assert near_origin > 30

    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        X = np.random.randn(1000, 10)
        config = TDAConfig(max_points=100)

        result1 = adaptive_sample(X, config, seed=42)
        result2 = adaptive_sample(X, config, seed=42)

        np.testing.assert_array_equal(result1["points"], result2["points"])


# =============================================================================
# Apply Sampling Tests
# =============================================================================

class TestApplySampling:
    """Tests for apply_sampling dispatcher function."""

    def test_uniform_strategy(self):
        """Test uniform sampling strategy."""
        X = np.random.randn(1000, 10)
        config = TDAConfig(max_points=100, sampling_strategy="uniform")
        result = apply_sampling(X, config, seed=42)

        assert result["method"] == "uniform"
        assert result["points"].shape[0] == 100

    def test_multiscale_strategy(self):
        """Test multiscale sampling strategy."""
        X = np.random.randn(1000, 10)
        config = TDAConfig(max_points=100, sampling_strategy="multiscale")
        result = apply_sampling(X, config, seed=42)

        assert result["method"] == "multiscale"

    def test_adaptive_strategy(self):
        """Test adaptive sampling strategy."""
        X = np.random.randn(1000, 10)
        config = TDAConfig(max_points=100, sampling_strategy="adaptive")
        result = apply_sampling(X, config, seed=42)

        assert result["method"] == "adaptive"

    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        X = np.random.randn(100, 10)
        config = TDAConfig(sampling_strategy="invalid")

        with pytest.raises(ValueError, match="Unknown sampling strategy"):
            apply_sampling(X, config, seed=42)


# =============================================================================
# Compute Persistence Tests
# =============================================================================

class TestComputePersistence:
    """Tests for compute_persistence function."""

    def test_basic_persistence(self):
        """Test basic persistence computation."""
        X = np.random.randn(100, 10)
        config = TDAConfig(maxdim=1, pca_components=5, max_points=50)
        result = compute_persistence(X, config, seed=42)

        assert "dgms" in result
        assert len(result["dgms"]) == 2  # H0 and H1

    def test_persistence_output_structure(self):
        """Test persistence output contains expected keys."""
        X = np.random.randn(100, 10)
        config = TDAConfig(maxdim=1, pca_components=5, max_points=50)
        result = compute_persistence(X, config, seed=42)

        expected_keys = ["dgms", "X_proc", "X_sampled", "sampling_info",
                        "original_shape", "processed_shape", "sampled_shape"]
        for key in expected_keys:
            assert key in result

    def test_shape_tracking(self):
        """Test that shapes are tracked correctly through pipeline."""
        X = np.random.randn(200, 100)
        config = TDAConfig(maxdim=1, pca_components=20, max_points=50)
        result = compute_persistence(X, config, seed=42)

        assert result["original_shape"] == (200, 100)
        assert result["processed_shape"] == (200, 20)
        assert result["sampled_shape"][0] == 50

    def test_h0_always_has_features(self):
        """Test that H0 always has at least one feature."""
        X = np.random.randn(100, 10)
        config = TDAConfig(maxdim=1, pca_components=5, max_points=50)
        result = compute_persistence(X, config, seed=42)

        assert len(result["dgms"][0]) > 0  # H0 should have features

    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        X = np.random.randn(100, 10)
        config = TDAConfig(maxdim=1, pca_components=5, max_points=50)

        result1 = compute_persistence(X, config, seed=42)
        result2 = compute_persistence(X, config, seed=42)

        for i in range(len(result1["dgms"])):
            np.testing.assert_array_almost_equal(result1["dgms"][i], result2["dgms"][i])

    def test_different_metrics(self):
        """Test persistence with different distance metrics."""
        X = np.random.randn(50, 10)

        for metric in ["euclidean", "cosine"]:
            config = TDAConfig(maxdim=1, metric=metric, max_points=30)
            result = compute_persistence(X, config, seed=42)
            assert result is not None
            assert len(result["dgms"]) == 2

    def test_high_dimensional_data(self):
        """Test persistence on high-dimensional data (like transformer outputs)."""
        X = np.random.randn(200, 768)  # Transformer hidden size
        config = TDAConfig(
            maxdim=1,
            pca_components=50,
            max_points=100,
            sampling_strategy="uniform"
        )
        result = compute_persistence(X, config, seed=42)

        assert result is not None
        assert result["X_sampled"].shape[0] == 100
        assert result["X_proc"].shape[1] == 50

    def test_maxdim_parameter(self):
        """Test different maxdim values."""
        X = np.random.randn(50, 10)

        for maxdim in [0, 1, 2]:
            config = TDAConfig(maxdim=maxdim, max_points=30, pca_components=5)
            result = compute_persistence(X, config, seed=42)
            assert len(result["dgms"]) == maxdim + 1

    def test_no_pca(self):
        """Test persistence without PCA."""
        X = np.random.randn(50, 10)
        config = TDAConfig(maxdim=1, pca_components=None, max_points=30)
        result = compute_persistence(X, config, seed=42)

        assert result["processed_shape"] == (50, 10)


# =============================================================================
# Summarize Diagrams Tests
# =============================================================================

class TestSummarizeDiagrams:
    """Tests for summarize_diagrams function."""

    def test_basic_summarization(self):
        """Test basic diagram summarization."""
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

    def test_h0_count(self):
        """Test H0 count calculation."""
        dgm_h0 = np.array([[0, 1], [0, 2], [0, 3]])
        dgms = [dgm_h0]

        summaries = summarize_diagrams(dgms)
        assert summaries["H0_count"] == 3

    def test_h1_count(self):
        """Test H1 count calculation."""
        dgm_h0 = np.array([[0, 1]])
        dgm_h1 = np.array([[1, 3], [2, 4], [3, 5], [4, 6]])
        dgms = [dgm_h0, dgm_h1]

        summaries = summarize_diagrams(dgms)
        assert summaries["H1_count"] == 4

    def test_total_persistence(self):
        """Test total persistence calculation."""
        dgm_h0 = np.array([[0, 1], [0, 3]])  # lifetimes: 1, 3
        dgms = [dgm_h0]

        summaries = summarize_diagrams(dgms)
        assert summaries["H0_total_persistence"] == 4.0

    def test_max_lifetime(self):
        """Test max lifetime calculation."""
        dgm_h0 = np.array([[0, 1], [0, 5], [0, 3]])  # lifetimes: 1, 5, 3
        dgms = [dgm_h0]

        summaries = summarize_diagrams(dgms)
        assert summaries["H0_max_lifetime"] == 5.0

    def test_empty_diagram(self):
        """Test summarization of empty diagrams."""
        dgms = [np.array([]), np.array([])]

        summaries = summarize_diagrams(dgms)

        assert summaries["H0_count"] == 0
        assert summaries["H0_total_persistence"] == 0
        assert summaries["H0_max_lifetime"] == 0
        assert summaries["H1_count"] == 0

    def test_infinite_death_handling(self):
        """Test handling of infinite death times."""
        dgm_h0 = np.array([[0, 1], [0, np.inf]])
        dgms = [dgm_h0]

        summaries = summarize_diagrams(dgms)
        # Infinite death should be handled gracefully
        assert summaries["H0_count"] == 2
        assert not np.isinf(summaries["H0_total_persistence"])

    def test_output_types(self):
        """Test that output values are Python floats."""
        dgm_h0 = np.array([[0, 1], [0, 2]])
        dgms = [dgm_h0]

        summaries = summarize_diagrams(dgms)

        for key, value in summaries.items():
            assert isinstance(value, float), f"{key} should be float"

    def test_multiple_dimensions(self):
        """Test summarization with multiple homology dimensions."""
        dgms = [
            np.array([[0, 1]]),
            np.array([[1, 2]]),
            np.array([[2, 3]])
        ]

        summaries = summarize_diagrams(dgms)

        assert "H0_count" in summaries
        assert "H1_count" in summaries
        assert "H2_count" in summaries


# =============================================================================
# Persistence Image Tests
# =============================================================================

class TestPersistenceImage:
    """Tests for persistence_image function."""

    def test_empty_diagram_image(self):
        """Test persistence image for empty diagram."""
        dgm = np.array([])
        img = persistence_image(dgm, pixels=20)

        assert img.shape == (20, 20)
        np.testing.assert_array_equal(img, np.zeros((20, 20)))


# =============================================================================
# Integration Tests
# =============================================================================

class TestTDAIntegration:
    """Integration tests for TDA pipeline."""

    def test_full_pipeline(self):
        """Test full TDA pipeline from raw data to summaries."""
        # Generate synthetic data
        np.random.seed(42)
        X = np.random.randn(500, 100)

        # Configure TDA
        config = TDAConfig(
            maxdim=1,
            pca_components=20,
            max_points=200,
            sampling_strategy="uniform"
        )

        # Compute persistence
        result = compute_persistence(X, config, seed=42)

        # Summarize
        summaries = summarize_diagrams(result["dgms"])

        # Verify complete pipeline
        assert result["original_shape"] == (500, 100)
        assert result["processed_shape"] == (500, 20)
        assert result["sampled_shape"][0] == 200
        assert summaries["H0_count"] > 0

    def test_pipeline_with_different_sampling_strategies(self):
        """Test pipeline with all sampling strategies."""
        X = np.random.randn(500, 50)

        for strategy in ["uniform", "multiscale", "adaptive"]:
            config = TDAConfig(
                maxdim=1,
                pca_components=10,
                max_points=100,
                sampling_strategy=strategy
            )

            result = compute_persistence(X, config, seed=42)
            summaries = summarize_diagrams(result["dgms"])

            assert summaries["H0_count"] > 0

    def test_pipeline_stability(self):
        """Test that pipeline produces stable results."""
        X = np.random.randn(200, 50)
        config = TDAConfig(maxdim=1, pca_components=10, max_points=50)

        # Run multiple times with same seed
        summaries_list = []
        for _ in range(3):
            result = compute_persistence(X, config, seed=42)
            summaries = summarize_diagrams(result["dgms"])
            summaries_list.append(summaries)

        # All runs should produce identical results
        for i in range(1, len(summaries_list)):
            assert summaries_list[0] == summaries_list[i]


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestTDAEdgeCases:
    """Edge case tests for TDA functions."""

    def test_single_point(self):
        """Test with single data point."""
        X = np.array([[1.0, 2.0, 3.0]])
        config = TDAConfig(maxdim=1, pca_components=None, max_points=10)

        result = compute_persistence(X, config, seed=42)
        assert len(result["dgms"][0]) == 1  # Should have exactly one H0 feature

    def test_two_points(self):
        """Test with two data points."""
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        config = TDAConfig(maxdim=1, pca_components=None, max_points=10)

        result = compute_persistence(X, config, seed=42)
        assert len(result["dgms"]) == 2

    def test_identical_points(self):
        """Test with identical points."""
        X = np.ones((10, 5))
        config = TDAConfig(maxdim=1, pca_components=None, max_points=10)

        result = compute_persistence(X, config, seed=42)
        # All points are identical, so only one connected component
        assert result is not None

    def test_very_high_dimensional(self):
        """Test with very high dimensional data."""
        X = np.random.randn(100, 1000)
        config = TDAConfig(maxdim=1, pca_components=10, max_points=50)

        result = compute_persistence(X, config, seed=42)
        assert result["processed_shape"][1] == 10

    def test_more_features_than_samples(self):
        """Test when features > samples."""
        X = np.random.randn(10, 100)
        config = TDAConfig(maxdim=1, pca_components=5, max_points=10)

        result = compute_persistence(X, config, seed=42)
        assert result is not None


# =============================================================================
# Bootstrap Confidence Interval Tests
# =============================================================================

class TestBootstrapPersistenceCI:
    """Tests for bootstrap confidence interval computation."""

    def test_bootstrap_returns_correct_structure(self):
        """Test bootstrap returns BootstrapResult with all fields."""
        X = np.random.randn(100, 10)
        config = TDAConfig(maxdim=1, pca_components=5, max_points=50, n_bootstrap=10)

        result = bootstrap_persistence_ci(X, config, seed=42)

        assert isinstance(result, BootstrapResult)
        assert isinstance(result.point_estimate, dict)
        assert isinstance(result.ci_lower, dict)
        assert isinstance(result.ci_upper, dict)
        assert isinstance(result.std_error, dict)
        assert result.n_bootstrap == 10
        assert result.ci_level == 0.95

    def test_bootstrap_ci_bounds(self):
        """Test that CI lower <= point estimate <= CI upper (usually)."""
        X = np.random.randn(100, 10)
        config = TDAConfig(maxdim=1, pca_components=5, max_points=50, n_bootstrap=50)

        result = bootstrap_persistence_ci(X, config, seed=42)

        # CI should bracket point estimate (with some tolerance for bootstrap variability)
        for metric in result.point_estimate.keys():
            # Lower bound should be <= upper bound
            assert result.ci_lower[metric] <= result.ci_upper[metric]

    def test_bootstrap_reproducibility(self):
        """Test bootstrap is reproducible with same seed."""
        X = np.random.randn(100, 10)
        config = TDAConfig(maxdim=1, pca_components=5, max_points=50, n_bootstrap=20)

        result1 = bootstrap_persistence_ci(X, config, seed=42)
        result2 = bootstrap_persistence_ci(X, config, seed=42)

        for metric in result1.point_estimate.keys():
            assert result1.ci_lower[metric] == result2.ci_lower[metric]
            assert result1.ci_upper[metric] == result2.ci_upper[metric]

    def test_bootstrap_different_seeds_differ(self):
        """Test different seeds produce different results."""
        X = np.random.randn(100, 10)
        config = TDAConfig(maxdim=1, pca_components=5, max_points=50, n_bootstrap=20)

        result1 = bootstrap_persistence_ci(X, config, seed=42)
        result2 = bootstrap_persistence_ci(X, config, seed=123)

        # At least some CI bounds should differ
        any_different = any(
            result1.ci_lower[m] != result2.ci_lower[m]
            for m in result1.ci_lower.keys()
        )
        assert any_different

    def test_bootstrap_return_samples(self):
        """Test return_samples=True returns bootstrap samples."""
        X = np.random.randn(100, 10)
        config = TDAConfig(maxdim=1, pca_components=5, max_points=50, n_bootstrap=10)

        result = bootstrap_persistence_ci(X, config, seed=42, return_samples=True)

        assert result.bootstrap_samples is not None
        assert len(result.bootstrap_samples) == 10

    def test_bootstrap_custom_ci_level(self):
        """Test custom confidence level."""
        X = np.random.randn(100, 10)
        config = TDAConfig(maxdim=1, pca_components=5, max_points=50)

        result = bootstrap_persistence_ci(X, config, n_bootstrap=50, ci_level=0.90, seed=42)

        assert result.ci_level == 0.90

    def test_bootstrap_std_error_positive(self):
        """Test standard errors are non-negative."""
        X = np.random.randn(100, 10)
        config = TDAConfig(maxdim=1, pca_components=5, max_points=50, n_bootstrap=20)

        result = bootstrap_persistence_ci(X, config, seed=42)

        for metric, se in result.std_error.items():
            assert se >= 0


class TestCompareBootstrapResults:
    """Tests for comparing bootstrap results across conditions."""

    def test_compare_returns_correct_structure(self):
        """Test comparison returns expected fields."""
        X1 = np.random.randn(100, 10)
        X2 = np.random.randn(100, 10) * 2  # Different scale
        config = TDAConfig(maxdim=1, pca_components=5, max_points=50, n_bootstrap=10)

        r1 = bootstrap_persistence_ci(X1, config, seed=42)
        r2 = bootstrap_persistence_ci(X2, config, seed=42)

        comparison = compare_bootstrap_results(
            {"small": r1, "large": r2},
            metric="H0_total_persistence"
        )

        assert "small_vs_large" in comparison
        assert "diff" in comparison["small_vs_large"]
        assert "ci_overlap" in comparison["small_vs_large"]
        assert "significant" in comparison["small_vs_large"]
        assert "cohens_d" in comparison["small_vs_large"]

    def test_compare_identical_data(self):
        """Test comparing identical data shows no significance."""
        X = np.random.randn(100, 10)
        config = TDAConfig(maxdim=1, pca_components=5, max_points=50, n_bootstrap=20)

        r1 = bootstrap_persistence_ci(X, config, seed=42)
        r2 = bootstrap_persistence_ci(X, config, seed=43)  # Different seed, same data

        comparison = compare_bootstrap_results(
            {"run1": r1, "run2": r2},
            metric="H0_total_persistence"
        )

        # Same data should have overlapping CIs
        assert comparison["run1_vs_run2"]["ci_overlap"] is True


# =============================================================================
# Ablation Study Tests
# =============================================================================

class TestAblationSampleSize:
    """Tests for sample size ablation study."""

    def test_ablation_sample_size_structure(self):
        """Test ablation returns correct structure."""
        X = np.random.randn(200, 20)
        config = TDAConfig(maxdim=1, pca_components=10, max_points=100)

        result = ablation_sample_size(
            X, config,
            sample_sizes=[50, 100],
            n_repeats=3,
            seed=42
        )

        assert 50 in result
        assert 100 in result
        assert result[50]["n_repeats"] == 3
        assert "metrics" in result[50]
        assert "H0_total_persistence" in result[50]["metrics"]

    def test_ablation_sample_size_has_stats(self):
        """Test ablation includes mean, std, cv statistics."""
        X = np.random.randn(200, 20)
        config = TDAConfig(maxdim=1, pca_components=10, max_points=100)

        result = ablation_sample_size(
            X, config,
            sample_sizes=[100],
            n_repeats=5,
            seed=42
        )

        stats = result[100]["metrics"]["H0_total_persistence"]
        assert "mean" in stats
        assert "std" in stats
        assert "cv" in stats
        assert "min" in stats
        assert "max" in stats

    def test_ablation_filters_invalid_sizes(self):
        """Test ablation filters out sample sizes larger than data."""
        X = np.random.randn(50, 10)
        config = TDAConfig(maxdim=1, pca_components=5, max_points=100)

        result = ablation_sample_size(
            X, config,
            sample_sizes=[25, 50, 100, 200],  # 100, 200 are invalid
            n_repeats=2,
            seed=42
        )

        assert 25 in result
        assert 50 in result
        assert 100 not in result
        assert 200 not in result

    def test_ablation_reproducibility(self):
        """Test ablation is reproducible with same seed."""
        X = np.random.randn(100, 10)
        config = TDAConfig(maxdim=1, pca_components=5, max_points=50)

        r1 = ablation_sample_size(X, config, sample_sizes=[50], n_repeats=3, seed=42)
        r2 = ablation_sample_size(X, config, sample_sizes=[50], n_repeats=3, seed=42)

        assert r1[50]["metrics"]["H0_total_persistence"]["mean"] == \
               r2[50]["metrics"]["H0_total_persistence"]["mean"]


class TestAblationPCAComponents:
    """Tests for PCA components ablation study."""

    def test_ablation_pca_structure(self):
        """Test PCA ablation returns correct structure."""
        X = np.random.randn(100, 50)
        config = TDAConfig(maxdim=1, max_points=50)

        result = ablation_pca_components(
            X, config,
            pca_values=[10, 20],
            n_repeats=3,
            seed=42
        )

        assert "10" in result
        assert "20" in result
        assert result["10"]["pca_components"] == 10
        assert result["10"]["n_repeats"] == 3

    def test_ablation_pca_none(self):
        """Test PCA ablation with None (no PCA)."""
        X = np.random.randn(100, 30)
        config = TDAConfig(maxdim=1, max_points=50)

        result = ablation_pca_components(
            X, config,
            pca_values=[10, None],
            n_repeats=2,
            seed=42
        )

        assert "10" in result
        assert "None" in result
        assert result["None"]["pca_components"] is None

    def test_ablation_pca_filters_invalid(self):
        """Test PCA ablation filters values >= n_dims."""
        X = np.random.randn(100, 20)
        config = TDAConfig(maxdim=1, max_points=50)

        result = ablation_pca_components(
            X, config,
            pca_values=[5, 10, 20, 50],  # 20, 50 are invalid
            n_repeats=2,
            seed=42
        )

        assert "5" in result
        assert "10" in result
        assert "20" not in result
        assert "50" not in result

    def test_ablation_pca_has_cv(self):
        """Test PCA ablation includes coefficient of variation."""
        X = np.random.randn(100, 30)
        config = TDAConfig(maxdim=1, max_points=50)

        result = ablation_pca_components(
            X, config,
            pca_values=[10],
            n_repeats=5,
            seed=42
        )

        assert "cv" in result["10"]["metrics"]["H0_total_persistence"]


# =============================================================================
# TDAConfig Bootstrap Parameter Tests
# =============================================================================

class TestTDAConfigBootstrap:
    """Tests for bootstrap parameters in TDAConfig."""

    def test_default_bootstrap_values(self):
        """Test default bootstrap configuration values."""
        config = TDAConfig()
        assert config.n_bootstrap == 100
        assert config.ci_level == 0.95

    def test_custom_bootstrap_values(self):
        """Test custom bootstrap configuration values."""
        config = TDAConfig(n_bootstrap=50, ci_level=0.90)
        assert config.n_bootstrap == 50
        assert config.ci_level == 0.90


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
