"""
Tests for pre-TDA geometry characterization module.

Tests intrinsic dimensionality estimation, k-NN structure analysis,
and geometry characterization utilities.
"""

import pytest
import numpy as np
import pandas as pd

from todacomm.analysis.geometry import (
    GeometryConfig,
    GeometryResult,
    compute_basic_stats,
    compute_correlation_matrix,
    estimate_mle_intrinsic_dim,
    estimate_correlation_dim,
    estimate_local_pca_dim,
    compute_pca_analysis,
    compute_knn_structure,
    characterize_geometry,
    summarize_geometry,
    recommend_pca_components,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def random_data():
    """Generate random high-dimensional data."""
    np.random.seed(42)
    return np.random.randn(500, 50).astype(np.float32)


@pytest.fixture
def low_rank_data():
    """Generate data lying on a low-dimensional subspace."""
    np.random.seed(42)
    # Generate 5-dimensional data embedded in 50D
    latent = np.random.randn(500, 5)
    projection = np.random.randn(5, 50)
    data = latent @ projection
    return data.astype(np.float32)


@pytest.fixture
def sparse_data():
    """Generate sparse activation data (many zeros)."""
    np.random.seed(42)
    data = np.random.randn(500, 50).astype(np.float32)
    # Apply ReLU-like sparsity
    data[data < 0] = 0
    return data


# =============================================================================
# GeometryConfig Tests
# =============================================================================

class TestGeometryConfig:
    """Tests for GeometryConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GeometryConfig()

        assert config.compute_basic_stats is True
        assert config.compute_intrinsic_dim is True
        assert config.compute_knn_structure is True
        assert config.k_neighbors == 20
        assert config.mle_k == 10
        assert config.max_samples_for_dim == 5000

    def test_custom_values(self):
        """Test custom configuration values."""
        config = GeometryConfig(
            k_neighbors=30,
            mle_k=20,
            max_samples_for_dim=1000,
            compute_knn_structure=False
        )

        assert config.k_neighbors == 30
        assert config.mle_k == 20
        assert config.max_samples_for_dim == 1000
        assert config.compute_knn_structure is False


# =============================================================================
# Basic Statistics Tests
# =============================================================================

class TestBasicStats:
    """Tests for basic statistics computation."""

    def test_compute_basic_stats(self, random_data):
        """Test basic statistics computation."""
        stats = compute_basic_stats(random_data)

        assert "mean" in stats
        assert "variance" in stats
        assert "total_variance" in stats
        assert "sparsity" in stats

        assert stats["mean"].shape == (50,)
        assert stats["variance"].shape == (50,)
        assert isinstance(stats["total_variance"], float)

    def test_mean_near_zero_for_random(self, random_data):
        """Test that mean is near zero for standard normal data."""
        stats = compute_basic_stats(random_data)
        assert np.abs(stats["mean"]).max() < 0.5

    def test_variance_near_one_for_random(self, random_data):
        """Test that variance is near one for standard normal data."""
        stats = compute_basic_stats(random_data)
        mean_var = np.mean(stats["variance"])
        assert 0.5 < mean_var < 2.0

    def test_sparsity_for_sparse_data(self, sparse_data):
        """Test sparsity detection for sparse data."""
        stats = compute_basic_stats(sparse_data)
        # ReLU-sparse data should have ~50% zeros
        assert stats["sparsity"] > 0.4

    def test_sparsity_for_dense_data(self, random_data):
        """Test sparsity near zero for dense random data."""
        stats = compute_basic_stats(random_data)
        assert stats["sparsity"] < 0.01


class TestCorrelationMatrix:
    """Tests for correlation matrix computation."""

    def test_correlation_shape(self, random_data):
        """Test correlation matrix shape."""
        corr = compute_correlation_matrix(random_data)
        assert corr.shape == (50, 50)

    def test_diagonal_is_one(self, random_data):
        """Test that diagonal is ones."""
        corr = compute_correlation_matrix(random_data)
        np.testing.assert_array_almost_equal(np.diag(corr), np.ones(50), decimal=5)

    def test_symmetric(self, random_data):
        """Test that correlation matrix is symmetric."""
        corr = compute_correlation_matrix(random_data)
        np.testing.assert_array_almost_equal(corr, corr.T)

    def test_max_dims_limit(self, random_data):
        """Test that max_dims limits output size."""
        corr = compute_correlation_matrix(random_data, max_dims=20)
        assert corr.shape == (20, 20)


# =============================================================================
# Intrinsic Dimensionality Tests
# =============================================================================

class TestMLEIntrinsicDim:
    """Tests for MLE intrinsic dimension estimator."""

    def test_returns_float(self, random_data):
        """Test that MLE returns a float."""
        dim = estimate_mle_intrinsic_dim(random_data)
        assert isinstance(dim, float)

    def test_low_rank_data(self, low_rank_data):
        """Test MLE on low-rank data estimates close to true dimension."""
        dim = estimate_mle_intrinsic_dim(low_rank_data, k=10)
        # Should estimate close to 5 (the true intrinsic dim)
        assert 2 < dim < 15

    def test_high_dim_random_data(self, random_data):
        """Test MLE on random data gives higher estimate."""
        dim = estimate_mle_intrinsic_dim(random_data, k=10)
        # Random 50D data should give high estimate
        assert dim > 10

    def test_small_k(self, random_data):
        """Test with small k value."""
        dim = estimate_mle_intrinsic_dim(random_data, k=3)
        assert dim > 0

    def test_handles_few_samples(self):
        """Test handling of very few samples."""
        X = np.random.randn(5, 10)
        dim = estimate_mle_intrinsic_dim(X, k=10)
        # Should return ambient dimension if too few samples
        assert dim == 10.0


class TestCorrelationDim:
    """Tests for correlation dimension estimator."""

    def test_returns_float(self, random_data):
        """Test that correlation dim returns a float."""
        dim = estimate_correlation_dim(random_data)
        assert isinstance(dim, float) or np.isnan(dim)

    def test_low_rank_data(self, low_rank_data):
        """Test correlation dimension on low-rank data."""
        dim = estimate_correlation_dim(low_rank_data)
        if not np.isnan(dim):
            # Should be in reasonable range
            assert 1 < dim < 20

    def test_handles_small_data(self):
        """Test handling of small dataset."""
        X = np.random.randn(5, 10)
        dim = estimate_correlation_dim(X)
        # Should return NaN for very small data
        assert np.isnan(dim)


class TestLocalPCADim:
    """Tests for local PCA dimension estimator."""

    def test_returns_float(self, random_data):
        """Test that local PCA returns a float."""
        dim = estimate_local_pca_dim(random_data, k=30, n_samples=50)
        assert isinstance(dim, float)

    def test_low_rank_data(self, low_rank_data):
        """Test local PCA on low-rank data."""
        dim = estimate_local_pca_dim(low_rank_data, k=30, n_samples=50)
        # Should estimate close to 5
        assert 2 < dim < 15

    def test_variance_threshold_effect(self, random_data):
        """Test that higher variance threshold gives higher dimension."""
        dim_low = estimate_local_pca_dim(random_data, variance_threshold=0.80)
        dim_high = estimate_local_pca_dim(random_data, variance_threshold=0.99)
        assert dim_high >= dim_low


class TestPCAAnalysis:
    """Tests for PCA analysis."""

    def test_explained_variance_sums_to_one(self, random_data):
        """Test that explained variance ratios sum to less than or equal to 1."""
        result = compute_pca_analysis(random_data)
        assert result["explained_variance_ratio"].sum() <= 1.01

    def test_cumulative_monotonic(self, random_data):
        """Test that cumulative variance is monotonically increasing."""
        result = compute_pca_analysis(random_data)
        cumsum = result["cumulative_variance"]
        assert np.all(np.diff(cumsum) >= 0)

    def test_dim_for_95_var(self, random_data):
        """Test dimension for 95% variance is reasonable."""
        result = compute_pca_analysis(random_data)
        assert 1 <= result["dim_for_95_var"] <= 50


# =============================================================================
# k-NN Structure Tests
# =============================================================================

class TestKNNStructure:
    """Tests for k-NN structure analysis."""

    def test_returns_expected_keys(self, random_data):
        """Test that kNN analysis returns expected keys."""
        result = compute_knn_structure(random_data, k=10)

        assert "distances" in result
        assert "distance_distribution" in result
        assert "hubness_score" in result
        assert "k_occurrences" in result

    def test_distance_shape(self, random_data):
        """Test that distances have correct shape."""
        result = compute_knn_structure(random_data, k=10)
        # Shape should be (n_samples, k) - excluding self
        assert result["distances"].shape == (500, 10)

    def test_distances_positive(self, random_data):
        """Test that all distances are positive."""
        result = compute_knn_structure(random_data, k=10)
        assert np.all(result["distances"] >= 0)

    def test_distance_distribution_stats(self, random_data):
        """Test distance distribution statistics."""
        result = compute_knn_structure(random_data, k=10)
        dist = result["distance_distribution"]

        assert "mean" in dist
        assert "std" in dist
        assert "median" in dist
        assert "iqr" in dist
        assert dist["mean"] > 0
        assert dist["std"] > 0

    def test_hubness_score(self, random_data):
        """Test hubness score is computed."""
        result = compute_knn_structure(random_data, k=10)
        assert isinstance(result["hubness_score"], float)

    def test_k_occurrences_sum(self, random_data):
        """Test that k-occurrences sum correctly."""
        result = compute_knn_structure(random_data, k=10)
        # Total occurrences should be n_samples * k
        assert np.sum(result["k_occurrences"]) == 500 * 10


# =============================================================================
# Full Characterization Tests
# =============================================================================

class TestCharacterizeGeometry:
    """Tests for full geometry characterization."""

    def test_returns_geometry_result(self, random_data):
        """Test that characterize_geometry returns GeometryResult."""
        result = characterize_geometry(random_data, "test_layer")
        assert isinstance(result, GeometryResult)

    def test_layer_name_stored(self, random_data):
        """Test that layer name is stored."""
        result = characterize_geometry(random_data, "hidden_0")
        assert result.layer_name == "hidden_0"

    def test_dimensions_stored(self, random_data):
        """Test that dimensions are stored."""
        result = characterize_geometry(random_data, "test")
        assert result.n_samples == 500
        assert result.n_dims == 50

    def test_all_components_computed(self, random_data):
        """Test that all components are computed by default."""
        result = characterize_geometry(random_data, "test")

        # Basic stats
        assert result.total_variance is not None
        assert result.sparsity is not None

        # Intrinsic dimension
        assert result.mle_intrinsic_dim is not None
        assert result.local_pca_intrinsic_dim is not None

        # k-NN structure
        assert result.hubness_score is not None
        assert result.distance_distribution is not None

    def test_selective_computation(self, random_data):
        """Test selective computation based on config."""
        config = GeometryConfig(
            compute_basic_stats=True,
            compute_intrinsic_dim=False,
            compute_knn_structure=False
        )
        result = characterize_geometry(random_data, "test", config)

        assert result.total_variance is not None
        assert result.mle_intrinsic_dim is None
        assert result.hubness_score is None

    def test_to_dict(self, random_data):
        """Test to_dict conversion."""
        result = characterize_geometry(random_data, "test")
        d = result.to_dict()

        assert isinstance(d, dict)
        assert d["layer_name"] == "test"
        assert "mle_intrinsic_dim" in d
        assert "hubness_score" in d

    def test_subsampling(self):
        """Test that large data is subsampled."""
        np.random.seed(42)
        large_data = np.random.randn(10000, 50).astype(np.float32)
        config = GeometryConfig(max_samples_for_dim=1000)

        # Should not raise memory error
        result = characterize_geometry(large_data, "test", config)
        assert result.mle_intrinsic_dim is not None


# =============================================================================
# Summary and Recommendation Tests
# =============================================================================

class TestSummarizeGeometry:
    """Tests for geometry summary functions."""

    def test_summarize_returns_dataframe(self, random_data):
        """Test that summarize returns a DataFrame."""
        results = [
            characterize_geometry(random_data, f"layer_{i}")
            for i in range(3)
        ]
        df = summarize_geometry(results)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_summary_columns(self, random_data):
        """Test that summary has expected columns."""
        results = [characterize_geometry(random_data, "test")]
        df = summarize_geometry(results)

        expected_cols = [
            "layer", "n_samples", "n_dims", "total_variance",
            "mle_intrinsic_dim", "hubness"
        ]
        for col in expected_cols:
            assert col in df.columns


class TestRecommendPCAComponents:
    """Tests for PCA component recommendation."""

    def test_returns_int(self, random_data):
        """Test that recommendation returns an integer."""
        results = [characterize_geometry(random_data, "test")]
        n_components = recommend_pca_components(results)
        assert isinstance(n_components, int)

    def test_minimum_20(self, low_rank_data):
        """Test that minimum recommendation is 20."""
        results = [characterize_geometry(low_rank_data, "test")]
        n_components = recommend_pca_components(results)
        assert n_components >= 20

    def test_based_on_mle(self, random_data):
        """Test that recommendation considers MLE dimension."""
        results = [characterize_geometry(random_data, "test")]
        n_components = recommend_pca_components(results)

        # Should be at least 2x MLE dim
        mle_dim = results[0].mle_intrinsic_dim
        if mle_dim is not None:
            assert n_components >= int(2 * mle_dim) or n_components >= 20


# =============================================================================
# Integration Tests
# =============================================================================

class TestGeometryIntegration:
    """Integration tests for geometry characterization."""

    def test_multi_layer_analysis(self, random_data, low_rank_data):
        """Test analyzing multiple layers."""
        results = [
            characterize_geometry(random_data, "layer_0"),
            characterize_geometry(low_rank_data, "layer_1"),
        ]

        df = summarize_geometry(results)
        assert len(df) == 2

        # Low-rank layer should have lower intrinsic dim
        layer0_dim = df[df["layer"] == "layer_0"]["mle_intrinsic_dim"].values[0]
        layer1_dim = df[df["layer"] == "layer_1"]["mle_intrinsic_dim"].values[0]
        assert layer1_dim < layer0_dim

    def test_reproducibility(self, random_data):
        """Test that results are reproducible with same seed."""
        config = GeometryConfig(seed=42)

        result1 = characterize_geometry(random_data, "test", config)
        result2 = characterize_geometry(random_data, "test", config)

        assert result1.mle_intrinsic_dim == result2.mle_intrinsic_dim
        assert result1.hubness_score == result2.hubness_score

    def test_edge_cases(self):
        """Test edge cases with small/unusual data."""
        # Very small data
        small = np.random.randn(10, 5).astype(np.float32)
        result = characterize_geometry(small, "small")
        assert result.n_samples == 10

        # Single sample (should not crash)
        single = np.random.randn(1, 5).astype(np.float32)
        config = GeometryConfig(
            compute_knn_structure=False  # k-NN needs multiple samples
        )
        result = characterize_geometry(single, "single", config)
        assert result.n_samples == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
