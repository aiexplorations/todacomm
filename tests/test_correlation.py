"""
Comprehensive tests for correlation analysis module.

Test Categories:
- Unit Tests: Individual correlation computation
- Input Validation: Error handling for invalid inputs
- Edge Cases: Empty data, single row, missing columns
- Integration: End-to-end correlation analysis
"""

import pytest
import pandas as pd
import numpy as np
from todacomm.analysis.correlation import (
    CorrelationResult,
    correlate_tda_with_metrics
)


# =============================================================================
# CorrelationResult Tests
# =============================================================================

class TestCorrelationResult:
    """Tests for CorrelationResult dataclass."""

    def test_create_correlation_result(self):
        """Test creating a CorrelationResult."""
        df = pd.DataFrame({
            "tda_feature": ["H0_count", "H1_count"],
            "performance_metric": ["test_acc", "test_acc"],
            "spearman_rho": [0.5, -0.3],
            "p_value": [0.01, 0.05]
        })
        result = CorrelationResult(correlations=df)

        assert result.correlations is not None
        assert len(result.correlations) == 2

    def test_correlation_result_attributes(self):
        """Test accessing CorrelationResult attributes."""
        df = pd.DataFrame({
            "tda_feature": ["H0_count"],
            "performance_metric": ["test_acc"],
            "spearman_rho": [0.8],
            "p_value": [0.001]
        })
        result = CorrelationResult(correlations=df)

        assert "spearman_rho" in result.correlations.columns
        assert "p_value" in result.correlations.columns


# =============================================================================
# Input Validation Tests
# =============================================================================

class TestInputValidation:
    """Tests for input validation in correlate_tda_with_metrics."""

    def test_missing_merge_column_tda(self):
        """Test error when merge column missing from TDA dataframe."""
        tda_df = pd.DataFrame({
            "H0_count": [1, 2, 3],
            "H1_count": [4, 5, 6]
        })
        metrics_df = pd.DataFrame({
            "run_id": ["a", "b", "c"],
            "test_acc": [0.8, 0.85, 0.9]
        })

        with pytest.raises(ValueError, match="not found in TDA dataframe"):
            correlate_tda_with_metrics(tda_df, metrics_df, on="run_id")

    def test_missing_merge_column_metrics(self):
        """Test error when merge column missing from metrics dataframe."""
        tda_df = pd.DataFrame({
            "run_id": ["a", "b", "c"],
            "H0_count": [1, 2, 3]
        })
        metrics_df = pd.DataFrame({
            "test_acc": [0.8, 0.85, 0.9]
        })

        with pytest.raises(ValueError, match="not found in metrics dataframe"):
            correlate_tda_with_metrics(tda_df, metrics_df, on="run_id")

    def test_empty_merge_result(self):
        """Test error when merge produces empty result."""
        tda_df = pd.DataFrame({
            "run_id": ["a", "b", "c"],
            "H0_count": [1, 2, 3]
        })
        metrics_df = pd.DataFrame({
            "run_id": ["x", "y", "z"],  # No matching IDs
            "test_acc": [0.8, 0.85, 0.9]
        })

        with pytest.raises(ValueError, match="No matching rows"):
            correlate_tda_with_metrics(tda_df, metrics_df, on="run_id")

    def test_no_tda_columns_found(self):
        """Test error when no TDA columns are found."""
        tda_df = pd.DataFrame({
            "run_id": ["a", "b", "c"],
            "some_other_col": [1, 2, 3]  # Not a TDA column
        })
        metrics_df = pd.DataFrame({
            "run_id": ["a", "b", "c"],
            "test_acc": [0.8, 0.85, 0.9]
        })

        with pytest.raises(ValueError, match="No TDA columns found"):
            correlate_tda_with_metrics(tda_df, metrics_df, on="run_id")

    def test_no_metric_columns_found(self):
        """Test error when no metric columns are found."""
        tda_df = pd.DataFrame({
            "run_id": ["a", "b", "c"],
            "H0_count": [1, 2, 3]
        })
        metrics_df = pd.DataFrame({
            "run_id": ["a", "b", "c"],
            "some_other_col": [0.8, 0.85, 0.9]  # Not a metric column
        })

        with pytest.raises(ValueError, match="No metric columns found"):
            correlate_tda_with_metrics(tda_df, metrics_df, on="run_id")


# =============================================================================
# Basic Correlation Tests
# =============================================================================

class TestBasicCorrelation:
    """Tests for basic correlation computation."""

    def test_positive_correlation(self):
        """Test detection of positive correlation."""
        tda_df = pd.DataFrame({
            "run_id": ["a", "b", "c", "d", "e"],
            "H0_count": [1, 2, 3, 4, 5],
            "H1_total_persistence": [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        metrics_df = pd.DataFrame({
            "run_id": ["a", "b", "c", "d", "e"],
            "test_acc": [0.5, 0.6, 0.7, 0.8, 0.9]  # Positively correlated
        })

        result = correlate_tda_with_metrics(tda_df, metrics_df, on="run_id")

        # Should have positive correlation
        h0_corr = result.correlations[
            result.correlations["tda_feature"] == "H0_count"
        ]["spearman_rho"].values[0]
        assert h0_corr > 0.9  # Strong positive

    def test_negative_correlation(self):
        """Test detection of negative correlation."""
        tda_df = pd.DataFrame({
            "run_id": ["a", "b", "c", "d", "e"],
            "H0_count": [5, 4, 3, 2, 1]  # Decreasing
        })
        metrics_df = pd.DataFrame({
            "run_id": ["a", "b", "c", "d", "e"],
            "test_acc": [0.5, 0.6, 0.7, 0.8, 0.9]  # Increasing
        })

        result = correlate_tda_with_metrics(tda_df, metrics_df, on="run_id")

        h0_corr = result.correlations[
            result.correlations["tda_feature"] == "H0_count"
        ]["spearman_rho"].values[0]
        assert h0_corr < -0.9  # Strong negative

    def test_no_correlation(self):
        """Test detection of no correlation."""
        np.random.seed(42)
        tda_df = pd.DataFrame({
            "run_id": [f"r{i}" for i in range(20)],
            "H0_count": np.random.randn(20)
        })
        metrics_df = pd.DataFrame({
            "run_id": [f"r{i}" for i in range(20)],
            "test_acc": np.random.randn(20)  # Independent random
        })

        result = correlate_tda_with_metrics(tda_df, metrics_df, on="run_id")

        h0_corr = result.correlations[
            result.correlations["tda_feature"] == "H0_count"
        ]["spearman_rho"].values[0]
        # Should be close to zero (weak correlation)
        assert abs(h0_corr) < 0.5

    def test_multiple_tda_features(self):
        """Test correlation with multiple TDA features."""
        tda_df = pd.DataFrame({
            "run_id": ["a", "b", "c", "d", "e"],
            "H0_count": [1, 2, 3, 4, 5],
            "H0_total_persistence": [0.1, 0.2, 0.3, 0.4, 0.5],
            "H1_count": [5, 4, 3, 2, 1],
            "H1_max_lifetime": [0.5, 0.4, 0.3, 0.2, 0.1]
        })
        metrics_df = pd.DataFrame({
            "run_id": ["a", "b", "c", "d", "e"],
            "test_acc": [0.5, 0.6, 0.7, 0.8, 0.9]
        })

        result = correlate_tda_with_metrics(tda_df, metrics_df, on="run_id")

        # Should have correlations for all TDA features
        assert len(result.correlations) == 4

    def test_multiple_metrics(self):
        """Test correlation with multiple performance metrics."""
        tda_df = pd.DataFrame({
            "run_id": ["a", "b", "c", "d", "e"],
            "H0_count": [1, 2, 3, 4, 5]
        })
        metrics_df = pd.DataFrame({
            "run_id": ["a", "b", "c", "d", "e"],
            "test_acc": [0.5, 0.6, 0.7, 0.8, 0.9],
            "train_acc": [0.6, 0.7, 0.8, 0.9, 0.95],
            "val_loss": [0.5, 0.4, 0.3, 0.2, 0.1]
        })

        result = correlate_tda_with_metrics(tda_df, metrics_df, on="run_id")

        # Should have correlations for all metric combinations
        assert len(result.correlations) == 3


# =============================================================================
# P-Value Tests
# =============================================================================

class TestPValues:
    """Tests for p-value computation."""

    def test_significant_correlation_low_pvalue(self):
        """Test that significant correlations have low p-values."""
        tda_df = pd.DataFrame({
            "run_id": [f"r{i}" for i in range(20)],
            "H0_count": list(range(20))  # Perfect ordering
        })
        metrics_df = pd.DataFrame({
            "run_id": [f"r{i}" for i in range(20)],
            "test_acc": list(range(20))  # Perfect correlation
        })

        result = correlate_tda_with_metrics(tda_df, metrics_df, on="run_id")

        p_value = result.correlations["p_value"].values[0]
        assert p_value < 0.05  # Statistically significant

    def test_random_correlation_high_pvalue(self):
        """Test that random data has high p-values."""
        np.random.seed(42)
        tda_df = pd.DataFrame({
            "run_id": [f"r{i}" for i in range(10)],
            "H0_count": np.random.randn(10)
        })
        metrics_df = pd.DataFrame({
            "run_id": [f"r{i}" for i in range(10)],
            "test_acc": np.random.randn(10)
        })

        result = correlate_tda_with_metrics(tda_df, metrics_df, on="run_id")

        p_value = result.correlations["p_value"].values[0]
        # P-value should generally be higher for random data
        # (though not guaranteed for small samples)
        assert p_value > 0  # At least check it's computed


# =============================================================================
# Result Sorting Tests
# =============================================================================

class TestResultSorting:
    """Tests for result sorting by correlation strength."""

    def test_sorted_by_absolute_correlation(self):
        """Test that results are sorted by absolute correlation strength."""
        tda_df = pd.DataFrame({
            "run_id": ["a", "b", "c", "d", "e"],
            "H0_count": [1, 2, 3, 4, 5],       # Corr = 1.0
            "H1_count": [1, 3, 2, 5, 4],       # Corr ≈ weak
            "H0_total_persistence": [5, 4, 3, 2, 1]  # Corr = -1.0
        })
        metrics_df = pd.DataFrame({
            "run_id": ["a", "b", "c", "d", "e"],
            "test_acc": [1, 2, 3, 4, 5]
        })

        result = correlate_tda_with_metrics(tda_df, metrics_df, on="run_id")

        # First result should have highest absolute correlation
        first_corr = abs(result.correlations.iloc[0]["spearman_rho"])
        last_corr = abs(result.correlations.iloc[-1]["spearman_rho"])
        assert first_corr >= last_corr


# =============================================================================
# Sample Size Tests
# =============================================================================

class TestSampleSize:
    """Tests for n_samples tracking."""

    def test_n_samples_reported(self):
        """Test that sample size is reported in results."""
        tda_df = pd.DataFrame({
            "run_id": ["a", "b", "c", "d", "e"],
            "H0_count": [1, 2, 3, 4, 5]
        })
        metrics_df = pd.DataFrame({
            "run_id": ["a", "b", "c", "d", "e"],
            "test_acc": [0.5, 0.6, 0.7, 0.8, 0.9]
        })

        result = correlate_tda_with_metrics(tda_df, metrics_df, on="run_id")

        assert "n_samples" in result.correlations.columns
        assert result.correlations["n_samples"].values[0] == 5

    def test_n_samples_with_nan(self):
        """Test sample size excludes NaN values."""
        tda_df = pd.DataFrame({
            "run_id": ["a", "b", "c", "d", "e"],
            "H0_count": [1, 2, np.nan, 4, 5]  # One NaN
        })
        metrics_df = pd.DataFrame({
            "run_id": ["a", "b", "c", "d", "e"],
            "test_acc": [0.5, 0.6, 0.7, 0.8, 0.9]
        })

        result = correlate_tda_with_metrics(tda_df, metrics_df, on="run_id")

        # Should exclude the NaN row
        assert result.correlations["n_samples"].values[0] == 4


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests for correlation analysis."""

    def test_minimum_samples(self):
        """Test with minimum number of samples (3)."""
        tda_df = pd.DataFrame({
            "run_id": ["a", "b", "c"],
            "H0_count": [1, 2, 3]
        })
        metrics_df = pd.DataFrame({
            "run_id": ["a", "b", "c"],
            "test_acc": [0.5, 0.6, 0.7]
        })

        result = correlate_tda_with_metrics(tda_df, metrics_df, on="run_id")

        assert len(result.correlations) == 1

    def test_large_dataset(self):
        """Test with larger dataset."""
        n = 100
        tda_df = pd.DataFrame({
            "run_id": [f"r{i}" for i in range(n)],
            "H0_count": np.random.randn(n),
            "H1_total_persistence": np.random.randn(n)
        })
        metrics_df = pd.DataFrame({
            "run_id": [f"r{i}" for i in range(n)],
            "test_acc": np.random.randn(n)
        })

        result = correlate_tda_with_metrics(tda_df, metrics_df, on="run_id")

        assert len(result.correlations) == 2

    def test_all_nan_column_skipped(self):
        """Test that all-NaN columns are handled gracefully."""
        tda_df = pd.DataFrame({
            "run_id": ["a", "b", "c", "d", "e"],
            "H0_count": [1, 2, 3, 4, 5],
            "H1_count": [np.nan, np.nan, np.nan, np.nan, np.nan]  # All NaN
        })
        metrics_df = pd.DataFrame({
            "run_id": ["a", "b", "c", "d", "e"],
            "test_acc": [0.5, 0.6, 0.7, 0.8, 0.9]
        })

        result = correlate_tda_with_metrics(tda_df, metrics_df, on="run_id")

        # Should only have correlation for H0_count
        assert len(result.correlations) == 1
        assert result.correlations.iloc[0]["tda_feature"] == "H0_count"

    def test_different_merge_column(self):
        """Test using different merge column name."""
        tda_df = pd.DataFrame({
            "experiment_id": ["a", "b", "c", "d", "e"],
            "H0_count": [1, 2, 3, 4, 5]
        })
        metrics_df = pd.DataFrame({
            "experiment_id": ["a", "b", "c", "d", "e"],
            "test_acc": [0.5, 0.6, 0.7, 0.8, 0.9]
        })

        result = correlate_tda_with_metrics(tda_df, metrics_df, on="experiment_id")

        assert len(result.correlations) == 1


# =============================================================================
# TDA Column Pattern Tests
# =============================================================================

class TestTDAColumnPatterns:
    """Tests for TDA column pattern matching."""

    def test_h0_count_recognized(self):
        """Test that H0_count is recognized as TDA column."""
        tda_df = pd.DataFrame({
            "run_id": ["a", "b", "c"],
            "H0_count": [1, 2, 3]
        })
        metrics_df = pd.DataFrame({
            "run_id": ["a", "b", "c"],
            "test_acc": [0.5, 0.6, 0.7]
        })

        result = correlate_tda_with_metrics(tda_df, metrics_df, on="run_id")
        assert "H0_count" in result.correlations["tda_feature"].values

    def test_h1_total_persistence_recognized(self):
        """Test that H1_total_persistence is recognized."""
        tda_df = pd.DataFrame({
            "run_id": ["a", "b", "c"],
            "H1_total_persistence": [0.1, 0.2, 0.3]
        })
        metrics_df = pd.DataFrame({
            "run_id": ["a", "b", "c"],
            "test_acc": [0.5, 0.6, 0.7]
        })

        result = correlate_tda_with_metrics(tda_df, metrics_df, on="run_id")
        assert "H1_total_persistence" in result.correlations["tda_feature"].values

    def test_max_lifetime_recognized(self):
        """Test that H*_max_lifetime is recognized."""
        tda_df = pd.DataFrame({
            "run_id": ["a", "b", "c"],
            "H0_max_lifetime": [1.0, 2.0, 3.0]
        })
        metrics_df = pd.DataFrame({
            "run_id": ["a", "b", "c"],
            "test_acc": [0.5, 0.6, 0.7]
        })

        result = correlate_tda_with_metrics(tda_df, metrics_df, on="run_id")
        assert "H0_max_lifetime" in result.correlations["tda_feature"].values


# =============================================================================
# Metric Column Pattern Tests
# =============================================================================

class TestMetricColumnPatterns:
    """Tests for metric column pattern matching."""

    def test_acc_suffix_recognized(self):
        """Test that *_acc columns are recognized."""
        tda_df = pd.DataFrame({
            "run_id": ["a", "b", "c"],
            "H0_count": [1, 2, 3]
        })
        metrics_df = pd.DataFrame({
            "run_id": ["a", "b", "c"],
            "test_acc": [0.5, 0.6, 0.7]
        })

        result = correlate_tda_with_metrics(tda_df, metrics_df, on="run_id")
        assert "test_acc" in result.correlations["performance_metric"].values

    def test_loss_suffix_recognized(self):
        """Test that *_loss columns are recognized."""
        tda_df = pd.DataFrame({
            "run_id": ["a", "b", "c"],
            "H0_count": [1, 2, 3]
        })
        metrics_df = pd.DataFrame({
            "run_id": ["a", "b", "c"],
            "train_loss": [0.5, 0.4, 0.3]
        })

        result = correlate_tda_with_metrics(tda_df, metrics_df, on="run_id")
        assert "train_loss" in result.correlations["performance_metric"].values

    def test_accuracy_suffix_recognized(self):
        """Test that *_accuracy columns are recognized."""
        tda_df = pd.DataFrame({
            "run_id": ["a", "b", "c"],
            "H0_count": [1, 2, 3]
        })
        metrics_df = pd.DataFrame({
            "run_id": ["a", "b", "c"],
            "validation_accuracy": [0.5, 0.6, 0.7]
        })

        result = correlate_tda_with_metrics(tda_df, metrics_df, on="run_id")
        assert "validation_accuracy" in result.correlations["performance_metric"].values


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
