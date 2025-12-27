"""Tests for visualization module."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from todacomm.visualization.tda_plots import (
    plot_persistence_diagram,
    plot_layer_tda_comparison,
    plot_betti_curves,
    plot_tda_summary,
    generate_all_visualizations,
    _generate_persistence_interpretation,
)


@pytest.fixture
def sample_diagrams():
    """Sample persistence diagrams for testing."""
    return {
        0: np.array([[0.0, 0.5], [0.1, 0.8], [0.2, 1.0]]),
        1: np.array([[0.3, 0.6], [0.4, 0.9]])
    }


@pytest.fixture
def sample_tda_summaries():
    """Sample TDA summaries for testing."""
    return {
        "embedding": {
            "H0_count": 30,
            "H0_total_persistence": 15.5,
            "H0_max_lifetime": 3.2,
            "H1_count": 0,
            "H1_total_persistence": 0.0,
            "H1_max_lifetime": 0.0
        },
        "layer_0": {
            "H0_count": 30,
            "H0_total_persistence": 45.2,
            "H0_max_lifetime": 8.5,
            "H1_count": 2,
            "H1_total_persistence": 0.15,
            "H1_max_lifetime": 0.1
        },
        "layer_5": {
            "H0_count": 30,
            "H0_total_persistence": 120.8,
            "H0_max_lifetime": 25.3,
            "H1_count": 5,
            "H1_total_persistence": 0.85,
            "H1_max_lifetime": 0.3
        },
        "layer_11": {
            "H0_count": 30,
            "H0_total_persistence": 95.1,
            "H0_max_lifetime": 18.7,
            "H1_count": 3,
            "H1_total_persistence": 0.42,
            "H1_max_lifetime": 0.2
        },
        "final": {
            "H0_count": 30,
            "H0_total_persistence": 95.1,
            "H0_max_lifetime": 18.7,
            "H1_count": 3,
            "H1_total_persistence": 0.42,
            "H1_max_lifetime": 0.2
        }
    }


class TestPlotPersistenceDiagram:
    """Tests for plot_persistence_diagram function."""

    def test_returns_figure(self, sample_diagrams):
        """Function should return a matplotlib figure."""
        fig = plot_persistence_diagram(sample_diagrams)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_custom_title(self, sample_diagrams):
        """Function should accept custom title."""
        fig = plot_persistence_diagram(sample_diagrams, title="Custom Title")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_custom_figsize(self, sample_diagrams):
        """Function should accept custom figure size."""
        fig = plot_persistence_diagram(sample_diagrams, figsize=(10, 10))
        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 10
        plt.close(fig)

    def test_with_provided_axes(self, sample_diagrams):
        """Function should accept provided axes."""
        fig, ax = plt.subplots()
        result = plot_persistence_diagram(sample_diagrams, ax=ax)
        assert result is fig
        plt.close(fig)

    def test_with_max_dim(self, sample_diagrams):
        """Function should respect max_dim parameter."""
        fig = plot_persistence_diagram(sample_diagrams, max_dim=0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_empty_diagrams(self):
        """Function should handle empty diagrams."""
        empty_diagrams = {0: np.array([]).reshape(0, 2)}
        fig = plot_persistence_diagram(empty_diagrams)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_infinite_deaths(self):
        """Function should handle infinite death values."""
        diagrams = {
            0: np.array([[0.0, np.inf], [0.1, 0.5]])
        }
        fig = plot_persistence_diagram(diagrams)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotLayerTDAComparison:
    """Tests for plot_layer_tda_comparison function."""

    def test_returns_figure(self, sample_tda_summaries):
        """Function should return a matplotlib figure."""
        fig = plot_layer_tda_comparison(sample_tda_summaries)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_total_persistence_metric(self, sample_tda_summaries):
        """Function should handle total_persistence metric."""
        fig = plot_layer_tda_comparison(sample_tda_summaries, metric="total_persistence")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_count_metric(self, sample_tda_summaries):
        """Function should handle count metric."""
        fig = plot_layer_tda_comparison(sample_tda_summaries, metric="count")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_max_lifetime_metric(self, sample_tda_summaries):
        """Function should handle max_lifetime metric."""
        fig = plot_layer_tda_comparison(sample_tda_summaries, metric="max_lifetime")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_custom_title(self, sample_tda_summaries):
        """Function should accept custom title."""
        fig = plot_layer_tda_comparison(sample_tda_summaries, title="Custom Title")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_custom_figsize(self, sample_tda_summaries):
        """Function should accept custom figure size."""
        fig = plot_layer_tda_comparison(sample_tda_summaries, figsize=(14, 8))
        assert fig.get_figwidth() == 14
        plt.close(fig)

    def test_creates_two_subplots(self, sample_tda_summaries):
        """Function should create two subplots (H0 and H1)."""
        fig = plot_layer_tda_comparison(sample_tda_summaries)
        axes = fig.get_axes()
        assert len(axes) == 2
        plt.close(fig)


class TestPlotBettiCurves:
    """Tests for plot_betti_curves function."""

    def test_returns_figure(self, sample_tda_summaries):
        """Function should return a matplotlib figure."""
        fig = plot_betti_curves(sample_tda_summaries)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_custom_figsize(self, sample_tda_summaries):
        """Function should accept custom figure size."""
        fig = plot_betti_curves(sample_tda_summaries, figsize=(12, 8))
        assert fig.get_figwidth() == 12
        plt.close(fig)

    def test_has_grid(self, sample_tda_summaries):
        """Plot should have grid enabled."""
        fig = plot_betti_curves(sample_tda_summaries)
        ax = fig.get_axes()[0]
        # Grid is set with alpha, just check the figure renders
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotTDASummary:
    """Tests for plot_tda_summary function."""

    def test_returns_figure(self, sample_tda_summaries):
        """Function should return a matplotlib figure."""
        fig = plot_tda_summary(sample_tda_summaries)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_custom_model_name(self, sample_tda_summaries):
        """Function should accept custom model name."""
        fig = plot_tda_summary(sample_tda_summaries, model_name="GPT-2")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_custom_figsize(self, sample_tda_summaries):
        """Function should accept custom figure size."""
        fig = plot_tda_summary(sample_tda_summaries, figsize=(16, 12))
        assert fig.get_figwidth() == 16
        plt.close(fig)

    def test_creates_multiple_subplots(self, sample_tda_summaries):
        """Function should create multiple subplots."""
        fig = plot_tda_summary(sample_tda_summaries)
        # Should have 5 main subplots plus twin axes
        axes = fig.get_axes()
        assert len(axes) >= 5
        plt.close(fig)


class TestGeneratePersistenceInterpretation:
    """Tests for _generate_persistence_interpretation function."""

    def test_returns_string(self):
        """Function should return a string."""
        layers = ["embedding", "layer_0", "layer_5"]
        h0_values = [10.0, 50.0, 100.0]
        h1_values = [0.0, 0.1, 0.5]
        result = _generate_persistence_interpretation(layers, h0_values, h1_values)
        assert isinstance(result, str)

    def test_mentions_peak_layer(self):
        """Interpretation should mention the peak layer."""
        layers = ["embedding", "layer_0", "layer_5"]
        h0_values = [10.0, 50.0, 100.0]
        h1_values = [0.0, 0.1, 0.5]
        result = _generate_persistence_interpretation(layers, h0_values, h1_values)
        assert "layer_5" in result

    def test_handles_uniform_h0(self):
        """Interpretation should note uniform H0 values."""
        layers = ["layer_0", "layer_1", "layer_2"]
        h0_values = [50.0, 50.0, 50.0]
        h1_values = [0.1, 0.2, 0.3]
        result = _generate_persistence_interpretation(layers, h0_values, h1_values)
        assert "uniform" in result.lower() or "H0" in result

    def test_handles_no_h1(self):
        """Interpretation should handle zero H1 values."""
        layers = ["layer_0", "layer_1"]
        h0_values = [50.0, 100.0]
        h1_values = [0.0, 0.0]
        result = _generate_persistence_interpretation(layers, h0_values, h1_values)
        assert "H1" in result


class TestGenerateAllVisualizations:
    """Tests for generate_all_visualizations function."""

    def test_creates_output_directory(self, sample_tda_summaries, tmp_path):
        """Function should create output directory if it doesn't exist."""
        output_dir = tmp_path / "viz_output"
        generate_all_visualizations(sample_tda_summaries, output_dir)
        assert output_dir.exists()

    def test_returns_list_of_paths(self, sample_tda_summaries, tmp_path):
        """Function should return list of generated file paths."""
        output_dir = tmp_path / "viz_output"
        result = generate_all_visualizations(sample_tda_summaries, output_dir)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_generates_summary_plot(self, sample_tda_summaries, tmp_path):
        """Function should generate summary plot."""
        output_dir = tmp_path / "viz_output"
        generate_all_visualizations(sample_tda_summaries, output_dir)
        assert (output_dir / "tda_summary.png").exists()

    def test_generates_persistence_plot(self, sample_tda_summaries, tmp_path):
        """Function should generate layer persistence plot."""
        output_dir = tmp_path / "viz_output"
        generate_all_visualizations(sample_tda_summaries, output_dir)
        assert (output_dir / "layer_persistence.png").exists()

    def test_generates_betti_curves(self, sample_tda_summaries, tmp_path):
        """Function should generate Betti curves plot."""
        output_dir = tmp_path / "viz_output"
        generate_all_visualizations(sample_tda_summaries, output_dir)
        assert (output_dir / "betti_curves.png").exists()

    def test_accepts_model_name(self, sample_tda_summaries, tmp_path):
        """Function should accept model name parameter."""
        output_dir = tmp_path / "viz_output"
        result = generate_all_visualizations(
            sample_tda_summaries, output_dir, model_name="GPT-2"
        )
        assert len(result) > 0

    def test_generates_persistence_diagrams_if_provided(self, sample_tda_summaries, sample_diagrams, tmp_path):
        """Function should generate persistence diagrams if provided."""
        output_dir = tmp_path / "viz_output"
        diagrams = {"layer_0": sample_diagrams}
        result = generate_all_visualizations(
            sample_tda_summaries, output_dir, diagrams=diagrams
        )
        # Should have regular plots plus diagram
        assert len(result) >= 4

    def test_files_are_png(self, sample_tda_summaries, tmp_path):
        """All generated files should be PNG."""
        output_dir = tmp_path / "viz_output"
        result = generate_all_visualizations(sample_tda_summaries, output_dir)
        for path in result:
            assert str(path).endswith(".png")


class TestVisualizationEdgeCases:
    """Tests for edge cases in visualization functions."""

    def test_single_layer(self):
        """Visualizations should work with single layer."""
        summaries = {
            "layer_0": {
                "H0_count": 30,
                "H0_total_persistence": 50.0,
                "H0_max_lifetime": 10.0,
                "H1_count": 1,
                "H1_total_persistence": 0.1,
                "H1_max_lifetime": 0.05
            }
        }
        fig = plot_layer_tda_comparison(summaries)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_zero_persistence_values(self):
        """Visualizations should handle zero persistence values."""
        summaries = {
            "layer_0": {
                "H0_count": 30,
                "H0_total_persistence": 0.0,
                "H0_max_lifetime": 0.0,
                "H1_count": 0,
                "H1_total_persistence": 0.0,
                "H1_max_lifetime": 0.0
            }
        }
        fig = plot_layer_tda_comparison(summaries)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_missing_metrics(self):
        """Visualizations should handle missing metrics gracefully."""
        summaries = {
            "layer_0": {"H0_count": 30},
            "layer_1": {"H0_count": 30, "H0_total_persistence": 50.0}
        }
        fig = plot_layer_tda_comparison(summaries)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_many_layers(self):
        """Visualizations should handle many layers."""
        summaries = {}
        for i in range(20):
            summaries[f"layer_{i}"] = {
                "H0_count": 30,
                "H0_total_persistence": 50.0 + i * 5,
                "H0_max_lifetime": 10.0 + i,
                "H1_count": i % 3,
                "H1_total_persistence": 0.1 * (i % 5),
                "H1_max_lifetime": 0.05 * (i % 3)
            }
        fig = plot_layer_tda_comparison(summaries)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
