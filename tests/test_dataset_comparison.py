"""Tests for dataset_comparison module."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from todacomm.analysis.dataset_comparison import (
    make_relative_path,
    find_dataset_experiments,
    load_dataset_tda_data,
    compute_dataset_summary,
    generate_dataset_comparison,
    generate_dataset_report_markdown,
    run_dataset_comparison_cli,
)


@pytest.fixture
def sample_tda_data():
    """Sample TDA data for testing."""
    return {
        "embedding": {
            "H0_total_persistence": 15.5,
            "H0_max_lifetime": 3.2,
            "H1_count": 0,
            "H1_total_persistence": 0.0
        },
        "layer_0": {
            "H0_total_persistence": 45.2,
            "H0_max_lifetime": 8.5,
            "H1_count": 2,
            "H1_total_persistence": 0.15
        },
        "layer_5": {
            "H0_total_persistence": 120.8,
            "H0_max_lifetime": 25.3,
            "H1_count": 5,
            "H1_total_persistence": 0.85
        },
        "final": {
            "H0_total_persistence": 95.1,
            "H0_max_lifetime": 18.7,
            "H1_count": 3,
            "H1_total_persistence": 0.42
        }
    }


@pytest.fixture
def sample_metrics():
    """Sample metrics for testing."""
    return {
        "perplexity": 25.5,
        "accuracy": 0.85
    }


@pytest.fixture
def experiment_dir(tmp_path, sample_tda_data, sample_metrics):
    """Create a mock experiment directory structure."""
    exp_dir = tmp_path / "experiments" / "gpt2_wikitext2_tda_20240101_120000"
    run_dir = exp_dir / "runs" / "run_0"
    run_dir.mkdir(parents=True)

    # Write TDA summaries
    with open(run_dir / "tda_summaries.json", "w") as f:
        json.dump(sample_tda_data, f)

    # Write metrics
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(sample_metrics, f)

    # Write config
    config = {"model": {"name": "gpt2"}, "dataset": {"name": "wikitext2"}}
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(config, f)

    # Create visualization directory
    viz_dir = run_dir / "visualizations"
    viz_dir.mkdir()
    (viz_dir / "tda_summary.png").touch()

    return exp_dir


class TestMakeRelativePath:
    """Tests for make_relative_path function."""

    def test_same_directory(self, tmp_path):
        """Relative path in same directory should be just filename."""
        from_file = tmp_path / "report.md"
        to_file = tmp_path / "data.json"
        result = make_relative_path(from_file, to_file)
        assert result == "data.json"

    def test_subdirectory(self, tmp_path):
        """Relative path to subdirectory should include folder."""
        from_file = tmp_path / "report.md"
        to_file = tmp_path / "data" / "file.json"
        (tmp_path / "data").mkdir()
        result = make_relative_path(from_file, to_file)
        assert "data" in result and "file.json" in result

    def test_parent_directory(self, tmp_path):
        """Relative path to parent directory should use .."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        from_file = subdir / "report.md"
        to_file = tmp_path / "data.json"
        result = make_relative_path(from_file, to_file)
        assert ".." in result


class TestFindDatasetExperiments:
    """Tests for find_dataset_experiments function."""

    def test_finds_matching_experiments(self, tmp_path):
        """Should find experiment directories matching pattern."""
        base_dir = tmp_path / "experiments"
        base_dir.mkdir()

        # Create matching directories
        (base_dir / "gpt2_wikitext2_tda_20240101").mkdir()
        (base_dir / "gpt2_squad_tda_20240102").mkdir()

        result = find_dataset_experiments("gpt2", ["wikitext2", "squad"], str(base_dir))
        assert "wikitext2" in result
        assert "squad" in result

    def test_returns_empty_if_no_match(self, tmp_path):
        """Should return empty dict if no matching experiments."""
        base_dir = tmp_path / "experiments"
        base_dir.mkdir()

        result = find_dataset_experiments("bert", ["wikitext2"], str(base_dir))
        assert result == {}

    def test_returns_empty_if_no_dir(self, tmp_path):
        """Should return empty dict if experiments directory doesn't exist."""
        result = find_dataset_experiments("gpt2", ["wikitext2"], str(tmp_path / "nonexistent"))
        assert result == {}

    def test_returns_most_recent(self, tmp_path):
        """Should return most recent experiment per dataset."""
        base_dir = tmp_path / "experiments"
        base_dir.mkdir()

        # Create multiple experiments (sorted in reverse, so 2 comes before 1)
        (base_dir / "gpt2_wikitext2_tda_20240101").mkdir()
        (base_dir / "gpt2_wikitext2_tda_20240102").mkdir()

        result = find_dataset_experiments("gpt2", ["wikitext2"], str(base_dir))
        assert "wikitext2" in result
        # Should get the more recent one (sorted reverse = 20240102 first)
        assert "20240102" in str(result["wikitext2"])


class TestLoadDatasetTDAData:
    """Tests for load_dataset_tda_data function."""

    def test_loads_tda_data(self, experiment_dir):
        """Should load TDA summaries from experiment directory."""
        result = load_dataset_tda_data(experiment_dir)
        assert result is not None
        assert "tda" in result
        assert "embedding" in result["tda"]

    def test_loads_metrics(self, experiment_dir):
        """Should load metrics from experiment directory."""
        result = load_dataset_tda_data(experiment_dir)
        assert "metrics" in result
        assert "perplexity" in result["metrics"]

    def test_loads_visualizations(self, experiment_dir):
        """Should collect visualization paths."""
        result = load_dataset_tda_data(experiment_dir)
        assert "visualizations" in result
        assert "tda_summary" in result["visualizations"]

    def test_returns_none_if_no_tda(self, tmp_path):
        """Should return None if TDA file doesn't exist."""
        exp_dir = tmp_path / "empty_exp"
        exp_dir.mkdir()
        result = load_dataset_tda_data(exp_dir)
        assert result is None


class TestComputeDatasetSummary:
    """Tests for compute_dataset_summary function."""

    def test_computes_embedding_h0(self, sample_tda_data, sample_metrics):
        """Should compute embedding H0 value."""
        data = {"tda": sample_tda_data, "metrics": sample_metrics}
        result = compute_dataset_summary(data)
        assert result["embedding_h0"] == 15.5

    def test_finds_peak_h0(self, sample_tda_data, sample_metrics):
        """Should find peak H0 layer."""
        data = {"tda": sample_tda_data, "metrics": sample_metrics}
        result = compute_dataset_summary(data)
        assert result["peak_h0_layer"] == "layer_5"
        assert result["peak_h0"] == 120.8

    def test_computes_expansion_ratio(self, sample_tda_data, sample_metrics):
        """Should compute expansion ratio correctly."""
        data = {"tda": sample_tda_data, "metrics": sample_metrics}
        result = compute_dataset_summary(data)
        expected_ratio = 120.8 / 15.5
        assert abs(result["expansion_ratio"] - expected_ratio) < 0.01

    def test_finds_max_h1(self, sample_tda_data, sample_metrics):
        """Should find maximum H1 layer."""
        data = {"tda": sample_tda_data, "metrics": sample_metrics}
        result = compute_dataset_summary(data)
        assert result["max_h1_layer"] == "layer_5"
        assert result["max_h1"] == 0.85

    def test_counts_h1_layers(self, sample_tda_data, sample_metrics):
        """Should count layers with H1 features."""
        data = {"tda": sample_tda_data, "metrics": sample_metrics}
        result = compute_dataset_summary(data)
        assert result["h1_layers_count"] == 3  # layer_0, layer_5, final

    def test_includes_performance_metrics(self, sample_tda_data, sample_metrics):
        """Should include perplexity and accuracy."""
        data = {"tda": sample_tda_data, "metrics": sample_metrics}
        result = compute_dataset_summary(data)
        assert result["perplexity"] == 25.5
        assert result["accuracy"] == 0.85


class TestGenerateDatasetComparison:
    """Tests for generate_dataset_comparison function."""

    def test_returns_report_if_no_data(self, tmp_path):
        """Should return minimal report if no data found."""
        result = generate_dataset_comparison("gpt2", ["wikitext2"], base_dir=str(tmp_path))
        assert "Dataset Comparison" in result
        assert "No experiment data found" in result

    def test_generates_markdown_report(self, tmp_path, sample_tda_data, sample_metrics):
        """Should generate markdown report."""
        # Create experiment directory
        base_dir = tmp_path / "experiments"
        exp_dir = base_dir / "gpt2_wikitext2_tda_20240101"
        run_dir = exp_dir / "runs" / "run_0"
        run_dir.mkdir(parents=True)

        with open(run_dir / "tda_summaries.json", "w") as f:
            json.dump(sample_tda_data, f)
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(sample_metrics, f)

        result = generate_dataset_comparison("gpt2", ["wikitext2"], base_dir=str(base_dir))
        assert "# Dataset Comparison" in result
        assert "wikitext2" in result

    def test_saves_to_file_if_path_provided(self, tmp_path, sample_tda_data, sample_metrics):
        """Should save report to file if output path provided."""
        # Create experiment directory
        base_dir = tmp_path / "experiments"
        exp_dir = base_dir / "gpt2_wikitext2_tda_20240101"
        run_dir = exp_dir / "runs" / "run_0"
        run_dir.mkdir(parents=True)

        with open(run_dir / "tda_summaries.json", "w") as f:
            json.dump(sample_tda_data, f)
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(sample_metrics, f)

        output_path = str(tmp_path / "report.md")
        generate_dataset_comparison("gpt2", ["wikitext2"], output_path=output_path, base_dir=str(base_dir))

        assert Path(output_path).exists()


class TestGenerateDatasetReportMarkdown:
    """Tests for generate_dataset_report_markdown function."""

    def test_includes_header(self):
        """Report should include header with model name."""
        dataset_data = {
            "wikitext2": {
                "summary": {
                    "embedding_h0": 15.5,
                    "peak_h0": 120.0,
                    "peak_h0_layer": "layer_5",
                    "avg_h0": 70.0,
                    "expansion_ratio": 7.7,
                    "max_h1": 0.85,
                    "max_h1_layer": "layer_5",
                    "total_h1": 1.5,
                    "h1_layers_count": 3,
                    "total_layers": 5,
                    "perplexity": 25.5,
                    "accuracy": None
                },
                "visualizations": {},
                "data_files": {},
                "experiment_dir": "experiments/gpt2_wikitext2_tda"
            }
        }
        result = generate_dataset_report_markdown("gpt2", dataset_data)
        assert "# Dataset Comparison: gpt2" in result

    def test_includes_performance_table(self):
        """Report should include performance table."""
        dataset_data = {
            "wikitext2": {
                "summary": {
                    "embedding_h0": 15.5,
                    "peak_h0": 120.0,
                    "peak_h0_layer": "layer_5",
                    "avg_h0": 70.0,
                    "expansion_ratio": 7.7,
                    "max_h1": 0.85,
                    "max_h1_layer": "layer_5",
                    "total_h1": 1.5,
                    "h1_layers_count": 3,
                    "total_layers": 5,
                    "perplexity": 25.5,
                    "accuracy": None
                },
                "visualizations": {},
                "data_files": {},
                "experiment_dir": "experiments/gpt2_wikitext2_tda"
            }
        }
        result = generate_dataset_report_markdown("gpt2", dataset_data)
        assert "Performance by Dataset" in result
        assert "Perplexity" in result

    def test_includes_h0_comparison(self):
        """Report should include H0 comparison section."""
        dataset_data = {
            "wikitext2": {
                "summary": {
                    "embedding_h0": 15.5,
                    "peak_h0": 120.0,
                    "peak_h0_layer": "layer_5",
                    "avg_h0": 70.0,
                    "expansion_ratio": 7.7,
                    "max_h1": 0.85,
                    "max_h1_layer": "layer_5",
                    "total_h1": 1.5,
                    "h1_layers_count": 3,
                    "total_layers": 5,
                    "perplexity": 25.5,
                    "accuracy": None
                },
                "visualizations": {},
                "data_files": {},
                "experiment_dir": "experiments/gpt2_wikitext2_tda"
            }
        }
        result = generate_dataset_report_markdown("gpt2", dataset_data)
        assert "H0 (Cluster Structure)" in result

    def test_includes_h1_comparison(self):
        """Report should include H1 comparison section."""
        dataset_data = {
            "wikitext2": {
                "summary": {
                    "embedding_h0": 15.5,
                    "peak_h0": 120.0,
                    "peak_h0_layer": "layer_5",
                    "avg_h0": 70.0,
                    "expansion_ratio": 7.7,
                    "max_h1": 0.85,
                    "max_h1_layer": "layer_5",
                    "total_h1": 1.5,
                    "h1_layers_count": 3,
                    "total_layers": 5,
                    "perplexity": 25.5,
                    "accuracy": None
                },
                "visualizations": {},
                "data_files": {},
                "experiment_dir": "experiments/gpt2_wikitext2_tda"
            }
        }
        result = generate_dataset_report_markdown("gpt2", dataset_data)
        assert "H1 (Cyclic Structure)" in result

    def test_compares_two_datasets(self):
        """Report should compare when given two datasets."""
        dataset_data = {
            "wikitext2": {
                "summary": {
                    "embedding_h0": 15.5,
                    "peak_h0": 120.0,
                    "peak_h0_layer": "layer_5",
                    "avg_h0": 70.0,
                    "expansion_ratio": 7.7,
                    "max_h1": 0.85,
                    "max_h1_layer": "layer_5",
                    "total_h1": 1.5,
                    "h1_layers_count": 3,
                    "total_layers": 5,
                    "perplexity": 25.5,
                    "accuracy": None
                },
                "visualizations": {},
                "data_files": {},
                "experiment_dir": "experiments/gpt2_wikitext2_tda"
            },
            "squad": {
                "summary": {
                    "embedding_h0": 18.0,
                    "peak_h0": 150.0,
                    "peak_h0_layer": "layer_5",
                    "avg_h0": 80.0,
                    "expansion_ratio": 8.3,
                    "max_h1": 1.2,
                    "max_h1_layer": "layer_5",
                    "total_h1": 2.0,
                    "h1_layers_count": 4,
                    "total_layers": 5,
                    "perplexity": None,
                    "accuracy": 0.85
                },
                "visualizations": {},
                "data_files": {},
                "experiment_dir": "experiments/gpt2_squad_tda"
            }
        }
        result = generate_dataset_report_markdown("gpt2", dataset_data)
        assert "Key Observations" in result


class TestRunDatasetComparisonCLI:
    """Tests for run_dataset_comparison_cli function."""

    def test_returns_output_path(self, tmp_path, monkeypatch):
        """Should return the output path."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "experiments").mkdir()

        result = run_dataset_comparison_cli("gpt2", ["wikitext2"])
        assert "dataset_comparison" in result
        assert result.endswith(".md")

    def test_uses_custom_output_name(self, tmp_path, monkeypatch):
        """Should use custom output name if provided."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "experiments").mkdir()

        result = run_dataset_comparison_cli("gpt2", ["wikitext2"], output_name="custom_report")
        assert "custom_report" in result
