"""Tests for meta_analysis module."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from todacomm.analysis.meta_analysis import (
    find_experiment_dirs,
    load_tda_data,
    compute_model_summary,
    make_relative_path,
    generate_meta_analysis,
    generate_report_markdown,
    run_meta_analysis_cli,
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
    exp_dir = tmp_path / "experiments" / "gpt2_tda_20240101_120000"
    run_dir = exp_dir / "runs" / "run_0"
    run_dir.mkdir(parents=True)

    # Write TDA summaries
    with open(run_dir / "tda_summaries.json", "w") as f:
        json.dump(sample_tda_data, f)

    # Write metrics
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(sample_metrics, f)

    # Create visualization directory
    viz_dir = run_dir / "visualizations"
    viz_dir.mkdir()
    (viz_dir / "tda_summary.png").touch()
    (viz_dir / "layer_persistence.png").touch()

    # Create artifacts directory
    artifacts_dir = exp_dir / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "experiment_data.csv").touch()

    return exp_dir


class TestFindExperimentDirs:
    """Tests for find_experiment_dirs function."""

    def test_finds_matching_experiments(self, tmp_path):
        """Should find experiment directories matching pattern."""
        base_dir = tmp_path / "experiments"
        base_dir.mkdir()

        # Create matching directories
        (base_dir / "gpt2_tda_20240101").mkdir()
        (base_dir / "bert_tda_20240102").mkdir()
        (base_dir / "other_dir").mkdir()  # Should be ignored

        result = find_experiment_dirs(str(base_dir), ["gpt2", "bert"])
        assert "gpt2" in result
        assert "bert" in result
        assert len(result) == 2

    def test_returns_empty_if_no_match(self, tmp_path):
        """Should return empty dict if no matching experiments."""
        base_dir = tmp_path / "experiments"
        base_dir.mkdir()
        (base_dir / "gpt2_tda_20240101").mkdir()

        result = find_experiment_dirs(str(base_dir), ["nonexistent"])
        assert result == {}

    def test_returns_empty_if_no_dir(self, tmp_path):
        """Should return empty dict if experiments directory doesn't exist."""
        result = find_experiment_dirs(str(tmp_path / "nonexistent"))
        assert result == {}

    def test_returns_most_recent(self, tmp_path):
        """Should return most recent experiment per model."""
        base_dir = tmp_path / "experiments"
        base_dir.mkdir()

        # Create multiple experiments (sorted in reverse, so 2 comes before 1)
        (base_dir / "gpt2_tda_20240101").mkdir()
        (base_dir / "gpt2_tda_20240102").mkdir()

        result = find_experiment_dirs(str(base_dir), ["gpt2"])
        assert "gpt2" in result
        # Should get the more recent one
        assert "20240102" in str(result["gpt2"])

    def test_finds_all_models_if_none_specified(self, tmp_path):
        """Should find all models if no filter specified."""
        base_dir = tmp_path / "experiments"
        base_dir.mkdir()

        (base_dir / "gpt2_tda_20240101").mkdir()
        (base_dir / "bert_tda_20240102").mkdir()

        result = find_experiment_dirs(str(base_dir))
        assert "gpt2" in result
        assert "bert" in result


class TestLoadTDAData:
    """Tests for load_tda_data function."""

    def test_loads_tda_data(self, experiment_dir):
        """Should load TDA summaries from experiment directory."""
        result = load_tda_data(experiment_dir)
        assert result is not None
        assert "tda" in result
        assert "embedding" in result["tda"]

    def test_loads_metrics(self, experiment_dir):
        """Should load metrics from experiment directory."""
        result = load_tda_data(experiment_dir)
        assert "metrics" in result
        assert "perplexity" in result["metrics"]

    def test_loads_visualizations(self, experiment_dir):
        """Should collect visualization paths."""
        result = load_tda_data(experiment_dir)
        assert "visualizations" in result
        assert "tda_summary" in result["visualizations"]
        assert "layer_persistence" in result["visualizations"]

    def test_loads_data_files(self, experiment_dir):
        """Should collect data file paths."""
        result = load_tda_data(experiment_dir)
        assert "data_files" in result
        assert "tda_summaries" in result["data_files"]
        assert "experiment_data" in result["data_files"]

    def test_returns_none_if_no_tda(self, tmp_path):
        """Should return None if TDA file doesn't exist."""
        exp_dir = tmp_path / "empty_exp"
        exp_dir.mkdir()
        result = load_tda_data(exp_dir)
        assert result is None

    def test_includes_experiment_dir(self, experiment_dir):
        """Should include experiment directory path."""
        result = load_tda_data(experiment_dir)
        assert "experiment_dir" in result
        assert str(experiment_dir) in result["experiment_dir"]


class TestComputeModelSummary:
    """Tests for compute_model_summary function."""

    def test_computes_embedding_h0(self, sample_tda_data, sample_metrics):
        """Should compute embedding H0 value."""
        data = {"tda": sample_tda_data, "metrics": sample_metrics}
        result = compute_model_summary(data)
        assert result["embedding_h0"] == 15.5

    def test_finds_peak_h0(self, sample_tda_data, sample_metrics):
        """Should find peak H0 layer."""
        data = {"tda": sample_tda_data, "metrics": sample_metrics}
        result = compute_model_summary(data)
        assert result["peak_h0_layer"] == "layer_5"
        assert result["peak_h0"] == 120.8

    def test_computes_expansion_ratio(self, sample_tda_data, sample_metrics):
        """Should compute expansion ratio correctly."""
        data = {"tda": sample_tda_data, "metrics": sample_metrics}
        result = compute_model_summary(data)
        expected_ratio = 120.8 / 15.5
        assert abs(result["expansion_ratio"] - expected_ratio) < 0.01

    def test_finds_max_h1(self, sample_tda_data, sample_metrics):
        """Should find maximum H1 layer."""
        data = {"tda": sample_tda_data, "metrics": sample_metrics}
        result = compute_model_summary(data)
        assert result["max_h1_layer"] == "layer_5"
        assert result["max_h1"] == 0.85

    def test_counts_h1_layers(self, sample_tda_data, sample_metrics):
        """Should count layers with H1 features."""
        data = {"tda": sample_tda_data, "metrics": sample_metrics}
        result = compute_model_summary(data)
        assert result["h1_layers_count"] == 3  # layer_0, layer_5, final

    def test_includes_performance_metrics(self, sample_tda_data, sample_metrics):
        """Should include perplexity and accuracy."""
        data = {"tda": sample_tda_data, "metrics": sample_metrics}
        result = compute_model_summary(data)
        assert result["perplexity"] == 25.5
        assert result["accuracy"] == 0.85

    def test_handles_missing_embedding(self, sample_metrics):
        """Should handle missing embedding layer gracefully."""
        tda_data = {
            "layer_0": {"H0_total_persistence": 50.0, "H1_count": 0, "H1_total_persistence": 0.0},
            "layer_1": {"H0_total_persistence": 100.0, "H1_count": 1, "H1_total_persistence": 0.1}
        }
        data = {"tda": tda_data, "metrics": sample_metrics}
        result = compute_model_summary(data)
        assert result["embedding_h0"] == 0
        assert result["expansion_ratio"] == 0


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


class TestGenerateMetaAnalysis:
    """Tests for generate_meta_analysis function."""

    def test_returns_report_if_no_data(self, tmp_path):
        """Should return minimal report if no data found."""
        result = generate_meta_analysis(["gpt2", "bert"], base_dir=str(tmp_path))
        assert "Meta-Analysis" in result
        assert "No experiment data found" in result

    def test_generates_markdown_report(self, experiment_dir, sample_tda_data, sample_metrics):
        """Should generate markdown report."""
        base_dir = experiment_dir.parent
        result = generate_meta_analysis(["gpt2"], base_dir=str(base_dir))
        assert "# Meta-Analysis" in result
        assert "gpt2" in result

    def test_saves_to_file_if_path_provided(self, experiment_dir, tmp_path):
        """Should save report to file if output path provided."""
        base_dir = experiment_dir.parent
        output_path = str(tmp_path / "meta_analysis.md")
        generate_meta_analysis(["gpt2"], output_path=output_path, base_dir=str(base_dir))
        assert Path(output_path).exists()


class TestGenerateReportMarkdown:
    """Tests for generate_report_markdown function."""

    def test_includes_header(self):
        """Report should include header."""
        model_data = {
            "gpt2": {
                "summary": {
                    "embedding_h0": 15.5,
                    "peak_h0": 120.0,
                    "peak_h0_layer": "layer_5",
                    "expansion_ratio": 7.7,
                    "max_h1": 0.85,
                    "max_h1_layer": "layer_5",
                    "h1_layers_count": 3,
                    "total_layers": 5,
                    "perplexity": 25.5,
                    "accuracy": None
                },
                "visualizations": {},
                "data_files": {},
                "experiment_dir": "experiments/gpt2_tda"
            }
        }
        result = generate_report_markdown(model_data)
        assert "# Meta-Analysis" in result

    def test_includes_model_table(self):
        """Report should include models analyzed table."""
        model_data = {
            "gpt2": {
                "summary": {
                    "embedding_h0": 15.5,
                    "peak_h0": 120.0,
                    "peak_h0_layer": "layer_5",
                    "expansion_ratio": 7.7,
                    "max_h1": 0.85,
                    "max_h1_layer": "layer_5",
                    "h1_layers_count": 3,
                    "total_layers": 5,
                    "perplexity": 25.5,
                    "accuracy": None
                },
                "visualizations": {},
                "data_files": {},
                "experiment_dir": "experiments/gpt2_tda"
            }
        }
        result = generate_report_markdown(model_data)
        assert "Models Analyzed" in result
        assert "gpt2" in result

    def test_includes_h0_analysis(self):
        """Report should include H0 analysis section."""
        model_data = {
            "gpt2": {
                "summary": {
                    "embedding_h0": 15.5,
                    "peak_h0": 120.0,
                    "peak_h0_layer": "layer_5",
                    "expansion_ratio": 7.7,
                    "max_h1": 0.85,
                    "max_h1_layer": "layer_5",
                    "h1_layers_count": 3,
                    "total_layers": 5,
                    "perplexity": 25.5,
                    "accuracy": None
                },
                "visualizations": {},
                "data_files": {},
                "experiment_dir": "experiments/gpt2_tda"
            }
        }
        result = generate_report_markdown(model_data)
        assert "H0 (Cluster Structure)" in result

    def test_includes_h1_analysis(self):
        """Report should include H1 analysis section."""
        model_data = {
            "gpt2": {
                "summary": {
                    "embedding_h0": 15.5,
                    "peak_h0": 120.0,
                    "peak_h0_layer": "layer_5",
                    "expansion_ratio": 7.7,
                    "max_h1": 0.85,
                    "max_h1_layer": "layer_5",
                    "h1_layers_count": 3,
                    "total_layers": 5,
                    "perplexity": 25.5,
                    "accuracy": None
                },
                "visualizations": {},
                "data_files": {},
                "experiment_dir": "experiments/gpt2_tda"
            }
        }
        result = generate_report_markdown(model_data)
        assert "H1 (Cyclic Structure)" in result

    def test_includes_key_findings(self):
        """Report should include key findings section."""
        model_data = {
            "gpt2": {
                "summary": {
                    "embedding_h0": 15.5,
                    "peak_h0": 120.0,
                    "peak_h0_layer": "layer_5",
                    "expansion_ratio": 7.7,
                    "max_h1": 0.85,
                    "max_h1_layer": "layer_5",
                    "h1_layers_count": 3,
                    "total_layers": 5,
                    "perplexity": 25.5,
                    "accuracy": None
                },
                "visualizations": {},
                "data_files": {},
                "experiment_dir": "experiments/gpt2_tda"
            }
        }
        result = generate_report_markdown(model_data)
        assert "Key Findings" in result

    def test_sorts_by_perplexity(self):
        """Report should sort models by perplexity."""
        model_data = {
            "bert": {
                "summary": {
                    "embedding_h0": 15.5,
                    "peak_h0": 120.0,
                    "peak_h0_layer": "layer_5",
                    "expansion_ratio": 7.7,
                    "max_h1": 0.85,
                    "max_h1_layer": "layer_5",
                    "h1_layers_count": 3,
                    "total_layers": 5,
                    "perplexity": 30.0,  # Higher (worse)
                    "accuracy": None
                },
                "visualizations": {},
                "data_files": {},
                "experiment_dir": "experiments/bert_tda"
            },
            "gpt2": {
                "summary": {
                    "embedding_h0": 15.5,
                    "peak_h0": 120.0,
                    "peak_h0_layer": "layer_5",
                    "expansion_ratio": 7.7,
                    "max_h1": 0.85,
                    "max_h1_layer": "layer_5",
                    "h1_layers_count": 3,
                    "total_layers": 5,
                    "perplexity": 20.0,  # Lower (better)
                    "accuracy": None
                },
                "visualizations": {},
                "data_files": {},
                "experiment_dir": "experiments/gpt2_tda"
            }
        }
        result = generate_report_markdown(model_data)
        # GPT-2 should appear first in sorted tables
        gpt2_pos = result.find("gpt2")
        bert_pos = result.find("bert")
        # In models analyzed table (first occurrence), gpt2 should come first
        assert gpt2_pos < bert_pos


class TestRunMetaAnalysisCLI:
    """Tests for run_meta_analysis_cli function."""

    def test_returns_output_path(self, tmp_path, monkeypatch):
        """Should return the output path."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "experiments").mkdir()

        result = run_meta_analysis_cli(["gpt2", "bert"])
        assert "meta_analysis" in result
        assert result.endswith(".md")

    def test_uses_custom_output_name(self, tmp_path, monkeypatch):
        """Should use custom output name if provided."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "experiments").mkdir()

        result = run_meta_analysis_cli(["gpt2", "bert"], output_name="custom_analysis")
        assert "custom_analysis" in result

    def test_creates_experiments_dir(self, tmp_path, monkeypatch):
        """Output path should be in experiments directory."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "experiments").mkdir()

        result = run_meta_analysis_cli(["gpt2"])
        assert "experiments" in result
