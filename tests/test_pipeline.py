"""
Comprehensive tests for the unified pipeline.

Test Categories:
- Configuration Tests: ExperimentConfig and ExperimentRun
- Orchestration Tests: Run matrix generation, directory setup
- Analysis Tests: Results analysis with synthetic data
- Report Tests: Report generation with synthetic data
- Integration Tests: Full pipeline execution (slow)
"""

import pytest
import os
import shutil
from pathlib import Path
import yaml
import json
import pandas as pd

from pipeline.unified_pipeline import (
    ExperimentConfig,
    ExperimentRun,
    setup_experiment_directory,
    generate_run_matrix,
    execute_single_run,
    analyze_experiment_results,
    generate_report,
    run_experiment
)


@pytest.fixture
def test_config_path(tmp_path):
    """Create a minimal test configuration."""
    config = {
        "experiment_name": "test_experiment",
        "experiment_type": "quick_test",
        "description": "Test configuration",
        "model": {
            "type": "gpt2",
            "name": "gpt2",
            "task": "lm"
        },
        "dataset": {
            "name": "wikitext2",
            "task": "lm",
            "tokenizer": "gpt2",
            "max_length": 64,
            "num_samples": 10,  # Very small for testing
            "batch_size": 2
        },
        "analysis_layers": ["embedding", "final"],
        "tda": {
            "maxdim": 1,
            "metric": "euclidean",
            "pca_components": 20,
            "sampling_strategy": "uniform",
            "max_points": 50
        },
        "extraction": {
            "max_samples": 10,
            "pool_strategy": "mean",
            "device": "cpu"
        },
        "output": {
            "generate_report": True,
            "save_artifacts": True
        },
        "device": "cpu",
        "random_seed": 42
    }
    
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return str(config_path)


@pytest.fixture
def cleanup_experiments():
    """Cleanup experiments directory after tests."""
    yield
    if Path("experiments").exists():
        shutil.rmtree("experiments")


def test_experiment_config_from_yaml(test_config_path):
    """Test loading configuration from YAML."""
    config = ExperimentConfig.from_yaml(test_config_path)
    
    assert config.experiment_name == "test_experiment"
    assert config.experiment_type == "quick_test"
    assert config.model["name"] == "gpt2"
    assert config.dataset["name"] == "wikitext2"
    assert len(config.analysis_layers) == 2


def test_experiment_config_to_dict():
    """Test configuration serialization."""
    config = ExperimentConfig(
        experiment_name="test",
        experiment_type="quick_test"
    )
    
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert config_dict["experiment_name"] == "test"


def test_setup_experiment_directory(cleanup_experiments):
    """Test experiment directory creation."""
    config = ExperimentConfig(experiment_name="test_setup")
    
    exp_dir = setup_experiment_directory(config)
    
    assert exp_dir.exists()
    assert (exp_dir / "runs").exists()
    assert (exp_dir / "reports").exists()
    assert (exp_dir / "artifacts").exists()
    assert (exp_dir / "experiment_config.yaml").exists()


def test_generate_run_matrix_single():
    """Test run matrix generation for single run."""
    config = ExperimentConfig(
        experiment_name="test",
        experiment_type="quick_test"
    )
    
    runs = generate_run_matrix(config)
    
    assert len(runs) == 1
    assert runs[0].run_id == "run_0"
    assert isinstance(runs[0], ExperimentRun)


def test_generate_run_matrix_multi_model():
    """Test run matrix generation for multi-model comparison."""
    config = ExperimentConfig(
        experiment_name="test",
        experiment_type="multi_model_comparison",
        model_variants=[
            {"type": "gpt2", "name": "gpt2", "task": "lm"},
            {"type": "gpt2", "name": "distilgpt2", "task": "lm"}
        ]
    )
    
    runs = generate_run_matrix(config)
    
    assert len(runs) == 2
    assert runs[0].model_config["name"] == "gpt2"
    assert runs[1].model_config["name"] == "distilgpt2"


@pytest.mark.slow
@pytest.mark.integration
def test_execute_single_run(test_config_path, cleanup_experiments):
    """Test executing a single experimental run."""
    config = ExperimentConfig.from_yaml(test_config_path)
    exp_dir = setup_experiment_directory(config)
    runs = generate_run_matrix(config)
    
    result = execute_single_run(runs[0], exp_dir)
    
    assert result["status"] == "success"
    assert "tda_summaries" in result
    assert "metrics" in result
    assert len(result["tda_summaries"]) == len(config.analysis_layers)
    
    # Check that files were created
    run_dir = exp_dir / "runs" / runs[0].run_id
    assert (run_dir / "run_config.json").exists()
    assert (run_dir / "activations.npz").exists()
    assert (run_dir / "tda_summaries.json").exists()
    assert (run_dir / "metrics.json").exists()


@pytest.mark.slow
@pytest.mark.integration
def test_analyze_experiment_results(test_config_path, cleanup_experiments):
    """Test experiment results analysis."""
    config = ExperimentConfig.from_yaml(test_config_path)
    exp_dir = setup_experiment_directory(config)
    runs = generate_run_matrix(config)

    # Execute runs
    run_results = []
    for run in runs:
        result = execute_single_run(run, exp_dir)
        run_results.append(result)

    # Analyze
    analysis_result = analyze_experiment_results(run_results, exp_dir, config)

    # Valid statuses: success, insufficient_data, or correlation_failed
    # (correlation_failed can happen when metric column patterns don't match)
    assert analysis_result["status"] in ["success", "insufficient_data", "correlation_failed"]
    assert (exp_dir / "artifacts" / "experiment_data.csv").exists()


@pytest.mark.slow
@pytest.mark.integration
def test_full_pipeline(test_config_path, cleanup_experiments):
    """Test full end-to-end pipeline execution."""
    exp_dir, analysis_result = run_experiment(test_config_path)
    
    assert exp_dir.exists()
    assert analysis_result is not None
    
    # Check that all expected files exist
    assert (exp_dir / "experiment_config.yaml").exists()
    assert (exp_dir / "reports" / "experiment_report.md").exists()
    assert (exp_dir / "artifacts" / "experiment_data.csv").exists()
    
    # Check report content
    report_path = exp_dir / "reports" / "experiment_report.md"
    with open(report_path, 'r') as f:
        report_content = f.read()
    
    assert "test_experiment" in report_content
    assert "Configuration" in report_content
    assert "Results" in report_content


def test_experiment_run_to_dict():
    """Test ExperimentRun serialization."""
    config = ExperimentConfig(experiment_name="test")
    run = ExperimentRun(
        run_id="test_run",
        model_config={"name": "gpt2"},
        dataset_config={"name": "wikitext2"},
        analysis_layers=["embedding"],
        tda_config={"maxdim": 1},
        extraction_config={"max_samples": 100},
        experiment_config=config
    )

    run_dict = run.to_dict()

    assert isinstance(run_dict, dict)
    assert run_dict["run_id"] == "test_run"
    assert run_dict["model_config"]["name"] == "gpt2"


# =============================================================================
# ExperimentConfig Tests
# =============================================================================

class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_default_values(self):
        """Test that defaults are set correctly."""
        config = ExperimentConfig(experiment_name="test")

        assert config.experiment_name == "test"
        assert config.experiment_type == "quick_test"
        assert config.model is not None
        assert config.dataset is not None
        assert config.analysis_layers is not None
        assert config.tda is not None

    def test_post_init_sets_defaults(self):
        """Test that __post_init__ sets nested defaults."""
        config = ExperimentConfig(experiment_name="test")

        assert config.model["type"] == "gpt2"
        assert config.dataset["name"] == "wikitext2"
        assert "embedding" in config.analysis_layers
        assert config.tda["maxdim"] == 1

    def test_custom_values_preserved(self):
        """Test that custom values are not overwritten."""
        config = ExperimentConfig(
            experiment_name="custom",
            model={"type": "bert", "name": "bert-base-uncased", "task": "classification"},
            analysis_layers=["layer_0", "layer_6"]
        )

        assert config.model["type"] == "bert"
        assert config.analysis_layers == ["layer_0", "layer_6"]

    def test_save_and_load(self, tmp_path):
        """Test saving and loading configuration."""
        config = ExperimentConfig(
            experiment_name="save_test",
            description="Testing save/load"
        )

        save_path = tmp_path / "config.yaml"
        config.save(str(save_path))

        loaded = ExperimentConfig.from_yaml(str(save_path))

        assert loaded.experiment_name == "save_test"
        assert loaded.description == "Testing save/load"

    def test_to_dict_complete(self):
        """Test that to_dict includes all fields."""
        config = ExperimentConfig(experiment_name="test")
        config_dict = config.to_dict()

        assert "experiment_name" in config_dict
        assert "experiment_type" in config_dict
        assert "model" in config_dict
        assert "dataset" in config_dict
        assert "tda" in config_dict
        assert "device" in config_dict
        assert "random_seed" in config_dict

    def test_all_experiment_types(self):
        """Test all valid experiment types."""
        for exp_type in ["quick_test", "multi_model_comparison", "training_comparison", "tda_config_sensitivity"]:
            config = ExperimentConfig(
                experiment_name="test",
                experiment_type=exp_type
            )
            assert config.experiment_type == exp_type


class TestExperimentRun:
    """Tests for ExperimentRun dataclass."""

    def test_to_dict_excludes_experiment_config(self):
        """Test that to_dict doesn't include full experiment config."""
        config = ExperimentConfig(experiment_name="test")
        run = ExperimentRun(
            run_id="run_0",
            model_config={"name": "gpt2"},
            dataset_config={"name": "wikitext2"},
            analysis_layers=["final"],
            tda_config={"maxdim": 1},
            extraction_config={},
            experiment_config=config
        )

        run_dict = run.to_dict()

        # Should not include full experiment_config (too verbose)
        assert "experiment_config" not in run_dict

    def test_all_fields_in_dict(self):
        """Test that to_dict includes all expected fields."""
        config = ExperimentConfig(experiment_name="test")
        run = ExperimentRun(
            run_id="run_0",
            model_config={"name": "gpt2"},
            dataset_config={"name": "wikitext2"},
            analysis_layers=["final"],
            tda_config={"maxdim": 1},
            extraction_config={"max_samples": 100},
            experiment_config=config
        )

        run_dict = run.to_dict()

        assert "run_id" in run_dict
        assert "model_config" in run_dict
        assert "dataset_config" in run_dict
        assert "analysis_layers" in run_dict
        assert "tda_config" in run_dict
        assert "extraction_config" in run_dict


# =============================================================================
# Run Matrix Generation Tests
# =============================================================================

class TestGenerateRunMatrix:
    """Tests for generate_run_matrix function."""

    def test_quick_test_single_run(self):
        """Test that quick_test generates single run."""
        config = ExperimentConfig(
            experiment_name="test",
            experiment_type="quick_test"
        )

        runs = generate_run_matrix(config)

        assert len(runs) == 1

    def test_multi_model_multiple_runs(self):
        """Test that multi_model_comparison generates multiple runs."""
        config = ExperimentConfig(
            experiment_name="test",
            experiment_type="multi_model_comparison",
            model_variants=[
                {"name": "gpt2", "task": "lm"},
                {"name": "distilgpt2", "task": "lm"},
                {"name": "bert-base-uncased", "task": "classification"}
            ]
        )

        runs = generate_run_matrix(config)

        assert len(runs) == 3

    def test_run_ids_are_unique(self):
        """Test that generated run IDs are unique."""
        config = ExperimentConfig(
            experiment_name="test",
            experiment_type="multi_model_comparison",
            model_variants=[
                {"name": "gpt2"},
                {"name": "distilgpt2"},
                {"name": "gpt2-medium"}
            ]
        )

        runs = generate_run_matrix(config)
        run_ids = [r.run_id for r in runs]

        assert len(run_ids) == len(set(run_ids))

    def test_dataset_tokenizer_updated(self):
        """Test that dataset tokenizer is updated to match model."""
        config = ExperimentConfig(
            experiment_name="test",
            experiment_type="multi_model_comparison",
            model_variants=[
                {"name": "bert-base-uncased", "task": "classification"}
            ]
        )

        runs = generate_run_matrix(config)

        assert runs[0].dataset_config["tokenizer"] == "bert-base-uncased"

    def test_empty_model_variants(self):
        """Test with empty model_variants falls back to single run."""
        config = ExperimentConfig(
            experiment_name="test",
            experiment_type="multi_model_comparison",
            model_variants=[]
        )

        runs = generate_run_matrix(config)

        # Should fall back to single run with default model
        assert len(runs) == 1

    def test_none_model_variants(self):
        """Test with None model_variants generates single run."""
        config = ExperimentConfig(
            experiment_name="test",
            experiment_type="multi_model_comparison",
            model_variants=None
        )

        runs = generate_run_matrix(config)

        assert len(runs) == 1


# =============================================================================
# Directory Setup Tests
# =============================================================================

class TestSetupExperimentDirectory:
    """Tests for setup_experiment_directory function."""

    def test_creates_all_subdirectories(self, cleanup_experiments):
        """Test that all required subdirectories are created."""
        config = ExperimentConfig(experiment_name="dir_test")

        exp_dir = setup_experiment_directory(config)

        assert (exp_dir / "runs").is_dir()
        assert (exp_dir / "reports").is_dir()
        assert (exp_dir / "artifacts").is_dir()

    def test_saves_config_file(self, cleanup_experiments):
        """Test that config file is saved."""
        config = ExperimentConfig(experiment_name="config_test")

        exp_dir = setup_experiment_directory(config)

        config_path = exp_dir / "experiment_config.yaml"
        assert config_path.exists()

        # Verify content
        with open(config_path, 'r') as f:
            saved_config = yaml.safe_load(f)
        assert saved_config["experiment_name"] == "config_test"

    def test_directory_name_contains_experiment_name(self, cleanup_experiments):
        """Test that directory name contains experiment name."""
        config = ExperimentConfig(experiment_name="my_experiment")

        exp_dir = setup_experiment_directory(config)

        assert "my_experiment" in str(exp_dir)

    def test_directory_name_contains_timestamp(self, cleanup_experiments):
        """Test that directory name contains timestamp."""
        config = ExperimentConfig(experiment_name="timestamp_test")

        exp_dir = setup_experiment_directory(config)

        # Should contain date pattern like 20241205
        import re
        assert re.search(r'\d{8}_\d{6}', str(exp_dir))


# =============================================================================
# Analyze Results Tests (with synthetic data)
# =============================================================================

class TestAnalyzeExperimentResults:
    """Tests for analyze_experiment_results with synthetic data."""

    @pytest.fixture
    def synthetic_run_results(self):
        """Create synthetic run results for testing."""
        return [
            {
                "run_id": "model_0_gpt2",
                "status": "success",
                "model": "gpt2",
                "tda_summaries": {
                    "embedding": {"H0_count": 10.0, "H0_total_persistence": 5.0, "H1_count": 3.0, "H1_total_persistence": 1.5},
                    "final": {"H0_count": 15.0, "H0_total_persistence": 7.0, "H1_count": 5.0, "H1_total_persistence": 2.5}
                },
                "metrics": {"perplexity": 25.0, "accuracy": 0.75}
            },
            {
                "run_id": "model_1_distilgpt2",
                "status": "success",
                "model": "distilgpt2",
                "tda_summaries": {
                    "embedding": {"H0_count": 12.0, "H0_total_persistence": 6.0, "H1_count": 4.0, "H1_total_persistence": 2.0},
                    "final": {"H0_count": 18.0, "H0_total_persistence": 8.0, "H1_count": 6.0, "H1_total_persistence": 3.0}
                },
                "metrics": {"perplexity": 30.0, "accuracy": 0.70}
            },
            {
                "run_id": "model_2_bert",
                "status": "success",
                "model": "bert",
                "tda_summaries": {
                    "embedding": {"H0_count": 8.0, "H0_total_persistence": 4.0, "H1_count": 2.0, "H1_total_persistence": 1.0},
                    "final": {"H0_count": 20.0, "H0_total_persistence": 10.0, "H1_count": 8.0, "H1_total_persistence": 4.0}
                },
                "metrics": {"perplexity": 20.0, "accuracy": 0.80}
            }
        ]

    def test_analyze_creates_csv(self, synthetic_run_results, cleanup_experiments):
        """Test that analysis creates experiment_data.csv."""
        config = ExperimentConfig(experiment_name="analyze_test")
        exp_dir = setup_experiment_directory(config)

        analyze_experiment_results(synthetic_run_results, exp_dir, config)

        csv_path = exp_dir / "artifacts" / "experiment_data.csv"
        assert csv_path.exists()

    def test_analyze_csv_content(self, synthetic_run_results, cleanup_experiments):
        """Test that CSV contains expected columns."""
        config = ExperimentConfig(experiment_name="csv_test")
        exp_dir = setup_experiment_directory(config)

        analyze_experiment_results(synthetic_run_results, exp_dir, config)

        df = pd.read_csv(exp_dir / "artifacts" / "experiment_data.csv")

        assert "run_id" in df.columns
        assert "model" in df.columns
        assert "layer" in df.columns
        assert "H0_count" in df.columns
        assert "perplexity" in df.columns

    def test_analyze_returns_status(self, synthetic_run_results, cleanup_experiments):
        """Test that analysis returns status dict."""
        config = ExperimentConfig(experiment_name="status_test")
        exp_dir = setup_experiment_directory(config)

        result = analyze_experiment_results(synthetic_run_results, exp_dir, config)

        assert "status" in result

    def test_analyze_with_failed_runs(self, cleanup_experiments):
        """Test analysis skips failed runs."""
        run_results = [
            {"run_id": "run_0", "status": "failed", "error": "Test error"},
            {
                "run_id": "run_1",
                "status": "success",
                "model": "gpt2",
                "tda_summaries": {"final": {"H0_count": 10.0}},
                "metrics": {"perplexity": 25.0}
            }
        ]
        config = ExperimentConfig(experiment_name="failed_test")
        exp_dir = setup_experiment_directory(config)

        result = analyze_experiment_results(run_results, exp_dir, config)

        # Should still work with one successful run
        assert (exp_dir / "artifacts" / "experiment_data.csv").exists()

    def test_analyze_all_failed(self, cleanup_experiments):
        """Test analysis when all runs failed."""
        run_results = [
            {"run_id": "run_0", "status": "failed", "error": "Error 1"},
            {"run_id": "run_1", "status": "failed", "error": "Error 2"}
        ]
        config = ExperimentConfig(experiment_name="all_failed_test")
        exp_dir = setup_experiment_directory(config)

        result = analyze_experiment_results(run_results, exp_dir, config)

        assert result["status"] == "no_data"

    def test_analyze_single_run(self, cleanup_experiments):
        """Test analysis with single successful run."""
        run_results = [
            {
                "run_id": "run_0",
                "status": "success",
                "model": "gpt2",
                "tda_summaries": {"final": {"H0_count": 10.0, "H1_count": 5.0}},
                "metrics": {"perplexity": 25.0, "accuracy": 0.75}
            }
        ]
        config = ExperimentConfig(experiment_name="single_run_test")
        exp_dir = setup_experiment_directory(config)

        result = analyze_experiment_results(run_results, exp_dir, config)

        assert result["status"] == "success"
        assert "note" in result  # Should note single run


# =============================================================================
# Report Generation Tests (with synthetic data)
# =============================================================================

class TestGenerateReport:
    """Tests for generate_report with synthetic data."""

    @pytest.fixture
    def synthetic_analysis_result(self):
        """Create synthetic analysis result for testing."""
        return {
            "status": "success",
            "num_runs": 3,
            "num_successful": 3,
            "correlations": [
                {"tda_feature": "H0_count", "performance_metric": "accuracy", "spearman_rho": 0.85, "p_value": 0.01},
                {"tda_feature": "H1_count", "performance_metric": "accuracy", "spearman_rho": -0.72, "p_value": 0.05},
                {"tda_feature": "H0_count", "performance_metric": "perplexity", "spearman_rho": -0.65, "p_value": 0.08}
            ]
        }

    def test_report_created(self, synthetic_analysis_result, cleanup_experiments):
        """Test that report file is created."""
        config = ExperimentConfig(
            experiment_name="report_test",
            description="Testing report generation"
        )
        exp_dir = setup_experiment_directory(config)

        generate_report(exp_dir, config, synthetic_analysis_result)

        report_path = exp_dir / "reports" / "experiment_report.md"
        assert report_path.exists()

    def test_report_contains_experiment_name(self, synthetic_analysis_result, cleanup_experiments):
        """Test that report contains experiment name."""
        config = ExperimentConfig(experiment_name="my_experiment_name")
        exp_dir = setup_experiment_directory(config)

        generate_report(exp_dir, config, synthetic_analysis_result)

        with open(exp_dir / "reports" / "experiment_report.md", 'r') as f:
            content = f.read()

        assert "my_experiment_name" in content

    def test_report_contains_correlations(self, synthetic_analysis_result, cleanup_experiments):
        """Test that report contains correlation data."""
        config = ExperimentConfig(experiment_name="corr_test")
        exp_dir = setup_experiment_directory(config)

        generate_report(exp_dir, config, synthetic_analysis_result)

        with open(exp_dir / "reports" / "experiment_report.md", 'r') as f:
            content = f.read()

        assert "H0_count" in content
        assert "0.85" in content or "0.850" in content

    def test_report_contains_configuration(self, synthetic_analysis_result, cleanup_experiments):
        """Test that report contains configuration section."""
        config = ExperimentConfig(experiment_name="config_in_report")
        exp_dir = setup_experiment_directory(config)

        generate_report(exp_dir, config, synthetic_analysis_result)

        with open(exp_dir / "reports" / "experiment_report.md", 'r') as f:
            content = f.read()

        assert "Configuration" in content
        assert "Model" in content
        assert "Dataset" in content

    def test_report_without_correlations(self, cleanup_experiments):
        """Test report generation when no correlations available."""
        analysis_result = {
            "status": "success",
            "num_runs": 1,
            "num_successful": 1,
            "note": "Single run - no correlation analysis"
        }
        config = ExperimentConfig(experiment_name="no_corr_test")
        exp_dir = setup_experiment_directory(config)

        generate_report(exp_dir, config, analysis_result)

        report_path = exp_dir / "reports" / "experiment_report.md"
        assert report_path.exists()


# =============================================================================
# Integration Tests (Slow)
# =============================================================================

@pytest.mark.slow
@pytest.mark.integration
def test_execute_single_run(test_config_path, cleanup_experiments):
    """Test executing a single experimental run."""
    config = ExperimentConfig.from_yaml(test_config_path)
    exp_dir = setup_experiment_directory(config)
    runs = generate_run_matrix(config)

    result = execute_single_run(runs[0], exp_dir)

    assert result["status"] == "success"
    assert "tda_summaries" in result
    assert "metrics" in result
    assert len(result["tda_summaries"]) == len(config.analysis_layers)

    # Check that files were created
    run_dir = exp_dir / "runs" / runs[0].run_id
    assert (run_dir / "run_config.json").exists()
    assert (run_dir / "activations.npz").exists()
    assert (run_dir / "tda_summaries.json").exists()
    assert (run_dir / "metrics.json").exists()


@pytest.mark.slow
@pytest.mark.integration
def test_analyze_experiment_results_integration(test_config_path, cleanup_experiments):
    """Test experiment results analysis (integration test)."""
    config = ExperimentConfig.from_yaml(test_config_path)
    exp_dir = setup_experiment_directory(config)
    runs = generate_run_matrix(config)

    # Execute runs
    run_results = []
    for run in runs:
        result = execute_single_run(run, exp_dir)
        run_results.append(result)

    # Analyze
    analysis_result = analyze_experiment_results(run_results, exp_dir, config)

    # Valid statuses: success, insufficient_data, or correlation_failed
    assert analysis_result["status"] in ["success", "insufficient_data", "correlation_failed"]
    assert (exp_dir / "artifacts" / "experiment_data.csv").exists()


@pytest.mark.slow
@pytest.mark.integration
def test_full_pipeline(test_config_path, cleanup_experiments):
    """Test full end-to-end pipeline execution."""
    exp_dir, analysis_result = run_experiment(test_config_path)

    assert exp_dir.exists()
    assert analysis_result is not None

    # Check that all expected files exist
    assert (exp_dir / "experiment_config.yaml").exists()
    assert (exp_dir / "reports" / "experiment_report.md").exists()
    assert (exp_dir / "artifacts" / "experiment_data.csv").exists()

    # Check report content
    report_path = exp_dir / "reports" / "experiment_report.md"
    with open(report_path, 'r') as f:
        report_content = f.read()

    assert "test_experiment" in report_content
    assert "Configuration" in report_content
    assert "Results" in report_content


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
