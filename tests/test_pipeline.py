"""
Integration tests for the unified pipeline.

Tests end-to-end workflow: config loading → model loading → extraction → TDA → analysis
"""

import pytest
import os
import shutil
from pathlib import Path
import yaml
import json

from pipeline.unified_pipeline import (
    ExperimentConfig,
    ExperimentRun,
    setup_experiment_directory,
    generate_run_matrix,
    execute_single_run,
    analyze_experiment_results,
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


def test_generate_run_matrix_perturbation():
    """Test run matrix generation for perturbation study."""
    config = ExperimentConfig(
        experiment_name="test",
        experiment_type="architecture_perturbation",
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
    
    assert analysis_result["status"] in ["success", "insufficient_data"]
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
