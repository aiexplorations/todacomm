"""Tests for CLI module."""

import argparse
import pytest
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import yaml

from todacomm.cli import (
    SUPPORTED_MODELS,
    get_all_layers,
    get_pool_strategy,
    create_parser,
    cmd_list_models,
    cmd_list_configs,
    cmd_init,
    cmd_run,
    cmd_compare,
    run_single_model,
    main,
)


class TestSupportedModels:
    """Tests for SUPPORTED_MODELS configuration."""

    def test_supported_models_not_empty(self):
        """SUPPORTED_MODELS should contain models."""
        assert len(SUPPORTED_MODELS) > 0

    def test_gpt2_in_supported_models(self):
        """GPT-2 should be in supported models."""
        assert "gpt2" in SUPPORTED_MODELS

    def test_bert_in_supported_models(self):
        """BERT should be in supported models."""
        assert "bert" in SUPPORTED_MODELS

    def test_model_has_required_fields(self):
        """Each model should have required configuration fields."""
        required_fields = ["type", "name", "task", "tokenizer", "description", "num_layers", "default_layers"]
        for model_key, model_info in SUPPORTED_MODELS.items():
            for field in required_fields:
                assert field in model_info, f"Model {model_key} missing field {field}"

    def test_model_num_layers_positive(self):
        """Each model should have positive number of layers."""
        for model_key, model_info in SUPPORTED_MODELS.items():
            assert model_info["num_layers"] > 0, f"Model {model_key} has invalid num_layers"

    def test_model_default_layers_not_empty(self):
        """Each model should have default layers specified."""
        for model_key, model_info in SUPPORTED_MODELS.items():
            assert len(model_info["default_layers"]) > 0, f"Model {model_key} has empty default_layers"

    def test_model_default_layers_includes_embedding(self):
        """Default layers should include embedding layer."""
        for model_key, model_info in SUPPORTED_MODELS.items():
            assert "embedding" in model_info["default_layers"], f"Model {model_key} missing embedding in default_layers"

    def test_model_default_layers_includes_final(self):
        """Default layers should include final layer."""
        for model_key, model_info in SUPPORTED_MODELS.items():
            assert "final" in model_info["default_layers"], f"Model {model_key} missing final in default_layers"


class TestGetAllLayers:
    """Tests for get_all_layers function."""

    def test_gpt2_all_layers(self):
        """GPT-2 should have 14 layers (embedding + 12 transformer + final)."""
        layers = get_all_layers("gpt2")
        assert len(layers) == 14  # embedding + layer_0 to layer_11 + final
        assert layers[0] == "embedding"
        assert layers[-1] == "final"

    def test_distilgpt2_all_layers(self):
        """DistilGPT-2 should have 8 layers."""
        layers = get_all_layers("distilgpt2")
        assert len(layers) == 8  # embedding + layer_0 to layer_5 + final

    def test_layer_names_format(self):
        """Layer names should follow expected format."""
        layers = get_all_layers("gpt2")
        assert layers[0] == "embedding"
        assert layers[1] == "layer_0"
        assert layers[-2] == "layer_11"
        assert layers[-1] == "final"

    def test_all_layers_includes_all_transformer_blocks(self):
        """All layers should include every transformer block."""
        for model_key, model_info in SUPPORTED_MODELS.items():
            layers = get_all_layers(model_key)
            num_layers = model_info["num_layers"]
            # Should have embedding + num_layers transformer blocks + final
            assert len(layers) == num_layers + 2


class TestGetPoolStrategy:
    """Tests for get_pool_strategy function."""

    def test_gpt2_uses_last_pooling(self):
        """GPT-2 (decoder) should use 'last' pooling."""
        assert get_pool_strategy("gpt2") == "last"

    def test_bert_uses_cls_pooling(self):
        """BERT (encoder) should use 'cls' pooling."""
        assert get_pool_strategy("bert") == "cls"

    def test_distilbert_uses_cls_pooling(self):
        """DistilBERT (encoder) should use 'cls' pooling."""
        assert get_pool_strategy("distilbert") == "cls"

    def test_pythia_uses_last_pooling(self):
        """Pythia (decoder) should use 'last' pooling."""
        assert get_pool_strategy("pythia") == "last"

    def test_opt_uses_last_pooling(self):
        """OPT (decoder) should use 'last' pooling."""
        assert get_pool_strategy("opt") == "last"

    def test_case_insensitive(self):
        """Pool strategy lookup should be case insensitive."""
        assert get_pool_strategy("BERT") == "cls"
        assert get_pool_strategy("GPT2") == "last"

    def test_unknown_model_uses_last(self):
        """Unknown model types should default to 'last' pooling."""
        assert get_pool_strategy("unknown_model") == "last"


class TestCreateParser:
    """Tests for argument parser creation."""

    def test_parser_creation(self):
        """Parser should be created successfully."""
        parser = create_parser()
        assert parser is not None
        assert isinstance(parser, argparse.ArgumentParser)

    def test_parser_prog_name(self):
        """Parser should have correct program name."""
        parser = create_parser()
        assert parser.prog == "todacomm"

    def test_run_command_exists(self):
        """Parser should have 'run' subcommand."""
        parser = create_parser()
        args = parser.parse_args(["run", "--model", "gpt2"])
        assert args.command == "run"
        assert args.model == "gpt2"

    def test_list_models_command_exists(self):
        """Parser should have 'list-models' subcommand."""
        parser = create_parser()
        args = parser.parse_args(["list-models"])
        assert args.command == "list-models"

    def test_list_configs_command_exists(self):
        """Parser should have 'list-configs' subcommand."""
        parser = create_parser()
        args = parser.parse_args(["list-configs"])
        assert args.command == "list-configs"

    def test_init_command_exists(self):
        """Parser should have 'init' subcommand."""
        parser = create_parser()
        args = parser.parse_args(["init", "test.yaml"])
        assert args.command == "init"
        assert args.filename == "test.yaml"

    def test_compare_command_exists(self):
        """Parser should have 'compare' subcommand."""
        parser = create_parser()
        args = parser.parse_args(["compare", "gpt2,bert"])
        assert args.command == "compare"
        assert args.models == "gpt2,bert"

    def test_run_with_config(self):
        """Run command should accept --config option."""
        parser = create_parser()
        args = parser.parse_args(["run", "--config", "test.yaml"])
        assert args.config == "test.yaml"

    def test_run_with_samples(self):
        """Run command should accept --samples option."""
        parser = create_parser()
        args = parser.parse_args(["run", "--model", "gpt2", "--samples", "500"])
        assert args.samples == 500

    def test_run_default_samples(self):
        """Run command should have default samples of 200."""
        parser = create_parser()
        args = parser.parse_args(["run", "--model", "gpt2"])
        assert args.samples == 200

    def test_run_with_layers_all(self):
        """Run command should accept --layers all option."""
        parser = create_parser()
        args = parser.parse_args(["run", "--model", "gpt2", "--layers", "all"])
        assert args.layers == "all"

    def test_run_with_layers_list(self):
        """Run command should accept comma-separated layers."""
        parser = create_parser()
        args = parser.parse_args(["run", "--model", "gpt2", "--layers", "embedding,layer_5,final"])
        assert args.layers == "embedding,layer_5,final"

    def test_run_with_device(self):
        """Run command should accept --device option."""
        parser = create_parser()
        args = parser.parse_args(["run", "--model", "gpt2", "--device", "cuda"])
        assert args.device == "cuda"

    def test_run_default_device(self):
        """Run command should default to CPU device."""
        parser = create_parser()
        args = parser.parse_args(["run", "--model", "gpt2"])
        assert args.device == "cpu"

    def test_run_with_pca(self):
        """Run command should accept --pca option."""
        parser = create_parser()
        args = parser.parse_args(["run", "--model", "gpt2", "--pca", "100"])
        assert args.pca == 100

    def test_run_default_pca(self):
        """Run command should default to 50 PCA components."""
        parser = create_parser()
        args = parser.parse_args(["run", "--model", "gpt2"])
        assert args.pca == 50

    def test_run_with_multiple_models(self):
        """Run command should accept --models option for multi-model analysis."""
        parser = create_parser()
        args = parser.parse_args(["run", "--models", "gpt2,bert,pythia-70m"])
        assert args.models == "gpt2,bert,pythia-70m"

    def test_run_with_datasets(self):
        """Run command should accept --datasets option."""
        parser = create_parser()
        args = parser.parse_args(["run", "--model", "gpt2", "--datasets", "wikitext2,squad"])
        assert args.datasets == "wikitext2,squad"

    def test_init_with_model(self):
        """Init command should accept --model option."""
        parser = create_parser()
        args = parser.parse_args(["init", "test.yaml", "--model", "bert"])
        assert args.model == "bert"

    def test_init_default_model(self):
        """Init command should default to gpt2."""
        parser = create_parser()
        args = parser.parse_args(["init", "test.yaml"])
        assert args.model == "gpt2"


class TestCmdListModels:
    """Tests for cmd_list_models function."""

    def test_returns_zero(self):
        """list-models command should return 0."""
        parser = create_parser()
        args = parser.parse_args(["list-models"])
        result = cmd_list_models(args)
        assert result == 0

    def test_prints_model_info(self, capsys):
        """list-models command should print model information."""
        parser = create_parser()
        args = parser.parse_args(["list-models"])
        cmd_list_models(args)
        captured = capsys.readouterr()
        assert "gpt2" in captured.out
        assert "bert" in captured.out
        assert "Supported Models" in captured.out


class TestCmdListConfigs:
    """Tests for cmd_list_configs function."""

    def test_returns_zero_no_configs_dir(self, tmp_path, monkeypatch):
        """list-configs should return 0 even if no configs directory."""
        monkeypatch.chdir(tmp_path)
        parser = create_parser()
        args = parser.parse_args(["list-configs"])
        result = cmd_list_configs(args)
        assert result == 0

    def test_prints_message_no_configs(self, tmp_path, monkeypatch, capsys):
        """list-configs should print helpful message if no configs."""
        monkeypatch.chdir(tmp_path)
        parser = create_parser()
        args = parser.parse_args(["list-configs"])
        cmd_list_configs(args)
        captured = capsys.readouterr()
        assert "No configs directory found" in captured.out or "No configuration files found" in captured.out

    def test_lists_yaml_files(self, tmp_path, monkeypatch, capsys):
        """list-configs should list YAML files in configs directory."""
        monkeypatch.chdir(tmp_path)
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()

        # Create a test config file
        config = {
            "experiment_name": "test_exp",
            "description": "Test experiment",
            "model": {"name": "gpt2"}
        }
        with open(configs_dir / "test.yaml", "w") as f:
            yaml.dump(config, f)

        parser = create_parser()
        args = parser.parse_args(["list-configs"])
        cmd_list_configs(args)
        captured = capsys.readouterr()
        assert "test.yaml" in captured.out


class TestCmdInit:
    """Tests for cmd_init function."""

    def test_creates_config_file(self, tmp_path, monkeypatch):
        """init command should create a config file."""
        monkeypatch.chdir(tmp_path)
        parser = create_parser()
        args = parser.parse_args(["init", "my_config.yaml", "--model", "gpt2"])
        result = cmd_init(args)

        assert result == 0
        config_path = tmp_path / "configs" / "my_config.yaml"
        assert config_path.exists()

    def test_adds_yaml_extension(self, tmp_path, monkeypatch):
        """init command should add .yaml extension if missing."""
        monkeypatch.chdir(tmp_path)
        parser = create_parser()
        args = parser.parse_args(["init", "my_config", "--model", "gpt2"])
        cmd_init(args)

        config_path = tmp_path / "configs" / "my_config.yaml"
        assert config_path.exists()

    def test_config_contains_model_info(self, tmp_path, monkeypatch):
        """Created config should contain correct model information."""
        monkeypatch.chdir(tmp_path)
        parser = create_parser()
        args = parser.parse_args(["init", "test.yaml", "--model", "bert"])
        cmd_init(args)

        config_path = tmp_path / "configs" / "test.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config["model"]["name"] == "bert-base-uncased"
        assert config["model"]["type"] == "bert"

    def test_config_uses_correct_pool_strategy(self, tmp_path, monkeypatch):
        """Created config should use correct pooling strategy for model type."""
        monkeypatch.chdir(tmp_path)
        parser = create_parser()

        # Test BERT (should use cls)
        args = parser.parse_args(["init", "bert_config.yaml", "--model", "bert"])
        cmd_init(args)
        with open(tmp_path / "configs" / "bert_config.yaml") as f:
            config = yaml.safe_load(f)
        assert config["extraction"]["pool_strategy"] == "cls"

        # Test GPT-2 (should use last)
        args = parser.parse_args(["init", "gpt2_config.yaml", "--model", "gpt2"])
        cmd_init(args)
        with open(tmp_path / "configs" / "gpt2_config.yaml") as f:
            config = yaml.safe_load(f)
        assert config["extraction"]["pool_strategy"] == "last"

    def test_config_respects_samples_arg(self, tmp_path, monkeypatch):
        """Created config should use specified number of samples."""
        monkeypatch.chdir(tmp_path)
        parser = create_parser()
        args = parser.parse_args(["init", "test.yaml", "--samples", "500"])
        cmd_init(args)

        with open(tmp_path / "configs" / "test.yaml") as f:
            config = yaml.safe_load(f)

        assert config["dataset"]["num_samples"] == 500
        assert config["extraction"]["max_samples"] == 500

    def test_warns_on_existing_file(self, tmp_path, monkeypatch):
        """init command should warn if file already exists."""
        monkeypatch.chdir(tmp_path)
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        existing_file = configs_dir / "test.yaml"
        existing_file.write_text("existing content")

        parser = create_parser()
        args = parser.parse_args(["init", "test.yaml"])

        # Mock input to return 'n' (don't overwrite)
        with patch('builtins.input', return_value='n'):
            result = cmd_init(args)
            assert result == 1


class TestMain:
    """Tests for main entry point."""

    def test_no_command_prints_help(self, capsys):
        """Running with no command should print help."""
        with patch('sys.argv', ['todacomm']):
            result = main()
        assert result == 0

    def test_list_models_command(self, capsys):
        """list-models command should work through main."""
        with patch('sys.argv', ['todacomm', 'list-models']):
            result = main()
        assert result == 0
        captured = capsys.readouterr()
        assert "gpt2" in captured.out

    def test_init_command(self, tmp_path, monkeypatch):
        """init command should work through main."""
        monkeypatch.chdir(tmp_path)
        with patch('sys.argv', ['todacomm', 'init', 'test.yaml']):
            result = main()
        assert result == 0
        assert (tmp_path / "configs" / "test.yaml").exists()

    def test_run_without_required_args(self, capsys):
        """run command without model or config should show error."""
        with patch('sys.argv', ['todacomm', 'run']):
            result = main()
        assert result == 1
        captured = capsys.readouterr()
        assert "Must specify" in captured.out

    def test_run_with_invalid_config(self, tmp_path, monkeypatch, capsys):
        """run command with non-existent config should show error."""
        monkeypatch.chdir(tmp_path)
        with patch('sys.argv', ['todacomm', 'run', '--config', 'nonexistent.yaml']):
            result = main()
        assert result == 1
        captured = capsys.readouterr()
        assert "not found" in captured.out


class TestRunCommandValidation:
    """Tests for run command argument validation."""

    def test_invalid_model_name(self, capsys):
        """Run with invalid model name should show error."""
        parser = create_parser()
        # This should raise an error during parsing
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "--model", "invalid_model_xyz"])

    def test_invalid_device_name(self, capsys):
        """Run with invalid device should show error."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "--model", "gpt2", "--device", "tpu"])

    def test_invalid_dataset_name(self):
        """Run with invalid dataset should show error."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "--model", "gpt2", "--dataset", "invalid_dataset"])


class TestCmdRunWithMocking:
    """Tests for cmd_run with mocked pipeline execution."""

    def test_run_with_config_file(self, tmp_path, monkeypatch):
        """cmd_run with config file should call run_experiment."""
        monkeypatch.chdir(tmp_path)

        # Create a config file
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        config_path = configs_dir / "test.yaml"
        config = {
            "experiment_name": "test",
            "model": {"name": "gpt2", "type": "gpt2"},
            "dataset": {"name": "wikitext2"}
        }
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        parser = create_parser()
        args = parser.parse_args(["run", "--config", str(config_path)])

        mock_run_experiment = MagicMock()
        with patch('pipeline.unified_pipeline.run_experiment', mock_run_experiment):
            result = cmd_run(args)

        assert result == 0
        mock_run_experiment.assert_called_once_with(str(config_path))

    def test_run_with_single_model(self, tmp_path, monkeypatch, capsys):
        """cmd_run with single model should create temp config and run."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "configs").mkdir(exist_ok=True)

        parser = create_parser()
        args = parser.parse_args(["run", "--model", "gpt2", "--samples", "100"])

        mock_run_experiment = MagicMock()
        with patch('pipeline.unified_pipeline.run_experiment', mock_run_experiment):
            result = cmd_run(args)

        assert result == 0
        mock_run_experiment.assert_called_once()
        captured = capsys.readouterr()
        assert "gpt2" in captured.out

    def test_run_with_layers_all(self, tmp_path, monkeypatch, capsys):
        """cmd_run with --layers all should use all layers."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "configs").mkdir(exist_ok=True)

        parser = create_parser()
        args = parser.parse_args(["run", "--model", "gpt2", "--layers", "all"])

        # Capture the config before it's deleted
        captured_config = {}

        def capture_config(config_path):
            with open(config_path) as f:
                captured_config.update(yaml.safe_load(f))

        mock_run_experiment = MagicMock(side_effect=capture_config)
        with patch('pipeline.unified_pipeline.run_experiment', mock_run_experiment):
            result = cmd_run(args)

        assert result == 0
        # GPT-2 has 12 layers, so all layers should be 14 (embedding + 12 + final)
        assert len(captured_config["analysis_layers"]) == 14

    def test_run_with_specific_layers(self, tmp_path, monkeypatch):
        """cmd_run with specific layers should use those layers."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "configs").mkdir(exist_ok=True)

        parser = create_parser()
        args = parser.parse_args(["run", "--model", "gpt2", "--layers", "embedding,layer_5,final"])

        captured_config = {}

        def capture_config(config_path):
            with open(config_path) as f:
                captured_config.update(yaml.safe_load(f))

        mock_run_experiment = MagicMock(side_effect=capture_config)
        with patch('pipeline.unified_pipeline.run_experiment', mock_run_experiment):
            result = cmd_run(args)

        assert result == 0
        assert captured_config["analysis_layers"] == ["embedding", "layer_5", "final"]

    def test_run_with_custom_pca(self, tmp_path, monkeypatch):
        """cmd_run with custom PCA should use specified value."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "configs").mkdir(exist_ok=True)

        parser = create_parser()
        args = parser.parse_args(["run", "--model", "gpt2", "--pca", "100"])

        captured_config = {}

        def capture_config(config_path):
            with open(config_path) as f:
                captured_config.update(yaml.safe_load(f))

        mock_run_experiment = MagicMock(side_effect=capture_config)
        with patch('pipeline.unified_pipeline.run_experiment', mock_run_experiment):
            result = cmd_run(args)

        assert result == 0
        assert captured_config["tda"]["pca_components"] == 100

    def test_run_with_device(self, tmp_path, monkeypatch):
        """cmd_run with device should use specified device."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "configs").mkdir(exist_ok=True)

        parser = create_parser()
        args = parser.parse_args(["run", "--model", "gpt2", "--device", "mps"])

        captured_config = {}

        def capture_config(config_path):
            with open(config_path) as f:
                captured_config.update(yaml.safe_load(f))

        mock_run_experiment = MagicMock(side_effect=capture_config)
        with patch('pipeline.unified_pipeline.run_experiment', mock_run_experiment):
            result = cmd_run(args)

        assert result == 0
        assert captured_config["device"] == "mps"


class TestCmdRunMultiModel:
    """Tests for multi-model analysis."""

    def test_run_with_multiple_models(self, tmp_path, monkeypatch, capsys):
        """cmd_run with multiple models should run each and generate meta-analysis."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "configs").mkdir(exist_ok=True)
        (tmp_path / "experiments").mkdir(exist_ok=True)

        parser = create_parser()
        args = parser.parse_args(["run", "--models", "gpt2,bert"])

        mock_run_experiment = MagicMock()
        mock_meta_analysis = MagicMock(return_value="experiments/meta_analysis.md")

        with patch('pipeline.unified_pipeline.run_experiment', mock_run_experiment):
            with patch('todacomm.analysis.meta_analysis.run_meta_analysis_cli', mock_meta_analysis):
                result = cmd_run(args)

        assert result == 0
        # Should have called run_experiment twice (once per model)
        assert mock_run_experiment.call_count == 2
        captured = capsys.readouterr()
        assert "Multi-Model Analysis" in captured.out

    def test_run_with_invalid_model_in_list(self, tmp_path, monkeypatch, capsys):
        """cmd_run with invalid model in list should show error."""
        monkeypatch.chdir(tmp_path)

        parser = create_parser()
        args = parser.parse_args(["run", "--models", "gpt2,invalid_model"])

        result = cmd_run(args)
        assert result == 1
        captured = capsys.readouterr()
        assert "Unknown model" in captured.out


class TestCmdRunMultiDataset:
    """Tests for multi-dataset analysis."""

    def test_run_with_multiple_datasets(self, tmp_path, monkeypatch, capsys):
        """cmd_run with multiple datasets should run each and generate comparison."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "configs").mkdir(exist_ok=True)
        (tmp_path / "experiments").mkdir(exist_ok=True)

        parser = create_parser()
        args = parser.parse_args(["run", "--model", "gpt2", "--datasets", "wikitext2,squad"])

        mock_run_experiment = MagicMock()
        mock_dataset_comparison = MagicMock(return_value="experiments/comparison.md")

        with patch('pipeline.unified_pipeline.run_experiment', mock_run_experiment):
            with patch('todacomm.analysis.dataset_comparison.run_dataset_comparison_cli', mock_dataset_comparison):
                result = cmd_run(args)

        assert result == 0
        assert mock_run_experiment.call_count == 2
        captured = capsys.readouterr()
        assert "Multi-Dataset Analysis" in captured.out

    def test_run_with_invalid_dataset_in_list(self, tmp_path, monkeypatch, capsys):
        """cmd_run with invalid dataset in list should show error."""
        monkeypatch.chdir(tmp_path)

        parser = create_parser()
        args = parser.parse_args(["run", "--model", "gpt2", "--datasets", "wikitext2,invalid"])

        result = cmd_run(args)
        assert result == 1
        captured = capsys.readouterr()
        assert "Unknown dataset" in captured.out


class TestCmdRunCustomModel:
    """Tests for custom HuggingFace model."""

    def test_run_with_hf_model(self, tmp_path, monkeypatch, capsys):
        """cmd_run with --hf-model should work."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "configs").mkdir(exist_ok=True)

        parser = create_parser()
        args = parser.parse_args(["run", "--hf-model", "microsoft/phi-1_5", "--num-layers", "24"])

        mock_run_experiment = MagicMock()
        with patch('pipeline.unified_pipeline.run_experiment', mock_run_experiment):
            result = cmd_run(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "microsoft/phi-1_5" in captured.out

    def test_run_with_hf_model_layers_all(self, tmp_path, monkeypatch):
        """cmd_run with --hf-model and --layers all should generate all layers."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "configs").mkdir(exist_ok=True)

        parser = create_parser()
        args = parser.parse_args(["run", "--hf-model", "test/model", "--num-layers", "6", "--layers", "all"])

        captured_config = {}

        def capture_config(config_path):
            with open(config_path) as f:
                captured_config.update(yaml.safe_load(f))

        mock_run_experiment = MagicMock(side_effect=capture_config)
        with patch('pipeline.unified_pipeline.run_experiment', mock_run_experiment):
            result = cmd_run(args)

        assert result == 0
        # Should be embedding + 6 layers + final = 8
        assert len(captured_config["analysis_layers"]) == 8


class TestCmdCompare:
    """Tests for cmd_compare function."""

    def test_compare_generates_meta_analysis(self, tmp_path, monkeypatch, capsys):
        """cmd_compare should generate meta-analysis."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "experiments").mkdir(exist_ok=True)

        parser = create_parser()
        args = parser.parse_args(["compare", "gpt2,bert"])

        mock_meta_analysis = MagicMock(return_value="experiments/meta_analysis.md")

        with patch('todacomm.analysis.meta_analysis.run_meta_analysis_cli', mock_meta_analysis):
            result = cmd_compare(args)

        assert result == 0
        mock_meta_analysis.assert_called_once_with(["gpt2", "bert"], None)
        captured = capsys.readouterr()
        assert "Meta-Analysis" in captured.out

    def test_compare_with_output_name(self, tmp_path, monkeypatch):
        """cmd_compare with output name should pass it to meta-analysis."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "experiments").mkdir(exist_ok=True)

        parser = create_parser()
        args = parser.parse_args(["compare", "gpt2,bert", "--output", "my_comparison"])

        mock_meta_analysis = MagicMock(return_value="experiments/my_comparison.md")

        with patch('todacomm.analysis.meta_analysis.run_meta_analysis_cli', mock_meta_analysis):
            result = cmd_compare(args)

        assert result == 0
        mock_meta_analysis.assert_called_once_with(["gpt2", "bert"], "my_comparison")


class TestRunSingleModel:
    """Tests for run_single_model function."""

    def test_run_single_model_success(self, tmp_path, monkeypatch):
        """run_single_model should return experiment directory on success."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "configs").mkdir(exist_ok=True)

        parser = create_parser()
        args = parser.parse_args(["run", "--model", "gpt2", "--samples", "100"])

        mock_run_experiment = MagicMock()

        with patch('pipeline.unified_pipeline.run_experiment', mock_run_experiment):
            result = run_single_model("gpt2", args)

        # Should return experiment directory (though mocked)
        assert result is not None or mock_run_experiment.called

    def test_run_single_model_with_dataset_override(self, tmp_path, monkeypatch):
        """run_single_model with dataset_override should use that dataset."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "configs").mkdir(exist_ok=True)

        parser = create_parser()
        args = parser.parse_args(["run", "--model", "gpt2"])

        captured_config = {}

        def capture_config(config_path):
            with open(config_path) as f:
                captured_config.update(yaml.safe_load(f))

        mock_run_experiment = MagicMock(side_effect=capture_config)

        with patch('pipeline.unified_pipeline.run_experiment', mock_run_experiment):
            run_single_model("gpt2", args, dataset_override="squad")

        assert captured_config["dataset"]["name"] == "squad"
        assert "squad" in captured_config["experiment_name"]


class TestCmdListConfigsEdgeCases:
    """Additional edge case tests for cmd_list_configs."""

    def test_list_configs_empty_directory(self, tmp_path, monkeypatch, capsys):
        """list-configs with empty configs dir should show message."""
        monkeypatch.chdir(tmp_path)
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()

        parser = create_parser()
        args = parser.parse_args(["list-configs"])
        result = cmd_list_configs(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "No configuration files found" in captured.out

    def test_list_configs_with_invalid_yaml(self, tmp_path, monkeypatch, capsys):
        """list-configs should handle invalid YAML gracefully."""
        monkeypatch.chdir(tmp_path)
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()

        # Create an invalid YAML file
        with open(configs_dir / "invalid.yaml", "w") as f:
            f.write("invalid: yaml: content: [")

        parser = create_parser()
        args = parser.parse_args(["list-configs"])
        result = cmd_list_configs(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "error reading" in captured.out

    def test_list_configs_with_yml_extension(self, tmp_path, monkeypatch, capsys):
        """list-configs should also find .yml files."""
        monkeypatch.chdir(tmp_path)
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()

        config = {"experiment_name": "test", "model": {"name": "gpt2"}}
        with open(configs_dir / "test.yml", "w") as f:
            yaml.dump(config, f)

        parser = create_parser()
        args = parser.parse_args(["list-configs"])
        cmd_list_configs(args)

        captured = capsys.readouterr()
        assert "test.yml" in captured.out


class TestMainEdgeCases:
    """Additional edge case tests for main function."""

    def test_main_unknown_command(self, capsys):
        """main with unknown command should print help."""
        # Create a mock args object with unknown command
        with patch('sys.argv', ['todacomm']):
            with patch('todacomm.cli.create_parser') as mock_parser:
                mock_args = MagicMock()
                mock_args.command = "unknown_command"
                mock_parser.return_value.parse_args.return_value = mock_args
                result = main()

        assert result == 1

    def test_main_compare_command(self, tmp_path, monkeypatch, capsys):
        """main with compare command should work."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "experiments").mkdir(exist_ok=True)

        mock_meta_analysis = MagicMock(return_value="experiments/meta.md")

        with patch('sys.argv', ['todacomm', 'compare', 'gpt2,bert']):
            with patch('todacomm.analysis.meta_analysis.run_meta_analysis_cli', mock_meta_analysis):
                result = main()

        assert result == 0
