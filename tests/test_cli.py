"""Tests for CLI module."""

import argparse
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml

from todacomm.cli import (
    SUPPORTED_MODELS,
    get_all_layers,
    get_pool_strategy,
    create_parser,
    cmd_list_models,
    cmd_list_configs,
    cmd_init,
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
