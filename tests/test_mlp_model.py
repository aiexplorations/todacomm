"""
Comprehensive tests for MLP model implementation.

Tests the MLPModel class implementing BaseModel interface,
including forward pass, layer extraction, configuration, and presets.
"""

import pytest
import torch
import torch.nn as nn
from typing import List

from todacomm.models.base import BaseModel, ModelConfig
from todacomm.models.mlp import (
    MLPModel,
    MLPConfig,
    PRESET_CONFIGS,
    create_mlp,
)


# =============================================================================
# MLPConfig Tests
# =============================================================================

class TestMLPConfig:
    """Tests for MLPConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MLPConfig()

        assert config.input_dim == 784
        assert config.hidden_dims == [256, 128]
        assert config.output_dim == 10
        assert config.activation == "relu"
        assert config.dropout == 0.0
        assert config.batch_norm is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = MLPConfig(
            input_dim=64,
            hidden_dims=[128, 64, 32],
            output_dim=5,
            activation="gelu",
            dropout=0.2,
            batch_norm=True,
        )

        assert config.input_dim == 64
        assert config.hidden_dims == [128, 64, 32]
        assert config.output_dim == 5
        assert config.activation == "gelu"
        assert config.dropout == 0.2
        assert config.batch_norm is True

    def test_num_layers_property(self):
        """Test num_layers property."""
        config1 = MLPConfig(hidden_dims=[256])
        assert config1.num_layers == 2  # 1 hidden + 1 output

        config2 = MLPConfig(hidden_dims=[256, 128])
        assert config2.num_layers == 3

        config3 = MLPConfig(hidden_dims=[256, 128, 64, 32])
        assert config3.num_layers == 5

    def test_to_dict(self):
        """Test to_dict serialization."""
        config = MLPConfig(input_dim=100, hidden_dims=[50, 25], output_dim=3)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["input_dim"] == 100
        assert config_dict["hidden_dims"] == [50, 25]
        assert config_dict["output_dim"] == 3

    def test_from_dict(self):
        """Test from_dict deserialization."""
        config_dict = {
            "input_dim": 100,
            "hidden_dims": [50, 25],
            "output_dim": 3,
            "activation": "tanh",
            "dropout": 0.1,
            "batch_norm": True,
        }
        config = MLPConfig.from_dict(config_dict)

        assert config.input_dim == 100
        assert config.hidden_dims == [50, 25]
        assert config.output_dim == 3
        assert config.activation == "tanh"
        assert config.dropout == 0.1
        assert config.batch_norm is True

    def test_roundtrip_serialization(self):
        """Test to_dict -> from_dict roundtrip."""
        original = MLPConfig(
            input_dim=64,
            hidden_dims=[128, 64],
            output_dim=10,
            activation="silu",
            dropout=0.3,
        )
        restored = MLPConfig.from_dict(original.to_dict())

        assert original.input_dim == restored.input_dim
        assert original.hidden_dims == restored.hidden_dims
        assert original.output_dim == restored.output_dim
        assert original.activation == restored.activation
        assert original.dropout == restored.dropout

    def test_inherits_from_model_config(self):
        """Test that MLPConfig inherits from ModelConfig."""
        config = MLPConfig()
        assert isinstance(config, ModelConfig)


# =============================================================================
# MLPModel Interface Tests
# =============================================================================

class TestMLPModelInterface:
    """Tests for MLPModel implementing BaseModel interface."""

    def test_inherits_from_base_model(self):
        """Test that MLPModel inherits from BaseModel."""
        config = MLPConfig(hidden_dims=[32, 16])
        model = MLPModel(config)

        assert isinstance(model, BaseModel)

    def test_inherits_from_nn_module(self):
        """Test that MLPModel inherits from nn.Module."""
        config = MLPConfig()
        model = MLPModel(config)

        assert isinstance(model, nn.Module)

    def test_has_required_methods(self):
        """Test that all required methods exist."""
        model = MLPModel(MLPConfig())

        assert hasattr(model, "forward")
        assert hasattr(model, "get_layer_names")
        assert hasattr(model, "extract_layer_features")
        assert hasattr(model, "forward_with_cache")
        assert hasattr(model, "validate_layer_name")


# =============================================================================
# MLPModel Forward Tests
# =============================================================================

class TestMLPModelForward:
    """Tests for forward pass."""

    def test_forward_output_shape(self):
        """Test forward output shape."""
        config = MLPConfig(input_dim=784, hidden_dims=[256, 128], output_dim=10)
        model = MLPModel(config)
        x = torch.randn(8, 784)

        output = model(x)

        assert output.shape == (8, 10)

    def test_forward_different_batch_sizes(self):
        """Test forward with different batch sizes."""
        config = MLPConfig(input_dim=784, output_dim=10)
        model = MLPModel(config)

        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 784)
            output = model(x)
            assert output.shape == (batch_size, 10)

    def test_forward_flattens_image_input(self):
        """Test that forward flattens image-like input."""
        config = MLPConfig(input_dim=784, output_dim=10)
        model = MLPModel(config)

        # Image-like input: (batch, channels, height, width)
        x = torch.randn(4, 1, 28, 28)  # 1*28*28 = 784
        output = model(x)

        assert output.shape == (4, 10)

    def test_forward_with_kwargs(self):
        """Test that forward accepts extra kwargs."""
        model = MLPModel(MLPConfig())
        x = torch.randn(4, 784)

        # Should not raise even with extra kwargs
        output = model(x, extra_param="ignored")
        assert output is not None

    def test_forward_maintains_grad(self):
        """Test that forward maintains gradient computation."""
        model = MLPModel(MLPConfig())
        x = torch.randn(4, 784, requires_grad=True)

        output = model(x)

        assert output.requires_grad

    def test_backward_works(self):
        """Test that backward pass works."""
        model = MLPModel(MLPConfig())
        x = torch.randn(4, 784)

        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None


# =============================================================================
# MLPModel Layer Names Tests
# =============================================================================

class TestMLPModelLayerNames:
    """Tests for get_layer_names method."""

    def test_returns_list(self):
        """Test that get_layer_names returns a list."""
        model = MLPModel(MLPConfig())
        names = model.get_layer_names()

        assert isinstance(names, list)

    def test_shallow_2_layer_names(self):
        """Test layer names for 2-layer (1 hidden) model."""
        config = MLPConfig(hidden_dims=[256])
        model = MLPModel(config)
        names = model.get_layer_names()

        assert names == ["input", "hidden_0", "output"]

    def test_shallow_3_layer_names(self):
        """Test layer names for 3-layer (2 hidden) model."""
        config = MLPConfig(hidden_dims=[256, 128])
        model = MLPModel(config)
        names = model.get_layer_names()

        assert names == ["input", "hidden_0", "hidden_1", "output"]

    def test_deep_6_layer_names(self):
        """Test layer names for 6-layer (5 hidden) model."""
        config = MLPConfig(hidden_dims=[256, 192, 128, 96, 64])
        model = MLPModel(config)
        names = model.get_layer_names()

        expected = ["input", "hidden_0", "hidden_1", "hidden_2", "hidden_3", "hidden_4", "output"]
        assert names == expected

    def test_layer_names_returns_copy(self):
        """Test that get_layer_names returns a copy."""
        model = MLPModel(MLPConfig())
        names1 = model.get_layer_names()
        names2 = model.get_layer_names()

        # Modifying one should not affect the other
        names1.append("extra")
        assert "extra" not in names2


# =============================================================================
# MLPModel Layer Extraction Tests
# =============================================================================

class TestMLPModelExtractLayerFeatures:
    """Tests for extract_layer_features method."""

    def test_extract_input_layer(self):
        """Test extracting input layer features."""
        config = MLPConfig(input_dim=784, hidden_dims=[256, 128])
        model = MLPModel(config)
        x = torch.randn(4, 784)

        features = model.extract_layer_features(x, "input")

        assert features.shape == (4, 784)

    def test_extract_first_hidden_layer(self):
        """Test extracting first hidden layer features."""
        config = MLPConfig(input_dim=784, hidden_dims=[256, 128])
        model = MLPModel(config)
        x = torch.randn(4, 784)

        features = model.extract_layer_features(x, "hidden_0")

        assert features.shape == (4, 256)

    def test_extract_second_hidden_layer(self):
        """Test extracting second hidden layer features."""
        config = MLPConfig(input_dim=784, hidden_dims=[256, 128])
        model = MLPModel(config)
        x = torch.randn(4, 784)

        features = model.extract_layer_features(x, "hidden_1")

        assert features.shape == (4, 128)

    def test_extract_output_layer(self):
        """Test extracting output layer features."""
        config = MLPConfig(input_dim=784, hidden_dims=[256, 128], output_dim=10)
        model = MLPModel(config)
        x = torch.randn(4, 784)

        features = model.extract_layer_features(x, "output")

        assert features.shape == (4, 10)

    def test_extract_invalid_layer_raises(self):
        """Test that extracting invalid layer raises error."""
        model = MLPModel(MLPConfig())
        x = torch.randn(4, 784)

        with pytest.raises(ValueError, match="not found"):
            model.extract_layer_features(x, "invalid_layer")

    def test_extract_with_image_input(self):
        """Test extraction with image-like input."""
        config = MLPConfig(input_dim=784, hidden_dims=[256])
        model = MLPModel(config)
        x = torch.randn(4, 1, 28, 28)  # Image input

        features = model.extract_layer_features(x, "hidden_0")

        assert features.shape == (4, 256)


# =============================================================================
# MLPModel Forward With Cache Tests
# =============================================================================

class TestMLPModelForwardWithCache:
    """Tests for forward_with_cache method."""

    def test_returns_dict(self):
        """Test that forward_with_cache returns a dictionary."""
        model = MLPModel(MLPConfig())
        x = torch.randn(4, 784)

        features = model.forward_with_cache(x)

        assert isinstance(features, dict)

    def test_extracts_all_layers_by_default(self):
        """Test forward_with_cache extracts all layers when None."""
        config = MLPConfig(hidden_dims=[256, 128])
        model = MLPModel(config)
        x = torch.randn(4, 784)

        features = model.forward_with_cache(x, layer_names=None)

        assert "input" in features
        assert "hidden_0" in features
        assert "hidden_1" in features
        assert "output" in features

    def test_extracts_specific_layers(self):
        """Test forward_with_cache with specific layers."""
        config = MLPConfig(hidden_dims=[256, 128])
        model = MLPModel(config)
        x = torch.randn(4, 784)

        features = model.forward_with_cache(x, layer_names=["hidden_0", "output"])

        assert "hidden_0" in features
        assert "output" in features
        assert "input" not in features
        assert "hidden_1" not in features

    def test_extracts_single_layer(self):
        """Test forward_with_cache with single layer."""
        model = MLPModel(MLPConfig())
        x = torch.randn(4, 784)

        features = model.forward_with_cache(x, layer_names=["hidden_1"])

        assert len(features) == 1
        assert "hidden_1" in features

    def test_empty_layer_list(self):
        """Test forward_with_cache with empty layer list."""
        model = MLPModel(MLPConfig())
        x = torch.randn(4, 784)

        features = model.forward_with_cache(x, layer_names=[])

        assert features == {}

    def test_feature_shapes(self):
        """Test that cached features have correct shapes."""
        config = MLPConfig(input_dim=784, hidden_dims=[256, 128], output_dim=10)
        model = MLPModel(config)
        x = torch.randn(4, 784)

        features = model.forward_with_cache(x)

        assert features["input"].shape == (4, 784)
        assert features["hidden_0"].shape == (4, 256)
        assert features["hidden_1"].shape == (4, 128)
        assert features["output"].shape == (4, 10)

    def test_consistency_with_extract(self):
        """Test that forward_with_cache matches extract_layer_features."""
        config = MLPConfig(hidden_dims=[256, 128])
        model = MLPModel(config)
        model.eval()
        x = torch.randn(4, 784)

        with torch.no_grad():
            cached = model.forward_with_cache(x)
            extracted = {
                name: model.extract_layer_features(x, name)
                for name in model.get_layer_names()
            }

        for name in model.get_layer_names():
            torch.testing.assert_close(cached[name], extracted[name])


# =============================================================================
# MLPModel Architecture Depth Tests
# =============================================================================

class TestMLPModelDepthConfigurations:
    """Tests for variable depth configurations (ablation study support)."""

    def test_shallow_2_layer(self):
        """Test 2-layer (1 hidden) configuration."""
        config = MLPConfig(hidden_dims=[256])
        model = MLPModel(config)

        assert len(model.layers) == 2  # input->hidden, hidden->output
        assert model.config.num_layers == 2

    def test_shallow_3_layer(self):
        """Test 3-layer (2 hidden) configuration."""
        config = MLPConfig(hidden_dims=[256, 128])
        model = MLPModel(config)

        assert len(model.layers) == 3
        assert model.config.num_layers == 3

    def test_medium_4_layer(self):
        """Test 4-layer configuration."""
        config = MLPConfig(hidden_dims=[256, 128, 64])
        model = MLPModel(config)

        assert len(model.layers) == 4
        assert model.config.num_layers == 4

    def test_deep_6_layer(self):
        """Test 6-layer configuration."""
        config = MLPConfig(hidden_dims=[256, 192, 128, 96, 64])
        model = MLPModel(config)

        assert len(model.layers) == 6
        assert model.config.num_layers == 6

    def test_very_deep_network(self):
        """Test very deep network (10 layers)."""
        config = MLPConfig(hidden_dims=[256] * 9)
        model = MLPModel(config)

        assert len(model.layers) == 10
        assert len(model.get_layer_names()) == 11  # input + 9 hidden + output


# =============================================================================
# MLPModel Activations Tests
# =============================================================================

class TestMLPModelActivations:
    """Tests for different activation functions."""

    @pytest.mark.parametrize("activation", ["relu", "tanh", "gelu", "silu"])
    def test_activation_types(self, activation):
        """Test that all activation types work."""
        config = MLPConfig(activation=activation, hidden_dims=[64])
        model = MLPModel(config)
        x = torch.randn(4, 784)

        output = model(x)

        assert output.shape == (4, 10)

    def test_relu_activation(self):
        """Test ReLU activation produces non-negative hidden values."""
        config = MLPConfig(activation="relu", hidden_dims=[64])
        model = MLPModel(config)
        x = torch.randn(4, 784)

        features = model.extract_layer_features(x, "hidden_0")

        # ReLU should produce non-negative values
        assert (features >= 0).all()

    def test_tanh_activation_range(self):
        """Test tanh activation produces values in [-1, 1]."""
        config = MLPConfig(activation="tanh", hidden_dims=[64])
        model = MLPModel(config)
        x = torch.randn(4, 784)

        features = model.extract_layer_features(x, "hidden_0")

        assert (features >= -1).all()
        assert (features <= 1).all()


# =============================================================================
# MLPModel Dropout and BatchNorm Tests
# =============================================================================

class TestMLPModelRegularization:
    """Tests for dropout and batch normalization."""

    def test_dropout_in_training_mode(self):
        """Test that dropout is applied in training mode."""
        config = MLPConfig(dropout=0.5, hidden_dims=[256])
        model = MLPModel(config)
        model.train()

        x = torch.randn(100, 784)

        # Run multiple times - with dropout, outputs should vary
        outputs = [model(x).detach() for _ in range(5)]

        # At least some outputs should differ (dropout is stochastic)
        all_same = all(torch.allclose(outputs[0], out) for out in outputs[1:])
        assert not all_same, "Dropout should cause variation in training mode"

    def test_no_dropout_in_eval_mode(self):
        """Test that dropout is not applied in eval mode."""
        config = MLPConfig(dropout=0.5, hidden_dims=[256])
        model = MLPModel(config)
        model.eval()

        x = torch.randn(4, 784)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        torch.testing.assert_close(output1, output2)

    def test_batch_norm_layers_created(self):
        """Test that batch norm layers are created when enabled."""
        config = MLPConfig(batch_norm=True, hidden_dims=[256, 128])
        model = MLPModel(config)

        assert model.batch_norms is not None
        assert len(model.batch_norms) == 2  # One for each hidden layer

    def test_no_batch_norm_by_default(self):
        """Test that batch norm is disabled by default."""
        config = MLPConfig(hidden_dims=[256, 128])
        model = MLPModel(config)

        assert model.batch_norms is None


# =============================================================================
# Preset Configurations Tests
# =============================================================================

class TestPresetConfigurations:
    """Tests for preset configuration dictionary."""

    def test_all_presets_exist(self):
        """Test that all expected presets exist."""
        expected = ["shallow_2", "shallow_3", "medium_4", "medium_5", "deep_6"]
        for preset in expected:
            assert preset in PRESET_CONFIGS

    def test_preset_depths(self):
        """Test preset depth configurations."""
        assert len(PRESET_CONFIGS["shallow_2"].hidden_dims) == 1
        assert len(PRESET_CONFIGS["shallow_3"].hidden_dims) == 2
        assert len(PRESET_CONFIGS["medium_4"].hidden_dims) == 3
        assert len(PRESET_CONFIGS["medium_5"].hidden_dims) == 4
        assert len(PRESET_CONFIGS["deep_6"].hidden_dims) == 5

    def test_presets_are_mlp_configs(self):
        """Test that presets are MLPConfig instances."""
        for name, config in PRESET_CONFIGS.items():
            assert isinstance(config, MLPConfig), f"{name} should be MLPConfig"


# =============================================================================
# create_mlp Factory Tests
# =============================================================================

class TestCreateMlpFactory:
    """Tests for create_mlp factory function."""

    def test_create_from_preset(self):
        """Test creating model from preset."""
        model = create_mlp("shallow_3")

        assert isinstance(model, MLPModel)
        assert len(model.config.hidden_dims) == 2

    def test_create_with_overrides(self):
        """Test creating model with config overrides."""
        model = create_mlp("shallow_3", input_dim=64, output_dim=5)

        assert model.config.input_dim == 64
        assert model.config.output_dim == 5
        assert model.config.hidden_dims == [256, 128]  # From preset

    def test_create_without_preset(self):
        """Test creating model without preset."""
        model = create_mlp(hidden_dims=[100, 50], input_dim=32, output_dim=3)

        assert model.config.hidden_dims == [100, 50]
        assert model.config.input_dim == 32
        assert model.config.output_dim == 3

    def test_invalid_preset_raises(self):
        """Test that invalid preset raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            create_mlp("nonexistent_preset")

    def test_create_all_presets(self):
        """Test that all presets can be instantiated."""
        for preset_name in PRESET_CONFIGS.keys():
            model = create_mlp(preset_name)
            assert isinstance(model, MLPModel)


# =============================================================================
# MLPModel Utility Methods Tests
# =============================================================================

class TestMLPModelUtilities:
    """Tests for utility methods."""

    def test_count_parameters(self):
        """Test parameter counting."""
        config = MLPConfig(input_dim=784, hidden_dims=[256, 128], output_dim=10)
        model = MLPModel(config)

        param_count = model.count_parameters()

        # Manual calculation:
        # Layer 1: 784 * 256 + 256 = 200,960
        # Layer 2: 256 * 128 + 128 = 32,896
        # Output:  128 * 10 + 10 = 1,290
        # Total: 235,146
        expected = 784 * 256 + 256 + 256 * 128 + 128 + 128 * 10 + 10
        assert param_count == expected

    def test_repr(self):
        """Test string representation."""
        config = MLPConfig(input_dim=784, hidden_dims=[256, 128], output_dim=10)
        model = MLPModel(config)

        repr_str = repr(model)

        assert "MLPModel" in repr_str
        assert "784" in repr_str
        assert "256" in repr_str
        assert "128" in repr_str
        assert "10" in repr_str
        assert "relu" in repr_str


# =============================================================================
# Integration Tests
# =============================================================================

class TestMLPModelIntegration:
    """Integration tests for MLPModel."""

    def test_full_extraction_pipeline(self):
        """Test extracting features from all layers."""
        config = MLPConfig(input_dim=784, hidden_dims=[256, 128], output_dim=10)
        model = MLPModel(config)
        x = torch.randn(8, 784)

        all_features = {}
        for layer_name in model.get_layer_names():
            features = model.extract_layer_features(x, layer_name)
            all_features[layer_name] = features

        assert len(all_features) == 4
        assert all_features["input"].shape == (8, 784)
        assert all_features["hidden_0"].shape == (8, 256)
        assert all_features["hidden_1"].shape == (8, 128)
        assert all_features["output"].shape == (8, 10)

    def test_forward_and_extract_consistency(self):
        """Test that forward and extract_layer_features are consistent."""
        config = MLPConfig(hidden_dims=[256, 128])
        model = MLPModel(config)
        model.eval()

        x = torch.randn(4, 784)

        with torch.no_grad():
            forward_output = model(x)
            extract_output = model.extract_layer_features(x, "output")

        torch.testing.assert_close(forward_output, extract_output)

    def test_eval_mode_consistency(self):
        """Test that eval mode gives consistent results."""
        model = MLPModel(MLPConfig(dropout=0.5))  # With dropout
        model.eval()

        x = torch.randn(4, 784)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        torch.testing.assert_close(output1, output2)

    def test_mnist_like_pipeline(self):
        """Test with MNIST-like dimensions."""
        config = MLPConfig(
            input_dim=784,
            hidden_dims=[256, 128],
            output_dim=10,
        )
        model = MLPModel(config)
        model.eval()

        # Simulate MNIST batch
        batch_size = 32
        x = torch.randn(batch_size, 1, 28, 28)

        # Forward pass
        output = model(x)
        assert output.shape == (batch_size, 10)

        # Feature extraction
        features = model.forward_with_cache(x)
        assert features["input"].shape == (batch_size, 784)
        assert features["hidden_0"].shape == (batch_size, 256)
        assert features["output"].shape == (batch_size, 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
