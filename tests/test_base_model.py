"""
Comprehensive tests for base model interface.

Tests the abstract BaseModel class using a concrete implementation
that provides minimal but real behavior for testing the base class logic.
"""

import pytest
import torch
import torch.nn as nn
from typing import List, Dict, Optional

from todacomm.models.base import BaseModel, ModelConfig


# =============================================================================
# Concrete Test Implementation
# =============================================================================

class SimpleTestModel(BaseModel):
    """
    Concrete implementation of BaseModel for testing.

    A minimal 2-layer network that implements all abstract methods
    with real (not mocked) behavior.
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 20, output_dim: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Real layers
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

        self._layer_names = ["embedding", "hidden", "output"]

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through the model."""
        x = self.relu(self.embedding(x))
        x = self.relu(self.hidden(x))
        x = self.output(x)
        return x

    def extract_layer_features(
        self,
        x: torch.Tensor,
        layer_name: str,
        **kwargs
    ) -> torch.Tensor:
        """Extract features from a specific layer."""
        self.validate_layer_name(layer_name)

        if layer_name == "embedding":
            return self.relu(self.embedding(x))
        elif layer_name == "hidden":
            x = self.relu(self.embedding(x))
            return self.relu(self.hidden(x))
        elif layer_name == "output":
            x = self.relu(self.embedding(x))
            x = self.relu(self.hidden(x))
            return self.output(x)
        else:
            raise ValueError(f"Unknown layer: {layer_name}")

    def get_layer_names(self) -> List[str]:
        """Get list of available layer names."""
        return self._layer_names


class SimpleTestConfig(ModelConfig):
    """Concrete implementation of ModelConfig for testing."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 20, output_dim: int = 5):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "SimpleTestConfig":
        """Create from dictionary."""
        return cls(**config_dict)


# =============================================================================
# BaseModel Tests
# =============================================================================

class TestBaseModelInterface:
    """Tests for BaseModel abstract interface."""

    def test_inherits_from_nn_module(self):
        """Test that BaseModel inherits from nn.Module."""
        model = SimpleTestModel()
        assert isinstance(model, nn.Module)

    def test_inherits_from_base_model(self):
        """Test that concrete implementation inherits from BaseModel."""
        model = SimpleTestModel()
        assert isinstance(model, BaseModel)

    def test_layer_names_initialized(self):
        """Test that _layer_names is initialized."""
        model = SimpleTestModel()
        assert hasattr(model, "_layer_names")
        assert isinstance(model._layer_names, list)


class TestBaseModelForward:
    """Tests for forward pass."""

    def test_forward_returns_tensor(self):
        """Test that forward returns a tensor."""
        model = SimpleTestModel(input_dim=10, output_dim=5)
        x = torch.randn(4, 10)
        output = model(x)

        assert isinstance(output, torch.Tensor)

    def test_forward_output_shape(self):
        """Test forward output shape."""
        model = SimpleTestModel(input_dim=10, output_dim=5)
        x = torch.randn(4, 10)
        output = model(x)

        assert output.shape == (4, 5)

    def test_forward_different_batch_sizes(self):
        """Test forward with different batch sizes."""
        model = SimpleTestModel(input_dim=10, output_dim=5)

        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 10)
            output = model(x)
            assert output.shape == (batch_size, 5)

    def test_forward_with_kwargs(self):
        """Test that forward accepts kwargs."""
        model = SimpleTestModel()
        x = torch.randn(4, 10)

        # Should not raise even with extra kwargs
        output = model(x, extra_param="ignored")
        assert output is not None


class TestBaseModelGetLayerNames:
    """Tests for get_layer_names method."""

    def test_returns_list(self):
        """Test that get_layer_names returns a list."""
        model = SimpleTestModel()
        names = model.get_layer_names()

        assert isinstance(names, list)

    def test_returns_expected_layers(self):
        """Test that expected layers are returned."""
        model = SimpleTestModel()
        names = model.get_layer_names()

        assert "embedding" in names
        assert "hidden" in names
        assert "output" in names

    def test_layer_names_not_empty(self):
        """Test that layer names list is not empty."""
        model = SimpleTestModel()
        names = model.get_layer_names()

        assert len(names) > 0


class TestBaseModelValidateLayerName:
    """Tests for validate_layer_name method."""

    def test_valid_layer_name_no_error(self):
        """Test that valid layer name doesn't raise."""
        model = SimpleTestModel()

        # Should not raise
        model.validate_layer_name("embedding")
        model.validate_layer_name("hidden")
        model.validate_layer_name("output")

    def test_invalid_layer_name_raises(self):
        """Test that invalid layer name raises ValueError."""
        model = SimpleTestModel()

        with pytest.raises(ValueError, match="not found"):
            model.validate_layer_name("nonexistent")

    def test_error_message_contains_available_layers(self):
        """Test that error message lists available layers."""
        model = SimpleTestModel()

        with pytest.raises(ValueError) as exc_info:
            model.validate_layer_name("invalid_layer")

        error_msg = str(exc_info.value)
        assert "embedding" in error_msg
        assert "hidden" in error_msg
        assert "output" in error_msg

    def test_case_sensitive_validation(self):
        """Test that layer name validation is case sensitive."""
        model = SimpleTestModel()

        with pytest.raises(ValueError):
            model.validate_layer_name("Embedding")  # Capital E

        with pytest.raises(ValueError):
            model.validate_layer_name("HIDDEN")  # All caps


class TestBaseModelExtractLayerFeatures:
    """Tests for extract_layer_features method."""

    def test_extract_embedding_layer(self):
        """Test extracting embedding layer features."""
        model = SimpleTestModel(input_dim=10, hidden_dim=20)
        x = torch.randn(4, 10)

        features = model.extract_layer_features(x, "embedding")

        assert features.shape == (4, 20)

    def test_extract_hidden_layer(self):
        """Test extracting hidden layer features."""
        model = SimpleTestModel(input_dim=10, hidden_dim=20)
        x = torch.randn(4, 10)

        features = model.extract_layer_features(x, "hidden")

        assert features.shape == (4, 20)

    def test_extract_output_layer(self):
        """Test extracting output layer features."""
        model = SimpleTestModel(input_dim=10, hidden_dim=20, output_dim=5)
        x = torch.randn(4, 10)

        features = model.extract_layer_features(x, "output")

        assert features.shape == (4, 5)

    def test_extract_invalid_layer_raises(self):
        """Test that extracting invalid layer raises error."""
        model = SimpleTestModel()
        x = torch.randn(4, 10)

        with pytest.raises(ValueError):
            model.extract_layer_features(x, "invalid")

    def test_extract_with_kwargs(self):
        """Test that extract_layer_features accepts kwargs."""
        model = SimpleTestModel()
        x = torch.randn(4, 10)

        # Should not raise even with extra kwargs
        features = model.extract_layer_features(x, "embedding", extra="ignored")
        assert features is not None


class TestBaseModelForwardWithCache:
    """Tests for forward_with_cache method."""

    def test_forward_with_cache_returns_dict(self):
        """Test that forward_with_cache returns a dictionary."""
        model = SimpleTestModel()
        x = torch.randn(4, 10)

        features = model.forward_with_cache(x)

        assert isinstance(features, dict)

    def test_forward_with_cache_all_layers(self):
        """Test forward_with_cache extracts all layers when None."""
        model = SimpleTestModel()
        x = torch.randn(4, 10)

        features = model.forward_with_cache(x, layer_names=None)

        assert "embedding" in features
        assert "hidden" in features
        assert "output" in features

    def test_forward_with_cache_specific_layers(self):
        """Test forward_with_cache with specific layers."""
        model = SimpleTestModel()
        x = torch.randn(4, 10)

        features = model.forward_with_cache(x, layer_names=["embedding", "output"])

        assert "embedding" in features
        assert "output" in features
        assert "hidden" not in features

    def test_forward_with_cache_single_layer(self):
        """Test forward_with_cache with single layer."""
        model = SimpleTestModel()
        x = torch.randn(4, 10)

        features = model.forward_with_cache(x, layer_names=["hidden"])

        assert len(features) == 1
        assert "hidden" in features

    def test_forward_with_cache_feature_shapes(self):
        """Test that cached features have correct shapes."""
        model = SimpleTestModel(input_dim=10, hidden_dim=20, output_dim=5)
        x = torch.randn(4, 10)

        features = model.forward_with_cache(x)

        assert features["embedding"].shape == (4, 20)
        assert features["hidden"].shape == (4, 20)
        assert features["output"].shape == (4, 5)

    def test_forward_with_cache_empty_list(self):
        """Test forward_with_cache with empty layer list."""
        model = SimpleTestModel()
        x = torch.randn(4, 10)

        features = model.forward_with_cache(x, layer_names=[])

        assert features == {}

    def test_forward_with_cache_with_kwargs(self):
        """Test forward_with_cache passes kwargs to extract."""
        model = SimpleTestModel()
        x = torch.randn(4, 10)

        # Should not raise
        features = model.forward_with_cache(x, layer_names=["embedding"], extra="ignored")
        assert features is not None


class TestBaseModelGradients:
    """Tests for gradient behavior."""

    def test_model_parameters_exist(self):
        """Test that model has trainable parameters."""
        model = SimpleTestModel()
        params = list(model.parameters())

        assert len(params) > 0

    def test_forward_maintains_grad(self):
        """Test that forward maintains gradient computation."""
        model = SimpleTestModel()
        x = torch.randn(4, 10, requires_grad=True)

        output = model(x)

        assert output.requires_grad

    def test_backward_works(self):
        """Test that backward pass works."""
        model = SimpleTestModel()
        x = torch.randn(4, 10)

        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None


# =============================================================================
# ModelConfig Tests
# =============================================================================

class TestModelConfig:
    """Tests for ModelConfig abstract class."""

    def test_to_dict(self):
        """Test to_dict method."""
        config = SimpleTestConfig(input_dim=15, hidden_dim=30, output_dim=8)
        config_dict = config.to_dict()

        assert config_dict["input_dim"] == 15
        assert config_dict["hidden_dim"] == 30
        assert config_dict["output_dim"] == 8

    def test_from_dict(self):
        """Test from_dict class method."""
        config_dict = {"input_dim": 20, "hidden_dim": 40, "output_dim": 10}
        config = SimpleTestConfig.from_dict(config_dict)

        assert config.input_dim == 20
        assert config.hidden_dim == 40
        assert config.output_dim == 10

    def test_roundtrip(self):
        """Test to_dict -> from_dict roundtrip."""
        original = SimpleTestConfig(input_dim=12, hidden_dim=24, output_dim=6)
        config_dict = original.to_dict()
        restored = SimpleTestConfig.from_dict(config_dict)

        assert original.input_dim == restored.input_dim
        assert original.hidden_dim == restored.hidden_dim
        assert original.output_dim == restored.output_dim

    def test_to_dict_returns_dict(self):
        """Test that to_dict returns a dictionary."""
        config = SimpleTestConfig()
        result = config.to_dict()

        assert isinstance(result, dict)


# =============================================================================
# Integration Tests
# =============================================================================

class TestBaseModelIntegration:
    """Integration tests for BaseModel."""

    def test_full_extraction_pipeline(self):
        """Test extracting features from all layers sequentially."""
        model = SimpleTestModel(input_dim=10, hidden_dim=20, output_dim=5)
        x = torch.randn(8, 10)

        all_features = {}
        for layer_name in model.get_layer_names():
            features = model.extract_layer_features(x, layer_name)
            all_features[layer_name] = features

        assert len(all_features) == 3
        assert all_features["embedding"].shape[0] == 8
        assert all_features["hidden"].shape[0] == 8
        assert all_features["output"].shape[0] == 8

    def test_forward_and_extract_consistency(self):
        """Test that forward and extract_layer_features are consistent."""
        model = SimpleTestModel(input_dim=10, hidden_dim=20, output_dim=5)
        model.eval()

        x = torch.randn(4, 10)

        # Get output via forward
        forward_output = model(x)

        # Get output via extract
        extract_output = model.extract_layer_features(x, "output")

        # Should be identical
        torch.testing.assert_close(forward_output, extract_output)

    def test_eval_mode_consistency(self):
        """Test that eval mode gives consistent results."""
        model = SimpleTestModel()
        model.eval()

        x = torch.randn(4, 10)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        torch.testing.assert_close(output1, output2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
