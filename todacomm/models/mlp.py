"""
Multi-layer perceptron model for TDA analysis.

Provides configurable-depth FC networks implementing the BaseModel interface
for consistent layer-wise feature extraction and ablation studies.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel, ModelConfig


@dataclass
class MLPConfig(ModelConfig):
    """
    Configuration for MLP models.

    Attributes:
        input_dim: Input feature dimension (784 for MNIST)
        hidden_dims: List of hidden layer dimensions (determines depth)
        output_dim: Output dimension (number of classes)
        activation: Activation function between layers
        dropout: Dropout probability (0 = no dropout)
        batch_norm: Whether to use batch normalization
    """
    input_dim: int = 784
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    output_dim: int = 10
    activation: Literal["relu", "tanh", "gelu", "silu"] = "relu"
    dropout: float = 0.0
    batch_norm: bool = False

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "MLPConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    @property
    def num_layers(self) -> int:
        """Total number of layers (hidden + output)."""
        return len(self.hidden_dims) + 1


class MLPModel(BaseModel):
    """
    Multi-layer perceptron with configurable depth for TDA analysis.

    Implements BaseModel interface for consistent layer-wise feature extraction.
    Supports 2-6+ layer configurations for depth ablation studies.

    Layer naming convention:
        - "input": Raw input features
        - "hidden_0", "hidden_1", ...: Post-activation hidden layer outputs
        - "output": Final logits (pre-softmax)

    Example:
        >>> config = MLPConfig(hidden_dims=[256, 128, 64])
        >>> model = MLPModel(config)
        >>> print(model.get_layer_names())
        ['input', 'hidden_0', 'hidden_1', 'hidden_2', 'output']
    """

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config

        # Build layers dynamically based on hidden_dims
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if config.batch_norm else None

        dims = [config.input_dim] + list(config.hidden_dims) + [config.output_dim]

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            # Add batch norm for all but the last layer
            if config.batch_norm and i < len(dims) - 2:
                self.batch_norms.append(nn.BatchNorm1d(dims[i + 1]))

        # Store layer names
        self._layer_names = self._build_layer_names()

        # Activation function
        self._activation = self._get_activation_fn()

    def _build_layer_names(self) -> List[str]:
        """Build list of extractable layer names."""
        names = ["input"]
        for i in range(len(self.config.hidden_dims)):
            names.append(f"hidden_{i}")
        names.append("output")
        return names

    def _get_activation_fn(self) -> nn.Module:
        """Get activation function based on config."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        return activations.get(self.config.activation, nn.ReLU())

    def _flatten_input(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten input if needed (for image data)."""
        if x.dim() > 2:
            return x.view(x.size(0), -1)
        return x

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch, *) - will be flattened to (batch, input_dim)

        Returns:
            Output logits of shape (batch, output_dim)
        """
        x = self._flatten_input(x)

        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            x = self._activation(x)
            if self.config.dropout > 0 and self.training:
                x = F.dropout(x, p=self.config.dropout, training=True)

        # Final layer (no activation, no dropout)
        x = self.layers[-1](x)
        return x

    def get_layer_names(self) -> List[str]:
        """
        Get list of available layer names for feature extraction.

        Returns:
            List of layer names: ['input', 'hidden_0', ..., 'hidden_n', 'output']
        """
        return self._layer_names.copy()

    def extract_layer_features(
        self,
        x: torch.Tensor,
        layer_name: str,
        **kwargs
    ) -> torch.Tensor:
        """
        Extract features from a specific layer.

        Args:
            x: Input tensor
            layer_name: Name of layer to extract from

        Returns:
            Features from the specified layer

        Raises:
            ValueError: If layer_name is not valid
        """
        self.validate_layer_name(layer_name)
        x = self._flatten_input(x)

        if layer_name == "input":
            return x

        # Forward through layers until we reach target
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            x = self._activation(x)

            if layer_name == f"hidden_{i}":
                return x

        # Output layer
        if layer_name == "output":
            x = self.layers[-1](x)
            return x

        raise ValueError(f"Layer '{layer_name}' not found")

    def forward_with_cache(
        self,
        x: torch.Tensor,
        layer_names: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Efficient single-pass extraction of multiple layers.

        Args:
            x: Input tensor
            layer_names: Layers to extract (default: all layers)

        Returns:
            Dictionary mapping layer names to feature tensors
        """
        if layer_names is None:
            layer_names = self.get_layer_names()

        # Validate all layer names upfront
        for name in layer_names:
            self.validate_layer_name(name)

        x = self._flatten_input(x)
        features = {}

        if "input" in layer_names:
            features["input"] = x.clone()

        current = x
        for i, layer in enumerate(self.layers[:-1]):
            current = layer(current)
            if self.batch_norms is not None:
                current = self.batch_norms[i](current)
            current = self._activation(current)

            layer_name = f"hidden_{i}"
            if layer_name in layer_names:
                features[layer_name] = current.clone()

        # Output layer
        if "output" in layer_names:
            output = self.layers[-1](current)
            features["output"] = output

        return features

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        dims = [self.config.input_dim] + list(self.config.hidden_dims) + [self.config.output_dim]
        dim_str = " -> ".join(str(d) for d in dims)
        return f"MLPModel({dim_str}, act={self.config.activation})"


# Preset configurations for common architectures
PRESET_CONFIGS = {
    "shallow_2": MLPConfig(hidden_dims=[256]),
    "shallow_3": MLPConfig(hidden_dims=[256, 128]),
    "medium_4": MLPConfig(hidden_dims=[256, 128, 64]),
    "medium_5": MLPConfig(hidden_dims=[256, 192, 128, 64]),
    "deep_6": MLPConfig(hidden_dims=[256, 192, 128, 96, 64]),
}


def create_mlp(
    preset: Optional[str] = None,
    **kwargs
) -> MLPModel:
    """
    Create an MLP model from a preset or custom config.

    Args:
        preset: One of 'shallow_2', 'shallow_3', 'medium_4', 'medium_5', 'deep_6'
        **kwargs: Override any config parameter

    Returns:
        Configured MLPModel instance

    Example:
        >>> model = create_mlp("shallow_3", input_dim=64, output_dim=3)
    """
    if preset is not None:
        if preset not in PRESET_CONFIGS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESET_CONFIGS.keys())}")
        config = PRESET_CONFIGS[preset]
        # Apply overrides
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        config = MLPConfig.from_dict(config_dict)
    else:
        config = MLPConfig(**kwargs)

    return MLPModel(config)
