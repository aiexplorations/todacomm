"""
Base model interface for unified feature extraction across architectures.

This module provides an abstract base class that all model implementations
should inherit from to ensure consistent layer-wise feature extraction.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """
    Abstract base class for models with layer-wise feature extraction.
    
    All model implementations (MLP, Transformer, etc.) should inherit from this
    class to provide a consistent interface for TDA analysis.
    """
    
    def __init__(self):
        super().__init__()
        self._layer_names: List[str] = []
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            **kwargs: Additional model-specific arguments
            
        Returns:
            Model output (logits, predictions, etc.)
        """
        pass
    
    @abstractmethod
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
            layer_name: Name of the layer to extract features from
            **kwargs: Additional model-specific arguments
            
        Returns:
            Features from the specified layer
            
        Raises:
            ValueError: If layer_name is not valid
        """
        pass
    
    @abstractmethod
    def get_layer_names(self) -> List[str]:
        """
        Get list of available layer names for feature extraction.
        
        Returns:
            List of layer names that can be used with extract_layer_features()
        """
        pass
    
    def forward_with_cache(
        self, 
        x: torch.Tensor,
        layer_names: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Efficient multi-layer feature extraction in a single forward pass.
        
        Args:
            x: Input tensor
            layer_names: List of layers to extract. If None, extracts all layers.
            **kwargs: Additional model-specific arguments
            
        Returns:
            Dictionary mapping layer names to their features
        """
        if layer_names is None:
            layer_names = self.get_layer_names()
        
        # Default implementation: extract each layer separately
        # Subclasses can override for more efficient implementation
        features = {}
        for layer_name in layer_names:
            features[layer_name] = self.extract_layer_features(x, layer_name, **kwargs)
        
        return features
    
    def validate_layer_name(self, layer_name: str) -> None:
        """
        Validate that a layer name is available for extraction.
        
        Args:
            layer_name: Layer name to validate
            
        Raises:
            ValueError: If layer_name is not valid
        """
        available = self.get_layer_names()
        if layer_name not in available:
            raise ValueError(
                f"Layer '{layer_name}' not found. "
                f"Available layers: {available}"
            )


class ModelConfig(ABC):
    """
    Abstract base class for model configurations.
    
    All model config classes should inherit from this to ensure
    consistent configuration interfaces.
    """
    
    @abstractmethod
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary for serialization."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, config_dict: Dict) -> "ModelConfig":
        """Create configuration from dictionary."""
        pass
