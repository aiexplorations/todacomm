"""
Basic tests for transformer model integration.
"""

import pytest
import torch
from tda_perturbation.models.transformer import (
    TransformerConfig,
    TransformerModel,
    load_pretrained_transformer
)


def test_transformer_config():
    """Test transformer configuration."""
    config = TransformerConfig(
        model_type="gpt2",
        model_name_or_path="gpt2",
        task="lm"
    )
    
    assert config.model_type == "gpt2"
    assert config.task == "lm"
    
    # Test serialization
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    
    # Test deserialization
    config2 = TransformerConfig.from_dict(config_dict)
    assert config2.model_type == config.model_type


@pytest.mark.slow
def test_load_gpt2_model():
    """Test loading GPT-2 model."""
    model = load_pretrained_transformer("gpt2", task="lm", device="cpu")
    
    assert model is not None
    assert isinstance(model, TransformerModel)
    
    # Check layer names
    layer_names = model.get_layer_names()
    assert "embedding" in layer_names
    assert "final" in layer_names
    assert len(layer_names) > 2


@pytest.mark.slow
def test_gpt2_forward():
    """Test GPT-2 forward pass."""
    model = load_pretrained_transformer("gpt2", task="lm", device="cpu")
    
    # Create dummy input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass
    outputs = model(input_ids)
    
    assert outputs is not None
    assert outputs.shape[0] == batch_size


@pytest.mark.slow
def test_extract_layer_features():
    """Test layer feature extraction."""
    model = load_pretrained_transformer("gpt2", task="lm", device="cpu")
    
    # Create dummy input
    input_ids = torch.randint(0, 1000, (2, 10))
    
    # Extract embedding layer
    features = model.extract_layer_features(input_ids, "embedding")
    assert features is not None
    assert features.shape[0] == 2  # batch size
    
    # Extract final layer
    features = model.extract_layer_features(input_ids, "final")
    assert features is not None


@pytest.mark.slow
def test_forward_with_cache():
    """Test efficient multi-layer extraction."""
    model = load_pretrained_transformer("gpt2", task="lm", device="cpu")
    
    input_ids = torch.randint(0, 1000, (2, 10))
    
    # Extract multiple layers
    layer_names = ["embedding", "layer_0", "final"]
    features = model.forward_with_cache(
        input_ids,
        layer_names=layer_names,
        pool_strategy="mean"
    )
    
    assert len(features) == len(layer_names)
    for layer_name in layer_names:
        assert layer_name in features
        assert features[layer_name].shape[0] == 2  # batch size
        assert features[layer_name].dim() == 2  # pooled to [batch, hidden]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
