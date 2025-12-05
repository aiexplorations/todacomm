"""
Transformer-specific activation extraction utilities.

Efficient extraction of layer-wise features from transformer models
with support for attention patterns and multi-layer batching.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Union
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class ActivationConfig:
    """Configuration for activation extraction."""
    
    layers: Union[str, List[str]] = "final"  # Layer(s) to extract
    max_samples: int = 2000  # Max samples to extract (for TDA efficiency)
    pool_strategy: Literal["mean", "cls", "last", "none"] = "mean"
    device: str = "cpu"
    batch_size: int = 8
    show_progress: bool = True


def extract_transformer_activations(
    model,
    dataloader: DataLoader,
    config: ActivationConfig
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract activations from transformer model.
    
    Args:
        model: TransformerModel instance
        dataloader: DataLoader with tokenized inputs
        config: Extraction configuration
        
    Returns:
        Dictionary mapping layer names to {"X": features, "y": labels}
        where features are [num_samples, hidden_dim]
    """
    device = torch.device(config.device)
    model.to(device)
    model.eval()
    
    # Handle single layer or multiple layers
    if isinstance(config.layers, str):
        target_layers = [config.layers]
    else:
        target_layers = config.layers
    
    # Validate layers
    available_layers = model.get_layer_names()
    for layer in target_layers:
        if layer not in available_layers:
            raise ValueError(
                f"Layer '{layer}' not found. Available: {available_layers}"
            )
    
    # Storage for each layer
    layer_features = {layer: [] for layer in target_layers}
    labels = []
    
    total_samples = 0
    
    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Extracting activations") if config.show_progress else dataloader
        
        for batch in iterator:
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Extract features from all target layers efficiently
            features = model.forward_with_cache(
                input_ids=input_ids,
                layer_names=target_layers,
                attention_mask=attention_mask,
                pool_strategy=config.pool_strategy
            )
            
            # Store features for each layer
            for layer in target_layers:
                layer_features[layer].append(features[layer].cpu().numpy())
            
            # Store labels if available
            if "labels" in batch:
                labels.append(batch["labels"].cpu().numpy())
            
            # Check sample limit
            total_samples += input_ids.size(0)
            if total_samples >= config.max_samples:
                break
    
    # Concatenate all batches
    result = {}
    
    for layer in target_layers:
        X = np.concatenate(layer_features[layer], axis=0)
        
        # Limit to max_samples
        if X.shape[0] > config.max_samples:
            X = X[:config.max_samples]
        
        # Handle labels
        if labels:
            y = np.concatenate(labels, axis=0)
            if y.shape[0] > config.max_samples:
                y = y[:config.max_samples]
        else:
            y = np.zeros(X.shape[0], dtype=np.int64)
        
        result[layer] = {
            "X": X.astype(np.float32),
            "y": y.astype(np.int64)
        }
    
    return result


def extract_attention_patterns(
    model,
    dataloader: DataLoader,
    layer_indices: Optional[List[int]] = None,
    max_samples: int = 100,
    device: str = "cpu"
) -> Dict[int, np.ndarray]:
    """
    Extract attention pattern matrices for topology analysis.
    
    Args:
        model: TransformerModel instance
        dataloader: DataLoader with tokenized inputs
        layer_indices: Which layers to extract attention from (None = all)
        max_samples: Maximum number of samples
        device: Device to use
        
    Returns:
        Dictionary mapping layer index to attention matrices
        [num_samples, seq_len, seq_len]
    """
    device = torch.device(device)
    model.to(device)
    model.eval()
    
    if layer_indices is None:
        # Extract from all layers
        layer_indices = list(range(model.config.num_layers))
    
    attention_storage = {idx: [] for idx in layer_indices}
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Get attention weights
            for layer_idx in layer_indices:
                layer_name = f"layer_{layer_idx}_attention"
                attn = model.extract_layer_features(
                    input_ids=input_ids,
                    layer_name=layer_name,
                    attention_mask=attention_mask
                )
                attention_storage[layer_idx].append(attn.cpu().numpy())
            
            total_samples += input_ids.size(0)
            if total_samples >= max_samples:
                break
    
    # Concatenate and limit
    result = {}
    for layer_idx in layer_indices:
        attn_matrices = np.concatenate(attention_storage[layer_idx], axis=0)
        if attn_matrices.shape[0] > max_samples:
            attn_matrices = attn_matrices[:max_samples]
        result[layer_idx] = attn_matrices
    
    return result


def extract_single_layer(
    model,
    dataloader: DataLoader,
    layer_name: str,
    max_samples: int = 2000,
    pool_strategy: Literal["mean", "cls", "last", "none"] = "mean",
    device: str = "cpu"
) -> Dict[str, np.ndarray]:
    """
    Convenience function to extract a single layer.
    
    Args:
        model: TransformerModel instance
        dataloader: DataLoader
        layer_name: Layer to extract
        max_samples: Maximum samples
        pool_strategy: Pooling strategy
        device: Device
        
    Returns:
        Dictionary with "X" (features) and "y" (labels)
    """
    config = ActivationConfig(
        layers=layer_name,
        max_samples=max_samples,
        pool_strategy=pool_strategy,
        device=device,
        show_progress=True
    )
    
    result = extract_transformer_activations(model, dataloader, config)
    return result[layer_name]
