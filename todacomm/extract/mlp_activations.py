"""
MLP activation extraction module.

Efficient extraction of activations from MLP models for TDA analysis.
Optimized for large sample counts (10k-50k) with batching and memory management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from todacomm.models.mlp import MLPModel


@dataclass
class MLPActivationConfig:
    """
    Configuration for MLP activation extraction.

    Attributes:
        layers: Which layers to extract ("all" or list of layer names)
        max_samples: Maximum number of samples to extract
        batch_size: Batch size for extraction (larger = faster but more memory)
        device: Device to run extraction on
        show_progress: Whether to show progress bar
        use_fp16: Use float16 to reduce memory (activations stored as float32)
    """
    layers: Union[str, List[str]] = "all"
    max_samples: int = 50000
    batch_size: int = 256
    device: str = "cpu"
    show_progress: bool = True
    use_fp16: bool = False


def extract_mlp_activations(
    model: MLPModel,
    dataloader: DataLoader,
    config: Optional[MLPActivationConfig] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract activations from MLP model.

    Efficiently extracts layer-wise activations in batches,
    optimized for large sample counts (10k-50k).

    Args:
        model: MLPModel instance
        dataloader: DataLoader providing input samples
        config: Extraction configuration (uses defaults if None)

    Returns:
        Dictionary mapping layer names to {"X": features, "y": labels}
        - X: np.ndarray of shape (n_samples, layer_dim)
        - y: np.ndarray of shape (n_samples,) with labels

    Example:
        >>> model = MLPModel(MLPConfig())
        >>> loader = create_dataloaders(datasets, batch_size=64)["train"]
        >>> activations = extract_mlp_activations(model, loader)
        >>> print(activations["hidden_0"]["X"].shape)
        (10000, 256)
    """
    if config is None:
        config = MLPActivationConfig()

    device = torch.device(config.device)
    model.to(device)
    model.eval()

    # Determine target layers
    if config.layers == "all":
        target_layers = model.get_layer_names()
    elif isinstance(config.layers, str):
        target_layers = [config.layers]
    else:
        target_layers = list(config.layers)

    # Validate layers
    for layer in target_layers:
        model.validate_layer_name(layer)

    # Storage (use lists for lazy concatenation to manage memory)
    layer_features: Dict[str, List[np.ndarray]] = {layer: [] for layer in target_layers}
    labels: List[np.ndarray] = []
    total_samples = 0

    # Dtype for model inference
    dtype = torch.float16 if config.use_fp16 else torch.float32

    with torch.no_grad():
        iterator: Iterator = dataloader
        if config.show_progress:
            iterator = tqdm(dataloader, desc="Extracting MLP activations")

        for batch in iterator:
            # Handle different batch formats
            if isinstance(batch, dict):
                # Language-style batch
                inputs = batch.get("input", batch.get("input_ids"))
                batch_labels = batch.get("labels")
            elif isinstance(batch, (list, tuple)):
                # Standard (inputs, labels) format
                inputs, batch_labels = batch[0], batch[1] if len(batch) > 1 else None
            else:
                inputs = batch
                batch_labels = None

            # Move to device
            inputs = inputs.to(device, dtype=dtype)

            # Flatten if needed (image data)
            if inputs.dim() > 2:
                inputs = inputs.view(inputs.size(0), -1)

            # Extract all target layers in single forward pass
            features = model.forward_with_cache(inputs, layer_names=target_layers)

            # Store features
            for layer in target_layers:
                feat = features[layer].cpu().numpy().astype(np.float32)
                layer_features[layer].append(feat)

            # Store labels
            if batch_labels is not None:
                if isinstance(batch_labels, torch.Tensor):
                    batch_labels = batch_labels.cpu().numpy()
                labels.append(batch_labels)

            total_samples += inputs.size(0)

            # Check if we've collected enough samples
            if total_samples >= config.max_samples:
                break

    # Concatenate and limit to max_samples
    result = {}
    for layer in target_layers:
        X = np.concatenate(layer_features[layer], axis=0)

        if X.shape[0] > config.max_samples:
            X = X[:config.max_samples]

        if labels:
            y = np.concatenate(labels, axis=0)
            if y.shape[0] > config.max_samples:
                y = y[:config.max_samples]
        else:
            y = np.zeros(X.shape[0], dtype=np.int64)

        result[layer] = {
            "X": X,
            "y": y
        }

    return result


def extract_single_layer(
    model: MLPModel,
    dataloader: DataLoader,
    layer_name: str,
    max_samples: int = 10000,
    device: str = "cpu",
    show_progress: bool = True
) -> Dict[str, np.ndarray]:
    """
    Extract activations from a single layer.

    Convenience function for extracting just one layer.

    Args:
        model: MLPModel instance
        dataloader: DataLoader providing input samples
        layer_name: Name of layer to extract
        max_samples: Maximum samples to extract
        device: Device for extraction
        show_progress: Show progress bar

    Returns:
        Dictionary with "X" (features) and "y" (labels)
    """
    config = MLPActivationConfig(
        layers=[layer_name],
        max_samples=max_samples,
        device=device,
        show_progress=show_progress
    )
    result = extract_mlp_activations(model, dataloader, config)
    return result[layer_name]


def get_activation_statistics(
    activations: Dict[str, Dict[str, np.ndarray]]
) -> Dict[str, Dict[str, any]]:
    """
    Compute quick statistics on extracted activations.

    Args:
        activations: Output from extract_mlp_activations

    Returns:
        Dictionary with per-layer statistics
    """
    stats = {}
    for layer_name, data in activations.items():
        X = data["X"]
        stats[layer_name] = {
            "n_samples": X.shape[0],
            "n_dims": X.shape[1],
            "mean": float(np.mean(X)),
            "std": float(np.std(X)),
            "min": float(np.min(X)),
            "max": float(np.max(X)),
            "sparsity": float(np.mean(np.abs(X) < 1e-6)),
            "memory_mb": X.nbytes / (1024 * 1024),
        }
    return stats


def subsample_activations(
    activations: Dict[str, Dict[str, np.ndarray]],
    n_samples: int,
    seed: int = 42
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Subsample activations to reduce size for expensive operations.

    Useful before running TDA (which is O(n³) in sample count).

    Args:
        activations: Output from extract_mlp_activations
        n_samples: Target number of samples
        seed: Random seed for reproducibility

    Returns:
        Subsampled activations dictionary
    """
    rng = np.random.RandomState(seed)

    # Get total samples from first layer
    first_layer = list(activations.keys())[0]
    total = activations[first_layer]["X"].shape[0]

    if total <= n_samples:
        return activations

    # Sample indices (same for all layers to maintain correspondence)
    indices = rng.choice(total, n_samples, replace=False)
    indices = np.sort(indices)

    result = {}
    for layer_name, data in activations.items():
        result[layer_name] = {
            "X": data["X"][indices],
            "y": data["y"][indices],
        }

    return result
