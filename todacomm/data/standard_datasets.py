"""
Standard dataset loaders for MLP models.

Supports MNIST, FashionMNIST (image classification) and UCI tabular datasets
for training and evaluating FC networks in TDA analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Literal, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split


@dataclass
class StandardDatasetConfig:
    """
    Configuration for standard (non-language) datasets.

    Attributes:
        dataset_name: Name of the dataset to load
        batch_size: Batch size for data loaders
        num_samples: Optional limit on number of samples
        normalize: Whether to normalize features
        val_split: Fraction of training data to use for validation
        seed: Random seed for reproducibility
        data_dir: Directory for downloading/caching data
    """
    dataset_name: Literal[
        "mnist", "fashion_mnist", "iris", "wine", "digits", "breast_cancer"
    ] = "mnist"
    batch_size: int = 64
    num_samples: Optional[int] = None
    normalize: bool = True
    val_split: float = 0.15
    seed: int = 42
    data_dir: str = "./data"


class TabularDataset(Dataset):
    """
    Generic tabular dataset wrapper.

    Wraps numpy arrays as a PyTorch Dataset with consistent interface.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        transform: Optional[callable] = None
    ):
        """
        Initialize tabular dataset.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Label vector (n_samples,)
            transform: Optional transform to apply to features
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Tuple of (features, label)
        """
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

    @property
    def input_dim(self) -> int:
        """Get input feature dimension."""
        return self.X.shape[1]

    @property
    def num_classes(self) -> int:
        """Get number of unique classes."""
        return len(torch.unique(self.y))


def _try_import_torchvision():
    """Try to import torchvision, raise helpful error if not available."""
    try:
        from torchvision import datasets, transforms
        return datasets, transforms
    except ImportError:
        raise ImportError(
            "torchvision is required for MNIST/FashionMNIST datasets. "
            "Install it with: pip install torchvision"
        )


def load_mnist(config: StandardDatasetConfig) -> Dict[str, Dataset]:
    """
    Load MNIST dataset.

    Args:
        config: Dataset configuration

    Returns:
        Dictionary with 'train', 'val', 'test' datasets
    """
    datasets_module, transforms = _try_import_torchvision()

    # Build transforms
    transform_list = [transforms.ToTensor()]
    if config.normalize:
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    transform = transforms.Compose(transform_list)

    # Load datasets
    train_full = datasets_module.MNIST(
        root=config.data_dir,
        train=True,
        download=True,
        transform=transform
    )
    test_data = datasets_module.MNIST(
        root=config.data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # Split train into train/val
    torch.manual_seed(config.seed)
    n_val = int(len(train_full) * config.val_split)
    n_train = len(train_full) - n_val
    train_data, val_data = random_split(train_full, [n_train, n_val])

    # Limit samples if specified
    if config.num_samples:
        n_train = min(config.num_samples, len(train_data))
        n_val = min(int(config.num_samples * config.val_split), len(val_data))
        n_test = min(int(config.num_samples * 0.2), len(test_data))

        train_data = Subset(train_data, range(n_train))
        val_data = Subset(val_data, range(n_val))
        test_data = Subset(test_data, range(n_test))

    return {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }


def load_fashion_mnist(config: StandardDatasetConfig) -> Dict[str, Dataset]:
    """
    Load FashionMNIST dataset.

    Args:
        config: Dataset configuration

    Returns:
        Dictionary with 'train', 'val', 'test' datasets
    """
    datasets_module, transforms = _try_import_torchvision()

    # Build transforms
    transform_list = [transforms.ToTensor()]
    if config.normalize:
        transform_list.append(transforms.Normalize((0.2860,), (0.3530,)))
    transform = transforms.Compose(transform_list)

    # Load datasets
    train_full = datasets_module.FashionMNIST(
        root=config.data_dir,
        train=True,
        download=True,
        transform=transform
    )
    test_data = datasets_module.FashionMNIST(
        root=config.data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # Split train into train/val
    torch.manual_seed(config.seed)
    n_val = int(len(train_full) * config.val_split)
    n_train = len(train_full) - n_val
    train_data, val_data = random_split(train_full, [n_train, n_val])

    # Limit samples if specified
    if config.num_samples:
        n_train = min(config.num_samples, len(train_data))
        n_val = min(int(config.num_samples * config.val_split), len(val_data))
        n_test = min(int(config.num_samples * 0.2), len(test_data))

        train_data = Subset(train_data, range(n_train))
        val_data = Subset(val_data, range(n_val))
        test_data = Subset(test_data, range(n_test))

    return {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }


def load_uci_dataset(config: StandardDatasetConfig) -> Dict[str, Dataset]:
    """
    Load UCI tabular datasets from sklearn.

    Supported datasets: iris, wine, digits, breast_cancer

    Args:
        config: Dataset configuration

    Returns:
        Dictionary with 'train', 'val', 'test' datasets
    """
    from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    loaders = {
        "iris": load_iris,
        "wine": load_wine,
        "digits": load_digits,
        "breast_cancer": load_breast_cancer,
    }

    if config.dataset_name not in loaders:
        raise ValueError(
            f"Unknown UCI dataset: {config.dataset_name}. "
            f"Available: {list(loaders.keys())}"
        )

    # Load data
    data = loaders[config.dataset_name]()
    X, y = data.data, data.target

    # Limit samples if specified
    if config.num_samples and config.num_samples < len(X):
        np.random.seed(config.seed)
        indices = np.random.choice(len(X), config.num_samples, replace=False)
        X, y = X[indices], y[indices]

    # Split into train/val/test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=config.seed,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=config.val_split / (1 - 0.2),  # Adjust for remaining data
        random_state=config.seed,
        stratify=y_train_val
    )

    # Normalize if requested
    if config.normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    return {
        "train": TabularDataset(X_train, y_train),
        "val": TabularDataset(X_val, y_val),
        "test": TabularDataset(X_test, y_test)
    }


def load_standard_dataset(config: StandardDatasetConfig) -> Dict[str, Dataset]:
    """
    Unified loader for standard datasets.

    Args:
        config: Dataset configuration

    Returns:
        Dictionary with 'train', 'val', 'test' datasets

    Raises:
        ValueError: If dataset_name is not recognized
    """
    if config.dataset_name == "mnist":
        return load_mnist(config)
    elif config.dataset_name == "fashion_mnist":
        return load_fashion_mnist(config)
    elif config.dataset_name in ["iris", "wine", "digits", "breast_cancer"]:
        return load_uci_dataset(config)
    else:
        raise ValueError(
            f"Unknown dataset: {config.dataset_name}. "
            f"Available: mnist, fashion_mnist, iris, wine, digits, breast_cancer"
        )


def create_dataloaders(
    datasets: Dict[str, Dataset],
    batch_size: int = 64,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders from datasets.

    Args:
        datasets: Dictionary of datasets
        batch_size: Batch size for loaders
        num_workers: Number of worker processes

    Returns:
        Dictionary of DataLoaders
    """
    return {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False
        )
        for split, dataset in datasets.items()
    }


def get_dataset_info(dataset_name: str) -> Dict:
    """
    Get metadata about a dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Dictionary with input_dim, num_classes, description
    """
    info = {
        "mnist": {
            "input_dim": 784,
            "num_classes": 10,
            "description": "Handwritten digits (28x28 grayscale)",
            "task": "classification"
        },
        "fashion_mnist": {
            "input_dim": 784,
            "num_classes": 10,
            "description": "Fashion articles (28x28 grayscale)",
            "task": "classification"
        },
        "iris": {
            "input_dim": 4,
            "num_classes": 3,
            "description": "Iris flower species classification",
            "task": "classification"
        },
        "wine": {
            "input_dim": 13,
            "num_classes": 3,
            "description": "Wine quality classification",
            "task": "classification"
        },
        "digits": {
            "input_dim": 64,
            "num_classes": 10,
            "description": "Handwritten digits (8x8 grayscale)",
            "task": "classification"
        },
        "breast_cancer": {
            "input_dim": 30,
            "num_classes": 2,
            "description": "Breast cancer diagnosis",
            "task": "classification"
        }
    }

    if dataset_name not in info:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return info[dataset_name]
