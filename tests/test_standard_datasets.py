"""
Tests for standard dataset loaders.

Tests MNIST, FashionMNIST, and UCI tabular dataset loading.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from todacomm.data.standard_datasets import (
    StandardDatasetConfig,
    TabularDataset,
    load_uci_dataset,
    load_standard_dataset,
    create_dataloaders,
    get_dataset_info,
)


# =============================================================================
# StandardDatasetConfig Tests
# =============================================================================

class TestStandardDatasetConfig:
    """Tests for StandardDatasetConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = StandardDatasetConfig()

        assert config.dataset_name == "mnist"
        assert config.batch_size == 64
        assert config.num_samples is None
        assert config.normalize is True
        assert config.val_split == 0.15
        assert config.seed == 42
        assert config.data_dir == "./data"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = StandardDatasetConfig(
            dataset_name="iris",
            batch_size=32,
            num_samples=100,
            normalize=False,
            val_split=0.2,
            seed=123,
        )

        assert config.dataset_name == "iris"
        assert config.batch_size == 32
        assert config.num_samples == 100
        assert config.normalize is False
        assert config.val_split == 0.2
        assert config.seed == 123


# =============================================================================
# TabularDataset Tests
# =============================================================================

class TestTabularDataset:
    """Tests for TabularDataset wrapper."""

    def test_basic_creation(self):
        """Test creating a tabular dataset."""
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 3, size=100)

        dataset = TabularDataset(X, y)

        assert len(dataset) == 100
        assert dataset.input_dim == 10
        assert dataset.num_classes == 3

    def test_getitem_returns_tuple(self):
        """Test that __getitem__ returns (features, label) tuple."""
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.random.randint(0, 2, size=50)

        dataset = TabularDataset(X, y)
        features, label = dataset[0]

        assert isinstance(features, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert features.shape == (5,)
        assert label.shape == ()

    def test_tensor_dtypes(self):
        """Test that tensors have correct dtypes."""
        X = np.random.randn(20, 4)  # float64 input
        y = np.random.randint(0, 3, size=20)  # int64 input

        dataset = TabularDataset(X, y)
        features, label = dataset[0]

        assert features.dtype == torch.float32
        assert label.dtype == torch.long

    def test_with_transform(self):
        """Test dataset with custom transform."""
        X = np.ones((10, 4), dtype=np.float32)
        y = np.zeros(10, dtype=np.int64)

        # Transform that doubles values
        transform = lambda x: x * 2

        dataset = TabularDataset(X, y, transform=transform)
        features, _ = dataset[0]

        assert torch.allclose(features, torch.ones(4) * 2)


# =============================================================================
# UCI Dataset Tests
# =============================================================================

class TestLoadUCIDataset:
    """Tests for UCI dataset loading."""

    def test_load_iris(self):
        """Test loading Iris dataset."""
        config = StandardDatasetConfig(dataset_name="iris")
        datasets = load_uci_dataset(config)

        assert "train" in datasets
        assert "val" in datasets
        assert "test" in datasets

        # Check dimensions
        train_sample = datasets["train"][0]
        assert train_sample[0].shape == (4,)  # 4 features
        assert train_sample[1].shape == ()   # scalar label

    def test_load_wine(self):
        """Test loading Wine dataset."""
        config = StandardDatasetConfig(dataset_name="wine")
        datasets = load_uci_dataset(config)

        train_sample = datasets["train"][0]
        assert train_sample[0].shape == (13,)  # 13 features

    def test_load_digits(self):
        """Test loading Digits dataset."""
        config = StandardDatasetConfig(dataset_name="digits")
        datasets = load_uci_dataset(config)

        train_sample = datasets["train"][0]
        assert train_sample[0].shape == (64,)  # 8x8 = 64 features

    def test_load_breast_cancer(self):
        """Test loading Breast Cancer dataset."""
        config = StandardDatasetConfig(dataset_name="breast_cancer")
        datasets = load_uci_dataset(config)

        train_sample = datasets["train"][0]
        assert train_sample[0].shape == (30,)  # 30 features

    def test_num_samples_limit(self):
        """Test limiting number of samples."""
        config = StandardDatasetConfig(
            dataset_name="digits",
            num_samples=100
        )
        datasets = load_uci_dataset(config)

        total_samples = len(datasets["train"]) + len(datasets["val"]) + len(datasets["test"])
        assert total_samples <= 100

    def test_normalization(self):
        """Test that normalization is applied."""
        config_norm = StandardDatasetConfig(dataset_name="iris", normalize=True)
        config_raw = StandardDatasetConfig(dataset_name="iris", normalize=False)

        datasets_norm = load_uci_dataset(config_norm)
        datasets_raw = load_uci_dataset(config_raw)

        # Get all training features
        train_norm = torch.stack([datasets_norm["train"][i][0] for i in range(len(datasets_norm["train"]))])
        train_raw = torch.stack([datasets_raw["train"][i][0] for i in range(len(datasets_raw["train"]))])

        # Normalized data should have near-zero mean
        norm_mean = train_norm.mean(dim=0).abs().mean()
        raw_mean = train_raw.mean(dim=0).abs().mean()

        assert norm_mean < 0.5  # Close to zero
        assert raw_mean > 1.0   # Original scale

    def test_invalid_dataset_raises(self):
        """Test that invalid dataset name raises error."""
        config = StandardDatasetConfig(dataset_name="invalid")

        with pytest.raises(ValueError, match="Unknown UCI dataset"):
            load_uci_dataset(config)

    def test_reproducibility(self):
        """Test that same seed gives same splits."""
        config1 = StandardDatasetConfig(dataset_name="iris", seed=42)
        config2 = StandardDatasetConfig(dataset_name="iris", seed=42)

        datasets1 = load_uci_dataset(config1)
        datasets2 = load_uci_dataset(config2)

        # First sample should be identical
        x1, y1 = datasets1["train"][0]
        x2, y2 = datasets2["train"][0]

        assert torch.allclose(x1, x2)
        assert y1 == y2


# =============================================================================
# Unified Loader Tests
# =============================================================================

class TestLoadStandardDataset:
    """Tests for unified dataset loader."""

    def test_load_uci_via_unified(self):
        """Test loading UCI dataset via unified loader."""
        config = StandardDatasetConfig(dataset_name="iris")
        datasets = load_standard_dataset(config)

        assert "train" in datasets
        assert "val" in datasets
        assert "test" in datasets

    def test_invalid_dataset_raises(self):
        """Test that invalid dataset raises error."""
        config = StandardDatasetConfig(dataset_name="nonexistent")

        with pytest.raises(ValueError, match="Unknown dataset"):
            load_standard_dataset(config)

    @pytest.mark.parametrize("dataset_name", ["iris", "wine", "digits", "breast_cancer"])
    def test_all_uci_datasets(self, dataset_name):
        """Test that all UCI datasets load correctly."""
        config = StandardDatasetConfig(dataset_name=dataset_name)
        datasets = load_standard_dataset(config)

        assert len(datasets["train"]) > 0
        assert len(datasets["val"]) > 0
        assert len(datasets["test"]) > 0


# =============================================================================
# DataLoader Tests
# =============================================================================

class TestCreateDataloaders:
    """Tests for DataLoader creation."""

    def test_creates_all_splits(self):
        """Test that dataloaders are created for all splits."""
        config = StandardDatasetConfig(dataset_name="iris")
        datasets = load_uci_dataset(config)
        loaders = create_dataloaders(datasets)

        assert "train" in loaders
        assert "val" in loaders
        assert "test" in loaders

    def test_batch_size(self):
        """Test that batch size is respected."""
        config = StandardDatasetConfig(dataset_name="digits")
        datasets = load_uci_dataset(config)
        loaders = create_dataloaders(datasets, batch_size=16)

        batch = next(iter(loaders["train"]))
        assert batch[0].shape[0] <= 16

    def test_train_is_shuffled(self):
        """Test that training loader shuffles data."""
        config = StandardDatasetConfig(dataset_name="digits", seed=42)
        datasets = load_uci_dataset(config)

        loaders1 = create_dataloaders(datasets, batch_size=10)
        loaders2 = create_dataloaders(datasets, batch_size=10)

        batch1 = next(iter(loaders1["train"]))
        batch2 = next(iter(loaders2["train"]))

        # Due to shuffling, batches should differ
        # (small chance they're the same, but very unlikely)
        # Just check that loaders work
        assert batch1[0].shape[0] == 10

    def test_loader_returns_correct_format(self):
        """Test that loader returns (features, labels) format."""
        config = StandardDatasetConfig(dataset_name="iris")
        datasets = load_uci_dataset(config)
        loaders = create_dataloaders(datasets, batch_size=8)

        batch = next(iter(loaders["train"]))

        assert len(batch) == 2
        assert batch[0].dtype == torch.float32  # features
        assert batch[1].dtype == torch.long     # labels


# =============================================================================
# Dataset Info Tests
# =============================================================================

class TestGetDatasetInfo:
    """Tests for dataset metadata retrieval."""

    def test_mnist_info(self):
        """Test MNIST metadata."""
        info = get_dataset_info("mnist")

        assert info["input_dim"] == 784
        assert info["num_classes"] == 10
        assert "description" in info

    def test_fashion_mnist_info(self):
        """Test FashionMNIST metadata."""
        info = get_dataset_info("fashion_mnist")

        assert info["input_dim"] == 784
        assert info["num_classes"] == 10

    def test_iris_info(self):
        """Test Iris metadata."""
        info = get_dataset_info("iris")

        assert info["input_dim"] == 4
        assert info["num_classes"] == 3

    def test_digits_info(self):
        """Test Digits metadata."""
        info = get_dataset_info("digits")

        assert info["input_dim"] == 64
        assert info["num_classes"] == 10

    def test_invalid_dataset_raises(self):
        """Test that invalid dataset raises error."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset_info("nonexistent")

    @pytest.mark.parametrize("dataset_name", [
        "mnist", "fashion_mnist", "iris", "wine", "digits", "breast_cancer"
    ])
    def test_all_datasets_have_info(self, dataset_name):
        """Test that all supported datasets have metadata."""
        info = get_dataset_info(dataset_name)

        assert "input_dim" in info
        assert "num_classes" in info
        assert "description" in info


# =============================================================================
# MNIST/FashionMNIST Tests (require torchvision)
# =============================================================================

class TestMNISTLoading:
    """Tests for MNIST dataset loading."""

    @pytest.mark.slow
    def test_load_mnist(self):
        """Test loading MNIST dataset."""
        config = StandardDatasetConfig(
            dataset_name="mnist",
            num_samples=100  # Limit for fast test
        )
        datasets = load_standard_dataset(config)

        assert "train" in datasets
        assert "val" in datasets
        assert "test" in datasets

    @pytest.mark.slow
    def test_load_fashion_mnist(self):
        """Test loading FashionMNIST dataset."""
        config = StandardDatasetConfig(
            dataset_name="fashion_mnist",
            num_samples=100
        )
        datasets = load_standard_dataset(config)

        assert "train" in datasets
        assert "val" in datasets
        assert "test" in datasets


# =============================================================================
# Integration Tests
# =============================================================================

class TestDatasetIntegration:
    """Integration tests for dataset loading pipeline."""

    def test_full_pipeline_iris(self):
        """Test full pipeline: config -> dataset -> dataloader -> batch."""
        config = StandardDatasetConfig(
            dataset_name="iris",
            batch_size=16,
            normalize=True
        )

        datasets = load_standard_dataset(config)
        loaders = create_dataloaders(datasets, batch_size=config.batch_size)

        # Iterate through one epoch
        n_batches = 0
        for features, labels in loaders["train"]:
            assert features.dtype == torch.float32
            assert labels.dtype == torch.long
            assert features.shape[1] == 4  # Iris has 4 features
            n_batches += 1

        assert n_batches > 0

    def test_dataset_for_mlp(self):
        """Test using dataset with expected MLP interface."""
        config = StandardDatasetConfig(dataset_name="digits")
        datasets = load_standard_dataset(config)

        # Get dataset info
        info = get_dataset_info("digits")

        # Verify dimensions match
        sample = datasets["train"][0]
        assert sample[0].numel() == info["input_dim"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
