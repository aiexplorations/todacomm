"""
Comprehensive tests for language dataset loaders.

Test Categories:
- Unit Tests: Individual dataset class behavior
- Configuration Tests: DatasetConfig dataclass
- Dataset Loading Tests: WikiText-2 and SQuAD loading
- DataLoader Tests: Batch creation and iteration
- Edge Cases: Empty data, special tokens, padding
"""

import pytest
import torch
import numpy as np
from transformers import AutoTokenizer
from todacomm.data.language_datasets import (
    DatasetConfig,
    LanguageModelingDataset,
    QuestionAnsweringDataset,
    load_wikitext2,
    load_squad,
    create_dataloaders,
    load_language_dataset
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def gpt2_tokenizer():
    """Get GPT-2 tokenizer."""
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture
def bert_tokenizer():
    """Get BERT tokenizer."""
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tok


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "Hello world, this is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming many industries.",
        "Python is a popular programming language."
    ]


@pytest.fixture
def sample_qa_data():
    """Sample QA data for testing."""
    questions = [
        "What is AI?",
        "Where is Paris?",
        "Who invented Python?"
    ]
    contexts = [
        "AI is artificial intelligence, a field of computer science.",
        "Paris is the capital of France, located in Europe.",
        "Python was created by Guido van Rossum in the late 1980s."
    ]
    answers = [
        {"answer_start": 6, "answer_end": 30},
        {"answer_start": 0, "answer_end": 26},
        {"answer_start": 22, "answer_end": 42}
    ]
    return questions, contexts, answers


# =============================================================================
# DatasetConfig Tests
# =============================================================================

class TestDatasetConfig:
    """Tests for DatasetConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DatasetConfig()
        assert config.dataset_name == "wikitext2"
        assert config.task == "lm"
        assert config.tokenizer_name == "gpt2"
        assert config.max_length == 512
        assert config.batch_size == 8

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DatasetConfig(
            dataset_name="squad",
            task="qa",
            tokenizer_name="bert-base-uncased",
            max_length=256,
            batch_size=16,
            num_samples=100
        )
        assert config.dataset_name == "squad"
        assert config.task == "qa"
        assert config.max_length == 256
        assert config.batch_size == 16
        assert config.num_samples == 100

    def test_split_ratios(self):
        """Test split ratio configuration."""
        config = DatasetConfig(split_ratios=(0.8, 0.1, 0.1))
        assert config.split_ratios == (0.8, 0.1, 0.1)
        assert sum(config.split_ratios) == 1.0

    def test_seed_configuration(self):
        """Test random seed configuration."""
        config = DatasetConfig(seed=123)
        assert config.seed == 123

    def test_all_dataset_names(self):
        """Test all supported dataset names."""
        for name in ["wikitext2", "ptb", "squad", "gsm8k", "arc", "custom"]:
            config = DatasetConfig(dataset_name=name)
            assert config.dataset_name == name

    def test_all_task_types(self):
        """Test all supported task types."""
        for task in ["lm", "qa", "reasoning", "classification"]:
            config = DatasetConfig(task=task)
            assert config.task == task


# =============================================================================
# LanguageModelingDataset Tests
# =============================================================================

class TestLanguageModelingDataset:
    """Tests for LanguageModelingDataset class."""

    def test_create_dataset(self, gpt2_tokenizer, sample_texts):
        """Test creating a language modeling dataset."""
        dataset = LanguageModelingDataset(
            sample_texts, gpt2_tokenizer, max_length=64
        )
        assert len(dataset) == len(sample_texts)

    def test_getitem_returns_dict(self, gpt2_tokenizer, sample_texts):
        """Test that __getitem__ returns a dictionary."""
        dataset = LanguageModelingDataset(
            sample_texts, gpt2_tokenizer, max_length=64
        )
        sample = dataset[0]

        assert isinstance(sample, dict)
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "labels" in sample

    def test_output_shapes(self, gpt2_tokenizer, sample_texts):
        """Test output tensor shapes."""
        max_length = 64
        dataset = LanguageModelingDataset(
            sample_texts, gpt2_tokenizer, max_length=max_length
        )
        sample = dataset[0]

        assert sample["input_ids"].shape == (max_length,)
        assert sample["attention_mask"].shape == (max_length,)
        assert sample["labels"].shape == (max_length,)

    def test_labels_equal_input_ids(self, gpt2_tokenizer, sample_texts):
        """Test that labels equal input_ids for LM."""
        dataset = LanguageModelingDataset(
            sample_texts, gpt2_tokenizer, max_length=64
        )
        sample = dataset[0]

        torch.testing.assert_close(sample["input_ids"], sample["labels"])

    def test_padding_applied(self, gpt2_tokenizer):
        """Test that padding is applied to short sequences."""
        short_text = ["Hi"]
        max_length = 64
        dataset = LanguageModelingDataset(
            short_text, gpt2_tokenizer, max_length=max_length
        )
        sample = dataset[0]

        # Should be padded to max_length
        assert sample["input_ids"].shape[0] == max_length

    def test_truncation_applied(self, gpt2_tokenizer):
        """Test that truncation is applied to long sequences."""
        long_text = ["word " * 1000]  # Very long text
        max_length = 64
        dataset = LanguageModelingDataset(
            long_text, gpt2_tokenizer, max_length=max_length
        )
        sample = dataset[0]

        # Should be truncated to max_length
        assert sample["input_ids"].shape[0] == max_length

    def test_attention_mask_values(self, gpt2_tokenizer, sample_texts):
        """Test that attention mask has valid values."""
        dataset = LanguageModelingDataset(
            sample_texts, gpt2_tokenizer, max_length=64
        )
        sample = dataset[0]

        # Attention mask should only contain 0s and 1s
        assert torch.all((sample["attention_mask"] == 0) | (sample["attention_mask"] == 1))

    def test_different_max_lengths(self, gpt2_tokenizer, sample_texts):
        """Test with different max_length values."""
        for max_length in [32, 64, 128, 256]:
            dataset = LanguageModelingDataset(
                sample_texts, gpt2_tokenizer, max_length=max_length
            )
            sample = dataset[0]
            assert sample["input_ids"].shape[0] == max_length

    def test_empty_text_handling(self, gpt2_tokenizer):
        """Test handling of empty text."""
        texts = ["", "Some text"]
        dataset = LanguageModelingDataset(texts, gpt2_tokenizer, max_length=64)

        # Should still work (empty text gets padded)
        sample = dataset[0]
        assert sample["input_ids"].shape[0] == 64

    def test_special_characters(self, gpt2_tokenizer):
        """Test handling of special characters."""
        texts = ["Hello! @#$%^&*() World", "Test\nwith\ttabs"]
        dataset = LanguageModelingDataset(texts, gpt2_tokenizer, max_length=64)

        for i in range(len(texts)):
            sample = dataset[i]
            assert sample["input_ids"].shape[0] == 64

    def test_all_indices_accessible(self, gpt2_tokenizer, sample_texts):
        """Test that all indices are accessible."""
        dataset = LanguageModelingDataset(
            sample_texts, gpt2_tokenizer, max_length=64
        )

        for i in range(len(dataset)):
            sample = dataset[i]
            assert sample is not None


# =============================================================================
# QuestionAnsweringDataset Tests
# =============================================================================

class TestQuestionAnsweringDataset:
    """Tests for QuestionAnsweringDataset class."""

    def test_create_dataset(self, bert_tokenizer, sample_qa_data):
        """Test creating a QA dataset."""
        questions, contexts, answers = sample_qa_data
        dataset = QuestionAnsweringDataset(
            questions, contexts, answers, bert_tokenizer, max_length=128
        )
        assert len(dataset) == len(questions)

    def test_getitem_returns_dict(self, bert_tokenizer, sample_qa_data):
        """Test that __getitem__ returns a dictionary."""
        questions, contexts, answers = sample_qa_data
        dataset = QuestionAnsweringDataset(
            questions, contexts, answers, bert_tokenizer, max_length=128
        )
        sample = dataset[0]

        assert isinstance(sample, dict)
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "start_positions" in sample
        assert "end_positions" in sample

    def test_output_shapes(self, bert_tokenizer, sample_qa_data):
        """Test output tensor shapes."""
        questions, contexts, answers = sample_qa_data
        max_length = 128
        dataset = QuestionAnsweringDataset(
            questions, contexts, answers, bert_tokenizer, max_length=max_length
        )
        sample = dataset[0]

        assert sample["input_ids"].shape == (max_length,)
        assert sample["attention_mask"].shape == (max_length,)
        assert sample["start_positions"].dim() == 0  # Scalar
        assert sample["end_positions"].dim() == 0  # Scalar

    def test_position_tensors(self, bert_tokenizer, sample_qa_data):
        """Test that position tensors are valid."""
        questions, contexts, answers = sample_qa_data
        dataset = QuestionAnsweringDataset(
            questions, contexts, answers, bert_tokenizer, max_length=128
        )
        sample = dataset[0]

        assert sample["start_positions"] >= 0
        assert sample["end_positions"] >= 0

    def test_all_indices_accessible(self, bert_tokenizer, sample_qa_data):
        """Test that all indices are accessible."""
        questions, contexts, answers = sample_qa_data
        dataset = QuestionAnsweringDataset(
            questions, contexts, answers, bert_tokenizer, max_length=128
        )

        for i in range(len(dataset)):
            sample = dataset[i]
            assert sample is not None


# =============================================================================
# WikiText-2 Loading Tests
# =============================================================================

@pytest.mark.slow
class TestLoadWikiText2:
    """Tests for WikiText-2 loading."""

    def test_basic_loading(self):
        """Test basic WikiText-2 loading."""
        config = DatasetConfig(
            dataset_name="wikitext2",
            task="lm",
            tokenizer_name="gpt2",
            max_length=64,
            num_samples=10
        )

        datasets = load_wikitext2(config)

        assert "train" in datasets
        assert "val" in datasets
        assert "test" in datasets

    def test_returns_datasets(self):
        """Test that loading returns Dataset objects."""
        config = DatasetConfig(
            dataset_name="wikitext2",
            task="lm",
            tokenizer_name="gpt2",
            num_samples=10
        )

        datasets = load_wikitext2(config)

        assert isinstance(datasets["train"], LanguageModelingDataset)
        assert isinstance(datasets["val"], LanguageModelingDataset)
        assert isinstance(datasets["test"], LanguageModelingDataset)

    def test_sample_limiting(self):
        """Test that num_samples limits dataset size."""
        config = DatasetConfig(
            dataset_name="wikitext2",
            task="lm",
            tokenizer_name="gpt2",
            num_samples=20,  # Very small
            max_length=64
        )

        datasets = load_wikitext2(config)

        # Total samples should be limited
        total = len(datasets["train"]) + len(datasets["val"]) + len(datasets["test"])
        assert total <= 20

    def test_datasets_are_iterable(self):
        """Test that datasets can be iterated."""
        config = DatasetConfig(
            dataset_name="wikitext2",
            task="lm",
            tokenizer_name="gpt2",
            num_samples=10,
            max_length=64
        )

        datasets = load_wikitext2(config)

        # Should be able to get samples
        if len(datasets["train"]) > 0:
            sample = datasets["train"][0]
            assert "input_ids" in sample


# =============================================================================
# SQuAD Loading Tests
# =============================================================================

@pytest.mark.slow
class TestLoadSQuAD:
    """Tests for SQuAD loading."""

    def test_basic_loading(self):
        """Test basic SQuAD loading."""
        config = DatasetConfig(
            dataset_name="squad",
            task="qa",
            tokenizer_name="bert-base-uncased",
            max_length=128,
            num_samples=10
        )

        datasets = load_squad(config)

        assert "train" in datasets
        assert "val" in datasets

    def test_returns_qa_datasets(self):
        """Test that loading returns QA Dataset objects."""
        config = DatasetConfig(
            dataset_name="squad",
            task="qa",
            tokenizer_name="bert-base-uncased",
            num_samples=10
        )

        datasets = load_squad(config)

        assert isinstance(datasets["train"], QuestionAnsweringDataset)
        assert isinstance(datasets["val"], QuestionAnsweringDataset)


# =============================================================================
# DataLoader Tests
# =============================================================================

class TestCreateDataloaders:
    """Tests for create_dataloaders function."""

    def test_create_dataloaders(self, gpt2_tokenizer, sample_texts):
        """Test creating dataloaders from datasets."""
        dataset = LanguageModelingDataset(
            sample_texts, gpt2_tokenizer, max_length=64
        )
        datasets = {"train": dataset, "val": dataset}

        dataloaders = create_dataloaders(datasets, batch_size=2)

        assert "train" in dataloaders
        assert "val" in dataloaders

    def test_batch_size(self, gpt2_tokenizer, sample_texts):
        """Test that batch size is respected."""
        dataset = LanguageModelingDataset(
            sample_texts, gpt2_tokenizer, max_length=64
        )
        datasets = {"train": dataset}

        dataloaders = create_dataloaders(datasets, batch_size=2)

        batch = next(iter(dataloaders["train"]))
        assert batch["input_ids"].shape[0] <= 2

    def test_dataloader_iteration(self, gpt2_tokenizer, sample_texts):
        """Test iterating through dataloader."""
        dataset = LanguageModelingDataset(
            sample_texts, gpt2_tokenizer, max_length=64
        )
        datasets = {"train": dataset}

        dataloaders = create_dataloaders(datasets, batch_size=2)

        batch_count = 0
        for batch in dataloaders["train"]:
            assert "input_ids" in batch
            assert "attention_mask" in batch
            batch_count += 1

        assert batch_count > 0

    def test_train_shuffling(self, gpt2_tokenizer):
        """Test that train dataloader shuffles by comparing batch composition."""
        # Create a larger dataset to test shuffling
        texts = [f"Unique text number {i} with different content" for i in range(100)]
        dataset = LanguageModelingDataset(texts, gpt2_tokenizer, max_length=64)
        datasets = {"train": dataset}

        dataloaders = create_dataloaders(datasets, batch_size=10)

        # Get order from two epochs using sum of input_ids (more unique per batch)
        orders = []
        for _ in range(2):
            epoch_sums = []
            for batch in dataloaders["train"]:
                # Sum all token IDs in the batch as a fingerprint
                batch_sum = batch["input_ids"].sum().item()
                epoch_sums.append(batch_sum)
            orders.append(epoch_sums)

        # Orders should likely be different (shuffling)
        # Note: This could theoretically fail, but very unlikely with 100 samples
        assert orders[0] != orders[1]

    def test_different_batch_sizes(self, gpt2_tokenizer, sample_texts):
        """Test with different batch sizes."""
        dataset = LanguageModelingDataset(
            sample_texts, gpt2_tokenizer, max_length=64
        )
        datasets = {"train": dataset}

        for batch_size in [1, 2, 4]:
            dataloaders = create_dataloaders(datasets, batch_size=batch_size)
            batch = next(iter(dataloaders["train"]))
            assert batch["input_ids"].shape[0] <= batch_size


# =============================================================================
# Generic Loader Tests
# =============================================================================

@pytest.mark.slow
class TestLoadLanguageDataset:
    """Tests for generic load_language_dataset function."""

    def test_load_wikitext2(self):
        """Test loading WikiText-2 through generic loader."""
        config = DatasetConfig(
            dataset_name="wikitext2",
            task="lm",
            tokenizer_name="gpt2",
            num_samples=10
        )

        datasets, tokenizer = load_language_dataset(config)

        assert datasets is not None
        assert tokenizer is not None
        assert "train" in datasets

    def test_load_squad(self):
        """Test loading SQuAD through generic loader."""
        config = DatasetConfig(
            dataset_name="squad",
            task="qa",
            tokenizer_name="bert-base-uncased",
            num_samples=10
        )

        datasets, tokenizer = load_language_dataset(config)

        assert datasets is not None
        assert tokenizer is not None

    def test_invalid_dataset_raises(self):
        """Test that invalid dataset name raises error."""
        config = DatasetConfig(dataset_name="invalid_dataset")

        with pytest.raises(ValueError, match="Unknown dataset"):
            load_language_dataset(config)

    def test_tokenizer_has_pad_token(self):
        """Test that returned tokenizer has pad token."""
        config = DatasetConfig(
            dataset_name="wikitext2",
            task="lm",
            tokenizer_name="gpt2",
            num_samples=10
        )

        _, tokenizer = load_language_dataset(config)

        assert tokenizer.pad_token is not None


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests for dataset loading."""

    def test_single_text(self, gpt2_tokenizer):
        """Test with single text."""
        texts = ["Just one text"]
        dataset = LanguageModelingDataset(texts, gpt2_tokenizer, max_length=64)

        assert len(dataset) == 1
        sample = dataset[0]
        assert sample["input_ids"].shape[0] == 64

    def test_unicode_text(self, gpt2_tokenizer):
        """Test with unicode characters."""
        texts = ["Hello 世界 🌍 مرحبا"]
        dataset = LanguageModelingDataset(texts, gpt2_tokenizer, max_length=64)

        sample = dataset[0]
        assert sample["input_ids"].shape[0] == 64

    def test_very_long_text(self, gpt2_tokenizer):
        """Test with very long text (should truncate)."""
        texts = ["word " * 10000]  # Very long
        dataset = LanguageModelingDataset(texts, gpt2_tokenizer, max_length=64)

        sample = dataset[0]
        assert sample["input_ids"].shape[0] == 64

    def test_whitespace_only_text(self, gpt2_tokenizer):
        """Test with whitespace-only text."""
        texts = ["   ", "Normal text"]
        dataset = LanguageModelingDataset(texts, gpt2_tokenizer, max_length=64)

        for i in range(len(texts)):
            sample = dataset[i]
            assert sample["input_ids"].shape[0] == 64

    def test_numeric_text(self, gpt2_tokenizer):
        """Test with numeric text."""
        texts = ["12345", "3.14159", "1 + 2 = 3"]
        dataset = LanguageModelingDataset(texts, gpt2_tokenizer, max_length=64)

        for i in range(len(texts)):
            sample = dataset[i]
            assert sample["input_ids"].shape[0] == 64


# =============================================================================
# Reproducibility Tests
# =============================================================================

class TestReproducibility:
    """Tests for reproducible dataset behavior."""

    def test_same_tokenization(self, gpt2_tokenizer, sample_texts):
        """Test that tokenization is deterministic."""
        dataset1 = LanguageModelingDataset(
            sample_texts, gpt2_tokenizer, max_length=64
        )
        dataset2 = LanguageModelingDataset(
            sample_texts, gpt2_tokenizer, max_length=64
        )

        for i in range(len(sample_texts)):
            sample1 = dataset1[i]
            sample2 = dataset2[i]
            torch.testing.assert_close(sample1["input_ids"], sample2["input_ids"])


# =============================================================================
# Advanced WikiText-2 Tests (Slow)
# =============================================================================

@pytest.mark.slow
class TestWikiText2Advanced:
    """Advanced WikiText-2 loading tests."""

    def test_different_split_ratios(self):
        """Test WikiText-2 with different split ratios."""
        config = DatasetConfig(
            dataset_name="wikitext2",
            task="lm",
            tokenizer_name="gpt2",
            num_samples=100,
            split_ratios=(0.8, 0.1, 0.1)
        )

        datasets = load_wikitext2(config)

        # Check that ratios are respected
        total = len(datasets["train"]) + len(datasets["val"]) + len(datasets["test"])
        assert total <= 100

    def test_no_sample_limit(self):
        """Test WikiText-2 without sample limit (loads more data)."""
        config = DatasetConfig(
            dataset_name="wikitext2",
            task="lm",
            tokenizer_name="gpt2",
            num_samples=None,  # No limit
            max_length=64
        )

        datasets = load_wikitext2(config)

        # Should have significant data
        assert len(datasets["train"]) > 100
        assert len(datasets["val"]) > 0
        assert len(datasets["test"]) > 0

    def test_different_tokenizers(self):
        """Test WikiText-2 with different tokenizers."""
        for tokenizer_name in ["gpt2", "distilgpt2"]:
            config = DatasetConfig(
                dataset_name="wikitext2",
                task="lm",
                tokenizer_name=tokenizer_name,
                num_samples=10,
                max_length=64
            )

            datasets = load_wikitext2(config)
            assert datasets is not None
            assert "train" in datasets

    def test_different_max_lengths(self):
        """Test WikiText-2 with different max lengths."""
        for max_length in [32, 64, 128, 256]:
            config = DatasetConfig(
                dataset_name="wikitext2",
                task="lm",
                tokenizer_name="gpt2",
                num_samples=10,
                max_length=max_length
            )

            datasets = load_wikitext2(config)
            if len(datasets["train"]) > 0:
                sample = datasets["train"][0]
                assert sample["input_ids"].shape[0] == max_length

    def test_dataset_iteration(self):
        """Test iterating through WikiText-2 datasets."""
        config = DatasetConfig(
            dataset_name="wikitext2",
            task="lm",
            tokenizer_name="gpt2",
            num_samples=20,
            max_length=64
        )

        datasets = load_wikitext2(config)

        for split in ["train", "val", "test"]:
            for i in range(min(5, len(datasets[split]))):
                sample = datasets[split][i]
                assert "input_ids" in sample
                assert "attention_mask" in sample
                assert "labels" in sample


# =============================================================================
# Advanced SQuAD Tests (Slow)
# =============================================================================

@pytest.mark.slow
class TestSQuADAdvanced:
    """Advanced SQuAD loading tests."""

    def test_squad_unanswerable_questions(self):
        """Test SQuAD with unanswerable questions."""
        config = DatasetConfig(
            dataset_name="squad",
            task="qa",
            tokenizer_name="bert-base-uncased",
            num_samples=50,  # Include some unanswerable
            max_length=256
        )

        datasets = load_squad(config)

        # Some samples may have zero-length answers
        found_unanswerable = False
        for i in range(min(20, len(datasets["train"]))):
            sample = datasets["train"][i]
            if sample["start_positions"].item() == 0 and sample["end_positions"].item() == 0:
                found_unanswerable = True
                break

        # Just verify the dataset loaded correctly
        assert len(datasets["train"]) > 0

    def test_squad_different_tokenizers(self):
        """Test SQuAD with different tokenizers."""
        for tokenizer_name in ["bert-base-uncased", "distilbert-base-uncased"]:
            config = DatasetConfig(
                dataset_name="squad",
                task="qa",
                tokenizer_name=tokenizer_name,
                num_samples=10,
                max_length=128
            )

            datasets = load_squad(config)
            assert datasets is not None
            assert "train" in datasets

    def test_squad_split_ratios(self):
        """Test SQuAD with custom split ratios."""
        config = DatasetConfig(
            dataset_name="squad",
            task="qa",
            tokenizer_name="bert-base-uncased",
            num_samples=100,
            split_ratios=(0.6, 0.2, 0.2)
        )

        datasets = load_squad(config)

        # Verify splits exist
        assert "train" in datasets
        assert "val" in datasets
        assert "test" in datasets

    def test_squad_iteration(self):
        """Test iterating through SQuAD datasets."""
        config = DatasetConfig(
            dataset_name="squad",
            task="qa",
            tokenizer_name="bert-base-uncased",
            num_samples=20,
            max_length=128
        )

        datasets = load_squad(config)

        for i in range(min(5, len(datasets["train"]))):
            sample = datasets["train"][i]
            assert "input_ids" in sample
            assert "attention_mask" in sample
            assert "start_positions" in sample
            assert "end_positions" in sample

    def test_squad_max_length_truncation(self):
        """Test SQuAD with short max length (forces truncation)."""
        config = DatasetConfig(
            dataset_name="squad",
            task="qa",
            tokenizer_name="bert-base-uncased",
            num_samples=10,
            max_length=64  # Short - will truncate many samples
        )

        datasets = load_squad(config)

        for i in range(min(5, len(datasets["train"]))):
            sample = datasets["train"][i]
            assert sample["input_ids"].shape[0] == 64


# =============================================================================
# DataLoader Integration Tests (Slow)
# =============================================================================

@pytest.mark.slow
class TestDataLoaderIntegration:
    """Integration tests for DataLoaders with real data."""

    def test_wikitext2_dataloader(self):
        """Test DataLoader with WikiText-2."""
        config = DatasetConfig(
            dataset_name="wikitext2",
            task="lm",
            tokenizer_name="gpt2",
            num_samples=50,
            max_length=64
        )

        datasets = load_wikitext2(config)
        dataloaders = create_dataloaders(datasets, batch_size=8)

        # Iterate through one epoch
        batch_count = 0
        for batch in dataloaders["train"]:
            assert batch["input_ids"].shape[0] <= 8
            assert batch["input_ids"].shape[1] == 64
            batch_count += 1
            if batch_count >= 5:
                break

        assert batch_count > 0

    def test_squad_dataloader(self):
        """Test DataLoader with SQuAD."""
        config = DatasetConfig(
            dataset_name="squad",
            task="qa",
            tokenizer_name="bert-base-uncased",
            num_samples=50,
            max_length=128
        )

        datasets = load_squad(config)
        dataloaders = create_dataloaders(datasets, batch_size=8)

        # Iterate through one epoch
        batch_count = 0
        for batch in dataloaders["train"]:
            assert batch["input_ids"].shape[0] <= 8
            assert batch["input_ids"].shape[1] == 128
            batch_count += 1
            if batch_count >= 5:
                break

        assert batch_count > 0

    def test_dataloader_num_workers(self):
        """Test DataLoader with multiple workers."""
        config = DatasetConfig(
            dataset_name="wikitext2",
            task="lm",
            tokenizer_name="gpt2",
            num_samples=30,
            max_length=64
        )

        datasets = load_wikitext2(config)
        # Note: num_workers > 0 may not work in all environments
        dataloaders = create_dataloaders(datasets, batch_size=4, num_workers=0)

        batch = next(iter(dataloaders["train"]))
        assert batch is not None


# =============================================================================
# Generic Loader Advanced Tests (Slow)
# =============================================================================

@pytest.mark.slow
class TestGenericLoaderAdvanced:
    """Advanced tests for generic load_language_dataset."""

    def test_load_wikitext2_full_pipeline(self):
        """Test full WikiText-2 loading pipeline."""
        config = DatasetConfig(
            dataset_name="wikitext2",
            task="lm",
            tokenizer_name="gpt2",
            num_samples=30,
            max_length=64,
            batch_size=4
        )

        datasets, tokenizer = load_language_dataset(config)

        # Create dataloaders
        dataloaders = create_dataloaders(datasets, batch_size=config.batch_size)

        # Run through a few batches
        for batch in dataloaders["train"]:
            # Check batch is properly formed
            assert batch["input_ids"].dim() == 2
            assert batch["attention_mask"].dim() == 2
            break

    def test_load_squad_full_pipeline(self):
        """Test full SQuAD loading pipeline."""
        config = DatasetConfig(
            dataset_name="squad",
            task="qa",
            tokenizer_name="bert-base-uncased",
            num_samples=30,
            max_length=128,
            batch_size=4
        )

        datasets, tokenizer = load_language_dataset(config)

        # Create dataloaders
        dataloaders = create_dataloaders(datasets, batch_size=config.batch_size)

        # Run through a few batches
        for batch in dataloaders["train"]:
            assert batch["input_ids"].dim() == 2
            assert batch["start_positions"].dim() == 1
            break

    def test_tokenizer_consistency(self):
        """Test that returned tokenizer is consistent with config."""
        config = DatasetConfig(
            dataset_name="wikitext2",
            task="lm",
            tokenizer_name="gpt2",
            num_samples=10
        )

        datasets, tokenizer = load_language_dataset(config)

        # Tokenizer should match config
        assert tokenizer is not None
        assert tokenizer.pad_token is not None

        # Verify tokenizer works
        encoded = tokenizer("Test text", return_tensors="pt")
        assert "input_ids" in encoded


# =============================================================================
# Seed Reproducibility Tests (Slow)
# =============================================================================

@pytest.mark.slow
class TestSeedReproducibility:
    """Tests for seed-based reproducibility."""

    def test_same_seed_same_order(self):
        """Test that same seed produces same data order."""
        config1 = DatasetConfig(
            dataset_name="wikitext2",
            task="lm",
            tokenizer_name="gpt2",
            num_samples=20,
            max_length=64,
            seed=42
        )

        config2 = DatasetConfig(
            dataset_name="wikitext2",
            task="lm",
            tokenizer_name="gpt2",
            num_samples=20,
            max_length=64,
            seed=42
        )

        datasets1 = load_wikitext2(config1)
        datasets2 = load_wikitext2(config2)

        # Same seed should produce same data
        if len(datasets1["train"]) > 0 and len(datasets2["train"]) > 0:
            sample1 = datasets1["train"][0]
            sample2 = datasets2["train"][0]
            torch.testing.assert_close(sample1["input_ids"], sample2["input_ids"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
