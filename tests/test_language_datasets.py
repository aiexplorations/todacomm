"""
Tests for language dataset loaders.
"""

import pytest
from tda_perturbation.data.language_datasets import (
    DatasetConfig,
    LanguageModelingDataset,
    QuestionAnsweringDataset,
    load_wikitext2,
    load_squad,
    create_dataloaders,
    load_language_dataset
)
from transformers import AutoTokenizer


@pytest.fixture
def tokenizer():
    """Get GPT-2 tokenizer."""
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def test_dataset_config():
    """Test dataset configuration."""
    config = DatasetConfig(
        dataset_name="wikitext2",
        task="lm",
        tokenizer_name="gpt2",
        max_length=128,
        num_samples=100
    )
    
    assert config.dataset_name == "wikitext2"
    assert config.task == "lm"
    assert config.max_length == 128


def test_language_modeling_dataset(tokenizer):
    """Test language modeling dataset."""
    texts = ["Hello world", "This is a test"]
    dataset = LanguageModelingDataset(texts, tokenizer, max_length=64)
    
    assert len(dataset) == 2
    
    sample = dataset[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample
    assert sample["input_ids"].shape[0] == 64  # max_length


def test_question_answering_dataset(tokenizer):
    """Test question answering dataset."""
    questions = ["What is AI?"]
    contexts = ["AI is artificial intelligence"]
    answers = [{"answer_start": 6, "answer_end": 30}]
    
    dataset = QuestionAnsweringDataset(
        questions, contexts, answers, tokenizer, max_length=64
    )
    
    assert len(dataset) == 1
    
    sample = dataset[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "start_positions" in sample
    assert "end_positions" in sample


@pytest.mark.slow
def test_load_wikitext2():
    """Test loading WikiText-2 dataset."""
    config = DatasetConfig(
        dataset_name="wikitext2",
        task="lm",
        tokenizer_name="gpt2",
        max_length=64,
        num_samples=10  # Small for testing
    )
    
    datasets = load_wikitext2(config)
    
    assert "train" in datasets
    assert "val" in datasets
    assert "test" in datasets
    assert len(datasets["train"]) > 0


@pytest.mark.slow
def test_load_squad():
    """Test loading SQuAD dataset."""
    config = DatasetConfig(
        dataset_name="squad",
        task="qa",
        tokenizer_name="bert-base-uncased",
        max_length=128,
        num_samples=10  # Small for testing
    )
    
    datasets = load_squad(config)
    
    assert "train" in datasets
    assert "val" in datasets
    assert len(datasets["train"]) > 0


@pytest.mark.slow
def test_load_language_dataset():
    """Test generic dataset loader."""
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


def test_create_dataloaders():
    """Test dataloader creation."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    texts = ["Hello world"] * 10
    dataset = LanguageModelingDataset(texts, tokenizer, max_length=64)
    
    datasets = {"train": dataset, "val": dataset}
    dataloaders = create_dataloaders(datasets, batch_size=4)
    
    assert "train" in dataloaders
    assert "val" in dataloaders
    
    # Test iteration
    batch = next(iter(dataloaders["train"]))
    assert "input_ids" in batch
    assert batch["input_ids"].shape[0] <= 4  # batch size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
