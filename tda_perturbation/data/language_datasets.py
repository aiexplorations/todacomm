"""
Language dataset loaders for transformer models.

Supports next token prediction, question answering, and reasoning tasks
with efficient tokenization and batching for M4 Pro.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Literal, List
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


@dataclass
class DatasetConfig:
    """Configuration for language datasets."""
    
    dataset_name: Literal[
        "wikitext2", "ptb", "squad", "gsm8k", "arc", "custom"
    ] = "wikitext2"
    task: Literal["lm", "qa", "reasoning", "classification"] = "lm"
    tokenizer_name: str = "gpt2"
    max_length: int = 512
    batch_size: int = 8
    num_samples: Optional[int] = None  # Limit dataset size for quick experiments
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)  # train/val/test
    seed: int = 42


class LanguageModelingDataset(Dataset):
    """Dataset for next token prediction / language modeling."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # For language modeling, labels are shifted input_ids
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class QuestionAnsweringDataset(Dataset):
    """Dataset for question answering tasks."""
    
    def __init__(
        self,
        questions: List[str],
        contexts: List[str],
        answers: List[Dict],
        tokenizer,
        max_length: int = 512
    ):
        self.questions = questions
        self.contexts = contexts
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        context = self.contexts[idx]
        answer = self.answers[idx]
        
        # Tokenize question + context
        encoding = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # Get answer span positions
        start_positions = torch.tensor(answer.get("answer_start", 0))
        end_positions = torch.tensor(answer.get("answer_end", 0))
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_positions": start_positions,
            "end_positions": end_positions
        }


def load_wikitext2(
    config: DatasetConfig
) -> Dict[str, Dataset]:
    """
    Load WikiText-2 dataset for language modeling.
    
    Args:
        config: Dataset configuration
        
    Returns:
        Dictionary with train/val/test datasets
    """
    # Load from HuggingFace
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Extract texts
    train_texts = [item["text"] for item in dataset["train"] if item["text"].strip()]
    val_texts = [item["text"] for item in dataset["validation"] if item["text"].strip()]
    test_texts = [item["text"] for item in dataset["test"] if item["text"].strip()]
    
    # Limit samples if specified
    if config.num_samples:
        train_texts = train_texts[:int(config.num_samples * config.split_ratios[0])]
        val_texts = val_texts[:int(config.num_samples * config.split_ratios[1])]
        test_texts = test_texts[:int(config.num_samples * config.split_ratios[2])]
    
    # Create datasets
    return {
        "train": LanguageModelingDataset(train_texts, tokenizer, config.max_length),
        "val": LanguageModelingDataset(val_texts, tokenizer, config.max_length),
        "test": LanguageModelingDataset(test_texts, tokenizer, config.max_length)
    }


def load_squad(
    config: DatasetConfig
) -> Dict[str, Dataset]:
    """
    Load SQuAD 2.0 dataset for question answering.
    
    Args:
        config: Dataset configuration
        
    Returns:
        Dictionary with train/val datasets
    """
    # Load from HuggingFace
    dataset = load_dataset("squad_v2")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def extract_qa_data(split_data):
        questions = []
        contexts = []
        answers = []
        
        for item in split_data:
            questions.append(item["question"])
            contexts.append(item["context"])
            
            # Handle answer positions
            if item["answers"]["text"]:
                answer_text = item["answers"]["text"][0]
                answer_start = item["answers"]["answer_start"][0]
                answers.append({
                    "text": answer_text,
                    "answer_start": answer_start,
                    "answer_end": answer_start + len(answer_text)
                })
            else:
                # No answer (SQuAD 2.0 includes unanswerable questions)
                answers.append({"text": "", "answer_start": 0, "answer_end": 0})
        
        return questions, contexts, answers
    
    # Extract train data
    train_q, train_c, train_a = extract_qa_data(dataset["train"])
    val_q, val_c, val_a = extract_qa_data(dataset["validation"])
    
    # Limit samples
    if config.num_samples:
        n_train = int(config.num_samples * config.split_ratios[0])
        n_val = int(config.num_samples * config.split_ratios[1])
        train_q, train_c, train_a = train_q[:n_train], train_c[:n_train], train_a[:n_train]
        val_q, val_c, val_a = val_q[:n_val], val_c[:n_val], val_a[:n_val]
    
    return {
        "train": QuestionAnsweringDataset(train_q, train_c, train_a, tokenizer, config.max_length),
        "val": QuestionAnsweringDataset(val_q, val_c, val_a, tokenizer, config.max_length),
        "test": QuestionAnsweringDataset(val_q, val_c, val_a, tokenizer, config.max_length)  # Use val as test
    }


def create_dataloaders(
    datasets: Dict[str, Dataset],
    batch_size: int = 8,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders from datasets.
    
    Args:
        datasets: Dictionary of datasets
        batch_size: Batch size
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
            pin_memory=False  # M4 Pro doesn't benefit from pinned memory
        )
        for split, dataset in datasets.items()
    }


def load_language_dataset(
    config: DatasetConfig
) -> Tuple[Dict[str, Dataset], AutoTokenizer]:
    """
    Load language dataset based on configuration.
    
    Args:
        config: Dataset configuration
        
    Returns:
        Tuple of (datasets dict, tokenizer)
    """
    if config.dataset_name == "wikitext2":
        datasets = load_wikitext2(config)
    elif config.dataset_name == "squad":
        datasets = load_squad(config)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return datasets, tokenizer
