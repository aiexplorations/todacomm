"""
MLP training utilities.

Simple training loop for training MLP models from scratch
on classification tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Literal, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from todacomm.models.mlp import MLPModel


@dataclass
class TrainingConfig:
    """
    Configuration for MLP training.

    Attributes:
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        weight_decay: L2 regularization strength
        optimizer: Optimizer type
        scheduler: Learning rate scheduler type
        device: Device to train on
        seed: Random seed for reproducibility
        early_stopping_patience: Epochs to wait before early stopping (0 = disabled)
        log_interval: Epochs between logging (0 = no intermediate logs)
    """
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: Literal["adam", "sgd", "adamw"] = "adam"
    scheduler: Optional[Literal["cosine", "step", "plateau"]] = None
    device: str = "cpu"
    seed: int = 42
    early_stopping_patience: int = 0
    log_interval: int = 0


@dataclass
class TrainingResult:
    """
    Results from training.

    Contains final metrics and training history.
    """
    final_train_loss: float
    final_val_loss: float
    final_train_acc: float
    final_val_acc: float
    best_val_loss: float
    best_val_acc: float
    best_epoch: int
    history: Dict[str, List[float]] = field(default_factory=dict)
    stopped_early: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


def train_mlp(
    model: MLPModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[TrainingConfig] = None
) -> TrainingResult:
    """
    Train MLP model from scratch.

    Args:
        model: MLPModel instance to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration

    Returns:
        TrainingResult with metrics and history
    """
    if config is None:
        config = TrainingConfig()

    device = torch.device(config.device)
    model.to(device)

    # Set random seed
    torch.manual_seed(config.seed)

    # Optimizer
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:  # sgd
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=0.9
        )

    # Scheduler
    scheduler = None
    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1
        )
    elif config.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )

    criterion = nn.CrossEntropyLoss()

    # History tracking
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch in train_loader:
            inputs, labels = _unpack_batch(batch, device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += inputs.size(0)

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = _unpack_batch(batch, device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += inputs.size(0)

        # Calculate epoch metrics
        epoch_train_loss = train_loss / train_total
        epoch_val_loss = val_loss / val_total
        epoch_train_acc = train_correct / train_total
        epoch_val_acc = val_correct / val_total
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)
        history["lr"].append(current_lr)

        # Track best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Logging
        if config.log_interval > 0 and (epoch + 1) % config.log_interval == 0:
            print(
                f"Epoch {epoch+1}/{config.epochs}: "
                f"train_loss={epoch_train_loss:.4f}, "
                f"val_loss={epoch_val_loss:.4f}, "
                f"val_acc={epoch_val_acc:.4f}"
            )

        # Learning rate scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

        # Early stopping
        if config.early_stopping_patience > 0:
            if epochs_without_improvement >= config.early_stopping_patience:
                stopped_early = True
                break

    return TrainingResult(
        final_train_loss=history["train_loss"][-1],
        final_val_loss=history["val_loss"][-1],
        final_train_acc=history["train_acc"][-1],
        final_val_acc=history["val_acc"][-1],
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
        history=history,
        stopped_early=stopped_early
    )


def _unpack_batch(
    batch,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unpack batch to (inputs, labels) and move to device."""
    if isinstance(batch, dict):
        inputs = batch.get("input", batch.get("input_ids"))
        labels = batch.get("labels")
    elif isinstance(batch, (list, tuple)):
        inputs, labels = batch[0], batch[1]
    else:
        raise ValueError(f"Unexpected batch type: {type(batch)}")

    inputs = inputs.to(device)
    labels = labels.to(device)

    return inputs, labels


def evaluate_model(
    model: MLPModel,
    dataloader: DataLoader,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: Trained MLPModel
        dataloader: Data to evaluate on
        device: Device for evaluation

    Returns:
        Dictionary with loss and accuracy
    """
    device = torch.device(device)
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = _unpack_batch(batch, device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += inputs.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "n_samples": total,
    }


def quick_train(
    model: MLPModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu"
) -> TrainingResult:
    """
    Quick training with minimal configuration.

    Args:
        model: MLPModel to train
        train_loader: Training data
        val_loader: Validation data
        epochs: Number of epochs
        lr: Learning rate
        device: Device

    Returns:
        TrainingResult
    """
    config = TrainingConfig(
        epochs=epochs,
        learning_rate=lr,
        device=device,
        optimizer="adam"
    )
    return train_mlp(model, train_loader, val_loader, config)
