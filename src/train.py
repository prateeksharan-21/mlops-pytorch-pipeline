"""Train a CIFAR-10 classifier and emit one JSON object per log line."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

try:
    from src.dataset import get_dataloaders
    from src.model import get_model
except ModuleNotFoundError:  # Supports `python src/train.py` inside containers.
    from dataset import get_dataloaders
    from model import get_model


def log_event(**values: Any) -> None:
    """Print a structured event for container log collectors."""
    print(json.dumps(values, sort_keys=True), flush=True)


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a YAML mapping")
    return config


def resolve_config_path() -> Path:
    """Use CONFIG_PATH when supplied, otherwise prefer the container mount path."""
    if configured_path := os.getenv("CONFIG_PATH"):
        return Path(configured_path)
    mounted_path = Path("/app/configs/training_config.yaml")
    return mounted_path if mounted_path.exists() else Path("configs/training_config.yaml")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one training or validation epoch and return average loss and accuracy."""
    is_training = optimizer is not None
    model.train(mode=is_training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.set_grad_enabled(is_training):
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_correct += (outputs.argmax(dim=1) == targets).sum().item()
            total_examples += targets.size(0)

    if total_examples == 0:
        raise ValueError("DataLoader returned no examples")
    return total_loss / total_examples, total_correct / total_examples


def main() -> None:
    config_path = resolve_config_path()
    config = load_config(config_path)
    seed = int(config["training"].get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    architecture = str(config["model"]["architecture"])
    num_classes = int(config["model"]["num_classes"])
    model = get_model(architecture, num_classes).to(device)

    train_loader, validation_loader = get_dataloaders(
        data_dir=str(config["data"]["data_dir"]),
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["data"].get("num_workers", 2)),
        download=bool(config["data"].get("download", True)),
        dataset_name=str(config["data"].get("dataset", "cifar10")),
        num_classes=num_classes,
        train_samples=int(config["data"].get("train_samples", 1024)),
        validation_samples=int(config["data"].get("validation_samples", 256)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / str(config["output"]["model_name"])
    patience = int(config["training"]["early_stopping_patience"])
    best_validation_loss = float("inf")
    stale_epochs = 0

    log_event(
        event="training_started",
        architecture=architecture,
        config_path=str(config_path),
        device=str(device),
        seed=seed,
    )

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, criterion, device
        )
        log_event(
            event="epoch_completed",
            epoch=epoch,
            train_accuracy=round(train_accuracy, 4),
            train_loss=round(train_loss, 4),
            validation_accuracy=round(validation_accuracy, 4),
            validation_loss=round(validation_loss, 4),
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            stale_epochs = 0
            torch.save(
                {
                    "architecture": architecture,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_accuracy": validation_accuracy,
                    "validation_loss": validation_loss,
                },
                checkpoint_path,
            )
            log_event(event="checkpoint_saved", path=str(checkpoint_path))
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                log_event(event="early_stopping", epoch=epoch, patience=patience)
                break

    log_event(
        event="training_complete",
        best_validation_loss=round(best_validation_loss, 4),
        checkpoint_path=str(checkpoint_path),
    )


if __name__ == "__main__":
    main()
