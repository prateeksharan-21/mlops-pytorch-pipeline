"""CIFAR-10 dataset and DataLoader helpers."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_transforms(train: bool = True) -> transforms.Compose:
    """Return augmentation for training and deterministic preprocessing otherwise."""
    operations: list[object] = []
    if train:
        operations.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    if not train:
        operations.append(transforms.Resize((32, 32)))
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )
    return transforms.Compose(operations)


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
    download: bool = True,
    dataset_name: str = "cifar10",
    num_classes: int = 10,
    train_samples: int = 1024,
    validation_samples: int = 256,
) -> tuple[DataLoader, DataLoader]:
    """Create shuffled training and deterministic validation loaders."""
    normalized_name = dataset_name.lower().strip()
    if normalized_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=download,
            transform=get_transforms(train=True),
        )
        validation_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=download,
            transform=get_transforms(train=False),
        )
    elif normalized_name == "fake":
        # Synthetic data is reserved for fast infrastructure smoke tests.
        train_dataset = datasets.FakeData(
            size=train_samples,
            image_size=(3, 32, 32),
            num_classes=num_classes,
            transform=get_transforms(train=True),
            random_offset=0,
        )
        validation_dataset = datasets.FakeData(
            size=validation_samples,
            image_size=(3, 32, 32),
            num_classes=num_classes,
            transform=get_transforms(train=False),
            random_offset=train_samples,
        )
    else:
        raise ValueError("dataset_name must be 'cifar10' or 'fake'")

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_options
    )
    return train_loader, validation_loader
