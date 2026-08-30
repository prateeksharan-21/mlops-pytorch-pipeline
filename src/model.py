"""Model definitions used by training and inference."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class SimpleCNN(nn.Module):
    """A compact CNN suited to 32x32 CIFAR-10 images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    """Build a supported classifier without downloading pretrained weights."""
    normalized_name = architecture.lower().strip()

    if normalized_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)

    if normalized_name == "resnet18":
        model = models.resnet18(weights=None)
        # The 3x3 stem retains more spatial information than the ImageNet 7x7 stem.
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    supported = "resnet18, simple_cnn"
    raise ValueError(f"Unsupported architecture '{architecture}'. Choose: {supported}")

