import pytest
import torch

from src.model import SimpleCNN, get_model


@pytest.mark.parametrize("architecture", ["simple_cnn", "resnet18"])
def test_model_returns_one_logit_per_class(architecture: str) -> None:
    model = get_model(architecture=architecture, num_classes=10)
    model.eval()
    with torch.inference_mode():
        output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 10)


def test_simple_cnn_supports_custom_class_count() -> None:
    model = SimpleCNN(num_classes=4)
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 32, 32))
    assert output.shape == (1, 4)


def test_unknown_architecture_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported architecture"):
        get_model("unknown")

