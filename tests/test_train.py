import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.train import run_epoch


def test_run_epoch_returns_finite_metrics() -> None:
    features = torch.randn(8, 3, 4, 4)
    targets = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(features, targets), batch_size=4)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    loss, accuracy = run_epoch(
        model=model,
        loader=loader,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        optimizer=optimizer,
    )

    assert loss >= 0
    assert 0 <= accuracy <= 1

