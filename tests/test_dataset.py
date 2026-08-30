from src.dataset import get_dataloaders


def test_fake_dataset_supports_fast_smoke_runs(tmp_path) -> None:
    train_loader, validation_loader = get_dataloaders(
        data_dir=str(tmp_path),
        batch_size=8,
        num_workers=0,
        download=False,
        dataset_name="fake",
        train_samples=16,
        validation_samples=8,
    )

    inputs, targets = next(iter(train_loader))
    assert inputs.shape == (8, 3, 32, 32)
    assert targets.shape == (8,)
    assert len(train_loader.dataset) == 16
    assert len(validation_loader.dataset) == 8

