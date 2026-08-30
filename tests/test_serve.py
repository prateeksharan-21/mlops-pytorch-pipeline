import asyncio
import io

import pytest
import torch
from fastapi import HTTPException, UploadFile
from PIL import Image
from starlette.datastructures import Headers

from src.model import get_model
from src.serve import app, health, lifespan, predict


def test_health_and_prediction_with_loaded_checkpoint(tmp_path, monkeypatch) -> None:
    checkpoint_path = tmp_path / "classifier.pt"
    model = get_model("simple_cnn", num_classes=10)
    torch.save(
        {
            "architecture": "simple_cnn",
            "num_classes": 10,
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    monkeypatch.setenv("MODEL_PATH", str(checkpoint_path))

    async def exercise_api() -> None:
        async with lifespan(app):
            assert health()["status"] == "healthy"
            image_buffer = io.BytesIO()
            Image.new("RGB", (32, 32), color=(80, 120, 160)).save(
                image_buffer, format="PNG"
            )
            image_buffer.seek(0)
            upload = UploadFile(
                file=image_buffer,
                filename="sample.png",
                headers=Headers({"content-type": "image/png"}),
            )
            response = await predict(upload)
            probabilities = response["probabilities"]
            assert len(probabilities) == 10
            assert response["predicted_class"] in probabilities
            assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-5)

    asyncio.run(exercise_api())


def test_health_is_unavailable_without_checkpoint(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.pt"))

    async def exercise_health() -> None:
        async with lifespan(app):
            with pytest.raises(HTTPException) as error:
                health()
            assert error.value.status_code == 503

    asyncio.run(exercise_health())

