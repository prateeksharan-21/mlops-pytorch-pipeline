"""FastAPI application that serves CIFAR-10 class probabilities."""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

try:
    from src.dataset import get_transforms
    from src.model import get_model
except ModuleNotFoundError:  # Supports `uvicorn serve:app` with /app/src on PATH.
    from dataset import get_transforms
    from model import get_model

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def load_checkpoint(checkpoint_path: Path) -> torch.nn.Module:
    """Load a trusted application checkpoint onto CPU for inference."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    architecture = str(checkpoint.get("architecture", "resnet18"))
    num_classes = int(checkpoint.get("num_classes", len(CIFAR10_CLASSES)))
    model = get_model(architecture=architecture, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint_path = Path(
        os.getenv("MODEL_PATH", "/app/checkpoints/classifier_v1.pt")
    )
    app.state.model = None
    app.state.load_error = None
    try:
        app.state.model = load_checkpoint(checkpoint_path)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        app.state.load_error = str(error)
    yield


app = FastAPI(
    title="CIFAR-10 Model API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Report readiness only after a model checkpoint has loaded."""
    if app.state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is not loaded: {app.state.load_error}",
        )
    return {"status": "healthy", "model": "loaded"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict[str, object]:
    """Return the predicted CIFAR-10 class and every class probability."""
    if app.state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded",
        )
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image")

    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="Invalid image data") from error

    inputs = get_transforms(train=False)(pil_image).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(app.state.model(inputs), dim=1)[0]

    scores = probabilities.tolist()
    class_names = (
        CIFAR10_CLASSES
        if len(scores) == len(CIFAR10_CLASSES)
        else tuple(f"class_{index}" for index in range(len(scores)))
    )
    predicted_index = int(probabilities.argmax().item())
    return {
        "predicted_class": class_names[predicted_index],
        "predicted_index": predicted_index,
        "probabilities": {
            class_name: round(score, 6)
            for class_name, score in zip(class_names, scores, strict=True)
        },
    }

