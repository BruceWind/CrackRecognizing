"""Serve a small web UI for single-image YOLO inference."""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "best.pt"))
).expanduser()
UPLOAD_DIR = Path(
    os.getenv("UPLOAD_DIR", str(PROJECT_ROOT / "web_data" / "uploads"))
).expanduser()
RESULT_DIR = Path(
    os.getenv("RESULT_DIR", str(PROJECT_ROOT / "web_data" / "results"))
).expanduser()
DEVICE = os.getenv("YOLO_DEVICE", "cpu")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024
MAX_MODEL_BYTES = int(os.getenv("MAX_MODEL_MB", "500")) * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MODEL_LOCK = threading.Lock()

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    if MODEL_PATH.is_file():
        try:
            app.state.model = load_model(MODEL_PATH)
        except Exception as error:
            print(f"Could not load model at {MODEL_PATH}: {error}")
    yield


app = FastAPI(title="Turbine Blade Inspector", lifespan=lifespan)
app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")


def load_model(path: Path):
    from ultralytics import YOLO

    return YOLO(str(path))


@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    return FileResponse(
        PROJECT_ROOT / "recognize_web" / "app" / "static" / "index.html"
    )


@app.get("/status")
async def status() -> dict[str, bool]:
    with MODEL_LOCK:
        return {"model_available": app.state.model is not None}


@app.post("/model")
async def upload_model(file: UploadFile = File(...)) -> dict[str, bool]:
    filename = Path(file.filename or "").name
    if filename.lower() != "best.pt":
        raise HTTPException(status_code=400, detail="Upload a model named best.pt")

    model_bytes = await file.read()
    await file.close()
    if not model_bytes:
        raise HTTPException(status_code=400, detail="The model file is empty")
    if len(model_bytes) > MAX_MODEL_BYTES:
        limit_mb = MAX_MODEL_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Model is larger than the {limit_mb} MB upload limit",
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = MODEL_PATH.with_name(f".{uuid4().hex}.pt")
    temporary_path.write_bytes(model_bytes)
    try:
        candidate = await run_in_threadpool(load_model, temporary_path)
        with MODEL_LOCK:
            os.replace(temporary_path, MODEL_PATH)
            app.state.model = candidate
    except Exception as error:
        raise HTTPException(status_code=400, detail="Could not load best.pt") from error
    finally:
        temporary_path.unlink(missing_ok=True)

    return {"model_available": True}


def run_prediction(input_path: Path, output_path: Path) -> None:
    import cv2

    with MODEL_LOCK:
        model = app.state.model
        if model is None:
            raise RuntimeError("No model has been uploaded")
        result = model.predict(
            source=str(input_path),
            device=DEVICE,
            verbose=False,
        )[0]
        annotated_image = result.plot()

    if not cv2.imwrite(str(output_path), annotated_image):
        raise RuntimeError(f"Could not write annotated image to {output_path}")


@app.post("/predict")
async def predict(file: UploadFile | None = File(None)) -> dict[str, str]:
    with MODEL_LOCK:
        if app.state.model is None:
            raise HTTPException(status_code=503, detail="Upload best.pt before analyzing images")

    if file is None:
        raise HTTPException(status_code=400, detail="Select an image before analyzing")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image")

    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        extension = ".jpg"

    image_bytes = await file.read()
    await file.close()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Image is larger than the {limit_mb} MB upload limit",
        )

    image_id = uuid4().hex
    input_path = UPLOAD_DIR / f"{image_id}{extension}"
    output_path = RESULT_DIR / f"{image_id}.jpg"
    input_path.write_bytes(image_bytes)

    try:
        await run_in_threadpool(
            run_prediction,
            input_path,
            output_path,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail="Inference failed") from error

    return {"image_url": f"/results/{output_path.name}"}
