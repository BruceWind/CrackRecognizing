# Recognition Web App

This directory contains the local FastAPI interface for single-image inference
with the trained YOLO model.

## Setup

From the repository root, optionally copy the trained checkpoint to
`models/best.pt`, then run:

```bash
./recognize_web/setup.sh
```

The setup uses CPU PyTorch for WSL on the AMD 7735HS. It installs dependencies
through the configured China PyPI mirror and stores the virtual environment in
the repository's `.venv/` directory.

## Start

```bash
./recognize_web/start.sh
```

Open [http://localhost:8000](http://localhost:8000) in a browser. If no model is
present, click **Upload model** and select `best.pt`. The same button remains
available to replace the loaded model. After a valid model is loaded, upload
one JPEG, PNG, or WebP image. The original is retained in `web_data/uploads/`,
and the annotated result is retained in `web_data/results/`.

To use a checkpoint elsewhere:

```bash
MODEL_PATH=/path/to/best.pt ./recognize_web/start.sh
```

## Docker

Docker Compose builds a CPU image and mounts the model and retained images from
the repository, so `best.pt` does not need to be baked into the image:

```bash
mkdir -p models web_data
# Optional: the model can also be uploaded from the web page.
cp /path/to/best.pt models/best.pt
docker compose -f recognize_web/compose.yaml up --build
```

Open [http://localhost:3005](http://localhost:3005). Stop the container with:

```bash
docker compose -f recognize_web/compose.yaml down
```

The model is mounted from `models/` so the web UI can replace `best.pt`.
Uploaded originals and results remain in `web_data/` on the host.

## Prediction response

`POST /predict` returns the annotated image URL plus prediction details:

```json
{
  "image_url": "/results/<id>.jpg",
  "filename": "blade.jpg",
  "detection_count": 1,
  "detections_by_class": {"damage": 1},
  "detections": [
    {
      "class_id": 0,
      "class_name": "damage",
      "confidence": 0.9732,
      "box": {"x1": 120.5, "y1": 80.0, "x2": 310.25, "y2": 240.75}
    }
  ],
  "image": {"width": 1280, "height": 720},
  "inference_time_ms": 42.1,
  "inference_speed_ms": {"preprocess": 3.2, "inference": 35.4, "postprocess": 3.5}
}
```

## Railway

On Railway, set the service's **Root Directory** to `recognize_web` so the
Docker build context is this directory and the `Dockerfile` is auto-detected.
The Dockerfile and its pinned requirements are self-contained here.

Mount or download `models/best.pt` at `/app/models/best.pt` in the service, and
open the generated public domain at port `3005`.
