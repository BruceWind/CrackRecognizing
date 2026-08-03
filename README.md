# Turbine Blade Inspector

Computer-vision tooling for detecting cracks and broken sections on wind-turbine blades.

The first milestone is downloading a YOLO-format dataset from Roboflow. Training and
inference will be added after the dataset classes and annotations have been reviewed.

## Project Layout

```text
.
├── data/                         # Downloaded datasets; ignored by Git
├── requirements/
│   ├── base.txt                  # Shared Python packages
│   ├── cpu.txt                   # PyTorch CPU wheels for WSL/Linux
│   └── directml.txt              # Native Windows DirectML alternative
├── training/
│   ├── AUTODL.md                 # AutoDL setup and CLI training guide
│   ├── download_dataset.py       # Roboflow dataset bootstrap
│   └── train.py                  # Python training entry point
├── recognize_web/                # Local single-image inference web app
└── src/turbine_blade_inspector/
```

## Environment Setup

Use Python 3.10 through 3.12. Python 3.10 is the safest choice if DirectML may be
needed later. These commands create a project-local virtual environment:

```bash
cd /home/fkubr/git/turbine-blade-inspector
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### WSL/Linux CPU Setup

Do not install `torch-directml` in WSL. Install the CPU PyTorch wheels first, then
install the shared packages from PyPI:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements/cpu.txt
python -m pip install -r requirements/base.txt
```

### Native Windows DirectML Setup

DirectML is not a WSL/Linux backend. In a native Windows PowerShell environment,
create a separate virtual environment and install the DirectML package instead of
`requirements/cpu.txt`:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/directml.txt
python -m pip install -r requirements/base.txt
```

The DirectML package currently supports PyTorch up to 2.3.1. Do not install the CPU
PyTorch requirements and DirectML requirements into the same environment.

## Roboflow Dataset

1. Create a Roboflow account and copy your private API key.
2. Copy `.env.example` to `.env` and set `ROBOFLOW_API_KEY`.
3. Review the dataset terms and class names before using the data for training.
4. Download any Roboflow Universe project in YOLOv8 layout by passing its URL:

```bash
cp .env.example .env
# Edit .env and set ROBOFLOW_API_KEY before continuing.
python training/download_dataset.py \
  --url "https://universe.roboflow.com/dtu-4hha9/wind-turbine-mpcbv/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true"
```

The downloader extracts `/dtu-4hha9/wind-turbine-mpcbv` from the URL and writes
the dataset to `data/roboflow/wind-turbine-mpcbv-v1/`. URL query parameters and
the `/browse` suffix are ignored.

The original dataset can be downloaded using its own URL:

```bash
python training/download_dataset.py \
  --url "https://universe.roboflow.com/shanghai-jiaotong-university-aru2u/wind-turbine-isdhe"
```

Existing complete folders are skipped when `data.yaml` and non-empty train and
valid image folders already exist. An existing incomplete folder is not overwritten
unless `--force` is supplied:

```bash
python training/download_dataset.py \
  --url "https://universe.roboflow.com/dtu-4hha9/wind-turbine-mpcbv" \
  --force
```

Use `--output` to choose a different destination or `--version` for another
Roboflow dataset version:

```bash
python training/download_dataset.py \
  --url "https://universe.roboflow.com/dtu-4hha9/wind-turbine-mpcbv" \
  --version 1 \
  --format yolov8 \
  --output data/roboflow/wind-turbine-mpcbv-v1
```

## Verify the Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import onnxruntime; print('ONNX Runtime:', onnxruntime.__version__)"
python training/download_dataset.py --help
```

## Train YOLO on AutoDL

For a complete AutoDL workflow using the `yolo` command directly, without the
custom training Python file, see [training/AUTODL.md](training/AUTODL.md).

Install the project dependencies in the AutoDL environment first. For a CUDA
environment, install a compatible PyTorch build, then install the shared packages:

```bash
python -m pip install -r requirements/base.txt
```

Start a YOLOv8-nano training run on GPU 0:

```bash
python training/train.py \
  --model yolov8n.pt \
  --epochs 100 \
  --batch 16 \
  --device 0
```

The pretrained `yolov8n.pt` checkpoint is downloaded automatically if it is not
already present. Results and the best checkpoint are written under
`runs/detect/wind-turbine-mpcbv-v1/`, with the trained weights in its `weights/`
subdirectory. Reduce `--batch` if CUDA runs out of memory. Larger models such as
`yolov8s.pt` can be selected with `--model` after confirming the GPU has enough memory.

To continue an interrupted run, pass its checkpoint as `--model` and add `--resume`:

```bash
python training/train.py \
  --model runs/detect/wind-turbine-mpcbv-v1/weights/last.pt \
  --resume \
  --device 0
```

## Local WSL Web App

The project includes a small FastAPI web interface for single-image CPU
inference. It keeps uploaded originals in `web_data/uploads/` and annotated
results in `web_data/results/`. Copy the trained checkpoint to `models/best.pt`
or upload it through the web page, then run:

```bash
./recognize_web/setup.sh
./recognize_web/start.sh
```

Open [http://localhost:8000](http://localhost:8000) in a Windows browser. The
WSL setup uses CPU PyTorch, which is appropriate for the AMD 7735HS integrated
GPU environment. To use another checkpoint location:

```bash
MODEL_PATH=/path/to/best.pt ./recognize_web/start.sh
```

The web service does not expose detection metadata in the UI; it displays the
annotated image returned by YOLO. See [recognize_web/README.md](recognize_web/README.md)
for web-specific setup and usage details.

To run the web app in Docker instead of the WSL virtual environment:

```bash
mkdir -p models web_data
cp /path/to/best.pt models/best.pt
docker compose -f recognize_web/compose.yaml up --build
```

The Docker deployment is available at [http://localhost:3005](http://localhost:3005).

The container uses CPU inference and keeps the model and retained images on the
host through volume mounts. See [recognize_web/README.md](recognize_web/README.md)
for Docker details.

## Next Steps

- Inspect the downloaded `data.yaml` and confirm the crack/broken-blade classes.
- Split-check the train, validation, and test images for leakage.
- Add YOLO training and evaluation commands after annotation review.
- Export the selected model to ONNX for deployment.
