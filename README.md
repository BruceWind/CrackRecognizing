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
├── scripts/
│   └── download_dataset.py       # Roboflow dataset bootstrap
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
python scripts/download_dataset.py \
  --url "https://universe.roboflow.com/dtu-4hha9/wind-turbine-mpcbv/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true"
```

The downloader extracts `/dtu-4hha9/wind-turbine-mpcbv` from the URL and writes
the dataset to `data/roboflow/wind-turbine-mpcbv-v1/`. URL query parameters and
the `/browse` suffix are ignored.

The original dataset can be downloaded using its own URL:

```bash
python scripts/download_dataset.py \
  --url "https://universe.roboflow.com/shanghai-jiaotong-university-aru2u/wind-turbine-isdhe"
```

Existing complete folders are skipped when `data.yaml` and non-empty train and
valid image folders already exist. An existing incomplete folder is not overwritten
unless `--force` is supplied:

```bash
python scripts/download_dataset.py \
  --url "https://universe.roboflow.com/dtu-4hha9/wind-turbine-mpcbv" \
  --force
```

Use `--output` to choose a different destination or `--version` for another
Roboflow dataset version:

```bash
python scripts/download_dataset.py \
  --url "https://universe.roboflow.com/dtu-4hha9/wind-turbine-mpcbv" \
  --version 1 \
  --format yolov8 \
  --output data/roboflow/wind-turbine-mpcbv-v1
```

## Verify the Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import onnxruntime; print('ONNX Runtime:', onnxruntime.__version__)"
python scripts/download_dataset.py --help
```

## Next Steps

- Inspect the downloaded `data.yaml` and confirm the crack/broken-blade classes.
- Split-check the train, validation, and test images for leakage.
- Add YOLO training and evaluation commands after annotation review.
- Export the selected model to ONNX for deployment.
