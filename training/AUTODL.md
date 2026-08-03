# AutoDL Guide

This guide runs the turbine-blade detector on an AutoDL NVIDIA GPU without
using the custom `training/train.py` file. It uses the Ultralytics `yolo` command
directly.

The commands assume that the repository is available at:

```text
/root/autodl-tmp/CrackRecognizing
```

Change `PROJECT_DIR` below if the repository is in another directory.

## 1. Create The Instance

Choose an AutoDL image with:

- An NVIDIA CUDA-enabled PyTorch environment
- Python 3.10 or 3.11
- CUDA 12.1 or newer
- An RTX 4090 or RTX 4090D 24G, or another supported NVIDIA GPU

The exact PyTorch version is not important as long as it is a CUDA build. Do
not choose a CPU-only image. The RTX 4090 and 4090D do not require special training
command.

## 2. Copy The Repository

Copy the repository to AutoDL's persistent data disk, or clone it if it is
available from Git:

```bash
source /etc/network_turbo
cd /root/autodl-tmp
git clone <repository-url> CrackRecognizing
cd /root/autodl-tmp/CrackRecognizing
```

If the repository was uploaded as an archive, extract it and make sure the
resulting directory contains `requirements/`, `training/`, `recognize_web/`, and
`pyproject.toml`.

Set the project directory for the remaining commands:

```bash
export PROJECT_DIR=/root/autodl-tmp/CrackRecognizing
cd "$PROJECT_DIR"
```

## 3. Check The GPU

Confirm that AutoDL exposes the GPU and that PyTorch can use it:

```bash
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

`CUDA available: True` and an NVIDIA GPU name should be shown. If PyTorch
reports `False`, fix the AutoDL image or PyTorch environment before training.

## 4. Install Python Packages

The AutoDL PyTorch image should already contain CUDA-enabled PyTorch. Install
the remaining project packages using a China PyPI mirror:

```bash
python -m pip install --upgrade pip \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -r requirements/base.txt \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

This installs the pinned Ultralytics version and its project dependencies. Do
not install `requirements/cpu.txt` on AutoDL because it is for CPU-only
PyTorch. Do not install `requirements/directml.txt` because DirectML is for
native Windows, not AutoDL Linux.

Check that the Ultralytics command is available:

```bash
yolo checks
```

## 5. Download The Dataset

The dataset is already usable if this file exists:

```bash
ls data/roboflow/wind-turbine-mpcbv-v1/data.yaml
```

If the dataset was not copied with the repository, download it from Roboflow.
First set the API key in the current shell. It is not written to a tracked
file this way:

```bash
export ROBOFLOW_API_KEY='paste-your-roboflow-api-key-here'
python training/download_dataset.py \
  --url "https://universe.roboflow.com/dtu-4hha9/wind-turbine-mpcbv" \
  --version 1 \
  --format yolov8 \
  --output data/roboflow/wind-turbine-mpcbv-v1
unset ROBOFLOW_API_KEY
```

Confirm the dataset layout:

```bash
ls data/roboflow/wind-turbine-mpcbv-v1/data.yaml
ls data/roboflow/wind-turbine-mpcbv-v1/train/images
ls data/roboflow/wind-turbine-mpcbv-v1/valid/images
```

The YAML file contains relative paths to `train/images`, `valid/images`, and
`test/images`. Keep the directory structure unchanged.

## 6. Train Without A Python Training File

Run Ultralytics directly from the shell:

```bash
cd "$PROJECT_DIR"
yolo detect train \
  data=data/roboflow/wind-turbine-mpcbv-v1/data.yaml \
  model=yolov8n.pt \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  workers=8 \
  device=0 \
  project=runs/detect \
  name=wind-turbine-mpcbv-v1
```

`yolov8n.pt` is downloaded automatically the first time. The `n` model is a
good starting point for this two-class dataset. A 4090/4090D 24G should normally
handle this command; reduce `batch` to `8` or `4` if CUDA runs out of memory.

Important options:

- `epochs=100`: number of complete passes through the training set
- `imgsz=640`: image size used by the model during training
- `batch=16`: images processed together on the GPU
- `device=0`: first NVIDIA GPU
- `workers=8`: data-loader worker processes

Training results are written to:

```text
runs/detect/wind-turbine-mpcbv-v1/
```

The trained checkpoints are in its `weights/` directory:

```text
runs/detect/wind-turbine-mpcbv-v1/weights/best.pt
runs/detect/wind-turbine-mpcbv-v1/weights/last.pt
```

## 6.1 Run Training In Background
Run training in the foreground:
cd /root/autodl-tmp/CrackRecognizing

yolo detect train \
  data=data/roboflow/wind-turbine-mpcbv-v1/data.yaml \
  model=yolov8n.pt \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0
You will see epoch progress, GPU memory, losses, precision, recall, and mAP in the terminal.

### RUnning in background
Use tmux so training continues after disconnecting:
```
tmux new -s yolo-training
```
Run the training command, then detach with:
Ctrl+B, then D

Reconnect later:
`tmux attach -t yolo-training`

Check Process and GPU
`nvidia-smi`
You can also check the process:
`ps aux | grep '[y]olo'`
## Training Outputs
Ultralytics usually saves results under:
```
runs/detect/train/
Important files:
runs/detect/train/weights/best.pt
runs/detect/train/weights/last.pt
runs/detect/train/results.csv
runs/detect/train/results.png
- best.pt: best model based on validation performance
- last.pt: checkpoint from the latest epoch, useful for resuming
- results.csv: metrics for every epoch
- results.png: training curves
```
The exact output directory is printed when training starts.
Use the trained model like this:
yolo detect predict \
  model=runs/detect/train/weights/best.pt \
  source=data/roboflow/wind-turbine-mpcbv-v1/test/images \
  device=0
The .pt file is the trained YOLO model.



## 7. Resume Training

If training stops, resume from the last checkpoint:

```bash
yolo detect train \
  model=runs/detect/wind-turbine-mpcbv-v1/weights/last.pt \
  resume=True \
  device=0
```

## 8. Validate The Trained Model

Evaluate the best checkpoint on the validation set:

```bash
yolo detect val \
  model=runs/detect/wind-turbine-mpcbv-v1/weights/best.pt \
  data=data/roboflow/wind-turbine-mpcbv-v1/data.yaml \
  device=0
```

Run inference on images and save the annotated results:

```bash
yolo detect predict \
  model=runs/detect/wind-turbine-mpcbv-v1/weights/best.pt \
  source=data/roboflow/wind-turbine-mpcbv-v1/test/images \
  device=0 \
  save=True
```

## 9. Download The Trained Model

If you only need the model for inference, archive `best.pt`:

```bash
cd /root/autodl-tmp/CrackRecognizing
zip wind-turbine-mpcbv-v1-best.zip \
  runs/detect/wind-turbine-mpcbv-v1/weights/best.pt
```

To keep the model, metrics, plots, and resume checkpoint, archive the complete
training run instead:

```bash
cd /root/autodl-tmp/CrackRecognizing
zip -r wind-turbine-mpcbv-v1-training.zip \
  runs/detect/train
```

Run this `scp` command on your local computer to download the complete archive:

```bash
scp -P 11273 \
  root@connect.bjb2.seetacloud.com:/root/autodl-tmp/CrackRecognizing/wind-turbine-mpcbv-v1-training.zip .
```

Replace `wind-turbine-mpcbv-v1-training.zip` with
`wind-turbine-mpcbv-v1-best.zip` when downloading only the trained model.
