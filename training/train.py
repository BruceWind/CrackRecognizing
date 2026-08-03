"""Train a YOLOv8 detector on the turbine blade dataset.

Example:
    python training/train.py --epochs 100 --batch 16 --device 0

For an RTX 4090D, choose:
- PyTorch: 2.3.1 or newer
- CUDA: 12.1 or newer
- Python: 3.10 or 3.11
- Ensure it is a CUDA/GPU image, not CPU-only
    
"""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    PROJECT_ROOT / "data" / "roboflow" / "wind-turbine-mpcbv-v1" / "data.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to a YOLO dataset YAML file (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Model checkpoint or model YAML (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: %(default)s)",
    )


    """
    --imgsz sets the image size used during training.
    --imgsz 640
    Images are resized/letterboxed to approximately 640×640 before being passed to YOLO.
    - Larger values, such as 1024: potentially better small-object detection, but slower and uses more GPU memory.
    - Smaller values, such as 416: faster and uses less memory, but may lose detail.
    - 640 is a good starting point for this dataset.
    
    """
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size (default: %(default)s)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size; lower it if the GPU runs out of memory (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Data-loader workers (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="CUDA device such as 0 or 0,1; auto-select when omitted",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=PROJECT_ROOT / "runs" / "detect",
        help="Directory for training outputs (default: %(default)s)",
    )
    parser.add_argument(
        "--name",
        default="wind-turbine-mpcbv-v1",
        help="Name of this training run (default: %(default)s)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the run identified by --model instead of starting fresh",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = args.data.expanduser().resolve()
    if not data.is_file():
        raise SystemExit(f"Dataset YAML does not exist: {data}")
    if args.epochs < 1:
        raise SystemExit("--epochs must be a positive integer")
    if args.imgsz < 1:
        raise SystemExit("--imgsz must be a positive integer")
    if args.batch == 0 or args.batch < -1:
        raise SystemExit("--batch must be -1 or a positive integer")
    if args.workers < 0:
        raise SystemExit("--workers cannot be negative")

    # Import lazily so --help and argument validation work without a GPU.
    from ultralytics import YOLO

    model = YOLO(args.model)
    train_args = {
        "data": str(data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(args.project.expanduser().resolve()),
        "name": args.name,
        "resume": args.resume,
    }
    if args.device is not None:
        train_args["device"] = args.device

    model.train(**train_args)


if __name__ == "__main__":
    main()
