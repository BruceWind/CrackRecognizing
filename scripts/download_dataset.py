"""Download a YOLO dataset from Roboflow."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_WORKSPACE = "shanghai-jiaotong-university-aru2u"
DEFAULT_PROJECT = "wind-turbine-isdhe"
DEFAULT_VERSION = 1
DEFAULT_FORMAT = "yolov8"
DEFAULT_OUTPUT = Path("data/roboflow/wind-turbine-isdhe-v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", default=DEFAULT_WORKSPACE, help="Roboflow workspace slug"
    )
    parser.add_argument(
        "--project", default=DEFAULT_PROJECT, help="Roboflow project slug"
    )
    parser.add_argument(
        "--version", type=int, default=DEFAULT_VERSION, help="Dataset version number"
    )
    parser.add_argument(
        "--format", default=DEFAULT_FORMAT, help="Roboflow export format"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory where the dataset will be downloaded",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Keep --help available before the optional project dependencies are installed.
    from dotenv import load_dotenv
    from roboflow import Roboflow

    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key or api_key == "your_roboflow_api_key_here":
        raise SystemExit(
            "ROBOFLOW_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    if args.version < 1:
        raise SystemExit("--version must be a positive integer")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading {args.workspace}/{args.project} version {args.version} "
        f"as {args.format} to {output}"
    )

    roboflow = Roboflow(api_key=api_key)
    project = roboflow.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download(args.format, location=str(output))
    print(f"Dataset downloaded to {dataset.location}")


if __name__ == "__main__":
    main()
