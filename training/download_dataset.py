"""Download a YOLO dataset from a Roboflow Universe project URL.

------------------------------------------------------------------------------------------------------------------------

python training/download_dataset.py \
  --url "https://universe.roboflow.com/dtu-4hha9/wind-turbine-mpcbv/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true"

Force re-download:
python training/download_dataset.py \
  --url "https://universe.roboflow.com/dtu-4hha9/wind-turbine-mpcbv" \
  --force

"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse


DEFAULT_FORMAT = "yolov8"
DEFAULT_VERSION = 1


@dataclass(frozen=True)
class DatasetConfig:
    workspace: str
    project: str
    version: int
    output: Path


def parse_project_url(url: str) -> tuple[str, str]:
    """Extract the workspace and project from a Roboflow Universe URL."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("--url must be an http(s) URL")
    if parsed.netloc.lower() not in {"universe.roboflow.com", "www.roboflow.com"}:
        raise ValueError("--url must point to Roboflow Universe")

    path_parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(path_parts) < 2:
        raise ValueError("--url must contain /<workspace>/<project>")

    return path_parts[0], path_parts[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        required=True,
        help="Roboflow Universe project URL",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=DEFAULT_VERSION,
        help="Dataset version number (default: %(default)s)",
    )
    parser.add_argument(
        "--format",
        default=DEFAULT_FORMAT,
        help="Roboflow export format (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory; defaults to data/roboflow/<project>-v<version>",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even when the destination folder exists",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> DatasetConfig:
    try:
        workspace, project = parse_project_url(args.url)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if args.version < 1:
        raise SystemExit("--version must be a positive integer")

    output = args.output or Path("data/roboflow") / f"{project}-v{args.version}"
    return DatasetConfig(workspace, project, args.version, output)


def dataset_is_complete(output: Path) -> bool:
    """Return whether the exported YOLO directory has its essential files."""
    if not (output / "data.yaml").is_file():
        return False

    for split in ("train", "valid"):
        image_dir = output / split / "images"
        if not image_dir.is_dir() or not any(image_dir.iterdir()):
            return False
    return True


def destination_is_ready(config: DatasetConfig, force: bool) -> bool:
    """Check an existing destination before importing or authenticating Roboflow."""
    output = config.output.resolve()
    if not output.exists() or force:
        return False
    if dataset_is_complete(output):
        print(f"Skipping: complete dataset already exists at {output}")
        return True
    raise SystemExit(
        f"Destination already exists but is incomplete: {output}. "
        "Remove it or use --force to download again."
    )


def download_dataset(
    roboflow, config: DatasetConfig, export_format: str, force: bool
) -> None:
    output = config.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading {config.workspace}/{config.project} version {config.version} "
        f"as {export_format} to {output}"
    )

    project = roboflow.workspace(config.workspace).project(config.project)
    dataset = project.version(config.version).download(
        export_format,
        location=str(output),
        overwrite=force,
    )
    print(f"Dataset downloaded to {getattr(dataset, 'location', output)}")


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    if destination_is_ready(config, args.force):
        return

    # Keep destination checks and --help usable before optional dependencies load.
    from dotenv import load_dotenv
    from roboflow import Roboflow

    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key or api_key == "your_roboflow_api_key_here":
        raise SystemExit(
            "ROBOFLOW_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    roboflow = Roboflow(api_key=api_key)
    download_dataset(roboflow, config, args.format, args.force)


if __name__ == "__main__":
    main()
