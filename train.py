#!/usr/bin/env python3
"""YOLO11n training pipeline for Jetson Nano.

Steps:
  1. Locate and extract the .tar dataset file.
  2. Locate data.yaml inside the extracted dataset.
  3. Train YOLO11n on the Jetson Nano GPU.
  4. Write all training outputs (weights, metrics, logs, plots) to output/.
"""

import argparse
import logging
import sys
import tarfile
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parent


def find_dataset_tar(explicit_path: Optional[str]) -> Path:
    if explicit_path:
        tar_path = Path(explicit_path).resolve()
        if not tar_path.is_file():
            raise FileNotFoundError(f"Dataset tar not found: {tar_path}")
        return tar_path

    candidates = sorted(ROOT.glob("*.tar"))
    if not candidates:
        raise FileNotFoundError(
            "No .tar dataset file found in the project root. "
            "Pass one explicitly with --tar <path>."
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple .tar files found in project root: {[c.name for c in candidates]}. "
            "Pass the one to use explicitly with --tar <path>."
        )
    return candidates[0]


def safe_extract(tar_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        # tarfile.data_filter guards against path traversal / unsafe members
        # (absolute paths, symlinks escaping dest_dir, device files, etc).
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest_dir, filter="data")
        else:  # pragma: no cover - Python < 3.12 fallback
            for member in tar.getmembers():
                member_path = (dest_dir / member.name).resolve()
                if not str(member_path).startswith(str(dest_dir.resolve())):
                    raise RuntimeError(f"Unsafe tar member path: {member.name}")
            tar.extractall(dest_dir)


def find_data_yaml(search_dir: Path) -> Path:
    matches = list(search_dir.rglob("data.yaml"))
    if not matches:
        raise FileNotFoundError(
            f"No data.yaml found under {search_dir} after extracting the dataset."
        )
    if len(matches) > 1:
        # Prefer the shallowest match.
        matches.sort(key=lambda p: len(p.parts))
    return matches[0]


def resolve_device(requested: str) -> str:
    try:
        import torch
    except ImportError:
        logging.warning("PyTorch not importable; falling back to CPU.")
        return "cpu"

    if requested != "cpu" and not torch.cuda.is_available():
        logging.warning(
            "CUDA device '%s' requested but not available; falling back to CPU. "
            "Check JetPack/CUDA/PyTorch installation on the Jetson Nano.",
            requested,
        )
        return "cpu"
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO11n on Jetson Nano.")
    parser.add_argument("--tar", help="Path to the dataset .tar file (auto-detected if omitted).")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"), help="Path to config.yaml.")
    parser.add_argument("--epochs", type=int, help="Override epochs from config.")
    parser.add_argument("--batch", type=int, help="Override batch size from config.")
    parser.add_argument("--imgsz", type=int, help="Override image size from config.")
    parser.add_argument("--device", help="Override device from config (e.g. 0 or cpu).")
    parser.add_argument("--run-name", default="train", help="Name of this run's output subfolder.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset_dir = ROOT / cfg["paths"]["dataset_dir"]
    output_dir = ROOT / cfg["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Locate + extract the .tar dataset.
    tar_path = find_dataset_tar(args.tar)
    logging.info("Extracting dataset: %s -> %s", tar_path, dataset_dir)
    safe_extract(tar_path, dataset_dir)

    # 2. Locate data.yaml inside the extracted dataset.
    data_yaml = find_data_yaml(dataset_dir)
    logging.info("Using dataset config: %s", data_yaml)

    # 3. Train YOLO11n using the Jetson Nano GPU.
    from ultralytics import YOLO

    device = resolve_device(str(args.device or cfg["model"]["device"]))
    logging.info("Training on device: %s", device)

    model = YOLO(cfg["model"]["weights"])
    model.train(
        data=str(data_yaml),
        epochs=args.epochs or cfg["train"]["epochs"],
        imgsz=args.imgsz or cfg["train"]["imgsz"],
        batch=args.batch or cfg["train"]["batch"],
        workers=cfg["train"]["workers"],
        patience=cfg["train"]["patience"],
        device=device,
        project=str(output_dir),
        name=args.run_name,
        exist_ok=True,
    )

    logging.info("Training complete. All outputs saved under: %s", output_dir / args.run_name)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        logging.error("Training pipeline failed: %s", exc)
        sys.exit(1)
