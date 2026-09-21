#!/usr/bin/env python3
"""YOLO11n training pipeline for NVIDIA CUDA GPUs.

Steps:
  1. Locate and extract the .tar dataset file.
  2. Locate data.yaml inside the extracted dataset.
  3. Train YOLO11n on the configured CUDA GPU.
  4. Write all training outputs (weights, metrics, logs, plots) to output/.
"""

import argparse
import logging
import re
import sys
import tarfile
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parent


def guard_jetson_allocator(device: str) -> None:
    """Reject the L4T release with NVIDIA's known CUDA allocation bug."""
    if device == "cpu":
        return

    release_file = Path("/etc/nv_tegra_release")
    if not release_file.is_file():
        return

    release = release_file.read_text(errors="replace").splitlines()[0]
    match = re.search(r"R(\d+).*REVISION:\s*(\d+)\.(\d+)", release)
    if match and tuple(map(int, match.groups())) == (36, 4, 7):
        raise RuntimeError(
            "Jetson Linux 36.4.7 has a known NvMap/CUDA allocation bug that "
            "prevents training even at batch=1. Upgrade to JetPack 6.2.2 "
            "(L4T 36.5) or newer and reboot before using the GPU."
        )


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
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install the CUDA 12.8 PyTorch build "
            "described in README.md before starting training."
        ) from exc

    if requested != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device '{requested}' was requested, but CUDA is unavailable. "
            "Training was stopped to avoid an accidental multi-day CPU run. "
            "Check the NVIDIA driver and CUDA-enabled PyTorch installation."
        )

    if requested.isdigit():
        device_index = int(requested)
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device '{requested}' does not exist; PyTorch detected "
                f"{torch.cuda.device_count()} CUDA device(s)."
            )
        properties = torch.cuda.get_device_properties(device_index)
        logging.info(
            "CUDA device %s: %s (%.1f GiB VRAM)",
            requested,
            properties.name,
            properties.total_memory / (1024**3),
        )

    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO11n on an NVIDIA CUDA GPU.")
    parser.add_argument("--tar", help="Path to the dataset .tar file (auto-detected if omitted).")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"), help="Path to config.yaml.")
    parser.add_argument("--epochs", type=int, help="Override epochs from config.")
    parser.add_argument("--batch", type=int, help="Override batch size from config.")
    parser.add_argument("--imgsz", type=int, help="Override image size from config.")
    parser.add_argument("--workers", type=int, help="Override data-loader workers from config.")
    parser.add_argument("--device", help="Override device from config (e.g. 0 or cpu).")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable automatic mixed precision (default: config value).",
    )
    parser.add_argument(
        "--amp-check",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable Ultralytics' separate AMP self-test (default: config value).",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract the dataset even if an extracted data.yaml already exists.",
    )
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

    # 1. Reuse an extracted dataset when possible; extraction can take several
    # minutes and needlessly increases disk I/O on retries.
    existing_yamls = list(dataset_dir.rglob("data.yaml")) if dataset_dir.is_dir() else []
    if existing_yamls and not args.force_extract and args.tar is None:
        data_yaml = find_data_yaml(dataset_dir)
        logging.info("Reusing extracted dataset (use --force-extract to refresh it).")
    else:
        tar_path = find_dataset_tar(args.tar)
        logging.info("Extracting dataset: %s -> %s", tar_path, dataset_dir)
        safe_extract(tar_path, dataset_dir)
        data_yaml = find_data_yaml(dataset_dir)

    # 2. Locate data.yaml inside the extracted dataset.
    logging.info("Using dataset config: %s", data_yaml)

    # 3. Train YOLO11n using the configured GPU.
    requested_device = str(args.device or cfg["model"]["device"])
    guard_jetson_allocator(requested_device)
    device = resolve_device(requested_device)

    from ultralytics import YOLO

    amp = args.amp if args.amp is not None else cfg["train"].get("amp", True)
    amp_check = args.amp_check if args.amp_check is not None else cfg["train"].get("amp_check", True)
    logging.info("Training on device: %s", device)
    logging.info("Automatic mixed precision: %s", "enabled" if amp else "disabled")

    if amp and not amp_check and device != "cpu":
        # Ultralytics 8.4.x checks AMP by loading a second YOLO model and
        # running an eight-image FP32/FP16 comparison. That transient workload
        # exhausts shared RAM on affected JetPack releases before training can
        # begin. Orin supports FP16, so retain low-memory AMP training while
        # skipping only that additional probe.
        from ultralytics.engine import trainer as ultralytics_trainer

        ultralytics_trainer.check_amp = lambda _model: True
        logging.warning("Skipping the Ultralytics AMP self-test (amp_check=false).")

    model = YOLO(cfg["model"]["weights"])
    model.train(
        data=str(data_yaml),
        epochs=args.epochs if args.epochs is not None else cfg["train"]["epochs"],
        imgsz=args.imgsz if args.imgsz is not None else cfg["train"]["imgsz"],
        batch=args.batch if args.batch is not None else cfg["train"]["batch"],
        workers=args.workers if args.workers is not None else cfg["train"]["workers"],
        patience=cfg["train"]["patience"],
        amp=amp,
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
