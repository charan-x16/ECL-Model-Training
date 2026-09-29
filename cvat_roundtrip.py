#!/usr/bin/env python3
"""Send flagged frames to CVAT for relabelling, then merge the fixes back.

Works on the output of find_false_frames.py. Two steps:

  1. package - zip the flagged original frames + labels in CVAT's "YOLO 1.1"
               format, ready for CVAT: Projects -> Import dataset.
  2. merge   - take the dataset CVAT exports ("YOLO 1.1" or "Ultralytics YOLO
               Detection", with images) and add it to the training split of
               data/, remapping class ids by name so CVAT label order can't
               corrupt them. Then run train.py again.

Usage:
  python cvat_roundtrip.py package                          # every flagged frame
  python cvat_roundtrip.py package --categories missed      # only missed objects
  python cvat_roundtrip.py merge path/to/cvat_export.zip --move-from-val
  python train.py --run-name train_v2
"""

import argparse
import csv
import logging
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

import yaml

from train import ROOT, find_data_yaml

CATEGORIES = ("fp_background", "wrong_class", "missed")
IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def dataset_class_names(data_yaml: Path) -> list[str]:
    """Class names in id order, exactly as training uses them."""
    from ultralytics.data.utils import check_det_dataset

    names = check_det_dataset(str(data_yaml))["names"]
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def read_report(report_path: Path) -> list[dict]:
    if not report_path.is_file():
        raise FileNotFoundError(
            f"{report_path} not found. Run find_false_frames.py first (without --no-originals)."
        )
    with open(report_path, newline="") as f:
        return list(csv.DictReader(f))


def package(args, cfg: dict, data_yaml: Path) -> None:
    run_dir = ROOT / cfg["paths"]["output_dir"] / args.run_name
    frames_dir = run_dir / f"false_frames_{args.split}"
    rows = read_report(frames_dir / "report.csv")

    selected = [
        r for r in rows
        if r["category"] in args.categories
        and r.get("training_image")
        and (not args.classes or r["predicted_class"] in args.classes or r["true_class"] in args.classes)
    ]
    # Most confident mistakes first; one entry per frame.
    selected.sort(key=lambda r: -float(r["confidence"]))
    images = list(dict.fromkeys(Path(r["training_image"]) for r in selected))
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise RuntimeError("No frames match the chosen categories/classes.")

    names = dataset_class_names(data_yaml)
    zip_path = Path(args.out) if args.out else run_dir / f"cvat_upload_{args.split}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("obj.names", "\n".join(names) + "\n")
        zf.writestr("obj.data", f"classes = {len(names)}\nnames = data/obj.names\n"
                                "train = data/train.txt\nbackup = backup/\n")
        zf.writestr("train.txt", "\n".join(f"data/obj_train_data/{p.name}" for p in images) + "\n")
        for image in images:
            label = image.parent.parent / "labels" / f"{image.stem}.txt"
            zf.write(image, f"obj_train_data/{image.name}")
            if label.is_file():
                zf.write(label, f"obj_train_data/{image.stem}.txt")
            else:
                zf.writestr(f"obj_train_data/{image.stem}.txt", "")

    logging.info("Packed %d frames (%s) into %s", len(images), ", ".join(args.categories), zip_path)
    logging.info("CVAT: create a project with labels %s, then Import dataset -> format 'YOLO 1.1'.", names)
    logging.info("Use %s/reference/ to see what is wrong in each frame.", frames_dir)


def read_export(zip_path: Path) -> tuple[list[str], dict[str, tuple[str, str | None]]]:
    """Return (class names in export id order, {stem: (image member, label member)})."""
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]
        names_member = next((m for m in members if PurePosixPath(m).name == "obj.names"), None)
        yaml_member = next((m for m in members if PurePosixPath(m).name == "data.yaml"), None)
        if names_member:
            export_names = [n.strip() for n in zf.read(names_member).decode().splitlines() if n.strip()]
        elif yaml_member:
            raw = yaml.safe_load(zf.read(yaml_member))["names"]
            export_names = [raw[i] for i in sorted(raw)] if isinstance(raw, dict) else list(raw)
        else:
            raise ValueError("Export has no obj.names or data.yaml; export as 'YOLO 1.1' "
                             "or 'Ultralytics YOLO Detection' with 'Save images' enabled.")

    images = {PurePosixPath(m).stem: m for m in members if PurePosixPath(m).suffix.lower() in IMG_SUFFIXES}
    labels = {PurePosixPath(m).stem: m for m in members
              if PurePosixPath(m).suffix == ".txt" and PurePosixPath(m).name != "train.txt"}
    if not images:
        raise ValueError("Export contains no images; re-export from CVAT with 'Save images' enabled.")
    return export_names, {stem: (img, labels.get(stem)) for stem, img in images.items()}


def remap_labels(text: str, id_map: dict[int, int]) -> str:
    lines = []
    for line in text.splitlines():
        values = line.split()
        if len(values) >= 5:
            lines.append(" ".join([str(id_map[int(float(values[0]))])] + values[1:]))
    return "\n".join(lines) + ("\n" if lines else "")


def train_images_dir(data_yaml: Path) -> Path:
    from ultralytics.data.utils import check_det_dataset

    train = check_det_dataset(str(data_yaml))["train"]
    train = Path(train[0] if isinstance(train, list) else train)
    if not train.is_dir():
        raise ValueError(f"Train split {train} is not a folder; pass --dest <images folder>.")
    return train


def merge(args, cfg: dict, data_yaml: Path) -> None:
    from ultralytics.data.utils import img2label_paths

    names = dataset_class_names(data_yaml)
    export_names, frames = read_export(Path(args.export))
    unknown = [n for n in export_names if n not in names]
    if unknown:
        raise ValueError(f"CVAT labels {unknown} are not dataset classes {names}. Rename them in CVAT.")
    id_map = {i: names.index(n) for i, n in enumerate(export_names)}
    if any(k != v for k, v in id_map.items()):
        logging.warning("CVAT label order differs from data.yaml; remapping ids by name: %s", id_map)

    dest_images = Path(args.dest) if args.dest else train_images_dir(data_yaml)
    run_dir = ROOT / cfg["paths"]["output_dir"] / args.run_name
    report = run_dir / f"false_frames_{args.split}" / "report.csv"
    # training/ copy stem -> original frame in the scanned split
    originals = {}
    if report.is_file():
        originals = {Path(r["training_image"]).stem: Path(r["image"])
                     for r in read_report(report) if r.get("training_image")}

    added = moved = 0
    backup_dir = run_dir / f"removed_from_{args.split}"
    with zipfile.ZipFile(args.export) as zf:
        for stem, (image_member, label_member) in sorted(frames.items()):
            suffix = PurePosixPath(image_member).suffix
            image_dest = dest_images / f"{stem}{suffix}"
            if image_dest.exists():
                image_dest = dest_images / f"{stem}_cvat{suffix}"
            label_dest = Path(img2label_paths([str(image_dest)])[0])
            label_text = remap_labels(zf.read(label_member).decode(), id_map) if label_member else ""

            original = originals.get(stem)
            if args.move_from_val and original and original.is_file():
                original_label = Path(img2label_paths([str(original)])[0])
                if not args.dry_run:
                    (backup_dir / "images").mkdir(parents=True, exist_ok=True)
                    (backup_dir / "labels").mkdir(parents=True, exist_ok=True)
                    shutil.move(str(original), backup_dir / "images" / original.name)
                    if original_label.is_file():
                        shutil.move(str(original_label), backup_dir / "labels" / original_label.name)
                moved += 1

            if not args.dry_run:
                image_dest.parent.mkdir(parents=True, exist_ok=True)
                label_dest.parent.mkdir(parents=True, exist_ok=True)
                image_dest.write_bytes(zf.read(image_member))
                label_dest.write_text(label_text)
            added += 1

    prefix = "[dry run] would add" if args.dry_run else "Added"
    logging.info("%s %d relabelled frames to %s", prefix, added, dest_images)
    if args.move_from_val:
        logging.info("%s %d originals out of %s (backup: %s)",
                     "[dry run] would move" if args.dry_run else "Moved", moved, args.split, backup_dir)
    elif args.split == "val":
        logging.warning("These frames are still in val as well; validation scores will be inflated. "
                        "Re-run with --move-from-val to move them out (they are backed up, not deleted).")
    logging.info("Next: python train.py --run-name <new run name>")


def main() -> None:
    parser = argparse.ArgumentParser(description="CVAT round trip for flagged frames.")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"), help="Path to config.yaml.")
    parser.add_argument("--data", help="data.yaml to use (default: auto-detect under dataset_dir).")
    parser.add_argument("--run-name", default="train", help="Run folder used by find_false_frames.py.")
    parser.add_argument("--split", default="val", help="Split find_false_frames.py scanned.")
    sub = parser.add_subparsers(dest="command", required=True)

    pack = sub.add_parser("package", help="Zip flagged frames for CVAT import (YOLO 1.1).")
    pack.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES),
                      help="Which mistakes to include (default: all).")
    pack.add_argument("--classes", nargs="+", help="Only frames involving these classes.")
    pack.add_argument("--limit", type=int, help="Maximum number of frames (most confident first).")
    pack.add_argument("--out", help="Zip path (default: output/<run>/cvat_upload_<split>.zip).")

    mrg = sub.add_parser("merge", help="Add a CVAT export to the dataset's training split.")
    mrg.add_argument("export", help="Zip exported from CVAT (YOLO 1.1 or Ultralytics YOLO, with images).")
    mrg.add_argument("--dest", help="Training images folder (default: train split of data.yaml).")
    mrg.add_argument("--move-from-val", action="store_true",
                     help="Move the original frames out of the scanned split into a backup folder.")
    mrg.add_argument("--dry-run", action="store_true", help="Show what would change without writing.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_yaml = Path(args.data) if args.data else find_data_yaml(ROOT / cfg["paths"]["dataset_dir"])

    if args.command == "package":
        package(args, cfg, data_yaml)
    else:
        merge(args, cfg, data_yaml)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        logging.error("CVAT round trip failed: %s", exc)
        sys.exit(1)
