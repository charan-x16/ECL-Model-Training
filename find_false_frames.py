#!/usr/bin/env python3
"""Find the exact frames behind every off-diagonal cell of the confusion matrix.

Reproduces the matching logic of Ultralytics' confusion matrix, so for every
class the exported frames correspond to these cells of confusion_matrix.png:

  * fp_background - predicted <class>, true background  (row <class>, col background)
  * wrong_class   - predicted <class>, true other class (row <class>, col other)
  * missed        - true <class>, predicted background  (row background, col <class>)

Output under output/<run>/false_frames_<split>/:

  reference/<class>/<category>/  annotated copies for review (green = ground
                                 truth, red = box at fault), confidence-prefixed
  training/images/, labels/      clean original frames and their YOLO label
                                 files, ready to fix and add to a training set
  report.csv                     one row per mistake, most confident first

Frames scanned from the val split are validation data: if they move into
training, replace them in val so validation scores stay honest.

Usage:
  python find_false_frames.py                          # all classes, val split, run "train"
  python find_false_frames.py --classes human pipe     # only these classes
  python find_false_frames.py --split train --conf 0.4
  python find_false_frames.py --weights output/train/weights/best.pt
"""

import argparse
import csv
import logging
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from train import ROOT, find_data_yaml, resolve_device

# Defaults used by Ultralytics' ConfusionMatrix during validation.
MATRIX_CONF = 0.25
MATRIX_IOU = 0.45

GT_COLOR = (0, 200, 0)
FAULT_COLOR = (0, 0, 255)


def list_split_images(data_yaml: Path, split: str) -> list[Path]:
    """Resolve a split from data.yaml into image paths (dirs, .txt lists, or lists)."""
    from ultralytics.data.utils import IMG_FORMATS, check_det_dataset

    data = check_det_dataset(str(data_yaml))
    if split not in data or not data[split]:
        raise ValueError(f"Split '{split}' is not defined in {data_yaml}.")

    sources = data[split] if isinstance(data[split], list) else [data[split]]
    images: list[Path] = []
    for source in map(Path, sources):
        if source.is_dir():
            images += [p for p in source.rglob("*") if p.suffix[1:].lower() in IMG_FORMATS]
        elif source.suffix == ".txt":
            for line in source.read_text().splitlines():
                line = line.strip()
                if line:
                    path = Path(line)
                    images.append(path if path.is_absolute() else source.parent / path)
        else:
            images.append(source)
    return sorted(images)


def load_labels(image_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Read YOLO labels for an image as (classes, xyxy pixel boxes)."""
    from ultralytics.data.utils import img2label_paths

    label_path = Path(img2label_paths([str(image_path)])[0])
    classes, boxes = [], []
    if label_path.is_file():
        for line in label_path.read_text().splitlines():
            values = line.split()
            if len(values) < 5:
                continue
            cls, coords = int(float(values[0])), np.array(values[1:], dtype=float)
            if len(coords) == 4:
                cx, cy, w, h = coords
                x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            else:  # segmentation polygon -> bounding box
                xs, ys = coords[0::2], coords[1::2]
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            classes.append(cls)
            boxes.append([x1 * width, y1 * height, x2 * width, y2 * height])
    return np.array(classes, dtype=int), np.array(boxes, dtype=float).reshape(-1, 4)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between xyxy boxes a (N, 4) and b (M, 4)."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_like_confusion_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray, iou_thres: float):
    """Class-agnostic one-to-one matching, identical to Ultralytics ConfusionMatrix.

    Returns (gt_to_pred, pred_to_gt) dicts and the IoU matrix.
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return {}, {}, np.zeros((len(gt_boxes), len(pred_boxes)))

    iou = box_iou(gt_boxes, pred_boxes)
    gt_idx, pred_idx = np.where(iou > iou_thres)
    matches = np.stack([gt_idx, pred_idx, iou[gt_idx, pred_idx]], axis=1)
    if len(matches):
        matches = matches[matches[:, 2].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[matches[:, 2].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

    gt_to_pred = {int(g): int(p) for g, p, _ in matches}
    pred_to_gt = {p: g for g, p in gt_to_pred.items()}
    return gt_to_pred, pred_to_gt, iou


def export_original(image_path: Path, training_dir: Path, used_names: set[str]) -> Path:
    """Copy an untouched frame and its label file into a YOLO images/labels layout.

    Frames with no label file are background images and get an empty label.
    """
    from ultralytics.data.utils import img2label_paths

    stem, n = image_path.stem, 1
    while stem in used_names:  # same file name from different source folders
        stem, n = f"{image_path.stem}_{n}", n + 1
    used_names.add(stem)

    image_dest = training_dir / "images" / f"{stem}{image_path.suffix}"
    label_dest = training_dir / "labels" / f"{stem}.txt"
    image_dest.parent.mkdir(parents=True, exist_ok=True)
    label_dest.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, image_dest)
    label_path = Path(img2label_paths([str(image_path)])[0])
    if label_path.is_file():
        shutil.copy2(label_path, label_dest)
    else:
        label_dest.touch()
    return image_dest


def draw_frame(image, gt_classes, gt_boxes, names, fault_box, fault_label):
    import cv2

    canvas = image.copy()
    for cls, box in zip(gt_classes, gt_boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GT_COLOR, 2)
        cv2.putText(canvas, f"GT {names.get(cls, cls)}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_COLOR, 1, cv2.LINE_AA)

    x1, y1, x2, y2 = map(int, fault_box)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), FAULT_COLOR, 3)
    cv2.putText(canvas, fault_label, (x1, min(y2 + 18, canvas.shape[0] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, FAULT_COLOR, 2, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Export frames behind confusion-matrix errors.")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"), help="Path to config.yaml.")
    parser.add_argument("--run-name", default="train", help="Training run folder under output/.")
    parser.add_argument("--weights", help="Weights to evaluate (default: output/<run>/weights/best.pt).")
    parser.add_argument("--data", help="data.yaml to use (default: auto-detect under dataset_dir).")
    parser.add_argument("--split", default="val", help="Dataset split to scan (val/train/test).")
    parser.add_argument("--classes", nargs="+", help="Only report these class names (default: all).")
    parser.add_argument("--conf", type=float, default=MATRIX_CONF, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=MATRIX_IOU, help="IoU threshold for matching.")
    parser.add_argument("--imgsz", type=int, help="Override image size from config.")
    parser.add_argument("--device", help="Override device from config (e.g. 0 or cpu).")
    parser.add_argument("--batch", type=int, default=16, help="Images per inference batch.")
    parser.add_argument("--no-missed", action="store_true", help="Skip exporting missed objects.")
    parser.add_argument("--no-images", action="store_true", help="Skip annotated reference images.")
    parser.add_argument("--no-originals", action="store_true",
                        help="Skip copying original frames and labels into training/.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = ROOT / cfg["paths"]["output_dir"]
    weights = Path(args.weights) if args.weights else output_dir / args.run_name / "weights" / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}. Pass --weights or --run-name.")
    data_yaml = Path(args.data) if args.data else find_data_yaml(ROOT / cfg["paths"]["dataset_dir"])
    device = resolve_device(str(args.device or cfg["model"]["device"]))
    imgsz = args.imgsz or cfg["train"]["imgsz"]

    import cv2
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = {int(k): v for k, v in model.names.items()}
    if args.classes:
        unknown = set(args.classes) - set(names.values())
        if unknown:
            raise ValueError(f"Unknown classes {sorted(unknown)}; model classes: {list(names.values())}")
        targets = {k for k, v in names.items() if v in args.classes}
    else:
        targets = set(names)

    images = list_split_images(data_yaml, args.split)
    if not images:
        raise FileNotFoundError(f"No images found for split '{args.split}' in {data_yaml}.")

    report_dir = output_dir / args.run_name / f"false_frames_{args.split}"
    report_dir.mkdir(parents=True, exist_ok=True)
    reference_dir, training_dir = report_dir / "reference", report_dir / "training"
    used_names: set[str] = set()
    categories = ["fp_background", "wrong_class"] + ([] if args.no_missed else ["missed"])

    logging.info("Model: %s | data: %s | split: %s (%d images)", weights, data_yaml, args.split, len(images))
    logging.info("Classes: %s | conf=%.2f iou=%.2f", [names[t] for t in sorted(targets)], args.conf, args.iou)

    rows = []
    for start in range(0, len(images), args.batch):
        chunk = images[start:start + args.batch]
        results = model.predict(
            source=[str(p) for p in chunk], conf=args.conf, imgsz=imgsz,
            device=device, verbose=False,
        )
        for image_path, result in zip(chunk, results):
            height, width = result.orig_shape
            gt_classes, gt_boxes = load_labels(image_path, width, height)
            pred_boxes = result.boxes.xyxy.cpu().numpy()
            pred_classes = result.boxes.cls.cpu().numpy().astype(int)
            pred_confs = result.boxes.conf.cpu().numpy()

            gt_to_pred, pred_to_gt, iou = match_like_confusion_matrix(gt_boxes, pred_boxes, args.iou)
            # (category, predicted class, true class, box, confidence, IoU with GT)
            faults = []

            for p, pred_cls in enumerate(pred_classes):
                g = pred_to_gt.get(p)
                if g is None:
                    best = float(iou[:, p].max()) if iou.size else 0.0
                    faults.append(("fp_background", pred_cls, None, pred_boxes[p], pred_confs[p], best))
                elif gt_classes[g] != pred_cls:
                    faults.append(("wrong_class", pred_cls, gt_classes[g], pred_boxes[p],
                                   pred_confs[p], float(iou[g, p])))

            if not args.no_missed:
                for g, true_cls in enumerate(gt_classes):
                    if g not in gt_to_pred:
                        faults.append(("missed", None, true_cls, gt_boxes[g], 0.0, 0.0))

            faults = [f for f in faults if f[1] in targets or f[2] in targets]
            if not faults:
                continue

            original = "" if args.no_originals else str(export_original(image_path, training_dir, used_names))
            image = None if args.no_images else cv2.imread(str(image_path))
            for n, (category, pred_cls, true_cls, box, conf, match_iou) in enumerate(faults):
                pred_name = "background" if pred_cls is None else names.get(pred_cls, str(pred_cls))
                true_name = "background" if true_cls is None else names.get(true_cls, str(true_cls))
                # Group by the class the mistake is about: the predicted class for
                # false positives (confusion-matrix row), the true class for misses.
                owner = true_name if category == "missed" else pred_name
                saved = ""
                if image is not None:
                    if category == "missed":
                        label = f"MISSED {true_name}"
                    else:
                        label = f"pred {pred_name} {conf:.2f} | true {true_name}"
                    suffix = f"_true-{true_name}" if category == "wrong_class" else ""
                    folder = reference_dir / owner / category
                    folder.mkdir(parents=True, exist_ok=True)
                    saved_path = folder / f"{conf:.2f}_{image_path.stem}_{n}{suffix}.jpg"
                    cv2.imwrite(str(saved_path), draw_frame(image, gt_classes, gt_boxes, names, box, label))
                    saved = str(saved_path)
                rows.append({
                    "category": category,
                    "predicted_class": pred_name,
                    "true_class": true_name,
                    "image": str(image_path),
                    "frame": image_path.stem,
                    "confidence": round(float(conf), 4),
                    "iou_with_gt": round(match_iou, 4),
                    "x1": round(float(box[0]), 1), "y1": round(float(box[1]), 1),
                    "x2": round(float(box[2]), 1), "y2": round(float(box[3]), 1),
                    "annotated_image": saved,
                    "training_image": original,
                })

        logging.info("Processed %d/%d images", min(start + args.batch, len(images)), len(images))

    rows.sort(key=lambda r: (r["category"], r["predicted_class"], r["true_class"], -r["confidence"]))
    report_path = report_dir / "report.csv"
    fieldnames = ["category", "predicted_class", "true_class", "image", "frame", "confidence",
                  "iou_with_gt", "x1", "y1", "x2", "y2", "annotated_image", "training_image"]
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary per confusion-matrix cell (predicted -> true).
    boxes, frames = Counter(), defaultdict(set)
    for r in rows:
        cell = (r["category"], r["predicted_class"], r["true_class"])
        boxes[cell] += 1
        frames[cell].add(r["image"])
    logging.info("%-14s %-12s %-12s %6s %6s", "category", "predicted", "true", "boxes", "frames")
    for category in categories:
        for cell in sorted(c for c in boxes if c[0] == category):
            logging.info("%-14s %-12s %-12s %6d %6d", *cell, boxes[cell], len(frames[cell]))
    if not args.no_originals:
        logging.info("Copied %d original frames + labels to: %s", len(used_names), training_dir)
    logging.info("Report written to: %s", report_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        logging.error("False-frame search failed: %s", exc)
        sys.exit(1)
