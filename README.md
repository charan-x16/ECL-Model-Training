# ECL-Model-Training

Trains a YOLO11n model on a Jetson Nano GPU.

## Requirements

Assumes JetPack, CUDA, cuDNN, and a Jetson-compatible PyTorch/torchvision wheel are
already installed on the device (install those via NVIDIA's Jetson PyTorch wheels —
`pip install ultralytics` alone will pull a generic `torch` build that does not use
the Jetson GPU). Requires Python 3.8+.

## Setup on Jetson Nano

Clone the repo, install dependencies, and copy the dataset `.tar` into the project root:

```bash
git clone https://github.com/charan-x16/ECL-Model-Training.git
cd ECL-Model-Training
pip install -r requirements.txt
```

Place the dataset `.tar` file in the project root (e.g. `cp /path/to/dataset.tar .`).

## Usage

Run the training pipeline:

```bash
python3 train.py
# or, to point at a specific tar file instead of auto-detecting it:
python3 train.py --tar path/to/dataset.tar
```

The pipeline will:

1. Extract the `.tar` dataset into `data/`.
2. Locate `data.yaml` inside the extracted dataset (standard YOLO format:
   `images/train`, `images/val`, `labels/train`, `labels/val`, `data.yaml`).
3. Train YOLO11n (`yolo11n.pt`) on the Jetson Nano GPU (CUDA device `0`).
4. Save all training outputs — trained weights (`best.pt`, `last.pt`), metrics
   (`results.csv`, `results.png`, `confusion_matrix.png`), and logs — under
   `output/train/`.

Hyperparameters (epochs, batch size, image size, device) are set in
[config.yaml](config.yaml) and can be overridden via CLI flags, e.g.:

```bash
python3 train.py --epochs 50 --batch 4 --device cpu
```
