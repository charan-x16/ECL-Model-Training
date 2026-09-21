# ECL Model Training

Trains a six-class YOLO11n object-detection model. The default configuration is
optimized for a single NVIDIA GeForce RTX 5070 with 12 GB of VRAM.

## Requirements

- Linux or Windows with Python 3.10–3.12.
- NVIDIA driver 570 or newer. Blackwell GPUs such as the RTX 5070 require a
  CUDA 12.8+ PyTorch build.
- At least 150 GB of free disk space for this 59 GB archive, its approximately
  59 GB extracted copy, caches, and training outputs.

## Setup for RTX 5070

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/charan-x16/ECL-Model-Training.git
cd ECL-Model-Training
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell, use `py -m venv .venv`, then activate it with
`.\.venv\Scripts\Activate.ps1` and upgrade pip with the same command above.

Install the CUDA 12.8 builds of PyTorch and torchvision, followed by the project
dependencies:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Verify that PyTorch detects the RTX 5070:

```bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.version.cuda)"
```

The result should identify the RTX 5070 and CUDA 12.8. Do not start a long run
if the command raises an error.

The NumPy and OpenCV upper bounds in `requirements.txt` are intentional and
keep the project compatible with both desktop CUDA and Jetson environments.

## Dataset

The dataset archive is intentionally excluded from Git. After cloning on the
training machine, copy one `.tar` dataset into the repository root:

```bash
cp /path/to/unified_dataset.tar .
```

On Windows PowerShell, the equivalent is
`Copy-Item C:\path\to\unified_dataset.tar .`.

The first run extracts it into `data/`. Later runs reuse the extracted dataset.
Passing `--tar /path/to/file.tar` always extracts that archive; use
`--force-extract` to re-extract an auto-detected archive.

## Training

Start the configured 60-epoch run:

```bash
python train.py
```

The configured device is CUDA GPU `0`. If CUDA is unavailable, the script stops
with an error instead of silently beginning a much slower CPU run.
Ultralytics downloads `yolo11n.pt` on the first run if it is not already
available, so that first run needs internet access (or a manually copied
checkpoint).

The RTX profile in `config.yaml` uses:

- YOLO11n pretrained weights and 640-pixel images.
- Automatic batch sizing (`batch: -1`) targeting approximately 60% GPU-memory
  utilization.
- Eight data-loader workers.
- Automatic mixed precision (FP16) with the standard safety check enabled.
- Early stopping after 50 epochs without improvement.

For this dataset (24,588 training and 5,163 validation images), allow roughly
3–6 hours on an RTX 5070. The first completed epoch provides the most accurate
estimate for the particular CPU, SSD, cooling, and selected automatic batch
size.

Training outputs are written under `output/train/`, including:

- `weights/best.pt` and `weights/last.pt`
- `results.csv` and `results.png`
- confusion matrices and other validation plots

Common overrides:

```bash
python train.py --epochs 20 --run-name trial
python train.py --batch 32 --workers 4
python train.py --tar /path/to/dataset.tar
```

## Jetson note

Jetson uses platform-specific PyTorch wheels; do not install the desktop CUDA
wheel above on Jetson. Also, do not train on Jetson Linux 36.4.7: NVIDIA
documents a CUDA allocation regression in that release. The script detects it
and requests JetPack 6.2.2 / L4T 36.5 or newer instead.
