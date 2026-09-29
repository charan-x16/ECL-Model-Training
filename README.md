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

## Improving the model: find, relabel, retrain

The confusion matrix (`output/train/confusion_matrix.png`) shows how many
mistakes the model makes, but not which frames they are in. Two scripts close
that loop:

- `find_false_frames.py` finds the exact frames behind every mistake.
- `cvat_roundtrip.py` sends those frames to CVAT for relabelling and merges
  the corrected labels back into the dataset.

### The workflow at a glance

| Step | Command | What it does |
|------|---------|--------------|
| 1 | `python train.py` | Train the model; creates `output/train/weights/best.pt`. |
| 2 | `python find_false_frames.py` | Find every wrong or missed detection in the validation set. |
| 3 | `python cvat_roundtrip.py package` | Zip the flagged frames and their labels for CVAT. |
| – | *CVAT (manual)* | Import the zip, fix the labels, export the dataset. |
| 4 | `python cvat_roundtrip.py merge <export.zip> --move-from-val` | Add the fixed frames to the training split. |
| 5 | `python train.py --run-name train_v2` | Retrain on the improved dataset. |

Steps 2–5 need the extracted dataset in `data/` and a finished training run.
Repeat them until the off-diagonal counts in the confusion matrix stop falling.

### Step 2: find the false frames

```bash
python find_false_frames.py
```

The model is run over the validation split and each prediction is matched to
the labels the same way Ultralytics builds its confusion matrix (confidence
0.25, IoU 0.45), so the results line up with the matrix cells:

| Category | Meaning | Confusion-matrix cell |
|----------|---------|-----------------------|
| `fp_background` | Predicted an object where nothing is labelled | row = class, column = background |
| `wrong_class` | Predicted a different class than the label | row = predicted, column = true class |
| `missed` | A labelled object was not detected | row = background, column = class |

Results are written to `output/train/false_frames_val/`:

```
false_frames_val/
├── reference/<class>/<category>/   annotated frames for review
├── training/images/, labels/       clean original frames and their label files
└── report.csv                      one row per mistake
```

- **`reference/`** – copies with boxes drawn: green = ground-truth label,
  red = the box at fault. File names start with the confidence
  (e.g. `0.91_frame_0877_0.jpg`), so the most confident mistakes sort first.
  These are for looking at only; never train on them.
- **`training/`** – untouched originals with their YOLO labels, used by the
  next step.
- **`report.csv`** – category, predicted class, true class, original image
  path, confidence, box coordinates, and the paths of both copies.

A summary per confusion-matrix cell is printed at the end.

Common options:

```bash
python find_false_frames.py --classes human pipe      # only these classes
python find_false_frames.py --run-name train_v2       # a different training run
python find_false_frames.py --weights path/to/best.pt # any weights file
python find_false_frames.py --split train             # scan the training split
python find_false_frames.py --device cpu              # no GPU available
python find_false_frames.py --no-missed               # skip missed objects
```

> Many high-confidence `fp_background` frames are usually **labelling
> mistakes** (for example a person who was never labelled) rather than model
> mistakes. Fixing those labels helps more than adding data.

### Step 3: package the frames for CVAT

```bash
python cvat_roundtrip.py package                          # all flagged frames
python cvat_roundtrip.py package --categories missed      # only missed objects
python cvat_roundtrip.py package --classes human          # only frames involving humans
python cvat_roundtrip.py package --limit 500              # the 500 most confident mistakes
```

This creates `output/train/cvat_upload_val.zip` in CVAT's **YOLO 1.1** format,
with each frame's current labels included.

### In CVAT (manual)

1. Create a **project** with exactly these labels:
   `pipe`, `trolley`, `gate1`, `gate2`, `human`, `others`.
2. On the project, choose **Import dataset**, select format **YOLO 1.1**, and
   upload the zip. The frames arrive with their existing boxes.
3. Correct the labels. Use `false_frames_val/reference/` to see what is wrong
   in each frame.
4. Choose **Export dataset**, select **YOLO 1.1** (or **Ultralytics YOLO
   Detection**), and turn **Save images** on.

### Step 4: merge the corrected frames into the dataset

Preview first, then merge:

```bash
python cvat_roundtrip.py merge path/to/cvat_export.zip --move-from-val --dry-run
python cvat_roundtrip.py merge path/to/cvat_export.zip --move-from-val
```

- The corrected images and labels are added to the training split in `data/`.
- Class ids are remapped **by name**, so a different label order in CVAT
  cannot corrupt the labels. An unknown label name stops the merge.
- `--move-from-val` moves the original copies out of the validation split into
  `output/train/removed_from_val/` (a backup, not a deletion). Without it the
  same frames would be in both training and validation, and validation scores
  would look better than the model really is.
- A frame whose name already exists in the training split is saved with a
  `_cvat` suffix instead of overwriting it.

> The merge edits `data/` in place. `--force-extract` or `--tar` re-extracts
> the original archive and discards merged frames, so keep the CVAT export zip
> to merge it again. If many frames move out of validation, add fresh labelled
> frames to it so the scores stay reliable.

### Step 5: retrain

```bash
python train.py --run-name train_v2
```

Use a new run name so the previous run's weights and confusion matrix are
kept for comparison.

### The next round

Pass the new run name to each step:

```bash
python find_false_frames.py --run-name train_v2
python cvat_roundtrip.py --run-name train_v2 package
python cvat_roundtrip.py --run-name train_v2 merge path/to/cvat_export.zip --move-from-val
python train.py --run-name train_v3
```

## Jetson note

Jetson uses platform-specific PyTorch wheels; do not install the desktop CUDA
wheel above on Jetson. Also, do not train on Jetson Linux 36.4.7: NVIDIA
documents a CUDA allocation regression in that release. The script detects it
and requests JetPack 6.2.2 / L4T 36.5 or newer instead.