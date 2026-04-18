# YOLOv10 + PPLCNet — Oyster Mushroom Detection

A lightweight object detection system that combines the **YOLOv10** detection paradigm with a custom **PP-LCNet** (Lightweight CPU Convolutional Network) backbone for single-class oyster mushroom detection.

> Published research: *"Oyster Mushroom Detection using YOLOv10 and PPLCNet"* — IEEE

---

## Project Structure

```
yolov10-pplcnet/
│
├── pplcnet.py          # PPLCNet backbone + DetectionHead + PPLCNetDetector
├── dataset.py          # OysterDataset (PyTorch Dataset, YOLO-format labels)
├── train.py            # Training script (CLI)
├── evaluate.py         # Validation loss + mean IoU
├── predict.py          # Single-image inference + bounding-box output
│
├── data.yaml           # Dataset config (paths, class names)
├── requirements.txt    # Python dependencies
│
├── YOLOv10_PPLCNet_OysterMushroom.ipynb   # End-to-end walkthrough notebook
└── dataset/            # ← NOT tracked in git (see below)
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
```

---

## Model Architecture

```
Input (3 × 640 × 640)
       │
  ┌────▼────────────────────────────────────┐
  │  PPLCNet Backbone                        │
  │  Stage 1: ConvBNReLU  3→32   stride 2   │
  │  Stage 2: DWConv      32→64  stride 2   │
  │  Stage 3: DWConv      64→128 stride 2   │
  │  Stage 4: DWConv      128→256 stride 2  │
  │  Stage 5: DWConv      256→512 stride 2  │
  └────────────────────────────┬────────────┘
                               │ (B, 512, 20, 20)
                  ┌────────────▼────────────┐
                  │  Detection Head          │
                  │  AdaptiveAvgPool2d(1)    │
                  │  Linear(512 → 6)         │
                  └────────────┬────────────┘
                               │
              [class, cx, cy, w, h, conf]
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/yolov10-pplcnet.git
cd yolov10-pplcnet
pip install -r requirements.txt
```

### 2. Prepare the dataset

Organise your data in YOLO format:

```
dataset/
  train/images/*.jpg
  train/labels/*.txt   # one line per image: class cx cy w h  (normalised)
  val/images/*.jpg
  val/labels/*.txt
```

Update `data.yaml` if your paths differ.

### 3. Train

```bash
python train.py \
  --train_img dataset/train/images \
  --train_lbl dataset/train/labels \
  --val_img   dataset/val/images   \
  --val_lbl   dataset/val/labels   \
  --epochs    20 \
  --lr        1e-3 \
  --batch     8
```

Weights are saved to `pplcnet_oyster_final.pt`.

### 4. Evaluate

```bash
python evaluate.py --weights pplcnet_oyster_final.pt
```

Outputs validation loss and mean IoU across all validation images.

### 5. Predict on a single image

```bash
python predict.py --image path/to/mushroom.jpg --weights pplcnet_oyster_final.pt
```

Result image is saved as `prediction.jpg`.

---

## Notebook

Open `YOLOv10_PPLCNet_OysterMushroom.ipynb` for a step-by-step walkthrough including visualisations of ground-truth vs predicted bounding boxes.

**Google Colab users:** Mount your Drive and unzip the dataset at the top of the notebook:

```python
from google.colab import drive
drive.mount('/content/drive')
!unzip /content/drive/MyDrive/oyster_mushroom_yolo.zip -d .
```

---

## Dataset

The dataset is **not included** in this repository due to size.  
Label format: YOLO (`.txt` files, one object per line — `class cx cy w h`, normalised 0–1).

---

## Results

| Metric          | Value  |
|-----------------|--------|
| Training Loss   | —      |
| Validation Loss | —      |
| Mean IoU        | —      |

*(Fill in after running training on your dataset.)*

---

## Dependencies

| Package       | Version  |
|---------------|----------|
| torch         | ≥ 2.0    |
| torchvision   | ≥ 0.15   |
| ultralytics   | ≥ 8.3    |
| opencv-python | ≥ 4.6    |
| numpy         | ≥ 1.23   |
| matplotlib    | ≥ 3.3    |

---

## Citation

If you use this work, please cite the associated IEEE paper:

```
@inproceedings{yourname2024oyster,
  title     = {Oyster Mushroom Detection using YOLOv10 and PPLCNet},
  author    = {<Your Name>},
  booktitle = {IEEE ...},
  year      = {2024}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
