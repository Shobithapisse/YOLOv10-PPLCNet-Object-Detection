"""
OysterDataset — PyTorch Dataset for YOLO-format oyster mushroom data.
"""

import os
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class OysterDataset(Dataset):
    """
    Loads images and YOLO-format bounding-box labels for oyster mushroom detection.

    Directory layout expected
    ─────────────────────────
    dataset/
      train/
        images/   *.jpg
        labels/   *.txt   (one line: class cx cy w h, normalised)
      val/
        images/
        labels/

    Parameters
    ----------
    image_dir : str | Path
        Path to the folder containing .jpg images.
    label_dir : str | Path
        Path to the folder containing matching .txt label files.
    img_size  : int
        Square size to which every image is resized (default 640).
    """

    def __init__(self, image_dir: str, label_dir: str, img_size: int = 640):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.images = sorted(os.listdir(self.image_dir))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_name = self.images[idx]
        img_path = self.image_dir / img_name
        label_path = self.label_dir / img_name.replace(".jpg", ".txt")

        # ── Load & preprocess image ──────────────────────────────────────
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img_tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0  # (3, H, W)

        # ── Load label (single object per image assumed) ──────────────────
        if label_path.exists():
            with open(label_path) as f:
                label = torch.tensor(list(map(float, f.readline().split())))
        else:
            label = torch.zeros(5)

        return img_tensor, label
