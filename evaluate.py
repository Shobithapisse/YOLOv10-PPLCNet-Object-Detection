"""
evaluate.py — Compute IoU-based evaluation metrics on the validation set.

Usage
-----
    python evaluate.py --val_img dataset/val/images \
                       --val_lbl dataset/val/labels \
                       --weights pplcnet_oyster_final.pt
"""

import argparse

import torch
from torch.utils.data import DataLoader

from dataset import OysterDataset
from pplcnet import PPLCNetDetector


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PPLCNet Oyster Detector")
    p.add_argument("--val_img",  default="dataset/val/images")
    p.add_argument("--val_lbl",  default="dataset/val/labels")
    p.add_argument("--weights",  default="pplcnet_oyster_final.pt")
    p.add_argument("--img_size", type=int, default=640)
    p.add_argument("--batch",    type=int, default=8)
    return p.parse_args()


def compute_iou(pred_box, gt_box):
    """
    Compute IoU between two boxes in [cx, cy, w, h] normalised format.
    """
    def to_corners(cx, cy, w, h):
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    px1, py1, px2, py2 = to_corners(*pred_box)
    gx1, gy1, gx2, gy2 = to_corners(*gt_box)

    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (px2 - px1) * (py2 - py1) + (gx2 - gx1) * (gy2 - gy1) - inter

    return inter / union if union > 0 else 0.0


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PPLCNetDetector(num_classes=1).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    val_ds     = OysterDataset(args.val_img, args.val_lbl, args.img_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    total_iou  = 0.0
    total_loss = 0.0
    criterion  = torch.nn.MSELoss()
    n_samples  = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)

            preds = model(imgs)  # (B, 6)

            conf   = torch.ones((labels.size(0), 1), device=device)
            labels_aug = torch.cat([labels, conf], dim=1)
            total_loss += criterion(preds, labels_aug).item()

            # Compute per-sample IoU
            for pred, gt in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                pred_box = pred[1:5]  # cx, cy, w, h
                gt_box   = gt[1:5]
                total_iou += compute_iou(pred_box, gt_box)
                n_samples += 1

    mean_iou  = total_iou / n_samples if n_samples else 0
    mean_loss = total_loss / len(val_loader)

    print(f"Validation Loss : {mean_loss:.6f}")
    print(f"Mean IoU        : {mean_iou:.4f}  ({n_samples} samples)")


if __name__ == "__main__":
    evaluate(parse_args())
