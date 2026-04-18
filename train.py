"""
train.py — Train the PPLCNet detector on the oyster mushroom dataset.

Usage
-----
    python train.py --train_img dataset/train/images \
                    --train_lbl dataset/train/labels \
                    --val_img   dataset/val/images   \
                    --val_lbl   dataset/val/labels   \
                    --epochs    20                    \
                    --lr        1e-3                  \
                    --batch     8
"""

import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import OysterDataset
from pplcnet import PPLCNetDetector


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train PPLCNet Oyster Detector")
    p.add_argument("--train_img", default="dataset/train/images")
    p.add_argument("--train_lbl", default="dataset/train/labels")
    p.add_argument("--val_img",   default="dataset/val/images")
    p.add_argument("--val_lbl",   default="dataset/val/labels")
    p.add_argument("--epochs",    type=int,   default=20)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--batch",     type=int,   default=8)
    p.add_argument("--img_size",  type=int,   default=640)
    p.add_argument("--save_path", default="pplcnet_oyster_final.pt")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Datasets & loaders
    train_ds = OysterDataset(args.train_img, args.train_lbl, args.img_size)
    val_ds   = OysterDataset(args.val_img,   args.val_lbl,   args.img_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2)

    # Model, optimiser, loss
    model     = PPLCNetDetector(num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for imgs, labels in train_loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)

            # Append dummy confidence=1 so shape matches model output [B, 6]
            conf   = torch.ones((labels.size(0), 1), device=device)
            labels = torch.cat([labels, conf], dim=1)

            preds = model(imgs)
            loss  = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device)
                labels = labels.to(device)

                conf   = torch.ones((labels.size(0), 1), device=device)
                labels = torch.cat([labels, conf], dim=1)

                preds    = model(imgs)
                val_loss += criterion(preds, labels).item()

        print(
            f"Epoch [{epoch:>3}/{args.epochs}]  "
            f"Train Loss: {train_loss / len(train_loader):.6f}  "
            f"Val Loss: {val_loss / len(val_loader):.6f}"
        )

    # Save weights
    torch.save(model.state_dict(), args.save_path)
    print(f"\n✅ Model saved to {args.save_path}")


if __name__ == "__main__":
    train(parse_args())
