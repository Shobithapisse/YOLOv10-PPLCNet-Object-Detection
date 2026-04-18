"""
predict.py — Run inference with the trained PPLCNet detector.

Usage
-----
    python predict.py --image path/to/image.jpg --weights pplcnet_oyster_final.pt
"""

import argparse

import cv2
import numpy as np
import torch

from pplcnet import PPLCNetDetector


def parse_args():
    p = argparse.ArgumentParser(description="PPLCNet Oyster Detector — Inference")
    p.add_argument("--image",    required=True, help="Path to input image")
    p.add_argument("--weights",  default="pplcnet_oyster_final.pt")
    p.add_argument("--img_size", type=int, default=640)
    p.add_argument("--conf",     type=float, default=0.5, help="Confidence threshold")
    p.add_argument("--output",   default="prediction.jpg")
    return p.parse_args()


def yolo_to_pixels(cx, cy, w, h, img_w, img_h):
    """Convert normalised YOLO coords → pixel (x1, y1, x2, y2)."""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2


def predict(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = PPLCNetDetector(num_classes=1).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # Load & preprocess image
    orig = cv2.imread(args.image)
    if orig is None:
        raise FileNotFoundError(f"Image not found: {args.image}")

    orig_h, orig_w = orig.shape[:2]

    img = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (args.img_size, args.img_size))
    tensor = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor = tensor.to(device)

    # Inference
    with torch.no_grad():
        pred = model(tensor).cpu().numpy()[0]  # [class, cx, cy, w, h, conf]

    _, cx, cy, w, h, conf = pred
    print(f"Prediction → cx:{cx:.4f}  cy:{cy:.4f}  w:{w:.4f}  h:{h:.4f}  conf:{conf:.4f}")

    if conf >= args.conf:
        x1, y1, x2, y2 = yolo_to_pixels(cx, cy, w, h, orig_w, orig_h)
        cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"oyster_mushroom {conf:.2f}"
        cv2.putText(orig, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        print(f"No detection above confidence threshold ({args.conf}).")

    cv2.imwrite(args.output, orig)
    print(f"✅ Result saved to {args.output}")


if __name__ == "__main__":
    predict(parse_args())
