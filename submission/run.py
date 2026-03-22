"""
submission/run.py

NorgesGruppen product detection — competition entry point.

Usage:
    python run.py --input /data/images --output /output/predictions.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.ops as ops

from model import DEIMv2Visual, load_model


# ── Constants ─────────────────────────────────────────────────────────────────

WEIGHTS_FILE = Path(__file__).parent / "deimv2_fp16.pt"

# ImageNet normalisation (used during training)
_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

# Inference hyper-parameters (from configs/deimv2.yaml)
SCORE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.5
TOP_K = 300
MAX_SIZE = 480    # longest-edge resize used during training


# ── Image pre-processing ──────────────────────────────────────────────────────

def preprocess(image_bgr: np.ndarray) -> tuple[torch.Tensor, int, int, int, int]:
    """BGR uint8 → normalised float tensor (1,3,H,W).

    Returns (tensor, pad_h, pad_w, new_h, new_w).
    new_h/new_w are the pre-pad resized dimensions (image content region).
    """
    h, w = image_bgr.shape[:2]
    scale = MAX_SIZE / max(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Pad to multiple of 32
    pad_h = ((new_h + 31) // 32) * 32
    pad_w = ((new_w + 31) // 32) * 32
    canvas = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = rgb

    tensor = torch.from_numpy(canvas).float() / 255.0    # (H, W, 3)
    tensor = (tensor - _MEAN) / _STD
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)        # (1, 3, H, W)
    return tensor, pad_h, pad_w, new_h, new_w


def postprocess(
    logits: torch.Tensor,   # (1, Q, C)
    boxes: torch.Tensor,    # (1, Q, 4) cx/cy/w/h in [0,1]
    orig_h: int,
    orig_w: int,
    pad_h: int,
    pad_w: int,
    new_h: int,
    new_w: int,
) -> list[dict]:
    """Convert model outputs to COCO prediction dicts."""
    probs = logits[0].sigmoid()              # (Q, C)
    scores, labels = probs.max(dim=-1)       # (Q,)

    keep = scores > SCORE_THRESHOLD
    scores = scores[keep]
    labels = labels[keep]
    b = boxes[0][keep]                        # (N, 4) cx/cy/w/h in [0,1]

    if scores.numel() == 0:
        return []

    # Top-K
    if scores.numel() > TOP_K:
        topk_idx = scores.topk(TOP_K).indices
        scores = scores[topk_idx]
        labels = labels[topk_idx]
        b = b[topk_idx]

    # Convert cx/cy/w/h (padded space) → x1/y1/x2/y2 (original image space)
    cx = b[:, 0] * pad_w
    cy = b[:, 1] * pad_h
    bw = b[:, 2] * pad_w
    bh = b[:, 3] * pad_h

    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    # Scale back to original image.
    # Boxes were normalized by padded dims during training, so denorm by pad_h/pad_w
    # gives pixel coords in the padded image. The image content occupies [0, new_w] x
    # [0, new_h] (top-left), so we scale by orig/new (not orig/pad).
    scale_x = orig_w / new_w
    scale_y = orig_h / new_h
    x1 = x1 * scale_x
    y1 = y1 * scale_y
    x2 = x2 * scale_x
    y2 = y2 * scale_y

    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

    # Class-aware NMS
    keep_idx = ops.batched_nms(boxes_xyxy, scores, labels, NMS_IOU_THRESHOLD)
    x1 = x1[keep_idx].clamp(min=0)
    y1 = y1[keep_idx].clamp(min=0)
    x2 = x2[keep_idx].clamp(max=orig_w)
    y2 = y2[keep_idx].clamp(max=orig_h)
    w_out = (x2 - x1).clamp(min=1e-3)
    h_out = (y2 - y1).clamp(min=1e-3)
    scores = scores[keep_idx]
    labels = labels[keep_idx]

    results = []
    for i in range(len(keep_idx)):
        results.append({
            "bbox": [
                float(x1[i]),
                float(y1[i]),
                float(w_out[i]),
                float(h_out[i]),
            ],
            "score": float(scores[i]),
            "label": int(labels[i]),
        })
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NorgesGruppen product detection inference"
    )
    parser.add_argument("--input", required=True, help="Directory of input images")
    parser.add_argument("--output", required=True, help="Path to write predictions.json")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = load_model(str(WEIGHTS_FILE), device=device)

    # Collect image files
    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    predictions = []

    with torch.no_grad():
        for img_path in image_files:
            # Extract numeric image_id from filename (img_00042.jpg → 42)
            stem = img_path.stem   # e.g. "img_00042"
            digits = re.sub(r"[^0-9]", "", stem)
            image_id = int(digits) if digits else 0

            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                continue

            orig_h, orig_w = image_bgr.shape[:2]
            tensor, pad_h, pad_w, new_h, new_w = preprocess(image_bgr)
            tensor = tensor.to(device)

            outputs = model(tensor)
            preds = postprocess(
                outputs["pred_logits"],
                outputs["pred_boxes"],
                orig_h=orig_h, orig_w=orig_w,
                pad_h=pad_h, pad_w=pad_w,
                new_h=new_h, new_w=new_w,
            )

            for p in preds:
                cat_id = int(p["label"])
                # Clamp to valid range [0, 355]
                cat_id = max(0, min(355, cat_id))
                predictions.append({
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": p["bbox"],
                    "score": p["score"],
                })

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
