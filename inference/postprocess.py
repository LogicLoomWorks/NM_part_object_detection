"""Post-processing utilities: score thresholding and NMS."""
from __future__ import annotations

from typing import Dict, List

import torch
import torchvision.ops as ops

from models.groundingdino.matcher import box_cxcywh_to_xyxy


def threshold_filter(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    threshold: float = 0.3,
    top_k: int = 300,
) -> List[Dict[str, torch.Tensor]]:
    """Filter predictions by score and keep the top-k per image.

    Args:
        logits:    (B, Q, C) raw class logits
        boxes:     (B, Q, 4) predicted boxes in cx/cy/w/h normalised [0, 1]
        threshold: minimum score to retain a detection
        top_k:     maximum detections per image after thresholding
    Returns:
        list of {'scores', 'labels', 'boxes'} tensors per image
    """
    probs = logits.sigmoid()   # (B, Q, C) — per-class probabilities
    results: List[Dict[str, torch.Tensor]] = []

    for prob, box in zip(probs, boxes):
        scores, labels = prob.max(dim=-1)   # (Q,)
        keep = scores > threshold
        scores, labels, box = scores[keep], labels[keep], box[keep]

        if scores.numel() > top_k:
            topk_idx = scores.topk(top_k).indices
            scores = scores[topk_idx]
            labels = labels[topk_idx]
            box = box[topk_idx]

        results.append({"scores": scores, "labels": labels, "boxes": box})

    return results


def apply_nms(
    predictions: List[Dict[str, torch.Tensor]],
    iou_threshold: float = 0.5,
) -> List[Dict[str, torch.Tensor]]:
    """Apply class-aware NMS to each image's predictions.

    Boxes are converted internally from cx/cy/w/h to x1/y1/x2/y2 for NMS,
    then returned in the original cx/cy/w/h format.
    """
    filtered: List[Dict[str, torch.Tensor]] = []
    for pred in predictions:
        if pred["scores"].numel() == 0:
            filtered.append(pred)
            continue

        boxes_xyxy = box_cxcywh_to_xyxy(pred["boxes"])
        keep = ops.batched_nms(
            boxes_xyxy,
            pred["scores"],
            pred["labels"],
            iou_threshold,
        )
        filtered.append(
            {
                "scores": pred["scores"][keep],
                "labels": pred["labels"][keep],
                "boxes": pred["boxes"][keep],
            }
        )
    return filtered
