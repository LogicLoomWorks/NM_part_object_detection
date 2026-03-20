from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1
    )


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(
        [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1
    )


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)

    enclose_x1 = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
    enclose_y1 = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
    enclose_x2 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
    enclose_y2 = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])

    enclose_w = (enclose_x2 - enclose_x1).clamp(min=0)
    enclose_h = (enclose_y2 - enclose_y1).clamp(min=0)
    enclose = enclose_w * enclose_h

    return iou - (enclose - union) / enclose.clamp(min=1e-6)


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ) -> None:
        super().__init__()
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        outputs: dict,
        targets: List[dict],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        pred_logits = outputs["pred_logits"]   # (B, Q, C)
        pred_boxes = outputs["pred_boxes"]     # (B, Q, 4)

        B, Q = pred_logits.shape[:2]
        sizes = [len(t["boxes"]) for t in targets]

        if sum(sizes) == 0:
            return [
                (
                    torch.empty(0, dtype=torch.long),
                    torch.empty(0, dtype=torch.long),
                )
                for _ in targets
            ]

        flat_logits = pred_logits.flatten(0, 1)   # (B*Q, C)
        flat_boxes = pred_boxes.flatten(0, 1)     # (B*Q, 4)

        tgt_labels = torch.cat([t["labels"] for t in targets], dim=0)
        tgt_boxes = torch.cat([t["boxes"] for t in targets], dim=0)

        probs = F.softmax(flat_logits, dim=-1)
        cost_class = -probs[:, tgt_labels]
        cost_bbox = torch.cdist(flat_boxes, tgt_boxes, p=1)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(flat_boxes),
            box_cxcywh_to_xyxy(tgt_boxes),
        )

        C = (
            self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
        ).view(B, Q, -1).cpu()

        indices = []
        start = 0
        for i, size in enumerate(sizes):
            if size == 0:
                indices.append((
                    torch.empty(0, dtype=torch.long),
                    torch.empty(0, dtype=torch.long),
                ))
                continue

            c_i = C[i, :, start:start + size]
            pred_idx, tgt_idx = linear_sum_assignment(c_i)
            indices.append((
                torch.as_tensor(pred_idx, dtype=torch.long),
                torch.as_tensor(tgt_idx, dtype=torch.long),
            ))
            start += size

        return indices