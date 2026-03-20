"""Hungarian matcher for DINO-style set prediction.

Assigns predicted queries to ground-truth boxes by minimising a joint cost:

    C = w_class * C_focal  +  w_bbox * C_L1  +  w_giou * C_GIoU

The assignment is solved per image with scipy's linear_sum_assignment.

Key correctness requirements
────────────────────────────
1.  Classification cost MUST use sigmoid focal, not softmax.
    - softmax assumes mutually-exclusive classes; focal does not.
    - Using softmax here while SetCriterion uses focal creates an inconsistency
      where the matcher optimises a different objective than the loss, leading
      to poor matching stability early in training.

2.  Boxes MUST be sanitised before GIoU.
    - An untrained model produces pred_boxes with arbitrary cx/cy/w/h values,
      including negative w or h.  After box_cxcywh_to_xyxy this gives x2 < x1,
      making the enclosing-box area zero and GIoU = 0/0 = NaN.
    - NaN in any cost entry makes the entire cost matrix invalid and causes
      scipy.optimize.linear_sum_assignment to raise ValueError.

3.  Everything must run in fp32.
    - Mixed-precision training passes fp16 tensors here.  sigmoid + log in fp16
      can overflow to ±inf for large-magnitude logits from an untrained model.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Box utilities
# ---------------------------------------------------------------------------

def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """(*, 4) cx,cy,w,h → x1,y1,x2,y2"""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1
    )


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """(*, 4) x1,y1,x2,y2 → cx,cy,w,h"""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(
        [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1
    )


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Generalised IoU between two sets of xyxy boxes.

    Args:
        boxes1: (M, 4)
        boxes2: (N, 4)
    Returns:
        giou:   (M, N)  values in [-1, 1]

    Both inputs must already be in xyxy format with x2 > x1 and y2 > y1.
    Use _safe_box_for_giou() to guarantee this for model outputs.
    """
    # Intersection
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter   = inter_w * inter_h                                    # (M, N)

    # Union
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
             * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))        # (M,)
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
             * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))        # (N,)
    union = area1[:, None] + area2[None, :] - inter               # (M, N)
    iou   = inter / union.clamp(min=1e-6)

    # Enclosing box
    enc_x1 = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
    enc_y1 = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
    enc_x2 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
    enc_y2 = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])

    enc_w   = (enc_x2 - enc_x1).clamp(min=0)
    enc_h   = (enc_y2 - enc_y1).clamp(min=0)
    enclose = enc_w * enc_h                                        # (M, N)

    return iou - (enclose - union) / enclose.clamp(min=1e-6)


def _safe_box_for_giou(boxes: torch.Tensor) -> torch.Tensor:
    """Convert cx,cy,w,h predictions to a valid xyxy box for GIoU computation.

    Two guards are applied:
      1. clamp(0, 1)           — keeps coordinates on the unit image canvas.
      2. x2 = max(x2, x1+eps) — ensures strictly positive width/height so
                                  the enclosing-box area is never zero,
                                  preventing the 0/0 = NaN in GIoU.

    This is needed because an untrained model outputs unconstrained cx/cy/w/h;
    a negative w produces x2 < x1 after conversion, making area = 0.
    """
    xyxy = box_cxcywh_to_xyxy(boxes).clamp(0.0, 1.0)
    x1, y1, x2, y2 = xyxy.unbind(-1)
    x2 = torch.maximum(x2, x1 + 1e-4)
    y2 = torch.maximum(y2, y1 + 1e-4)
    return torch.stack([x1, y1, x2, y2], dim=-1)


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------

class HungarianMatcher(nn.Module):
    """Optimal bipartite matching between predictions and ground-truth boxes.

    Args:
        cost_class: weight for the focal classification cost term.
        cost_bbox:  weight for the L1 box regression cost term.
        cost_giou:  weight for the GIoU box regression cost term.
        focal_alpha: α parameter of the focal cost (mirrors SetCriterion).
        focal_gamma: γ parameter of the focal cost (mirrors SetCriterion).
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox:  float = 5.0,
        cost_giou:  float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, \
            "At least one cost term must be non-zero."
        self.cost_class  = cost_class
        self.cost_bbox   = cost_bbox
        self.cost_giou   = cost_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def forward(
        self,
        outputs: dict,
        targets: List[dict],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Compute the optimal assignment for a batch.

        Args:
            outputs: dict with keys
                "pred_logits": (B, Q, C)  raw class logits
                "pred_boxes":  (B, Q, 4)  cx,cy,w,h — NOT required to be in [0,1]
            targets: list of B dicts, each with
                "labels": (N,)   int64 class indices
                "boxes":  (N, 4) float32 cx,cy,w,h in [0, 1]

        Returns:
            List of B tuples (pred_indices, tgt_indices), each a pair of
            1-D LongTensors giving the matched rows in pred and target.
        """
        # Cast to fp32: mixed-precision passes fp16, which overflows in
        # sigmoid + log for large-magnitude logits from an untrained model.
        pred_logits = outputs["pred_logits"].float()   # (B, Q, C)
        pred_boxes  = outputs["pred_boxes"].float()    # (B, Q, 4)

        B, Q    = pred_logits.shape[:2]
        sizes   = [len(t["boxes"]) for t in targets]

        # Fast path: no ground-truth boxes in the entire batch
        if sum(sizes) == 0:
            return [
                (torch.empty(0, dtype=torch.long),
                 torch.empty(0, dtype=torch.long))
                for _ in targets
            ]

        # Flatten batch dimension for vectorised cost computation
        flat_logits = pred_logits.flatten(0, 1)        # (B*Q, C)
        flat_boxes  = pred_boxes.flatten(0, 1)         # (B*Q, 4)

        tgt_labels  = torch.cat([t["labels"] for t in targets], dim=0)   # (total_N,)
        tgt_boxes   = torch.cat([t["boxes"]  for t in targets], dim=0).float()  # (total_N, 4)

        # ── Classification cost ───────────────────────────────────────────────
        # Focal-style cost that mirrors the sigmoid_focal_loss in SetCriterion.
        #
        # For each query q and target n, the cost is:
        #   C_class[q,n] = -alpha*(1-p)^gamma*log(p)        ← positive-class term
        #                  +(1-alpha)*p^gamma*log(p)         ← subtract negative-class term
        #
        # We only evaluate at the column corresponding to tgt_labels[n] because
        # all other classes contribute identically to every assignment and cancel
        # out in the optimisation.
        #
        # The prob clamp keeps log inputs away from 0 to prevent -inf,
        # which would make the cost matrix invalid for linear_sum_assignment.
        alpha, gamma = self.focal_alpha, self.focal_gamma
        prob     = flat_logits.sigmoid().clamp(1e-6, 1.0 - 1e-6)  # (B*Q, C)
        neg_cost = (1 - alpha) * (prob ** gamma) * torch.log(prob)
        pos_cost = alpha * ((1 - prob) ** gamma) * torch.log(1 - prob)
        # Select columns for each target's true class
        cost_class = (-pos_cost + neg_cost)[:, tgt_labels]        # (B*Q, total_N)

        # ── L1 box cost ───────────────────────────────────────────────────────
        # Simple L1 distance in cx,cy,w,h space between every pair.
        cost_bbox = torch.cdist(flat_boxes, tgt_boxes, p=1)       # (B*Q, total_N)

        # ── GIoU box cost ─────────────────────────────────────────────────────
        # Negated so lower cost = higher overlap.
        # _safe_box_for_giou handles degenerate predictions (negative w/h).
        cost_giou = -generalized_box_iou(
            _safe_box_for_giou(flat_boxes),
            _safe_box_for_giou(tgt_boxes),
        )                                                          # (B*Q, total_N)

        # ── Combined cost matrix ──────────────────────────────────────────────
        C = (
            self.cost_class * cost_class
            + self.cost_bbox  * cost_bbox
            + self.cost_giou  * cost_giou
        )                                                          # (B*Q, total_N)

        # Last-resort NaN/Inf guard.
        # Should never fire after the guards above; if it does, a warning is
        # emitted so the caller knows to investigate the model outputs.
        if not torch.isfinite(C).all():
            import warnings
            warnings.warn(
                "[HungarianMatcher] Non-finite values in cost matrix after all "
                "guards — replacing with large finite penalty. This should not "
                "happen; inspect pred_logits and pred_boxes for NaN/Inf.",
                RuntimeWarning,
                stacklevel=2,
            )
            C = torch.nan_to_num(C, nan=1e4, posinf=1e4, neginf=-1e4)

        C = C.view(B, Q, -1).cpu()                                # (B, Q, total_N)

        # ── Per-image Hungarian assignment ────────────────────────────────────
        indices = []
        start   = 0
        for i, size in enumerate(sizes):
            if size == 0:
                # Image has no GT boxes — return empty assignment
                indices.append((
                    torch.empty(0, dtype=torch.long),
                    torch.empty(0, dtype=torch.long),
                ))
                continue

            c_i = C[i, :, start : start + size]                   # (Q, size)
            pred_idx, tgt_idx = linear_sum_assignment(c_i.numpy())
            indices.append((
                torch.as_tensor(pred_idx, dtype=torch.long),
                torch.as_tensor(tgt_idx,  dtype=torch.long),
            ))
            start += size

        return indices