"""
Hungarian matcher — pairs predicted boxes/logits with GT targets.
Self-contained: _generalised_iou is defined locally.
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


def _generalised_iou(boxes1: torch.Tensor,
                     boxes2: torch.Tensor) -> torch.Tensor:
    """GIoU between two sets of boxes in normalised cx/cy/w/h format."""
    b1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
    b1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
    b1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
    b1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2

    b2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
    b2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
    b2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
    b2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2

    inter_x1 = torch.max(b1_x1[:, None], b2_x1[None, :])
    inter_y1 = torch.max(b1_y1[:, None], b2_y1[None, :])
    inter_x2 = torch.min(b1_x2[:, None], b2_x2[None, :])
    inter_y2 = torch.min(b1_y2[:, None], b2_y2[None, :])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter   = inter_w * inter_h

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = area1[:, None] + area2[None, :] - inter

    iou = inter / union.clamp(min=1e-6)

    enc_x1 = torch.min(b1_x1[:, None], b2_x1[None, :])
    enc_y1 = torch.min(b1_y1[:, None], b2_y1[None, :])
    enc_x2 = torch.max(b1_x2[:, None], b2_x2[None, :])
    enc_y2 = torch.max(b1_y2[:, None], b2_y2[None, :])

    enc_area = ((enc_x2 - enc_x1).clamp(min=0) *
                (enc_y2 - enc_y1).clamp(min=0))

    return iou - (enc_area - union) / enc_area.clamp(min=1e-6)


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox  = cost_bbox
        self.cost_giou  = cost_giou

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list) -> list:
        B, Q, C = outputs["pred_logits"].shape
        indices = []

        for b in range(B):
            logits     = outputs["pred_logits"][b]   # (Q, C)
            boxes      = outputs["pred_boxes"][b]    # (Q, 4)
            tgt_labels = targets[b]["labels"]
            tgt_boxes  = targets[b]["boxes"]

            N = tgt_labels.shape[0]
            if N == 0:
                indices.append((
                    torch.zeros(0, dtype=torch.long),
                    torch.zeros(0, dtype=torch.long),
                ))
                continue

            probs    = logits.sigmoid()
            alpha, gamma = 0.25, 2.0
            neg_cost = (1 - alpha) * (probs ** gamma) * (-(1 - probs + 1e-8).log())
            pos_cost = alpha * ((1 - probs) ** gamma) * (-(probs + 1e-8).log())
            cost_cls = pos_cost[:, tgt_labels] - neg_cost[:, tgt_labels]

            cost_l1 = torch.cdist(boxes, tgt_boxes, p=1)
            cost_g  = -_generalised_iou(boxes, tgt_boxes)

            C_mat = (
                self.cost_class * cost_cls +
                self.cost_bbox  * cost_l1 +
                self.cost_giou  * cost_g
            ).cpu().numpy()

            row_ind, col_ind = linear_sum_assignment(C_mat)
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.long),
                torch.as_tensor(col_ind, dtype=torch.long),
            ))

        return indices


def build_matcher(cfg) -> HungarianMatcher:
    return HungarianMatcher(
        cost_class=cfg.cost_class,
        cost_bbox=cfg.cost_bbox,
        cost_giou=cfg.cost_giou,
    )