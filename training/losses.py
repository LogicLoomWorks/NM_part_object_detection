"""Set prediction losses: sigmoid focal + L1 + GIoU with Hungarian matching.

SetCriterion follows the DETR/DINO convention:
  1. Run HungarianMatcher to assign predictions → targets
  2. Compute per-matched-pair box losses (L1 + GIoU)
  3. Compute focal classification loss over all queries
  4. Repeat for each intermediate decoder layer (auxiliary loss)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.groundingdino.matcher import (
    HungarianMatcher,
    box_cxcywh_to_xyxy,
    generalized_box_iou,
)


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Sigmoid focal loss (per element, summed over all elements)."""
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * ce).sum()


class SetCriterion(nn.Module):
    """Detection loss combining focal classification, L1, and GIoU box losses.

    ``weight_dict`` maps each loss key to its scalar coefficient. The total
    loss is the weighted sum; individual terms are also logged for monitoring.
    """

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: Dict[str, float],
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _src_idx(
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = torch.cat([torch.full_like(s, i) for i, (s, _) in enumerate(indices)])
        src = torch.cat([s for s, _ in indices])
        return batch, src

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def _loss_class(
        self,
        outputs: Dict,
        targets: List[Dict],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int,
    ) -> torch.Tensor:
        pred_logits = outputs["pred_logits"]          # (B, Q, C)
        b_idx, s_idx = self._src_idx(indices)
        tgt_labels = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)])

        # One-hot target tensor; matched positions get 1 at the true class
        target_onehot = torch.zeros_like(pred_logits)
        target_onehot[b_idx, s_idx, tgt_labels] = 1.0

        return sigmoid_focal_loss(
            pred_logits,
            target_onehot,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        ) / num_boxes

    def _loss_boxes(
        self,
        outputs: Dict,
        targets: List[Dict],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b_idx, s_idx = self._src_idx(indices)
        pred_boxes = outputs["pred_boxes"][b_idx, s_idx]          # (M, 4)
        tgt_boxes = torch.cat(
            [t["boxes"][j] for t, (_, j) in zip(targets, indices)]
        )                                                           # (M, 4)

        loss_l1 = F.l1_loss(pred_boxes, tgt_boxes, reduction="sum") / num_boxes

        giou = generalized_box_iou(
            box_cxcywh_to_xyxy(pred_boxes),
            box_cxcywh_to_xyxy(tgt_boxes),
        )
        loss_giou = (1 - giou.diag()).sum() / num_boxes
        return loss_l1, loss_giou

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(
        self,
        outputs: Dict,
        targets: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        indices = self.matcher(outputs, targets)
        num_boxes = max(1, sum(len(t["labels"]) for t in targets))

        loss_ce = self._loss_class(outputs, targets, indices, num_boxes)
        loss_l1, loss_giou = self._loss_boxes(outputs, targets, indices, num_boxes)

        losses: Dict[str, torch.Tensor] = {
            "loss_ce": loss_ce,
            "loss_bbox": loss_l1,
            "loss_giou": loss_giou,
        }

        # Auxiliary decoder layer losses
        for i, aux in enumerate(outputs.get("aux_outputs", [])):
            aux_idx = self.matcher(aux, targets)
            aux_ce = self._loss_class(aux, targets, aux_idx, num_boxes)
            aux_l1, aux_giou = self._loss_boxes(aux, targets, aux_idx, num_boxes)
            losses[f"loss_ce_{i}"] = aux_ce
            losses[f"loss_bbox_{i}"] = aux_l1
            losses[f"loss_giou_{i}"] = aux_giou

        losses["loss_total"] = sum(
            self.weight_dict.get(k, 1.0) * v for k, v in losses.items()
        )
        return losses
