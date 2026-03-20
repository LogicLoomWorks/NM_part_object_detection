"""Set prediction losses: sigmoid focal + L1 + GIoU with Hungarian matching.

SetCriterion follows the DETR/DINO convention:
  1. Run HungarianMatcher to assign predictions → targets.
  2. Compute focal classification loss over ALL queries (matched get positive
     target, unmatched get all-zero target → background focal penalty).
  3. Compute L1 + GIoU losses over MATCHED pairs only.
  4. Repeat steps 2-3 for each intermediate decoder layer (auxiliary loss).
  5. Return a dict of all named losses plus a weighted loss_total.

Numerical stability notes
──────────────────────────
- All tensors are cast to fp32 at entry.  Mixed-precision training passes fp16,
  which overflows in sigmoid + log for large untrained-model logits.
- sigmoid_focal_loss uses F.binary_cross_entropy_with_logits for the CE term,
  which is numerically stable (uses the log-sum-exp trick internally).  We only
  clamp prob for the modulating factor (1-p_t)^gamma, not for the CE term,
  so gradients remain correct.
- GIoU is computed via _safe_box_for_giou (imported from matcher) to handle
  degenerate matched boxes during early training.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.groundingdino.matcher import (
    HungarianMatcher,
    _safe_box_for_giou,
    generalized_box_iou,
)


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

def sigmoid_focal_loss(
    inputs:  torch.Tensor,   # (*, C) raw logits
    targets: torch.Tensor,   # (*, C) binary float targets in {0, 1}
    alpha:   float = 0.25,
    gamma:   float = 2.0,
) -> torch.Tensor:
    """Sigmoid focal loss, summed over all elements.

    Uses F.binary_cross_entropy_with_logits for the cross-entropy term so that
    log(sigmoid(x)) is computed as -softplus(-x), which is numerically stable
    for arbitrarily large or small logits.

    The modulating factor (1 - p_t)^gamma is computed from a clamped sigmoid
    to avoid NaN in the rare case where prob is exactly 0 or 1 due to extreme
    logits combined with fp32 rounding.

    Returns the raw sum (not yet divided by num_boxes); the caller normalises.
    """
    prob    = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    # p_t: probability of the correct class (positive or negative)
    p_t     = prob * targets + (1.0 - prob) * (1.0 - targets)
    p_t     = p_t.clamp(1e-6, 1.0 - 1e-6)   # guard (1-p_t)^gamma from NaN

    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss    = alpha_t * (1.0 - p_t) ** gamma * ce_loss

    # Defensive guard: should never fire after the clamp above
    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    return loss.sum()


# ---------------------------------------------------------------------------
# SetCriterion
# ---------------------------------------------------------------------------

class SetCriterion(nn.Module):
    """Detection loss combining focal classification, L1, and GIoU box losses.

    Args:
        num_classes:  number of object categories (excluding background).
        matcher:      HungarianMatcher instance (shared with the trainer).
        weight_dict:  maps each loss key (e.g. "loss_ce") to its scalar weight.
                      Auxiliary-layer keys follow the pattern "loss_ce_0", etc.
        focal_alpha:  α for sigmoid_focal_loss (should match matcher).
        focal_gamma:  γ for sigmoid_focal_loss (should match matcher).
    """

    def __init__(
        self,
        num_classes:  int,
        matcher:      HungarianMatcher,
        weight_dict:  Dict[str, float],
        focal_alpha:  float = 0.25,
        focal_gamma:  float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes  = num_classes
        self.matcher      = matcher
        self.weight_dict  = weight_dict
        self.focal_alpha  = focal_alpha
        self.focal_gamma  = focal_gamma

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _src_idx(
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert per-image match lists → flat (batch_idx, query_idx) tensors."""
        batch = torch.cat(
            [torch.full_like(s, i) for i, (s, _) in enumerate(indices)]
        )
        src = torch.cat([s for s, _ in indices])
        return batch, src

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def _loss_class(
        self,
        outputs:   Dict,
        targets:   List[Dict],
        indices:   List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float,
    ) -> torch.Tensor:
        """Sigmoid focal classification loss over all queries.

        Matched queries get a one-hot target at their assigned GT class.
        All other queries get an all-zero target, contributing a small
        background focal penalty that encourages low confidence everywhere
        except at matched positions.
        """
        pred_logits = outputs["pred_logits"]              # (B, Q, C)
        b_idx, s_idx = self._src_idx(indices)

        # Gather assigned GT class labels for matched (batch, query) pairs
        tgt_labels = torch.cat(
            [t["labels"][j] for t, (_, j) in zip(targets, indices)]
        )                                                  # (total_matched,)

        # Build dense (B, Q, C) binary target: 1 at matched class, 0 elsewhere
        target_onehot = torch.zeros_like(pred_logits)
        target_onehot[b_idx, s_idx, tgt_labels] = 1.0

        loss = sigmoid_focal_loss(
            pred_logits,
            target_onehot,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        )
        return loss / num_boxes

    def _loss_boxes(
        self,
        outputs:   Dict,
        targets:   List[Dict],
        indices:   List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """L1 and GIoU losses computed only over matched query-GT pairs.

        Returns:
            loss_l1:   scalar L1 loss normalised by num_boxes
            loss_giou: scalar GIoU loss normalised by num_boxes
        """
        b_idx, s_idx = self._src_idx(indices)
        pred_boxes = outputs["pred_boxes"][b_idx, s_idx]  # (M, 4) cx,cy,w,h
        tgt_boxes  = torch.cat(
            [t["boxes"][j] for t, (_, j) in zip(targets, indices)]
        )                                                  # (M, 4)

        loss_l1 = F.l1_loss(pred_boxes, tgt_boxes, reduction="sum") / num_boxes

        # GIoU on matched pairs only: take the diagonal of the (M, M) matrix
        # _safe_box_for_giou guards against degenerate boxes from the model
        # during early training (e.g. negative w/h giving x2 < x1).
        giou = generalized_box_iou(
            _safe_box_for_giou(pred_boxes),
            _safe_box_for_giou(tgt_boxes),
        )                                                  # (M, M)
        loss_giou = (1.0 - giou.diag()).sum() / num_boxes

        return loss_l1, loss_giou

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        outputs: Dict,
        targets: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """Compute the total detection loss for one batch.

        Args:
            outputs: dict produced by GroundingDINO.forward(), containing
                "pred_logits": (B, Q, C)
                "pred_boxes":  (B, Q, 4)  cx,cy,w,h in [0, 1]
                "aux_outputs": optional list of dicts with same keys,
                               one per intermediate decoder layer
            targets: list of B dicts, each with
                "labels": (N,)   int64
                "boxes":  (N, 4) float32 cx,cy,w,h in [0, 1]

        Returns:
            Dict with keys: loss_ce, loss_bbox, loss_giou,
                            loss_ce_i / loss_bbox_i / loss_giou_i  (aux layers),
                            loss_total (weighted sum, used for .backward()).
        """
        # Cast main outputs to fp32.  Auxiliary outputs are cast inside the loop.
        # We deliberately avoid modifying the original dict so the caller's
        # tensors are not affected.
        main = {
            k: v.float() if isinstance(v, torch.Tensor) else v
            for k, v in outputs.items()
            if k != "aux_outputs"
        }

        # num_boxes: normaliser shared across final and auxiliary losses so that
        # the total loss magnitude is independent of batch size.
        num_boxes = float(max(1, sum(len(t["labels"]) for t in targets)))

        # ── Final decoder layer losses ─────────────────────────────────────
        indices  = self.matcher(main, targets)
        loss_ce             = self._loss_class(main, targets, indices, num_boxes)
        loss_l1, loss_giou  = self._loss_boxes(main, targets, indices, num_boxes)

        losses: Dict[str, torch.Tensor] = {
            "loss_ce":   loss_ce,
            "loss_bbox": loss_l1,
            "loss_giou": loss_giou,
        }

        # ── Auxiliary decoder layer losses ────────────────────────────────
        # Each intermediate layer gets its own matching and its own loss terms.
        # They share the same num_boxes normaliser so their scale is comparable.
        for i, aux in enumerate(outputs.get("aux_outputs", [])):
            aux_fp32 = {
                k: v.float() if isinstance(v, torch.Tensor) else v
                for k, v in aux.items()
            }
            aux_indices            = self.matcher(aux_fp32, targets)
            aux_ce                 = self._loss_class(aux_fp32, targets, aux_indices, num_boxes)
            aux_l1, aux_giou       = self._loss_boxes(aux_fp32, targets, aux_indices, num_boxes)
            losses[f"loss_ce_{i}"]   = aux_ce
            losses[f"loss_bbox_{i}"] = aux_l1
            losses[f"loss_giou_{i}"] = aux_giou

        # ── Weighted total ────────────────────────────────────────────────
        # Only keys present in weight_dict are included; unknown keys are
        # logged individually but excluded from the gradient signal.
        losses["loss_total"] = sum(
            self.weight_dict.get(k, 0.0) * v
            for k, v in losses.items()
            if k != "loss_total"
        )

        return losses