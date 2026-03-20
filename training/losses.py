"""
Detection loss: Focal CE + L1 + GIoU with Hungarian matching.

Supports auxiliary decoder outputs (aux_loss=True).

Usage
-----
  criterion = build_criterion(cfg)
  loss_dict  = criterion(outputs, targets)
  total_loss = sum(loss_dict.values())
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.groundingdino.matcher import HungarianMatcher, build_matcher


# ──────────────────────────────────────────────────────────────────────────────
# Focal loss helper
# ──────────────────────────────────────────────────────────────────────────────

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


def sigmoid_focal_loss(logits: torch.Tensor,
                       targets: torch.Tensor,
                       alpha: float = 0.25,
                       gamma: float = 2.0,
                       reduction: str = "sum") -> torch.Tensor:
    """
    Binary focal loss applied per element.

    Args:
        logits  : (N, C) raw logits.
        targets : (N, C) binary targets in {0, 1}.
    """
    prob = logits.sigmoid()
    ce   = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t  = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


# ──────────────────────────────────────────────────────────────────────────────
# Main criterion
# ──────────────────────────────────────────────────────────────────────────────

class SetCriterion(nn.Module):
    """
    Args:
        num_classes  (int):   number of detection categories.
        matcher      : HungarianMatcher instance.
        weight_dict  (dict):  loss_ce / loss_bbox / loss_giou weights.
        focal_alpha  (float): focal loss alpha.
        focal_gamma  (float): focal loss gamma.
        aux_loss     (bool):  include intermediate decoder layer losses.
    """

    def __init__(self,
                 num_classes: int,
                 matcher: HungarianMatcher,
                 weight_dict: dict,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 aux_loss: bool = True):
        super().__init__()
        self.num_classes  = num_classes
        self.matcher      = matcher
        self.weight_dict  = weight_dict
        self.focal_alpha  = focal_alpha
        self.focal_gamma  = focal_gamma
        self.aux_loss     = aux_loss

    # ------------------------------------------------------------------
    def _match(self, outputs, targets):
        return self.matcher(
            {
                "pred_logits": outputs["pred_logits"].detach(),
                "pred_boxes":  outputs["pred_boxes"].detach(),
            },
            targets,
        )

    # ------------------------------------------------------------------
    def loss_labels(self, outputs, targets, indices) -> torch.Tensor:
        logits = outputs["pred_logits"]              # (B, Q, C)
        B, Q, C = logits.shape
        device  = logits.device

        tgt_cls = torch.zeros(B, Q, C, device=device)
        for b, (pred_i, tgt_i) in enumerate(indices):
            if pred_i.numel() == 0:
                continue
            tgt_cls[b, pred_i, targets[b]["labels"][tgt_i]] = 1.0

        loss = sigmoid_focal_loss(
            logits.view(-1, C),
            tgt_cls.view(-1, C),
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="sum",
        )
        num_boxes = max(sum(t["labels"].numel() for t in targets), 1)
        return loss / num_boxes

    # ------------------------------------------------------------------
    def loss_boxes(self, outputs, targets, indices):
        pred_boxes = outputs["pred_boxes"]           # (B, Q, 4)
        device     = pred_boxes.device

        pred_list, tgt_list = [], []
        for b, (pred_i, tgt_i) in enumerate(indices):
            if pred_i.numel() == 0:
                continue
            pred_list.append(pred_boxes[b][pred_i])
            tgt_list.append(targets[b]["boxes"][tgt_i].to(device))

        if not pred_list:
            zero = pred_boxes.sum() * 0
            return zero, zero

        src = torch.cat(pred_list)
        tgt = torch.cat(tgt_list)
        num_boxes = max(src.shape[0], 1)

        loss_l1   = F.l1_loss(src, tgt, reduction="sum") / num_boxes

        giou = _generalised_iou(src, tgt)            # (N, N)
        loss_giou = (1 - giou.diag()).sum() / num_boxes

        return loss_l1, loss_giou

    # ------------------------------------------------------------------
    def _single(self, outputs, targets) -> dict:
        indices = self._match(outputs, targets)
        l_cls           = self.loss_labels(outputs, targets, indices)
        l_bbox, l_giou  = self.loss_boxes(outputs, targets, indices)
        return {
            "loss_ce":   self.weight_dict["loss_ce"]   * l_cls,
            "loss_bbox": self.weight_dict["loss_bbox"] * l_bbox,
            "loss_giou": self.weight_dict["loss_giou"] * l_giou,
        }

    # ------------------------------------------------------------------
    def forward(self, outputs: dict, targets: list) -> dict:
        loss_dict = self._single(outputs, targets)

        if self.aux_loss and "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                aux_losses = self._single(aux, targets)
                for k, v in aux_losses.items():
                    loss_dict[f"{k}_aux_{i}"] = v

        return loss_dict


def build_criterion(cfg) -> SetCriterion:
    matcher = build_matcher(cfg.training.matcher)
    return SetCriterion(
        num_classes=cfg.model.num_classes,
        matcher=matcher,
        weight_dict={
            "loss_ce":   cfg.training.loss_weights.loss_ce,
            "loss_bbox": cfg.training.loss_weights.loss_bbox,
            "loss_giou": cfg.training.loss_weights.loss_giou,
        },
        focal_alpha=cfg.training.focal_alpha,
        focal_gamma=cfg.training.focal_gamma,
        aux_loss=cfg.model.aux_loss,
    )