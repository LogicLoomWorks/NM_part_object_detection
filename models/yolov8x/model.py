"""
models/yolov8x/model.py

YOLOv8xDetector — ultralytics YOLOv8x adapted to the project's common
detection interface.

Interface (identical to GroundingDINOVisual / DEIMv2Visual / SamDetector)
-------------------------------------------------------------------------
  build_model(cfg: DictConfig) -> YOLOv8xDetector

  model.forward(images, masks=None) -> {
      "pred_logits": Tensor(B, Q, num_classes),  # raw logits
      "pred_boxes":  Tensor(B, Q, 4),            # cx/cy/w/h in [0, 1]
      "aux_outputs": []
  }

Architecture
------------
  Ultralytics YOLOv8x (pretrained on COCO-80) with the classification
  sub-layer of its Detect head replaced to emit num_classes channels.

  self.backbone  — layers 0..N-2 of the underlying torch model
                   (Conv+C2f backbone + PANet neck)
  self._head     — Detect layer with patched nc

Number of classes
-----------------
  Read from run_config.TRAINING["num_classes"] when present, otherwise
  from cfg.model.num_classes (the model's own yaml config).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

import run_config as _rc


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_num_classes(cfg_model: DictConfig) -> int:
    """Return num_classes from run_config.TRAINING (if present) else yaml cfg."""
    if "num_classes" in _rc.TRAINING:
        return int(_rc.TRAINING["num_classes"])
    return int(cfg_model.num_classes)


class _BackboneProxy(nn.Module):
    """Thin nn.Module wrapper around the backbone+neck layers of the
    ultralytics model.  Exposes .parameters() so the trainer's
    configure_optimizers can set a separate backbone learning rate.
    """

    def __init__(self, layer_list: list[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layer_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Not called during normal training; kept for completeness.
        for layer in self.layers:
            x = layer(x)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Main detector
# ──────────────────────────────────────────────────────────────────────────────

class YOLOv8xDetector(nn.Module):
    """YOLOv8x repurposed as a DETR-compatible detector.

    The underlying ultralytics model is kept in eval mode throughout so
    that the Detect head always returns the decoded ``(B, 4+nc, Q)``
    tensor rather than per-stride raw feature maps.  Gradients still
    flow through this path, enabling end-to-end fine-tuning.

    Args:
        cfg: OmegaConf DictConfig — the ``model`` sub-tree from the yaml
             config (i.e. cfg.model, not the full cfg).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        try:
            from ultralytics import YOLO
            from ultralytics.nn.modules.head import Detect
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for YOLOv8xDetector. "
                "Install it with:  pip install ultralytics"
            ) from exc

        self.num_classes = _resolve_num_classes(cfg)

        # ── Load pretrained model ──────────────────────────────────────────
        yolo = YOLO("yolov8x.pt")
        self._torch_model = yolo.model

        # ── Replace classification head for our num_classes ───────────────
        head = self._torch_model.model[-1]
        assert isinstance(head, Detect), (
            f"Expected final layer to be Detect, got {type(head).__name__}"
        )
        if head.nc != self.num_classes:
            head.nc = self.num_classes
            head.no = head.reg_max * 4 + self.num_classes
            for i in range(len(head.cv3)):
                old_conv = head.cv3[i][-1]          # Conv2d(c_in, old_nc, 1)
                head.cv3[i][-1] = nn.Conv2d(
                    old_conv.in_channels, self.num_classes, kernel_size=1
                )
        self._head = head

        # ── Expose backbone for the trainer's configure_optimizers ────────
        # Layers 0..(N-2) are backbone + PANet neck; layer N-1 is Detect.
        self.backbone = _BackboneProxy(list(self._torch_model.model)[:-1])

        # Always keep the underlying model in eval mode so output format
        # is consistent regardless of Lightning's train/eval state.
        self._torch_model.eval()

    # ── Override train() to keep the underlying model in eval mode ────────

    def train(self, mode: bool = True) -> "YOLOv8xDetector":
        super().train(mode)
        self._torch_model.eval()
        return self

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        images: torch.Tensor,
        masks: Optional[torch.Tensor] = None,   # noqa: ARG002 — not used by YOLO
    ) -> dict[str, torch.Tensor]:
        """Run YOLOv8x detection.

        Args:
            images: ``(B, 3, H, W)`` float in [0, 1].
            masks:  Ignored — YOLO handles padding internally.
                    Kept for interface compatibility.

        Returns:
            A dict with ``pred_logits``, ``pred_boxes``, ``aux_outputs``.
        """
        B, _, H, W = images.shape

        # YOLOv8 eval-mode output: (pred, feature_maps)
        # pred: (B, 4 + nc, total_anchors)
        #   [:, :4, :]  cx/cy/w/h in PIXEL SPACE (not normalised)
        #   [:, 4:, :]  sigmoid class scores ∈ [0, 1]
        out = self._torch_model(images)
        pred = out[0]                               # (B, 4+nc, Q)

        # ── Boxes: pixel cx/cy/w/h → normalised cx/cy/w/h ─────────────────
        raw_boxes = pred[:, :4, :]                  # (B, 4, Q)
        scale = torch.tensor(
            [W, H, W, H], dtype=pred.dtype, device=pred.device
        ).view(1, 4, 1)
        norm_boxes = (raw_boxes / scale).permute(0, 2, 1).clamp(0.0, 1.0)
        # norm_boxes: (B, Q, 4)

        # ── Logits: undo sigmoid → raw logits for SetCriterion ────────────
        sig_scores = pred[:, 4:, :].permute(0, 2, 1)   # (B, Q, nc) in [0,1]
        logits = sig_scores.clamp(1e-6, 1.0 - 1e-6).logit()

        return {
            "pred_logits": logits,
            "pred_boxes":  norm_boxes,
            "aux_outputs": [],
        }


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(cfg: DictConfig) -> YOLOv8xDetector:
    """Construct a YOLOv8xDetector from the top-level OmegaConf config."""
    return YOLOv8xDetector(cfg.model)
