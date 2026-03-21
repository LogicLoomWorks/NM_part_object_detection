"""
models/rtdetr/model.py

RTDETRDetector — ultralytics RT-DETR-X adapted to the project's common
detection interface.

Interface (identical to GroundingDINOVisual / DEIMv2Visual / SamDetector)
-------------------------------------------------------------------------
  build_model(cfg: DictConfig) -> RTDETRDetector

  model.forward(images, masks=None) -> {
      "pred_logits": Tensor(B, num_queries, num_classes),  # raw logits
      "pred_boxes":  Tensor(B, num_queries, 4),            # cx/cy/w/h in [0, 1]
      "aux_outputs": []
  }

Architecture
------------
  Ultralytics RT-DETR-X (pretrained on COCO-80) with three heads inside
  RTDETRDecoder replaced to emit num_classes:
    • dec_score_head  — ModuleList[6 × Linear(256, nc)]  (decoder cls head)
    • enc_score_head  — Linear(256, nc)                   (query-selection cls)
    • denoising_class_embed — Embedding(nc+1, 256)        (CDN label embedding)

  num_queries = 300 (fixed by the RT-DETR architecture).

Output tensors from the underlying model in eval mode
  out[1][0]  (num_dec_layers, B, 300, 4)   decoder boxes  per layer
  out[1][1]  (num_dec_layers, B, 300, nc)  decoder logits per layer (RAW)
  out[1][2]  (B, 300, 4)                   encoder boxes
  out[1][3]  (B, 300, nc)                  encoder scores

  We take the LAST decoder layer:
    pred_boxes  = out[1][0][-1]   (B, 300, 4)
    pred_logits = out[1][1][-1]   (B, 300, nc)

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

class RTDETRDetector(nn.Module):
    """RT-DETR-X repurposed as a DETR-compatible detector.

    RT-DETR's decoder already outputs cx/cy/w/h normalised boxes and raw
    class logits, making it a near-zero-adaptation fit for SetCriterion.

    The underlying ultralytics model is kept in eval mode so that the
    RTDETRDecoder returns the decoded (num_layers, B, 300, *) tensors
    needed by SetCriterion.  Gradients still flow through this path.

    Args:
        cfg: OmegaConf DictConfig — the ``model`` sub-tree from the yaml
             config (i.e. cfg.model, not the full cfg).
    """

    NUM_QUERIES: int = 300  # fixed by RT-DETR architecture

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        try:
            from ultralytics import RTDETR
            from ultralytics.nn.modules.head import RTDETRDecoder
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for RTDETRDetector. "
                "Install it with:  pip install ultralytics"
            ) from exc

        self.num_classes = _resolve_num_classes(cfg)
        self.num_queries = self.NUM_QUERIES

        # ── Load pretrained model ──────────────────────────────────────────
        rtdetr = RTDETR("rtdetr-x.pt")
        self._torch_model = rtdetr.model

        # ── Find the RTDETRDecoder head ────────────────────────────────────
        decoder_head: RTDETRDecoder | None = None
        for module in self._torch_model.modules():
            if isinstance(module, RTDETRDecoder):
                decoder_head = module
                break
        if decoder_head is None:
            raise RuntimeError(
                "Could not find RTDETRDecoder inside the RT-DETR model."
            )
        self._decoder_head = decoder_head

        # ── Replace all classification heads to match num_classes ─────────
        nc = self.num_classes
        old_nc = self._decoder_head.nc

        if old_nc != nc:
            self._decoder_head.nc = nc

            # 1. dec_score_head: ModuleList[num_decoder_layers × Linear(256, nc)]
            for i, lin in enumerate(self._decoder_head.dec_score_head):
                self._decoder_head.dec_score_head[i] = nn.Linear(
                    lin.in_features, nc
                )

            # 2. enc_score_head: Linear(256, nc) used for top-k query selection
            old_enc = self._decoder_head.enc_score_head
            self._decoder_head.enc_score_head = nn.Linear(
                old_enc.in_features, nc
            )

            # 3. denoising_class_embed: Embedding(nc+1, d_model) for CDN
            old_dn = self._decoder_head.denoising_class_embed
            self._decoder_head.denoising_class_embed = nn.Embedding(
                nc + 1, old_dn.embedding_dim
            )

        # ── Expose backbone for the trainer's configure_optimizers ────────
        # Layers 0..(N-2) are backbone + AIFI/RepC3 neck; layer N-1 is
        # the RTDETRDecoder.
        self.backbone = _BackboneProxy(list(self._torch_model.model)[:-1])

        # Always keep the underlying model in eval mode.
        self._torch_model.eval()

    # ── Override train() to keep the underlying model in eval mode ────────

    def train(self, mode: bool = True) -> "RTDETRDetector":
        super().train(mode)
        self._torch_model.eval()
        return self

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        images: torch.Tensor,
        masks: Optional[torch.Tensor] = None,   # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        """Run RT-DETR detection.

        Args:
            images: ``(B, 3, H, W)`` float in [0, 1].
            masks:  Ignored — kept for interface compatibility.

        Returns:
            A dict with ``pred_logits``, ``pred_boxes``, ``aux_outputs``.

        Underlying model eval-mode output layout::

            out[0]    (B, 300, 4+nc)               combined (post-processed)
            out[1][0] (num_layers, B, 300, 4)      decoder boxes  per layer
            out[1][1] (num_layers, B, 300, nc)     decoder logits per layer (RAW)
            out[1][2] (B, 300, 4)                  encoder boxes
            out[1][3] (B, 300, nc)                 encoder scores

        We take the last decoder layer (index -1) for pred_boxes / pred_logits.
        """
        out = self._torch_model(images)

        # Last decoder-layer boxes and raw logits
        # out[1][0]: (num_layers, B, 300, 4)  → take [-1] → (B, 300, 4)
        # out[1][1]: (num_layers, B, 300, nc) → take [-1] → (B, 300, nc)
        pred_boxes  = out[1][0][-1].clamp(0.0, 1.0)   # (B, 300, 4)
        pred_logits = out[1][1][-1]                    # (B, 300, nc) — raw logits

        return {
            "pred_logits": pred_logits,
            "pred_boxes":  pred_boxes,
            "aux_outputs": [],
        }


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(cfg: DictConfig) -> RTDETRDetector:
    """Construct an RTDETRDetector from the top-level OmegaConf config."""
    return RTDETRDetector(cfg.model)
