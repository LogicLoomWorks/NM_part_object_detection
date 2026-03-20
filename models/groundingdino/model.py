"""GroundingDINO-inspired object detector with configurable CNN backbone.

Architecture:
  1. Backbone  – ResNet / ConvNeXt via timm; selected by cfg.model.backbone.name
  2. Neck      – FPN; unifies channel dimension, adds P6
  3. Encoder   – multi-head self-attention over flattened multi-scale features
  4. Decoder   – query-based cross-attention with iterative box refinement
  5. Heads     – per-layer class linear + box MLP

Checkpoint compatibility:
  save_checkpoint stores the backbone name and num_classes.
  load_checkpoint verifies these match the current config before loading weights.
  Switching backbones requires a fresh checkpoint; all other hyperparameters
  (neck, transformer depth, num_queries) must also match.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from models.groundingdino.backbone import TimmBackbone, build_backbone
from models.groundingdino.neck import FPN, build_neck
from models.groundingdino.transformer import (
    MLP,
    PositionEmbeddingSine,
    TransformerDecoder,
    TransformerEncoder,
)


class GroundingDINO(nn.Module):
    """DINO-style multi-scale object detector."""

    def __init__(
        self,
        backbone: TimmBackbone,
        neck: FPN,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder,
        num_classes: int,
        num_queries: int,
        d_model: int,
        aux_loss: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.encoder = encoder
        self.decoder = decoder
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model
        self.aux_loss = aux_loss

        self.pos_embed = PositionEmbeddingSine(d_model // 2)

        # Learnable query content and positional embeddings
        self.tgt_embed = nn.Embedding(num_queries, d_model)
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Learnable initial reference points (anchor boxes, cx/cy/w/h)
        self.reference_points = nn.Embedding(num_queries, 4)

        # Per-level input projections (FPN outputs are already d_model-wide)
        num_levels = neck.num_levels
        self.input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(d_model, d_model, 1),
                    nn.GroupNorm(32, d_model),
                )
                for _ in range(num_levels)
            ]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.reference_points.weight, 0.05, 0.95)
        for proj in self.input_proj:
            for m in proj.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_masks_and_pos(
        self,
        features: List[torch.Tensor],
        image_mask: Optional[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Downsample the image padding mask to each FPN level and compute pos embeds."""
        masks_list: List[torch.Tensor] = []
        pos_list: List[torch.Tensor] = []
        for feat in features:
            B, _, H, W = feat.shape
            if image_mask is not None:
                m = F.interpolate(
                    image_mask[:, None].float(), size=(H, W), mode="nearest"
                ).squeeze(1).bool()
            else:
                m = torch.ones(B, H, W, dtype=torch.bool, device=feat.device)
            masks_list.append(m)
            pos_list.append(self.pos_embed(feat, m))  # (B, d_model, H, W)
        return masks_list, pos_list

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Args:
            images:     (B, 3, H, W) normalised input images
            image_mask: (B, H, W) bool; True = valid pixel (not padding)
        Returns:
            {
              'pred_logits': (B, Q, num_classes),  # last decoder layer
              'pred_boxes':  (B, Q, 4),            # cx/cy/w/h in [0, 1]
              'aux_outputs': list of dicts          # intermediate layers
            }
        """
        B = images.size(0)

        # 1. Multi-scale backbone features — unpack (features, masks) tuple
        backbone_feats, _ = self.backbone(images)     # [C3, C4, C5]
        neck_feats = self.neck(backbone_feats)        # [P3, P4, P5, P6]

        # 2. Input projection + padding masks + positional embeddings
        proj_feats = [proj(f) for proj, f in zip(self.input_proj, neck_feats)]
        masks_list, pos_list = self._build_masks_and_pos(proj_feats, image_mask)

        # 3. Flatten all scales: (B, sum(Hi*Wi), d_model)
        src = torch.cat(
            [f.flatten(2).permute(0, 2, 1) for f in proj_feats], dim=1
        )
        # key_padding_mask: True means the position should be *ignored*
        key_padding_mask = torch.cat(
            [~m.flatten(1) for m in masks_list], dim=1
        )
        pos_enc = torch.cat(
            [p.flatten(2).permute(0, 2, 1) for p in pos_list], dim=1
        )

        # 4. Encoder
        memory = self.encoder(src, key_padding_mask, pos_enc)

        # 5. Decoder
        tgt = self.tgt_embed.weight.unsqueeze(0).expand(B, -1, -1)
        query_pos = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        ref_pts = self.reference_points.weight.sigmoid().unsqueeze(0).expand(B, -1, -1)

        all_logits, all_boxes = self.decoder(
            tgt, memory, ref_pts, key_padding_mask, pos_enc, query_pos
        )

        result: Dict = {
            "pred_logits": all_logits[-1],
            "pred_boxes": all_boxes[-1],
        }
        if self.aux_loss:
            result["aux_outputs"] = [
                {"pred_logits": cl, "pred_boxes": bx}
                for cl, bx in zip(all_logits[:-1], all_boxes[:-1])
            ]
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg: DictConfig) -> GroundingDINO:
    """Build a GroundingDINO model from an OmegaConf config."""
    mc = cfg.model
    tc = mc.transformer

    backbone = build_backbone(mc.backbone)
    neck = build_neck(mc.neck, backbone.out_channels)

    encoder = TransformerEncoder(
        d_model=tc.d_model,
        nhead=tc.nhead,
        dim_feedforward=tc.dim_feedforward,
        num_layers=tc.num_encoder_layers,
        dropout=tc.dropout,
    )
    decoder = TransformerDecoder(
        d_model=tc.d_model,
        nhead=tc.nhead,
        dim_feedforward=tc.dim_feedforward,
        num_layers=tc.num_decoder_layers,
        dropout=tc.dropout,
        num_classes=mc.num_classes,
    )
    return GroundingDINO(
        backbone=backbone,
        neck=neck,
        encoder=encoder,
        decoder=decoder,
        num_classes=mc.num_classes,
        num_queries=mc.num_queries,
        d_model=tc.d_model,
        aux_loss=mc.get("aux_loss", True),
    )


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: GroundingDINO,
    path: str,
    cfg: DictConfig,
    epoch: int,
    metrics: dict,
) -> None:
    """Save model weights and metadata needed to verify future loads."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "backbone_name": cfg.model.backbone.name,
            "num_classes": cfg.model.num_classes,
            "num_queries": cfg.model.num_queries,
            "epoch": epoch,
            "metrics": metrics,
            "cfg": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )


def load_checkpoint(
    path: str,
    cfg: DictConfig,
) -> Tuple[GroundingDINO, dict]:
    """Load weights from a checkpoint, verifying backbone/class compatibility.

    Raises ValueError if the checkpoint was trained with a different backbone
    or a different number of classes, preventing silent architecture mismatches.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if ckpt["backbone_name"] != cfg.model.backbone.name:
        raise ValueError(
            f"Checkpoint backbone '{ckpt['backbone_name']}' does not match "
            f"config backbone '{cfg.model.backbone.name}'. "
            "Use the same backbone config that was used during training."
        )
    if ckpt["num_classes"] != cfg.model.num_classes:
        raise ValueError(
            f"Checkpoint num_classes={ckpt['num_classes']} != "
            f"config num_classes={cfg.model.num_classes}"
        )

    model = build_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt.get("metrics", {})