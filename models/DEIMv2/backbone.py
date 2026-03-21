"""CNN backbone for DEIMv2 object detection.

Components:
  - TimmBackbone: wraps a timm model and extracts multi-scale feature maps
  - build_backbone: factory function driven by OmegaConf config

Notes:
  - Outputs are returned in ascending stride order matching out_indices.
  - Channel counts and strides are introspected from timm at init time via
    feature_info — no manual channel list needed in the config.
  - Mask convention follows transformer.py:
      True  = padded / invalid position
      False = valid position
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

import timm


class TimmBackbone(nn.Module):
    """Multi-scale CNN backbone backed by a timm feature extractor.

    Args:
        name:        timm model name, e.g. "resnet50" or "convnext_base".
        out_indices: FPN stages to return, e.g. [1, 2, 3].
        pretrained:  load ImageNet weights when True.
        freeze_at:   freeze all stages with index <= freeze_at.
                     -1 = nothing frozen, 0 = stem only.
    """

    def __init__(
        self,
        name: str,
        out_indices: List[int],
        pretrained: bool = True,
        freeze_at: int = -1,
    ) -> None:
        super().__init__()

        self.out_indices = out_indices
        self.model = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )

        # feature_info covers ALL stages; slice to only the requested indices.
        all_info = self.model.feature_info.info  # list of dicts for every stage
        self._out_channels: List[int] = [
            all_info[i]["num_chs"] for i in out_indices
        ]
        self._out_strides: List[int] = [
            all_info[i]["reduction"] for i in out_indices
        ]

        if freeze_at >= 0:
            self._freeze_stages(freeze_at)

    # ------------------------------------------------------------------
    # Public properties (read by the neck / input projection)
    # ------------------------------------------------------------------

    @property
    def out_channels(self) -> List[int]:
        """Channel count for each returned feature map, in stage order."""
        return self._out_channels

    @property
    def out_strides(self) -> List[int]:
        """Spatial stride for each returned feature map, in stage order."""
        return self._out_strides

    @property
    def num_feature_levels(self) -> int:
        return len(self._out_channels)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> List[torch.Tensor]:
        """Extract multi-scale feature maps.

        Args:
            x:    (B, 3, H, W) input image tensor, normalised to ImageNet stats.
            mask: optional (B, H, W) bool tensor for the input image,
                  True = padded / invalid position.  Kept as a parameter so
                  callers can pass the collate mask without extra unpacking,
                  but mask propagation to each scale is handled by the caller
                  (model.py) after the backbone returns.

        Returns:
            features: list of (B, C_i, H_i, W_i), one per stage in
                      out_indices order (ascending stride).
        """
        assert x.dim() == 4 and x.size(1) == 3, (
            f"Expected input (B, 3, H, W), got {tuple(x.shape)}"
        )
        return self.model(x)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _freeze_stages(self, freeze_at: int) -> None:
        """Freeze stem and all stages with index <= freeze_at.

        timm feature extractors expose children in the order:
          stem (index 0), stage0 (1), stage1 (2), ...
        so we freeze the first freeze_at + 2 children.
        """
        children = list(self.model.children())
        n_freeze = min(freeze_at + 2, len(children))  # +1 stem, +1 inclusive
        for child in children[:n_freeze]:
            for param in child.parameters():
                param.requires_grad_(False)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_backbone(cfg: DictConfig) -> TimmBackbone:
    """Construct a TimmBackbone from an OmegaConf config node.

    Accepts either the full model config (with a "backbone" sub-key) or the
    backbone sub-node directly — both forms work:

        build_backbone(cfg)           # cfg.backbone.name exists
        build_backbone(cfg.backbone)  # pass sub-node directly

    Expected config keys:
        backbone:
          name:        "resnet50"    # required
          out_indices: [1, 2, 3]     # default [1, 2, 3]
          pretrained:  true          # default true
          freeze_at:   -1            # default -1 (nothing frozen)
    """
    bcfg = cfg.get("backbone", cfg)

    name: str = bcfg.name
    out_indices: List[int] = list(bcfg.get("out_indices", [1, 2, 3]))
    pretrained: bool = bool(bcfg.get("pretrained", True))
    freeze_at: int = int(bcfg.get("freeze_at", -1))

    assert isinstance(name, str) and name, (
        "backbone.name must be a non-empty string"
    )
    assert len(out_indices) >= 1, (
        "backbone.out_indices must contain at least one stage index"
    )

    return TimmBackbone(
        name=name,
        out_indices=out_indices,
        pretrained=pretrained,
        freeze_at=freeze_at,
    )
