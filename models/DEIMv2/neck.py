"""Feature Pyramid Network neck for DEIMv2 multi-scale object detection.

Components:
  - FPN: lateral connections + top-down pathway + extra stride-2 levels
  - build_neck: factory function driven by OmegaConf config
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class FPN(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        num_levels: int = 4,
    ) -> None:
        super().__init__()
        assert num_levels >= len(in_channels), (
            "num_levels must be >= number of input feature maps"
        )

        self.num_levels = num_levels
        self.num_in = len(in_channels)

        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels]
        )
        self.output_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                for _ in in_channels
            ]
        )

        num_extra = num_levels - self.num_in
        self.extra = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
                for _ in range(num_extra)
            ]
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(features) == self.num_in, (
            f"Expected {self.num_in} feature maps, got {len(features)}"
        )

        laterals = [conv(f) for conv, f in zip(self.lateral, features)]

        for i in range(self.num_in - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[-2:],
                mode="nearest",
            )

        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]

        current = outputs[-1]
        for extra_conv in self.extra:
            current = extra_conv(F.relu(current))
            outputs.append(current)

        return outputs


def build_neck(cfg: DictConfig, in_channels: List[int]) -> FPN:
    """Construct an FPN neck from config.

    Args:
        cfg:         full model config or neck sub-node.
        in_channels: channel counts from the backbone, e.g. [512, 1024, 2048].
                     Typically sourced from backbone.out_channels.
    """
    ncfg = cfg.get("neck", cfg)
    return FPN(
        in_channels=in_channels,
        out_channels=int(ncfg.get("out_channels", 256)),
        num_levels=int(ncfg.get("num_levels", 4)),
    )
