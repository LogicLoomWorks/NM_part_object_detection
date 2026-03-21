"""
models/sam/classification_head.py

Maps SAM mask-decoder token embeddings to per-class logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Two-layer MLP that maps decoder token embeddings → class logits.

    Architecture::

        Linear(embed_dim, hidden_dim) → ReLU → Linear(hidden_dim, num_classes)

    The output is raw logits (no softmax / sigmoid applied here).  Downstream
    losses (focal, cross-entropy) and the matcher operate on raw logits, exactly
    as the other models in this project do.

    Args:
        embed_dim:   Input feature dimension.  For SAM ViT-B this is 256
                     (the ``iou_token`` output of the mask decoder).
        num_classes: Number of detection classes.
        hidden_dim:  Width of the intermediate MLP layer (default 256).

    Shape:
        - Input  ``tokens``: ``(B, num_queries, embed_dim)``
        - Output ``logits``: ``(B, num_queries, num_classes)``
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Map token embeddings to class logits.

        Args:
            tokens: ``(B, num_queries, embed_dim)`` — iou/mask-token outputs
                    from the SAM mask decoder (or the simplified inline decoder
                    used in SamDetector).

        Returns:
            logits: ``(B, num_queries, num_classes)`` — raw (pre-activation)
                    class scores.
        """
        return self.mlp(tokens)
