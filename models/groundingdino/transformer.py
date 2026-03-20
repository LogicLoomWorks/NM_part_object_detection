"""Transformer encoder-decoder for DINO-style object detection.

Components:
  - PositionEmbeddingSine: 2D sinusoidal positional encoding
  - TransformerEncoder: multi-scale self-attention over flattened feature maps
  - TransformerDecoder: query-based cross-attention with iterative box refinement
  - MLP: shared multi-layer perceptron for box prediction heads
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Numerically stable inverse sigmoid (logit)."""
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


class MLP(nn.Module):
    """Feed-forward MLP used as the bounding-box prediction head."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> None:
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(dims[:-1], dims[1:])]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


# ---------------------------------------------------------------------------
# Positional embedding
# ---------------------------------------------------------------------------

class PositionEmbeddingSine(nn.Module):
    """2D sinusoidal positional encoding for image feature maps (DETR-style)."""

    def __init__(
        self,
        num_pos_feats: int = 128,
        temperature: int = 10_000,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, C, H, W) feature map
            mask: (B, H, W) bool; True = valid pixel (not padding)
        Returns:
            pos:  (B, 2*num_pos_feats, H, W)
        """
        B, _, H, W = x.shape
        if mask is None:
            mask = torch.ones(B, H, W, dtype=torch.bool, device=x.device)

        y_embed = mask.float().cumsum(1)
        x_embed = mask.float().cumsum(2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, None] / dim_t  # (B, H, W, num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = k = src + pos if pos is not None else src
        attn, _ = self.self_attn(q, k, src, key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout1(attn))
        ff = self.linear2(self.dropout(F.relu(self.linear1(src))))
        return self.norm2(src + self.dropout2(ff))


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_key_padding_mask, pos)
        return self.norm(output)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention among queries
        q = k = tgt + query_pos if query_pos is not None else tgt
        self_out, _ = self.self_attn(q, k, tgt)
        tgt = self.norm1(tgt + self.dropout1(self_out))

        # Cross-attention: queries attend to encoder memory
        q = tgt + query_pos if query_pos is not None else tgt
        k = memory + pos if pos is not None else memory
        cross_out, _ = self.cross_attn(
            q, k, memory, key_padding_mask=memory_key_padding_mask
        )
        tgt = self.norm2(tgt + self.dropout2(cross_out))

        ff = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout3(ff))


class TransformerDecoder(nn.Module):
    """DINO-style decoder with per-layer iterative box refinement.

    Each decoder layer predicts class logits and box deltas. The deltas are
    applied to the current reference points in logit space (inverse-sigmoid),
    producing refined boxes that become the next layer's reference points.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

        # Per-layer prediction heads
        self.class_heads = nn.ModuleList(
            [nn.Linear(d_model, num_classes) for _ in range(num_layers)]
        )
        self.bbox_heads = nn.ModuleList(
            [MLP(d_model, d_model, 4, 3) for _ in range(num_layers)]
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        reference_points: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            tgt:              (B, Q, d_model) initial query content features
            memory:           (B, N, d_model) encoder output
            reference_points: (B, Q, 4) initial anchor boxes, cx/cy/w/h in [0,1]
            memory_key_padding_mask: (B, N) True = padded position
            pos:              (B, N, d_model) positional embeddings for memory
            query_pos:        (B, Q, d_model) positional embeddings for queries
        Returns:
            all_logits: list of (B, Q, num_classes) per layer
            all_boxes:  list of (B, Q, 4) per layer, cx/cy/w/h in [0,1]
        """
        output = tgt
        ref = reference_points  # (B, Q, 4)

        all_logits: List[torch.Tensor] = []
        all_boxes: List[torch.Tensor] = []

        for layer, cls_head, box_head in zip(
            self.layers, self.class_heads, self.bbox_heads
        ):
            output = layer(output, memory, memory_key_padding_mask, pos, query_pos)
            normed = self.norm(output)

            # Class prediction
            cls_logits = cls_head(normed)  # (B, Q, num_classes)

            # Box refinement: predict delta in logit space, refine reference
            box_delta = box_head(normed)  # (B, Q, 4)
            refined = (inverse_sigmoid(ref) + box_delta).sigmoid()

            all_logits.append(cls_logits)
            all_boxes.append(refined)

            # Detach so gradients don't flow between refinement steps
            ref = refined.detach()

        return all_logits, all_boxes
