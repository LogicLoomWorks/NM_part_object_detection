"""
models/groundingdino/transformer.py

Exports: MLP, PositionEmbeddingSine,
         TransformerEncoderLayer, TransformerEncoder,
         TransformerDecoderLayer, TransformerDecoder

Bug fixed: cross_attn / self_attn in TransformerDecoderLayer were built with
batch_first=True but tensors flow as (seq_len, batch, embed) throughout.
Fix: batch_first=False on all nn.MultiheadAttention instances.

Bug fixed: PositionEmbeddingSine.forward() did not accept a mask argument.
Fix: add mask=None parameter (mask is unused but callers may pass it).

Bug fixed: TransformerDecoder returned a raw tensor instead of (all_logits,
all_boxes). Fix: class head + box MLP wired inside the decoder; forward()
now returns (all_logits, all_boxes) each shaped (num_layers, B, Q, C).
"""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)
        )
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10_000,
                 normalize: bool = True, scale: float | None = None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale or (2 * math.pi)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # mask is accepted for API compatibility but not used —
        # we always embed the full spatial grid.
        B, _, H, W = x.shape
        not_mask = torch.ones(B, H, W, device=x.device, dtype=torch.float32)
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=4).flatten(3)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=4).flatten(3)
        return torch.cat([pos_y, pos_x], dim=3).permute(0, 3, 1, 2)  # (B, C, H, W)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_mha(embed_dim: int, num_heads: int, dropout: float = 0.0) -> nn.MultiheadAttention:
    return nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        batch_first=False,  # tensors are (seq, B, C) throughout
    )


def _seq_first_guard(t: torch.Tensor, name: str) -> torch.Tensor:
    if t.dim() == 3 and t.shape[0] <= 16 and t.shape[1] > t.shape[0] * 4:
        warnings.warn(
            f"[Transformer] '{name}' looks batch-first {list(t.shape)} — "
            "transposing to seq-first. Fix the call site.",
            stacklevel=3,
        )
        return t.transpose(0, 1).contiguous()
    return t


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = _build_mha(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor,
                src_key_padding_mask=None,
                pos: torch.Tensor | None = None) -> torch.Tensor:
        q = k = src + pos if pos is not None else src
        src2, _ = self.self_attn(q, k, src, key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        return self.norm2(src + self.dropout2(src2))


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor,
                src_key_padding_mask=None,
                pos: torch.Tensor | None = None) -> torch.Tensor:
        # src arrives as (B, S, C) from model.py — transpose to seq-first
        x = src.transpose(0, 1)                          # (S, B, C)
        if pos is not None:
            pos = pos.transpose(0, 1)                    # (S, B, C)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask, pos=pos)
        x = self.norm(x)
        return x.transpose(0, 1)                         # back to (B, S, C)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = _build_mha(d_model, nhead, dropout)
        self.cross_attn = _build_mha(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                memory_key_padding_mask=None,
                pos: torch.Tensor | None = None,
                query_pos: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            tgt:       (Q, B, C) query tokens
            memory:    (S, B, C) encoder output
            pos:       (S, B, C) encoder positional embeddings
            query_pos: (Q, B, C) query positional embeddings
        """
        # self-attention
        q = k = tgt + query_pos if query_pos is not None else tgt
        tgt2, _ = self.self_attn(q, k, tgt)
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        # cross-attention
        q = tgt    + query_pos if query_pos is not None else tgt
        k = memory + pos       if pos       is not None else memory
        cross_out, _ = self.cross_attn(q, k, memory,
                                       key_padding_mask=memory_key_padding_mask)
        tgt = self.norm2(tgt + self.dropout2(cross_out))

        # FFN
        ff = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout3(ff))


class TransformerDecoder(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 num_layers: int = 6, num_classes: int = 80,
                 return_intermediate: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

        # Per-layer prediction heads (one set per decoder layer)
        self.class_heads = nn.ModuleList([
            nn.Linear(d_model, num_classes) for _ in range(num_layers)
        ])
        self.box_heads = nn.ModuleList([
            MLP(d_model, d_model, 4, num_layers=3) for _ in range(num_layers)
        ])

    def forward(self,
                tgt:       torch.Tensor,        # (B, Q, C)
                memory:    torch.Tensor,         # (B, S, C)
                ref_pts:   torch.Tensor,         # (B, Q, 4)  sigmoid space
                memory_key_padding_mask=None,    # (B, S)
                pos:       torch.Tensor | None = None,        # (B, S, C)
                query_pos: torch.Tensor | None = None):       # (B, Q, C)
        """
        Returns:
            all_logits: list of (B, Q, num_classes), one per layer
            all_boxes:  list of (B, Q, 4),           one per layer, values in [0,1]
        """
        # Transpose to seq-first for the MHA layers
        x   = tgt.transpose(0, 1)                           # (Q, B, C)
        mem = memory.transpose(0, 1)                        # (S, B, C)
        if pos       is not None: pos       = pos.transpose(0, 1)        # (S, B, C)
        if query_pos is not None: query_pos = query_pos.transpose(0, 1)  # (Q, B, C)

        all_logits = []
        all_boxes  = []

        for layer, cls_head, box_head in zip(self.layers, self.class_heads, self.box_heads):
            x = layer(x, mem,
                      memory_key_padding_mask=memory_key_padding_mask,
                      pos=pos, query_pos=query_pos)

            out = self.norm(x).transpose(0, 1)              # (B, Q, C)

            logits = cls_head(out)                           # (B, Q, num_classes)

            # Box refinement: predict delta in logit space, add to reference,
            # then sigmoid back to [0, 1]
            box_delta = box_head(out)                        # (B, Q, 4)
            ref_logit = torch.log(ref_pts / (1.0 - ref_pts + 1e-8))  # inverse sigmoid
            boxes = (ref_logit + box_delta).sigmoid()        # (B, Q, 4)  in [0,1]

            all_logits.append(logits)
            all_boxes.append(boxes)

        return all_logits, all_boxes