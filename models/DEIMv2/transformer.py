"""
models/groundingdino/transformer.py

Key contracts
─────────────
TransformerEncoder.forward(src, key_padding_mask, pos)
    src  : (S, B, C)   batch_first=False
    returns (S, B, C)

TransformerDecoder.forward(tgt, memory, memory_key_padding_mask, pos, query_pos)
    tgt / memory : (S, B, C)   batch_first=False
    returns      : (num_layers, S, B, C)  when return_intermediate=True
                   (S, B, C)              otherwise

Both modules use standard nn.TransformerEncoderLayer /
nn.TransformerDecoderLayer with batch_first=False.
"""

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────────────────────────────────────

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )
        self.linear1    = nn.Linear(d_model, dim_feedforward)
        self.linear2    = nn.Linear(dim_feedforward, d_model)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def with_pos(self, x, pos):
        return x if pos is None else x + pos

    def forward(self, src,
                key_padding_mask=None, pos=None):
        q = k = self.with_pos(src, pos)
        src2, _ = self.self_attn(q, k, src,
                                  key_padding_mask=key_padding_mask)
        src = self.norm1(src + self.dropout(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src  = self.norm2(src + self.dropout(src2))
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, key_padding_mask=None, pos=None):
        """src: (S, B, C)  →  (S, B, C)"""
        out = src
        for layer in self.layers:
            out = layer(out, key_padding_mask=key_padding_mask, pos=pos)
        return self.norm(out)


# ──────────────────────────────────────────────────────────────────────────────
# Decoder
# ──────────────────────────────────────────────────────────────────────────────

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )
        self.linear1    = nn.Linear(d_model, dim_feedforward)
        self.linear2    = nn.Linear(dim_feedforward, d_model)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def with_pos(self, x, pos):
        return x if pos is None else x + pos

    def forward(self, tgt, memory,
                memory_key_padding_mask=None,
                pos=None, query_pos=None):
        # self-attention on queries
        q = k = self.with_pos(tgt, query_pos)
        tgt2, _ = self.self_attn(q, k, tgt)
        tgt  = self.norm1(tgt + self.dropout(tgt2))
        # cross-attention: queries attend to encoder memory
        tgt2, _ = self.cross_attn(
            query = self.with_pos(tgt, query_pos),
            key   = self.with_pos(memory, pos),
            value = memory,
            key_padding_mask = memory_key_padding_mask,
        )
        tgt  = self.norm2(tgt + self.dropout(tgt2))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt  = self.norm3(tgt + self.dropout(tgt2))
        return tgt


class TransformerDecoder(nn.Module):
    """
    Decoder with optional intermediate-state return.

    forward() returns
    -----------------
    return_intermediate=True  → stacked tensor (num_layers, S, B, C)
    return_intermediate=False → tensor (S, B, C)
    """

    def __init__(self, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 return_intermediate=True):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm                = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

        # NOTE: class_heads / box_heads are defined on the parent model,
        # NOT here, so the decoder stays architecture-agnostic.

    def forward(self, tgt, memory,
                memory_key_padding_mask=None,
                pos=None, query_pos=None):
        """
        tgt    : (Q, B, C)
        memory : (S, B, C)
        returns: (num_layers, Q, B, C)  or  (Q, B, C)
        """
        out = tgt
        intermediates = []
        for layer in self.layers:
            out = layer(out, memory,
                        memory_key_padding_mask=memory_key_padding_mask,
                        pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediates.append(self.norm(out))

        if self.return_intermediate:
            return torch.stack(intermediates)   # (num_layers, Q, B, C)
        return self.norm(out)                   # (Q, B, C)