"""
models/deimv2/transformer.py

Pure-PyTorch deformable transformer for DEIMv2.

Key contracts
─────────────
DeformableTransformerEncoder.forward(
    src, spatial_shapes, level_start_idx, key_padding_mask, pos)
    src  : (N_total, B, C)   batch_first=False
    returns (N_total, B, C)

DINODecoder.forward(
    tgt, memory, reference_points, spatial_shapes, level_start_idx,
    memory_key_padding_mask, query_pos, attn_mask)
    tgt / query_pos : (Q, B, C)   batch_first=False
    reference_points: (Q, B, 4)   sigmoid [0,1]
    returns (num_layers, Q, B, C), list[ref_pts_per_layer]

All shapes use batch_first=False (seq, batch, channels).
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Multi-scale deformable attention (pure PyTorch / F.grid_sample)
# ──────────────────────────────────────────────────────────────────────────────

class DeformableAttention(nn.Module):
    """Multi-scale deformable self/cross attention.

    For each query the module predicts num_heads * num_points sampling offsets
    relative to a reference point, samples from all feature levels with
    F.grid_sample, then produces a weighted sum.

    Args:
        d_model:     embedding dimension.
        num_heads:   attention heads.
        num_levels:  number of feature map scales.
        num_points:  sampling points per head per level.
        dropout:     dropout probability.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model    = d_model
        self.num_heads  = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim   = d_model // num_heads

        # Sampling offset predictor
        self.sampling_offsets = nn.Linear(
            d_model, num_heads * num_levels * num_points * 2
        )
        # Attention weight predictor
        self.attention_weights = nn.Linear(
            d_model, num_heads * num_levels * num_points
        )
        self.value_proj  = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout     = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid = torch.stack([thetas.cos(), thetas.sin()], dim=-1)  # (H, 2)
        bias = grid.view(self.num_heads, 1, 1, 2).expand(
            self.num_heads, self.num_levels, self.num_points, 2
        ).reshape(-1)
        self.sampling_offsets.bias = nn.Parameter(bias)

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,                        # (Q, B, d_model)
        reference_points: torch.Tensor,             # (Q, B, L, 2) or (Q, B, 4)
        value: torch.Tensor,                        # (N_total, B, d_model)
        spatial_shapes: torch.Tensor,               # (num_levels, 2) — (H_l, W_l)
        level_start_idx: torch.Tensor,              # (num_levels,)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, N_total) True=pad
    ) -> torch.Tensor:
        """Returns: (Q, B, d_model)"""
        Q, B, _ = query.shape
        N_total  = value.shape[0]

        # Project value: (N_total, B, d_model) → (B, N_total, num_heads, head_dim)
        value = self.value_proj(value).permute(1, 0, 2)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        value = value.view(B, N_total, self.num_heads, self.head_dim)

        query_bq = query.permute(1, 0, 2)  # (B, Q, d_model)

        # Sampling offsets: (B, Q, num_heads, num_levels, num_points, 2)
        offsets = self.sampling_offsets(query_bq).view(
            B, Q, self.num_heads, self.num_levels, self.num_points, 2
        )

        # Attention weights: (B, Q, num_heads, num_levels, num_points)
        attn_w = self.attention_weights(query_bq).view(
            B, Q, self.num_heads, self.num_levels * self.num_points
        )
        attn_w = attn_w.softmax(dim=-1).view(
            B, Q, self.num_heads, self.num_levels, self.num_points
        )

        # Normalise reference_points to (B, Q, num_levels, 2) in [0,1]
        if reference_points.shape[-1] == 4:
            # cx/cy/w/h — use centre (cx, cy)
            ref_cxcy = reference_points[..., :2]
            if ref_cxcy.dim() == 3:
                # (Q, B, 2) → (B, Q, num_levels, 2)
                ref_pt = ref_cxcy.permute(1, 0, 2).unsqueeze(2).expand(
                    -1, -1, self.num_levels, -1
                )
            else:
                # (Q, B, L, 2)
                ref_pt = ref_cxcy.permute(1, 0, 2, 3)
                if ref_pt.shape[2] != self.num_levels:
                    ref_pt = ref_pt[:, :, :1, :].expand(-1, -1, self.num_levels, -1)
        elif reference_points.shape[-1] == 2:
            if reference_points.dim() == 3:
                # (Q, B, 2)
                ref_pt = reference_points.permute(1, 0, 2).unsqueeze(2).expand(
                    -1, -1, self.num_levels, -1
                )
            else:
                # (Q, B, num_levels, 2)
                ref_pt = reference_points.permute(1, 0, 2, 3)
        else:
            raise ValueError(
                f"Unexpected reference_points shape: {reference_points.shape}"
            )

        # Scale offsets relative to feature-map resolution
        # spatial_shapes is (H, W); we need (W, H) for (x, y) offsets
        level_shapes = spatial_shapes.float()
        offset_normaliser = torch.stack(
            [level_shapes[:, 1], level_shapes[:, 0]], dim=-1
        ).view(1, 1, 1, self.num_levels, 1, 2)   # (1,1,1,L,1,2)

        ref_expanded  = ref_pt.unsqueeze(2).unsqueeze(4)  # (B, Q, 1, L, 1, 2)
        sampling_locs = ref_expanded + offsets / (offset_normaliser + 1e-6)
        sampling_locs = sampling_locs.clamp(0.0, 1.0)

        # Sample from each level using F.grid_sample
        outputs = []
        for lvl, (H_l, W_l) in enumerate(spatial_shapes.tolist()):
            H_l, W_l = int(H_l), int(W_l)
            start     = int(level_start_idx[lvl].item())
            end       = start + H_l * W_l

            # (B, H_l*W_l, num_heads, head_dim) → (B*num_heads, head_dim, H_l, W_l)
            v_lvl = value[:, start:end, :, :]
            v_lvl = v_lvl.view(B, H_l, W_l, self.num_heads, self.head_dim)
            v_lvl = v_lvl.permute(0, 3, 4, 1, 2)                  # (B, H, hd, Hl, Wl)
            v_lvl = v_lvl.reshape(B * self.num_heads, self.head_dim, H_l, W_l)

            # (B, Q, num_heads, num_points, 2) → remap to [-1,1]
            s_locs = sampling_locs[:, :, :, lvl, :, :]  # (B, Q, H, P, 2)
            s_locs = s_locs * 2.0 - 1.0
            # → (B*num_heads, Q*num_points, 1, 2) for grid_sample
            s_locs = s_locs.permute(0, 2, 1, 3, 4)                # (B, H, Q, P, 2)
            s_locs = s_locs.reshape(B * self.num_heads, Q * self.num_points, 1, 2)

            # (B*H, hd, Q*P, 1) → (B, Q, H, P, hd)
            sampled = F.grid_sample(
                v_lvl, s_locs,
                mode="bilinear", padding_mode="zeros", align_corners=False,
            )
            sampled = sampled.view(
                B, self.num_heads, self.head_dim, Q, self.num_points
            )
            sampled = sampled.permute(0, 3, 1, 4, 2)              # (B, Q, H, P, hd)

            w_lvl   = attn_w[:, :, :, lvl, :]                     # (B, Q, H, P)
            out_lvl = (sampled * w_lvl.unsqueeze(-1)).sum(dim=3)   # (B, Q, H, hd)
            outputs.append(out_lvl)

        # Sum over levels → (B, Q, d_model)
        out = torch.stack(outputs, dim=0).sum(dim=0).reshape(B, Q, self.d_model)
        out = self.dropout(self.output_proj(out))
        return out.permute(1, 0, 2)  # (Q, B, d_model)


# ──────────────────────────────────────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────────────────────────────────────

class DeformableTransformerEncoderLayer(nn.Module):
    """Single encoder layer: deformable self-attention + FFN."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_attn = DeformableAttention(
            d_model=d_model,
            num_heads=nhead,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout,
        )
        self.linear1    = nn.Linear(d_model, dim_feedforward)
        self.linear2    = nn.Linear(dim_feedforward, d_model)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout1   = nn.Dropout(dropout)
        self.dropout2   = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        src: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_idx: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q    = src if pos is None else src + pos
        src2 = self.self_attn(
            query=q,
            reference_points=reference_points,
            value=src,
            spatial_shapes=spatial_shapes,
            level_start_idx=level_start_idx,
            key_padding_mask=key_padding_mask,
        )
        src  = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src  = self.norm2(src + self.dropout2(src2))
        return src


class DeformableTransformerEncoder(nn.Module):
    """Stack of deformable encoder layers.

    Generates normalised grid reference points from spatial_shapes and
    passes them to each layer.  Returns (N_total, B, d_model).
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        num_levels: int = 4,
        num_points: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DeformableTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                num_levels=num_levels,
                num_points=num_points,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _get_reference_points(
        spatial_shapes: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Build normalised grid reference points for encoder self-attention.

        Returns: (N_total, 1, num_levels, 2) — (cx, cy) in [0, 1]
        """
        ref_pts_list = []
        for H_l, W_l in spatial_shapes.tolist():
            H_l, W_l = int(H_l), int(W_l)
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_l - 0.5, H_l, device=device) / H_l,
                torch.linspace(0.5, W_l - 0.5, W_l, device=device) / W_l,
                indexing="ij",
            )
            ref_pts = torch.stack(
                [ref_x.reshape(-1), ref_y.reshape(-1)], dim=-1
            )  # (H_l*W_l, 2)
            ref_pts_list.append(ref_pts)

        ref_pts_all = torch.cat(ref_pts_list, dim=0)   # (N_total, 2)
        num_levels  = len(spatial_shapes)
        return ref_pts_all[:, None, None, :].expand(-1, 1, num_levels, -1)

    def forward(
        self,
        src: torch.Tensor,                      # (N_total, B, d_model)
        spatial_shapes: torch.Tensor,            # (num_levels, 2)
        level_start_idx: torch.Tensor,           # (num_levels,)
        key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B      = src.shape[1]
        device = src.device
        # (N_total, 1, num_levels, 2) → (N_total, B, num_levels, 2)
        reference_points = self._get_reference_points(
            spatial_shapes, device
        ).expand(-1, B, -1, -1)

        out = src
        for layer in self.layers:
            out = layer(
                src=out,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_idx=level_start_idx,
                key_padding_mask=key_padding_mask,
                pos=pos,
            )
        return self.norm(out)


# ──────────────────────────────────────────────────────────────────────────────
# Decoder
# ──────────────────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(num_layers):
        i_dim = in_dim  if i == 0             else hidden_dim
        o_dim = out_dim if i == num_layers - 1 else hidden_dim
        layers.append(nn.Linear(i_dim, o_dim))
        if i < num_layers - 1:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class DINODecoderLayer(nn.Module):
    """Single DINO-style decoder layer.

    1. Self-attention among queries (standard MHA, supports CDN attn_mask).
    2. Deformable cross-attention to encoder memory.
    3. FFN.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )
        self.cross_attn = DeformableAttention(
            d_model=d_model,
            num_heads=nhead,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout,
        )
        self.linear1    = nn.Linear(d_model, dim_feedforward)
        self.linear2    = nn.Linear(dim_feedforward, d_model)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout1   = nn.Dropout(dropout)
        self.dropout2   = nn.Dropout(dropout)
        self.dropout3   = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        tgt: torch.Tensor,                         # (Q, B, d_model)
        memory: torch.Tensor,                      # (N_total, B, d_model)
        reference_points: torch.Tensor,            # (Q, B, 4)
        spatial_shapes: torch.Tensor,
        level_start_idx: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,  # (Q, B, d_model)
        attn_mask: Optional[torch.Tensor] = None,  # (Q, Q) bool: True=block
    ) -> torch.Tensor:

        # 1. Self-attention
        q = k = tgt if query_pos is None else tgt + query_pos
        tgt2, _ = self.self_attn(q, k, tgt, attn_mask=attn_mask)
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        # 2. Cross-deformable attention
        q_with_pos = tgt if query_pos is None else tgt + query_pos
        tgt2 = self.cross_attn(
            query=q_with_pos,
            reference_points=reference_points,
            value=memory,
            spatial_shapes=spatial_shapes,
            level_start_idx=level_start_idx,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # 3. FFN
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt  = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


class DINODecoder(nn.Module):
    """DINO-style decoder with iterative box refinement.

    Each layer refines reference_points by predicting a delta in logit space,
    then converting back to [0, 1] via sigmoid.

    Box heads are injected from DEIMv2Visual via set_box_heads() so the
    parent model owns all trainable parameters.

    forward() returns:
        hs            : (num_layers, Q, B, d_model)
        ref_pts_list  : list[Tensor(B, Q, 4)] — refined ref pts per layer
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        num_levels: int = 4,
        num_points: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        return_intermediate: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DINODecoderLayer(
                d_model=d_model,
                nhead=nhead,
                num_levels=num_levels,
                num_points=num_points,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.norm                = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate
        self.num_layers          = num_layers
        self.box_heads: Optional[nn.ModuleList] = None

    def set_box_heads(self, box_heads: nn.ModuleList) -> None:
        """Register per-layer box refinement heads (owned by DEIMv2Visual)."""
        self.box_heads = box_heads

    def forward(
        self,
        tgt: torch.Tensor,                         # (Q, B, d_model)
        memory: torch.Tensor,                      # (N_total, B, d_model)
        reference_points: torch.Tensor,            # (Q, B, 4) — [0,1]
        spatial_shapes: torch.Tensor,
        level_start_idx: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        out                       = tgt
        ref                       = reference_points
        intermediates:     list[torch.Tensor] = []
        ref_pts_per_layer: list[torch.Tensor] = []

        for lvl, layer in enumerate(self.layers):
            out = layer(
                tgt=out,
                memory=memory,
                reference_points=ref,
                spatial_shapes=spatial_shapes,
                level_start_idx=level_start_idx,
                memory_key_padding_mask=memory_key_padding_mask,
                query_pos=query_pos,
                attn_mask=attn_mask,
            )
            out_normed = self.norm(out)

            if self.return_intermediate:
                intermediates.append(out_normed)

            # Iterative box refinement in logit space
            if self.box_heads is not None:
                ref_bq    = ref.permute(1, 0, 2)         # (B, Q, 4)
                ref_logit = torch.log(
                    ref_bq.clamp(1e-6, 1 - 1e-6) /
                    (1.0 - ref_bq.clamp(1e-6, 1 - 1e-6))
                )
                out_bq  = out_normed.permute(1, 0, 2)    # (B, Q, d_model)
                delta   = self.box_heads[lvl](out_bq)    # (B, Q, 4)
                ref_new = (ref_logit + delta).sigmoid().clamp(0.0, 1.0)
                ref_pts_per_layer.append(ref_new)
                ref = ref_new.permute(1, 0, 2)           # (Q, B, 4)
            else:
                ref_pts_per_layer.append(ref.permute(1, 0, 2))

        if self.return_intermediate:
            return torch.stack(intermediates), ref_pts_per_layer

        out_normed = self.norm(out)
        return out_normed, ref_pts_per_layer
