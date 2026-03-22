"""
submission/model.py

Consolidated DEIMv2Visual model for inference.
All components inlined from models/DEIMv2/{backbone,neck,transformer,cdn,model}.py
No omegaconf dependency — config is hardcoded for the trained checkpoint.
"""
from __future__ import annotations

import math
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm


# ──────────────────────────────────────────────────────────────────────────────
# Hardcoded model config (from configs/deimv2.yaml)
# ──────────────────────────────────────────────────────────────────────────────

NUM_QUERIES       = 300
NUM_CLASSES       = 356
D_MODEL           = 256
NHEAD             = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD   = 1024
DROPOUT           = 0.0
NUM_FEATURE_LEVELS = 4
NUM_POINTS        = 4
BACKBONE_NAME     = "resnet50"
BACKBONE_OUT_INDICES = [2, 3, 4]
NECK_OUT_CHANNELS = 256
NECK_NUM_LEVELS   = 4
NUM_CDN_GROUPS    = 5
CDN_LABEL_NOISE   = 0.5
CDN_BOX_NOISE     = 1.0
AUX_LOSS          = True


# ──────────────────────────────────────────────────────────────────────────────
# Backbone
# ──────────────────────────────────────────────────────────────────────────────

class TimmBackbone(nn.Module):
    def __init__(
        self,
        name: str,
        out_indices: List[int],
        pretrained: bool = False,
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
        all_info = self.model.feature_info.info
        self._out_channels: List[int] = [all_info[i]["num_chs"] for i in out_indices]
        self._out_strides: List[int] = [all_info[i]["reduction"] for i in out_indices]

    @property
    def out_channels(self) -> List[int]:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.model(x)


# ──────────────────────────────────────────────────────────────────────────────
# Neck (FPN)
# ──────────────────────────────────────────────────────────────────────────────

class FPN(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        num_levels: int = 4,
    ) -> None:
        super().__init__()
        assert num_levels >= len(in_channels)
        self.num_levels = num_levels
        self.num_in = len(in_channels)
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels]
        )
        self.output_convs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
             for _ in in_channels]
        )
        num_extra = num_levels - self.num_in
        self.extra = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
             for _ in range(num_extra)]
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        laterals = [conv(f) for conv, f in zip(self.lateral, features)]
        for i in range(self.num_in - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:], mode="nearest",
            )
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        current = outputs[-1]
        for extra_conv in self.extra:
            current = extra_conv(F.relu(current))
            outputs.append(current)
        return outputs


# ──────────────────────────────────────────────────────────────────────────────
# Positional encoding
# ──────────────────────────────────────────────────────────────────────────────

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats: int = 128,
                 temperature: int = 10_000, normalize: bool = True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * 3.141592653589793

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, H, W = x.shape
        if mask is None:
            mask = torch.zeros(B, H, W, dtype=torch.bool, device=x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(),
                              pos_x[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(),
                              pos_y[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos = torch.cat([pos_y, pos_x], dim=3).permute(0, 3, 1, 2)
        return pos


# ──────────────────────────────────────────────────────────────────────────────
# Level embedding
# ──────────────────────────────────────────────────────────────────────────────

class LevelEmbed(nn.Module):
    def __init__(self, num_levels: int, d_model: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_levels, d_model)

    def forward(self, feats, level_start_idx, spatial_shapes):
        out_parts = []
        for lvl, feat in enumerate(feats):
            flat = feat.flatten(2).permute(2, 0, 1)
            flat = flat + self.embed.weight[lvl][None, None, :]
            out_parts.append(flat)
        return torch.cat(out_parts, dim=0)


# ──────────────────────────────────────────────────────────────────────────────
# Deformable attention
# ──────────────────────────────────────────────────────────────────────────────

class DeformableAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = d_model // num_heads
        self.sampling_offsets = nn.Linear(d_model, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(d_model, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
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
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_idx: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Q, B, _ = query.shape
        N_total = value.shape[0]

        value = self.value_proj(value).permute(1, 0, 2)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        value = value.view(B, N_total, self.num_heads, self.head_dim)

        query_bq = query.permute(1, 0, 2)
        offsets = self.sampling_offsets(query_bq).view(
            B, Q, self.num_heads, self.num_levels, self.num_points, 2
        )
        attn_w = self.attention_weights(query_bq).view(
            B, Q, self.num_heads, self.num_levels * self.num_points
        )
        attn_w = attn_w.softmax(dim=-1).view(
            B, Q, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 4:
            ref_cxcy = reference_points[..., :2]
            if ref_cxcy.dim() == 3:
                ref_pt = ref_cxcy.permute(1, 0, 2).unsqueeze(2).expand(
                    -1, -1, self.num_levels, -1
                )
            else:
                ref_pt = ref_cxcy.permute(1, 0, 2, 3)
                if ref_pt.shape[2] != self.num_levels:
                    ref_pt = ref_pt[:, :, :1, :].expand(-1, -1, self.num_levels, -1)
        elif reference_points.shape[-1] == 2:
            if reference_points.dim() == 3:
                ref_pt = reference_points.permute(1, 0, 2).unsqueeze(2).expand(
                    -1, -1, self.num_levels, -1
                )
            else:
                ref_pt = reference_points.permute(1, 0, 2, 3)
        else:
            raise ValueError(f"Unexpected reference_points shape: {reference_points.shape}")

        level_shapes = spatial_shapes.float()
        offset_normaliser = torch.stack(
            [level_shapes[:, 1], level_shapes[:, 0]], dim=-1
        ).view(1, 1, 1, self.num_levels, 1, 2)

        ref_expanded = ref_pt.unsqueeze(2).unsqueeze(4)
        sampling_locs = ref_expanded + offsets / (offset_normaliser + 1e-6)
        sampling_locs = sampling_locs.clamp(0.0, 1.0)

        outputs = []
        for lvl, (H_l, W_l) in enumerate(spatial_shapes.tolist()):
            H_l, W_l = int(H_l), int(W_l)
            start = int(level_start_idx[lvl].item())
            end = start + H_l * W_l
            v_lvl = value[:, start:end, :, :]
            v_lvl = v_lvl.view(B, H_l, W_l, self.num_heads, self.head_dim)
            v_lvl = v_lvl.permute(0, 3, 4, 1, 2)
            v_lvl = v_lvl.reshape(B * self.num_heads, self.head_dim, H_l, W_l)
            s_locs = sampling_locs[:, :, :, lvl, :, :]
            s_locs = s_locs * 2.0 - 1.0
            s_locs = s_locs.permute(0, 2, 1, 3, 4)
            s_locs = s_locs.reshape(B * self.num_heads, Q * self.num_points, 1, 2)
            sampled = F.grid_sample(
                v_lvl, s_locs, mode="bilinear", padding_mode="zeros", align_corners=False,
            )
            sampled = sampled.view(B, self.num_heads, self.head_dim, Q, self.num_points)
            sampled = sampled.permute(0, 3, 1, 4, 2)
            w_lvl = attn_w[:, :, :, lvl, :]
            out_lvl = (sampled * w_lvl.unsqueeze(-1)).sum(dim=3)
            outputs.append(out_lvl)

        out = torch.stack(outputs, dim=0).sum(dim=0).reshape(B, Q, self.d_model)
        out = self.dropout(self.output_proj(out))
        return out.permute(1, 0, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────────────────────────────────────

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_levels=4, num_points=4,
                 dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.self_attn = DeformableAttention(
            d_model=d_model, num_heads=nhead, num_levels=num_levels,
            num_points=num_points, dropout=dropout,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, src, reference_points, spatial_shapes, level_start_idx,
                key_padding_mask=None, pos=None):
        q = src if pos is None else src + pos
        src2 = self.self_attn(
            query=q, reference_points=reference_points, value=src,
            spatial_shapes=spatial_shapes, level_start_idx=level_start_idx,
            key_padding_mask=key_padding_mask,
        )
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, num_levels=4,
                 num_points=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            DeformableTransformerEncoderLayer(
                d_model=d_model, nhead=nhead, num_levels=num_levels,
                num_points=num_points, dim_feedforward=dim_feedforward, dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _get_reference_points(spatial_shapes, device):
        ref_pts_list = []
        for H_l, W_l in spatial_shapes.tolist():
            H_l, W_l = int(H_l), int(W_l)
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_l - 0.5, H_l, device=device) / H_l,
                torch.linspace(0.5, W_l - 0.5, W_l, device=device) / W_l,
                indexing="ij",
            )
            ref_pts = torch.stack([ref_x.reshape(-1), ref_y.reshape(-1)], dim=-1)
            ref_pts_list.append(ref_pts)
        ref_pts_all = torch.cat(ref_pts_list, dim=0)
        num_levels = len(spatial_shapes)
        return ref_pts_all[:, None, None, :].expand(-1, 1, num_levels, -1)

    def forward(self, src, spatial_shapes, level_start_idx,
                key_padding_mask=None, pos=None):
        B = src.shape[1]
        device = src.device
        reference_points = self._get_reference_points(
            spatial_shapes, device
        ).expand(-1, B, -1, -1)
        out = src
        for layer in self.layers:
            out = layer(
                src=out, reference_points=reference_points,
                spatial_shapes=spatial_shapes, level_start_idx=level_start_idx,
                key_padding_mask=key_padding_mask, pos=pos,
            )
        return self.norm(out)


# ──────────────────────────────────────────────────────────────────────────────
# MLP helper
# ──────────────────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> nn.Sequential:
    layers: list = []
    for i in range(num_layers):
        i_dim = in_dim if i == 0 else hidden_dim
        o_dim = out_dim if i == num_layers - 1 else hidden_dim
        layers.append(nn.Linear(i_dim, o_dim))
        if i < num_layers - 1:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# ──────────────────────────────────────────────────────────────────────────────
# Decoder
# ──────────────────────────────────────────────────────────────────────────────

class DINODecoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_levels=4, num_points=4,
                 dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )
        self.cross_attn = DeformableAttention(
            d_model=d_model, num_heads=nhead, num_levels=num_levels,
            num_points=num_points, dropout=dropout,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, tgt, memory, reference_points, spatial_shapes,
                level_start_idx, memory_key_padding_mask=None,
                query_pos=None, attn_mask=None):
        q = k = tgt if query_pos is None else tgt + query_pos
        tgt2, _ = self.self_attn(q, k, tgt, attn_mask=attn_mask)
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        q_with_pos = tgt if query_pos is None else tgt + query_pos
        tgt2 = self.cross_attn(
            query=q_with_pos, reference_points=reference_points, value=memory,
            spatial_shapes=spatial_shapes, level_start_idx=level_start_idx,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


class DINODecoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, num_levels=4,
                 num_points=4, dim_feedforward=1024, dropout=0.0,
                 return_intermediate=True):
        super().__init__()
        self.layers = nn.ModuleList([
            DINODecoderLayer(
                d_model=d_model, nhead=nhead, num_levels=num_levels,
                num_points=num_points, dim_feedforward=dim_feedforward, dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate
        self.num_layers = num_layers
        self.box_heads: Optional[nn.ModuleList] = None

    def set_box_heads(self, box_heads: nn.ModuleList) -> None:
        self.box_heads = box_heads

    def forward(self, tgt, memory, reference_points, spatial_shapes,
                level_start_idx, memory_key_padding_mask=None,
                query_pos=None, attn_mask=None):
        out = tgt
        ref = reference_points
        intermediates: list = []
        ref_pts_per_layer: list = []

        for lvl, layer in enumerate(self.layers):
            out = layer(
                tgt=out, memory=memory, reference_points=ref,
                spatial_shapes=spatial_shapes, level_start_idx=level_start_idx,
                memory_key_padding_mask=memory_key_padding_mask,
                query_pos=query_pos, attn_mask=attn_mask,
            )
            out_normed = self.norm(out)
            if self.return_intermediate:
                intermediates.append(out_normed)
            if self.box_heads is not None:
                ref_bq = ref.permute(1, 0, 2)
                ref_logit = torch.log(
                    ref_bq.clamp(1e-6, 1 - 1e-6) /
                    (1.0 - ref_bq.clamp(1e-6, 1 - 1e-6))
                )
                out_bq = out_normed.permute(1, 0, 2)
                delta = self.box_heads[lvl](out_bq)
                ref_new = (ref_logit + delta).sigmoid().clamp(0.0, 1.0)
                ref_pts_per_layer.append(ref_new)
                ref = ref_new.permute(1, 0, 2)
            else:
                ref_pts_per_layer.append(ref.permute(1, 0, 2))

        if self.return_intermediate:
            return torch.stack(intermediates), ref_pts_per_layer
        out_normed = self.norm(out)
        return out_normed, ref_pts_per_layer


# ──────────────────────────────────────────────────────────────────────────────
# CDN (only postprocess needed for inference)
# ──────────────────────────────────────────────────────────────────────────────

class ContrastiveDenoisingTraining(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def prepare(self, targets, num_queries, d_model, label_enc, cfg):
        if targets is None or not self.training:
            return None, None, None, None
        return None, None, None, None

    def postprocess(self, outputs, dn_meta):
        if dn_meta is None:
            return outputs, None
        num_dn = dn_meta["num_dn_queries"]
        def _split(logits, boxes):
            return (
                {"pred_logits": logits[:, num_dn:], "pred_boxes": boxes[:, num_dn:]},
                {"pred_logits": logits[:, :num_dn], "pred_boxes": boxes[:, :num_dn]},
            )
        clean, dn_out = _split(outputs["pred_logits"], outputs["pred_boxes"])
        if "aux_outputs" in outputs:
            clean_aux = []
            for aux in outputs["aux_outputs"]:
                c, _ = _split(aux["pred_logits"], aux["pred_boxes"])
                clean_aux.append(c)
            clean["aux_outputs"] = clean_aux
        return clean, dn_out


# ──────────────────────────────────────────────────────────────────────────────
# DEIMv2Visual
# ──────────────────────────────────────────────────────────────────────────────

class DEIMv2Visual(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # backbone
        self.backbone = TimmBackbone(
            name=BACKBONE_NAME,
            out_indices=BACKBONE_OUT_INDICES,
            pretrained=False,
        )
        bb_channels = self.backbone.out_channels

        # neck
        self.neck = FPN(
            in_channels=bb_channels,
            out_channels=NECK_OUT_CHANNELS,
            num_levels=NECK_NUM_LEVELS,
        )

        # input projection
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(NECK_OUT_CHANNELS, D_MODEL, 1),
                nn.GroupNorm(32, D_MODEL),
            )
            for _ in range(NECK_NUM_LEVELS)
        ])

        # positional encoding
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=D_MODEL // 2)
        self.level_embed = LevelEmbed(NECK_NUM_LEVELS, D_MODEL)

        # encoder
        self.encoder = DeformableTransformerEncoder(
            d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_ENCODER_LAYERS,
            num_levels=NUM_FEATURE_LEVELS, num_points=NUM_POINTS,
            dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT,
        )

        # query embeddings
        self.tgt_embed = nn.Embedding(NUM_QUERIES, D_MODEL)
        self.query_pos_embed = nn.Embedding(NUM_QUERIES, D_MODEL)

        # encoder->query selection
        self.enc_score_head = nn.Linear(D_MODEL, NUM_CLASSES)
        self.enc_bbox_head = _mlp(D_MODEL, D_MODEL, 4, num_layers=3)

        # per-decoder-layer heads
        self.class_heads = nn.ModuleList([
            nn.Linear(D_MODEL, NUM_CLASSES) for _ in range(NUM_DECODER_LAYERS)
        ])
        self.box_heads = nn.ModuleList([
            _mlp(D_MODEL, D_MODEL, 4, num_layers=3) for _ in range(NUM_DECODER_LAYERS)
        ])

        # decoder
        self.decoder = DINODecoder(
            d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_DECODER_LAYERS,
            num_levels=NUM_FEATURE_LEVELS, num_points=NUM_POINTS,
            dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT,
            return_intermediate=True,
        )
        self.decoder.set_box_heads(self.box_heads)

        # CDN (inference-only: always returns None)
        self.cdn = ContrastiveDenoisingTraining()

        # label embedding (shared with CDN)
        self.label_enc = nn.Embedding(NUM_CLASSES + 1, D_MODEL)

        self.num_queries = NUM_QUERIES
        self.num_classes = NUM_CLASSES
        self.aux_loss = AUX_LOSS
        self.num_levels = NECK_NUM_LEVELS
        self.d_model = D_MODEL
        self._num_cdn_groups = NUM_CDN_GROUPS
        self._cdn_label_noise_ratio = CDN_LABEL_NOISE
        self._cdn_box_noise_scale = CDN_BOX_NOISE

        self._init_weights()

    def _init_weights(self) -> None:
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight)
            nn.init.zeros_(proj[0].bias)
        nn.init.uniform_(self.tgt_embed.weight)
        nn.init.uniform_(self.query_pos_embed.weight)

    def _build_multi_scale(self, proj_feats, src_mask):
        B = proj_feats[0].shape[0]
        masks_list, pos_flat_list, feat_flat_list = [], [], []
        spatial_shapes_list = []

        for lvl, feat in enumerate(proj_feats):
            _, _, H, W = feat.shape
            m = F.interpolate(
                src_mask.unsqueeze(1).float(), size=(H, W), mode="nearest"
            ).squeeze(1).bool()
            p = self.pos_embed(feat, m)
            masks_list.append(m.flatten(1))
            pos_flat_list.append(p.flatten(2).permute(2, 0, 1))
            feat_flat_list.append(feat.flatten(2).permute(2, 0, 1))
            spatial_shapes_list.append((H, W))

        level_start = []
        cursor = 0
        for lvl in range(self.num_levels):
            level_start.append(cursor)
            cursor += spatial_shapes_list[lvl][0] * spatial_shapes_list[lvl][1]

        level_start_idx = torch.tensor(level_start, dtype=torch.long, device=proj_feats[0].device)
        spatial_shapes = torch.tensor(spatial_shapes_list, dtype=torch.long, device=proj_feats[0].device)

        src_parts = []
        for lvl, flat in enumerate(feat_flat_list):
            flat = flat + self.level_embed.embed.weight[lvl][None, None, :]
            src_parts.append(flat)
        src = torch.cat(src_parts, dim=0)

        pos_enc = torch.cat(pos_flat_list, dim=0)
        key_padding_mask = torch.cat(masks_list, dim=1)

        return src, pos_enc, key_padding_mask, spatial_shapes, level_start_idx

    def _mixed_query_selection(self, memory, spatial_shapes, level_start_idx):
        N_total, B, _ = memory.shape
        Q = self.num_queries
        enc_scores = self.enc_score_head(memory)
        enc_boxes = self.enc_bbox_head(memory).sigmoid()
        topk_scores, topk_idx = enc_scores.max(-1)[0].topk(Q, dim=0)
        topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, 4)
        ref_pts = enc_boxes.gather(0, topk_idx_exp)
        tgt = self.tgt_embed.weight.unsqueeze(1).expand(-1, B, -1)
        return tgt, ref_pts

    def forward(self, images, masks=None, targets=None):
        B = images.shape[0]
        if masks is None:
            masks = torch.zeros(
                B, images.shape[2], images.shape[3],
                dtype=torch.bool, device=images.device,
            )

        bb_feats = self.backbone(images)
        neck_feats = self.neck(bb_feats)
        proj_feats = [proj(f) for proj, f in zip(self.input_proj, neck_feats)]

        (src, pos_enc, key_padding_mask,
         spatial_shapes, level_start_idx) = self._build_multi_scale(proj_feats, masks)

        memory = self.encoder(
            src=src, spatial_shapes=spatial_shapes,
            level_start_idx=level_start_idx,
            key_padding_mask=key_padding_mask, pos=pos_enc,
        )

        tgt, ref_pts = self._mixed_query_selection(
            memory, spatial_shapes, level_start_idx
        )
        query_pos = self.query_pos_embed.weight.unsqueeze(1).expand(-1, B, -1)

        dn_tgt, dn_ref_pts, attn_mask, dn_meta = self.cdn.prepare(
            targets=None, num_queries=self.num_queries,
            d_model=self.d_model, label_enc=self.label_enc, cfg=self,
        )

        hs, ref_pts_list = self.decoder(
            tgt=tgt, memory=memory, reference_points=ref_pts,
            spatial_shapes=spatial_shapes, level_start_idx=level_start_idx,
            memory_key_padding_mask=key_padding_mask,
            query_pos=query_pos, attn_mask=None,
        )

        outputs_classes, outputs_coords = [], []
        for lvl in range(hs.shape[0]):
            out = hs[lvl].permute(1, 0, 2)
            logits = self.class_heads[lvl](out)
            coords = ref_pts_list[lvl]
            outputs_classes.append(logits)
            outputs_coords.append(coords)

        out_dict = {
            "pred_logits": outputs_classes[-1],
            "pred_boxes": outputs_coords[-1],
        }
        if self.aux_loss:
            out_dict["aux_outputs"] = [
                {"pred_logits": lc, "pred_boxes": bc}
                for lc, bc in zip(outputs_classes[:-1], outputs_coords[:-1])
            ]

        out_dict, _ = self.cdn.postprocess(out_dict, dn_meta)
        return out_dict

    @property
    def num_cdn_groups(self) -> int:
        return self._num_cdn_groups

    @property
    def cdn_label_noise_ratio(self) -> float:
        return self._cdn_label_noise_ratio

    @property
    def cdn_box_noise_scale(self) -> float:
        return self._cdn_box_noise_scale


# ──────────────────────────────────────────────────────────────────────────────
# Load helper
# ──────────────────────────────────────────────────────────────────────────────

def load_model(weights_path: str, device: str = "cpu") -> DEIMv2Visual:
    """Build DEIMv2Visual and load weights from a checkpoint file.

    Handles Lightning checkpoints (model. prefix) and raw state dicts.
    """
    model = DEIMv2Visual()

    ckpt = torch.load(weights_path, map_location=device, weights_only=True)

    # Lightning checkpoint: {"state_dict": {...}}
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    # Strip leading "model." prefix from Lightning-wrapped weights
    if any(k.startswith("model.") for k in state):
        state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}

    # Cast FP16 → FP32 for inference stability on CPU; keep FP16 on CUDA
    if device == "cpu":
        state = {k: v.float() if v.dtype == torch.float16 else v
                 for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.to(device).train(False)
    return model
