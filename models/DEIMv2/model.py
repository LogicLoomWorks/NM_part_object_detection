"""
models/deimv2/model.py

DEIMv2Visual — pure-visual deformable DETR-style object detector.

Architecture
────────────
  TimmBackbone  →  FPN  →  input_proj (Conv1x1 + GroupNorm per level)
  →  PositionEmbeddingSine
  →  DeformableTransformerEncoder  (deformable self-attention on all levels)
  →  mixed query selection (top-K encoder outputs → initial reference points;
                            learned content queries)
  →  DINODecoder  (deformable cross-attention + iterative refinement)
  →  per-layer class + box heads
  →  ContrastiveDenoisingTraining  (CDN, training only)

Interface
─────────
  build_model(cfg: DictConfig) -> DEIMv2Visual
  model.forward(images, masks=None, targets=None) -> {
      "pred_logits": (B, Q, num_classes),
      "pred_boxes" : (B, Q, 4),          # cx/cy/w/h in [0,1]
      "aux_outputs": [{"pred_logits":…, "pred_boxes":…}, …]
  }
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from models.DEIMv2.backbone    import TimmBackbone
from models.DEIMv2.neck        import FPN
from models.DEIMv2.transformer import (
    DeformableTransformerEncoder,
    DINODecoder,
    _mlp,
)
from models.DEIMv2.cdn import ContrastiveDenoisingTraining


# ──────────────────────────────────────────────────────────────────────────────
# Positional encoding
# ──────────────────────────────────────────────────────────────────────────────

class PositionEmbeddingSine(nn.Module):
    """Sine/cosine 2-D positional encoding (DETR-style)."""

    def __init__(self, num_pos_feats: int = 128,
                 temperature: int = 10_000, normalize: bool = True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature   = temperature
        self.normalize     = normalize
        self.scale         = 2 * 3.141592653589793

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
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
        pos = torch.cat([pos_y, pos_x], dim=3).permute(0, 3, 1, 2)  # (B,C,H,W)
        return pos


# ──────────────────────────────────────────────────────────────────────────────
# Level-embedding (one learnable vector per feature-map scale)
# ──────────────────────────────────────────────────────────────────────────────

class LevelEmbed(nn.Module):
    """Learnable per-scale additive embedding (D-DETR style)."""
    def __init__(self, num_levels: int, d_model: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_levels, d_model)

    def forward(
        self,
        feats: list[torch.Tensor],
        level_start_idx: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        """Returns (N_total, B, d_model) flat tensor with level embeds added."""
        out_parts = []
        for lvl, feat in enumerate(feats):
            # feat: (B, d_model, H_l, W_l) → (H_l*W_l, B, d_model)
            N_l = feat.shape[2] * feat.shape[3]
            flat = feat.flatten(2).permute(2, 0, 1)   # (N_l, B, d_model)
            flat = flat + self.embed.weight[lvl][None, None, :]
            out_parts.append(flat)
        return torch.cat(out_parts, dim=0)


# ──────────────────────────────────────────────────────────────────────────────
# DEIMv2Visual
# ──────────────────────────────────────────────────────────────────────────────

class DEIMv2Visual(nn.Module):
    """
    Pure-visual deformable DETR-style detector (DEIMv2).

    Input
    -----
    images  : (B, 3, H, W)
    masks   : (B, H, W)   True = padded pixel, False = valid pixel
    targets : list of B dicts (only used during training for CDN)
                each dict has "labels" (Long[N]) and "boxes" (Float[N, 4])

    Output
    ------
    {
      "pred_logits": (B, Q, num_classes),
      "pred_boxes" : (B, Q, 4),          # cx/cy/w/h in [0,1]
      "aux_outputs": [ {"pred_logits":…, "pred_boxes":…}, … ]
    }
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        mc = cfg
        tc = mc.transformer
        d  = int(tc.d_model)

        # ── backbone ──────────────────────────────────────────────────
        self.backbone = TimmBackbone(
            name        = mc.backbone.name,
            pretrained  = mc.backbone.pretrained,
            freeze_at   = mc.backbone.freeze_at,
            out_indices = list(mc.backbone.out_indices),
        )
        bb_channels = self.backbone.out_channels  # e.g. [512, 1024, 2048]

        # ── neck ──────────────────────────────────────────────────────
        self.neck = FPN(
            in_channels  = bb_channels,
            out_channels = int(mc.neck.out_channels),
            num_levels   = int(mc.neck.num_levels),
        )

        # ── projection (Conv1x1 + GroupNorm) per FPN level ────────────
        num_levels = int(mc.neck.num_levels)
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(int(mc.neck.out_channels), d, 1),
                nn.GroupNorm(32, d),
            )
            for _ in range(num_levels)
        ])

        # ── positional encoding ───────────────────────────────────────
        self.pos_embed   = PositionEmbeddingSine(num_pos_feats=d // 2)
        self.level_embed = LevelEmbed(num_levels, d)

        # ── encoder ───────────────────────────────────────────────────
        self.encoder = DeformableTransformerEncoder(
            d_model         = d,
            nhead           = int(tc.nhead),
            num_layers      = int(tc.num_encoder_layers),
            num_levels      = int(tc.num_feature_levels),
            num_points      = int(tc.num_points),
            dim_feedforward = int(tc.dim_feedforward),
            dropout         = float(tc.dropout),
        )

        # ── query embeddings (content + positional) ───────────────────
        num_queries = int(mc.num_queries)
        self.tgt_embed       = nn.Embedding(num_queries, d)  # content queries
        self.query_pos_embed = nn.Embedding(num_queries, d)  # positional queries

        # ── encoder→query selection ───────────────────────────────────
        # Linear to score each encoder token for top-K selection
        self.enc_score_head = nn.Linear(d, int(mc.num_classes))
        self.enc_bbox_head  = _mlp(d, d, 4, num_layers=3)

        # ── per-decoder-layer class + box heads ───────────────────────
        num_dec = int(tc.num_decoder_layers)
        self.class_heads = nn.ModuleList([
            nn.Linear(d, int(mc.num_classes)) for _ in range(num_dec)
        ])
        self.box_heads = nn.ModuleList([
            _mlp(d, d, 4, num_layers=3) for _ in range(num_dec)
        ])

        # ── decoder ───────────────────────────────────────────────────
        self.decoder = DINODecoder(
            d_model         = d,
            nhead           = int(tc.nhead),
            num_layers      = num_dec,
            num_levels      = int(tc.num_feature_levels),
            num_points      = int(tc.num_points),
            dim_feedforward = int(tc.dim_feedforward),
            dropout         = float(tc.dropout),
            return_intermediate = True,
        )
        self.decoder.set_box_heads(self.box_heads)

        # ── CDN ───────────────────────────────────────────────────────
        self.cdn = ContrastiveDenoisingTraining()

        # ── label embedding (shared with CDN) ─────────────────────────
        # num_classes + 1 for the "unknown/noise" class in CDN
        self.label_enc = nn.Embedding(int(mc.num_classes) + 1, d)

        # ── misc ──────────────────────────────────────────────────────
        self.num_queries  = num_queries
        self.num_classes  = int(mc.num_classes)
        self.aux_loss     = bool(mc.aux_loss)
        self.num_levels   = num_levels
        self.d_model      = d

        # CDN hyper-parameters (plain floats, not nn.Parameters)
        self._num_cdn_groups        = int(mc.num_cdn_groups)
        self._cdn_label_noise_ratio = float(mc.cdn_label_noise_ratio)
        self._cdn_box_noise_scale   = float(mc.cdn_box_noise_scale)

        self._init_weights()

    # ── initialisation ────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight)
            nn.init.zeros_(proj[0].bias)
        nn.init.uniform_(self.tgt_embed.weight)
        nn.init.uniform_(self.query_pos_embed.weight)

    # ── helpers ───────────────────────────────────────────────────────

    def _build_multi_scale(
        self,
        proj_feats: list[torch.Tensor],
        src_mask:   torch.Tensor,
    ):
        """Build per-level masks, positional encodings, and flat tensors.

        Returns:
            src              : (N_total, B, d_model)
            pos_enc          : (N_total, B, d_model)
            key_padding_mask : (B, N_total)  True = padded
            spatial_shapes   : LongTensor(num_levels, 2)  (H_l, W_l)
            level_start_idx  : LongTensor(num_levels,)
        """
        B = proj_feats[0].shape[0]
        masks_list, pos_flat_list, feat_flat_list = [], [], []
        spatial_shapes_list = []

        for lvl, feat in enumerate(proj_feats):
            _, _, H, W = feat.shape
            m = F.interpolate(
                src_mask.unsqueeze(1).float(), size=(H, W), mode="nearest"
            ).squeeze(1).bool()                         # (B, H, W)
            p = self.pos_embed(feat, m)                 # (B, d, H, W)

            masks_list.append(m.flatten(1))             # (B, H*W)
            pos_flat_list.append(p.flatten(2).permute(2, 0, 1))   # (H*W, B, d)
            feat_flat_list.append(feat.flatten(2).permute(2, 0, 1))  # (H*W, B, d)
            spatial_shapes_list.append((H, W))

        # Add level embeddings to feature tokens
        level_start = []
        cursor = 0
        for lvl in range(self.num_levels):
            level_start.append(cursor)
            cursor += spatial_shapes_list[lvl][0] * spatial_shapes_list[lvl][1]

        level_start_idx = torch.tensor(
            level_start, dtype=torch.long, device=proj_feats[0].device
        )
        spatial_shapes = torch.tensor(
            spatial_shapes_list, dtype=torch.long, device=proj_feats[0].device
        )  # (num_levels, 2)

        # Flat concatenation with level embeddings
        src_parts = []
        for lvl, flat in enumerate(feat_flat_list):
            flat = flat + self.level_embed.embed.weight[lvl][None, None, :]
            src_parts.append(flat)
        src = torch.cat(src_parts, dim=0)               # (N_total, B, d_model)

        pos_enc          = torch.cat(pos_flat_list, dim=0)            # (N_total, B, d)
        key_padding_mask = torch.cat(masks_list, dim=1)               # (B, N_total)

        return src, pos_enc, key_padding_mask, spatial_shapes, level_start_idx

    def _mixed_query_selection(
        self,
        memory: torch.Tensor,           # (N_total, B, d_model)
        spatial_shapes: torch.Tensor,
        level_start_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select top-K encoder tokens as initial reference points.

        Returns:
            tgt        : (Q, B, d_model)  — content queries (learned)
            ref_pts    : (Q, B, 4)        — reference boxes [0,1] cx/cy/w/h
        """
        N_total, B, _ = memory.shape
        Q = self.num_queries

        # Score every encoder token
        enc_scores = self.enc_score_head(memory)     # (N_total, B, num_classes)
        enc_boxes  = self.enc_bbox_head(memory).sigmoid()  # (N_total, B, 4)

        # Top-K by max-class score
        topk_scores, topk_idx = enc_scores.max(-1)[0].topk(Q, dim=0)
        # topk_idx: (Q, B)

        # Gather reference boxes
        topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, 4)  # (Q, B, 4)
        ref_pts = enc_boxes.gather(0, topk_idx_exp)              # (Q, B, 4)

        # Content queries: learned embeddings, broadcast over batch
        tgt = self.tgt_embed.weight.unsqueeze(1).expand(-1, B, -1)   # (Q, B, d)

        return tgt, ref_pts

    # ── forward ───────────────────────────────────────────────────────

    def forward(
        self,
        images:  torch.Tensor,
        masks:   torch.Tensor | None = None,
        targets: list | None         = None,
    ) -> dict[str, torch.Tensor]:

        B = images.shape[0]

        if masks is None:
            masks = torch.zeros(
                B, images.shape[2], images.shape[3],
                dtype=torch.bool, device=images.device,
            )

        # 1. Backbone → FPN → projection
        bb_feats   = self.backbone(images)
        neck_feats = self.neck(bb_feats)
        proj_feats = [proj(f) for proj, f in zip(self.input_proj, neck_feats)]

        # 2. Build multi-scale flat tensors
        (src, pos_enc,
         key_padding_mask,
         spatial_shapes,
         level_start_idx) = self._build_multi_scale(proj_feats, masks)

        # 3. Encoder
        memory = self.encoder(
            src              = src,
            spatial_shapes   = spatial_shapes,
            level_start_idx  = level_start_idx,
            key_padding_mask = key_padding_mask,
            pos              = pos_enc,
        )  # (N_total, B, d_model)

        # 4. Mixed query selection
        tgt, ref_pts = self._mixed_query_selection(
            memory, spatial_shapes, level_start_idx
        )  # tgt: (Q, B, d); ref_pts: (Q, B, 4)

        query_pos = self.query_pos_embed.weight.unsqueeze(1).expand(-1, B, -1)

        # 5. CDN — prepend denoising queries during training
        dn_tgt, dn_ref_pts, attn_mask, dn_meta = self.cdn.prepare(
            targets     = targets if self.training else None,
            num_queries = self.num_queries,
            d_model     = self.d_model,
            label_enc   = self.label_enc,
            cfg         = self,  # exposes num_cdn_groups etc.
        )

        if dn_tgt is not None:
            # Prepend DN queries
            tgt       = torch.cat([dn_tgt, tgt], dim=0)
            ref_pts   = torch.cat([dn_ref_pts, ref_pts], dim=0)
            # query_pos: pad with zeros for DN slots
            zero_pos  = torch.zeros_like(dn_tgt)
            query_pos = torch.cat([zero_pos, query_pos], dim=0)

        # 6. Decoder
        hs, ref_pts_list = self.decoder(
            tgt                      = tgt,
            memory                   = memory,
            reference_points         = ref_pts,
            spatial_shapes           = spatial_shapes,
            level_start_idx          = level_start_idx,
            memory_key_padding_mask  = key_padding_mask,
            query_pos                = query_pos,
            attn_mask                = attn_mask,
        )
        # hs: (num_layers, Q_total, B, d_model)

        # 7. Per-layer class heads (box refinement done inside decoder)
        outputs_classes, outputs_coords = [], []
        for lvl in range(hs.shape[0]):
            out    = hs[lvl].permute(1, 0, 2)          # (B, Q_total, d)
            logits = self.class_heads[lvl](out)         # (B, Q_total, num_classes)
            coords = ref_pts_list[lvl]                  # (B, Q_total, 4) — already sigmoid
            outputs_classes.append(logits)
            outputs_coords.append(coords)

        # 8. Build output dict (covers full Q_total including any DN slots)
        out_dict: dict = {
            "pred_logits": outputs_classes[-1],
            "pred_boxes" : outputs_coords[-1],
        }
        if self.aux_loss:
            out_dict["aux_outputs"] = [
                {"pred_logits": lc, "pred_boxes": bc}
                for lc, bc in zip(outputs_classes[:-1], outputs_coords[:-1])
            ]

        # 9. Split DN queries out (training only)
        out_dict, _ = self.cdn.postprocess(out_dict, dn_meta)

        return out_dict

    # ── expose CDN config attributes so cdn.prepare() can read them ───

    @property
    def num_cdn_groups(self) -> int:
        return self._num_cdn_groups

    @property
    def cdn_label_noise_ratio(self) -> float:
        return self._cdn_label_noise_ratio

    @property
    def cdn_box_noise_scale(self) -> float:
        return self._cdn_box_noise_scale


# ── factory + checkpoint utils ────────────────────────────────────────────────

def build_model(cfg: DictConfig) -> DEIMv2Visual:
    return DEIMv2Visual(cfg.model)


def save_checkpoint(
    model: DEIMv2Visual,
    path: str,
    cfg: DictConfig | None = None,
) -> None:
    """Save model weights and optionally the config."""
    payload = {"model_state_dict": model.state_dict()}
    if cfg is not None:
        payload["cfg"] = OmegaConf.to_container(cfg, resolve=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    cfg: DictConfig,
    device: Optional[str] = None,
) -> tuple[DEIMv2Visual, dict]:
    """Load a checkpoint.

    Returns:
        model   — DEIMv2Visual instance with weights loaded, in eval mode
        payload — the raw checkpoint dict
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=device)

    model = build_model(cfg)

    if "state_dict" in payload:
        state = {
            k.removeprefix("model."): v
            for k, v in payload["state_dict"].items()
        }
    elif "model_state_dict" in payload:
        state = payload["model_state_dict"]
    else:
        state = payload

    model.load_state_dict(state)
    model.to(device).eval()
    return model, payload
