"""
models/groundingdino/model.py

Fixes applied
─────────────
1. _build_masks_and_pos  – key_padding_mask was inverted.
     OLD:  key_padding_mask = ~mask.flatten(1)   # flips valid→True (BAD)
     NEW:  key_padding_mask =  mask.flatten(1)   # True = padded  (GOOD)

2. TransformerDecoder wiring – decoder must expose per-layer class/box heads
   and return (intermediate_hs, aux_outputs) so the model can build
   pred_logits + pred_boxes + aux_outputs correctly.

3. PositionEmbeddingSine.forward signature   forward(self, x, mask=None)
   (was forward(self, x) – crashed when called with two args).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from models.groundingdino.backbone import TimmBackbone
from models.groundingdino.neck     import FPN
from models.groundingdino.transformer import TransformerEncoder, TransformerDecoder


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
        # mask: True = padded  →  not_mask: True = valid
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)   # (B,H,W)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)   # (B,H,W)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32,
                              device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t   # (B,H,W,C/2)
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(),
                              pos_x[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(),
                              pos_y[:, :, :, 1::2].cos()], dim=4).flatten(3)

        pos = torch.cat([pos_y, pos_x], dim=3).permute(0, 3, 1, 2)  # (B,C,H,W)
        return pos


# ──────────────────────────────────────────────────────────────────────────────
# Detection head helpers
# ──────────────────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(num_layers):
        i_dim = in_dim if i == 0 else hidden_dim
        o_dim = out_dim if i == num_layers - 1 else hidden_dim
        layers.append(nn.Linear(i_dim, o_dim))
        if i < num_layers - 1:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# ──────────────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────────────

class GroundingDINOVisual(nn.Module):
    """
    Pure-visual DETR-style detector (no text grounding).

    Input
    -----
    images : (B, 3, H, W)
    masks  : (B, H, W)   True = padded pixel, False = valid pixel

    Output
    ------
    {
      'pred_logits': (B, Q, num_classes),
      'pred_boxes' : (B, Q, 4),          # cx/cy/w/h in [0,1]
      'aux_outputs': [ {'pred_logits':…, 'pred_boxes':…}, … ]
    }
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        mc   = cfg
        tc   = mc.transformer
        d    = tc.d_model                   # 256

        # ── backbone ──────────────────────────────────────────────────────────
        self.backbone = TimmBackbone(
            name       = mc.backbone.name,
            pretrained = mc.backbone.pretrained,
            freeze_at  = mc.backbone.freeze_at,
            out_indices= list(mc.backbone.out_indices),
        )
        bb_channels = self.backbone.out_channels   # e.g. [256, 512, 1024]

        # ── neck ──────────────────────────────────────────────────────────────
        self.neck = FPN(
            in_channels  = bb_channels,
            out_channels = mc.neck.out_channels,
            num_levels   = mc.neck.num_levels,
        )

        # ── projection (Conv1×1 + GN) per FPN level ───────────────────────────
        num_levels = mc.neck.num_levels
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mc.neck.out_channels, d, 1),
                nn.GroupNorm(32, d),
            )
            for _ in range(num_levels)
        ])

        # ── positional encoding ───────────────────────────────────────────────
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=d // 2)

        # ── transformer ───────────────────────────────────────────────────────
        self.encoder = TransformerEncoder(
            d_model         = d,
            nhead           = tc.nhead,
            num_layers      = tc.num_encoder_layers,
            dim_feedforward = tc.dim_feedforward,
            dropout         = tc.dropout,
        )

        self.decoder = TransformerDecoder(
            d_model            = d,
            nhead              = tc.nhead,
            num_layers         = tc.num_decoder_layers,
            dim_feedforward    = tc.dim_feedforward,
            dropout            = tc.dropout,
            return_intermediate= True,
        )

        # ── query embeddings ──────────────────────────────────────────────────
        self.tgt_embed         = nn.Embedding(mc.num_queries, d)
        self.query_embed       = nn.Embedding(mc.num_queries, d)
        self.reference_points  = nn.Embedding(mc.num_queries, 4)
        nn.init.uniform_(self.reference_points.weight, 0.05, 0.95)

        # ── per-decoder-layer classification + box heads ──────────────────────
        num_dec = tc.num_decoder_layers
        self.class_heads = nn.ModuleList([
            nn.Linear(d, mc.num_classes) for _ in range(num_dec)
        ])
        self.box_heads = nn.ModuleList([
            _mlp(d, d, 4, num_layers=3) for _ in range(num_dec)
        ])

        self.num_queries  = mc.num_queries
        self.num_classes  = mc.num_classes
        self.aux_loss     = mc.aux_loss

    # ── internal helpers ──────────────────────────────────────────────────────

    def _build_masks_and_pos(
        self,
        proj_feats: list[torch.Tensor],
        src_mask:   torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        masks_list, pos_list = [], []
        for feat in proj_feats:
            _, _, H, W = feat.shape
            m = F.interpolate(
                src_mask.unsqueeze(1).float(), size=(H, W), mode="nearest"
            ).squeeze(1).bool()
            masks_list.append(m)
            pos_list.append(self.pos_embed(feat, m))
        return masks_list, pos_list

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        images: torch.Tensor,
        masks:  torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:

        B = images.shape[0]

        # Default to all-valid mask if none provided
        if masks is None:
            masks = torch.zeros(
                B, images.shape[2], images.shape[3],
                dtype=torch.bool, device=images.device,
            )

        # 1. Backbone → FPN → projection
        bb_feats   = self.backbone(images)
        neck_feats = self.neck(bb_feats)
        proj_feats = [proj(f) for proj, f in zip(self.input_proj, neck_feats)]

        # 2. Masks + positional encodings at each level
        masks_list, pos_list = self._build_masks_and_pos(proj_feats, masks)

        # 3. Flatten + concatenate all levels → (B, N, C)
        src = torch.cat(
            [f.flatten(2).permute(0, 2, 1) for f in proj_feats], dim=1
        )
        pos_enc = torch.cat(
            [p.flatten(2).permute(0, 2, 1) for p in pos_list], dim=1
        )
        key_padding_mask = torch.cat(
            [m.flatten(1) for m in masks_list], dim=1
        )  # (B, N)  True = padded

        # 4. Encoder  (S, B, C)
        memory = self.encoder(
            src.transpose(0, 1),
            key_padding_mask = key_padding_mask,
            pos              = pos_enc.transpose(0, 1),
        )  # (N, B, C)

        # 5. Query initialisation
        tgt       = self.tgt_embed.weight.unsqueeze(1).expand(-1, B, -1)
        query_pos = self.query_embed.weight.unsqueeze(1).expand(-1, B, -1)
        ref_pts   = (
            self.reference_points.weight
            .sigmoid()
            .clamp(1e-4, 1 - 1e-4)
            .unsqueeze(1).expand(-1, B, -1)
        )

        # 6. Decoder → (num_layers, Q, B, C)
        hs = self.decoder(
            tgt                      = tgt,
            memory                   = memory,
            memory_key_padding_mask  = key_padding_mask,
            pos                      = pos_enc.transpose(0, 1),
            query_pos                = query_pos,
        )

        # 7. Per-layer heads
        ref_pts_bq = ref_pts.permute(1, 0, 2)
        ref_logit  = torch.log(ref_pts_bq / (1.0 - ref_pts_bq))

        outputs_classes, outputs_coords = [], []
        for lvl in range(hs.shape[0]):
            out    = hs[lvl].permute(1, 0, 2)
            logits = self.class_heads[lvl](out)
            delta  = self.box_heads[lvl](out)
            coords = (ref_logit + delta).sigmoid().clamp(0.0, 1.0)
            outputs_classes.append(logits)
            outputs_coords.append(coords)

        out_dict: dict = {
            "pred_logits": outputs_classes[-1],
            "pred_boxes" : outputs_coords[-1],
        }
        if self.aux_loss:
            out_dict["aux_outputs"] = [
                {"pred_logits": lc, "pred_boxes": bc}
                for lc, bc in zip(outputs_classes[:-1], outputs_coords[:-1])
            ]
        return out_dict


# ── convenience alias ─────────────────────────────────────────────────────────
GroundingDINO = GroundingDINOVisual


# ── factory + checkpoint utils ───────────────────────────────────────────────

def build_model(cfg: DictConfig) -> GroundingDINOVisual:
    return GroundingDINOVisual(cfg)


def save_checkpoint(
    model: GroundingDINOVisual,
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
) -> tuple[GroundingDINOVisual, dict]:
    """Load a checkpoint saved by save_checkpoint or Lightning's ModelCheckpoint.

    Returns:
        model   — GroundingDINOVisual instance with weights loaded, in eval mode
        payload — the raw checkpoint dict
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=device)

    model = build_model(cfg)

    # Support both raw state dicts and Lightning checkpoints
    if "state_dict" in payload:
        # Lightning saves weights with 'model.' prefix from DetectionLightningModule
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