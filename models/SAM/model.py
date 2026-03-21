"""
models/sam/model.py

SamDetector — SAM ViT-B used as a prompt-guided object detector.

Architecture overview
─────────────────────
1. Image encoder  : SAM ViT-B (loaded from segment_anything if available,
                    otherwise falls back to timm vit_base_patch16_224).
2. Prompt encoder : LearnedGridPrompts — a set of learnable 2-D point coords
                    that produce ``num_queries`` sparse embeddings.
3. Mask decoder   : Simplified cross-attention decoder (implemented inline).
                    For each query, cross-attends to the image embeddings and
                    produces an iou-token vector + a mask logit map.
4. Class head     : ClassificationHead maps iou-token → class logits.
5. Box extraction : Tight bounding boxes derived from predicted masks
                    (cx/cy/w/h in [0, 1]).

Output interface (identical to GroundingDINOVisual)
────────────────────────────────────────────────────
{
  "pred_logits": (B, Q, num_classes),   # raw logits
  "pred_boxes":  (B, Q, 4),             # cx/cy/w/h in [0, 1]
  "aux_outputs": []                      # empty — SAM has no aux layers
}
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from models.sam.prompt_encoder import LearnedGridPrompts
from models.sam.classification_head import ClassificationHead

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Utility: masks → bounding boxes
# ──────────────────────────────────────────────────────────────────────────────

def masks_to_boxes_normalized(
    mask_logits: torch.Tensor,
    img_h: int,
    img_w: int,
) -> torch.Tensor:
    """Convert binary mask logits to normalised cx/cy/w/h bounding boxes.

    Args:
        mask_logits: ``(B, Q, H, W)`` — raw mask scores (threshold at 0.5
                     after sigmoid).
        img_h:       Image height used for normalisation.
        img_w:       Image width used for normalisation.

    Returns:
        boxes: ``(B, Q, 4)`` in cx/cy/w/h format, values in [0, 1].
               Queries whose mask is all-zero get box ``[0.5, 0.5, 0.0, 0.0]``.
    """
    B, Q, H, W = mask_logits.shape
    device = mask_logits.device

    # Threshold
    masks = (mask_logits.sigmoid() > 0.5).float()  # (B, Q, H, W)

    # Build pixel grids once
    ys = torch.arange(H, device=device, dtype=torch.float32) / max(H - 1, 1)
    xs = torch.arange(W, device=device, dtype=torch.float32) / max(W - 1, 1)

    boxes = torch.zeros(B, Q, 4, device=device, dtype=torch.float32)
    # Default for empty masks
    boxes[..., 0] = 0.5
    boxes[..., 1] = 0.5

    for b in range(B):
        for q in range(Q):
            m = masks[b, q]                         # (H, W)
            if m.sum() == 0:
                continue

            # Rows and columns that have any foreground pixel
            row_mask = m.any(dim=1)                 # (H,)
            col_mask = m.any(dim=0)                 # (W,)

            y_min = ys[row_mask].min()
            y_max = ys[row_mask].max()
            x_min = xs[col_mask].min()
            x_max = xs[col_mask].max()

            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            w  = x_max - x_min
            h  = y_max - y_min

            boxes[b, q] = torch.stack([cx, cy, w, h])

    return boxes.clamp(0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Simplified inline mask decoder
# ──────────────────────────────────────────────────────────────────────────────

class SimpleMaskDecoder(nn.Module):
    """Lightweight cross-attention decoder that mimics SAM's mask decoder.

    For each query (point embedding) the decoder cross-attends to flattened
    image features and produces:
    - An iou-token vector used for classification.
    - A mask logit map (upsampled to the spatial size of the image embeddings).

    Args:
        embed_dim:      Feature dimension (256 for SAM ViT-B).
        nhead:          Number of attention heads.
        dim_feedforward: FFN width.
        dropout:        Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Cross-attention: queries attend to image features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # Self-attention among queries
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(embed_dim)

        # Project iou-token to a mask (applied on top of image features)
        self.mask_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the decoder.

        Args:
            image_embeddings: ``(B, C, H, W)`` image feature map from the
                              image encoder.
            point_embeddings: ``(Q, 1, C)`` per-query sparse embeddings from
                              LearnedGridPrompts.

        Returns:
            iou_tokens: ``(B, Q, C)`` — per-query summary vectors for
                         classification.
            mask_logits: ``(B, Q, H, W)`` — per-query mask predictions.
        """
        B, C, H, W = image_embeddings.shape
        Q = point_embeddings.shape[0]

        # Flatten image features: (B, H*W, C)
        img_flat = image_embeddings.flatten(2).permute(0, 2, 1)  # (B, HW, C)

        # Expand point embeddings for the batch: (B, Q, C)
        # point_embeddings shape: (Q, 1, C) → squeeze middle dim → (Q, C)
        q_emb = point_embeddings.squeeze(1)                       # (Q, C)
        q_emb = q_emb.unsqueeze(0).expand(B, -1, -1)             # (B, Q, C)

        # Self-attention among queries
        q2, _ = self.self_attn(q_emb, q_emb, q_emb)
        q_emb = self.norm2(q_emb + q2)

        # Cross-attention: queries attend to image features
        q3, _ = self.cross_attn(q_emb, img_flat, img_flat)
        q_emb = self.norm1(q_emb + q3)

        # FFN
        q_emb = self.norm3(q_emb + self.ffn(q_emb))

        # iou tokens are the final query vectors
        iou_tokens = q_emb                                        # (B, Q, C)

        # Mask prediction: dot product between projected queries and image feats
        mask_keys = self.mask_proj(iou_tokens)                    # (B, Q, C)
        # (B, Q, C) x (B, C, HW) → (B, Q, HW) → (B, Q, H, W)
        mask_logits = torch.bmm(mask_keys, img_flat.permute(0, 2, 1))
        mask_logits = mask_logits.view(B, Q, H, W)

        return iou_tokens, mask_logits


# ──────────────────────────────────────────────────────────────────────────────
# Timm fallback image encoder (used when segment_anything is not installed)
# ──────────────────────────────────────────────────────────────────────────────

class _TimmFallbackEncoder(nn.Module):
    """Minimal ViT-Base wrapper (timm) that projects to 256-channel output.

    This is used when the ``segment_anything`` package is unavailable.
    SAM weights are NOT loaded; timm ImageNet pretrained weights are used.
    """

    def __init__(self, image_size: int = 1024, out_channels: int = 256) -> None:
        super().__init__()
        import timm  # lazy import — checked at construction, not at module load

        self.image_size = image_size

        # timm ViT-B/16 — use 224 input size in the model config but we resize
        # the input before passing it so the actual spatial output doesn't depend
        # on image_size.
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0,           # remove classification head
            global_pool="",          # keep all patch tokens
        )
        vit_dim = self.backbone.embed_dim   # 768 for ViT-B

        # Project from ViT feature dim → SAM-compatible 256 channels
        self.proj = nn.Sequential(
            nn.Conv2d(vit_dim, out_channels, kernel_size=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

        # Patch size used by the backbone
        self.patch_size = 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract image features.

        Args:
            x: ``(B, 3, H, W)`` — images, already resized to ``image_size``.

        Returns:
            ``(B, 256, H//patch, W//patch)`` feature map.
        """
        # Resize to the backbone's expected 224×224
        x224 = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # Forward through timm ViT — returns (B, N+1, C) where N=patch tokens,
        # +1 is the CLS token.  Drop CLS token.
        tokens = self.backbone.forward_features(x224)    # (B, N+1, C)
        # timm ViT returns CLS at position 0
        patch_tokens = tokens[:, 1:, :]                  # (B, N, C)

        side = int(patch_tokens.shape[1] ** 0.5)
        B, N, C = patch_tokens.shape
        feat = patch_tokens.permute(0, 2, 1).reshape(B, C, side, side)

        # Project to out_channels (256)
        feat = self.proj(feat)                           # (B, 256, side, side)

        # Upsample back to H//16, W//16 of the original (SAM-style output)
        target_h = x.shape[2] // self.patch_size
        target_w = x.shape[3] // self.patch_size
        if feat.shape[2] != target_h or feat.shape[3] != target_w:
            feat = F.interpolate(feat, size=(target_h, target_w), mode="bilinear", align_corners=False)

        return feat


# ──────────────────────────────────────────────────────────────────────────────
# Dummy prompt encoder shim (for fallback path without segment_anything)
# ──────────────────────────────────────────────────────────────────────────────

class _DummyPromptEncoder(nn.Module):
    """Minimal prompt encoder shim used in the timm fallback path.

    LearnedGridPrompts.forward() needs a ``prompt_encoder`` argument that at
    minimum has no ``pe_layer`` attribute (so the fallback sinusoidal embedding
    is used).
    """

    def __init__(self) -> None:
        super().__init__()


# ──────────────────────────────────────────────────────────────────────────────
# Main detector
# ──────────────────────────────────────────────────────────────────────────────

class SamDetector(nn.Module):
    """SAM ViT-B repurposed as a prompt-free object detector.

    The model accepts batched images and returns detection outputs that match
    the interface produced by GroundingDINOVisual:

    .. code-block:: python

        {
            "pred_logits": Tensor(B, Q, num_classes),   # raw logits
            "pred_boxes":  Tensor(B, Q, 4),             # cx/cy/w/h in [0,1]
            "aux_outputs": []
        }

    Args:
        cfg: OmegaConf DictConfig node (the ``model`` sub-tree from
             ``configs/sam.yaml``).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.num_queries  = int(cfg.num_queries)
        self.num_classes  = int(cfg.num_classes)
        self.embed_dim    = int(cfg.embed_dim)
        self.image_size   = int(cfg.image_size)
        self.freeze_image_encoder = bool(cfg.freeze_image_encoder)
        self._using_sam   = False          # set to True if segment_anything loads OK

        weights_path: str = str(cfg.weights_path)

        # ── 1. Image encoder ─────────────────────────────────────────────────
        self.image_encoder, self._prompt_encoder_for_embed = (
            self._build_image_encoder(weights_path)
        )

        if self.freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad_(False)

        # ── 2. Learned prompt grid ────────────────────────────────────────────
        self.learned_prompts = LearnedGridPrompts(
            num_queries=self.num_queries,
            embed_dim=self.embed_dim,
        )

        # ── 3. Inline mask decoder ────────────────────────────────────────────
        tc = cfg.transformer
        self.mask_decoder = SimpleMaskDecoder(
            embed_dim=self.embed_dim,
            nhead=int(tc.nhead),
            dim_feedforward=int(tc.dim_feedforward),
            dropout=float(tc.dropout),
        )

        # ── 4. Classification head ────────────────────────────────────────────
        self.class_head = ClassificationHead(
            embed_dim=self.embed_dim,
            num_classes=self.num_classes,
        )

    # ------------------------------------------------------------------ #
    # Construction helpers                                                 #
    # ------------------------------------------------------------------ #

    def _build_image_encoder(
        self, weights_path: str
    ) -> tuple[nn.Module, nn.Module]:
        """Try to load the SAM ViT-B image encoder.

        Precedence:
        1. ``segment_anything`` package + weights file  → full SAM encoder.
        2. ``segment_anything`` not installed           → timm fallback (warns).

        Returns:
            (image_encoder, prompt_encoder_or_shim)
        """
        # Check weights file first — raise early before touching segment_anything.
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"SAM weights not found at {weights_path}. "
                "Download sam_vit_b_01ec64.pth from the SAM repo and place it "
                "at models/pretrained/sam_vit_b.pth"
            )

        try:
            from segment_anything import sam_model_registry  # lazy import

            sam = sam_model_registry["vit_b"](checkpoint=weights_path)
            self._using_sam = True
            logger.info("SamDetector: loaded SAM ViT-B from %s", weights_path)
            return sam.image_encoder, sam.prompt_encoder

        except ImportError:
            logger.warning(
                "SamDetector: 'segment_anything' package not found. "
                "Falling back to timm ViT-B (SAM weights NOT loaded). "
                "Install segment_anything for full SAM support."
            )
            encoder = _TimmFallbackEncoder(
                image_size=self.image_size,
                out_channels=self.embed_dim,
            )
            return encoder, _DummyPromptEncoder()

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        images: torch.Tensor,
        masks: Optional[torch.Tensor] = None,  # noqa: ARG002 — ignored, kept for interface compat
    ) -> dict[str, torch.Tensor]:
        """Run SAM-based detection.

        Args:
            images: ``(B, 3, H, W)`` — input images.  Resized to
                    ``self.image_size`` internally.
            masks:  Ignored.  SAM handles padding internally.  Kept for
                    interface compatibility with GroundingDINOVisual.

        Returns:
            A dict with keys ``pred_logits``, ``pred_boxes``, ``aux_outputs``.
        """
        B = images.shape[0]

        # ── Resize to SAM's native resolution ────────────────────────────────
        if images.shape[2] != self.image_size or images.shape[3] != self.image_size:
            images_resized = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        else:
            images_resized = images

        # ── 1. Image encoder ─────────────────────────────────────────────────
        if self.freeze_image_encoder:
            with torch.no_grad():
                image_embeddings = self.image_encoder(images_resized)
        else:
            image_embeddings = self.image_encoder(images_resized)

        # Ensure (B, C, H, W) — SAM's image encoder already outputs this shape;
        # the timm fallback also produces this.
        if image_embeddings.dim() == 3:
            # Some ViT variants return (B, N, C) — reshape to 2D spatial.
            BN, N, C = image_embeddings.shape
            side = int(N ** 0.5)
            image_embeddings = image_embeddings.permute(0, 2, 1).reshape(BN, C, side, side)

        _, C, H_feat, W_feat = image_embeddings.shape

        # ── 2. Learned prompts ────────────────────────────────────────────────
        # sparse_embeddings: (Q, 1, embed_dim)
        sparse_embeddings = self.learned_prompts(self._prompt_encoder_for_embed)

        # ── 3. Mask decoder ───────────────────────────────────────────────────
        # iou_tokens:  (B, Q, embed_dim)
        # mask_logits: (B, Q, H_feat, W_feat)
        iou_tokens, mask_logits = self.mask_decoder(image_embeddings, sparse_embeddings)

        # ── 4. Classification logits ──────────────────────────────────────────
        pred_logits = self.class_head(iou_tokens)           # (B, Q, num_classes)

        # ── 5. Bounding boxes from masks ──────────────────────────────────────
        pred_boxes = masks_to_boxes_normalized(
            mask_logits, img_h=H_feat, img_w=W_feat
        )                                                   # (B, Q, 4)

        return {
            "pred_logits": pred_logits,
            "pred_boxes":  pred_boxes,
            "aux_outputs": [],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_model(cfg: DictConfig) -> SamDetector:
    """Construct a SamDetector from the top-level OmegaConf config.

    Args:
        cfg: The full config (``cfg.model`` is the relevant sub-tree).

    Returns:
        An initialised SamDetector instance.
    """
    return SamDetector(cfg.model)
