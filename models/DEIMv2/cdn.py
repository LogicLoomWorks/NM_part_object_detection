"""
models/deimv2/cdn.py

Contrastive Denoising (CDN) training module for DEIMv2.

During training, CDN prepends (num_cdn_groups * 2 * num_known) noisy
"denoising queries" ahead of the regular learned queries in the decoder.
Each CDN group contains:
  - A positive sub-group: boxes + labels with small Gaussian noise
  - A negative sub-group: same boxes but labels randomly flipped

An attention mask is constructed so that:
  - DN queries within the same group can attend to each other.
  - DN queries CANNOT attend to queries from other groups or to real queries.
  - Real queries CANNOT attend to any DN queries.

During inference (training=False) or when targets is None, prepare() returns
four Nones and postprocess() returns the output unchanged.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ContrastiveDenoisingTraining(nn.Module):
    """Contrastive denoising helper.

    This module holds no trainable parameters of its own; the label
    embedding (label_enc) is owned by DEIMv2Visual and passed in at
    call time so that the two share the same embedding table.
    """

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # prepare
    # ------------------------------------------------------------------

    def prepare(
        self,
        targets: list[dict],
        num_queries: int,
        d_model: int,
        label_enc: nn.Embedding,
        cfg,
    ):
        """Build denoising queries and the associated attention mask.

        Args:
            targets:     list of B target dicts, each with keys
                         "labels" (Long[N_i]) and "boxes" (Float[N_i, 4])
                         in cx/cy/w/h [0,1] format.
            num_queries: number of regular (non-DN) queries.
            d_model:     embedding dimension.
            label_enc:   shared nn.Embedding(num_classes+1, d_model).
            cfg:         model config with fields:
                           num_cdn_groups        (int)
                           cdn_label_noise_ratio (float)
                           cdn_box_noise_scale   (float)
                           num_classes           (int)

        Returns:
            dn_tgt      : (num_dn, B, d_model)  or None
            dn_ref_pts  : (num_dn, B, 4)         or None  — sigmoid space [0,1]
            attn_mask   : (num_dn + num_queries,
                           num_dn + num_queries)  or None  — bool, True = block
            dn_meta     : dict with "group_size" and "num_dn_queries"  or None
        """
        if targets is None or not self.training:
            return None, None, None, None

        num_groups        = int(cfg.num_cdn_groups)
        label_noise_ratio = float(cfg.cdn_label_noise_ratio)
        box_noise_scale   = float(cfg.cdn_box_noise_scale)
        num_classes       = int(cfg.num_classes)

        device = label_enc.weight.device

        # Collect all GT across the batch to find the maximum per image.
        batch_size = len(targets)
        num_known_list = [t["labels"].shape[0] for t in targets]
        max_known = max(num_known_list) if num_known_list else 0

        if max_known == 0:
            return None, None, None, None

        # group_size = num_groups * 2 (positive + negative)
        group_size   = num_groups * 2
        # Total DN slots per image (padded to max_known across batch)
        num_dn       = group_size * max_known

        # ------------------------------------------------------------------
        # Build padded label / box tensors: (B, group_size * max_known)
        # ------------------------------------------------------------------
        dn_labels = torch.zeros(batch_size, num_dn, dtype=torch.long, device=device)
        dn_boxes  = torch.zeros(batch_size, num_dn, 4, device=device)

        for bi, tgt in enumerate(targets):
            labels = tgt["labels"]          # (N_i,)
            boxes  = tgt["boxes"].float()   # (N_i, 4)
            N      = labels.shape[0]
            if N == 0:
                continue

            # Repeat labels/boxes for each group
            rep_labels = labels.repeat(group_size)               # (group_size*N,)
            rep_boxes  = boxes.repeat(group_size, 1)              # (group_size*N, 4)

            # Flip labels for negative groups (groups num_groups..2*num_groups-1)
            neg_start = num_groups * N
            flip_mask = torch.rand(rep_labels[neg_start:].shape, device=device)
            flipped   = torch.randint_like(
                rep_labels[neg_start:], low=0, high=num_classes
            )
            flip_where = flip_mask < label_noise_ratio
            rep_labels[neg_start:] = torch.where(
                flip_where, flipped, rep_labels[neg_start:]
            )

            # Add label noise to positive groups as well
            pos_flip_mask = torch.rand(rep_labels[:neg_start].shape, device=device)
            pos_flipped   = torch.randint_like(
                rep_labels[:neg_start], low=0, high=num_classes
            )
            rep_labels[:neg_start] = torch.where(
                pos_flip_mask < label_noise_ratio,
                pos_flipped,
                rep_labels[:neg_start],
            )

            # Add box noise (Gaussian, then clamp)
            box_noise = (
                torch.randn_like(rep_boxes) * box_noise_scale * 0.5
            )
            rep_boxes = (rep_boxes + box_noise).clamp(0.0, 1.0)

            # Pack into the padded tensor (interleaved by group × known)
            # Layout: [g0_k0, g0_k1, …, g0_kN-1, g1_k0, …]
            # We fill slot [bi, g*max_known : g*max_known+N] for each group g
            for g in range(group_size):
                dn_labels[bi, g * max_known: g * max_known + N] = rep_labels[g * N: (g + 1) * N]
                dn_boxes[bi, g * max_known: g * max_known + N]  = rep_boxes[g * N: (g + 1) * N]

        # ------------------------------------------------------------------
        # Embed labels → content embeddings  (B, num_dn, d_model)
        # ------------------------------------------------------------------
        # Clamp to valid embedding indices
        dn_labels_clamped = dn_labels.clamp(0, num_classes - 1)
        dn_tgt = label_enc(dn_labels_clamped)           # (B, num_dn, d_model)
        dn_tgt = dn_tgt.permute(1, 0, 2)               # (num_dn, B, d_model)

        # Reference points = noisy boxes in [0,1]
        dn_ref_pts = dn_boxes.permute(1, 0, 2)         # (num_dn, B, 4)

        # ------------------------------------------------------------------
        # Attention mask  (num_dn + num_queries, num_dn + num_queries)
        # True = block that attention edge
        # ------------------------------------------------------------------
        total = num_dn + num_queries
        attn_mask = torch.ones(total, total, dtype=torch.bool, device=device)

        # DN queries can attend to queries within the same group only.
        # Group g occupies slice [g*max_known : (g+1)*max_known].
        for g in range(group_size):
            s = g * max_known
            e = (g + 1) * max_known
            attn_mask[s:e, s:e] = False          # same group — allow

        # Real queries can attend to each other (bottom-right block).
        attn_mask[num_dn:, num_dn:] = False

        # Real queries CANNOT attend to DN queries → already True above.
        # DN queries CANNOT attend to real queries → already True above.

        dn_meta = {
            "group_size":     group_size,
            "num_dn_queries": num_dn,
            "max_known":      max_known,
        }

        return dn_tgt, dn_ref_pts, attn_mask, dn_meta

    # ------------------------------------------------------------------
    # postprocess
    # ------------------------------------------------------------------

    def postprocess(
        self,
        outputs: dict,
        dn_meta: Optional[dict],
    ) -> tuple[dict, Optional[dict]]:
        """Split denoising outputs from regular outputs.

        Args:
            outputs: dict with "pred_logits" (B, num_dn+Q, C) and
                     "pred_boxes" (B, num_dn+Q, 4), plus optional
                     "aux_outputs".
            dn_meta: dict returned by prepare(), or None.

        Returns:
            (clean_outputs, dn_outputs)
            clean_outputs has pred_logits/pred_boxes for the Q regular queries.
            dn_outputs    has pred_logits/pred_boxes for the num_dn DN queries,
                          or None if dn_meta is None.
        """
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
            dn_aux    = []
            for aux in outputs["aux_outputs"]:
                c, d = _split(aux["pred_logits"], aux["pred_boxes"])
                clean_aux.append(c)
                dn_aux.append(d)
            clean["aux_outputs"]  = clean_aux
            dn_out["aux_outputs"] = dn_aux

        return clean, dn_out
