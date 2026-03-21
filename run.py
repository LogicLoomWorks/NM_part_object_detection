#!/usr/bin/env python3
"""Submission entry-point.

Usage:
    python run.py --input /path/to/images --output predictions.json

The script runs entirely offline.  All weights are loaded from paths
relative to this file:
    checkpoints/deimv2_best.pt   — DEIMv2 detector checkpoint
    siglip_weights/              — SigLIP vision encoder weights (HF or timm format)
    gallery/                     — pre-computed product embedding gallery
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict, Tuple

import cv2
import numpy as np
import timm
import torch
import torchvision.ops as tv_ops
from safetensors.torch import load_file as safetensors_load

from models.DEIMv2.model import build_model, DEIMv2Visual

# ── Paths (all relative to this file) ────────────────────────────────────────

HERE        = Path(__file__).parent
CKPT_PATH   = HERE / "checkpoints" / "deimv2_best.pt"
SIGLIP_DIR  = HERE / "siglip_weights"
GALLERY_DIR = HERE / "gallery"

# ── DEIMv2 architecture config ────────────────────────────────────────────────
# Mirrors configs/deimv2.yaml as a plain SimpleNamespace (no third-party config lib needed).
# backbone.pretrained=False — weights come from checkpoint, no net download.

def _ns(d):
    """Recursively convert a nested dict to SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_ns(v) for v in d]
    return d


_DEIMV2_CFG = _ns({
    "model": {
        "name": "deimv2",
        "num_queries": 300,
        "num_cdn_groups": 5,
        "cdn_label_noise_ratio": 0.5,
        "cdn_box_noise_scale": 1.0,
        "aux_loss": True,
        "num_classes": 356,
        "backbone": {
            "name": "resnet50",
            "pretrained": False,   # loaded from checkpoint; no network download
            "freeze_at": 1,
            "out_indices": [2, 3, 4],
        },
        "neck": {
            "out_channels": 256,
            "num_levels": 4,
        },
        "transformer": {
            "d_model": 256,
            "nhead": 8,
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "dim_feedforward": 1024,
            "dropout": 0.0,
            "num_feature_levels": 4,
            "num_points": 4,
        },
    }
})

# ImageNet normalisation constants (match training transforms)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# SigLIP normalisation constants
_SIGLIP_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_SIGLIP_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_SIGLIP_SIZE = 224

# Detection hyper-parameters
_SCORE_THRESH = 0.05
_NMS_IOU      = 0.50
_TOP_K        = 300
_DEIMV2_SIZE  = 480   # longest side in pixels (matches training max_size)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_deimv2(device: torch.device) -> DEIMv2Visual:
    """Build DEIMv2 from inline config; load weights from checkpoint."""
    model = build_model(_DEIMV2_CFG)

    payload = torch.load(
        str(CKPT_PATH),
        map_location=device,
        weights_only=False,
    )
    if "state_dict" in payload:
        state = {
            k.removeprefix("model."): v
            for k, v in payload["state_dict"].items()
        }
    elif "model_state_dict" in payload:
        state = payload["model_state_dict"]
    else:
        state = payload

    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _remap_siglip_hf_to_timm(hf_sd: dict) -> dict:
    """Remap HuggingFace SiglipVisionModel keys → timm vit_base_patch16_siglip keys.

    HF layout (as saved by SiglipModel.vision_model.state_dict()):
      vision_model.embeddings.patch_embedding.*
      vision_model.embeddings.position_embedding.weight
      vision_model.encoder.layers.N.{layer_norm1,layer_norm2}.*
      vision_model.encoder.layers.N.self_attn.{q,k,v,out}_proj.*
      vision_model.encoder.layers.N.mlp.{fc1,fc2}.*
      vision_model.post_layernorm.*
      vision_model.head.probe
      vision_model.head.attention.{in_proj_weight,in_proj_bias,out_proj.*}
      vision_model.head.layernorm.*
      vision_model.head.mlp.*

    timm layout (vit_base_patch16_siglip_224, num_classes=0):
      pos_embed / patch_embed.proj.*
      blocks.N.{norm1,norm2}.*
      blocks.N.attn.qkv.*   (fused Q+K+V)
      blocks.N.attn.proj.*
      blocks.N.mlp.{fc1,fc2}.*
      norm.*
      attn_pool.{latent,q.*,kv.*,proj.*,norm.*,mlp.*}
    """
    # ── strip optional 'vision_model.' outer prefix ───────────────────────────
    sd = {}
    for k, v in hf_sd.items():
        sd[k[len("vision_model."):] if k.startswith("vision_model.") else k] = v

    out: dict = {}
    # temporary buffer for per-block QKV fusion
    _qkv_buf: dict = {}   # (block_idx, 'q'|'k'|'v', 'weight'|'bias') → tensor

    for k, v in sd.items():

        # ── patch embedding ───────────────────────────────────────────────────
        if k == "embeddings.patch_embedding.weight":
            out["patch_embed.proj.weight"] = v
        elif k == "embeddings.patch_embedding.bias":
            out["patch_embed.proj.bias"] = v

        # ── position embedding: HF is nn.Embedding (196, D) → (1, 196, D) ────
        elif k == "embeddings.position_embedding.weight":
            out["pos_embed"] = v.unsqueeze(0)

        # ── transformer blocks ────────────────────────────────────────────────
        elif k.startswith("encoder.layers."):
            rest = k[len("encoder.layers."):]          # "N.something"
            idx_str, tail = rest.split(".", 1)
            n = int(idx_str)

            if tail == "layer_norm1.weight":
                out[f"blocks.{n}.norm1.weight"] = v
            elif tail == "layer_norm1.bias":
                out[f"blocks.{n}.norm1.bias"] = v
            elif tail == "layer_norm2.weight":
                out[f"blocks.{n}.norm2.weight"] = v
            elif tail == "layer_norm2.bias":
                out[f"blocks.{n}.norm2.bias"] = v

            elif tail == "self_attn.out_proj.weight":
                out[f"blocks.{n}.attn.proj.weight"] = v
            elif tail == "self_attn.out_proj.bias":
                out[f"blocks.{n}.attn.proj.bias"] = v

            elif tail in ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                          "self_attn.v_proj.weight",
                          "self_attn.q_proj.bias",   "self_attn.k_proj.bias",
                          "self_attn.v_proj.bias"):
                # buffer for later fusion
                letter  = tail[10]              # 'q', 'k', or 'v'
                wb      = tail.split(".")[-1]   # 'weight' or 'bias'
                _qkv_buf[(n, letter, wb)] = v

            elif tail.startswith("mlp."):
                out[f"blocks.{n}.{tail}"] = v

            else:
                out[f"blocks.{n}.{tail}"] = v

        # ── final layer norm ──────────────────────────────────────────────────
        elif k == "post_layernorm.weight":
            out["norm.weight"] = v
        elif k == "post_layernorm.bias":
            out["norm.bias"] = v

        # ── attention pooling head ────────────────────────────────────────────
        elif k == "head.probe":
            out["attn_pool.latent"] = v                      # (1, 1, D)

        elif k == "head.attention.in_proj_weight":
            D = v.shape[0] // 3
            out["attn_pool.q.weight"]  = v[:D]               # W_q
            out["attn_pool.kv.weight"] = v[D:]               # W_k concat W_v

        elif k == "head.attention.in_proj_bias":
            D = v.shape[0] // 3
            out["attn_pool.q.bias"]  = v[:D]
            out["attn_pool.kv.bias"] = v[D:]

        elif k == "head.attention.out_proj.weight":
            out["attn_pool.proj.weight"] = v
        elif k == "head.attention.out_proj.bias":
            out["attn_pool.proj.bias"] = v

        elif k == "head.layernorm.weight":
            out["attn_pool.norm.weight"] = v
        elif k == "head.layernorm.bias":
            out["attn_pool.norm.bias"] = v

        elif k.startswith("head.mlp."):
            out["attn_pool." + k[len("head."):]] = v

        # ── anything else: pass through unchanged ─────────────────────────────
        else:
            out[k] = v

    # ── fuse per-block Q, K, V → qkv ─────────────────────────────────────────
    block_indices = {idx for (idx, _l, _wb) in _qkv_buf}
    for n in block_indices:
        for wb in ("weight", "bias"):
            q = _qkv_buf.get((n, "q", wb))
            k = _qkv_buf.get((n, "k", wb))
            v = _qkv_buf.get((n, "v", wb))
            if q is not None and k is not None and v is not None:
                out[f"blocks.{n}.attn.qkv.{wb}"] = torch.cat([q, k, v], dim=0)

    return out


def load_siglip(device: torch.device) -> torch.nn.Module:
    """Load SigLIP vision encoder from local safetensors weights.

    Handles two formats automatically:
      • HuggingFace keys (vision_model.* prefix) — remapped to timm layout
      • timm keys (patch_embed.*, blocks.*, attn_pool.*) — loaded directly
    """
    model = timm.create_model(
        "vit_base_patch16_siglip_224",
        pretrained=False,
        num_classes=0,
    ).to(device).eval()

    raw_sd = safetensors_load(
        str(SIGLIP_DIR / "vision_model.safetensors"),
        device=str(device),
    )

    # Auto-detect format by checking for a known HF key
    is_hf_format = any(
        k.startswith("vision_model.") or k.startswith("embeddings.")
        for k in raw_sd
    )

    if is_hf_format:
        print("  SigLIP weights: HF format detected — remapping keys …")
        mapped_sd = _remap_siglip_hf_to_timm(raw_sd)
    else:
        print("  SigLIP weights: timm format detected — loading directly …")
        mapped_sd = dict(raw_sd)

    # Verify before loading
    model_keys  = set(model.state_dict().keys())
    loaded_keys = set(mapped_sd.keys())
    missing  = model_keys - loaded_keys
    surplus  = loaded_keys - model_keys
    if missing:
        print(f"  WARNING: {len(missing)} keys missing from checkpoint: "
              f"{sorted(missing)[:5]} …")
    if surplus:
        print(f"  WARNING: {len(surplus)} unexpected keys in checkpoint: "
              f"{sorted(surplus)[:5]} …")

    model.load_state_dict(mapped_sd, strict=True)
    return model


def load_gallery(device: torch.device) -> Tuple[torch.Tensor, np.ndarray]:
    """Load pre-computed gallery embeddings and category IDs.

    Returns:
        gallery_emb  — (N, D) float32 L2-normalised tensor on device
        gallery_cats — (N,) int32 numpy array of category_id values
    """
    emb  = np.load(str(GALLERY_DIR / "gallery_embeddings.npy"))   # (N, D)
    cats = np.load(str(GALLERY_DIR / "gallery_category_ids.npy")) # (N,)
    return torch.from_numpy(emb).to(device), cats


# ── Image preprocessing ───────────────────────────────────────────────────────

def preprocess_for_deimv2(
    bgr: np.ndarray,
    max_size: int = _DEIMV2_SIZE,
) -> Tuple[torch.Tensor, float, int, int]:
    """Resize → pad to 32-multiple → normalise → tensor.

    Returns:
        tensor   — (1, 3, H_pad, W_pad) float32
        scale    — resize scale factor applied to the original image
        pad_h    — pixels of zero-padding added on the bottom
        pad_w    — pixels of zero-padding added on the right
    """
    h_orig, w_orig = bgr.shape[:2]
    scale = max_size / max(h_orig, w_orig)
    new_h = int(round(h_orig * scale))
    new_w = int(round(w_orig * scale))

    img = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Pad to next multiple of 32 (same as training val transforms)
    pad_h = (32 - new_h % 32) % 32
    pad_w = (32 - new_w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                                 cv2.BORDER_CONSTANT, value=0)

    img = img.astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)  # (1, 3, H, W)
    return tensor, scale, pad_h, pad_w


def preprocess_siglip(crops_bgr: List[np.ndarray]) -> torch.Tensor:
    """Resize crops to 224×224 and normalise for SigLIP.

    Args:
        crops_bgr: list of (H, W, 3) uint8 BGR arrays
    Returns:
        (N, 3, 224, 224) float32 tensor (CPU)
    """
    size = _SIGLIP_SIZE
    out = np.empty((len(crops_bgr), 3, size, size), dtype=np.float32)
    for i, crop in enumerate(crops_bgr):
        img = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - _SIGLIP_MEAN) / _SIGLIP_STD
        out[i] = img.transpose(2, 0, 1)
    return torch.from_numpy(out)


# ── Detection post-processing ─────────────────────────────────────────────────

def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """(N, 4) cx/cy/w/h → x1/y1/x2/y2."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2,
                        cx + w / 2, cy + h / 2], dim=-1)


def filter_and_nms(
    logits: torch.Tensor,   # (1, Q, C)
    boxes:  torch.Tensor,   # (1, Q, 4) cx/cy/w/h in [0,1]
    score_thresh: float = _SCORE_THRESH,
    iou_thresh:   float = _NMS_IOU,
    top_k:        int   = _TOP_K,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (scores, labels, boxes_cxcywh) after threshold + NMS."""
    prob = logits[0].sigmoid()               # (Q, C)
    scores, labels = prob.max(dim=-1)        # (Q,)
    keep = scores > score_thresh
    scores, labels, boxes = scores[keep], labels[keep], boxes[0][keep]

    if scores.numel() > top_k:
        idx = scores.topk(top_k).indices
        scores, labels, boxes = scores[idx], labels[idx], boxes[idx]

    if scores.numel() == 0:
        return scores, labels, boxes

    keep_nms = tv_ops.batched_nms(
        _cxcywh_to_xyxy(boxes), scores, labels, iou_thresh
    )
    return scores[keep_nms], labels[keep_nms], boxes[keep_nms]


# ── SigLIP embedding ──────────────────────────────────────────────────────────

@torch.no_grad()
def embed_crops(
    crops_bgr: List[np.ndarray],
    vision_model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    """Embed a list of BGR crops.  Returns (N, D) L2-normalised float32."""
    if not crops_bgr:
        return torch.zeros(0, 768, device=device)

    all_vecs: List[torch.Tensor] = []
    for i in range(0, len(crops_bgr), batch_size):
        batch_bgr = crops_bgr[i: i + batch_size]
        pixel_values = preprocess_siglip(batch_bgr).to(device)
        vecs = vision_model(pixel_values)                             # (B, D)
        vecs = vecs / vecs.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        all_vecs.append(vecs)

    return torch.cat(all_vecs, dim=0)  # (N, D)


# ── Gallery lookup ────────────────────────────────────────────────────────────

def gallery_lookup(
    crop_embeddings: torch.Tensor,   # (N, D) L2-normalised
    gallery_emb: torch.Tensor,       # (G, D) L2-normalised
    gallery_cats: np.ndarray,        # (G,) int32
) -> List[int]:
    """Return category_id for each crop via cosine nearest-neighbour."""
    if crop_embeddings.shape[0] == 0:
        return []
    sims = crop_embeddings @ gallery_emb.T     # (N, G)
    best = sims.argmax(dim=-1).cpu().numpy()   # (N,)
    return [int(gallery_cats[i]) for i in best]


# ── Main pipeline ─────────────────────────────────────────────────────────────

@torch.no_grad()
def process_images(
    image_paths: List[Path],
    deimv2: DEIMv2Visual,
    vision_model: torch.nn.Module,
    gallery_emb: torch.Tensor,
    gallery_cats: np.ndarray,
    device: torch.device,
) -> List[Dict]:
    """Run the full detection + classification pipeline."""
    results: List[Dict] = []
    use_amp = device.type == "cuda"

    for img_path in image_paths:
        # ── parse image_id from filename ──────────────────────────────────────
        try:
            image_id = int(img_path.stem)
        except ValueError:
            digits = "".join(c for c in img_path.stem if c.isdigit())
            image_id = int(digits) if digits else 0

        # ── load image ────────────────────────────────────────────────────────
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        orig_h, orig_w = bgr.shape[:2]

        # ── DEIMv2 forward ────────────────────────────────────────────────────
        tensor, scale, pad_h, pad_w = preprocess_for_deimv2(bgr)
        tensor   = tensor.to(device)
        input_h  = tensor.shape[2]    # new_h + pad_h
        input_w  = tensor.shape[3]    # new_w + pad_w

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = deimv2(tensor)

        scores, labels, boxes_norm = filter_and_nms(
            out["pred_logits"], out["pred_boxes"]
        )

        if scores.numel() == 0:
            continue

        # ── convert boxes: normalised padded-input → original pixels ──────────
        # box coords are [0,1] relative to the padded tensor; dividing by scale
        # maps them back to original image pixels.
        cx, cy, bw, bh = boxes_norm.cpu().float().unbind(-1)
        x1    = ((cx - bw / 2) * input_w / scale).clamp(0, orig_w)
        y1    = ((cy - bh / 2) * input_h / scale).clamp(0, orig_h)
        x2    = ((cx + bw / 2) * input_w / scale).clamp(0, orig_w)
        y2    = ((cy + bh / 2) * input_h / scale).clamp(0, orig_h)
        abs_w = (x2 - x1).clamp(min=0)
        abs_h = (y2 - y1).clamp(min=0)

        x1_np = x1.numpy()
        y1_np = y1.numpy()
        x2_np = x2.numpy()
        y2_np = y2.numpy()

        # ── crop original image for each detection ────────────────────────────
        crops:    List[np.ndarray] = []
        valid_idx: List[int]       = []
        for i in range(len(scores)):
            xi1, yi1 = int(x1_np[i]), int(y1_np[i])
            xi2, yi2 = int(x2_np[i]), int(y2_np[i])
            if xi2 <= xi1 or yi2 <= yi1:
                continue
            crop = bgr[yi1:yi2, xi1:xi2]
            if crop.size == 0:
                continue
            crops.append(crop)
            valid_idx.append(i)

        if not crops:
            continue

        # ── SigLIP classify crops ─────────────────────────────────────────────
        crop_emb = embed_crops(crops, vision_model, device)
        cat_ids  = gallery_lookup(crop_emb, gallery_emb, gallery_cats)

        # ── build prediction dicts ────────────────────────────────────────────
        scores_np = scores.cpu().numpy()
        for j, orig_i in enumerate(valid_idx):
            results.append({
                "image_id":    image_id,
                "category_id": cat_ids[j],
                "bbox": [
                    float(x1_np[orig_i]),
                    float(y1_np[orig_i]),
                    float(abs_w[orig_i].item()),
                    float(abs_h[orig_i].item()),
                ],
                "score": float(scores_np[orig_i]),
            })

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DEIMv2 + SigLIP product detection pipeline"
    )
    parser.add_argument("--input",  required=True,
                        help="Folder containing JPEG images")
    parser.add_argument("--output", required=True,
                        help="Path to write predictions.json")
    args = parser.parse_args()

    input_dir   = Path(args.input)
    output_path = Path(args.output)

    image_paths: List[Path] = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg"}
    )
    print(f"Found {len(image_paths)} images in {input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading DEIMv2 …")
    deimv2 = load_deimv2(device)

    print("Loading SigLIP …")
    vision_model = load_siglip(device)

    print("Loading gallery …")
    gallery_emb, gallery_cats = load_gallery(device)
    print(f"Gallery: {gallery_emb.shape[0]} products, "
          f"{gallery_emb.shape[1]}-dim embeddings")

    predictions = process_images(
        image_paths,
        deimv2,
        vision_model,
        gallery_emb,
        gallery_cats,
        device,
    )
    print(f"Total predictions: {len(predictions)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(predictions, indent=2),
        encoding="utf-8",
    )
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
