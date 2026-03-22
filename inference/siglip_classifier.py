"""SigLIP-based product classifier.

Two public functions:
    build_gallery()     — embed all reference images and save to disk
    classify_crops()    — embed crop PIL images and return category_ids

For LOCAL use only (requires transformers + network for first download).
The submission sandbox uses timm + pre-computed .npy galleries instead.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import SiglipModel, SiglipProcessor

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT             = Path(__file__).parent.parent
PRODUCT_IMG_DIR  = ROOT / "data" / "raw" / "product_images"
ANNOTATIONS_FILE = ROOT / "data" / "raw" / "coco_dataset" / "train" / "annotations.json"
GALLERY_DIR      = ROOT / "models" / "final" / "siglip"
PROMPT_DIR       = ROOT / "prompt_data"

MODEL_ID        = "google/siglip-base-patch16-224"
PREFERRED_VIEWS = ["main.jpg", "front.jpg", "back.jpg"]

# ── Module-level cache (lazy-loaded) ─────────────────────────────────────────

_model:      Optional[SiglipModel]     = None
_processor:  Optional[SiglipProcessor] = None
_gallery_emb_np:  Optional[np.ndarray] = None   # (N, D) float32, L2-normalised
_gallery_cats_np: Optional[np.ndarray] = None   # (N,) int32


# ── Internal helpers ──────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    """Normalise a product name to lowercase alphanumeric for fuzzy matching.

    Strips Norwegian/special characters and encoding artefacts so that
    'FRØKRISP' and 'FR\ufffdKRISP' both collapse to 'frkrisp'.
    """
    name = name.replace("\ufffd", "")
    name = unicodedata.normalize("NFKD", name)
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _load_metadata() -> dict[str, str]:
    """Return {product_code: product_name} from metadata.json."""
    data = json.loads(
        (PRODUCT_IMG_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    return {entry["product_code"]: entry["product_name"]
            for entry in data.get("products", [])}


def _build_norm_to_cat_id() -> dict[str, int]:
    """Return {normalised_name: category_id} from COCO annotations.json.

    Tries UTF-8 first; falls back to latin-1 for files with legacy encoding.
    """
    try:
        text = ANNOTATIONS_FILE.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = ANNOTATIONS_FILE.read_text(encoding="latin-1")
    cats = json.loads(text)["categories"]
    return {_norm(c["name"]): c["id"] for c in cats}


def _select_images(product_dir: Path) -> List[Path]:
    """Return up to 3 image paths; prefer main → front → back, then any jpg."""
    chosen: List[Path] = []
    for fname in PREFERRED_VIEWS:
        p = product_dir / fname
        if p.exists():
            chosen.append(p)
        if len(chosen) == 3:
            return chosen
    for p in sorted(product_dir.glob("*.jpg")):
        if p not in chosen:
            chosen.append(p)
        if len(chosen) == 3:
            break
    return chosen


def _get_model(device: torch.device) -> tuple[SiglipModel, SiglipProcessor]:
    """Lazy-load and cache the SigLIP model + processor."""
    global _model, _processor
    if _model is None:
        print(f"  Loading SigLIP: {MODEL_ID}")
        _processor = SiglipProcessor.from_pretrained(MODEL_ID)
        _model = SiglipModel.from_pretrained(MODEL_ID).eval()
    # Move to requested device (no-op if already there)
    _model = _model.to(device)
    return _model, _processor


@torch.no_grad()
def _embed_pil_images(
    pil_images: List[Image.Image],
    model: SiglipModel,
    processor: SiglipProcessor,
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    """Embed a list of PIL images; returns (N, D) L2-normalised float32."""
    all_vecs: List[torch.Tensor] = []
    for i in range(0, len(pil_images), batch_size):
        batch  = pil_images[i: i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        out    = model.vision_model(**inputs)
        vecs   = out.pooler_output.float()              # (B, D)
        vecs   = vecs / vecs.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        all_vecs.append(vecs)
    return torch.cat(all_vecs, dim=0)                   # (N, D)


# ── Public API ────────────────────────────────────────────────────────────────

def build_gallery(device: Optional[str] = None) -> dict:
    """Build and save the SigLIP reference gallery.

    For each product folder in data/raw/product_images/:
      - Loads up to 3 images (main, front, back)
      - Embeds them with the SigLIP vision encoder
      - Averages the per-image vectors → one L2-normalised vector per product
      - Matches product name (via metadata.json) to COCO category_id (via annotations.json)

    Saves to models/final/siglip/:
      gallery_embeddings.npy     — (N, embed_dim) float32, L2-normalised
      gallery_category_ids.npy   — (N,) int32  (-1 for unmatched products)
      gallery_product_codes.json — list of product_code strings (for debugging)

    Also copies the two .npy files to prompt_data/ for reference.

    Returns a summary dict.
    """
    _device = torch.device(
        device if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {_device}")

    model, processor = _get_model(_device)

    metadata       = _load_metadata()            # {product_code: product_name}
    norm_to_cat_id = _build_norm_to_cat_id()     # {normalised_name: category_id}

    product_dirs = sorted(
        p for p in PRODUCT_IMG_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    print(f"Found {len(product_dirs)} product directories\n")

    embeddings:    List[np.ndarray] = []
    category_ids:  List[int]        = []
    product_codes: List[str]        = []
    unmatched:     List[str]        = []
    skipped:       int              = 0

    for i, pdir in enumerate(product_dirs, 1):
        code      = pdir.name
        img_paths = _select_images(pdir)

        if not img_paths:
            skipped += 1
            continue

        try:
            pil_imgs = [Image.open(p).convert("RGB") for p in img_paths]
            vecs = _embed_pil_images(pil_imgs, model, processor, _device)
            mean = vecs.mean(dim=0)
            mean = mean / mean.norm(p=2).clamp(min=1e-12)
        except Exception as exc:
            print(f"  [{i:>3}/{len(product_dirs)}] {code}: embed error — {exc}")
            skipped += 1
            continue

        product_name = metadata.get(code, "")
        cat_id       = norm_to_cat_id.get(_norm(product_name), -1)

        embeddings.append(mean.cpu().numpy())
        category_ids.append(cat_id)
        product_codes.append(code)

        if cat_id == -1:
            unmatched.append(f"{code} | {product_name!r}")

        if i % 50 == 0 or i == len(product_dirs):
            n_matched = sum(c != -1 for c in category_ids)
            print(f"  [{i:>3}/{len(product_dirs)}]  "
                  f"embedded {len(embeddings)}, "
                  f"matched {n_matched}/{len(embeddings)}")

    if not embeddings:
        raise RuntimeError("No embeddings produced — check PRODUCT_IMG_DIR.")

    emb_array  = np.stack(embeddings, axis=0).astype(np.float32)  # (N, D)
    cats_array = np.array(category_ids, dtype=np.int32)            # (N,)

    # ── save to models/final/siglip/ ──────────────────────────────────────────
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    emb_path   = GALLERY_DIR / "gallery_embeddings.npy"
    cats_path  = GALLERY_DIR / "gallery_category_ids.npy"
    codes_path = GALLERY_DIR / "gallery_product_codes.json"

    np.save(str(emb_path),  emb_array)
    np.save(str(cats_path), cats_array)
    codes_path.write_text(
        json.dumps(product_codes, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── copy .npy files to prompt_data/ ──────────────────────────────────────
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    for src in (emb_path, cats_path):
        (PROMPT_DIR / src.name).write_bytes(src.read_bytes())

    # ── summary ───────────────────────────────────────────────────────────────
    n_total   = len(embeddings)
    n_matched = int((cats_array != -1).sum())
    embed_dim = emb_array.shape[1]

    print(f"\n{'=' * 60}")
    print(f"  Products embedded      : {n_total}")
    print(f"  Matched to category_id : {n_matched} / {n_total}")
    print(f"  Unmatched (cat_id=-1)  : {n_total - n_matched}")
    print(f"  Skipped (no images)    : {skipped}")
    print(f"  Embedding dimension    : {embed_dim}")
    print(f"\n  Saved → {GALLERY_DIR}")
    print(f"    gallery_embeddings.npy      "
          f"{emb_path.stat().st_size / 1e6:.2f} MB  shape={emb_array.shape}")
    print(f"    gallery_category_ids.npy    "
          f"{cats_path.stat().st_size / 1e3:.1f} KB")
    print(f"    gallery_product_codes.json  "
          f"{codes_path.stat().st_size / 1e3:.1f} KB")
    print(f"\n  Copied .npy → {PROMPT_DIR}")

    if unmatched:
        print(f"\n  Unmatched products ({len(unmatched)}):")
        for u in unmatched[:12]:
            print(f"    {u}")
        if len(unmatched) > 12:
            print(f"    … and {len(unmatched) - 12} more")
    print(f"{'=' * 60}")

    return {
        "n_products_embedded": n_total,
        "n_matched":           n_matched,
        "n_unmatched":         n_total - n_matched,
        "n_skipped":           skipped,
        "embed_dim":           embed_dim,
        "gallery_shape":       list(emb_array.shape),
        "gallery_dir":         str(GALLERY_DIR),
    }


@torch.no_grad()
def classify_crops(
    pil_images: List[Image.Image],
    device: Optional[str] = None,
    batch_size: int = 32,
) -> List[int]:
    """Classify cropped PIL images using the pre-computed gallery.

    Loads the gallery from models/final/siglip/ on the first call
    (subsequent calls reuse the cached numpy arrays).

    Args:
        pil_images:  list of RGB PIL.Image objects (one per detection crop)
        device:      torch device string, or None to auto-detect
        batch_size:  embedding batch size

    Returns:
        list of int category_id per crop (nearest-neighbour in cosine space).
        A value of -1 means the matched gallery entry had no COCO category
        (product was not found in annotations.json during gallery build).
    """
    global _gallery_emb_np, _gallery_cats_np

    if not pil_images:
        return []

    _device = torch.device(
        device if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ── lazy-load gallery ─────────────────────────────────────────────────────
    if _gallery_emb_np is None:
        emb_path  = GALLERY_DIR / "gallery_embeddings.npy"
        cats_path = GALLERY_DIR / "gallery_category_ids.npy"
        if not emb_path.exists():
            raise FileNotFoundError(
                f"Gallery not found at {GALLERY_DIR}. "
                "Run build_gallery() first."
            )
        _gallery_emb_np  = np.load(str(emb_path))
        _gallery_cats_np = np.load(str(cats_path))

    gallery_emb = torch.from_numpy(_gallery_emb_np).to(_device)  # (G, D)

    # ── embed crops ───────────────────────────────────────────────────────────
    model, processor = _get_model(_device)
    crop_emb = _embed_pil_images(pil_images, model, processor, _device,
                                 batch_size)                       # (N, D)

    # ── cosine nearest-neighbour (both sides L2-normalised → dot == cosine) ──
    sims = crop_emb @ gallery_emb.T                               # (N, G)
    best = sims.argmax(dim=-1).cpu().numpy()                      # (N,)

    return [int(_gallery_cats_np[i]) for i in best]
