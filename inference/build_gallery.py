"""Build a SigLIP reference gallery for product classification.

Loads google/siglip-base-patch16-224, embeds up to 3 images per product,
averages the vectors, L2-normalises, then saves everything needed for
offline nearest-neighbour lookup at submission time.

Usage (network required, run once):
    python inference/build_gallery.py

Outputs are written to models/final/siglip/ and copied to prompt_data/siglip/.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file as safetensors_save
from transformers import SiglipModel, SiglipProcessor


# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT             = Path(__file__).parent.parent
PRODUCT_IMG_DIR  = ROOT / "data" / "raw" / "product_images"
ANNOTATIONS_FILE = ROOT / "data" / "raw" / "coco_dataset" / "train" / "annotations.json"
OUTPUT_DIR       = ROOT / "models" / "final" / "siglip"
PROMPT_DIR       = ROOT / "prompt_data" / "siglip"

MODEL_ID         = "google/siglip-base-patch16-224"
PREFERRED_VIEWS  = ["main.jpg", "front.jpg", "back.jpg"]


# ── Name normalisation ────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    """Strip to lowercase alphanumeric only — survives encoding corruption.

    Both 'FRØKRISP' and 'FR\ufffdKRISP' collapse to 'frkrisp', so mismatches
    caused by mojibake in the COCO annotations file are handled automatically.
    """
    name = name.replace("\ufffd", "")
    name = unicodedata.normalize("NFKD", name)
    return re.sub(r"[^a-z0-9]", "", name.lower())


# ── Data loading ──────────────────────────────────────────────────────────────

def load_metadata(metadata_path: Path) -> dict[str, str]:
    """Return {product_code: product_name} from metadata.json."""
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        entry["product_code"]: entry["product_name"]
        for entry in data.get("products", [])
    }


def build_norm_to_cat_id(annotations_path: Path) -> dict[str, int]:
    """Return {normalised_name: category_id} from COCO annotations.json.

    Tries UTF-8 first; falls back to latin-1 for files with legacy encoding.
    """
    try:
        text = annotations_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = annotations_path.read_text(encoding="latin-1")
    data = json.loads(text)
    return {_norm(cat["name"]): cat["id"] for cat in data["categories"]}


# ── Image selection ───────────────────────────────────────────────────────────

def select_images(product_dir: Path) -> list[Path]:
    """Return up to 3 image paths; prefer main → front → back, then any jpg."""
    chosen: list[Path] = []
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


# ── Embedding ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def embed_images(
    paths: list[Path],
    model: SiglipModel,
    processor: SiglipProcessor,
    device: torch.device,
) -> np.ndarray:
    """Embed images, average, L2-normalise → (embed_dim,) float32 numpy."""
    pil_images = [Image.open(p).convert("RGB") for p in paths]
    inputs = processor(images=pil_images, return_tensors="pt").to(device)
    outputs = model.vision_model(**inputs)
    vecs = outputs.pooler_output           # (N, embed_dim)
    mean_vec = vecs.mean(dim=0)            # (embed_dim,)
    mean_vec = mean_vec / mean_vec.norm(p=2).clamp(min=1e-12)
    return mean_vec.cpu().float().numpy()


# ── File copy (no shutil) ─────────────────────────────────────────────────────

def _copy(src: Path, dst: Path) -> None:
    dst.write_bytes(src.read_bytes())


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device          : {device}")
    print(f"Loading model   : {MODEL_ID}")

    processor: SiglipProcessor = SiglipProcessor.from_pretrained(MODEL_ID)
    model: SiglipModel = SiglipModel.from_pretrained(MODEL_ID).to(device).eval()

    print(f"Model loaded.\n")

    # ── load name tables ──────────────────────────────────────────────────────
    metadata       = load_metadata(PRODUCT_IMG_DIR / "metadata.json")
    norm_to_cat_id = build_norm_to_cat_id(ANNOTATIONS_FILE)

    # ── enumerate product folders ─────────────────────────────────────────────
    product_dirs = sorted(
        p for p in PRODUCT_IMG_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    print(f"Product directories found : {len(product_dirs)}")

    embeddings:    list[np.ndarray] = []
    category_ids:  list[int]        = []
    product_codes: list[str]        = []
    unmatched:     list[str]        = []
    skipped:       int              = 0

    for i, pdir in enumerate(product_dirs, 1):
        code      = pdir.name
        img_paths = select_images(pdir)

        if not img_paths:
            skipped += 1
            continue

        try:
            vec = embed_images(img_paths, model, processor, device)
        except Exception as exc:
            print(f"  [{i:>3}/{len(product_dirs)}] {code}: embed error — {exc}")
            skipped += 1
            continue

        # match product name → category_id
        product_name = metadata.get(code, "")
        cat_id       = norm_to_cat_id.get(_norm(product_name), -1)

        embeddings.append(vec)
        category_ids.append(cat_id)
        product_codes.append(code)

        if cat_id == -1:
            unmatched.append(f"{code} | {product_name!r}")

        if i % 50 == 0 or i == len(product_dirs):
            n_matched = sum(c != -1 for c in category_ids)
            print(f"  [{i:>3}/{len(product_dirs)}] embedded {len(embeddings)}, "
                  f"matched {n_matched}/{len(embeddings)}")

    if not embeddings:
        print("No embeddings produced — aborting.")
        return

    emb_array = np.stack(embeddings, axis=0).astype(np.float32)  # (N, D)
    cat_array = np.array(category_ids, dtype=np.int32)            # (N,)

    # ── save outputs ──────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    emb_path   = OUTPUT_DIR / "gallery_embeddings.npy"
    cat_path   = OUTPUT_DIR / "gallery_category_ids.npy"
    code_path  = OUTPUT_DIR / "gallery_product_codes.json"
    weights_path = OUTPUT_DIR / "vision_model.safetensors"
    config_path  = OUTPUT_DIR / "preprocessor_config.json"

    np.save(emb_path, emb_array)
    np.save(cat_path, cat_array)
    code_path.write_text(
        json.dumps(product_codes, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Vision encoder weights — move to CPU first
    vision_state = {k: v.cpu().contiguous() for k, v in model.vision_model.state_dict().items()}
    safetensors_save(vision_state, str(weights_path))

    # Preprocessor config — use to_dict() so we control the JSON serialisation
    config_dict = processor.image_processor.to_dict()
    config_path.write_text(
        json.dumps(config_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── copy data files to prompt_data/siglip/ ────────────────────────────────
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    for src in (emb_path, cat_path, code_path, config_path):
        _copy(src, PROMPT_DIR / src.name)

    # ── summary ───────────────────────────────────────────────────────────────
    n_total   = len(embeddings)
    n_matched = int((cat_array != -1).sum())

    print(f"\n{'=' * 60}")
    print(f"  Products embedded         : {n_total}")
    print(f"  Matched to category_id    : {n_matched} / {n_total}")
    print(f"  Unmatched (cat_id = -1)   : {n_total - n_matched}")
    print(f"  Skipped (no images/error) : {skipped}")
    print(f"\n  Output : {OUTPUT_DIR}")
    print(f"    gallery_embeddings.npy       "
          f"{emb_path.stat().st_size / 1e6:.2f} MB  shape={emb_array.shape}")
    print(f"    gallery_category_ids.npy     "
          f"{cat_path.stat().st_size / 1e3:.1f} KB")
    print(f"    gallery_product_codes.json   "
          f"{code_path.stat().st_size / 1e3:.1f} KB")
    print(f"    vision_model.safetensors     "
          f"{weights_path.stat().st_size / 1e6:.1f} MB")
    print(f"    preprocessor_config.json     "
          f"{config_path.stat().st_size / 1e3:.1f} KB")
    print(f"\n  Also copied to : {PROMPT_DIR}")

    if unmatched:
        print(f"\n  Unmatched products ({len(unmatched)}):")
        for entry in unmatched[:15]:
            print(f"    {entry}")
        if len(unmatched) > 15:
            print(f"    … and {len(unmatched) - 15} more")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
