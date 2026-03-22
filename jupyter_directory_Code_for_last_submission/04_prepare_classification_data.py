"""
04_prepare_classification_data.py — Build image classifier dataset for product classes.

FIXES:
  - Uses metadata.json to map product barcode folders → COCO category IDs
  - aug_product_images path corrected to data/aug_product_images

Sources (in priority order):
  1. COCO annotations  → crop each annotated bbox from shelf images
  2. product_images/   → per-product reference images (folder name = barcode)
  3. aug_product_images/ → augmented per-product images (folder name = barcode)

Output layout:
  data/classifier/
    train/
      0/   ← images of classifier class 0
      ...
    val/
      0/
      ...
    classifier_idx_to_category_id.json  ← maps classifier index → COCO category_id

Usage:
    python 04_prepare_classification_data.py
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2

# ── Config ────────────────────────────────────────────────────────────────────
COCO_DIR    = Path("data/raw/coco_dataset/train")
ANN_FILE    = COCO_DIR / "annotations.json"
COCO_IMGS   = COCO_DIR / "images"

PROD_DIR    = Path("data/raw/product_images")
METADATA    = PROD_DIR / "metadata.json"
AUG_DIR     = Path("data/aug_product_images")

OUT_DIR     = Path("data/classifier")
VAL_FRAC    = 0.15
SEED        = 42
MIN_CROP_PX = 20

random.seed(SEED)


def copy_file(src: Path, dst: Path) -> None:
    """Copy a file using pathlib (no shutil needed)."""
    if not dst.exists():
        dst.write_bytes(src.read_bytes())


# ── Load COCO annotations ─────────────────────────────────────────────────────
print("=" * 60)
print("  PREPARE CLASSIFIER DATA")
print("=" * 60)

data        = json.loads(ANN_FILE.read_text(encoding="utf-8"))
images      = data["images"]
annotations = data["annotations"]
categories  = sorted(data["categories"], key=lambda c: c["id"])

cat_id_to_name = {c["id"]: c["name"] for c in categories}
cat_name_to_id = {c["name"].strip(): c["id"] for c in categories}

cat_ids_sorted = [c["id"] for c in categories]
cat_id_to_cls  = {cid: i for i, cid in enumerate(cat_ids_sorted)}
cls_to_cat_id  = {str(i): cid for i, cid in enumerate(cat_ids_sorted)}
n_classes      = len(cat_ids_sorted)

print(f"  {n_classes} classes (COCO cat IDs {cat_ids_sorted[0]}-{cat_ids_sorted[-1]})")

# Create output dirs
for split in ("train", "val"):
    for cls_idx in range(n_classes):
        (OUT_DIR / split / str(cls_idx)).mkdir(parents=True, exist_ok=True)


# ── Build barcode → category_id mapping from metadata.json ────────────────────
print("\n  Building barcode -> category_id mapping from metadata.json ...")

barcode_to_catid = {}

if METADATA.exists():
    meta = json.loads(METADATA.read_text(encoding="utf-8"))
    products = meta.get("products", [])
    missing  = meta.get("missing", [])
    all_products = products + missing

    matched_count = 0
    unmatched_names = []

    for p in all_products:
        barcode = p.get("product_code", "").strip()
        pname   = p.get("product_name", "").strip()
        if not barcode or not pname:
            continue

        cid = cat_name_to_id.get(pname)
        if cid is not None:
            barcode_to_catid[barcode] = cid
            matched_count += 1
        else:
            # Try case-insensitive match
            for cat_name, cat_id in cat_name_to_id.items():
                if pname.lower() == cat_name.lower():
                    barcode_to_catid[barcode] = cat_id
                    matched_count += 1
                    break
            else:
                unmatched_names.append(pname)

    print(f"  Metadata products: {len(all_products)}")
    print(f"  Matched to COCO categories: {matched_count}")
    if unmatched_names:
        print(f"  Unmatched product names ({len(unmatched_names)}): {unmatched_names[:5]}...")
else:
    print(f"  WARNING: {METADATA} not found — product images won't be mapped!")


# ── SOURCE 1: COCO crops ──────────────────────────────────────────────────────
print("\nSource 1: COCO bbox crops")

id_to_img  = {im["id"]: im for im in images}
ann_by_img = defaultdict(list)
for a in annotations:
    ann_by_img[a["image_id"]].append(a)

crop_counts   = defaultdict(int)
skipped_crops = 0

for img_id, anns in ann_by_img.items():
    im    = id_to_img[img_id]
    fname = im["file_name"]
    src   = COCO_IMGS / fname
    if not src.exists():
        stem = Path(fname).stem
        alts = list(COCO_IMGS.glob(f"{stem}.*"))
        src  = alts[0] if alts else None
    if src is None:
        continue

    bgr = cv2.imread(str(src))
    if bgr is None:
        continue
    H, W = bgr.shape[:2]

    for a in anns:
        cid = a["category_id"]
        if cid not in cat_id_to_cls:
            continue
        cls_idx = cat_id_to_cls[cid]

        bx, by, bw, bh = a["bbox"]
        x1 = max(0, int(bx))
        y1 = max(0, int(by))
        x2 = min(W, int(bx + bw))
        y2 = min(H, int(by + bh))

        if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
            skipped_crops += 1
            continue

        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        split    = "val" if a["id"] % 100 < int(VAL_FRAC * 100) else "train"
        out_path = OUT_DIR / split / str(cls_idx) / f"coco_{img_id}_{a['id']}.jpg"
        if not out_path.exists():
            cv2.imwrite(str(out_path), crop)
        crop_counts[cls_idx] += 1

total_coco_crops = sum(crop_counts.values())
print(f"  Extracted {total_coco_crops} crops ({skipped_crops} skipped as too small)")


# ── SOURCE 2: product_images/ ─────────────────────────────────────────────────
print("\nSource 2: product_images/")

prod_matched     = 0
prod_unmatched   = 0
prod_imgs_copied = 0

if not PROD_DIR.exists():
    print(f"  WARNING: {PROD_DIR} not found — skipping")
else:
    subdirs = [d for d in PROD_DIR.iterdir() if d.is_dir()]
    print(f"  Found {len(subdirs)} product folders")

    for d in subdirs:
        barcode = d.name.strip()

        matched_cid = barcode_to_catid.get(barcode)
        if matched_cid is None:
            prod_unmatched += 1
            continue

        prod_matched += 1
        cls_idx = cat_id_to_cls[matched_cid]

        img_files = (
            list(d.glob("*.jpg")) +
            list(d.glob("*.jpeg")) +
            list(d.glob("*.png"))
        )
        random.shuffle(img_files)
        n_val_imgs = max(1, int(len(img_files) * VAL_FRAC))

        for i, img_f in enumerate(img_files):
            split = "val" if i < n_val_imgs else "train"
            dst   = OUT_DIR / split / str(cls_idx) / f"prod_{barcode}_{img_f.name}"
            copy_file(img_f, dst)
            prod_imgs_copied += 1

    print(f"  Matched   : {prod_matched} / {len(subdirs)} folders")
    print(f"  Unmatched : {prod_unmatched}")
    print(f"  Copied    : {prod_imgs_copied} images")


# ── SOURCE 3: aug_product_images/ ────────────────────────────────────────────
print("\nSource 3: aug_product_images/")

aug_matched     = 0
aug_unmatched   = 0
aug_imgs_copied = 0

if not AUG_DIR.exists():
    print(f"  WARNING: {AUG_DIR} not found — skipping")
else:
    aug_subdirs = [d for d in AUG_DIR.iterdir() if d.is_dir()]
    print(f"  Found {len(aug_subdirs)} aug folders")

    if not aug_subdirs:
        print("  No subfolders found")
    else:
        for d in aug_subdirs:
            barcode = d.name.strip()

            matched_cid = barcode_to_catid.get(barcode)
            if matched_cid is None:
                aug_unmatched += 1
                continue

            aug_matched += 1
            cls_idx = cat_id_to_cls[matched_cid]

            img_files = (
                list(d.glob("*.jpg")) +
                list(d.glob("*.jpeg")) +
                list(d.glob("*.png"))
            )
            random.shuffle(img_files)
            n_val_imgs = max(1, int(len(img_files) * VAL_FRAC))

            for i, img_f in enumerate(img_files):
                split = "val" if i < n_val_imgs else "train"
                dst   = OUT_DIR / split / str(cls_idx) / f"aug_{barcode}_{img_f.name}"
                copy_file(img_f, dst)
                aug_imgs_copied += 1

        print(f"  Matched   : {aug_matched} / {len(aug_subdirs)} folders")
        print(f"  Unmatched : {aug_unmatched}")
        print(f"  Copied    : {aug_imgs_copied} images")


# ── Write classifier_idx_to_category_id.json ─────────────────────────────────
out_map = OUT_DIR / "classifier_idx_to_category_id.json"
out_map.write_text(json.dumps(cls_to_cat_id, indent=2), encoding="utf-8")
print(f"\n  Wrote {out_map}")

# ── Final stats ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  CLASSIFIER DATASET SUMMARY")
print("=" * 60)

n_train_imgs = sum(
    len(list((OUT_DIR / "train" / str(i)).iterdir()))
    for i in range(n_classes)
    if (OUT_DIR / "train" / str(i)).exists()
)
n_val_imgs = sum(
    len(list((OUT_DIR / "val" / str(i)).iterdir()))
    for i in range(n_classes)
    if (OUT_DIR / "val" / str(i)).exists()
)
non_empty_classes = sum(
    1 for i in range(n_classes)
    if (OUT_DIR / "train" / str(i)).exists()
    and len(list((OUT_DIR / "train" / str(i)).iterdir())) > 0
)

print(f"  Classes with >=1 train image : {non_empty_classes} / {n_classes}")
print(f"  Total train images           : {n_train_imgs}")
print(f"  Total val images             : {n_val_imgs}")
print(f"  COCO crops                   : {total_coco_crops}")
print(f"  Product images               : {prod_imgs_copied} ({prod_matched} folders)")
print(f"  Aug images                   : {aug_imgs_copied} ({aug_matched} folders)")
print(f"  Output dir                   : {OUT_DIR.resolve()}")
print(f"\nNext: python 05_train_classifier.py")