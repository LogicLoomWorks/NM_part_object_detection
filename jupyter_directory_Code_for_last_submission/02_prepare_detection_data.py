"""
02_prepare_detection_data.py — Build YOLOv8 detection dataset from COCO + SKU110K + aug_product_images.

Output layout:
  data/yolo_detection/
    images/train/   ← COCO train images + SKU110K train images + aug_product_images (copied)
    images/val/     ← COCO val images + SKU110K val images + aug_product_images
    labels/train/   ← YOLO .txt label files
    labels/val/
    dataset.yaml
    idx_to_category_id.json   ← maps YOLO class index → COCO category_id

Rules:
  - COCO images: use their actual category indices (0..N-1)
  - SKU110K images: ALL boxes mapped to class index 0 (generic "product")
  - aug_product_images: Images from each product folder → bounding box covering entire image, 
    class index 0 (generic "product") — teaches detector to find product regions
    (augmented version of classifier/ data)
    If train has > 5000 images → subsample 5000 randomly (seed=42)

Usage:
    python 02_prepare_detection_data.py
"""
import json
import random
from collections import defaultdict
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
COCO_DIR       = Path("data/raw/coco_dataset/train")
ANN_FILE       = COCO_DIR / "annotations.json"
COCO_IMGS      = COCO_DIR / "images"

SKU_BASE       = Path("data/raw/extra_data/SKU110K_fixed")
SKU_ANN        = SKU_BASE / "annotations"
SKU_IMGS       = SKU_BASE / "images"

AUG_PROD_DATA  = Path("data/aug_product_images")   # ← augmented classifier data

OUT_DIR        = Path("data/yolo_detection")
VAL_FRAC       = 0.20
SEED           = 42
SKU_MAX_TRAIN  = 5000   # subsample if more
AUG_MAX_TRAIN  = 5000   # subsample if more

random.seed(SEED)


def copy_file(src: Path, dst: Path) -> None:
    """Copy a file using pathlib (no shutil needed)."""
    if not dst.exists():
        dst.write_bytes(src.read_bytes())


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — COCO  →  YOLO
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Converting COCO dataset")
print("=" * 60)

data        = json.loads(ANN_FILE.read_text(encoding="utf-8"))
images      = data["images"]
annotations = data["annotations"]
categories  = sorted(data["categories"], key=lambda c: c["id"])

cat_id_to_idx = {c["id"]: i for i, c in enumerate(categories)}
idx_to_cat_id = {str(i): c["id"] for i, c in enumerate(categories)}
nc    = len(categories)
names = [c["name"] for c in categories]

print(f"  {nc} categories, IDs {categories[0]['id']}–{categories[-1]['id']}")

# Train/val split (COCO)
img_ids = [im["id"] for im in images]
random.shuffle(img_ids)
n_val   = max(1, int(len(img_ids) * VAL_FRAC))
val_set = set(img_ids[:n_val])
print(f"  COCO split: {len(img_ids)-n_val} train / {n_val} val")

# Build annotation index
ann_by_img = defaultdict(list)
for a in annotations:
    ann_by_img[a["image_id"]].append(a)

id_to_img = {im["id"]: im for im in images}

# Create output dirs
for split in ("train", "val"):
    (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

skipped      = 0
total_labels = 0

for img_id in img_ids:
    im    = id_to_img[img_id]
    fname = im["file_name"]
    W, H  = im["width"], im["height"]
    split = "val" if img_id in val_set else "train"

    src = COCO_IMGS / fname
    if not src.exists():
        stem = Path(fname).stem
        alts = list(COCO_IMGS.glob(f"{stem}.*"))
        src  = alts[0] if alts else None
    if src is None:
        skipped += 1
        continue

    dst_img = OUT_DIR / "images" / split / src.name
    copy_file(src, dst_img)

    anns  = ann_by_img.get(img_id, [])
    lines = []
    for a in anns:
        bx, by, bw, bh = a["bbox"]
        if bw <= 0 or bh <= 0:
            continue
        cid = a["category_id"]
        if cid not in cat_id_to_idx:
            continue
        cls_idx = cat_id_to_idx[cid]
        cx = max(0.0, min(1.0, (bx + bw / 2) / W))
        cy = max(0.0, min(1.0, (by + bh / 2) / H))
        nw = max(1e-6, min(1.0, bw / W))
        nh = max(1e-6, min(1.0, bh / H))
        lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        total_labels += 1

    label_path = OUT_DIR / "labels" / split / (src.stem + ".txt")
    label_path.write_text("\n".join(lines), encoding="utf-8")

print(f"  COCO: {total_labels} labels written, {skipped} images skipped (missing)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — SKU110K  →  YOLO  (class 0 for all boxes)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("STEP 2: Converting SKU110K dataset")
print("=" * 60)


def parse_sku_csv(csv_path: Path):
    """Parse SKU110K CSV. Returns (boxes_by_img, img_dims)."""
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found — skipping")
        return {}, {}

    lines  = csv_path.read_text(encoding="utf-8").splitlines()
    header = lines[0].lower().split(",")
    print(f"  Columns: {header}")

    def col(names_list):
        for nm in names_list:
            if nm in header:
                return header.index(nm)
        return None

    i_name = col(["image_name", "name", "filename"])
    i_x1   = col(["x1"])
    i_y1   = col(["y1"])
    i_x2   = col(["x2"])
    i_y2   = col(["y2"])
    i_imgw = col(["image_width",  "width",  "img_width"])
    i_imgh = col(["image_height", "height", "img_height"])

    if None in (i_name, i_x1, i_y1, i_x2, i_y2):
        i_name, i_x1, i_y1, i_x2, i_y2 = 0, 1, 2, 3, 4
        i_imgw, i_imgh = 6, 7
        print("  Falling back to positional column indices")

    boxes_by_img = defaultdict(list)
    img_dims     = {}

    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) <= max(filter(lambda x: x is not None, [i_name, i_x1, i_y1, i_x2, i_y2])):
            continue
        try:
            img_name = parts[i_name].strip()
            x1 = float(parts[i_x1])
            y1 = float(parts[i_y1])
            x2 = float(parts[i_x2])
            y2 = float(parts[i_y2])
            boxes_by_img[img_name].append((x1, y1, x2, y2))
            if i_imgw and i_imgh and img_name not in img_dims:
                try:
                    img_dims[img_name] = (float(parts[i_imgw]), float(parts[i_imgh]))
                except (ValueError, IndexError):
                    pass
        except (ValueError, IndexError):
            continue

    return boxes_by_img, img_dims


def convert_sku_split(csv_path: Path, split: str, max_imgs: int = None):
    """Convert one SKU110K split to YOLO labels."""
    boxes_by_img, img_dims = parse_sku_csv(csv_path)
    img_names = list(boxes_by_img.keys())

    if not img_names:
        print(f"  No images found in {csv_path}")
        return 0

    if max_imgs and len(img_names) > max_imgs:
        random.shuffle(img_names)
        img_names = img_names[:max_imgs]
        print(f"  Subsampled {split} to {len(img_names)} images (from {len(boxes_by_img)})")
    else:
        print(f"  {split}: {len(img_names)} images")

    total_lbl = 0
    missing   = 0

    for img_name in img_names:
        img_path = SKU_IMGS / img_name
        if not img_path.exists():
            stem = Path(img_name).stem
            alts = list(SKU_IMGS.glob(f"{stem}.*"))
            img_path = alts[0] if alts else None
        if img_path is None:
            missing += 1
            continue

        dst_img = OUT_DIR / "images" / split / img_path.name
        copy_file(img_path, dst_img)

        if img_name in img_dims:
            W, H = img_dims[img_name]
        else:
            import cv2
            img = cv2.imread(str(img_path))
            if img is None:
                missing += 1
                continue
            H, W = img.shape[:2]

        lines = []
        for (x1, y1, x2, y2) in boxes_by_img[img_name]:
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0 or W <= 0 or H <= 0:
                continue
            cx = max(0.0, min(1.0, (x1 + bw / 2) / W))
            cy = max(0.0, min(1.0, (y1 + bh / 2) / H))
            nw = max(1e-6, min(1.0, bw / W))
            nh = max(1e-6, min(1.0, bh / H))
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            total_lbl += 1

        lbl_path = OUT_DIR / "labels" / split / (img_path.stem + ".txt")
        lbl_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"  {split}: {total_lbl} labels, {missing} images not found on disk")
    return len(img_names)


convert_sku_split(SKU_ANN / "annotations_train.csv", "train", max_imgs=SKU_MAX_TRAIN)
convert_sku_split(SKU_ANN / "annotations_val.csv",   "val",   max_imgs=None)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — aug_product_images  →  YOLO  (class 0, full-image bounding box)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("STEP 3: Converting aug_product_images (augmented classifier data)")
print("=" * 60)

if not AUG_PROD_DATA.exists():
    print(f"  WARNING: {AUG_PROD_DATA} not found — skipping")
else:
    # Collect all images from all product folders
    all_img_files = []
    for prod_folder in sorted(AUG_PROD_DATA.iterdir()):
        if prod_folder.is_dir():
            for img_file in prod_folder.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    all_img_files.append(img_file)

    print(f"  Total augmented product images found: {len(all_img_files)}")

    if not all_img_files:
        print(f"  No images found in {AUG_PROD_DATA}")
    else:
        # Split train/val FIRST
        random.shuffle(all_img_files)
        n_val_aug = max(1, int(len(all_img_files) * VAL_FRAC))
        
        all_val_imgs = all_img_files[:n_val_aug]
        all_train_imgs = all_img_files[n_val_aug:]

        # Subsample train if needed
        if len(all_train_imgs) > AUG_MAX_TRAIN:
            random.shuffle(all_train_imgs)
            all_train_imgs = all_train_imgs[:AUG_MAX_TRAIN]
            print(f"  Subsampled train to {len(all_train_imgs)} images (from {len(all_img_files) - n_val_aug})")
        else:
            print(f"  Split: {len(all_train_imgs)} train / {len(all_val_imgs)} val")

        total_aug_lbl = 0
        missing_aug = 0

        # Process ONLY train and val images (after subsampling)
        for split_name, img_list in [("train", all_train_imgs), ("val", all_val_imgs)]:
            for img_file in img_list:
                dst_img = OUT_DIR / "images" / split_name / img_file.name
                copy_file(img_file, dst_img)

                # Get image dimensions
                try:
                    import cv2
                    img = cv2.imread(str(img_file))
                    if img is None:
                        missing_aug += 1
                        continue
                    H, W = img.shape[:2]
                except Exception:
                    missing_aug += 1
                    continue

                # Create one label: full-image bounding box with class 0
                # Normalized coords: center=(0.5, 0.5), width=1.0, height=1.0
                label_line = "0 0.5 0.5 1.0 1.0"
                
                lbl_path = OUT_DIR / "labels" / split_name / (img_file.stem + ".txt")
                lbl_path.write_text(label_line, encoding="utf-8")
                total_aug_lbl += 1

        print(f"  Wrote {total_aug_lbl} labels, {missing_aug} images not readable")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Write dataset.yaml + idx_to_category_id.json
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("STEP 4: Writing config files")
print("=" * 60)

map_path = OUT_DIR / "idx_to_category_id.json"
map_path.write_text(json.dumps(idx_to_cat_id, indent=2), encoding="utf-8")
print(f"  Wrote {map_path}")

abs_out = str(OUT_DIR.resolve()).replace("\\", "/")
yaml_content = (
    f"# YOLOv8 dataset — COCO + SKU110K + aug_product_images (auto-generated by 02_prepare_detection_data.py)\n"
    f"path: {abs_out}\n"
    f"train: images/train\n"
    f"val:   images/val\n"
    f"nc:    {nc}\n"
    f"names: {json.dumps(names, ensure_ascii=False)}\n"
)
yaml_path = OUT_DIR / "dataset.yaml"
yaml_path.write_text(yaml_content, encoding="utf-8")
print(f"  Wrote {yaml_path}")

# ── Final stats ────────────────────────────────────────────────────────────────
n_tr_imgs = len(list((OUT_DIR / "images" / "train").iterdir()))
n_vl_imgs = len(list((OUT_DIR / "images" / "val").iterdir()))
n_tr_lbls = len(list((OUT_DIR / "labels" / "train").iterdir()))
n_vl_lbls = len(list((OUT_DIR / "labels" / "val").iterdir()))

print(f"\n{'='*60}")
print("  DATASET READY")
print(f"{'='*60}")
print(f"  Train images : {n_tr_imgs}  (labels: {n_tr_lbls})")
print(f"  Val   images : {n_vl_imgs}  (labels: {n_vl_lbls})")
print(f"  Classes      : {nc}")
print(f"  Output dir   : {OUT_DIR.resolve()}")
print(f"\nNext: python 03_train_detector.py")