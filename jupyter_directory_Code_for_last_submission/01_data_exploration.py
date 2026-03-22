"""
01_data_exploration.py — Summarize datasets before training.

Checks:
  1. COCO dataset (train folder)
  2. SKU110K_fixed (if available)
  3. aug_product_images (augmented classifier data — augmented version of classifier/)
  4. classifier (original classifier data — product folders)
"""
import json
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
COCO_DIR       = Path("data/raw/coco_dataset/train")
ANN_FILE       = COCO_DIR / "annotations.json"
COCO_IMGS      = COCO_DIR / "images"

SKU_BASE       = Path("data/raw/extra_data/SKU110K_fixed")
SKU_ANN        = SKU_BASE / "annotations"
SKU_IMGS       = SKU_BASE / "images"

AUG_PROD_DATA  = Path("data/aug_product_images")  # ← augmented classifier data (augmented version of classifier/)
PROD_DATA      = Path("data/classifier")          # ← original classifier data


# ══════════════════════════════════════════════════════════════════════════════
# 1. COCO DATASET
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  1. COCO DATASET")
print("=" * 60)

if not ANN_FILE.exists():
    print(f"  ERROR: {ANN_FILE} not found")
else:
    data        = json.loads(ANN_FILE.read_text(encoding="utf-8"))
    images      = data["images"]
    annotations = data["annotations"]
    categories  = data["categories"]

    print(f"  Images      : {len(images)}")
    print(f"  Annotations : {len(annotations)}")
    print(f"  Categories  : {len(categories)}  (IDs {min(c['id'] for c in categories)}–{max(c['id'] for c in categories)})")

    # Check for ID gaps
    cat_ids = sorted([c["id"] for c in categories])
    gaps = []
    for i in range(len(cat_ids) - 1):
        if cat_ids[i+1] - cat_ids[i] > 1:
            gaps.append(f"{cat_ids[i]}–{cat_ids[i+1]}")
    print(f"  ID gaps     : {', '.join(gaps) if gaps else 'none'}")

    # Images with no annotations
    ann_img_ids = set(a["image_id"] for a in annotations)
    all_img_ids = set(im["id"] for im in images)
    no_ann = len(all_img_ids - ann_img_ids)
    print(f"  Images with no annotations: {no_ann}")

    if annotations:
        print(f"  Avg annotations/image: {len(annotations) / len(images):.1f}")

    # Image dimensions
    widths = [im["width"] for im in images]
    heights = [im["height"] for im in images]
    print(f"  Width  range: {min(widths)}–{max(widths)} px  avg={sum(widths)//len(widths)}")
    print(f"  Height range: {min(heights)}–{max(heights)} px  avg={sum(heights)//len(heights)}")

    # Top/bottom categories by annotation count
    cat_counts = defaultdict(int)
    for a in annotations:
        cat_counts[a["category_id"]] += 1

    sorted_cats = sorted(cat_counts.items(), key=lambda x: -x[1])
    print("\n  Top 10 most common categories:")
    for cat_id, count in sorted_cats[:10]:
        cat_name = next((c["name"] for c in categories if c["id"] == cat_id), "?")
        print(f"    [{cat_id:3d}] {count:5d}  {cat_name}")

    print("\n  Bottom 10 least common categories:")
    for cat_id, count in sorted_cats[-10:]:
        cat_name = next((c["name"] for c in categories if c["id"] == cat_id), "?")
        print(f"    [{cat_id:3d}] {count:5d}  {cat_name}")

    # Categories with 0 annotations
    zero_cats = [c["id"] for c in categories if c["id"] not in cat_counts]
    print(f"\n  Categories with 0 annotations: {len(zero_cats)}")

    # Check for degenerate boxes
    degen = sum(1 for a in annotations if a["bbox"][2] <= 0 or a["bbox"][3] <= 0)
    print(f"  Degenerate boxes (w<=0 or h<=0): {degen}")

    # Check if images exist on disk
    if COCO_IMGS.exists():
        img_files = set(f.name for f in COCO_IMGS.iterdir() if f.is_file())
        file_names = set(im["file_name"] for im in images)
        missing = len(file_names - img_files)
        print(f"\n  Images on disk : {len(images)}")
        print(f"  Missing files  : {missing}")
    else:
        print(f"\n  Images dir not found: {COCO_IMGS}")

    # Sample category names
    print(f"\n  Sample category names:")
    for i, c in enumerate(categories[:8]):
        print(f"    [{i:3d}] {c['name']}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. SKU110K_fixed
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("  2. SKU110K_fixed")
print("=" * 60)

sku_train_csv = SKU_ANN / "annotations_train.csv"
sku_val_csv   = SKU_ANN / "annotations_val.csv"
sku_test_csv  = SKU_ANN / "annotations_test.csv"

sku_counts = {"train": 0, "val": 0, "test": 0}
for split, csv_path in [("train", sku_train_csv), ("val", sku_val_csv), ("test", sku_test_csv)]:
    if csv_path.exists():
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        sku_counts[split] = len(lines) - 1  # subtract header
    else:
        print(f"  WARNING: {csv_path} not found")

print(f"  Total SKU110K images: {sum(sku_counts.values())}")
if not SKU_IMGS.exists():
    print(f"  WARNING: SKU110K images dir not found: {SKU_IMGS}")
else:
    sku_img_count = len(list(SKU_IMGS.iterdir()))
    print(f"  SKU110K images on disk: {sku_img_count}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. aug_product_images (augmented classifier data)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("  3. aug_product_images (augmented classifier data)")
print("=" * 60)

if AUG_PROD_DATA.exists():
    # Count product folders and images
    product_folders = [d for d in AUG_PROD_DATA.iterdir() if d.is_dir()]
    total_aug_images = 0
    folder_img_counts = []

    for folder in product_folders:
        img_count = len([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        total_aug_images += img_count
        folder_img_counts.append(img_count)

    print(f"  Product folders  : {len(product_folders)}")
    print(f"  Total images     : {total_aug_images}")
    if folder_img_counts:
        avg = total_aug_images / len(product_folders)
        print(f"  Avg images/folder: {avg:.1f}")
        print(f"  Min/Max images/folder: {min(folder_img_counts)} / {max(folder_img_counts)}")
        
        # Sample folder names
        sample_names = [f.name for f in sorted(product_folders)[:8]]
        print(f"  Sample folder names: {sample_names}")
else:
    print(f"  WARNING: {AUG_PROD_DATA} not found")


# ══════════════════════════════════════════════════════════════════════════════
# 4. product_images (original classifier data)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("  4. classifier (original classifier data)")
print("=" * 60)

if PROD_DATA.exists():
    # Count product folders and images
    product_folders = [d for d in PROD_DATA.iterdir() if d.is_dir()]
    total_prod_images = 0
    folder_img_counts = []

    for folder in product_folders:
        img_count = len([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        total_prod_images += img_count
        folder_img_counts.append(img_count)

    print(f"  Product folders  : {len(product_folders)}")
    print(f"  Total images     : {total_prod_images}")
    if folder_img_counts:
        avg = total_prod_images / len(product_folders)
        print(f"  Avg images/folder: {avg:.1f}")
        print(f"  Min/Max images/folder: {min(folder_img_counts)} / {max(folder_img_counts)}")
        
        # Sample folder names
        sample_names = [f.name for f in sorted(product_folders)[:8]]
        print(f"  Sample folder names: {sample_names}")

    # Check if any folders match COCO categories
    if ANN_FILE.exists():
        data = json.loads(ANN_FILE.read_text(encoding="utf-8"))
        coco_cat_ids = {str(c["id"]) for c in data["categories"]}
        prod_folder_names = {f.name for f in product_folders}
        matched = len(coco_cat_ids & prod_folder_names)
        print(f"  Folders matching COCO categories: {matched}/{len(product_folders)}")
else:
    print(f"  WARNING: {PROD_DATA} not found")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print("  DATA SUMMARY")
print("=" * 64)

coco_img_count = len(images) if ANN_FILE.exists() else 0
coco_ann_count = len(annotations) if ANN_FILE.exists() else 0
coco_cat_count = len(categories) if ANN_FILE.exists() else 0

aug_prod_count = len([d for d in AUG_PROD_DATA.iterdir() if d.is_dir()]) if AUG_PROD_DATA.exists() else 0
aug_prod_images = sum(
    len([f for f in d.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    for d in AUG_PROD_DATA.iterdir() if d.is_dir()
) if AUG_PROD_DATA.exists() else 0

prod_count = len([d for d in PROD_DATA.iterdir() if d.is_dir()]) if PROD_DATA.exists() else 0
prod_images = sum(
    len([f for f in d.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    for d in PROD_DATA.iterdir() if d.is_dir()
) if PROD_DATA.exists() else 0

print(f"  coco_dataset:    {coco_img_count} images, {coco_ann_count} annotations, {coco_cat_count} categories")
print(f"  SKU110K:         {sku_counts['train']} train / {sku_counts['val']} val / {sku_counts['test']} test images")
print(f"  aug_product:     {aug_prod_count} folders, {aug_prod_images} images (augmented classifier)")
print(f"  classifier:      {prod_count} folders, {prod_images} images (original classifier)")
print("=" * 64)

print("\nNext: python 02_prepare_detection_data.py")