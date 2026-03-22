"""
prepare_upload.py — Lag optimalisert data-zip for opplasting til HPC.

Kjør lokalt FØR du laster opp til HPC:
    python prepare_upload.py

Lager:
    data_main.zip   — COCO + product_images + aug_product_images (~1.3 GB)
    data_sku.zip    — SKU110K 5000-bilde-subset (~2-5 GB, valgfri)

Deretter last opp til HPC:
    scp data_main.zip training_pipeline.zip user@hpc.uit.no:~/nm_detection/
    scp data_sku.zip  user@hpc.uit.no:~/nm_detection/   # valgfri, men bedre detektor
"""
import json
import random
import zipfile
from pathlib import Path

SEED = 42
random.seed(SEED)

SKU_MAX = 5000  # maks SKU110K-bilder å inkludere

# ── Kilder ────────────────────────────────────────────────────────────────────
COCO_DIR  = Path("data/raw/coco_dataset/train")
PROD_DIR  = Path("data/raw/product_images")
AUG_DIR   = Path("data/augmented_data/aug_product_images")
SKU_BASE  = Path("data/raw/extra_data/SKU110K_fixed")
SKU_ANN   = SKU_BASE / "annotations"
SKU_IMGS  = SKU_BASE / "images"

OUT_MAIN  = Path("data_main.zip")
OUT_SKU   = Path("data_sku.zip")


def add_dir(zf: zipfile.ZipFile, src: Path, arc_prefix: Path, exts=None):
    """Legg til alle filer i src-mappe med arc_prefix som rot i zip."""
    n = 0
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        if exts and f.suffix.lower() not in exts:
            continue
        arcname = arc_prefix / f.relative_to(src)
        zf.write(f, arcname=str(arcname))
        n += 1
    return n


# ══════════════════════════════════════════════════════════════════════════════
# data_main.zip — alltid nødvendig
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  Lager data_main.zip ...")
print("=" * 60)

if OUT_MAIN.exists():
    OUT_MAIN.unlink()

IMG_EXTS = {".jpg", ".jpeg", ".png"}

with zipfile.ZipFile(OUT_MAIN, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
    # COCO annotations
    ann_file = COCO_DIR / "annotations.json"
    if ann_file.exists():
        zf.write(ann_file, arcname="data/raw/coco_dataset/train/annotations.json")
        print(f"  + COCO annotations.json")

    # COCO images
    if (COCO_DIR / "images").exists():
        n = add_dir(zf, COCO_DIR / "images",
                    Path("data/raw/coco_dataset/train/images"), IMG_EXTS)
        print(f"  + COCO images: {n} filer")

    # product_images
    if PROD_DIR.exists():
        n = add_dir(zf, PROD_DIR, Path("data/raw/product_images"), IMG_EXTS)
        print(f"  + product_images: {n} filer")

    # aug_product_images
    if AUG_DIR.exists():
        n = add_dir(zf, AUG_DIR, Path("data/augmented_data/aug_product_images"), IMG_EXTS)
        print(f"  + aug_product_images: {n} filer")

mb = OUT_MAIN.stat().st_size / 1024**2
print(f"\n  data_main.zip: {mb:.0f} MB")


# ══════════════════════════════════════════════════════════════════════════════
# data_sku.zip — valgfri, bedre detektor med denne
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print(f"  Lager data_sku.zip (maks {SKU_MAX} bilder per split) ...")
print("=" * 60)

def get_sku_images(csv_path: Path, max_imgs: int) -> list:
    """Returner liste med bildenavn fra SKU110K CSV."""
    if not csv_path.exists():
        print(f"  Ikke funnet: {csv_path}")
        return []
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    seen = set()
    imgs = []
    for line in lines[1:]:
        name = line.split(",")[0].strip()
        if name and name not in seen:
            seen.add(name)
            imgs.append(name)
    random.shuffle(imgs)
    return imgs[:max_imgs]


if not SKU_BASE.exists():
    print("  SKU110K ikke funnet lokalt — hopper over data_sku.zip")
else:
    if OUT_SKU.exists():
        OUT_SKU.unlink()

    with zipfile.ZipFile(OUT_SKU, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for split in ("train", "val"):
            csv_path = SKU_ANN / f"annotations_{split}.csv"
            if not csv_path.exists():
                continue

            # Alltid inkluder CSV
            zf.write(csv_path,
                     arcname=f"data/raw/extra_data/SKU110K_fixed/annotations/annotations_{split}.csv")
            print(f"  + SKU110K annotations_{split}.csv")

            # Subsample bilder
            limit = SKU_MAX if split == "train" else None
            img_names = get_sku_images(csv_path, limit or 99999)

            added = 0
            for name in img_names:
                img_path = SKU_IMGS / name
                if not img_path.exists():
                    stem = Path(name).stem
                    alts = list(SKU_IMGS.glob(f"{stem}.*"))
                    img_path = alts[0] if alts else None
                if img_path:
                    zf.write(img_path,
                             arcname=f"data/raw/extra_data/SKU110K_fixed/images/{img_path.name}")
                    added += 1

            print(f"  + SKU110K {split} images: {added} stk")

    mb_sku = OUT_SKU.stat().st_size / 1024**2
    print(f"\n  data_sku.zip: {mb_sku:.0f} MB")


# ── Oppsummering ──────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  FERDIG — last opp disse filene til HPC:")
print("=" * 60)
print(f"  {OUT_MAIN}   ({OUT_MAIN.stat().st_size/1024**2:.0f} MB)  ← alltid nødvendig")
if OUT_SKU.exists():
    print(f"  {OUT_SKU}    ({OUT_SKU.stat().st_size/1024**2:.0f} MB)  ← valgfri (+SKU110K-detektor)")
print(f"  training_pipeline.zip  (21 KB)  ← treningskoden")
print()
print("  scp-kommandoer (tilpass bruker/hostname):")
print(f"  scp data_main.zip training_pipeline.zip BRUKER@HOSTNAME:~/nm_detection/")
if OUT_SKU.exists():
    print(f"  scp data_sku.zip BRUKER@HOSTNAME:~/nm_detection/")
print()
print("  Åpne så ZZZZ_train_remote.ipynb på HPC og kjør alle celler.")
