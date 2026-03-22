"""
Build ZZZ_submission/ by copying exactly the files needed for competition submission.
Uses pathlib + shutil only — no shell commands.
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).parent
DEST = ROOT / "ZZZ_submission"

FILES = [
    "run.py",
    "models/__init__.py",
    "models/DEIMv2/__init__.py",
    "models/DEIMv2/model.py",
    "models/DEIMv2/backbone.py",
    "models/DEIMv2/neck.py",
    "models/DEIMv2/transformer.py",
    "models/DEIMv2/cdn.py",
    "checkpoints/deimv2_best.pt",
    "siglip_weights/vision_model.safetensors",
    "siglip_weights/preprocessor_config.json",
    "gallery/gallery_embeddings.npy",
    "gallery/gallery_category_ids.npy",
]

# ── clean slate ──────────────────────────────────────────────────────────────
if DEST.exists():
    shutil.rmtree(DEST)
DEST.mkdir(parents=True)

# ── copy ─────────────────────────────────────────────────────────────────────
missing = []
copied  = []

for rel in FILES:
    src = ROOT / rel
    dst = DEST / rel
    if not src.exists():
        missing.append(rel)
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(dst)

# ── report ───────────────────────────────────────────────────────────────────
print("\n=== ZZZ_submission contents ===\n")

all_files = sorted(DEST.rglob("*"))
total_bytes = 0
py_count     = 0
weight_count = 0

for f in all_files:
    if not f.is_file():
        continue
    size_bytes = f.stat().st_size
    size_mb    = size_bytes / (1024 ** 2)
    total_bytes += size_bytes
    rel_path = f.relative_to(DEST)
    print(f"  {str(rel_path):<55}  {size_mb:>9.4f} MB")
    if f.suffix == ".py":
        py_count += 1
    if f.suffix in (".pt", ".safetensors"):
        weight_count += 1

total_mb = total_bytes / (1024 ** 2)
print(f"\n{'-'*70}")
print(f"  Total size   : {total_mb:.4f} MB")
print(f"  .py files    : {py_count}")
print(f"  weight files : {weight_count}  (.pt / .safetensors)")

if missing:
    print(f"\nWARNING - Missing source files (skipped):")
    for m in missing:
        print(f"     {m}")
