#!/usr/bin/env python3
"""Rebuild ZZZ_submission/ cleanly with FP16 weights and no omegaconf.

Uses pathlib for all I/O — no shutil.

Gallery strategy
----------------
.npz is NOT in the allowed extension set (.py .json .yaml .yml .cfg
.pt .pth .onnx .safetensors .npy).  Instead the two gallery arrays are
packed into a single .npy file as a (N, D+1) float32 matrix:
  columns 0..D-1  — L2-normalised embeddings (float32)
  column  D       — category_ids cast to float32

load_gallery() in the copied run.py is patched to match.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
DEST = ROOT / "ZZZ_submission"

WEIGHT_EXTS   = {".pt", ".pth", ".onnx", ".safetensors", ".npy"}
ALLOWED_EXTS  = {".py", ".json", ".yaml", ".yml", ".cfg",
                 ".pt", ".pth", ".onnx", ".safetensors", ".npy"}


# ── helpers ───────────────────────────────────────────────────────────────────

def copy_file(src: Path, dst: Path) -> None:
    """Copy any file using pathlib (binary read/write)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def copy_text(src: Path, dst: Path, replacements: dict[str, str]) -> None:
    """Copy a text file applying string substitutions."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8")
    for old, new in replacements.items():
        text = text.replace(old, new)
    dst.write_text(text, encoding="utf-8")


# ── Step 1: delete existing ZZZ_submission/ ──────────────────────────────────

def rmtree(p: Path) -> None:
    """Recursively delete a directory tree using pathlib only."""
    if not p.exists():
        return
    for child in p.iterdir():
        if child.is_dir():
            rmtree(child)
        else:
            child.unlink()
    p.rmdir()


if DEST.exists():
    print(f"Removing existing {DEST} ...")
    rmtree(DEST)
DEST.mkdir(parents=True)
print(f"Created fresh {DEST}")


# ── Step 2: copy Python helper files ─────────────────────────────────────────

PY_FILES = [
    "models/__init__.py",
    "models/DEIMv2/__init__.py",
    "models/DEIMv2/backbone.py",
    "models/DEIMv2/cdn.py",
    "models/DEIMv2/model.py",
    "models/DEIMv2/neck.py",
    "models/DEIMv2/transformer.py",
]

for rel in PY_FILES:
    src = ROOT / rel
    dst = DEST / rel
    copy_file(src, dst)
    print(f"  Copied {rel}")


# ── Step 3: copy run.py with filename patches ─────────────────────────────────
#
# Patches applied:
#   1. CKPT_PATH  — deimv2_best.pt  → deimv2_fp16.pt
#   2. safetensors_load path — vision_model.safetensors → vision_model_fp16.safetensors
#   3. load_gallery() body — two separate np.load calls → single stacked .npy

RUN_SRC = ROOT / "run.py"
RUN_DST = DEST / "run.py"

OLD_LOAD_GALLERY = '''\
    emb  = np.load(str(GALLERY_DIR / "gallery_embeddings.npy"))   # (N, D)
    cats = np.load(str(GALLERY_DIR / "gallery_category_ids.npy")) # (N,)
    return torch.from_numpy(emb).to(device), cats'''

NEW_LOAD_GALLERY = '''\
    data = np.load(str(GALLERY_DIR / "gallery_combined.npy"))
    emb  = np.ascontiguousarray(data[:, :-1])      # (N, D) float32
    cats = data[:, -1].astype(np.int32)             # (N,)
    return torch.from_numpy(emb).to(device), cats'''

copy_text(RUN_SRC, RUN_DST, {
    # docstring comment
    "checkpoints/deimv2_best.pt   — DEIMv2 detector checkpoint":
        "checkpoints/deimv2_fp16.pt   — DEIMv2 detector checkpoint (FP16)",
    # path constant
    '"checkpoints" / "deimv2_best.pt"':           '"checkpoints" / "deimv2_fp16.pt"',
    # safetensors load call
    '"vision_model.safetensors"':                  '"vision_model_fp16.safetensors"',
    # load_gallery body
    OLD_LOAD_GALLERY:                              NEW_LOAD_GALLERY,
})
print("  Copied run.py  (with filename + load_gallery patches)")

# Verify patches landed
run_text = RUN_DST.read_text(encoding="utf-8")
assert "deimv2_fp16.pt"             in run_text, "PATCH FAILED: deimv2_fp16.pt"
assert "vision_model_fp16.safetensors" in run_text, "PATCH FAILED: vision_model_fp16.safetensors"
assert "gallery_combined.npy"       in run_text, "PATCH FAILED: gallery_combined.npy"
assert "gallery_embeddings.npy"  not in run_text, "OLD REF REMAINS: gallery_embeddings.npy"
assert "gallery_category_ids.npy" not in run_text, "OLD REF REMAINS: gallery_category_ids.npy"
assert "deimv2_best.pt"          not in run_text, "OLD REF REMAINS: deimv2_best.pt"
print("  run.py patch assertions OK")


# ── Step 4: copy FP16 weight files ───────────────────────────────────────────

WEIGHT_FILES = {
    "checkpoints/deimv2_fp16.pt":                      "checkpoints/deimv2_fp16.pt",
    "siglip_weights/vision_model_fp16.safetensors":    "siglip_weights/vision_model_fp16.safetensors",
}

for src_rel, dst_rel in WEIGHT_FILES.items():
    src = ROOT / src_rel
    dst = DEST / dst_rel
    copy_file(src, dst)
    mb = dst.stat().st_size / 1e6
    print(f"  Copied {dst_rel}  ({mb:.1f} MB)")


# ── Step 5: build gallery_combined.npy ───────────────────────────────────────
#
# Pack embeddings (N, D) float32 and category_ids (N,) int32 into a
# single (N, D+1) float32 array.  Last column = category_ids as float32.
# No pickle needed; plain np.save / np.load with allow_pickle=False works.

EMB_SRC = ROOT / "gallery" / "gallery_embeddings.npy"
CAT_SRC = ROOT / "gallery" / "gallery_category_ids.npy"
GAL_DST = DEST / "gallery" / "gallery_combined.npy"

embeddings   = np.load(str(EMB_SRC), allow_pickle=False)   # (N, D) float32
category_ids = np.load(str(CAT_SRC), allow_pickle=False)   # (N,)   int32

combined = np.hstack([
    embeddings,
    category_ids.reshape(-1, 1).astype(np.float32),
])  # (N, D+1) float32

GAL_DST.parent.mkdir(parents=True, exist_ok=True)
np.save(str(GAL_DST), combined)

mb = GAL_DST.stat().st_size / 1e6
print(f"  Created gallery/gallery_combined.npy  "
      f"({combined.shape}, {mb:.2f} MB)")

# Quick round-trip sanity check
_check = np.load(str(GAL_DST), allow_pickle=False)
assert _check.shape == combined.shape
assert _check[:, :-1].dtype == np.float32
assert np.array_equal(_check[:, -1].astype(np.int32), category_ids)
print("  gallery_combined.npy round-trip OK")


# ── Step 6: six submission checks ────────────────────────────────────────────

print("\n=== Submission checks ===\n")

all_files    = [f for f in DEST.rglob("*") if f.is_file()]
py_files     = [f for f in all_files if f.suffix == ".py"]
weight_files = [f for f in all_files if f.suffix in WEIGHT_EXTS]
weight_bytes = sum(f.stat().st_size for f in weight_files)
weight_mb    = weight_bytes / 1e6
bad_ext      = [f for f in all_files if f.suffix not in ALLOWED_EXTS]
run_py       = DEST / "run.py"

checks = [
    (
        "1. Total files <= 1000",
        len(all_files) <= 1000,
        f"{len(all_files)} files",
    ),
    (
        "2. Python files <= 10",
        len(py_files) <= 10,
        f"{len(py_files)} .py files: {[f.name for f in py_files]}",
    ),
    (
        "3. Weight files <= 3",
        len(weight_files) <= 3,
        f"{len(weight_files)} weight files: "
        f"{[str(f.relative_to(DEST)) for f in weight_files]}",
    ),
    (
        "4. Total weight size <= 420 MB",
        weight_mb <= 420.0,
        f"{weight_mb:.2f} MB",
    ),
    (
        "5. All extensions in allowed set",
        len(bad_ext) == 0,
        "OK" if not bad_ext
        else f"DISALLOWED: {[str(f.relative_to(DEST)) for f in bad_ext]}",
    ),
    (
        "6. run.py at zip root",
        run_py.exists() and run_py.parent == DEST,
        "ZZZ_submission/run.py exists" if run_py.exists() else "MISSING",
    ),
]

all_passed = True
for label, passed, detail in checks:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}]  {label}")
    print(f"         {detail}")
    if not passed:
        all_passed = False

print()
if all_passed:
    print("All six checks PASSED.")
else:
    print("One or more checks FAILED — see above.")

# ── File listing ──────────────────────────────────────────────────────────────

print("\n=== ZZZ_submission/ contents ===\n")
for f in sorted(all_files):
    rel  = f.relative_to(DEST)
    mb   = f.stat().st_size / 1e6
    print(f"  {str(rel):<60}  {mb:>8.3f} MB")
total_mb = sum(f.stat().st_size for f in all_files) / 1e6
print(f"\n  {'TOTAL':<60}  {total_mb:>8.3f} MB")
