"""
06_build_submission.py — Package submission/ into submission_onnx.zip and validate.

Files packed (all at zip root — run.py MUST be at root, never in a subdir):
  run.py
  detector.onnx
  classifier.onnx
  idx_to_category_id.json
  classifier_idx_to_category_id.json

Usage:
    python 06_build_submission.py
"""
import json
import zipfile
from pathlib import Path

SUBMISSION_DIR = Path("submission")
ZIP_PATH       = Path("submission_onnx.zip")

REQUIRED_FILES = [
    "run.py",
    "detector.onnx",
    "classifier.onnx",
    "idx_to_category_id.json",
    "classifier_idx_to_category_id.json",
]

# ── Pre-flight checks ─────────────────────────────────────────────────────────
print("=" * 60)
print("  PRE-FLIGHT CHECKS")
print("=" * 60)
all_ok   = True
total_mb = 0.0

for fname in REQUIRED_FILES:
    p = SUBMISSION_DIR / fname
    if p.exists():
        size_mb  = p.stat().st_size / 1024**2
        total_mb += size_mb
        print(f"  OK  {fname:45s}  ({size_mb:7.1f} MB)")
    else:
        print(f"  MISSING: {fname}")
        all_ok = False

# Count Python files
py_files = list(SUBMISSION_DIR.glob("*.py"))
print(f"\n  Python files in submission/: {[p.name for p in py_files]}")
if len(py_files) > 10:
    print(f"  WARNING: {len(py_files)} .py files — max allowed is 10")
    all_ok = False

# Count weight files
weight_files = (
    list(SUBMISSION_DIR.glob("*.pt")) +
    list(SUBMISSION_DIR.glob("*.pth")) +
    list(SUBMISSION_DIR.glob("*.onnx")) +
    list(SUBMISSION_DIR.glob("*.safetensors")) +
    list(SUBMISSION_DIR.glob("*.npy"))
)
weight_count = len(weight_files)
print(f"  Weight files: {weight_count}  (max 3 allowed)")
if weight_count > 3:
    print(f"  ERROR: {weight_count} weight files — max allowed is 3")
    print(f"  Files: {[f.name for f in weight_files]}")
    all_ok = False

print(f"\n  Total uncompressed size: {total_mb:.1f} MB  (limit: 420 MB)")
if total_mb > 420:
    print("  ERROR: total size exceeds 420 MB limit!")
    all_ok = False

if not all_ok:
    print("\nERROR: Fix issues above before submitting.")
    raise SystemExit(1)

# ── Check run.py for banned imports ──────────────────────────────────────────
BANNED = [
    "import os", "import sys", "import subprocess", "import socket",
    "import ctypes", "import builtins", "import importlib",
    "import pickle", "import marshal", "import shelve", "import shutil",
    "import yaml", "import requests", "import urllib",
    "import http.client", "import multiprocessing", "import threading",
    "import signal", "import gc", "import code", "import codeop", "import pty",
]
BANNED_CALLS = ["eval(", "exec(", "compile(", "__import__("]

run_py = SUBMISSION_DIR / "run.py"
if run_py.exists():
    run_text = run_py.read_text(encoding="utf-8")
    found_banned = [b for b in BANNED if b in run_text]
    found_calls  = [b for b in BANNED_CALLS if b in run_text]
    if found_banned:
        print(f"\n  ERROR: run.py has banned imports: {found_banned}")
        all_ok = False
    if found_calls:
        print(f"\n  ERROR: run.py has banned calls: {found_calls}")
        all_ok = False
    if not found_banned and not found_calls:
        print("\n  run.py import check: PASS")

if not all_ok:
    print("\nERROR: Fix issues above before submitting.")
    raise SystemExit(1)

# ── Build zip ─────────────────────────────────────────────────────────────────
print(f"\nBuilding {ZIP_PATH} ...")
if ZIP_PATH.exists():
    ZIP_PATH.unlink()

with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for fname in REQUIRED_FILES:
        p = SUBMISSION_DIR / fname
        zf.write(p, arcname=fname)   # arcname ensures file is at zip root
        print(f"  Added {fname}")

zip_size_mb = ZIP_PATH.stat().st_size / 1024**2
print(f"\nZip created: {ZIP_PATH}  ({zip_size_mb:.1f} MB compressed)")

# ── Verify zip structure ──────────────────────────────────────────────────────
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    names = zf.namelist()

print(f"Files in zip: {names}")

if "run.py" not in names:
    print("ERROR: run.py not at zip root!")
    raise SystemExit(1)

subdirs_in_zip = [n for n in names if "/" in n]
if subdirs_in_zip:
    print(f"ERROR: Subdirectories found in zip: {subdirs_in_zip}")
    raise SystemExit(1)

print("\n" + "=" * 60)
print("  GO — submission_onnx.zip is valid!")
print("=" * 60)
print(f"  File    : {ZIP_PATH.resolve()}")
print(f"  Size    : {zip_size_mb:.1f} MB compressed / {total_mb:.1f} MB uncompressed")
print(f"  Files   : {names}")
print(f"\n  Download submission_onnx.zip and upload to the competition portal.")
print("=" * 60)
