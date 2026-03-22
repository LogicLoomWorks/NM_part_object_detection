"""
map_directory.py — List every file and folder in the project as a flat JSON array.

Usage:
    python map_directory.py
"""
import json
import time
from pathlib import Path

ROOT = Path(".").resolve()
OUTPUT = Path("directory_map.json")

SKIP = {"__pycache__", ".git", ".ipynb_checkpoints", "node_modules", ".venv"}

print("Scanning", ROOT, "...")
t0 = time.time()

files = []
dirs = []

for item in sorted(ROOT.rglob("*")):
    # Skip excluded folders and anything inside them
    parts = item.relative_to(ROOT).parts
    if any(p in SKIP for p in parts):
        continue

    rel = str(item.relative_to(ROOT))

    if item.is_dir():
        dirs.append(rel)
    elif item.is_file():
        files.append(rel)

output = {
    "root": str(ROOT),
    "total_files": len(files),
    "total_dirs": len(dirs),
    "dirs": dirs,
    "files": files,
}

OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s — {len(files)} files, {len(dirs)} dirs")
print(f"Saved to {OUTPUT}")