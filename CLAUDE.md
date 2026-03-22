# CLAUDE.md — NorgesGruppen Submission Rules (ENFORCE STRICTLY)

## HARD LIMITS — Never violate these
- `run.py` MUST be at the zip root (never in a subfolder)
- Max 10 Python files total
- Max 3 weight files (.pt, .pth, .onnx, .safetensors, .npy)
- Max 420 MB total (uncompressed)
- Allowed extensions ONLY: `.py`, `.json`, `.yaml`, `.yml`, `.cfg`, `.pt`, `.pth`, `.onnx`, `.safetensors`, `.npy`
- No `.bin` files (rename to `.pt` or convert to `.safetensors`)

## BANNED IMPORTS — Never write these in any .py file
```
os, sys, subprocess, socket, ctypes, builtins, importlib,
pickle, marshal, shelve, shutil, yaml,
requests, urllib, http.client,
multiprocessing, threading, signal, gc,
code, codeop, pty
```
- Use `pathlib` instead of `os`
- Use `json` instead of `yaml`

## BANNED CALLS — Never use these
```
eval(), exec(), compile(), __import__()
```
- No symlinks, no path traversal, no ELF/Mach-O/PE binaries

## PINNED PACKAGE VERSIONS — Never exceed these
| Package | Version |
|---|---|
| ultralytics | ==8.1.0 (NOT 8.2+) |
| torch | ==2.6.0 (NOT 2.7+) |
| torchvision | ==0.21.0 |
| timm | ==0.9.12 (NOT 1.0+) |
| ONNX opset | ≤ 20 (use 17 to be safe) |

## WEIGHT SAVING — Always use state_dict
```python
# CORRECT
torch.save(model.state_dict(), "model.pt")

# WRONG — never do this
torch.save(model, "model.pt")
```

## run.py INTERFACE — Must match exactly
```bash
python run.py --input /data/images --output /output/predictions.json
```
- Input: JPEG files named `img_xxxx.jpg`
- Parse image_id as: `int("img_00042.jpg".split("_")[1].split(".")[0])` → `42`

## OUTPUT FORMAT — Must match exactly
```json
[
  {
    "image_id": 42,
    "category_id": 0,
    "bbox": [120.5, 45.0, 80.0, 110.0],
    "score": 0.923
  }
]
```
- `category_id`: integer 0–355 (NOT 356 unless truly unknown)
- `bbox`: COCO format [x, y, width, height]
- `score`: float 0.0–1.0

## GPU — Always use, never skip
```python
device = "cuda" if torch.cuda.is_available() else "cpu"  # always cuda in sandbox
# For ONNX:
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
```

## NO RUNTIME INSTALLS — Never do this
```python
# WRONG — network is offline, pip is blocked
import subprocess; subprocess.run(["pip", "install", ...])
```
All packages must be pre-bundled or already in the sandbox.

## SANDBOX SPECS (read-only reference)
- Python 3.11, CUDA 12.4, GPU: NVIDIA L4 (24 GB VRAM)
- Timeout: 300 seconds for all images
- Memory: 8 GB RAM

## ZIP CREATION — Use this exact command
```bash
cd my_submission/
zip -r ../submission.zip . -x ".*" "__MACOSX/*"
```
Verify with: `unzip -l submission.zip | head -5` — `run.py` must appear at root level.
