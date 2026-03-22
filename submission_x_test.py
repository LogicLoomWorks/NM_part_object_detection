"""
Submission Validator — run this in the same directory as your submission.zip
Checks everything the competition validator checks before you waste an attempt.

Usage:  python test_submission.py
   or:  paste cells into a Jupyter notebook
"""
from pathlib import Path
import zipfile
import ast
import re

# ── CONFIG ────────────────────────────────────────────────────────────────────
ZIP_PATH = Path("submission_x.zip")  # adjust if your zip has a different name

ALLOWED_EXTENSIONS = {".py", ".json", ".yaml", ".yml", ".cfg", ".pt", ".pth", ".onnx", ".safetensors", ".npy"}
WEIGHT_EXTENSIONS = {".pt", ".pth", ".onnx", ".safetensors", ".npy"}
MAX_FILES = 1000
MAX_PY_FILES = 10
MAX_WEIGHT_FILES = 3
MAX_WEIGHT_SIZE_MB = 420
MAX_ZIP_SIZE_MB = 420

BLOCKED_IMPORTS = {
    "os", "sys", "subprocess", "socket", "ctypes", "builtins", "importlib",
    "pickle", "marshal", "shelve", "shutil",
    "yaml",
    "requests", "urllib", "http", "http.client",
    "multiprocessing", "threading", "signal", "gc",
    "code", "codeop", "pty",
}

BLOCKED_CALLS = {"eval", "exec", "compile", "__import__"}

# ── HELPERS ───────────────────────────────────────────────────────────────────
pass_count = 0
fail_count = 0
warn_count = 0

def ok(msg):
    global pass_count
    pass_count += 1
    print(f"  ✅ {msg}")

def fail(msg):
    global fail_count
    fail_count += 1
    print(f"  ❌ {msg}")

def warn(msg):
    global warn_count
    warn_count += 1
    print(f"  ⚠️  {msg}")

# ── 1. ZIP EXISTS ─────────────────────────────────────────────────────────────
print("=" * 60)
print("SUBMISSION VALIDATOR")
print("=" * 60)

print("\n[1/7] Checking zip file exists...")
if not ZIP_PATH.exists():
    fail(f"{ZIP_PATH} not found! Place this script next to your zip.")
    print("\n🛑 Cannot continue without zip file.")
    raise SystemExit(1)
ok(f"Found {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1e6:.1f} MB compressed)")

# ── 2. ZIP STRUCTURE ─────────────────────────────────────────────────────────
print("\n[2/7] Checking zip structure...")
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    names = zf.namelist()
    infos = zf.infolist()

    # Filter out directory entries
    file_names = [n for n in names if not n.endswith("/")]
    total_uncompressed = sum(i.file_size for i in infos)

    print(f"       Files in zip: {file_names}")

    # run.py at root?
    if "run.py" in file_names:
        ok("run.py is at the root of the zip")
    else:
        # Check if it's nested
        nested = [n for n in file_names if n.endswith("run.py")]
        if nested:
            fail(f"run.py is NESTED inside a folder: {nested[0]}  ← THIS IS YOUR PROBLEM")
        else:
            fail("run.py not found anywhere in the zip!")

    # Any subdirectories with files?
    dirs_with_files = set()
    for n in file_names:
        parts = Path(n).parts
        if len(parts) > 1:
            dirs_with_files.add(parts[0])
    if dirs_with_files:
        fail(f"Files inside subdirectories: {dirs_with_files} — all files must be at root level")
    else:
        ok("All files are at root level (no subdirectories)")

# ── 3. FILE TYPES ─────────────────────────────────────────────────────────────
print("\n[3/7] Checking file types...")
py_files = []
weight_files = []
weight_size = 0
bad_types = []

for n in file_names:
    ext = Path(n).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        bad_types.append(n)
    if ext == ".py":
        py_files.append(n)
    if ext in WEIGHT_EXTENSIONS:
        weight_files.append(n)
        # get size
        for i in infos:
            if i.filename == n:
                weight_size += i.file_size

if bad_types:
    fail(f"Disallowed file types: {bad_types}")
else:
    ok("All file types are allowed")

# ── 4. LIMITS ─────────────────────────────────────────────────────────────────
print("\n[4/7] Checking limits...")

if len(file_names) <= MAX_FILES:
    ok(f"File count: {len(file_names)} / {MAX_FILES}")
else:
    fail(f"Too many files: {len(file_names)} > {MAX_FILES}")

if len(py_files) <= MAX_PY_FILES:
    ok(f"Python files: {len(py_files)} / {MAX_PY_FILES}")
else:
    fail(f"Too many .py files: {len(py_files)} > {MAX_PY_FILES}")

if len(weight_files) <= MAX_WEIGHT_FILES:
    ok(f"Weight files: {len(weight_files)} / {MAX_WEIGHT_FILES}")
else:
    fail(f"Too many weight files: {len(weight_files)} > {MAX_WEIGHT_FILES}")

weight_mb = weight_size / (1024 * 1024)
if weight_mb <= MAX_WEIGHT_SIZE_MB:
    ok(f"Weight size: {weight_mb:.1f} MB / {MAX_WEIGHT_SIZE_MB} MB")
else:
    fail(f"Weight size too large: {weight_mb:.1f} MB > {MAX_WEIGHT_SIZE_MB} MB")

total_mb = total_uncompressed / (1024 * 1024)
if total_mb <= MAX_ZIP_SIZE_MB:
    ok(f"Total uncompressed: {total_mb:.1f} MB / {MAX_ZIP_SIZE_MB} MB")
else:
    fail(f"Uncompressed too large: {total_mb:.1f} MB > {MAX_ZIP_SIZE_MB} MB")

# ── 5. SECURITY SCAN ─────────────────────────────────────────────────────────
print("\n[5/7] Security scan (blocked imports & calls)...")
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    for py_file in py_files:
        source = zf.read(py_file).decode("utf-8")
        try:
            tree = ast.parse(source, filename=py_file)
        except SyntaxError as e:
            fail(f"{py_file}: syntax error — {e}")
            continue

        found_blocked = []
        found_calls = []

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in BLOCKED_IMPORTS:
                        found_blocked.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split(".")[0]
                    if top in BLOCKED_IMPORTS:
                        found_blocked.append(node.module)
            # Check dangerous calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in BLOCKED_CALLS:
                        found_calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in BLOCKED_CALLS:
                        found_calls.append(node.func.attr)

        if found_blocked:
            fail(f"{py_file}: BLOCKED imports → {found_blocked}")
        else:
            ok(f"{py_file}: no blocked imports")

        if found_calls:
            fail(f"{py_file}: BLOCKED calls → {found_calls}")
        else:
            ok(f"{py_file}: no blocked calls (eval/exec/compile/__import__)")

# ── 6. RUN.PY CONTRACT ───────────────────────────────────────────────────────
print("\n[6/7] Checking run.py contract...")
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    if "run.py" in file_names:
        source = zf.read("run.py").decode("utf-8")

        # Check --input and --output args
        if "--input" in source:
            ok("run.py accepts --input argument")
        else:
            fail("run.py missing --input argument")

        if "--output" in source:
            ok("run.py accepts --output argument")
        else:
            fail("run.py missing --output argument")

        # Check output format keys
        for key in ["image_id", "category_id", "bbox", "score"]:
            if key in source:
                ok(f'Output includes "{key}" field')
            else:
                warn(f'Could not find "{key}" in run.py — make sure output JSON has it')

        # Check weight file reference matches an actual file in zip
        weight_refs = re.findall(r'["\']([^"\']+\.(?:pt|pth|onnx|safetensors|npy))["\']', source)
        for ref in weight_refs:
            basename = Path(ref).name
            if basename in file_names:
                ok(f"Weight reference '{basename}' found in zip")
            else:
                warn(f"Weight reference '{ref}' — basename '{basename}' not found in zip (may use Path logic)")

# ── 7. OUTPUT FORMAT SPOT-CHECK ───────────────────────────────────────────────
print("\n[7/7] Checking output JSON field mapping...")
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    if "run.py" in file_names:
        source = zf.read("run.py").decode("utf-8")

        # The output must use "category_id" not "label"
        if '"category_id"' in source:
            ok('Output uses "category_id" (correct COCO field name)')
        elif '"label"' in source and '"category_id"' not in source:
            fail('Output uses "label" instead of "category_id" — competition expects "category_id"!')
        else:
            warn('Could not determine category field name — verify manually')

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  ✅ Passed: {pass_count}")
print(f"  ❌ Failed: {fail_count}")
print(f"  ⚠️  Warnings: {warn_count}")

if fail_count == 0:
    print("\n🎉 Your submission looks good to upload!")
else:
    print(f"\n🛑 Fix {fail_count} failure(s) before uploading.")