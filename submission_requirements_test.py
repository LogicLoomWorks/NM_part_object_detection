"""
CLAUDE CODE IS NOT ALLOWED TO MODIFY THIS FILE


Submission validator for NorgesGruppen Object Detection competition.
Checks ALL rules from the submission format specification in one file.

Usage:
    python validate_submission.py                        # validates ./submission/ directory
    python validate_submission.py --dir my_submission/   # validates a directory
    python validate_submission.py --zip submission.zip   # validates a zip file
"""

import argparse
import ast
import re
import struct
import zipfile
import tempfile
import shutil
import json
from pathlib import Path
from collections import defaultdict

# ── Constants from rules.txt ──────────────────────────────────────────────

MAX_UNCOMPRESSED_SIZE_MB = 420
MAX_FILES = 1000
MAX_PYTHON_FILES = 10
MAX_WEIGHT_FILES = 3
MAX_WEIGHT_SIZE_MB = 420
MAX_ONNX_OPSET = 20
TIMEOUT_SECONDS = 300

ALLOWED_EXTENSIONS = {
    ".py", ".json", ".yaml", ".yml", ".cfg",
    ".pt", ".pth", ".onnx", ".safetensors", ".npy",
}

WEIGHT_EXTENSIONS = {".pt", ".pth", ".onnx", ".safetensors", ".npy"}

# Extensions people commonly try that are NOT allowed
RENAME_HINTS = {
    ".bin": "Rename .bin → .pt (identical format) or convert with safetensors",
    ".h5": "Convert HDF5 weights to .pt via torch.save(state_dict)",
    ".pkl": "Pickle files blocked — use .pt or .safetensors",
    ".tar": "Extract and include individual weight files",
    ".gz": "Extract and include individual weight files",
    ".zip": "Nested zips not allowed — flatten into submission root",
    ".txt": "Use .json or .cfg instead",
    ".csv": "Not an allowed extension — embed data in .json",
    ".md": "Not an allowed extension",
}

BLOCKED_IMPORTS = {
    "os", "sys", "subprocess", "socket", "ctypes", "builtins", "importlib",
    "pickle", "marshal", "shelve", "shutil",
    "yaml",
    "requests", "urllib", "http",
    "multiprocessing", "threading", "signal", "gc",
    "code", "codeop", "pty",
}

BLOCKED_CALLS = {"eval", "exec", "compile", "__import__"}

# Sandbox pre-installed packages (for version-mismatch warnings)
SANDBOX_VERSIONS = {
    "ultralytics": "8.1.0",
    "torch": "2.6.0",
    "torchvision": "0.21.0",
    "timm": "0.9.12",
    "onnxruntime": "1.20.0",
    "numpy": "1.26.4",
    "albumentations": "1.3.1",
    "Pillow": "10.2.0",
    "scipy": "1.12.0",
    "scikit-learn": "1.4.0",
    "pycocotools": "2.0.7",
    "supervision": "0.18.0",
    "safetensors": "0.4.2",
    "opencv-python-headless": "4.9.0.80",
    "ensemble-boxes": "1.0.9",
}

# Required fields in each prediction object
PREDICTION_SCHEMA = {
    "image_id": int,
    "category_id": int,
    "bbox": list,
    "score": (int, float),
}


# ── Helpers ───────────────────────────────────────────────────────────────

class Result:
    def __init__(self):
        self.passed = []
        self.warnings = []
        self.failed = []

    def ok(self, msg):
        self.passed.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    def fail(self, msg):
        self.failed.append(msg)

    @property
    def success(self):
        return len(self.failed) == 0


def fmt_mb(n_bytes):
    return f"{n_bytes / (1024 ** 2):.1f} MB"


def collect_files(root: Path):
    """Yield (relative_path, absolute_path) for every file under root."""
    for p in sorted(root.rglob("*")):
        if p.is_file():
            yield p.relative_to(root), p


# ── AST-based import / call checker ──────────────────────────────────────

def check_python_safety(filepath: Path, source: str, res: Result):
    """Parse a .py file and flag blocked imports and calls."""
    rel = filepath
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as e:
        res.warn(f"  Could not parse {rel} (syntax error line {e.lineno}) — skipping AST checks")
        return

    for node in ast.walk(tree):
        # ── import X / import X.Y ────────────────────────────────
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in BLOCKED_IMPORTS:
                    res.fail(
                        f"  {rel}:{node.lineno} — blocked import: `import {alias.name}`"
                    )

        # ── from X import Y ──────────────────────────────────────
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in BLOCKED_IMPORTS:
                    res.fail(
                        f"  {rel}:{node.lineno} — blocked import: `from {node.module} import ...`"
                    )

        # ── dangerous calls: eval(), exec(), compile(), __import__() ─
        elif isinstance(node, ast.Call):
            fn = node.func
            name = None
            if isinstance(fn, ast.Name):
                name = fn.id
            elif isinstance(fn, ast.Attribute):
                name = fn.attr

            if name in BLOCKED_CALLS:
                res.fail(
                    f"  {rel}:{node.lineno} — blocked call: `{name}()`"
                )

            # getattr with dangerous second arg
            if name == "getattr" and len(node.args) >= 2:
                arg = node.args[1]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value.startswith("__") or arg.value in BLOCKED_CALLS:
                        res.fail(
                            f"  {rel}:{node.lineno} — suspicious getattr: `getattr(..., '{arg.value}')`"
                        )


def check_run_py_contract(source: str, res: Result):
    """Verify run.py accepts --input and --output via argparse."""
    has_input = bool(re.search(r"""['"]--input['"]""", source))
    has_output = bool(re.search(r"""['"]--output['"]""", source))

    if has_input and has_output:
        res.ok("run.py accepts --input and --output arguments")
    else:
        missing = []
        if not has_input:
            missing.append("--input")
        if not has_output:
            missing.append("--output")
        res.fail(f"run.py missing CLI arguments: {', '.join(missing)}")

    # Check that it writes JSON (looks for json.dump or json.dumps or Path.write_text)
    writes_json = bool(
        re.search(r"json\.dump", source)
        or re.search(r"\.write_text\(", source)
        or re.search(r"open\(.*['\"]w['\"]", source)
    )
    if writes_json:
        res.ok("run.py appears to write output (json.dump / write_text / open)")
    else:
        res.warn("run.py — couldn't confirm it writes JSON output (look for json.dump)")

    # Check for pathlib usage (recommended over os)
    uses_pathlib = "pathlib" in source or "Path(" in source
    if uses_pathlib:
        res.ok("run.py uses pathlib for file operations")
    else:
        res.warn("run.py doesn't import pathlib — recommended over os for file operations")


def check_sandbox_version_warnings(py_sources: list[tuple[Path, str]], res: Result):
    """Warn if Python files import packages where version pinning matters."""
    IMPORT_TO_PACKAGE = {
        "ultralytics": "ultralytics",
        "torch": "torch",
        "torchvision": "torchvision",
        "timm": "timm",
        "onnxruntime": "onnxruntime",
        "numpy": "numpy",
        "np": "numpy",
        "albumentations": "albumentations",
        "PIL": "Pillow",
        "scipy": "scipy",
        "sklearn": "scikit-learn",
        "pycocotools": "pycocotools",
        "supervision": "supervision",
        "safetensors": "safetensors",
        "cv2": "opencv-python-headless",
        "ensemble_boxes": "ensemble-boxes",
    }
    imported = set()
    for rel, src in py_sources:
        try:
            tree = ast.parse(src, filename=str(rel))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in IMPORT_TO_PACKAGE:
                        imported.add(IMPORT_TO_PACKAGE[top])
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                if top in IMPORT_TO_PACKAGE:
                    imported.add(IMPORT_TO_PACKAGE[top])

    if not imported:
        return

    version_critical = {"ultralytics", "torch", "torchvision", "timm"}
    for pkg in sorted(imported):
        ver = SANDBOX_VERSIONS.get(pkg)
        if ver and pkg in version_critical:
            res.warn(
                f"  Uses '{pkg}' — sandbox has {pkg}=={ver}. "
                f"Pin this version during training to avoid load failures."
            )
        elif ver:
            res.ok(f"Uses '{pkg}' (sandbox: {ver})")


def check_yaml_config_gotcha(files: list[tuple[Path, Path]], res: Result):
    """Warn if .yaml/.yml files are included but yaml import is blocked."""
    yaml_files = [r for r, _ in files if r.suffix in {".yaml", ".yml"}]
    if yaml_files:
        res.warn(
            f"  Found {len(yaml_files)} YAML config file(s) "
            f"({', '.join(str(r) for r in yaml_files[:5])}), but `import yaml` is blocked "
            f"in the sandbox. Use json.load() or hard-code config values instead."
        )


# ── ONNX opset checker ───────────────────────────────────────────────────

def check_onnx_opset(filepath: Path, res: Result):
    """Read ONNX opset version from the protobuf header without importing onnx."""
    try:
        import onnx
        model = onnx.load(str(filepath), load_external_data=False)
        for opset in model.opset_import:
            if opset.domain == "" or opset.domain == "ai.onnx":
                if opset.version > MAX_ONNX_OPSET:
                    res.fail(
                        f"  {filepath.name}: ONNX opset {opset.version} > {MAX_ONNX_OPSET} "
                        f"— re-export with opset_version=17"
                    )
                    return
                else:
                    res.ok(f"{filepath.name}: ONNX opset {opset.version} (limit {MAX_ONNX_OPSET})")
                    return
        res.warn(f"  {filepath.name}: could not determine ONNX opset version")
    except ImportError:
        try:
            raw = filepath.read_bytes()[:4096]
            res.warn(
                f"  {filepath.name}: install `onnx` to validate opset version "
                f"(must be ≤ {MAX_ONNX_OPSET})"
            )
        except Exception:
            res.warn(f"  {filepath.name}: could not read ONNX file for opset check")
    except Exception as e:
        res.warn(f"  {filepath.name}: ONNX opset check failed — {e}")


# ── Output schema validator ──────────────────────────────────────────────

def check_predictions_json(filepath: Path, res: Result):
    """Validate a predictions.json file against the required schema."""
    try:
        data = json.loads(filepath.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        res.fail(f"  {filepath.name}: invalid JSON — {e}")
        return
    except Exception as e:
        res.warn(f"  {filepath.name}: could not read — {e}")
        return

    if not isinstance(data, list):
        res.fail(f"  {filepath.name}: top-level must be a JSON array, got {type(data).__name__}")
        return

    if len(data) == 0:
        res.warn(f"  {filepath.name}: empty predictions array")
        return

    errors = 0
    max_errors = 10
    all_category_zero = True

    for i, pred in enumerate(data):
        if errors >= max_errors:
            res.warn(f"  ... ({len(data) - i} more predictions not checked, stopping at {max_errors} errors)")
            break

        if not isinstance(pred, dict):
            res.fail(f"  {filepath.name}[{i}]: expected object, got {type(pred).__name__}")
            errors += 1
            continue

        # Check required fields
        for field, expected_type in PREDICTION_SCHEMA.items():
            if field not in pred:
                res.fail(f"  {filepath.name}[{i}]: missing field '{field}'")
                errors += 1
            elif not isinstance(pred[field], expected_type):
                res.fail(
                    f"  {filepath.name}[{i}]: '{field}' should be "
                    f"{expected_type.__name__ if isinstance(expected_type, type) else expected_type}, "
                    f"got {type(pred[field]).__name__}"
                )
                errors += 1

        # Validate bbox shape
        if "bbox" in pred and isinstance(pred["bbox"], list):
            if len(pred["bbox"]) != 4:
                res.fail(f"  {filepath.name}[{i}]: bbox must have 4 elements [x,y,w,h], got {len(pred['bbox'])}")
                errors += 1
            elif not all(isinstance(v, (int, float)) for v in pred["bbox"]):
                res.fail(f"  {filepath.name}[{i}]: bbox values must be numeric")
                errors += 1

        # Validate score range
        if "score" in pred and isinstance(pred["score"], (int, float)):
            if not (0.0 <= pred["score"] <= 1.0):
                res.fail(f"  {filepath.name}[{i}]: score {pred['score']} outside [0, 1]")
                errors += 1

        # Validate category_id range
        if "category_id" in pred and isinstance(pred["category_id"], int):
            if not (0 <= pred["category_id"] <= 355):
                res.fail(f"  {filepath.name}[{i}]: category_id {pred['category_id']} outside [0, 355]")
                errors += 1
            if pred["category_id"] != 0:
                all_category_zero = False

        # Validate bbox values are non-negative and w/h are positive
        if "bbox" in pred and isinstance(pred["bbox"], list) and len(pred["bbox"]) == 4:
            if all(isinstance(v, (int, float)) for v in pred["bbox"]):
                bx, by, bw, bh = pred["bbox"]
                if bx < 0 or by < 0:
                    res.fail(f"  {filepath.name}[{i}]: bbox x/y must be ≥ 0, got [{bx}, {by}, ...]")
                    errors += 1
                if bw <= 0 or bh <= 0:
                    res.fail(f"  {filepath.name}[{i}]: bbox w/h must be > 0, got [..., {bw}, {bh}]")
                    errors += 1

        # Validate image_id is non-negative (derived from img_XXXXX.jpg filenames)
        if "image_id" in pred and isinstance(pred["image_id"], int):
            if pred["image_id"] < 0:
                res.fail(f"  {filepath.name}[{i}]: image_id must be ≥ 0, got {pred['image_id']}")
                errors += 1

    if errors == 0:
        res.ok(f"{filepath.name}: all {len(data)} predictions match required schema")

    if all_category_zero and len(data) > 0:
        res.warn(
            f"  {filepath.name}: all predictions use category_id=0 — "
            f"detection-only (max 70% score). Add classification for the remaining 30%."
        )


# ── Binary sniff (ELF / Mach-O / PE) ─────────────────────────────────────

BINARY_MAGICS = {
    b"\x7fELF": "ELF binary",
    b"\xfe\xed\xfa\xce": "Mach-O binary (32-bit)",
    b"\xfe\xed\xfa\xcf": "Mach-O binary (64-bit)",
    b"\xce\xfa\xed\xfe": "Mach-O binary (32-bit, reversed)",
    b"\xcf\xfa\xed\xfe": "Mach-O binary (64-bit, reversed)",
    b"MZ": "PE/Windows binary",
}


def is_blocked_binary(filepath: Path) -> str | None:
    try:
        with open(filepath, "rb") as f:
            header = f.read(4)
        for magic, label in BINARY_MAGICS.items():
            if header[: len(magic)] == magic:
                return label
    except Exception:
        pass
    return None


# ── Main validation ───────────────────────────────────────────────────────

def validate_directory(root: Path, zip_compressed_size: int | None = None) -> Result:
    res = Result()
    files = list(collect_files(root))
    rel_paths = [r for r, _ in files]

    # ── 1. run.py at root ─────────────────────────────────────────────
    print("\n[1/10] Checking run.py exists at root...")
    if Path("run.py") in rel_paths:
        res.ok("run.py found at root")
    else:
        nested = [r for r in rel_paths if r.name == "run.py"]
        if nested:
            res.fail(f"run.py found but nested inside subfolder: {nested[0]}")
        else:
            res.fail("run.py not found anywhere")

    # ── 2. File counts ────────────────────────────────────────────────
    print("[2/10] Checking file counts...")
    total = len(files)
    py_files = [r for r, _ in files if r.suffix == ".py"]
    weight_files = [(r, p) for r, p in files if r.suffix in WEIGHT_EXTENSIONS]

    if total <= MAX_FILES:
        res.ok(f"Total files: {total} (limit {MAX_FILES})")
    else:
        res.fail(f"Too many files: {total} (limit {MAX_FILES})")

    if len(py_files) <= MAX_PYTHON_FILES:
        res.ok(f"Python files: {len(py_files)} (limit {MAX_PYTHON_FILES})")
    else:
        res.fail(f"Too many Python files: {len(py_files)} (limit {MAX_PYTHON_FILES})")
        for p in py_files:
            print(f"    {p}")

    if len(weight_files) <= MAX_WEIGHT_FILES:
        res.ok(f"Weight files: {len(weight_files)} (limit {MAX_WEIGHT_FILES})")
    else:
        res.fail(f"Too many weight files: {len(weight_files)} (limit {MAX_WEIGHT_FILES})")
        for r, _ in weight_files:
            print(f"    {r}")

    # ── 3. Size checks ────────────────────────────────────────────────
    print("[3/10] Checking sizes...")
    total_size = sum(p.stat().st_size for _, p in files)
    weight_size = sum(p.stat().st_size for _, p in weight_files)

    if total_size <= MAX_UNCOMPRESSED_SIZE_MB * 1024 ** 2:
        res.ok(f"Total uncompressed size: {fmt_mb(total_size)} (limit {MAX_UNCOMPRESSED_SIZE_MB} MB)")
    else:
        res.fail(f"Total uncompressed too large: {fmt_mb(total_size)} (limit {MAX_UNCOMPRESSED_SIZE_MB} MB)")

    if weight_size <= MAX_WEIGHT_SIZE_MB * 1024 ** 2:
        res.ok(f"Total weight size: {fmt_mb(weight_size)} (limit {MAX_WEIGHT_SIZE_MB} MB)")
    else:
        res.fail(f"Weights too large: {fmt_mb(weight_size)} (limit {MAX_WEIGHT_SIZE_MB} MB)")

    if zip_compressed_size is not None:
        res.ok(f"Compressed zip size: {fmt_mb(zip_compressed_size)}")

    # Per-weight file breakdown
    for r, p in weight_files:
        sz = p.stat().st_size
        print(f"    {r}: {fmt_mb(sz)}")

    # ── 4. Allowed file types + rename hints ──────────────────────────
    print("[4/10] Checking file extensions...")
    bad_ext = defaultdict(list)
    for r, _ in files:
        if r.suffix.lower() not in ALLOWED_EXTENSIONS:
            bad_ext[r.suffix.lower()].append(r)

    if not bad_ext:
        res.ok("All file extensions are allowed")
    else:
        for ext, paths in bad_ext.items():
            hint = RENAME_HINTS.get(ext, "")
            hint_str = f" — {hint}" if hint else ""
            res.fail(
                f"Disallowed extension '{ext}': "
                f"{', '.join(str(p) for p in paths[:5])}"
                + (f" (+{len(paths)-5} more)" if len(paths) > 5 else "")
                + hint_str
            )

    # ── 5. No symlinks or path traversal ──────────────────────────────
    print("[5/10] Checking for symlinks and path traversal...")
    symlinks = [r for r, p in files if p.is_symlink()]
    traversal = [r for r in rel_paths if ".." in r.parts]
    if not symlinks and not traversal:
        res.ok("No symlinks or path traversal found")
    else:
        for s in symlinks:
            res.fail(f"Symlink found: {s}")
        for t in traversal:
            res.fail(f"Path traversal found: {t}")

    # ── 6. No blocked binaries ────────────────────────────────────────
    print("[6/10] Checking for blocked binaries...")
    found_bins = []
    for r, p in files:
        if r.suffix in {".py", ".json", ".yaml", ".yml", ".cfg"}:
            continue
        label = is_blocked_binary(p)
        if label:
            found_bins.append((r, label))
    if not found_bins:
        res.ok("No blocked binary executables found")
    else:
        for r, label in found_bins:
            res.fail(f"Blocked binary ({label}): {r}")

    # ── 7. Python security scan ───────────────────────────────────────
    print("[7/10] Scanning Python files for blocked imports/calls...")
    py_sources = []
    for r, p in files:
        if r.suffix == ".py":
            try:
                src = p.read_text(encoding="utf-8")
                py_sources.append((r, src))
            except Exception as e:
                res.warn(f"  Could not read {r}: {e}")

    blocked_before = len(res.failed)
    for r, src in py_sources:
        check_python_safety(r, src, res)
    blocked_after = len(res.failed)

    if blocked_after == blocked_before:
        res.ok("No blocked imports or calls found in Python files")

    # ── 8. run.py contract ────────────────────────────────────────────
    print("[8/10] Checking run.py contract (--input / --output / writes JSON)...")
    run_sources = [(r, src) for r, src in py_sources if r == Path("run.py")]
    if run_sources:
        check_run_py_contract(run_sources[0][1], res)
    else:
        res.warn("Skipped run.py contract check (file not readable)")

    # ── 8b. Sandbox version warnings ──────────────────────────────────
    print("[8b/10] Checking sandbox package version compatibility...")
    check_sandbox_version_warnings(py_sources, res)

    # ── 8c. YAML config gotcha ────────────────────────────────────────
    print("[8c/10] Checking for YAML config files vs blocked yaml import...")
    check_yaml_config_gotcha(files, res)

    # ── 9. ONNX opset version ─────────────────────────────────────────
    onnx_files = [(r, p) for r, p in files if r.suffix == ".onnx"]
    if onnx_files:
        print("[9/10] Checking ONNX opset versions...")
        for r, p in onnx_files:
            check_onnx_opset(p, res)
    else:
        print("[9/10] No ONNX files — skipping opset check")

    # ── 10. Predictions.json schema (if present) ──────────────────────
    pred_files = [(r, p) for r, p in files if r.name == "predictions.json"]
    if pred_files:
        print("[10/10] Validating predictions.json schema...")
        for r, p in pred_files:
            check_predictions_json(p, res)
    else:
        print("[10/10] No predictions.json found — skipping output schema check")
        res.warn(
            "No predictions.json in submission to validate. "
            "Run your pipeline once and include output to verify schema, or ignore this."
        )

    return res


# ── Report ────────────────────────────────────────────────────────────────

def print_report(res: Result):
    print("\n" + "=" * 64)
    print("  SUBMISSION VALIDATION REPORT")
    print("=" * 64)

    if res.passed:
        print(f"\n  PASSED ({len(res.passed)})")
        for m in res.passed:
            print(f"    ✅  {m}")

    if res.warnings:
        print(f"\n  WARNINGS ({len(res.warnings)})")
        for m in res.warnings:
            print(f"    ⚠️   {m}")

    if res.failed:
        print(f"\n  FAILED ({len(res.failed)})")
        for m in res.failed:
            print(f"    ❌  {m}")

    print("\n" + "-" * 64)
    print(f"  Sandbox reminder: Python 3.11 | L4 GPU 24GB | 300s timeout")
    print(f"  Scoring: 70% detection mAP + 30% classification mAP")
    print("-" * 64)

    if res.success:
        print("  ✅  ALL CHECKS PASSED — ready to submit!")
    else:
        print(f"  ❌  {len(res.failed)} check(s) failed — fix before submitting.")
    print("-" * 64 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate NorgesGruppen submission against all competition rules"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dir", type=str, default=None, help="Path to submission directory")
    group.add_argument("--zip", type=str, default=None, help="Path to submission .zip file")
    group.add_argument(
        "--check-output", type=str, default=None,
        help="Path to a predictions.json to validate schema only"
    )
    args = parser.parse_args()

    # ── Schema-only mode ──────────────────────────────────────────────
    if args.check_output:
        p = Path(args.check_output)
        if not p.exists():
            print(f"Error: file not found: {p}")
            return
        print(f"Validating predictions schema: {p}")
        res = Result()
        check_predictions_json(p, res)
        print_report(res)
        return

    # ── Zip mode ──────────────────────────────────────────────────────
    if args.zip:
        zip_path = Path(args.zip)
        if not zip_path.exists():
            print(f"Error: zip file not found: {zip_path}")
            return

        compressed_size = zip_path.stat().st_size
        tmp = tempfile.mkdtemp(prefix="validate_sub_")
        try:
            print(f"Extracting {zip_path} ({fmt_mb(compressed_size)} compressed)...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Check for zip-bomb / nested folders before extracting
                names = zf.namelist()
                nested_run = [n for n in names if n.endswith("run.py") and "/" in n]
                root_run = "run.py" in names
                if nested_run and not root_run:
                    print(f"\n  ⚠️  run.py is nested: {nested_run[0]}")
                    print(f"     Common mistake — zip the *contents*, not the folder.\n")

                zf.extractall(tmp)
            res = validate_directory(Path(tmp), zip_compressed_size=compressed_size)
            print_report(res)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ── Directory mode ────────────────────────────────────────────────
    else:
        d = Path(args.dir) if args.dir else Path("submission")
        if not d.exists():
            print(f"Error: directory not found: {d}")
            print("Usage:")
            print("  python validate_submission.py --dir <path>")
            print("  python validate_submission.py --zip <path>")
            print("  python validate_submission.py --check-output predictions.json")
            return
        print(f"Validating directory: {d}")
        res = validate_directory(d)
        print_report(res)


if __name__ == "__main__":
    main()