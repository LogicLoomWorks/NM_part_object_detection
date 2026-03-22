"""
run.py Contract Compliance Audit -- NorgesGruppen Data competition.

Statically analyses run.py with the AST module and, when all model weights
are present, runs a lightweight dynamic end-to-end smoke-test.

Exit codes
----------
0 -- all checks pass (SKIP results are informational, not failures)
1 -- one or more checks fail
"""

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_PY       = PROJECT_ROOT / "run.py"

# All files that run.py requires at start-up; used to gate the dynamic test.
_MODEL_PATHS: list[Path] = [
    PROJECT_ROOT / "checkpoints"   / "deimv2_best.pt",
    PROJECT_ROOT / "siglip_weights" / "vision_model.safetensors",
    PROJECT_ROOT / "gallery"       / "gallery_embeddings.npy",
    PROJECT_ROOT / "gallery"       / "gallery_category_ids.npy",
]

# The two required CLI flags per the run.py contract.
_REQUIRED_FLAGS = {"--input", "--output"}

# Test filenames  ->  expected integer image_id values.
_TEST_IMAGES: list[tuple[str, int]] = [
    ("img_00001.jpg",  1),
    ("img_00042.jpg", 42),
    ("img_00100.jpg", 100),
]

# ---------------------------------------------------------------------------
# Minimal valid JPEG factory
# ---------------------------------------------------------------------------

def _make_jpeg_bytes() -> bytes:
    """Return bytes for a tiny JPEG that OpenCV can decode.

    Tries PIL first (Pillow is pre-installed in the sandbox / venv);
    falls back to a hardcoded 1x1 white JPEG if PIL is unavailable.
    """
    try:
        import io
        from PIL import Image  # type: ignore[import]
        img = Image.new("RGB", (32, 32), color=(100, 149, 237))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=50)
        return buf.getvalue()
    except Exception:
        pass

    # Hardcoded 1x1 white JPEG (SOI + JFIF + quantisation + SOF + Huffman +
    # SOS + compressed data + EOI) -- known to be decodable by libjpeg/OpenCV.
    return bytes([
        0xFF,0xD8,0xFF,0xE0,0x00,0x10,0x4A,0x46,0x49,0x46,0x00,0x01,
        0x01,0x00,0x00,0x01,0x00,0x01,0x00,0x00,0xFF,0xDB,0x00,0x43,
        0x00,0x08,0x06,0x06,0x07,0x06,0x05,0x08,0x07,0x07,0x07,0x09,
        0x09,0x08,0x0A,0x0C,0x14,0x0D,0x0C,0x0B,0x0B,0x0C,0x19,0x12,
        0x13,0x0F,0x14,0x1D,0x1A,0x1F,0x1E,0x1D,0x1A,0x1C,0x1C,0x20,
        0x24,0x2E,0x27,0x20,0x22,0x2C,0x23,0x1C,0x1C,0x28,0x37,0x29,
        0x2C,0x30,0x31,0x34,0x34,0x34,0x1F,0x27,0x39,0x3D,0x38,0x32,
        0x3C,0x2E,0x33,0x34,0x32,0xFF,0xC0,0x00,0x0B,0x08,0x00,0x01,
        0x00,0x01,0x01,0x01,0x11,0x00,0xFF,0xC4,0x00,0x1F,0x00,0x00,
        0x01,0x05,0x01,0x01,0x01,0x01,0x01,0x01,0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
        0x09,0x0A,0x0B,0xFF,0xC4,0x00,0xB5,0x10,0x00,0x02,0x01,0x03,
        0x03,0x02,0x04,0x03,0x05,0x05,0x04,0x04,0x00,0x00,0x01,0x7D,
        0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,
        0x13,0x51,0x61,0x07,0x22,0x71,0x14,0x32,0x81,0x91,0xA1,0x08,
        0x23,0x42,0xB1,0xC1,0x15,0x52,0xD1,0xF0,0x24,0x33,0x62,0x72,
        0x82,0x09,0x0A,0x16,0x17,0x18,0x19,0x1A,0x25,0x26,0x27,0x28,
        0x29,0x2A,0x34,0x35,0x36,0x37,0x38,0x39,0x3A,0x43,0x44,0x45,
        0x46,0x47,0x48,0x49,0x4A,0x53,0x54,0x55,0x56,0x57,0x58,0x59,
        0x5A,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6A,0x73,0x74,0x75,
        0x76,0x77,0x78,0x79,0x7A,0x83,0x84,0x85,0x86,0x87,0x88,0x89,
        0x8A,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9A,0xA2,0xA3,
        0xA4,0xA5,0xA6,0xA7,0xA8,0xA9,0xAA,0xB2,0xB3,0xB4,0xB5,0xB6,
        0xB7,0xB8,0xB9,0xBA,0xC2,0xC3,0xC4,0xC5,0xC6,0xC7,0xC8,0xC9,
        0xCA,0xD2,0xD3,0xD4,0xD5,0xD6,0xD7,0xD8,0xD9,0xDA,0xE1,0xE2,
        0xE3,0xE4,0xE5,0xE6,0xE7,0xE8,0xE9,0xEA,0xF1,0xF2,0xF3,0xF4,
        0xF5,0xF6,0xF7,0xF8,0xF9,0xFA,0xFF,0xDA,0x00,0x08,0x01,0x01,
        0x00,0x00,0x3F,0x00,0xFB,0xD7,0xFF,0xD9,
    ])


# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------

_STATUS_LABEL = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}


class Results:
    def __init__(self) -> None:
        self._rows: list[tuple[str, str, str]] = []  # (name, status, detail)

    def pass_(self, name: str, detail: str = "") -> None:
        self._rows.append((name, "PASS", detail))

    def fail(self, name: str, detail: str = "") -> None:
        self._rows.append((name, "FAIL", detail))

    def skip(self, name: str, detail: str = "") -> None:
        self._rows.append((name, "SKIP", detail))

    def has_failures(self) -> bool:
        return any(s == "FAIL" for _, s, _ in self._rows)

    def fail_count(self) -> int:
        return sum(1 for _, s, _ in self._rows if s == "FAIL")

    def print_table(self) -> None:
        col = max((len(n) for n, _, _ in self._rows), default=4)
        header = f"  {'Check':<{col}}  Status  Detail"
        print(header)
        print("  " + "-" * max(len(header) - 2, 60))
        for name, status, detail in self._rows:
            label = _STATUS_LABEL[status]
            tail  = f"  {detail}" if detail else ""
            print(f"  {name:<{col}}  {label}{tail}")


# ---------------------------------------------------------------------------
# AST visitors
# ---------------------------------------------------------------------------

class _ArgparseVisitor(ast.NodeVisitor):
    """Collect argparse imports and all add_argument() flag names."""

    def __init__(self) -> None:
        self.has_import: bool      = False
        self.flags: list[str]      = []   # every first-arg string starting with '-'

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".")[0] == "argparse":
                self.has_import = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.module.split(".")[0] == "argparse":
            self.has_import = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # parser.add_argument("--xxx", ...) or ArgumentParser().add_argument(...)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
        ):
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                self.flags.append(first.value)
        self.generic_visit(node)


class _JsonWriteVisitor(ast.NodeVisitor):
    """Detect json.dumps/json.dump calls and write_text() calls."""

    def __init__(self) -> None:
        self.dumps_args: list[ast.expr] = []   # first arg of each json.dumps(...)
        self.dump_calls: int            = 0    # count of json.dump(...) calls
        self.write_text_calls: int      = 0

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == "json":
                if func.attr == "dumps" and node.args:
                    self.dumps_args.append(node.args[0])
                elif func.attr == "dump":
                    self.dump_calls += 1
            if func.attr == "write_text":
                self.write_text_calls += 1
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# AST analysis helpers
# ---------------------------------------------------------------------------

def _infer_is_list(arg: ast.expr, tree: ast.AST) -> bool | None:
    """Best-effort: is `arg` (passed to json.dumps) a list?

    Returns True/False if deterministic, None if inconclusive.
    """
    if isinstance(arg, ast.List):
        return True
    if not isinstance(arg, ast.Name):
        return None

    var = arg.id
    init_as_list   = False
    has_dict_append = False

    for node in ast.walk(tree):
        # var = []  or  var = list()
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == var:
                    val = node.value
                    if isinstance(val, ast.List):
                        init_as_list = True
                    if (isinstance(val, ast.Call)
                            and isinstance(val.func, ast.Name)
                            and val.func.id == "list"):
                        init_as_list = True

        # var.append({...})
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "append"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == var
                and call.args
                and isinstance(call.args[0], ast.Dict)
            ):
                has_dict_append = True

    if init_as_list or has_dict_append:
        return True
    return None


def _dict_append_keys(tree: ast.AST, var: str) -> set[str] | None:
    """Return all constant string keys used in var.append({...}) calls.

    Returns None if no such calls found.
    """
    keys: set[str] = set()
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "append"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == var
                and call.args
                and isinstance(call.args[0], ast.Dict)
            ):
                found = True
                for k in call.args[0].keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        keys.add(k.value)
    return keys if found else None


def _check_image_id_int(tree: ast.AST) -> tuple[bool | None, str]:
    """Check that image_id is extracted as int(), not float or string.

    Returns (passed, detail_message).
    """
    int_on_stem   = False
    float_of_id   = False

    for node in ast.walk(tree):
        # int(path.stem) or int(some_digit_var)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "int"
            and node.args
        ):
            arg = node.args[0]
            # int(x.stem)
            if isinstance(arg, ast.Attribute) and arg.attr in ("stem", "name"):
                int_on_stem = True
            # int(digits) where the variable name suggests digit extraction
            if isinstance(arg, ast.Name) and any(
                kw in arg.id.lower() for kw in ("digit", "num", "stem", "id_str")
            ):
                int_on_stem = True

        # Catch image_id = int(...) assignments
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and "image_id" in tgt.id.lower():
                    val = node.value
                    if isinstance(val, ast.Call):
                        fname = ""
                        if isinstance(val.func, ast.Name):
                            fname = val.func.id
                        if fname == "int":
                            int_on_stem = True
                        if fname == "float":
                            float_of_id = True

        # float(image_id) -- bad
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "float"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and "image_id" in node.args[0].id.lower()
        ):
            float_of_id = True

    if float_of_id:
        return False, "image_id cast to float() somewhere -- must be int"
    if int_on_stem:
        return True, "int() applied to filename stem / digit string"
    return None, "inconclusive -- could not find explicit int() extraction"


# ---------------------------------------------------------------------------
# Static (AST) checks
# ---------------------------------------------------------------------------

def run_ast_checks(tree: ast.AST, results: Results) -> None:
    # -- 1. argparse imported -------------------------------------------------
    ap = _ArgparseVisitor()
    ap.visit(tree)

    if ap.has_import:
        results.pass_("argparse imported")
    else:
        results.fail("argparse imported", "no 'import argparse' or 'from argparse import' found")

    # -- 2. --input and --output declared; no unexpected long flags -----------
    long_flags = {f for f in ap.flags if f.startswith("--")}
    missing    = _REQUIRED_FLAGS - long_flags
    extra      = long_flags - _REQUIRED_FLAGS

    if "--input" in long_flags:
        results.pass_("--input argument declared")
    else:
        results.fail("--input argument declared", "add_argument('--input', ...) not found")

    if "--output" in long_flags:
        results.pass_("--output argument declared")
    else:
        results.fail("--output argument declared", "add_argument('--output', ...) not found")

    if not extra:
        results.pass_("no extra --flags", f"flags found: {sorted(long_flags)}")
    else:
        results.fail("no extra --flags", f"unexpected: {sorted(extra)}")

    # -- 3. JSON serialisation ------------------------------------------------
    jw = _JsonWriteVisitor()
    jw.visit(tree)

    if jw.dumps_args or jw.dump_calls:
        results.pass_("json.dumps / json.dump called")
    else:
        results.fail("json.dumps / json.dump called", "no JSON serialisation found")

    if jw.write_text_calls > 0 or jw.dump_calls > 0:
        results.pass_("output written to file (write_text or json.dump)")
    else:
        results.fail(
            "output written to file (write_text or json.dump)",
            "no path.write_text() or json.dump(fp) detected",
        )

    # -- 4. Output is a JSON array of objects ---------------------------------
    serialised_var: str | None = None
    list_result: bool | None   = None

    if jw.dumps_args:
        arg = jw.dumps_args[0]
        list_result = _infer_is_list(arg, tree)
        if isinstance(arg, ast.Name):
            serialised_var = arg.id

    if list_result is True:
        results.pass_("JSON output is a list (array)")
        if serialised_var:
            keys = _dict_append_keys(tree, serialised_var)
            required = {"image_id", "category_id", "bbox", "score"}
            if keys is not None:
                missing_k = required - keys
                if not missing_k:
                    results.pass_(
                        "output dicts contain required keys",
                        str(sorted(keys & required)),
                    )
                else:
                    results.fail(
                        "output dicts contain required keys",
                        f"missing: {sorted(missing_k)}",
                    )
            else:
                results.skip(
                    "output dicts contain required keys",
                    "no .append({...}) calls found on serialised variable",
                )
        else:
            results.skip(
                "output dicts contain required keys",
                "serialised value is not a simple name -- check dynamically",
            )
    elif list_result is None:
        results.skip("JSON output is a list (array)", "inconclusive -- check dynamically")
        results.skip("output dicts contain required keys", "depends on list check")
    else:
        results.fail("JSON output is a list (array)", "serialised value does not appear to be a list")
        results.skip("output dicts contain required keys", "depends on list check")

    # -- 5. image_id extracted as int, not float ------------------------------
    passed, detail = _check_image_id_int(tree)
    if passed is True:
        results.pass_("image_id extracted as int()", detail)
    elif passed is False:
        results.fail("image_id extracted as int()", detail)
    else:
        results.skip("image_id extracted as int()", detail)


# ---------------------------------------------------------------------------
# Dynamic test
# ---------------------------------------------------------------------------

def _model_files_present() -> tuple[bool, list[str]]:
    missing = [
        str(p.relative_to(PROJECT_ROOT))
        for p in _MODEL_PATHS
        if not p.exists()
    ]
    return len(missing) == 0, missing


def _dynamic_skip_all(results: Results, reason: str) -> None:
    for name in (
        "dynamic: run.py exits 0",
        "dynamic: output file is valid JSON list",
        "dynamic: every element has required keys",
        "dynamic: image_id is correct int (not float/string)",
        "dynamic: image_id matches filename (img_XXXXX -> int)",
        "dynamic: bbox is a list of 4 numbers",
        "dynamic: score is float in [0, 1]",
    ):
        results.skip(name, reason)


def run_dynamic_test(results: Results) -> None:
    present, missing_files = _model_files_present()
    if not present:
        reason = f"model weights absent: {', '.join(missing_files)}"
        _dynamic_skip_all(results, reason)
        print(f"  [SKIP] Dynamic test skipped -- {reason}")
        return

    print("  Model weights found -- running dynamic smoke-test ...")

    # Build temp directory with three tiny test JPEGs.
    jpeg_bytes = _make_jpeg_bytes()
    tmp = tempfile.TemporaryDirectory(prefix="audit_contract_")
    try:
        input_dir   = Path(tmp.name)
        output_file = input_dir / "predictions.json"
        for name, _ in _TEST_IMAGES:
            (input_dir / name).write_bytes(jpeg_bytes)

        print(f"  Test images written to {input_dir}")
        print(f"  Invoking: {sys.executable} run.py --input <tmp> --output <tmp>/predictions.json")

        proc = subprocess.run(
            [
                sys.executable, str(RUN_PY),
                "--input",  str(input_dir),
                "--output", str(output_file),
            ],
            capture_output=True,
            text=True,
            timeout=240,
            cwd=str(PROJECT_ROOT),
        )

        if proc.stdout:
            print("\n  --- run.py stdout (last 20 lines) ---")
            for line in proc.stdout.splitlines()[-20:]:
                print(f"  | {line}")
        if proc.stderr:
            print("\n  --- run.py stderr (last 10 lines) ---")
            for line in proc.stderr.splitlines()[-10:]:
                print(f"  ! {line}")
        print()

        if proc.returncode != 0:
            results.fail(
                "dynamic: run.py exits 0",
                f"exit code {proc.returncode}; see stderr above",
            )
            _dynamic_skip_all_except_first(results)
            return

        results.pass_("dynamic: run.py exits 0")

        # -- Validate output file ---------------------------------------------
        if not output_file.exists():
            results.fail(
                "dynamic: output file is valid JSON list",
                "output file was not created",
            )
            _dynamic_skip_all_except_first(results)
            return

        raw = output_file.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            results.fail("dynamic: output file is valid JSON list", f"parse error: {exc}")
            _dynamic_skip_all_except_first(results)
            return

        if not isinstance(data, list):
            results.fail(
                "dynamic: output file is valid JSON list",
                f"top-level type is {type(data).__name__}, expected list",
            )
            _dynamic_skip_all_except_first(results)
            return

        results.pass_("dynamic: output file is valid JSON list", f"{len(data)} prediction(s)")

        if not data:
            reason = "no detections produced (tiny dummy image -- expected)"
            for name in (
                "dynamic: every element has required keys",
                "dynamic: image_id is correct int (not float/string)",
                "dynamic: image_id matches filename (img_XXXXX -> int)",
                "dynamic: bbox is a list of 4 numbers",
                "dynamic: score is float in [0, 1]",
            ):
                results.skip(name, reason)
            return

        # -- Per-element validation --------------------------------------------
        required_keys = {"image_id", "category_id", "bbox", "score"}
        valid_image_ids = {img_id for _, img_id in _TEST_IMAGES}

        key_fails:   list[str] = []
        id_type_fails: list[str] = []
        id_value_fails: list[str] = []
        bbox_fails:  list[str] = []
        score_fails: list[str] = []

        for i, elem in enumerate(data):
            tag = f"elem[{i}]"

            if not isinstance(elem, dict):
                key_fails.append(f"{tag} is {type(elem).__name__}, not a dict")
                continue

            missing_k = required_keys - elem.keys()
            if missing_k:
                key_fails.append(f"{tag} missing keys: {sorted(missing_k)}")

            # image_id -- must be plain int
            iid = elem.get("image_id")
            if not isinstance(iid, int) or isinstance(iid, bool):
                id_type_fails.append(
                    f"{tag} image_id={iid!r} (type={type(iid).__name__}, expected int)"
                )
            else:
                if iid not in valid_image_ids:
                    id_value_fails.append(
                        f"{tag} image_id={iid} not in expected set {sorted(valid_image_ids)}"
                    )

            # category_id -- must be int
            cat = elem.get("category_id")
            if not isinstance(cat, int) or isinstance(cat, bool):
                key_fails.append(
                    f"{tag} category_id={cat!r} (type={type(cat).__name__}, expected int)"
                )

            # bbox -- list of exactly 4 numbers
            bbox = elem.get("bbox")
            if not (
                isinstance(bbox, list)
                and len(bbox) == 4
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in bbox)
            ):
                bbox_fails.append(f"{tag} bbox={bbox!r}")

            # score -- float/int in [0, 1]
            sc = elem.get("score")
            if not isinstance(sc, (int, float)) or isinstance(sc, bool):
                score_fails.append(f"{tag} score={sc!r} not numeric")
            elif not (0.0 <= float(sc) <= 1.0):
                score_fails.append(f"{tag} score={sc} outside [0, 1]")

        _record(results, "dynamic: every element has required keys",            key_fails)
        _record(results, "dynamic: image_id is correct int (not float/string)", id_type_fails)
        _record(results, "dynamic: image_id matches filename (img_XXXXX -> int)", id_value_fails)
        _record(results, "dynamic: bbox is a list of 4 numbers",                 bbox_fails)
        _record(results, "dynamic: score is float in [0, 1]",                    score_fails)

    except subprocess.TimeoutExpired:
        results.fail("dynamic: run.py exits 0", "timed out after 240 s")
        _dynamic_skip_all_except_first(results)
    except Exception as exc:
        results.fail("dynamic: run.py exits 0", f"unexpected error: {exc}")
        _dynamic_skip_all_except_first(results)
    finally:
        tmp.cleanup()


def _dynamic_skip_all_except_first(results: Results) -> None:
    """Skip the remaining dynamic sub-checks after a fatal failure."""
    for name in (
        "dynamic: output file is valid JSON list",
        "dynamic: every element has required keys",
        "dynamic: image_id is correct int (not float/string)",
        "dynamic: image_id matches filename (img_XXXXX -> int)",
        "dynamic: bbox is a list of 4 numbers",
        "dynamic: score is float in [0, 1]",
    ):
        results.skip(name, "skipped after earlier dynamic failure")


def _record(results: Results, name: str, failures: list[str]) -> None:
    if not failures:
        results.pass_(name)
    else:
        results.fail(name, "; ".join(failures[:3]))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("  NorgesGruppen Data -- run.py Contract Compliance Audit")
    print(f"  run.py : {RUN_PY}")
    print("=" * 70)

    if not RUN_PY.is_file():
        print(f"\n[FATAL] run.py not found at {RUN_PY}")
        return 1

    source = RUN_PY.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(RUN_PY))
    except SyntaxError as exc:
        print(f"\n[FATAL] Syntax error in run.py: {exc}")
        return 1

    results = Results()

    # -- Static checks --------------------------------------------------------
    print("\n-- Static (AST) checks ----------------------------------------------\n")
    run_ast_checks(tree, results)

    # -- Dynamic checks -------------------------------------------------------
    print("\n-- Dynamic checks ---------------------------------------------------\n")
    run_dynamic_test(results)

    # -- Summary --------------------------------------------------------------
    print("\n")
    print("  Check Summary")
    print("  " + "=" * 66)
    results.print_table()
    print()

    if results.has_failures():
        print("=" * 70)
        print(f"  RESULT: FAIL  -- {results.fail_count()} check(s) failed")
        print("=" * 70)
        return 1

    print("=" * 70)
    print("  RESULT: PASS  -- all checks passed (SKIP = informational only)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
