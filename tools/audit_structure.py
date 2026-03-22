"""
Submission structure and size audit for NorgesGruppen Data competition.

Validates all structural and size constraints from rules.txt.

Usage:
  python tools/audit_structure.py [--dir <submission_dir>]

Exit codes:
  0 — all checks pass
  1 — one or more checks fail
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Limits from rules.txt
# ---------------------------------------------------------------------------

MAX_TOTAL_FILES   = 1000
MAX_PY_FILES      = 10
MAX_WEIGHT_FILES  = 3
MAX_WEIGHT_BYTES  = 420 * 1024 * 1024   # 420 MB
MAX_TOTAL_BYTES   = 420 * 1024 * 1024   # 420 MB

WEIGHT_EXTENSIONS = {".pt", ".pth", ".onnx", ".safetensors", ".npy"}
ALLOWED_EXTENSIONS = {
    ".py", ".json", ".yaml", ".yml", ".cfg",
    ".pt", ".pth", ".onnx", ".safetensors", ".npy",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_bytes(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024**3:.2f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024**2:.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.2f} KB"
    return f"{n} B"


def collect_files(root: Path) -> list[Path]:
    """Return all regular files (not symlinks, not dirs) under root."""
    return [p for p in root.rglob("*") if p.is_file() and not p.is_symlink()]


# ---------------------------------------------------------------------------
# Check results accumulator
# ---------------------------------------------------------------------------

class Results:
    def __init__(self) -> None:
        self._rows: list[tuple[str, str, str, bool]] = []

    def add(self, check: str, value: str, limit: str, passed: bool) -> None:
        self._rows.append((check, value, limit, passed))

    def print_table(self) -> None:
        # Convert to all-string rows for display
        str_rows = [
            (check, value, limit, "PASS" if passed else "FAIL")
            for check, value, limit, passed in self._rows
        ]
        header = ("Check", "Value", "Limit", "Status")
        all_rows = [header] + str_rows
        col_w = [max(len(r[i]) for r in all_rows) for i in range(4)]
        col_w = [max(c, h) for c, h in zip(col_w, [5, 5, 5, 6])]

        def row(cells: tuple[str, str, str, str]) -> str:
            return "  " + "  ".join(c.ljust(w) for c, w in zip(cells, col_w))

        sep = "  " + "  ".join("-" * w for w in col_w)
        print(row(header))
        print(sep)
        for cells in str_rows:
            print(row(cells))

    def all_passed(self) -> bool:
        return all(p for _, _, _, p in self._rows)

    def fail_count(self) -> int:
        return sum(1 for _, _, _, p in self._rows if not p)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_run_py_at_root(root: Path, results: Results) -> None:
    """run.py must exist at the directory root (not in a subfolder)."""
    run_py = root / "run.py"
    exists_at_root = run_py.is_file()
    results.add(
        "run.py at root",
        str(run_py.relative_to(root)) if exists_at_root else "(missing)",
        "required",
        exists_at_root,
    )

    # Also check whether run.py is accidentally nested in a subfolder
    nested = [
        p for p in root.rglob("run.py")
        if p != run_py and p.is_file()
    ]
    if nested:
        for p in nested:
            results.add(
                "run.py not nested",
                str(p.relative_to(root)),
                "root only",
                False,
            )


def check_total_files(files: list[Path], results: Results) -> None:
    count = len(files)
    results.add(
        "Total files",
        str(count),
        f"<= {MAX_TOTAL_FILES}",
        count <= MAX_TOTAL_FILES,
    )


def check_py_files(files: list[Path], results: Results) -> None:
    py_files = [f for f in files if f.suffix.lower() == ".py"]
    count = len(py_files)
    results.add(
        "Python (.py) files",
        str(count),
        f"<= {MAX_PY_FILES}",
        count <= MAX_PY_FILES,
    )
    if py_files:
        print("\n  Python files found:")
        for p in sorted(py_files):
            print(f"    {p}")


def check_weight_files(files: list[Path], results: Results) -> None:
    weight_files = [f for f in files if f.suffix.lower() in WEIGHT_EXTENSIONS]
    count = len(weight_files)
    total_size = sum(f.stat().st_size for f in weight_files)

    results.add(
        "Weight files (count)",
        str(count),
        f"<= {MAX_WEIGHT_FILES}",
        count <= MAX_WEIGHT_FILES,
    )
    results.add(
        "Weight files (total size)",
        fmt_bytes(total_size),
        f"<= {fmt_bytes(MAX_WEIGHT_BYTES)}",
        total_size <= MAX_WEIGHT_BYTES,
    )

    if weight_files:
        print("\n  Weight files found:")
        for p in sorted(weight_files):
            size = p.stat().st_size
            print(f"    {p}  ({fmt_bytes(size)})")


def check_total_size(files: list[Path], results: Results) -> None:
    total = sum(f.stat().st_size for f in files)
    results.add(
        "Total uncompressed size",
        fmt_bytes(total),
        f"<= {fmt_bytes(MAX_TOTAL_BYTES)}",
        total <= MAX_TOTAL_BYTES,
    )


def check_allowed_extensions(root: Path, files: list[Path], results: Results) -> None:
    bad = [f for f in files if f.suffix.lower() not in ALLOWED_EXTENSIONS]
    results.add(
        "All file types allowed",
        f"{len(bad)} disallowed" if bad else "clean",
        "allowed set only",
        len(bad) == 0,
    )
    if bad:
        print("\n  Disallowed file types:")
        for p in sorted(bad):
            print(f"    {p.relative_to(root)}  ('{p.suffix}')")


def check_zip_simulation(root: Path, results: Results) -> None:
    """
    Simulate what a zip of the directory would look like.
    run.py must appear as a top-level entry (no parent folder prefix).
    """
    run_py = root / "run.py"
    if not run_py.is_file():
        results.add("zip simulation: run.py at zip root", "(missing)", "run.py", False)
        return

    # In a correct zip (cd submission_dir && zip -r ../sub.zip .)
    # run.py gets the path "run.py", not "submission_dir/run.py"
    zip_path = run_py.relative_to(root)
    at_root = len(zip_path.parts) == 1

    results.add(
        "zip simulation: run.py at zip root",
        str(zip_path),
        "run.py  (no prefix)",
        at_root,
    )

    # Show what the first few entries would look like
    all_files = collect_files(root)
    print("\n  Simulated zip entries (first 10):")
    for p in sorted(all_files)[:10]:
        print(f"    {p.relative_to(root)}")
    if len(all_files) > 10:
        print(f"    ... and {len(all_files) - 10} more")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate submission structure against competition rules."
    )
    parser.add_argument(
        "--dir",
        default=".",
        help="Submission directory to audit (default: current directory)",
    )
    args = parser.parse_args()

    root = Path(args.dir).resolve()
    if not root.is_dir():
        print(f"ERROR: '{root}' is not a directory.")
        return 1

    files = collect_files(root)
    results = Results()

    print("=" * 70)
    print("  NorgesGruppen Data — Submission Structure Audit")
    print(f"  Directory: {root}")
    print(f"  Files found: {len(files)}")
    print("=" * 70)

    # Run all checks (file listings are printed inline)
    check_run_py_at_root(root, results)
    check_total_files(files, results)
    check_py_files(files, results)
    check_weight_files(files, results)
    check_total_size(files, results)
    check_allowed_extensions(root, files, results)
    check_zip_simulation(root, results)

    # Summary table
    print("\n")
    print("  Check Summary")
    print("  " + "=" * 66)
    results.print_table()
    print()

    if results.all_passed():
        print("=" * 70)
        print("  RESULT: PASS  — all checks passed")
        print("=" * 70)
        return 0
    else:
        print("=" * 70)
        print(f"  RESULT: FAIL  -- {results.fail_count()} check(s) failed")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
