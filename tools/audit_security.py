"""
Security and import audit for NorgesGruppen Data competition submissions.

Scans all .py files (excluding tools/) for banned imports, banned calls,
disallowed file types, and binary/symlink/path-traversal issues.

Exit codes:
  0 — clean (no violations)
  1 — one or more violations found
"""

import ast
import os
import struct
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BANNED_IMPORTS = {
    "os", "sys", "subprocess", "socket", "ctypes", "builtins", "importlib",
    "pickle", "marshal", "shelve", "shutil", "yaml",
    "requests", "urllib", "http.client",
    "multiprocessing", "threading", "signal", "gc",
    "code", "codeop", "pty",
}

BANNED_CALLS = {"eval", "exec", "compile", "__import__"}

# Dangerous names that must not be passed as string literals to getattr()
DANGEROUS_GETATTR_NAMES = BANNED_IMPORTS | {
    "system", "popen", "execve", "execvp", "spawn", "fork",
    "loads", "load", "dumps", "dump",
    "environ", "getenv", "putenv",
}

ALLOWED_EXTENSIONS = {
    ".py", ".json", ".yaml", ".yml", ".cfg",
    ".pt", ".pth", ".onnx", ".safetensors", ".npy",
}

# Binary magic bytes: (offset, magic_bytes, label)
BINARY_SIGNATURES = [
    (0, b"\x7fELF", "ELF binary"),
    (0, b"\xfe\xed\xfa\xce", "Mach-O 32-bit binary"),
    (0, b"\xfe\xed\xfa\xcf", "Mach-O 64-bit binary"),
    (0, b"\xce\xfa\xed\xfe", "Mach-O 32-bit binary (reversed)"),
    (0, b"\xcf\xfa\xed\xfe", "Mach-O 64-bit binary (reversed)"),
    (0, b"MZ", "PE/Windows binary"),
]


# ---------------------------------------------------------------------------
# AST-based checkers
# ---------------------------------------------------------------------------

class ImportVisitor(ast.NodeVisitor):
    """Collect all banned imports and banned calls in one AST walk."""

    def __init__(self, filename: str):
        self.filename = filename
        self.violations: list[str] = []

    # -- import / from-import ------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top in BANNED_IMPORTS or alias.name in BANNED_IMPORTS:
                self.violations.append(
                    f"  line {node.lineno}: banned import '{alias.name}'"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            top = node.module.split(".")[0]
            full = node.module
            if top in BANNED_IMPORTS or full in BANNED_IMPORTS:
                self.violations.append(
                    f"  line {node.lineno}: banned from-import 'from {node.module} import ...'"
                )
        self.generic_visit(node)

    # -- banned calls --------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        func_name = self._call_name(node.func)

        if func_name in BANNED_CALLS:
            self.violations.append(
                f"  line {node.lineno}: banned call '{func_name}()'"
            )

        # getattr(obj, "dangerous_name") or getattr(obj, "dangerous_name", default)
        if func_name == "getattr" and len(node.args) >= 2:
            attr_arg = node.args[1]
            if isinstance(attr_arg, ast.Constant) and isinstance(attr_arg.value, str):
                if attr_arg.value in DANGEROUS_GETATTR_NAMES:
                    self.violations.append(
                        f"  line {node.lineno}: getattr() with dangerous name '{attr_arg.value}'"
                    )

        self.generic_visit(node)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _call_name(node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""


# ---------------------------------------------------------------------------
# File-level checks
# ---------------------------------------------------------------------------

def is_binary(path: Path) -> tuple[bool, str]:
    """Return (True, label) if the file matches a known binary signature."""
    try:
        with open(path, "rb") as f:
            header = f.read(8)
    except OSError:
        return False, ""
    for offset, magic, label in BINARY_SIGNATURES:
        end = offset + len(magic)
        if len(header) >= end and header[offset:end] == magic:
            return True, label
    return False, ""


def check_python_file(path: Path) -> list[str]:
    """Parse and audit a single .py file; return list of violation strings."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [f"  could not read file: {exc}"]

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"  syntax error (cannot parse): {exc}"]

    visitor = ImportVisitor(str(path))
    visitor.visit(tree)
    return visitor.violations


# ---------------------------------------------------------------------------
# Directory walker
# ---------------------------------------------------------------------------

SKIP_DIRS = {
    "tools", ".venv", "venv", ".env", ".git", "__pycache__",
    "node_modules", ".tox", ".mypy_cache", ".pytest_cache",
    "dist", "build", "eggs", ".eggs",
}


def collect_files(root: Path) -> list[Path]:
    """Recursively collect all files under root, skipping excluded directories."""
    files: list[Path] = []

    def _walk(directory: Path) -> None:
        try:
            entries = list(directory.iterdir())
        except PermissionError:
            return
        for entry in entries:
            # Skip excluded directory names
            if entry.is_dir() and not entry.is_symlink():
                if entry.name in SKIP_DIRS:
                    continue
                _walk(entry)
            else:
                files.append(entry)

    _walk(root)
    return files


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def audit(root: Path) -> bool:
    """
    Run the full audit.  Returns True if clean, False if violations found.
    """
    all_files = collect_files(root)
    violations_by_file: dict[str, list[str]] = {}
    global_notes: list[str] = []

    def add(filepath: str, msg: str) -> None:
        violations_by_file.setdefault(filepath, []).append(msg)

    for path in sorted(all_files):
        rel = path.relative_to(root)
        rel_str = str(rel)

        # 1. Symlink check
        if path.is_symlink():
            add(rel_str, "  symlink detected (not allowed)")
            continue  # don't follow symlinks for further checks

        # 2. Path-traversal check
        if ".." in rel.parts:
            add(rel_str, "  path contains '..' (path traversal)")

        # 3. Allowed extension check (skip rules.txt at root explicitly)
        if rel_str == "rules.txt":
            pass  # explicitly permitted
        elif path.suffix.lower() not in ALLOWED_EXTENSIONS:
            add(rel_str, f"  disallowed file type '{path.suffix}'")

        # 4. Binary file check (ELF / Mach-O / PE)
        found_binary, binary_label = is_binary(path)
        if found_binary:
            add(rel_str, f"  {binary_label} detected")

        # 5. YAML outside tools/ — warn about yaml import being banned
        if path.suffix.lower() in {".yaml", ".yml"}:
            add(
                rel_str,
                "  WARNING: yaml/yml file found — 'yaml' import is banned; "
                "consider converting to .json",
            )

        # 6. Python AST audit
        if path.suffix.lower() == ".py":
            py_viols = check_python_file(path)
            for v in py_viols:
                add(rel_str, v)

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    print("=" * 70)
    print("  NorgesGruppen Data — Security & Import Audit")
    print(f"  Scanned root: {root}")
    print("=" * 70)

    if not violations_by_file:
        print("\n[PASS] No violations found.\n")
        return True

    total = sum(len(v) for v in violations_by_file.values())
    print(f"\n[FAIL] {total} violation(s) in {len(violations_by_file)} file(s):\n")
    for filepath, msgs in sorted(violations_by_file.items()):
        print(f"  {filepath}")
        for msg in msgs:
            print(msg)
        print()

    print("=" * 70)
    print(f"  RESULT: FAIL  ({total} violation(s))")
    print("=" * 70)
    return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    clean = audit(project_root)
    sys.exit(0 if clean else 1)
