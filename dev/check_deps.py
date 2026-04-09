#!/usr/bin/env python3
"""
ODLabTracker dependency checker.
Scans all .py source files and reports:
  - Platform-specific imports (blocking)
  - Imports not declared in pyproject.toml (warning)
  - Declared deps never imported anywhere (warning)
  - Python version compatibility

Usage:
    python dev/check_deps.py
    SKIP_DEP_CHECK=1 git commit -m "..."   # bypass pre-commit hook
"""

import ast
import sys
import os
import re
from pathlib import Path
from datetime import datetime

# ── Python version check ──────────────────────────────────────────────────────
MIN_PYTHON = (3, 9)
if sys.version_info < MIN_PYTHON:
    print(f"ERROR: Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required "
          f"(running {sys.version_info.major}.{sys.version_info.minor})")
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent

SCAN_DIRS = [
    REPO_ROOT / "src",
    REPO_ROOT / "build" / "lib",
]
SCAN_ROOT_GLOBS = list(REPO_ROOT.glob("*.py"))  # root-level .py files

EXCLUDE_DIRS = {".ipynb_checkpoints", "__pycache__", ".git"}

# Packages known to be platform-specific
WINDOWS_ONLY = {
    "winreg", "winrt", "win32api", "win32con", "win32gui", "win32com",
    "pywintypes", "pythoncom", "pywin32", "wmi", "comtypes", "msvcrt",
    "ctypes.windll",
}
MAC_ONLY = {
    "AppKit", "Foundation", "Cocoa", "objc", "PyObjC",
    "tensorflow_macos", "tensorflow_metal",
}
LINUX_ONLY = {
    "termios", "tty", "pty", "fcntl", "grp", "pwd",
}

# Mapping from import name → pyproject.toml package name (where they differ)
IMPORT_TO_PACKAGE = {
    "cv2": "opencv-python",
    "PIL": "pillow",
    "PIL.Image": "pillow",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "iio": "imageio",
    "imageio": "imageio",
    "tifffile": "tifffile",
    "tp": "trackpy",
    "mpl": "matplotlib",
    "plt": "matplotlib",
    "np": "numpy",
    "pd": "pandas",
    "sp": "scipy",
    "readlif": "readlif",
    "ipyfilechooser": "ipyfilechooser",
    "av": "av",
    "imagecodecs": "imagecodecs",
}

# Stdlib modules to ignore (not third-party)
STDLIB = {
    "os", "sys", "re", "ast", "time", "math", "json", "csv", "io",
    "pathlib", "argparse", "subprocess", "multiprocessing", "functools",
    "itertools", "collections", "typing", "dataclasses", "abc", "copy",
    "datetime", "hashlib", "logging", "random", "string", "struct",
    "threading", "queue", "shutil", "tempfile", "glob", "fnmatch",
    "contextlib", "warnings", "traceback", "inspect", "importlib",
    "unittest", "enum", "signal", "socket", "urllib", "http", "email",
    "html", "xml", "zipfile", "tarfile", "gzip", "bz2", "lzma",
    "pickle", "shelve", "sqlite3", "platform", "textwrap", "pprint",
    "gc", "weakref", "array", "bisect", "heapq", "operator",
    "tkinter",  # stdlib but not always installed — checked separately below
    "IPython",  # treat as known/benign — not platform-specific
    "ODLabTracker",  # own package
}

# Packages declared in pyproject.toml as indirect backends (never imported directly)
# These are legitimate deps but won't appear in import scans
INDIRECT_DEPS = {
    "av",           # imageio video backend
    "imagecodecs",  # imageio codec backend
    "imageio-ffmpeg",  # imageio ffmpeg backend
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def collect_py_files():
    files = list(SCAN_ROOT_GLOBS)
    for d in SCAN_DIRS:
        if not d.exists():
            continue
        for f in d.rglob("*.py"):
            if not any(ex in f.parts for ex in EXCLUDE_DIRS):
                files.append(f)
    return sorted(set(files))


def extract_imports(filepath):
    """Return set of top-level module names imported in a file."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as e:
        return None, str(e)

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports, None


def parse_declared_deps():
    """Parse dependencies from pyproject.toml using regex (no extra deps needed)."""
    toml_path = REPO_ROOT / "pyproject.toml"
    text = toml_path.read_text(encoding="utf-8")

    # Extract the dependencies block
    match = re.search(r"dependencies\s*=\s*\[(.*?)\]", text, re.DOTALL)
    if not match:
        return set()

    block = match.group(1)
    deps = set()
    for line in block.splitlines():
        line = line.strip().strip('",').strip()
        if line and not line.startswith("#"):
            # Strip version specifiers: numpy>=1.0 → numpy
            name = re.split(r"[><=!;]", line)[0].strip().lower()
            if name:
                deps.add(name)
    return deps


def normalize_import(name):
    """Map import name to pyproject package name (lowercase)."""
    mapped = IMPORT_TO_PACKAGE.get(name)
    if mapped:
        return mapped.lower()
    return name.lower().replace("_", "-")


# ── Main check ────────────────────────────────────────────────────────────────

def run_check():
    files = collect_py_files()
    declared_deps = parse_declared_deps()

    all_imports = set()
    file_results = {}
    syntax_errors = {}
    platform_issues = []

    for f in files:
        imports, err = extract_imports(f)
        rel = f.relative_to(REPO_ROOT)
        if err:
            syntax_errors[str(rel)] = err
            continue

        third_party = imports - STDLIB
        file_results[str(rel)] = sorted(third_party)
        all_imports |= third_party

        # Check platform-specific
        win_hits = third_party & WINDOWS_ONLY
        mac_hits = third_party & MAC_ONLY
        linux_hits = third_party & LINUX_ONLY
        if win_hits or mac_hits or linux_hits:
            platform_issues.append((str(rel), win_hits, mac_hits, linux_hits))

    # Undeclared: imported but not in pyproject.toml
    undeclared = set()
    for imp in all_imports:
        pkg = normalize_import(imp)
        if pkg not in declared_deps and imp not in STDLIB:
            undeclared.add(imp)

    # Unused declared: in pyproject.toml but never imported (exclude known indirect backends)
    imported_normalized = {normalize_import(i) for i in all_imports}
    unused_declared = {
        d for d in declared_deps
        if d not in imported_normalized and d not in INDIRECT_DEPS
    }

    # Check if tkinter is used (stdlib but not always present)
    tkinter_used = "tkinter" in all_imports

    blocked = len(platform_issues) > 0

    return {
        "blocked": blocked,
        "platform_issues": platform_issues,
        "undeclared": sorted(undeclared),
        "unused_declared": sorted(unused_declared),
        "tkinter_used": tkinter_used,
        "file_results": file_results,
        "syntax_errors": syntax_errors,
        "declared_deps": sorted(declared_deps),
        "files_scanned": [str(f.relative_to(REPO_ROOT)) for f in files],
    }


def write_report(results):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    python_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    status = "BLOCKED" if results["blocked"] else "PASS"

    lines = [
        "# Dependency Report",
        f"",
        f"**Generated:** {now}  ",
        f"**Python:** {python_ver}  ",
        f"**Status:** {status}  ",
        "",
    ]

    # Platform issues (blocking)
    if results["platform_issues"]:
        lines += ["## Platform-Specific Imports (BLOCKING)", ""]
        for filepath, win, mac, linux in results["platform_issues"]:
            lines.append(f"### `{filepath}`")
            if win:
                lines.append(f"- **Windows-only:** {', '.join(sorted(win))}")
            if mac:
                lines.append(f"- **Mac-only:** {', '.join(sorted(mac))}")
            if linux:
                lines.append(f"- **Linux-only:** {', '.join(sorted(linux))}")
        lines.append("")
    else:
        lines += ["## Platform Compatibility", "", "No platform-specific imports detected.", ""]

    # Undeclared imports
    if results["undeclared"]:
        lines += ["## Undeclared Imports (WARNING)", ""]
        lines.append("Imported in code but not listed in `pyproject.toml`:")
        for pkg in results["undeclared"]:
            lines.append(f"- `{pkg}`")
        lines.append("")
    else:
        lines += ["## Undeclared Imports", "", "All imports are declared in `pyproject.toml`.", ""]

    # Unused declared deps
    if results["unused_declared"]:
        lines += ["## Unused Declared Dependencies (WARNING)", ""]
        lines.append("Listed in `pyproject.toml` but never imported in scanned files:")
        for pkg in results["unused_declared"]:
            lines.append(f"- `{pkg}`")
        lines.append("")
    else:
        lines += ["## Unused Declared Dependencies", "", "All declared dependencies are imported.", ""]

    # tkinter note
    if results["tkinter_used"]:
        lines += [
            "## tkinter (NOTE)",
            "",
            "`tkinter` is used and is part of the Python standard library, but is not always "
            "bundled on all platforms (notably some minimal Linux installs). "
            "Mac and Windows users should be unaffected.",
            "",
        ]

    # Syntax errors
    if results["syntax_errors"]:
        lines += ["## Syntax Errors", ""]
        for filepath, err in results["syntax_errors"].items():
            lines.append(f"- `{filepath}`: {err}")
        lines.append("")

    # Per-file breakdown
    lines += ["## Per-File Import Summary", ""]
    for filepath, imports in sorted(results["file_results"].items()):
        lines.append(f"### `{filepath}`")
        if imports:
            for imp in imports:
                lines.append(f"- `{imp}`")
        else:
            lines.append("- *(no third-party imports)*")
        lines.append("")

    # Files scanned
    lines += ["## Files Scanned", ""]
    for f in results["files_scanned"]:
        lines.append(f"- `{f}`")
    lines.append("")

    report_path = REPO_ROOT / "dependency_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def print_summary(results):
    print("\n── ODLabTracker Dependency Check ──────────────────────────")
    if results["blocked"]:
        print("STATUS: BLOCKED — platform-specific imports detected\n")
        for filepath, win, mac, linux in results["platform_issues"]:
            print(f"  {filepath}")
            if win:
                print(f"    Windows-only: {', '.join(sorted(win))}")
            if mac:
                print(f"    Mac-only:     {', '.join(sorted(mac))}")
            if linux:
                print(f"    Linux-only:   {', '.join(sorted(linux))}")
    else:
        print("STATUS: PASS — no platform-specific imports detected")

    if results["undeclared"]:
        print(f"\nWARNING: Undeclared imports: {', '.join(results['undeclared'])}")
    if results["unused_declared"]:
        print(f"WARNING: Unused declared deps: {', '.join(results['unused_declared'])}")
    if results["tkinter_used"]:
        print("NOTE: tkinter detected — stdlib but may be missing on minimal Linux installs")
    if results["syntax_errors"]:
        print(f"WARNING: Syntax errors in: {', '.join(results['syntax_errors'].keys())}")

    print(f"\nFiles scanned: {len(results['files_scanned'])}")
    print("────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    results = run_check()
    report_path = write_report(results)
    print_summary(results)
    print(f"Report written to: {report_path.relative_to(REPO_ROOT)}\n")
    sys.exit(1 if results["blocked"] else 0)
