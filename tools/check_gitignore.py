"""Fail if an ignore rule would hide source, tests, docs or benchmark files.

The root .gitignore once hid ``website/src/lib`` because of an unanchored ``lib/`` rule.
This script asks git about a set of probe paths that must never be ignored, and about
every real file under the protected folders.

Run from anywhere inside the repository:  python tools/check_gitignore.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROTECTED = ("src", "tests", "docs", "benchmarks", "tools", "constraints")

# Folder names that the stock Python ignore template hides when its rules are not anchored.
RISKY_NAMES = (
    "build",
    "dist",
    "lib",
    "lib64",
    "env",
    "venv",
    "eggs",
    "parts",
    "sdist",
    "var",
    "wheels",
    "downloads",
    "target",
    "instance",
    "cover",
    "data",
    "datasets",
    "results",
)


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
    )
    return Path(out.stdout.strip())


def ignored(root: Path, paths: list[str]) -> list[str]:
    """Return the subset of ``paths`` that git would ignore."""
    if not paths:
        return []
    proc = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        input="\n".join(paths),
        capture_output=True,
        text=True,
        cwd=root,
        check=False,
    )
    if proc.returncode not in (0, 1):  # 0: some ignored, 1: none ignored
        raise RuntimeError(proc.stderr.strip() or "git check-ignore failed")
    return [line for line in proc.stdout.splitlines() if line]


def probe_paths() -> list[str]:
    probes = []
    for top in PROTECTED:
        for name in RISKY_NAMES:
            probes.append(f"{top}/{name}/probe.py")
            probes.append(f"{top}/pkg/{name}/probe.json")
    probes.append("src/obscuradp/lib/probe.py")
    return probes


def existing_files(root: Path) -> list[str]:
    skip = {"__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"}
    files = []
    for top in PROTECTED:
        base = root / top
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and not (skip & set(path.parts)) and path.suffix != ".pyc":
                files.append(path.relative_to(root).as_posix())
    return files


def main() -> int:
    root = repo_root()
    bad = ignored(root, probe_paths() + existing_files(root))
    if bad:
        print("These paths would be hidden by an ignore rule:")
        for path in bad:
            print(f"  {path}")
        print("Anchor the offending rule in .gitignore to the repository root (for example /lib/).")
        return 1
    print("ignore rules: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
