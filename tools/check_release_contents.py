"""Fail if a built release contains anything that must never be shipped.

Datasets, experiment folders, result files, the website and the project's internal documents
stay in the repository. A release holds the library, its tests and a few top-level files.

    python tools/check_release_contents.py dist
"""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path

FORBIDDEN_DIRS = ("Trial 07", "website", "benchmarks", "node_modules", "results", "data", "docs")
FORBIDDEN_SUFFIXES = (".csv", ".parquet", ".pt", ".pth", ".pkl", ".npz", ".png")
WHEEL_ALLOWED_TOP = ("obscuradp/", "obscuradp-")


def names_in(archive: Path) -> list[str]:
    if archive.suffix == ".whl":
        with zipfile.ZipFile(archive) as zf:
            return zf.namelist()
    with tarfile.open(archive) as tf:
        return tf.getnames()


def problems(archive: Path) -> list[str]:
    found = []
    for name in names_in(archive):
        inner = name.split("/", 1)[1] if archive.suffix != ".whl" and "/" in name else name
        parts = inner.split("/")[:-1]
        if any(part in FORBIDDEN_DIRS or part.startswith("Experiment ") for part in parts):
            found.append(f"{archive.name}: forbidden path {name}")
        elif inner.lower().endswith(FORBIDDEN_SUFFIXES):
            found.append(f"{archive.name}: data or binary file {name}")
        elif archive.suffix == ".whl" and not inner.startswith(WHEEL_ALLOWED_TOP):
            found.append(f"{archive.name}: unexpected top-level entry {name}")
    return found


def main(argv: list[str]) -> int:
    dist = Path(argv[1] if len(argv) > 1 else "dist")
    archives = sorted([*dist.glob("*.whl"), *dist.glob("*.tar.gz")])
    if not archives:
        print(f"no archives found in {dist}")
        return 1
    bad = [p for archive in archives for p in problems(archive)]
    for line in bad:
        print(line)
    if bad:
        return 1
    for archive in archives:
        print(f"{archive.name}: {len(names_in(archive))} files, ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
