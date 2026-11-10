"""Record or check fingerprints of the frozen research folders.

``Experiment 1`` to ``3`` and ``Trial 07/27`` are the reference the library is compared
against, so they must not change by accident. This script keeps a list of every tracked
file in those folders with a sha256 fingerprint of its content.

    python tools/freeze_research.py --check     # exit 1 if anything changed (default)
    python tools/freeze_research.py --write     # record the current state on purpose

Line endings are normalized before hashing, so a Windows checkout gives the same
fingerprints. Changing a research file is allowed only to fix a documented error: make the
change, run ``--write``, and explain the change in the pull request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

FROZEN = ("Experiment 1", "Experiment 2", "Experiment 3", "Trial 07")
MANIFEST = Path("tests/golden/research_manifest.json")


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
    )
    return Path(out.stdout.strip())


def fingerprint(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def tracked_files(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "-c", "core.quotepath=off", "ls-files", "-z", "--", *FROZEN],
        check=True,
        capture_output=True,
        cwd=root,
    )
    return sorted(name for name in out.stdout.decode("utf-8").split("\0") if name)


def current(root: Path) -> dict[str, str]:
    return {name: fingerprint(root / name) for name in tracked_files(root)}


def differences(recorded: dict[str, str], now: dict[str, str]) -> list[str]:
    lines = [f"changed: {name}" for name in recorded if name in now and now[name] != recorded[name]]
    lines += [f"missing: {name}" for name in recorded if name not in now]
    lines += [f"new, not recorded: {name}" for name in now if name not in recorded]
    return sorted(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="compare with the recorded list")
    group.add_argument("--write", action="store_true", help="record the current state")
    args = parser.parse_args()

    root = repo_root()
    now = current(root)
    manifest = root / MANIFEST
    if args.write:
        manifest.write_text(
            json.dumps({"files": now}, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"recorded {len(now)} files in {MANIFEST.as_posix()}")
        return 0

    recorded = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    problems = differences(recorded, now)
    for line in problems:
        print(line)
    if problems:
        print("The research folders are frozen. See the top of tools/freeze_research.py.")
        return 1
    print(f"research folders: {len(now)} files unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
