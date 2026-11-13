"""Checks that the written plan came first and has not been changed since.

The plan (``benchmarks/PREREGISTRATION.md``) was added by one commit. Its text above the
line "Amendments" may never change; later decisions go below that line with a date. Every
record of a run stores the commit of the code that made it, and that commit has to come
after the plan's.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "benchmarks" / "PREREGISTRATION.md"
PLAN_COMMIT = "ad2289c208ea4387116d5d37732a4745714b37e8"
AMENDMENTS = "## Amendments\n"
# sha256 of the plan's text up to and including the line "## Amendments", Unix line ends.
PLAN_SHA256 = "87e4c3b26b1240a69eb7fb1288a8889b8f89ed23357ec57d2a499ad93a197313"


def fixed_part(text: str) -> str:
    """The part of the plan that may not change: everything up to the amendments."""
    if AMENDMENTS not in text:
        raise ValueError('the plan has no line "## Amendments"')
    return text[: text.index(AMENDMENTS) + len(AMENDMENTS)]


def fingerprint(text: str) -> str:
    return hashlib.sha256(fixed_part(text).encode("utf-8")).hexdigest()


def plan_is_unchanged(path: Path = PLAN) -> bool:
    """Whether the plan's text above the amendments is what the plan's commit added."""
    return fingerprint(path.read_text(encoding="utf-8")) == PLAN_SHA256


def comes_after_plan(commit: str | None, directory: Path = ROOT) -> bool | None:
    """Whether ``commit`` has the plan's commit in its history.

    ``None`` if that cannot be told here: no git, or a copy of the repository without its
    history. A commit marked ``+dirty`` (changed files that were not committed) is never
    accepted: its record does not name the code that made it.
    """
    if not commit or commit.endswith("+dirty"):
        return False
    git = shutil.which("git")
    if git is None:
        return None
    try:
        for name in (PLAN_COMMIT, commit):
            known = subprocess.run(  # noqa: S603 - fixed arguments, no shell
                [git, "cat-file", "-e", f"{name}^{{commit}}"],
                capture_output=True,
                cwd=directory,
                check=False,
                timeout=10,
            )
            if known.returncode != 0:
                return None
        answer = subprocess.run(  # noqa: S603 - fixed arguments, no shell
            [git, "merge-base", "--is-ancestor", PLAN_COMMIT, commit],
            capture_output=True,
            cwd=directory,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return answer.returncode == 0
