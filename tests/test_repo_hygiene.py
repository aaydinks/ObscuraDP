"""Checks on the repository itself. Skipped when the tests run outside a git checkout."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_PACKAGES = ("obscura", "stroke_fl", "credit_fl", "fedml_he")

in_checkout = (ROOT / ".git").exists() and shutil.which("git") is not None
needs_checkout = pytest.mark.skipif(not in_checkout, reason="not running inside a git checkout")


@needs_checkout
def test_ignore_rules_hide_no_source() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_gitignore.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_library_does_not_import_research_code() -> None:
    pattern = re.compile(
        r"^\s*(?:from|import)\s+(" + "|".join(RESEARCH_PACKAGES) + r")(?:\.|\s|$)", re.MULTILINE
    )
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src").rglob("*.py")
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_no_em_dashes_in_library_or_docs() -> None:
    em_dash = chr(0x2014)
    checked = [*(ROOT / "src").rglob("*.py"), *(ROOT / "tests").rglob("*.py")]
    if (ROOT / "docs").is_dir():
        checked += list((ROOT / "docs").rglob("*.md"))
    checked += list(ROOT.glob("*.md"))
    offenders = [
        p.relative_to(ROOT).as_posix() for p in checked if em_dash in p.read_text(encoding="utf-8")
    ]
    assert offenders == []


@needs_checkout
def test_research_folders_are_unchanged() -> None:
    """The comparison tests are only meaningful if the reference itself never moves."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "freeze_research.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
