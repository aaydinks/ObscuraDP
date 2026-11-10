"""Run the Trial 07/27 research code as the reference ("oracle") for comparison tests.

The research package ``obscura`` lives in a folder whose name contains a space, is not
installed, and reads its data through paths relative to its own folder. The fixture below
puts that folder on ``sys.path``, imports the package, and offers ``call`` to run any of
its functions with that folder as the working directory.

These tests need the ``legacy`` extra and a full checkout of the repository. Without them
they are skipped, unless ``OBSCURADP_REQUIRE_GOLDEN=1`` is set (as it is in the CI job that
exists to run them), in which case a missing piece is an error.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import os
import sys
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from obscuradp.legacy import LegacyTrialConfig

ROOT = Path(__file__).resolve().parents[2]
TRIAL_DIR = ROOT / "Trial 07" / "27"
REQUIRED = os.environ.get("OBSCURADP_REQUIRE_GOLDEN") == "1"
EXTRAS = ("pandas", "sklearn", "imblearn", "opacus", "tenseal")


def _unavailable(reason: str) -> None:
    if REQUIRED:
        pytest.fail(f"comparison tests are required here but cannot run: {reason}")
    pytest.skip(reason)


class Oracle:
    """Handle on the research package and a way to call into it."""

    def __init__(self, package: ModuleType) -> None:
        self.package = package

    def module(self, name: str) -> ModuleType:
        return importlib.import_module(f"obscura.{name}")

    def call(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call a research function from inside the research folder, with warnings silenced."""
        with contextlib.chdir(TRIAL_DIR), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return function(*args, **kwargs)


def _check_available() -> None:
    if not (TRIAL_DIR / "obscura" / "__init__.py").is_file():
        _unavailable("the Trial 07/27 research folder is not present (not a full checkout)")
    missing = [name for name in EXTRAS if importlib.util.find_spec(name) is None]
    if missing:
        _unavailable(f"missing optional packages: {', '.join(missing)} (install the legacy extra)")


@pytest.fixture(scope="session")
def legacy_available() -> None:
    """Skip (or fail, when comparisons are required) if the research code cannot be run here."""
    _check_available()


@pytest.fixture(autouse=True)
def _comparison_environment(legacy_available: None) -> Iterator[None]:
    """Every test in this folder needs the research code's packages, and runs on one thread.

    One thread keeps floating-point sums in a fixed order, which bit-for-bit comparisons need.
    """
    import torch

    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope="session")
def oracle(legacy_available: None) -> Iterator[Oracle]:

    sys.path.insert(0, str(TRIAL_DIR))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            package = importlib.import_module("obscura")
            for name in ("config", "data", "models", "metrics", "aggregate", "he"):
                importlib.import_module(f"obscura.{name}")
        yield Oracle(package)
    finally:
        sys.path.remove(str(TRIAL_DIR))


@pytest.fixture(scope="session")
def cfg(legacy_available: None) -> LegacyTrialConfig:
    """The library's copy of the research settings, pointed at the research data files."""
    return LegacyTrialConfig(data_dir=TRIAL_DIR / "data")
