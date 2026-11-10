"""Result records that say where they came from.

A number is only as trustworthy as the trail behind it. Every result this library writes
carries a manifest: which code produced it (package version and, in a checkout, the git
commit), on what (Python, operating system, CPU count, threads, library versions), from
which settings (a fingerprint of the configuration) and with which seeds. With that, a
result file can be regenerated and compared; without it, it can only be believed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCHEMA = "obscuradp.result/1"

# Libraries whose versions can change numerical results. Missing ones are recorded as None.
_TRACKED_PACKAGES = (
    "obscuradp",
    "numpy",
    "torch",
    "dp-accounting",
    "opacus",
    "tenseal",
    "scikit-learn",
    "pandas",
    "scipy",
    "imbalanced-learn",
    "cryptography",
    "flwr",
)


def to_jsonable(obj: Any) -> Any:
    """Convert numpy and torch values, dataclasses, paths and mappings to plain JSON types.

    NaN and infinity become ``None``: strict JSON has no way to write them.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, Mapping):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, torch.Tensor):
        return to_jsonable(obj.detach().cpu().numpy())
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, Path):
        return obj.as_posix()
    if obj is None or isinstance(obj, (int, str)):
        return obj
    raise TypeError(f"cannot store a value of type {type(obj).__name__} in a result")


def config_hash(config: Any) -> str:
    """A sha256 fingerprint of a configuration, independent of key order."""
    canonical = json.dumps(to_jsonable(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_commit(directory: str | os.PathLike[str] | None = None) -> str | None:
    """The current git commit of ``directory`` (default: the working directory), if any.

    A trailing ``+dirty`` marks uncommitted changes. Returns ``None`` outside a checkout or
    when git is not installed. This reads local files only; nothing is sent anywhere.
    """
    git = shutil.which("git")
    if git is None:
        return None
    cwd = os.fspath(directory) if directory is not None else None
    try:
        head = subprocess.run(  # noqa: S603 - fixed arguments, no shell
            [git, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            check=False,
            timeout=10,
        )
        if head.returncode != 0:
            return None
        status = subprocess.run(  # noqa: S603 - fixed arguments, no shell
            [git, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            cwd=cwd,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = head.stdout.strip()
    return f"{commit}+dirty" if status.returncode == 0 and status.stdout.strip() else commit


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_manifest(directory: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Describe the code and machine a result is being produced on."""
    return {
        "git_commit": git_commit(directory),
        "packages": _package_versions(),
        "python": platform.python_version(),
        "implementation": sys.implementation.name,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_threads": int(torch.get_num_threads()),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def make_result(
    *,
    kind: str,
    config: Any,
    seeds: Sequence[int],
    data: Mapping[str, Any],
    directory: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Assemble a result record: schema tag, manifest, settings, seeds and the results.

    Args:
        kind: A short name for what was run, for example ``"arm_matrix"``.
        config: The settings of the run. Stored in full and fingerprinted.
        seeds: Every seed the run used.
        data: The results themselves.
        directory: Where to look for a git checkout (default: the working directory).
    """
    if not kind:
        raise ValueError("kind must not be empty")
    stored_config = to_jsonable(config)
    return {
        "schema": SCHEMA,
        "kind": kind,
        "manifest": {
            **environment_manifest(directory),
            "config_sha256": config_hash(stored_config),
            "seeds": [int(seed) for seed in seeds],
        },
        "config": stored_config,
        "data": to_jsonable(data),
    }


def validate_result(record: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` if ``record`` is not a well-formed, self-consistent result."""
    if record.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} record (schema is {record.get('schema')!r})")
    for key in ("kind", "manifest", "config", "data"):
        if key not in record:
            raise ValueError(f"result record is missing {key!r}")
    manifest = record["manifest"]
    for key in (
        "git_commit",
        "packages",
        "python",
        "platform",
        "cpu_count",
        "config_sha256",
        "seeds",
    ):
        if key not in manifest:
            raise ValueError(f"result manifest is missing {key!r}")
    if manifest["config_sha256"] != config_hash(record["config"]):
        raise ValueError("the stored settings do not match their fingerprint")


def write_result(path: str | os.PathLike[str], record: Mapping[str, Any]) -> None:
    """Validate ``record`` and write it as strict, sorted JSON."""
    validate_result(record)
    text = json.dumps(record, indent=1, sort_keys=True, allow_nan=False)
    Path(path).write_text(text + "\n", encoding="utf-8")


def read_result(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a result file and check that it is well formed."""
    record: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_result(record)
    return record
