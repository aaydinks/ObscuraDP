"""Fetch public dataset files once, check them against a recorded fingerprint, keep them.

The library itself never opens a network connection. This module belongs to the benchmark
scripts and is the only place in the repository that downloads anything. A file is taken
from the cache if it is there and its sha256 fingerprint matches; otherwise it is
downloaded to a scratch name, checked, and only then moved into place. A file whose
fingerprint does not match is never used.

The cache is ``~/.cache/obscuradp-benchmarks`` unless the environment variable
``OBSCURADP_BENCHMARK_CACHE`` names another folder. It is outside the repository, so
nothing downloaded can be committed by accident.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

CACHE_VARIABLE = "OBSCURADP_BENCHMARK_CACHE"
_CHUNK = 1 << 20


@dataclass(frozen=True)
class Source:
    """One public file: where it is, what it must hash to, and where it comes from.

    Attributes:
        name: The file's name in the cache.
        url: Where to download it (https only).
        sha256: The fingerprint the file must have.
        size: Its size in bytes, for a quick first check.
        origin: Who publishes it and under which terms, in a sentence.
    """

    name: str
    url: str
    sha256: str
    size: int
    origin: str


class FingerprintMismatch(RuntimeError):
    """A file does not have the fingerprint that was recorded for it."""


def cache_directory() -> Path:
    """The folder downloaded files are kept in."""
    chosen = os.environ.get(CACHE_VARIABLE)
    return Path(chosen) if chosen else Path.home() / ".cache" / "obscuradp-benchmarks"


def fingerprint(path: Path) -> str:
    """The sha256 of a file, as hexadecimal text."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _open(url: str) -> BinaryIO:
    if not url.startswith("https://"):
        raise ValueError("only https addresses are downloaded")
    response: BinaryIO = urllib.request.urlopen(url, timeout=120)  # noqa: S310 - https only, checked above
    return response


def fetch(
    source: Source,
    directory: Path | None = None,
    *,
    opener: Callable[[str], BinaryIO] = _open,
) -> Path:
    """The local path of ``source``, downloading it first if the cache does not have it.

    Args:
        source: The file to get.
        directory: The cache folder (default: ``cache_directory()``).
        opener: How to open the address; replaced in tests, which must not use the network.

    Raises:
        FingerprintMismatch: If what was downloaded is not the recorded file. Nothing is
            kept in that case.
    """
    folder = directory or cache_directory()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / source.name
    if path.is_file() and path.stat().st_size == source.size and fingerprint(path) == source.sha256:
        return path
    scratch = path.with_name(path.name + ".part")
    with opener(source.url) as response, scratch.open("wb") as handle:
        shutil.copyfileobj(response, handle, _CHUNK)
    found = fingerprint(scratch)
    if scratch.stat().st_size != source.size or found != source.sha256:
        scratch.unlink()
        raise FingerprintMismatch(
            f"{source.name}: expected sha256 {source.sha256} ({source.size} bytes), got {found}"
        )
    scratch.replace(path)
    return path
