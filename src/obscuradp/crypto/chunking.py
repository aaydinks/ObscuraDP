"""Cutting a long vector into pieces that fit one ciphertext each."""

from __future__ import annotations

from collections.abc import Iterator


def chunk_bounds(n: int, slots: int) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` pairs that cover ``[0, n)`` in pieces of at most ``slots``."""
    if slots < 1:
        raise ValueError("slots must be at least 1")
    for start in range(0, n, slots):
        yield start, min(start + slots, n)


def chunk_count(n: int, slots: int) -> int:
    """Number of pieces ``chunk_bounds(n, slots)`` yields."""
    if slots < 1:
        raise ValueError("slots must be at least 1")
    return -(-max(n, 0) // slots)
