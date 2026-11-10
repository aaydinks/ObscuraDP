"""Writing whole numbers with exactly as many bits as they need.

A message of $d$ values of $k$ bits each takes $\\lceil d k / 8 \\rceil$ bytes. Byte counts
reported by the library are the lengths of these messages, not estimates.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

IntVector = npt.NDArray[np.uint64]


def packed_size(length: int, bits: int) -> int:
    """Bytes needed for ``length`` values of ``bits`` bits each."""
    if length < 0:
        raise ValueError("length must not be negative")
    _check_bits(bits)
    return math.ceil(length * bits / 8)


def float32_size(length: int) -> int:
    """Bytes of the same vector sent as 32-bit numbers: the baseline for the bytes target."""
    return 4 * length


def pack_bits(values: npt.ArrayLike, bits: int) -> bytes:
    """Pack whole numbers below $2^{bits}$ into ``packed_size`` bytes."""
    _check_bits(bits)
    numbers = np.asarray(values, dtype=np.uint64).ravel()
    if numbers.size and int(numbers.max()) >> bits:
        raise ValueError(f"a value does not fit in {bits} bits")
    shifts = np.arange(bits, dtype=np.uint64)
    table = ((numbers[:, None] >> shifts) & np.uint64(1)).astype(np.uint8)
    return np.packbits(table.ravel(), bitorder="little").tobytes()


def unpack_bits(data: bytes, bits: int, length: int) -> IntVector:
    """Inverse of ``pack_bits``."""
    _check_bits(bits)
    if len(data) != packed_size(length, bits):
        raise ValueError("the message has the wrong length for this many values")
    flat = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")[: length * bits]
    table = flat.reshape(length, bits).astype(np.uint64)
    weights = np.uint64(1) << np.arange(bits, dtype=np.uint64)
    return np.asarray((table * weights).sum(axis=1, dtype=np.uint64), dtype=np.uint64)


def _check_bits(bits: int) -> None:
    if not 1 <= bits <= 62:
        raise ValueError("bits must be between 1 and 62")
