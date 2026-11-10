"""Rounding an update to small whole numbers.

A secure sum adds whole numbers, not floating-point values, and fewer bits per number means
smaller messages. ``Quantizer`` cuts each value off at a declared range and rounds it to one
of $2^b$ levels. The rounding is random in such a way that it is right on average, so the
sum of many rounded updates is an unbiased estimate of the true sum.

Rounding happens after any privacy noise has been added. It is post-processing and cannot
weaken a privacy guarantee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatVector = npt.NDArray[np.float64]
IntVector = npt.NDArray[np.uint64]


def sum_bits(value_bits: int, participants: int) -> int:
    """Bits needed to add ``participants`` values of ``value_bits`` bits without overflow.

    $k = b + \\lceil \\log_2 N \\rceil$.
    """
    if value_bits < 1:
        raise ValueError("value_bits must be at least 1")
    if participants < 1:
        raise ValueError("participants must be at least 1")
    return value_bits + math.ceil(math.log2(participants))


@dataclass(frozen=True)
class Quantizer:
    """Maps values in $[-R, R]$ to whole numbers $0 \\dots 2^b - 1$ and sums of them back.

    Args:
        bits: Number of bits $b$ per value (2 to 24).
        value_range: $R$. Values outside $[-R, R]$ are cut off at the range.
    """

    bits: int
    value_range: float

    def __post_init__(self) -> None:
        if not 2 <= self.bits <= 24:
            raise ValueError("bits must be between 2 and 24")
        if not self.value_range > 0 or not math.isfinite(self.value_range):
            raise ValueError("value_range must be positive")

    @property
    def levels(self) -> int:
        """The largest whole number a value can become: $2^b - 1$."""
        return int(2**self.bits - 1)

    @property
    def step(self) -> float:
        """The spacing of the levels in the original units."""
        return 2.0 * self.value_range / self.levels

    def quantize(self, values: npt.ArrayLike, generator: np.random.Generator) -> IntVector:
        """Round each value to a neighbouring level, at random but right on average."""
        x = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(x)):
            raise ValueError("values must be finite")
        position = (np.clip(x, -self.value_range, self.value_range) + self.value_range) / self.step
        below = np.floor(position)
        rounded = below + (generator.random(x.shape) < position - below)
        return np.asarray(np.clip(rounded, 0, self.levels), dtype=np.uint64)

    def dequantize_sum(self, total: npt.ArrayLike, count: int) -> FloatVector:
        """Turn the sum of ``count`` quantized vectors back into the sum of the values."""
        if count < 1:
            raise ValueError("count must be at least 1")
        summed = np.asarray(total, dtype=np.float64)
        return np.asarray(summed * self.step - count * self.value_range, dtype=np.float64)

    def saturated_share(self, values: npt.ArrayLike) -> float:
        """Share of values that lie outside the range and would be cut off."""
        x = np.asarray(values, dtype=np.float64)
        return float(np.mean(np.abs(x) > self.value_range)) if x.size else 0.0
