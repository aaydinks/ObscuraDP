"""Whole numbers for a secure sum whose messages carry only a slice of the noise.

``Quantizer`` cuts values off at a declared range. That is harmless when every message
already carries all the noise its records need. It is not harmless when the noise is shared:
a participant whose values are cut off no longer contributes its slice, and the records of
the others are left with less noise than the budget assumes.

``SumGrid`` therefore never cuts off. A value is rounded to a multiple of the grid step, at
random but right on average, and written modulo $2^k$. Numbers that do not fit wrap around.
Wrapping cannot hurt privacy: the sum of wrapped numbers, wrapped again, is exactly the
wrapped true sum, so everything the server sees is computed from the true sum of the rounded
values. If a sum ever wraps, only accuracy suffers, and the range is planned so that this
practically never happens.

The grid is planned from public numbers only (see ``plan_sum_grid``): the size of one noise
slice and a limit on a participant's clipped sum. The step is a fixed fraction of one slice.
Rounding on so fine a grid changes what the sum reveals by an amount far below anything
measurable; ``docs/SHARED_NOISE.md`` gives the argument and what remains unproven until the
whole-number noise of a later version.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from obscuradp.compression.packing import packed_size
from obscuradp.compression.quantize import sum_bits

FloatVector = npt.NDArray[np.float64]
IntVector = npt.NDArray[np.uint64]

# One noise slice spans at least this many grid steps (standard deviation over step).
STEPS_PER_SHARE = 4.0
# Room for the noise beyond the limit of the clipped sum, in slice standard deviations.
TAIL_SHARES = 10.0
MAX_VALUE_BITS = 24


@dataclass(frozen=True)
class SumGrid:
    """Signed fixed-point numbers that add up modulo $2^k$.

    Args:
        step: The spacing of the grid in the original units.
        value_bits: Bits $b$ planned for one participant's numbers: the range
            $\\pm 2^{b-1}$ steps holds them except in the far tail.
        participants: Number of messages in one sum, $N$. The sum uses
            $k = b + \\lceil \\log_2 N \\rceil$ bits per number.
    """

    step: float
    value_bits: int
    participants: int

    def __post_init__(self) -> None:
        if not self.step > 0.0 or not math.isfinite(self.step):
            raise ValueError("step must be positive")
        if not 2 <= self.value_bits <= MAX_VALUE_BITS:
            raise ValueError(f"value_bits must be between 2 and {MAX_VALUE_BITS}")
        if self.participants < 1:
            raise ValueError("participants must be at least 1")

    @property
    def modulus_bits(self) -> int:
        """Bits per number on the wire, $k$."""
        return sum_bits(self.value_bits, self.participants)

    @property
    def reach(self) -> float:
        """The largest size of one participant's number that is certain not to wrap."""
        return float(2 ** (self.value_bits - 1)) * self.step

    def message_bytes(self, length: int) -> int:
        """Bytes one participant uploads for ``length`` numbers."""
        return packed_size(length, self.modulus_bits)

    def encode(self, values: npt.ArrayLike, generator: np.random.Generator) -> IntVector:
        """Round each value to a neighbouring grid point and write it modulo $2^k$."""
        x = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(x)):
            raise ValueError("values must be finite")
        position = x / self.step
        if x.size and float(np.max(np.abs(position))) >= 2.0**52:
            raise ValueError("a value is too large for this grid")
        below = np.floor(position)
        rounded = (below + (generator.random(x.shape) < position - below)).astype(np.int64)
        return np.asarray(rounded.astype(np.uint64) & self._mask, dtype=np.uint64)

    def decode_sum(self, total: npt.ArrayLike) -> FloatVector:
        """Turn a sum modulo $2^k$ back into the sum of the values.

        The result is read as a signed number: sums up to $\\pm 2^{k-1}$ steps come back
        exactly (up to the rounding of ``encode``).
        """
        numbers = np.asarray(total, dtype=np.uint64) & self._mask
        signed = numbers.astype(np.int64)
        half = np.int64(2 ** (self.modulus_bits - 1))
        signed = np.where(signed >= half, signed - np.int64(2**self.modulus_bits), signed)
        return np.asarray(signed * self.step, dtype=np.float64)

    def outside_share(self, values: npt.ArrayLike) -> float:
        """Share of one participant's values beyond the planned range (a diagnostic)."""
        x = np.asarray(values, dtype=np.float64)
        return float(np.mean(np.abs(x) >= self.reach)) if x.size else 0.0

    @property
    def _mask(self) -> np.uint64:
        return np.uint64(2**self.modulus_bits - 1)


def plan_sum_grid(share_std: float, sum_bound: float, participants: int) -> SumGrid:
    """The grid for a sum of ``participants`` messages, from public numbers only.

    The step is ``share_std / STEPS_PER_SHARE``. The range of one participant's numbers
    covers ``sum_bound`` (the limit of its clipped sum) plus ``TAIL_SHARES`` standard
    deviations of its noise slice, with the smallest number of bits that does.

    Args:
        share_std: Standard deviation of one participant's noise slice.
        sum_bound: Limit on the size of each number of one participant's clipped sum.
        participants: Number of messages in a sum.

    Raises:
        ValueError: If the range would need more than 24 bits per value. That happens when
            the noise is tiny compared with the sums; such a run is better served by a
            larger noise level or a smaller batch.
    """
    if not share_std > 0.0 or not math.isfinite(share_std):
        raise ValueError("share_std must be positive")
    if not sum_bound > 0.0 or not math.isfinite(sum_bound):
        raise ValueError("sum_bound must be positive")
    step = share_std / STEPS_PER_SHARE
    reach = sum_bound + TAIL_SHARES * share_std
    value_bits = max(2, math.ceil(math.log2(reach / step)) + 1)
    if value_bits > MAX_VALUE_BITS:
        raise ValueError(
            f"the shared noise is too small for the sums it protects: one message would need "
            f"{value_bits} bits per number (at most {MAX_VALUE_BITS} are supported)"
        )
    return SumGrid(step=step, value_bits=value_bits, participants=participants)
