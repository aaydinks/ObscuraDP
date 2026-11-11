"""Whole-number noise for the shared-noise mode: rounding, the exact length check, the noise.

With Gaussian noise every participant rounds its noisy numbers before the secure sum, and
that this rounding is harmless has to be argued (``docs/SHARED_NOISE.md``). Here the order is
turned around and nothing has to be argued:

1. Each record's clipped gradient is rounded to whole numbers on its own.
2. Its length is then checked with exact whole-number arithmetic against a fixed limit. So
   the most one record can move the sum is known exactly, whatever floating-point arithmetic
   did before.
3. The noise is whole numbers too (Skellam noise, the difference of two Poisson counts).
   Slices of it add up to noise of the same kind, exactly.

Everything after step 1 is whole-number arithmetic. There are no low-order bits of a
floating-point number for an attacker to inspect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

IntMatrix = npt.NDArray[np.int64]

# Rounding may lengthen a vector. With this chance of failure per try the limit below holds,
# and a record that fails is simply rounded again.
_BETA = math.exp(-0.5)
# The planner makes the grid fine enough that rounding lengthens a record by at most this.
MAX_INFLATION = 1.05
_ROUNDING_TRIES = 16


def rounded_length_limit(scale: float, dimension: int) -> int:
    """The squared length a rounded record may have, as a whole number.

    A vector of length at most $c$ (``scale``) with $m$ numbers (``dimension``), each rounded
    at random to a neighbouring whole number, has squared length at most

    $$\\min\\Bigl((c + \\sqrt{m})^2,\\ c^2 + \\frac{m}{4}
    + \\sqrt{2 \\ln(1/\\beta)}\\,\\bigl(c + \\tfrac{\\sqrt{m}}{2}\\bigr)\\Bigr)$$

    always (first term) or with probability at least $1 - \\beta$ (second term; Kairouz, Liu
    and Steinke, 2021, on the distributed discrete Gaussian mechanism). The limit only decides
    how often a record has to be rounded again. That no record exceeds it is enforced by the
    exact check in ``round_records``, not by this formula.
    """
    if not scale > 0.0 or not math.isfinite(scale):
        raise ValueError("scale must be positive")
    if dimension < 1:
        raise ValueError("dimension must be at least 1")
    root = math.sqrt(dimension)
    always = (scale + root) ** 2
    usually = (
        scale**2 + dimension / 4.0 + math.sqrt(2.0 * math.log(1.0 / _BETA)) * (scale + root / 2.0)
    )
    return max(1, math.floor(min(always, usually)))


@dataclass(frozen=True)
class IntegerStep:
    """The whole-number grid of one run, planned from public numbers only.

    Attributes:
        scale: Grid points per clipping norm, $c$. A clipped gradient of length $C$ becomes a
            vector of length about $c$ in whole numbers.
        dimension: Number of model numbers, $m$.
        l2_squared: The squared length no rounded record may exceed (a whole number).
    """

    scale: float
    dimension: int
    l2_squared: int

    @property
    def l2_sensitivity(self) -> float:
        """$\\Delta_2$: the most one record can change the whole-number sum, in length."""
        return math.sqrt(self.l2_squared)

    @property
    def l1_sensitivity(self) -> float:
        """$\\Delta_1$: the most one record can change the sum of absolute values.

        For whole numbers $|x| \\le x^2$, so $\\Delta_1 \\le \\Delta_2^2$; and always
        $\\Delta_1 \\le \\sqrt{m}\\,\\Delta_2$.
        """
        return float(min(self.l2_squared, math.sqrt(self.dimension) * self.l2_sensitivity))

    @property
    def inflation(self) -> float:
        """How much longer a rounded record may be than the clipping norm, as a factor."""
        return self.l2_sensitivity / self.scale


def plan_integer_step(dimension: int, max_inflation: float = MAX_INFLATION) -> IntegerStep:
    """The coarsest grid on which rounding lengthens a record by at most ``max_inflation``.

    A finer grid costs more bits per number; a coarser one needs more noise, because the
    noise has to cover the rounded length. Only the size of the model enters.
    """
    if dimension < 1:
        raise ValueError("dimension must be at least 1")
    if not max_inflation > 1.0:
        raise ValueError("max_inflation must be above 1")
    low, high = 1.0, 2.0
    while math.sqrt(rounded_length_limit(high, dimension)) / high > max_inflation:
        low, high = high, high * 2.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        if math.sqrt(rounded_length_limit(middle, dimension)) / middle > max_inflation:
            low = middle
        else:
            high = middle
    scale = float(math.ceil(high))
    return IntegerStep(scale, dimension, rounded_length_limit(scale, dimension))


def round_records(
    records: npt.ArrayLike, l2_squared: int, generator: np.random.Generator
) -> IntMatrix:
    """Round each row to whole numbers so that no row is longer than the limit. Always.

    Every number is rounded to a neighbouring whole number, at random and right on average.
    The squared length of each rounded row is then computed in whole-number arithmetic. A row
    above ``l2_squared`` is rounded again; after several tries it is rounded towards zero,
    which cannot lengthen it; and a row that is still too long (its input was longer than
    the limit to begin with) contributes nothing. So the limit holds for every row, whatever
    the input.

    Args:
        records: One row per record, already scaled to grid units.
        l2_squared: The largest allowed squared length of a rounded row.
        generator: The random stream of the rounding.
    """
    values = np.asarray(records, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("records must be a matrix with one row per record")
    if l2_squared < 1:
        raise ValueError("l2_squared must be at least 1")
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError("records must be finite")
    if values.size and float(np.max(np.abs(values))) >= 2.0**31:
        raise ValueError("a record is too large for the whole-number grid")
    below = np.floor(values)
    fraction = values - below
    rounded = np.zeros(values.shape, dtype=np.int64)
    pending = np.arange(values.shape[0])
    for _ in range(_ROUNDING_TRIES):
        if pending.size == 0:
            break
        up = generator.random((pending.size, values.shape[1])) < fraction[pending]
        candidate = (below[pending] + up).astype(np.int64)
        fits = (candidate * candidate).sum(axis=1) <= l2_squared
        rounded[pending[fits]] = candidate[fits]
        pending = pending[~fits]
    if pending.size:
        towards_zero = np.trunc(values[pending]).astype(np.int64)
        fits = (towards_zero * towards_zero).sum(axis=1) <= l2_squared
        rounded[pending[fits]] = towards_zero[fits]
    return rounded


def skellam_noise(
    shape: int | tuple[int, ...], variance: float, generator: np.random.Generator
) -> IntMatrix:
    """Independent Skellam noise with mean 0 and the given variance, as 64-bit whole numbers.

    The difference of two independent Poisson counts with mean ``variance / 2`` each. Noises
    of this kind add up to noise of the same kind with the variances added.
    """
    if not variance >= 0.0 or not math.isfinite(variance):
        raise ValueError("variance must not be negative")
    if variance == 0.0:
        return np.zeros(shape, dtype=np.int64)
    half = variance / 2.0
    plus = generator.poisson(half, shape).astype(np.int64)
    minus = generator.poisson(half, shape).astype(np.int64)
    return np.asarray(plus - minus, dtype=np.int64)
