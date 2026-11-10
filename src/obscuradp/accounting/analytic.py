"""Exact privacy of plain Gaussian noise (no subsampling).

For Gaussian noise with noise multiplier $\\sigma$ (standard deviation over sensitivity) the
exact trade-off between $\\varepsilon$ and $\\delta$ is known in closed form (Balle and Wang,
2018):

    delta(eps) = Phi(1/(2 sigma) - eps sigma) - exp(eps) Phi(-1/(2 sigma) - eps sigma)

Repeating the mechanism $k$ times is exactly one Gaussian mechanism with noise multiplier
$\\sigma / \\sqrt{k}$. These formulas are the anchor the numerical accountants are tested
against.
"""

from __future__ import annotations

import math


def _phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def gaussian_delta(noise_multiplier: float, epsilon: float, count: int = 1) -> float:
    """Exact $\\delta$ of ``count`` Gaussian releases at a given $\\varepsilon$."""
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive")
    if count < 1:
        raise ValueError("count must be at least 1")
    sigma = noise_multiplier / math.sqrt(count)
    a = 1.0 / (2.0 * sigma)
    b = epsilon * sigma
    # exp(eps) * Phi(-a - b) can overflow one factor while the product is tiny: use logs.
    tail = _phi(-a - b)
    second = math.exp(epsilon + math.log(tail)) if tail > 0.0 else 0.0
    return max(0.0, _phi(a - b) - second)


def gaussian_epsilon(noise_multiplier: float, delta: float, count: int = 1) -> float:
    """Exact budget of ``count`` Gaussian releases at a given $\\delta$ (0 if none is needed)."""
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if gaussian_delta(noise_multiplier, 0.0, count) <= delta:
        return 0.0
    low, high = 0.0, 1.0
    while gaussian_delta(noise_multiplier, high, count) > delta:
        high *= 2.0
        if high > 1e6:
            raise ValueError("no finite epsilon found; the noise is too small")
    for _ in range(200):
        middle = 0.5 * (low + high)
        if gaussian_delta(noise_multiplier, middle, count) > delta:
            low = middle
        else:
            high = middle
        if high - low < 1e-12 * max(1.0, high):
            break
    return high


def gaussian_total_variation(noise_multiplier: float, count: int = 1) -> float:
    """Largest possible advantage of any attacker against ``count`` Gaussian releases.

    This is $\\delta$ at $\\varepsilon = 0$: the true-positive rate minus the false-positive
    rate of the best possible test for "was this record used".
    """
    return gaussian_delta(noise_multiplier, 0.0, count)
