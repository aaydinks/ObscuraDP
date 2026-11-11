"""Finding the noise level that fits a total privacy budget.

The user states a budget for the whole training run. Calibration answers: how much noise
must every step carry so that all steps together stay inside it? The answer is found by
bisection and is always rounded towards more noise, so the stated budget is never exceeded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from obscuradp.accounting.accountants import epsilon_pld, epsilon_rdp
from obscuradp.accounting.events import SkellamEvent, SubsampledGaussianEvent

_SIGMA_FLOOR = 0.05
_SIGMA_CEILING = 1e4


@dataclass(frozen=True)
class Calibration:
    """A noise level and the budget it was certified to spend."""

    noise_multiplier: float
    sampling_rate: float
    steps: int
    target_epsilon: float
    delta: float
    epsilon: float
    accountant: str

    @property
    def event(self) -> SubsampledGaussianEvent:
        """The event describing the whole run at this noise level."""
        return SubsampledGaussianEvent(self.noise_multiplier, self.sampling_rate, self.steps)


def steps_per_round(local_epochs: float, sampling_rate: float) -> int:
    """Sampling draws in one round so that each record is expected ``local_epochs`` times.

    $S = \\lceil E / q \\rceil$.
    """
    if local_epochs <= 0:
        raise ValueError("local_epochs must be positive")
    if not 0.0 < sampling_rate <= 1.0:
        raise ValueError("sampling_rate must be in (0, 1]")
    return math.ceil(local_epochs / sampling_rate - 1e-12)


def total_steps(rounds: int, local_epochs: float, sampling_rate: float) -> int:
    """Sampling draws in the whole run: $T = R \\cdot \\lceil E / q \\rceil$."""
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    return rounds * steps_per_round(local_epochs, sampling_rate)


def calibrate_noise_multiplier(
    target_epsilon: float,
    delta: float,
    sampling_rate: float,
    steps: int,
    *,
    accountant: str = "pld",
    relative_tolerance: float = 1e-3,
) -> Calibration:
    """Smallest noise multiplier (within ``relative_tolerance``) whose total budget fits.

    See ``_calibrate`` for the details; results are cached because the search takes seconds.
    """
    return _calibrate(
        float(target_epsilon),
        float(delta),
        float(sampling_rate),
        int(steps),
        accountant,
        relative_tolerance,
    )


@lru_cache(maxsize=512)
def _calibrate(
    target_epsilon: float,
    delta: float,
    sampling_rate: float,
    steps: int,
    accountant: str,
    relative_tolerance: float,
) -> Calibration:
    """Smallest noise multiplier (within ``relative_tolerance``) whose total budget fits.

    The returned noise level is certified: the chosen accountant, run on exactly the
    returned event, reports a budget of at most ``target_epsilon``.

    Args:
        target_epsilon: Budget for all ``steps`` together.
        delta: The $\\delta$ of the budget.
        sampling_rate: Probability that a record joins a step.
        steps: Number of sampling draws in the whole run.
        accountant: ``"pld"`` (tight, default) or ``"rdp"`` (looser, more noise).
        relative_tolerance: How close to the smallest sufficient noise level to get.

    Raises:
        ValueError: If no noise level up to 10,000 fits the budget.
    """
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if accountant not in ("pld", "rdp"):
        raise ValueError("accountant must be 'pld' or 'rdp'")
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must be in (0, 1)")

    def spent_rdp(sigma: float) -> float:
        return epsilon_rdp([SubsampledGaussianEvent(sigma, sampling_rate, steps)], delta)

    def spent(sigma: float) -> float:
        if accountant == "rdp":
            return spent_rdp(sigma)
        return epsilon_pld([SubsampledGaussianEvent(sigma, sampling_rate, steps)], delta)

    # Bracket with the cheap RDP limit first. RDP needs a little more noise than PLD, so its
    # answer is a starting point from above; PLD is then only evaluated near plausible values.
    low, high = _SIGMA_FLOOR, 1.0
    while spent_rdp(high) > target_epsilon:
        low, high = high, high * 2.0
        if high > _SIGMA_CEILING:
            raise ValueError("no noise level up to 10,000 fits this budget; it is too small")
    while high / low > 1.0 + relative_tolerance:
        middle = math.sqrt(low * high)
        if spent_rdp(middle) > target_epsilon:
            low = middle
        else:
            high = middle

    if accountant == "pld":
        # Walk up if PLD does not accept the RDP answer (not expected), then walk down to
        # find a level PLD refuses, and bisect between the two.
        while spent(high) > target_epsilon:
            high *= 1.0 + 10.0 * relative_tolerance
            if high > _SIGMA_CEILING:
                raise ValueError("no noise level up to 10,000 fits this budget; it is too small")
        low = high
        while low > _SIGMA_FLOOR:
            low = max(_SIGMA_FLOOR, low * 0.9)
            if spent(low) > target_epsilon:
                break
            high = low
        while high / low > 1.0 + relative_tolerance and spent(low) > target_epsilon:
            middle = math.sqrt(low * high)
            if spent(middle) > target_epsilon:
                low = middle
            else:
                high = middle
            if high / low <= 1.0 + relative_tolerance:
                break
    return Calibration(
        noise_multiplier=high,
        sampling_rate=sampling_rate,
        steps=steps,
        target_epsilon=target_epsilon,
        delta=delta,
        epsilon=spent(high),
        accountant=accountant,
    )


@dataclass(frozen=True)
class SkellamCalibration:
    """A level of whole-number noise and the budget it was certified to spend.

    ``variance`` is the variance of the summed noise in every coordinate, in the units of the
    whole-number grid. ``noise_multiplier`` is its standard deviation over the length one
    record can move the sum, which makes it comparable with Gaussian noise.
    """

    variance: float
    l2_sensitivity: float
    l1_sensitivity: float
    steps: int
    target_epsilon: float
    delta: float
    epsilon: float
    accountant: str = "rdp"
    sampling_rate: float = 1.0

    @property
    def noise_multiplier(self) -> float:
        return math.sqrt(self.variance) / self.l2_sensitivity

    @property
    def event(self) -> SkellamEvent:
        """The event describing the whole run at this noise level."""
        return SkellamEvent(self.variance, self.l2_sensitivity, self.l1_sensitivity, self.steps)


def calibrate_skellam(
    target_epsilon: float,
    delta: float,
    steps: int,
    l2_sensitivity: float,
    l1_sensitivity: float,
    *,
    relative_tolerance: float = 1e-3,
) -> SkellamCalibration:
    """Smallest whole-number noise (within ``relative_tolerance``) whose total budget fits.

    Every record takes part in every one of the ``steps`` releases. The budget is counted
    with the Renyi curve of ``obscuradp.accounting.skellam``, the one proven method for this
    noise. The result is certified: that method, run on exactly the returned event, reports
    at most ``target_epsilon``.

    Raises:
        ValueError: If no noise multiplier up to 10,000 fits the budget.
    """
    return _calibrate_skellam(
        float(target_epsilon),
        float(delta),
        int(steps),
        float(l2_sensitivity),
        float(l1_sensitivity),
        relative_tolerance,
    )


@lru_cache(maxsize=512)
def _calibrate_skellam(
    target_epsilon: float,
    delta: float,
    steps: int,
    l2_sensitivity: float,
    l1_sensitivity: float,
    relative_tolerance: float,
) -> SkellamCalibration:
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must be in (0, 1)")

    def event(multiplier: float) -> SkellamEvent:
        variance = (multiplier * l2_sensitivity) ** 2
        return SkellamEvent(variance, l2_sensitivity, l1_sensitivity, steps)

    def spent(multiplier: float) -> float:
        return epsilon_rdp([event(multiplier)], delta)

    low, high = _SIGMA_FLOOR, 1.0
    while spent(high) > target_epsilon:
        low, high = high, high * 2.0
        if high > _SIGMA_CEILING:
            raise ValueError("no noise level up to 10,000 fits this budget; it is too small")
    while high / low > 1.0 + relative_tolerance:
        middle = math.sqrt(low * high)
        if spent(middle) > target_epsilon:
            low = middle
        else:
            high = middle
    chosen = event(high)
    return SkellamCalibration(
        variance=chosen.variance,
        l2_sensitivity=l2_sensitivity,
        l1_sensitivity=l1_sensitivity,
        steps=steps,
        target_epsilon=target_epsilon,
        delta=delta,
        epsilon=spent(high),
    )
