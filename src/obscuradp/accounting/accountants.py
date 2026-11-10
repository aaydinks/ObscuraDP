"""Turning a list of events into a privacy budget, three independent ways.

- **PLD** (privacy loss distributions, from Google's ``dp-accounting``) is the tight one and
  the one the library calibrates noise with. It can report an upper estimate (never below the
  truth, up to its discretization) and a lower estimate.
- **RDP** (Renyi differential privacy, the older method Opacus uses by default) is always an
  upper limit, and a looser one. It serves as an independent cross-check.
- For plain Gaussian noise without subsampling the exact answer is known (see ``analytic``).

All functions use the add-or-remove-one-record neighbouring relation.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from obscuradp.accounting.events import (
    Event,
    GaussianEvent,
    SubsampledGaussianEvent,
    merge_events,
)

# Grid spacing of the PLD accountant for the upper estimate. Smaller is tighter and slower.
DISCRETIZATION = 1e-4
# The lower estimate uses a cruder algorithm whose error grows with the number of repetitions,
# so its grid is refined as the run gets longer (bounded so that it stays fast).
_LOWER_ERROR_BUDGET = 0.05
_FINEST_DISCRETIZATION = 2e-6
# Budgets above this are reported as infinite: they promise nothing and cost gigabytes to compute.
MEANINGLESS_EPSILON = 200.0


@contextmanager
def _quiet_library_logging() -> Iterator[None]:
    """dp-accounting reports routine choices through the root logger; keep them out of user logs."""
    root = logging.getLogger()
    absl = logging.getLogger("absl")
    before = (root.level, absl.level)
    root.setLevel(logging.ERROR)
    absl.setLevel(logging.ERROR)
    try:
        yield
    finally:
        root.setLevel(before[0])
        absl.setLevel(before[1])


def _event_parts(event: Event) -> tuple[float, float, int]:
    if isinstance(event, SubsampledGaussianEvent):
        return event.noise_multiplier, event.sampling_rate, event.steps
    if isinstance(event, GaussianEvent):
        return event.noise_multiplier, 1.0, event.count
    raise TypeError(f"unsupported event type {type(event).__name__}")


@lru_cache(maxsize=256)
def _pld_single(
    noise_multiplier: float,
    sampling_rate: float,
    repetitions: int,
    pessimistic: bool,
    interval: float,
) -> Any:
    from dp_accounting.pld import privacy_loss_distribution as pldlib

    with _quiet_library_logging():
        base = pldlib.from_gaussian_mechanism(
            standard_deviation=noise_multiplier,
            sensitivity=1.0,
            pessimistic_estimate=pessimistic,
            value_discretization_interval=interval,
            sampling_prob=sampling_rate,
            use_connect_dots=pessimistic,
        )
        return base.self_compose(repetitions) if repetitions > 1 else base


def _interval(merged: Sequence[Event], pessimistic: bool) -> float:
    if pessimistic:
        return DISCRETIZATION
    repetitions = sum(event.repetitions for event in merged)
    return min(
        DISCRETIZATION, max(_FINEST_DISCRETIZATION, _LOWER_ERROR_BUDGET / max(1, repetitions))
    )


def _pld(events: Sequence[Event], pessimistic: bool) -> Any:
    merged = merge_events(events)
    if not merged:
        return None
    interval = _interval(merged, pessimistic)
    with _quiet_library_logging():
        total = None
        for event in merged:
            sigma, rate, repetitions = _event_parts(event)
            part = _pld_single(sigma, rate, repetitions, pessimistic, interval)
            total = part if total is None else total.compose(part)
        return total


def epsilon_pld(events: Sequence[Event], delta: float, *, pessimistic: bool = True) -> float:
    """Budget of ``events`` at ``delta`` by privacy loss distributions.

    With ``pessimistic=True`` (the default, used for all guarantees) the result errs on the
    high side. With ``pessimistic=False`` it errs on the low side; the truth lies between.
    """
    _check_delta(delta)
    merged = merge_events(events)
    if not merged:
        return 0.0
    return _epsilon_pld(merged, delta, pessimistic)


@lru_cache(maxsize=1024)
def _epsilon_pld(merged: tuple[Event, ...], delta: float, pessimistic: bool) -> float:
    """Cached by the merged releases: many participants and rounds repeat the same settings."""
    if _beyond_meaning(merged, delta):
        return math.inf
    total = _pld(merged, pessimistic)
    with _quiet_library_logging():
        return float(total.get_epsilon_for_delta(delta))


def delta_pld(events: Sequence[Event], epsilon: float, *, pessimistic: bool = True) -> float:
    """$\\delta$ of ``events`` at a given $\\varepsilon$ by privacy loss distributions."""
    if epsilon < 0:
        raise ValueError("epsilon must not be negative")
    if not merge_events(events):
        return 0.0
    if _beyond_meaning(events, 1e-5):
        return 1.0
    total = _pld(events, pessimistic)
    with _quiet_library_logging():
        return float(total.get_delta_for_epsilon(epsilon))


def _beyond_meaning(events: Sequence[Event], delta: float) -> bool:
    """True if the budget is so large that it protects nothing and is costly to compute exactly.

    The PLD accountant needs memory in proportion to the size of the budget. A run whose RDP
    limit is above ``MEANINGLESS_EPSILON`` is reported as unbounded without running it.
    """
    return epsilon_rdp(events, delta) > MEANINGLESS_EPSILON


def epsilon_rdp(events: Sequence[Event], delta: float) -> float:
    """Budget of ``events`` at ``delta`` by Renyi differential privacy (an upper limit)."""
    _check_delta(delta)
    merged = merge_events(events)
    if not merged:
        return 0.0
    return _epsilon_rdp(merged, delta)


@lru_cache(maxsize=1024)
def _epsilon_rdp(merged: tuple[Event, ...], delta: float) -> float:
    import dp_accounting as dpa
    from dp_accounting import rdp

    with _quiet_library_logging():
        accountant = rdp.RdpAccountant()
        for event in merged:
            sigma, rate, repetitions = _event_parts(event)
            base: Any = dpa.GaussianDpEvent(sigma)
            if rate < 1.0:
                base = dpa.PoissonSampledDpEvent(rate, base)
            accountant.compose(dpa.SelfComposedDpEvent(base, repetitions))
        value = float(accountant.get_epsilon(delta))
    return math.inf if math.isnan(value) else value


def attacker_advantage_ceiling(events: Sequence[Event]) -> float:
    """Largest advantage any attacker can have against exactly these events.

    Advantage is the true-positive rate minus the false-positive rate when guessing whether a
    record was used. For a set of noise mechanisms the best possible attacker reaches the
    total variation distance, which is $\\delta$ at $\\varepsilon = 0$. This is tighter than
    the general ceiling that holds for any mechanism with the same budget.
    """
    return delta_pld(events, 0.0, pessimistic=True)


def general_advantage_ceiling(epsilon: float, delta: float) -> float:
    """Largest advantage any attacker can have against ANY mechanism with this budget.

    $(e^{\\varepsilon} - 1 + 2\\delta) / (e^{\\varepsilon} + 1)$, capped at 1.
    """
    if epsilon < 0:
        raise ValueError("epsilon must not be negative")
    if not 0.0 <= delta <= 1.0:
        raise ValueError("delta must be in [0, 1]")
    if epsilon > 700.0:
        return 1.0
    grown = math.exp(epsilon)
    return min(1.0, (grown - 1.0 + 2.0 * delta) / (grown + 1.0))


def _check_delta(delta: float) -> None:
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
