"""What was released, written down in a form anyone can re-check.

Every time something that depends on private data leaves a participant, the library records
an event: which noise mechanism protected it and with what settings. A privacy budget is
nothing more than the result of adding these events up, so storing them (and not only the
final number) is what lets a third party recompute the budget from a report.

Events are small frozen records with plain numbers. They carry no data.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

MECHANISM_GAUSSIAN = "gaussian"
MECHANISM_SUBSAMPLED_GAUSSIAN = "poisson_subsampled_gaussian"


@dataclass(frozen=True)
class GaussianEvent:
    """Gaussian noise added to a value whose sensitivity is 1, repeated ``count`` times.

    ``noise_multiplier`` is the noise standard deviation divided by the sensitivity: how much
    one record can change the value, measured in the L2 norm.
    """

    noise_multiplier: float
    count: int = 1

    def __post_init__(self) -> None:
        _check_noise(self.noise_multiplier)
        _check_count(self.count, "count")

    @property
    def key(self) -> tuple[Any, ...]:
        return (MECHANISM_GAUSSIAN, self.noise_multiplier)

    @property
    def repetitions(self) -> int:
        return self.count

    def with_repetitions(self, count: int) -> GaussianEvent:
        return GaussianEvent(self.noise_multiplier, count)

    def with_noise_multiplier(self, noise_multiplier: float) -> GaussianEvent:
        """The same releases at another noise level (for "what if" tables in a report)."""
        return GaussianEvent(noise_multiplier, self.count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanism": MECHANISM_GAUSSIAN,
            "noise_multiplier": self.noise_multiplier,
            "count": self.count,
        }


@dataclass(frozen=True)
class SubsampledGaussianEvent:
    """DP-SGD steps: each record joins a step with probability ``sampling_rate``; then noise.

    ``steps`` counts every sampling draw, including draws that happened to select no record.
    """

    noise_multiplier: float
    sampling_rate: float
    steps: int

    def __post_init__(self) -> None:
        _check_noise(self.noise_multiplier)
        if not 0.0 < self.sampling_rate <= 1.0:
            raise ValueError("sampling_rate must be in (0, 1]")
        _check_count(self.steps, "steps")

    @property
    def key(self) -> tuple[Any, ...]:
        return (MECHANISM_SUBSAMPLED_GAUSSIAN, self.noise_multiplier, self.sampling_rate)

    @property
    def repetitions(self) -> int:
        return self.steps

    def with_repetitions(self, count: int) -> SubsampledGaussianEvent:
        return SubsampledGaussianEvent(self.noise_multiplier, self.sampling_rate, count)

    def with_noise_multiplier(self, noise_multiplier: float) -> SubsampledGaussianEvent:
        """The same releases at another noise level (for "what if" tables in a report)."""
        return SubsampledGaussianEvent(noise_multiplier, self.sampling_rate, self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanism": MECHANISM_SUBSAMPLED_GAUSSIAN,
            "noise_multiplier": self.noise_multiplier,
            "sampling_rate": self.sampling_rate,
            "steps": self.steps,
        }


Event = GaussianEvent | SubsampledGaussianEvent


def event_from_dict(data: Mapping[str, Any]) -> Event:
    """Rebuild an event from ``to_dict`` output. Unknown mechanisms are refused."""
    mechanism = data.get("mechanism")
    if mechanism == MECHANISM_GAUSSIAN:
        return GaussianEvent(float(data["noise_multiplier"]), int(data.get("count", 1)))
    if mechanism == MECHANISM_SUBSAMPLED_GAUSSIAN:
        return SubsampledGaussianEvent(
            float(data["noise_multiplier"]), float(data["sampling_rate"]), int(data["steps"])
        )
    raise ValueError(f"unknown mechanism {mechanism!r}")


def merge_events(events: Iterable[Event]) -> tuple[Event, ...]:
    """Combine events with identical settings by adding their repetitions.

    The combined budget of a set of events does not depend on their order, so merging changes
    nothing except the cost of computing it. The order of first appearance is kept.
    """
    merged: dict[tuple[Any, ...], Event] = {}
    for event in events:
        if event.repetitions == 0:
            continue
        seen = merged.get(event.key)
        total = event.repetitions + (seen.repetitions if seen is not None else 0)
        merged[event.key] = event.with_repetitions(total)
    return tuple(merged.values())


def covers(planned: Sequence[Event], used: Sequence[Event]) -> bool:
    """True if ``used`` fits inside ``planned``: same settings and no more repetitions."""
    capacity = {event.key: event.repetitions for event in merge_events(planned)}
    return all(event.repetitions <= capacity.get(event.key, 0) for event in merge_events(used))


def _check_noise(noise_multiplier: float) -> None:
    if not noise_multiplier > 0.0:
        raise ValueError("noise_multiplier must be positive")


def _check_count(count: int, name: str) -> None:
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"{name} must be a non-negative whole number")
