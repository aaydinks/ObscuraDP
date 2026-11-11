"""Sharing the privacy noise among the participants.

In the local mode every participant adds all the noise its records need. In the shared mode
every participant adds only a slice, and the slices add up inside a secure sum, where nobody
can look at a single participant's message. The noise the group needs is then paid once, not
once per participant.

Three numbers describe a group: $N$ participants, of which at most $t$ pool what they know
with the server (they know their own slices and can subtract them) and at most $d$ are
missing from a sum. The slices of the remaining $N - t - d$ participants must add up to the
full noise on their own, so each slice has standard deviation

$$\\frac{\\sigma C}{\\sqrt{N - t - d}}$$

for a noise multiplier $\\sigma$ and a clipping norm $C$.

A single slice is far too small to protect anyone by itself. Everything in this module is
only safe behind a secure sum. See ``docs/SHARED_NOISE.md`` for the argument.
"""

from __future__ import annotations

import math


def _check_group(participants: int, max_colluding: int, max_dropouts: int) -> int:
    for name, value in (
        ("participants", participants),
        ("max_colluding", max_colluding),
        ("max_dropouts", max_dropouts),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be a whole number")
    if participants < 2:
        raise ValueError("shared noise needs at least two participants")
    if max_colluding < 0 or max_dropouts < 0:
        raise ValueError("max_colluding and max_dropouts must not be negative")
    honest = participants - max_colluding - max_dropouts
    if honest < 1:
        raise ValueError(
            f"with {participants} participants, {max_colluding} that may collude and "
            f"{max_dropouts} that may be missing, nobody is left to supply the noise"
        )
    return honest


def honest_shares(participants: int, max_colluding: int, max_dropouts: int) -> int:
    """How many slices the guarantee may count on: $N - t - d$ (at least 1)."""
    return _check_group(participants, max_colluding, max_dropouts)


def share_noise_multiplier(
    noise_multiplier: float, participants: int, max_colluding: int, max_dropouts: int
) -> float:
    """The noise multiplier of one participant's slice: $\\sigma / \\sqrt{N - t - d}$."""
    if not noise_multiplier > 0.0:
        raise ValueError("noise_multiplier must be positive")
    return noise_multiplier / math.sqrt(_check_group(participants, max_colluding, max_dropouts))


def share_std(
    noise_multiplier: float,
    clip_norm: float,
    participants: int,
    max_colluding: int,
    max_dropouts: int,
) -> float:
    """Standard deviation of one participant's slice: $\\sigma C / \\sqrt{N - t - d}$."""
    if not clip_norm > 0.0:
        raise ValueError("clip_norm must be positive")
    return clip_norm * share_noise_multiplier(
        noise_multiplier, participants, max_colluding, max_dropouts
    )


def effective_noise_multiplier(
    noise_multiplier: float,
    participants: int,
    max_colluding: int,
    max_dropouts: int,
    colluding: int,
) -> float:
    """The noise that is really left when ``colluding`` participants pool with the server.

    A sum is opened only when all $N$ messages are there. If $t'$ of the participants give
    their slices away, the other $N - t'$ slices remain, so the noise multiplier that
    protects a record is

    $$\\sigma_{t'} = \\sigma \\sqrt{\\frac{N - t'}{N - t - d}}.$$

    With $t' = t$ and $d = 0$ this is $\\sigma$ itself. A participant on its own knows its
    own slice, so against a single curious participant the value for $t' = 1$ applies.

    Args:
        noise_multiplier: $\\sigma$, the multiplier the run was calibrated for.
        participants: $N$.
        max_colluding: $t$, the assumed largest number of colluding participants.
        max_dropouts: $d$, the assumed largest number of missing participants.
        colluding: $t'$, the real number of colluding participants, from 0 to $N - 1$.
    """
    honest = _check_group(participants, max_colluding, max_dropouts)
    if not noise_multiplier > 0.0:
        raise ValueError("noise_multiplier must be positive")
    if isinstance(colluding, bool) or not isinstance(colluding, int):
        raise ValueError("colluding must be a whole number")
    if not 0 <= colluding < participants:
        raise ValueError("colluding must be between 0 and participants - 1")
    return noise_multiplier * math.sqrt((participants - colluding) / honest)


def default_sum_bound(clip_norm: float, sampling_rate: float, largest_table: int) -> float:
    """A limit for each number of one participant's clipped gradient sum, from public facts.

    No number of a sum of $m$ clipped gradients can exceed $m C$ in size. The number of rows
    $m$ a participant draws is random; for the largest table of $n$ rows it stays below
    $q n + 10 \\sqrt{q n (1 - q)} + 10$ except with a probability far below $10^{-15}$, and it
    can never exceed $n$. The limit only uses the declared table size, the sampling rate and
    the clipping norm, all of which are public.
    """
    if not clip_norm > 0.0:
        raise ValueError("clip_norm must be positive")
    if not 0.0 < sampling_rate <= 1.0:
        raise ValueError("sampling_rate must be in (0, 1]")
    if largest_table < 1:
        raise ValueError("largest_table must be at least 1")
    mean = sampling_rate * largest_table
    spread = math.sqrt(mean * (1.0 - sampling_rate))
    return clip_norm * min(float(largest_table), mean + 10.0 * spread + 10.0)
