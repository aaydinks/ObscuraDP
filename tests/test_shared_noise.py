"""Shared noise: the size of a slice, the grid of the secure sum, and why rounding is safe.

In the shared-noise mode a single message carries only a slice of the noise. These tests
check the arithmetic that mode stands on: slices add up to the full noise, the whole-number
grid never cuts a value off, and the sum of separately rounded messages reveals no more than
the true noisy sum does.
"""

import math

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import ndtr

from obscuradp.compression import Quantizer, SumGrid, plan_sum_grid
from obscuradp.compression.grid import MAX_VALUE_BITS, STEPS_PER_SHARE, TAIL_SHARES
from obscuradp.crypto import setup_pairwise_masking
from obscuradp.mechanisms import (
    default_sum_bound,
    effective_noise_multiplier,
    gaussian_noise,
    honest_shares,
    share_noise_multiplier,
    share_std,
)

# --------------------------------------------------------------------------- slice sizes


@pytest.mark.parametrize(
    ("participants", "colluding", "dropouts"),
    [(2, 0, 0), (5, 1, 0), (5, 0, 2), (5, 2, 2), (10, 3, 1), (3, 2, 0)],
)
def test_slice_size_matches_the_formula(participants: int, colluding: int, dropouts: int) -> None:
    sigma, clip = 3.2, 0.5
    honest = participants - colluding - dropouts
    assert honest_shares(participants, colluding, dropouts) == honest
    assert share_noise_multiplier(sigma, participants, colluding, dropouts) == pytest.approx(
        sigma / math.sqrt(honest)
    )
    assert share_std(sigma, clip, participants, colluding, dropouts) == pytest.approx(
        sigma * clip / math.sqrt(honest)
    )


def test_five_participants_and_one_colluder_halve_the_slice() -> None:
    assert share_std(2.0, 1.0, 5, 1, 0) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("participants", "colluding", "dropouts", "message"),
    [
        (5, 5, 0, "nobody is left"),
        (5, 3, 2, "nobody is left"),
        (5, -1, 0, "must not be negative"),
        (5, 0, -1, "must not be negative"),
        (1, 0, 0, "at least two participants"),
        (5, True, 0, "whole number"),
    ],
)
def test_impossible_groups_are_refused(
    participants: int, colluding: int, dropouts: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        honest_shares(participants, colluding, dropouts)
    with pytest.raises(ValueError, match=message):
        share_std(1.0, 1.0, participants, colluding, dropouts)


def test_slice_arguments_are_checked() -> None:
    with pytest.raises(ValueError, match="noise_multiplier"):
        share_noise_multiplier(0.0, 5, 1, 0)
    with pytest.raises(ValueError, match="clip_norm"):
        share_std(1.0, 0.0, 5, 1, 0)


def test_the_assumed_honest_slices_add_up_to_the_full_noise() -> None:
    """The $N - t - d$ slices the guarantee counts on carry $\\sigma C$ together."""
    sigma, clip, participants, colluding, dropouts = 2.5, 0.8, 6, 1, 1
    honest = honest_shares(participants, colluding, dropouts)
    slice_std = share_std(sigma, clip, participants, colluding, dropouts)
    generator = torch.Generator().manual_seed(0)
    slices = [
        gaussian_noise((200_000,), slice_std, generator, dtype=torch.float64)
        for _ in range(participants)
    ]
    assert float(slices[0].std()) == pytest.approx(slice_std, rel=0.01)
    assert float(torch.stack(slices[:honest]).sum(dim=0).std()) == pytest.approx(
        sigma * clip, rel=0.01
    )
    # When nobody is missing and nobody colludes, there is more noise than needed.
    assert float(torch.stack(slices).sum(dim=0).std()) == pytest.approx(
        sigma * clip * math.sqrt(participants / honest), rel=0.01
    )


def test_the_effective_noise_falls_as_more_participants_collude() -> None:
    sigma, participants, colluding = 2.0, 5, 1
    values = [
        effective_noise_multiplier(sigma, participants, colluding, 0, real)
        for real in range(participants)
    ]
    assert values[colluding] == pytest.approx(sigma)
    assert values == sorted(values, reverse=True)
    assert values[0] == pytest.approx(sigma * math.sqrt(5 / 4))
    # One honest participant left: only its own slice protects it.
    assert values[-1] == pytest.approx(share_noise_multiplier(sigma, participants, colluding, 0))
    # An allowance for missing participants makes every slice larger.
    assert effective_noise_multiplier(sigma, 5, 1, 1, 1) == pytest.approx(sigma * math.sqrt(4 / 3))
    with pytest.raises(ValueError, match="between 0 and participants - 1"):
        effective_noise_multiplier(sigma, 5, 1, 0, 5)
    with pytest.raises(ValueError, match="whole number"):
        effective_noise_multiplier(sigma, 5, 1, 0, 1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="noise_multiplier"):
        effective_noise_multiplier(-1.0, 5, 1, 0, 1)


def test_the_limit_of_a_clipped_sum_uses_public_numbers_only() -> None:
    # Full batch: a sum of n clipped gradients cannot exceed n times the clipping norm.
    assert default_sum_bound(1.0, 1.0, 50) == 50.0
    assert default_sum_bound(0.5, 1.0, 50) == 25.0
    # A small rate: the expected number of rows plus a wide margin, never more than n.
    bound = default_sum_bound(1.0, 0.05, 1600)
    expected = 0.05 * 1600
    assert expected + 10 * math.sqrt(expected * 0.95) + 10 == pytest.approx(bound)
    assert bound < 1600
    assert default_sum_bound(1.0, 0.9, 12) == 12.0
    for arguments, message in (
        ((0.0, 0.5, 10), "clip_norm"),
        ((1.0, 0.0, 10), "sampling_rate"),
        ((1.0, 1.5, 10), "sampling_rate"),
        ((1.0, 0.5, 0), "largest_table"),
    ):
        with pytest.raises(ValueError, match=message):
            default_sum_bound(*arguments)


# --------------------------------------------------------------------------- the grid


def test_the_grid_is_planned_from_the_slice_and_the_limit() -> None:
    grid = plan_sum_grid(share_std=1.0, sum_bound=168.0, participants=5)
    assert grid.step == pytest.approx(1.0 / STEPS_PER_SHARE)
    assert grid.reach >= 168.0 + TAIL_SHARES * 1.0
    # One bit fewer would not hold the planned range.
    assert grid.reach / 2 < 168.0 + TAIL_SHARES * 1.0
    assert grid.modulus_bits == grid.value_bits + 3
    assert grid.message_bytes(1000) == math.ceil(1000 * grid.modulus_bits / 8)
    # Fewer than half the bits of a 32-bit number for these sizes.
    assert grid.modulus_bits <= 15


def test_a_grid_that_would_need_too_many_bits_is_refused() -> None:
    with pytest.raises(ValueError, match="too small for the sums it protects"):
        plan_sum_grid(share_std=1e-6, sum_bound=100.0, participants=3)
    for arguments, message in (
        ((0.0, 1.0, 3), "share_std"),
        ((1.0, 0.0, 3), "sum_bound"),
        ((math.inf, 1.0, 3), "share_std"),
    ):
        with pytest.raises(ValueError, match=message):
            plan_sum_grid(*arguments)
    for grid_arguments, message in (
        ((0.0, 8, 3), "step must be positive"),
        ((1.0, 1, 3), "value_bits"),
        ((1.0, MAX_VALUE_BITS + 1, 3), "value_bits"),
        ((1.0, 8, 0), "participants"),
    ):
        with pytest.raises(ValueError, match=message):
            SumGrid(*grid_arguments)


def test_rounding_is_right_on_average() -> None:
    grid = SumGrid(step=0.25, value_bits=8, participants=3)
    values = np.full(200_000, 0.4)  # between the grid points 0.25 and 0.5
    generator = np.random.default_rng(0)
    decoded = grid.decode_sum(grid.encode(values, generator))
    assert set(np.unique(decoded)) == {0.25, 0.5}
    assert decoded.mean() == pytest.approx(0.4, abs=1e-3)
    assert grid.decode_sum(grid.encode([-3.0, 0.0, 7.75], generator)).tolist() == [
        -3.0,
        0.0,
        7.75,
    ]


def test_encode_refuses_what_it_cannot_represent() -> None:
    grid = SumGrid(step=1e-6, value_bits=8, participants=2)
    generator = np.random.default_rng(0)
    with pytest.raises(ValueError, match="finite"):
        grid.encode([math.nan], generator)
    with pytest.raises(ValueError, match="too large"):
        grid.encode([1e12], generator)
    assert grid.outside_share([]) == 0.0
    assert grid.outside_share([0.0, 1.0, -1.0, 1e-5]) == 0.5


@settings(max_examples=60, deadline=None)
@given(
    participants=st.integers(2, 6),
    value_bits=st.integers(3, 12),
    seed=st.integers(0, 2**32 - 1),
)
def test_wrapping_single_values_never_changes_the_sum(
    participants: int, value_bits: int, seed: int
) -> None:
    """Values may lie far outside one participant's range: only the sum has to fit."""
    grid = SumGrid(step=0.5, value_bits=value_bits, participants=participants)
    generator = np.random.default_rng(seed)
    room = 2.0 ** (grid.modulus_bits - 1) * grid.step
    # Large values of both signs whose sum is small: each wraps, the total does not.
    big = generator.uniform(-4 * grid.reach, 4 * grid.reach, size=(participants - 1, 7))
    last = generator.uniform(-0.25 * room, 0.25 * room, size=7) - big.sum(axis=0)
    values = np.vstack([big, last])
    modulus = np.uint64(2**grid.modulus_bits)
    total = np.zeros(7, dtype=np.uint64)
    for row in values:
        total = (total + grid.encode(row, generator)) % modulus
    opened = grid.decode_sum(total)
    # Each rounding moves a value by less than one step.
    assert np.all(np.abs(opened - values.sum(axis=0)) < participants * grid.step)


def test_the_masked_sum_opens_to_the_plain_sum() -> None:
    names = ["a", "b", "c", "d"]
    grid = plan_sum_grid(share_std=1.5, sum_bound=40.0, participants=len(names))
    clients, aggregator = setup_pairwise_masking(names, grid.modulus_bits)
    generator = np.random.default_rng(1)
    values = {name: generator.normal(scale=20.0, size=300) for name in names}
    values["a"][:5] = 5.0 * grid.reach  # far outside one participant's range
    values["b"][:5] = -5.0 * grid.reach
    messages = {
        name: clients[name].protect(0, grid.encode(values[name], generator)) for name in names
    }
    assert all(len(message) == grid.message_bytes(300) for message in messages.values())
    opened = grid.decode_sum(aggregator.combine(0, messages, 300))
    plain = np.sum(list(values.values()), axis=0)
    assert np.all(np.abs(opened - plain) < len(names) * grid.step)


def test_cutting_values_off_would_erase_noise_and_the_grid_does_not() -> None:
    """Why the shared-noise mode cannot use the rounding of the local mode."""
    generator = np.random.default_rng(2)
    slices = generator.normal(scale=2.5, size=100_000)
    cut = Quantizer(bits=12, value_range=1.0)
    kept = cut.dequantize_sum(cut.quantize(slices, generator), 1)
    assert kept.std() < 1.0  # most of the slice is gone
    grid = plan_sum_grid(share_std=2.5, sum_bound=1.0, participants=3)
    assert grid.decode_sum(grid.encode(slices, generator)).std() == pytest.approx(2.5, rel=0.02)


# --------------------------------------------------------------------------- rounding


def _rounded_pmf(mean: float, std: float, points: np.ndarray) -> np.ndarray:
    """Exact chance that a value from $\\mathcal{N}(m, s^2)$ is rounded at random to each point.

    Random rounding to a neighbouring whole number picks $k$ with probability
    $\\max(0, 1 - |x - k|)$. That hat is $(x - k + 1)^+ - 2 (x - k)^+ + (x - k - 1)^+$, and
    $E[(X - a)^+]$ has a closed form for a Gaussian $X$.
    """

    def partial(a: np.ndarray) -> np.ndarray:
        z = (mean - a) / std
        return (mean - a) * ndtr(z) + std * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

    return partial(points - 1) - 2 * partial(points) + partial(points + 1)


def _smoothed_pmf(mean: float, std: float, hats: int, points: np.ndarray) -> np.ndarray:
    """The same for a Gaussian value plus ``hats - 1`` independent small noises, rounded once.

    Each extra noise is the difference of two independent uniform numbers in [0, 1]. Written
    through the Fourier transform, where the Gaussian contributes
    $e^{-2 \\pi^2 s^2 \\xi^2}$ and every hat $\\mathrm{sinc}^2(\\xi)$.
    """
    xi = np.linspace(-6.0 / std, 6.0 / std, 20_001)
    weight = np.exp(-2.0 * math.pi**2 * std**2 * xi**2) * np.sinc(xi) ** (2 * hats)
    width = xi[1] - xi[0]
    return np.array(
        [np.sum(weight * np.cos(2.0 * math.pi * xi * (k - mean))) * width for k in points]
    )


def _distance_to_one_rounding(slice_steps: float, offsets: tuple[float, ...]) -> float:
    """Total variation between two ways of producing a whole-number sum, in one dimension.

    First way (what the protocol does): every participant rounds its own noisy value and the
    whole numbers are added. Second way: the unrounded noisy values are added, small noise
    that depends on no data is added, and the result is rounded once. The second is computed
    from the true noisy sum alone, so it can reveal no more than that sum.

    Args:
        slice_steps: Standard deviation of one noise slice, in grid steps.
        offsets: The participants' clean values, in grid steps.
    """
    count = len(offsets)
    span = int(12 * slice_steps * math.sqrt(count) + 12 + abs(sum(offsets)))
    points = np.arange(-span, span + 1, dtype=np.float64)
    separately = np.array([1.0])
    for offset in offsets:
        separately = np.convolve(separately, _rounded_pmf(offset, slice_steps, points))
    sums = np.arange(-span * count, span * count + 1, dtype=np.float64)
    once = _smoothed_pmf(sum(offsets), slice_steps * math.sqrt(count), count, sums)
    assert separately.sum() == pytest.approx(1.0, abs=1e-9)
    return 0.5 * float(np.abs(separately - once).sum())


@pytest.mark.parametrize(
    "offsets",
    [(0.0, 0.0), (0.3, -0.41, 0.77), (0.5, 0.5, 0.5, 0.25), (17.2, -3.9, 0.01, 4.4, -8.6)],
)
def test_separately_rounded_messages_reveal_no_more_than_the_noisy_sum(
    offsets: tuple[float, ...],
) -> None:
    """At the planned grid the two ways agree to the limit of double precision."""
    assert _distance_to_one_rounding(STEPS_PER_SHARE, offsets) < 1e-10


def test_on_a_coarse_grid_the_two_ways_differ() -> None:
    """The control: with a step of two slice widths the difference is plain to see."""
    assert _distance_to_one_rounding(0.5, (0.3, -0.41, 0.77)) > 1e-3
    assert _distance_to_one_rounding(0.25, (0.0, 0.0)) > 1e-2
