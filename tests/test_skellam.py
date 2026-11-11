"""Whole-number (Skellam) noise: the leak formula against brute force, rounding, the noise.

The formula for what Skellam noise can leak is taken from a paper. These tests do not take
it on trust: they compute the same quantity exactly from the Skellam probabilities and
check that the formula is never below it and not far above it.
"""

import math

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import logsumexp
from scipy.stats import skellam
from torch import nn

from obscuradp.accounting import (
    GaussianEvent,
    SkellamCalibration,
    SkellamEvent,
    SubsampledGaussianEvent,
    attacker_advantage_ceiling,
    calibrate_skellam,
    cross_check,
    delta_pld,
    epsilon_pld,
    epsilon_rdp,
    event_from_dict,
    merge_events,
    skellam_rdp,
)
from obscuradp.accounting.skellam import ORDERS
from obscuradp.client import integer_sum_step
from obscuradp.compression import SumGrid, plan_integer_grid
from obscuradp.ledger import BudgetExceeded, PrivacyLedger
from obscuradp.mechanisms import (
    clipped_records,
    clipped_sum,
    per_record_gradients,
    plan_integer_step,
    round_records,
    skellam_noise,
)
from obscuradp.mechanisms.skellam import MAX_INFLATION, rounded_length_limit
from obscuradp.models import TabularMLP
from obscuradp.policy import PrivacyPolicy
from obscuradp.state import get_state

DELTA = 1e-5


# --------------------------------------------------------------------------- brute force


def _log_pmf(variance: float, shift: int, points: np.ndarray) -> np.ndarray:
    """Log-probabilities of Skellam noise of the given variance, moved by ``shift``."""
    return np.asarray(skellam.logpmf(points - shift, variance / 2.0, variance / 2.0))


def _points(variance: float, shift: int) -> np.ndarray:
    span = int(abs(shift) + 30 * math.sqrt(variance) + 40)
    return np.arange(-span, span + 1)


def _exact_renyi(order: float, variance: float, shift: int) -> float:
    """Exact Renyi divergence between Skellam noise moved by ``shift`` and unmoved noise."""
    points = _points(variance, shift)
    moved, still = _log_pmf(variance, shift, points), _log_pmf(variance, 0, points)
    usable = np.isfinite(moved) & np.isfinite(still)
    total = logsumexp(order * moved[usable] + (1.0 - order) * still[usable])
    return float(total) / (order - 1.0)


def _exact_delta(epsilon: float, variance: float, shift: int) -> float:
    """Exact $\\delta$ at ``epsilon`` for one release of a number moved by ``shift``."""
    points = _points(variance, shift)
    moved, still = np.exp(_log_pmf(variance, shift, points)), np.exp(_log_pmf(variance, 0, points))
    return float(np.clip(moved - math.exp(epsilon) * still, 0.0, None).sum())


@pytest.mark.parametrize("variance", [1.0, 2.0, 4.0, 10.0, 40.0, 200.0, 2000.0])
@pytest.mark.parametrize("shift", [1, 2, 3, 5, 10, 30])
def test_the_formula_is_never_below_the_exact_value(variance: float, shift: int) -> None:
    for order in (2, 3, 5, 8, 16, 32, 64):
        exact = _exact_renyi(order, variance, shift)
        limit = skellam_rdp(order, variance, float(shift), float(shift))
        assert exact <= limit * (1.0 + 1e-9), (order, exact, limit)


def test_the_formula_is_close_to_the_exact_value_for_ample_noise() -> None:
    """Not wasteful: within 1% once the noise is large compared with order times shift."""
    for shift in (1, 5, 30):
        for order in (2, 8):
            exact = _exact_renyi(order, 2000.0, shift)
            limit = skellam_rdp(order, 2000.0, float(shift), float(shift))
            assert exact == pytest.approx(limit, rel=0.01)


def test_the_formula_holds_in_many_dimensions() -> None:
    """Coordinates are independent, so the exact value is the sum over coordinates."""
    generator = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        variance = float(generator.choice([4.0, 20.0, 100.0, 800.0]))
        shift = generator.integers(-4, 5, size=int(generator.integers(2, 12)))
        order = int(generator.choice([2, 4, 16, 40]))
        if not shift.any():
            continue
        exact = sum(_exact_renyi(order, variance, int(part)) for part in shift if part)
        limit = skellam_rdp(
            order,
            variance,
            float(np.sqrt((shift**2).sum())),
            float(np.abs(shift).sum()),
        )
        assert exact <= limit * (1.0 + 1e-9)
        worst = max(worst, exact / limit)
    assert worst > 0.9  # and the limit is nearly reached somewhere


@pytest.mark.parametrize(("variance", "shift"), [(50.0, 2), (200.0, 5), (14.0 * 900.0, 30)])
def test_the_reported_budget_covers_the_exact_leak_of_one_release(
    variance: float, shift: int
) -> None:
    event = SkellamEvent(variance, float(shift), float(shift), 1)
    epsilon = epsilon_rdp([event], DELTA)
    assert _exact_delta(epsilon, variance, shift) <= DELTA
    # The same at the level of the attacker's advantage (the leak at a budget of zero).
    assert _exact_delta(0.0, variance, shift) <= attacker_advantage_ceiling([event])


def test_the_formula_has_the_form_of_the_paper() -> None:
    """Corollary 3.6 of Agarwal, Kairouz and Liu (2021), with the variance as its parameter."""
    order, variance, l2, l1 = 7, 50.0, 3.0, 5.0
    first = order * l2**2 / (2 * variance)
    second = min(((2 * order - 1) * l2**2 + 6 * l1) / (4 * variance**2), 3 * l1 / (2 * variance))
    assert skellam_rdp(order, variance, l2, l1) == pytest.approx(first + second)
    # The first part is the curve of Gaussian noise; the rest fades as the noise grows.
    assert skellam_rdp(order, 1e9, l2, l1) == pytest.approx(order * l2**2 / 2e9, rel=1e-6)
    # Little noise: the second form of the correction is the smaller one.
    assert skellam_rdp(order, 0.5, l2, l1) == pytest.approx(order * l2**2 / 1.0 + 3 * l1 / 1.0)


def test_only_whole_number_orders_are_accepted() -> None:
    """The result is proven for whole-number orders, so no other order is ever used."""
    assert all(isinstance(order, int) and order >= 2 for order in ORDERS)
    assert list(ORDERS) == sorted(set(ORDERS))
    for order in (1, 2.5, True, 0):
        with pytest.raises(ValueError, match="whole number above 1"):
            skellam_rdp(order, 10.0, 1.0, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="variance"):
        skellam_rdp(2, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="sensitivities"):
        skellam_rdp(2, 1.0, 0.0, 1.0)


# --------------------------------------------------------------------------- accounting


def _release(multiplier: float, count: int = 1, l2: float = 30.0) -> SkellamEvent:
    return SkellamEvent((multiplier * l2) ** 2, l2, l2 * l2, count)


def test_a_whole_number_release_is_a_record_like_the_others() -> None:
    event = _release(4.0, 3)
    assert event.noise_multiplier == pytest.approx(4.0)
    assert event.repetitions == 3
    assert event.with_repetitions(7).count == 7
    assert event.with_noise_multiplier(8.0).variance == pytest.approx(4.0 * event.variance)
    assert event_from_dict(event.to_dict()) == event
    assert event.to_dict()["mechanism"] == "skellam"
    merged = merge_events([event, _release(4.0, 2), GaussianEvent(4.0, 1)])
    assert merged == (_release(4.0, 5), GaussianEvent(4.0, 1))
    for arguments, message in (
        ((0.0, 1.0, 1.0), "variance"),
        ((math.inf, 1.0, 1.0), "variance"),
        ((1.0, 0.0, 1.0), "sensitivities"),
        ((1.0, 2.0, 1.0), "cannot be below"),
    ):
        with pytest.raises(ValueError, match=message):
            SkellamEvent(*arguments)
    with pytest.raises(ValueError, match="count"):
        SkellamEvent(1.0, 1.0, 1.0, -1)


@pytest.mark.parametrize(("multiplier", "count"), [(3.73, 1), (17.0, 20), (40.0, 100)])
def test_whole_number_noise_costs_a_little_more_than_gaussian_noise(
    multiplier: float, count: int
) -> None:
    whole, ordinary = [_release(multiplier, count)], [GaussianEvent(multiplier, count)]
    budget = epsilon_rdp(whole, DELTA)
    # Never below Gaussian noise of the same size by the same method, and barely above it.
    assert budget >= epsilon_rdp(ordinary, DELTA) - 1e-9
    assert budget == pytest.approx(epsilon_rdp(ordinary, DELTA), rel=0.02)
    # The tight method for Gaussian noise is what the looser curve gives away.
    assert epsilon_pld(ordinary, DELTA) < budget < 1.15 * epsilon_pld(ordinary, DELTA)
    # For whole-number noise the reported figure is the curve's.
    assert epsilon_pld(whole, DELTA) == budget
    assert epsilon_pld(whole, DELTA, pessimistic=False) == epsilon_pld(
        ordinary, DELTA, pessimistic=False
    )


def test_the_cross_check_knows_there_is_one_method_only() -> None:
    check = cross_check([_release(17.0, 20)], DELTA)
    assert check.single_method
    assert check.problems() == []
    assert check.epsilon == check.rdp
    assert check.pld_lower < check.pld_upper
    assert "one counting method is proven" in check.warnings()[0]
    assert not cross_check([GaussianEvent(17.0, 20)], DELTA).single_method
    # The remaining rule still bites: a reference figure above the reported one is an error.
    broken = type(check)(DELTA, pld_upper=1.0, pld_lower=1.2, rdp=1.0, single_method=True)
    assert any("above its upper estimate" in text for text in broken.problems())
    wide = type(check)(DELTA, pld_upper=1.0, pld_lower=0.5, rdp=1.0, single_method=True)
    assert wide.problems() == []


def test_leak_at_a_given_budget_and_the_attacker_limit() -> None:
    events = [_release(17.0, 20)]
    budget = epsilon_rdp(events, DELTA)
    assert delta_pld(events, budget) <= DELTA * 1.001
    assert delta_pld(events, budget / 2) > DELTA
    assert delta_pld(events, 0.0) > delta_pld(events, 0.5) > delta_pld(events, 2.0)
    limit = attacker_advantage_ceiling(events)
    gaussian = attacker_advantage_ceiling([GaussianEvent(17.0, 20)])
    assert gaussian < limit < 1.0
    assert delta_pld(events, 0.0, pessimistic=False) == pytest.approx(
        delta_pld([GaussianEvent(17.0, 20)], 0.0, pessimistic=False)
    )
    assert attacker_advantage_ceiling([_release(0.01, 5)]) == 1.0
    assert (
        math.isinf(epsilon_rdp([_release(0.05, 50)], DELTA))
        or epsilon_rdp([_release(0.05, 50)], DELTA) > 200
    )


def test_whole_number_and_gaussian_releases_add_up() -> None:
    whole = _release(17.0, 20)
    sampled = SubsampledGaussianEvent(1.5, 0.05, 100)
    together = epsilon_rdp([whole, sampled], DELTA)
    assert together > epsilon_rdp([whole], DELTA)
    assert together > epsilon_rdp([sampled], DELTA) - 0.05
    assert together < epsilon_rdp([whole], DELTA) + epsilon_rdp([sampled], DELTA)
    assert cross_check([whole, sampled], DELTA).problems() == []


def test_the_noise_level_is_searched_for_the_whole_run() -> None:
    result = calibrate_skellam(1.0, DELTA, 20, 30.0, 900.0)
    assert isinstance(result, SkellamCalibration)
    assert result.event == SkellamEvent(result.variance, 30.0, 900.0, 20)
    assert result.sampling_rate == 1.0
    assert 0.995 < result.epsilon <= 1.0
    assert epsilon_rdp([result.event], DELTA) == result.epsilon
    # Not wasteful: two percent less noise no longer fits.
    less = result.event.with_noise_multiplier(0.98 * result.noise_multiplier)
    assert epsilon_rdp([less], DELTA) > 1.0
    # About eight to nine percent more noise than Gaussian noise counted the tight way.
    assert 16.0 < result.noise_multiplier < 19.5
    assert calibrate_skellam(1.0, DELTA, 20, 30.0, 900.0) is result
    assert calibrate_skellam(0.5, DELTA, 20, 30.0, 900.0).noise_multiplier > result.noise_multiplier
    for arguments, message in (
        ((0.0, DELTA, 20, 30.0, 900.0), "target_epsilon"),
        ((1.0, DELTA, 0, 30.0, 900.0), "steps"),
        ((1e-9, DELTA, 20, 30.0, 900.0), "no noise level"),
    ):
        with pytest.raises(ValueError, match=message):
            calibrate_skellam(*arguments)
    with pytest.raises(ValueError, match="relative_tolerance"):
        calibrate_skellam(1.0, DELTA, 20, 30.0, 900.0, relative_tolerance=2.0)


def test_the_ledger_refuses_one_release_too_many() -> None:
    planned = calibrate_skellam(1.0, DELTA, 5, 30.0, 900.0)
    ledger = PrivacyLedger(PrivacyPolicy(1.0, DELTA))
    ledger.plan("site", [planned.event])
    one = planned.event.with_repetitions(1)
    for _ in range(5):
        ledger.charge("site", one)
    assert 0.995 < ledger.spent("site") <= 1.0
    with pytest.raises(BudgetExceeded):
        ledger.charge("site", one)
    assert ledger.events("site") == (planned.event,)
    rebuilt = PrivacyLedger.from_dict(ledger.to_dict())
    assert rebuilt.events("site") == (planned.event,)


# --------------------------------------------------------------------------- rounding


def test_the_grid_is_planned_from_the_size_of_the_model_only() -> None:
    for dimension in (1, 129, 3000, 20000):
        step = plan_integer_step(dimension)
        assert step.dimension == dimension
        assert step.scale == float(int(step.scale))
        assert step.l2_squared == rounded_length_limit(step.scale, dimension)
        assert 1.0 < step.inflation <= MAX_INFLATION
        assert step.l1_sensitivity == min(
            step.l2_squared, math.sqrt(dimension) * step.l2_sensitivity
        )
        assert step.l1_sensitivity >= step.l2_sensitivity
    # A coarser grid would lengthen records by more than the planner allows.
    step = plan_integer_step(3000)
    coarser = step.scale - 2.0
    assert math.sqrt(rounded_length_limit(coarser, 3000)) / coarser > MAX_INFLATION
    assert plan_integer_step(129, max_inflation=1.5).scale < plan_integer_step(129).scale
    for arguments, message in (((0,), "dimension"), ((10, 1.0), "max_inflation")):
        with pytest.raises(ValueError, match=message):
            plan_integer_step(*arguments)
    with pytest.raises(ValueError, match="scale"):
        rounded_length_limit(0.0, 5)
    with pytest.raises(ValueError, match="dimension"):
        rounded_length_limit(5.0, 0)


@settings(max_examples=60, deadline=None)
@given(
    dimension=st.integers(1, 40),
    rows=st.integers(0, 12),
    stretch=st.sampled_from([0.0, 0.3, 1.0, 1.0 + 1e-6, 1.04, 3.0, 50.0]),
    seed=st.integers(0, 2**32 - 1),
)
def test_no_rounded_row_is_ever_longer_than_the_limit(
    dimension: int, rows: int, stretch: float, seed: int
) -> None:
    """Whatever comes in, including rows longer than the clipping allows."""
    generator = np.random.default_rng(seed)
    step = plan_integer_step(dimension)
    directions = generator.normal(size=(rows, dimension))
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    values = directions / np.where(lengths == 0, 1.0, lengths) * step.scale * stretch
    rounded = round_records(values, step.l2_squared, generator)
    assert rounded.dtype == np.int64
    assert rounded.shape == values.shape
    squares = (rounded * rounded).sum(axis=1)
    assert np.all(squares <= step.l2_squared)
    assert np.all(np.abs(rounded).sum(axis=1) <= step.l1_sensitivity + 1e-9)
    # Every number moved by less than one grid point, unless the row was dropped.
    kept = squares > 0
    assert np.all(np.abs(rounded[kept] - values[kept]) < 1.0)
    if stretch <= 1.0:
        assert np.all(np.abs(rounded - values) < 1.0)


def test_rounding_is_right_on_average_and_rows_that_are_too_long_count_for_nothing() -> None:
    generator = np.random.default_rng(1)
    step = plan_integer_step(50)
    row = generator.normal(size=50)
    row = row / np.linalg.norm(row) * step.scale
    many = round_records(np.tile(row, (20_000, 1)), step.l2_squared, generator)
    assert np.abs(many.mean(axis=0) - row).max() < 0.03
    assert len({tuple(r) for r in many[:50]}) > 1  # the rounding is random
    assert not round_records(np.tile(row * 40, (3, 1)), step.l2_squared, generator).any()
    assert round_records(np.zeros((0, 4)), 9, generator).shape == (0, 4)
    for values, limit, message in (
        (np.zeros(4), 9, "matrix"),
        (np.zeros((2, 4)), 0, "l2_squared"),
        (np.full((1, 2), np.nan), 9, "finite"),
        (np.full((1, 2), 1e12), 9, "too large"),
    ):
        with pytest.raises(ValueError, match=message):
            round_records(values, limit, generator)


def test_a_row_that_keeps_failing_is_rounded_towards_zero() -> None:
    """A row right at the limit in one coordinate: rounding up can never pass."""
    generator = np.random.default_rng(2)
    values = np.array([[3.999999, 0.0, 0.0]])
    rounded = round_records(values, 9, generator)  # 4 squared is 16, above 9
    assert rounded.tolist() == [[3, 0, 0]]


# --------------------------------------------------------------------------- the noise


def test_whole_number_noise_has_the_stated_spread_and_slices_add_up() -> None:
    generator = np.random.default_rng(3)
    noise = skellam_noise((400_000,), 37.0, generator)
    assert noise.dtype == np.int64
    assert abs(noise.mean()) < 0.05
    assert noise.var() == pytest.approx(37.0, rel=0.01)
    # Four slices of a quarter each are noise of the full size: compare with the exact
    # probabilities of that noise.
    total = sum(skellam_noise((400_000,), 37.0 / 4, generator) for _ in range(4))
    values = np.arange(-15, 16)
    observed = np.array([(total == value).mean() for value in values])
    expected = skellam.pmf(values, 18.5, 18.5)
    assert np.abs(observed - expected).max() < 0.002
    assert not skellam_noise((5,), 0.0, generator).any()
    assert skellam_noise((2, 3), 4.0, generator).shape == (2, 3)
    for variance in (-1.0, math.inf):
        with pytest.raises(ValueError, match="variance"):
            skellam_noise((5,), variance, generator)


# --------------------------------------------------------------------------- one message


def _table(n: int = 30, d: int = 4, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=generator)
    return x, (x[:, 0] + 0.5 * x[:, 1] > 0).float()


def _model(seed: int = 0) -> TabularMLP:
    torch.manual_seed(seed)
    return TabularMLP(4, (6,))


def _message(model: nn.Module, x: torch.Tensor, y: torch.Tensor, **overrides: object) -> np.ndarray:
    step = plan_integer_step(sum(p.numel() for p in model.parameters()))
    arguments: dict[str, object] = {
        "loss_fn": nn.BCELoss(),
        "ledger": PrivacyLedger(PrivacyPolicy(1e6, DELTA)),
        "scope": "site",
        "clip_norm": 1.0,
        "step": step,
        "variance": 4000.0,
        "share_variance": 1000.0,
        "rounding_generator": np.random.default_rng(1),
        "noise_generator": np.random.default_rng(2),
        **overrides,
    }
    return integer_sum_step(model, x, y, **arguments)  # type: ignore[arg-type]


def test_the_message_is_rounded_records_plus_one_slice() -> None:
    x, y = _table()
    model = _model()
    before = get_state(model)
    step = plan_integer_step(sum(p.numel() for p in model.parameters()))
    message = _message(model, x, y)
    assert message.dtype == np.int64
    assert message.shape == (step.dimension,)
    # Replay: clip each record, scale, round, add up, add the slice.
    rows = clipped_records(per_record_gradients(model, nn.BCELoss(), x, y), 1.0)
    rounded = round_records((rows * step.scale).numpy(), step.l2_squared, np.random.default_rng(1))
    noise = skellam_noise((step.dimension,), 1000.0, np.random.default_rng(2))
    assert np.array_equal(message, rounded.sum(axis=0) + noise)
    # The clipped records are the ones the ordinary step adds up.
    total = clipped_sum(per_record_gradients(model, nn.BCELoss(), x, y), 1.0)
    flat = torch.cat([value.double().flatten() for value in total.values()])
    assert torch.allclose(rows.sum(dim=0), flat, atol=1e-6)
    assert float(rows.norm(dim=1).max()) <= 1.0
    for name, value in get_state(model).items():
        assert torch.equal(value, before[name])
    # Limiting each number of the clean sum happens before the noise.
    limited = _message(model, x, y, sum_bound=3)
    assert np.array_equal(limited, np.clip(rounded.sum(axis=0), -3, 3) + noise)


def test_the_message_is_charged_before_any_data_is_used() -> None:
    x, y = _table()
    model = _model()
    step = plan_integer_step(sum(p.numel() for p in model.parameters()))
    rounding, noise = np.random.default_rng(1), np.random.default_rng(2)
    untouched = (rounding.bit_generator.state, noise.bit_generator.state)
    tight = PrivacyLedger(PrivacyPolicy(0.01, DELTA))
    with pytest.raises(BudgetExceeded):
        _message(model, x, y, ledger=tight, rounding_generator=rounding, noise_generator=noise)
    assert tight.scopes == ()
    assert (rounding.bit_generator.state, noise.bit_generator.state) == untouched
    ledger = PrivacyLedger(PrivacyPolicy(1e6, DELTA))
    _message(model, x, y, ledger=ledger)
    # Charged with the noise of all honest slices together and the exact sensitivities.
    assert ledger.events("site") == (
        SkellamEvent(4000.0, step.l2_sensitivity, step.l1_sensitivity, 1),
    )


def test_a_message_needs_noise_rows_and_the_right_grid() -> None:
    x, y = _table()
    with pytest.raises(ValueError, match="without noise is not private"):
        _message(_model(), x, y, share_variance=0.0)
    with pytest.raises(ValueError, match="at least one record"):
        _message(_model(), x[:0], y[:0])
    with pytest.raises(ValueError, match="same number of records"):
        _message(_model(), x, y[:-1])
    with pytest.raises(ValueError, match="sum_bound"):
        _message(_model(), x, y, sum_bound=0)
    with pytest.raises(ValueError, match="another size"):
        _message(_model(), x, y, step=plan_integer_step(7))
    with pytest.raises(ValueError, match="no trainable parameter"):
        clipped_records({}, 1.0)


# --------------------------------------------------------------------------- the wire


def test_whole_numbers_travel_modulo_the_grid() -> None:
    grid = plan_integer_grid(unit=0.04, sum_bound=1000, share_std=50.0, participants=4)
    assert grid.step == 0.04
    assert 2 ** (grid.value_bits - 1) >= 1000 + 10 * 50.0 > 2 ** (grid.value_bits - 2)
    values = np.array([0, 1, -1, 1499, -1500, 10**6, -(10**6)], dtype=np.int64)
    wrapped = grid.wrap(values)
    assert wrapped.dtype == np.uint64
    assert int(wrapped.max()) < 2**grid.modulus_bits
    modulus = np.uint64(2**grid.modulus_bits)
    total = (wrapped + grid.wrap(-values)) % modulus
    assert not grid.decode_sum(total).any()
    assert grid.decode_sum(grid.wrap(values[:5])).tolist() == pytest.approx(
        [0.0, 0.04, -0.04, 59.96, -60.0]
    )
    with pytest.raises(ValueError, match="whole numbers"):
        grid.wrap(np.array([0.5]))
    for arguments, message in (
        ((0.0, 10, 1.0, 3), "unit"),
        ((1.0, 0, 1.0, 3), "sum_bound"),
        ((1.0, 10, 0.0, 3), "share_std"),
        ((1.0, 10**9, 1.0, 3), "declare a smaller sum_bound"),
    ):
        with pytest.raises(ValueError, match=message):
            plan_integer_grid(*arguments)
    assert isinstance(grid, SumGrid)
