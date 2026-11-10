"""Turning attack results into lower limits on the privacy budget."""

import math

import numpy as np
import pytest

from obscuradp.accounting import gaussian_epsilon
from obscuradp.audit import (
    clopper_pearson,
    epsilon_lower_bound,
    gaussian_epsilon_lower_bound,
    gaussian_separation_interval,
)


def test_exact_interval_known_values() -> None:
    # No successes in n trials: the upper limit solves (1 - p)^n = 0.025.
    lower, upper = clopper_pearson(0, 100, 0.95)
    assert lower == 0.0
    assert upper == pytest.approx(1 - 0.025 ** (1 / 100), rel=1e-9)
    lower, upper = clopper_pearson(100, 100, 0.95)
    assert upper == 1.0
    assert lower == pytest.approx(0.025 ** (1 / 100), rel=1e-9)
    # A textbook case: 5 of 20 at 95% gives (0.0866, 0.4910).
    lower, upper = clopper_pearson(5, 20, 0.95)
    assert (lower, upper) == pytest.approx((0.0866, 0.4910), abs=5e-4)
    # More confidence, wider interval; mirrored counts, mirrored interval.
    wide = clopper_pearson(5, 20, 0.999)
    assert wide[0] < lower < upper < wide[1]
    mirrored = clopper_pearson(15, 20, 0.95)
    assert mirrored == pytest.approx((1 - upper, 1 - lower), rel=1e-9)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [((1, 0, 0.95), "trials"), ((5, 4, 0.95), "successes"), ((1, 4, 1.0), "confidence")],
)
def test_exact_interval_refuses_bad_input(arguments: tuple[int, int, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        clopper_pearson(*arguments)


def test_an_attack_that_guesses_shows_nothing() -> None:
    assert epsilon_lower_bound(500, 500, 1000, 1000, delta=1e-5) == 0.0
    assert epsilon_lower_bound(1000, 0, 1000, 1000, delta=1e-5) == 0.0  # always says "used"


def test_a_perfect_attack_is_limited_only_by_the_number_of_trials() -> None:
    small = epsilon_lower_bound(0, 0, 100, 100, delta=1e-5)
    large = epsilon_lower_bound(0, 0, 10_000, 10_000, delta=1e-5)
    upper = clopper_pearson(0, 100, 0.975)[1]
    assert small == pytest.approx(math.log((1 - 1e-5 - upper) / upper), rel=1e-9)
    assert 3.0 < small < large < 9.0
    with pytest.raises(ValueError, match="delta"):
        epsilon_lower_bound(0, 0, 10, 10, delta=1.0)


def test_the_limit_uses_whichever_direction_of_error_is_rarer() -> None:
    one_way = epsilon_lower_bound(2, 400, 1000, 1000, delta=0.0)
    other_way = epsilon_lower_bound(400, 2, 1000, 1000, delta=0.0)
    assert one_way == pytest.approx(other_way, rel=1e-12)
    assert one_way > 3.0


def test_the_limit_never_exceeds_the_truth_for_gaussian_noise() -> None:
    """Simulated attacks on Gaussian noise with a known budget: the limit stays below it."""
    sigma, delta, trials = 1.0, 1e-5, 20_000
    truth = gaussian_epsilon(sigma, delta)  # about 4.4
    rng = np.random.default_rng(0)
    without = rng.normal(0.0, sigma, trials)
    with_record = rng.normal(1.0, sigma, trials)
    best = 0.0
    for threshold in np.linspace(-2.0, 4.0, 25):
        limit = epsilon_lower_bound(
            int((without > threshold).sum()),
            int((with_record <= threshold).sum()),
            trials,
            trials,
            delta,
            confidence=1 - 0.05 / 25,  # 25 thresholds tried
        )
        assert limit <= truth
        best = max(best, limit)
    assert best > 1.5  # and the attack is not toothless


def test_gaussian_separation_is_recovered_with_an_honest_interval() -> None:
    rng = np.random.default_rng(1)
    truth = 0.4
    without = rng.normal(3.0, 2.0, 5000)
    with_record = rng.normal(3.0 + truth * 2.0, 2.0, 5000)
    estimate, lower, upper = gaussian_separation_interval(without, with_record)
    assert lower < truth < upper
    assert estimate == pytest.approx(truth, abs=0.05)
    assert upper - lower == pytest.approx(2 * 1.96 * math.sqrt(2 / 5000), rel=0.02)
    tighter = gaussian_separation_interval(without, with_record, confidence=0.5)
    assert tighter[2] - tighter[1] < upper - lower
    with pytest.raises(ValueError, match="at least two"):
        gaussian_separation_interval([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="do not vary"):
        gaussian_separation_interval([1.0, 1.0], [1.0, 1.0])


def test_gaussian_limit_is_below_the_truth_and_close_to_it() -> None:
    rng = np.random.default_rng(2)
    sigma, delta = 2.0, 1e-5
    truth = gaussian_epsilon(sigma, delta)
    without = rng.normal(0.0, sigma, 20_000)
    with_record = rng.normal(1.0, sigma, 20_000)
    limit = gaussian_epsilon_lower_bound(without, with_record, delta)
    assert 0.85 * truth < limit <= truth
    # No visible difference between the worlds: nothing is shown.
    assert gaussian_epsilon_lower_bound(without, rng.normal(0.0, sigma, 20_000), delta) == 0.0
