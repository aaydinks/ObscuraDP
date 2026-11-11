"""Turning attack results into lower limits on the privacy budget."""

import math

import numpy as np
import pytest
from scipy import stats

from obscuradp.accounting import gaussian_epsilon
from obscuradp.audit import (
    clopper_pearson,
    epsilon_lower_bound,
    gaussian_epsilon_lower_bound,
    gaussian_separation_interval,
    one_run_epsilon_lower_bound,
    one_run_p_value,
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


# --------------------------------------------------------------------------- one training run


def _listing_p_value(m: int, r: int, v: int, eps: float, delta: float) -> float:
    """The paper's own listing (Steinke, Nasr and Jagielski 2023, appendix), line for line."""
    q = 1 / (1 + math.exp(-eps))
    beta = stats.binom.sf(v - 1, r, q)
    alpha = 0.0
    total = 0.0
    for i in range(1, v + 1):
        total = total + stats.binom.pmf(v - i, r, q)
        if total > i * alpha:
            alpha = total / i
    return float(min(beta + alpha * delta * 2 * m, 1))


def _listing_epsilon(m: int, r: int, v: int, delta: float, p: float) -> float:
    eps_min, eps_max = 0.0, 1.0
    while _listing_p_value(m, r, v, eps_max, delta) < p:
        eps_max = eps_max + 1
    for _ in range(30):
        eps = (eps_min + eps_max) / 2
        if _listing_p_value(m, r, v, eps, delta) < p:
            eps_min = eps
        else:
            eps_max = eps
    return eps_min


@pytest.mark.parametrize(
    ("m", "r", "v"),
    [(700, 700, 400), (700, 200, 150), (100, 100, 100), (1000, 100, 99), (21000, 6000, 3500)],
)
def test_the_one_run_routine_is_the_papers(m: int, r: int, v: int) -> None:
    for eps in (0.0, 0.5, 1.0, 3.0):
        for delta in (0.0, 1e-5, 1e-3):
            ours = one_run_p_value(m, r, v, eps, delta)
            assert ours == pytest.approx(_listing_p_value(m, r, v, eps, delta), rel=1e-9, abs=1e-15)
    assert one_run_epsilon_lower_bound(m, r, v, 1e-5) == pytest.approx(
        _listing_epsilon(m, r, v, 1e-5, 0.05), abs=1e-6
    )


def test_one_run_guesses_that_are_no_better_than_coins_show_nothing() -> None:
    assert one_run_epsilon_lower_bound(700, 700, 350, 1e-5) == 0.0
    assert one_run_epsilon_lower_bound(700, 700, 0, 1e-5) == 0.0
    assert one_run_epsilon_lower_bound(700, 0, 0, 1e-5) == 0.0
    assert one_run_p_value(700, 700, 0, 1.0, 1e-5) == 1.0


def test_one_run_limit_grows_with_right_guesses_and_is_capped_by_their_number() -> None:
    limits = [one_run_epsilon_lower_bound(400, 400, v, 1e-5) for v in (220, 260, 300, 360, 400)]
    assert limits == sorted(limits)
    assert limits[0] < limits[-1]
    # Even 100 right guesses out of 100 cannot prove more than about 3.5.
    assert 3.0 < one_run_epsilon_lower_bound(100, 100, 100, 1e-5) < 4.0
    # More confidence asked for, less proven.
    assert one_run_epsilon_lower_bound(400, 400, 300, 1e-5, confidence=0.999) < limits[2]
    # A larger delta excuses more.
    assert one_run_epsilon_lower_bound(400, 400, 300, 1e-2) < limits[2]


def test_one_run_refusals() -> None:
    with pytest.raises(ValueError, match="correct <= guesses <= canaries"):
        one_run_p_value(10, 11, 5, 1.0, 1e-5)
    with pytest.raises(ValueError, match="correct <= guesses <= canaries"):
        one_run_p_value(10, 5, 6, 1.0, 1e-5)
    with pytest.raises(ValueError, match="epsilon"):
        one_run_p_value(10, 5, 5, -1.0, 1e-5)
    with pytest.raises(ValueError, match="delta"):
        one_run_p_value(10, 5, 5, 1.0, 2.0)
    with pytest.raises(ValueError, match="confidence"):
        one_run_epsilon_lower_bound(10, 5, 5, 1e-5, confidence=1.0)


def test_one_run_limit_does_not_overstate_a_mechanism_that_leaks_all_it_may() -> None:
    """Brute force: answer each coin truthfully with probability e^eps / (e^eps + 1).

    That is the leakiest thing a budget of ``eps`` allows for one record, so the limit may
    exceed ``eps`` in at most 5% of audits. Pooling audits must not change that.
    """
    truth = 1.0
    right = math.exp(truth) / (math.exp(truth) + 1.0)
    generator = np.random.default_rng(5)
    single = pooled = 0
    audits = 400
    for _ in range(audits):
        correct = generator.binomial(300, right, size=4)
        single += one_run_epsilon_lower_bound(300, 300, int(correct[0]), 0.0) > truth
        pooled += one_run_epsilon_lower_bound(1200, 1200, int(correct.sum()), 0.0) > truth
    assert single / audits <= 0.08
    assert pooled / audits <= 0.08
    # And the limit is not empty: pooled, it typically proves most of the budget.
    typical = one_run_epsilon_lower_bound(1200, 1200, round(1200 * right), 0.0)
    assert 0.7 * truth < typical <= truth
