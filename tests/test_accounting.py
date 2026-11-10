"""Events, the three accountants, calibration and cross-checks."""

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from obscuradp.accounting import (
    AccountingMismatch,
    CrossCheck,
    GaussianEvent,
    SubsampledGaussianEvent,
    attacker_advantage_ceiling,
    calibrate_noise_multiplier,
    covers,
    cross_check,
    delta_pld,
    epsilon_pld,
    epsilon_rdp,
    event_from_dict,
    gaussian_delta,
    gaussian_epsilon,
    gaussian_total_variation,
    general_advantage_ceiling,
    merge_events,
    steps_per_round,
    total_steps,
)

DELTA = 1e-5


def _phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


# --------------------------------------------------------------------------- events


def test_events_round_trip_through_dictionaries() -> None:
    for event in (GaussianEvent(1.5, 3), SubsampledGaussianEvent(2.0, 0.05, 400)):
        assert event_from_dict(event.to_dict()) == event
    assert event_from_dict({"mechanism": "gaussian", "noise_multiplier": 2}) == GaussianEvent(
        2.0, 1
    )
    with pytest.raises(ValueError, match="unknown mechanism"):
        event_from_dict({"mechanism": "laplace"})


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: GaussianEvent(0.0), "noise_multiplier"),
        (lambda: GaussianEvent(-1.0), "noise_multiplier"),
        (lambda: GaussianEvent(1.0, -1), "count"),
        (lambda: GaussianEvent(1.0, True), "count"),
        (lambda: SubsampledGaussianEvent(1.0, 0.0, 5), "sampling_rate"),
        (lambda: SubsampledGaussianEvent(1.0, 1.5, 5), "sampling_rate"),
        (lambda: SubsampledGaussianEvent(1.0, 0.5, 2.5), "steps"),  # type: ignore[arg-type]
        (lambda: SubsampledGaussianEvent(float("nan"), 0.5, 2), "noise_multiplier"),
    ],
)
def test_impossible_events_are_refused(build, message: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match=message):
        build()


def test_merging_adds_repetitions_of_identical_settings() -> None:
    a = SubsampledGaussianEvent(2.0, 0.05, 100)
    b = SubsampledGaussianEvent(2.0, 0.05, 50)
    c = SubsampledGaussianEvent(3.0, 0.05, 10)
    g = GaussianEvent(2.0)
    merged = merge_events([a, c, b, g, GaussianEvent(2.0, 2), SubsampledGaussianEvent(9.0, 0.5, 0)])
    assert merged == (SubsampledGaussianEvent(2.0, 0.05, 150), c, GaussianEvent(2.0, 3))
    assert merge_events([]) == ()


def test_covers() -> None:
    plan = [SubsampledGaussianEvent(2.0, 0.05, 100), GaussianEvent(1.0, 2)]
    assert covers(plan, [])
    assert covers(
        plan, [SubsampledGaussianEvent(2.0, 0.05, 60), SubsampledGaussianEvent(2.0, 0.05, 40)]
    )
    assert not covers(plan, [SubsampledGaussianEvent(2.0, 0.05, 101)])
    assert not covers(plan, [SubsampledGaussianEvent(1.9, 0.05, 1)])  # less noise was not planned
    assert not covers(plan, [GaussianEvent(1.0, 3)])


# --------------------------------------------------------------------------- exact Gaussian


def test_exact_gaussian_formula_by_hand() -> None:
    expected = _phi(-0.5) - math.e * _phi(-1.5)
    assert gaussian_delta(1.0, 1.0) == pytest.approx(expected, rel=1e-12)
    assert gaussian_delta(1.0, 1.0) == pytest.approx(0.126937, abs=1e-6)
    assert gaussian_total_variation(1.0) == pytest.approx(2 * _phi(0.5) - 1, rel=1e-12)
    assert gaussian_delta(1.0, 800.0) == 0.0  # no overflow at huge epsilon


def test_exact_gaussian_epsilon_inverts_delta() -> None:
    for sigma in (0.5, 1.0, 4.0):
        epsilon = gaussian_epsilon(sigma, DELTA)
        assert gaussian_delta(sigma, epsilon) == pytest.approx(DELTA, rel=1e-6)
    assert gaussian_epsilon(1e6, 0.5) == 0.0


def test_repeating_gaussian_noise_equals_dividing_it() -> None:
    assert gaussian_epsilon(4.0, DELTA, count=16) == pytest.approx(
        gaussian_epsilon(1.0, DELTA), rel=1e-9
    )


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1])
def test_exact_gaussian_refuses_bad_inputs(bad: float) -> None:
    with pytest.raises(ValueError, match="delta"):
        gaussian_epsilon(1.0, bad)
    with pytest.raises(ValueError, match="noise_multiplier"):
        gaussian_delta(0.0, 1.0)
    with pytest.raises(ValueError, match="count"):
        gaussian_delta(1.0, 1.0, count=0)


# --------------------------------------------------------------------------- accountants


@pytest.mark.parametrize(("sigma", "count"), [(1.0, 1), (0.7, 1), (4.0, 9), (20.0, 100)])
def test_pld_matches_the_exact_answer_without_subsampling(sigma: float, count: int) -> None:
    events = [GaussianEvent(sigma, count)]
    exact = gaussian_epsilon(sigma, DELTA, count)
    assert epsilon_pld(events, DELTA) == pytest.approx(exact, rel=1e-3)
    assert epsilon_pld(events, DELTA, pessimistic=False) == pytest.approx(exact, rel=2e-2)
    assert epsilon_pld(events, DELTA, pessimistic=False) <= epsilon_pld(events, DELTA) + 1e-6
    assert epsilon_rdp(events, DELTA) >= exact
    assert attacker_advantage_ceiling(events) == pytest.approx(
        gaussian_total_variation(sigma, count), rel=1e-3
    )


def test_subsampling_rate_one_is_the_same_as_no_subsampling() -> None:
    assert epsilon_pld([SubsampledGaussianEvent(2.0, 1.0, 5)], DELTA) == pytest.approx(
        epsilon_pld([GaussianEvent(2.0, 5)], DELTA), rel=1e-9
    )
    assert epsilon_rdp([SubsampledGaussianEvent(2.0, 1.0, 5)], DELTA) == pytest.approx(
        epsilon_rdp([GaussianEvent(2.0, 5)], DELTA), rel=1e-9
    )


GRID = [
    (sigma, rate, steps)
    for sigma in (1.0, 3.0)
    for rate in (0.01, 0.145, 1.0)
    for steps in (60, 412)
]


@pytest.mark.parametrize(("sigma", "rate", "steps"), GRID)
def test_the_three_accountants_are_consistent(sigma: float, rate: float, steps: int) -> None:
    events = [SubsampledGaussianEvent(sigma, rate, steps)]
    result = cross_check(events, DELTA)  # raises if any rule is broken
    assert result.problems() == []
    assert result.epsilon == result.pld_upper
    assert result.to_dict()["delta"] == DELTA
    if result.unbounded:
        # Only the weakest cell: 412 full-batch steps at noise multiplier 1.
        assert (sigma, rate, steps) == (1.0, 1.0, 412)
        assert result.rdp > 200
        assert result.warnings() == [
            "the budget is too large to mean anything and is reported as unbounded"
        ]
        return
    assert result.pld_lower <= result.pld_upper + 1e-6
    assert result.pld_upper - result.pld_lower <= 0.10 * result.pld_upper + 0.02
    assert result.pld_lower <= result.rdp + 1e-6
    assert result.pld_upper <= result.rdp + 1e-6  # RDP is an upper limit, PLD the tight one


def test_rdp_is_much_looser_for_short_runs_which_is_a_warning_not_an_error() -> None:
    """Measured: 1.88 times the PLD estimate at 60 steps, rate 0.01, noise multiplier 1."""
    result = cross_check([SubsampledGaussianEvent(1.0, 0.01, 60)], DELTA)
    assert result.rdp / result.pld_upper == pytest.approx(1.88, abs=0.02)
    assert result.problems() == []
    assert len(result.warnings()) == 1
    assert "far above the PLD estimate" in result.warnings()[0]


@pytest.mark.parametrize(("n_rows", "steps"), [(683, 1300), (220, 420), (2482, 4660)])
def test_rdp_stays_inside_its_measured_band_for_calibrated_runs(n_rows: int, steps: int) -> None:
    """For the runs the library plans, RDP is 1.08 to 1.24 times PLD; outside 1.35 warns."""
    event = calibrate_noise_multiplier(1.0, DELTA, 32 / n_rows, steps).event
    result = cross_check([event], DELTA)
    assert 1.05 <= result.rdp / result.pld_upper <= 1.27
    assert result.warnings() == []


def test_order_of_events_does_not_matter_and_merging_changes_nothing() -> None:
    a = SubsampledGaussianEvent(2.0, 0.05, 100)
    b = SubsampledGaussianEvent(1.5, 0.1, 30)
    whole = epsilon_pld([a, b], DELTA)
    assert epsilon_pld([b, a], DELTA) == pytest.approx(whole, rel=1e-9)
    halves = [SubsampledGaussianEvent(2.0, 0.05, 50)] * 2 + [b]
    assert epsilon_pld(halves, DELTA) == pytest.approx(whole, rel=1e-12)
    assert whole > max(epsilon_pld([a], DELTA), epsilon_pld([b], DELTA))


def test_nothing_released_costs_nothing() -> None:
    assert epsilon_pld([], DELTA) == 0.0
    assert epsilon_rdp([], DELTA) == 0.0
    assert delta_pld([], 0.0) == 0.0
    assert attacker_advantage_ceiling([]) == 0.0


def test_absurdly_weak_noise_is_reported_as_unbounded_without_exhausting_memory() -> None:
    events = [SubsampledGaussianEvent(0.05, 0.5, 5000)]
    assert epsilon_pld(events, DELTA) == math.inf
    assert delta_pld(events, 1.0) == 1.0


def test_accountants_refuse_bad_inputs() -> None:
    event = [GaussianEvent(1.0)]
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError, match="delta"):
            epsilon_pld(event, bad)
        with pytest.raises(ValueError, match="delta"):
            epsilon_rdp(event, bad)
    with pytest.raises(ValueError, match="epsilon"):
        delta_pld(event, -1.0)


@settings(max_examples=12, deadline=None)
@given(
    sigma=st.floats(0.8, 6.0),
    rate=st.floats(0.01, 0.5),
    steps=st.integers(10, 400),
    factor=st.floats(1.1, 2.0),
)
def test_more_noise_fewer_steps_or_rarer_sampling_never_cost_more(
    sigma: float, rate: float, steps: int, factor: float
) -> None:
    base = epsilon_pld([SubsampledGaussianEvent(sigma, rate, steps)], DELTA)
    slack = 1e-3 + 2e-3 * base  # the accountant works on a grid
    assert (
        epsilon_pld([SubsampledGaussianEvent(sigma * factor, rate, steps)], DELTA) <= base + slack
    )
    assert (
        epsilon_pld([SubsampledGaussianEvent(sigma, rate / factor, steps)], DELTA) <= base + slack
    )
    fewer = max(1, int(steps / factor))
    assert epsilon_pld([SubsampledGaussianEvent(sigma, rate, fewer)], DELTA) <= base + slack


# --------------------------------------------------------------------------- attacker ceilings


def test_general_ceiling_values() -> None:
    assert general_advantage_ceiling(1.0, DELTA) == pytest.approx(0.4621, abs=1e-4)
    assert general_advantage_ceiling(0.0, 0.0) == 0.0
    assert general_advantage_ceiling(1000.0, DELTA) == 1.0
    # A guaranteed 5% needs a budget near 0.1, as the audit says.
    assert general_advantage_ceiling(2 * math.atanh(0.05), 0.0) == pytest.approx(0.05, rel=1e-9)
    with pytest.raises(ValueError, match="epsilon"):
        general_advantage_ceiling(-1.0, DELTA)
    with pytest.raises(ValueError, match="delta"):
        general_advantage_ceiling(1.0, 2.0)


def test_mechanism_ceiling_is_far_below_the_general_one() -> None:
    """Table 3 of docs/KPI_MAPPING.md at a total budget of 1 (loan setting)."""
    calibration = calibrate_noise_multiplier(1.0, DELTA, 32 / 683, 1300)
    ceiling = attacker_advantage_ceiling([calibration.event])
    assert ceiling == pytest.approx(0.106, abs=0.002)
    assert ceiling < general_advantage_ceiling(1.0, DELTA) / 4


# --------------------------------------------------------------------------- calibration


def test_step_counts_by_hand() -> None:
    assert steps_per_round(3, 32 / 683) == 65  # 3 / 0.04685 = 64.03
    assert total_steps(20, 3, 32 / 683) == 1300
    assert steps_per_round(1, 1.0) == 1
    assert steps_per_round(2, 0.5) == 4  # exact division is not rounded up
    for call in (
        lambda: steps_per_round(0, 0.5),
        lambda: steps_per_round(1, 0.0),
        lambda: total_steps(0, 1, 0.5),
    ):
        with pytest.raises(ValueError, match="must be"):
            call()


@pytest.mark.parametrize(
    ("n_rows", "steps", "expected"),
    [(683, 1300, 6.39), (220, 420, 11.22), (2482, 4660, 3.38)],
)
def test_calibration_reproduces_the_reference_table(
    n_rows: int, steps: int, expected: float
) -> None:
    """Table 1 of docs/KPI_MAPPING.md at a total budget of 1."""
    rate = 32 / n_rows
    assert total_steps(20, 3, rate) == steps
    result = calibrate_noise_multiplier(1.0, DELTA, rate, steps)
    assert result.noise_multiplier == pytest.approx(expected, abs=0.02)
    # Certified: the returned noise level fits, and only just.
    assert result.epsilon <= 1.0
    assert result.epsilon > 0.995
    assert epsilon_pld([result.event], DELTA) == result.epsilon
    assert result.event == SubsampledGaussianEvent(result.noise_multiplier, rate, steps)


def test_rdp_calibration_needs_more_noise_than_pld() -> None:
    tight = calibrate_noise_multiplier(1.0, DELTA, 32 / 683, 1300)
    loose = calibrate_noise_multiplier(1.0, DELTA, 32 / 683, 1300, accountant="rdp")
    assert loose.noise_multiplier == pytest.approx(6.93, abs=0.02)
    assert loose.noise_multiplier > tight.noise_multiplier * 1.05
    assert epsilon_rdp([loose.event], DELTA) <= 1.0


def test_a_larger_budget_needs_less_noise() -> None:
    levels = [
        calibrate_noise_multiplier(eps, DELTA, 0.1, 200).noise_multiplier
        for eps in (0.5, 1.0, 2.0, 4.0)
    ]
    assert levels == sorted(levels, reverse=True)
    assert len(set(levels)) == 4


def test_calibration_refuses_what_it_cannot_deliver() -> None:
    with pytest.raises(ValueError, match="too small"):
        calibrate_noise_multiplier(1e-3, DELTA, 1.0, 10**6, accountant="rdp")
    for kwargs in (
        {"target_epsilon": 0.0},
        {"steps": 0},
        {"accountant": "magic"},
        {"relative_tolerance": 0.0},
    ):
        arguments = {
            "target_epsilon": 1.0,
            "delta": DELTA,
            "sampling_rate": 0.1,
            "steps": 10,
            **kwargs,
        }
        with pytest.raises(ValueError, match="must be"):
            calibrate_noise_multiplier(**arguments)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- cross-check failures


def test_cross_check_names_each_broken_rule() -> None:
    assert CrossCheck(DELTA, pld_upper=1.0, pld_lower=0.97, rdp=1.09).problems() == []
    above = CrossCheck(DELTA, pld_upper=1.0, pld_lower=1.2, rdp=1.09).problems()
    assert any("above its upper estimate" in text for text in above)
    assert any("above the RDP limit" in text for text in above)
    assert any("too far apart" in text for text in CrossCheck(DELTA, 1.0, 0.5, 1.09).problems())
    loose = CrossCheck(DELTA, pld_upper=1.0, pld_lower=0.97, rdp=2.0)
    assert loose.problems() == []
    assert any("far above the PLD" in text for text in loose.warnings())
    unbounded = CrossCheck(DELTA, pld_upper=math.inf, pld_lower=math.inf, rdp=302.0)
    assert unbounded.unbounded
    assert unbounded.problems() == []


def test_cross_check_stops_a_run_when_accountants_disagree(monkeypatch: pytest.MonkeyPatch) -> None:
    """If one accountant were given different events, the run must not report a number."""
    from obscuradp.accounting import crosscheck

    monkeypatch.setattr(crosscheck, "epsilon_rdp", lambda events, delta: 0.1)
    events = [SubsampledGaussianEvent(2.0, 0.1, 100)]
    with pytest.raises(AccountingMismatch, match="above the RDP limit"):
        cross_check(events, DELTA)
    assert cross_check(events, DELTA, strict=False).rdp == 0.1


# --------------------------------------------------------------------------- full reference tables


@pytest.mark.slow
def test_full_noise_table_of_the_target_definitions() -> None:
    """Every cell of table 1 in docs/KPI_MAPPING.md (PLD column), within its printed precision."""
    table = {
        220: {0.5: 21.05, 1: 11.22, 2: 6.05, 4: 3.36, 8: 1.96},
        393: {0.5: 15.65, 1: 8.35, 2: 4.53, 4: 2.54, 8: 1.53},
        2482: {0.5: 6.26, 1: 3.38, 2: 1.91, 4: 1.18, 8: 0.83},
        683: {0.5: 11.94, 1: 6.39, 2: 3.49, 4: 1.99, 8: 1.24},
    }
    for n_rows, row in table.items():
        rate = 32 / n_rows
        steps = total_steps(20, 3, rate)
        for epsilon, expected in row.items():
            result = calibrate_noise_multiplier(float(epsilon), DELTA, rate, steps)
            assert result.noise_multiplier == pytest.approx(expected, abs=0.02), (n_rows, epsilon)
            assert result.epsilon <= epsilon


@pytest.mark.slow
def test_full_ceiling_table_of_the_target_definitions() -> None:
    """Table 3 in docs/KPI_MAPPING.md: both ceilings at five budgets (loan setting)."""
    table = {
        0.5: (0.245, 0.056),
        1: (0.462, 0.106),
        2: (0.762, 0.195),
        4: (0.964, 0.346),
        8: (0.999, 0.563),
    }
    for epsilon, (general, mechanism) in table.items():
        result = calibrate_noise_multiplier(float(epsilon), DELTA, 32 / 683, 1300)
        assert general_advantage_ceiling(epsilon, DELTA) == pytest.approx(general, abs=1e-3)
        assert attacker_advantage_ceiling([result.event]) == pytest.approx(mechanism, abs=2e-3)
