"""The privacy policy and the ledger that enforces it."""

import pytest

import obscuradp
from obscuradp.accounting import (
    GaussianEvent,
    SubsampledGaussianEvent,
    calibrate_noise_multiplier,
    epsilon_pld,
    epsilon_rdp,
)
from obscuradp.ledger import BudgetExceeded, PrivacyLedger
from obscuradp.policy import PrivacyPolicy

DELTA = 1e-5
RATE = 0.1
ROUNDS = 10
STEPS_PER_ROUND = 20


@pytest.fixture(scope="module")
def sigma() -> float:
    """Noise level for a total budget of 1 over the whole run (10 rounds of 20 steps)."""
    result = calibrate_noise_multiplier(1.0, DELTA, RATE, ROUNDS * STEPS_PER_ROUND)
    return result.noise_multiplier


def _round(sigma: float, steps: int = STEPS_PER_ROUND) -> SubsampledGaussianEvent:
    return SubsampledGaussianEvent(sigma, RATE, steps)


# --------------------------------------------------------------------------- policy


def test_the_top_level_package_exports_the_public_pieces() -> None:
    assert obscuradp.PrivacyPolicy is PrivacyPolicy
    assert obscuradp.PrivacyLedger is PrivacyLedger
    assert obscuradp.BudgetExceeded is BudgetExceeded


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"epsilon": 0.0}, "epsilon must be positive"),
        ({"epsilon": float("nan")}, "epsilon must be positive"),
        ({"delta": 0.0}, "delta"),
        ({"delta": 1.0}, "delta"),
        ({"unit": "person"}, "unit"),
        ({"adjacency": "replace_one"}, "adjacency"),
        ({"accountant": "magic"}, "accountant"),
        ({"max_records_per_person": 0}, "max_records_per_person"),
        ({"per_scope_epsilon": {"a": 0.0}}, "per-scope"),
    ],
)
def test_impossible_policies_are_refused(kwargs: dict[str, object], message: str) -> None:
    arguments: dict[str, object] = {"epsilon": 1.0, "delta": DELTA, **kwargs}
    with pytest.raises(ValueError, match=message):
        PrivacyPolicy(**arguments)  # type: ignore[arg-type]


def test_policy_budgets_per_scope_and_weak_delta() -> None:
    policy = PrivacyPolicy(2.0, DELTA, per_scope_epsilon={"clinic": 0.5})
    assert policy.epsilon_for("clinic") == 0.5
    assert policy.epsilon_for("anyone else") == 2.0
    with pytest.raises(TypeError):
        policy.per_scope_epsilon["clinic"] = 9.0  # type: ignore[index]
    assert not policy.delta_is_weak_for(683)  # 1e-5 is well below 1/683
    assert policy.delta_is_weak_for(100_000)  # 1e-5 is exactly 1/100000: too large
    assert not policy.delta_is_weak_for(0)


def test_policy_describes_itself_in_plain_words() -> None:
    text = PrivacyPolicy(1.0, DELTA).describe()
    assert "one record" in text
    assert "adding or removing" in text
    assert "epsilon = 1, delta = 1e-05" in text
    assert "group privacy" not in text
    many = PrivacyPolicy(
        1.0, DELTA, unit="client", per_scope_epsilon={"b": 0.5}, max_records_per_person=4
    ).describe()
    assert "one participant" in many
    assert "b: 0.5" in many
    assert "up to 4 records" in many


def test_policy_round_trips_through_a_dictionary() -> None:
    policy = PrivacyPolicy(
        1.5, 1e-6, accountant="rdp", per_scope_epsilon={"a": 0.25}, max_records_per_person=3
    )
    assert PrivacyPolicy.from_dict(policy.to_dict()) == policy
    assert PrivacyPolicy.from_dict({"epsilon": 1, "delta": DELTA}) == PrivacyPolicy(1.0, DELTA)


# --------------------------------------------------------------------------- ledger


def test_a_whole_run_fits_and_one_more_round_is_refused(sigma: float) -> None:
    ledger = PrivacyLedger(PrivacyPolicy(1.0, DELTA))
    for _ in range(ROUNDS):
        ledger.charge("client-0", _round(sigma))
    spent = ledger.spent("client-0")
    assert 0.99 < spent <= 1.0
    assert ledger.remaining("client-0") == pytest.approx(1.0 - spent)
    assert ledger.events("client-0") == (_round(sigma, ROUNDS * STEPS_PER_ROUND),)

    assert not ledger.can_afford("client-0", _round(sigma))
    with pytest.raises(BudgetExceeded, match="more than its budget of 1"):
        ledger.charge("client-0", _round(sigma))
    # A refused release is not recorded.
    assert ledger.events("client-0") == (_round(sigma, ROUNDS * STEPS_PER_ROUND),)
    assert ledger.spent("client-0") == spent


def test_scopes_are_independent_and_the_run_reports_the_largest(sigma: float) -> None:
    ledger = PrivacyLedger(PrivacyPolicy(1.0, DELTA))
    assert ledger.scopes == ()
    assert ledger.spent_max() == 0.0
    assert ledger.spent("nobody") == 0.0
    assert ledger.remaining("nobody") == 1.0
    ledger.charge("a", _round(sigma, 150))
    ledger.charge("b", _round(sigma, 40))
    assert ledger.scopes == ("a", "b")
    assert ledger.spent("a") > ledger.spent("b") > 0.0
    assert ledger.spent_max() == ledger.spent("a")
    # Running out in one scope does not block another.
    assert not ledger.can_afford("a", _round(sigma, 100))
    assert ledger.can_afford("b", _round(sigma, 100))


def test_a_stricter_scope_is_refused_sooner(sigma: float) -> None:
    ledger = PrivacyLedger(PrivacyPolicy(1.0, DELTA, per_scope_epsilon={"strict": 0.3}))
    whole_run = _round(sigma, ROUNDS * STEPS_PER_ROUND)
    ledger.charge("normal", whole_run)
    with pytest.raises(BudgetExceeded, match=r"budget of 0\.3"):
        ledger.charge("strict", whole_run)
    assert "strict" not in ledger.scopes


def test_releases_with_no_repetitions_are_ignored(sigma: float) -> None:
    ledger = PrivacyLedger(PrivacyPolicy(1.0, DELTA))
    ledger.charge("a", _round(sigma, 0))
    ledger.charge("a", GaussianEvent(5.0, 0))
    assert ledger.scopes == ()


def test_a_plan_is_checked_once_and_then_trusted(
    sigma: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = PrivacyLedger(PrivacyPolicy(1.0, DELTA))
    planned = ledger.plan("a", [_round(sigma, ROUNDS * STEPS_PER_ROUND)])
    assert 0.99 < planned <= 1.0
    assert ledger.planned("a") == (_round(sigma, ROUNDS * STEPS_PER_ROUND),)
    assert ledger.planned("b") == ()
    assert ledger.scopes == ()  # planning records no release

    calls = []
    original = PrivacyLedger._epsilon

    def counting(self: PrivacyLedger, events: object) -> float:
        calls.append(events)
        return original(self, events)  # type: ignore[arg-type]

    monkeypatch.setattr(PrivacyLedger, "_epsilon", counting)
    for _ in range(ROUNDS):
        assert ledger.can_afford("a", _round(sigma))
        ledger.charge("a", _round(sigma))
    assert calls == []  # every charge stayed inside the plan: no budget was recomputed

    # One step beyond the plan is checked in full, and refused.
    with pytest.raises(BudgetExceeded):
        ledger.charge("a", _round(sigma, 20))
    assert len(calls) == 1
    # Less noise than planned is not covered by the plan either.
    assert not ledger.can_afford("a", _round(sigma * 0.5, 1))


def test_an_impossible_plan_is_refused_and_reserves_nothing(sigma: float) -> None:
    ledger = PrivacyLedger(PrivacyPolicy(1.0, DELTA))
    with pytest.raises(BudgetExceeded, match="planned releases"):
        ledger.plan("a", [_round(sigma, 2 * ROUNDS * STEPS_PER_ROUND)])
    assert ledger.planned("a") == ()
    # A plan counts what was already charged.
    ledger.charge("a", _round(sigma, 150))
    with pytest.raises(BudgetExceeded):
        ledger.plan("a", [_round(sigma, 100)])
    assert ledger.plan("a", [_round(sigma, 50)]) <= 1.0


def test_the_rdp_accountant_charges_more_for_the_same_releases(sigma: float) -> None:
    events = [_round(sigma, 150)]
    tight = PrivacyLedger(PrivacyPolicy(1.0, DELTA))
    loose = PrivacyLedger(PrivacyPolicy(1.0, DELTA, accountant="rdp"))
    assert tight.epsilon_of(events) == epsilon_pld(events, DELTA)
    assert loose.epsilon_of(events) == epsilon_rdp(events, DELTA)
    assert loose.epsilon_of(events) > tight.epsilon_of(events)
    assert tight.epsilon_of([]) == 0.0
    # The whole run fits under PLD but not under the looser method.
    whole_run = _round(sigma, ROUNDS * STEPS_PER_ROUND)
    tight.charge("a", whole_run)
    with pytest.raises(BudgetExceeded):
        loose.charge("a", whole_run)


def test_mixed_releases_are_counted_together() -> None:
    ledger = PrivacyLedger(PrivacyPolicy(2.0, DELTA))
    ledger.charge("a", SubsampledGaussianEvent(3.0, 0.05, 100))
    after_training = ledger.spent("a")
    ledger.charge("a", GaussianEvent(6.0))  # for example one noisy statistic
    assert ledger.spent("a") > after_training
    assert len(ledger.events("a")) == 2


def test_ledger_round_trips_through_a_dictionary(sigma: float) -> None:
    ledger = PrivacyLedger(PrivacyPolicy(1.0, DELTA, per_scope_epsilon={"b": 0.5}))
    ledger.charge("a", _round(sigma, 100))
    ledger.charge("a", GaussianEvent(40.0))
    ledger.charge("b", _round(sigma, 30))
    copy = PrivacyLedger.from_dict(ledger.to_dict())
    assert copy.policy == ledger.policy
    assert copy.scopes == ledger.scopes
    for scope in ledger.scopes:
        assert copy.events(scope) == ledger.events(scope)
        assert copy.spent(scope) == ledger.spent(scope)
    assert PrivacyLedger.from_dict({"policy": ledger.policy.to_dict()}).scopes == ()
