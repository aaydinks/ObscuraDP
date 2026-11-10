"""The simulation: a total budget for the whole run, clean evaluation, unsafe settings refused."""

from typing import Any

import numpy as np
import pytest
import torch

import obscuradp as odp
from obscuradp.accounting import (
    SubsampledGaussianEvent,
    calibrate_noise_multiplier,
    steps_per_round,
)
from obscuradp.imbalance import ClassWeighted, NoRebalancing, UnsafeOversampling
from obscuradp.metrics import best_threshold
from obscuradp.models import TabularMLP
from obscuradp.results import validate_result
from obscuradp.state import set_state
from obscuradp.tables import ClientData, make_client_data
from obscuradp.unsafe import UnsafeConfiguration

DELTA = 1e-5
ROUNDS = 4
SIZES = (900, 600, 400)


def _client(name: str, n: int, seed: int) -> ClientData:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 6)).astype(np.float32)
    signal = x[:, 0] + 0.6 * x[:, 1] - 0.4 * x[:, 2] + rng.normal(scale=0.5, size=n)
    return make_client_data(name, x, (signal > 1.0).astype(np.float32), seed=seed)


@pytest.fixture(scope="module")
def clients() -> list[ClientData]:
    return [_client(f"site-{i}", n, i) for i, n in enumerate(SIZES)]


def _model() -> TabularMLP:
    return TabularMLP(6, (16,))


def _simulation(clients: list[ClientData], **overrides: Any) -> odp.Simulation:
    arguments: dict[str, Any] = {
        "model_fn": _model,
        "clients": clients,
        "plan": odp.Plan(dp=odp.LocalDPSGD(clip_norm=1.0), imbalance=ClassWeighted(0.2)),
        "policy": odp.PrivacyPolicy(1.0, DELTA),
        "rounds": ROUNDS,
        "local_epochs": 1.0,
        "batch_size": 32,
        "learning_rate": 0.1,
        **overrides,
    }
    return odp.Simulation(**arguments)


def _baseline(clients: list[ClientData], **overrides: Any) -> odp.Simulation:
    plan = overrides.pop("plan", odp.Plan(dp=None, imbalance=ClassWeighted(0.2)))
    return _simulation(clients, plan=plan, policy=None, **overrides)


@pytest.fixture(scope="module")
def private_result(clients: list[ClientData]) -> odp.SimulationResult:
    return _simulation(clients, choose_threshold=True).run(seed=1)


# --------------------------------------------------------------------------- the private run


def test_the_noise_is_calibrated_once_for_the_whole_run(
    clients: list[ClientData], private_result: odp.SimulationResult
) -> None:
    report = private_result.report
    assert report is not None
    assert set(private_result.noise_multipliers) == {"site-0", "site-1", "site-2"}
    for client, scope in zip(clients, report.scopes, strict=True):
        rate = 32 / client.n_train
        steps = ROUNDS * steps_per_round(1.0, rate)
        expected = calibrate_noise_multiplier(1.0, DELTA, rate, steps).noise_multiplier
        assert private_result.noise_multipliers[client.name] == expected
        # The ledger holds exactly the steps that were run: all rounds, nothing else.
        assert scope.events == (SubsampledGaussianEvent(expected, rate, steps),)
        assert 0.99 < scope.epsilon <= 1.0
    # Smaller participants sample each row more often and need more noise.
    levels = [private_result.noise_multipliers[c.name] for c in clients]
    assert levels == sorted(levels)


def test_the_report_is_complete_and_can_be_recomputed(private_result: odp.SimulationResult) -> None:
    report = private_result.report
    assert report is not None
    assert 0.99 < report.epsilon <= 1.0
    assert report.within_budget
    assert report.is_safe
    assert odp.verify_report(report.to_dict()) == []
    text = report.summary()
    assert "adds its own noise before anything leaves its machine" in text
    assert "Validation and test rows are not protected" in text
    assert "drawn from the run's seed" in text
    assert "threshold was chosen on validation rows" in text
    assert "UNSAFE" not in text


def test_scores_come_from_validation_rows_and_test_rows_are_scored_once(
    clients: list[ClientData], private_result: odp.SimulationResult
) -> None:
    assert [entry["round"] for entry in private_result.history] == [1, 2, 3, 4]
    final = {k: v for k, v in private_result.history[-1].items() if k != "round"}
    assert final == private_result.validation
    assert set(private_result.test) == set(private_result.validation)
    assert private_result.test != private_result.validation
    assert private_result.test["auc"] > 0.85  # the model did learn

    # The threshold is the best one on the validation rows, then applied to the test rows.
    model = _model()
    set_state(model, private_result.state)
    model.eval()
    with torch.no_grad():
        validation_scores = model(
            torch.as_tensor(np.concatenate([c.validation_x for c in clients]))
        )
    labels = np.concatenate([c.validation_y for c in clients])
    assert private_result.threshold == best_threshold(validation_scores.numpy(), labels)
    assert private_result.test_at_chosen_threshold is not None
    assert private_result.test_at_chosen_threshold["auc"] == private_result.test["auc"]
    assert private_result.test_at_chosen_threshold["f1"] != private_result.test["f1"]


def test_the_result_record_carries_its_origin(
    private_result: odp.SimulationResult, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    record = private_result.to_record(directory=str(tmp_path))
    validate_result(record)
    assert record["kind"] == "simulation"
    assert record["manifest"]["seeds"] == [1]
    assert record["config"]["plan"] == {
        "dp": {"clip_norm": 1.0},
        "imbalance": "ClassWeighted",
        "positive_prior": 0.2,
        "proximal_mu": 0.0,
    }
    assert record["config"]["clients"] == {"site-0": 630, "site-1": 420, "site-2": 280}
    assert record["data"]["privacy_report"]["schema"] == "obscuradp.privacy_report/1"
    assert odp.verify_report(record["data"]["privacy_report"]) == []
    assert len(record["data"]["history"]) == ROUNDS


def test_same_seed_same_run_other_seed_other_run(
    clients: list[ClientData], private_result: odp.SimulationResult
) -> None:
    again = _simulation(clients, choose_threshold=True).run(seed=1)
    other = _simulation(clients, choose_threshold=True).run(seed=2)
    for name, value in private_result.state.items():
        assert torch.equal(value, again.state[name])
    assert any(not torch.equal(v, other.state[n]) for n, v in private_result.state.items())
    assert again.test == private_result.test


def test_operating_system_noise_is_not_repeatable_and_drops_the_seed_caveat(
    clients: list[ClientData], private_result: odp.SimulationResult
) -> None:
    secure = _simulation(clients, secure_noise=True).run(seed=1)
    assert any(not torch.equal(v, secure.state[n]) for n, v in private_result.state.items())
    assert secure.report is not None
    assert "drawn from the run's seed" not in secure.report.summary()
    assert secure.noise_multipliers == private_result.noise_multipliers


def test_class_weights_take_effect_under_clipping(clients: list[ClientData]) -> None:
    """Without the per-class clipping limit, clipping would cancel the weights."""
    plain = _simulation(clients, plan=odp.Plan(imbalance=NoRebalancing())).run(seed=1)
    weighted = _simulation(clients).run(seed=1)
    assert weighted.test["recall"] > plain.test["recall"] + 0.3
    assert weighted.noise_multipliers == plain.noise_multipliers  # same privacy cost


def test_a_stricter_participant_gets_more_noise(clients: list[ClientData]) -> None:
    policy = odp.PrivacyPolicy(1.0, DELTA, accountant="rdp", per_scope_epsilon={"site-1": 0.5})
    simulation = _simulation(clients, policy=policy)
    default = _simulation(clients, policy=odp.PrivacyPolicy(1.0, DELTA, accountant="rdp"))
    assert (
        simulation.calibrations()["site-1"].noise_multiplier
        > 1.5 * default.calibrations()["site-1"].noise_multiplier
    )
    assert simulation.calibrations()["site-0"] == default.calibrations()["site-0"]
    report = simulation.run(seed=1).report
    assert report is not None
    budgets = {scope.scope: (scope.budget, scope.epsilon) for scope in report.scopes}
    assert budgets["site-1"][0] == 0.5
    assert budgets["site-1"][1] <= 0.5
    assert budgets["site-0"][0] == 1.0


def test_a_budget_that_cannot_be_met_stops_before_training(clients: list[ClientData]) -> None:
    built = []

    def counting_model() -> TabularMLP:
        built.append(1)
        return _model()

    simulation = _simulation(
        clients, model_fn=counting_model, policy=odp.PrivacyPolicy(1e-4, DELTA, accountant="rdp")
    )
    with pytest.raises(ValueError, match="too small"):
        simulation.run(seed=1)
    assert len(built) == 2  # models were built, but no round was run


# --------------------------------------------------------------------------- the baseline


def test_the_unprotected_baseline_has_no_report_and_learns(clients: list[ClientData]) -> None:
    result = _baseline(clients).run(seed=1)
    assert result.report is None
    assert result.noise_multipliers == {}
    assert result.test["auc"] > 0.93
    assert result.to_record()["data"]["privacy_report"] is None
    assert _baseline(clients).calibrations() == {}
    same = _baseline(clients).run(seed=1)
    assert all(torch.equal(v, same.state[n]) for n, v in result.state.items())


def test_averaging_and_the_proximal_term_change_the_result(clients: list[ClientData]) -> None:
    base = _baseline(clients).run(seed=1)
    unweighted = _baseline(clients, weighted_average=False).run(seed=1)
    proximal = _baseline(
        clients, plan=odp.Plan(dp=None, imbalance=ClassWeighted(0.2), proximal_mu=1.0)
    ).run(seed=1)
    adam = _baseline(clients, optimizer="adam", learning_rate=0.01).run(seed=1)
    for other in (unweighted, proximal, adam):
        assert any(not torch.equal(v, other.state[n]) for n, v in base.state.items())
        assert other.test["auc"] > 0.9


def test_models_that_return_raw_scores_are_scored_as_probabilities(
    clients: list[ClientData],
) -> None:
    result = _baseline(
        clients, model_fn=lambda: TabularMLP(6, (16,), probability_output=False)
    ).run(seed=1)
    assert result.test["auc"] > 0.93
    assert 0.0 < result.test["loss"] < 1.0
    forced = _baseline(clients, outputs="probability").run(seed=1)
    assert forced.test["auc"] > 0.93


def test_evaluation_schedule_and_missing_parts() -> None:
    clients = [_client("a", 200, 0), _client("b", 200, 1)]
    final_only = _baseline(clients, evaluate_every=0).run(seed=1)
    assert [entry["round"] for entry in final_only.history] == [ROUNDS]
    every_third = _baseline(clients, evaluate_every=3).run(seed=1)
    assert [entry["round"] for entry in every_third.history] == [3, 4]

    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 6)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.float32)
    no_holdout = [make_client_data("a", x, y, validation=0.0, test=0.0, seed=0)]
    result = _baseline(no_holdout, choose_threshold=True).run(seed=1)
    assert result.history == ()
    assert result.validation == {}
    assert result.test == {}
    assert result.threshold == 0.5
    assert result.test_at_chosen_threshold is None


# --------------------------------------------------------------------------- unsafe settings


def _double(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate([x, x]), np.concatenate([y, y])


def test_oversampling_in_a_private_run_is_refused(clients: list[ClientData]) -> None:
    plan = odp.Plan(imbalance=UnsafeOversampling(_double, name="doubling"))
    with pytest.raises(UnsafeConfiguration, match="one real record influences many training rows"):
        _simulation(clients, plan=plan)


def test_forced_oversampling_runs_and_is_flagged_in_the_report(clients: list[ClientData]) -> None:
    plan = odp.Plan(imbalance=UnsafeOversampling(_double, name="doubling"))
    policy = odp.PrivacyPolicy(1.0, DELTA, accountant="rdp")
    simulation = _simulation(clients, plan=plan, policy=policy, allow_unsafe=True)
    # The noise is sized for the enlarged tables, which is exactly why the budget is void.
    assert simulation.calibrations()["site-0"].sampling_rate == pytest.approx(32 / 1260)
    report = simulation.run(seed=1).report
    assert report is not None
    assert not report.is_safe
    assert report.within_budget
    assert len(report.unsafe_flags) == 1
    assert "UNSAFE: the stated guarantee is weakened or void" in report.summary()
    assert "Synthetic rows were built from real rows" in report.summary()
    assert report.to_dict()["unsafe_flags"] == list(report.unsafe_flags)


def test_oversampling_without_a_privacy_budget_needs_no_flag(clients: list[ClientData]) -> None:
    plan = odp.Plan(dp=None, imbalance=UnsafeOversampling(_double))
    result = _baseline(clients, plan=plan).run(seed=1)
    assert result.report is None
    assert result.test["auc"] > 0.9


def test_weights_from_private_labels_need_the_flag_twice(clients: list[ClientData]) -> None:
    weights = ClassWeighted.from_private_labels(clients[0].train_y, allow_unsafe=True)
    with pytest.raises(UnsafeConfiguration, match="private label counts"):
        _simulation(clients, plan=odp.Plan(imbalance=weights))
    policy = odp.PrivacyPolicy(1.0, DELTA, accountant="rdp")
    report = (
        _simulation(clients, plan=odp.Plan(imbalance=weights), policy=policy, allow_unsafe=True)
        .run(seed=1)
        .report
    )
    assert report is not None
    assert any("private label counts" in flag for flag in report.unsafe_flags)


def test_a_failure_probability_that_is_too_large_is_refused(clients: list[ClientData]) -> None:
    weak = odp.PrivacyPolicy(1.0, 1 / 500, accountant="rdp")  # two participants have more rows
    with pytest.raises(UnsafeConfiguration, match="not below one over the number of records"):
        _simulation(clients, policy=weak)
    report = _simulation(clients, policy=weak, allow_unsafe=True).run(seed=1).report
    assert report is not None
    assert any("affected: site-0" in flag and "site-2" not in flag for flag in report.unsafe_flags)


# --------------------------------------------------------------------------- refused settings


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"policy": None}, "needs a PrivacyPolicy"),
        ({"plan": odp.Plan(dp=None)}, "the plan adds no noise"),
        ({"policy": odp.PrivacyPolicy(1.0, DELTA, unit="client")}, "only record-level"),
        ({"rounds": 0}, "rounds"),
        ({"local_epochs": 0.0}, "local_epochs"),
        ({"batch_size": 0}, "batch_size"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"optimizer": "lbfgs"}, "optimizer"),
        ({"evaluate_every": -1}, "evaluate_every"),
        ({"outputs": "scores"}, "outputs"),
        ({"clients": []}, "at least one participant"),
    ],
)
def test_bad_settings_are_refused(
    clients: list[ClientData], overrides: dict[str, Any], message: str
) -> None:
    chosen = overrides.get("clients", clients)
    rest = {key: value for key, value in overrides.items() if key != "clients"}
    with pytest.raises(ValueError, match=message):
        _simulation(chosen, **rest)


def test_bad_participants_are_refused(clients: list[ClientData]) -> None:
    with pytest.raises(ValueError, match="names must be distinct"):
        _simulation([clients[0], clients[0]])
    rng = np.random.default_rng(0)
    narrow = make_client_data(
        "narrow", rng.normal(size=(40, 5)), (rng.random(40) > 0.5).astype(float), seed=0
    )
    with pytest.raises(ValueError, match="same columns"):
        _simulation([clients[0], narrow])
    with pytest.raises(ValueError, match="clip_norm"):
        odp.LocalDPSGD(clip_norm=0.0)
    with pytest.raises(ValueError, match="proximal_mu"):
        odp.Plan(proximal_mu=-1.0)
