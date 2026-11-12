"""The shadow-model attack: each record judged against its own history in other runs."""

import dataclasses
from collections.abc import Mapping, Sequence

import numpy as np
import pytest
import torch

import obscuradp as odp
from obscuradp.audit import selftest, shadow
from obscuradp.audit.membership import leave_out, measured_advantage
from obscuradp.models import TabularMLP
from obscuradp.state import flatten_state, get_state
from obscuradp.tables import ClientData


def _made_up_runs(
    runs: int, records: int, shift: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Confidence values in which records differ far more from each other than in from out."""
    generator = np.random.default_rng(seed)
    difficulty = generator.normal(scale=3.0, size=records)  # the same in every run
    kept = generator.random((runs, records)) < 0.5
    values = difficulty + shift * kept + generator.normal(scale=0.5, size=(runs, records))
    return values, kept


def test_confidence_is_the_log_odds_of_the_true_label() -> None:
    features = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]], dtype=np.float32)
    labels = np.array([1.0, 0.0, 1.0])
    for probability_output in (True, False):
        model = TabularMLP(2, (), probability_output=probability_output)
        state = get_state(model)
        state["net.0.weight"] = torch.tensor([[2.0, -1.0]])
        state["net.0.bias"] = torch.tensor([0.5])
        found = shadow.confidence(
            model, state, features, labels, from_logits=not probability_output
        )
        # Raw scores 2.5, -1.5, -2.5; a record with label 0 counts the other way round.
        assert found == pytest.approx([2.5, 1.5, -2.5], abs=1e-5)
    # A model that is certain stays finite.
    state["net.0.bias"] = torch.tensor([500.0])
    sure = shadow.confidence(model, state, features, labels, from_logits=True)
    assert np.all(np.isfinite(sure))
    assert abs(float(sure[0])) < 20.0


def test_judging_each_record_against_itself_finds_what_one_cut_off_misses() -> None:
    values, kept = _made_up_runs(runs=33, records=3000, shift=1.0, seed=0)
    scores = shadow.likelihood_ratio_scores(values[0], values[1:], kept[1:])
    plain = measured_advantage(values[0], kept[0])
    informed = measured_advantage(scores, kept[0])
    # One cut-off for all records sees almost nothing; the records' own histories do.
    assert plain < 0.2
    assert informed > 0.6
    # Two bell curves a full two spreads apart could give 2 * Phi(1) - 1 = 0.68 at best.
    assert informed < 0.75


def test_records_that_do_not_react_to_being_used_give_nothing() -> None:
    values, kept = _made_up_runs(runs=33, records=3000, shift=0.0, seed=1)
    scores = shadow.likelihood_ratio_scores(values[0], values[1:], kept[1:])
    assert abs(measured_advantage(scores, kept[0])) < 0.06


def test_a_record_with_too_little_history_is_judged_like_all_records() -> None:
    values, kept = _made_up_runs(runs=9, records=50, shift=1.0, seed=2)
    kept[1:, 0] = True  # never out in the other runs
    kept[1:, 1] = False  # never in
    kept[1:, 2] = [True] + [False] * 7  # in only once
    scores = shadow.likelihood_ratio_scores(values[0], values[1:], kept[1:])
    assert np.all(np.isfinite(scores))


def test_scores_need_tables_that_fit() -> None:
    values, kept = _made_up_runs(runs=9, records=50, shift=1.0, seed=3)
    with pytest.raises(ValueError, match="one row per shadow run"):
        shadow.likelihood_ratio_scores(values[0, :-1], values[1:], kept[1:])
    with pytest.raises(ValueError, match="one row per shadow run"):
        shadow.likelihood_ratio_scores(values[0], values[1:], kept[2:])
    with pytest.raises(ValueError, match="one value per record"):
        shadow.likelihood_ratio_scores(values[:1], values[1:, :1], kept[1:, :1])


def test_every_run_is_attacked_with_the_others_or_with_a_group_set_apart() -> None:
    values, kept = _made_up_runs(runs=12, records=400, shift=1.0, seed=4)
    everyone = shadow.attack_each_run(values, kept)
    assert sorted(everyone) == list(range(12))
    assert everyone[3] == pytest.approx(
        shadow.likelihood_ratio_scores(values[3], np.delete(values, 3, 0), np.delete(kept, 3, 0))
    )
    apart = shadow.attack_each_run(values, kept, targets=range(6), shadows=range(6, 12))
    assert sorted(apart) == list(range(6))
    assert apart[0] == pytest.approx(
        shadow.likelihood_ratio_scores(values[0], values[6:], kept[6:])
    )
    with pytest.raises(ValueError, match="own shadow"):
        shadow.attack_each_run(values, kept, targets=[0], shadows=[0, 1, 2, 3])
    with pytest.raises(ValueError, match="too few shadow runs"):
        shadow.attack_each_run(values, kept, targets=[0], shadows=[1, 2, 3])
    with pytest.raises(ValueError, match="one row per run"):
        shadow.attack_each_run(values, kept[:-1])


@pytest.fixture(scope="module")
def clients() -> list[ClientData]:
    return selftest.made_up_participants(seed=21)


def test_a_run_on_a_random_half_judges_every_row(clients: list[ClientData]) -> None:
    half = shadow.random_half_run(selftest.correct_pipeline, clients, seed=3)
    assert half.kept.shape == half.labels.shape == half.confidence.shape == (720,)
    assert 0.4 < float(half.kept.mean()) < 0.6
    assert np.all(np.isfinite(half.confidence))
    assert np.array_equal(half.labels, np.concatenate([c.train_y for c in clients]))
    assert half.result.report is not None
    assert half.result.report.epsilon == pytest.approx(1.0, abs=0.01)
    # It is the run one gets by leaving the same rows out by hand, with the sizes declared.
    reduced, declared = leave_out(clients, shadow.all_rows(clients, 3))
    assert sum(client.n_train for client in reduced) == int(half.kept.sum())
    plain = selftest.correct_pipeline(reduced, declared).run(seed=3)
    assert np.array_equal(flatten_state(plain.state), flatten_state(half.result.state))
    # Another seed, other coins.
    assert not np.array_equal(shadow.all_rows(clients, 4).coins(), half.kept)

    def forgetful(rows: Sequence[ClientData], declared: Mapping[str, int]) -> odp.Simulation:
        return dataclasses.replace(selftest.correct_pipeline(rows, declared), declared_rows=None)

    with pytest.raises(ValueError, match="declared_rows"):
        shadow.random_half_run(forgetful, clients, seed=3)


def _memorizing(rows: Sequence[ClientData], declared: Mapping[str, int]) -> odp.Simulation:
    """A network far too large for its table, trained hard and without protection."""
    return odp.Simulation(
        model_fn=lambda: TabularMLP(selftest.WIDTH, (64,)),
        clients=rows,
        plan=odp.Plan(dp=None),
        policy=None,
        rounds=30,
        local_epochs=2.0,
        batch_size=16,
        learning_rate=0.3,
        evaluate_every=0,
        declared_rows=declared,
    )


def test_a_model_that_learns_its_rows_by_heart_is_found_out() -> None:
    small = [
        dataclasses.replace(client, train_x=client.train_x[:40], train_y=client.train_y[:40])
        for client in selftest.made_up_participants(seed=5)
    ]
    runs = [shadow.random_half_run(_memorizing, small, seed) for seed in range(12)]
    values = np.stack([run.confidence for run in runs])
    kept = np.stack([run.kept for run in runs])
    scores = shadow.attack_each_run(values, kept)
    advantage = float(np.mean([measured_advantage(scores[run], kept[run]) for run in scores]))
    assert advantage > 0.3
