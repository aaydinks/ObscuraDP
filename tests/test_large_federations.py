"""What makes thousands of participants practical to play on one machine.

Participation by chance for unprotected baselines, no work for participants that sit a
round out, a simulated secure sum that gives what the real one gives, and scoring in pieces.
"""

import dataclasses
from typing import Any

import numpy as np
import pytest

import obscuradp as odp
from obscuradp.audit.mechanism import LinearProbe
from obscuradp.crypto import PairwiseMasking
from obscuradp.models import TabularMLP
from obscuradp.report import FULL_TABLE_UP_TO, collusion_counts
from obscuradp.sim import RoundRecord
from obscuradp.sim import simulation as simulation_module
from obscuradp.state import flatten_state
from obscuradp.tables import ClientData, make_client_data

DELTA = 1e-5

pytestmark = pytest.mark.filterwarnings("ignore:protecting whole participants with only")


def _client(name: str, rows: int, seed: int, width: int = 6) -> ClientData:
    generator = np.random.default_rng(seed)
    features = generator.normal(size=(rows, width)).astype(np.float32)
    signal = features[:, 0] + 0.5 * features[:, 1] + generator.normal(scale=0.5, size=rows)
    return make_client_data(name, features, (signal > 0).astype(np.float32), seed=seed)


@pytest.fixture(scope="module")
def clients() -> list[ClientData]:
    return [_client(f"site-{index}", 120, index) for index in range(8)]


def _plain(clients: list[ClientData], **overrides: Any) -> odp.Simulation:
    arguments: dict[str, Any] = {
        "model_fn": lambda: TabularMLP(6, (8,)),
        "clients": clients,
        "plan": odp.Plan(dp=None),
        "policy": None,
        "rounds": 6,
        "learning_rate": 0.1,
        "evaluate_every": 0,
        **overrides,
    }
    return odp.Simulation(**arguments)


# --------------------------------------------------------------------------- participation


def test_only_some_participants_take_part_in_each_round(clients: list[ClientData]) -> None:
    records: list[RoundRecord] = []
    result = _plain(clients, participation=0.5).run(seed=1, observer=records.append)
    counts = [len(record.messages) for record in records]
    assert len(counts) == 6
    assert 0 < min(counts) < max(counts) <= 8  # it varies from round to round
    assert 12 <= sum(counts) <= 36  # about half of 48
    assert result.config["participation"] == 0.5
    # The same seed, the same participants; and the run still learns.
    again: list[RoundRecord] = []
    repeated = _plain(clients, participation=0.5).run(seed=1, observer=again.append)
    assert [sorted(record.messages) for record in again] == [sorted(r.messages) for r in records]
    assert np.array_equal(flatten_state(repeated.state), flatten_state(result.state))
    assert result.test["auc"] > 0.7


def test_everybody_taking_part_is_the_run_as_it_always_was(clients: list[ClientData]) -> None:
    usual = _plain(clients).run(seed=2)
    stated = _plain(clients, participation=1.0).run(seed=2)
    assert np.array_equal(flatten_state(usual.state), flatten_state(stated.state))
    assert "participation" not in usual.config


def test_a_round_in_which_nobody_takes_part_leaves_the_model_alone(
    clients: list[ClientData],
) -> None:
    records: list[RoundRecord] = []
    _plain(clients[:2], participation=0.01, rounds=3).run(seed=3, observer=records.append)
    assert all(len(record.messages) == 0 for record in records)
    assert all(not np.any(record.total) for record in records)  # type: ignore[arg-type]


def test_participation_is_refused_where_the_plan_decides_who_takes_part(
    clients: list[ClientData],
) -> None:
    with pytest.raises(ValueError, match=r"participation must be in \(0, 1\]"):
        _plain(clients, participation=0.0)
    with pytest.raises(ValueError, match="plans without protection"):
        _plain(
            clients,
            participation=0.5,
            plan=odp.Plan(dp=odp.LocalDPSGD()),
            policy=odp.PrivacyPolicy(1.0, DELTA),
        )
    with pytest.raises(ValueError, match="plans without protection"):
        _plain(clients, participation=0.5, plan=odp.Plan(dp=None, secure_sum=PairwiseMasking()))


# --------------------------------------------------------------------------- simulated secure sum


def _silent(count: int, width: int) -> list[ClientData]:
    """Participants whose rows pull nowhere: what they send is noise and nothing else."""
    labels = np.tile(np.array([0.0, 1.0], dtype=np.float32), 20)
    return [
        make_client_data(f"site-{i}", np.zeros((40, width), dtype=np.float32), labels, seed=i)
        for i in range(count)
    ]


def _whole(clients: list[ClientData], width: int, **overrides: Any) -> odp.Simulation:
    arguments: dict[str, Any] = {
        "model_fn": lambda: LinearProbe(width),
        "clients": clients,
        "plan": odp.Plan(
            dp=odp.ClientLevelDP(sampling_rate=1.0, clip_norm=1.0, max_colluding=1),
            secure_sum=PairwiseMasking(),
        ),
        "policy": odp.PrivacyPolicy(1.0, DELTA, unit="client"),
        "rounds": 1,
        "learning_rate": 1.0,
        "weighted_average": False,
        "evaluate_every": 0,
        "outputs": "logit",
        **overrides,
    }
    return odp.Simulation(**arguments)


def _opened(simulation: odp.Simulation, seed: int) -> np.ndarray:
    seen: list[RoundRecord] = []
    simulation.run(seed=seed, observer=seen.append)
    assert seen[0].total is not None
    return seen[0].total


def test_the_simulated_secure_sum_opens_what_the_real_one_opens() -> None:
    """Six participants: the noise in the opened sum has the same size either way."""
    width, count = 4000, 6
    group = _silent(count, width)
    real = _whole(group, width)
    played = _whole(group, width, crypto_in_loop=False)
    real_result = real.run(seed=1)
    played_result = played.run(seed=1)
    sigma = real_result.noise_multipliers["all participants"]
    assert played_result.noise_multipliers == real_result.noise_multipliers
    # Six slices sized for five honest ones: the opened sum has 6/5 of the charged noise.
    expected = sigma**2 * count / (count - 1)
    assert float(np.var(_opened(real, 1))) == pytest.approx(expected, rel=0.1)
    assert float(np.var(_opened(played, 1))) == pytest.approx(expected, rel=0.1)
    # The same budget is reported, the same assumptions, and the report recomputes.
    assert real_result.report is not None
    assert played_result.report is not None
    assert played_result.report.epsilon == real_result.report.epsilon
    assert played_result.report.mode == real_result.report.mode
    assert odp.verify_report(played_result.report.to_dict()) == []
    assert played_result.communication == pytest.approx(real_result.communication, abs=1.0)
    # What differs is said in the report and in the record of the run.
    assert any("simulated" in text for text in played_result.report.caveats)
    assert not any("simulated" in text for text in real_result.report.caveats)
    assert played_result.config["crypto_in_loop"] is False
    assert "crypto_in_loop" not in real_result.config
    assert "simulated" in played_result.report.summary()


def test_what_a_participant_contributes_arrives_the_same_way() -> None:
    """One participant pulls one way with the full clipping norm: the sums shift alike."""
    width = 6

    def group(pull: float) -> list[ClientData]:
        made = _silent(4, width)
        features = np.zeros((40, width), dtype=np.float32)
        features[:, 0] = pull
        labels = np.ones(40, dtype=np.float32)
        made[0] = make_client_data("site-0", features, labels, seed=0)
        return made

    # A loose budget, so that the noise is small enough to see the shift in 100 runs.
    loose = odp.PrivacyPolicy(8.0, DELTA, unit="client")

    def first_number(pull: float, in_loop: bool, seed: int) -> float:
        simulation = _whole(
            group(pull), width, crypto_in_loop=in_loop, learning_rate=5.0, policy=loose
        )
        return float(_opened(simulation, seed)[0])

    shifts = {}
    for in_loop in (True, False):
        with_pull = [first_number(3.0, in_loop, seed) for seed in range(100)]
        without = [first_number(0.0, in_loop, seed) for seed in range(100, 200)]
        shifts[in_loop] = float(np.mean(with_pull) - np.mean(without))
    # The shortened change is exactly one clipping norm in that direction, either way.
    assert shifts[True] == pytest.approx(1.0, abs=0.35)
    assert shifts[False] == pytest.approx(1.0, abs=0.35)


def test_only_the_secure_sum_of_whole_participants_can_be_simulated(
    clients: list[ClientData],
) -> None:
    with pytest.raises(ValueError, match="only the secure sum of ClientLevelDP"):
        _plain(clients, crypto_in_loop=False)
    with pytest.raises(ValueError, match="only the secure sum of ClientLevelDP"):
        _whole(
            clients,
            6,
            crypto_in_loop=False,
            plan=odp.Plan(dp=odp.ClientLevelDP(sampling_rate=1.0)),
            allow_unsafe=True,
        )
    with pytest.raises(ValueError, match="missing participants cannot be played"):
        _whole(clients, 6, crypto_in_loop=False, dropouts={0: ["site-1"]})


def test_a_simulated_secure_sum_runs_none_of_the_real_ones_machinery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What makes hundreds of participants quick: no key agreements, no masks, no messages."""

    def forbidden(*arguments: Any, **settings: Any) -> None:
        raise AssertionError("the simulated secure sum must not set up the real one")

    monkeypatch.setattr(simulation_module, "setup_pairwise_masking", forbidden)
    crowd = [_client(f"p-{index}", 30, index) for index in range(200)]
    simulation = odp.Simulation(
        model_fn=lambda: TabularMLP(6, (8,)),
        clients=crowd,
        plan=odp.Plan(
            dp=odp.ClientLevelDP(sampling_rate=0.1, clip_norm=1.0, max_colluding=20),
            secure_sum=PairwiseMasking(),
        ),
        policy=odp.PrivacyPolicy(8.0, DELTA, unit="client"),
        rounds=5,
        learning_rate=0.1,
        evaluate_every=0,
        crypto_in_loop=False,
    )
    records: list[RoundRecord] = []
    result = simulation.run(seed=1, observer=records.append)
    assert result.report is not None
    assert result.report.within_budget
    assert result.report.mode is not None
    assert result.report.mode.participants == 200
    assert len(result.report.mode.table) == len(collusion_counts(200, 20))
    assert "a selection of numbers" in result.report.summary()
    assert all(record.masked and not record.messages for record in records)
    # A run with the real secure sum does set it up (shown on five participants).
    real = dataclasses.replace(
        simulation,
        crypto_in_loop=True,
        clients=crowd[:5],
        plan=odp.Plan(
            dp=odp.ClientLevelDP(sampling_rate=0.1, clip_norm=1.0, max_colluding=1),
            secure_sum=PairwiseMasking(),
        ),
    )
    with pytest.raises(AssertionError, match="must not set up"):
        real.run(seed=1)


def test_those_who_sit_a_round_out_send_nothing_to_a_trusted_server(
    clients: list[ClientData],
) -> None:
    records: list[RoundRecord] = []
    odp.Simulation(
        model_fn=lambda: TabularMLP(6, (8,)),
        clients=clients,
        plan=odp.Plan(dp=odp.ClientLevelDP(sampling_rate=0.4, clip_norm=1.0)),
        policy=odp.PrivacyPolicy(8.0, DELTA, unit="client"),
        rounds=4,
        learning_rate=0.1,
        evaluate_every=0,
        allow_unsafe=True,
    ).run(seed=2, observer=records.append)
    silent = [
        sum(1 for message in record.messages.values() if not np.any(message)) for record in records
    ]
    assert all(len(record.messages) == 8 for record in records)
    assert 0 < sum(silent) < 32  # some sat out, some took part


# --------------------------------------------------------------------------- scoring in pieces


def test_scoring_in_pieces_gives_the_same_scores(
    clients: list[ClientData], monkeypatch: pytest.MonkeyPatch
) -> None:
    whole = _plain(clients).run(seed=5)
    monkeypatch.setattr(simulation_module, "SCORING_CHUNK", 7)
    pieces = _plain(clients).run(seed=5)
    assert pieces.test == pytest.approx(whole.test, abs=1e-6)
    assert pieces.validation == pytest.approx(whole.validation, abs=1e-6)


# --------------------------------------------------------------------------- the report's table


def test_small_groups_keep_a_row_for_every_number_of_colluders() -> None:
    assert collusion_counts(5, 1) == (0, 1, 2, 3, 4)
    assert collusion_counts(FULL_TABLE_UP_TO, 3) == tuple(range(FULL_TABLE_UP_TO))


def test_large_groups_keep_a_selection_of_rows() -> None:
    counts = collusion_counts(3400, 340)
    assert len(counts) < 30
    assert counts == tuple(sorted(set(counts)))
    # The first few, the assumed number and the one after it, shares of the group, the last.
    assert set(range(9)) <= set(counts)
    assert {340, 341, 34, 1700, 3060, 3399} <= set(counts)
    assert max(counts) == 3399
    # An assumed number near the end does not run past the group.
    assert max(collusion_counts(100, 99)) == 99
