"""``ClientLevelDP``: a budget that protects a whole participant, not one row.

What must hold: the plan only runs with a budget for whole participants, no participant can
move the sum by more than the clipping norm, the noise is sized for one whole participant
and survives the assumed colluders, a trusted server has to be asked for, and small groups
are warned.
"""

import warnings
from typing import Any

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import nn

import obscuradp as odp
from obscuradp.accounting import SubsampledGaussianEvent, calibrate_noise_multiplier
from obscuradp.audit import gaussian_epsilon_lower_bound
from obscuradp.audit.mechanism import whole_participant_scores
from obscuradp.client import clipped_model_change
from obscuradp.crypto import PairwiseMasking
from obscuradp.imbalance import ClassWeighted
from obscuradp.mechanisms import gaussian_noise, share_std
from obscuradp.models import TabularMLP
from obscuradp.plan import SMALL_FEDERATION, SmallFederationWarning
from obscuradp.results import validate_result
from obscuradp.seeding import derive_seed, torch_generator
from obscuradp.sim import simulation as simulation_module
from obscuradp.state import flatten_state, get_state
from obscuradp.tables import ClientData, make_client_data
from obscuradp.unsafe import UnsafeConfiguration

# Matched by its text, not by its class: another test re-imports the package, after which a
# filter that names the class would look at a different class object than the one raised.
pytestmark = pytest.mark.filterwarnings("ignore:protecting whole participants with only")

DELTA = 1e-5
ROUNDS = 6
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


def _shared(rate: float = 1.0, colluding: int = 1, **settings: Any) -> odp.Plan:
    return odp.Plan(
        dp=odp.ClientLevelDP(sampling_rate=rate, max_colluding=colluding, **settings),
        imbalance=ClassWeighted(0.2),
        secure_sum=PairwiseMasking(),
    )


def _served(rate: float = 1.0, **settings: Any) -> odp.Plan:
    return odp.Plan(
        dp=odp.ClientLevelDP(sampling_rate=rate, **settings), imbalance=ClassWeighted(0.2)
    )


def _simulation(clients: list[ClientData], **overrides: Any) -> odp.Simulation:
    arguments: dict[str, Any] = {
        "model_fn": _model,
        "clients": clients,
        "plan": _shared(),
        "policy": odp.PrivacyPolicy(8.0, DELTA, unit="client"),
        "rounds": ROUNDS,
        "learning_rate": 0.1,
        **overrides,
    }
    return odp.Simulation(**arguments)


@pytest.fixture(scope="module")
def shared_result(clients: list[ClientData]) -> odp.SimulationResult:
    return _simulation(clients).run(seed=1)


# --------------------------------------------------------------------------- refusals


def test_the_plan_checks_its_own_settings() -> None:
    for arguments, message in (
        ({"sampling_rate": 0.0}, "sampling_rate"),
        ({"sampling_rate": 1.5}, "sampling_rate"),
        ({"sampling_rate": 0.5, "clip_norm": 0.0}, "clip_norm"),
        ({"sampling_rate": 0.5, "max_colluding": -1}, "max_colluding"),
        ({"sampling_rate": 0.5, "max_colluding": 1.5}, "max_colluding"),
        ({"sampling_rate": 0.5, "max_dropouts": True}, "max_dropouts"),
    ):
        with pytest.raises(ValueError, match=message):
            odp.ClientLevelDP(**arguments)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        odp.ClientLevelDP()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="max_colluding only has a meaning"):
        odp.Plan(dp=odp.ClientLevelDP(sampling_rate=0.5, max_colluding=1))
    with pytest.raises(ValueError, match="needs max_colluding"):
        odp.Plan(dp=odp.ClientLevelDP(sampling_rate=0.5), secure_sum=PairwiseMasking())
    with pytest.raises(ValueError, match="plans the range and rounding of its messages itself"):
        odp.Plan(
            dp=odp.ClientLevelDP(sampling_rate=0.5, max_colluding=1),
            secure_sum=PairwiseMasking(value_bits=8),
        )
    alone = odp.ClientLevelDP(sampling_rate=0.5)
    with pytest.raises(ValueError, match="max_colluding is needed"):
        alone.honest_shares(5)
    with pytest.raises(ValueError, match="max_colluding is needed"):
        alone.share_noise_multiplier(1.0, 5)
    assert not _shared().single_step


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"policy": odp.PrivacyPolicy(8.0, DELTA)}, "goes with ClientLevelDP"),
        (
            {"policy": odp.PrivacyPolicy(8.0, DELTA, unit="client", per_scope_epsilon={"a": 1.0})},
            "per_scope_epsilon does not apply",
        ),
        ({"plan": _shared(colluding=3)}, "nobody is left to supply the noise"),
        (
            {"plan": _served(), "allow_unsafe": True, "dropouts": {0: ["site-0"]}},
            "where the participants share the noise",
        ),
    ],
)
def test_settings_that_do_not_fit_are_refused(
    clients: list[ClientData], overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _simulation(clients, **overrides)


def test_a_budget_for_whole_participants_needs_this_plan(clients: list[ClientData]) -> None:
    policy = odp.PrivacyPolicy(8.0, DELTA, unit="client")
    with pytest.raises(ValueError, match="goes with ClientLevelDP"):
        odp.Simulation(_model, clients, odp.Plan(), policy, rounds=2)
    with pytest.raises(ValueError, match="at least two of them"):
        _simulation(clients[:1], plan=_served(), allow_unsafe=True)


def test_a_trusted_server_must_be_asked_for(clients: list[ClientData]) -> None:
    with pytest.raises(UnsafeConfiguration, match="not protected from the server"):
        _simulation(clients, plan=_served())
    result = _simulation(clients, plan=_served(), allow_unsafe=True).run(seed=1)
    report = result.report
    assert report is not None
    assert report.within_budget
    assert not report.is_safe
    text = report.summary()
    assert "by the server, which sees every participant's change to the model" in text
    assert "Participants are not protected from the server." in text
    assert report.mode is not None
    assert report.mode.to_dict() == {
        "mode": "participant",
        "participants": 3,
        "attempts": ROUNDS,
        "completed_rounds": ROUNDS,
        "sampling_rate": 1.0,
    }
    assert odp.verify_report(report.to_dict()) == []
    assert result.communication["ratio"] == 1.0


def test_a_small_group_is_warned(clients: list[ClientData]) -> None:
    with pytest.warns(SmallFederationWarning, match="only 3 of them"):
        _simulation(clients)
    # A thousand participants: no warning, and no such line in the settings check.
    table = _client("site", 40, 0)
    crowd = [
        ClientData(
            f"site-{index}",
            table.train_x,
            table.train_y,
            table.validation_x,
            table.validation_y,
            table.test_x,
            table.test_y,
        )
        for index in range(SMALL_FEDERATION)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        large = _simulation(crowd, plan=_served(rate=0.05), allow_unsafe=True)
    assert not any("participants, hiding one whole" in text for text in large._caveats())
    assert any("participants, hiding one whole" in t for t in _simulation(clients)._caveats())


# --------------------------------------------------------------------------- one change


@settings(max_examples=100, deadline=None)
@given(
    before=st.lists(st.floats(-1e6, 1e6), min_size=5, max_size=5),
    after=st.lists(st.floats(-1e6, 1e6), min_size=5, max_size=5),
    clip=st.floats(0.01, 100.0),
    weight=st.floats(0.01, 1.0),
)
def test_no_participant_moves_the_sum_by_more_than_the_clipping_norm(
    before: list[float], after: list[float], clip: float, weight: float
) -> None:
    model = nn.Linear(4, 1)
    start = [torch.tensor(before[:4]).view(1, 4), torch.tensor(before[4:])]
    with torch.no_grad():
        model.weight.copy_(torch.tensor(after[:4]).view(1, 4))
        model.bias.copy_(torch.tensor(after[4:]))
    change = clipped_model_change(model, start, clip, weight)
    assert change.dtype == torch.float64
    assert float(change.norm()) <= clip * weight
    moved = (
        torch.tensor(after, dtype=torch.float32).double()
        - torch.tensor(before, dtype=torch.float32).double()
    )
    if float(moved.norm()) <= 0.99 * clip:
        assert torch.allclose(change, moved * weight)


def test_a_change_that_is_not_a_number_counts_for_nothing() -> None:
    model = nn.Linear(2, 1)
    start = [value.detach().clone() for value in model.parameters()]
    with torch.no_grad():
        model.weight.fill_(float("nan"))
    assert not clipped_model_change(model, start, 1.0).any()
    for arguments, message in (
        ((start, 0.0), "clip_norm"),
        ((start, 1.0, 0.0), "weight"),
        ((start, 1.0, 1.5), "weight"),
        ((start[:1], 1.0), "one tensor per trainable parameter"),
    ):
        with pytest.raises(ValueError, match=message):
            clipped_model_change(model, *arguments)


# --------------------------------------------------------------------------- the budget


def test_one_release_per_round_for_the_whole_group(
    clients: list[ClientData], shared_result: odp.SimulationResult
) -> None:
    report = shared_result.report
    assert report is not None
    expected = calibrate_noise_multiplier(8.0, DELTA, 1.0, ROUNDS)
    assert shared_result.noise_multipliers == {"all participants": expected.noise_multiplier}
    assert [scope.scope for scope in report.scopes] == ["all participants"]
    assert report.scopes[0].events == (
        SubsampledGaussianEvent(expected.noise_multiplier, 1.0, ROUNDS),
    )
    assert 0.99 * 8.0 < report.epsilon <= 8.0
    assert report.is_safe
    assert report.policy.unit == "client"
    stored = report.to_dict()
    assert odp.verify_report(stored) == []
    assert stored["mode"]["mode"] == "participant"
    assert stored["mode"]["sampling_rate"] == 1.0
    assert len(stored["mode"]["collusion_table"]) == 3
    # The stated rate has to be the one the releases were made at.
    stored["mode"]["sampling_rate"] = 0.5
    assert any("sampling rate does not match" in text for text in odp.verify_report(stored))
    record = shared_result.to_record()
    validate_result(record)
    assert record["config"]["plan"]["dp"] == {
        "kind": "ClientLevelDP",
        "clip_norm": 1.0,
        "sampling_rate": 1.0,
        "max_colluding": 1,
        "max_dropouts": 0,
        "dropouts": {},
    }
    again = _simulation(clients).run(seed=1)
    assert np.array_equal(flatten_state(again.state), flatten_state(shared_result.state))


def test_taking_part_less_often_needs_less_noise(clients: list[ClientData]) -> None:
    result = _simulation(clients, plan=_shared(rate=0.5)).run(seed=1)
    full = calibrate_noise_multiplier(8.0, DELTA, 1.0, ROUNDS).noise_multiplier
    assert result.noise_multipliers["all participants"] < full
    assert result.report is not None
    assert result.report.scopes[0].events[0] == SubsampledGaussianEvent(
        result.noise_multipliers["all participants"], 0.5, ROUNDS
    )
    assert odp.verify_report(result.report.to_dict()) == []


def test_the_report_speaks_about_participants(shared_result: odp.SimulationResult) -> None:
    assert shared_result.report is not None
    text = shared_result.report.summary()
    assert "Privacy unit: one participant" in text
    assert "guessing whether a participant contributed" in text
    assert "The budget protects a whole participant" in text
    assert "Every participant adds a slice every round" in text
    assert "takes part in a round with probability 1.0" in text
    assert "whether it takes part or not" in text
    assert "Expect little accuracy below about 1000 participants" in text
    assert "public keys are not checked" in text
    assert "not yet by a machine-checked proof" in text
    assert chr(0x2014) not in text


def test_a_failed_round_is_charged_and_does_not_move_the_model(
    clients: list[ClientData],
) -> None:
    absent = {attempt: ["site-1"] for attempt in range(ROUNDS)}
    result = _simulation(clients, dropouts=absent).run(seed=1)
    torch.manual_seed(derive_seed(1, "init"))
    assert np.array_equal(flatten_state(result.state), flatten_state(get_state(_model())))
    report = result.report
    assert report is not None
    assert report.mode is not None
    assert (report.mode.attempts, report.mode.completed_rounds) == (ROUNDS, 0)
    assert report.scopes[0].events[0].repetitions == ROUNDS
    assert odp.verify_report(report.to_dict()) == []
    assert result.to_record()["config"]["plan"]["dp"]["dropouts"]["0"] == ["site-1"]


# --------------------------------------------------------------------------- the noise


class _Silent(nn.Module):
    """Loss is linear in the weights; all-zero rows leave the model where it is."""

    probability_output = False

    def __init__(self, width: int) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w


def _rows(name: str, width: int, value: float, seed: int) -> ClientData:
    """Forty rows whose first column is ``value``; most outcomes are 1, so they pull one way."""
    features = np.zeros((40, width), dtype=np.float32)
    features[:, 0] = value
    labels = np.ones(40, dtype=np.float32)
    labels[::10] = 0.0
    return make_client_data(name, features, labels, seed=seed)


WIDTH = 4000


@pytest.mark.parametrize(("colluding", "dropouts"), [(1, 0), (2, 1), (0, 0)])
def test_the_opened_sum_carries_noise_for_one_whole_participant(
    colluding: int, dropouts: int
) -> None:
    count, clip = 5, 0.7
    silent = [_rows(f"site-{i}", WIDTH, 0.0, i) for i in range(count)]
    result = odp.Simulation(
        model_fn=lambda: _Silent(WIDTH),
        clients=silent,
        plan=odp.Plan(
            dp=odp.ClientLevelDP(
                sampling_rate=1.0, clip_norm=clip, max_colluding=colluding, max_dropouts=dropouts
            ),
            secure_sum=PairwiseMasking(),
        ),
        policy=odp.PrivacyPolicy(1.0, DELTA, unit="client"),
        rounds=1,
        weighted_average=False,
    ).run(seed=3)
    sigma = result.noise_multipliers["all participants"]
    one_slice = share_std(sigma, clip, count, colluding, dropouts)
    # The server added the opened sum divided by the number of participants.
    opened = np.asarray(flatten_state(result.state), dtype=np.float64) * count
    rounding = 1.0 + 1.0 / 96.0
    assert opened.var() == pytest.approx(count * one_slice**2 * rounding, rel=0.1)
    known = sum(
        gaussian_noise(
            (WIDTH,), one_slice, torch_generator(3, "noise", 0, index), dtype=torch.float64
        ).numpy()
        for index in range(colluding)
    )
    left = (opened - known).var()
    assert left == pytest.approx((count - colluding) * one_slice**2 * rounding, rel=0.1)
    assert left >= 0.95 * (sigma * clip) ** 2


def test_the_trusted_server_adds_noise_for_one_whole_participant() -> None:
    silent = [_rows(f"site-{i}", WIDTH, 0.0, i) for i in range(3)]
    result = odp.Simulation(
        model_fn=lambda: _Silent(WIDTH),
        clients=silent,
        plan=odp.Plan(dp=odp.ClientLevelDP(sampling_rate=1.0, clip_norm=0.7)),
        policy=odp.PrivacyPolicy(1.0, DELTA, unit="client"),
        rounds=1,
        weighted_average=False,
        allow_unsafe=True,
    ).run(seed=3)
    sigma = result.noise_multipliers["all participants"]
    opened = np.asarray(flatten_state(result.state), dtype=np.float64) * 3
    assert opened.std() == pytest.approx(sigma * 0.7, rel=0.05)


def test_who_takes_part_is_not_drawn_from_the_seed_for_real_data(
    clients: list[ClientData], monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded = simulation_module.torch_generator

    def watched(seed: int, purpose: str, *path: int) -> torch.Generator:
        assert purpose not in ("noise", "sampling"), f"{purpose} was drawn from the seed"
        return seeded(seed, purpose, *path)

    monkeypatch.setattr(simulation_module, "torch_generator", watched)
    first = _simulation(clients, plan=_shared(rate=0.5), secure_noise=True).run(seed=1)
    second = _simulation(clients, plan=_shared(rate=0.5), secure_noise=True).run(seed=1)
    assert not np.array_equal(flatten_state(first.state), flatten_state(second.state))
    with pytest.raises(AssertionError, match="was drawn from the seed"):
        _simulation(clients, plan=_shared(rate=0.5)).run(seed=1)


# --------------------------------------------------------------------------- an attack


def _attack_scores(
    with_target: bool, trials: int, first_seed: int, epsilon: float = 1.0
) -> tuple[list[float], float]:
    """The attack itself is public: ``obscuradp.audit.mechanism.whole_participant_scores``."""
    return whole_participant_scores(with_target, trials, first_seed=first_seed, epsilon=epsilon)


def test_a_participant_that_pulls_hard_arrives_shortened_to_the_clipping_norm() -> None:
    """Strong rows and a large step size: unshortened, the change would be about 6."""
    without, spent = _attack_scores(False, 100, first_seed=1_000, epsilon=8.0)
    with_target, _ = _attack_scores(True, 100, first_seed=2_000, epsilon=8.0)
    assert 7.9 < spent <= 8.0
    assert np.mean(with_target) - np.mean(without) == pytest.approx(1.0, abs=0.35)


@pytest.mark.slow
def test_the_attack_on_a_whole_participant_proves_no_more_than_the_ledger() -> None:
    without, spent = _attack_scores(False, 1500, first_seed=10_000)
    with_target, _ = _attack_scores(True, 1500, first_seed=20_000)
    assert spent == pytest.approx(1.0, abs=0.01)
    assert np.mean(with_target) - np.mean(without) == pytest.approx(1.0, abs=0.35)
    proven = gaussian_epsilon_lower_bound(without, with_target, DELTA, confidence=0.999)
    assert proven <= spent
    estimate = gaussian_epsilon_lower_bound(without, with_target, DELTA, confidence=1e-9)
    assert estimate == pytest.approx(spent, rel=0.4)
