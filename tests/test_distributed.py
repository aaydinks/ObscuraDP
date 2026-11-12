"""The one-step modes: shared noise (``DistributedDPSGD``) and a trusted server (``CentralDPSGD``).

What must hold: a single message is never exposed (no secure sum, no run), one noise level
covers the whole run and every participant, the opened sum carries the stated noise even
after the assumed colluders remove their slices, a failed round is paid for, and the local
mode behaves exactly as before.
"""

import dataclasses
import math
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import obscuradp as odp
from obscuradp.accounting import (
    SkellamEvent,
    SubsampledGaussianEvent,
    calibrate_noise_multiplier,
    calibrate_skellam,
)
from obscuradp.audit import gaussian_epsilon_lower_bound
from obscuradp.audit.mechanism import shared_sum_scores, whole_number_sum_scores
from obscuradp.crypto import PairwiseMasking, pairwise_masking, setup_pairwise_masking
from obscuradp.imbalance import ClassWeighted
from obscuradp.ledger import PrivacyLedger
from obscuradp.mechanisms import (
    gaussian_noise,
    plan_integer_step,
    share_std,
    skellam_noise,
)
from obscuradp.models import TabularMLP
from obscuradp.results import validate_result
from obscuradp.seeding import numpy_generator, torch_generator
from obscuradp.sim import simulation as simulation_module
from obscuradp.state import flatten_state
from obscuradp.tables import ClientData, make_client_data
from obscuradp.unsafe import UnsafeConfiguration

DELTA = 1e-5
ROUNDS = 12
SIZES = (900, 600, 400)
TRAIN_ROWS = (630, 420, 280)
BATCH = 190


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


def _shared(colluding: int = 1, dropouts: int = 0, clip_norm: float = 1.0) -> odp.Plan:
    return odp.Plan(
        dp=odp.DistributedDPSGD(
            max_colluding=colluding, max_dropouts=dropouts, clip_norm=clip_norm
        ),
        imbalance=ClassWeighted(0.2),
        secure_sum=PairwiseMasking(),
    )


def _central() -> odp.Plan:
    return odp.Plan(dp=odp.CentralDPSGD(clip_norm=1.0), imbalance=ClassWeighted(0.2))


def _simulation(clients: list[ClientData], **overrides: Any) -> odp.Simulation:
    arguments: dict[str, Any] = {
        "model_fn": _model,
        "clients": clients,
        "plan": _shared(),
        "policy": odp.PrivacyPolicy(1.0, DELTA),
        "rounds": ROUNDS,
        "batch_size": BATCH,
        "learning_rate": 0.5,
        **overrides,
    }
    return odp.Simulation(**arguments)


@pytest.fixture(scope="module")
def shared_result(clients: list[ClientData]) -> odp.SimulationResult:
    return _simulation(clients, choose_threshold=True).run(seed=1)


# --------------------------------------------------------------------------- refusals


def test_shared_noise_without_a_secure_sum_is_refused_and_cannot_be_forced(
    clients: list[ClientData],
) -> None:
    dp = odp.DistributedDPSGD(max_colluding=1)
    with pytest.raises(ValueError, match="needs a secure sum") as refusal:
        odp.Plan(dp=dp)
    # Not one of the settings that allow_unsafe can switch on.
    assert not isinstance(refusal.value, UnsafeConfiguration)
    assert "No setting overrides this" in str(refusal.value)
    # A frozen plan cannot be edited into that state either.
    with pytest.raises(dataclasses.FrozenInstanceError):
        _shared().secure_sum = None  # type: ignore[misc]


def test_the_number_of_colluders_must_be_stated() -> None:
    with pytest.raises(TypeError, match="max_colluding"):
        odp.DistributedDPSGD()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        odp.DistributedDPSGD(1)  # type: ignore[misc]
    for arguments, message in (
        ({"max_colluding": -1}, "max_colluding must be a non-negative whole number"),
        ({"max_colluding": 1.0}, "max_colluding must be a non-negative whole number"),
        ({"max_colluding": True}, "max_colluding must be a non-negative whole number"),
        ({"max_colluding": 1, "max_dropouts": -1}, "max_dropouts must be a non-negative"),
        ({"max_colluding": 1, "clip_norm": 0.0}, "clip_norm must be positive"),
    ):
        with pytest.raises(ValueError, match=message):
            odp.DistributedDPSGD(**arguments)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="clip_norm"):
        odp.CentralDPSGD(clip_norm=-1.0)


def test_plans_that_make_no_sense_are_refused() -> None:
    dp = odp.DistributedDPSGD(max_colluding=1)
    with pytest.raises(ValueError, match="plans the range and rounding of its messages itself"):
        odp.Plan(dp=dp, secure_sum=PairwiseMasking(value_bits=8))
    with pytest.raises(ValueError, match="plans the range and rounding of its messages itself"):
        odp.Plan(dp=dp, secure_sum=PairwiseMasking(value_range=5.0))
    with pytest.raises(ValueError, match="proximal_mu has no effect"):
        odp.Plan(dp=dp, secure_sum=PairwiseMasking(), proximal_mu=0.1)
    with pytest.raises(ValueError, match="takes no secure sum"):
        odp.Plan(dp=odp.CentralDPSGD(), secure_sum=PairwiseMasking())
    with pytest.raises(ValueError, match="proximal_mu has no effect"):
        odp.Plan(dp=odp.CentralDPSGD(), proximal_mu=0.1)
    assert _shared().single_step
    assert _central().single_step
    assert not odp.Plan().single_step
    assert not odp.Plan(dp=None).single_step


def test_partial_encryption_cannot_be_combined_with_a_plan() -> None:
    """Shared noise with partly encrypted updates must be refused.

    Today no plan can ask for partial encryption at all, so the rule holds by construction.
    If a field is ever added to ``Plan``, this test fails and the refusal has to be written.
    """
    assert {f.name for f in dataclasses.fields(odp.Plan)} == {
        "dp",
        "imbalance",
        "secure_sum",
        "proximal_mu",
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"plan": _shared(colluding=3)}, "nobody is left to supply the noise"),
        ({"plan": _shared(colluding=1, dropouts=2)}, "nobody is left to supply the noise"),
        ({"local_epochs": 3.0}, "one step per round"),
        ({"weighted_average": False}, "weighted_average=False is not available"),
        ({"dropouts": {ROUNDS: ["site-0"]}}, "outside 0 to"),
        ({"dropouts": {0: ["nobody"]}}, "unknown participants"),
        ({"dropouts": {0: "site-0"}}, "collection of names"),
        ({"dropouts": {"0": ["site-0"]}}, "round numbers"),
        ({"plan": _central(), "allow_unsafe": True, "local_epochs": 2.0}, "one step per round"),
        (
            {"plan": _central(), "allow_unsafe": True, "dropouts": {0: ["site-0"]}},
            "only be played where the participants share the noise",
        ),
        (
            {"plan": odp.Plan(imbalance=ClassWeighted(0.2)), "dropouts": {0: ["site-0"]}},
            "only be played where the participants share the noise",
        ),
    ],
)
def test_settings_that_do_not_fit_the_one_step_modes_are_refused(
    clients: list[ClientData], overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _simulation(clients, **overrides)


def test_a_trusted_server_must_be_asked_for(clients: list[ClientData]) -> None:
    with pytest.raises(UnsafeConfiguration, match="not protected from the server"):
        _simulation(clients, plan=_central())
    result = _simulation(clients, plan=_central(), allow_unsafe=True).run(seed=1)
    assert result.report is not None
    assert not result.report.is_safe
    assert result.report.within_budget
    summary = result.report.summary()
    assert "UNSAFE" in summary
    assert "baseline for comparisons" in summary
    assert "by the server, which sees every participant's clean sums" in summary
    assert "Records are not protected from the server." in summary
    assert odp.verify_report(result.report.to_dict()) == []
    assert result.report.mode is not None
    assert result.report.mode.to_dict() == {
        "mode": "central",
        "participants": 3,
        "attempts": ROUNDS,
        "completed_rounds": ROUNDS,
    }
    assert result.communication["ratio"] == 1.0


# --------------------------------------------------------------------------- one budget


def test_one_noise_level_covers_the_whole_run_and_everyone(
    clients: list[ClientData], shared_result: odp.SimulationResult
) -> None:
    report = shared_result.report
    assert report is not None
    rate = BATCH / sum(TRAIN_ROWS)
    expected = calibrate_noise_multiplier(1.0, DELTA, rate, ROUNDS)
    assert set(shared_result.noise_multipliers.values()) == {expected.noise_multiplier}
    simulation = _simulation(clients)
    assert simulation.calibrations()["site-2"] == expected
    # One noisy release per round: the count is the rounds, not rounds times local steps.
    for scope in report.scopes:
        assert scope.events == (SubsampledGaussianEvent(expected.noise_multiplier, rate, ROUNDS),)
        assert 0.99 < scope.epsilon <= 1.0
    assert report.is_safe
    assert odp.verify_report(report.to_dict()) == []


def test_the_strictest_participant_sets_the_noise_for_all(clients: list[ClientData]) -> None:
    policy = odp.PrivacyPolicy(2.0, DELTA, per_scope_epsilon={"site-1": 0.5})
    result = _simulation(clients, policy=policy, rounds=4).run(seed=1)
    strict = calibrate_noise_multiplier(0.5, DELTA, BATCH / sum(TRAIN_ROWS), 4)
    assert set(result.noise_multipliers.values()) == {strict.noise_multiplier}
    assert result.report is not None
    spent = {scope.scope: scope.epsilon for scope in result.report.scopes}
    assert 0.495 < spent["site-1"] <= 0.5
    assert spent["site-0"] == spent["site-1"]  # the others spend less than their 2.0
    assert result.report.is_safe


def test_a_budget_that_cannot_be_met_stops_before_training(clients: list[ClientData]) -> None:
    with pytest.raises(ValueError, match="no noise level"):
        _simulation(clients, policy=odp.PrivacyPolicy(1e-7, DELTA)).run(seed=1)


def test_the_run_learns_and_reports_like_any_other(
    clients: list[ClientData], shared_result: odp.SimulationResult
) -> None:
    assert shared_result.test["auc"] > 0.85
    assert shared_result.history[-1]["round"] == ROUNDS
    assert shared_result.history[-1]["auc"] == shared_result.validation["auc"]
    assert shared_result.test_at_chosen_threshold is not None
    communication = shared_result.communication
    assert communication["ratio"] < 0.5
    assert communication["upload_bytes_per_round"] == math.ceil(
        129 * communication["bits_per_value"] / 8
    )
    assert communication["plain_bytes_per_round"] == 4 * 129
    assert communication["setup_bytes"] == 32 * 3
    assert communication["saturated_share"] == 0.0
    record = shared_result.to_record()
    validate_result(record)
    assert record["config"]["plan"]["dp"] == {
        "kind": "DistributedDPSGD",
        "clip_norm": 1.0,
        "max_colluding": 1,
        "max_dropouts": 0,
        "noise": "gaussian",
        "sum_bound": None,
        "dropouts": {},
    }
    assert record["data"]["privacy_report"]["mode"]["mode"] == "distributed"
    # Repeatable from the seed although the keys of the secure sum are fresh every time.
    again = _simulation(clients, choose_threshold=True).run(seed=1)
    assert np.array_equal(flatten_state(again.state), flatten_state(shared_result.state))


def test_the_server_keeps_one_optimizer_for_the_whole_run(clients: list[ClientData]) -> None:
    """With Adam the server's running averages carry over from round to round."""
    result = _simulation(clients, optimizer="adam", learning_rate=0.05).run(seed=1)
    assert result.test["auc"] > 0.8
    assert result.report is not None
    assert result.report.is_safe


def test_the_trusted_server_baseline_reaches_a_similar_model(
    clients: list[ClientData], shared_result: odp.SimulationResult
) -> None:
    central = _simulation(clients, plan=_central(), allow_unsafe=True).run(seed=1)
    assert central.noise_multipliers == shared_result.noise_multipliers
    assert central.test["auc"] == pytest.approx(shared_result.test["auc"], abs=0.05)
    assert central.to_record()["config"]["plan"]["dp"] == {
        "kind": "CentralDPSGD",
        "clip_norm": 1.0,
    }


def test_the_report_says_who_is_protected_from_whom(
    clients: list[ClientData], shared_result: odp.SimulationResult
) -> None:
    assert shared_result.report is not None
    text = shared_result.report.summary()
    assert "How the noise was added: shared. Each of the 3 participants adds a slice" in text
    assert "at most 1 participant(s) pool what they know with the server" in text
    assert "One message alone carries far too little noise" in text
    assert "The same budget holds against any one participant" in text
    assert "not yet by a machine-checked proof" in text
    assert "public keys are not checked" in text
    assert "the stated budget would then not hold" in text
    assert chr(0x2014) not in text
    # Nobody assumed to collude: a participant on its own still knows its own slice.
    alone = _simulation(clients, plan=_shared(colluding=0), rounds=4).run(seed=1)
    assert alone.report is not None
    assert "the table row for 1 applies, not the stated budget" in alone.report.summary()
    several = _simulation(
        [*clients, _client("site-3", 300, 3)], plan=_shared(colluding=2), rounds=4
    ).run(seed=1)
    assert several.report is not None
    assert "any group of up to 2 participants" in several.report.summary()


# --------------------------------------------------------------------------- the noise


class _Silent(nn.Module):
    """A model whose gradient is zero on all-zero rows, so a message is pure noise."""

    probability_output = False

    def __init__(self, width: int) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w


WIDTH = 4000


def _silent_clients(count: int) -> list[ClientData]:
    labels = np.tile(np.array([0.0, 1.0], dtype=np.float32), 30)
    return [
        make_client_data(f"site-{i}", np.zeros((60, WIDTH), dtype=np.float32), labels, seed=i)
        for i in range(count)
    ]


def _opened_sum(result: odp.SimulationResult, divisor: float, rate: float) -> np.ndarray:
    """Back the one opened sum out of the model: the server took one plain step from zero."""
    return -np.asarray(flatten_state(result.state), dtype=np.float64) * divisor / rate


@pytest.mark.parametrize(("colluding", "dropouts"), [(1, 0), (2, 1), (0, 0)])
def test_the_opened_sum_carries_the_stated_noise(colluding: int, dropouts: int) -> None:
    count, clip, rate = 5, 0.7, 0.25
    silent = _silent_clients(count)
    rows = sum(client.n_train for client in silent)
    plan = odp.Plan(
        dp=odp.DistributedDPSGD(max_colluding=colluding, max_dropouts=dropouts, clip_norm=clip),
        secure_sum=PairwiseMasking(),
    )
    result = odp.Simulation(
        model_fn=lambda: _Silent(WIDTH),
        clients=silent,
        plan=plan,
        policy=odp.PrivacyPolicy(1.0, DELTA),
        rounds=1,
        batch_size=rows,
        learning_rate=rate,
    ).run(seed=3)
    sigma = result.noise_multipliers["site-0"]
    one_slice = share_std(sigma, clip, count, colluding, dropouts)
    opened = _opened_sum(result, divisor=float(rows), rate=rate)
    # Rounding on a grid of a quarter slice adds one part in 96 to each slice's variance.
    rounding = 1.0 + 1.0 / 96.0
    assert opened.var() == pytest.approx(count * one_slice**2 * rounding, rel=0.1)
    assert opened.var() >= 0.95 * (sigma * clip) ** 2

    # The assumed colluders know their own slices and take them out of the sum.
    known = sum(
        gaussian_noise(
            (WIDTH,), one_slice, torch_generator(3, "noise", 0, index), dtype=torch.float64
        ).numpy()
        for index in range(colluding)
    )
    left = (opened - known).var()
    assert left == pytest.approx((count - colluding) * one_slice**2 * rounding, rel=0.1)
    # What remains is at least the full noise the budget was computed for.
    assert left >= 0.95 * (sigma * clip) ** 2


def test_the_trusted_server_adds_exactly_the_full_noise() -> None:
    silent = _silent_clients(3)
    rows = sum(client.n_train for client in silent)
    result = odp.Simulation(
        model_fn=lambda: _Silent(WIDTH),
        clients=silent,
        plan=odp.Plan(dp=odp.CentralDPSGD(clip_norm=0.7)),
        policy=odp.PrivacyPolicy(1.0, DELTA),
        rounds=1,
        batch_size=rows,
        learning_rate=0.25,
        allow_unsafe=True,
    ).run(seed=3)
    sigma = result.noise_multipliers["site-0"]
    opened = _opened_sum(result, divisor=float(rows), rate=0.25)
    assert opened.std() == pytest.approx(sigma * 0.7, rel=0.05)


def test_secure_noise_takes_nothing_secret_from_the_seed(
    clients: list[ClientData], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The noise and the choice of rows must both come from the operating system."""
    seeded = simulation_module.torch_generator

    def watched(seed: int, purpose: str, *path: int) -> torch.Generator:
        assert purpose not in ("noise", "sampling"), f"{purpose} was drawn from the seed"
        return seeded(seed, purpose, *path)

    monkeypatch.setattr(simulation_module, "torch_generator", watched)
    for plan, batch in ((_shared(), BATCH), (odp.Plan(imbalance=ClassWeighted(0.2)), 32)):
        settings: dict[str, Any] = {"plan": plan, "rounds": 4, "batch_size": batch}
        first = _simulation(clients, secure_noise=True, **settings).run(seed=1)
        second = _simulation(clients, secure_noise=True, **settings).run(seed=1)
        assert not np.array_equal(flatten_state(first.state), flatten_state(second.state))
        assert first.report is not None
        assert "drawn from the run's seed" not in first.report.summary()
    with pytest.raises(AssertionError, match="was drawn from the seed"):
        _simulation(clients, rounds=4).run(seed=1)


# --------------------------------------------------------------------------- the local mode


def test_the_local_mode_is_unchanged(clients: list[ClientData]) -> None:
    """Numbers recorded before the one-step modes were added (same seed, same settings)."""
    result = odp.Simulation(
        model_fn=_model,
        clients=clients,
        plan=odp.Plan(dp=odp.LocalDPSGD(clip_norm=1.0), imbalance=ClassWeighted(0.2)),
        policy=odp.PrivacyPolicy(1.0, DELTA),
        rounds=4,
        local_epochs=1.0,
        batch_size=32,
        learning_rate=0.1,
    ).run(seed=1)
    assert result.noise_multipliers == pytest.approx(
        {"site-0": 2.000062, "site-1": 2.432795, "site-2": 2.882479}, abs=1e-5
    )
    assert flatten_state(result.state)[:6] == pytest.approx(
        [-0.416337, -0.401345, -0.206163, -0.221003, -0.175966, -0.10466], abs=1e-3
    )
    assert result.test["auc"] == pytest.approx(0.935732, abs=5e-3)
    assert result.report is not None
    assert "mode" not in result.report.to_dict()
    assert result.to_record()["config"]["plan"]["dp"] == {"clip_norm": 1.0}


# --------------------------------------------------------------------------- dropouts


def test_a_failed_round_is_charged_to_those_who_sent_and_never_moves_the_model(
    clients: list[ClientData],
) -> None:
    absent = {attempt: ["site-1"] for attempt in range(4)}
    result = _simulation(clients, rounds=4, dropouts=absent).run(seed=1)
    torch.manual_seed(odp.seeding.derive_seed(1, "init"))
    untouched = flatten_state(odp.state.get_state(_model()))
    assert np.array_equal(flatten_state(result.state), untouched)
    report = result.report
    assert report is not None
    assert {scope.scope: scope.events[0].repetitions for scope in report.scopes} == {
        "site-0": 4,
        "site-2": 4,
    }
    assert report.mode is not None
    assert (report.mode.attempts, report.mode.completed_rounds) == (4, 0)
    assert result.history == ({"round": 0, **result.validation},)
    assert odp.verify_report(report.to_dict()) == []


def test_failed_rounds_use_up_the_budget_and_never_exceed_it(clients: list[ClientData]) -> None:
    result = _simulation(clients, rounds=4, dropouts={1: ["site-0"], 3: []}).run(seed=1)
    report = result.report
    assert report is not None
    assert report.mode is not None
    assert (report.mode.attempts, report.mode.completed_rounds) == (4, 3)
    assert report.within_budget
    releases = {scope.scope: scope.events[0].repetitions for scope in report.scopes}
    assert releases == {"site-0": 3, "site-1": 4, "site-2": 4}
    assert max(scope.epsilon for scope in report.scopes) <= 1.0
    assert "1 further round(s) failed because a participant was missing" in report.summary()
    assert result.history[-1]["round"] == 3
    assert result.to_record()["config"]["plan"]["dp"]["dropouts"] == {"1": ["site-0"]}
    assert odp.verify_report(report.to_dict()) == []


def test_every_attempt_has_its_own_masks(
    clients: list[ClientData], monkeypatch: pytest.MonkeyPatch
) -> None:
    used: list[tuple[str, int]] = []
    protect = pairwise_masking.MaskingClient.protect

    def recording(self: pairwise_masking.MaskingClient, sum_index: int, values: Any) -> bytes:
        used.append((self.name, sum_index))
        return protect(self, sum_index, values)

    monkeypatch.setattr(pairwise_masking.MaskingClient, "protect", recording)
    _simulation(clients, rounds=4, dropouts={0: ["site-2"], 2: ["site-0"]}).run(seed=1)
    assert used == [
        ("site-0", 0),
        ("site-1", 0),
        ("site-0", 1),
        ("site-1", 1),
        ("site-2", 1),
        ("site-1", 2),
        ("site-2", 2),
        ("site-0", 3),
        ("site-1", 3),
        ("site-2", 3),
    ]


def _honest_messages(
    missing: str,
) -> tuple[dict[str, pairwise_masking.MaskingClient], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Three participants prepare a sum and ``missing`` never sends its message."""
    names = ["alice", "bob", "carol"]
    bits = 16
    clients, _ = setup_pairwise_masking(names, bits)
    generator = np.random.default_rng(0)
    values = {name: generator.integers(0, 2**10, size=2000).astype(np.uint64) for name in names}
    sent = {
        name: pairwise_masking.unpack_bits(clients[name].protect(7, values[name]), bits, 2000)
        for name in names
        if name != missing
    }
    return clients, values, sent


def _strip(message: np.ndarray, owner: str, insider: pairwise_masking.MaskingClient) -> np.ndarray:
    """Remove from ``owner``'s message the mask it shares with ``insider``."""
    mask = pairwise_masking._mask(insider._pair_keys[owner], 7, message.size, 16)
    stripped = message - mask if owner < insider.name else message + mask
    return np.asarray(stripped & np.uint64(2**16 - 1), dtype=np.uint64)


def test_a_missing_colluder_lets_the_group_read_the_others_sum() -> None:
    """Why a failed round is charged: the sum did not open, yet it was not secret."""
    clients, values, sent = _honest_messages(missing="carol")
    # Carol works with the server. She kept her message back but knows her pair secrets.
    total = _strip(sent["alice"], "alice", clients["carol"]) + _strip(
        sent["bob"], "bob", clients["carol"]
    )
    total &= np.uint64(2**16 - 1)
    assert np.array_equal(total, values["alice"] + values["bob"])


def test_a_missing_honest_participant_keeps_everything_hidden() -> None:
    clients, values, sent = _honest_messages(missing="carol")
    # Bob works with the server; Carol is honest and simply offline.
    exposed = _strip(sent["alice"], "alice", clients["bob"])
    assert not np.array_equal(exposed, values["alice"])
    # What is left is still covered by the mask Alice shares with Carol: it looks uniform.
    counts = np.bincount((exposed >> np.uint64(12)).astype(np.int64), minlength=16)
    expected = exposed.size / 16
    assert float(((counts - expected) ** 2 / expected).sum()) < 60.0


# --------------------------------------------------------------------------- an attack


def _attack_scores(with_target: bool, trials: int, sigma: float, seed: int) -> list[float]:
    """The attack itself is public: ``obscuradp.audit.mechanism.shared_sum_scores``."""
    return shared_sum_scores(with_target, trials, noise_multiplier=sigma, seed=seed)[0]


ATTACK_SIGMA = 3.73  # one full-batch release at this noise level costs a budget of 1


def test_the_strongest_attack_with_the_assumed_colluder_proves_no_more_than_the_ledger() -> None:
    spent = PrivacyLedger(odp.PrivacyPolicy(50.0, DELTA)).epsilon_of(
        [SubsampledGaussianEvent(ATTACK_SIGMA, 1.0, 1)]
    )
    assert spent == pytest.approx(1.0, abs=0.01)
    without = _attack_scores(False, 2000, ATTACK_SIGMA, seed=10)
    with_target = _attack_scores(True, 2000, ATTACK_SIGMA, seed=20)
    proven = gaussian_epsilon_lower_bound(without, with_target, DELTA, confidence=0.999)
    assert proven <= spent
    # The attack is sharp, so the accounting is not loose either.
    estimate = gaussian_epsilon_lower_bound(without, with_target, DELTA, confidence=1e-9)
    assert estimate == pytest.approx(spent, rel=0.3)


# --------------------------------------------------------------------------- whole-number noise

FULL = sum(TRAIN_ROWS)


def _whole(colluding: int = 1, dropouts: int = 0, **settings: Any) -> odp.Plan:
    return odp.Plan(
        dp=odp.DistributedDPSGD(
            max_colluding=colluding, max_dropouts=dropouts, noise="skellam", **settings
        ),
        imbalance=ClassWeighted(0.2),
        secure_sum=PairwiseMasking(),
    )


@pytest.fixture(scope="module")
def whole_result(clients: list[ClientData]) -> odp.SimulationResult:
    return _simulation(clients, plan=_whole(), batch_size=FULL, choose_threshold=True).run(seed=1)


def test_whole_number_noise_needs_full_rounds_and_a_known_name(
    clients: list[ClientData],
) -> None:
    with pytest.raises(ValueError, match="every row takes part"):
        _simulation(clients, plan=_whole(), batch_size=FULL - 1)
    with pytest.raises(ValueError, match="noise must be one of"):
        odp.DistributedDPSGD(max_colluding=1, noise="laplace")
    with pytest.raises(ValueError, match="sum_bound must be positive"):
        odp.DistributedDPSGD(max_colluding=1, sum_bound=0.0)
    # Still refused without a secure sum, like the Gaussian path.
    with pytest.raises(ValueError, match="needs a secure sum"):
        odp.Plan(dp=odp.DistributedDPSGD(max_colluding=1, noise="skellam"))


def test_a_run_with_whole_number_noise_fits_its_budget(
    clients: list[ClientData], whole_result: odp.SimulationResult
) -> None:
    report = whole_result.report
    assert report is not None
    step = plan_integer_step(129)
    expected = calibrate_skellam(1.0, DELTA, ROUNDS, step.l2_sensitivity, step.l1_sensitivity)
    assert _simulation(clients, plan=_whole(), batch_size=FULL).calibrations()["site-1"] == (
        expected
    )
    assert set(whole_result.noise_multipliers.values()) == {expected.noise_multiplier}
    for scope in report.scopes:
        assert scope.events == (
            SkellamEvent(expected.variance, step.l2_sensitivity, step.l1_sensitivity, ROUNDS),
        )
        assert 0.995 < scope.epsilon <= 1.0
        assert scope.check.single_method
    assert report.is_safe
    stored = report.to_dict()
    assert odp.verify_report(stored) == []
    assert stored["scopes"][0]["events"][0]["mechanism"] == "skellam"
    assert stored["mode"]["noise"] == "skellam"
    assert stored["mode"]["steps_per_share"] is None
    # Editing the stored noise is noticed like any other edit.
    stored["scopes"][0]["events"][0]["variance"] *= 2.0
    assert odp.verify_report(stored) != []


def test_the_report_of_whole_number_noise_drops_the_rounding_caveat(
    whole_result: odp.SimulationResult, shared_result: odp.SimulationResult
) -> None:
    assert whole_result.report is not None
    assert shared_result.report is not None
    text = whole_result.report.summary()
    assert "The noise is made of whole numbers." in text
    assert "rests on no argument about rounding" in text
    assert "not yet by a machine-checked proof" not in text
    assert "the one method proven for whole-number noise" in text
    assert "numpy's Poisson routine" in text
    assert "Budget by the real number of participants who pool with the server:" in text
    assert chr(0x2014) not in text
    # The Gaussian path keeps its caveat and is not told about Poisson draws.
    other = shared_result.report.summary()
    assert "not yet by a machine-checked proof" in other
    assert "numpy's Poisson routine" not in other
    table = whole_result.report.mode.table  # type: ignore[union-attr]
    assert table[1].epsilon == pytest.approx(whole_result.report.epsilon)
    assert [row.epsilon for row in table] == sorted(row.epsilon for row in table)


def test_a_run_with_whole_number_noise_learns_and_is_repeatable(
    clients: list[ClientData], whole_result: odp.SimulationResult
) -> None:
    assert whole_result.test["auc"] > 0.85
    record = whole_result.to_record()
    validate_result(record)
    assert record["config"]["plan"]["dp"]["noise"] == "skellam"
    communication = whole_result.communication
    assert communication["saturated_share"] == 0.0
    assert communication["upload_bytes_per_round"] == math.ceil(
        129 * communication["bits_per_value"] / 8
    )
    again = _simulation(clients, plan=_whole(), batch_size=FULL, choose_threshold=True).run(seed=1)
    assert np.array_equal(flatten_state(again.state), flatten_state(whole_result.state))
    # A declared limit on the sums gives smaller messages; here it cuts nothing off.
    bounded = _simulation(clients, plan=_whole(sum_bound=40.0), batch_size=FULL).run(seed=1)
    assert bounded.communication["bits_per_value"] < communication["bits_per_value"]
    assert bounded.test["auc"] == pytest.approx(whole_result.test["auc"], abs=0.02)
    assert bounded.to_record()["config"]["plan"]["dp"]["sum_bound"] == 40.0
    # A limit far too small cuts the sums off, which the result reports. The budget holds
    # all the same, because the limit is applied before the noise.
    tiny = _simulation(clients, plan=_whole(sum_bound=0.5), batch_size=FULL).run(seed=1)
    assert tiny.communication["saturated_share"] > 0.3
    assert tiny.report is not None
    assert tiny.report.is_safe
    cut = _simulation(
        clients,
        plan=odp.Plan(
            dp=odp.DistributedDPSGD(max_colluding=1, sum_bound=0.05),
            imbalance=ClassWeighted(0.2),
            secure_sum=PairwiseMasking(),
        ),
    ).run(seed=1)
    assert cut.communication["saturated_share"] > 0.3


def test_a_failed_round_with_whole_number_noise_is_charged(clients: list[ClientData]) -> None:
    result = _simulation(clients, plan=_whole(), batch_size=FULL, dropouts={2: ["site-0"]}).run(
        seed=1
    )
    report = result.report
    assert report is not None
    assert report.mode is not None
    assert (report.mode.attempts, report.mode.completed_rounds) == (ROUNDS, ROUNDS - 1)
    releases = {scope.scope: scope.events[0].repetitions for scope in report.scopes}
    assert releases == {"site-0": ROUNDS - 1, "site-1": ROUNDS, "site-2": ROUNDS}
    assert report.within_budget
    assert odp.verify_report(report.to_dict()) == []


@pytest.mark.parametrize(("colluding", "dropouts"), [(1, 0), (2, 1)])
def test_the_opened_sum_carries_the_stated_whole_number_noise(
    colluding: int, dropouts: int
) -> None:
    count, clip, rate = 5, 0.7, 0.25
    silent = _silent_clients(count)
    rows = sum(client.n_train for client in silent)
    plan = odp.Plan(
        dp=odp.DistributedDPSGD(
            max_colluding=colluding, max_dropouts=dropouts, clip_norm=clip, noise="skellam"
        ),
        secure_sum=PairwiseMasking(),
    )
    result = odp.Simulation(
        model_fn=lambda: _Silent(WIDTH),
        clients=silent,
        plan=plan,
        policy=odp.PrivacyPolicy(1.0, DELTA),
        rounds=1,
        batch_size=rows,
        learning_rate=rate,
    ).run(seed=3)
    assert result.report is not None
    event = result.report.scopes[0].events[0]
    assert isinstance(event, SkellamEvent)
    step = plan_integer_step(WIDTH)
    unit = clip / step.scale
    slice_variance = event.variance / (count - colluding - dropouts)
    # In grid points the opened sum is whole numbers: nothing was rounded after the noise.
    opened = _opened_sum(result, divisor=float(rows), rate=rate) / unit
    assert np.abs(opened - np.round(opened)).max() < 1e-3
    assert opened.var() == pytest.approx(count * slice_variance, rel=0.1)
    known = sum(
        skellam_noise((WIDTH,), slice_variance, numpy_generator(3, "noise", 0, index))
        for index in range(colluding)
    )
    left = (np.round(opened) - known).var()
    assert left == pytest.approx((count - colluding) * slice_variance, rel=0.1)
    # What the honest participants supply is at least the noise the ledger was charged with.
    assert left >= 0.95 * event.variance


def _whole_attack_scores(with_target: bool, trials: int, seed: int) -> tuple[list[float], float]:
    """The attack itself is public: ``obscuradp.audit.mechanism.whole_number_sum_scores``."""
    return whole_number_sum_scores(with_target, trials, seed=seed)


def test_the_attack_on_whole_number_noise_proves_no_more_than_the_ledger() -> None:
    without, spent = _whole_attack_scores(False, 2000, seed=10)
    with_target, _ = _whole_attack_scores(True, 2000, seed=20)
    assert spent == pytest.approx(1.0, abs=0.01)
    proven = gaussian_epsilon_lower_bound(without, with_target, DELTA, confidence=0.999)
    assert proven <= spent
    # The attack gets most of the way: the count is not loose by much.
    estimate = gaussian_epsilon_lower_bound(without, with_target, DELTA, confidence=1e-9)
    assert estimate == pytest.approx(spent, rel=0.35)
