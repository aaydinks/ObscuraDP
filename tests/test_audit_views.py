"""Watching a run from outside, and what each attacker is allowed to see of it.

``Simulation.run(observer=...)`` hands out everything that was sent in a round. The views
of ``obscuradp.audit.views`` cut that down to one attacker. What must hold: watching
changes nothing, the record is complete and consistent, and a view never contains a single
message its attacker could not have read.
"""

from typing import Any

import numpy as np
import pytest
import torch

import obscuradp as odp
from obscuradp.audit.views import MODEL_OBSERVER, SERVER, model_observer_view, server_view
from obscuradp.crypto import PairwiseMasking
from obscuradp.models import TabularMLP
from obscuradp.sim import RECORD_CHANGES, RECORD_MODELS, RECORD_SUMS, RoundRecord
from obscuradp.state import flatten_state
from obscuradp.tables import ClientData, make_client_data

DELTA = 1e-5
ROUNDS = 3
NAMES = ("site-0", "site-1", "site-2")

# Matched by its text, not by its class: another test re-imports the package, after which a
# filter that names the class would look at a different class object than the one raised.
pytestmark = pytest.mark.filterwarnings("ignore:protecting whole participants with only")


def _client(name: str, n: int, seed: int) -> ClientData:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 6)).astype(np.float32)
    signal = x[:, 0] + 0.6 * x[:, 1] + rng.normal(scale=0.5, size=n)
    return make_client_data(name, x, (signal > 0.5).astype(np.float32), seed=seed)


@pytest.fixture(scope="module")
def clients() -> list[ClientData]:
    return [_client(name, 300, index) for index, name in enumerate(NAMES)]


def _simulation(clients: list[ClientData], plan: odp.Plan, **overrides: Any) -> odp.Simulation:
    arguments: dict[str, Any] = {
        "model_fn": lambda: TabularMLP(6, (8,)),
        "clients": clients,
        "plan": plan,
        "policy": None if plan.dp is None else odp.PrivacyPolicy(4.0, DELTA),
        "rounds": ROUNDS,
        "batch_size": 64,
        "learning_rate": 0.1,
        "evaluate_every": 0,
        **overrides,
    }
    return odp.Simulation(**arguments)


def _shared(noise: str = "gaussian") -> odp.Plan:
    return odp.Plan(
        dp=odp.DistributedDPSGD(max_colluding=1, noise=noise), secure_sum=PairwiseMasking()
    )


PLANS: dict[str, tuple[odp.Plan, dict[str, Any], str, bool]] = {
    # name: (plan, settings, kind of record, masked)
    "plain": (odp.Plan(dp=None), {}, RECORD_MODELS, False),
    "local": (odp.Plan(dp=odp.LocalDPSGD()), {}, RECORD_MODELS, False),
    "local, secure sum": (
        odp.Plan(dp=odp.LocalDPSGD(), secure_sum=PairwiseMasking()),
        {},
        RECORD_MODELS,
        True,
    ),
    "shared": (_shared(), {}, RECORD_SUMS, True),
    "shared, whole numbers": (_shared("skellam"), {"batch_size": 10**9}, RECORD_SUMS, True),
    "trusted server": (odp.Plan(dp=odp.CentralDPSGD()), {"allow_unsafe": True}, RECORD_SUMS, False),
    "whole participants": (
        odp.Plan(
            dp=odp.ClientLevelDP(sampling_rate=1.0, max_colluding=1), secure_sum=PairwiseMasking()
        ),
        {"policy": odp.PrivacyPolicy(4.0, DELTA, unit="client")},
        RECORD_CHANGES,
        True,
    ),
}


@pytest.mark.parametrize("name", list(PLANS))
def test_watching_a_run_changes_nothing_and_misses_nothing(
    clients: list[ClientData], name: str
) -> None:
    plan, settings, kind, masked = PLANS[name]
    simulation = _simulation(clients, plan, **settings)
    unwatched = simulation.run(seed=7)
    records: list[RoundRecord] = []
    watched = simulation.run(seed=7, observer=records.append)
    assert np.array_equal(flatten_state(unwatched.state), flatten_state(watched.state))
    assert [record.round for record in records] == list(range(ROUNDS))
    assert {record.kind for record in records} == {kind}
    assert {record.masked for record in records} == {masked}
    for record in records:
        assert set(record.messages) == set(NAMES)
        assert record.total is not None
        if masked:
            # What goes through the secure sum adds up to what the sum opens.
            summed = sum(record.messages[participant] for participant in NAMES)
            assert np.allclose(summed, record.total, atol=2e-3)
    # Each round starts where the last one ended, and the last one ends at the result.
    starts = [flatten_state(record.start) for record in records]
    starts.append(flatten_state(watched.state))
    assert not np.array_equal(starts[0], starts[-1])
    if kind == RECORD_MODELS:
        for record, before, after in zip(records, starts[:-1], starts[1:], strict=True):
            assert record.total is not None
            assert np.allclose(before + record.total, after, atol=1e-5)


def test_an_observer_that_draws_random_numbers_is_not_the_runs_business(
    clients: list[ClientData],
) -> None:
    """Audits restore the random stream themselves; an observer that reads only is safe."""
    plan, settings, _, _ = PLANS["local"]
    simulation = _simulation(clients, plan, **settings)
    plain = simulation.run(seed=3)

    def careful(record: RoundRecord) -> None:
        saved = torch.random.get_rng_state()
        torch.rand(5)
        torch.random.set_rng_state(saved)

    assert np.array_equal(
        flatten_state(plain.state), flatten_state(simulation.run(seed=3, observer=careful).state)
    )


def test_a_failed_round_has_a_record_without_a_total(clients: list[ClientData]) -> None:
    records: list[RoundRecord] = []
    _simulation(clients, _shared(), dropouts={1: ["site-1"]}).run(seed=1, observer=records.append)
    assert [record.total is None for record in records] == [False, True, False]
    assert set(records[1].messages) == {"site-0", "site-2"}
    assert server_view(records[1], ["site-2"]).honest_total() is None


def _records(clients: list[ClientData], name: str) -> list[RoundRecord]:
    plan, settings, _, _ = PLANS[name]
    records: list[RoundRecord] = []
    _simulation(clients, plan, **settings).run(seed=5, observer=records.append)
    return records


@pytest.mark.parametrize("name", ["local, secure sum", "shared", "shared, whole numbers"])
def test_behind_the_secure_sum_the_server_reads_no_single_message(
    clients: list[ClientData], name: str
) -> None:
    record = _records(clients, name)[0]
    alone = server_view(record)
    assert alone.attacker == SERVER
    assert dict(alone.messages) == {}
    assert alone.total is not None
    assert np.array_equal(alone.total, record.total)  # type: ignore[arg-type]
    assert np.array_equal(alone.honest_total(), record.total)  # type: ignore[arg-type]
    # A participant that works with the server adds its own message, and no other.
    helped = server_view(record, colluders=["site-2"])
    assert set(helped.messages) == {"site-2"}
    left = helped.honest_total()
    assert left is not None
    assert np.allclose(left, record.total - record.messages["site-2"])
    # The view cannot be widened afterwards.
    with pytest.raises(TypeError):
        helped.messages["site-0"] = record.messages["site-0"]  # type: ignore[index]


@pytest.mark.parametrize("name", ["plain", "local"])
def test_updates_in_the_clear_are_all_read_by_the_server(
    clients: list[ClientData], name: str
) -> None:
    record = _records(clients, name)[0]
    assert set(server_view(record).messages) == set(NAMES)


@pytest.mark.parametrize("name", list(PLANS))
def test_a_model_observer_reads_no_message_at_all(clients: list[ClientData], name: str) -> None:
    record = _records(clients, name)[0]
    view = model_observer_view(record)
    assert view.attacker == MODEL_OBSERVER
    assert dict(view.messages) == {}
    assert view.colluders == frozenset()
    assert view.round == record.round
    assert view.divisor == record.divisor
    assert np.array_equal(view.honest_total(), record.total)  # type: ignore[arg-type]


def test_the_noise_the_attacker_is_left_with_is_what_the_plan_counts_on(
    clients: list[ClientData],
) -> None:
    """Shared noise, one colluder of three: two slices remain, together the charged noise."""
    width = 3000
    silent = [
        make_client_data(
            name,
            np.zeros((60, width), dtype=np.float32),
            np.tile(np.array([0.0, 1.0], dtype=np.float32), 30),
            seed=index,
        )
        for index, name in enumerate(NAMES)
    ]

    class Silent(torch.nn.Module):
        probability_output = False

        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(width))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x @ self.w

    records: list[RoundRecord] = []
    result = odp.Simulation(
        model_fn=Silent,
        clients=silent,
        plan=_shared(),
        policy=odp.PrivacyPolicy(1.0, DELTA),
        rounds=1,
        batch_size=10**9,
        learning_rate=0.1,
        evaluate_every=0,
    ).run(seed=2, observer=records.append)
    sigma = result.noise_multipliers["site-0"]
    left = server_view(records[0], colluders=["site-2"]).honest_total()
    assert left is not None
    # All-zero rows have no gradient, so what is left is the honest participants' noise.
    assert float(np.var(left)) == pytest.approx(sigma**2, rel=0.1)
    assert float(np.var(records[0].total)) == pytest.approx(1.5 * sigma**2, rel=0.1)  # type: ignore[arg-type]
