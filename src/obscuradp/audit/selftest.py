"""Does the auditor itself work? It must catch a broken pipeline and pass a correct one.

An audit that never raises a flag proves nothing, and one that flags correct code is
useless. ``self_test`` therefore runs both auditors of this package on pipelines whose
answer is known:

- the attack on a whole training run (``obscuradp.audit.membership``) against a correct
  private run, which must **not** be flagged, and against training that states a budget of
  1 but adds no noise at all, which **must** be flagged;
- the strongest-attacker check (``obscuradp.audit.mechanism``) against the library's
  private training step, which must not be flagged, and against the same step with half
  the noise its ledger is told about, which must be flagged.

Everything runs on a small made-up table that is built here. No real row is involved.
From the command line: ``obscuradp audit self-test``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from obscuradp.accounting.events import SubsampledGaussianEvent
from obscuradp.audit.mechanism import audit_local_step
from obscuradp.audit.membership import ATTACK_UPDATES, audit_run, summarize, summarize_run
from obscuradp.client.trainer import train_private_round
from obscuradp.ledger import PrivacyLedger
from obscuradp.models import TabularMLP
from obscuradp.plan import LocalDPSGD, Plan
from obscuradp.policy import PrivacyPolicy
from obscuradp.sim.simulation import Simulation
from obscuradp.tables.split import ClientData

CLAIMED_EPSILON = 1.0
DELTA = 1e-5
WIDTH = 20
ROWS = 240
PARTICIPANTS = 3


@dataclass(frozen=True)
class SelfTestCheck:
    """One pipeline put before an auditor.

    Attributes:
        name: What was audited.
        must_flag: Whether a working auditor has to flag it.
        flagged: Whether the auditor flagged it.
        detail: The auditor's finding in plain words.
    """

    name: str
    must_flag: bool
    flagged: bool
    detail: str

    @property
    def passed(self) -> bool:
        """True if the auditor did what a working auditor has to do."""
        return self.flagged == self.must_flag


def made_up_participants(seed: int = 0) -> list[ClientData]:
    """A small random table with a made-up rule between columns and outcome."""
    generator = np.random.default_rng(seed)
    rule = generator.normal(size=WIDTH)
    clients = []
    empty_x = np.zeros((0, WIDTH), dtype=np.float32)
    empty_y = np.zeros(0, dtype=np.float32)
    for index in range(PARTICIPANTS):
        features = generator.normal(size=(ROWS, WIDTH)).astype(np.float32)
        outcome = features @ rule + generator.logistic(size=ROWS)
        clients.append(
            ClientData(
                name=f"site-{index}",
                train_x=features,
                train_y=(outcome > 0).astype(np.float32),
                validation_x=empty_x,
                validation_y=empty_y,
                test_x=empty_x,
                test_y=empty_y,
            )
        )
    return clients


def _model() -> TabularMLP:
    return TabularMLP(WIDTH, (16,))


def correct_pipeline(clients: Sequence[ClientData], declared: Mapping[str, int]) -> Simulation:
    """The default plan with a budget of 1: every participant adds all of its own noise."""
    return Simulation(
        model_fn=_model,
        clients=clients,
        plan=Plan(dp=LocalDPSGD(clip_norm=1.0)),
        policy=PrivacyPolicy(CLAIMED_EPSILON, DELTA),
        rounds=10,
        local_epochs=1.0,
        batch_size=16,
        learning_rate=0.05,
        evaluate_every=0,
        declared_rows=declared,
    )


def silent_pipeline(clients: Sequence[ClientData], declared: Mapping[str, int]) -> Simulation:
    """Training without any noise. The self-test pretends it states a budget of 1."""
    return Simulation(
        model_fn=_model,
        clients=clients,
        plan=Plan(dp=None),
        policy=None,
        rounds=10,
        local_epochs=1.0,
        batch_size=16,
        learning_rate=0.01,
        evaluate_every=0,
        declared_rows=declared,
    )


def half_noise_round(
    model: torch.nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    ledger: PrivacyLedger,
    scope: str,
    sampling_rate: float,
    steps: int,
    noise_multiplier: float,
    **settings: Any,
) -> Any:
    """A broken training round: it reports its noise to the ledger and adds half of it."""
    ledger.charge(scope, SubsampledGaussianEvent(noise_multiplier, sampling_rate, steps))
    unread = PrivacyLedger(PrivacyPolicy(1e9, ledger.policy.delta))
    return train_private_round(
        model,
        features,
        targets,
        ledger=unread,
        scope=scope,
        sampling_rate=sampling_rate,
        steps=steps,
        noise_multiplier=noise_multiplier / 2.0,
        **settings,
    )


def self_test(*, trials: int = 2000, seed: int = 0) -> list[SelfTestCheck]:
    """Run both auditors on a correct and on a broken pipeline each.

    Args:
        trials: Runs per world for the strongest-attacker check.
        seed: Seed of the made-up table and of the audited runs.

    Returns:
        One entry per audited pipeline. The self-test is passed if every entry is.
    """
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        clients = made_up_participants(seed)
        checks = []
        for name, build, must_flag in (
            ("whole run, correct pipeline (default plan, budget 1)", correct_pipeline, False),
            ("whole run, no noise added but a budget of 1 stated", silent_pipeline, True),
        ):
            run = summarize_run(audit_run(build, clients, seed))
            found = summarize(
                [run], ATTACK_UPDATES, 1.0, claimed_epsilon=CLAIMED_EPSILON, delta=DELTA
            )
            checks.append(SelfTestCheck(name, must_flag, found.flagged, found.describe()))
        for name, train_round, must_flag in (
            ("single step, the library's private training step", train_private_round, False),
            ("single step, half the noise its ledger is told about", half_noise_round, True),
        ):
            step = audit_local_step(trials, train_round=train_round)
            checks.append(SelfTestCheck(name, must_flag, step.flagged, step.describe()))
        return checks
    finally:
        torch.set_num_threads(threads)


def passed(checks: Sequence[SelfTestCheck]) -> bool:
    """True if the auditor behaved as it must on every pipeline."""
    return all(check.passed for check in checks)
