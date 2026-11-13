"""The benchmark's attack tests (section 7 of the written plan and its first amendment).

    python -m benchmarks.phase5.audit health
    python -m benchmarks.phase5.audit writers

**Health table, by class.** The whole-run test of step S15 on the default plan, with every
training row of the rare class a test record beside equally many rows of the other class,
30 runs, read by class. Step S16 showed why: with weighted classes an average over all
rows hides the rare ones.

**Handwriting, whole writers.** One run in which whole writers are the test units: each of
400 writers is kept in or left out of the run by a fair coin. The attacker sees the sum
that is opened in every round and knows the images of the test writers, but not who was
drawn in which round. For every round it works out what each test writer would have sent
from that round's starting model and takes out the direction all test writers share.
The scores of all test writers are then worked out together: the set of scores that best
explains what was left of the opened sums (the same decoding step S15 used for rows).
What the other participants send is noise to this attacker. This is one attack, built on a
small made-up group; a weak result from it is no proof of safety.

What is stored are counts and summary figures, never a single record's or writer's score.
"""

from __future__ import annotations

import json
import multiprocessing
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import obscuradp as odp
from benchmarks.phase4.more import _rare_and_as_many_common
from benchmarks.phase5.common import DELTA, PARTICIPANT_ARM, Setting, build_simulation, load
from benchmarks.phase5.main import chosen_settings, reported
from benchmarks.phase5.run import RESULTS, write_record
from obscuradp.audit.bounds import one_run_epsilon_lower_bound
from obscuradp.audit.membership import (
    ATTACK_FINAL_MODEL,
    ATTACK_UPDATES,
    AuditRun,
    _Decoder,
    audit_run,
    best_case_lower_bound,
    count_guesses,
    measured_advantage,
)
from obscuradp.client.trainer import clipped_model_change, train_plain_round
from obscuradp.plan import ClientLevelDP
from obscuradp.results import make_result
from obscuradp.seeding import numpy_generator, torch_generator
from obscuradp.sim import RoundRecord
from obscuradp.state import set_state
from obscuradp.tables import ClientData
from obscuradp.tasks import ManyClasses

AUDIT = RESULTS / "audit"
HEALTH_SEEDS = tuple(range(1100, 1130))
HEALTH_MODELS = ("network", "logistic")  # the slower one first
HEALTH_BUDGETS = (1.0, 8.0)
HEALTH_ARM = "local"
GUESS_SHARE = 0.05  # fixed by the pilot of step S15
ATTACKS = (ATTACK_UPDATES, ATTACK_FINAL_MODEL)
GROUPS = ("all", "label_1", "label_0")
WRITER_SEED = 1100
WRITER_BUDGET = 8.0
TEST_WRITERS = 400
STAGES = ("health", "writers")


# ---------------------------------------------------------------------- the health table by class


def audited(
    setting: Setting, clients: Sequence[ClientData], eligible: Mapping[str, Any]
) -> AuditRun:
    """One audited run of ``setting`` in which every eligible row is a test record."""
    torch.set_num_threads(1)

    def build(rows: Sequence[ClientData], declared: Mapping[str, int]) -> odp.Simulation:
        simulation = build_simulation(setting, list(rows))
        # The audit needs no chosen cut-off.
        return replace(simulation, declared_rows=dict(declared), choose_threshold=False)

    return audit_run(build, clients, setting.seed, eligible=eligible, fraction=1.0, cap=10**9)


def by_group(run: AuditRun, attack: str) -> dict[str, dict[str, float | int]]:
    """Guesses and measured advantage of one attack: all test records, and each class apart."""
    scores, kept = np.asarray(run.scores[attack]), np.asarray(run.kept)
    rare = np.asarray(run.labels) > 0.5
    figures: dict[str, dict[str, float | int]] = {}
    for group, members in (("all", np.ones_like(rare)), ("label_1", rare), ("label_0", ~rare)):
        guesses, right = count_guesses(scores[members], kept[members], GUESS_SHARE)
        figures[group] = {
            "records": int(members.sum()),
            "guesses": guesses,
            "right": right,
            "advantage": measured_advantage(scores[members], kept[members]),
        }
    return figures


def health_cell(setting: Setting, clients: Sequence[ClientData]) -> dict[str, Any]:
    """What is stored of one audited run on the health table."""
    run = audited(setting, clients, _rare_and_as_many_common(list(clients), setting.seed))
    report = run.result.report
    assert report is not None
    return {
        "epsilon": report.epsilon,
        "ceiling_mechanism": report.advantage_ceiling,
        "ceiling_general": report.general_advantage_ceiling,
        "attacker": run.attacker,
        "seconds": round(run.result.seconds, 1),
        "attacks": {attack: by_group(run, attack) for attack in ATTACKS},
    }


def _health_task(task: tuple[str, int, str, dict[str, dict[str, Any]]]) -> str:
    model, seed, directory, chosen = task
    warnings.simplefilter("ignore")
    stem = f"health_{model}_seed-{seed}"
    path = Path(directory) / f"{stem}.json"
    if path.exists():
        return stem
    clients = load("health", seed)
    cells = {}
    for epsilon in HEALTH_BUDGETS:
        setting = reported("health", model, HEALTH_ARM, epsilon, seed, chosen)
        cells[f"{epsilon:g}"] = health_cell(setting, clients)
    config = {"dataset": "health", "model": model, "arm": HEALTH_ARM, "guess_share": GUESS_SHARE}
    write_record(
        path, make_result(kind="phase5_audit_health", config=config, seeds=[seed], data=cells)
    )
    return stem


def health(chosen: dict[str, dict[str, Any]], directory: Path = AUDIT, processes: int = 4) -> int:
    """The audited runs on the health table: two models, two budgets, thirty seeds."""
    directory.mkdir(parents=True, exist_ok=True)
    tasks = [
        (model, seed, str(directory), chosen) for model in HEALTH_MODELS for seed in HEALTH_SEEDS
    ]
    with multiprocessing.Pool(processes) as pool:
        for stem in pool.imap_unordered(_health_task, tasks, chunksize=1):
            print(f"finished {stem}", flush=True)
    return len(tasks)


# ---------------------------------------------------------------------- whole writers


def choose_test_units(participants: int, units: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Which participants are test units, and each one's coin (true: it stays in the run)."""
    if not 2 <= units <= participants:
        raise ValueError("need at least two test units and no more than there are participants")
    generator = numpy_generator(seed, "canary", 0)
    picked = np.sort(generator.choice(participants, size=units, replace=False))
    return picked, generator.random(units) < 0.5


class WriterAttack:
    """Scores whole participants from the sums a run opens. See the module's description."""

    def __init__(
        self,
        simulation: odp.Simulation,
        units: Sequence[ClientData],
        weights: Sequence[float],
        seed: int,
    ) -> None:
        dp = simulation.plan.dp
        if not isinstance(dp, ClientLevelDP) or not isinstance(simulation.task, ManyClasses):
            raise ValueError("this attack is written for whole participants and several classes")
        if simulation.optimizer != "sgd":
            raise ValueError("this attack retraces plain gradient steps only")
        self._simulation = simulation
        self._clip = dp.clip_norm
        self._rate = dp.sampling_rate
        self._decoder = _Decoder.empty(len(units))
        self._weights = list(weights)
        self._worker = simulation.model_fn()
        self._tables = [
            (
                torch.as_tensor(np.asarray(unit.train_x), dtype=torch.float32),
                torch.as_tensor(np.asarray(unit.train_y).astype(np.int64)),
            )
            for unit in units
        ]
        # The attacker's own shuffling: it does not know the participants' secret one.
        self._generator = torch_generator(seed, "canary", 1)
        self.rounds = 0

    def _would_send(self, record: RoundRecord) -> np.ndarray:
        """What each test unit would have sent this round, one row per unit."""
        simulation, worker = self._simulation, self._worker
        rows = []
        for (features, targets), weight in zip(self._tables, self._weights, strict=True):
            set_state(worker, record.start)
            start = [v.detach().clone() for v in worker.parameters() if v.requires_grad]
            train_plain_round(
                worker,
                features,
                targets,
                loss_fn=lambda output, target: nn.functional.cross_entropy(
                    output, target, reduction="none"
                ),
                optimizer=torch.optim.SGD(worker.parameters(), lr=simulation.learning_rate),
                epochs=simulation.local_epochs,
                batch_size=simulation.batch_size,
                generator=self._generator,
            )
            rows.append(clipped_model_change(worker, start, self._clip, weight).numpy())
        return np.stack(rows)

    def observe(self, record: RoundRecord) -> None:
        if record.total is None:
            return
        # The run's own random numbers must not notice that it is being watched.
        state = torch.get_rng_state()
        try:
            sent = self._would_send(record)
        finally:
            torch.set_rng_state(state)
        opened = np.asarray(record.total, dtype=np.float64)
        # The direction all test units share says nothing about any one of them, and the
        # other participants move the model along it too. It is taken out of both sides.
        shared = sent.mean(axis=0)
        energy = float(shared @ shared)
        if energy > 0.0:
            sent = sent - np.outer(sent @ shared / energy, shared)
            opened = opened - (opened @ shared / energy) * shared
        # A unit that is in the run is drawn in a round with the public chance ``rate``.
        self._decoder.add(self._rate * sent, opened)
        self.rounds += 1

    def scores(self) -> np.ndarray:
        """One score per test unit; higher means "more likely in the run"."""
        return np.asarray(self._decoder.solve(), dtype=np.float64)


def writer_audit(
    setting: Setting,
    clients: Sequence[ClientData],
    *,
    units: int = TEST_WRITERS,
    share: float = GUESS_SHARE,
    trials: int = 200,
) -> dict[str, Any]:
    """Train once with whole participants flipped in and out, and attack the opened sums."""
    if setting.arm != PARTICIPANT_ARM:
        raise ValueError("whole participants are test units only where they are protected")
    torch.set_num_threads(1)
    started = time.perf_counter()
    picked, coins = choose_test_units(len(clients), units, setting.seed)
    left_out = {int(index) for index, stays in zip(picked, coins, strict=True) if not stays}
    taking_part = [client for index, client in enumerate(clients) if index not in left_out]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulation = build_simulation(setting, taking_part)
        # Public weights, as the run computes them: a share of the largest table that takes part.
        largest = max(client.n_train for client in taking_part)
        test_units = [clients[int(index)] for index in picked]
        weights = [min(1.0, unit.n_train / largest) for unit in test_units]
        attack = WriterAttack(simulation, test_units, weights, setting.seed)
        result = simulation.run(seed=setting.seed, observer=attack.observe)
    report = result.report
    assert report is not None
    scores = attack.scores()
    guesses, right = count_guesses(scores, coins, share)
    return {
        "test_units": units,
        "kept": int(coins.sum()),
        "participants_in_the_run": len(taking_part),
        "rounds_seen": attack.rounds,
        "guess_share": share,
        "guesses": guesses,
        "right": right,
        "advantage": measured_advantage(scores, coins),
        "epsilon": report.epsilon,
        "epsilon_lower_limit": one_run_epsilon_lower_bound(units, guesses, right, DELTA),
        "most_the_test_could_show": best_case_lower_bound(
            units, 1, share, report.advantage_ceiling, DELTA, trials=trials
        ),
        "ceiling_mechanism": report.advantage_ceiling,
        "ceiling_general": report.general_advantage_ceiling,
        "validation": dict(result.validation),
        "seconds": round(time.perf_counter() - started, 1),
    }


def writers(chosen: dict[str, dict[str, Any]], directory: Path = AUDIT) -> str:
    """The one audited handwriting run."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"femnist_writers_seed-{WRITER_SEED}"
    path = directory / f"{stem}.json"
    if not path.exists():
        setting = reported("femnist", "cnn", PARTICIPANT_ARM, WRITER_BUDGET, WRITER_SEED, chosen)
        data = writer_audit(setting, load("femnist", WRITER_SEED))
        config = {"benchmark": setting.__dict__, "test_units": TEST_WRITERS}
        write_record(
            path,
            make_result(kind="phase5_audit_writers", config=config, seeds=[WRITER_SEED], data=data),
        )
    print(json.loads(path.read_text(encoding="utf-8"))["data"])
    return stem


def main(arguments: list[str] | None = None) -> None:
    stages = list(sys.argv[1:] if arguments is None else arguments) or list(STAGES)
    unknown = sorted(set(stages) - set(STAGES))
    if unknown:
        raise SystemExit(f"unknown stage {unknown[0]!r}; the stages are {', '.join(STAGES)}")
    chosen = chosen_settings()
    for stage in stages:
        if stage == "health":
            health(chosen)
        else:
            writers(chosen)
        print(f"stage {stage} is complete", flush=True)


if __name__ == "__main__":
    main()
