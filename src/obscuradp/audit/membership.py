"""Attack a whole training run and see how well "was this record used?" can be answered.

The method trains **once** (Steinke, Nasr and Jagielski, "Privacy Auditing with One (1)
Training Run", 2023). Some of the participants' real training rows are picked as test
records. Each is kept in or left out by a fair coin. After the run an attacker gives every
test record a score and guesses, for the records it is most sure about, which way the coin
fell. Two things come out:

- a **lower limit on the privacy budget**: guessing right more often than a budget
  $\\varepsilon$ allows shows that the true budget is larger
  (``obscuradp.audit.one_run_epsilon_lower_bound``);
- the **measured advantage**: the attacker's true-positive rate minus its false-positive
  rate. 0 means no better than a coin.

A small limit or a small advantage is **not** proof of safety. It only says that these
attacks, on these records, found little. ``best_case_lower_bound`` says how much the test
could have shown at most, and every summary carries that number.

Two attacks are built in. Both may only use what their attacker can see:

- ``ATTACK_FINAL_MODEL``: an outsider who gets the final model and knows the test records.
  The score is how well the model fits the record.
- ``ATTACK_UPDATES``: the attacker the plan is written for (``obscuradp.audit.views``),
  with the knowledge the guarantee has to withstand: it sees what the plan lets it see in
  every round, and it knows every other row. It works out what each round's messages
  would look like without the test records and solves for the coins that best explain
  what is left over.

The participants' public sizes are held fixed with ``Simulation(declared_rows=...)``, so
that leaving a record out changes nothing an observer could read off directly.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from scipy import stats
from torch import nn

from obscuradp.accounting.accountants import general_advantage_ceiling
from obscuradp.accounting.calibrate import Calibration
from obscuradp.aggregation.fedavg import fedavg_weights
from obscuradp.audit.bounds import one_run_epsilon_lower_bound
from obscuradp.audit.views import (
    MODEL_OBSERVER,
    SERVER,
    AttackerView,
    model_observer_view,
    server_view,
)
from obscuradp.mechanisms.dpsgd import clipped_records, per_record_gradients
from obscuradp.plan import CentralDPSGD, ClientLevelDP, DistributedDPSGD, LocalDPSGD
from obscuradp.seeding import numpy_generator
from obscuradp.sim.simulation import (
    RECORD_MODELS,
    RECORD_SUMS,
    RoundRecord,
    Simulation,
    SimulationResult,
)
from obscuradp.state import State, flatten_state, get_state, set_state, state_layout
from obscuradp.tables.split import ClientData

ATTACK_FINAL_MODEL = "final model"
ATTACK_UPDATES = "updates"
ATTACKS = (ATTACK_FINAL_MODEL, ATTACK_UPDATES)

# Shares of the test records the attacker guesses for, the surest first. Counts are kept
# for each of them; which one a report uses has to be fixed before the runs it reports.
GUESS_SHARES = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)

# An interval from resampling runs means little with a handful of runs.
MIN_RUNS_FOR_INTERVAL = 10

_CHUNK = 2048

BoolVector = npt.NDArray[np.bool_]
FloatVector = npt.NDArray[np.float64]
SimulationBuilder = Callable[[Sequence[ClientData], Mapping[str, int]], Simulation]


# ---------------------------------------------------------------------- the test records


@dataclass(frozen=True)
class Canaries:
    """The test records of one run.

    Attributes:
        rows: For each participant, the positions of its test records among its training
            rows.
        kept: For each participant, one coin per test record: true if the record stays in.
    """

    rows: Mapping[str, npt.NDArray[np.int64]]
    kept: Mapping[str, BoolVector]

    @property
    def count(self) -> int:
        """Number of test records."""
        return int(sum(len(rows) for rows in self.rows.values()))

    def coins(self) -> BoolVector:
        """All coins in one vector, participant after participant."""
        parts = [self.kept[name] for name in self.rows]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=bool)


def choose_canaries(
    clients: Sequence[ClientData],
    seed: int,
    *,
    fraction: float = 0.25,
    cap: int = 250,
    skip: Collection[str] = (),
) -> Canaries:
    """Pick test records among the participants' training rows and flip their coins.

    Args:
        clients: The participants with all of their rows.
        seed: Seed of the choice and of the coins.
        fraction: Share of a participant's training rows that become test records.
        cap: Largest number of test records per participant.
        skip: Participants without test records (those that work with the attacker know
            their own rows anyway).
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if cap < 1:
        raise ValueError("cap must be at least 1")
    rows: dict[str, npt.NDArray[np.int64]] = {}
    kept: dict[str, BoolVector] = {}
    for index, client in enumerate(clients):
        if client.name in skip:
            continue
        count = min(cap, int(fraction * client.n_train))
        if count < 1:
            continue
        generator = numpy_generator(seed, "canary", index)
        rows[client.name] = np.sort(generator.choice(client.n_train, size=count, replace=False))
        kept[client.name] = generator.random(count) < 0.5
    return Canaries(rows=rows, kept=kept)


def leave_out(
    clients: Sequence[ClientData], canaries: Canaries
) -> tuple[list[ClientData], dict[str, int]]:
    """The participants without the test records whose coin says "out".

    Returns:
        The participants as the run will see them, and each one's public size, which is
        its full number of training rows whatever the coins say.
    """
    reduced = []
    declared = {}
    for client in clients:
        declared[client.name] = client.n_train
        if client.name not in canaries.rows:
            reduced.append(client)
            continue
        stays = np.ones(client.n_train, dtype=bool)
        stays[canaries.rows[client.name][~canaries.kept[client.name]]] = False
        if not stays.any():
            raise ValueError(f"{client.name}: the coins left no training row")
        reduced.append(
            replace(client, train_x=client.train_x[stays], train_y=client.train_y[stays])
        )
    return reduced, declared


# ---------------------------------------------------------------------- the attacks


def _trainable_positions(model: nn.Module) -> npt.NDArray[np.int64]:
    """Where the trainable parameters sit in a flattened state, in the order of the gradients."""
    layout = {entry.name: entry for entry in state_layout(get_state(model))}
    positions = []
    for name, value in model.named_parameters():
        if value.requires_grad:
            entry = layout[name.replace("_module.", "")]
            positions.append(np.arange(entry.start, entry.stop, dtype=np.int64))
    if not positions:
        raise ValueError("the model has no trainable parameter")
    return np.concatenate(positions)


class _RowGradients:
    """Each row's gradient as the run would use it: shortened if the plan shortens."""

    def __init__(self, simulation: Simulation, model: nn.Module) -> None:
        self.model = model
        outputs = simulation.outputs
        from_logits = (
            outputs == "logit"
            if outputs is not None
            else not bool(getattr(model, "probability_output", True))
        )
        self.loss_fn = simulation.plan.imbalance.loss(from_logits=from_logits)
        dp = simulation.plan.dp
        self.clip_norm = None if dp is None else dp.clip_norm
        self.clip_scale = simulation.plan.imbalance.clip_scale

    def rows(self, state: State, features: torch.Tensor, targets: torch.Tensor) -> FloatVector:
        """One row of numbers per record."""
        set_state(self.model, state)
        self.model.train()
        pieces = []
        for begin in range(0, int(features.shape[0]), _CHUNK):
            x, y = features[begin : begin + _CHUNK], targets[begin : begin + _CHUNK]
            gradients = per_record_gradients(self.model, self.loss_fn, x, y)
            if self.clip_norm is None:
                flat = torch.cat([g.detach().double().flatten(1) for g in gradients.values()], 1)
            else:
                flat = clipped_records(gradients, self.clip_norm, self.clip_scale(y))
            pieces.append(flat.numpy())
        if not pieces:
            return np.zeros((0, len(_trainable_positions(self.model))))
        return np.concatenate(pieces)

    def total(self, state: State, features: torch.Tensor, targets: torch.Tensor) -> FloatVector:
        """The sum over the records."""
        result: FloatVector = self.rows(state, features, targets).sum(axis=0)
        return result

    def losses(self, state: State, features: torch.Tensor, targets: torch.Tensor) -> FloatVector:
        """Each record's loss under ``state``."""
        set_state(self.model, state)
        self.model.eval()
        with torch.no_grad():
            values = self.loss_fn(self.model(features), targets)
        self.model.train()
        result: FloatVector = values.detach().double().flatten().numpy()
        return result


@dataclass
class _Decoder:
    """Solve for the coins that best explain what the attacker could not otherwise explain.

    Round after round the attacker has a vector it cannot explain (``residual``) and, for
    every test record, the vector that record would add if its coin said "in" rather than
    "undecided" (a column of ``columns``). The best guess of the coins that is linear in
    the residuals is a ridge regression, with a penalty that grows with the noise; its
    solution is the score.

    The size of the noise is **measured from the residuals**, not taken from what the
    pipeline says about itself. An audit exists to catch a pipeline whose noise is not
    what it states, and an attacker that believed the stated noise would look away exactly
    then.
    """

    size: int
    gram: FloatVector
    moment: FloatVector
    energy: float = 0.0
    numbers: int = 0

    @classmethod
    def empty(cls, size: int) -> _Decoder:
        return cls(size=size, gram=np.zeros((size, size)), moment=np.zeros(size))

    def add(self, columns: FloatVector, residual: FloatVector) -> None:
        """``columns`` has one row per test record."""
        self.gram += columns @ columns.T
        self.moment += columns @ residual
        self.energy += float(residual @ residual)
        self.numbers += int(residual.size)

    def solve(self) -> FloatVector:
        if self.size == 0:
            return np.zeros(0)
        typical = float(np.trace(self.gram)) / self.size
        if typical <= 0.0 or self.numbers == 0:
            return np.zeros(self.size)
        # A coin is plus or minus one half around "undecided": variance one quarter. What
        # is left unexplained per number is treated as noise, which overstates it a little.
        ridge = max(4.0 * self.energy / self.numbers, 1e-6 * typical)
        solution = np.linalg.solve(self.gram + ridge * np.eye(self.size), self.moment)
        return np.asarray(solution, dtype=np.float64)


class _UpdateAttack:
    """``ATTACK_UPDATES``, fed one attacker's view per round."""

    def __init__(
        self,
        simulation: Simulation,
        clients: Sequence[ClientData],
        canaries: Canaries,
        attacker: str,
        colluders: frozenset[str],
        path_points: int = 1,
    ) -> None:
        if path_points < 1:
            raise ValueError("path_points must be at least 1")
        self.path_points = path_points
        dp = simulation.plan.dp
        if isinstance(dp, ClientLevelDP):
            raise ValueError(
                "this audit asks about single records; a plan that protects whole "
                "participants needs whole participants as test units"
            )
        if dp is not None and not isinstance(dp, DistributedDPSGD | CentralDPSGD | LocalDPSGD):
            raise ValueError(f"no update attack is written for {type(dp).__name__}")
        if not simulation.plan.single_step and simulation.optimizer != "sgd":
            raise ValueError("the update attack on local training is written for the sgd optimizer")
        if simulation.plan.proximal_mu > 0.0:
            raise ValueError("the update attack does not model the proximal term")
        if colluders & set(canaries.rows):
            raise ValueError("participants that work with the attacker cannot hold test records")
        self.simulation = simulation
        self.attacker = attacker
        self.colluders = colluders
        self.names = [client.name for client in clients]
        self.canaries = canaries
        self.calibrations = simulation.calibrations()
        sizes = [client.n_train for client in clients]
        self.sizes = dict(zip(self.names, sizes, strict=True))
        self.weights = dict(
            zip(
                self.names,
                fedavg_weights(sizes, weighted=simulation.weighted_average),
                strict=True,
            )
        )
        # Model building draws from the global random stream, so it happens before the run.
        self.gradients = _RowGradients(simulation, simulation.model_fn())
        self.positions = _trainable_positions(self.gradients.model)
        self.known: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.tested: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for client in clients:
            x = torch.as_tensor(np.asarray(client.train_x, dtype=np.float32))
            y = torch.as_tensor(np.asarray(client.train_y, dtype=np.float32))
            picked = np.zeros(client.n_train, dtype=bool)
            if client.name in canaries.rows:
                picked[canaries.rows[client.name]] = True
                self.tested[client.name] = (x[picked], y[picked])
            self.known[client.name] = (x[~picked], y[~picked])
        self.offsets: dict[str, int] = {}
        offset = 0
        for name in canaries.rows:
            self.offsets[name] = offset
            offset += len(canaries.rows[name])
        self.count = offset
        # Where the server reads every participant's change, each participant's test
        # records are solved for from that participant's own change.
        self.per_participant = (
            attacker == SERVER
            and simulation.plan.secure_sum is None
            and not simulation.plan.single_step
        )
        self.decoders = (
            {name: _Decoder.empty(len(canaries.rows[name])) for name in canaries.rows}
            if self.per_participant
            else {"": _Decoder.empty(self.count)}
        )

    # -- how much one row moves a participant's change in local training

    def _local_scale(self, name: str) -> float:
        sim = self.simulation
        if sim.plan.dp is None:
            # Every row is met once per pass, in a batch of B rows whose mean is followed.
            batch = min(sim.batch_size, self.sizes[name])
            return sim.learning_rate * sim.local_epochs / batch
        calibration = self.calibrations[name]
        assert isinstance(calibration, Calibration)
        steps = calibration.steps // sim.rounds
        # A row joins a step with probability q and is divided by q times the public size.
        return sim.learning_rate * steps / self.sizes[name]

    # -- one round

    def observe(self, record: RoundRecord) -> None:
        view = (
            server_view(record, self.colluders)
            if self.attacker == SERVER
            else model_observer_view(record)
        )
        saved = torch.random.get_rng_state()
        try:
            if view.kind == RECORD_MODELS:
                self._local_round(view)
            elif view.kind == RECORD_SUMS:
                self._sum_round(view)
            else:
                raise ValueError(f"no update attack is written for rounds of {view.kind}")
        finally:
            # The attacker's arithmetic must not move the run's own random stream.
            torch.random.set_rng_state(saved)

    def _along(self, start: State, change: FloatVector) -> list[State]:
        """Models spread evenly along ``change``, the straight line a round's training took.

        Local training moves the model while it runs, and with it every row's gradient.
        The attacker sees where the round started and where it ended, so it takes the
        gradients at points in between rather than at the start alone.
        """
        origin = flatten_state(start)
        states = []
        for point in range(self.path_points):
            flat = origin + (point + 0.5) / self.path_points * change
            state = {}
            for entry in state_layout(start):
                piece = flat[entry.start : entry.stop].reshape(entry.shape)
                state[entry.name] = torch.as_tensor(piece, dtype=start[entry.name].dtype)
            states.append(state)
        return states

    def _mean_total(self, states: Sequence[State], name: str) -> FloatVector:
        parts = [self.gradients.total(state, *self.known[name]) for state in states]
        result: FloatVector = np.mean(parts, axis=0)
        return result

    def _mean_rows(self, states: Sequence[State], name: str) -> FloatVector:
        parts = [self.gradients.rows(state, *self.tested[name]) for state in states]
        result: FloatVector = np.mean(parts, axis=0)
        return result

    def _local_round(self, view: AttackerView) -> None:
        if view.total is None:
            return
        if self.per_participant:
            for name, decoder in self.decoders.items():
                change = view.messages[name]
                scale = self._local_scale(name)
                at = self._along(view.start, change)
                background = self._mean_total(at, name)
                columns = -scale * self._mean_rows(at, name)
                expected = -scale * background + 0.5 * columns.sum(axis=0)
                decoder.add(columns, change[self.positions] - expected)
            return
        # Only the average change is visible: all test records are solved for together.
        at = self._along(view.start, view.total)
        expected = np.zeros(len(self.positions))
        columns = np.zeros((self.count, len(self.positions)))
        for name in self.names:
            scale = self._local_scale(name)
            weight = float(self.weights[name])
            expected += -weight * scale * self._mean_total(at, name)
            if name in self.tested:
                rows = -weight * scale * self._mean_rows(at, name)
                begin = self.offsets[name]
                columns[begin : begin + rows.shape[0]] = rows
        expected += 0.5 * columns.sum(axis=0)
        self.decoders[""].add(columns, view.total[self.positions] - expected)

    def _sum_round(self, view: AttackerView) -> None:
        left = view.honest_total()
        if left is None:
            return
        rate = self.calibrations[self.names[0]].sampling_rate
        expected = np.zeros(len(self.positions))
        columns = np.zeros((self.count, len(self.positions)))
        for name in self.names:
            if name in self.colluders:
                continue
            expected += rate * self.gradients.total(view.start, *self.known[name])
            if name in self.tested:
                rows = rate * self.gradients.rows(view.start, *self.tested[name])
                begin = self.offsets[name]
                columns[begin : begin + rows.shape[0]] = rows
        expected += 0.5 * columns.sum(axis=0)
        self.decoders[""].add(columns, left - expected)

    def scores(self) -> FloatVector:
        if self.per_participant:
            parts = [self.decoders[name].solve() for name in self.canaries.rows]
            return np.concatenate(parts) if parts else np.zeros(0)
        return self.decoders[""].solve()

    def final_model_scores(self, state: State) -> FloatVector:
        """``ATTACK_FINAL_MODEL``: a record the model fits well was more likely used."""
        parts = [-self.gradients.losses(state, *self.tested[name]) for name in self.canaries.rows]
        return np.concatenate(parts) if parts else np.zeros(0)


# ---------------------------------------------------------------------- one run


@dataclass(frozen=True)
class AuditRun:
    """One audited training run.

    Attributes:
        kept: One coin per test record: true if the record was in the training rows.
        scores: For each attack, one score per test record; higher means "more likely in".
        result: The run's own result, with its privacy report.
        attacker: Whose view the update attack used.
        colluders: The participants that worked with that attacker.
    """

    kept: BoolVector
    scores: Mapping[str, FloatVector]
    result: SimulationResult
    attacker: str
    colluders: tuple[str, ...]


def default_attacker(simulation: Simulation) -> str:
    """The attacker a plan is written for: the server, unless the plan trusts the server."""
    return MODEL_OBSERVER if isinstance(simulation.plan.dp, CentralDPSGD) else SERVER


def default_colluders(simulation: Simulation) -> tuple[str, ...]:
    """As many participants as the plan assumes may work with the server: the last ones."""
    dp = simulation.plan.dp
    if not isinstance(dp, DistributedDPSGD) or dp.max_colluding == 0:
        return ()
    return tuple(client.name for client in simulation.clients)[-dp.max_colluding :]


def audit_run(
    build: SimulationBuilder,
    clients: Sequence[ClientData],
    seed: int,
    *,
    fraction: float = 0.25,
    cap: int = 250,
    attacker: str | None = None,
    colluders: Collection[str] | None = None,
    path_points: int = 1,
) -> AuditRun:
    """Train once with test records flipped in and out, and attack the run.

    Args:
        build: Makes the simulation to audit from the participants and their public sizes:
            ``build(clients, declared_rows)`` must return
            ``Simulation(clients=clients, declared_rows=declared_rows, ...)``.
        clients: The participants with all of their rows.
        seed: Seed of the test records, their coins and the run.
        fraction: Share of each participant's training rows that become test records.
        cap: Largest number of test records per participant.
        attacker: ``obscuradp.audit.views.SERVER`` or ``MODEL_OBSERVER``. By default the
            attacker the plan is written for.
        colluders: Participants that work with the server. By default as many as the plan
            assumes. They hold no test records.
        path_points: For plans with local training: at how many points between a round's
            start and end the update attack takes the rows' gradients. More points follow
            a round's training more closely and cost that many times the arithmetic.
    """
    full = [client.n_train for client in clients]
    probe = build(clients, dict(zip([c.name for c in clients], full, strict=True)))
    who = default_attacker(probe) if attacker is None else attacker
    if who not in (SERVER, MODEL_OBSERVER):
        raise ValueError(f"attacker must be {SERVER!r} or {MODEL_OBSERVER!r}")
    together = frozenset(default_colluders(probe) if colluders is None else colluders)
    if who == MODEL_OBSERVER and together:
        raise ValueError("a model observer has no colluders")
    canaries = choose_canaries(clients, seed, fraction=fraction, cap=cap, skip=together)
    if canaries.count == 0:
        raise ValueError("no test records: the participants have too few rows")
    reduced, declared = leave_out(clients, canaries)
    simulation = build(reduced, declared)
    if simulation.declared_rows != declared:
        raise ValueError("build must pass declared_rows on to the Simulation")
    # The attacker works from the full tables: it knows every row and every test record.
    attack = _UpdateAttack(probe, clients, canaries, who, together, path_points)
    result = simulation.run(seed, observer=attack.observe)
    return AuditRun(
        kept=canaries.coins(),
        scores={
            ATTACK_FINAL_MODEL: attack.final_model_scores(result.state),
            ATTACK_UPDATES: attack.scores(),
        },
        result=result,
        attacker=who,
        colluders=tuple(sorted(together)),
    )


# ---------------------------------------------------------------------- reading the scores


def count_guesses(scores: npt.ArrayLike, kept: npt.ArrayLike, share: float) -> tuple[int, int]:
    """Guess for the surest ``share`` of the test records; return (guesses, right guesses).

    Half of the guesses go to the highest scores ("in"), half to the lowest ("out"). With
    a share of 1 every record is guessed for.
    """
    if not 0.0 < share <= 1.0:
        raise ValueError("share must be in (0, 1]")
    values = np.asarray(scores, dtype=np.float64)
    coins = np.asarray(kept, dtype=bool)
    if values.shape != coins.shape or values.ndim != 1:
        raise ValueError("scores and coins must be two vectors of the same length")
    each = int(share * values.size) // 2
    if each < 1:
        return 0, 0
    # Ties are broken by position, which carries no information about the coins.
    order = np.argsort(values, kind="stable")
    lowest, highest = order[:each], order[-each:]
    right = int(np.sum(coins[highest])) + int(np.sum(~coins[lowest]))
    return 2 * each, right


def _advantage_at(scores: FloatVector, coins: BoolVector, threshold: float) -> float:
    inside, outside = scores[coins], scores[~coins]
    if inside.size == 0 or outside.size == 0:
        return 0.0
    return float(np.mean(inside > threshold) - np.mean(outside > threshold))


def measured_advantage(scores: npt.ArrayLike, kept: npt.ArrayLike) -> float:
    """True-positive rate minus false-positive rate at a cut-off chosen on other records.

    The test records are split into two fixed halves by position. The cut-off that looks
    best on one half is applied to the other half, and the two results are averaged. So
    the figure is not flattered by a cut-off fitted to the records it is measured on.
    """
    values = np.asarray(scores, dtype=np.float64)
    coins = np.asarray(kept, dtype=bool)
    if values.shape != coins.shape or values.ndim != 1:
        raise ValueError("scores and coins must be two vectors of the same length")
    first = np.arange(values.size) % 2 == 0
    results = []
    for chooser in (first, ~first):
        candidates = np.unique(values[chooser])
        if candidates.size == 0:
            continue
        gains = [_advantage_at(values[chooser], coins[chooser], float(t)) for t in candidates]
        best = float(candidates[int(np.argmax(gains))])
        results.append(_advantage_at(values[~chooser], coins[~chooser], best))
    return float(np.mean(results)) if results else 0.0


@dataclass(frozen=True)
class RunSummary:
    """What is kept of one audited run: counts and figures, no single record's score.

    Attributes:
        canaries: Number of test records.
        guesses: For each attack and each share of ``GUESS_SHARES`` (as text), the number of
            guesses and of right guesses.
        advantage: For each attack, the measured advantage of this run.
        epsilon: The budget the run's privacy report states (``None`` without a report).
        delta: The report's $\\delta$.
        ceiling_mechanism: The proven limit on the advantage for the noise the run added.
    """

    canaries: int
    guesses: Mapping[str, Mapping[str, tuple[int, int]]]
    advantage: Mapping[str, float]
    epsilon: float | None
    delta: float | None
    ceiling_mechanism: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "canaries": self.canaries,
            "guesses": {
                attack: {share: list(pair) for share, pair in by_share.items()}
                for attack, by_share in self.guesses.items()
            },
            "advantage": dict(self.advantage),
            "epsilon": self.epsilon,
            "delta": self.delta,
            "ceiling_mechanism": self.ceiling_mechanism,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunSummary:
        return cls(
            canaries=int(data["canaries"]),
            guesses={
                attack: {share: (int(pair[0]), int(pair[1])) for share, pair in by_share.items()}
                for attack, by_share in data["guesses"].items()
            },
            advantage={attack: float(value) for attack, value in data["advantage"].items()},
            epsilon=None if data["epsilon"] is None else float(data["epsilon"]),
            delta=None if data["delta"] is None else float(data["delta"]),
            ceiling_mechanism=None
            if data["ceiling_mechanism"] is None
            else float(data["ceiling_mechanism"]),
        )


def summarize_run(run: AuditRun) -> RunSummary:
    """Reduce one audited run to its counts."""
    report = run.result.report
    return RunSummary(
        canaries=int(run.kept.size),
        guesses={
            attack: {f"{share:g}": count_guesses(scores, run.kept, share) for share in GUESS_SHARES}
            for attack, scores in run.scores.items()
        },
        advantage={
            attack: measured_advantage(scores, run.kept) for attack, scores in run.scores.items()
        },
        epsilon=None if report is None else report.epsilon,
        delta=None if report is None else report.policy.delta,
        ceiling_mechanism=None if report is None else report.advantage_ceiling,
    )


def best_case_lower_bound(
    canaries: int,
    runs: int,
    share: float,
    advantage_ceiling: float,
    delta: float,
    *,
    confidence: float = 0.95,
    trials: int = 200,
    seed: int = 0,
) -> float:
    """The lower limit this test would show against an attacker that could do no better.

    The proven limit on the advantage says how far apart "in" and "out" can be at most.
    This plays an attacker whose score for every test record is exactly that far apart
    (two bell curves, as for plain Gaussian noise), with the same number of records, runs
    and guesses, and returns the typical (median) lower limit. A real attack on real
    records stays below it; the gap between this number and the stated budget is what the
    test cannot see whatever the attack.

    Args:
        canaries: Test records per run.
        runs: Number of pooled runs.
        share: Share of the records that are guessed for.
        advantage_ceiling: The proven limit on the advantage, between 0 and 1.
        delta: The $\\delta$ of the guarantee.
        confidence: Confidence of the lower limit.
        trials: Number of simulated audits.
        seed: Seed of the simulation.
    """
    if canaries < 1 or runs < 1:
        raise ValueError("need at least one record and one run")
    if not 0.0 <= advantage_ceiling <= 1.0:
        raise ValueError("advantage_ceiling must be in [0, 1]")
    limit = min(advantage_ceiling, 1.0 - 1e-12)
    separation = 2.0 * float(stats.norm.ppf((1.0 + limit) / 2.0))
    generator = np.random.default_rng(seed)
    limits = []
    for _ in range(trials):
        guesses = right = 0
        for _ in range(runs):
            coins = generator.random(canaries) < 0.5
            scores = generator.normal(size=canaries) + separation * (coins - 0.5)
            made, correct = count_guesses(scores, coins, share)
            guesses += made
            right += correct
        limits.append(
            one_run_epsilon_lower_bound(canaries * runs, guesses, right, delta, confidence)
        )
    return float(np.median(limits))


@dataclass(frozen=True)
class MembershipAudit:
    """The outcome of one attack over several audited runs.

    The three numbers the roadmap asks for side by side are ``ceiling_any`` (true for any
    method with the stated budget), ``ceiling_mechanism`` (true for the noise that was
    added) and ``advantage`` (what this attack achieved).

    Attributes:
        attack: The attack's name.
        runs: Number of runs.
        canaries: Test records over all runs.
        share: The share of records guessed for.
        guesses: Guesses over all runs.
        correct: Right guesses over all runs.
        epsilon_claimed: The budget the pipeline states.
        delta: Its $\\delta$.
        epsilon_lower: Lower limit on the budget from the pooled guesses.
        epsilon_lower_median_run: The median of the limits of the single runs.
        best_case: The lower limit an attacker at the proven ceiling would typically show
            with the same counts (``None`` without a proven ceiling).
        advantage: Mean measured advantage of the runs.
        advantage_low: Lower end of its interval (resampling whole runs).
        advantage_high: Upper end.
        ceiling_any: Largest advantage any method with the stated budget allows.
        ceiling_mechanism: Largest advantage the added noise allows.
        confidence: Confidence of the limit and of the interval.
        flagged: True if the lower limit is above the stated budget: the pipeline leaks
            more than it says.
        above_noise_limit: True if even the lower end of the advantage interval is above
            ``ceiling_mechanism``: the attack does better than the stated noise allows, so
            the noise is not what the pipeline says, whether or not the budget limit shows
            it. Only judged from ``MIN_RUNS_FOR_INTERVAL`` runs on, because the interval
            comes from resampling runs.
    """

    attack: str
    runs: int
    canaries: int
    share: float
    guesses: int
    correct: int
    epsilon_claimed: float
    delta: float
    epsilon_lower: float
    epsilon_lower_median_run: float
    best_case: float | None
    advantage: float
    advantage_low: float
    advantage_high: float
    ceiling_any: float
    ceiling_mechanism: float | None
    confidence: float
    flagged: bool
    above_noise_limit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def describe(self) -> str:
        """The outcome in plain words."""
        limit = "none proven" if self.ceiling_mechanism is None else f"{self.ceiling_mechanism:.1%}"
        reach = "" if self.best_case is None else f" (the most it could show: {self.best_case:.2f})"
        verdict = "FLAGGED: leaks more than stated" if self.flagged else "not flagged"
        if self.above_noise_limit:
            verdict += "; ABOVE THE LIMIT FOR THE STATED NOISE: the noise is not as stated"
        return (
            f"{self.attack} attack, {self.runs} run(s), {self.canaries} test records: "
            f"stated budget {self.epsilon_claimed:g}, lower limit shown "
            f"{self.epsilon_lower:.2f}{reach}; advantage {self.advantage:.1%} "
            f"({self.advantage_low:.1%} to {self.advantage_high:.1%}), limit for the noise "
            f"used {limit}, limit for any method {self.ceiling_any:.1%}. {verdict}."
        )


def summarize(
    runs: Sequence[RunSummary],
    attack: str,
    share: float,
    *,
    claimed_epsilon: float | None = None,
    delta: float | None = None,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> MembershipAudit:
    """Put several audited runs of one pipeline together.

    Args:
        runs: The runs. They must be independent: each with its own seed.
        attack: Which attack to report.
        share: The share of test records guessed for, one of ``GUESS_SHARES``. It must be
            fixed before the runs are looked at.
        claimed_epsilon: The budget the pipeline states, if the runs carry no privacy
            report (a pipeline without protection has none).
        delta: Its $\\delta$, likewise.
        confidence: Confidence of the lower limit and of the advantage interval.
        resamples: Number of resamples of whole runs for the interval.
        seed: Seed of the resampling.
    """
    if not runs:
        raise ValueError("no runs to summarize")
    key = f"{share:g}"
    reported = [run.epsilon for run in runs if run.epsilon is not None]
    if claimed_epsilon is None:
        if len(reported) != len(runs):
            raise ValueError("runs without a privacy report need claimed_epsilon and delta")
        claimed_epsilon = max(reported)
    deltas = [run.delta for run in runs if run.delta is not None]
    if delta is None:
        if len(deltas) != len(runs):
            raise ValueError("runs without a privacy report need claimed_epsilon and delta")
        delta = max(deltas)
    canaries = sum(run.canaries for run in runs)
    made = sum(run.guesses[attack][key][0] for run in runs)
    right = sum(run.guesses[attack][key][1] for run in runs)
    lower = one_run_epsilon_lower_bound(canaries, made, right, delta, confidence)
    single = [
        one_run_epsilon_lower_bound(
            run.canaries,
            run.guesses[attack][key][0],
            run.guesses[attack][key][1],
            delta,
            confidence,
        )
        for run in runs
    ]
    advantages = np.array([run.advantage[attack] for run in runs], dtype=np.float64)
    generator = np.random.default_rng(seed)
    if len(runs) > 1:
        picks = generator.integers(0, len(runs), size=(resamples, len(runs)))
        means = advantages[picks].mean(axis=1)
        tail = (1.0 - confidence) / 2.0
        low, high = (float(x) for x in np.quantile(means, [tail, 1.0 - tail]))
    else:
        low = high = float(advantages[0])
    ceilings = [run.ceiling_mechanism for run in runs if run.ceiling_mechanism is not None]
    ceiling = max(ceilings) if len(ceilings) == len(runs) else None
    best = None
    if ceiling is not None:
        per_run = max(1, round(canaries / len(runs)))
        best = best_case_lower_bound(
            per_run, len(runs), share, ceiling, delta, confidence=confidence, seed=seed
        )
    return MembershipAudit(
        attack=attack,
        runs=len(runs),
        canaries=canaries,
        share=share,
        guesses=made,
        correct=right,
        epsilon_claimed=float(claimed_epsilon),
        delta=float(delta),
        epsilon_lower=lower,
        epsilon_lower_median_run=float(np.median(single)),
        best_case=best,
        advantage=float(advantages.mean()),
        advantage_low=low,
        advantage_high=high,
        ceiling_any=general_advantage_ceiling(float(claimed_epsilon), float(delta)),
        ceiling_mechanism=ceiling,
        confidence=confidence,
        flagged=lower > float(claimed_epsilon),
        above_noise_limit=(
            ceiling is not None and len(runs) >= MIN_RUNS_FOR_INTERVAL and low > ceiling
        ),
    )
