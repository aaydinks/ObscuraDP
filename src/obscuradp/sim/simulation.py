"""Run a whole federation on one machine.

``Simulation`` plays every role itself: the participants, who train on their own rows, and
the server, which averages what they send. It is the tool for research and for trying a
configuration before a real deployment.

Two kinds of round exist. With ``LocalDPSGD`` (and without privacy) every participant
trains for a while on its own rows and the server averages the resulting models. With
``DistributedDPSGD`` and ``CentralDPSGD`` a round is a single step: every participant sends
one clipped gradient sum computed at the shared model, and the server applies the noisy
total.

What it guarantees for a private plan:

- The noise each participant needs is worked out once, for the **whole run**, from the
  policy's total budget. The whole run is reserved in the privacy ledger before any data is
  touched, so a run that cannot fit its budget never starts.
- Every round of every participant is charged to the ledger before it happens.
- Settings are judged on validation rows. Test rows are scored once, at the end.
- Anything that weakens the guarantee is refused unless ``allow_unsafe=True``, and is then
  written into the privacy report.
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from obscuradp.accounting.calibrate import (
    Calibration,
    SkellamCalibration,
    calibrate_noise_multiplier,
    calibrate_skellam,
    steps_per_round,
)
from obscuradp.accounting.events import SubsampledGaussianEvent
from obscuradp.aggregation.fedavg import aggregate_states, fedavg_weights
from obscuradp.client.trainer import (
    clipped_model_change,
    clipped_sum_step,
    integer_sum_step,
    private_sum_step,
    train_plain_round,
    train_private_round,
)
from obscuradp.compression.grid import SumGrid, plan_integer_grid, plan_sum_grid
from obscuradp.compression.packing import float32_size
from obscuradp.compression.quantize import Quantizer
from obscuradp.crypto.pairwise_masking import (
    PUBLIC_KEY_BYTES,
    MaskingAggregator,
    MaskingClient,
    setup_pairwise_masking,
)
from obscuradp.crypto.roles import MissingParticipants
from obscuradp.imbalance import UnsafeOversampling
from obscuradp.ledger import PrivacyLedger
from obscuradp.mechanisms.dpsgd import check_per_record_model
from obscuradp.mechanisms.noise import entropy_generator, gaussian_noise
from obscuradp.mechanisms.shared_noise import default_sum_bound
from obscuradp.mechanisms.skellam import IntegerStep, plan_integer_step
from obscuradp.metrics import best_threshold, binary_metrics
from obscuradp.plan import (
    SMALL_FEDERATION,
    CentralDPSGD,
    ClientLevelDP,
    DistributedDPSGD,
    LocalDPSGD,
    Plan,
    SmallFederationWarning,
)
from obscuradp.policy import PrivacyPolicy
from obscuradp.report import (
    MODE_CENTRAL,
    MODE_DISTRIBUTED,
    MODE_PARTICIPANT,
    ModeReport,
    PrivacyReport,
    collusion_table,
)
from obscuradp.results import make_result
from obscuradp.seeding import derive_seed, numpy_generator, torch_generator
from obscuradp.state import (
    State,
    flatten_state,
    get_state,
    set_state,
    state_size,
    unflatten_state,
)
from obscuradp.tables.split import ClientData
from obscuradp.unsafe import UnsafeFlags

OPTIMIZERS = ("sgd", "adam")
# The ledger scope of a run that protects whole participants: one budget for the group.
GROUP_SCOPE = "all participants"


# What a round's messages are: each participant's change to the shared model after its local
# training, or one clipped gradient sum per participant (the one-step modes), or each
# participant's shortened change (whole participants protected).
RECORD_MODELS = "model changes"
RECORD_SUMS = "gradient sums"
RECORD_CHANGES = "shortened changes"


@dataclass(frozen=True)
class RoundRecord:
    """Everything that left a participant or the server in one round. **For audits only.**

    A record holds more than any single party sees. Whoever plays an attacker has to cut
    it down to that attacker's view first (``obscuradp.audit.views``); handing a whole
    record to anyone would break the plan's protection.

    Attributes:
        round: The round's number, counted from 0. In the modes that share the noise it is
            the number of the attempt: a failed round has a record too.
        kind: What the messages are, one of ``RECORD_MODELS``, ``RECORD_SUMS`` and
            ``RECORD_CHANGES``.
        start: The shared model at the start of the round.
        messages: What each participant sent, before any masking, as one 64-bit vector
            per participant. Participants that were missing have no entry. For
            ``RECORD_MODELS`` it is the participant's change to the shared model; where the
            messages are masked it is already multiplied by the participant's public
            averaging weight, as it enters the sum. Masked messages add up to ``total``
            (up to rounding).
        masked: True if the messages went through the secure sum, so that the server saw
            none of them on its own.
        total: What moved the shared model in this round. For ``RECORD_MODELS`` the average
            change of the model, which is all the server gets where the messages are
            masked. Otherwise the opened secure sum, or the noised total of a trusted
            server. ``None`` for a round whose sum could not be opened.
        divisor: The public number ``total`` is divided by before it moves the model (1
            where it is used as it is).
    """

    round: int
    kind: str
    start: State
    messages: Mapping[str, np.ndarray[Any, Any]]
    masked: bool
    total: np.ndarray[Any, Any] | None
    divisor: float = 1.0


RoundObserver = Callable[[RoundRecord], None]


@dataclass(frozen=True)
class SimulationResult:
    """Everything one run produced.

    Attributes:
        report: The privacy report, or ``None`` for a run without privacy protection.
        history: One entry per evaluated round: the round number and the measures on the
            pooled validation rows.
        validation: Measures of the final model on the pooled validation rows.
        test: Measures of the final model on the pooled test rows, at threshold 0.5.
        test_at_chosen_threshold: The same at the threshold chosen on validation rows, if
            that was requested.
        threshold: The decision threshold chosen on validation rows, or 0.5.
        noise_multipliers: The noise multiplier of each participant (empty if not private).
        communication: What one participant uploads per round: ``upload_bytes_per_round``,
            the same update as 32-bit numbers (``plain_bytes_per_round``), their ``ratio``,
            ``bits_per_value``, one-time ``setup_bytes``, and ``saturated_share`` (the largest
            share of numbers in any update that fell outside the planned range: cut off
            before rounding in the local mode; in the shared-noise mode limited by
            ``sum_bound`` before the noise, or wrapped around afterwards).
        state: The final shared model.
        seed: The run's seed.
        config: The settings of the run.
        seconds: Wall-clock time of the run.
    """

    report: PrivacyReport | None
    history: tuple[dict[str, Any], ...]
    validation: dict[str, float]
    test: dict[str, float]
    test_at_chosen_threshold: dict[str, float] | None
    threshold: float
    noise_multipliers: dict[str, float]
    communication: dict[str, float]
    state: State
    seed: int
    config: dict[str, Any]
    seconds: float

    def to_record(self, directory: str | None = None) -> dict[str, Any]:
        """A result record with the code version, machine, settings fingerprint and seed."""
        data: dict[str, Any] = {
            "history": list(self.history),
            "validation": self.validation,
            "test": self.test,
            "test_at_chosen_threshold": self.test_at_chosen_threshold,
            "threshold": self.threshold,
            "noise_multipliers": self.noise_multipliers,
            "communication": self.communication,
            "seconds": self.seconds,
            "privacy_report": None if self.report is None else self.report.to_dict(),
        }
        return make_result(
            kind="simulation", config=self.config, seeds=[self.seed], data=data, directory=directory
        )


@dataclass
class Simulation:
    """A federation of participants and a server, run in one process.

    Args:
        model_fn: Builds a fresh model. It is called once per run; every participant starts
            each round from the shared weights.
        clients: The participants' data, already encoded and cut into training, validation
            and test rows (see ``obscuradp.tables``).
        plan: The means of protection (see ``obscuradp.plan``).
        policy: The total privacy budget. Required for a private plan; must be ``None`` for
            a plan without protection.
        rounds: Number of rounds $R$. With ``DistributedDPSGD`` it is the number of rounds
            that are started: a round that fails because a participant is missing still
            counts against the budget and is not repeated for free.
        local_epochs: Passes over a participant's rows per round, $E$. In a private run each
            participant takes $\\lceil E / q \\rceil$ steps per round. The one-step modes
            (``DistributedDPSGD``, ``CentralDPSGD``) only accept 1.
        batch_size: Batch size $B$. In a private run it is the expected batch size: each row
            joins each step with probability $q = \\min(1, B / n)$ for a participant with
            $n$ rows. In the one-step modes there is one batch per round for the whole
            group: $n$ is the number of rows of all participants together and every row
            joins with the same probability.
        learning_rate: Step size of the local optimizer.
        optimizer: ``"sgd"`` or ``"adam"``. A fresh optimizer is made for every participant
            in every round.
        weighted_average: Average the participants' models by their number of training rows
            (true) or equally (false). The one-step modes treat every row alike, which is
            the weighted average, and refuse ``False``.
        evaluate_every: Score the shared model on validation rows every this many rounds
            (the last round is always scored). 0 scores only the final model.
        choose_threshold: Also choose the decision threshold that maximizes F1 on the
            validation rows, and report test measures at it.
        allow_unsafe: Run settings that weaken the privacy guarantee, and flag them in the
            report, instead of refusing them.
        secure_noise: Draw the noise, and the choice of rows for every private step, from
            the operating system instead of from the run's seed. The run is then not
            repeatable. Use this for anything but experiments.
        outputs: ``"probability"`` if the model returns probabilities, ``"logit"`` if it
            returns raw scores. By default this is read from the model's
            ``probability_output`` attribute (probabilities if it has none).
        dropouts: For ``DistributedDPSGD`` only: which participants are missing in which
            round, as a mapping from the round's number (counted from 0) to their names.
            Such a round fails: the secure sum cannot be opened, the model does not move,
            and everyone who sent a message is still charged.
        declared_rows: The public number of training rows of each participant, by name,
            where it differs from the number of rows actually given. Every place that
            would use a participant's size (the noise level, the chance of a row joining a
            step, the averaging weights) uses this number instead, so that leaving a row
            out changes nothing an observer could see directly. It must not be smaller
            than the real number. Meant for audits that leave some rows out by coin flip.
    """

    model_fn: Callable[[], nn.Module]
    clients: Sequence[ClientData]
    plan: Plan
    policy: PrivacyPolicy | None
    rounds: int
    local_epochs: float = 1.0
    batch_size: int = 32
    learning_rate: float = 0.05
    optimizer: str = "sgd"
    weighted_average: bool = True
    evaluate_every: int = 1
    choose_threshold: bool = False
    allow_unsafe: bool = False
    secure_noise: bool = False
    outputs: str | None = None
    dropouts: Mapping[int, Collection[str]] | None = None
    declared_rows: Mapping[str, int] | None = None
    _flags: UnsafeFlags = field(init=False, repr=False)
    _missing: dict[int, frozenset[str]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.clients = tuple(self.clients)
        if len(self.clients) == 0:
            raise ValueError("need at least one participant")
        names = [client.name for client in self.clients]
        if len(set(names)) != len(names):
            raise ValueError("participant names must be distinct")
        if len({client.train_x.shape[1] for client in self.clients}) != 1:
            raise ValueError("all participants must have the same columns")
        if self.rounds < 1:
            raise ValueError("rounds must be at least 1")
        if not self.local_epochs > 0:
            raise ValueError("local_epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not self.learning_rate > 0:
            raise ValueError("learning_rate must be positive")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
        if self.evaluate_every < 0:
            raise ValueError("evaluate_every must not be negative")
        if self.outputs not in (None, "probability", "logit"):
            raise ValueError("outputs must be 'probability' or 'logit'")
        if self.plan.dp is not None and self.policy is None:
            raise ValueError("a private plan needs a PrivacyPolicy with the total budget")
        if self.plan.dp is None and self.policy is not None:
            raise ValueError(
                "a privacy budget was given but the plan adds no noise; use Plan(dp=LocalDPSGD()) "
                "or pass policy=None for an unprotected baseline"
            )
        if self.declared_rows is not None:
            if isinstance(self.plan.imbalance, UnsafeOversampling):
                raise ValueError("declared_rows cannot be combined with oversampling")
            if set(self.declared_rows) != set(names):
                raise ValueError("declared_rows must name every participant exactly once")
            for client in self.clients:
                declared = self.declared_rows[client.name]
                if isinstance(declared, bool) or not isinstance(declared, int):
                    raise ValueError("declared_rows must hold whole numbers")
                if declared < client.n_train:
                    raise ValueError(
                        f"{client.name} declares {declared} rows but has {client.n_train}; "
                        "a declared size must not be smaller than the real one"
                    )
            self.declared_rows = dict(self.declared_rows)
        whole = isinstance(self.plan.dp, ClientLevelDP)
        if self.policy is not None and whole != (self.policy.unit == "client"):
            raise ValueError(
                "a budget for whole participants (PrivacyPolicy(unit='client')) goes with "
                "ClientLevelDP, and a budget for single rows (unit='record') with the other "
                "plans"
            )
        if whole and self.policy is not None and self.policy.per_scope_epsilon:
            raise ValueError(
                "a budget for whole participants is one budget for the group; "
                "per_scope_epsilon does not apply"
            )
        if whole and len(self.clients) < 2:
            raise ValueError("protecting whole participants needs at least two of them")
        if self.plan.secure_sum is not None and len(self.clients) < 2:
            raise ValueError("a secure sum needs at least two participants")
        if self.plan.single_step:
            if self.local_epochs != 1.0:
                raise ValueError(
                    "this plan takes one step per round at the shared model; leave "
                    "local_epochs at 1 and use more rounds"
                )
            if not self.weighted_average:
                raise ValueError(
                    "this plan draws every row of every participant with the same "
                    "probability, which is the weighted average; weighted_average=False is "
                    "not available"
                )
        if isinstance(self.plan.dp, ClientLevelDP):
            if self.plan.secure_sum is not None:
                self.plan.dp.honest_shares(len(self.clients))
            if len(self.clients) < SMALL_FEDERATION:
                warnings.warn(
                    f"protecting whole participants with only {len(self.clients)} of them: "
                    "the noise has to hide one participant among the others, which leaves "
                    f"little to learn from below about {SMALL_FEDERATION} participants",
                    SmallFederationWarning,
                    stacklevel=3,
                )
        if isinstance(self.plan.dp, DistributedDPSGD):
            self.plan.dp.honest_shares(len(self.clients))
            rows = sum(self._public_size(client) for client in self.clients)
            if self.plan.dp.noise == "skellam" and self.batch_size < rows:
                raise ValueError(
                    "whole-number noise is counted for rounds in which every row takes part; "
                    f"set batch_size to at least the number of training rows ({rows}), or "
                    "use noise='gaussian'"
                )
        self._missing = self._check_dropouts(names)
        self._flags = UnsafeFlags(self.allow_unsafe)
        self._check_unsafe()

    # ------------------------------------------------------------------ safety checks

    @property
    def private(self) -> bool:
        """True if the run adds privacy noise."""
        return self.plan.dp is not None

    def _check_dropouts(self, names: Sequence[str]) -> dict[int, frozenset[str]]:
        if self.dropouts is None:
            return {}
        shared = isinstance(self.plan.dp, DistributedDPSGD) or (
            isinstance(self.plan.dp, ClientLevelDP) and self.plan.secure_sum is not None
        )
        if not shared:
            raise ValueError(
                "dropouts can only be played where the participants share the noise "
                "(DistributedDPSGD, or ClientLevelDP with a secure sum)"
            )
        missing: dict[int, frozenset[str]] = {}
        for attempt, absent in self.dropouts.items():
            if isinstance(attempt, bool) or not isinstance(attempt, int):
                raise ValueError("dropouts must map round numbers to participant names")
            if not 0 <= attempt < self.rounds:
                raise ValueError(f"dropouts names round {attempt}, outside 0 to {self.rounds - 1}")
            if isinstance(absent, str):
                raise ValueError("dropouts must map each round to a collection of names")
            unknown = sorted(set(absent) - set(names))
            if unknown:
                raise ValueError(f"dropouts names unknown participants: {unknown}")
            if absent:
                missing[attempt] = frozenset(absent)
        return missing

    def _check_unsafe(self) -> None:
        if self.policy is None:
            return
        if isinstance(self.plan.dp, CentralDPSGD) or (
            isinstance(self.plan.dp, ClientLevelDP) and self.plan.secure_sum is None
        ):
            self._flags.raise_or_record("trusted_server")
        code = self.plan.imbalance.unsafe_code
        if code is not None:
            self._flags.raise_or_record(code)
        if isinstance(self.plan.dp, ClientLevelDP):
            # The unit is a participant: delta has to be small against their number.
            if self.policy.delta_is_weak_for(len(self.clients)):
                self._flags.raise_or_record(
                    "weak_delta",
                    f"delta = {self.policy.delta:g} with {len(self.clients)} participants",
                )
            return
        weak = [c.name for c in self.clients if self.policy.delta_is_weak_for(self._public_size(c))]
        if weak:
            self._flags.raise_or_record(
                "weak_delta", f"delta = {self.policy.delta:g}; affected: {', '.join(weak)}"
            )

    # ------------------------------------------------------------------ planning

    def sampling_rate(self, n_train: int) -> float:
        """The probability with which each of a participant's rows joins each step.

        In the one-step modes ``n_train`` is the number of rows of all participants.
        """
        return min(1.0, self.batch_size / n_train)

    def calibrations(self) -> dict[str, Calibration | SkellamCalibration]:
        """The noise each participant needs so the whole run fits its budget.

        Call this before ``run`` to see what a budget costs. Empty for an unprotected plan.
        """
        return self._calibrate(self._training_sizes())

    def _integer_step(self) -> IntegerStep:
        """The whole-number grid for this model: it depends on the model's size only."""
        model = self.model_fn()
        return plan_integer_step(
            sum(int(value.numel()) for value in model.parameters() if value.requires_grad)
        )

    def _calibrate(self, sizes: Sequence[int]) -> dict[str, Calibration | SkellamCalibration]:
        if self.policy is None:
            return {}
        if isinstance(self.plan.dp, ClientLevelDP):
            # One release per round for the whole group, at the participants' sampling rate.
            return {
                GROUP_SCOPE: calibrate_noise_multiplier(
                    self.policy.epsilon,
                    self.policy.delta,
                    self.plan.dp.sampling_rate,
                    self.rounds,
                    accountant=self.policy.accountant,
                )
            }
        if self.plan.single_step:
            # One release per round, one noise level for everybody: the strictest budget
            # among the participants decides.
            strictest = min(self.policy.epsilon_for(client.name) for client in self.clients)
            rate = self.sampling_rate(sum(sizes))
            shared: Calibration | SkellamCalibration
            if isinstance(self.plan.dp, DistributedDPSGD) and self.plan.dp.noise == "skellam":
                if rate < 1.0:
                    raise ValueError(
                        "whole-number noise is counted for rounds in which every row takes "
                        f"part; set batch_size to at least {sum(sizes)}"
                    )
                step = self._integer_step()
                shared = calibrate_skellam(
                    strictest,
                    self.policy.delta,
                    self.rounds,
                    step.l2_sensitivity,
                    step.l1_sensitivity,
                )
            else:
                shared = calibrate_noise_multiplier(
                    strictest,
                    self.policy.delta,
                    rate,
                    self.rounds,
                    accountant=self.policy.accountant,
                )
            return {client.name: shared for client in self.clients}
        result: dict[str, Calibration | SkellamCalibration] = {}
        for client, rows in zip(self.clients, sizes, strict=True):
            rate = self.sampling_rate(rows)
            result[client.name] = calibrate_noise_multiplier(
                self.policy.epsilon_for(client.name),
                self.policy.delta,
                rate,
                self.rounds * steps_per_round(self.local_epochs, rate),
                accountant=self.policy.accountant,
            )
        return result

    def _public_size(self, client: ClientData) -> int:
        """The number of training rows a participant is taken to have."""
        if self.declared_rows is None:
            return client.n_train
        return self.declared_rows[client.name]

    def _training_sizes(self) -> list[int]:
        """Training rows per participant. Oversampling changes them, so it is applied first."""
        if not isinstance(self.plan.imbalance, UnsafeOversampling):
            return [self._public_size(client) for client in self.clients]
        return [int(x.shape[0]) for x, _ in self._training_tables(seed=0)]

    def _training_tables(self, seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        tables = []
        for index, client in enumerate(self.clients):
            x, y = client.train_x, client.train_y
            if isinstance(self.plan.imbalance, UnsafeOversampling):
                x, y = self.plan.imbalance.resample(x, y, derive_seed(seed, "data", index))
            tables.append(
                (
                    torch.as_tensor(np.asarray(x, dtype=np.float32)),
                    torch.as_tensor(np.asarray(y, dtype=np.float32)),
                )
            )
        return tables

    def _dp_config(self) -> dict[str, Any] | None:
        dp = self.plan.dp
        if dp is None:
            return None
        if isinstance(dp, LocalDPSGD):
            return {"clip_norm": dp.clip_norm}
        settings: dict[str, Any] = {"kind": type(dp).__name__, "clip_norm": dp.clip_norm}
        if isinstance(dp, DistributedDPSGD):
            settings.update(
                max_colluding=dp.max_colluding,
                max_dropouts=dp.max_dropouts,
                noise=dp.noise,
                sum_bound=dp.sum_bound,
            )
            settings["dropouts"] = {
                str(attempt): sorted(absent) for attempt, absent in sorted(self._missing.items())
            }
        if isinstance(dp, ClientLevelDP):
            settings.update(
                sampling_rate=dp.sampling_rate,
                max_colluding=dp.max_colluding,
                max_dropouts=dp.max_dropouts,
            )
            settings["dropouts"] = {
                str(attempt): sorted(absent) for attempt, absent in sorted(self._missing.items())
            }
        return settings

    def config(self) -> dict[str, Any]:
        """The settings of this simulation, for result records."""
        return {
            "rounds": self.rounds,
            "local_epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "weighted_average": self.weighted_average,
            "choose_threshold": self.choose_threshold,
            "secure_noise": self.secure_noise,
            "allow_unsafe": self.allow_unsafe,
            "plan": {
                "dp": self._dp_config(),
                "imbalance": type(self.plan.imbalance).__name__,
                "positive_prior": getattr(self.plan.imbalance, "positive_prior", None),
                "proximal_mu": self.plan.proximal_mu,
                "secure_sum": None
                if self.plan.secure_sum is None
                else {
                    "kind": type(self.plan.secure_sum).__name__,
                    "value_bits": self.plan.secure_sum.value_bits,
                    "value_range": self.plan.secure_sum.value_range,
                },
            },
            "policy": None if self.policy is None else self.policy.to_dict(),
            "clients": {client.name: client.n_train for client in self.clients},
            # Only present when it was set, so that earlier records keep their fingerprint.
            **({} if self.declared_rows is None else {"declared_rows": dict(self.declared_rows)}),
        }

    # ------------------------------------------------------------------ running

    def run(self, seed: int, observer: RoundObserver | None = None) -> SimulationResult:
        """Run all rounds and return the final model, its scores and the privacy report.

        Args:
            seed: The run's seed.
            observer: Optional function called once per round with a ``RoundRecord`` of
                everything that was sent in that round. It is meant for audits, which play
                an attacker; it must not change anything it is given. The run itself is
                the same with and without it.

        Raises:
            BudgetExceeded: If the run cannot fit the policy's budget. Nothing is trained.
            ValueError: If no noise level can meet the budget.
        """
        started = time.perf_counter()
        torch.manual_seed(derive_seed(seed, "init"))
        shared = self.model_fn()
        worker = self.model_fn()
        from_logits = self._from_logits(shared)
        loss_fn = self.plan.imbalance.loss(from_logits=from_logits)
        tables = self._training_tables(seed)
        sizes = [int(x.shape[0]) for x, _ in tables]
        if self.declared_rows is not None:
            sizes = [self.declared_rows[client.name] for client in self.clients]

        ledger: PrivacyLedger | None = None
        calibrations: dict[str, Calibration | SkellamCalibration] = {}
        if self.policy is not None:
            ledger = PrivacyLedger(self.policy)
            calibrations = self._calibrate(sizes)
            # Reserve the whole run first. A run that cannot fit never touches the data.
            for scope, calibration in calibrations.items():
                ledger.plan(scope, [calibration.event])

        history: list[dict[str, Any]] = []
        state = get_state(shared)
        names = [client.name for client in self.clients]
        length = state_size(state)
        communication: dict[str, float] = {
            "upload_bytes_per_round": float(float32_size(length)),
            "plain_bytes_per_round": float(float32_size(length)),
            "ratio": 1.0,
            "bits_per_value": 32.0,
            "setup_bytes": 0.0,
            "saturated_share": 0.0,
        }
        maskers: dict[str, MaskingClient] = {}
        aggregator: MaskingAggregator | None = None
        mode: ModeReport | None = None
        one_step = self.plan.single_step or isinstance(self.plan.dp, ClientLevelDP)
        if self.plan.single_step:
            assert ledger is not None
            state, mode = self._single_step_rounds(
                shared,
                tables,
                sizes,
                loss_fn,
                ledger,
                calibrations,
                seed,
                history,
                communication,
                observer,
            )
        elif isinstance(self.plan.dp, ClientLevelDP):
            assert ledger is not None
            state, mode = self._participant_rounds(
                shared,
                worker,
                tables,
                sizes,
                loss_fn,
                ledger,
                calibrations[GROUP_SCOPE],
                seed,
                history,
                communication,
                observer,
            )
        elif self.plan.secure_sum is not None:
            maskers, aggregator = self.plan.secure_sum.build(names)
            upload = self.plan.secure_sum.message_bytes(length, len(names))
            communication.update(
                upload_bytes_per_round=float(upload),
                ratio=upload / float32_size(length),
                bits_per_value=float(self.plan.secure_sum.modulus_bits(len(names))),
                setup_bytes=float(self.plan.secure_sum.setup_bytes(len(names))),
            )
        for round_index in range(0 if one_step else self.rounds):
            updates = []
            for index, client in enumerate(self.clients):
                set_state(worker, state)
                torch.manual_seed(derive_seed(seed, "train", round_index, index))
                self._train_one(
                    worker,
                    tables[index],
                    client.name,
                    loss_fn,
                    state,
                    ledger,
                    calibrations.get(client.name),
                    seed,
                    round_index,
                    index,
                    None if self.declared_rows is None else sizes[index],
                )
                updates.append(get_state(worker))
            start = state
            if aggregator is None:
                state = aggregate_states(updates, sizes, weighted=self.weighted_average)
            else:
                state, saturated = self._secure_average(
                    state, updates, sizes, maskers, aggregator, seed, round_index
                )
                communication["saturated_share"] = max(communication["saturated_share"], saturated)
            if observer is not None:
                origin = flatten_state(start)
                factors = (
                    np.ones(len(names))
                    if aggregator is None
                    else fedavg_weights(sizes, weighted=self.weighted_average)
                )
                observer(
                    RoundRecord(
                        round=round_index,
                        kind=RECORD_MODELS,
                        start=start,
                        messages={
                            name: factor * (flatten_state(update) - origin)
                            for name, factor, update in zip(names, factors, updates, strict=True)
                        },
                        masked=aggregator is not None,
                        total=flatten_state(state) - origin,
                    )
                )
            last = round_index == self.rounds - 1
            due = self.evaluate_every > 0 and (round_index + 1) % self.evaluate_every == 0
            if (due or last) and self._has_rows("validation"):
                set_state(shared, state)
                scores = self._score(shared, "validation", from_logits)
                history.append({"round": round_index + 1, **scores[0]})

        set_state(shared, state)
        validation: dict[str, float] = {}
        threshold = 0.5
        if self._has_rows("validation"):
            validation, probabilities, labels = self._score(shared, "validation", from_logits)
            if self.choose_threshold:
                threshold = best_threshold(probabilities, labels)
        test: dict[str, float] = {}
        test_at_chosen: dict[str, float] | None = None
        if self._has_rows("test"):
            test, probabilities, labels = self._score(shared, "test", from_logits)
            if self.choose_threshold:
                test_at_chosen = binary_metrics(probabilities, labels, threshold=threshold)

        report = None
        if ledger is not None and self.plan.dp is not None:
            report = PrivacyReport.from_ledger(
                ledger,
                protection=self._protection(),
                unsafe_flags=self._flags.messages,
                caveats=self._caveats(),
                mode=mode,
            )
        return SimulationResult(
            report=report,
            history=tuple(history),
            validation=validation,
            test=test,
            test_at_chosen_threshold=test_at_chosen,
            threshold=threshold,
            noise_multipliers={name: c.noise_multiplier for name, c in calibrations.items()},
            communication=communication,
            state=state,
            seed=seed,
            config=self.config(),
            seconds=time.perf_counter() - started,
        )

    def _train_one(
        self,
        model: nn.Module,
        table: tuple[torch.Tensor, torch.Tensor],
        name: str,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        reference: State,
        ledger: PrivacyLedger | None,
        calibration: Calibration | SkellamCalibration | None,
        seed: int,
        round_index: int,
        index: int,
        declared: int | None = None,
    ) -> None:
        features, targets = table
        optimizer: torch.optim.Optimizer
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        mu = self.plan.proximal_mu
        if ledger is None or calibration is None or self.plan.dp is None:
            train_plain_round(
                model,
                features,
                targets,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=self.local_epochs,
                batch_size=self.batch_size,
                generator=torch_generator(seed, "sampling", round_index, index),
                proximal_mu=mu,
                reference=reference if mu > 0 else None,
            )
            return
        noise = self._torch_stream(seed, "noise", round_index, index)
        train_private_round(
            model,
            features,
            targets,
            loss_fn=loss_fn,
            optimizer=optimizer,
            ledger=ledger,
            scope=name,
            sampling_rate=calibration.sampling_rate,
            steps=calibration.steps // self.rounds,
            clip_norm=self.plan.dp.clip_norm,
            noise_multiplier=calibration.noise_multiplier,
            # The budget counts on the choice of rows being secret, like the noise.
            sampling_generator=self._torch_stream(seed, "sampling", round_index, index),
            noise_generator=noise,
            declared_records=declared,
            clip_scale=self.plan.imbalance.clip_scale,
            proximal_mu=mu,
            reference=reference if mu > 0 else None,
        )

    def _torch_stream(self, seed: int, purpose: str, *path: int) -> torch.Generator:
        """A random stream for something the privacy guarantee relies on being secret."""
        if self.secure_noise:
            return entropy_generator()
        return torch_generator(seed, purpose, *path)

    def _single_step_rounds(
        self,
        shared: nn.Module,
        tables: Sequence[tuple[torch.Tensor, torch.Tensor]],
        sizes: Sequence[int],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        ledger: PrivacyLedger,
        calibrations: Mapping[str, Calibration | SkellamCalibration],
        seed: int,
        history: list[dict[str, Any]],
        communication: dict[str, float],
        observer: RoundObserver | None = None,
    ) -> tuple[State, ModeReport]:
        """The rounds of ``DistributedDPSGD`` and ``CentralDPSGD``: one step each.

        Every participant computes one clipped gradient sum at the shared model. In the
        distributed mode it adds its slice of the noise and the sums go through the secure
        sum; in the central mode the server adds the noise to the clean total. The server
        divides by the expected batch size and takes one optimizer step.
        """
        dp = self.plan.dp
        assert isinstance(dp, (DistributedDPSGD, CentralDPSGD))
        names = [client.name for client in self.clients]
        calibration = calibrations[names[0]]
        rate, sigma = calibration.sampling_rate, calibration.noise_multiplier
        divisor = rate * sum(sizes)
        check_per_record_model(shared)
        trainable = [value for value in shared.parameters() if value.requires_grad]
        length = sum(int(value.numel()) for value in trainable)
        optimizer = self._make_optimizer(shared)
        from_logits = self._from_logits(shared)
        event = SubsampledGaussianEvent(sigma, rate, 1)

        grid: SumGrid | None = None
        maskers: dict[str, MaskingClient] = {}
        aggregator: MaskingAggregator | None = None
        share = bound = 0.0
        plain = float(float32_size(length))
        communication.update(upload_bytes_per_round=plain, plain_bytes_per_round=plain)
        integer: IntegerStep | None = None
        share_variance = 0.0
        integer_bound = 0
        if isinstance(dp, DistributedDPSGD):
            bound = (
                dp.sum_bound
                if dp.sum_bound is not None
                else default_sum_bound(dp.clip_norm, rate, max(sizes))
            )
            if isinstance(calibration, SkellamCalibration):
                # Whole numbers throughout: one grid point is clip_norm / scale.
                integer = plan_integer_step(length)
                unit = dp.clip_norm / integer.scale
                share_variance = calibration.variance / dp.honest_shares(len(names))
                integer_bound = (
                    math.ceil(max(sizes) * integer.l2_sensitivity)
                    if dp.sum_bound is None
                    else max(1, math.ceil(dp.sum_bound / unit))
                )
                grid = plan_integer_grid(unit, integer_bound, math.sqrt(share_variance), len(names))
            else:
                share = dp.clip_norm * dp.share_noise_multiplier(sigma, len(names))
                grid = plan_sum_grid(share, bound, len(names))
            maskers, aggregator = setup_pairwise_masking(names, grid.modulus_bits)
            communication.update(
                upload_bytes_per_round=float(grid.message_bytes(length)),
                ratio=grid.message_bytes(length) / plain,
                bits_per_value=float(grid.modulus_bits),
                setup_bytes=float(PUBLIC_KEY_BYTES * len(names)),
            )

        completed = 0
        for attempt in range(self.rounds):
            total: np.ndarray[Any, Any] | None = None
            sent: dict[str, np.ndarray[Any, Any]] = {}
            start = get_state(shared) if observer is not None else None
            if grid is not None and aggregator is not None:
                absent = self._missing.get(attempt, frozenset())
                messages: dict[str, bytes] = {}
                for index, client in enumerate(self.clients):
                    if client.name in absent:
                        continue
                    torch.manual_seed(derive_seed(seed, "train", attempt, index))
                    rounding = self._numpy_stream(seed, "quantize", attempt, index)
                    local: dict[str, float] = {}
                    if integer is not None and isinstance(calibration, SkellamCalibration):
                        whole = integer_sum_step(
                            shared,
                            *tables[index],
                            loss_fn=loss_fn,
                            ledger=ledger,
                            scope=client.name,
                            clip_norm=dp.clip_norm,
                            step=integer,
                            variance=calibration.variance,
                            share_variance=share_variance,
                            rounding_generator=rounding,
                            noise_generator=self._numpy_stream(seed, "noise", attempt, index),
                            clip_scale=self.plan.imbalance.clip_scale,
                            sum_bound=integer_bound,
                            diagnostics=local,
                        )
                        vector = whole * grid.step
                        encoded = grid.wrap(whole)
                    else:
                        vector = private_sum_step(
                            shared,
                            *tables[index],
                            loss_fn=loss_fn,
                            ledger=ledger,
                            scope=client.name,
                            sampling_rate=rate,
                            clip_norm=dp.clip_norm,
                            noise_multiplier=sigma,
                            share_std=share,
                            sampling_generator=self._torch_stream(seed, "sampling", attempt, index),
                            noise_generator=self._torch_stream(seed, "noise", attempt, index),
                            clip_scale=self.plan.imbalance.clip_scale,
                            sum_bound=bound,
                            diagnostics=local,
                        ).numpy()
                        encoded = grid.encode(vector, rounding)
                    communication["saturated_share"] = max(
                        communication["saturated_share"],
                        grid.outside_share(vector),
                        local.get("limited_share", 0.0),
                    )
                    if observer is not None:
                        # The message as the sum will count it: on the grid, before masking.
                        sent[client.name] = grid.decode_sum(encoded)
                    # Every attempt has its own sum index: masks are never used twice.
                    messages[client.name] = maskers[client.name].protect(attempt, encoded)
                try:
                    total = grid.decode_sum(aggregator.combine(attempt, messages, length))
                except MissingParticipants:
                    # The sum stays closed and the model does not move. Those who sent a
                    # message stay charged: if the missing participant works with the
                    # server, the two can strip the masks from the others' messages.
                    total = None
            else:
                sums = []
                for index, client in enumerate(self.clients):
                    torch.manual_seed(derive_seed(seed, "train", attempt, index))
                    ledger.charge(client.name, event)
                    sums.append(
                        clipped_sum_step(
                            shared,
                            *tables[index],
                            loss_fn=loss_fn,
                            sampling_rate=rate,
                            clip_norm=dp.clip_norm,
                            sampling_generator=self._torch_stream(seed, "sampling", attempt, index),
                            clip_scale=self.plan.imbalance.clip_scale,
                        )
                    )
                noise = gaussian_noise(
                    (length,),
                    sigma * dp.clip_norm,
                    self._torch_stream(seed, "noise", attempt, len(names)),
                    dtype=torch.float64,
                )
                total = (torch.stack(sums).sum(dim=0) + noise).numpy()
                if observer is not None:
                    sent = {
                        name: clean.numpy().copy() for name, clean in zip(names, sums, strict=True)
                    }

            if observer is not None and start is not None:
                observer(
                    RoundRecord(
                        round=attempt,
                        kind=RECORD_SUMS,
                        start=start,
                        messages=sent,
                        masked=grid is not None,
                        total=None if total is None else total.copy(),
                        divisor=divisor,
                    )
                )
            if total is not None:
                self._apply_step(trainable, optimizer, total / divisor)
                completed += 1
            last = attempt == self.rounds - 1
            due = (
                total is not None
                and self.evaluate_every > 0
                and completed % self.evaluate_every == 0
            )
            scored = bool(history) and history[-1]["round"] == completed
            if (due or last) and not scored and self._has_rows("validation"):
                scores = self._score(shared, "validation", from_logits)
                history.append({"round": completed, **scores[0]})

        if isinstance(dp, DistributedDPSGD):
            assert grid is not None
            mode = ModeReport(
                mode=MODE_DISTRIBUTED,
                participants=len(names),
                attempts=self.rounds,
                completed_rounds=completed,
                max_colluding=dp.max_colluding,
                max_dropouts=dp.max_dropouts,
                steps_per_share=None if integer is not None else share / grid.step,
                noise=dp.noise,
                table=collusion_table(
                    {scope: ledger.events(scope) for scope in ledger.scopes},
                    ledger.policy.delta,
                    len(names),
                    dp.max_colluding,
                    dp.max_dropouts,
                ),
            )
        else:
            mode = ModeReport(
                mode=MODE_CENTRAL,
                participants=len(names),
                attempts=self.rounds,
                completed_rounds=completed,
            )
        return get_state(shared), mode

    def _participant_rounds(
        self,
        shared: nn.Module,
        worker: nn.Module,
        tables: Sequence[tuple[torch.Tensor, torch.Tensor]],
        sizes: Sequence[int],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        ledger: PrivacyLedger,
        calibration: Calibration | SkellamCalibration,
        seed: int,
        history: list[dict[str, Any]],
        communication: dict[str, float],
        observer: RoundObserver | None = None,
    ) -> tuple[State, ModeReport]:
        """The rounds of ``ClientLevelDP``: the budget protects whole participants.

        Each participant takes part with its own secret coin, trains without noise and
        shortens its change to the shared model. Noise sized for one whole participant is
        added to the sum of the changes: in slices through the secure sum, or by the server.
        """
        dp = self.plan.dp
        assert isinstance(dp, ClientLevelDP)
        names = [client.name for client in self.clients]
        rate, sigma = dp.sampling_rate, calibration.noise_multiplier
        # Public weights: a participant counts by its share of the largest declared table.
        weights = [size / max(sizes) if self.weighted_average else 1.0 for size in sizes]
        divisor = rate * sum(weights)
        trainable = [value for value in shared.parameters() if value.requires_grad]
        length = sum(int(value.numel()) for value in trainable)
        from_logits = self._from_logits(shared)
        event = SubsampledGaussianEvent(sigma, rate, 1)

        grid: SumGrid | None = None
        maskers: dict[str, MaskingClient] = {}
        aggregator: MaskingAggregator | None = None
        share = 0.0
        plain = float(float32_size(length))
        communication.update(upload_bytes_per_round=plain, plain_bytes_per_round=plain)
        if self.plan.secure_sum is not None:
            share = dp.clip_norm * dp.share_noise_multiplier(sigma, len(names))
            # No number of a shortened change can be larger than the clipping norm.
            grid = plan_sum_grid(share, dp.clip_norm, len(names))
            maskers, aggregator = setup_pairwise_masking(names, grid.modulus_bits)
            communication.update(
                upload_bytes_per_round=float(grid.message_bytes(length)),
                ratio=grid.message_bytes(length) / plain,
                bits_per_value=float(grid.modulus_bits),
                setup_bytes=float(PUBLIC_KEY_BYTES * len(names)),
            )

        completed = 0
        for attempt in range(self.rounds):
            absent = self._missing.get(attempt, frozenset())
            # One release for the group, charged before any participant does anything.
            ledger.charge(GROUP_SCOPE, event)
            state = get_state(shared)
            start = [value.detach().clone() for value in trainable]
            total = torch.zeros(length, dtype=torch.float64)
            messages: dict[str, bytes] = {}
            sent: dict[str, np.ndarray[Any, Any]] = {}
            for index, client in enumerate(self.clients):
                if client.name in absent:
                    continue
                coin = self._torch_stream(seed, "sampling", attempt, index)
                takes_part = bool(torch.rand(1, generator=coin, dtype=torch.float64) < rate)
                change = torch.zeros(length, dtype=torch.float64)
                if takes_part:
                    set_state(worker, state)
                    torch.manual_seed(derive_seed(seed, "train", attempt, index))
                    mu = self.plan.proximal_mu
                    train_plain_round(
                        worker,
                        *tables[index],
                        loss_fn=loss_fn,
                        optimizer=self._make_optimizer(worker),
                        epochs=self.local_epochs,
                        batch_size=self.batch_size,
                        generator=coin,
                        proximal_mu=mu,
                        reference=state if mu > 0 else None,
                    )
                    change = clipped_model_change(worker, start, dp.clip_norm, weights[index])
                if grid is None:
                    total += change
                    if observer is not None:
                        sent[client.name] = change.numpy().copy()
                    continue
                noise = gaussian_noise(
                    (length,),
                    share,
                    self._torch_stream(seed, "noise", attempt, index),
                    dtype=torch.float64,
                )
                vector = (change + noise).numpy()
                communication["saturated_share"] = max(
                    communication["saturated_share"], grid.outside_share(vector)
                )
                encoded = grid.encode(vector, self._numpy_stream(seed, "quantize", attempt, index))
                if observer is not None:
                    sent[client.name] = grid.decode_sum(encoded)
                messages[client.name] = maskers[client.name].protect(attempt, encoded)
            opened: np.ndarray[Any, Any] | None
            if grid is not None and aggregator is not None:
                try:
                    opened = grid.decode_sum(aggregator.combine(attempt, messages, length))
                except MissingParticipants:
                    opened = None
            else:
                server_noise = gaussian_noise(
                    (length,),
                    sigma * dp.clip_norm,
                    self._torch_stream(seed, "noise", attempt, len(names)),
                    dtype=torch.float64,
                )
                opened = (total + server_noise).numpy()
            if observer is not None:
                observer(
                    RoundRecord(
                        round=attempt,
                        kind=RECORD_CHANGES,
                        start=state,
                        messages=sent,
                        masked=grid is not None,
                        total=None if opened is None else opened.copy(),
                        divisor=divisor,
                    )
                )
            if opened is not None:
                offset = 0
                with torch.no_grad():
                    for value in trainable:
                        size = int(value.numel())
                        piece = torch.as_tensor(
                            opened[offset : offset + size] / divisor, dtype=value.dtype
                        )
                        value.add_(piece.view_as(value).to(value.device))
                        offset += size
                completed += 1
            last = attempt == self.rounds - 1
            due = (
                opened is not None
                and self.evaluate_every > 0
                and completed % self.evaluate_every == 0
            )
            scored = bool(history) and history[-1]["round"] == completed
            if (due or last) and not scored and self._has_rows("validation"):
                scores = self._score(shared, "validation", from_logits)
                history.append({"round": completed, **scores[0]})

        shared_noise = grid is not None
        mode = ModeReport(
            mode=MODE_PARTICIPANT,
            participants=len(names),
            attempts=self.rounds,
            completed_rounds=completed,
            sampling_rate=rate,
            max_colluding=dp.max_colluding if shared_noise else None,
            max_dropouts=dp.max_dropouts if shared_noise else None,
            steps_per_share=share / grid.step if grid is not None else None,
            noise="gaussian" if shared_noise else None,
            table=collusion_table(
                {GROUP_SCOPE: ledger.events(GROUP_SCOPE)},
                ledger.policy.delta,
                len(names),
                dp.max_colluding,
                dp.max_dropouts,
            )
            if shared_noise and dp.max_colluding is not None
            else (),
        )
        return get_state(shared), mode

    def _numpy_stream(self, seed: int, purpose: str, *path: int) -> np.random.Generator:
        """Like ``_torch_stream``, for the parts that work on whole numbers."""
        if self.secure_noise:
            return np.random.default_rng()
        return numpy_generator(seed, purpose, *path)

    def _make_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        if self.optimizer == "adam":
            return torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        return torch.optim.SGD(model.parameters(), lr=self.learning_rate)

    @staticmethod
    def _apply_step(
        trainable: Sequence[torch.Tensor],
        optimizer: torch.optim.Optimizer,
        gradient: np.ndarray[Any, Any],
    ) -> None:
        """One optimizer step with ``gradient``, a flat vector over the trainable parameters."""
        optimizer.zero_grad(set_to_none=True)
        offset = 0
        with torch.no_grad():
            for value in trainable:
                size = int(value.numel())
                piece = torch.as_tensor(gradient[offset : offset + size], dtype=value.dtype)
                value.grad = piece.view_as(value).to(value.device)
                offset += size
        optimizer.step()

    def _secure_average(
        self,
        state: State,
        updates: Sequence[State],
        sizes: Sequence[int],
        maskers: dict[str, MaskingClient],
        aggregator: MaskingAggregator,
        seed: int,
        round_index: int,
    ) -> tuple[State, float]:
        """Average the updates through the secure sum: the server only ever sees the total.

        Each participant sends its change to the shared model, already multiplied by its
        averaging weight, rounded to whole numbers and masked. The sum of those messages is
        the weighted average change.
        """
        assert self.plan.secure_sum is not None
        quantizer = Quantizer(self.plan.secure_sum.value_bits, self.plan.secure_sum.value_range)
        weights = fedavg_weights(sizes, weighted=self.weighted_average)
        start = flatten_state(state)
        messages: dict[str, bytes] = {}
        saturated = 0.0
        for index, (client, update) in enumerate(zip(self.clients, updates, strict=True)):
            change = weights[index] * (flatten_state(update) - start)
            saturated = max(saturated, quantizer.saturated_share(change))
            rounding = numpy_generator(seed, "quantize", round_index, index)
            messages[client.name] = maskers[client.name].protect(
                round_index, quantizer.quantize(change, rounding)
            )
        total = aggregator.combine(round_index, messages, start.size)
        average_change = quantizer.dequantize_sum(total, len(self.clients))
        return unflatten_state(start + average_change, state), saturated

    def _protection(self) -> tuple[str, ...]:
        assert self.plan.dp is not None
        if isinstance(self.plan.dp, CentralDPSGD):
            return self.plan.dp.protection()
        if isinstance(self.plan.dp, DistributedDPSGD):
            return self.plan.dp.protection(len(self.clients))
        if isinstance(self.plan.dp, ClientLevelDP):
            return self.plan.dp.protection(
                len(self.clients), shared=self.plan.secure_sum is not None
            )
        if self.plan.secure_sum is None:
            return self.plan.dp.protection(updates_in_the_clear=True)
        return (
            *self.plan.dp.protection(updates_in_the_clear=False),
            *self.plan.secure_sum.protection(len(self.clients)),
        )

    # ------------------------------------------------------------------ scoring

    def _from_logits(self, model: nn.Module) -> bool:
        if self.outputs is not None:
            return self.outputs == "logit"
        return not bool(getattr(model, "probability_output", True))

    def _has_rows(self, part: str) -> bool:
        return any(getattr(client, f"{part}_x").shape[0] > 0 for client in self.clients)

    def _score(
        self, model: nn.Module, part: str, from_logits: bool
    ) -> tuple[dict[str, float], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        features = np.concatenate([getattr(client, f"{part}_x") for client in self.clients])
        labels = np.concatenate([getattr(client, f"{part}_y") for client in self.clients])
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                output = model(torch.as_tensor(features))
                if from_logits:
                    output = torch.sigmoid(output)
                probabilities = output.detach().cpu().numpy()
        finally:
            model.train(was_training)
        return binary_metrics(probabilities, labels), probabilities, labels

    def _caveats(self) -> tuple[str, ...]:
        caveats = [
            "Validation and test rows are not protected by this budget; only training rows are."
        ]
        if isinstance(self.plan.dp, ClientLevelDP):
            caveats = [
                "A participant's validation and test rows are not protected by this budget; "
                "only what it contributes to training is."
            ]
        if not self.secure_noise:
            caveats.append(
                "The noise was drawn from the run's seed so that the experiment can be repeated. "
                "Anyone who knows the seed can remove it. Real data needs secure_noise=True."
            )
        if isinstance(self.plan.dp, DistributedDPSGD) and self.plan.dp.noise == "skellam":
            caveats.append(
                "The whole-number noise is drawn with numpy's Poisson routine, which has not "
                "been certified as an exact sampler."
            )
        if isinstance(self.plan.dp, ClientLevelDP) and len(self.clients) < SMALL_FEDERATION:
            caveats.append(
                f"With {len(self.clients)} participants, hiding one whole participant takes "
                "noise of about the size of everything one participant contributes. Expect "
                f"little accuracy below about {SMALL_FEDERATION} participants."
            )
        if isinstance(self.plan.dp, DistributedDPSGD) or (
            isinstance(self.plan.dp, ClientLevelDP) and self.plan.secure_sum is not None
        ):
            caveats.append(
                "The participants' public keys are not checked. A server that swaps them could "
                "read single messages, and a single message carries only a slice of the noise: "
                "the stated budget would then not hold."
            )
        if self.choose_threshold:
            caveats.append("The decision threshold was chosen on validation rows.")
        return tuple(caveats)
