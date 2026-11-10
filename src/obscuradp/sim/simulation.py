"""Run a whole federation on one machine.

``Simulation`` plays every role itself: the participants, who train on their own rows, and
the server, which averages what they send. It is the tool for research and for trying a
configuration before a real deployment.

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

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from obscuradp.accounting.calibrate import Calibration, calibrate_noise_multiplier, steps_per_round
from obscuradp.aggregation.fedavg import aggregate_states
from obscuradp.client.trainer import train_plain_round, train_private_round
from obscuradp.imbalance import UnsafeOversampling
from obscuradp.ledger import PrivacyLedger
from obscuradp.mechanisms.noise import entropy_generator
from obscuradp.metrics import best_threshold, binary_metrics
from obscuradp.plan import Plan
from obscuradp.policy import PrivacyPolicy
from obscuradp.report import PrivacyReport
from obscuradp.results import make_result
from obscuradp.seeding import derive_seed, torch_generator
from obscuradp.state import State, get_state, set_state
from obscuradp.tables.split import ClientData
from obscuradp.unsafe import UnsafeFlags

OPTIMIZERS = ("sgd", "adam")


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
        rounds: Number of rounds $R$.
        local_epochs: Passes over a participant's rows per round, $E$. In a private run each
            participant takes $\\lceil E / q \\rceil$ steps per round.
        batch_size: Batch size $B$. In a private run it is the expected batch size: each row
            joins each step with probability $q = \\min(1, B / n)$.
        learning_rate: Step size of the local optimizer.
        optimizer: ``"sgd"`` or ``"adam"``. A fresh optimizer is made for every participant
            in every round.
        weighted_average: Average the participants' models by their number of training rows
            (true) or equally (false).
        evaluate_every: Score the shared model on validation rows every this many rounds
            (the last round is always scored). 0 scores only the final model.
        choose_threshold: Also choose the decision threshold that maximizes F1 on the
            validation rows, and report test measures at it.
        allow_unsafe: Run settings that weaken the privacy guarantee, and flag them in the
            report, instead of refusing them.
        secure_noise: Draw the noise from the operating system instead of from the run's
            seed. The run is then not repeatable. Use this for anything but experiments.
        outputs: ``"probability"`` if the model returns probabilities, ``"logit"`` if it
            returns raw scores. By default this is read from the model's
            ``probability_output`` attribute (probabilities if it has none).
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
    _flags: UnsafeFlags = field(init=False, repr=False)

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
        if self.policy is not None and self.policy.unit != "record":
            raise ValueError("only record-level privacy is implemented so far")
        self._flags = UnsafeFlags(self.allow_unsafe)
        self._check_unsafe()

    # ------------------------------------------------------------------ safety checks

    @property
    def private(self) -> bool:
        """True if the run adds privacy noise."""
        return self.plan.dp is not None

    def _check_unsafe(self) -> None:
        if self.policy is None:
            return
        code = self.plan.imbalance.unsafe_code
        if code is not None:
            self._flags.raise_or_record(code)
        weak = [c.name for c in self.clients if self.policy.delta_is_weak_for(c.n_train)]
        if weak:
            self._flags.raise_or_record(
                "weak_delta", f"delta = {self.policy.delta:g}; affected: {', '.join(weak)}"
            )

    # ------------------------------------------------------------------ planning

    def sampling_rate(self, n_train: int) -> float:
        """The probability with which each of a participant's rows joins each step."""
        return min(1.0, self.batch_size / n_train)

    def calibrations(self) -> dict[str, Calibration]:
        """The noise each participant needs so the whole run fits its budget.

        Call this before ``run`` to see what a budget costs. Empty for an unprotected plan.
        """
        return self._calibrate(self._training_sizes())

    def _calibrate(self, sizes: Sequence[int]) -> dict[str, Calibration]:
        if self.policy is None:
            return {}
        result = {}
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

    def _training_sizes(self) -> list[int]:
        """Training rows per participant. Oversampling changes them, so it is applied first."""
        if not isinstance(self.plan.imbalance, UnsafeOversampling):
            return [client.n_train for client in self.clients]
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
                "dp": None if self.plan.dp is None else {"clip_norm": self.plan.dp.clip_norm},
                "imbalance": type(self.plan.imbalance).__name__,
                "positive_prior": getattr(self.plan.imbalance, "positive_prior", None),
                "proximal_mu": self.plan.proximal_mu,
            },
            "policy": None if self.policy is None else self.policy.to_dict(),
            "clients": {client.name: client.n_train for client in self.clients},
        }

    # ------------------------------------------------------------------ running

    def run(self, seed: int) -> SimulationResult:
        """Run all rounds and return the final model, its scores and the privacy report.

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

        ledger: PrivacyLedger | None = None
        calibrations: dict[str, Calibration] = {}
        if self.policy is not None:
            ledger = PrivacyLedger(self.policy)
            calibrations = self._calibrate(sizes)
            # Reserve the whole run first. A run that cannot fit never touches the data.
            for client in self.clients:
                ledger.plan(client.name, [calibrations[client.name].event])

        history: list[dict[str, Any]] = []
        state = get_state(shared)
        for round_index in range(self.rounds):
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
                )
                updates.append(get_state(worker))
            state = aggregate_states(updates, sizes, weighted=self.weighted_average)
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
                protection=self.plan.dp.protection(),
                unsafe_flags=self._flags.messages,
                caveats=self._caveats(),
            )
        return SimulationResult(
            report=report,
            history=tuple(history),
            validation=validation,
            test=test,
            test_at_chosen_threshold=test_at_chosen,
            threshold=threshold,
            noise_multipliers={name: c.noise_multiplier for name, c in calibrations.items()},
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
        calibration: Calibration | None,
        seed: int,
        round_index: int,
        index: int,
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
        noise = (
            entropy_generator()
            if self.secure_noise
            else torch_generator(seed, "noise", round_index, index)
        )
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
            sampling_generator=torch_generator(seed, "sampling", round_index, index),
            noise_generator=noise,
            clip_scale=self.plan.imbalance.clip_scale,
            proximal_mu=mu,
            reference=reference if mu > 0 else None,
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
        if not self.secure_noise:
            caveats.append(
                "The noise was drawn from the run's seed so that the experiment can be repeated. "
                "Anyone who knows the seed can remove it. Real data needs secure_noise=True."
            )
        if self.choose_threshold:
            caveats.append("The decision threshold was chosen on validation rows.")
        return tuple(caveats)
