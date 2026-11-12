"""Attack single releases with the strongest attacker the guarantee allows.

Each check here builds the worst case for one way of adding the noise: a record (or a
whole participant) that is shortened to exactly the clipping norm, an attacker that sees
every release it is allowed to see and knows everything else, and two worlds, one with the
record and one without. The check is run many times in both worlds. What the attacker can
tell apart is a **lower limit** on the true privacy budget
(``obscuradp.audit.bounds``); a correct implementation never shows a limit above what its
ledger reports, and a sharp attack shows a limit close to it.

Unlike ``obscuradp.audit.membership`` these checks do not look at a real training run on
real rows. They look at the noise itself, where the best attack is known exactly, so they
can tell a mechanism that adds half the noise it reports from a correct one.

The ``*_scores`` functions return the attacker's score for every trial in one world and
the budget the ledger reports. ``compare_worlds`` turns the scores of both worlds into a
result. ``audit_local_step``, ``audit_shared_sum``, ``audit_whole_number_sum`` and
``audit_whole_participant`` do both.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from obscuradp.accounting.calibrate import calibrate_skellam
from obscuradp.audit.bounds import epsilon_lower_bound, gaussian_epsilon_lower_bound
from obscuradp.audit.views import server_view
from obscuradp.client.trainer import integer_sum_step, private_sum_step, train_private_round
from obscuradp.compression.grid import plan_integer_grid, plan_sum_grid
from obscuradp.crypto.pairwise_masking import PairwiseMasking
from obscuradp.ledger import PrivacyLedger
from obscuradp.mechanisms.dpsgd import clipped_sum, per_record_gradients
from obscuradp.mechanisms.shared_noise import default_sum_bound, share_std
from obscuradp.mechanisms.skellam import plan_integer_step
from obscuradp.models import TabularMLP
from obscuradp.plan import ClientLevelDP, Plan, SmallFederationWarning
from obscuradp.policy import PrivacyPolicy
from obscuradp.sim.simulation import RoundRecord, Simulation
from obscuradp.tables.split import ClientData, make_client_data

DELTA = 1e-5
# One full-batch release at this noise level costs a budget of 1 at delta = 1e-5.
UNIT_BUDGET_NOISE = 3.73

Scores = tuple[list[float], float]


class LinearProbe(nn.Module):
    """A model whose loss is linear in its weights.

    Every record's gradient is then the same at every step, whatever the weights, and
    all-zero rows have no gradient at all. That makes the best attack a plain sum.
    """

    probability_output = False

    def __init__(self, width: int) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = x @ self.w
        return result


def linear_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The loss that goes with ``LinearProbe``."""
    return -output * target


@dataclass(frozen=True)
class MechanismAudit:
    """What one check found.

    Attributes:
        name: Which mechanism was attacked.
        trials: Number of runs in each of the two worlds.
        epsilon_reported: The budget the ledger reports for what was released.
        delta: The $\\delta$ of that budget.
        epsilon_lower: The lower limit the attack proves, at ``confidence``.
        epsilon_estimate: The attack's best estimate of the budget (no safety margin).
            ``None`` for checks that count errors instead of fitting bell curves.
        confidence: Confidence of the lower limit.
        flagged: True if the lower limit is above the reported budget: the mechanism
            leaks more than its ledger says.
    """

    name: str
    trials: int
    epsilon_reported: float
    delta: float
    epsilon_lower: float
    epsilon_estimate: float | None
    confidence: float
    flagged: bool

    def describe(self) -> str:
        """The outcome in plain words."""
        estimate = (
            "" if self.epsilon_estimate is None else f", best estimate {self.epsilon_estimate:.2f}"
        )
        verdict = "FLAGGED: leaks more than reported" if self.flagged else "not flagged"
        return (
            f"{self.name}: reported budget {self.epsilon_reported:.2f}, the strongest attack "
            f"proves at least {self.epsilon_lower:.2f}{estimate} "
            f"({self.trials} runs per world). {verdict}."
        )


def compare_worlds(
    name: str,
    without: Sequence[float],
    with_target: Sequence[float],
    reported: float,
    *,
    delta: float = DELTA,
    confidence: float = 0.999,
) -> MechanismAudit:
    """Judge a check whose scores are bell-shaped with the same spread in both worlds."""
    lower = gaussian_epsilon_lower_bound(without, with_target, delta, confidence=confidence)
    estimate = gaussian_epsilon_lower_bound(without, with_target, delta, confidence=1e-9)
    return MechanismAudit(
        name=name,
        trials=min(len(without), len(with_target)),
        epsilon_reported=reported,
        delta=delta,
        epsilon_lower=lower,
        epsilon_estimate=estimate,
        confidence=confidence,
        flagged=lower > reported,
    )


def counted_epsilon_lower_bound(
    without: Sequence[float],
    with_target: Sequence[float],
    delta: float = DELTA,
    confidence: float = 0.999,
) -> float:
    """A lower limit from error counts, for scores of any shape.

    The cut-off between "in" and "out" is chosen on the first half of the trials and the
    errors are counted on the second half, so the choice does not flatter the result.
    """
    trials = min(len(without), len(with_target))
    half = trials // 2
    if half < 1:
        raise ValueError("at least two trials are needed in each world")
    candidates = sorted([*without[:half], *with_target[:half]])[:: max(1, half // 50)]
    best_threshold = max(
        candidates,
        key=lambda t: epsilon_lower_bound(
            sum(score > t for score in without[:half]),
            sum(score <= t for score in with_target[:half]),
            half,
            half,
            delta,
        ),
    )
    return epsilon_lower_bound(
        sum(score > best_threshold for score in without[half:trials]),
        sum(score <= best_threshold for score in with_target[half:trials]),
        trials - half,
        trials - half,
        delta,
        confidence=confidence,
    )


# ---------------------------------------------------------------------- the private step


def local_step_scores(
    with_canary: bool,
    trials: int,
    *,
    noise_multiplier: float,
    steps: int,
    seed: int,
    train_round: Callable[..., Any] = train_private_round,
) -> Scores:
    """Attack the private training step (``LocalDPSGD``) where every record joins every step.

    The releases are then plain Gaussian noise and the best possible attack is known:
    project each noisy gradient on the canary's direction and add the projections up. Its
    score is Gaussian with standard deviation $\\sigma \\sqrt{T}$, shifted by $T$ when the
    canary is present.

    Args:
        with_canary: Which world: with the canary record or without it.
        trials: Number of runs.
        noise_multiplier: $\\sigma$ of every step.
        steps: Number of steps $T$ per run.
        seed: Seed of the sampling and the noise.
        train_round: The training round to attack; the library's by default. A self-test
            passes a deliberately broken one.

    Returns:
        The attacker's score of every run, and the budget the ledger reports for one run.
    """
    width, others, clip = 6, 5, 1.0
    data = torch.Generator().manual_seed(0)
    features = torch.randn(others, width, generator=data)
    canary = torch.zeros(1, width)
    canary[0, 0] = 3.0  # longer than the clipping norm, so it is clipped to exactly that
    if with_canary:
        features = torch.cat([features, canary])
    targets = torch.ones(features.shape[0])
    declared = others + 1  # the public size is the same in both worlds
    sampling = torch.Generator().manual_seed(seed)
    noise = torch.Generator().manual_seed(seed + 1)
    scores = []
    spent = 0.0
    for _ in range(trials):
        model = LinearProbe(width)
        ledger = PrivacyLedger(PrivacyPolicy(50.0, DELTA))
        seen: list[float] = []
        train_round(
            model,
            features,
            targets,
            loss_fn=linear_loss,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            ledger=ledger,
            scope="audited",
            sampling_rate=1.0,
            steps=steps,
            clip_norm=clip,
            noise_multiplier=noise_multiplier,
            sampling_generator=sampling,
            noise_generator=noise,
            declared_records=declared,
            on_step=lambda step, indices, noisy, sink=seen: sink.append(
                -float(noisy["w"][0]) * declared / clip
            ),
        )
        scores.append(sum(seen))
        spent = ledger.spent("audited")
    return scores, spent


def sampled_training_scores(with_canary: bool, trials: int, seed: int) -> Scores:
    """The same attack on a small network whose steps draw their records at random.

    The scores are not bell-shaped here, so they are judged with
    ``counted_epsilon_lower_bound``.
    """
    width, others, clip, rate, steps, sigma = 4, 12, 1.0, 0.5, 8, 0.8
    data = torch.Generator().manual_seed(0)
    features = torch.randn(others, width, generator=data)
    targets = (torch.rand(others, generator=data) > 0.5).float()
    canary_x = torch.full((1, width), 4.0)
    canary_y = torch.zeros(1)
    if with_canary:
        features = torch.cat([features, canary_x])
        targets = torch.cat([targets, canary_y])
    sampling = torch.Generator().manual_seed(seed)
    noise = torch.Generator().manual_seed(seed + 1)
    loss_fn = nn.BCELoss()
    scores = []
    spent = 0.0
    for _ in range(trials):
        torch.manual_seed(0)
        model = TabularMLP(width, (4,))
        ledger = PrivacyLedger(PrivacyPolicy(1e6, DELTA, accountant="rdp"))
        seen: list[float] = []

        def watch(
            step: int,
            indices: torch.Tensor,
            noisy: dict[str, torch.Tensor],
            model: nn.Module = model,
            sink: list[float] = seen,
        ) -> None:
            # The attacker knows the weights at this step, so it knows the canary's gradient.
            direction = clipped_sum(per_record_gradients(model, loss_fn, canary_x, canary_y), clip)
            sink.append(sum(float((noisy[name] * direction[name]).sum()) for name in noisy))

        train_private_round(
            model,
            features,
            targets,
            loss_fn=loss_fn,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
            ledger=ledger,
            scope="audited",
            sampling_rate=rate,
            steps=steps,
            clip_norm=clip,
            noise_multiplier=sigma,
            sampling_generator=sampling,
            noise_generator=noise,
            declared_records=others + 1,
            on_step=watch,
        )
        scores.append(sum(seen))
        spent = PrivacyLedger(PrivacyPolicy(1e6, DELTA)).epsilon_of(ledger.events("audited"))
    return scores, spent


def audit_local_step(
    trials: int = 2000,
    *,
    steps: int = 2,
    train_round: Callable[..., Any] = train_private_round,
    confidence: float = 0.999,
) -> MechanismAudit:
    """Check the private training step at a reported budget of about 1."""
    sigma = UNIT_BUDGET_NOISE * math.sqrt(steps)
    without, spent = local_step_scores(
        False, trials, noise_multiplier=sigma, steps=steps, seed=10, train_round=train_round
    )
    with_canary, _ = local_step_scores(
        True, trials, noise_multiplier=sigma, steps=steps, seed=20, train_round=train_round
    )
    return compare_worlds(
        "private training step", without, with_canary, spent, confidence=confidence
    )


# ---------------------------------------------------------------------- shared noise


def shared_sum_scores(
    with_target: bool,
    trials: int,
    *,
    noise_multiplier: float,
    seed: int,
    sum_step: Callable[..., torch.Tensor] = private_sum_step,
) -> Scores:
    """Attack one step of the shared-noise mode with ordinary noise.

    Three participants, one of them (the last) works with the server. The attacker sees
    the opened sum, knows the colluder's whole message and every record except the target,
    and must decide whether the target record was in the first participant's table. Every
    row joins (full batch), so the best attack is known: remove the colluder's message and
    project what is left on the target's gradient.

    Returns:
        The attacker's score of every run, and the budget the ledger reports for one run.
    """
    width, clip, count = 6, 1.0, 3
    one_slice = share_std(noise_multiplier, clip, count, 1, 0)
    grid = plan_sum_grid(one_slice, default_sum_bound(clip, 1.0, 4), count)
    modulus = np.uint64(2**grid.modulus_bits)
    data = torch.Generator().manual_seed(0)
    tables = [torch.randn(3, width, generator=data) for _ in range(count)]
    target = torch.zeros(1, width)
    target[0, 0] = 3.0  # longer than the clipping norm, so it is clipped to exactly that
    if with_target:
        tables[0] = torch.cat([tables[0], target])
    sampling = torch.Generator().manual_seed(seed)
    noise = torch.Generator().manual_seed(seed + 1)
    rounding = np.random.default_rng(seed + 2)
    model = LinearProbe(width)
    scores = []
    spent = 0.0
    for _ in range(trials):
        ledger = PrivacyLedger(PrivacyPolicy(50.0, DELTA))
        total = np.zeros(width, dtype=np.uint64)
        messages = []
        for index, table in enumerate(tables):
            message = sum_step(
                model,
                table,
                torch.ones(table.shape[0]),
                loss_fn=linear_loss,
                ledger=ledger,
                scope=f"site-{index}",
                sampling_rate=1.0,
                clip_norm=clip,
                noise_multiplier=noise_multiplier,
                share_std=one_slice,
                sampling_generator=sampling,
                noise_generator=noise,
            ).numpy()
            messages.append(message)
            total = (total + grid.encode(message, rounding)) % modulus
        honest = grid.decode_sum(total) - messages[-1]
        scores.append(-float(honest[0]) / clip)
        spent = ledger.spent("site-0")
    return scores, spent


def audit_shared_sum(
    trials: int = 2000,
    *,
    sum_step: Callable[..., torch.Tensor] = private_sum_step,
    confidence: float = 0.999,
) -> MechanismAudit:
    """Check the shared-noise step (ordinary noise, one colluder) at a budget of about 1."""
    without, spent = shared_sum_scores(
        False, trials, noise_multiplier=UNIT_BUDGET_NOISE, seed=10, sum_step=sum_step
    )
    with_target, _ = shared_sum_scores(
        True, trials, noise_multiplier=UNIT_BUDGET_NOISE, seed=20, sum_step=sum_step
    )
    return compare_worlds(
        "shared noise, ordinary", without, with_target, spent, confidence=confidence
    )


def whole_number_sum_scores(with_target: bool, trials: int, *, seed: int) -> Scores:
    """The attack of ``shared_sum_scores`` against one release with whole-number noise."""
    width, clip, count = 6, 1.0, 3
    step = plan_integer_step(width)
    planned = calibrate_skellam(1.0, DELTA, 1, step.l2_sensitivity, step.l1_sensitivity)
    slice_variance = planned.variance / (count - 1)
    grid = plan_integer_grid(
        clip / step.scale, math.ceil(4 * step.l2_sensitivity), math.sqrt(slice_variance), count
    )
    modulus = np.uint64(2**grid.modulus_bits)
    data = torch.Generator().manual_seed(0)
    tables = [torch.randn(3, width, generator=data) for _ in range(count)]
    target = torch.zeros(1, width)
    target[0, 0] = 3.0
    if with_target:
        tables[0] = torch.cat([tables[0], target])
    rounding = np.random.default_rng(seed)
    noise = np.random.default_rng(seed + 1)
    model = LinearProbe(width)
    scores = []
    spent = 0.0
    for _ in range(trials):
        ledger = PrivacyLedger(PrivacyPolicy(50.0, DELTA))
        total = np.zeros(width, dtype=np.uint64)
        messages = []
        for index, table in enumerate(tables):
            message = integer_sum_step(
                model,
                table,
                torch.ones(table.shape[0]),
                loss_fn=linear_loss,
                ledger=ledger,
                scope=f"site-{index}",
                clip_norm=clip,
                step=step,
                variance=planned.variance,
                share_variance=slice_variance,
                rounding_generator=rounding,
                noise_generator=noise,
            )
            messages.append(message)
            total = (total + grid.wrap(message)) % modulus
        honest = grid.decode_sum(total) / grid.step - messages[-1]
        scores.append(-float(honest[0]))
        spent = ledger.spent("site-0")
    return scores, spent


def audit_whole_number_sum(trials: int = 2000, *, confidence: float = 0.999) -> MechanismAudit:
    """Check the shared-noise step with whole-number noise at a budget of about 1."""
    without, spent = whole_number_sum_scores(False, trials, seed=10)
    with_target, _ = whole_number_sum_scores(True, trials, seed=20)
    return compare_worlds(
        "shared noise, whole numbers", without, with_target, spent, confidence=confidence
    )


# ---------------------------------------------------------------------- whole participants


def _pulling_rows(name: str, width: int, value: float, seed: int) -> ClientData:
    """Forty rows whose first column is ``value``; most outcomes are 1, so they pull one way."""
    features = np.zeros((40, width), dtype=np.float32)
    features[:, 0] = value
    labels = np.ones(40, dtype=np.float32)
    labels[::10] = 0.0
    return make_client_data(name, features, labels, seed=seed)


def whole_participant_scores(
    with_target: bool, trials: int, *, first_seed: int, epsilon: float = 1.0
) -> Scores:
    """Attack ``ClientLevelDP``: decide whether one whole participant contributed.

    Three participants, the last works with the server. The target's rows all pull the
    first weight the same way, so its shortened change is exactly the clipping norm in
    that direction; in the other world its rows pull nowhere. The attacker opens the sum,
    takes out the colluder's message and reads the first number. This check runs the whole
    ``Simulation`` and watches it through the server's eyes.

    Returns:
        The attacker's score of every run, and the budget the run's privacy report states.
    """
    width, clip = 6, 1.0
    group = [
        _pulling_rows("site-0", width, 3.0 if with_target else 0.0, 0),
        _pulling_rows("site-1", width, 0.0, 1),
        _pulling_rows("site-2", width, 0.0, 2),
    ]
    with warnings.catch_warnings():
        # Three participants are far too few for real use; here they are the worst case.
        warnings.simplefilter("ignore", SmallFederationWarning)
        simulation = Simulation(
            model_fn=lambda: LinearProbe(width),
            clients=group,
            plan=Plan(
                dp=ClientLevelDP(sampling_rate=1.0, clip_norm=clip, max_colluding=1),
                secure_sum=PairwiseMasking(),
            ),
            policy=PrivacyPolicy(epsilon, DELTA, unit="client"),
            rounds=1,
            learning_rate=5.0,
            weighted_average=False,
            evaluate_every=0,
            outputs="logit",
        )
    scores = []
    spent = 0.0
    for trial in range(trials):
        seen: list[float] = []

        def watch(record: RoundRecord, sink: list[float] = seen) -> None:
            left = server_view(record, colluders=["site-2"]).honest_total()
            assert left is not None
            sink.append(float(left[0]))

        result = simulation.run(seed=first_seed + trial, observer=watch)
        scores.append(seen[0])
        assert result.report is not None
        spent = result.report.epsilon
    return scores, spent


def audit_whole_participant(trials: int = 1500, *, confidence: float = 0.999) -> MechanismAudit:
    """Check the mode that protects whole participants at a budget of 1."""
    without, spent = whole_participant_scores(False, trials, first_seed=10_000)
    with_target, _ = whole_participant_scores(True, trials, first_seed=20_000)
    return compare_worlds(
        "whole participants protected", without, with_target, spent, confidence=confidence
    )


def audit_all(trials: int = 2000) -> list[MechanismAudit]:
    """Every check of this module at a reported budget of about 1."""
    return [
        audit_local_step(trials),
        audit_shared_sum(trials),
        audit_whole_number_sum(trials),
        audit_whole_participant(trials),
    ]
