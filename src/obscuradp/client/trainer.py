"""One participant's local training, private or plain.

``train_private_round`` runs a fixed number of private gradient steps on one participant's
records. It writes the whole round into the privacy ledger **before** it touches the data:
if the round does not fit the participant's remaining budget, ``BudgetExceeded`` is raised
and nothing is trained.

The function takes the participant's records as tensors, not as a data loader. It draws
every batch itself with Poisson sampling, because the privacy accounting is only valid for
that way of drawing batches.

``train_plain_round`` is the ordinary, unprotected counterpart, used for baselines.

``private_sum_step`` is one participant's part of a round in the shared-noise mode: one
clipped gradient sum at the shared model plus this participant's slice of the noise. Its
result is only safe inside a secure sum. ``clipped_sum_step`` is the same without any noise,
for a server that is trusted to add the noise itself.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from obscuradp.accounting.events import SkellamEvent, SubsampledGaussianEvent
from obscuradp.ledger import PrivacyLedger
from obscuradp.mechanisms.dpsgd import (
    Gradients,
    LossFunction,
    check_per_record_model,
    clipped_records,
    clipped_sum,
    per_record_gradients,
    private_gradient,
)
from obscuradp.mechanisms.noise import gaussian_noise
from obscuradp.mechanisms.sampling import poisson_sample
from obscuradp.mechanisms.skellam import IntegerStep, round_records, skellam_noise

StepObserver = Callable[[int, torch.Tensor, Gradients], None]
ClipScale = Callable[[torch.Tensor], torch.Tensor | None]


@dataclass(frozen=True)
class RoundStats:
    """What happened in one private round.

    These numbers describe the participant's own random draws. They are for local
    diagnostics and must not be sent to anyone: the actual batch sizes are not covered by
    the privacy guarantee.
    """

    steps: int
    empty_batches: int
    records_drawn: int


def train_private_round(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_fn: LossFunction,
    optimizer: torch.optim.Optimizer,
    ledger: PrivacyLedger,
    scope: str,
    sampling_rate: float,
    steps: int,
    clip_norm: float,
    noise_multiplier: float,
    sampling_generator: torch.Generator,
    noise_generator: torch.Generator,
    declared_records: int | None = None,
    clip_scale: ClipScale | None = None,
    proximal_mu: float = 0.0,
    reference: Mapping[str, torch.Tensor] | None = None,
    on_step: StepObserver | None = None,
) -> RoundStats:
    """Train ``model`` in place for ``steps`` private steps on one participant's records.

    Args:
        model: The model, already loaded with the round's starting weights.
        features: All of the participant's training records, one per row.
        targets: Their targets.
        loss_fn: Per-record loss, called as ``loss_fn(output, target)`` on one record.
        optimizer: Any torch optimizer over ``model``'s parameters.
        ledger: The privacy ledger. The round is charged to it before any data is used.
        scope: The ledger scope to charge, normally the participant's name.
        sampling_rate: Probability $q$ with which each record joins each step.
        steps: Number of steps. It must be fixed in advance, never chosen from the data.
        clip_norm: Clipping norm $C$.
        noise_multiplier: $\\sigma$; the noise on each summed gradient has standard
            deviation $\\sigma C$.
        sampling_generator: Random stream for drawing batches.
        noise_generator: Random stream for the noise. For real data use
            ``obscuradp.mechanisms.entropy_generator()``.
        declared_records: The dataset size the participant has declared. Every summed
            gradient is divided by ``sampling_rate * declared_records``. The guarantee treats
            this number as public, so it must not change when one record is added or
            removed. Defaults to the actual number of records.
        clip_scale: Optional function from a batch's targets to one factor in (0, 1] per
            record; record $i$ is clipped to $C \\cdot s_i$. The noise stays sized for $C$.
        proximal_mu: Strength of the FedProx term that pulls the weights towards
            ``reference``. It uses no data, so it costs no privacy.
        reference: The weights to pull towards (normally the round's starting weights).
            Required if ``proximal_mu`` is positive.
        on_step: Optional observer called as ``on_step(step, batch_indices, noisy_gradient)``
            before each update. Meant for audits that play an attacker who sees every step.

    Raises:
        BudgetExceeded: If the round does not fit the scope's remaining budget. Nothing is
            trained in that case.
    """
    n_records = int(features.shape[0])
    if n_records < 1:
        raise ValueError("a participant needs at least one record")
    if targets.shape[0] != n_records:
        raise ValueError("features and targets must have the same number of records")
    if steps < 0:
        raise ValueError("steps must not be negative")
    if declared_records is not None and declared_records < 1:
        raise ValueError("declared_records must be at least 1")
    if proximal_mu < 0.0:
        raise ValueError("proximal_mu must not be negative")
    if proximal_mu > 0.0 and reference is None:
        raise ValueError("reference weights are required when proximal_mu is positive")
    check_per_record_model(model)
    event = SubsampledGaussianEvent(noise_multiplier, sampling_rate, steps)

    # The charge comes first. If it is refused, no data has been touched.
    ledger.charge(scope, event)

    expected_batch_size = sampling_rate * (
        n_records if declared_records is None else declared_records
    )
    trainable = {name: value for name, value in model.named_parameters() if value.requires_grad}
    empty = 0
    drawn = 0
    model.train()
    for step in range(steps):
        indices = poisson_sample(n_records, sampling_rate, sampling_generator)
        empty += int(indices.numel() == 0)
        drawn += int(indices.numel())
        chosen = indices.to(features.device)
        noisy = private_gradient(
            model,
            loss_fn,
            features[chosen],
            targets[chosen],
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier,
            expected_batch_size=expected_batch_size,
            generator=noise_generator,
            clip_scale=None if clip_scale is None else clip_scale(targets[chosen]),
        )
        if on_step is not None:
            on_step(step, indices, noisy)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, value in trainable.items():
                update = noisy[name]
                if proximal_mu > 0.0 and reference is not None:
                    anchor = reference[name].to(value.device)
                    update = update + proximal_mu * (value.detach() - anchor)
                value.grad = update
        optimizer.step()
    return RoundStats(steps=steps, empty_batches=empty, records_drawn=drawn)


def flatten_gradients(gradients: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """All gradients as one 64-bit vector, in the order of the mapping."""
    if not gradients:
        raise ValueError("the model has no trainable parameter")
    return torch.cat([value.detach().double().flatten() for value in gradients.values()])


def clipped_sum_step(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_fn: LossFunction,
    sampling_rate: float,
    clip_norm: float,
    sampling_generator: torch.Generator,
    clip_scale: ClipScale | None = None,
    sum_bound: float | None = None,
    diagnostics: dict[str, float] | None = None,
) -> torch.Tensor:
    """One participant's clipped gradient sum at the model's current weights. **No noise.**

    Each record joins with probability ``sampling_rate``; the gradients of the drawn records
    are clipped and added up. The result is not private. It may only be handed to a party
    that is trusted to add the noise (``CentralDPSGD``), or be noised first
    (``private_sum_step``). The model is not changed.

    Args:
        model: The model at the shared weights.
        features: All of the participant's training records, one per row.
        targets: Their targets.
        loss_fn: Per-record loss, called as ``loss_fn(output, target)`` on one record.
        sampling_rate: Probability $q$ with which each record joins.
        clip_norm: Clipping norm $C$.
        sampling_generator: Random stream for drawing the records.
        clip_scale: Optional function from the drawn targets to one factor in (0, 1] per
            record, see ``train_private_round``.
        sum_bound: If given, every number of the sum is limited to plus or minus this value.
            Limiting each number can only bring two sums closer together, so one record
            still moves the result by at most $C$.
        diagnostics: If given, ``"limited_share"`` is set to the share of numbers that
            ``sum_bound`` cut off. It describes private data: for local use only, never to
            be sent.

    Returns:
        The sum as one 64-bit vector over the trainable parameters, in the order of
        ``model.named_parameters()``.
    """
    n_records = int(features.shape[0])
    if n_records < 1:
        raise ValueError("a participant needs at least one record")
    if targets.shape[0] != n_records:
        raise ValueError("features and targets must have the same number of records")
    if sum_bound is not None and not sum_bound > 0.0:
        raise ValueError("sum_bound must be positive")
    check_per_record_model(model)
    model.train()
    indices = poisson_sample(n_records, sampling_rate, sampling_generator)
    chosen = indices.to(features.device)
    total = clipped_sum(
        per_record_gradients(model, loss_fn, features[chosen], targets[chosen]),
        clip_norm,
        None if clip_scale is None else clip_scale(targets[chosen]),
    )
    flat = flatten_gradients(total)
    if sum_bound is not None:
        if diagnostics is not None:
            diagnostics["limited_share"] = float((flat.abs() > sum_bound).double().mean())
        flat = flat.clamp(min=-sum_bound, max=sum_bound)
    return flat


def private_sum_step(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_fn: LossFunction,
    ledger: PrivacyLedger,
    scope: str,
    sampling_rate: float,
    clip_norm: float,
    noise_multiplier: float,
    share_std: float,
    sampling_generator: torch.Generator,
    noise_generator: torch.Generator,
    clip_scale: ClipScale | None = None,
    sum_bound: float | None = None,
    diagnostics: dict[str, float] | None = None,
) -> torch.Tensor:
    """One participant's message in the shared-noise mode: clipped sum plus its noise slice.

    The release is charged to the ledger **before** any data is used: one step at
    ``noise_multiplier``, the noise that all honest slices give together. This participant
    only adds ``share_std`` of it. **The result is therefore only protected inside a secure
    sum** that hides single messages; sent in the clear it would be protected by one slice
    alone, far less than the ledger says.

    Args:
        model: The model at the shared weights. It is not changed.
        features: All of the participant's training records, one per row.
        targets: Their targets.
        loss_fn: Per-record loss, called as ``loss_fn(output, target)`` on one record.
        ledger: The privacy ledger.
        scope: The ledger scope to charge, normally the participant's name.
        sampling_rate: Probability $q$ with which each record joins.
        clip_norm: Clipping norm $C$.
        noise_multiplier: $\\sigma$, what the ledger is charged with.
        share_std: Standard deviation of this participant's slice, normally
            $\\sigma C / \\sqrt{N - t - d}$ (see ``obscuradp.mechanisms.shared_noise``).
        sampling_generator: Random stream for drawing the records.
        noise_generator: Random stream for the noise. For real data use
            ``obscuradp.mechanisms.entropy_generator()``.
        clip_scale: Optional per-record clip factors, see ``train_private_round``.
        sum_bound: Optional limit for every number of the clipped sum, applied before the
            noise is added.
        diagnostics: Optional place for local diagnostics, see ``clipped_sum_step``.

    Raises:
        BudgetExceeded: If the release does not fit the scope's remaining budget. No data is
            used in that case.
    """
    if not share_std > 0.0:
        raise ValueError("share_std must be positive: a message without noise is not private")
    if int(features.shape[0]) < 1:
        raise ValueError("a participant needs at least one record")
    check_per_record_model(model)
    event = SubsampledGaussianEvent(noise_multiplier, sampling_rate, 1)

    # The charge comes first. If it is refused, no data has been touched.
    ledger.charge(scope, event)

    total = clipped_sum_step(
        model,
        features,
        targets,
        loss_fn=loss_fn,
        sampling_rate=sampling_rate,
        clip_norm=clip_norm,
        sampling_generator=sampling_generator,
        clip_scale=clip_scale,
        sum_bound=sum_bound,
        diagnostics=diagnostics,
    )
    noise = gaussian_noise(total.shape, share_std, noise_generator, dtype=torch.float64)
    return total + noise.to(total.device)


def integer_sum_step(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_fn: LossFunction,
    ledger: PrivacyLedger,
    scope: str,
    clip_norm: float,
    step: IntegerStep,
    variance: float,
    share_variance: float,
    rounding_generator: np.random.Generator,
    noise_generator: np.random.Generator,
    clip_scale: ClipScale | None = None,
    sum_bound: int | None = None,
    diagnostics: dict[str, float] | None = None,
) -> npt.NDArray[np.int64]:
    """One participant's message with whole-number noise: rounded records plus its noise slice.

    Every record takes part. Each record's clipped gradient is scaled to the whole-number
    grid, rounded on its own, and checked in whole-number arithmetic against
    ``step.l2_squared``; the rounded records are added up; then this participant's slice of
    Skellam noise is added. The release is charged to the ledger **before** any data is used:
    one release at ``variance``, the noise of all honest slices together. **The result is
    only protected inside a secure sum**, like the message of ``private_sum_step``.

    Args:
        model: The model at the shared weights. It is not changed.
        features: All of the participant's training records, one per row.
        targets: Their targets.
        loss_fn: Per-record loss, called as ``loss_fn(output, target)`` on one record.
        ledger: The privacy ledger.
        scope: The ledger scope to charge, normally the participant's name.
        clip_norm: Clipping norm $C$.
        step: The whole-number grid (see ``obscuradp.mechanisms.plan_integer_step``).
        variance: Variance of the summed honest noise, in grid units; what the ledger is
            charged with.
        share_variance: Variance of this participant's slice, normally
            ``variance / (N - t - d)``.
        rounding_generator: Random stream for the rounding.
        noise_generator: Random stream for the noise. For real data it must be seeded from
            the operating system.
        clip_scale: Optional per-record clip factors, see ``train_private_round``.
        sum_bound: Optional limit, in grid units, for every number of the sum of rounded
            records, applied before the noise. Limiting each number cannot move two sums
            apart, so the sensitivities stay as they are.
        diagnostics: Optional place for local diagnostics, see ``clipped_sum_step``.

    Returns:
        The message as 64-bit whole numbers over the trainable parameters, in the order of
        ``model.named_parameters()``.

    Raises:
        BudgetExceeded: If the release does not fit the scope's remaining budget. No data is
            used in that case.
    """
    n_records = int(features.shape[0])
    if n_records < 1:
        raise ValueError("a participant needs at least one record")
    if targets.shape[0] != n_records:
        raise ValueError("features and targets must have the same number of records")
    if not share_variance > 0.0:
        raise ValueError("share_variance must be positive: a message without noise is not private")
    if sum_bound is not None and sum_bound < 1:
        raise ValueError("sum_bound must be at least 1")
    check_per_record_model(model)
    event = SkellamEvent(variance, step.l2_sensitivity, step.l1_sensitivity, 1)

    # The charge comes first. If it is refused, no data has been touched.
    ledger.charge(scope, event)

    model.train()
    rows = clipped_records(
        per_record_gradients(model, loss_fn, features, targets),
        clip_norm,
        None if clip_scale is None else clip_scale(targets),
    )
    if rows.shape[1] != step.dimension:
        raise ValueError("the grid was planned for a model of another size")
    scaled = (rows * (step.scale / clip_norm)).cpu().numpy()
    total = round_records(scaled, step.l2_squared, rounding_generator).sum(axis=0)
    if sum_bound is not None:
        if diagnostics is not None:
            diagnostics["limited_share"] = float(np.mean(np.abs(total) > sum_bound))
        total = np.clip(total, -sum_bound, sum_bound)
    noise = skellam_noise(total.shape, share_variance, noise_generator)
    return np.asarray(total + noise, dtype=np.int64)


def train_plain_round(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_fn: LossFunction,
    optimizer: torch.optim.Optimizer,
    epochs: float,
    batch_size: int,
    generator: torch.Generator,
    proximal_mu: float = 0.0,
    reference: Mapping[str, torch.Tensor] | None = None,
) -> int:
    """Train ``model`` in place with ordinary minibatch steps. **No privacy protection.**

    Rows are shuffled and cut into batches of ``batch_size``; ``epochs`` passes are made
    (a fraction is allowed). ``loss_fn`` returns one value per row and its mean is minimized.

    Returns:
        The number of steps taken.
    """
    n_records = int(features.shape[0])
    if n_records < 1:
        raise ValueError("a participant needs at least one record")
    if targets.shape[0] != n_records:
        raise ValueError("features and targets must have the same number of records")
    if not epochs > 0:
        raise ValueError("epochs must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if proximal_mu < 0.0:
        raise ValueError("proximal_mu must not be negative")
    if proximal_mu > 0.0 and reference is None:
        raise ValueError("reference weights are required when proximal_mu is positive")
    # A pass is ceil(n / B) batches (the last one may be smaller), so whole passes see every row.
    steps = math.ceil(epochs * math.ceil(n_records / batch_size) - 1e-9)
    model.train()
    order = torch.empty(0, dtype=torch.int64)
    for _ in range(steps):
        if order.numel() == 0:
            order = torch.randperm(n_records, generator=generator)
        chosen, order = order[:batch_size].to(features.device), order[batch_size:]
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(features[chosen]), targets[chosen]).mean()
        if proximal_mu > 0.0 and reference is not None:
            for name, value in model.named_parameters():
                if value.requires_grad:
                    anchor = reference[name].to(value.device)
                    loss = loss + 0.5 * proximal_mu * (value - anchor).pow(2).sum()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    return steps
