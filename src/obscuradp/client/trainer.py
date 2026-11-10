"""One participant's local training, private or plain.

``train_private_round`` runs a fixed number of private gradient steps on one participant's
records. It writes the whole round into the privacy ledger **before** it touches the data:
if the round does not fit the participant's remaining budget, ``BudgetExceeded`` is raised
and nothing is trained.

The function takes the participant's records as tensors, not as a data loader. It draws
every batch itself with Poisson sampling, because the privacy accounting is only valid for
that way of drawing batches.

``train_plain_round`` is the ordinary, unprotected counterpart, used for baselines.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
from torch import nn

from obscuradp.accounting.events import SubsampledGaussianEvent
from obscuradp.ledger import PrivacyLedger
from obscuradp.mechanisms.dpsgd import (
    Gradients,
    LossFunction,
    check_per_record_model,
    private_gradient,
)
from obscuradp.mechanisms.sampling import poisson_sample

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
