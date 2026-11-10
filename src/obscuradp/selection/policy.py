"""Choosing which numbers of an update get encrypted, without leaking through the choice.

With partial encryption only some numbers of each update are encrypted and the rest travel
in the clear. Which numbers are chosen is public: the server sees it. If the choice is
computed from private data, the choice itself reveals something about that data, and
nothing has paid for that.

Three ways to choose are offered:

- ``public_mask``: a random choice from a public seed. It depends on no data at all. This is
  the default.
- ``private_scores`` followed by ``mask_from_scores``: scores computed from private data with
  noise, charged to the privacy ledger like any other release.
- ``mask_from_private_data``: scores taken from private data without noise. Refused unless
  unsafe settings are allowed, and then flagged in the privacy report.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from obscuradp.accounting.events import GaussianEvent
from obscuradp.ledger import PrivacyLedger
from obscuradp.mechanisms.dpsgd import LossFunction, check_per_record_model, private_gradient
from obscuradp.selection.masks import build_mask, random_mask
from obscuradp.state import get_state
from obscuradp.unsafe import UnsafeFlags

Mask = npt.NDArray[np.bool_]


def public_mask(n_total: int, fraction: float, seed: int) -> Mask:
    """Mark a random ``fraction`` of ``n_total`` numbers, using only a public seed."""
    if n_total < 1:
        raise ValueError("n_total must be at least 1")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    return random_mask(n_total, fraction, seed)


def private_scores(
    model: nn.Module,
    loss_fn: LossFunction,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    clip_norm: float,
    noise_multiplier: float,
    ledger: PrivacyLedger,
    scope: str,
    generator: torch.Generator,
) -> dict[str, float]:
    """One noisy score per parameter tensor, paid for from the privacy budget.

    The score of a tensor is the mean absolute value of one private gradient over all of
    the participant's records: every record's gradient is clipped to ``clip_norm``, the
    clipped gradients are summed, noise with standard deviation ``noise_multiplier *
    clip_norm`` is added to the sum, and the result is divided by the number of records.
    The release is charged to ``ledger`` first; if it does not fit the budget,
    ``BudgetExceeded`` is raised and nothing is computed.

    (The research code divided the noise by the number of records of a small batch while
    describing it as covering the whole table, finding G14. Here the noise is always sized
    for exactly the records that are used.)
    """
    n_records = int(features.shape[0])
    if n_records < 1:
        raise ValueError("at least one record is needed")
    check_per_record_model(model)
    ledger.charge(scope, GaussianEvent(noise_multiplier))
    noisy = private_gradient(
        model,
        loss_fn,
        features,
        targets,
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
        expected_batch_size=float(n_records),
        generator=generator,
    )
    return {name: float(value.abs().mean()) for name, value in noisy.items()}


def mask_from_scores(model: nn.Module, scores: Mapping[str, float], fraction: float) -> Mask:
    """Mark the ``fraction`` of numbers whose tensors have the highest scores."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    return build_mask(get_state(model), scores, fraction)


def mask_from_private_data(
    model: nn.Module,
    loss_fn: LossFunction,
    features: torch.Tensor,
    targets: torch.Tensor,
    fraction: float,
    flags: UnsafeFlags,
) -> Mask:
    """A mask from exact gradients of private data. Unsafe: the choice itself leaks.

    Raises:
        UnsafeConfiguration: Unless ``flags`` allows unsafe settings; then the reason is
            recorded for the privacy report.
    """
    flags.raise_or_record("mask_from_private_data")
    model.zero_grad(set_to_none=True)
    loss_fn(model(features), targets).mean().backward()  # type: ignore[no-untyped-call]
    scores = {
        name: float(value.grad.abs().mean())
        for name, value in model.named_parameters()
        if value.grad is not None
    }
    model.zero_grad(set_to_none=True)
    return mask_from_scores(model, scores, fraction)
