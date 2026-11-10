"""Handling a rare outcome without breaking the privacy guarantee.

When few rows are positive (4% of the stroke data), a model can score well by always
saying "no". Three ways to deal with that are offered:

- ``ClassWeighted``: errors on the rare class count more in the loss. The weights come from
  a **public** figure, such as a published prevalence. This is the supported way.
- ``NoRebalancing``: do nothing in training; choose the decision threshold on validation
  rows afterwards.
- ``UnsafeOversampling``: build synthetic rows (SMOTE and similar) before training. In a
  private run this is refused unless ``allow_unsafe=True``, because each real row then
  shapes many training rows and the budget no longer describes real records.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.nn import functional

from obscuradp.unsafe import UnsafeConfiguration

LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Resampler = Callable[[npt.NDArray[Any], npt.NDArray[Any], int], tuple[Any, Any]]


def _cross_entropy(output: torch.Tensor, target: torch.Tensor, from_logits: bool) -> torch.Tensor:
    if from_logits:
        return functional.binary_cross_entropy_with_logits(output, target, reduction="none")
    return functional.binary_cross_entropy(output, target, reduction="none")


@dataclass(frozen=True)
class NoRebalancing:
    """Plain cross-entropy: every row counts the same."""

    unsafe_code: str | None = None

    def loss(self, *, from_logits: bool) -> LossFunction:
        """A loss that returns one value per row."""
        return lambda output, target: _cross_entropy(output, target, from_logits)

    def clip_scale(self, targets: torch.Tensor) -> torch.Tensor | None:
        """No per-record clipping factors: every row has the full clipping norm."""
        return None


@dataclass(frozen=True)
class ClassWeighted:
    """Cross-entropy in which the rare class counts more.

    In private training every row's gradient is clipped to the same length, which would
    cancel the weights. So the weights are also applied to the clipping norm: rows of the
    heavier class may use the full norm $C$, rows of the lighter class only
    $C \\cdot w_{\\text{light}} / w_{\\text{heavy}}$. No row exceeds $C$, so the privacy
    guarantee is the same as without weights.

    Args:
        positive_prior: The share of positive rows expected in the population, taken from a
            public source. Positive rows get weight $0.5 / p$ and negative rows
            $0.5 / (1 - p)$, so both classes carry the same total weight and the average
            weight is 1.
        unsafe_code: Set by ``from_private_labels``. Leave it alone otherwise.
    """

    positive_prior: float
    unsafe_code: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.positive_prior < 1.0:
            raise ValueError("positive_prior must be in (0, 1)")

    @classmethod
    def from_private_labels(
        cls, labels: npt.ArrayLike, *, allow_unsafe: bool = False
    ) -> ClassWeighted:
        """Weights from the actual label counts. This releases those counts without paying.

        Raises:
            UnsafeConfiguration: Unless ``allow_unsafe`` is true. A private run that uses the
                result carries an unsafe flag in its report.
        """
        if not allow_unsafe:
            raise UnsafeConfiguration(
                "Computing class weights from private labels releases the label counts without "
                "charging the privacy budget. Use a public prior, or pass allow_unsafe=True; "
                "the privacy report will then carry a warning."
            )
        y = np.asarray(labels, dtype=np.float64).ravel()
        if y.size == 0:
            raise ValueError("need at least one label")
        share = float(np.clip(np.mean(y > 0.5), 1e-6, 1 - 1e-6))
        return cls(positive_prior=share, unsafe_code="class_weights_from_private_labels")

    @property
    def weights(self) -> tuple[float, float]:
        """``(weight of a negative row, weight of a positive row)``."""
        return 0.5 / (1.0 - self.positive_prior), 0.5 / self.positive_prior

    def loss(self, *, from_logits: bool) -> LossFunction:
        """A loss that returns one weighted value per row."""
        negative, positive = self.weights

        def weighted(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            weight = torch.where(target > 0.5, positive, negative).to(output.dtype)
            return weight * _cross_entropy(output, target, from_logits)

        return weighted

    def clip_scale(self, targets: torch.Tensor) -> torch.Tensor | None:
        """One factor in (0, 1] per row: each class's weight over the larger weight."""
        negative, positive = self.weights
        largest = max(negative, positive)
        return torch.where(targets > 0.5, positive / largest, negative / largest)


@dataclass(frozen=True)
class UnsafeOversampling:
    """Synthetic rows built from real ones before training (SMOTE and similar).

    Args:
        resample: Called as ``resample(features, labels, seed)`` on one participant's
            training rows; returns the enlarged ``(features, labels)``.
        name: What to call the method in messages.
    """

    resample: Resampler
    name: str = "oversampling"
    unsafe_code: str | None = "oversampling_before_private_training"

    def loss(self, *, from_logits: bool) -> LossFunction:
        """Plain cross-entropy; the balancing happened in the data."""
        return lambda output, target: _cross_entropy(output, target, from_logits)

    def clip_scale(self, targets: torch.Tensor) -> torch.Tensor | None:
        """No per-record clipping factors."""
        return None


Imbalance = NoRebalancing | ClassWeighted | UnsafeOversampling
