"""Small ready-made models.

Any ``torch.nn.Module`` can be trained with this library. The models here exist so that
examples, tests and the comparison against the earlier research code have a fixed,
well-understood architecture to use.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class TabularMLP(nn.Module):
    """A plain feed-forward network for tables: Linear, ReLU, ..., Linear, optional Sigmoid.

    It contains no layer that mixes information across the rows of a batch (no batch
    normalization), which is what private training needs.

    With ``hidden=(64, 32)`` on 15 inputs this is the 3,137-parameter network of the stroke
    experiments; with ``hidden=(32, 16)`` on 11 inputs it is the 929-parameter network of the
    loan experiments.

    Args:
        in_features: Number of input columns.
        hidden: Width of each hidden layer, in order. May be empty (logistic regression).
        probability_output: If true, apply a sigmoid so the output is a probability in (0, 1).
            If false, the output is the raw score (logit).
    """

    def __init__(
        self,
        in_features: int,
        hidden: Sequence[int] = (64, 32),
        *,
        probability_output: bool = True,
    ) -> None:
        super().__init__()
        if in_features < 1:
            raise ValueError("in_features must be at least 1")
        if any(width < 1 for width in hidden):
            raise ValueError("every hidden width must be at least 1")
        layers: list[nn.Module] = []
        previous = in_features
        for width in hidden:
            layers.append(nn.Linear(previous, width))
            layers.append(nn.ReLU())
            previous = width
        layers.append(nn.Linear(previous, 1))
        if probability_output:
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        self.in_features = in_features
        self.hidden = tuple(hidden)
        self.probability_output = probability_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one value per row."""
        out: torch.Tensor = self.net(x).squeeze(-1)
        return out
