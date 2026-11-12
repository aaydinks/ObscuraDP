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


class SmallCNN(nn.Module):
    """A small network for square grey images, such as handwritten characters.

    Two convolutions with pooling, then two linear layers. It takes each image as one flat
    row of ``side * side`` numbers (the way tables are stored here) and returns one raw
    score per class. With the defaults (28 by 28 pixels, 62 classes) it has 44,318 numbers.
    It has no layer that mixes the rows of a batch.

    Args:
        classes: Number of classes.
        side: Width and height of the images in pixels.
    """

    probability_output = False

    def __init__(self, classes: int = 62, side: int = 28) -> None:
        super().__init__()
        if classes < 2:
            raise ValueError("classes must be at least 2")
        reduced = ((side - 4) // 2 - 4) // 2
        if reduced < 1:
            raise ValueError("side must be at least 16")
        self.side = side
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Linear(16 * reduced * reduced, 128), nn.ReLU(), nn.Linear(128, classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one row of raw scores per image."""
        images = x.reshape(-1, 1, self.side, self.side)
        out: torch.Tensor = self.head(self.features(images).flatten(1))
        return out


class CharGRU(nn.Module):
    """A small network that reads a short text and scores what comes next.

    An embedding, one recurrent layer (GRU) and a linear layer. It takes each text as one
    row of symbol numbers (stored as ordinary numbers, the way tables are stored here) and
    returns one raw score per symbol for the position after the last one. With the
    defaults it has 65,322 numbers.

    Args:
        vocabulary: Number of different symbols.
        embedding: Size of each symbol's embedding.
        hidden: Size of the recurrent layer.
    """

    probability_output = False

    def __init__(self, vocabulary: int = 90, embedding: int = 8, hidden: int = 128) -> None:
        super().__init__()
        if vocabulary < 2:
            raise ValueError("vocabulary must be at least 2")
        self.vocabulary = vocabulary
        self.embed = nn.Embedding(vocabulary, embedding)
        self.gru = nn.GRU(embedding, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocabulary)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one row of raw scores per text."""
        symbols = x.long().clamp(0, self.vocabulary - 1)
        states, _ = self.gru(self.embed(symbols))
        out: torch.Tensor = self.head(states[:, -1])
        return out
