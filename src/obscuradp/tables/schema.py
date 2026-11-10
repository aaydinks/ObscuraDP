"""Turning a table into numbers using only what was declared in advance.

Scaling a column by its own minimum, maximum, mean or median uses the private data twice:
once to train and once, unaccounted, to compute those statistics. It also lets one unusual
row change how every other row is encoded. Here the caller declares each column's range or
its list of categories up front, from public knowledge (a form's allowed values, a medical
reference range). Each row is then encoded on its own, so preprocessing costs no privacy
budget and no row influences another.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

FloatTable = npt.NDArray[np.float32]


@dataclass(frozen=True)
class NumericFeature:
    """A number with a declared range.

    Values are cut off at the range and mapped linearly to [-1, 1]. A missing or unreadable
    value becomes ``missing`` (by default the middle of the range).
    """

    name: str
    lower: float
    upper: float
    missing: float | None = None

    def __post_init__(self) -> None:
        if not self.upper > self.lower:
            raise ValueError(f"{self.name}: upper must be greater than lower")
        if self.missing is not None and not self.lower <= self.missing <= self.upper:
            raise ValueError(f"{self.name}: the value for missing entries must be inside the range")

    @property
    def columns(self) -> tuple[str, ...]:
        return (self.name,)

    def encode(self, values: Sequence[Any] | npt.NDArray[Any]) -> FloatTable:
        numbers = _as_numbers(values)
        fill = (self.lower + self.upper) / 2.0 if self.missing is None else self.missing
        numbers = np.where(np.isfinite(numbers), numbers, fill)
        clipped = np.clip(numbers, self.lower, self.upper)
        scaled = 2.0 * (clipped - self.lower) / (self.upper - self.lower) - 1.0
        return np.asarray(scaled, dtype=np.float32).reshape(-1, 1)


@dataclass(frozen=True)
class CategoricalFeature:
    """A label from a declared list. Encoded as one 0/1 column per listed category.

    A value that is not on the list (including a missing one) gets zeros in every column.
    """

    name: str
    categories: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "categories", tuple(str(c) for c in self.categories))
        if len(self.categories) == 0:
            raise ValueError(f"{self.name}: at least one category is needed")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError(f"{self.name}: categories must be distinct")

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(f"{self.name}={category}" for category in self.categories)

    def encode(self, values: Sequence[Any] | npt.NDArray[Any]) -> FloatTable:
        labels = np.asarray([str(value) for value in np.asarray(values, dtype=object).ravel()])
        out = np.zeros((labels.size, len(self.categories)), dtype=np.float32)
        for index, category in enumerate(self.categories):
            out[:, index] = labels == category
        return out


Feature = NumericFeature | CategoricalFeature


@dataclass(frozen=True)
class TableSchema:
    """The declared columns of a table, in the order they are encoded."""

    features: tuple[Feature, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", tuple(self.features))
        names = [feature.name for feature in self.features]
        if len(names) == 0:
            raise ValueError("a schema needs at least one feature")
        if len(set(names)) != len(names):
            raise ValueError("feature names must be distinct")

    @property
    def columns(self) -> tuple[str, ...]:
        """Names of the encoded columns."""
        return tuple(column for feature in self.features for column in feature.columns)

    @property
    def width(self) -> int:
        """Number of encoded columns."""
        return len(self.columns)

    def encode(self, table: Mapping[str, Any]) -> FloatTable:
        """Encode ``table`` (column name to values; a pandas DataFrame works too).

        Every output row depends only on the same input row and on this declaration.
        """
        parts = []
        length: int | None = None
        for feature in self.features:
            if feature.name not in table:
                raise KeyError(f"the table has no column {feature.name!r}")
            part = feature.encode(np.asarray(table[feature.name], dtype=object))
            if length is not None and part.shape[0] != length:
                raise ValueError("all columns must have the same number of rows")
            length = part.shape[0]
            parts.append(part)
        return np.concatenate(parts, axis=1)


def _as_numbers(values: Sequence[Any] | npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    raw = np.asarray(values, dtype=object).ravel()
    out = np.full(raw.size, np.nan, dtype=np.float64)
    for index, value in enumerate(raw):
        try:
            out[index] = float(value)
        except (TypeError, ValueError):
            continue
    return out
