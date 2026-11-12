"""Can the model give away one hidden column of a record?

The attacker knows a record except for one sensitive column (say, a yes/no flag), knows
its outcome, and has the model. It tries every possible value of the hidden column and
asks which one makes the known outcome most plausible under the model.

Whether that is a privacy leak depends on two comparisons, and both are built in:

- **Against an attacker without the model.** Columns of a table are related, so a hidden
  column can often be guessed from the others with nothing but rows from the same
  population. ``impute_without_model`` is that attacker. Only what the model adds on top
  counts against the model.
- **Rows that were used against rows that were not.** If the model helps equally for
  both, it has learned something about the population (which is its job), not about the
  rows it was trained on.

Quality is reported as balanced accuracy: the mean, over the possible values, of how often
that value is recognized. 0.5 means no better than guessing for a two-valued column,
whatever the mix of values.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from obscuradp.state import State, set_state
from obscuradp.tables.schema import CategoricalFeature, NumericFeature, TableSchema

FloatTable = npt.NDArray[np.float64]
IntVector = npt.NDArray[np.int64]

_EDGE = 1e-7


@dataclass(frozen=True)
class Attribute:
    """A hidden column: where it sits in the encoded table and what it can be.

    Attributes:
        name: The column's name.
        columns: The positions of its encoded columns.
        values: Every possible encoding, one tuple per value.
        labels: A name for each value.
    """

    name: str
    columns: tuple[int, ...]
    values: tuple[tuple[float, ...], ...]
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.values) < 2:
            raise ValueError("a hidden column needs at least two possible values")
        if len(self.labels) != len(self.values):
            raise ValueError("one label per value is needed")
        if any(len(value) != len(self.columns) for value in self.values):
            raise ValueError("every value needs one number per encoded column")

    def read(self, features: npt.ArrayLike) -> IntVector:
        """Which value each record has (the index into ``values``; -1 if none of them)."""
        table = np.asarray(features, dtype=np.float64)[:, list(self.columns)]
        found = np.full(table.shape[0], -1, dtype=np.int64)
        for index, value in enumerate(self.values):
            found[np.all(np.isclose(table, np.asarray(value)), axis=1)] = index
        return found

    def write(self, features: npt.ArrayLike, index: int) -> npt.NDArray[np.float32]:
        """A copy of the records in which the column has value number ``index``."""
        table = np.array(features, dtype=np.float32, copy=True)
        table[:, list(self.columns)] = np.asarray(self.values[index], dtype=np.float32)
        return table

    def without(self, features: npt.ArrayLike) -> FloatTable:
        """The records without the hidden column."""
        table = np.asarray(features, dtype=np.float64)
        keep = [column for column in range(table.shape[1]) if column not in self.columns]
        return table[:, keep]


def attribute_from_schema(schema: TableSchema, name: str) -> Attribute:
    """A yes/no flag or a column of listed categories as a hidden column.

    A numeric column is taken as a flag with the two values at the ends of its declared
    range. A column of categories has one value per listed category.
    """
    position = 0
    for feature in schema.features:
        width = len(feature.columns)
        if feature.name == name:
            columns = tuple(range(position, position + width))
            if isinstance(feature, NumericFeature):
                return Attribute(name, columns, ((-1.0,), (1.0,)), ("low", "high"))
            assert isinstance(feature, CategoricalFeature)
            values = tuple(
                tuple(1.0 if other == index else 0.0 for other in range(width))
                for index in range(width)
            )
            return Attribute(name, columns, values, tuple(feature.categories))
        position += width
    raise KeyError(f"the table has no column {name!r}")


def balanced_accuracy(truth: npt.ArrayLike, guess: npt.ArrayLike, values: int) -> float:
    """The mean, over the values that occur, of the share of their records guessed right."""
    real = np.asarray(truth, dtype=np.int64)
    said = np.asarray(guess, dtype=np.int64)
    if real.shape != said.shape:
        raise ValueError("one guess per record is needed")
    shares = [
        float(np.mean(said[real == value] == value))
        for value in range(values)
        if np.any(real == value)
    ]
    if not shares:
        raise ValueError("no record has a known value")
    return float(np.mean(shares))


def _fit_logistic(inputs: FloatTable, target: npt.NDArray[np.float64], ridge: float) -> FloatTable:
    """Weights of a ridge-penalized logistic regression (Newton steps), bias last."""
    design = np.column_stack([inputs, np.ones(inputs.shape[0])])
    weights = np.zeros(design.shape[1])
    penalty = ridge * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    for _ in range(50):
        probability = 1.0 / (1.0 + np.exp(-np.clip(design @ weights, -30.0, 30.0)))
        gradient = design.T @ (probability - target) + penalty @ weights
        curvature = (design * (probability * (1.0 - probability))[:, None]).T @ design + penalty
        step = np.linalg.solve(curvature + 1e-9 * np.eye(design.shape[1]), gradient)
        weights = weights - step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return np.asarray(weights, dtype=np.float64)


@dataclass(frozen=True)
class Imputer:
    """Guesses a hidden column from the other columns, learned from rows the attacker has.

    One logistic regression per value (that value against the rest). ``log_odds`` gives,
    for each record and value, how plausible the value is.
    """

    attribute: Attribute
    weights: FloatTable
    use_labels: bool

    def _inputs(self, features: npt.ArrayLike, labels: npt.ArrayLike | None) -> FloatTable:
        rest = self.attribute.without(features)
        if not self.use_labels:
            return rest
        if labels is None:
            raise ValueError("this imputer was fitted with labels and needs them")
        return np.column_stack([rest, np.asarray(labels, dtype=np.float64)])

    def log_odds(self, features: npt.ArrayLike, labels: npt.ArrayLike | None = None) -> FloatTable:
        inputs = self._inputs(features, labels)
        design = np.column_stack([inputs, np.ones(inputs.shape[0])])
        return np.asarray(design @ self.weights.T, dtype=np.float64)

    def guess(self, features: npt.ArrayLike, labels: npt.ArrayLike | None = None) -> IntVector:
        return np.asarray(np.argmax(self.log_odds(features, labels), axis=1), dtype=np.int64)


def fit_imputer(
    attribute: Attribute,
    features: npt.ArrayLike,
    labels: npt.ArrayLike | None = None,
    *,
    ridge: float = 1.0,
) -> Imputer:
    """Learn to guess the hidden column from rows whose value the attacker knows.

    Args:
        attribute: The hidden column.
        features: Rows the attacker has in full (not the rows under attack).
        labels: Their outcomes. If given, the outcome is used as one more column.
        ridge: Strength of the penalty that keeps the fit stable on small tables.
    """
    truth = attribute.read(features)
    known = truth >= 0
    if int(known.sum()) < 2 * len(attribute.values):
        raise ValueError("too few rows with a known value to learn from")
    use_labels = labels is not None
    rest = attribute.without(features)
    inputs = (
        np.column_stack([rest, np.asarray(labels, dtype=np.float64)]) if use_labels else rest
    )[known]
    weights = np.stack(
        [
            _fit_logistic(inputs, (truth[known] == value).astype(np.float64), ridge)
            for value in range(len(attribute.values))
        ]
    )
    return Imputer(attribute=attribute, weights=weights, use_labels=use_labels)


def outcome_log_likelihood(
    model: nn.Module,
    state: State,
    features: npt.ArrayLike,
    labels: npt.ArrayLike,
    attribute: Attribute,
    *,
    from_logits: bool,
) -> FloatTable:
    """For each record and each possible hidden value: how plausible the known outcome is.

    Returns:
        One row per record and one column per value: the log of the model's probability of
        the record's true outcome when the hidden column is set to that value.
    """
    set_state(model, state)
    was_training = model.training
    model.eval()
    positive = np.asarray(labels, dtype=np.float64) > 0.5
    columns = []
    try:
        with torch.no_grad():
            for index in range(len(attribute.values)):
                output = model(torch.as_tensor(attribute.write(features, index))).double()
                probability = torch.sigmoid(output) if from_logits else output
                probability = probability.clamp(_EDGE, 1.0 - _EDGE).numpy()
                columns.append(np.log(np.where(positive, probability, 1.0 - probability)))
    finally:
        model.train(was_training)
    return np.column_stack(columns)


def infer_with_model(
    likelihood: npt.ArrayLike, prior_log_odds: npt.ArrayLike | None = None
) -> IntVector:
    """The attacker's guess with the model: the value under which the outcome fits best.

    Args:
        likelihood: The result of ``outcome_log_likelihood``.
        prior_log_odds: What the attacker believed before looking at the model, one column
            per value (for example ``Imputer.log_odds`` of an imputer fitted **without**
            labels, since the model's part already accounts for the outcome). Without it
            every value counts as equally likely beforehand.
    """
    total = np.asarray(likelihood, dtype=np.float64)
    if prior_log_odds is not None:
        prior = np.asarray(prior_log_odds, dtype=np.float64)
        if prior.shape != total.shape:
            raise ValueError("the prior needs one column per value and one row per record")
        # Log-odds of "this value against the rest" to a log-probability.
        total = total - np.logaddexp(0.0, -prior)
    return np.asarray(np.argmax(total, axis=1), dtype=np.int64)


def attribute_names(schema: TableSchema) -> Sequence[str]:
    """The columns of a table that can serve as a hidden column with two values."""
    names = []
    for feature in schema.features:
        if isinstance(feature, CategoricalFeature) and len(feature.categories) == 2:
            names.append(feature.name)
    return names
