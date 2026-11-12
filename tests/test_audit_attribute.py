"""Guessing a hidden column: with the model, and without it as the yardstick."""

import numpy as np
import pytest
import torch

from obscuradp.audit.attribute import (
    Attribute,
    attribute_from_schema,
    attribute_names,
    balanced_accuracy,
    fit_imputer,
    infer_with_model,
    outcome_log_likelihood,
)
from obscuradp.models import TabularMLP
from obscuradp.state import get_state
from obscuradp.tables import CategoricalFeature, NumericFeature, TableSchema

SCHEMA = TableSchema(
    (
        NumericFeature("age", 0.0, 100.0),
        NumericFeature("flag", 0.0, 1.0),
        CategoricalFeature("married", ("Yes", "No")),
        CategoricalFeature("work", ("a", "b", "c")),
    )
)


def test_a_hidden_column_is_found_in_the_declared_table() -> None:
    flag = attribute_from_schema(SCHEMA, "flag")
    assert (flag.columns, flag.values) == ((1,), ((-1.0,), (1.0,)))
    married = attribute_from_schema(SCHEMA, "married")
    assert (married.columns, married.labels) == ((2, 3), ("Yes", "No"))
    assert married.values == ((1.0, 0.0), (0.0, 1.0))
    work = attribute_from_schema(SCHEMA, "work")
    assert work.columns == (4, 5, 6)
    assert len(work.values) == 3
    assert list(attribute_names(SCHEMA)) == ["married"]
    with pytest.raises(KeyError, match="no column"):
        attribute_from_schema(SCHEMA, "missing")
    with pytest.raises(ValueError, match="at least two"):
        Attribute("x", (0,), ((1.0,),), ("only",))
    with pytest.raises(ValueError, match="one label per value"):
        Attribute("x", (0,), ((1.0,), (0.0,)), ("only",))
    with pytest.raises(ValueError, match="one number per encoded column"):
        Attribute("x", (0, 1), ((1.0,), (0.0,)), ("a", "b"))


def test_reading_writing_and_removing_the_hidden_column() -> None:
    table = SCHEMA.encode(
        {
            "age": [50, 20, 80],
            "flag": [0, 1, 1],
            "married": ["Yes", "No", "unknown"],
            "work": ["a", "c", "b"],
        }
    )
    flag = attribute_from_schema(SCHEMA, "flag")
    married = attribute_from_schema(SCHEMA, "married")
    assert flag.read(table).tolist() == [0, 1, 1]
    assert married.read(table).tolist() == [0, 1, -1]  # an unlisted value matches nothing
    changed = married.write(table, 1)
    assert married.read(changed).tolist() == [1, 1, 1]
    assert np.array_equal(np.delete(changed, [2, 3], axis=1), np.delete(table, [2, 3], axis=1))
    assert married.read(table).tolist() == [0, 1, -1]  # the original is untouched
    assert married.without(table).shape == (3, 5)
    assert flag.without(table).shape == (3, 6)


def test_balanced_accuracy_does_not_reward_always_saying_the_common_value() -> None:
    truth = np.array([0] * 90 + [1] * 10)
    assert balanced_accuracy(truth, np.zeros(100, dtype=int), 2) == pytest.approx(0.5)
    assert balanced_accuracy(truth, truth, 2) == 1.0
    half_right = np.array([0] * 90 + [1] * 5 + [0] * 5)
    assert balanced_accuracy(truth, half_right, 2) == pytest.approx(0.75)
    # Records whose value is unknown (-1) count for nothing.
    assert balanced_accuracy(np.array([0, 1, -1]), np.array([0, 1, 0]), 2) == 1.0
    with pytest.raises(ValueError, match="one guess per record"):
        balanced_accuracy(truth, truth[:-1], 2)
    with pytest.raises(ValueError, match="no record has a known value"):
        balanced_accuracy(np.array([-1, -1]), np.array([0, 1]), 2)


def _table(rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A made-up table: the flag depends on age, and the outcome depends on the flag."""
    generator = np.random.default_rng(seed)
    age = generator.uniform(0, 100, rows)
    flag = (generator.random(rows) < 1.0 / (1.0 + np.exp(-(age - 50.0) / 8.0))).astype(int)
    outcome = (generator.random(rows) < np.where(flag == 1, 0.85, 0.15)).astype(np.float64)
    table = SCHEMA.encode(
        {
            "age": age,
            "flag": flag,
            "married": generator.choice(["Yes", "No"], rows),
            "work": generator.choice(["a", "b", "c"], rows),
        }
    )
    return table, outcome


def test_without_any_model_related_columns_already_give_a_column_away() -> None:
    flag = attribute_from_schema(SCHEMA, "flag")
    known, known_outcome = _table(600, seed=0)
    hidden, outcome = _table(2000, seed=1)
    truth = flag.read(hidden)
    blind = fit_imputer(flag, known)
    assert not blind.use_labels
    from_columns = balanced_accuracy(truth, blind.guess(hidden), 2)
    assert from_columns > 0.8  # age alone says a lot
    with_outcome = fit_imputer(flag, known, known_outcome)
    assert with_outcome.use_labels
    better = balanced_accuracy(truth, with_outcome.guess(hidden, outcome), 2)
    assert better > from_columns + 0.02  # and the outcome says more
    assert blind.log_odds(hidden).shape == (2000, 2)
    with pytest.raises(ValueError, match="needs them"):
        with_outcome.guess(hidden)
    with pytest.raises(ValueError, match="too few rows"):
        fit_imputer(flag, known[:3])
    # A column that depends on nothing is not given away.
    married = attribute_from_schema(SCHEMA, "married")
    guess = fit_imputer(married, known, known_outcome).guess(hidden, outcome)
    assert abs(balanced_accuracy(married.read(hidden), guess, 2) - 0.5) < 0.06


def test_the_model_is_asked_under_which_value_the_outcome_fits_best() -> None:
    flag = attribute_from_schema(SCHEMA, "flag")
    hidden, outcome = _table(2000, seed=2)
    truth = flag.read(hidden)
    for probability_output in (True, False):
        model = TabularMLP(SCHEMA.width, (), probability_output=probability_output)
        state = get_state(model)
        weights = torch.zeros(1, SCHEMA.width)
        weights[0, 1] = 0.87  # the outcome's log-odds are 1.73 times the flag (plus or minus 1)
        state["net.0.weight"] = weights
        state["net.0.bias"] = torch.zeros(1)
        likelihood = outcome_log_likelihood(
            model, state, hidden, outcome, flag, from_logits=not probability_output
        )
        assert likelihood.shape == (2000, 2)
        assert np.all(likelihood <= 0.0)
        # Each record is guessed to have the value that fits its outcome: right 85% of the time.
        alone = infer_with_model(likelihood)
        assert alone.tolist() == outcome.astype(int).tolist()
        assert balanced_accuracy(truth, alone, 2) == pytest.approx(0.85, abs=0.03)
    # With what the other columns say as the starting point, the two sources add up.
    known, _ = _table(600, seed=3)
    prior = fit_imputer(flag, known).log_odds(hidden)
    combined = balanced_accuracy(truth, infer_with_model(likelihood, prior), 2)
    assert combined > balanced_accuracy(truth, alone, 2)
    assert combined > balanced_accuracy(truth, fit_imputer(flag, known).guess(hidden), 2)
    with pytest.raises(ValueError, match="one column per value"):
        infer_with_model(likelihood, prior[:, :1])
