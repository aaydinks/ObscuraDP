"""Encoding from declared ranges, and each participant's three-way split."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from obscuradp.data import (
    CategoricalFeature,
    ClientData,
    NumericFeature,
    TableSchema,
    make_client_data,
    split_indices,
)

SCHEMA = TableSchema(
    (
        NumericFeature("age", 0.0, 100.0),
        NumericFeature("glucose", 50.0, 300.0, missing=100.0),
        CategoricalFeature("smoker", ("never", "former", "current")),
    )
)


# --------------------------------------------------------------------------- encoding


def test_encoding_by_hand() -> None:
    table = {
        "age": [0, 50, 100, 250, -5, None],
        "glucose": [50, 300, "175", "n/a", float("nan"), 100],
        "smoker": ["never", "current", "former", "unknown", None, "never"],
    }
    encoded = SCHEMA.encode(table)
    assert encoded.dtype == np.float32
    assert encoded.shape == (6, 5)
    assert SCHEMA.width == 5
    assert SCHEMA.columns == ("age", "glucose", "smoker=never", "smoker=former", "smoker=current")
    # Out-of-range values are cut off; a missing age becomes the middle of the range.
    assert encoded[:, 0].tolist() == [-1.0, 0.0, 1.0, 1.0, -1.0, 0.0]
    # Missing glucose becomes the declared 100, which maps to -0.6.
    assert encoded[:, 1].tolist() == pytest.approx([-1.0, 1.0, 0.0, -0.6, -0.6, -0.6])
    assert encoded[:, 2:].tolist() == [
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 0, 0],  # not on the declared list
        [0, 0, 0],  # missing
        [1, 0, 0],
    ]


@settings(max_examples=30, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    n=st.integers(2, 30),
    extreme=st.floats(-1e12, 1e12),
)
def test_a_row_is_encoded_on_its_own(seed: int, n: int, extreme: float) -> None:
    """Changing or removing other rows never changes how a row is encoded."""
    rng = np.random.default_rng(seed)
    table = {
        "age": rng.uniform(-20, 140, n),
        "glucose": rng.uniform(0, 500, n),
        "smoker": rng.choice(["never", "former", "current", "other"], n),
    }
    whole = SCHEMA.encode(table)
    assert np.all(np.abs(whole) <= 1.0)
    alone = SCHEMA.encode({name: values[3 % n : 3 % n + 1] for name, values in table.items()})
    assert np.array_equal(whole[3 % n], alone[0])
    changed = {name: values.copy() for name, values in table.items()}
    changed["age"][0] = extreme
    changed["glucose"][0] = extreme
    assert np.array_equal(SCHEMA.encode(changed)[1:], whole[1:])


def test_encoding_accepts_a_dataframe_if_pandas_is_installed() -> None:
    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame(
        {"age": [10, 20], "glucose": [100.0, None], "smoker": ["never", "former"], "extra": [1, 2]}
    )
    encoded = SCHEMA.encode(frame)
    assert encoded.shape == (2, 5)
    assert encoded[1, 1] == pytest.approx(-0.6)


def test_bad_declarations_and_tables_are_refused() -> None:
    with pytest.raises(ValueError, match="upper must be greater"):
        NumericFeature("x", 1.0, 1.0)
    with pytest.raises(ValueError, match="inside the range"):
        NumericFeature("x", 0.0, 1.0, missing=2.0)
    with pytest.raises(ValueError, match="at least one category"):
        CategoricalFeature("c", ())
    with pytest.raises(ValueError, match="distinct"):
        CategoricalFeature("c", ("a", "a"))
    with pytest.raises(ValueError, match="at least one feature"):
        TableSchema(())
    with pytest.raises(ValueError, match="names must be distinct"):
        TableSchema((NumericFeature("x", 0, 1), CategoricalFeature("x", ("a",))))
    with pytest.raises(KeyError, match="no column 'glucose'"):
        SCHEMA.encode({"age": [1], "smoker": ["never"]})
    with pytest.raises(ValueError, match="same number of rows"):
        SCHEMA.encode({"age": [1, 2], "glucose": [100], "smoker": ["never", "never"]})
    assert CategoricalFeature("c", (1, 2)).categories == ("1", "2")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- splitting


@settings(max_examples=40, deadline=None)
@given(
    n=st.integers(1, 300),
    validation=st.floats(0.0, 0.45),
    test=st.floats(0.0, 0.45),
    seed=st.integers(0, 1000),
    positives=st.floats(0.0, 1.0),
)
def test_every_row_lands_in_exactly_one_part(
    n: int, validation: float, test: float, seed: int, positives: float
) -> None:
    labels = (np.random.default_rng(seed).random(n) < positives).astype(int)
    for stratify in (None, labels):
        train, val, held_out = split_indices(
            n, validation=validation, test=test, seed=seed, labels=stratify
        )
        together = np.concatenate([train, val, held_out])
        assert sorted(together.tolist()) == list(range(n))
        assert train.size >= 1
        if stratify is not None:
            for value in np.unique(labels):  # every class keeps at least one training row
                assert np.any(labels[train] == value)


def test_split_sizes_stratification_and_repeatability() -> None:
    labels = np.array([1] * 20 + [0] * 180)
    train, val, held_out = split_indices(200, validation=0.2, test=0.1, seed=7, labels=labels)
    assert (train.size, val.size, held_out.size) == (140, 40, 20)
    assert (labels[train].sum(), labels[val].sum(), labels[held_out].sum()) == (14, 4, 2)
    again = split_indices(200, validation=0.2, test=0.1, seed=7, labels=labels)
    assert all(np.array_equal(a, b) for a, b in zip((train, val, held_out), again, strict=True))
    other = split_indices(200, validation=0.2, test=0.1, seed=8, labels=labels)
    assert not np.array_equal(train, other[0])
    everything, none_a, none_b = split_indices(5, validation=0.0, test=0.0, seed=0)
    assert (everything.size, none_a.size, none_b.size) == (5, 0, 0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_rows": 0}, "at least one row"),
        ({"validation": 0.6, "test": 0.4}, "less than 1"),
        ({"validation": -0.1}, "non-negative"),
        ({"labels": [0, 1]}, "one label per row"),
    ],
)
def test_bad_splits_are_refused(kwargs: dict[str, object], message: str) -> None:
    arguments: dict[str, object] = {"n_rows": 10, "seed": 0, **kwargs}
    with pytest.raises(ValueError, match=message):
        split_indices(**arguments)  # type: ignore[arg-type]


def test_client_data_is_cut_and_checked() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 4))
    y = (rng.random(100) < 0.2).astype(int)
    client = make_client_data("clinic", x, y, validation=0.2, test=0.2, seed=1)
    assert client.name == "clinic"
    assert client.n_train == 60
    assert client.train_x.dtype == np.float32
    assert client.train_y.dtype == np.float32
    assert client.validation_x.shape == (20, 4)
    assert client.test_y.shape == (20,)
    rows = np.concatenate([client.train_x, client.validation_x, client.test_x])
    assert sorted(map(tuple, rows.tolist())) == sorted(map(tuple, x.astype(np.float32).tolist()))
    with pytest.raises(ValueError, match="one label per row"):
        make_client_data("clinic", x, y[:5], seed=1)
    empty = np.zeros((0, 4), dtype=np.float32)
    none = np.zeros(0, dtype=np.float32)
    with pytest.raises(ValueError, match="at least one training row"):
        ClientData("c", empty, none, empty, none, empty, none)
    with pytest.raises(ValueError, match="same columns"):
        ClientData("c", client.train_x, client.train_y, empty[:, :3], none, empty, none)
    with pytest.raises(ValueError, match="one label per row"):
        ClientData("c", client.train_x, client.train_y[:3], empty, none, empty, none)
