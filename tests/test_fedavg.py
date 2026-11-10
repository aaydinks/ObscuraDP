"""Averaging participants' states in the clear."""

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from obscuradp.aggregation import aggregate_states, fedavg_weights, weighted_mean
from obscuradp.models import TabularMLP
from obscuradp.state import flatten_state, get_state


def test_weights_follow_row_counts() -> None:
    w = fedavg_weights([10, 30, 60])
    assert np.allclose(w, [0.1, 0.3, 0.6])
    assert w.dtype == np.float64
    assert w.sum() == pytest.approx(1.0)


def test_unweighted_ignores_row_counts() -> None:
    assert np.array_equal(fedavg_weights([1, 1000, 5], weighted=False), np.full(3, 1 / 3))


@pytest.mark.parametrize("counts", [[], [0, 0], [3, -1]])
def test_bad_counts_are_refused(counts: list[float]) -> None:
    with pytest.raises(ValueError, match="counts"):
        fedavg_weights(counts)


def test_weighted_mean_by_hand() -> None:
    vectors = [np.array([0.0, 10.0]), np.array([10.0, 0.0])]
    assert np.allclose(weighted_mean(vectors), [5.0, 5.0])
    assert np.allclose(weighted_mean(vectors, [3, 1]), [2.5, 7.5])


def test_weights_are_rescaled() -> None:
    vectors = [np.array([1.0]), np.array([3.0])]
    assert np.allclose(weighted_mean(vectors, [2, 2]), weighted_mean(vectors, [0.5, 0.5]))


def test_weighted_mean_needs_matching_weights() -> None:
    with pytest.raises(ValueError, match="one weight per vector"):
        weighted_mean([np.zeros(2), np.zeros(2)], [1.0])
    with pytest.raises(ValueError, match="at least one vector"):
        weighted_mean([])


def test_aggregate_states_matches_the_float64_mean_stored_as_float32() -> None:
    torch.manual_seed(0)
    states = [get_state(TabularMLP(6, (5, 3))) for _ in range(4)]
    counts = [10, 20, 30, 40]
    out = aggregate_states(states, counts)
    reference = weighted_mean([flatten_state(s) for s in states], fedavg_weights(counts))
    assert list(out) == list(states[0])
    assert all(v.dtype == torch.float32 for v in out.values())
    assert np.array_equal(flatten_state(out), reference.astype(np.float32).astype(np.float64))


def test_identical_states_average_to_themselves() -> None:
    torch.manual_seed(1)
    state = get_state(TabularMLP(4, (3,)))
    out = aggregate_states([state, state, state], [1, 2, 3])
    for key in state:
        assert torch.allclose(out[key], state[key], atol=1e-7)


def test_aggregate_states_checks_its_inputs() -> None:
    state = get_state(TabularMLP(2, (2,)))
    with pytest.raises(ValueError, match="at least one state"):
        aggregate_states([], [])
    with pytest.raises(ValueError, match="one count per state"):
        aggregate_states([state, state], [1])


@settings(max_examples=50, deadline=None)
@given(
    values=st.lists(st.lists(st.floats(-1e3, 1e3), min_size=3, max_size=3), min_size=1, max_size=6),
    data=st.data(),
)
def test_mean_stays_between_the_extremes(values: list[list[float]], data: st.DataObject) -> None:
    weights = data.draw(
        st.lists(st.floats(0.01, 100.0), min_size=len(values), max_size=len(values))
    )
    vectors = [np.array(v) for v in values]
    mean = weighted_mean(vectors, weights)
    stacked = np.stack(vectors)
    assert np.all(mean >= stacked.min(axis=0) - 1e-9)
    assert np.all(mean <= stacked.max(axis=0) + 1e-9)
