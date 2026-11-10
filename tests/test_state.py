"""Flattening a model's state into one vector and back."""

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import nn

from obscuradp.models import TabularMLP
from obscuradp.state import (
    count_parameters,
    flatten_state,
    get_state,
    set_state,
    state_layout,
    state_size,
    unflatten_state,
)


def test_round_trip_is_exact_for_float32_models() -> None:
    torch.manual_seed(0)
    state = get_state(TabularMLP(15, (64, 32)))
    flat = flatten_state(state)
    assert flat.dtype == np.float64
    assert flat.shape == (3137,)
    rebuilt = unflatten_state(flat, state)
    assert list(rebuilt) == list(state)
    for key in state:
        assert rebuilt[key].dtype == state[key].dtype
        assert torch.equal(rebuilt[key], state[key])


def test_layout_matches_flat_vector() -> None:
    torch.manual_seed(1)
    state = get_state(TabularMLP(11, (32, 16)))
    flat = flatten_state(state)
    layout = state_layout(state)
    assert [e.name for e in layout] == list(state)
    assert layout[0].start == 0
    assert layout[-1].stop == state_size(state) == flat.size == 929
    for entry in layout:
        expected = state[entry.name].numpy().ravel().astype(np.float64)
        assert np.array_equal(flat[entry.start : entry.stop], expected)
        assert entry.size == int(np.prod(entry.shape))


def test_buffers_are_part_of_the_state() -> None:
    """Running statistics of a normalization layer must travel with the weights (finding G20)."""
    model = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3))
    state = get_state(model)
    assert "1.running_mean" in state
    assert "1.num_batches_tracked" in state
    assert state_size(state) > count_parameters(model)

    state["1.running_mean"] += 0.25
    state["1.num_batches_tracked"] += 7
    rebuilt = unflatten_state(flatten_state(state), state)
    assert torch.equal(rebuilt["1.running_mean"], state["1.running_mean"])
    assert rebuilt["1.num_batches_tracked"].dtype == torch.int64
    assert int(rebuilt["1.num_batches_tracked"]) == 7


def test_integer_entries_are_rounded_not_truncated() -> None:
    template = {"steps": torch.tensor(0, dtype=torch.int64)}
    assert int(unflatten_state(np.array([6.6]), template)["steps"]) == 7


def test_get_state_strips_the_wrapper_prefix_and_copies() -> None:
    class Wrapper(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self._module = inner

    inner = nn.Linear(2, 1)
    state = get_state(Wrapper(inner))
    assert set(state) == {"weight", "bias"}
    state["weight"] += 1.0
    assert not torch.equal(state["weight"], inner.weight.detach())


def test_set_state_loads_values() -> None:
    torch.manual_seed(2)
    source, target = TabularMLP(5, (4,)), TabularMLP(5, (4,))
    set_state(target, get_state(source))
    for a, b in zip(source.parameters(), target.parameters(), strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("length", [0, 5, 100])
def test_wrong_length_is_refused(length: int) -> None:
    template = get_state(nn.Linear(3, 2))  # 8 numbers
    with pytest.raises(ValueError, match="template needs 8"):
        unflatten_state(np.zeros(length), template)


def test_empty_state() -> None:
    assert flatten_state({}).shape == (0,)
    assert unflatten_state(np.zeros(0), {}) == {}


@settings(max_examples=50, deadline=None)
@given(
    shapes=st.lists(
        st.lists(st.integers(1, 4), min_size=0, max_size=3).map(tuple), min_size=1, max_size=5
    ),
    seed=st.integers(0, 2**31 - 1),
)
def test_round_trip_property(shapes: list[tuple[int, ...]], seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    state = {f"t{i}": torch.randn(shape, generator=generator) for i, shape in enumerate(shapes)}
    flat = flatten_state(state)
    assert flat.size == sum(int(np.prod(s)) for s in shapes)
    rebuilt = unflatten_state(flat, state)
    for key, value in state.items():
        assert rebuilt[key].shape == value.shape
        assert torch.equal(rebuilt[key], value)
