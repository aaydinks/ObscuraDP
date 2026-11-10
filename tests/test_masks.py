"""Masks over a model's flat vector."""

import math

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from obscuradp.models import TabularMLP
from obscuradp.selection import (
    build_mask,
    mask_agreement,
    random_jaccard_expected,
    random_mask,
    score_rank_correlation,
    tensor_selection,
    tensor_selection_agreement,
)
from obscuradp.state import get_state, state_layout, state_size


def _state() -> dict[str, torch.Tensor]:
    return {"a": torch.zeros(2, 3), "b": torch.zeros(10), "c": torch.zeros(4)}  # 6 + 10 + 4


def test_mask_takes_whole_high_score_tensors_first() -> None:
    state = _state()
    mask = build_mask(state, {"a": 0.1, "b": 0.5, "c": 0.9}, fraction=0.7)  # 14 of 20
    layout = {e.name: e for e in state_layout(state)}
    assert mask.dtype == bool
    assert mask.sum() == 14
    assert mask[layout["c"].start : layout["c"].stop].all()
    assert mask[layout["b"].start : layout["b"].stop].all()
    assert not mask[layout["a"].start : layout["a"].stop].any()


@pytest.mark.parametrize(
    ("fraction", "expected"), [(0.0, 0), (0.25, 5), (0.5, 10), (1.0, 20), (1.5, 20)]
)
def test_mask_size_follows_the_fraction(fraction: float, expected: int) -> None:
    mask = build_mask(_state(), {"a": 1.0, "b": 2.0, "c": 3.0}, fraction)
    assert mask.sum() == expected


def test_missing_scores_count_as_zero() -> None:
    state = _state()
    mask = build_mask(state, {"c": 1.0}, fraction=0.2)  # 4 of 20: exactly tensor c
    layout = {e.name: e for e in state_layout(state)}
    assert mask[layout["c"].start : layout["c"].stop].all()
    assert mask.sum() == 4


def test_tensor_selection_orders_by_score_then_name() -> None:
    state = _state()
    assert tensor_selection(state, {"a": 0.1, "b": 0.5, "c": 0.9}, 0.2) == ("c",)
    assert tensor_selection(state, {"a": 0.1, "b": 0.5, "c": 0.9}, 0.5) == ("c", "b")
    assert tensor_selection(state, {"a": 1.0, "b": 1.0, "c": 1.0}, 0.5) == ("a", "b")
    assert tensor_selection(state, {"a": 1.0}, 0.0) == ()


def test_tensor_selection_agreement() -> None:
    state = _state()
    same = tensor_selection_agreement(state, {"c": 2.0, "b": 1.0}, {"c": 9.0, "b": 3.0}, 0.5)
    assert same == {"same_selection": 1.0, "tensor_jaccard": 1.0, "n_a": 2, "n_b": 2, "n_both": 2}
    differ = tensor_selection_agreement(state, {"c": 2.0, "b": 1.0}, {"a": 9.0, "b": 3.0}, 0.5)
    assert differ["same_selection"] == 0.0
    assert differ["tensor_jaccard"] == pytest.approx(1 / 3)


def test_random_mask_is_seeded_and_sized() -> None:
    a, b, c = random_mask(1000, 0.25, 1), random_mask(1000, 0.25, 1), random_mask(1000, 0.25, 2)
    assert a.sum() == 250
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert random_mask(10, 0.0, 0).sum() == 0
    assert random_mask(10, 1.0, 0).all()


def test_random_jaccard_matches_simulation() -> None:
    assert random_jaccard_expected(0.25) == pytest.approx(1 / 7)
    assert random_jaccard_expected(1.0) == 1.0
    fixed = np.zeros(4000, dtype=bool)
    fixed[:1000] = True
    measured = np.mean(
        [mask_agreement(fixed, random_mask(4000, 0.25, s))["jaccard"] for s in range(200)]
    )
    assert measured == pytest.approx(1 / 7, abs=0.01)


def test_mask_agreement_by_hand() -> None:
    a = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    b = np.array([0, 1, 1, 1, 0, 0], dtype=bool)
    out = mask_agreement(a, b)
    assert out == {"jaccard": 0.5, "overlap": pytest.approx(2 / 3), "n_a": 3, "n_b": 3, "n_both": 2}
    empty = np.zeros(4, dtype=bool)
    assert mask_agreement(empty, empty)["jaccard"] == 1.0
    assert mask_agreement(empty, ~empty)["overlap"] == 0.0
    with pytest.raises(ValueError, match="same shape"):
        mask_agreement(a, empty)


def test_rank_correlation() -> None:
    up = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    assert score_rank_correlation(
        up, {"a": 10.0, "b": 20.0, "c": 35.0, "d": 90.0}
    ) == pytest.approx(1.0)
    assert score_rank_correlation(up, {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}) == pytest.approx(
        -1.0
    )
    assert math.isnan(score_rank_correlation(up, {"a": 1.0, "b": 2.0}))
    assert math.isnan(score_rank_correlation(up, {"a": 5.0, "b": 5.0, "c": 5.0, "d": 5.0}))
    # ties share the average rank: ranks (1.5, 1.5, 3, 4) against (1, 2, 3, 4)
    tied = score_rank_correlation({"a": 1.0, "b": 1.0, "c": 2.0, "d": 3.0}, up)
    assert tied == pytest.approx(4.5 / math.sqrt(4.5 * 5.0))


def test_mask_on_a_real_model_has_the_right_length() -> None:
    torch.manual_seed(0)
    state = get_state(TabularMLP(15, (64, 32)))
    scores = {name: float(i) for i, name in enumerate(state)}
    mask = build_mask(state, scores, 0.25)
    assert mask.shape == (state_size(state),) == (3137,)
    assert mask.sum() == round(0.25 * 3137)


@settings(max_examples=60, deadline=None)
@given(
    sizes=st.lists(st.integers(1, 30), min_size=1, max_size=6),
    fraction=st.floats(0.0, 1.0),
    seed=st.integers(0, 10_000),
)
def test_mask_properties(sizes: list[int], fraction: float, seed: int) -> None:
    state = {f"t{i}": torch.zeros(n) for i, n in enumerate(sizes)}
    rng = np.random.default_rng(seed)
    scores = {name: float(rng.random()) for name in state}
    n_total = sum(sizes)
    mask = build_mask(state, scores, fraction)
    assert mask.sum() == min(n_total, round(fraction * n_total))
    # Every tensor named by tensor_selection contributes at least one marked entry, and no
    # tensor outside it contributes any (scores are distinct with probability one).
    chosen = set(tensor_selection(state, scores, fraction))
    for entry in state_layout(state):
        marked = int(mask[entry.start : entry.stop].sum())
        assert (marked > 0) == (entry.name in chosen)
