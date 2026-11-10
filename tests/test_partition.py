"""Splitting rows among simulated participants."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from obscuradp.partition import dirichlet_partition, iid_partition


def _labels(n: int = 600, positive_share: float = 0.2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(n) < positive_share).astype(np.int64)


def _assert_is_a_partition(parts: list[np.ndarray], n: int) -> None:
    joined = np.concatenate(parts)
    assert joined.size == n
    assert np.array_equal(np.sort(joined), np.arange(n))


def test_iid_covers_every_row_once_with_near_equal_sizes() -> None:
    parts, info = iid_partition(103, 5, seed=42)
    _assert_is_a_partition(parts, 103)
    assert info.kind == "iid"
    assert info.sizes == tuple(len(p) for p in parts)
    assert max(info.sizes) - min(info.sizes) <= 1


def test_iid_is_deterministic_and_seed_dependent() -> None:
    a, _ = iid_partition(200, 4, seed=1)
    b, _ = iid_partition(200, 4, seed=1)
    c, _ = iid_partition(200, 4, seed=2)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c, strict=True))


def test_dirichlet_covers_every_row_once_and_reports_class_counts() -> None:
    y = _labels()
    parts, info = dirichlet_partition(y, 5, alpha=0.5, seed=42, min_per_class=5)
    _assert_is_a_partition(parts, y.size)
    assert info.kind == "dirichlet"
    assert info.attempts >= 1
    assert info.sizes == tuple(len(p) for p in parts)
    for label in (0, 1):
        counts = tuple(int((y[p] == label).sum()) for p in parts)
        assert info.class_counts[label] == counts
        assert min(counts) >= 5
        assert sum(counts) == int((y == label).sum())


def test_dirichlet_is_deterministic_and_seed_dependent() -> None:
    y = _labels()
    a, _ = dirichlet_partition(y, 4, alpha=0.5, seed=7)
    b, _ = dirichlet_partition(y, 4, alpha=0.5, seed=7)
    c, _ = dirichlet_partition(y, 4, alpha=0.5, seed=8)
    assert all(np.array_equal(x, z) for x, z in zip(a, b, strict=True))
    assert not all(np.array_equal(x, z) for x, z in zip(a, c, strict=True))


def test_small_alpha_gives_more_uneven_class_mixes_than_large_alpha() -> None:
    y = _labels(4000, 0.5, seed=3)

    def spread(alpha: float) -> float:
        values = []
        for seed in range(10):
            parts, _ = dirichlet_partition(y, 5, alpha=alpha, seed=seed)
            values.append(np.std([y[p].mean() for p in parts if p.size]))
        return float(np.mean(values))

    assert spread(0.1) > spread(100.0) * 3


def test_impossible_floor_raises_instead_of_looping_forever() -> None:
    y = np.array([0] * 50 + [1] * 3)
    with pytest.raises(RuntimeError, match="enough rows of every class"):
        dirichlet_partition(y, 5, alpha=0.5, seed=0, min_per_class=2, max_attempts=20)


@pytest.mark.parametrize("bad", [0, -1])
def test_bad_arguments_are_refused(bad: int) -> None:
    with pytest.raises(ValueError, match="num_clients"):
        iid_partition(10, bad, seed=0)
    with pytest.raises(ValueError, match="num_clients"):
        dirichlet_partition(_labels(50), bad, alpha=0.5, seed=0)
    with pytest.raises(ValueError, match="alpha"):
        dirichlet_partition(_labels(50), 2, alpha=float(bad), seed=0)


def test_more_participants_than_rows_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        iid_partition(3, 5, seed=0)


@settings(max_examples=40, deadline=None)
@given(
    n=st.integers(20, 300),
    clients=st.integers(1, 8),
    alpha=st.floats(0.05, 50.0),
    seed=st.integers(0, 10_000),
)
def test_dirichlet_partition_property(n: int, clients: int, alpha: float, seed: int) -> None:
    y = _labels(n, 0.3, seed)
    parts, info = dirichlet_partition(y, clients, alpha=alpha, seed=seed)
    assert len(parts) == clients
    _assert_is_a_partition(parts, n)
    assert sum(info.sizes) == n
