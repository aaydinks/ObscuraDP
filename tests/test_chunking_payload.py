"""Cutting vectors into ciphertext-sized pieces, and the size arithmetic that follows."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from obscuradp.bench import payload_ratio_asymptote, payload_ratio_model
from obscuradp.crypto import chunk_bounds, chunk_count


def test_chunks_by_hand() -> None:
    assert list(chunk_bounds(10, 4)) == [(0, 4), (4, 8), (8, 10)]
    assert list(chunk_bounds(8, 4)) == [(0, 4), (4, 8)]
    assert list(chunk_bounds(0, 4)) == []
    assert chunk_count(10, 4) == 3
    assert chunk_count(8, 4) == 2
    assert chunk_count(0, 4) == 0
    assert chunk_count(1, 4096) == 1


@pytest.mark.parametrize("slots", [0, -3])
def test_bad_slot_counts_are_refused(slots: int) -> None:
    with pytest.raises(ValueError, match="slots"):
        list(chunk_bounds(5, slots))
    with pytest.raises(ValueError, match="slots"):
        chunk_count(5, slots)
    with pytest.raises(ValueError, match="slots"):
        payload_ratio_asymptote(100.0, slots)


@given(n=st.integers(0, 50_000), slots=st.integers(1, 5000))
def test_chunks_cover_the_range_exactly(n: int, slots: int) -> None:
    bounds = list(chunk_bounds(n, slots))
    assert len(bounds) == chunk_count(n, slots)
    position = 0
    for start, end in bounds:
        assert start == position
        assert 0 < end - start <= slots
        position = end
    assert position == n


def test_payload_ratio_reproduces_the_measured_research_values() -> None:
    """One 4096-slot ciphertext serializes to 235,359 bytes in the research setting."""
    quantum, slots = 235_359, 4096
    assert payload_ratio_model(3137, 3137, quantum, slots) == pytest.approx(18.757, abs=1e-3)
    assert payload_ratio_model(929, 929, quantum, slots) == pytest.approx(63.3, abs=0.1)
    assert payload_ratio_model(2_131_969, 2_131_969, quantum, slots) == pytest.approx(
        14.379, abs=1e-3
    )
    assert payload_ratio_asymptote(quantum, slots) == pytest.approx(14.365, abs=1e-3)


def test_small_models_pay_for_a_whole_ciphertext_at_any_fraction() -> None:
    quantum, slots = 235_359, 4096
    ratios = {payload_ratio_model(k, 3137, quantum, slots) for k in (1, 314, 784, 1568, 3137)}
    assert len(ratios) == 1
    assert payload_ratio_model(0, 3137, quantum, slots) == 0.0
    assert payload_ratio_model(10, 0, quantum, slots) == 0.0


@given(n=st.integers(1, 10_000_000), slots=st.integers(1, 8192))
def test_ratio_is_never_below_the_asymptote(n: int, slots: int) -> None:
    quantum = 235_359.0
    assert (
        payload_ratio_model(n, n, quantum, slots) >= payload_ratio_asymptote(quantum, slots) - 1e-9
    )
