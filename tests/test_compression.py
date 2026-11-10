"""Rounding updates to whole numbers and packing them tightly."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from obscuradp.compression import (
    Quantizer,
    float32_size,
    pack_bits,
    packed_size,
    sum_bits,
    unpack_bits,
)


def test_bits_on_the_wire_match_the_reference_table() -> None:
    """Table 2 of docs/KPI_MAPPING.md: value bits plus room for the sum of N values."""
    assert sum_bits(8, 5) == 11
    assert sum_bits(12, 5) == 15
    assert sum_bits(12, 10) == 16
    assert sum_bits(8, 256) == 16
    assert sum_bits(12, 1) == 12
    assert sum_bits(12, 2) == 13
    # 12-bit values at 5 sites: 15 of 32 bits, under the 50% target. 16-bit values are not.
    assert sum_bits(12, 5) / 32 < 0.5 < sum_bits(16, 5) / 32
    with pytest.raises(ValueError, match="value_bits"):
        sum_bits(0, 5)
    with pytest.raises(ValueError, match="participants"):
        sum_bits(8, 0)


def test_the_sum_of_n_values_always_fits() -> None:
    for bits in (2, 8, 12):
        for n in (1, 2, 3, 5, 8, 9, 100):
            largest_sum = n * (2**bits - 1)
            assert largest_sum < 2 ** sum_bits(bits, n)


def test_quantizer_levels_by_hand() -> None:
    quantizer = Quantizer(bits=2, value_range=1.5)  # levels 0, 1, 2, 3 at -1.5, -0.5, 0.5, 1.5
    assert quantizer.levels == 3
    assert quantizer.step == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    exact = quantizer.quantize([-1.5, -0.5, 0.5, 1.5, 7.0, -7.0], rng)
    assert exact.tolist() == [0, 1, 2, 3, 3, 0]  # on a level: no randomness; outside: cut off
    assert exact.dtype == np.uint64
    assert quantizer.dequantize_sum(exact[:4], 1).tolist() == pytest.approx([-1.5, -0.5, 0.5, 1.5])
    # Halfway between two levels: each neighbour about half the time.
    draws = quantizer.quantize(np.zeros(20_000), rng)
    assert set(draws.tolist()) == {1, 2}
    assert float(np.mean(draws == 2)) == pytest.approx(0.5, abs=0.02)
    assert quantizer.saturated_share([0.0, 2.0, -3.0, 1.0]) == 0.5
    assert quantizer.saturated_share([]) == 0.0


def test_rounding_is_right_on_average_and_never_off_by_more_than_one_step() -> None:
    quantizer = Quantizer(bits=6, value_range=1.0)
    rng = np.random.default_rng(1)
    values = rng.uniform(-1.0, 1.0, size=500)
    repeats = np.stack(
        [quantizer.dequantize_sum(quantizer.quantize(values, rng), 1) for _ in range(400)]
    )
    assert float(np.abs(repeats - values).max()) <= quantizer.step * (1 + 1e-9)
    # The average of 400 draws is within 5 standard errors of the true value everywhere.
    standard_error = quantizer.step / 2 / np.sqrt(400)
    assert float(np.abs(repeats.mean(axis=0) - values).max()) < 5 * standard_error


def test_a_sum_of_quantized_updates_estimates_the_true_sum() -> None:
    quantizer = Quantizer(bits=12, value_range=0.5)
    rng = np.random.default_rng(2)
    updates = [rng.normal(0.0, 0.1, size=2000) for _ in range(5)]
    total = sum(quantizer.quantize(update, rng) for update in updates)
    estimate = quantizer.dequantize_sum(total, 5)
    assert float(np.abs(estimate - sum(updates)).max()) <= 5 * quantizer.step


@pytest.mark.parametrize(
    ("bits", "value_range", "message"),
    [
        (1, 1.0, "bits"),
        (25, 1.0, "bits"),
        (8, 0.0, "value_range"),
        (8, float("inf"), "value_range"),
    ],
)
def test_quantizer_refuses_bad_settings(bits: int, value_range: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Quantizer(bits, value_range)


def test_quantizer_refuses_bad_values() -> None:
    quantizer = Quantizer(8, 1.0)
    with pytest.raises(ValueError, match="finite"):
        quantizer.quantize([0.0, float("nan")], np.random.default_rng(0))
    with pytest.raises(ValueError, match="count"):
        quantizer.dequantize_sum([1, 2], 0)


@settings(max_examples=60, deadline=None)
@given(bits=st.integers(1, 62), length=st.integers(0, 200), seed=st.integers(0, 10_000))
def test_packing_round_trips_and_uses_exactly_the_stated_bytes(
    bits: int, length: int, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 2**bits, size=length, dtype=np.uint64)
    packed = pack_bits(values, bits)
    assert len(packed) == packed_size(length, bits) == -(-length * bits // 8)
    assert np.array_equal(unpack_bits(packed, bits, length), values)


def test_packing_edges() -> None:
    assert pack_bits([1, 0, 1, 1], 1) == bytes([0b1101])
    assert pack_bits([5, 3], 4) == bytes([0x35])
    assert packed_size(3137, 15) == 5882
    assert float32_size(3137) == 12548
    with pytest.raises(ValueError, match="does not fit in 4 bits"):
        pack_bits([16], 4)
    with pytest.raises(ValueError, match="wrong length"):
        unpack_bits(b"\x00\x00", 4, 5)
    for bad in (0, 63):
        with pytest.raises(ValueError, match="bits must be between"):
            pack_bits([1], bad)
    with pytest.raises(ValueError, match="length"):
        packed_size(-1, 8)
