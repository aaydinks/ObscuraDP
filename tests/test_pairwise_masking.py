"""The default secure sum: exact totals, unreadable single messages, no secrets on the server."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from obscuradp.compression import float32_size, unpack_bits
from obscuradp.crypto import (
    MaskingAggregator,
    MaskingClient,
    MissingParticipants,
    PairwiseMasking,
    setup_pairwise_masking,
)

NAMES = ("clinic-a", "clinic-b", "clinic-c", "clinic-d")


def _values(length: int, bits: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {name: rng.integers(0, 2**bits, size=length, dtype=np.uint64) for name in NAMES}


@settings(max_examples=40, deadline=None)
@given(
    participants=st.integers(2, 7),
    bits=st.integers(1, 62),
    length=st.integers(0, 60),
    seed=st.integers(0, 10_000),
    sum_index=st.integers(0, 2**40),
)
def test_the_masks_cancel_and_the_sum_is_exact(
    participants: int, bits: int, length: int, seed: int, sum_index: int
) -> None:
    names = [f"p{i}" for i in range(participants)]
    clients, aggregator = setup_pairwise_masking(names, bits)
    rng = np.random.default_rng(seed)
    values = {name: rng.integers(0, 2**bits, size=length, dtype=np.uint64) for name in names}
    messages = {name: clients[name].protect(sum_index, values[name]) for name in names}
    expected = np.zeros(length, dtype=np.uint64)
    for vector in values.values():
        expected = expected + vector
    expected = expected & np.uint64(2**bits - 1)  # the sum modulo 2**bits
    assert np.array_equal(aggregator.combine(sum_index, messages, length), expected)


def test_a_single_message_looks_like_random_numbers() -> None:
    """All-zero input, 4-bit numbers: each of the 16 values should appear equally often."""
    clients, _ = setup_pairwise_masking(NAMES, 4)
    length = 40_000
    message = clients["clinic-b"].protect(0, np.zeros(length, dtype=np.uint64))
    seen = unpack_bits(message, 4, length)
    counts = np.bincount(seen.astype(np.int64), minlength=16)
    expected = length / 16
    chi_square = float(((counts - expected) ** 2 / expected).sum())
    # 15 degrees of freedom: 60 is exceeded by truly random numbers about once in 4 million runs.
    assert chi_square < 60.0
    assert counts.min() > 0


def test_messages_do_not_repeat_and_do_not_match_between_participants() -> None:
    clients, _ = setup_pairwise_masking(NAMES, 16)
    values = np.arange(500, dtype=np.uint64)
    first = clients["clinic-a"].protect(0, values)
    second = clients["clinic-a"].protect(1, values)  # same numbers, another sum: new masks
    other = clients["clinic-b"].protect(0, values)
    plain = unpack_bits(first, 16, 500)
    assert first != second
    assert first != other
    assert float(np.mean(plain == values)) < 0.01


def test_a_sum_index_is_never_used_twice() -> None:
    clients, _ = setup_pairwise_masking(NAMES, 16)
    clients["clinic-a"].protect(7, [1, 2, 3])
    with pytest.raises(ValueError, match="masks are never reused"):
        clients["clinic-a"].protect(7, [4, 5, 6])
    clients["clinic-a"].protect(
        8, [4, 5, 6]
    )  # a fresh index is fine: this is how a round is retried


def test_without_every_participant_nothing_can_be_opened() -> None:
    clients, aggregator = setup_pairwise_masking(NAMES, 16)
    values = _values(300, 12, seed=0)
    messages = {name: clients[name].protect(0, values[name]) for name in NAMES}
    partial = {name: messages[name] for name in NAMES[:3]}
    with pytest.raises(MissingParticipants, match=r"missing: \['clinic-d'\]"):
        aggregator.combine(0, partial, 300)
    with pytest.raises(MissingParticipants, match="unexpected"):
        aggregator.combine(0, {**messages, "stranger": messages["clinic-a"]}, 300)
    # Adding three of the four messages by hand gives noise, not the sum of those three.
    added = np.zeros(300, dtype=np.uint64)
    truth = np.zeros(300, dtype=np.uint64)
    for name in NAMES[:3]:
        added = (added + unpack_bits(messages[name], 16, 300)) & np.uint64(2**16 - 1)
        truth = truth + values[name]
    assert float(np.mean(added == truth)) < 0.01


def test_the_aggregator_holds_no_secret() -> None:
    clients, aggregator = setup_pairwise_masking(NAMES, 16)
    assert set(vars(aggregator)) == {"participants", "modulus_bits"}
    assert isinstance(aggregator, MaskingAggregator)
    # The only bytes a participant publishes are its public key; its secrets stay inside.
    client = clients["clinic-a"]
    assert len(client.public_key()) == 32
    assert client.public_key() != clients["clinic-b"].public_key()
    assert not any(isinstance(value, bytes) for value in vars(aggregator).values())


def test_both_ends_of_a_pair_derive_the_same_secret_and_pairs_differ() -> None:
    clients, _ = setup_pairwise_masking(NAMES, 16)
    a, b, c = (clients[name] for name in NAMES[:3])
    assert a._pair_keys["clinic-b"] == b._pair_keys["clinic-a"]
    assert a._pair_keys["clinic-b"] != a._pair_keys["clinic-c"]
    assert a._pair_keys["clinic-c"] == c._pair_keys["clinic-a"]
    assert set(a._pair_keys) == {"clinic-b", "clinic-c", "clinic-d"}
    # A fresh setup gives fresh secrets.
    again, _ = setup_pairwise_masking(NAMES, 16)
    assert again["clinic-a"]._pair_keys["clinic-b"] != a._pair_keys["clinic-b"]


def test_misuse_is_refused() -> None:
    lonely = MaskingClient("clinic-a", 16)
    with pytest.raises(RuntimeError, match="receive_public_keys"):
        lonely.protect(0, [1])
    with pytest.raises(ValueError, match="must include this participant"):
        lonely.receive_public_keys({"x": bytes(32), "y": bytes(32)})
    clients, aggregator = setup_pairwise_masking(NAMES, 8)
    with pytest.raises(ValueError, match="does not fit in 8 bits"):
        clients["clinic-a"].protect(0, [256])
    with pytest.raises(ValueError, match="sum_index"):
        clients["clinic-a"].protect(-1, [1])
    messages = {name: clients[name].protect(1, [1, 2]) for name in NAMES}
    messages["clinic-d"] = clients["clinic-d"].protect(2, [1, 2, 3])
    with pytest.raises(ValueError, match="must be 2 bytes for 2 numbers"):
        aggregator.combine(1, messages, 2)
    for names, message in (
        (["only"], "at least two"),
        (["a", "a"], "distinct"),
        (["a", "b|c"], "must not contain"),
        (["a", ""], "non-empty"),
    ):
        with pytest.raises(ValueError, match=message):
            setup_pairwise_masking(names, 16)
    for bad in (0, 63):
        with pytest.raises(ValueError, match="modulus_bits"):
            setup_pairwise_masking(NAMES, bad)


def test_settings_object_reports_sizes_and_protection() -> None:
    masking = PairwiseMasking(value_bits=12, value_range=0.5)
    assert masking.modulus_bits(5) == 15
    assert masking.message_bytes(3137, 5) == 5882
    # The bytes target: under half of the same update as 32-bit numbers.
    assert masking.message_bytes(3137, 5) / float32_size(3137) == pytest.approx(0.469, abs=0.001)
    assert masking.setup_bytes(5) == 160
    assert masking.unsafe_code is None
    clients, aggregator = masking.build(NAMES)
    assert aggregator.modulus_bits == 14
    assert set(clients) == set(NAMES)
    text = " ".join(masking.protection(5))
    assert "no single one" in text
    assert "up to 3 of the 5 participants" in text
    assert "must be repeated" in text
    assert "up to 0 of the 2" in " ".join(masking.protection(2))
    for kwargs, message in (
        ({"value_bits": 1}, "value_bits"),
        ({"value_range": 0.0}, "value_range"),
    ):
        with pytest.raises(ValueError, match=message):
            PairwiseMasking(**kwargs)  # type: ignore[arg-type]
