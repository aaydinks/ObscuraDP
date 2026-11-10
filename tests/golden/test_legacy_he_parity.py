"""Encrypted aggregation: the copy behaves like the research code.

Encryption uses fresh randomness that cannot be seeded, so two encrypted runs never agree
bit for bit. What must hold instead: everything structural is exactly equal, entries that
are averaged in the clear are exactly equal, and decrypted entries agree with each other
and with the plain average far inside the tolerance the research code set for itself.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import numpy as np
import pytest
import torch

from obscuradp.aggregation import fedavg_weights, weighted_mean
from obscuradp.bench import payload_ratio_asymptote, payload_ratio_model
from obscuradp.legacy import LegacyTrialConfig
from obscuradp.legacy import aggregate as new_aggregate
from obscuradp.legacy import bench as new_bench
from obscuradp.legacy import he as new_he
from obscuradp.legacy import models as new_models
from obscuradp.state import flatten_state, get_state

pytestmark = [pytest.mark.integration, pytest.mark.filterwarnings("ignore")]

EXACT_KEYS = ("n_chunks", "n_encrypted", "encrypted_fraction", "n_params", "plaintext_bytes")
STROKE_COUNTS = [484, 234, 1120, 4920, 738]
DECRYPT_TOLERANCE = 1e-6


def _quiet(function: Any, *args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


@pytest.fixture(scope="module")
def ctx(cfg: LegacyTrialConfig) -> Any:
    return _quiet(new_he.make_context, cfg)


@pytest.fixture(scope="module")
def vectors() -> list[np.ndarray[Any, Any]]:
    rng = np.random.default_rng(0)
    return [rng.normal(0.0, 0.1, 3137) for _ in range(5)]


@pytest.fixture(scope="module")
def mask() -> np.ndarray[Any, Any]:
    chosen = np.zeros(3137, dtype=bool)
    chosen[np.random.default_rng(1).choice(3137, size=784, replace=False)] = True
    return chosen


def _close_bytes(a: float, b: float) -> bool:
    return abs(a - b) <= 0.001 * max(a, b)


# --------------------------------------------------------------------------- context


def test_context_has_the_research_parameters(oracle: Any, cfg: LegacyTrialConfig, ctx: Any) -> None:
    import tenseal as ts

    old_ctx = oracle.call(oracle.module("he").make_context)
    assert ctx.global_scale == old_ctx.global_scale == 2**40
    # The secret key is inside the one shared object (finding G5).
    assert ctx.is_private()
    assert old_ctx.is_private()
    one = ts.ckks_vector(ctx, [1.0] * 4096)
    assert len(one.decrypt()) == 4096


def test_the_modulus_chain_allows_exactly_one_multiplication(
    cfg: LegacyTrialConfig, ctx: Any
) -> None:
    """Weighted aggregation fits with zero headroom, as the research write-up states."""
    encrypted = new_he.encrypt_chunk(ctx, np.full(8, 0.5))
    once = encrypted * 0.25
    assert np.allclose(once.decrypt()[:8], 0.125, atol=DECRYPT_TOLERANCE)
    with pytest.raises(ValueError, match="scale out of bounds"):
        _ = once * 0.5


# --------------------------------------------------------------------------- aggregation


@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("slots", [None, 1024])
def test_timed_aggregation_matches(
    oracle: Any,
    cfg: LegacyTrialConfig,
    ctx: Any,
    vectors: list[Any],
    mask: Any,
    weighted: bool,
    slots: int | None,
) -> None:
    old = oracle.module("he")
    weights = fedavg_weights(STROKE_COUNTS, weighted=weighted) if weighted else None
    reference = weighted_mean(vectors, weights)

    old_out, old_stats = old.fedavg_he_timed(vectors, mask, ctx, weights=weights, slots=slots)
    new_out, new_stats = new_he.fedavg_he_timed(
        cfg, vectors, mask, ctx, weights=weights, slots=slots
    )

    assert set(old_stats) == set(new_stats)
    for key in EXACT_KEYS:
        assert old_stats[key] == new_stats[key], key
    assert _close_bytes(old_stats["ciphertext_bytes"], new_stats["ciphertext_bytes"])
    assert new_stats["n_chunks"] == (1 if slots is None else -(-784 // 1024))
    assert new_stats["n_encrypted"] == 784

    # Entries averaged in the clear are the same numbers, bit for bit.
    assert np.array_equal(old_out[~mask], new_out[~mask])
    assert np.array_equal(new_out[~mask], reference[~mask])
    # Decrypted entries agree with each other and with the plain average.
    assert np.max(np.abs(old_out[mask] - new_out[mask])) < DECRYPT_TOLERANCE
    assert np.max(np.abs(new_out - reference)) < DECRYPT_TOLERANCE

    # What the client really sends is more than the ciphertext while part stays in the clear.
    assert new_stats["wire_payload_ratio"] > new_stats["payload_ratio"]
    expected_wire = (new_stats["ciphertext_bytes"] + 4 * (3137 - 784)) / (4 * 3137)
    assert new_stats["wire_payload_ratio"] == pytest.approx(expected_wire)


def test_full_encryption_and_no_encryption(
    oracle: Any, cfg: LegacyTrialConfig, ctx: Any, vectors: list[Any]
) -> None:
    old = oracle.module("he")
    everything = np.ones(3137, dtype=bool)
    nothing = np.zeros(3137, dtype=bool)
    reference = weighted_mean(vectors)

    out, stats = new_he.fedavg_he_timed(cfg, vectors, everything, ctx)
    assert np.max(np.abs(out - reference)) < DECRYPT_TOLERANCE
    assert stats["wire_payload_ratio"] == stats["payload_ratio"]
    assert stats["encrypted_fraction"] == 1.0

    old_out, old_stats = old.fedavg_he_timed(vectors, nothing, ctx)
    new_out, new_stats = new_he.fedavg_he_timed(cfg, vectors, nothing, ctx)
    assert np.array_equal(old_out, new_out)
    assert old_stats == new_stats
    assert new_stats["wire_payload_ratio"] == 1.0
    assert new_stats["payload_ratio"] == 0.0


def test_untimed_paths_match(
    oracle: Any, cfg: LegacyTrialConfig, ctx: Any, vectors: list[Any], mask: Any
) -> None:
    old = oracle.module("he")
    weights = fedavg_weights(STROKE_COUNTS)
    reference = weighted_mean(vectors, weights)
    old_out = old.fedavg_he(vectors, mask, ctx, weights=weights)
    new_out = new_he.fedavg_he(cfg, vectors, mask, ctx, weights=weights)
    assert np.array_equal(old_out[~mask], new_out[~mask])
    assert np.max(np.abs(old_out - new_out)) < DECRYPT_TOLERANCE
    assert np.max(np.abs(new_out - reference)) < DECRYPT_TOLERANCE

    idx = np.where(mask)[0]
    old_sub = old.aggregate_indices(vectors, idx, ctx)
    new_sub = new_he.aggregate_indices(cfg, vectors, idx, ctx)
    assert np.max(np.abs(old_sub - new_sub)) < DECRYPT_TOLERANCE
    assert np.max(np.abs(new_sub - weighted_mean(vectors)[idx])) < DECRYPT_TOLERANCE

    assert np.array_equal(
        old.fedavg_plaintext(vectors, weights), new_he.fedavg_plaintext(vectors, weights)
    )


def test_unequal_weights_are_really_applied_under_encryption(
    cfg: LegacyTrialConfig, ctx: Any, vectors: list[Any]
) -> None:
    """The research smoke test only used equal weights, which cannot tell the two apart."""
    weights = fedavg_weights(STROKE_COUNTS)
    everything = np.ones(3137, dtype=bool)
    out, _ = new_he.fedavg_he_timed(cfg, vectors, everything, ctx, weights=weights)
    weighted_reference = weighted_mean(vectors, weights)
    unweighted_reference = weighted_mean(vectors)
    assert np.max(np.abs(out - weighted_reference)) < DECRYPT_TOLERANCE
    assert np.max(np.abs(out - unweighted_reference)) > 1e-2


def test_encrypt_vector_chunks_like_the_research_code(
    oracle: Any, cfg: LegacyTrialConfig, ctx: Any
) -> None:
    old = oracle.module("he")
    vec = np.linspace(-0.5, 0.5, 9000)
    for slots in (None, 1024):
        old_cts, _, old_bytes = old.encrypt_vector(ctx, vec, slots=slots)
        new_cts, _, new_bytes = new_he.encrypt_vector(cfg, ctx, vec, slots=slots)
        assert len(old_cts) == len(new_cts) == -(-9000 // (slots or 4096))
        assert _close_bytes(old_bytes, new_bytes)
        first = np.asarray(new_cts[0].decrypt()[:5])
        assert np.max(np.abs(first - vec[:5])) < DECRYPT_TOLERANCE
    assert list(old.chunk_bounds(9000)) == [(0, 4096), (4096, 8192), (8192, 9000)]


def test_the_longer_modulus_chain_of_experiment_2_also_works(
    cfg: LegacyTrialConfig, vectors: list[Any]
) -> None:
    """Experiment 2's smoke test: encrypted averaging within 1e-3 of the plain mean."""
    cfg2 = dataclasses.replace(cfg, ckks_coeff_mod_bit_sizes=(60, 40, 40, 60))
    ctx2 = _quiet(new_he.make_context, cfg2)
    short = [v[:929] for v in vectors]
    out, stats = new_he.fedavg_he_timed(cfg2, short, np.ones(929, dtype=bool), ctx2)
    assert np.max(np.abs(out - weighted_mean(short))) < 1e-3
    # 929 numbers in one padded ciphertext: the longer chain costs more bytes than the short one.
    _, short_chain = new_he.fedavg_he_timed(
        cfg, short, np.ones(929, dtype=bool), _quiet(new_he.make_context, cfg)
    )
    assert stats["ciphertext_bytes"] > short_chain["ciphertext_bytes"]


# --------------------------------------------------------------------------- the aggregation switch


@pytest.fixture(scope="module")
def states(cfg: LegacyTrialConfig) -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(5)
    return [get_state(new_models.new_model(cfg, 15, "stroke")) for _ in range(5)]


@pytest.mark.parametrize("weighted", [True, False])
def test_switch_in_plain_mode_is_identical(
    oracle: Any, cfg: LegacyTrialConfig, states: list[Any], weighted: bool
) -> None:
    old = oracle.module("aggregate")
    old_state, old_stats = old.aggregate_states(
        states, STROKE_COUNTS, mode="plaintext", weighted=weighted
    )
    new_state, new_stats = new_aggregate.aggregate_states(
        cfg, states, STROKE_COUNTS, mode="plaintext", weighted=weighted
    )
    for key in old_state:
        assert torch.equal(old_state[key], new_state[key])
    old_stats.pop("agg_sec"), new_stats.pop("agg_sec")
    assert old_stats == new_stats
    assert np.array_equal(
        old.plaintext_reference(states, STROKE_COUNTS, weighted=weighted),
        new_aggregate.plaintext_reference(states, STROKE_COUNTS, weighted=weighted),
    )


@pytest.mark.parametrize("mode", ["he", "spe"])
def test_switch_in_encrypted_modes_matches_the_plain_reference(
    oracle: Any, cfg: LegacyTrialConfig, ctx: Any, states: list[Any], mask: Any, mode: str
) -> None:
    """Both codes get the same recorded client states, so differences cannot compound."""
    old = oracle.module("aggregate")
    reference = new_aggregate.plaintext_reference(states, STROKE_COUNTS, weighted=False)
    old_state, old_stats = old.aggregate_states(
        states, STROKE_COUNTS, mode=mode, mask=mask, ctx=ctx, weighted=False
    )
    new_state, new_stats = new_aggregate.aggregate_states(
        cfg, states, STROKE_COUNTS, mode=mode, mask=mask, ctx=ctx, weighted=False
    )
    old_flat, new_flat = flatten_state(old_state), flatten_state(new_state)
    assert np.max(np.abs(old_flat - new_flat)) < DECRYPT_TOLERANCE
    assert np.max(np.abs(new_flat - reference)) < DECRYPT_TOLERANCE
    for key in EXACT_KEYS:
        assert old_stats[key] == new_stats[key], key
    assert new_stats["n_encrypted"] == (3137 if mode == "he" else 784)


def test_switch_falls_back_and_refuses_like_the_research_code(
    oracle: Any, cfg: LegacyTrialConfig, ctx: Any, states: list[Any]
) -> None:
    old = oracle.module("aggregate")
    for empty in (None, np.zeros(3137, dtype=bool)):
        old_state, old_stats = old.aggregate_states(
            states, STROKE_COUNTS, mode="spe", mask=empty, ctx=ctx
        )
        new_state, new_stats = new_aggregate.aggregate_states(
            cfg, states, STROKE_COUNTS, mode="spe", mask=empty, ctx=ctx
        )
        assert torch.equal(old_state["net.0.weight"], new_state["net.0.weight"])
        old_stats.pop("agg_sec"), new_stats.pop("agg_sec")
        assert old_stats == new_stats
        assert new_stats["wire_payload_ratio"] == 1.0
    for call in (
        lambda: old.aggregate_states(states, STROKE_COUNTS, mode="banana"),
        lambda: new_aggregate.aggregate_states(cfg, states, STROKE_COUNTS, mode="banana"),
    ):
        with pytest.raises(ValueError, match="Unknown mode"):
            call()
    for call in (
        lambda: old.aggregate_states(states, STROKE_COUNTS, mode="he"),
        lambda: new_aggregate.aggregate_states(cfg, states, STROKE_COUNTS, mode="he"),
    ):
        with pytest.raises(ValueError, match="requires ctx"):
            call()


# --------------------------------------------------------------------------- sizes and benchmarks


def test_ciphertext_size_explains_the_published_payload_ratios(
    oracle: Any, cfg: LegacyTrialConfig, ctx: Any
) -> None:
    old = oracle.module("bench")
    quantum = new_bench.ciphertext_quantum(cfg, ctx)
    assert _close_bytes(quantum, old.ciphertext_quantum(ctx))
    assert _close_bytes(quantum, 235_359)
    one_slot = new_he.serialized_size(new_he.encrypt_chunk(ctx, np.zeros(1)))
    assert abs(one_slot - quantum) < 0.001 * quantum  # a nearly empty ciphertext costs the same
    assert payload_ratio_model(3137, 3137, quantum, cfg.ckks_slots) == pytest.approx(
        18.757, abs=0.02
    )
    assert payload_ratio_asymptote(quantum, cfg.ckks_slots) == pytest.approx(14.365, abs=0.02)
    assert old.payload_ratio_model(784, 3137, 1000.0) == payload_ratio_model(
        784, 3137, 1000.0, cfg.ckks_slots
    )
    assert old.payload_ratio_model(9000, 9000, 1000.0, slots=1024) == payload_ratio_model(
        9000, 9000, 1000.0, 1024
    )


def test_benchmarks_report_the_same_structure(
    oracle: Any, cfg: LegacyTrialConfig, ctx: Any
) -> None:
    old = oracle.module("bench")
    for fraction in (0.25, 1.0):
        old_row = old.bench_encryption(10_369, ctx, fraction=fraction, repeats=1)
        new_row = new_bench.bench_encryption(cfg, 10_369, ctx, fraction=fraction, repeats=1)
        assert set(old_row) == set(new_row)
        for key in ("n_params", "fraction", "n_encrypted", "n_chunks", "plaintext_bytes"):
            assert old_row[key] == new_row[key], key
        assert _close_bytes(old_row["ciphertext_bytes"], new_row["ciphertext_bytes"])
        assert new_row["predicted_payload_ratio"] == pytest.approx(
            new_row["payload_ratio"], rel=1e-3
        )

    assert set(old.bench_plaintext(5000, repeats=1)) == set(
        new_bench.bench_plaintext(cfg, 5000, repeats=1)
    )

    old_sizes = old.bench_model_sizes([1, 2], "stroke", 15, ctx)
    new_sizes = new_bench.bench_model_sizes(cfg, [1, 2], "stroke", 15, ctx)
    assert [r["n_params"] for r in new_sizes] == [r["n_params"] for r in old_sizes] == [3137, 10369]
    assert [r["n_chunks"] for r in new_sizes] == [r["n_chunks"] for r in old_sizes] == [1.0, 3.0]

    old_tax = oracle.call(old.bench_dp_tax, "stroke", 15, 1, 1.0, 256, repeats=1)
    new_tax = _quiet(new_bench.bench_dp_tax, cfg, "stroke", 15, 1, 1.0, 256, repeats=1)
    assert set(old_tax) == set(new_tax)
    for key in ("n_params", "width_scale", "n_samples"):
        assert old_tax[key] == new_tax[key]
    assert new_tax["dp_multiplier"] > 0
