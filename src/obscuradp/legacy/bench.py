"""The Trial 07/27 overhead benchmarks (comparison only, see the package docstring).

Timing harnesses for encrypted aggregation and for private training. No model is trained to
convergence here; the numbers are costs, not accuracies.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from obscuradp.bench.payload import payload_ratio_model
from obscuradp.legacy import he, training
from obscuradp.legacy import models as legacy_models
from obscuradp.legacy.config import LegacyTrialConfig
from obscuradp.state import count_parameters, get_state


def ciphertext_quantum(cfg: LegacyTrialConfig, ctx: Any, slots: int | None = None) -> int:
    """Serialized bytes of one full ciphertext, measured, not assumed."""
    slots = slots or cfg.ckks_slots
    return he.serialized_size(he.encrypt_chunk(ctx, np.zeros(slots)))


def bench_encryption(
    cfg: LegacyTrialConfig,
    n_params: int,
    ctx: Any,
    fraction: float = 1.0,
    repeats: int | None = None,
) -> dict[str, Any]:
    """Median cost of encrypting, adding and decrypting synthetic updates of that size."""
    if repeats is None:
        repeats = cfg.overhead_repeats

    # Deterministic synthetic data: float64 vectors ~0.1 magnitude.
    # RandomState on purpose: same draws as the research code
    rng = np.random.RandomState(cfg.seed)
    client_vecs = [
        rng.normal(0.0, 0.1, n_params).astype(np.float64) for _ in range(cfg.num_clients)
    ]

    # Build mask: encrypt the first round(fraction * n_params) entries.
    n_encrypted = round(fraction * n_params)
    mask = np.zeros(n_params, dtype=bool)
    mask[:n_encrypted] = True

    all_stats = []
    for _ in range(repeats):
        _, stats = he.fedavg_he_timed(cfg, client_vecs, mask, ctx)
        all_stats.append(stats)

    median_stats: dict[str, Any] = {}
    for key in all_stats[0]:
        values = [s[key] for s in all_stats]
        if isinstance(values[0], (int, float)):
            median_stats[key] = float(np.median(values))
        else:
            median_stats[key] = values[0]

    us_per_param = 1e6 * median_stats["encrypt_sec"] / max(1, n_encrypted)
    n_chunks = median_stats["n_chunks"]
    bytes_per_ciphertext = median_stats["ciphertext_bytes"] / n_chunks if n_chunks else 0.0

    return {
        "n_params": n_params,
        "fraction": fraction,
        "n_encrypted": n_encrypted,
        "encrypt_sec": median_stats["encrypt_sec"],
        "aggregate_sec": median_stats["aggregate_sec"],
        "decrypt_sec": median_stats["decrypt_sec"],
        "crypto_sec": median_stats["crypto_sec"],
        "ciphertext_bytes": median_stats["ciphertext_bytes"],
        "plaintext_bytes": median_stats["plaintext_bytes"],
        "payload_ratio": median_stats["payload_ratio"],
        "wire_payload_ratio": median_stats["wire_payload_ratio"],
        "predicted_payload_ratio": payload_ratio_model(
            n_encrypted, n_params, bytes_per_ciphertext, cfg.ckks_slots
        ),
        "bytes_per_ciphertext": float(bytes_per_ciphertext),
        "n_chunks": n_chunks,
        "us_per_param": us_per_param,
    }


def bench_plaintext(
    cfg: LegacyTrialConfig, n_params: int, repeats: int | None = None
) -> dict[str, Any]:
    """Median time of the plain mean of synthetic updates of ``n_params`` numbers."""
    if repeats is None:
        repeats = cfg.overhead_repeats
    # RandomState on purpose: same draws as the research code
    rng = np.random.RandomState(cfg.seed)
    client_vecs = [
        rng.normal(0.0, 0.1, n_params).astype(np.float64) for _ in range(cfg.num_clients)
    ]
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = np.mean(np.stack(client_vecs, axis=0), axis=0)
        times.append(time.perf_counter() - t0)
    return {"n_params": n_params, "plain_sec": float(np.median(times))}


def bench_dp_tax(
    cfg: LegacyTrialConfig,
    dataset: str,
    in_features: int,
    width_scale: int,
    sigma: float,
    n_samples: int,
    repeats: int | None = None,
) -> dict[str, Any]:
    """Median time of one local round with and without private training, on synthetic rows."""
    if repeats is None:
        repeats = cfg.overhead_repeats

    # RandomState on purpose: same draws as the research code
    rng = np.random.RandomState(cfg.seed)
    X_syn = rng.normal(0.5, 0.2, (n_samples, in_features)).astype(np.float32)
    y_syn = rng.binomial(1, 0.5, n_samples).astype(np.int64)

    model = legacy_models.new_model(cfg, in_features, dataset=dataset, width_scale=width_scale)
    global_state = get_state(model)
    n_params = count_parameters(model)

    plain_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        training.local_train(
            cfg, global_state, X_syn, y_syn, in_features,
            dataset=dataset, dp=False, width_scale=width_scale, seed=cfg.seed,
        )  # fmt: skip
        plain_times.append(time.perf_counter() - t0)
    plain_sec = float(np.median(plain_times))

    dp_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        training.local_train(
            cfg, global_state, X_syn, y_syn, in_features,
            dataset=dataset, dp=True, sigma=sigma, width_scale=width_scale, seed=cfg.seed,
        )  # fmt: skip
        dp_times.append(time.perf_counter() - t0)
    dp_sec = float(np.median(dp_times))

    return {
        "plain_sec": plain_sec,
        "dp_sec": dp_sec,
        "dp_multiplier": dp_sec / max(plain_sec, 1e-9),
        "n_params": n_params,
        "width_scale": width_scale,
        "n_samples": n_samples,
    }


def bench_model_sizes(
    cfg: LegacyTrialConfig,
    width_scales: Sequence[int],
    dataset: str,
    in_features: int,
    ctx: Any,
    fraction: float = 1.0,
) -> list[dict[str, Any]]:
    """``bench_encryption`` for the research network at each width multiplier."""
    results = []
    for scale in width_scales:
        model = legacy_models.new_model(cfg, in_features, dataset=dataset, width_scale=scale)
        bench = bench_encryption(cfg, count_parameters(model), ctx, fraction=fraction)
        bench["width_scale"] = scale
        results.append(bench)
    return results
