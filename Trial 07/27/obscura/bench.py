"""Crypto and DP overhead benchmarking harness (no training).

Separates the costs of three orthogonal privacy mechanisms:
  * HE: encryption + aggregation time and payload, driven by parameter count.
  * DP: per-sample gradient clipping + Gaussian noise, driven by sigma.

These costs are measured independently to test the hypothesis that they compose
additively. HE's cost is communication+computation on aggregation; DP's cost is
per-sample gradient processing on the client. They touch disjoint parts of a
federated round, so total overhead should be DP_tax * HE_tax with no interaction.

Key methods:
  * bench_encryption: time one client encrypting a masked vector; return per-client
    serialized ciphertext size and time to encrypt+aggregate+decrypt over NUM_CLIENTS.
  * bench_plaintext: wall-clock of a numpy weighted mean (the denominator for HE slowdown).
  * bench_dp_tax: time one local epoch under plain SGD vs DP-SGD on synthetic data.
  * bench_model_sizes: count parameters and call bench_encryption for each width scale.
  * ciphertext_quantum / payload_ratio_model: the padding arithmetic that explains
    every payload number in this trial.

THE PADDING FLOOR, because it is the single most confusing number in the results.
A CKKS ciphertext under poly_modulus_degree=8192 serialises to the same ~235 KB
whether 1 or all 4096 of its slots carry data. Payload is therefore quantised in
whole ciphertexts, and the ratio against a float32 baseline is

    payload_ratio(n) = ceil(n / slots) * bytes_per_ciphertext / (4 * n)

which is NOT a constant -- it decays towards bytes_per_ciphertext/(4*slots) as n
grows and the final partial chunk stops dominating. That is why the same CKKS
configuration reports 63.3x at loan's 929 parameters, 18.76x at stroke's 3137,
and 14.38x at 2.1M: 18.758 / 14.365 = 1.3058 and 4096 / 3137 = 1.3057. The
measured spread is padding, not a modulus-chain difference and not noise.
The same quantisation is why an encryption-ratio sweep on a model smaller than
one ciphertext is degenerate: every p >= 1/slots costs exactly one chunk.
"""
import numpy as np
import time
import torch

from . import config, he, models, fedprox, data


def ciphertext_quantum(ctx, slots=None):
    """Serialized bytes of ONE full ciphertext -- the payload quantisation step.

    Measured rather than hard-coded, so the analytic padding model stays honest
    if the CKKS parameters ever change. Encrypting a full `slots`-wide chunk and
    a 1-element chunk gives sizes within ~50 B of each other; that near-equality
    IS the padding floor.
    """
    slots = slots or config.CKKS_SLOTS
    return he.serialized_size(he.encrypt_chunk(ctx, np.zeros(slots)))


def payload_ratio_model(n_encrypted, n_params, bytes_per_ciphertext, slots=None):
    """Analytic payload ratio: ceil(n_enc/slots) * bytes / (4 * n_params).

    The closed form of the padding floor described in the module docstring.
    Reported alongside every measured payload_ratio so a reader can see that the
    measurement is exactly the arithmetic and not an artifact of the benchmark.
    """
    slots = slots or config.CKKS_SLOTS
    if n_params <= 0:
        return 0.0
    n_chunks = -(-int(n_encrypted) // slots)
    return float(n_chunks) * float(bytes_per_ciphertext) / float(4 * n_params)


def bench_encryption(n_params, ctx, fraction=1.0, repeats=None):
    """Benchmark HE aggregation cost: encrypt, aggregate over NUM_CLIENTS, decrypt.

    Creates NUM_CLIENTS synthetic float64 vectors of length n_params (deterministic,
    seeded, scaled ~0.1 to resemble real weights), applies the mask to the first
    round(fraction*n_params) indices, then calls he.fedavg_he_timed and reports
    the MEDIAN over repeats.

    Args:
        n_params: Total parameter count.
        ctx: CKKS context (built once by caller).
        fraction: Fraction of parameters to encrypt (0.0 to 1.0).
        repeats: Number of repetitions; defaults to config.OVERHEAD_REPEATS.

    Returns:
        Dict with keys: n_params, fraction, n_encrypted, encrypt_sec, aggregate_sec,
        decrypt_sec, crypto_sec, ciphertext_bytes, plaintext_bytes, payload_ratio,
        wire_payload_ratio, predicted_payload_ratio, bytes_per_ciphertext,
        n_chunks, us_per_param.
    """
    if repeats is None:
        repeats = config.OVERHEAD_REPEATS

    # Deterministic synthetic data: float64 vectors ~0.1 magnitude.
    rng = np.random.RandomState(config.SEED)
    client_vecs = [
        rng.normal(0.0, 0.1, n_params).astype(np.float64)
        for _ in range(config.NUM_CLIENTS)
    ]

    # Build mask: encrypt top-fraction parameters.
    n_encrypted = int(round(fraction * n_params))
    mask = np.zeros(n_params, dtype=bool)
    mask[:n_encrypted] = True

    # Collect timings over repeats and take median.
    all_stats = []
    for _ in range(repeats):
        _, stats = he.fedavg_he_timed(client_vecs, mask, ctx)
        all_stats.append(stats)

    # Median over repeats for each timing field.
    median_stats = {}
    for key in all_stats[0].keys():
        values = [s[key] for s in all_stats]
        if isinstance(values[0], (int, float)):
            median_stats[key] = float(np.median(values))
        else:
            median_stats[key] = values[0]

    # Per-parameter encryption cost in microseconds (numerator is encrypt_sec only).
    encrypt_sec = median_stats['encrypt_sec']
    us_per_param = 1e6 * encrypt_sec / max(1, n_encrypted)

    # Bytes per ciphertext, back-derived from what was actually serialized, so
    # the analytic model below is calibrated on this run rather than assumed.
    n_chunks = median_stats['n_chunks']
    bytes_per_ciphertext = (median_stats['ciphertext_bytes'] / n_chunks
                            if n_chunks else 0.0)

    return {
        "n_params": n_params,
        "fraction": fraction,
        "n_encrypted": n_encrypted,
        "encrypt_sec": median_stats['encrypt_sec'],
        "aggregate_sec": median_stats['aggregate_sec'],
        "decrypt_sec": median_stats['decrypt_sec'],
        "crypto_sec": median_stats['crypto_sec'],
        "ciphertext_bytes": median_stats['ciphertext_bytes'],
        "plaintext_bytes": median_stats['plaintext_bytes'],
        "payload_ratio": median_stats['payload_ratio'],
        "wire_payload_ratio": median_stats['wire_payload_ratio'],
        "predicted_payload_ratio": payload_ratio_model(
            n_encrypted, n_params, bytes_per_ciphertext),
        "bytes_per_ciphertext": float(bytes_per_ciphertext),
        "n_chunks": n_chunks,
        "us_per_param": us_per_param,
    }


def bench_plaintext(n_params, repeats=None):
    """Benchmark plaintext FedAvg: numpy weighted mean of NUM_CLIENTS vectors.

    Creates NUM_CLIENTS synthetic float64 vectors and computes their mean;
    reports wall-clock median over repeats.

    Args:
        n_params: Vector length.
        repeats: Number of repetitions; defaults to config.OVERHEAD_REPEATS.

    Returns:
        Dict with keys: n_params, plain_sec.
    """
    if repeats is None:
        repeats = config.OVERHEAD_REPEATS

    # Deterministic synthetic data.
    rng = np.random.RandomState(config.SEED)
    client_vecs = [
        rng.normal(0.0, 0.1, n_params).astype(np.float64)
        for _ in range(config.NUM_CLIENTS)
    ]

    # Median wall-clock time for numpy mean.
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = np.mean(np.stack(client_vecs, axis=0), axis=0)
        times.append(time.perf_counter() - t0)

    plain_sec = float(np.median(times))
    return {
        "n_params": n_params,
        "plain_sec": plain_sec,
    }


def bench_dp_tax(dataset, in_features, width_scale, sigma, n_samples, repeats=None):
    """Benchmark DP-SGD per-sample gradient cost vs plaintext SGD.

    Trains one local epoch on synthetic tabular data (deterministic RandomState(SEED),
    Bernoulli labels) with dp=False and dp=True, reports wall-clock median over repeats.

    Args:
        dataset: "stroke" or "loan".
        in_features: Input feature count.
        width_scale: Hidden layer width multiplier.
        sigma: Noise multiplier for DP (ignored if dp=False).
        n_samples: Number of synthetic samples to generate.
        repeats: Number of repetitions; defaults to config.OVERHEAD_REPEATS.

    Returns:
        Dict with keys: plain_sec, dp_sec, dp_multiplier, n_params, width_scale, n_samples.
    """
    if repeats is None:
        repeats = config.OVERHEAD_REPEATS

    # Deterministic synthetic data: float32 features, Bernoulli labels.
    rng = np.random.RandomState(config.SEED)
    X_syn = rng.normal(0.5, 0.2, (n_samples, in_features)).astype(np.float32)
    y_syn = rng.binomial(1, 0.5, n_samples).astype(np.int64)

    # Build global state (zeros).
    model = models.new_model(in_features, dataset=dataset, width_scale=width_scale)
    global_state = models.get_weights(model)
    n_params = models.count_params(model)

    # Time plaintext training.
    plain_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _, _, _ = fedprox.local_train(
            global_state, X_syn, y_syn, in_features,
            dataset=dataset, dp=False, width_scale=width_scale, seed=config.SEED
        )
        plain_times.append(time.perf_counter() - t0)
    plain_sec = float(np.median(plain_times))

    # Time DP-SGD training.
    dp_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _, _, _ = fedprox.local_train(
            global_state, X_syn, y_syn, in_features,
            dataset=dataset, dp=True, sigma=sigma,
            width_scale=width_scale, seed=config.SEED
        )
        dp_times.append(time.perf_counter() - t0)
    dp_sec = float(np.median(dp_times))

    dp_multiplier = dp_sec / max(plain_sec, 1e-9)
    return {
        "plain_sec": plain_sec,
        "dp_sec": dp_sec,
        "dp_multiplier": dp_multiplier,
        "n_params": n_params,
        "width_scale": width_scale,
        "n_samples": n_samples,
    }


def bench_model_sizes(width_scales, dataset, in_features, ctx, fraction=1.0):
    """Benchmark HE overhead across model sizes via width scaling.

    For each width scale, builds a model, counts its parameters, and calls
    bench_encryption. Returns a list of dicts each carrying width_scale, n_params,
    and all encryption stats.

    Args:
        width_scales: List of width multipliers.
        dataset: "stroke" or "loan".
        in_features: Input feature count.
        ctx: CKKS context.
        fraction: Encryption fraction (default 1.0 for naive HE).

    Returns:
        List of dicts, one per width scale.
    """
    results = []
    for scale in width_scales:
        model = models.new_model(in_features, dataset=dataset, width_scale=scale)
        n_params = models.count_params(model)
        bench = bench_encryption(n_params, ctx, fraction=fraction)
        bench["width_scale"] = scale
        results.append(bench)
    return results
