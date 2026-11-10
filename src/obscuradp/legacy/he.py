"""CKKS aggregation exactly as in Trial 07/27 (comparison only, see the package docstring).

Known defects reproduced on purpose: one context object holds the secret key and encrypts,
adds and decrypts (finding G5); decrypted approximate values are handed back as 64-bit
floats (finding G15); under a partial mask the unmasked entries are averaged in the clear.

Needs TenSEAL (the ``legacy`` or ``he`` extra).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from obscuradp.aggregation.fedavg import weighted_mean
from obscuradp.crypto.chunking import chunk_bounds
from obscuradp.legacy.config import LegacyTrialConfig

Vector = npt.NDArray[np.float64]


def make_context(cfg: LegacyTrialConfig) -> Any:
    """Build the CKKS context, secret key included (done once before training)."""
    import tenseal as ts

    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=cfg.ckks_poly_modulus_degree,
        coeff_mod_bit_sizes=list(cfg.ckks_coeff_mod_bit_sizes),
    )
    ctx.generate_galois_keys()
    ctx.global_scale = cfg.ckks_global_scale
    return ctx


def encrypt_chunk(ctx: Any, arr: npt.ArrayLike) -> Any:
    """Encrypt a 1-D array (at most one ciphertext long)."""
    import tenseal as ts

    return ts.ckks_vector(ctx, np.asarray(arr, dtype=np.float64).tolist())


def serialized_size(enc: Any) -> int:
    """Serialized ciphertext size in bytes."""
    return len(enc.serialize())


def encrypt_vector(
    cfg: LegacyTrialConfig, ctx: Any, vec: npt.ArrayLike, slots: int | None = None
) -> tuple[list[Any], float, int]:
    """Encrypt a flat vector. Returns (ciphertexts, seconds, total serialized bytes)."""
    slots = slots or cfg.ckks_slots
    vec = np.asarray(vec, dtype=np.float64)
    t0 = time.perf_counter()
    ciphertexts = []
    total_bytes = 0
    for s, e in chunk_bounds(len(vec), slots):
        enc = encrypt_chunk(ctx, vec[s:e])
        ciphertexts.append(enc)
        total_bytes += serialized_size(enc)
    elapsed = time.perf_counter() - t0
    return ciphertexts, elapsed, total_bytes


def fedavg_plaintext(
    client_vecs: Sequence[npt.ArrayLike], weights: npt.ArrayLike | None = None
) -> Vector:
    """Weighted or unweighted mean of client vectors in the clear."""
    return weighted_mean(client_vecs, weights)


def aggregate_indices(
    cfg: LegacyTrialConfig,
    client_vecs: Sequence[Vector],
    idx: npt.NDArray[np.intp],
    ctx: Any,
    weights: npt.ArrayLike | None = None,
    slots: int | None = None,
) -> Vector:
    """Aggregate the clients' values at flat indices ``idx`` under encryption, then decrypt."""
    slots = slots or cfg.ckks_slots
    n = len(idx)
    N = len(client_vecs)
    sub = [v[idx] for v in client_vecs]  # each client's selected params

    w: Vector | None
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w = w / np.sum(w)  # defensive normalization
    else:
        w = None

    out = np.empty(n, dtype=np.float64)
    for s, e in chunk_bounds(n, slots):
        encs = [encrypt_chunk(ctx, sub[c][s:e]) for c in range(N)]
        agg = encs[0]
        if w is not None:
            agg = agg * w[0]
            for k in range(1, N):
                agg = agg + (encs[k] * w[k])
        else:
            for k in range(1, N):
                agg = agg + encs[k]
            agg = agg * (1.0 / N)
        out[s:e] = np.asarray(agg.decrypt()[: e - s], dtype=np.float64)
    return out


def fedavg_he(
    cfg: LegacyTrialConfig,
    client_vecs: Sequence[Vector],
    mask: npt.NDArray[np.bool_],
    ctx: Any,
    weights: npt.ArrayLike | None = None,
    slots: int | None = None,
) -> Vector:
    """Aggregate masked entries under encryption and the rest in the clear."""
    enc_idx = np.where(mask)[0]
    plain_idx = np.where(~mask)[0]
    out = np.empty(len(mask), dtype=np.float64)

    if len(plain_idx):
        plain_stack = np.stack([v[plain_idx] for v in client_vecs], axis=0)
        out[plain_idx] = fedavg_plaintext(list(plain_stack), weights=weights)

    if len(enc_idx):
        out[enc_idx] = aggregate_indices(
            cfg, client_vecs, enc_idx, ctx, weights=weights, slots=slots
        )

    return out


def fedavg_he_timed(
    cfg: LegacyTrialConfig,
    client_vecs: Sequence[Vector],
    mask: npt.NDArray[np.bool_],
    ctx: Any,
    weights: npt.ArrayLike | None = None,
    slots: int | None = None,
) -> tuple[Vector, dict[str, Any]]:
    """As ``fedavg_he``, with timing and size statistics.

    ``payload_ratio`` counts ciphertext bytes against the whole model as 32-bit numbers.
    ``wire_payload_ratio`` also counts the unencrypted remainder the client still sends.
    """
    slots = slots or cfg.ckks_slots
    n_params = len(mask)
    n_encrypted = np.sum(mask)
    encrypted_fraction = float(n_encrypted) / n_params if n_params > 0 else 0.0
    n_chunks = (n_encrypted + slots - 1) // slots if n_encrypted > 0 else 0

    # Plaintext bytes: float32 wire format for the full vector per client.
    plaintext_bytes = n_params * 4

    if n_encrypted == 0:
        # No encryption: pure plaintext mean.
        out = fedavg_plaintext(client_vecs, weights=weights)
        stats: dict[str, Any] = {
            "encrypt_sec": 0.0,
            "aggregate_sec": 0.0,
            "decrypt_sec": 0.0,
            "crypto_sec": 0.0,
            "ciphertext_bytes": 0,
            "plaintext_bytes": int(plaintext_bytes),
            "payload_ratio": 0.0,
            "wire_payload_ratio": 1.0,
            "n_chunks": 0,
            "n_encrypted": 0,
            "encrypted_fraction": 0.0,
            "n_params": int(n_params),
        }
        return out, stats

    # Encrypt only the masked subset (SPE).
    t_enc_start = time.perf_counter()
    enc_idx = np.where(mask)[0]
    enc_vecs = [v[enc_idx] for v in client_vecs]
    ctx_ciphertexts = []
    total_ciphertext_bytes = 0

    for vec in enc_vecs:
        cts, _, total_bytes = encrypt_vector(cfg, ctx, vec, slots=slots)
        ctx_ciphertexts.append(cts)
        total_ciphertext_bytes += total_bytes

    encrypt_sec = time.perf_counter() - t_enc_start

    # Aggregate on ciphertext (chunk-wise for memory efficiency).
    w: Vector | None
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w = w / np.sum(w)
    else:
        w = None

    agg_result = np.empty(len(enc_idx), dtype=np.float64)
    t_agg_start = time.perf_counter()
    agg_ciphertexts = []
    for chunk_idx, (s, e) in enumerate(chunk_bounds(len(enc_idx), slots)):
        encs = [ctx_ciphertexts[c][chunk_idx] for c in range(len(client_vecs))]
        agg = encs[0]
        if w is not None:
            agg = agg * w[0]
            for k in range(1, len(client_vecs)):
                agg = agg + (encs[k] * w[k])
        else:
            for k in range(1, len(client_vecs)):
                agg = agg + encs[k]
            agg = agg * (1.0 / len(client_vecs))
        agg_ciphertexts.append((agg, s, e))
    aggregate_sec = time.perf_counter() - t_agg_start

    # Decrypt chunk-wise and time separately.
    t_dec_start = time.perf_counter()
    for agg, s, e in agg_ciphertexts:
        agg_result[s:e] = np.asarray(agg.decrypt()[: e - s], dtype=np.float64)
    decrypt_sec = time.perf_counter() - t_dec_start

    # For per-client ciphertext_bytes, average across clients.
    ciphertext_bytes_per_client = total_ciphertext_bytes // len(client_vecs)

    # Plaintext indices: averaged in plaintext.
    plain_idx = np.where(~mask)[0]
    if len(plain_idx):
        plain_stack = np.stack([v[plain_idx] for v in client_vecs], axis=0)
        plain_result = fedavg_plaintext(list(plain_stack), weights=weights)
    else:
        plain_result = np.array([])

    crypto_sec = encrypt_sec + aggregate_sec + decrypt_sec

    # Assemble output.
    out = np.empty(n_params, dtype=np.float64)
    out[enc_idx] = agg_result
    if len(plain_idx):
        out[plain_idx] = plain_result

    payload_ratio = (
        float(ciphertext_bytes_per_client) / float(plaintext_bytes) if plaintext_bytes > 0 else 0.0
    )
    plaintext_remainder_bytes = int(n_params - n_encrypted) * 4
    wire_payload_ratio = (
        (float(ciphertext_bytes_per_client) + float(plaintext_remainder_bytes))
        / float(plaintext_bytes)
        if plaintext_bytes > 0
        else 0.0
    )

    stats = {
        "encrypt_sec": float(encrypt_sec),
        "aggregate_sec": float(aggregate_sec),
        "decrypt_sec": float(decrypt_sec),
        "crypto_sec": float(crypto_sec),
        "ciphertext_bytes": int(ciphertext_bytes_per_client),
        "plaintext_bytes": int(plaintext_bytes),
        "payload_ratio": float(payload_ratio),
        "wire_payload_ratio": float(wire_payload_ratio),
        "n_chunks": int(n_chunks),
        "n_encrypted": int(n_encrypted),
        "encrypted_fraction": float(encrypted_fraction),
        "n_params": int(n_params),
    }

    return out, stats
