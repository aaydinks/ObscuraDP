"""CKKS primitives and weighted homomorphic aggregation (TenSEAL).

This module streams parameters in chunks to keep peak memory at O(N*slots),
not O(N*model). It combines Experiment 3's SPE mask-driven selective encryption
with Experiment 2's payload/latency instrumentation, and adds weighted
homomorphic aggregation -- a contribution of this trial.

Weighted HE: To aggregate under encryption without ever decrypting an individual
client's vector, we multiply each client's ciphertext by its plaintext scalar
weight (w_c) before summing. This remains multiplicative depth 1 (same as the
plain 1/N scale factor), so the [60,40,60] modulus chain in config still
suffices. The sum is then the encrypted weighted sum; decryption yields the
plaintext-equivalent result. This is essential when the data is non-IID
(Dirichlet partition) and clients must be weighted by their sample counts.
Independent audit note: [60,40,60] leaves exactly ZERO multiplicative depth after
that weight multiply -- two scalar multiplies on a fresh ciphertext raise
"scale out of bounds". The chain suffices for this aggregation and nothing more;
a second homomorphic operation would need [60,40,40,60] and ~20x payload.

TWO PAYLOAD RATIOS, and they answer different questions:

  payload_ratio      = ciphertext_bytes / (4 * n_params)
      Ciphertext cost measured against a full-plaintext float32 baseline. This is
      the number Experiments 2 and 3 report, and it is what you want when asking
      "how expensive is a CKKS ciphertext relative to the model it covers".
  wire_payload_ratio = (ciphertext_bytes + 4 * n_plaintext_params) / (4 * n_params)
      What the client ACTUALLY puts on the wire. Under SPE at p < 1 the client
      still has to send the unencrypted parameters in the clear, and
      payload_ratio omits them. At n_params = 3137 the two read 18.758 vs 19.658
      at p = 0.1, and coincide at p = 1.0 where there is no plaintext remainder.

Both are reported; wire_payload_ratio is the honest total-bandwidth figure and
payload_ratio is kept unchanged so earlier results stay comparable.
"""
import time

import numpy as np
import tenseal as ts

from . import config


def make_context():
    """Build the CKKS context + keys (done once before training)."""
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=config.CKKS_POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=config.CKKS_COEFF_MOD_BIT_SIZES,
    )
    ctx.generate_galois_keys()
    ctx.global_scale = config.CKKS_GLOBAL_SCALE
    return ctx


def chunk_bounds(n, slots=None):
    """Yield (start, end) slices covering [0, n) in `slots`-sized chunks."""
    slots = slots or config.CKKS_SLOTS
    for s in range(0, n, slots):
        yield s, min(s + slots, n)


def encrypt_chunk(ctx, arr):
    """Encrypt a 1-D array (<= slots long) as one CKKS ciphertext."""
    return ts.ckks_vector(ctx, np.asarray(arr, dtype=np.float64).tolist())


def serialized_size(enc):
    """Serialized ciphertext size in bytes (the per-chunk wire payload)."""
    return len(enc.serialize())


def encrypt_vector(ctx, vec, slots=None):
    """Encrypt a flat parameter vector and return ciphertexts + timing.

    `slots` must be threaded through: callers that aggregate the returned
    ciphertexts chunk-by-chunk index them with chunk_bounds(n, slots), so if this
    function chunked at config.CKKS_SLOTS regardless, the two would disagree for
    any non-default slot count and the aggregation would run off the end of the
    list. Defaults to config.CKKS_SLOTS, so existing callers are unaffected.

    Args:
        ctx: CKKS context.
        vec: Flat 1-D parameter vector (float64).
        slots: CKKS slot count per ciphertext; defaults to config.CKKS_SLOTS.

    Returns:
        (list of CKKSVector chunks, seconds elapsed, total serialized bytes).
    """
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


def fedavg_plaintext(client_vecs, weights=None):
    """Plaintext FedAvg: weighted or unweighted mean of client vectors.

    Args:
        client_vecs: List of flat parameter vectors (float64).
        weights: Optional array of sample weights; if None, unweighted mean.
                 Defensively normalized to sum to 1 if provided.

    Returns:
        Float64 mean vector.
    """
    vecs = np.stack(client_vecs, axis=0)
    if weights is None:
        return np.mean(vecs, axis=0)
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.sum(w)  # defensive normalization
    return np.average(vecs, axis=0, weights=w)


def aggregate_indices(client_vecs, idx, ctx, weights=None, slots=None):
    """Homomorphically aggregate client values at flat indices `idx`.

    Streams chunk-by-chunk: for each chunk, encrypts every client's slice,
    combines the ciphertexts (scaling by weight w_c per client, or by 1/N for
    unweighted), decrypts. Returns the aggregated plaintext values aligned with
    `idx`. Peak memory is O(N*slots) because we hold at most N ciphertexts of
    size `slots` at once, not O(N*model).

    Args:
        client_vecs: List of flat parameter vectors.
        idx: Indices to aggregate (e.g., encrypted parameters only).
        ctx: CKKS context.
        weights: Optional per-client weights (sum to 1). If None, unweighted 1/N.
        slots: CKKS slot count; defaults to config.CKKS_SLOTS.

    Returns:
        Float64 aggregated values at indices `idx`.
    """
    slots = slots or config.CKKS_SLOTS
    n = len(idx)
    N = len(client_vecs)
    sub = [v[idx] for v in client_vecs]  # each client's selected params

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


def fedavg_he(client_vecs, mask, ctx, weights=None, slots=None):
    """SPE/naive FedAvg aggregation over flat parameter vectors.

    `mask` is a boolean array: True positions are aggregated under CKKS,
    False positions are averaged in plaintext, then scattered back together.
    A full-True mask reproduces naive HE.

    Args:
        client_vecs: List of flat parameter vectors.
        mask: Boolean array over flattened parameters.
        ctx: CKKS context.
        weights: Optional per-client weights. If None, unweighted 1/N.
        slots: CKKS slot count; defaults to config.CKKS_SLOTS.

    Returns:
        Float64 aggregated vector (same shape as client_vecs[0]).
    """
    enc_idx = np.where(mask)[0]
    plain_idx = np.where(~mask)[0]
    out = np.empty(len(mask), dtype=np.float64)

    if len(plain_idx):
        plain_stack = np.stack([v[plain_idx] for v in client_vecs], axis=0)
        out[plain_idx] = fedavg_plaintext(plain_stack, weights=weights)

    if len(enc_idx):
        out[enc_idx] = aggregate_indices(
            client_vecs, enc_idx, ctx, weights=weights, slots=slots
        )

    return out


def fedavg_he_timed(client_vecs, mask, ctx, weights=None, slots=None):
    """Instrumented SPE/naive FedAvg with comprehensive timing and stats.

    Returns:
        (aggregated float64 vector, stats dict with keys: encrypt_sec,
         aggregate_sec, decrypt_sec, crypto_sec, ciphertext_bytes (per client),
         plaintext_bytes (per client, float32 wire format), payload_ratio,
         wire_payload_ratio, n_chunks, n_encrypted, encrypted_fraction,
         n_params). See the module docstring for the difference between the two
         payload ratios.
    """
    slots = slots or config.CKKS_SLOTS
    n_params = len(mask)
    n_encrypted = np.sum(mask)
    encrypted_fraction = float(n_encrypted) / n_params if n_params > 0 else 0.0
    n_chunks = (
        (n_encrypted + slots - 1) // slots if n_encrypted > 0 else 0
    )

    # Plaintext bytes: float32 wire format for the full vector per client.
    plaintext_bytes = n_params * 4

    if n_encrypted == 0:
        # No encryption: pure plaintext mean.
        out = fedavg_plaintext(client_vecs, weights=weights)
        stats = {
            "encrypt_sec": 0.0,
            "aggregate_sec": 0.0,
            "decrypt_sec": 0.0,
            "crypto_sec": 0.0,
            "ciphertext_bytes": 0,
            "plaintext_bytes": int(plaintext_bytes),
            "payload_ratio": 0.0,
            # Nothing is encrypted, so the client sends the whole model in the
            # clear: exactly 1.0x the plaintext baseline, not 0.0x.
            "wire_payload_ratio": 1.0,
            "n_chunks": 0,
            "n_encrypted": 0,
            "encrypted_fraction": 0.0,
            "n_params": int(n_params),
        }
        return out, stats

    # Timing the full workflow.
    t_total_start = time.perf_counter()

    # Encrypt only the masked subset (SPE).
    t_enc_start = time.perf_counter()
    enc_idx = np.where(mask)[0]
    enc_vecs = [v[enc_idx] for v in client_vecs]
    ctx_ciphertexts = []
    total_ciphertext_bytes = 0

    for vec_idx, vec in enumerate(enc_vecs):
        cts, _, total_bytes = encrypt_vector(ctx, vec, slots=slots)
        ctx_ciphertexts.append(cts)
        total_ciphertext_bytes += total_bytes

    encrypt_sec = time.perf_counter() - t_enc_start

    # Aggregate on ciphertext (chunk-wise for memory efficiency).
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
        plain_result = fedavg_plaintext(plain_stack, weights=weights)
    else:
        plain_result = np.array([])

    crypto_sec = encrypt_sec + aggregate_sec + decrypt_sec

    # Assemble output.
    out = np.empty(n_params, dtype=np.float64)
    out[enc_idx] = agg_result
    if len(plain_idx):
        out[plain_idx] = plain_result

    payload_ratio = (
        float(ciphertext_bytes_per_client) / float(plaintext_bytes)
        if plaintext_bytes > 0
        else 0.0
    )
    # The p*n encrypted params ride as ciphertext; the (1-p)*n unencrypted ones
    # still have to be transmitted, as float32. payload_ratio omits the second
    # term, which is why it looks flat in p on a model small enough to need one
    # chunk at any p -- the wire ratio does not.
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
