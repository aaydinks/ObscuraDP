"""Timing / payload benchmark harness for naive HE, SPE, and plaintext FedAvg.

All measurements stream chunk-by-chunk and discard each ciphertext immediately, so
peak memory is O(N * slots) rather than O(N * model) -- essential for the ~25M-param
ResNets and the ~86M-param ViT.

What is timed, per communication round:
  * client crypto  (enc_ms)  -- one client encrypting its (masked) update.
  * server crypto  (agg_ms)  -- homomorphic sum of the N clients + 1/N scale +
                                decrypt. The N client ciphertexts of a chunk are
                                simulated by adding one encrypted chunk N times;
                                the homomorphic add cost is identical to summing
                                distinct ciphertexts, which keeps the harness fast.
  * payload        (bytes)   -- serialized ciphertext size (HE) vs float32 (plain).

A full-coverage mask reproduces naive HE; a p-fraction mask reproduces SPE at p.
"""
import time

import numpy as np

from . import config, he


def benchmark_plaintext(vec, N):
    """Plaintext FedAvg cost: float32 serialization + N-way mean."""
    n = vec.size
    f32 = vec.astype(np.float32)
    t0 = time.perf_counter()
    _ = f32.tobytes()                              # what crosses the wire
    prep_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    acc = f32.astype(np.float64).copy()
    for _ in range(N - 1):
        acc += f32
    acc /= N
    agg_ms = (time.perf_counter() - t0) * 1e3
    return {"prep_ms": prep_ms, "agg_ms": agg_ms,
            "payload_bytes": n * 4, "n_params": n}


def benchmark_he(ctx, vec, N, n_encrypt=None, slots=None):
    """Crypto cost of encrypting `n_encrypt` params (default all) of `vec`.

    Returns per-round client encryption time, server aggregation+decrypt time, and
    total serialized ciphertext payload for the encrypted portion. The plaintext
    remainder (n - n_encrypt params) is averaged in the clear and timed too."""
    slots = slots or config.CKKS_SLOTS
    n = vec.size
    n_encrypt = n if n_encrypt is None else int(n_encrypt)
    enc_part = vec[:n_encrypt]

    enc_ms = agg_ms = 0.0
    ct_bytes = 0
    for s, e in he.chunk_bounds(n_encrypt, slots):
        chunk = enc_part[s:e]
        t0 = time.perf_counter()
        enc = he.encrypt_chunk(ctx, chunk)
        enc_ms += (time.perf_counter() - t0) * 1e3
        ct_bytes += he.serialized_size(enc)
        t0 = time.perf_counter()
        agg = enc
        for _ in range(N - 1):
            agg = agg + enc
        agg = agg * (1.0 / N)
        _ = agg.decrypt()
        agg_ms += (time.perf_counter() - t0) * 1e3

    # plaintext remainder (the (1-p) fraction handled in the clear under SPE)
    n_plain = n - n_encrypt
    plain_payload = n_plain * 4
    if n_plain:
        rem = vec[n_encrypt:].astype(np.float32)
        t0 = time.perf_counter()
        acc = rem.astype(np.float64).copy()
        for _ in range(N - 1):
            acc += rem
        acc /= N
        agg_ms += (time.perf_counter() - t0) * 1e3

    return {"enc_ms": enc_ms, "agg_ms": agg_ms,
            "ciphertext_bytes": ct_bytes, "plain_payload_bytes": plain_payload,
            "total_payload_bytes": ct_bytes + plain_payload,
            "n_encrypt": n_encrypt, "n_params": n}
