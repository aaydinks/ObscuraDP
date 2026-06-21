"""CKKS primitives and chunked homomorphic aggregation (TenSEAL).

A model has millions of parameters but one CKKS ciphertext packs only
CKKS_SLOTS (4096) reals, so a flattened parameter vector is split into
ceil(n / slots) chunks, each encrypted independently. Aggregation is chunk-wise:
the server sums the N clients' ciphertexts for each chunk and scales by 1/N --
pure homomorphic adds plus one scalar multiply (depth 1), so the decrypted FedAvg
mean matches plaintext to CKKS rounding (~1e-6).

Two aggregation modes:
  * naive  -- every parameter encrypted (full mask).
  * SPE    -- only the masked (sensitive) parameters are encrypted; the rest are
              averaged in plaintext. Overhead falls ~linearly with the mask size.

Functions are written to stream chunk-by-chunk so peak memory stays at O(N*slots)
ciphertexts rather than O(N*model), which matters for the ~25M-param ResNets.
"""
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


def aggregate_indices(client_vecs, idx, ctx, slots=None):
    """Homomorphically average client values at flat indices `idx`.

    Streams chunk-by-chunk: for each chunk, encrypts every client's slice, sums
    the ciphertexts, scales by 1/N, decrypts. Returns the averaged plaintext
    values aligned with `idx`. Used for both naive (idx = all) and SPE (idx =
    sensitive subset) aggregation, and by the benchmark harness.
    """
    slots = slots or config.CKKS_SLOTS
    n = len(idx)
    N = len(client_vecs)
    sub = [v[idx] for v in client_vecs]          # each client's selected params
    out = np.empty(n, dtype=np.float64)
    for s, e in chunk_bounds(n, slots):
        encs = [encrypt_chunk(ctx, sub[c][s:e]) for c in range(N)]
        agg = encs[0]
        for k in range(1, N):
            agg = agg + encs[k]
        agg = agg * (1.0 / N)
        out[s:e] = np.asarray(agg.decrypt()[: e - s], dtype=np.float64)
    return out


def fedavg_plaintext(client_vecs):
    """Plaintext FedAvg: unweighted mean of the client parameter vectors."""
    return np.mean(np.stack(client_vecs, axis=0), axis=0)


def fedavg_he(client_vecs, mask, ctx, slots=None):
    """SPE/naive FedAvg aggregation over flat parameter vectors.

    `mask` is a boolean array over the flattened parameters: True positions are
    aggregated under CKKS, False positions are averaged in plaintext, then the
    two are scattered back together. A full-True mask reproduces naive HE.
    """
    enc_idx = np.where(mask)[0]
    plain_idx = np.where(~mask)[0]
    out = np.empty(len(mask), dtype=np.float64)
    if len(plain_idx):
        plain_stack = np.stack([v[plain_idx] for v in client_vecs], axis=0)
        out[plain_idx] = plain_stack.mean(axis=0)
    if len(enc_idx):
        out[enc_idx] = aggregate_indices(client_vecs, enc_idx, ctx, slots)
    return out
