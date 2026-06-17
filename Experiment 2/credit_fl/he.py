"""CKKS homomorphic encryption for the HE-FL arm (TenSEAL).

Threat model reproduced from the paper: clients encrypt their flattened model
update under a shared CKKS public context; the server aggregates *on ciphertext*
(adds the five encrypted vectors and scales by 1/N) and only the decrypted mean
is ever revealed. The server never sees an individual plaintext update, and no
client sees another's update.

CKKS is approximate arithmetic over reals, but the only homomorphic operations
here are additions plus one scalar multiply (1/N), so the decrypted mean matches
the plaintext FedAvg mean to ~6-7 decimal places -- hence HE-FL reproduces
Standard FL accuracy within floating-point rounding, which is the paper's claim.
"""
import time

import numpy as np
import tenseal as ts

from . import config


def make_context():
    """Build the CKKS context (key generation -- done once before training)."""
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=config.CKKS_POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=config.CKKS_COEFF_MOD_BIT_SIZES,
    )
    context.generate_galois_keys()
    context.global_scale = config.CKKS_GLOBAL_SCALE
    return context


def encrypt_vector(context, flat):
    """Encrypt a flat plaintext weight vector. Returns (ciphertext, seconds)."""
    t0 = time.perf_counter()
    enc = ts.ckks_vector(context, flat.tolist())
    return enc, time.perf_counter() - t0


def ciphertext_bytes(enc):
    """Serialized ciphertext size in bytes (what would cross the wire)."""
    return len(enc.serialize())


def aggregate_encrypted(enc_list):
    """Server-side: sum the ciphertexts and scale by 1/N. Returns (ct, seconds).

    Pure homomorphic ops -- the server holds no secret key and never decrypts."""
    t0 = time.perf_counter()
    agg = enc_list[0]
    for e in enc_list[1:]:
        agg = agg + e
    agg = agg * (1.0 / len(enc_list))
    return agg, time.perf_counter() - t0


def decrypt_vector(enc, size):
    """Decrypt and truncate to the model's parameter count."""
    return np.array(enc.decrypt()[:size], dtype=np.float64)
