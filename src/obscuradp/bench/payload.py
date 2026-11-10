"""How large encrypted messages are compared with plain ones.

A ciphertext has a fixed size whether one of its slots is used or all of them. So the size
of an encrypted update grows in whole-ciphertext steps, and small models pay for mostly
empty ciphertexts. The functions here are that arithmetic.
"""

from __future__ import annotations

from obscuradp.crypto.chunking import chunk_count


def payload_ratio_model(
    n_encrypted: int, n_params: int, bytes_per_ciphertext: float, slots: int
) -> float:
    """Encrypted bytes divided by the bytes of the whole model sent as 32-bit numbers.

    ``ceil(n_encrypted / slots) * bytes_per_ciphertext / (4 * n_params)``.
    """
    if n_params <= 0:
        return 0.0
    chunks = chunk_count(int(n_encrypted), slots)
    return float(chunks) * float(bytes_per_ciphertext) / float(4 * n_params)


def payload_ratio_asymptote(bytes_per_ciphertext: float, slots: int) -> float:
    """The ratio a fully encrypted model approaches as it grows: bytes per slot over 4."""
    if slots < 1:
        raise ValueError("slots must be at least 1")
    return float(bytes_per_ciphertext) / float(4 * slots)
