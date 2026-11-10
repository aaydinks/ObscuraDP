"""CKKS encryption as a secure sum. **Research only.**

CKKS is a homomorphic encryption scheme: the server can add encrypted updates without being
able to read them, and whoever holds the key decrypts only the sum. It is kept here so the
earlier experiments can be compared with the default secure sum, with three known problems
made explicit:

- Its results are approximate, and published decrypted approximations can help an attacker
  recover the key (Li and Micciancio, 2021). This wrapper rounds every decrypted value to a
  declared number of decimals and returns 32-bit numbers, which removes the low-order digits
  that attack relies on. That is a mitigation, not a proof of safety.
- Its messages are several times larger than plain ones (measured about 14 times in the
  earlier experiments), so it misses the project's bytes target.
- Somebody must hold the key and must refuse to decrypt anything but the final sum.

For these reasons every entry point takes an ``UnsafeFlags`` object and refuses to work
unless unsafe settings are allowed; the reason is then written into the privacy report.

The roles are separate objects. ``CkksKeyHolder`` owns the secret key. Clients and the
aggregator get a public context without it, so the aggregator cannot decrypt. No rotation
or multiplication keys are generated: adding ciphertexts does not need them.

Needs TenSEAL (the ``he`` extra).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from obscuradp.crypto.chunking import chunk_bounds
from obscuradp.crypto.roles import MissingParticipants
from obscuradp.unsafe import UnsafeFlags

Float32Vector = npt.NDArray[np.float32]

POLY_MODULUS_DEGREE = 8192
COEFF_MOD_BIT_SIZES = (60, 40, 60)
GLOBAL_SCALE = 2**40
SLOTS = POLY_MODULUS_DEGREE // 2


def _tenseal() -> Any:
    try:
        import tenseal
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ImportError("the CKKS backend needs TenSEAL: pip install 'obscuradp[he]'") from error
    return tenseal


class CkksKeyHolder:
    """Owns the secret key. Must not be the party that adds the updates.

    Args:
        flags: The run's unsafe-setting collector. Construction is refused unless it allows
            unsafe settings.
        decimals: Decrypted sums are rounded to this many decimals before they are returned.
    """

    def __init__(self, flags: UnsafeFlags, *, decimals: int = 4) -> None:
        flags.raise_or_record("ckks_research_only")
        if not 0 <= decimals <= 6:
            raise ValueError("decimals must be between 0 and 6")
        ts = _tenseal()
        self.decimals = decimals
        self._context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=POLY_MODULUS_DEGREE,
            coeff_mod_bit_sizes=list(COEFF_MOD_BIT_SIZES),
        )
        self._context.global_scale = GLOBAL_SCALE

    def public_context(self) -> bytes:
        """What clients and the aggregator receive: everything except the secret key."""
        serialized: bytes = self._context.serialize(
            save_public_key=True,
            save_secret_key=False,
            save_galois_keys=False,
            save_relin_keys=False,
        )
        return serialized

    def open_sum(self, ciphertexts: Sequence[bytes], length: int) -> Float32Vector:
        """Decrypt a sum and return it rounded, as 32-bit numbers. Never call this on a
        single participant's message."""
        ts = _tenseal()
        values: list[float] = []
        for blob in ciphertexts:
            values.extend(ts.ckks_vector_from(self._context, blob).decrypt())
        if len(values) < length:
            raise ValueError("the ciphertexts hold fewer values than expected")
        return np.round(np.asarray(values[:length], dtype=np.float64), self.decimals).astype(
            np.float32
        )


class CkksClient:
    """Encrypts one participant's update with the public context."""

    def __init__(self, name: str, public_context: bytes) -> None:
        self.name = name
        self._context = _tenseal().context_from(public_context)

    def protect(self, values: npt.ArrayLike) -> list[bytes]:
        """Encrypt a vector of numbers, 4,096 per ciphertext."""
        ts = _tenseal()
        vector = np.asarray(values, dtype=np.float64).ravel()
        return [
            ts.ckks_vector(self._context, vector[start:stop].tolist()).serialize()
            for start, stop in chunk_bounds(vector.size, SLOTS)
        ]


class CkksAggregator:
    """Adds encrypted updates. Holds only the public context, so it cannot decrypt."""

    def __init__(self, participants: Sequence[str], public_context: bytes) -> None:
        self.participants = tuple(participants)
        self._context = _tenseal().context_from(public_context)

    @property
    def can_decrypt(self) -> bool:
        """Whether this object holds a secret key. It never should."""
        return bool(self._context.is_private())

    def combine(self, messages: Mapping[str, Sequence[bytes]]) -> list[bytes]:
        """Add one encrypted update per participant, ciphertext by ciphertext."""
        if set(messages) != set(self.participants):
            raise MissingParticipants("a message from every participant is needed")
        ts = _tenseal()
        counts = {len(blobs) for blobs in messages.values()}
        if len(counts) != 1:
            raise ValueError("all updates must have the same number of ciphertexts")
        totals = []
        for index in range(counts.pop()):
            total = ts.ckks_vector_from(self._context, messages[self.participants[0]][index])
            for name in self.participants[1:]:
                total = total + ts.ckks_vector_from(self._context, messages[name][index])
            totals.append(total.serialize())
        return totals
