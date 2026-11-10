"""A secure sum by pairwise masking.

Every pair of participants agrees on a secret that the server never sees. From it both
derive the same long stream of random numbers, a mask. One of the two adds the mask to its
update and the other subtracts it. Each message on its own then looks like random numbers,
but in the sum of all messages every mask appears once with a plus and once with a minus,
so the masks cancel and the true sum comes out exactly.

Building blocks, all from the ``cryptography`` package: X25519 for the shared secret,
HKDF-SHA256 to turn it into a key, ChaCha20 to stretch the key into a mask. The arithmetic
is on whole numbers modulo $2^k$, so nothing is approximate.

What this version does not do, stated in every privacy report that uses it:

- It assumes the server passes the participants' public keys on unchanged. Keys are not
  authenticated, so a server that swaps keys could read updates.
- It protects against the server and against up to $N - 2$ participants working with it.
  With two participants each one can work out the other's update from the sum.
- It has no dropout recovery: if a participant's message is missing, the sum cannot be
  opened and the round must be repeated with a new sum index.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from obscuradp.compression.packing import pack_bits, packed_size, unpack_bits
from obscuradp.compression.quantize import sum_bits
from obscuradp.crypto.roles import MissingParticipants

IntVector = npt.NDArray[np.uint64]

_INFO = b"obscuradp pairwise masking v1"
PUBLIC_KEY_BYTES = 32


def _check_names(participants: Sequence[str]) -> tuple[str, ...]:
    names = tuple(participants)
    if len(names) < 2:
        raise ValueError("a secure sum needs at least two participants")
    if len(set(names)) != len(names):
        raise ValueError("participant names must be distinct")
    if any("|" in name or not name for name in names):
        raise ValueError("participant names must be non-empty and must not contain '|'")
    return names


def _check_modulus(modulus_bits: int) -> None:
    if not 1 <= modulus_bits <= 62:
        raise ValueError("modulus_bits must be between 1 and 62")


def _mask(pair_key: bytes, sum_index: int, length: int, modulus_bits: int) -> IntVector:
    """The mask of one pair for one sum: ``length`` uniform numbers below $2^k$."""
    nonce = (0).to_bytes(4, "little") + sum_index.to_bytes(12, "little")
    stream = Cipher(algorithms.ChaCha20(pair_key, nonce), mode=None).encryptor()
    raw = np.frombuffer(stream.update(bytes(8 * length)), dtype="<u8")
    return np.asarray(raw & np.uint64(2**modulus_bits - 1), dtype=np.uint64)


class MaskingClient:
    """One participant's side of the protocol. Lives on that participant's machine."""

    def __init__(self, name: str, modulus_bits: int) -> None:
        _check_modulus(modulus_bits)
        self.name = name
        self.modulus_bits = modulus_bits
        self._private = X25519PrivateKey.generate()
        self._pair_keys: dict[str, bytes] = {}
        self._used: set[int] = set()

    def public_key(self) -> bytes:
        """The 32 bytes this participant publishes. Safe for anyone to see."""
        return self._private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )

    def receive_public_keys(self, public_keys: Mapping[str, bytes]) -> None:
        """Take in everyone's public key and derive one secret per other participant."""
        names = _check_names(list(public_keys))
        if self.name not in public_keys:
            raise ValueError("the key list must include this participant")
        self._pair_keys = {}
        for other in names:
            if other == self.name:
                continue
            shared = self._private.exchange(X25519PublicKey.from_public_bytes(public_keys[other]))
            pair = "|".join(sorted((self.name, other))).encode()
            self._pair_keys[other] = HKDF(
                algorithm=hashes.SHA256(), length=32, salt=None, info=_INFO + b"|" + pair
            ).derive(shared)

    def protect(self, sum_index: int, values: npt.ArrayLike) -> bytes:
        """Mask this participant's numbers for sum number ``sum_index``.

        Each sum index can be used once. Masking two different updates with the same masks
        would reveal their difference, so a second call with the same index is refused.
        """
        if not self._pair_keys:
            raise RuntimeError("receive_public_keys must be called before protect")
        if sum_index < 0:
            raise ValueError("sum_index must not be negative")
        if sum_index in self._used:
            raise ValueError(f"sum index {sum_index} was already used; masks are never reused")
        numbers = np.asarray(values, dtype=np.uint64).ravel()
        modulus = np.uint64(2**self.modulus_bits - 1)
        if numbers.size and int(numbers.max()) >> self.modulus_bits:
            raise ValueError(f"a value does not fit in {self.modulus_bits} bits")
        total = numbers.copy()
        for other, key in self._pair_keys.items():
            mask = _mask(key, sum_index, numbers.size, self.modulus_bits)
            # The alphabetically first of the two adds the mask, the other subtracts it.
            # uint64 arithmetic wraps modulo 2**64, of which 2**k is a divisor.
            total = total + mask if self.name < other else total - mask
        self._used.add(sum_index)
        return pack_bits(total & modulus, self.modulus_bits)


class MaskingAggregator:
    """The server's side. It knows who takes part and how wide the numbers are, nothing else."""

    def __init__(self, participants: Sequence[str], modulus_bits: int) -> None:
        _check_modulus(modulus_bits)
        self.participants = _check_names(participants)
        self.modulus_bits = modulus_bits

    def combine(self, sum_index: int, messages: Mapping[str, bytes], length: int) -> IntVector:
        """Add one message per participant. The masks cancel and the true sum remains.

        Args:
            sum_index: Which sum this is. Only used in error messages here.
            messages: One message from every participant.
            length: How many numbers each update holds (the size of the model, which is
                public). It cannot be read off a message, whose last byte may be padding.

        Raises:
            MissingParticipants: If a message is missing or comes from an unknown sender.
                The sum cannot be opened without every participant.
        """
        if set(messages) != set(self.participants):
            missing = sorted(set(self.participants) - set(messages))
            extra = sorted(set(messages) - set(self.participants))
            raise MissingParticipants(
                f"sum {sum_index} needs a message from every participant "
                f"(missing: {missing}, unexpected: {extra})"
            )
        expected = packed_size(length, self.modulus_bits)
        if any(len(message) != expected for message in messages.values()):
            raise ValueError(
                f"every message of this sum must be {expected} bytes for {length} numbers"
            )
        total = np.zeros(length, dtype=np.uint64)
        for name in self.participants:
            total = total + unpack_bits(messages[name], self.modulus_bits, length)
        return np.asarray(total & np.uint64(2**self.modulus_bits - 1), dtype=np.uint64)


def setup_pairwise_masking(
    participants: Sequence[str], modulus_bits: int
) -> tuple[dict[str, MaskingClient], MaskingAggregator]:
    """Create every participant's client and the server's aggregator, and exchange keys.

    In a real deployment each client is created on its own machine and the public keys
    travel through the server. Here that exchange is done in one place for simulations.
    """
    names = _check_names(participants)
    clients = {name: MaskingClient(name, modulus_bits) for name in names}
    public_keys = {name: client.public_key() for name, client in clients.items()}
    for client in clients.values():
        client.receive_public_keys(public_keys)
    return clients, MaskingAggregator(names, modulus_bits)


@dataclass(frozen=True)
class PairwiseMasking:
    """Settings for a secure sum by pairwise masking, for use in a ``Plan``.

    Args:
        value_bits: Bits $b$ each number of an update is rounded to. The message uses
            $b + \\lceil \\log_2 N \\rceil$ bits per number for $N$ participants.
        value_range: Each number of a (weighted) update is cut off at plus or minus this
            value before rounding. The result of a run reports how many were cut off.
    """

    value_bits: int = 12
    value_range: float = 1.0
    unsafe_code: str | None = None

    def __post_init__(self) -> None:
        if not 2 <= self.value_bits <= 24:
            raise ValueError("value_bits must be between 2 and 24")
        if not self.value_range > 0:
            raise ValueError("value_range must be positive")

    def modulus_bits(self, participants: int) -> int:
        """Bits per number on the wire for this many participants."""
        return sum_bits(self.value_bits, participants)

    def message_bytes(self, length: int, participants: int) -> int:
        """Bytes one participant uploads per sum for an update of ``length`` numbers."""
        return packed_size(length, self.modulus_bits(participants))

    def setup_bytes(self, participants: int) -> int:
        """One-time bytes per participant: its own public key up, the others' keys down."""
        return PUBLIC_KEY_BYTES * participants

    def build(
        self, participants: Sequence[str]
    ) -> tuple[dict[str, MaskingClient], MaskingAggregator]:
        """Create the clients and the aggregator for these participants."""
        return setup_pairwise_masking(participants, self.modulus_bits(len(participants)))

    def protection(self, participants: int) -> tuple[str, ...]:
        """Who is protected from whom, in plain words."""
        colluders = max(0, participants - 2)
        return (
            "The server receives every update hidden under masks that only cancel in the sum. "
            "It can read the sum of all updates and no single one.",
            f"This holds if the server passes the participants' public keys on unchanged, and "
            f"against the server working together with up to {colluders} of the {participants} "
            "participants.",
            "If one participant's message is missing, the sum cannot be opened and the round "
            "must be repeated.",
        )
