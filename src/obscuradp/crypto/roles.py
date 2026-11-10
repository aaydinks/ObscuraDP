"""Who holds what in a secure sum.

The whole point of a secure sum is that the party who adds the updates cannot read any
single one. That only holds if the pieces of the protocol are kept apart, so the library
models them as separate objects with separate knowledge:

- a ``ClientEncryptor`` runs on a participant's machine and turns that participant's
  numbers into a protected message;
- an ``Aggregator`` runs on the server, adds protected messages, and holds no secret;
- a ``KeyHolder`` (encryption backends only) holds the decryption key and is asked only for
  the final sum. It must not be the aggregator.

Pairwise masking needs no key holder: the masks cancel in the sum, so the aggregator can
read the sum, and only the sum, by itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np
import numpy.typing as npt

IntVector = npt.NDArray[np.uint64]


class MissingParticipants(RuntimeError):
    """A sum was requested without a message from every expected participant."""


class ClientEncryptor(Protocol):
    """Runs on a participant's machine."""

    name: str

    def protect(self, sum_index: int, values: npt.ArrayLike) -> bytes:
        """Turn this participant's whole numbers into a message only the sum can use."""
        ...


class Aggregator(Protocol):
    """Runs on the server. Holds nothing secret."""

    participants: tuple[str, ...]

    def combine(self, sum_index: int, messages: Mapping[str, bytes], length: int) -> IntVector:
        """Add one message from every participant and return the sum of ``length`` numbers."""
        ...
