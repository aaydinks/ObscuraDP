"""Settings that weaken or void the privacy guarantee, and how the library treats them.

Some things people routinely do around private training quietly break the guarantee: making
synthetic rows from real ones before the private step, computing statistics from private
data without paying for them, choosing a failure probability that is too large. The library
refuses each of these by default. A caller who insists passes ``allow_unsafe=True``; the run
then proceeds, and the reason is written into the privacy report in plain words, where
whoever reads the report will see it.
"""

from __future__ import annotations

UNSAFE_REASONS: dict[str, str] = {
    "oversampling_before_private_training": (
        "Synthetic rows were built from real rows before the private step, so one real record "
        "influences many training rows. The stated budget does not hold for real records."
    ),
    "class_weights_from_private_labels": (
        "Class weights were computed from private label counts, and that release was not "
        "charged to the budget."
    ),
    "weak_delta": (
        "delta is not below one over the number of records of at least one participant, so "
        "the guarantee allows a real chance that a whole record is exposed."
    ),
    "statistics_from_private_data": (
        "Preprocessing statistics (for example means, medians or category lists) were computed "
        "from private data and not charged to the budget."
    ),
    "mask_from_private_data": (
        "The choice of which numbers to encrypt was computed from private data without noise. "
        "That choice is visible to the server and was not charged to the budget."
    ),
    "ckks_research_only": (
        "CKKS encryption gives approximate results, and decrypted approximate results can help "
        "an attacker recover the key. This backend is for research with all roles under one "
        "roof, not for protecting real data."
    ),
}


class UnsafeConfiguration(ValueError):
    """A setting that weakens the privacy guarantee was used without ``allow_unsafe=True``."""


class UnsafeFlags:
    """Collects the unsafe settings of one run, or refuses them."""

    def __init__(self, allow_unsafe: bool = False) -> None:
        self.allow_unsafe = allow_unsafe
        self._raised: dict[str, str] = {}

    def raise_or_record(self, code: str, detail: str = "") -> None:
        """Refuse the setting ``code``, or record it if unsafe settings are allowed.

        Raises:
            UnsafeConfiguration: If unsafe settings are not allowed.
            KeyError: If ``code`` is not a known reason.
        """
        reason = UNSAFE_REASONS[code]
        message = f"{reason} ({detail})" if detail else reason
        if not self.allow_unsafe:
            raise UnsafeConfiguration(
                f"{message} Pass allow_unsafe=True to run anyway; the privacy report will then "
                "carry this warning."
            )
        self._raised.setdefault(code, message)

    @property
    def codes(self) -> tuple[str, ...]:
        """The reasons recorded so far, as short codes."""
        return tuple(self._raised)

    @property
    def messages(self) -> tuple[str, ...]:
        """The reasons recorded so far, in plain words, for the privacy report."""
        return tuple(self._raised.values())
