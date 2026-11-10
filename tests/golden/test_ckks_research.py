"""The research-only CKKS wrapper: refused by default, roles kept apart, rounded output.

Needs TenSEAL, which comes with the ``legacy`` extra, so these tests run with the comparison
tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from obscuradp.compression import float32_size
from obscuradp.crypto.ckks import CkksAggregator, CkksClient, CkksKeyHolder
from obscuradp.crypto.roles import MissingParticipants
from obscuradp.legacy import LegacyTrialConfig
from obscuradp.legacy import he as legacy_he
from obscuradp.unsafe import UnsafeConfiguration, UnsafeFlags

pytestmark = [pytest.mark.integration, pytest.mark.filterwarnings("ignore")]

NAMES = ("a", "b", "c")


@pytest.fixture(scope="module")
def holder() -> CkksKeyHolder:
    return CkksKeyHolder(UnsafeFlags(allow_unsafe=True))


def test_ckks_is_refused_unless_unsafe_settings_are_allowed() -> None:
    with pytest.raises(UnsafeConfiguration, match="not for protecting real data"):
        CkksKeyHolder(UnsafeFlags())
    flags = UnsafeFlags(allow_unsafe=True)
    CkksKeyHolder(flags)
    assert flags.codes == ("ckks_research_only",)
    with pytest.raises(ValueError, match="decimals"):
        CkksKeyHolder(flags, decimals=9)


def test_the_encrypted_sum_matches_the_plain_sum_after_rounding(holder: CkksKeyHolder) -> None:
    public = holder.public_context()
    rng = np.random.default_rng(0)
    updates = {name: rng.normal(0.0, 0.1, size=5000) for name in NAMES}  # two ciphertexts each
    clients = {name: CkksClient(name, public) for name in NAMES}
    aggregator = CkksAggregator(NAMES, public)
    messages = {name: clients[name].protect(updates[name]) for name in NAMES}
    assert all(len(blobs) == 2 for blobs in messages.values())
    opened = holder.open_sum(aggregator.combine(messages), 5000)
    truth = sum(updates.values())
    assert opened.shape == (5000,)
    # Never 64-bit decrypted values: 32-bit numbers on a grid of four decimals.
    assert opened.dtype == np.float32
    assert np.array_equal(opened, np.round(opened.astype(np.float64), 4).astype(np.float32))
    assert float(np.abs(opened - truth).max()) < 1e-4
    with pytest.raises(ValueError, match="fewer values"):
        holder.open_sum(aggregator.combine(messages), 9000)


def test_the_aggregator_cannot_decrypt(holder: CkksKeyHolder) -> None:
    import tenseal

    public = holder.public_context()
    aggregator = CkksAggregator(NAMES, public)
    client = CkksClient("a", public)
    assert not aggregator.can_decrypt
    message = client.protect([0.25, -0.5])
    for party in (aggregator, client):  # neither the server nor a participant holds the key
        with pytest.raises(ValueError, match="secret_key"):
            tenseal.ckks_vector_from(party._context, message[0]).decrypt()


def test_no_unused_keys_are_generated_or_sent(holder: CkksKeyHolder) -> None:
    """The research code generated rotation keys it never used; they dominate the setup size."""
    with_rotation_keys = legacy_he.make_context(LegacyTrialConfig()).serialize(
        save_public_key=True, save_secret_key=False, save_galois_keys=True, save_relin_keys=True
    )
    assert len(holder.public_context()) * 20 < len(with_rotation_keys)


def test_every_participant_is_needed_and_updates_must_match(holder: CkksKeyHolder) -> None:
    public = holder.public_context()
    aggregator = CkksAggregator(NAMES, public)
    clients = {name: CkksClient(name, public) for name in NAMES}
    messages = {name: clients[name].protect([0.1, 0.2]) for name in NAMES}
    with pytest.raises(MissingParticipants):
        aggregator.combine({name: messages[name] for name in NAMES[:2]})
    messages["c"] = clients["c"].protect(np.zeros(5000))
    with pytest.raises(ValueError, match="same number of ciphertexts"):
        aggregator.combine(messages)


def test_ckks_messages_are_many_times_larger_than_plain_ones(holder: CkksKeyHolder) -> None:
    """Why this backend cannot meet the bytes target (finding G7)."""
    client = CkksClient("a", holder.public_context())
    message = client.protect(np.zeros(3137))  # the stroke network: one ciphertext
    ratio = sum(len(blob) for blob in message) / float32_size(3137)
    assert 10 < ratio < 40
