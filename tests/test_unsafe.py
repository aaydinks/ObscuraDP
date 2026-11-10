"""Settings that weaken the guarantee are refused unless explicitly allowed."""

import pytest

from obscuradp.unsafe import UNSAFE_REASONS, UnsafeConfiguration, UnsafeFlags


def test_unsafe_settings_are_refused_by_default() -> None:
    flags = UnsafeFlags()
    with pytest.raises(UnsafeConfiguration, match="allow_unsafe=True") as refusal:
        flags.raise_or_record("oversampling_before_private_training")
    assert "one real record influences many training rows" in str(refusal.value)
    assert flags.codes == ()
    assert flags.messages == ()
    assert issubclass(UnsafeConfiguration, ValueError)


def test_allowed_unsafe_settings_are_recorded_in_plain_words() -> None:
    flags = UnsafeFlags(allow_unsafe=True)
    flags.raise_or_record("weak_delta", "delta = 0.01; affected: clinic")
    flags.raise_or_record("weak_delta", "a second time")
    flags.raise_or_record("statistics_from_private_data")
    assert flags.codes == ("weak_delta", "statistics_from_private_data")
    assert flags.messages[0].endswith("(delta = 0.01; affected: clinic)")
    assert flags.messages[1] == UNSAFE_REASONS["statistics_from_private_data"]


def test_unknown_reasons_are_a_programming_error() -> None:
    with pytest.raises(KeyError):
        UnsafeFlags(allow_unsafe=True).raise_or_record("made_up")


def test_every_reason_is_a_full_sentence_without_long_dashes() -> None:
    for code, reason in UNSAFE_REASONS.items():
        assert code == code.lower()
        assert reason.endswith(".")
        assert chr(0x2014) not in reason
