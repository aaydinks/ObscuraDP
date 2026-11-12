"""Attack the private trainer and compare what the attack proves with what the ledger says.

The attacker here is the strongest the guarantee allows for: it sees the noisy gradient of
every step, knows every other record, and only has to decide whether one known record (the
"canary") was in the training table. Whatever such an attack can prove is a lower limit on
the true privacy budget. A correct trainer never shows a limit above the ledger's figure.
A trainer that adds half the noise it reports must be caught.
"""

import math

import pytest

from obscuradp.accounting import gaussian_epsilon
from obscuradp.audit import gaussian_epsilon_lower_bound
from obscuradp.audit.mechanism import (
    DELTA,
    counted_epsilon_lower_bound,
    local_step_scores,
    sampled_training_scores,
)
from obscuradp.mechanisms import dpsgd as dpsgd_module


def _full_batch_scores(
    with_canary: bool, trials: int, sigma: float, steps: int, seed: int
) -> tuple[list[float], float]:
    """The attack itself is public: ``obscuradp.audit.mechanism.local_step_scores``."""
    return local_step_scores(with_canary, trials, noise_multiplier=sigma, steps=steps, seed=seed)


STEPS = 2
# Two full-batch steps at this noise level cost a budget of 1 at delta = 1e-5.
SIGMA = 3.73 * math.sqrt(STEPS)
TRIALS = 2000


def test_the_ledger_figure_is_the_exact_gaussian_answer_here() -> None:
    _, spent = _full_batch_scores(True, 1, SIGMA, STEPS, seed=0)
    assert spent == pytest.approx(gaussian_epsilon(SIGMA, DELTA, count=STEPS), rel=1e-3)
    assert spent == pytest.approx(1.0, abs=0.01)


def test_the_strongest_attack_proves_no_more_than_the_ledger_reports() -> None:
    without, spent = _full_batch_scores(False, TRIALS, SIGMA, STEPS, seed=10)
    with_canary, _ = _full_batch_scores(True, TRIALS, SIGMA, STEPS, seed=20)
    proven = gaussian_epsilon_lower_bound(without, with_canary, DELTA, confidence=0.999)
    assert proven <= spent
    # The attack is sharp: it proves most of the budget, so the accounting is not loose either.
    estimate = gaussian_epsilon_lower_bound(without, with_canary, DELTA, confidence=1e-9)
    assert estimate == pytest.approx(spent, rel=0.3)


def test_half_the_reported_noise_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation test: break the noise inside the mechanism and the audit must notice."""
    honest = dpsgd_module.gaussian_noise

    def half_noise(shape, standard_deviation, generator, **kwargs):  # type: ignore[no-untyped-def]
        return honest(shape, standard_deviation / 2.0, generator, **kwargs)

    monkeypatch.setattr(dpsgd_module, "gaussian_noise", half_noise)
    without, spent = _full_batch_scores(False, TRIALS, SIGMA, STEPS, seed=10)
    with_canary, _ = _full_batch_scores(True, TRIALS, SIGMA, STEPS, seed=20)
    assert spent == pytest.approx(1.0, abs=0.01)  # the ledger still believes the full noise
    proven = gaussian_epsilon_lower_bound(without, with_canary, DELTA, confidence=0.999)
    assert proven > 1.5 * spent


def test_unclipped_gradients_are_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Second mutation: clip to three times the reported norm."""
    honest = dpsgd_module.clipped_sum
    monkeypatch.setattr(
        dpsgd_module,
        "clipped_sum",
        lambda gradients, clip_norm, clip_scale=None: honest(
            gradients, 3.0 * clip_norm, clip_scale
        ),
    )
    without, spent = _full_batch_scores(False, TRIALS, SIGMA, STEPS, seed=10)
    with_canary, _ = _full_batch_scores(True, TRIALS, SIGMA, STEPS, seed=20)
    proven = gaussian_epsilon_lower_bound(without, with_canary, DELTA, confidence=0.999)
    assert proven > 1.5 * spent


@pytest.mark.slow
def test_white_box_attack_on_sampled_training_stays_below_the_ledger() -> None:
    """Nightly check with Poisson sampling and a real network, using exact error counts."""
    trials = 2000
    without, spent = sampled_training_scores(False, trials, seed=100)
    with_canary, _ = sampled_training_scores(True, trials, seed=200)
    # The cut-off is chosen on the first half, the errors are counted on the second half.
    proven = counted_epsilon_lower_bound(without, with_canary, DELTA, confidence=0.999)
    assert proven <= spent
    assert proven > 0.3  # the attack does prove something, so the comparison has teeth
