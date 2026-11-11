"""The strongest-attacker checks as a public tool (``obscuradp.audit.mechanism``).

The attacks themselves are exercised at full strength where they have always been
(``test_dpsgd_audit``, ``test_distributed``, ``test_participant``), together with the
planted defects. Here: the result object, the judging of two worlds, and that each ready
check passes the library's mechanism and flags a broken one handed to it.
"""

from typing import Any

import numpy as np
import pytest
import torch

from obscuradp.accounting import gaussian_epsilon
from obscuradp.audit import mechanism
from obscuradp.audit.mechanism import (
    DELTA,
    UNIT_BUDGET_NOISE,
    LinearProbe,
    audit_all,
    audit_local_step,
    audit_shared_sum,
    audit_whole_number_sum,
    audit_whole_participant,
    compare_worlds,
    counted_epsilon_lower_bound,
    linear_loss,
)
from obscuradp.audit.selftest import half_noise_round
from obscuradp.client import private_sum_step


def test_the_probe_model_has_one_gradient_per_record_whatever_the_weights() -> None:
    model = LinearProbe(3)
    rows = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    loss = linear_loss(model(rows), torch.ones(2)).sum()
    loss.backward()  # type: ignore[no-untyped-call]
    assert model.w.grad is not None
    assert model.w.grad.tolist() == [-1.0, -2.0, -3.0]


def test_two_worlds_of_bell_curves_are_judged_against_the_reported_budget() -> None:
    generator = np.random.default_rng(0)
    truth = gaussian_epsilon(UNIT_BUDGET_NOISE, DELTA)
    without = generator.normal(0.0, UNIT_BUDGET_NOISE, size=4000).tolist()
    with_target = generator.normal(1.0, UNIT_BUDGET_NOISE, size=4000).tolist()
    honest = compare_worlds("made up", without, with_target, truth)
    assert honest.trials == 4000
    assert honest.epsilon_lower <= truth
    assert honest.epsilon_estimate == pytest.approx(truth, rel=0.3)
    assert not honest.flagged
    assert "not flagged" in honest.describe()
    assert "best estimate" in honest.describe()
    # The same scores from a mechanism that reports a third of what it leaks.
    overstated = compare_worlds("made up", without, with_target, truth / 3)
    assert overstated.flagged
    assert "FLAGGED" in overstated.describe()
    assert chr(0x2014) not in overstated.describe()


def test_counting_errors_gives_a_limit_for_scores_of_any_shape() -> None:
    generator = np.random.default_rng(1)
    # One Gaussian release at noise multiplier 1: the true budget at delta = 1e-5 is known.
    truth = gaussian_epsilon(1.0, DELTA)
    without = generator.normal(0.0, 1.0, size=3000).tolist()
    with_target = generator.normal(1.0, 1.0, size=3000).tolist()
    proven = counted_epsilon_lower_bound(without, with_target)
    assert 0.3 < proven <= truth
    # Scores that carry nothing prove nothing.
    assert counted_epsilon_lower_bound(without, without[::-1]) == 0.0
    with pytest.raises(ValueError, match="at least two trials"):
        counted_epsilon_lower_bound([0.0], [1.0])


def test_the_private_step_passes_and_a_step_with_half_its_noise_is_flagged() -> None:
    honest = audit_local_step(1000)
    assert honest.epsilon_reported == pytest.approx(1.0, abs=0.01)
    assert not honest.flagged
    broken = audit_local_step(2000, train_round=half_noise_round)
    assert broken.epsilon_reported == pytest.approx(1.0, abs=0.01)  # what its ledger was told
    assert broken.flagged
    assert broken.epsilon_lower > 1.3


def _thin_slice(*arguments: Any, share_std: float, **settings: Any) -> torch.Tensor:
    """A broken shared-noise step: it adds a third of the slice the plan counts on."""
    return private_sum_step(*arguments, share_std=share_std / 3.0, **settings)


def test_the_shared_sum_passes_and_a_thin_slice_is_flagged() -> None:
    honest = audit_shared_sum(1000)
    assert honest.epsilon_reported == pytest.approx(1.0, abs=0.01)
    assert not honest.flagged
    assert audit_shared_sum(2000, sum_step=_thin_slice).flagged


def test_the_whole_number_sum_and_the_whole_participant_mode_pass() -> None:
    whole = audit_whole_number_sum(1000)
    assert whole.epsilon_reported == pytest.approx(1.0, abs=0.01)
    assert not whole.flagged
    participant = audit_whole_participant(60)
    assert participant.epsilon_reported == pytest.approx(1.0, abs=0.01)
    assert participant.trials == 60
    assert not participant.flagged


@pytest.mark.slow
def test_every_check_at_full_strength() -> None:
    """Nightly: no check flags the library, and each attack proves most of the budget."""
    results = audit_all()
    assert [result.name for result in results] == [
        "private training step",
        "shared noise, ordinary",
        "shared noise, whole numbers",
        "whole participants protected",
    ]
    for result in results:
        assert not result.flagged, result.describe()
        assert result.epsilon_estimate == pytest.approx(result.epsilon_reported, rel=0.4)
    assert mechanism.DELTA == DELTA
