"""Choosing what to encrypt without leaking through the choice."""

import pytest
import torch
from torch import nn

from obscuradp.accounting import GaussianEvent
from obscuradp.ledger import BudgetExceeded, PrivacyLedger
from obscuradp.mechanisms import clipped_sum, per_record_gradients
from obscuradp.models import TabularMLP
from obscuradp.policy import PrivacyPolicy
from obscuradp.selection import (
    mask_from_private_data,
    mask_from_scores,
    private_scores,
    public_mask,
)
from obscuradp.unsafe import UnsafeConfiguration, UnsafeFlags

DELTA = 1e-5


def _setup(seed: int = 0) -> tuple[TabularMLP, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    model = TabularMLP(4, (6,))
    x = torch.randn(30, 4)
    y = (x[:, 0] > 0).float()
    return model, x, y


def _loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy(output, target, reduction="none")


def test_the_default_mask_depends_on_nothing_but_a_public_seed() -> None:
    mask = public_mask(1000, 0.25, seed=3)
    assert mask.dtype == bool
    assert int(mask.sum()) == 250
    assert (mask == public_mask(1000, 0.25, seed=3)).all()
    assert (mask != public_mask(1000, 0.25, seed=4)).any()
    assert int(public_mask(10, 0.0, 0).sum()) == 0
    assert int(public_mask(10, 1.0, 0).sum()) == 10
    with pytest.raises(ValueError, match="fraction"):
        public_mask(10, 1.5, 0)
    with pytest.raises(ValueError, match="n_total"):
        public_mask(0, 0.5, 0)


def test_noisy_scores_are_charged_before_they_are_computed() -> None:
    model, x, y = _setup()
    generator = torch.Generator().manual_seed(1)
    untouched = generator.get_state().clone()
    tight = PrivacyLedger(PrivacyPolicy(0.1, DELTA))
    with pytest.raises(BudgetExceeded):
        private_scores(
            model, _loss, x, y, clip_norm=1.0, noise_multiplier=1.0, ledger=tight, scope="a",
            generator=generator,
        )  # fmt: skip
    assert tight.scopes == ()
    assert torch.equal(generator.get_state(), untouched)

    ledger = PrivacyLedger(PrivacyPolicy(5.0, DELTA))
    scores = private_scores(
        model, _loss, x, y, clip_norm=1.0, noise_multiplier=4.0, ledger=ledger, scope="a",
        generator=generator,
    )  # fmt: skip
    assert ledger.events("a") == (GaussianEvent(4.0, 1),)
    assert set(scores) == {name for name, _ in model.named_parameters()}
    assert all(value > 0 for value in scores.values())


def test_noisy_scores_follow_the_stated_formula() -> None:
    """Noise sized for exactly the records used (the fix for finding G14)."""
    model, x, y = _setup()
    ledger = PrivacyLedger(PrivacyPolicy(50.0, DELTA))
    scores = private_scores(
        model, _loss, x, y, clip_norm=0.5, noise_multiplier=2.0, ledger=ledger, scope="a",
        generator=torch.Generator().manual_seed(5),
    )  # fmt: skip
    total = clipped_sum(per_record_gradients(model, _loss, x, y), 0.5)
    replay = torch.Generator().manual_seed(5)
    for name, value in total.items():
        noise = torch.normal(0.0, 2.0 * 0.5, size=value.shape, generator=replay)
        assert scores[name] == pytest.approx(float(((value + noise) / 30).abs().mean()), rel=1e-5)


def test_a_mask_from_scores_takes_the_highest_scoring_tensors() -> None:
    model, _, _ = _setup()
    scores = {"net.0.weight": 0.1, "net.0.bias": 9.0, "net.2.weight": 5.0, "net.2.bias": 0.0}
    mask = mask_from_scores(model, scores, 12 / 37)  # 24 + 6 + 6 + 1 = 37 numbers
    assert int(mask.sum()) == 12
    assert mask[24:30].all()  # net.0.bias, the highest score
    assert mask[30:36].all()  # net.2.weight, the second highest
    assert not mask[:24].any()
    with pytest.raises(ValueError, match="fraction"):
        mask_from_scores(model, scores, -0.1)


def test_a_mask_from_private_data_is_refused_unless_forced() -> None:
    model, x, y = _setup()
    with pytest.raises(UnsafeConfiguration, match="visible to the server"):
        mask_from_private_data(model, _loss, x, y, 0.3, UnsafeFlags())
    flags = UnsafeFlags(allow_unsafe=True)
    mask = mask_from_private_data(model, _loss, x, y, 0.3, flags)
    assert flags.codes == ("mask_from_private_data",)
    assert int(mask.sum()) == round(0.3 * 37)
    assert all(parameter.grad is None for parameter in model.parameters())
