"""Handling a rare outcome: weights from a public figure, and what is refused."""

import numpy as np
import pytest
import torch
from torch import nn

from obscuradp.imbalance import ClassWeighted, NoRebalancing, UnsafeOversampling
from obscuradp.mechanisms import CLIP_MARGIN, clipped_sum, per_record_gradients
from obscuradp.models import TabularMLP
from obscuradp.unsafe import UnsafeConfiguration


def test_plain_loss_is_cross_entropy_per_row() -> None:
    output = torch.tensor([0.9, 0.2, 0.5])
    target = torch.tensor([1.0, 0.0, 1.0])
    loss = NoRebalancing().loss(from_logits=False)(output, target)
    assert loss.shape == (3,)
    assert torch.allclose(loss, nn.BCELoss(reduction="none")(output, target))
    logits = torch.tensor([2.0, -1.0, 0.0])
    from_logits = NoRebalancing().loss(from_logits=True)(logits, target)
    assert torch.allclose(from_logits, nn.BCEWithLogitsLoss(reduction="none")(logits, target))
    assert NoRebalancing().clip_scale(target) is None
    assert NoRebalancing().unsafe_code is None


def test_weights_from_a_public_prior() -> None:
    weighting = ClassWeighted(positive_prior=0.05)
    negative, positive = weighting.weights
    assert positive == pytest.approx(10.0)
    assert negative == pytest.approx(0.5 / 0.95)
    # In a population with that prior the average weight is 1 and both classes weigh the same.
    assert 0.05 * positive + 0.95 * negative == pytest.approx(1.0)
    assert 0.05 * positive == pytest.approx(0.95 * negative)
    assert weighting.unsafe_code is None
    output = torch.tensor([0.9, 0.2])
    target = torch.tensor([1.0, 0.0])
    plain = nn.BCELoss(reduction="none")(output, target)
    weighted = weighting.loss(from_logits=False)(output, target)
    assert torch.allclose(weighted, plain * torch.tensor([positive, negative]))
    for bad in (0.0, 1.0, -0.2):
        with pytest.raises(ValueError, match="positive_prior"):
            ClassWeighted(bad)


def test_weights_also_scale_the_clipping_norm_and_never_exceed_it() -> None:
    weighting = ClassWeighted(positive_prior=0.2)  # weights 0.625 and 2.5
    target = torch.tensor([1.0, 0.0, 0.0, 1.0])
    scale = weighting.clip_scale(target)
    assert scale is not None
    assert scale.tolist() == pytest.approx([1.0, 0.25, 0.25, 1.0])
    # A common positive class is the lighter one: then it gets the smaller limit.
    flipped = ClassWeighted(positive_prior=0.8).clip_scale(target)
    assert flipped is not None
    assert flipped.tolist() == pytest.approx([0.25, 1.0, 1.0, 0.25])

    # Four long gradients (length 10 each): every one is cut to its own limit.
    gradients = {"a": torch.full((4, 4), 5.0)}
    for i, factor in enumerate(scale.tolist()):
        alone = clipped_sum({"a": gradients["a"][i : i + 1]}, 1.0, scale[i : i + 1])
        length = float(alone["a"].double().norm())
        assert length == pytest.approx(factor * (1 - CLIP_MARGIN), rel=1e-5)
    total = clipped_sum(gradients, 1.0, scale)
    assert float(total["a"].double().norm()) == pytest.approx(2.5 * (1 - CLIP_MARGIN), rel=1e-5)

    # On real gradients no record exceeds its own limit, and none exceeds the clipping norm.
    torch.manual_seed(0)
    model = TabularMLP(3, (4,))
    x = torch.randn(4, 3) * 50.0
    real = per_record_gradients(model, weighting.loss(from_logits=False), x, target)
    for i, factor in enumerate(scale.tolist()):
        alone = clipped_sum({k: v[i : i + 1] for k, v in real.items()}, 1.0, scale[i : i + 1])
        length = float(torch.sqrt(sum(v.double().pow(2).sum() for v in alone.values())))
        assert length <= factor <= 1.0


def test_clip_scale_must_be_a_factor_per_record_between_zero_and_one() -> None:
    gradients = {"a": torch.ones(3, 2)}
    assert clipped_sum(gradients, 1.0, torch.ones(3))["a"].shape == (2,)
    with pytest.raises(ValueError, match="one factor per record"):
        clipped_sum(gradients, 1.0, torch.ones(2))
    for bad in ([1.0, 1.5, 1.0], [1.0, 0.0, 1.0], [1.0, -0.5, 1.0], [1.0, float("nan"), 1.0]):
        with pytest.raises(ValueError, match=r"in \(0, 1\]"):
            clipped_sum(gradients, 1.0, torch.tensor(bad))


def test_weights_from_private_labels_are_refused_unless_forced() -> None:
    labels = np.array([1, 0, 0, 0])
    with pytest.raises(UnsafeConfiguration, match="releases the label counts"):
        ClassWeighted.from_private_labels(labels)
    forced = ClassWeighted.from_private_labels(labels, allow_unsafe=True)
    assert forced.positive_prior == pytest.approx(0.25)
    assert forced.unsafe_code == "class_weights_from_private_labels"
    assert 0 < ClassWeighted.from_private_labels([0, 0], allow_unsafe=True).positive_prior < 1e-5
    with pytest.raises(ValueError, match="at least one label"):
        ClassWeighted.from_private_labels([], allow_unsafe=True)


def test_oversampling_is_marked_unsafe() -> None:
    def double(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
        return np.concatenate([x, x]), np.concatenate([y, y])

    method = UnsafeOversampling(double, name="doubling")
    assert method.unsafe_code == "oversampling_before_private_training"
    assert method.clip_scale(torch.ones(2)) is None
    output, target = torch.tensor([0.7]), torch.tensor([1.0])
    assert torch.allclose(
        method.loss(from_logits=False)(output, target), nn.BCELoss(reduction="none")(output, target)
    )
