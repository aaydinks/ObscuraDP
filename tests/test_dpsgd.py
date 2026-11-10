"""The private gradient step: per-record gradients, clipping and noise."""

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import nn

from obscuradp.mechanisms import (
    CLIP_MARGIN,
    check_per_record_model,
    clipped_sum,
    per_record_gradients,
    private_gradient,
    record_norms,
)
from obscuradp.mechanisms.dpsgd import Gradients
from obscuradp.models import TabularMLP


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _data(n: int, d: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = _generator(seed)
    x = torch.randn(n, d, generator=generator)
    y = (torch.rand(n, generator=generator) > 0.5).float()
    return x, y


def _model(d: int = 5, seed: int = 0, **kwargs: object) -> TabularMLP:
    torch.manual_seed(seed)
    return TabularMLP(d, (8, 4), **kwargs)  # type: ignore[arg-type]


def _one_at_a_time(
    model: nn.Module, loss_fn: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> Gradients:
    """The slow, obviously correct way: one backward pass per record."""
    rows: dict[str, list[torch.Tensor]] = {name: [] for name, _ in model.named_parameters()}
    for i in range(x.shape[0]):
        model.zero_grad(set_to_none=True)
        loss_fn(model(x[i : i + 1]), y[i : i + 1]).sum().backward()
        for name, value in model.named_parameters():
            assert value.grad is not None
            rows[name].append(value.grad.detach().clone())
    model.zero_grad(set_to_none=True)
    return {name: torch.stack(values) for name, values in rows.items()}


def _length(gradients: Gradients) -> float:
    return math.sqrt(sum(float(value.double().pow(2).sum()) for value in gradients.values()))


# --------------------------------------------------------------------------- per-record gradients


@pytest.mark.parametrize(
    ("probability_output", "loss_fn"),
    [
        (True, nn.BCELoss()),
        (True, nn.BCELoss(reduction="sum")),
        (True, nn.BCELoss(reduction="none")),
        (False, nn.BCEWithLogitsLoss()),
        (False, nn.MSELoss()),
    ],
)
def test_per_record_gradients_equal_one_backward_pass_per_record(
    probability_output: bool, loss_fn: nn.Module
) -> None:
    model = _model(probability_output=probability_output)
    x, y = _data(9, 5, seed=1)
    fast = per_record_gradients(model, loss_fn, x, y)
    slow = _one_at_a_time(model, loss_fn, x, y)
    assert list(fast) == [name for name, _ in model.named_parameters()]
    for name, value in fast.items():
        assert value.shape == (9, *dict(model.named_parameters())[name].shape)
        assert torch.allclose(value, slow[name], rtol=0.0, atol=1e-6), name
    # Computing them leaves the model untouched.
    assert all(value.grad is None for value in model.parameters())


def test_a_record_gradient_does_not_depend_on_the_other_records() -> None:
    model = _model()
    x, y = _data(6, 5, seed=2)
    whole = per_record_gradients(model, nn.BCELoss(), x, y)
    alone = per_record_gradients(model, nn.BCELoss(), x[2:3], y[2:3])
    other_x, other_y = _data(6, 5, seed=3)
    other_x[2], other_y[2] = x[2], y[2]
    mixed = per_record_gradients(model, nn.BCELoss(), other_x, other_y)
    for name in whole:
        assert torch.allclose(whole[name][2], alone[name][0], atol=1e-7)
        assert torch.allclose(whole[name][2], mixed[name][2], atol=1e-7)


def test_empty_batches_frozen_parameters_and_bad_input() -> None:
    model = _model()
    x, y = _data(4, 5, seed=4)
    empty = per_record_gradients(model, nn.BCELoss(), x[:0], y[:0])
    assert {name: value.shape[0] for name, value in empty.items()} == dict.fromkeys(empty, 0)
    assert empty["net.0.weight"].shape == (0, 8, 5)

    model.net[0].weight.requires_grad_(False)
    assert "net.0.weight" not in per_record_gradients(model, nn.BCELoss(), x, y)
    with pytest.raises(ValueError, match="same number of records"):
        per_record_gradients(model, nn.BCELoss(), x, y[:3])


def test_dropout_gets_its_own_draw_for_every_record() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(5, 16), nn.Dropout(0.5), nn.Linear(16, 1), nn.Flatten(0))
    model.train()
    x, y = _data(1, 5, seed=5)
    same_x, same_y = x.repeat(8, 1), y.repeat(8)
    gradients = per_record_gradients(model, nn.BCEWithLogitsLoss(), same_x, same_y)
    # Identical records, different dropout masks: their gradients differ.
    assert not torch.allclose(gradients["0.weight"][0], gradients["0.weight"][1])


@pytest.mark.parametrize(
    ("layer", "refused"),
    [
        (nn.BatchNorm1d(4), True),
        (nn.LazyBatchNorm1d(), True),
        (nn.InstanceNorm1d(4, track_running_stats=True), True),
        (nn.InstanceNorm1d(4), False),
        (nn.GroupNorm(2, 4), False),
        (nn.LayerNorm(4), False),
        (nn.Dropout(0.1), False),
    ],
)
def test_layers_that_mix_records_are_refused(layer: nn.Module, refused: bool) -> None:
    model = nn.Sequential(nn.Linear(3, 4), layer, nn.Linear(4, 1))
    if refused:
        with pytest.raises(ValueError, match="mixes the records of a batch"):
            check_per_record_model(model)
    else:
        check_per_record_model(model)


# --------------------------------------------------------------------------- clipping


def test_norms_are_taken_over_all_parameters_together() -> None:
    model = _model()
    x, y = _data(5, 5, seed=6)
    gradients = per_record_gradients(model, nn.BCELoss(), x, y)
    norms = record_norms(gradients)
    assert norms.dtype == torch.float64
    for i in range(5):
        by_hand = math.sqrt(sum(float(v[i].double().pow(2).sum()) for v in gradients.values()))
        assert float(norms[i]) == pytest.approx(by_hand, rel=1e-12)
    # Larger than any single layer's norm: clipping layer by layer would be a different rule.
    assert float(norms[0]) > float(gradients["net.0.weight"][0].norm())
    with pytest.raises(ValueError, match="no trainable parameter"):
        record_norms({})


def test_long_gradients_are_shrunk_and_short_ones_left_alone() -> None:
    gradients = {
        "a": torch.tensor([[3.0, 0.0], [0.1, 0.0], [0.0, 0.0]]),
        "b": torch.tensor([[4.0], [0.0], [0.0]]),
    }
    assert record_norms(gradients).tolist() == pytest.approx([5.0, 0.1, 0.0])
    total = clipped_sum(gradients, clip_norm=1.0)
    scale = (1.0 - CLIP_MARGIN) / 5.0
    assert total["a"].tolist() == pytest.approx([3.0 * scale + 0.1, 0.0])
    assert total["b"].tolist() == pytest.approx([4.0 * scale])
    # With a generous limit nothing is shrunk: the result is the plain sum.
    plain = clipped_sum(gradients, clip_norm=100.0)
    assert plain["a"].tolist() == pytest.approx([3.1, 0.0])
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="clip_norm"):
            clipped_sum(gradients, bad)


def test_a_record_with_a_broken_gradient_contributes_nothing() -> None:
    gradients = {
        "a": torch.tensor([[float("nan"), 1.0], [float("inf"), 0.0], [0.3, 0.4]]),
    }
    total = clipped_sum(gradients, clip_norm=1.0)
    assert total["a"].tolist() == pytest.approx([0.3, 0.4])


@settings(max_examples=40, deadline=None)
@given(
    n=st.integers(1, 12),
    seed=st.integers(0, 10_000),
    clip_norm=st.floats(0.01, 5.0),
    scale=st.floats(1e-3, 1e3),
    drop=st.integers(0, 11),
)
def test_one_record_never_moves_the_sum_by_more_than_the_clipping_norm(
    n: int, seed: int, clip_norm: float, scale: float, drop: int
) -> None:
    """The fact the whole guarantee rests on, checked on real gradients of any size."""
    model = _model(seed=seed)
    x, y = _data(n, 5, seed=seed + 1)
    x = x * scale
    gradients = per_record_gradients(model, nn.BCELoss(), x, y)
    with_all = clipped_sum(gradients, clip_norm)
    keep = [i for i in range(n) if i != drop % n]
    without_one = clipped_sum({name: value[keep] for name, value in gradients.items()}, clip_norm)
    moved = _length({name: with_all[name] - without_one[name] for name in with_all})
    # float32 sums of up to 12 terms: allow a few rounding errors on top of the exact bound.
    assert moved <= clip_norm * (1.0 + 1e-5)
    # And each record alone is strictly inside the limit, thanks to the safety margin.
    for i in range(n):
        alone = clipped_sum(
            {name: value[i : i + 1] for name, value in gradients.items()}, clip_norm
        )
        assert _length(alone) <= clip_norm


# --------------------------------------------------------------------------- the noisy step


def test_the_private_gradient_is_the_clipped_sum_plus_noise_over_a_fixed_divisor() -> None:
    model = _model()
    x, y = _data(6, 5, seed=7)
    settings_ = {"clip_norm": 0.5, "noise_multiplier": 1.5, "expected_batch_size": 3.7}
    noisy = private_gradient(model, nn.BCELoss(), x, y, generator=_generator(8), **settings_)
    total = clipped_sum(per_record_gradients(model, nn.BCELoss(), x, y), 0.5)
    replay = _generator(8)
    for name, value in total.items():  # the noise is drawn parameter by parameter, in order
        noise = torch.normal(0.0, 1.5 * 0.5, size=value.shape, generator=replay)
        assert torch.allclose(noisy[name], (value + noise) / 3.7, atol=1e-7), name
    again = private_gradient(model, nn.BCELoss(), x, y, generator=_generator(8), **settings_)
    assert all(torch.equal(noisy[name], again[name]) for name in noisy)


def test_noise_level_is_sigma_times_clip_over_expected_batch_size() -> None:
    """Statistical check on 60,000 noisy coordinates, with a fixed seed."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(40, 30), nn.Flatten(0))  # 1,230 parameters
    model[0].weight.data.zero_()
    x = torch.randn(4, 40)
    y = torch.zeros(4)
    sigma, clip, divisor = 2.0, 0.7, 12.5
    expected_spread = sigma * clip / divisor  # 0.112
    generator = _generator(9)
    total = clipped_sum(per_record_gradients(model, _row_sum_loss, x, y), clip)
    centre = torch.cat([value.flatten() for value in total.values()]) / divisor
    draws = []
    for _ in range(50):
        noisy = private_gradient(
            model,
            _row_sum_loss,
            x,
            y,
            clip_norm=clip,
            noise_multiplier=sigma,
            expected_batch_size=divisor,
            generator=generator,
        )
        draws.append(torch.cat([value.flatten() for value in noisy.values()]))
    stacked = torch.stack(draws).double()  # 50 x 1,230
    residual = stacked - centre.double()
    count = residual.numel()
    # Mean of the residual: 0 with standard deviation spread / sqrt(count).
    assert abs(float(residual.mean())) < 4.5 * expected_spread / math.sqrt(count)
    # Standard deviation: within 1.5% (five sigma for 61,500 draws).
    assert float(residual.std()) == pytest.approx(expected_spread, rel=0.015)
    # Each coordinate is centred on the clipped sum divided by the fixed divisor.
    per_coordinate_error = (stacked.mean(dim=0) - centre.double()).abs().max()
    assert float(per_coordinate_error) < 5.0 * expected_spread / math.sqrt(50)
    # Different coordinates get independent noise.
    assert abs(float(torch.corrcoef(residual[:, :2].T)[0, 1])) < 0.6
    assert abs(
        float((residual[:, 0::2] * residual[:, 1::2]).mean())
    ) < 4.5 * expected_spread**2 / math.sqrt(count / 2)


def _row_sum_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return output.sum() + 0.0 * target.sum()


def test_an_empty_batch_still_releases_noise() -> None:
    model = _model()
    x, y = _data(3, 5, seed=10)
    noisy = private_gradient(
        model,
        nn.BCELoss(),
        x[:0],
        y[:0],
        clip_norm=1.0,
        noise_multiplier=3.0,
        expected_batch_size=2.0,
        generator=_generator(11),
    )
    flat = torch.cat([value.flatten() for value in noisy.values()])
    assert flat.numel() == 89
    assert float(flat.std()) == pytest.approx(1.5, rel=0.25)  # 3.0 * 1.0 / 2.0
    assert float(flat.abs().min()) > 0.0


def test_a_step_without_noise_or_without_a_divisor_is_refused() -> None:
    model = _model()
    x, y = _data(3, 5, seed=12)
    for sigma in (0.0, -1.0):
        with pytest.raises(ValueError, match="noise_multiplier"):
            private_gradient(
                model,
                nn.BCELoss(),
                x,
                y,
                clip_norm=1.0,
                noise_multiplier=sigma,
                expected_batch_size=2.0,
                generator=_generator(0),
            )
    with pytest.raises(ValueError, match="expected_batch_size"):
        private_gradient(
            model,
            nn.BCELoss(),
            x,
            y,
            clip_norm=1.0,
            noise_multiplier=1.0,
            expected_batch_size=0.0,
            generator=_generator(0),
        )
