"""Three independent ways to compute per-record gradients must agree.

The library's own engine (``torch.func``), Opacus (the library the research code used) and
a plain loop with one backward pass per record share no code. If all three give the same
numbers, a bug in any one of them would have to be matched by the same bug in the others.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from obscuradp.mechanisms import clipped_sum, per_record_gradients, record_norms
from obscuradp.models import TabularMLP

pytestmark = [pytest.mark.integration, pytest.mark.filterwarnings("ignore")]

CASES = [
    # (inputs, hidden layers, records, probability output)
    (15, (64, 32), 32, True),  # the stroke network of the research code
    (11, (32, 16), 32, True),  # the loan network
    (11, (), 17, True),  # logistic regression
    (6, (8, 8, 4), 5, False),  # deeper, trained on raw scores
]


def _setup(
    inputs: int, hidden: tuple[int, ...], records: int, probability_output: bool, seed: int
) -> tuple[TabularMLP, nn.Module, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    model = TabularMLP(inputs, hidden, probability_output=probability_output)
    x = torch.randn(records, inputs) * 2.0
    y = (torch.rand(records) > 0.6).float()
    loss_fn: nn.Module = nn.BCELoss() if probability_output else nn.BCEWithLogitsLoss()
    return model, loss_fn, x, y


def _with_opacus(
    model: nn.Module, loss_fn: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> dict[str, torch.Tensor]:
    from opacus import GradSampleModule

    # Opacus must be told how the loss was reduced; with "sum" it reports raw per-record values.
    wrapped = GradSampleModule(model, loss_reduction="sum")
    wrapped.zero_grad()
    type(loss_fn)(reduction="sum")(wrapped(x), y).backward()
    result = {
        name.removeprefix("_module."): parameter.grad_sample.detach().clone()
        for name, parameter in wrapped.named_parameters()
    }
    wrapped.to_standard_module()
    model.zero_grad(set_to_none=True)
    return result


def _one_at_a_time(
    model: nn.Module, loss_fn: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> dict[str, torch.Tensor]:
    rows: dict[str, list[torch.Tensor]] = {name: [] for name, _ in model.named_parameters()}
    for i in range(x.shape[0]):
        model.zero_grad(set_to_none=True)
        loss_fn(model(x[i : i + 1]), y[i : i + 1]).backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None
            rows[name].append(parameter.grad.detach().clone())
    model.zero_grad(set_to_none=True)
    return {name: torch.stack(values) for name, values in rows.items()}


@pytest.mark.parametrize(("inputs", "hidden", "records", "probability_output"), CASES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_three_implementations_agree(
    inputs: int, hidden: tuple[int, ...], records: int, probability_output: bool, seed: int
) -> None:
    model, loss_fn, x, y = _setup(inputs, hidden, records, probability_output, seed)
    ours = per_record_gradients(model, loss_fn, x, y)
    loop = _one_at_a_time(model, loss_fn, x, y)
    opacus = _with_opacus(model, loss_fn, x, y)
    assert set(ours) == set(loop) == set(opacus)
    for name in ours:
        assert ours[name].shape == loop[name].shape == opacus[name].shape
        assert torch.allclose(ours[name], loop[name], rtol=0.0, atol=1e-6), name
        assert torch.allclose(ours[name], opacus[name], rtol=0.0, atol=1e-6), name
    # The lengths that decide the clipping agree too, and so does the clipped sum.
    assert torch.allclose(record_norms(ours), record_norms(opacus), rtol=1e-5, atol=1e-7)
    ours_sum, opacus_sum = clipped_sum(ours, 1.0), clipped_sum(opacus, 1.0)
    for name in ours_sum:
        assert torch.allclose(ours_sum[name], opacus_sum[name], rtol=0.0, atol=1e-5), name


def test_the_clipped_average_matches_what_opacus_feeds_its_optimizer() -> None:
    """Opacus's own clipping (flat per-record clipping to norm C), without noise."""
    from opacus.optimizers import DPOptimizer

    model, loss_fn, x, y = _setup(11, (32, 16), 24, True, seed=3)
    clip = 0.5
    ours = clipped_sum(per_record_gradients(model, loss_fn, x, y), clip)

    from opacus import GradSampleModule

    wrapped = GradSampleModule(model)
    optimizer = DPOptimizer(
        torch.optim.SGD(wrapped.parameters(), lr=0.0),
        noise_multiplier=0.0,
        max_grad_norm=clip,
        expected_batch_size=24,
    )
    optimizer.zero_grad()
    nn.BCELoss()(wrapped(x), y).backward()
    optimizer.clip_and_accumulate()
    for (name, _), summed in zip(wrapped.named_parameters(), optimizer.params, strict=True):
        key = name.removeprefix("_module.")
        # Ours clips to C * (1 - 1e-6); Opacus to C / (norm + 1e-6). Both differences are tiny.
        assert torch.allclose(ours[key], summed.summed_grad, rtol=1e-4, atol=1e-6), key
    wrapped.to_standard_module()
