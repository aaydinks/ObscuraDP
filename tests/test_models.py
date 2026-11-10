"""The ready-made tabular network."""

import pytest
import torch
from torch import nn

from obscuradp.models import TabularMLP
from obscuradp.state import count_parameters, get_state


@pytest.mark.parametrize(
    ("in_features", "hidden", "expected"),
    [(15, (64, 32), 3137), (11, (32, 16), 929), (15, (), 16), (4, (8, 8, 8), 193)],
)
def test_parameter_counts(in_features: int, hidden: tuple[int, ...], expected: int) -> None:
    assert count_parameters(TabularMLP(in_features, hidden)) == expected


def test_layout_and_keys_match_the_research_network() -> None:
    model = TabularMLP(15, (64, 32))
    kinds = [type(layer) for layer in model.net]
    assert kinds == [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear, nn.Sigmoid]
    assert list(get_state(model)) == [
        "net.0.weight",
        "net.0.bias",
        "net.2.weight",
        "net.2.bias",
        "net.4.weight",
        "net.4.bias",
    ]


def test_output_is_one_probability_per_row() -> None:
    torch.manual_seed(0)
    out = TabularMLP(7, (5,))(torch.randn(13, 7))
    assert out.shape == (13,)
    assert bool(((out > 0) & (out < 1)).all())


def test_raw_scores_when_asked() -> None:
    torch.manual_seed(0)
    model = TabularMLP(3, (4,), probability_output=False)
    assert not any(isinstance(layer, nn.Sigmoid) for layer in model.net)
    out = model(torch.randn(50, 3) * 20)
    assert bool(((out < 0) | (out > 1)).any())


def test_no_layer_mixes_rows() -> None:
    """Each row's output must not depend on the other rows in the batch."""
    torch.manual_seed(0)
    model = TabularMLP(6, (8, 4))
    batch = torch.randn(10, 6)
    together = model(batch)
    alone = torch.stack([model(row.unsqueeze(0)).squeeze(0) for row in batch])
    assert torch.allclose(together, alone, atol=1e-7)


@pytest.mark.parametrize(("in_features", "hidden"), [(0, (4,)), (3, (0,)), (3, (4, -1))])
def test_bad_sizes_are_refused(in_features: int, hidden: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        TabularMLP(in_features, hidden)
