"""Reading a record back out of an update: exact when it stands alone, lost under noise."""

import numpy as np
import pytest
import torch
from torch import nn

from obscuradp.audit.inversion import (
    by_name,
    first_layer,
    reconstruct_input,
    reconstruction_error,
)
from obscuradp.client.trainer import flatten_gradients
from obscuradp.mechanisms import clipped_sum, per_record_gradients, private_gradient
from obscuradp.models import TabularMLP
from obscuradp.state import get_state


def _loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy(output, target, reduction="none")


def _records(count: int, width: int = 9, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    features = torch.rand(count, width, generator=generator) * 2.0 - 1.0
    labels = (torch.rand(count, generator=generator) > 0.5).float()
    return features, labels


def _gradient(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
    """The plain sum of the records' gradients (the clipping norm is out of reach)."""
    return clipped_sum(per_record_gradients(model, _loss, x, y), 1e9)


@pytest.mark.parametrize("hidden", [(), (16,), (32, 8)])
def test_a_record_alone_in_an_update_is_read_back_exactly(hidden: tuple[int, ...]) -> None:
    torch.manual_seed(1)
    model = TabularMLP(9, hidden)
    features, labels = _records(20)
    for index in range(20):
        x, y = features[index : index + 1], labels[index : index + 1]
        found = reconstruct_input(_gradient(model, x, y), model)
        assert reconstruction_error(x[0].numpy(), found) < 1e-4
    # Any multiple of the gradient gives the same record: a clipped one, or an averaged one.
    x, y = features[:1], labels[:1]
    shortened = clipped_sum(per_record_gradients(model, _loss, x, y), 0.01)
    assert reconstruction_error(x[0].numpy(), reconstruct_input(shortened, model)) < 1e-4
    # So does the change of the model after one plain step on that record.
    before = get_state(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    _loss(model(x), y).sum().backward()  # type: ignore[no-untyped-call]
    optimizer.step()
    change = {name: value - before[name] for name, value in get_state(model).items()}
    assert reconstruction_error(x[0].numpy(), reconstruct_input(change, model)) < 1e-3


def test_a_flat_vector_can_be_cut_back_into_named_pieces() -> None:
    torch.manual_seed(2)
    model = TabularMLP(9, (16,))
    features, labels = _records(1)
    gradient = _gradient(model, features, labels)
    pieces = by_name(flatten_gradients(gradient).numpy(), model)
    assert set(pieces) == set(gradient)
    found = reconstruct_input(pieces, model)
    assert reconstruction_error(features[0].numpy(), found) < 1e-4
    with pytest.raises(ValueError, match="one number per trainable parameter"):
        by_name(np.zeros(5), model)


def test_other_records_in_the_same_update_blur_it() -> None:
    torch.manual_seed(3)
    model = TabularMLP(9, (16,))
    features, labels = _records(200)
    average = features.double().mean(dim=0).numpy()
    mixed, guessed = [], []
    for index in range(40):
        rows = [index, *range(100 + index, 131 + index)]
        found = reconstruct_input(_gradient(model, features[rows], labels[rows]), model)
        mixed.append(reconstruction_error(features[index].numpy(), found))
        guessed.append(reconstruction_error(features[index].numpy(), average))
    # With 31 other records the exact method is no better than answering the average record.
    assert np.median(mixed) > 0.5
    assert np.median(mixed) > 0.7 * np.median(guessed)


def test_the_noise_of_the_private_step_buries_a_record_that_stands_alone() -> None:
    torch.manual_seed(4)
    model = TabularMLP(9, (16,))
    features, labels = _records(60)
    noise = torch.Generator().manual_seed(5)
    errors = []
    for index in range(60):
        x, y = features[index : index + 1], labels[index : index + 1]
        noisy = private_gradient(
            model,
            _loss,
            x,
            y,
            clip_norm=1.0,
            noise_multiplier=1.0,
            expected_batch_size=1.0,
            generator=noise,
        )
        errors.append(reconstruction_error(x[0].numpy(), reconstruct_input(noisy, model)))
    assert np.median(errors) > 0.5


def test_models_and_updates_that_cannot_be_inverted_are_refused() -> None:
    model = TabularMLP(4, (3,))
    assert first_layer(model) == ("net.0.weight", "net.0.bias")
    with pytest.raises(ValueError, match="no bias"):
        first_layer(nn.Sequential(nn.Linear(4, 2, bias=False)))
    with pytest.raises(ValueError, match="no linear layer"):
        first_layer(nn.Sequential(nn.ReLU()))
    zero = {"net.0.weight": np.ones((3, 4)), "net.0.bias": np.zeros(3)}
    with pytest.raises(ValueError, match="bias part of the update is zero"):
        reconstruct_input(zero, model)
    crooked = {"net.0.weight": np.ones((2, 4)), "net.0.bias": np.ones(3)}
    with pytest.raises(ValueError, match="do not fit together"):
        reconstruct_input(crooked, model)


def test_the_error_is_a_share_of_the_records_own_length() -> None:
    record = np.array([3.0, 4.0])
    assert reconstruction_error(record, record) == 0.0
    assert reconstruction_error(record, np.zeros(2)) == pytest.approx(1.0)
    assert reconstruction_error(record, [3.0, 3.0]) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="same length"):
        reconstruction_error(record, [1.0])
    with pytest.raises(ValueError, match="all zeros"):
        reconstruction_error(np.zeros(2), record)
