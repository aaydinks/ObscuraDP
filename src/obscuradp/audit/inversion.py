"""Can whoever sees an update read a record back out of it?

For the models of this library there is an exact way when a single record stands alone in
an update. The first layer computes $W x + b$. For one record, the gradient of the loss
with respect to row $j$ of $W$ is the gradient with respect to $b_j$ times the record
itself: $\\partial L / \\partial W_j = (\\partial L / \\partial b_j) \\, x$. So dividing one
by the other gives $x$, exactly, whatever the rest of the network looks like (Phong, Aono,
Hayashi, Wang and Moriai, 2018; Geiping, Bauermeister, Droge and Moeller, 2020). With
logistic regression the first layer is the whole model.

``reconstruct_input`` does that division, using all rows of the first layer at once (a
least-squares fit, which is the exact answer when there is one, and the best single
answer when noise or other records are mixed in). ``reconstruction_error`` says how far
the result is from the truth. What an attacker may feed into it is decided elsewhere:
only what its plan lets it see (``obscuradp.audit.views``).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

FloatVector = npt.NDArray[np.float64]


def first_layer(model: nn.Module) -> tuple[str, str]:
    """The names of the weights and the bias of the model's first linear layer."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module.bias is None:
                raise ValueError("the first linear layer has no bias, so the ratio is not defined")
            prefix = f"{name}." if name else ""
            return f"{prefix}weight", f"{prefix}bias"
    raise ValueError("the model has no linear layer")


def reconstruct_input(
    update: Mapping[str, torch.Tensor | npt.ArrayLike], model: nn.Module
) -> FloatVector:
    """The record that best explains the first layer's part of an update.

    Args:
        update: A gradient, a sum of gradients or a change of the model, by parameter name
            (any multiple of a single record's gradient gives the same answer).
        model: The model the update belongs to (only its layout is used).

    Returns:
        The reconstructed record. For an update that comes from one record alone, without
        noise, it is that record.

    Raises:
        ValueError: If the bias part of the update is all zero (nothing can be divided).
    """
    weight_name, bias_name = first_layer(model)
    weight = np.asarray(_numbers(update[weight_name]), dtype=np.float64)
    bias = np.asarray(_numbers(update[bias_name]), dtype=np.float64).reshape(-1)
    if weight.ndim != 2 or weight.shape[0] != bias.shape[0]:
        raise ValueError("the first layer's weights and bias do not fit together")
    energy = float(bias @ bias)
    if energy <= 0.0:
        raise ValueError("the bias part of the update is zero")
    # Least squares over the rows: minimize the sum of |W_j - b_j x|^2 over x.
    result: FloatVector = (bias @ weight) / energy
    return result


def _numbers(value: torch.Tensor | npt.ArrayLike) -> npt.ArrayLike:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value


def by_name(flat: npt.ArrayLike, model: nn.Module) -> dict[str, FloatVector]:
    """Cut a flat vector over the trainable parameters into one piece per parameter."""
    vector = np.asarray(flat, dtype=np.float64).reshape(-1)
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    if sum(int(value.numel()) for _, value in trainable) != vector.size:
        raise ValueError("the vector does not have one number per trainable parameter")
    pieces = {}
    offset = 0
    for name, value in trainable:
        size = int(value.numel())
        pieces[name] = vector[offset : offset + size].reshape(tuple(value.shape))
        offset += size
    return pieces


def reconstruction_error(truth: npt.ArrayLike, guess: npt.ArrayLike) -> float:
    """How far a reconstruction is from the record, as a share of the record's own length.

    0 is a perfect copy. 1 is as far off as answering with all zeros.
    """
    real = np.asarray(truth, dtype=np.float64).reshape(-1)
    said = np.asarray(guess, dtype=np.float64).reshape(-1)
    if real.shape != said.shape:
        raise ValueError("record and reconstruction must have the same length")
    length = float(np.linalg.norm(real))
    if length == 0.0:
        raise ValueError("the record is all zeros")
    return float(np.linalg.norm(said - real)) / length
