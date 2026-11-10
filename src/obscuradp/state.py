"""Turning a model's state into one flat vector and back.

Secure sums, encryption and averaging all work on a single vector of numbers per participant.
These helpers define that vector: every entry of ``module.state_dict()``, in its iteration
order, flattened and concatenated. Buffers (for example the running statistics of a
normalization layer) are part of the state dict and therefore part of the vector, so they
are combined across participants like any weight.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

State = dict[str, torch.Tensor]
FloatVector = npt.NDArray[np.float64]

# Opacus wraps a model and prefixes every key with this. Stripping it keeps keys stable.
_WRAPPER_PREFIX = "_module."


class Entry(NamedTuple):
    """One tensor of a state: its key, its shape, how many numbers it holds, and where they sit."""

    name: str
    shape: tuple[int, ...]
    size: int
    start: int

    @property
    def stop(self) -> int:
        return self.start + self.size


def get_state(module: nn.Module) -> State:
    """Return a detached CPU copy of the module's state dict.

    Keys lose a leading ``_module.`` (added when a model is wrapped for private training),
    so the same model has the same keys whether or not it is wrapped.
    """
    return {
        key.replace(_WRAPPER_PREFIX, ""): value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def set_state(module: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Load ``state`` into ``module``. Keys and shapes must match exactly."""
    module.load_state_dict(dict(state))


def count_parameters(module: nn.Module) -> int:
    """Number of trainable and non-trainable parameters (buffers are not counted)."""
    return sum(p.numel() for p in module.parameters())


def state_layout(state: Mapping[str, torch.Tensor]) -> list[Entry]:
    """Describe where each tensor of ``state`` sits in the flat vector."""
    layout = []
    start = 0
    for name, value in state.items():
        size = value.numel()
        layout.append(Entry(name, tuple(value.shape), size, start))
        start += size
    return layout


def state_size(state: Mapping[str, torch.Tensor]) -> int:
    """Length of the flat vector for ``state``."""
    return sum(value.numel() for value in state.values())


def flatten_state(state: Mapping[str, torch.Tensor]) -> FloatVector:
    """Concatenate every tensor of ``state`` into one float64 vector.

    The order is the state's own iteration order, the same order ``state_layout`` reports.
    """
    parts = [value.numpy().ravel().astype(np.float64) for value in state.values()]
    if not parts:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(parts).astype(np.float64)


def unflatten_state(flat: npt.ArrayLike, template: Mapping[str, torch.Tensor]) -> State:
    """Rebuild a state from a flat vector, using ``template`` for keys, shapes and types.

    Floating-point tensors are stored back in the template's precision (float32 for ordinary
    models), which rounds the incoming float64 values. Integer tensors, such as a step
    counter, are rounded to the nearest integer.
    """
    vector = np.asarray(flat)
    expected = state_size(template)
    if vector.ndim != 1 or vector.shape[0] != expected:
        raise ValueError(
            f"flat vector has shape {vector.shape}, but the template needs {expected} numbers"
        )
    out: State = {}
    start = 0
    for name, value in template.items():
        size = value.numel()
        chunk = vector[start : start + size]
        target = value.numpy().dtype
        if np.issubdtype(target, np.floating):
            array = np.asarray(chunk, dtype=target)
        else:
            array = np.rint(np.asarray(chunk, dtype=np.float64)).astype(target)
        out[name] = torch.from_numpy(array.reshape(tuple(value.shape)).copy())
        start += size
    return out
