"""Federated averaging in the clear.

This is the reference every secure way of summing must agree with: the weighted mean of the
participants' flat state vectors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import torch

from obscuradp.state import FloatVector, State, flatten_state, unflatten_state


def fedavg_weights(counts: Sequence[float], *, weighted: bool = True) -> FloatVector:
    """Return one weight per participant, summing to 1.

    With ``weighted=True`` a participant's weight is its share of all rows. With
    ``weighted=False`` every participant counts the same.
    """
    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("counts must be a non-empty list of numbers")
    if weighted:
        if np.any(values < 0) or np.sum(values) <= 0:
            raise ValueError("counts must be non-negative and not all zero")
        return np.asarray(values / np.sum(values), dtype=np.float64)
    return np.ones(len(values), dtype=np.float64) / len(values)


def weighted_mean(
    vectors: Sequence[npt.ArrayLike], weights: npt.ArrayLike | None = None
) -> FloatVector:
    """Mean of equally long vectors, optionally weighted. Weights are rescaled to sum to 1."""
    if len(vectors) == 0:
        raise ValueError("need at least one vector")
    stacked = np.stack([np.asarray(v, dtype=np.float64) for v in vectors], axis=0)
    if weights is None:
        return np.asarray(np.mean(stacked, axis=0), dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (stacked.shape[0],):
        raise ValueError("need exactly one weight per vector")
    w = w / np.sum(w)
    return np.asarray(np.average(stacked, axis=0, weights=w), dtype=np.float64)


def aggregate_states(
    states: Sequence[Mapping[str, torch.Tensor]],
    counts: Sequence[float],
    *,
    weighted: bool = True,
) -> State:
    """Average participants' states in the clear and return the result as a state.

    The average is computed in float64 and stored back in the precision of the first state
    (float32 for ordinary models).
    """
    if len(states) == 0:
        raise ValueError("need at least one state")
    if len(states) != len(counts):
        raise ValueError("need exactly one count per state")
    flat = [flatten_state(state) for state in states]
    mean = weighted_mean(flat, fedavg_weights(counts, weighted=weighted))
    return unflatten_state(mean, states[0])
