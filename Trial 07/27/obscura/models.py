"""Parameterised feed-forward networks for tabular prediction.

This module unifies Experiment 1's FFNN (15->64->32->1 on stroke data) and
Experiment 2's MLP (11->32->16->1 on loan data) into one parameterised family,
plus Experiment 3's flatten/unflatten helpers so the parameter vector can be
encrypted and aggregated homomorphically.

Architecture: input -> hidden1 (ReLU) -> hidden2 (ReLU) -> 1 (Sigmoid), trained
with BCELoss. No BatchNorm (Opacus-incompatible; see Exp 1-2); plain Linear+ReLU
is fully DP-SGD friendly.

CRITICAL ALIGNMENT INVARIANT: For these plain nn.Sequential(Linear, ReLU, ...)
models, state_dict() order == named_parameters() order. The SPE mask is built
over param_index_map(state) and applied to flatten_state(state). This seam is
where Experiment 3's sensitivity code meets Experiment 1/2's state-dict code:
a mask derived from per-named-parameter gradient scores indexes the flattened
state correctly because both iterate in the same order.
"""
import numpy as np
import torch
import torch.nn as nn

from . import config


class FFNN(nn.Module):
    """Feed-forward network with two hidden layers and sigmoid output.

    Args:
        in_features: Input dimension.
        hidden1: First hidden layer dimension.
        hidden2: Second hidden layer dimension.
    """
    def __init__(self, in_features, hidden1, hidden2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def new_model(in_features, dataset="stroke", width_scale=1):
    """Create a model with dataset-specific hidden dimensions, scaled by width_scale.

    The hidden dimensions are read from config.HIDDEN[dataset], then scaled:
    hidden = tuple(h * width_scale for h in config.HIDDEN[dataset]).
    This drives the overhead-vs-model-size sweep: WIDTH_SCALES spans 3.3k to ~1M
    parameters on the SAME architecture family, exposing O(n) crypto scaling.

    Args:
        in_features: Input dimension.
        dataset: "stroke" or "loan".
        width_scale: Multiplier for hidden dimensions.

    Returns:
        FFNN instance.
    """
    hidden = tuple(h * width_scale for h in config.HIDDEN[dataset])
    return FFNN(in_features, hidden[0], hidden[1])


def count_params(model):
    """Return total parameter count."""
    return sum(p.numel() for p in model.parameters())


def get_weights(model):
    """Return a CPU state_dict of plain tensors, handling Opacus '_module.' prefix.

    Strips '_module.' (added by Opacus DP-SGD wrapping) and returns detached CPU
    clones so the dict can be safely encrypted / aggregated without triggering
    autograd.
    """
    sd = model.state_dict()
    return {k.replace("_module.", ""): v.detach().cpu().clone() for k, v in sd.items()}


def set_weights(model, state):
    """Load a state dict into the model."""
    model.load_state_dict(state)


def state_template(in_features, dataset="stroke", width_scale=1):
    """Return a state dict template by constructing a fresh model.

    This is used by unflatten_state() to recover shapes and keys; by calling
    new_model() here, we ensure the template's keys match those in get_weights().
    """
    model = new_model(in_features, dataset=dataset, width_scale=width_scale)
    return get_weights(model)


def param_index_map(state):
    """Return [(name, shape_tuple, numel)] in the exact iteration order of state dict.

    This is the ground-truth ordering: param_index_map(state) drives both
    flatten_state() and SPE mask construction (sensitivity.py). For these plain
    Sequential models, state_dict() order == named_parameters() order, so a
    mask built over these indices indexes flatten_state(state) correctly.

    Args:
        state: A state dict (output of get_weights()).

    Returns:
        List of (name, shape_tuple, numel) in dict iteration order.
    """
    result = []
    for k, v in state.items():
        result.append((k, tuple(v.shape), v.numel()))
    return result


def flatten_state(state):
    """Concatenate state dict into a single 1-D float64 numpy vector.

    Order matches param_index_map(state), so the result can be indexed by
    masks built from param_index_map() and gradient sensitivities.

    Args:
        state: A state dict.

    Returns:
        1-D float64 numpy array.
    """
    parts = []
    for k in state:
        v = state[k]
        parts.append(v.numpy().ravel().astype(np.float64))
    return np.concatenate(parts).astype(np.float64)


def unflatten_state(flat, template):
    """Rebuild a state dict from a flat vector using template for keys and shapes.

    Args:
        flat: 1-D array (float64 or similar).
        template: A state dict; keys and shapes are used, values are ignored.

    Returns:
        Dict of str -> torch.Tensor (float32), same keys and shapes as template.
    """
    out, i = {}, 0
    for k, v in template.items():
        n = v.numel()
        chunk = np.asarray(flat[i:i + n], dtype=np.float32).reshape(v.shape)
        out[k] = torch.from_numpy(chunk)
        i += n
    return out
