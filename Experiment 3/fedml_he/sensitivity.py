"""Selective Parameter Encryption: sensitivity scoring + mask construction.

The paper's idea: parameters differ in how much they leak under gradient-inversion
attacks, so encrypt only the most sensitive top-p fraction. Sensitivity is scored
once before training from the gradient magnitude on a small local batch:

    sensitivity[name] = param.grad.abs().mean()

Every scalar in a tensor inherits its tensor's score (matching the paper's
per-named-parameter computation); we then rank all scalars by score and flag the
top-p fraction in a boolean mask over the flattened parameter vector, so the mask
encrypts *exactly* a p fraction of the parameters.
"""
import numpy as np
import torch
import torch.nn as nn

from . import config
from .model_utils import flatten_params


def sensitivity_scores(model, batch, criterion=None):
    """Per-named-parameter gradient-magnitude scores from one backward pass."""
    model.zero_grad(set_to_none=True)
    x, y = batch
    out = model(x)
    if criterion is None:
        # default: cross-entropy on the final logits (handles seq models too)
        if out.dim() == 3:                       # (B, T, V) language-model logits
            out = out.reshape(-1, out.size(-1)); y = y.reshape(-1)
        criterion = nn.CrossEntropyLoss()
    loss = criterion(out, y)
    loss.backward()
    scores = {}
    for name, p in model.named_parameters():
        scores[name] = (p.grad.abs().mean().item() if p.grad is not None else 0.0)
    return scores


def build_mask(model, scores, fraction):
    """Boolean mask over the flattened params flagging the top-`fraction` scalars.

    Each scalar gets its tensor's sensitivity score; the global top-`fraction` by
    score are set True. Returns a bool array aligned with flatten_params(model).
    """
    _, template = flatten_params(model)
    n_total = sum(numel for _, _, numel in template)
    per_scalar = np.empty(n_total, dtype=np.float64)
    i = 0
    for name, _, numel in template:
        per_scalar[i:i + numel] = scores.get(name, 0.0)
        i += numel

    mask = np.zeros(n_total, dtype=bool)
    k = int(round(fraction * n_total))
    if k >= n_total:
        mask[:] = True
    elif k > 0:
        # indices of the k highest-sensitivity scalars (unsorted top-k via argpartition)
        top = np.argpartition(per_scalar, n_total - k)[n_total - k:]
        mask[top] = True
    return mask


def random_batch(name, n=64):
    """A throwaway input batch for sensitivity scoring (shape matches the model).

    The overhead a mask induces depends only on how many params it selects, not on
    which, so a random batch is sufficient for Experiments A/B; real CIFAR-10 is
    used only for the accuracy run (Experiment C)."""
    g = torch.Generator().manual_seed(config.SEED)
    if name == "LSTM":
        x = torch.randint(0, 10000, (n, 32), generator=g)
        y = torch.randint(0, 10000, (n, 32), generator=g)
        return x, y
    img = 224 if name == "ViT-B/16" else 32
    x = torch.randn(n, 3, img, img, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return x, y
