"""Flatten / unflatten a model's parameters to and from a single 1-D vector.

The encryption and aggregation code operates on one contiguous parameter vector;
these helpers convert between that vector and the live nn.Module parameters, in a
fixed (named_parameters) order so every client/round lines up.
"""
import numpy as np
import torch


def flatten_params(model):
    """Return (flat float64 vector, template) where template = [(name, shape, numel)]."""
    parts, template = [], []
    for name, p in model.named_parameters():
        d = p.detach().cpu().numpy().ravel().astype(np.float64)
        parts.append(d)
        template.append((name, tuple(p.shape), p.numel()))
    return np.concatenate(parts), template


def get_flat_params(model):
    return flatten_params(model)[0]


def set_flat_params(model, flat):
    """Write a flat vector back into the model's parameters (in named order)."""
    i = 0
    with torch.no_grad():
        for _, p in model.named_parameters():
            n = p.numel()
            chunk = np.asarray(flat[i:i + n], dtype=np.float32).reshape(p.shape)
            p.copy_(torch.from_numpy(chunk).to(p.device, p.dtype))
            i += n
