"""Mask scoring exactly as in Trial 07/27 (comparison only, see the package docstring).

Known defects reproduced on purpose: the default "clean" scores are plain gradients of one
participant's raw rows with no noise (finding G4), and the "dp" scores divide the noise by
the scoring batch (256 rows) and not by the training batch, so they are about 8 times less
noisy than training (finding G14). Neither release is charged to any privacy budget.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from obscuradp.legacy.config import LegacyTrialConfig
from obscuradp.selection.masks import build_mask, random_mask
from obscuradp.state import state_size

Batch = tuple[torch.Tensor, torch.Tensor]


def sensitivity_scores(model: nn.Module, batch: Batch, criterion: Any = None) -> dict[str, float]:
    """One score per named parameter: mean absolute gradient from one backward pass."""
    model.zero_grad(set_to_none=True)
    x, y = batch
    out = model(x)
    if criterion is None:
        criterion = nn.BCELoss()
    loss = criterion(out, y)
    loss.backward()
    scores = {}
    for name, p in model.named_parameters():
        scores[name] = p.grad.abs().mean().item() if p.grad is not None else 0.0
    return scores


def dp_sensitivity_scores(
    cfg: LegacyTrialConfig,
    model: nn.Module,
    batch: Batch,
    sigma: float,
    max_grad_norm: float | None = None,
    criterion: Any = None,
    seed: int | None = None,
) -> dict[str, float]:
    """Scores from per-row clipped gradients, averaged, with Gaussian noise added.

    The noise standard deviation is ``sigma * max_grad_norm / B`` where ``B`` is the number
    of rows in ``batch`` (see the module docstring for why that is a defect).
    """
    if max_grad_norm is None:
        max_grad_norm = cfg.max_grad_norm
    if criterion is None:
        criterion = nn.BCELoss()

    x, y = batch
    B = x.shape[0]
    device = x.device

    # Collect per-sample gradients as a list of flat tensors.
    per_sample_grads = []

    for i in range(B):
        model.zero_grad(set_to_none=True)
        x_i = x[i : i + 1]
        y_i = y[i : i + 1]
        out_i = model(x_i)
        loss_i = criterion(out_i, y_i)
        loss_i.backward()

        # Collect and flatten gradients.
        grad_flat = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        grad_norm = torch.norm(grad_flat, p=2)

        # Clip to max_grad_norm via standard DP-SGD clipping factor.
        clip_coeff = torch.clamp(
            torch.tensor(max_grad_norm, device=device, dtype=grad_flat.dtype) / (grad_norm + 1e-10),
            max=1.0,
        )
        per_sample_grads.append(grad_flat * clip_coeff)

    # Average the clipped per-sample gradients.
    avg_grad_flat = torch.stack(per_sample_grads).mean(dim=0)

    # Add Gaussian noise with std sigma * max_grad_norm / B.
    if seed is not None:
        g: torch.Generator | None = torch.Generator(device=device)
        assert g is not None
        g.manual_seed(seed)
    else:
        g = None
    noise_std = sigma * max_grad_norm / B
    noise = torch.randn_like(avg_grad_flat, generator=g) * noise_std
    noised_grad_flat = avg_grad_flat + noise

    # Map back to named parameters and compute abs().mean() for each.
    scores = {}
    offset = 0
    for name, p in model.named_parameters():
        numel = p.numel()
        grad_param = noised_grad_flat[offset : offset + numel].reshape(p.shape)
        scores[name] = grad_param.abs().mean().item()
        offset += numel

    return scores


def make_mask(
    cfg: LegacyTrialConfig,
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    batch: Batch,
    fraction: float,
    mode: str,
    sigma: float | None = None,
    seed: int | None = None,
) -> tuple[npt.NDArray[np.bool_], dict[str, float]]:
    """Build a mask in mode ``"clean"``, ``"dp"`` or ``"random"``. Returns (mask, scores)."""
    n_total = state_size(state)
    if mode == "clean":
        scores = sensitivity_scores(model, batch)
        mask = build_mask(state, scores, fraction)
    elif mode == "dp":
        if sigma is None:
            raise ValueError("mode='dp' requires sigma to be specified")
        scores = dp_sensitivity_scores(cfg, model, batch, sigma, seed=seed)
        mask = build_mask(state, scores, fraction)
    elif mode == "random":
        if seed is None:
            raise ValueError("mode='random' requires a seed")
        scores = {}
        mask = random_mask(n_total, fraction, seed)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return mask, scores


def tabular_batch(
    X: np.ndarray[Any, Any], y: np.ndarray[Any, Any], n: int = 256, seed: int | None = None
) -> Batch:
    """Up to ``n`` rows as float32 tensors: the first rows if ``seed`` is None, else a sample."""
    n_samples = X.shape[0]
    n_take = min(n, n_samples)
    if seed is None:
        indices = np.arange(n_take)
    else:
        # RandomState on purpose: same draws as the research code
        rng = np.random.RandomState(seed)
        indices = rng.choice(n_samples, size=n_take, replace=False)
    x_out = torch.from_numpy(X[indices].astype(np.float32))
    y_out = torch.from_numpy(y[indices].astype(np.float32))
    return x_out, y_out
