"""Selective Parameter Encryption: sensitivity scoring and mask construction.

This module extends Experiment 3's gradient-magnitude-based SPE heuristic to the
DP-SGD regime. Under clean-data training, parameters are ranked by |p.grad|.mean(),
and the top-p fraction are encrypted under homomorphic encryption (SPE). But
DP-SGD clips each per-sample gradient to norm C, then adds Gaussian noise — if
this destroys the ranking, Experiment 3's sensitivity-based mask becomes unreliable
in a DP+HE stack.

This module computes BOTH scorings (clean and DP-aware) and the tools to compare
them (mask Jaccard, Spearman rank correlation, tabular batch sampling). These
measurements let Trial 07/27 claim that sensitivity ranking fidelity survives the
DP budget, provided the mask is selected with DP-scored parameters instead of clean
ones.

Core idea: DP-SGD's per-sample clipping and Gaussian noise are applied in
sequence, so we simulate both via micro-batching: accumulate per-sample gradients
by looping over the batch one sample at a time, clip each ||g_i||_2 to C, average
them, then add noise with std sigma*C/B to every coordinate.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from . import config
from . import models


def sensitivity_scores(model, batch, criterion=None):
    """Per-named-parameter clean gradient-magnitude scores from one backward pass.

    One forward pass on the full batch, one backward, then for each named
    parameter return p.grad.abs().mean().item(). This is Experiment 3's recipe.
    """
    model.zero_grad(set_to_none=True)
    x, y = batch
    out = model(x)
    if criterion is None:
        criterion = nn.BCELoss()
    loss = criterion(out, y)
    loss.backward()
    scores = {}
    for name, p in model.named_parameters():
        scores[name] = (p.grad.abs().mean().item() if p.grad is not None else 0.0)
    return scores


def dp_sensitivity_scores(model, batch, sigma, max_grad_norm=None, criterion=None, seed=None):
    """Per-named-parameter sensitivity scores seen through DP-SGD's lens.

    Simulate DP-SGD gradient clipping and Gaussian noise via micro-batching:
    1. For each sample in the batch:
       - compute per-sample gradient via backward on that sample alone
       - clip its L2 norm to max_grad_norm (default config.MAX_GRAD_NORM)
    2. Average the clipped per-sample gradients
    3. Add Gaussian noise N(0, (sigma * max_grad_norm / B)^2) to each coordinate
    4. For each named parameter, return abs().mean()

    We implement per-sample clipping manually via a loop over the batch (no Opacus),
    because we need the raw clipped-and-noised mean gradient and the loop is
    transparent at batch size O(100). This avoids an extra dependency and makes
    the DP operations explicit.
    """
    if max_grad_norm is None:
        max_grad_norm = config.MAX_GRAD_NORM
    if criterion is None:
        criterion = nn.BCELoss()

    x, y = batch
    B = x.shape[0]
    device = x.device

    # Collect per-sample gradients as a list of flat tensors.
    per_sample_grads = []

    for i in range(B):
        model.zero_grad(set_to_none=True)
        x_i = x[i:i+1]
        y_i = y[i:i+1]
        out_i = model(x_i)
        # out_i is (B,) since FFNN.forward squeezes; y_i is (1,).
        loss_i = criterion(out_i, y_i)
        loss_i.backward()

        # Collect and flatten gradients.
        grad_flat = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        grad_norm = torch.norm(grad_flat, p=2)

        # Clip to max_grad_norm via standard DP-SGD clipping factor.
        clip_coeff = torch.clamp(
            torch.tensor(max_grad_norm, device=device, dtype=grad_flat.dtype) / (grad_norm + 1e-10),
            max=1.0
        )
        grad_flat_clipped = grad_flat * clip_coeff

        per_sample_grads.append(grad_flat_clipped)

    # Average the clipped per-sample gradients.
    avg_grad_flat = torch.stack(per_sample_grads).mean(dim=0)

    # Add Gaussian noise with std sigma * max_grad_norm / B.
    if seed is not None:
        g = torch.Generator(device=device)
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
        grad_param = noised_grad_flat[offset:offset+numel].reshape(p.shape)
        scores[name] = grad_param.abs().mean().item()
        offset += numel

    return scores


def build_mask(state, scores, fraction):
    """Boolean mask over flattened parameters flagging the top-fraction by score.

    Every scalar in a tensor inherits its tensor's score. The global top-k with
    k = int(round(fraction * n_total)) scalars are set True. Uses np.argpartition
    for O(n) selection without full sort.
    """
    param_map = models.param_index_map(state)
    n_total = sum(numel for _, _, numel in param_map)

    # Assign each scalar its tensor's score.
    per_scalar = np.empty(n_total, dtype=np.float64)
    i = 0
    for name, _, numel in param_map:
        per_scalar[i:i+numel] = scores.get(name, 0.0)
        i += numel

    # Compute top-k.
    mask = np.zeros(n_total, dtype=bool)
    k = int(round(fraction * n_total))

    if k >= n_total:
        mask[:] = True
    elif k > 0:
        top = np.argpartition(per_scalar, n_total - k)[n_total - k:]
        mask[top] = True

    return mask


def random_mask(n_total, fraction, seed):
    """Uniformly random top-k selection as a control.

    Select k = int(round(fraction * n_total)) random indices without replacement.
    """
    k = int(round(fraction * n_total))
    rng = np.random.RandomState(seed)

    mask = np.zeros(n_total, dtype=bool)
    if k >= n_total:
        mask[:] = True
    elif k > 0:
        indices = rng.choice(n_total, size=k, replace=False)
        mask[indices] = True

    return mask


def make_mask(model, state, batch, fraction, mode, sigma=None, seed=None):
    """Dispatch to sensitivity or random masking and return (mask, scores).

    mode in config.MASK_MODES:
      "clean"  -> sensitivity_scores
      "dp"     -> dp_sensitivity_scores (raises ValueError if sigma is None)
      "random" -> random_mask with empty scores dict
    """
    n_total = sum(numel for _, _, numel in models.param_index_map(state))

    if mode == "clean":
        scores = sensitivity_scores(model, batch)
        mask = build_mask(state, scores, fraction)
    elif mode == "dp":
        if sigma is None:
            raise ValueError("mode='dp' requires sigma to be specified")
        scores = dp_sensitivity_scores(model, batch, sigma, seed=seed)
        mask = build_mask(state, scores, fraction)
    elif mode == "random":
        scores = {}
        mask = random_mask(n_total, fraction, seed)
    else:
        raise ValueError("Unknown mode: {}".format(mode))

    return mask, scores


def mask_agreement(mask_a, mask_b):
    """Compute Jaccard, overlap, and set sizes of two boolean masks.

    Returns dict with keys:
      jaccard: |A and B| / |A or B|
      overlap: |A and B| / |A|  (0 if |A| == 0)
      n_a: count of True in mask_a
      n_b: count of True in mask_b
      n_both: count of True in both
    """
    n_a = mask_a.sum()
    n_b = mask_b.sum()
    n_both = (mask_a & mask_b).sum()
    n_union = (mask_a | mask_b).sum()

    jaccard = n_both / n_union if n_union > 0 else 1.0
    overlap = n_both / n_a if n_a > 0 else 0.0

    return {
        "jaccard": float(jaccard),
        "overlap": float(overlap),
        "n_a": int(n_a),
        "n_b": int(n_b),
        "n_both": int(n_both),
    }


def score_rank_correlation(scores_a, scores_b):
    """Spearman rank correlation over the intersection of parameter names.

    Returns float('nan') if fewer than 3 shared keys or if either side is constant.
    """
    keys_shared = set(scores_a.keys()) & set(scores_b.keys())

    if len(keys_shared) < 3:
        return float('nan')

    keys_sorted = sorted(keys_shared)
    vals_a = np.array([scores_a[k] for k in keys_sorted])
    vals_b = np.array([scores_b[k] for k in keys_sorted])

    # Check for constant scores.
    if np.allclose(vals_a, vals_a[0]) or np.allclose(vals_b, vals_b[0]):
        return float('nan')

    corr, _ = stats.spearmanr(vals_a, vals_b)
    return float(corr) if not np.isnan(corr) else float('nan')


def tabular_batch(X, y, n=256, seed=None):
    """Sample up to n rows from tabular data and return as torch float32 tensors.

    If seed is None, return the first n rows deterministically.
    Otherwise, use np.random.RandomState(seed) to sample n rows uniformly.
    """
    n_samples = X.shape[0]
    n_take = min(n, n_samples)

    if seed is None:
        # Deterministic: first n_take rows.
        indices = np.arange(n_take)
    else:
        # Random sample.
        rng = np.random.RandomState(seed)
        indices = rng.choice(n_samples, size=n_take, replace=False)

    x_out = torch.from_numpy(X[indices].astype(np.float32))
    y_out = torch.from_numpy(y[indices].astype(np.float32))

    return x_out, y_out
