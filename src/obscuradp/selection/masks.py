"""Masks over a model's flat vector, and ways to compare two masks.

A mask is a boolean array with one entry per number in the flat vector (see
``obscuradp.state``). ``True`` marks the entries chosen for encryption. The functions here
only do the selection arithmetic. Where the scores come from, and whether computing them
reveals anything about private data, is the caller's responsibility: a score computed from
private data is itself a release of information.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import torch

from obscuradp.state import state_layout, state_size

Mask = npt.NDArray[np.bool_]


def _count(fraction: float, n_total: int) -> int:
    return round(fraction * n_total)


def build_mask(
    state: Mapping[str, torch.Tensor], scores: Mapping[str, float], fraction: float
) -> Mask:
    """Mark the top ``fraction`` of entries, where every entry carries its tensor's score.

    Tensors missing from ``scores`` count as 0. Because all entries of one tensor share a
    score, the selection is whole tensors plus part of one boundary tensor; which entries of
    the boundary tensor are taken is decided by tie-breaking, not by the scores.
    """
    layout = state_layout(state)
    n_total = state_size(state)
    per_entry = np.empty(n_total, dtype=np.float64)
    for entry in layout:
        per_entry[entry.start : entry.stop] = scores.get(entry.name, 0.0)
    mask = np.zeros(n_total, dtype=bool)
    k = _count(fraction, n_total)
    if k >= n_total:
        mask[:] = True
    elif k > 0:
        top = np.argpartition(per_entry, n_total - k)[n_total - k :]
        mask[top] = True
    return mask


def tensor_selection(
    state: Mapping[str, torch.Tensor], scores: Mapping[str, float], fraction: float
) -> tuple[str, ...]:
    """Names of the tensors a top-``fraction`` mask draws from, highest score first.

    Unlike ``build_mask`` this does not depend on tie-breaking inside the boundary tensor:
    tensors are sorted by score (ties by name) and taken until ``fraction`` of all entries is
    covered.
    """
    layout = state_layout(state)
    k = _count(fraction, state_size(state))
    if k <= 0:
        return ()
    ordered = sorted(layout, key=lambda entry: (-scores.get(entry.name, 0.0), entry.name))
    selected = []
    filled = 0
    for entry in ordered:
        selected.append(entry.name)
        filled += entry.size
        if filled >= k:
            break
    return tuple(selected)


def tensor_selection_agreement(
    state: Mapping[str, torch.Tensor],
    scores_a: Mapping[str, float],
    scores_b: Mapping[str, float],
    fraction: float,
) -> dict[str, float]:
    """Do two scorings select the same tensors? Free of tie-breaking effects.

    Keys: ``same_selection`` (1.0 or 0.0), ``tensor_jaccard`` (shared tensors over all
    selected tensors), ``n_a``, ``n_b``, ``n_both``.
    """
    sel_a = set(tensor_selection(state, scores_a, fraction))
    sel_b = set(tensor_selection(state, scores_b, fraction))
    n_both = len(sel_a & sel_b)
    n_union = len(sel_a | sel_b)
    return {
        "same_selection": 1.0 if sel_a == sel_b else 0.0,
        "tensor_jaccard": float(n_both) / n_union if n_union > 0 else 1.0,
        "n_a": len(sel_a),
        "n_b": len(sel_b),
        "n_both": n_both,
    }


def random_jaccard_expected(fraction: float) -> float:
    """Expected overlap score between a top-k mask and a random mask of the same size.

    Both hold a share ``f`` of the entries, so the expected value is ``f / (2 - f)``.
    """
    f = float(fraction)
    return f / (2.0 - f) if f < 2.0 else 1.0


def random_mask(n_total: int, fraction: float, seed: int) -> Mask:
    """Mark a uniformly random ``fraction`` of ``n_total`` entries (a control)."""
    k = _count(fraction, n_total)
    # RandomState on purpose: same draws as the research code
    rng = np.random.RandomState(seed)
    mask = np.zeros(n_total, dtype=bool)
    if k >= n_total:
        mask[:] = True
    elif k > 0:
        mask[rng.choice(n_total, size=k, replace=False)] = True
    return mask


def mask_agreement(mask_a: npt.ArrayLike, mask_b: npt.ArrayLike) -> dict[str, float]:
    """Overlap of two masks.

    Keys: ``jaccard`` (shared entries over all marked entries; 1.0 if both are empty),
    ``overlap`` (shared entries over the entries of the first mask; 0.0 if it is empty),
    ``n_a``, ``n_b``, ``n_both``.
    """
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("masks must have the same shape")
    n_a = a.sum()
    n_b = b.sum()
    n_both = (a & b).sum()
    n_union = (a | b).sum()
    jaccard = n_both / n_union if n_union > 0 else 1.0
    overlap = n_both / n_a if n_a > 0 else 0.0
    return {
        "jaccard": float(jaccard),
        "overlap": float(overlap),
        "n_a": int(n_a),
        "n_b": int(n_b),
        "n_both": int(n_both),
    }


def score_rank_correlation(scores_a: Mapping[str, float], scores_b: Mapping[str, float]) -> float:
    """Spearman rank correlation of two scorings over the names they share.

    NaN if they share fewer than three names or if either side is constant.
    """
    shared = sorted(set(scores_a) & set(scores_b))
    if len(shared) < 3:
        return math.nan
    values_a = np.array([scores_a[name] for name in shared], dtype=np.float64)
    values_b = np.array([scores_b[name] for name in shared], dtype=np.float64)
    if np.allclose(values_a, values_a[0]) or np.allclose(values_b, values_b[0]):
        return math.nan
    ranks_a = _average_ranks(values_a)
    ranks_b = _average_ranks(values_b)
    centered_a = ranks_a - ranks_a.mean()
    centered_b = ranks_b - ranks_b.mean()
    denominator = math.sqrt(float(np.sum(centered_a**2)) * float(np.sum(centered_b**2)))
    if denominator == 0.0:
        return math.nan
    return float(np.sum(centered_a * centered_b)) / denominator


def _average_ranks(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Rank from 1, giving equal values the average of the ranks they span."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    last = np.cumsum(counts)
    average = last - (counts - 1) / 2.0
    ranks: npt.NDArray[np.float64] = average[inverse].astype(np.float64)
    return ranks
