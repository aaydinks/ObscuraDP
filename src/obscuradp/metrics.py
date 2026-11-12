"""Measuring how good a yes/no model is, with numpy only.

All functions take the model's predicted probabilities and the true 0/1 labels. Nothing here
looks at training data or spends privacy budget: these are for evaluation on held-out rows.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

# torch.nn.BCELoss clamps each log term at -100 so a probability of exactly 0 or 1 stays finite.
_LOG_CLAMP = -100.0


def multiclass_metrics(scores: npt.ArrayLike, labels: npt.ArrayLike) -> dict[str, float]:
    """Return the measures for a prediction among several classes.

    Args:
        scores: One row per record and one raw score per class (before any softmax).
        labels: The true class of each record, as whole numbers from 0.

    Keys: ``accuracy`` (share of records whose highest score is the true class) and
    ``loss`` (mean cross-entropy).
    """
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels).ravel()
    if values.ndim != 2 or values.shape[0] != truth.shape[0]:
        raise ValueError("need one row of scores per label")
    if values.shape[0] == 0:
        raise ValueError("need at least one row")
    classes = truth.astype(np.int64)
    if not np.all(classes == truth) or np.min(classes) < 0 or np.max(classes) >= values.shape[1]:
        raise ValueError("labels must be whole numbers from 0 to the number of classes minus 1")
    shifted = values - np.max(values, axis=1, keepdims=True)
    log_share = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    picked = log_share[np.arange(classes.size), classes]
    return {
        "accuracy": float(np.mean(np.argmax(values, axis=1) == classes)),
        "loss": float(-np.mean(np.maximum(picked, _LOG_CLAMP))),
    }


def binary_metrics(
    probabilities: npt.ArrayLike, labels: npt.ArrayLike, *, threshold: float = 0.5
) -> dict[str, float]:
    """Return the standard measures for a yes/no prediction task.

    Keys: ``accuracy``, ``recall`` (share of true positives that were found), ``precision``
    (share of predicted positives that were right), ``f1``, ``auc`` (area under the ROC
    curve), ``ap`` (average precision), ``loss`` (mean binary cross-entropy) and
    ``pred_pos_rate`` (share of rows predicted positive).

    ``auc`` and ``ap`` do not depend on ``threshold``. They are NaN when the labels contain
    only one class. A ratio whose denominator is zero is reported as 0.
    """
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel()
    if p.shape != y.shape:
        raise ValueError("probabilities and labels must have the same length")
    if p.size == 0:
        raise ValueError("need at least one row")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be 0 or 1")
    truth = y.astype(bool)
    predicted = p >= threshold

    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))

    with np.errstate(divide="ignore"):
        log_p = np.maximum(np.log(p), _LOG_CLAMP)
        log_not_p = np.maximum(np.log1p(-p), _LOG_CLAMP)
    loss = float(-np.mean(np.where(truth, log_p, log_not_p)))

    return {
        "accuracy": float(np.mean(predicted == truth)),
        "recall": _ratio(tp, tp + fn),
        "precision": _ratio(tp, tp + fp),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "auc": roc_auc(p, truth),
        "ap": average_precision(p, truth),
        "loss": loss,
        "pred_pos_rate": float(np.mean(predicted)),
    }


def roc_auc(scores: npt.ArrayLike, labels: npt.ArrayLike) -> float:
    """Chance that a random positive row gets a higher score than a random negative row.

    Ties count as half. Returns NaN if only one class is present.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    truth = np.asarray(labels).ravel().astype(bool)
    n_pos = int(truth.sum())
    n_neg = int(truth.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return math.nan
    # Average rank of each score (1-based), shared among equal scores.
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    last_rank = np.cumsum(counts)
    average_rank = last_rank - (counts - 1) / 2.0
    ranks = average_rank[inverse]
    rank_sum = float(np.sum(ranks[truth]))
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(scores: npt.ArrayLike, labels: npt.ArrayLike) -> float:
    """Average precision: precision at each score cut-off, weighted by the gain in recall.

    Returns NaN if only one class is present.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    truth = np.asarray(labels).ravel().astype(bool)
    n_pos = int(truth.sum())
    if n_pos == 0 or n_pos == truth.size:
        return math.nan
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    hits = truth[order].astype(np.float64)
    # Evaluate once per distinct score: the last position of each run of equal scores.
    cut = np.r_[np.where(np.diff(s_sorted) != 0)[0], s_sorted.size - 1]
    tp = np.cumsum(hits)[cut]
    precision = tp / (cut + 1.0)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def evaluate(
    model: nn.Module, features: npt.ArrayLike, labels: npt.ArrayLike, *, threshold: float = 0.5
) -> dict[str, float]:
    """Run ``model`` on held-out rows and return ``binary_metrics`` of its output.

    The model must return one probability per row.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(features, dtype=np.float32))
            probabilities = model(x).detach().cpu().numpy()
    finally:
        model.train(was_training)
    return binary_metrics(probabilities, labels, threshold=threshold)


def best_threshold(
    probabilities: npt.ArrayLike, labels: npt.ArrayLike, *, metric: str = "f1"
) -> float:
    """The decision threshold that maximizes ``metric`` on the given rows.

    Use validation rows, never test rows. Every distinct predicted value is tried as a
    threshold; ties go to the lowest threshold. ``metric`` is any threshold-dependent key of
    ``binary_metrics``: ``f1``, ``accuracy``, ``recall`` or ``precision``.
    """
    if metric not in ("f1", "accuracy", "recall", "precision"):
        raise ValueError("metric must be one of f1, accuracy, recall, precision")
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    best_value, best_cut = -1.0, 0.5
    for cut in np.unique(p):
        value = binary_metrics(p, labels, threshold=float(cut))[metric]
        if value > best_value:
            best_value, best_cut = value, float(cut)
    return best_cut


def summarize(runs: Iterable[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    """Mean and spread of each numeric measure across runs.

    ``std`` is the sample standard deviation (it divides by the number of runs minus one),
    which is the honest choice for a handful of runs. It is NaN for a single run. NaN values
    of a measure are left out of its mean and spread; ``n`` says how many values were used.
    """
    collected: dict[str, list[float]] = {}
    for run in runs:
        for key, value in run.items():
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                continue
            collected.setdefault(key, []).append(float(value))
    summary: dict[str, dict[str, float]] = {}
    for key, values in collected.items():
        finite = np.asarray([v for v in values if not math.isnan(v)], dtype=np.float64)
        if finite.size == 0:
            summary[key] = {"mean": math.nan, "std": math.nan, "n": 0.0}
        else:
            std = float(np.std(finite, ddof=1)) if finite.size > 1 else math.nan
            summary[key] = {"mean": float(np.mean(finite)), "std": std, "n": float(finite.size)}
    return summary


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0
