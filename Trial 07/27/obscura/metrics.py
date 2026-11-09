"""Evaluation metrics for binary classification on test sets.

This trial composes Experiments 1 and 2: Experiment 1 reported recall on a
class-balanced stroke test set (reflecting the clinical focus on detecting
stroke cases); Experiment 2 reported accuracy on a natural loan test set
(where majority-class bias matters). This trial runs both tasks, so it
computes the full metric set for both and lets primary_metric(dataset)
pick the headline number: recall for stroke, accuracy for loan. This lets
each arm's utility be measured on its dataset-specific metric while the
full evaluation dict enables offline re-aggregation and cross-dataset analysis.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score,
)

from . import models


def evaluate(state, X_test, y_test, in_features, dataset="stroke", width_scale=1):
    """Forward a test set through the model and return all binary metrics.

    Builds the model, loads the state, and computes accuracy, recall, precision,
    f1, auc, ap, loss, and pred_pos_rate at a 0.5 threshold. AUC and AP are
    computed from probabilities (sklearn); if a class is missing, they are NaN.
    Loss is BCELoss computed on the probabilities.
    """
    model = models.new_model(in_features, dataset=dataset, width_scale=width_scale)
    models.set_weights(model, state)
    model.eval()

    with torch.no_grad():
        X_t = torch.from_numpy(X_test)
        probs = model(X_t).numpy()

    preds = (probs >= 0.5).astype(int)
    y_t = y_test.astype(np.float32)

    try:
        auc = float(roc_auc_score(y_test, probs))
        ap = float(average_precision_score(y_test, probs))
    except ValueError:
        auc = float('nan')
        ap = float('nan')

    loss = float(nn.BCELoss()(torch.from_numpy(probs), torch.from_numpy(y_t)))

    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "auc": auc,
        "ap": ap,
        "loss": loss,
        "pred_pos_rate": float(preds.mean()),
    }


def primary_metric(dataset):
    """Return the headline metric name for a dataset.

    Stroke uses recall (clinical focus on detecting positives); loan uses
    accuracy (overall correctness on natural imbalance).
    """
    if dataset == "stroke":
        return "recall"
    elif dataset == "loan":
        return "accuracy"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def summarize(list_of_metric_dicts):
    """Aggregate a list of metric dicts into mean and std per key.

    Uses np.nanmean and np.nanstd to ignore NaN values. Skips keys that
    are not numeric in at least one dict.
    """
    if not list_of_metric_dicts:
        return {}

    all_keys = set()
    for d in list_of_metric_dicts:
        all_keys.update(d.keys())

    result = {}
    for key in all_keys:
        values = []
        for d in list_of_metric_dicts:
            if key in d:
                v = d[key]
                if isinstance(v, (int, float, np.number)):
                    values.append(float(v))

        if values:
            result[key] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
            }

    return result
