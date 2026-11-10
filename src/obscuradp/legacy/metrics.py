"""Evaluation exactly as in Trial 07/27 (comparison only, see the package docstring).

This uses scikit-learn, as the research code did, so that results match it bit for bit.
The library's own measures are in ``obscuradp.metrics`` and need numpy only.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn

from obscuradp.legacy.config import LegacyTrialConfig
from obscuradp.legacy.models import new_model
from obscuradp.state import set_state


def evaluate(
    cfg: LegacyTrialConfig,
    state: Mapping[str, torch.Tensor],
    X_test: np.ndarray[Any, Any],
    y_test: np.ndarray[Any, Any],
    in_features: int,
    dataset: str = "stroke",
    width_scale: int = 1,
) -> dict[str, float]:
    """Run the test rows through the model and return all yes/no measures at a 0.5 cut-off."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    model = new_model(cfg, in_features, dataset=dataset, width_scale=width_scale)
    set_state(model, state)
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
        auc = float("nan")
        ap = float("nan")

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


def primary_metric(dataset: str) -> str:
    """Headline measure of the research write-up: recall for stroke, accuracy for loan."""
    if dataset == "stroke":
        return "recall"
    if dataset == "loan":
        return "accuracy"
    raise ValueError(f"Unknown dataset: {dataset}")
