"""The numpy-only measures agree with scikit-learn."""

from __future__ import annotations

import math

import numpy as np
import pytest

from obscuradp.metrics import binary_metrics

pytestmark = [pytest.mark.integration, pytest.mark.filterwarnings("ignore")]

sklearn_metrics = pytest.importorskip("sklearn.metrics")


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("ties", [False, True])
def test_agrees_with_scikit_learn(seed: int, ties: bool) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(20, 400))
    y = (rng.random(n) < rng.uniform(0.1, 0.9)).astype(np.int64)
    if y.min() == y.max():
        y[0] = 1 - y[0]
    p = np.clip(rng.beta(2, 2, n) * 0.6 + y * 0.25, 1e-6, 1 - 1e-6)
    if ties:
        p = np.round(p, 1).clip(0.05, 0.95)

    m = binary_metrics(p, y)
    predicted = (p >= 0.5).astype(int)
    sk = sklearn_metrics
    assert m["accuracy"] == pytest.approx(sk.accuracy_score(y, predicted), abs=1e-12)
    assert m["recall"] == pytest.approx(sk.recall_score(y, predicted, zero_division=0), abs=1e-12)
    assert m["precision"] == pytest.approx(
        sk.precision_score(y, predicted, zero_division=0), abs=1e-12
    )
    assert m["f1"] == pytest.approx(sk.f1_score(y, predicted, zero_division=0), abs=1e-12)
    assert m["auc"] == pytest.approx(sk.roc_auc_score(y, p), abs=1e-12)
    assert m["ap"] == pytest.approx(sk.average_precision_score(y, p), abs=1e-12)
    assert m["loss"] == pytest.approx(sk.log_loss(y, p), abs=1e-12)
    assert not math.isnan(m["auc"])
