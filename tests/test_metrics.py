"""Yes/no measures computed with numpy only."""

import math

import numpy as np
import pytest
import torch

from obscuradp.metrics import (
    average_precision,
    best_threshold,
    binary_metrics,
    evaluate,
    roc_auc,
    summarize,
)
from obscuradp.models import TabularMLP


def test_hand_checked_example() -> None:
    #            TP    FN    FP    TN    TP    TN
    p = np.array([0.9, 0.2, 0.7, 0.1, 0.6, 0.4])
    y = np.array([1, 1, 0, 0, 1, 0])
    m = binary_metrics(p, y)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["pred_pos_rate"] == pytest.approx(0.5)
    # Positive scores 0.9, 0.2, 0.6 against negative 0.7, 0.1, 0.4: 6 of 9 pairs ordered right.
    assert m["auc"] == pytest.approx(6 / 9)
    # Sorted by score: 1, 0, 1, 0, 1(0.2 is fifth), ... precision at each hit: 1, 2/3, 3/5.
    assert m["ap"] == pytest.approx((1 + 2 / 3 + 3 / 5) / 3)
    expected_loss = -np.mean(np.log([0.9, 0.2, 0.3, 0.9, 0.6, 0.6]))
    assert m["loss"] == pytest.approx(expected_loss)


def test_perfect_and_reversed_rankings() -> None:
    y = np.array([0, 0, 1, 1])
    assert roc_auc([0.1, 0.2, 0.8, 0.9], y) == 1.0
    assert roc_auc([0.9, 0.8, 0.2, 0.1], y) == 0.0
    assert average_precision([0.1, 0.2, 0.8, 0.9], y) == 1.0


def test_ties_count_as_half() -> None:
    assert roc_auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    assert average_precision([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5


def test_one_class_gives_nan_for_ranking_measures() -> None:
    m = binary_metrics([0.2, 0.8], [1, 1])
    assert math.isnan(m["auc"])
    assert math.isnan(m["ap"])
    assert m["recall"] == 0.5


def test_zero_denominators_give_zero() -> None:
    m = binary_metrics([0.1, 0.2], [0, 0])
    assert m["recall"] == 0.0
    assert m["precision"] == 0.0
    assert m["f1"] == 0.0


def test_extreme_probabilities_give_a_finite_loss() -> None:
    m = binary_metrics([0.0, 1.0], [1, 0])
    assert m["loss"] == 100.0


def test_threshold_moves_predictions_but_not_ranking_measures() -> None:
    p, y = [0.3, 0.6, 0.8], [0, 1, 1]
    low, high = binary_metrics(p, y, threshold=0.2), binary_metrics(p, y, threshold=0.7)
    assert low["pred_pos_rate"] == 1.0
    assert high["pred_pos_rate"] == pytest.approx(1 / 3)
    assert low["auc"] == high["auc"]
    assert low["ap"] == high["ap"]


@pytest.mark.parametrize(
    ("p", "y", "message"),
    [([0.5], [0, 1], "same length"), ([], [], "at least one row"), ([0.5, 0.5], [0, 2], "0 or 1")],
)
def test_bad_inputs_are_refused(p: list[float], y: list[int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        binary_metrics(p, y)


def test_evaluate_runs_the_model_and_restores_training_mode() -> None:
    torch.manual_seed(0)
    model = TabularMLP(4, (3,))
    model.train()
    x = np.random.default_rng(0).normal(size=(20, 4)).astype(np.float32)
    y = (np.arange(20) % 2).astype(np.int64)
    m = evaluate(model, x, y)
    assert model.training
    with torch.no_grad():
        expected = binary_metrics(model.eval()(torch.from_numpy(x)).numpy(), y)
    assert m == expected


def test_summarize_uses_the_sample_standard_deviation() -> None:
    runs = [{"recall": 0.5, "name": "a"}, {"recall": 0.7, "name": "b"}, {"recall": 0.9, "ok": True}]
    s = summarize(runs)
    assert set(s) == {"recall"}
    assert s["recall"]["mean"] == pytest.approx(0.7)
    assert s["recall"]["std"] == pytest.approx(0.2)  # population value would be 0.163
    assert s["recall"]["n"] == 3


def test_summarize_handles_nan_and_single_runs() -> None:
    s = summarize([{"auc": math.nan, "acc": 0.9}, {"auc": 0.8}])
    assert s["auc"]["mean"] == 0.8
    assert s["auc"]["n"] == 1
    assert math.isnan(s["auc"]["std"])
    assert math.isnan(s["acc"]["std"])
    assert summarize([]) == {}


def test_best_threshold_by_hand() -> None:
    scores = [0.1, 0.4, 0.35, 0.8]
    labels = [0, 0, 1, 1]
    # At 0.35 both positives are found with one false alarm (F1 0.8); at 0.8 one is missed (0.67).
    assert best_threshold(scores, labels) == 0.35
    assert best_threshold(scores, labels, metric="precision") == 0.8
    assert best_threshold(scores, labels, metric="recall") == 0.1  # ties go to the lowest
    assert best_threshold(scores, labels, metric="accuracy") == 0.35
    chosen = best_threshold(scores, labels)
    assert binary_metrics(scores, labels, threshold=chosen)["f1"] == pytest.approx(0.8)
    with pytest.raises(ValueError, match="metric must be one of"):
        best_threshold(scores, labels, metric="auc")


def test_best_threshold_is_never_worse_than_one_half() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        labels = (rng.random(200) < 0.1).astype(int)
        labels[0] = 1
        scores = np.clip(rng.normal(0.2 + 0.2 * labels, 0.1), 0.01, 0.99)
        chosen = best_threshold(scores, labels)
        assert (
            binary_metrics(scores, labels, threshold=chosen)["f1"]
            >= binary_metrics(scores, labels, threshold=0.5)["f1"]
        )
