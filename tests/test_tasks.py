"""Outcomes with more than two classes: labels, loss, scoring, and the two small models."""

import dataclasses
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import obscuradp as odp
from obscuradp.imbalance import ClassWeighted
from obscuradp.mechanisms.dpsgd import check_per_record_model
from obscuradp.metrics import multiclass_metrics
from obscuradp.models import CharGRU, SmallCNN, TabularMLP
from obscuradp.tables import ClientData, make_client_data
from obscuradp.tasks import ManyClasses, YesNo

DELTA = 1e-5
CLASSES = 4

pytestmark = pytest.mark.filterwarnings("ignore:protecting whole participants with only")


def _client(name: str, rows: int, seed: int) -> ClientData:
    """Six columns; the class is the quarter of the plane the first two columns point to."""
    generator = np.random.default_rng(seed)
    features = generator.normal(size=(rows, 6)).astype(np.float32)
    labels = (features[:, 0] > 0).astype(np.int64) + 2 * (features[:, 1] > 0)
    return make_client_data(name, features, labels, seed=seed)


@pytest.fixture(scope="module")
def clients() -> list[ClientData]:
    return [_client(f"site-{index}", 600, index) for index in range(3)]


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, CLASSES))


def _simulation(clients: list[ClientData], **overrides: Any) -> odp.Simulation:
    arguments: dict[str, Any] = {
        "model_fn": _model,
        "clients": clients,
        "plan": odp.Plan(dp=None),
        "policy": None,
        "rounds": 10,
        "local_epochs": 2.0,
        "learning_rate": 0.2,
        "evaluate_every": 0,
        "task": ManyClasses(CLASSES),
        **overrides,
    }
    return odp.Simulation(**arguments)


def test_measures_for_several_classes() -> None:
    scores = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0], [5.0, 0.0, 0.0]])
    found = multiclass_metrics(scores, [0, 1, 2, 1])
    assert found["accuracy"] == 0.75
    # Mean cross-entropy, by hand: the log of each row's share for its true class.
    shares = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    assert found["loss"] == pytest.approx(-np.mean(np.log(shares[[0, 1, 2, 3], [0, 1, 2, 1]])))
    # Huge scores do not overflow.
    assert np.isfinite(multiclass_metrics(scores * 1000.0, [0, 1, 2, 1])["loss"])
    with pytest.raises(ValueError, match="one row of scores per label"):
        multiclass_metrics(scores, [0, 1])
    with pytest.raises(ValueError, match="whole numbers from 0"):
        multiclass_metrics(scores, [0, 1, 2, 3])
    with pytest.raises(ValueError, match="whole numbers from 0"):
        multiclass_metrics(scores, [0, 1, 2, 0.5])
    with pytest.raises(ValueError, match="at least one row"):
        multiclass_metrics(np.zeros((0, 3)), [])


def test_a_task_needs_at_least_two_classes() -> None:
    assert ManyClasses(2).classes == 2
    assert YesNo() == YesNo()
    with pytest.raises(ValueError, match="at least 2"):
        ManyClasses(1)
    with pytest.raises(ValueError, match="whole number"):
        ManyClasses(2.5)  # type: ignore[arg-type]


def test_a_run_with_several_classes_learns_and_is_scored_by_accuracy(
    clients: list[ClientData],
) -> None:
    result = _simulation(clients).run(seed=1)
    assert set(result.test) == set(result.validation) == {"accuracy", "loss"}
    assert result.test["accuracy"] > 0.85  # guessing gives 0.25
    assert result.test_at_chosen_threshold is None
    assert result.config["task"] == {"classes": CLASSES}
    assert result.report is None
    again = _simulation(clients).run(seed=1)
    assert again.test == result.test


def test_runs_with_a_yes_no_outcome_do_not_mention_a_task() -> None:
    generator = np.random.default_rng(0)
    features = generator.normal(size=(200, 6)).astype(np.float32)
    plain = [make_client_data("a", features, features[:, 0] > 0, seed=0)]
    result = odp.Simulation(
        model_fn=lambda: TabularMLP(6, (4,)),
        clients=plain,
        plan=odp.Plan(dp=None),
        policy=None,
        rounds=1,
    ).run(seed=0)
    assert "task" not in result.config
    assert "auc" in result.test


def test_single_rows_can_be_protected_with_several_classes(clients: list[ClientData]) -> None:
    result = _simulation(
        clients,
        plan=odp.Plan(dp=odp.LocalDPSGD(clip_norm=1.0)),
        policy=odp.PrivacyPolicy(8.0, DELTA),
        rounds=5,
        local_epochs=1.0,
    ).run(seed=2)
    assert result.report is not None
    assert result.report.within_budget
    assert odp.verify_report(result.report.to_dict()) == []
    assert result.test["accuracy"] > 0.5


def test_whole_participants_can_be_protected_with_several_classes(
    clients: list[ClientData],
) -> None:
    result = _simulation(
        clients,
        plan=odp.Plan(dp=odp.ClientLevelDP(sampling_rate=1.0, clip_norm=1.0)),
        policy=odp.PrivacyPolicy(8.0, DELTA, unit="client"),
        rounds=3,
        local_epochs=1.0,
        allow_unsafe=True,
    ).run(seed=3)
    assert result.report is not None
    assert result.report.within_budget
    assert set(result.test) == {"accuracy", "loss"}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"outputs": "probability"}, "raw scores"),
        ({"choose_threshold": True}, "decision threshold"),
        ({"plan": odp.Plan(dp=None, imbalance=ClassWeighted(0.2))}, "yes/no outcomes"),
        ({"task": ManyClasses(3)}, "labels must be whole numbers from 0 to 2"),
    ],
)
def test_settings_that_only_make_sense_for_yes_no_are_refused(
    clients: list[ClientData], overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _simulation(clients, **overrides)


def test_labels_that_are_not_whole_numbers_are_refused(clients: list[ClientData]) -> None:
    crooked = dataclasses.replace(clients[0], train_y=clients[0].train_y + 0.5)
    with pytest.raises(ValueError, match="whole numbers"):
        _simulation([crooked, *clients[1:]])


def test_the_small_image_network() -> None:
    model = SmallCNN()
    assert sum(value.numel() for value in model.parameters()) == 44_318
    assert model(torch.rand(5, 784)).shape == (5, 62)
    assert model.probability_output is False
    check_per_record_model(model)  # no layer mixes the rows of a batch
    assert SmallCNN(classes=10, side=20)(torch.rand(2, 400)).shape == (2, 10)
    with pytest.raises(ValueError, match="at least 2"):
        SmallCNN(classes=1)
    with pytest.raises(ValueError, match="at least 16"):
        SmallCNN(side=12)


def test_the_small_text_network() -> None:
    model = CharGRU()
    assert sum(value.numel() for value in model.parameters()) == 65_322
    texts = torch.randint(0, 90, (4, 80)).float()  # symbol numbers stored as ordinary numbers
    assert model(texts).shape == (4, 90)
    check_per_record_model(model)
    # A symbol number outside the vocabulary is read as the nearest one, not as an error.
    assert torch.isfinite(model(torch.full((1, 5), 500.0))).all()
    with pytest.raises(ValueError, match="at least 2"):
        CharGRU(vocabulary=1)


def test_images_and_texts_run_through_the_simulation() -> None:
    generator = np.random.default_rng(4)
    # Images: class 0 is bright on the left half, class 1 on the right half.
    labels = generator.integers(0, 2, 240)
    images = generator.random((240, 16, 16)).astype(np.float32) * 0.2
    for index, label in enumerate(labels):
        images[index, :, : 8 if label == 0 else 16][:, 0 if label == 0 else 8 :] += 0.8
    writers = [
        make_client_data(f"writer-{i}", images[i::3].reshape(-1, 256), labels[i::3], seed=i)
        for i in range(3)
    ]
    seen = odp.Simulation(
        model_fn=lambda: SmallCNN(classes=2, side=16),
        clients=writers,
        plan=odp.Plan(dp=None),
        policy=None,
        rounds=6,
        local_epochs=2.0,
        learning_rate=0.1,
        evaluate_every=0,
        task=ManyClasses(2),
    ).run(seed=4)
    assert seen.test["accuracy"] > 0.9
    # Texts: the next symbol is always the one after the last symbol (counting round to 5).
    texts = generator.integers(0, 5, (300, 6))
    following = (texts[:, -1] + 1) % 5
    roles = [
        make_client_data(f"role-{i}", texts[i::3].astype(np.float32), following[i::3], seed=i)
        for i in range(3)
    ]
    read = odp.Simulation(
        model_fn=lambda: CharGRU(vocabulary=5, embedding=4, hidden=16),
        clients=roles,
        plan=odp.Plan(dp=None),
        policy=None,
        rounds=10,
        local_epochs=3.0,
        learning_rate=1.0,
        evaluate_every=0,
        task=ManyClasses(5),
    ).run(seed=5)
    assert read.test["accuracy"] > 0.9
