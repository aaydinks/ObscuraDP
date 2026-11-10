"""The library, wired the old way, gives exactly what the Trial 07/27 research code gives.

"Exactly" means bit for bit: every array is compared with ``array_equal``, every state with
``torch.equal`` and every number with ``==``. These parts of the pipeline are deterministic,
so any difference is a porting mistake.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from obscuradp.aggregation import aggregate_states, fedavg_weights, weighted_mean
from obscuradp.legacy import LegacyTrialConfig
from obscuradp.legacy import data as new_data
from obscuradp.legacy import metrics as new_metrics
from obscuradp.legacy import models as new_models
from obscuradp.partition import dirichlet_partition, iid_partition
from obscuradp.state import flatten_state, get_state, state_layout, unflatten_state

pytestmark = [pytest.mark.integration, pytest.mark.filterwarnings("ignore")]

SEEDS = (42, 43, 44, 45, 46)
DATASETS = ("stroke", "loan")


def _same_arrays(a: Any, b: Any) -> None:
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype
    assert a.shape == b.shape
    assert np.array_equal(a, b)


def _same_states(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> None:
    assert list(a) == list(b)
    for key in a:
        assert a[key].dtype == b[key].dtype
        assert torch.equal(a[key], b[key])


def _quiet(function: Any, *args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


# --------------------------------------------------------------------------- settings


def test_every_research_setting_has_the_same_value(oracle: Any, cfg: LegacyTrialConfig) -> None:
    old = oracle.module("config")
    names = [n for n in vars(old) if n.isupper()]
    assert len(names) >= 45

    def normal(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return tuple(normal(v) for v in value)
        if isinstance(value, dict) or hasattr(value, "items"):
            return {k: normal(v) for k, v in value.items()}
        return value

    special = {
        "DATA_DIR": cfg.data_dir.name,
        "RESULTS_DIR": str(LegacyTrialConfig().results_dir),
        "STROKE_PATH": (Path(old.DATA_DIR) / cfg.stroke_file).as_posix(),
        "LOAN_PATH": (Path(old.DATA_DIR) / cfg.loan_file).as_posix(),
    }
    for name in names:
        expected = normal(getattr(old, name))
        actual = special[name] if name in special else normal(getattr(cfg, name.lower()))
        assert actual == expected, name
        if name not in special:
            assert type(actual) is type(expected), name


# --------------------------------------------------------------------------- splitting rows


@pytest.mark.parametrize("seed", SEEDS)
def test_dirichlet_partition_is_identical(oracle: Any, cfg: LegacyTrialConfig, seed: int) -> None:
    old = oracle.module("data")
    y = (np.random.RandomState(seed).rand(3926) < 0.0426).astype(np.int64)
    old_parts, old_info = old.dirichlet_partition(y, 5, alpha=0.5, min_pos=12, seed=seed)
    new_parts, new_info = dirichlet_partition(y, 5, alpha=0.5, seed=seed, min_per_class=12)
    assert len(old_parts) == len(new_parts)
    for a, b in zip(old_parts, new_parts, strict=True):
        _same_arrays(a, b)
    assert new_info.attempts == old_info["attempts"]
    assert list(new_info.sizes) == old_info["sizes"]
    assert list(new_info.class_counts[1]) == old_info["pos_counts"]
    assert list(new_info.class_counts[0]) == old_info["neg_counts"]


@pytest.mark.parametrize("seed", SEEDS)
def test_iid_partition_is_identical(oracle: Any, seed: int) -> None:
    old_parts, old_info = oracle.module("data").iid_partition(3415, 5, seed)
    new_parts, new_info = iid_partition(3415, 5, seed)
    for a, b in zip(old_parts, new_parts, strict=True):
        _same_arrays(a, b)
    assert list(new_info.sizes) == old_info["sizes"]


# --------------------------------------------------------------------------- data


@pytest.mark.parametrize("dataset", DATASETS)
def test_loading_is_identical(oracle: Any, cfg: LegacyTrialConfig, dataset: str) -> None:
    old_x, old_y, old_names = oracle.call(oracle.module("data").load_dataset, dataset)
    new_x, new_y, new_names = _quiet(new_data.load_dataset, cfg, dataset)
    _same_arrays(old_x, new_x)
    _same_arrays(old_y, new_y)
    assert old_names == new_names


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("seed", [42, 44])
def test_prepared_data_is_identical(
    oracle: Any, cfg: LegacyTrialConfig, dataset: str, seed: int
) -> None:
    """Split, scaling, partition and per-participant rows (resampled for stroke)."""
    old = oracle.call(oracle.module("orchestrator").prepare, dataset, seed=seed)
    new = _quiet(new_data.prepare, cfg, dataset, seed=seed)
    assert set(old) == set(new)
    for key in ("X_tr", "y_tr", "X_te", "y_te"):
        _same_arrays(old[key], new[key])
    assert old["info"] == new["info"]
    assert old["part_info"] == new["part_info"]
    assert old["in_features"] == new["in_features"]
    for a, b in zip(old["parts"], new["parts"], strict=True):
        _same_arrays(a, b)
    assert len(old["client_data"]) == len(new["client_data"]) == cfg.num_clients
    for (old_x, old_y), (new_x, new_y) in zip(old["client_data"], new["client_data"], strict=True):
        _same_arrays(old_x, new_x)
        _same_arrays(old_y, new_y)


def test_resampling_can_be_switched_off_identically(oracle: Any, cfg: LegacyTrialConfig) -> None:
    old = oracle.call(oracle.module("orchestrator").prepare, "stroke", seed=42, use_smote=False)
    new = _quiet(new_data.prepare, cfg, "stroke", seed=42, use_smote=False)
    for (old_x, old_y), (new_x, new_y) in zip(old["client_data"], new["client_data"], strict=True):
        _same_arrays(old_x, new_x)
        _same_arrays(old_y, new_y)
    assert [len(y) for _, y in new["client_data"]] == new["part_info"]["sizes"]


# --------------------------------------------------------------------------- models and state


@pytest.mark.parametrize(
    ("dataset", "in_features", "width_scale", "n_params"),
    [("stroke", 15, 1, 3137), ("loan", 11, 1, 929), ("stroke", 15, 4, 37121)],
)
def test_models_start_from_identical_weights(
    oracle: Any,
    cfg: LegacyTrialConfig,
    dataset: str,
    in_features: int,
    width_scale: int,
    n_params: int,
) -> None:
    old_models = oracle.module("models")
    torch.manual_seed(42)
    old_model = old_models.new_model(in_features, dataset=dataset, width_scale=width_scale)
    torch.manual_seed(42)
    new_model = new_models.new_model(cfg, in_features, dataset=dataset, width_scale=width_scale)

    old_state, new_state = old_models.get_weights(old_model), get_state(new_model)
    _same_states(old_state, new_state)
    assert old_models.count_params(old_model) == n_params

    old_flat, new_flat = old_models.flatten_state(old_state), flatten_state(new_state)
    _same_arrays(old_flat, new_flat)
    assert old_models.param_index_map(old_state) == [
        (e.name, e.shape, e.size) for e in state_layout(new_state)
    ]

    noisy = old_flat + np.random.RandomState(0).normal(0, 1e-3, old_flat.shape)
    _same_states(old_models.unflatten_state(noisy, old_state), unflatten_state(noisy, new_state))

    x = torch.randn(32, in_features, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert torch.equal(old_model(x), new_model(x))


def test_state_template_is_identical(oracle: Any, cfg: LegacyTrialConfig) -> None:
    torch.manual_seed(7)
    old_state = oracle.module("models").state_template(15, dataset="stroke")
    torch.manual_seed(7)
    _same_states(old_state, new_models.state_template(cfg, 15, dataset="stroke"))


# --------------------------------------------------------------------------- averaging


@pytest.mark.parametrize("weighted", [True, False])
def test_plain_averaging_is_identical(oracle: Any, cfg: LegacyTrialConfig, weighted: bool) -> None:
    old_agg, old_he = oracle.module("aggregate"), oracle.module("he")
    torch.manual_seed(3)
    states = [get_state(new_models.new_model(cfg, 15, "stroke")) for _ in range(5)]
    counts = [484, 234, 1120, 4920, 738]

    _same_arrays(
        old_agg.fedavg_weights(counts, weighted=weighted), fedavg_weights(counts, weighted=weighted)
    )

    flat = [flatten_state(s) for s in states]
    weights = fedavg_weights(counts, weighted=weighted)
    _same_arrays(old_he.fedavg_plaintext(flat, weights=weights), weighted_mean(flat, weights))
    _same_arrays(old_he.fedavg_plaintext(flat), weighted_mean(flat))
    _same_arrays(
        old_agg.plaintext_reference(states, counts, weighted=weighted), weighted_mean(flat, weights)
    )

    old_state, old_stats = old_agg.aggregate_states(
        states, counts, mode="plaintext", weighted=weighted
    )
    _same_states(old_state, aggregate_states(states, counts, weighted=weighted))
    assert old_stats["n_params"] == flat[0].size


# --------------------------------------------------------------------------- evaluation


@pytest.mark.parametrize("dataset", DATASETS)
def test_evaluation_is_identical(oracle: Any, cfg: LegacyTrialConfig, dataset: str) -> None:
    prepared = _quiet(new_data.prepare, cfg, dataset, seed=42)
    torch.manual_seed(42)
    state = get_state(new_models.new_model(cfg, prepared["in_features"], dataset))
    args = (state, prepared["X_te"], prepared["y_te"], prepared["in_features"])
    old = oracle.call(oracle.module("metrics").evaluate, *args, dataset=dataset)
    new = _quiet(new_metrics.evaluate, cfg, *args, dataset=dataset)
    assert old == new
    assert oracle.module("metrics").primary_metric(dataset) == new_metrics.primary_metric(dataset)


# --------------------------------------------------------------------------- refusals


def test_unknown_names_are_refused_like_the_research_code(
    oracle: Any, cfg: LegacyTrialConfig
) -> None:
    old_data, old_metrics = oracle.module("data"), oracle.module("metrics")
    y = np.zeros(10, dtype=np.int64)
    for old_call, new_call in [
        (lambda: old_data.load_dataset("iris"), lambda: new_data.load_dataset(cfg, "iris")),
        (lambda: old_data.split_and_scale("iris"), lambda: new_data.split_and_scale(cfg, "iris")),
        (lambda: old_data.partition("iris", y), lambda: new_data.partition(cfg, "iris", y)),
        (lambda: old_metrics.primary_metric("iris"), lambda: new_metrics.primary_metric("iris")),
    ]:
        with pytest.raises(ValueError, match="Unknown dataset"):
            old_call()
        with pytest.raises(ValueError, match="Unknown dataset"):
            new_call()
    with pytest.raises(ValueError, match="Unknown dataset"):
        cfg.hidden_for("iris")


def test_default_seed_is_the_configured_one(oracle: Any, cfg: LegacyTrialConfig) -> None:
    old = oracle.call(oracle.module("data").split_and_scale, "loan")
    new = _quiet(new_data.split_and_scale, cfg, "loan")
    for a, b in zip(old[:4], new[:4], strict=True):
        _same_arrays(a, b)
    old_parts, _ = oracle.module("data").partition("loan", old[1])
    new_parts, _ = new_data.partition(cfg, "loan", new[1])
    for a, b in zip(old_parts, new_parts, strict=True):
        _same_arrays(a, b)
