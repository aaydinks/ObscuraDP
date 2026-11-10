"""Accounting, local training and mask scoring: the copy equals the research code bit for bit."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
import torch

from obscuradp.legacy import LegacyTrialConfig
from obscuradp.legacy import data as new_data
from obscuradp.legacy import dp as new_dp
from obscuradp.legacy import models as new_models
from obscuradp.legacy import sensitivity as new_sens
from obscuradp.legacy import training as new_training
from obscuradp.selection import masks as new_masks
from obscuradp.state import get_state

pytestmark = [pytest.mark.integration, pytest.mark.filterwarnings("ignore")]

# The resampled sizes of the five stroke participants at seed 42, and the loan size.
CLIENT_SIZES = (484, 234, 1120, 4920, 738, 683)


def _quiet(function: Any, *args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def _same_states(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> None:
    assert list(a) == list(b)
    for key in a:
        assert torch.equal(a[key], b[key]), key


@pytest.fixture(scope="module")
def stroke(cfg: LegacyTrialConfig) -> Any:
    return _quiet(new_data.prepare, cfg, "stroke", seed=42)


@pytest.fixture(scope="module")
def loan(cfg: LegacyTrialConfig) -> Any:
    return _quiet(new_data.prepare, cfg, "loan", seed=42)


# --------------------------------------------------------------------------- accounting


@pytest.mark.parametrize("epsilon", [0.5, 1.0, 9.0, 50.0])
@pytest.mark.parametrize("n_local", CLIENT_SIZES)
def test_noise_multiplier_and_budgets_are_identical(
    oracle: Any, cfg: LegacyTrialConfig, epsilon: float, n_local: int
) -> None:
    old = oracle.module("dp")
    old_sigma = oracle.call(old.get_sigma, epsilon, n_local)
    new_sigma = _quiet(new_dp.get_sigma, cfg, epsilon, n_local)
    assert new_sigma == old_sigma
    assert _quiet(new_dp.per_round_epsilon, cfg, new_sigma, n_local) == oracle.call(
        old.per_round_epsilon, old_sigma, n_local
    )
    assert _quiet(new_dp.cumulative_epsilon, cfg, new_sigma, n_local) == oracle.call(
        old.cumulative_epsilon, old_sigma, n_local
    )
    assert new_dp.dp_noise_std(cfg, new_sigma) == old.dp_noise_std(old_sigma)
    assert _quiet(new_dp.noise_floor, cfg, epsilon, n_local) == oracle.call(
        old.noise_floor, epsilon, n_local
    )


def test_non_default_epochs_rounds_and_batch_are_identical(
    oracle: Any, cfg: LegacyTrialConfig
) -> None:
    old = oracle.module("dp")
    kwargs = {"epochs": 1, "batch_size": 64}
    old_sigma = oracle.call(old.get_sigma, 2.0, 700, **kwargs)
    assert _quiet(new_dp.get_sigma, cfg, 2.0, 700, **kwargs) == old_sigma
    assert _quiet(
        new_dp.cumulative_epsilon, cfg, old_sigma, 700, rounds=3, **kwargs
    ) == oracle.call(old.cumulative_epsilon, old_sigma, 700, rounds=3, **kwargs)
    assert new_dp.dp_noise_std(cfg, 1.5, batch_size=64, max_grad_norm=2.0) == old.dp_noise_std(
        1.5, batch_size=64, max_grad_norm=2.0
    )


def test_the_documented_per_round_defect_is_reproduced(cfg: LegacyTrialConfig) -> None:
    """A labelled budget of 9 is spent every round, so 20 rounds spend far more (finding G2)."""
    sigma = _quiet(new_dp.get_sigma, cfg, 9.0, 484)
    assert _quiet(new_dp.per_round_epsilon, cfg, sigma, 484) == pytest.approx(9.0, abs=0.05)
    assert _quiet(new_dp.cumulative_epsilon, cfg, sigma, 484) > 30.0


# --------------------------------------------------------------------------- local training


@pytest.mark.parametrize("dataset", ["stroke", "loan"])
@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("mu", [0.0, 0.01])
def test_one_local_round_gives_identical_weights(
    oracle: Any,
    cfg: LegacyTrialConfig,
    request: pytest.FixtureRequest,
    dataset: str,
    private: bool,
    mu: float,
) -> None:
    prepared = request.getfixturevalue(dataset)
    in_features = prepared["in_features"]
    # The smallest participant keeps this fast; private training on it takes about a second.
    Xc, yc = min(prepared["client_data"], key=lambda pair: len(pair[1]))
    torch.manual_seed(7)
    global_state = get_state(new_models.new_model(cfg, in_features, dataset))
    sigma = _quiet(new_dp.get_sigma, cfg, 9.0, len(yc)) if private else None
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "dp": private,
        "sigma": sigma,
        "mu": mu,
        "local_epochs": 2,
        "seed": 42_013,
    }
    old_state, old_n, old_stats = oracle.call(
        oracle.module("fedprox").local_train, global_state, Xc, yc, in_features, **kwargs
    )
    new_state, new_n, new_stats = _quiet(
        new_training.local_train, cfg, global_state, Xc, yc, in_features, **kwargs
    )
    _same_states(old_state, new_state)
    assert old_n == new_n == len(yc)
    assert old_stats["n_batches"] == new_stats["n_batches"]
    assert old_stats["n_samples"] == new_stats["n_samples"]
    # Training really moved the weights, so the equality above is not trivial.
    assert not torch.equal(new_state["net.0.weight"], global_state["net.0.weight"])


def test_private_training_needs_a_noise_multiplier(cfg: LegacyTrialConfig, loan: Any) -> None:
    Xc, yc = loan["client_data"][0]
    state = get_state(new_models.new_model(cfg, loan["in_features"], "loan"))
    with pytest.raises(ValueError, match="sigma is required"):
        new_training.local_train(cfg, state, Xc, yc, loan["in_features"], dataset="loan", dp=True)


# --------------------------------------------------------------------------- mask scoring


def _mask_inputs(cfg: LegacyTrialConfig, stroke: Any) -> tuple[Any, Any, Any]:
    torch.manual_seed(42)
    model = new_models.new_model(cfg, stroke["in_features"], "stroke")
    state = get_state(model)
    Xc, yc = stroke["client_data"][0]
    return model, state, (Xc, yc)


def test_scoring_batch_is_identical(oracle: Any, cfg: LegacyTrialConfig, stroke: Any) -> None:
    old = oracle.module("sensitivity")
    Xc, yc = stroke["client_data"][0]
    for seed in (None, 42):
        old_x, old_y = old.tabular_batch(Xc, yc, n=256, seed=seed)
        new_x, new_y = new_sens.tabular_batch(Xc, yc, n=256, seed=seed)
        assert torch.equal(old_x, new_x)
        assert torch.equal(old_y, new_y)


@pytest.mark.parametrize("fraction", [0.1, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("mode", ["clean", "dp", "random"])
def test_masks_and_scores_are_identical(
    oracle: Any, cfg: LegacyTrialConfig, stroke: Any, mode: str, fraction: float
) -> None:
    old = oracle.module("sensitivity")
    model, state, (Xc, yc) = _mask_inputs(cfg, stroke)
    batch = new_sens.tabular_batch(Xc, yc, n=64, seed=42)
    sigma = 0.7 if mode == "dp" else None
    old_mask, old_scores = oracle.call(
        old.make_mask, model, state, batch, fraction=fraction, mode=mode, sigma=sigma, seed=42
    )
    new_mask, new_scores = _quiet(
        new_sens.make_mask,
        cfg,
        model,
        state,
        batch,
        fraction=fraction,
        mode=mode,
        sigma=sigma,
        seed=42,
    )
    assert np.array_equal(old_mask, new_mask)
    assert old_scores == new_scores
    if mode != "random":
        assert len(new_scores) == 6


def test_mask_helpers_are_identical(oracle: Any, cfg: LegacyTrialConfig, stroke: Any) -> None:
    old = oracle.module("sensitivity")
    model, state, (Xc, yc) = _mask_inputs(cfg, stroke)
    batch = new_sens.tabular_batch(Xc, yc, n=64, seed=1)
    clean = new_sens.sensitivity_scores(model, batch)
    assert clean == old.sensitivity_scores(model, batch)
    noisy = new_sens.dp_sensitivity_scores(cfg, model, batch, 0.7, seed=3)
    assert noisy == old.dp_sensitivity_scores(model, batch, 0.7, seed=3)

    for fraction in (0.1, 0.25, 0.5):
        assert np.array_equal(
            old.build_mask(state, clean, fraction), new_masks.build_mask(state, clean, fraction)
        )
        assert old.tensor_selection(state, noisy, fraction) == new_masks.tensor_selection(
            state, noisy, fraction
        )
        assert old.tensor_selection_agreement(
            state, clean, noisy, fraction
        ) == new_masks.tensor_selection_agreement(state, clean, noisy, fraction)
        assert old.random_jaccard_expected(fraction) == new_masks.random_jaccard_expected(fraction)
        assert np.array_equal(
            old.random_mask(3137, fraction, 9), new_masks.random_mask(3137, fraction, 9)
        )
    a, b = new_masks.build_mask(state, clean, 0.25), new_masks.build_mask(state, noisy, 0.25)
    assert old.mask_agreement(a, b) == new_masks.mask_agreement(a, b)
    # The library's rank correlation needs no scipy; the research code used scipy.
    assert new_masks.score_rank_correlation(clean, noisy) == pytest.approx(
        old.score_rank_correlation(clean, noisy), abs=1e-12
    )


def test_mask_modes_that_cannot_work_are_refused(cfg: LegacyTrialConfig, stroke: Any) -> None:
    model, state, (Xc, yc) = _mask_inputs(cfg, stroke)
    batch = new_sens.tabular_batch(Xc, yc, n=8)
    with pytest.raises(ValueError, match="requires sigma"):
        new_sens.make_mask(cfg, model, state, batch, 0.25, "dp")
    with pytest.raises(ValueError, match="requires a seed"):
        new_sens.make_mask(cfg, model, state, batch, 0.25, "random")
    with pytest.raises(ValueError, match="Unknown mode"):
        new_sens.make_mask(cfg, model, state, batch, 0.25, "banana")
