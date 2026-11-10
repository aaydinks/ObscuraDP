"""Privacy accounting exactly as in Trial 07/27 (comparison only, see the package docstring).

Known defect reproduced on purpose (finding G2): ``epsilon`` here is a budget for EACH
ROUND. The noise multiplier is solved for one round's local epochs and reused every round,
so the budget for a whole run is several times the labelled value. The accountant is also
told a slightly different sampling rate and step count than training actually uses.

Needs Opacus (the ``legacy`` extra).
"""

from __future__ import annotations

import warnings
from typing import Any

from obscuradp.legacy.config import LegacyTrialConfig

_SIGMA_CACHE: dict[tuple[float, int, int, int, float], float] = {}


def get_sigma(
    cfg: LegacyTrialConfig,
    epsilon: float,
    n_local: int,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> float:
    """Noise multiplier that spends ``epsilon`` over one round's local epochs."""
    from opacus.accountants.utils import get_noise_multiplier

    if epochs is None:
        epochs = cfg.local_epochs
    if batch_size is None:
        batch_size = cfg.batch_size
    key = (epsilon, n_local, epochs, batch_size, cfg.target_delta)
    if key in _SIGMA_CACHE:
        return _SIGMA_CACHE[key]
    sample_rate = min(1.0, batch_size / n_local)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        sigma = float(
            get_noise_multiplier(
                target_epsilon=epsilon,
                target_delta=cfg.target_delta,
                sample_rate=sample_rate,
                epochs=epochs,
                accountant="rdp",
            )
        )
    _SIGMA_CACHE[key] = sigma
    return sigma


def per_round_epsilon(
    cfg: LegacyTrialConfig,
    sigma: float,
    n_local: int,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> float:
    """Budget spent in one round with this noise multiplier and participant size."""
    from opacus.accountants import RDPAccountant

    if epochs is None:
        epochs = cfg.local_epochs
    if batch_size is None:
        batch_size = cfg.batch_size
    sample_rate = min(1.0, batch_size / n_local)
    steps = int(epochs / sample_rate)
    acc = RDPAccountant()
    acc.history = [(sigma, sample_rate, steps)]
    return float(acc.get_epsilon(delta=cfg.target_delta))


def cumulative_epsilon(
    cfg: LegacyTrialConfig,
    sigma: float,
    n_local: int,
    rounds: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> float:
    """Budget spent over all rounds, assuming the same noise multiplier every round."""
    from opacus.accountants import RDPAccountant

    if rounds is None:
        rounds = cfg.rounds
    if epochs is None:
        epochs = cfg.local_epochs
    if batch_size is None:
        batch_size = cfg.batch_size
    sample_rate = min(1.0, batch_size / n_local)
    steps = int(rounds * epochs / sample_rate)
    acc = RDPAccountant()
    acc.history = [(sigma, sample_rate, steps)]
    return float(acc.get_epsilon(delta=cfg.target_delta))


def make_private(
    cfg: LegacyTrialConfig, model: Any, optimizer: Any, loader: Any, sigma: float
) -> tuple[Any, Any, Any, Any]:
    """Wrap model, optimizer and loader with a fresh Opacus engine (one per call)."""
    from opacus import PrivacyEngine

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        engine = PrivacyEngine(accountant="rdp")
        model, optimizer, loader = engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=sigma,
            max_grad_norm=cfg.max_grad_norm,
        )
    return model, optimizer, loader, engine


def dp_noise_std(
    cfg: LegacyTrialConfig,
    sigma: float,
    batch_size: int | None = None,
    max_grad_norm: float | None = None,
) -> float:
    """``sigma * C / B`` with the nominal batch size (Opacus uses its expected batch size)."""
    if batch_size is None:
        batch_size = cfg.batch_size
    if max_grad_norm is None:
        max_grad_norm = cfg.max_grad_norm
    return sigma * max_grad_norm / batch_size


def noise_floor(cfg: LegacyTrialConfig, epsilon: float, n_local: int) -> float:
    """``dp_noise_std`` of the noise multiplier for a per-round ``epsilon``."""
    return dp_noise_std(cfg, get_sigma(cfg, epsilon, n_local))
