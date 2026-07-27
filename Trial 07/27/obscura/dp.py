"""Opacus DP-SGD integration: noise multiplier solving, epsilon accounting, and private training.

This module reconciles two prior epsilon conventions:
  - Experiment 1 (arXiv:2508.10017) treated epsilon as a PER-ROUND budget: sigma is
    solved for config.LOCAL_EPOCHS at the client's sampling rate and reused every round.
  - Experiment 2 (Sci. Rep. 10.1038/s41598-025-34536-9) treated epsilon as CUMULATIVE
    over ROUNDS*LOCAL_EPOCHS total epochs.

Trial 07/27 adopts Experiment 1's per-round convention so its epsilon grid is directly
comparable to Experiment 1's frontier. Additionally, cumulative_epsilon() reports the
composed privacy spend over all rounds, so both per-round and cumulative budgets are
visible in every result record -- nothing is hidden.
"""
import warnings

import numpy as np
import torch
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier

from . import config

warnings.filterwarnings("ignore", category=UserWarning)

_SIGMA_CACHE = {}


def get_sigma(epsilon, n_local, epochs=None, batch_size=None):
    """Noise multiplier sigma achieving epsilon over LOCAL_EPOCHS at this client's
    sampling rate (per-round privacy budget). Cached on (epsilon, n_local, epochs, batch_size).

    epochs defaults to config.LOCAL_EPOCHS, batch_size to config.BATCH_SIZE.
    sample_rate = min(1.0, batch_size / n_local). Uses opacus.accountants.utils.
    get_noise_multiplier with target_delta=config.TARGET_DELTA and accountant="rdp".
    Solving sigma is slow (root-finding); the cache is critical when sweeps call this
    thousands of times.
    """
    if epochs is None:
        epochs = config.LOCAL_EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    key = (epsilon, n_local, epochs, batch_size)
    if key in _SIGMA_CACHE:
        return _SIGMA_CACHE[key]
    sample_rate = min(1.0, batch_size / n_local)
    sigma = get_noise_multiplier(
        target_epsilon=epsilon,
        target_delta=config.TARGET_DELTA,
        sample_rate=sample_rate,
        epochs=epochs,
        accountant="rdp",
    )
    _SIGMA_CACHE[key] = sigma
    return sigma


def per_round_epsilon(sigma, n_local, epochs=None, batch_size=None):
    """Epsilon consumed in one round with this (sigma, n_local) pair.

    Builds an RDPAccountant, sets acc.history = [(sigma, sample_rate, steps)] where
    steps = int(epochs / sample_rate), and returns acc.get_epsilon(delta=config.TARGET_DELTA).
    Mirrors Opacus's own step convention exactly so that get_sigma -> per_round_epsilon
    reproduces the target epsilon to numerical precision.
    """
    if epochs is None:
        epochs = config.LOCAL_EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    sample_rate = min(1.0, batch_size / n_local)
    steps = int(epochs / sample_rate)
    acc = RDPAccountant()
    acc.history = [(sigma, sample_rate, steps)]
    return float(acc.get_epsilon(delta=config.TARGET_DELTA))


def cumulative_epsilon(sigma, n_local, rounds=None, epochs=None, batch_size=None):
    """Epsilon consumed across all rounds, assuming the same sigma every round.

    Builds an RDPAccountant, sets acc.history = [(sigma, sample_rate, steps)] where
    steps = int(rounds * epochs / sample_rate), and returns acc.get_epsilon(delta=config.TARGET_DELTA).
    rounds defaults to config.ROUNDS, epochs to config.LOCAL_EPOCHS, batch_size to config.BATCH_SIZE.
    """
    if rounds is None:
        rounds = config.ROUNDS
    if epochs is None:
        epochs = config.LOCAL_EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    sample_rate = min(1.0, batch_size / n_local)
    steps = int(rounds * epochs / sample_rate)
    acc = RDPAccountant()
    acc.history = [(sigma, sample_rate, steps)]
    return float(acc.get_epsilon(delta=config.TARGET_DELTA))


def make_private(model, optimizer, loader, sigma):
    """Wrap model, optimizer, and loader with Opacus PrivacyEngine for DP-SGD.

    Returns the 4-tuple (model, optimizer, loader, engine) as expected by local_train.
    """
    engine = PrivacyEngine(accountant="rdp")
    model, optimizer, loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=sigma,
        max_grad_norm=config.MAX_GRAD_NORM,
    )
    return model, optimizer, loader, engine


def dp_noise_std(sigma, batch_size=None, max_grad_norm=None):
    """Standard deviation of the Gaussian noise added to the AVERAGED per-sample gradient.

    Returns sigma * C / B where C is max_grad_norm (clipping threshold) and B is batch_size.
    This is the quantity Trial 07/27 compares CKKS rounding error against: the headline
    claim is that HE's ~1e-7 aggregation error is orders of magnitude below the noise DP
    already injects, so HE is free *relative to* DP.
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if max_grad_norm is None:
        max_grad_norm = config.MAX_GRAD_NORM
    return sigma * max_grad_norm / batch_size


def noise_floor(epsilon, n_local):
    """Convenience: dp_noise_std(get_sigma(epsilon, n_local)).

    The noise magnitude that DP-SGD injects for a given per-round epsilon budget.
    """
    sigma = get_sigma(epsilon, n_local)
    return dp_noise_std(sigma)
