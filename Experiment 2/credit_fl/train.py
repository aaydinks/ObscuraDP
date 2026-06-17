"""Local training (plaintext + DP-SGD), FedAvg aggregation, and evaluation.

The federation is simulated in-process (5 synchronous clients). Each round every
client copies the global weights, trains locally for LOCAL_EPOCHS, and the server
averages the returned weights. Three privacy regimes share this code path:

  * Standard FL  -- plaintext local SGD, plaintext averaging.
  * DP-FL        -- Opacus DP-SGD locally (Gaussian mechanism + RDP accounting),
                    plaintext averaging.
  * HE-FL        -- identical plaintext local SGD, but averaging happens on CKKS
                    ciphertexts (see he.py); this module supplies the plaintext
                    training + the reference plaintext aggregation.

Averaging is the unweighted mean over the 5 clients (sum / N), matching the
paper's HE description "sum the ciphertexts, divide by 5". With the IID,
near-equal partition this is equivalent to sample-weighted FedAvg, and using the
same rule everywhere lets HE-FL reproduce Standard FL within CKKS rounding error.
"""
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier

from . import config
from .model import new_model, get_weights, set_weights

warnings.filterwarnings("ignore", category=UserWarning)

_SIGMA_CACHE = {}


def _make_loader(Xc, yc):
    ds = TensorDataset(torch.from_numpy(Xc),
                       torch.from_numpy(yc.astype(np.float32)))
    return DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True)


def local_train_plain(global_state, Xc, yc, in_features):
    """One client's plaintext local SGD for LOCAL_EPOCHS. Returns (state, n)."""
    model = new_model(in_features)
    set_weights(model, global_state)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    criterion = nn.BCELoss()
    loader = _make_loader(Xc, yc)

    for _ in range(config.LOCAL_EPOCHS):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    return get_weights(model), len(yc)


def total_dp_epochs():
    """Cumulative local epochs over which each client spends its privacy budget."""
    return config.ROUNDS * config.LOCAL_EPOCHS


def get_sigma(epsilon, n_local):
    """Noise multiplier sigma achieving cumulative `epsilon` over all rounds.

    The budget is composed across ROUNDS*LOCAL_EPOCHS epochs at this client's
    Poisson sampling rate, so a single fixed sigma is reused every round and the
    total privacy spend equals the target. Cached per (eps, n)."""
    key = (epsilon, n_local)
    if key in _SIGMA_CACHE:
        return _SIGMA_CACHE[key]
    sample_rate = min(1.0, config.BATCH_SIZE / n_local)
    sigma = get_noise_multiplier(
        target_epsilon=epsilon,
        target_delta=config.TARGET_DELTA,
        sample_rate=sample_rate,
        epochs=total_dp_epochs(),
        accountant="rdp",
    )
    _SIGMA_CACHE[key] = sigma
    return sigma


def actual_epsilon(sigma, n_local):
    """Re-derive the cumulative epsilon actually consumed for (sigma, n_local).

    Mirrors Opacus's get_noise_multiplier step convention exactly
    (steps = int(epochs/sample_rate), one composed RDP history entry) so the
    reported value matches the target the sigma was solved for."""
    sample_rate = min(1.0, config.BATCH_SIZE / n_local)
    steps = int(total_dp_epochs() / sample_rate)
    acc = RDPAccountant()
    acc.history = [(sigma, sample_rate, steps)]
    return float(acc.get_epsilon(delta=config.TARGET_DELTA))


def local_train_dp(global_state, Xc, yc, sigma, in_features):
    """One client's DP-SGD local training for LOCAL_EPOCHS. Returns (state, n)."""
    model = new_model(in_features)
    set_weights(model, global_state)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    criterion = nn.BCELoss()
    loader = _make_loader(Xc, yc)

    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=sigma,
        max_grad_norm=config.MAX_GRAD_NORM,
    )

    for _ in range(config.LOCAL_EPOCHS):
        for xb, yb in loader:
            if xb.numel() == 0:        # Poisson sampling can yield empty batches
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    return get_weights(model), len(yc)


def aggregate(states):
    """Unweighted FedAvg: plain mean of client state_dicts (sum / N)."""
    keys = states[0].keys()
    n = len(states)
    return {k: torch.stack([s[k].float() for s in states], 0).sum(0) / n
            for k in keys}


def evaluate(state, X_test, y_test, in_features):
    """Held-out accuracy + mean BCE loss at a 0.5 threshold."""
    model = new_model(in_features)
    set_weights(model, state)
    model.eval()
    with torch.no_grad():
        probs = model(torch.from_numpy(X_test))
        y = torch.from_numpy(y_test.astype(np.float32))
        loss = float(nn.BCELoss()(probs, y))
        probs = probs.numpy()
    preds = (probs >= 0.5).astype(int)
    acc = float((preds == y_test).mean())
    return {"accuracy": acc, "loss": loss}
