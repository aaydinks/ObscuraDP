"""Federated training: SMOTETomek + FedProx/FedAvg under DP-SGD (Opacus).

The federation is simulated in-process (5 clients, synchronous rounds). Each
round every client trains locally on its (optionally SMOTETomek-balanced) data
with DP-SGD, and the server does FedAvg weighted aggregation. FedProx adds a
proximal pull toward the global weights.

Why the proximal term is injected manually: under Opacus DP-SGD the optimizer
overwrites `p.grad` with the clipped+noised per-sample gradient during its
private step, so a proximal term merely *added to the loss* would be silently
discarded. We therefore run Opacus's private step (`pre_step`), then add the
proximal gradient mu*(w - w_global) to `p.grad`, then let Adam apply the sum.
The proximal term is data-independent (depends only on public global weights),
so adding it post-clipping does not affect the privacy guarantee.
"""
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score,
)

from . import config
from .model import new_model, get_weights, set_weights

warnings.filterwarnings("ignore", category=UserWarning)

_SIGMA_CACHE = {}


def smote_resample(X, y):
    """Apply SMOTETomek to a client's local data (oversample minority + clean).

    k_neighbors is clamped to the smallest class size so resampling never fails
    on a small / skewed client (SMOTE needs k_neighbors < n_minority)."""
    n_min = int(np.bincount(y).min())
    k = max(1, min(5, n_min - 1))
    smt = SMOTETomek(
        random_state=config.SEED,
        smote=SMOTE(random_state=config.SEED, k_neighbors=k),
    )
    Xr, yr = smt.fit_resample(X, y)
    return Xr.astype(np.float32), yr.astype(np.int64)


def get_sigma(epsilon, n_local):
    """Noise multiplier sigma achieving `epsilon` over LOCAL_EPOCHS at this
    client's sampling rate (per-round privacy budget). Cached per (eps, n)."""
    key = (epsilon, n_local)
    if key in _SIGMA_CACHE:
        return _SIGMA_CACHE[key]
    sample_rate = min(1.0, config.BATCH_SIZE / n_local)
    sigma = get_noise_multiplier(
        target_epsilon=epsilon,
        target_delta=config.TARGET_DELTA,
        sample_rate=sample_rate,
        epochs=config.LOCAL_EPOCHS,
        accountant="rdp",
    )
    _SIGMA_CACHE[key] = sigma
    return sigma


def local_train(global_state, Xc, yc, epsilon, mu, in_features):
    """One client's local DP-SGD training for LOCAL_EPOCHS. Returns (state, n)."""
    model = new_model(in_features)
    set_weights(model, global_state)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    ds = TensorDataset(
        torch.from_numpy(Xc),
        torch.from_numpy(yc.astype(np.float32)),
    )
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True)

    sigma = get_sigma(epsilon, len(yc))
    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=sigma,
        max_grad_norm=config.MAX_GRAD_NORM,
    )

    # Snapshot global weights (constant) for the proximal term, in the exact
    # parameter order the update loop iterates.
    global_params = [p.detach().clone() for p in model.parameters()]
    criterion = nn.BCELoss()

    for _ in range(config.LOCAL_EPOCHS):
        for xb, yb in loader:
            if xb.numel() == 0:        # poisson sampling can yield empty batches
                continue
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            if mu > 0.0:
                # Opacus private step writes clipped+noised grad into p.grad ...
                if optimizer.pre_step():
                    with torch.no_grad():
                        for p, g in zip(model.parameters(), global_params):
                            if p.grad is not None:
                                p.grad.add_(p.detach() - g, alpha=mu)  # + proximal
                    optimizer.original_optimizer.step()                # Adam applies sum
            else:
                optimizer.step()

    return get_weights(model), len(yc)


def aggregate(states, counts):
    """FedAvg: sample-weighted average of client state_dicts."""
    total = float(sum(counts))
    keys = states[0].keys()
    agg = {}
    for k in keys:
        stacked = torch.stack([
            s[k].float() * (n / total) for s, n in zip(states, counts)
        ], dim=0)
        agg[k] = stacked.sum(dim=0)
    return agg


def evaluate(state, X_test, y_test, in_features):
    model = new_model(in_features)
    set_weights(model, state)
    model.eval()
    with torch.no_grad():
        probs = model(torch.from_numpy(X_test)).numpy()
    preds = (probs >= 0.5).astype(int)
    # AUC/AP are threshold-free utility metrics; defined only if both classes
    # are present in y_test (they are, for our balanced test set).
    try:
        auc = float(roc_auc_score(y_test, probs))
        ap = float(average_precision_score(y_test, probs))
    except ValueError:
        auc = ap = float("nan")
    return {
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "pred_pos_rate": float(preds.mean()),
        "auc": auc,
        "ap": ap,
    }


def run_experiment(epsilon, mu, use_smote, partition_idx,
                   X_train, y_train, X_test, y_test, in_features,
                   rounds=None, local_epochs=None, log=print):
    """Run the full FL pipeline for one (strategy, epsilon) config.

    `mu>0` -> FedProx, `mu==0` -> FedAvg. `use_smote=False` reproduces the
    raw-imbalanced failure baseline. Returns the held-out metrics dict.
    """
    if rounds is not None:
        config.ROUNDS = rounds
    if local_epochs is not None:
        config.LOCAL_EPOCHS = local_epochs

    # Determinism + shared global init across configs.
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # Per-client local data (SMOTETomek is deterministic -> compute once; the
    # paper's "before every round" yields identical data each round).
    client_data = []
    for idx in partition_idx:
        Xc, yc = X_train[idx], y_train[idx]
        if use_smote:
            Xc, yc = smote_resample(Xc, yc)
        client_data.append((Xc, yc))

    global_state = get_weights(new_model(in_features))

    for rnd in range(config.ROUNDS):
        states, counts = [], []
        for (Xc, yc) in client_data:
            st, n = local_train(global_state, Xc, yc, epsilon, mu, in_features)
            states.append(st)
            counts.append(n)
        global_state = aggregate(states, counts)
        if log and (rnd + 1) % 5 == 0:
            m = evaluate(global_state, X_test, y_test, in_features)
            log(f"      round {rnd+1:2d}/{config.ROUNDS}  recall={m['recall']:.3f} "
                f"f1={m['f1']:.3f}")

    return evaluate(global_state, X_test, y_test, in_features)
