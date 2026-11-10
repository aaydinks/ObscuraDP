"""One participant's local round exactly as in Trial 07/27 (comparison only).

Known defects reproduced on purpose: ``lr``, ``local_epochs`` and ``batch_size`` fall back to
the defaults when given as 0 (finding G18), and an empty sampled batch is skipped where a
noise-only step should be taken (finding G27).

Private training needs Opacus (the ``legacy`` extra).
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from obscuradp.legacy import dp as dp_mod
from obscuradp.legacy.config import LegacyTrialConfig
from obscuradp.legacy.models import new_model
from obscuradp.state import State, get_state, set_state

Array = np.ndarray[Any, Any]


def local_train(
    cfg: LegacyTrialConfig,
    global_state: Mapping[str, torch.Tensor],
    Xc: Array,
    yc: Array,
    in_features: int,
    dataset: str = "stroke",
    dp: bool = False,
    sigma: float | None = None,
    mu: float = 0.0,
    width_scale: int = 1,
    lr: float | None = None,
    local_epochs: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
) -> tuple[State, int, dict[str, Any]]:
    """Train one participant for one round, plain or with DP-SGD, with an optional FedProx term.

    Returns ``(state, number of local rows, stats)``.
    """
    if dp and sigma is None:
        raise ValueError("sigma is required when dp=True")

    lr = lr or cfg.lr[dataset]
    local_epochs = local_epochs or cfg.local_epochs
    batch_size = batch_size or cfg.batch_size

    # Build and initialize model
    model: Any = new_model(cfg, in_features, dataset, width_scale)
    set_state(model, global_state)
    model.train()

    # Create data loader
    if seed is not None:
        torch.manual_seed(seed)
    ds = TensorDataset(
        torch.from_numpy(Xc),
        torch.from_numpy(yc.astype(np.float32)),
    )
    loader: Any = DataLoader(ds, batch_size=batch_size, shuffle=True)

    optimizer: Any = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    start_time = time.time()
    n_batches = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        if not dp:
            for _ in range(local_epochs):
                for xb, yb in loader:
                    if xb.numel() == 0:
                        continue
                    n_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    out = model(xb)
                    loss = criterion(out, yb)
                    loss.backward()

                    if mu > 0.0:
                        # Add proximal gradient before optimizer.step()
                        with torch.no_grad():
                            for p, g in zip(
                                model.parameters(), global_state.values(), strict=False
                            ):
                                if p.grad is not None:
                                    p.grad.add_(p.detach() - g, alpha=mu)

                    optimizer.step()
        else:
            assert sigma is not None
            model, optimizer, loader, _ = dp_mod.make_private(cfg, model, optimizer, loader, sigma)

            # Snapshot global weights for the proximal term, in exact parameter order
            global_params = [p.detach().clone() for p in model.parameters()]

            for _ in range(local_epochs):
                for xb, yb in loader:
                    if xb.numel() == 0:  # Poisson sampling can yield empty batches
                        continue
                    n_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    out = model(xb)
                    loss = criterion(out, yb)
                    loss.backward()

                    if mu > 0.0:
                        # Inject proximal term after Opacus's private step
                        if optimizer.pre_step():
                            with torch.no_grad():
                                for p, g in zip(model.parameters(), global_params, strict=False):
                                    if p.grad is not None:
                                        p.grad.add_(p.detach() - g, alpha=mu)
                            optimizer.original_optimizer.step()
                    else:
                        optimizer.step()

    elapsed = time.time() - start_time

    state = get_state(model)
    stats = {
        "train_sec": elapsed,
        "n_batches": n_batches,
        "n_samples": len(yc),
    }

    return state, len(yc), stats
