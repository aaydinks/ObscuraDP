"""One client's local round under a FedAvg or FedProx objective, plain or DP-SGD.

This is the single local-training entry point for all five privacy stacks in the
trial (plain / dp / he / dp_he / dp_spe). Encryption happens at aggregation time,
so the HE stacks reuse the plaintext path here unchanged -- that is exactly why
the arms are comparable.

The FedProx proximal term mu*(w - w_global) is injected differently depending on
the privacy regime:

  * Plain (dp=False): added to p.grad after loss.backward() and before
    optimizer.step(), a standard proximal-gradient technique.
  * DP (dp=True): Opacus OVERWRITES p.grad with the clipped+noised per-sample
    gradient during its private step, so a proximal term merely added to the loss
    would be silently discarded. We therefore run optimizer.pre_step() to invoke
    Opacus's private logic, then add mu*(w - w_global) to p.grad, then let Adam
    apply the sum via optimizer.original_optimizer.step(). The proximal term is
    data-independent (depends only on public global weights), so adding it
    post-clipping does not affect the privacy guarantee.

When mu == 0, both paths reduce to plain FedAvg local training.
"""
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import config, models
from . import dp as dp_mod

warnings.filterwarnings("ignore", category=UserWarning)


def local_train(global_state, Xc, yc, in_features, dataset="stroke", dp=False,
                sigma=None, mu=0.0, width_scale=1, lr=None, local_epochs=None,
                batch_size=None, seed=None):
    """One client's local training round, plain or DP-SGD, with optional FedProx.

    Args:
        global_state: dict of global weights (state_dict-like).
        Xc: client features (float32, shape (n_local, in_features)).
        yc: client labels (int64 or float32, shape (n_local,)).
        in_features: number of input features.
        dataset: "stroke" or "loan" (default "stroke").
        dp: if True, use Opacus DP-SGD; if False, plain SGD.
        sigma: noise multiplier (required if dp=True, ignored if dp=False).
        mu: FedProx coefficient (0.0 -> FedAvg, > 0 -> FedProx).
        width_scale: hidden layer width multiplier (default 1).
        lr: learning rate (default config.LR[dataset]).
        local_epochs: number of local epochs (default config.LOCAL_EPOCHS).
        batch_size: batch size (default config.BATCH_SIZE).
        seed: if not None, torch.manual_seed(seed) before DataLoader.

    Returns:
        (state, n_samples, stats) where:
          state: dict of local weights after training.
          n_samples: number of local samples (len(yc)).
          stats: dict with keys {"train_sec", "n_batches", "n_samples"}.

    Raises:
        ValueError: if dp=True and sigma is None.
    """
    if dp and sigma is None:
        raise ValueError("sigma is required when dp=True")

    lr = lr or config.LR[dataset]
    local_epochs = local_epochs or config.LOCAL_EPOCHS
    batch_size = batch_size or config.BATCH_SIZE

    # Build and initialize model
    model = models.new_model(in_features, dataset, width_scale)
    models.set_weights(model, global_state)
    model.train()

    # Create data loader
    if seed is not None:
        torch.manual_seed(seed)
    ds = TensorDataset(
        torch.from_numpy(Xc),
        torch.from_numpy(yc.astype(np.float32)),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    start_time = time.time()
    n_batches = 0

    if not dp:
        # Plain SGD path
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
                        for p, g in zip(model.parameters(), global_state.values()):
                            if p.grad is not None:
                                p.grad.add_(p.detach() - g, alpha=mu)

                optimizer.step()
    else:
        # DP-SGD path via Opacus
        model, optimizer, loader, _ = dp_mod.make_private(model, optimizer, loader, sigma)

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
                            for p, g in zip(model.parameters(), global_params):
                                if p.grad is not None:
                                    p.grad.add_(p.detach() - g, alpha=mu)
                        optimizer.original_optimizer.step()
                else:
                    optimizer.step()

    elapsed = time.time() - start_time

    state = models.get_weights(model)
    stats = {
        "train_sec": elapsed,
        "n_batches": n_batches,
        "n_samples": len(yc),
    }

    return state, len(yc), stats
