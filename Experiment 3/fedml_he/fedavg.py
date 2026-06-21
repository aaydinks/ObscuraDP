"""Real FedAvg training on CIFAR-10 for the accuracy / losslessness check (Exp C).

Three aggregation modes share one training loop:
  * "plaintext" -- mean of the client parameter vectors in the clear.
  * "he"        -- naive HE: every parameter aggregated under CKKS.
  * "spe"       -- only the top-p sensitive parameters under CKKS, rest in clear.

Because CKKS is exact up to ~1e-6 rounding, all three produce numerically the same
aggregated model, so their accuracy curves coincide. `aggregation_agreement()`
measures that per-round equality directly -- a length-independent proof of
losslessness -- while `run_federated()` produces the actual accuracy curves.
"""
import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from . import config, he
from .models import build_model
from .model_utils import get_flat_params, set_flat_params
from .sensitivity import sensitivity_scores, build_mask


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _synthetic_cifar(n_train, n_test):
    """Deterministic CIFAR-shaped 10-class task (per-class prototype + noise).

    Fallback when the 170 MB CIFAR-10 tarball is unavailable. It is genuinely
    learnable (accuracy rises well above chance), which is all Experiment C needs:
    losslessness is a property of the *aggregation*, so the three modes coincide on
    any data. The exact agreement check in aggregation_agreement() does not depend
    on this at all."""
    g = torch.Generator().manual_seed(config.SEED)
    protos = torch.randn(10, 3, 32, 32, generator=g)

    def make(n):
        y = torch.randint(0, 10, (n,), generator=g)
        x = protos[y] + 0.6 * torch.randn(n, 3, 32, 32, generator=g)
        return x, y

    return make(n_train or 5000), make(n_test or 2000)


def get_cifar10(train_subset=None, test_subset=None):
    """Load CIFAR-10 as CPU tensors; fall back to a synthetic task if unavailable.

    Returns (train, test, source) where source is "CIFAR-10" or "synthetic"."""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    # Only touch torchvision when the data is actually present (extracted folder or
    # a complete tarball) -- never trigger its network download, which both fails on
    # the macOS SSL path and would clobber an in-progress manual download.
    extracted = os.path.isdir(os.path.join(config.DATA_DIR, "cifar-10-batches-py"))
    tarball = os.path.join(config.DATA_DIR, "cifar-10-python.tar.gz")
    complete_tar = os.path.exists(tarball) and os.path.getsize(tarball) >= 170_000_000
    try:
        if not (extracted or complete_tar):
            raise FileNotFoundError("CIFAR-10 data not present")
        tr = datasets.CIFAR10(config.DATA_DIR, train=True, download=not extracted, transform=tf)
        te = datasets.CIFAR10(config.DATA_DIR, train=False, download=not extracted, transform=tf)
    except Exception as exc:
        print(f"  [warn] CIFAR-10 unavailable ({type(exc).__name__}); "
              f"using synthetic fallback. Drop cifar-10-python.tar.gz into "
              f"{config.DATA_DIR}/ for the real dataset.")
        return (*_synthetic_cifar(train_subset, test_subset), "synthetic")

    def stack(ds, k):
        idx = np.arange(len(ds))
        if k:
            rng = np.random.RandomState(config.SEED)
            idx = rng.choice(idx, size=k, replace=False)
        xs = torch.stack([ds[i][0] for i in idx])
        ys = torch.tensor([ds[i][1] for i in idx])
        return xs, ys

    return stack(tr, train_subset), stack(te, test_subset), "CIFAR-10"


def iid_partition(n, num_clients, seed):
    rng = np.random.RandomState(seed)
    idx = np.arange(n); rng.shuffle(idx)
    return [np.array(p) for p in np.array_split(idx, num_clients)]


def local_train(model, X, y, dev, epochs, lr, batch):
    """One client's local SGD; returns its flat parameter vector."""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    n = len(y)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, batch):
            bi = perm[s:s + batch]
            xb, yb = X[bi].to(dev), y[bi].to(dev)
            opt.zero_grad(set_to_none=True)
            crit(model(xb), yb).backward()
            opt.step()
    return get_flat_params(model)


@torch.no_grad()
def evaluate(model, X, y, dev, batch=256):
    model.eval()
    correct = 0
    for s in range(0, len(y), batch):
        xb = X[s:s + batch].to(dev)
        pred = model(xb).argmax(1).cpu()
        correct += (pred == y[s:s + batch]).sum().item()
    return correct / len(y)


def _aggregate(mode, client_vecs, mask, ctx):
    if mode == "plaintext":
        return he.fedavg_plaintext(client_vecs)
    if mode == "he":
        full = np.ones(client_vecs[0].size, dtype=bool)
        return he.fedavg_he(client_vecs, full, ctx)
    if mode == "spe":
        return he.fedavg_he(client_vecs, mask, ctx)
    raise ValueError(mode)


def run_federated(mode, model_name, rounds, local_epochs, lr, batch,
                  spe_fraction, train_data, test_data, log=print):
    """Train `model_name` on CIFAR-10 under aggregation `mode`; return acc curve."""
    torch.manual_seed(config.SEED); np.random.seed(config.SEED)
    dev = device()
    (Xtr, ytr), (Xte, yte) = train_data, test_data
    parts = iid_partition(len(ytr), config.NUM_CLIENTS, config.SEED)

    model = build_model(model_name).to(dev)
    ctx = he.make_context() if mode in ("he", "spe") else None

    # SPE mask from pre-training gradient sensitivity on one client's batch.
    mask = None
    if mode == "spe":
        scratch = build_model(model_name).to(dev)
        set_flat_params(scratch, get_flat_params(model))
        b = parts[0][:min(len(parts[0]), 256)]
        scores = sensitivity_scores(scratch, (Xtr[b].to(dev), ytr[b].to(dev)))
        mask = build_mask(scratch, scores, spe_fraction)

    # Re-seed right before training so the client-update RNG (batch shuffling) is
    # IDENTICAL across all three modes -- the SPE sensitivity pass above consumes
    # the global RNG, so without this the modes would shuffle differently and the
    # curves would diverge for reasons unrelated to the aggregation. With it, the
    # ONLY difference between modes is the aggregation arithmetic, which is exact
    # to ~1e-7 (see aggregation_agreement); residual drift is pure FP chaos.
    torch.manual_seed(config.SEED + 1)
    global_vec = get_flat_params(model)
    curve = []
    for r in range(rounds):
        client_vecs = []
        for idx in parts:
            set_flat_params(model, global_vec)
            client_vecs.append(
                local_train(model, Xtr[idx], ytr[idx], dev, local_epochs, lr, batch))
        global_vec = _aggregate(mode, client_vecs, mask, ctx)
        set_flat_params(model, global_vec)
        acc = evaluate(model, Xte, yte, dev)
        curve.append(acc)
        if log:
            log(f"    [{mode:9s}] round {r+1:2d}/{rounds}  acc={acc:.4f}")
    return curve


def aggregation_agreement(model_name, spe_fraction, train_data):
    """Exact per-round losslessness check: aggregate identical client updates via
    plaintext / naive-HE / SPE and report the max abs deviation from plaintext."""
    torch.manual_seed(config.SEED); np.random.seed(config.SEED)
    dev = device()
    (Xtr, ytr), _ = train_data, None
    parts = iid_partition(len(ytr), config.NUM_CLIENTS, config.SEED)
    model = build_model(model_name).to(dev)
    ctx = he.make_context()

    g = get_flat_params(model)
    client_vecs = []
    for idx in parts:
        set_flat_params(model, g)
        client_vecs.append(local_train(model, Xtr[idx], ytr[idx], dev, 1,
                                       config.ACC_LR, config.ACC_BATCH_SIZE))
    scores = sensitivity_scores(build_model(model_name).to(dev),
                                (Xtr[parts[0][:256]].to(dev), ytr[parts[0][:256]].to(dev)))
    mask = build_mask(model, scores, spe_fraction)

    plain = he.fedavg_plaintext(client_vecs)
    full = np.ones(plain.size, dtype=bool)
    he_agg = he.fedavg_he(client_vecs, full, ctx)
    spe_agg = he.fedavg_he(client_vecs, mask, ctx)
    return {"he_vs_plain_max_abs": float(np.max(np.abs(he_agg - plain))),
            "spe_vs_plain_max_abs": float(np.max(np.abs(spe_agg - plain))),
            "encrypted_fraction": float(mask.mean())}
