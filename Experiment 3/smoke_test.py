"""Fast sanity check for the FedML-HE package (no CIFAR download, ~seconds).

Verifies: model build + param count, flatten/unflatten round-trip, sensitivity
scoring + mask fraction, and -- critically -- that naive-HE and SPE aggregation
reproduce the plaintext FedAvg mean to CKKS rounding (~1e-6).
"""
import numpy as np
import torch

from fedml_he import he, bench, config
from fedml_he.models import build_model, num_params
from fedml_he.model_utils import flatten_params, set_flat_params, get_flat_params
from fedml_he.sensitivity import sensitivity_scores, build_mask, random_batch

torch.manual_seed(0)

m = build_model("SmallCNN")
n = num_params(m)
print(f"SmallCNN params: {n:,}")

# flatten / unflatten round-trip
flat, template = flatten_params(m)
assert flat.size == n
m2 = build_model("SmallCNN"); set_flat_params(m2, flat)
assert np.max(np.abs(get_flat_params(m2) - flat)) < 1e-5, "flatten round-trip failed"
print("flatten/unflatten OK")

# sensitivity + mask
scores = sensitivity_scores(build_model("SmallCNN"), random_batch("SmallCNN"))
for frac in (0.1, 0.25, 0.5, 1.0):
    mask = build_mask(m, scores, frac)
    assert abs(mask.mean() - frac) < 0.02, (frac, mask.mean())
print("sensitivity + mask fractions OK")

# CKKS aggregation correctness: naive + SPE vs plaintext mean
ctx = he.make_context()
N = config.NUM_CLIENTS
clients = [flat + 0.01 * np.random.randn(n) for _ in range(N)]
plain = he.fedavg_plaintext(clients)
he_full = he.fedavg_he(clients, np.ones(n, bool), ctx)
mask = build_mask(m, scores, 0.25)
spe = he.fedavg_he(clients, mask, ctx)
print(f"naive-HE vs plaintext max abs err: {np.max(np.abs(he_full-plain)):.2e}")
print(f"SPE(0.25) vs plaintext max abs err: {np.max(np.abs(spe-plain)):.2e}")
assert np.max(np.abs(he_full - plain)) < 1e-4
assert np.max(np.abs(spe - plain)) < 1e-4

# benchmark harness returns sane numbers
pt = bench.benchmark_plaintext(flat, N)
hb = bench.benchmark_he(ctx, flat, N)
ratio = hb["ciphertext_bytes"] / pt["payload_bytes"]
print(f"payload: plaintext {pt['payload_bytes']/1e3:.1f} KB | "
      f"HE {hb['ciphertext_bytes']/1e3:.1f} KB | {ratio:.1f}x")
print("SMOKE OK")
