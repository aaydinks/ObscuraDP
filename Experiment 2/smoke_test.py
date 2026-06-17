"""Fast end-to-end sanity check (1 round) for all three FL regimes + CKKS exactness.

Verifies the data pipeline, that DP/HE training run, and -- critically -- that the
CKKS homomorphic mean matches the plaintext mean to ~1e-4. Run before the full
sweep: `python smoke_test.py`.
"""
import numpy as np

from credit_fl import config, data, he
from credit_fl import train as T
from credit_fl.model import (new_model, get_weights, flatten_state,
                             unflatten_state)

config.ROUNDS = 1
config.LOCAL_EPOCHS = 1

X_tr, y_tr, X_te, y_te, info = data.split_and_scale()
nf = info["n_features"]
parts, pinfo = data.iid_partition(len(y_tr), config.NUM_CLIENTS, config.SEED)
print("features:", nf, "| client sizes:", pinfo["sizes"],
      "| train pos-rate:", round(info["train_pos_rate"], 3))

g = get_weights(new_model(nf))

# plaintext FedAvg
states = [T.local_train_plain(g, X_tr[i], y_tr[i], nf)[0] for i in parts]
plain_mean = T.aggregate(states)
print("plain FedAvg ok, eval:", T.evaluate(plain_mean, X_te, y_te, nf))

# DP one round
s = T.get_sigma(8.65, len(parts[0]))
print("sigma(eps=8.65):", round(s, 3), "| actual eps:",
      round(T.actual_epsilon(s, len(parts[0])), 2))
dp_states = [T.local_train_dp(g, X_tr[i], y_tr[i], s, nf)[0] for i in parts]
print("DP FedAvg ok, eval:", T.evaluate(T.aggregate(dp_states), X_te, y_te, nf))

# HE: encrypted mean vs plaintext mean
ctx = he.make_context()
flats = [flatten_state(st) for st in states]
enc = [he.encrypt_vector(ctx, f)[0] for f in flats]
agg_ct, _ = he.aggregate_encrypted(enc)
he_flat = he.decrypt_vector(agg_ct, flats[0].size)
plain_flat = flatten_state(plain_mean)
max_err = float(np.max(np.abs(he_flat - plain_flat)))
print(f"CKKS vs plaintext max abs err: {max_err:.2e}  "
      f"(ciphertext {he.ciphertext_bytes(enc[0])/1e6:.3f} MB)")
assert max_err < 1e-3, "CKKS aggregation diverged from plaintext!"
print("SMOKE OK")
