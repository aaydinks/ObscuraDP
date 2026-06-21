"""Experiment A -- naive HE vs plaintext FedAvg overhead across model sizes.

For every model: flatten its parameters, then benchmark plaintext aggregation and
full CKKS encryption (per-client encrypt + server homomorphic-sum/decrypt + payload
size). Emits results/overhead.json with the per-model timings and ratios used to
draw the O(n) scaling curve and confirm the ~15x payload blow-up.
"""
import json
import os
import time

import numpy as np
import torch

from fedml_he import config, he, bench
from fedml_he.models import build_model, num_params
from fedml_he.model_utils import flatten_params


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    torch.manual_seed(config.SEED)
    ctx = he.make_context()
    N = config.NUM_CLIENTS

    models = list(config.OVERHEAD_MODELS)
    if config.RUN_VIT:
        models.append("ViT-B/16")

    rows = []
    for name in models:
        model = build_model(name)
        n = num_params(model)
        flat, _ = flatten_params(model)
        del model

        t0 = time.perf_counter()
        pt = bench.benchmark_plaintext(flat, N)
        hb = bench.benchmark_he(ctx, flat, N)
        wall = time.perf_counter() - t0

        plain_ms = pt["prep_ms"] + pt["agg_ms"]
        he_ms = hb["enc_ms"] + hb["agg_ms"]
        row = {
            "model": name, "n_params": n,
            "plain_prep_ms": pt["prep_ms"], "plain_agg_ms": pt["agg_ms"],
            "plain_ms": plain_ms, "plain_payload_bytes": pt["payload_bytes"],
            "he_enc_ms": hb["enc_ms"], "he_agg_ms": hb["agg_ms"], "he_ms": he_ms,
            "he_ciphertext_bytes": hb["ciphertext_bytes"],
            "time_ratio": he_ms / max(plain_ms, 1e-9),
            "payload_ratio": hb["ciphertext_bytes"] / max(pt["payload_bytes"], 1),
            "n_chunks": int(np.ceil(n / config.CKKS_SLOTS)),
        }
        rows.append(row)
        print(f"{name:10s} {n:>11,}p  "
              f"HE enc {hb['enc_ms']/1e3:6.2f}s agg {hb['agg_ms']/1e3:6.2f}s  "
              f"payload {hb['ciphertext_bytes']/1e6:7.1f} MB "
              f"({row['payload_ratio']:.1f}x)  "
              f"time {row['time_ratio']:.0f}x  [{wall:.1f}s]", flush=True)

    out = {"config": {"ckks": config.CKKS_COEFF_MOD_BIT_SIZES,
                      "slots": config.CKKS_SLOTS, "clients": N},
           "rows": rows}
    with open(os.path.join(config.RESULTS_DIR, "overhead.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {config.RESULTS_DIR}/overhead.json")


if __name__ == "__main__":
    main()
