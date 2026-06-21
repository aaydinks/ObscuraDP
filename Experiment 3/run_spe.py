"""Experiment B -- Selective Parameter Encryption overhead vs encryption ratio p.

For each model and each p in SPE_FRACTIONS, encrypt only the top-p sensitive
fraction of parameters (rest aggregated in plaintext) and measure the crypto time
and payload. Overhead should fall ~linearly with p, so SPE at p=0.25 keeps ~1/4 of
the naive cost. Emits results/spe.json.

The sensitivity mask selects *which* parameters are encrypted; the overhead depends
only on *how many* (p * n), so the harness encrypts a p-fraction-sized subset.
"""
import json
import os

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

    results = {"fractions": config.SPE_FRACTIONS, "models": {}}
    for name in config.SPE_MODELS:
        model = build_model(name)
        n = num_params(model)
        flat, _ = flatten_params(model)
        del model
        # SPE encrypts the top-p sensitive parameters; the *overhead* depends only
        # on how many are encrypted (p*n), not which, so the harness measures a
        # p-fraction-sized subset. (Sensitivity selection is exercised in Exp C.)

        plain = bench.benchmark_plaintext(flat, N)
        plain_ms = plain["prep_ms"] + plain["agg_ms"]
        per_p = []
        for p in config.SPE_FRACTIONS:
            n_enc = int(round(p * n))
            hb = bench.benchmark_he(ctx, flat, N, n_encrypt=n_enc)
            he_ms = hb["enc_ms"] + hb["agg_ms"]
            per_p.append({
                "p": p, "n_encrypt": n_enc,
                "he_ms": he_ms, "payload_bytes": hb["total_payload_bytes"],
                "ciphertext_bytes": hb["ciphertext_bytes"],
                "time_ratio_vs_plain": he_ms / max(plain_ms, 1e-9),
                "payload_ratio_vs_plain": hb["total_payload_bytes"] / max(plain["payload_bytes"], 1),
            })
            print(f"{name:10s} p={p:<4}  enc {hb['enc_ms']/1e3:6.2f}s  "
                  f"payload {hb['total_payload_bytes']/1e6:7.1f} MB  "
                  f"time {per_p[-1]['time_ratio_vs_plain']:6.0f}x  "
                  f"payload {per_p[-1]['payload_ratio_vs_plain']:.1f}x", flush=True)
        # overhead relative to naive (p=1.0)
        naive = per_p[-1]
        for r in per_p:
            r["he_ms_vs_naive"] = r["he_ms"] / max(naive["he_ms"], 1e-9)
            r["payload_vs_naive"] = r["payload_bytes"] / max(naive["payload_bytes"], 1)
        results["models"][name] = {"n_params": n, "plain_ms": plain_ms,
                                   "plain_payload_bytes": plain["payload_bytes"],
                                   "sweep": per_p}

    with open(os.path.join(config.RESULTS_DIR, "spe.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {config.RESULTS_DIR}/spe.json")


if __name__ == "__main__":
    main()
