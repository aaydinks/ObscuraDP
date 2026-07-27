"""Experiment E -- crypto overhead composition: HE vs p, vs model size, and DP tax.

Tests the hypothesis that DP and HE costs compose additively by measuring them
independently:
  * crypto_vs_p: How HE overhead scales with the encryption ratio p (SPE).
  * crypto_vs_size: How HE overhead scales with total parameter count (O(n)).
  * dp_tax: How DP-SGD per-sample gradient processing multiplies per-epoch time.

The stroke model at its stroke baseline size (width_scale=1, 15 input features)
is used to measure crypto_vs_p, isolating the effect of the encryption fraction.
Width scaling sweeps the full model family to expose O(n) communication overhead.
DP tax is measured per width scale to capture the model-size-dependent DP cost.

Emits results/overhead.json with three independent sweeps and metadata about
the CKKS context, seed, and client count used.
"""
import json
import os
import time

import numpy as np

from obscura import config, he, bench, dp


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Build CKKS context once; reuse across all benchmarks.
    print("Building CKKS context...", flush=True)
    ctx = he.make_context()

    # ========================================================================
    # crypto_vs_p: How overhead scales with encryption fraction on stroke baseline.
    # ========================================================================
    print("\n--- Benchmark 1: crypto vs encryption fraction (stroke baseline) ---",
          flush=True)
    in_features = 15  # stroke
    dataset = "stroke"
    width_scale = 1

    # Get plaintext baseline (one time).
    model = __import__('obscura.models', fromlist=['new_model']).new_model(
        in_features, dataset=dataset, width_scale=width_scale
    )
    n_params = __import__('obscura.models', fromlist=['count_params']).count_params(model)
    plain_result = bench.bench_plaintext(n_params)
    plain_sec = plain_result["plain_sec"]

    crypto_vs_p = []
    for fraction in config.OVERHEAD_FRACTIONS:
        print(f"  fraction={fraction:.2f} ({int(round(fraction*n_params))} params)...",
              end=" ", flush=True)
        t0 = time.perf_counter()
        enc_result = bench.bench_encryption(n_params, ctx, fraction=fraction)
        elapsed = time.perf_counter() - t0
        print(f"[{elapsed:.1f}s]", flush=True)

        record = enc_result.copy()
        record["plain_sec"] = plain_sec
        record["slowdown"] = enc_result["crypto_sec"] / max(plain_sec, 1e-9)
        crypto_vs_p.append(record)

    # ========================================================================
    # crypto_vs_size: How overhead scales with parameter count (full naive HE and SPE).
    # ========================================================================
    print("\n--- Benchmark 2: crypto vs model size ---", flush=True)
    crypto_vs_size = []

    # First pass: fraction=1.0 (naive full HE).
    print("  Pass 1 (naive HE, fraction=1.0):", flush=True)
    for scale in config.WIDTH_SCALES:
        n_enc = 0  # placeholder; will be computed
        model = __import__('obscura.models', fromlist=['new_model']).new_model(
            in_features, dataset=dataset, width_scale=scale
        )
        n_params_scaled = __import__('obscura.models', fromlist=['count_params']).count_params(model)
        n_encrypted = int(round(1.0 * n_params_scaled))
        n_chunks = (n_encrypted + config.CKKS_SLOTS - 1) // config.CKKS_SLOTS if n_encrypted > 0 else 0

        # Guard the ~1M-parameter case.
        if scale == 32:
            print(f"    width_scale={scale}: {n_params_scaled:,} params -> {n_chunks} chunks (estimated)", flush=True)

        print(f"    width_scale={scale}: {n_params_scaled:,} params...", end=" ", flush=True)
        t0 = time.perf_counter()
        results = bench.bench_model_sizes([scale], dataset, in_features, ctx, fraction=1.0)
        elapsed = time.perf_counter() - t0
        print(f"[{elapsed:.1f}s]", flush=True)

        for r in results:
            r["fraction"] = 1.0
            crypto_vs_size.append(r)

    # Second pass: fraction=config.SPE_DEFAULT (SPE baseline).
    print(f"  Pass 2 (SPE, fraction={config.SPE_DEFAULT}):", flush=True)
    for scale in config.WIDTH_SCALES:
        model = __import__('obscura.models', fromlist=['new_model']).new_model(
            in_features, dataset=dataset, width_scale=scale
        )
        n_params_scaled = __import__('obscura.models', fromlist=['count_params']).count_params(model)
        print(f"    width_scale={scale}: {n_params_scaled:,} params...", end=" ", flush=True)
        t0 = time.perf_counter()
        results = bench.bench_model_sizes([scale], dataset, in_features, ctx,
                                         fraction=config.SPE_DEFAULT)
        elapsed = time.perf_counter() - t0
        print(f"[{elapsed:.1f}s]", flush=True)

        for r in results:
            r["fraction"] = config.SPE_DEFAULT
            crypto_vs_size.append(r)

    # ========================================================================
    # dp_tax: How DP-SGD per-sample gradient cost scales with model size.
    # ========================================================================
    print("\n--- Benchmark 3: DP tax (per-epoch training time) ---", flush=True)
    n_samples = 800  # synthetic training set size
    sigma = dp.get_sigma(config.EPSILON_DEFAULT, n_samples)

    dp_tax = []
    for scale in config.WIDTH_SCALES:
        model = __import__('obscura.models', fromlist=['new_model']).new_model(
            in_features, dataset=dataset, width_scale=scale
        )
        n_params_scaled = __import__('obscura.models', fromlist=['count_params']).count_params(model)
        print(f"  width_scale={scale}: {n_params_scaled:,} params (sigma={sigma:.4f})...",
              end=" ", flush=True)
        t0 = time.perf_counter()
        tax_result = bench.bench_dp_tax(dataset, in_features, scale, sigma, n_samples)
        elapsed = time.perf_counter() - t0
        print(f"[{elapsed:.1f}s] multiplier={tax_result['dp_multiplier']:.2f}x", flush=True)
        dp_tax.append(tax_result)

    # ========================================================================
    # Assemble and write results.
    # ========================================================================
    print("\n--- Writing results ---", flush=True)

    meta = {
        "trial": "07/27",
        "seed": config.SEED,
        "num_clients": config.NUM_CLIENTS,
        "ckks_poly_modulus_degree": config.CKKS_POLY_MODULUS_DEGREE,
        "ckks_coeff_mod_bit_sizes": config.CKKS_COEFF_MOD_BIT_SIZES,
        "ckks_global_scale": config.CKKS_GLOBAL_SCALE,
        "ckks_slots": config.CKKS_SLOTS,
        "generated_by": "run_overhead.py",
    }

    results = {
        "meta": meta,
        "crypto_vs_p": crypto_vs_p,
        "crypto_vs_size": crypto_vs_size,
        "dp_tax": dp_tax,
    }

    output_path = os.path.join(config.RESULTS_DIR, "overhead.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved -> {output_path}", flush=True)

    # Print summary table.
    print("\n--- Summary ---", flush=True)
    print(f"crypto_vs_p: {len(crypto_vs_p)} records", flush=True)
    print(f"crypto_vs_size: {len(crypto_vs_size)} records (2 passes x {len(config.WIDTH_SCALES)} scales)", flush=True)
    print(f"dp_tax: {len(dp_tax)} records", flush=True)


if __name__ == "__main__":
    main()
