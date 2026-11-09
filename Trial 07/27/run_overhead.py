"""Experiment E -- crypto overhead composition: HE vs p, vs model size, and DP tax.

Tests the hypothesis that DP and HE costs compose additively by measuring them
independently:
  * crypto_vs_p: How HE overhead scales with the encryption ratio p (SPE).
  * crypto_vs_size: How HE overhead scales with total parameter count (O(n)).
  * dp_tax: How DP-SGD per-sample gradient processing multiplies per-epoch time.

crypto_vs_p is measured at config.OVERHEAD_P_WIDTH_SCALE, NOT at the deployed
stroke width. This matters. A CKKS ciphertext costs the same ~235 KB whether 1 or
all 4096 of its slots carry data, so on the 3137-parameter stroke model every
p >= 0.1 needs exactly one chunk: payload_ratio is pinned at 18.76 and crypto_sec
is flat timing noise (0.047 / 0.026 / 0.025 / 0.027 / 0.038 s at p = 0.1 ... 1.0).
Published that way the sweep reads as "SPE saves nothing", which is a statement
about the model being smaller than a single ciphertext, not about SPE. At
width_scale 16 (541,697 params) the same sweep spans 14 chunks at p=0.1 to 133 at
p=1.0 and both cost axes move linearly with p. The width-1 sweep is retained as an
explicitly labelled degenerate control, because it is the clearest illustration of
the padding floor that also explains the inflated small-model payload ratios.

Width scaling sweeps the full model family to expose O(n) communication overhead.
DP tax is measured per width scale to capture the model-size-dependent DP cost.

Emits results/overhead.json with three independent sweeps and metadata about
the CKKS context, seed, and client count used. Every crypto record carries
width_scale, n_params, n_chunks, the measured payload ratio, the analytic
prediction from the padding model, and the true wire ratio, so a plot can
separate the regimes and a reader can check the arithmetic.
"""
import json
import os
import time

import numpy as np

from obscura import config, he, bench, dp, models


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Build CKKS context once; reuse across all benchmarks.
    print("Building CKKS context...", flush=True)
    ctx = he.make_context()

    # ========================================================================
    # crypto_vs_p: How overhead scales with encryption fraction on stroke baseline.
    # ========================================================================
    print("\n--- Benchmark 1: crypto vs encryption fraction ---", flush=True)
    in_features = 15  # stroke
    dataset = "stroke"

    bytes_per_ct = bench.ciphertext_quantum(ctx)
    print(f"  padding quantum: one full {config.CKKS_SLOTS}-slot ciphertext "
          f"serialises to {bytes_per_ct:,} B, and a 1-slot one costs the same",
          flush=True)

    # The sweep runs at a chunked width plus the deployed width as a control, so
    # the degenerate regime is visible next to the informative one rather than
    # standing in for it.
    p_widths = [
        (config.OVERHEAD_P_WIDTH_SCALE, "chunked"),
        (config.OVERHEAD_P_CONTROL_WIDTH_SCALE, "degenerate_single_chunk"),
    ]

    crypto_vs_p = []
    for width_scale, regime in p_widths:
        model = models.new_model(in_features, dataset=dataset, width_scale=width_scale)
        n_params = models.count_params(model)
        chunks_at_full = -(-n_params // config.CKKS_SLOTS)
        if regime == "chunked" and chunks_at_full < 4:
            print(f"  WARNING: width_scale={width_scale} is only {chunks_at_full} "
                  f"chunk(s) at p=1.0 -- the p sweep will be degenerate", flush=True)
        print(f"  width_scale={width_scale} [{regime}]: {n_params:,} params, "
              f"{chunks_at_full} chunks at p=1.0", flush=True)

        plain_sec = bench.bench_plaintext(n_params)["plain_sec"]

        for fraction in config.OVERHEAD_FRACTIONS:
            print(f"    fraction={fraction:.2f} "
                  f"({int(round(fraction*n_params)):,} params)...",
                  end=" ", flush=True)
            t0 = time.perf_counter()
            enc_result = bench.bench_encryption(n_params, ctx, fraction=fraction)
            elapsed = time.perf_counter() - t0
            print(f"[{elapsed:.1f}s] {enc_result['n_chunks']} chunks, "
                  f"payload {enc_result['payload_ratio']:.3f}x, "
                  f"wire {enc_result['wire_payload_ratio']:.3f}x", flush=True)

            record = enc_result.copy()
            record["width_scale"] = width_scale
            record["regime"] = regime
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
        model = models.new_model(in_features, dataset=dataset, width_scale=scale)
        n_params_scaled = models.count_params(model)
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
        model = models.new_model(in_features, dataset=dataset, width_scale=scale)
        n_params_scaled = models.count_params(model)
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
        model = models.new_model(in_features, dataset=dataset, width_scale=scale)
        n_params_scaled = models.count_params(model)
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

    # The padding floor, tabulated analytically so nobody has to reverse-engineer
    # why the same CKKS configuration reports 63x, 18.8x and 14.4x. Costs nothing
    # to compute -- it is closed form, not a benchmark.
    asymptote = bytes_per_ct / (4.0 * config.CKKS_SLOTS)
    padding_model = []
    reference_sizes = [
        (929, "loan model (width 1)"),
        (3137, "stroke model (width 1)"),
    ] + [(models.count_params(
        models.new_model(in_features, dataset=dataset, width_scale=s)),
        "stroke width %d" % s) for s in config.WIDTH_SCALES if s > 1]
    for n, label in reference_sizes:
        padding_model.append({
            "n_params": int(n),
            "label": label,
            "n_chunks": int(-(-n // config.CKKS_SLOTS)),
            "predicted_payload_ratio": bench.payload_ratio_model(
                n, n, bytes_per_ct),
            "slot_utilisation": float(n) / (config.CKKS_SLOTS
                                            * (-(-n // config.CKKS_SLOTS))),
        })

    print("\n--- The padding floor (analytic) ---", flush=True)
    print("  payload_ratio(n) = ceil(n/slots) * bytes_per_ciphertext / (4n)",
          flush=True)
    print(f"  slots={config.CKKS_SLOTS}, bytes_per_ciphertext={bytes_per_ct:,}, "
          f"asymptote={asymptote:.3f}x", flush=True)
    print("  | model | n_params | chunks | slot use | predicted ratio |", flush=True)
    print("  |---|---:|---:|---:|---:|", flush=True)
    for row in padding_model:
        print(f"  | {row['label']} | {row['n_params']:,} | {row['n_chunks']} | "
              f"{row['slot_utilisation']*100:.1f}% | "
              f"{row['predicted_payload_ratio']:.3f}x |", flush=True)
    print("  The whole spread is zero-padding of the final chunk: at 3137 params "
          "one\n  ciphertext carries 4096 slots, and 4096/3137 = 1.306 is exactly "
          "18.758/14.365.", flush=True)

    print("\n--- crypto_vs_p: measured vs predicted ---", flush=True)
    print("  | width | n_params | p | chunks | measured | predicted | wire | crypto_sec |",
          flush=True)
    print("  |---|---:|---:|---:|---:|---:|---:|---:|", flush=True)
    for r in crypto_vs_p:
        print(f"  | {r['width_scale']} | {r['n_params']:,} | {r['fraction']:.2f} | "
              f"{r['n_chunks']} | {r['payload_ratio']:.3f} | "
              f"{r['predicted_payload_ratio']:.3f} | "
              f"{r['wire_payload_ratio']:.3f} | {r['crypto_sec']:.4f} |", flush=True)

    meta = {
        "trial": "07/27",
        "seed": config.SEED,
        "num_clients": config.NUM_CLIENTS,
        "ckks_poly_modulus_degree": config.CKKS_POLY_MODULUS_DEGREE,
        "ckks_coeff_mod_bit_sizes": config.CKKS_COEFF_MOD_BIT_SIZES,
        "ckks_global_scale": config.CKKS_GLOBAL_SCALE,
        "ckks_slots": config.CKKS_SLOTS,
        "bytes_per_ciphertext": float(bytes_per_ct),
        "payload_ratio_asymptote": float(asymptote),
        "overhead_p_width_scale": config.OVERHEAD_P_WIDTH_SCALE,
        "overhead_p_control_width_scale": config.OVERHEAD_P_CONTROL_WIDTH_SCALE,
        "generated_by": "run_overhead.py",
    }

    results = {
        "meta": meta,
        "crypto_vs_p": crypto_vs_p,
        "crypto_vs_size": crypto_vs_size,
        "padding_model": padding_model,
        "dp_tax": dp_tax,
    }

    output_path = os.path.join(config.RESULTS_DIR, "overhead.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved -> {output_path}", flush=True)

    # Print summary table.
    print("\n--- Summary ---", flush=True)
    print(f"crypto_vs_p: {len(crypto_vs_p)} records "
          f"({len(p_widths)} widths x {len(config.OVERHEAD_FRACTIONS)} fractions)",
          flush=True)
    print(f"crypto_vs_size: {len(crypto_vs_size)} records (2 passes x {len(config.WIDTH_SCALES)} scales)", flush=True)
    print(f"dp_tax: {len(dp_tax)} records", flush=True)


if __name__ == "__main__":
    main()
