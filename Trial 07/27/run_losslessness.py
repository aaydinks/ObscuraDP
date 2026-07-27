"""Experiment C: losslessness of HE and SPE when the updates are DP-noised.

Experiment 3 proved CKKS aggregation is lossless (~1e-7 rounding error) on CLEAN
client updates. This gap asks: does losslessness survive when the updates are
DP-noised? If yes, HE's arithmetic error is orders of magnitude below the noise
DP injects, so composing HE on top of DP has no utility cost.

Procedure: for each epsilon, generate ONE set of DP-noised updates from a shared
global init, then aggregate those SAME updates in plaintext, under full HE, and
under SPE at each p, recording the max absolute deviation from plaintext. Reusing
the identical updates isolates aggregation arithmetic from training variability.

The full-HE aggregation does not depend on p, so it is computed ONCE per epsilon
and emitted as a single record. It used to sit inside the p loop, which emitted
two indistinguishable (epsilon, 1.0, "he") records per epsilon and silently
double-counted any per-p grouping of lossless.json.

TWO NORMALISATIONS OF THE DEVIATION, and only one of them is defensible as the
headline:

  rel_to_noise_floor  = max_abs_dev / dp_noise_std(sigma)
      Kept for continuity with earlier results, but it is dimensionally
      inconsistent: max_abs_dev is measured on AGGREGATED MODEL WEIGHTS while
      dp_noise_std is a PER-STEP GRADIENT noise standard deviation. Those live in
      different spaces, which is why this ratio wanders ~14x across the epsilon
      grid while its numerator barely moves -- all of the variation comes from
      the denominator.
  rel_to_update_scale = max_abs_dev / update_scale
      Divides by mean|aggregated weight|, i.e. the scale of the very thing being
      encrypted. Same space, same units, scale-free. It sits at a stable
      ~5.6e-07 to 7.6e-07 across every epsilon, which is the honest way to say
      "CKKS error is seven orders of magnitude below the signal it is carrying".

The verdict table leads with rel_to_update_scale for that reason.
"""
import json
import os

import numpy as np
import torch

from obscura import config, data, models, dp, he, sensitivity, fedprox, aggregate


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Prepare the dataset once (stroke)
    print("Preparing stroke dataset...")
    dataset = "stroke"
    X_tr, y_tr, X_te, y_te, info = data.split_and_scale(dataset, seed=config.SEED)
    in_features = info["n_features"]
    parts, part_info = data.partition(
        dataset, y_tr, num_clients=config.NUM_CLIENTS, seed=config.SEED
    )
    client_data = data.build_client_data(
        dataset, X_tr, y_tr, parts, use_smote=None, seed=config.SEED
    )

    # Build ONE shared global init
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    global_state = models.get_weights(
        models.new_model(in_features, dataset=dataset, width_scale=1)
    )
    n_params = models.count_params(
        models.new_model(in_features, dataset=dataset, width_scale=1)
    )

    # Build CKKS context once
    ctx = he.make_context()

    # Build clean sensitivity scores once from initial model + client 0 batch
    mask_model = models.new_model(in_features, dataset=dataset, width_scale=1)
    models.set_weights(mask_model, global_state)
    Xc_sample, yc_sample = client_data[0]
    batch_data = sensitivity.tabular_batch(
        Xc_sample, yc_sample, n=256, seed=config.SEED
    )
    clean_scores = sensitivity.sensitivity_scores(mask_model, batch_data)

    # Collect all records
    lossless_records = []

    # Sweep epsilons and fractions
    for epsilon in config.LOSSLESS_EPSILONS:
        print(f"\nEpsilon = {epsilon}")

        # Compute per-client sigma
        sigmas = {}
        for client_idx, (Xc, yc) in enumerate(client_data):
            sigmas[client_idx] = dp.get_sigma(
                epsilon, len(yc), epochs=config.LOCAL_EPOCHS,
                batch_size=config.BATCH_SIZE
            )
        mean_sigma = float(np.mean(list(sigmas.values())))
        noise_std = dp.dp_noise_std(mean_sigma, batch_size=config.BATCH_SIZE)

        # Generate DP-noised updates from ONE shared global init
        # (reuse this for every p)
        states = []
        counts = []
        for client_idx, (Xc, yc) in enumerate(client_data):
            sigma = sigmas[client_idx]
            state, n_samples, _ = fedprox.local_train(
                global_state, Xc, yc, in_features,
                dataset=dataset,
                dp=True,
                sigma=sigma,
                mu=config.MU,
                width_scale=1,
                lr=config.LR[dataset],
                local_epochs=config.LOCAL_EPOCHS,
                batch_size=config.BATCH_SIZE,
                seed=config.SEED * 1000 + client_idx
            )
            states.append(state)
            counts.append(n_samples)

        # Plaintext reference
        ref_flat = aggregate.plaintext_reference(states, counts, weighted=True)
        update_scale = float(np.mean(np.abs(ref_flat)))

        # Full HE does not depend on p -- aggregate once per epsilon.
        he_flat, he_stats = he.fedavg_he_timed(
            [models.flatten_state(s) for s in states],
            np.ones(n_params, dtype=bool),
            ctx,
            weights=aggregate.fedavg_weights(counts, weighted=True)
        )
        he_dev = he_flat - ref_flat
        he_max_abs_dev = float(np.max(np.abs(he_dev)))
        he_mean_abs_dev = float(np.mean(np.abs(he_dev)))
        lossless_records.append({
            "epsilon": epsilon,
            "sigma": mean_sigma,
            "noise_std": noise_std,
            "spe_fraction": 1.0,
            "mode": "he",
            "max_abs_dev": he_max_abs_dev,
            "mean_abs_dev": he_mean_abs_dev,
            "rel_to_noise_floor": he_max_abs_dev / noise_std if noise_std > 0 else 0.0,
            "rel_to_update_scale": (he_max_abs_dev / update_scale
                                    if update_scale > 0 else 0.0),
            "update_scale": update_scale,
            "n_params": n_params,
            "payload_ratio": float(he_stats["payload_ratio"]),
            "wire_payload_ratio": float(he_stats["wire_payload_ratio"]),
        })
        print(f"  full HE: max|HE-plain|={he_max_abs_dev:.2e}")

        # Now aggregate those SAME updates under SPE at each p
        for p in config.LOSSLESS_FRACTIONS:
            print(f"  p = {p}", end="")

            # SPE at fraction p (using pre-computed clean scores)
            spe_mask = sensitivity.build_mask(global_state, clean_scores, p)
            spe_flat, spe_stats = he.fedavg_he_timed(
                [models.flatten_state(s) for s in states],
                spe_mask,
                ctx,
                weights=aggregate.fedavg_weights(counts, weighted=True)
            )

            # Compute deviations
            spe_dev = spe_flat - ref_flat
            spe_max_abs_dev = float(np.max(np.abs(spe_dev)))
            spe_mean_abs_dev = float(np.mean(np.abs(spe_dev)))

            # SPE record
            spe_record = {
                "epsilon": epsilon,
                "sigma": mean_sigma,
                "noise_std": noise_std,
                "spe_fraction": p,
                "mode": "spe",
                "max_abs_dev": spe_max_abs_dev,
                "mean_abs_dev": spe_mean_abs_dev,
                "rel_to_noise_floor": spe_max_abs_dev / noise_std if noise_std > 0 else 0.0,
                "rel_to_update_scale": (spe_max_abs_dev / update_scale
                                        if update_scale > 0 else 0.0),
                "update_scale": update_scale,
                "n_params": n_params,
                "payload_ratio": float(spe_stats["payload_ratio"]),
                "wire_payload_ratio": float(spe_stats["wire_payload_ratio"]),
            }
            lossless_records.append(spe_record)

            print(f"  max|SPE-plain|={spe_max_abs_dev:.2e}  "
                  f"wire_payload_ratio={spe_stats['wire_payload_ratio']:.3f}")

    # Build and save results
    results = {
        "meta": {
            "trial": "07/27",
            "seed": config.SEED,
            "rounds": config.ROUNDS,
            "local_epochs": config.LOCAL_EPOCHS,
            "num_clients": config.NUM_CLIENTS,
            "generated_by": "run_losslessness.py",
            "epsilons": config.LOSSLESS_EPSILONS,
            "fractions": config.LOSSLESS_FRACTIONS,
        },
        "lossless": lossless_records,
    }

    out_path = os.path.join(config.RESULTS_DIR, "lossless.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # Print verdict table per epsilon
    print("\n" + "="*80)
    print("LOSSLESSNESS VERDICT PER EPSILON")
    print("="*80)
    print("Leading with rel_to_update_scale: it divides the weight-space deviation")
    print("by the weight-space update magnitude, so it is scale-free and stable.")
    print("rel_to_noise_floor is shown second and mixes weight space with gradient")
    print("space, which is why only IT moves across epsilon -- see the docstring.\n")
    for epsilon in config.LOSSLESS_EPSILONS:
        # Find records for this epsilon
        eps_records = [r for r in lossless_records if r["epsilon"] == epsilon]
        if eps_records:
            # Take HE at p=1.0
            he_rec = next((r for r in eps_records if r["mode"] == "he"), None)
            if he_rec:
                noise_std = he_rec["noise_std"]
                max_dev = he_rec["max_abs_dev"]
                ratio = max_dev / noise_std if noise_std > 0 else 0.0
                print(f"eps={epsilon:5.1f}  max|HE-plain|={max_dev:.2e}  "
                      f"->  {he_rec['rel_to_update_scale']:.2e} x the update scale "
                      f"({he_rec['update_scale']:.3e})"
                      f"   [legacy: {ratio:.2e} x the DP noise floor "
                      f"{noise_std:.2e}]")

    rus = [r["rel_to_update_scale"] for r in lossless_records]
    rnf = [r["rel_to_noise_floor"] for r in lossless_records]
    print(f"\nrel_to_update_scale spans {min(rus):.2e} .. {max(rus):.2e} "
          f"({max(rus)/min(rus):.2f}x) over all {len(rus)} records")
    print(f"rel_to_noise_floor  spans {min(rnf):.2e} .. {max(rnf):.2e} "
          f"({max(rnf)/min(rnf):.2f}x) -- the spread is the denominator, not the error")

    # Check against CKKS_TOLERANCE
    print("\n" + "="*80)
    print("CKKS_TOLERANCE CHECK")
    print("="*80)
    all_pass = True
    for record in lossless_records:
        if record["max_abs_dev"] > config.CKKS_TOLERANCE:
            print(f"FAIL: eps={record['epsilon']} p={record['spe_fraction']} "
                  f"mode={record['mode']} max_abs_dev={record['max_abs_dev']:.2e} "
                  f"> CKKS_TOLERANCE={config.CKKS_TOLERANCE:.2e}")
            all_pass = False

    if all_pass:
        print(f"PASS: All max_abs_dev values are below CKKS_TOLERANCE={config.CKKS_TOLERANCE:.2e}")
    else:
        print(f"FAIL: Some max_abs_dev values exceed CKKS_TOLERANCE={config.CKKS_TOLERANCE:.2e}")


if __name__ == "__main__":
    main()
