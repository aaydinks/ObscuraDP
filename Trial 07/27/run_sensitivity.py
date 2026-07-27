"""Experiment D: does the SPE mask survive DP noise?

Experiment 3 chose the top-p sensitivity mask from CLEAN gradient magnitudes.
Under DP-SGD, gradients are clipped and Gaussian-noised. If the ranking does not
survive, the SPE heuristic does not transfer to a DP+HE stack.

Two parts:
  1. Mask agreement: for each (epsilon, p, repeat), build clean/DP/random masks
     and measure their Jaccard similarity and Spearman rank correlation vs epsilon.
  2. Utility: run the full stack with different mask modes (clean/dp/random) and
     verify utility is essentially identical -- the mask choice is privacy, not utility.
"""
import json
import os

import numpy as np
import torch

from obscura import config, data, models, dp, sensitivity, orchestrator, he


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # ========================================================================
    # PART 1: Mask Agreement
    # ========================================================================
    print("="*80)
    print("PART 1: Mask Agreement (clean vs DP vs random)")
    print("="*80)

    # Prepare stroke dataset once
    dataset = "stroke"
    prepared = orchestrator.prepare(dataset, seed=config.SEED)
    X_tr = prepared["X_tr"]
    y_tr = prepared["y_tr"]
    client_data = prepared["client_data"]
    in_features = prepared["in_features"]

    # Build ONE shared global init and batch for mask building
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    global_state = models.get_weights(
        models.new_model(in_features, dataset=dataset, width_scale=1)
    )

    # Get a batch from client 0 for mask building
    Xc_sample, yc_sample = client_data[0]
    batch_data = sensitivity.tabular_batch(
        Xc_sample, yc_sample, n=256, seed=config.SEED
    )

    agreement_records = []

    for epsilon in config.SENS_EPSILONS:
        print(f"\nEpsilon = {epsilon}")

        # Get sigma for this epsilon (from client 0)
        n_samples_client0 = len(client_data[0][1])
        sigma = dp.get_sigma(
            epsilon, n_samples_client0, epochs=config.LOCAL_EPOCHS,
            batch_size=config.BATCH_SIZE
        )

        for fraction in config.SENS_FRACTIONS:
            print(f"  Fraction p = {fraction}")

            for repeat in range(config.SENS_REPEATS):
                # Build clean mask + scores
                mask_model = models.new_model(
                    in_features, dataset=dataset, width_scale=1
                )
                models.set_weights(mask_model, global_state)
                clean_mask, clean_scores = sensitivity.make_mask(
                    mask_model, global_state, batch_data,
                    fraction=fraction, mode="clean", seed=config.SEED + repeat
                )

                # Build DP mask + scores with this epsilon's sigma
                mask_model = models.new_model(
                    in_features, dataset=dataset, width_scale=1
                )
                models.set_weights(mask_model, global_state)
                dp_mask, dp_scores = sensitivity.make_mask(
                    mask_model, global_state, batch_data,
                    fraction=fraction, mode="dp", sigma=sigma,
                    seed=config.SEED + repeat
                )

                # Build random mask as control
                n_total = sum(numel for _, _, numel in models.param_index_map(global_state))
                random_mask = sensitivity.random_mask(
                    n_total, fraction, seed=config.SEED + repeat
                )

                # Measure agreement: clean vs DP
                agreement = sensitivity.mask_agreement(clean_mask, dp_mask)
                jaccard_clean_dp = agreement["jaccard"]
                overlap_clean_dp = agreement["overlap"]

                # Measure Spearman rank correlation
                spearman_clean_dp = sensitivity.score_rank_correlation(
                    clean_scores, dp_scores
                )

                # Compare against random: build random scores (all zeros, so rank is random)
                random_agreement = sensitivity.mask_agreement(clean_mask, random_mask)
                random_jaccard = random_agreement["jaccard"]

                record = {
                    "epsilon": epsilon,
                    "sigma": float(sigma),
                    "fraction": fraction,
                    "repeat": repeat,
                    "jaccard": jaccard_clean_dp,
                    "overlap": overlap_clean_dp,
                    "spearman": spearman_clean_dp,
                    "random_jaccard": random_jaccard,
                }
                agreement_records.append(record)

                print(f"    repeat {repeat}: jaccard={jaccard_clean_dp:.4f} "
                      f"(random={random_jaccard:.4f}) spearman={spearman_clean_dp:.4f}")

    # ========================================================================
    # PART 2: Utility Across Mask Modes
    # ========================================================================
    print("\n" + "="*80)
    print("PART 2: Utility Across Mask Modes")
    print("="*80)

    # Build CKKS context once
    ctx = he.make_context()

    utility_runs = []

    for mask_mode in config.MASK_MODES:
        print(f"\nMask mode: {mask_mode}")
        print("  Running dp_spe stack...")

        arm = orchestrator.run_stack(
            dataset="stroke",
            stack="dp_spe",
            algo="fedprox",
            epsilon=config.EPSILON_DEFAULT,
            spe_fraction=config.SPE_DEFAULT,
            mask_mode=mask_mode,
            seed=config.SEED,
            prepared=prepared,
            ctx=ctx,
            log=None  # Suppress per-round logs
        )
        utility_runs.append(arm)
        m = arm["metrics"]
        print(f"    accuracy={m.get('accuracy', 0):.4f} "
              f"recall={m.get('recall', 0):.4f}")

    # ========================================================================
    # Save Results
    # ========================================================================
    results = {
        "meta": {
            "trial": "07/27",
            "seed": config.SEED,
            "rounds": config.ROUNDS,
            "local_epochs": config.LOCAL_EPOCHS,
            "num_clients": config.NUM_CLIENTS,
            "generated_by": "run_sensitivity.py",
            "epsilons": config.SENS_EPSILONS,
            "fractions": config.SENS_FRACTIONS,
            "repeats": config.SENS_REPEATS,
            "mask_modes": config.MASK_MODES,
        },
        "sensitivity": agreement_records,
        "utility": utility_runs,
    }

    out_path = os.path.join(config.RESULTS_DIR, "sensitivity.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # ========================================================================
    # Print Tables
    # ========================================================================
    print("\n" + "="*80)
    print("MASK AGREEMENT TABLE")
    print("="*80)
    print("Rows = epsilon, Cols = fraction, Cells = 'clean vs DP Jaccard (random Jaccard)'")
    print()

    # Build table
    fractions = sorted(set(r["fraction"] for r in agreement_records))
    epsilons_sorted = sorted(config.SENS_EPSILONS)

    header = "| Epsilon |"
    for frac in fractions:
        header += f" p={frac} |"
    print(header)

    sep = "|---|"
    for _ in fractions:
        sep += "---|"
    print(sep)

    for epsilon in epsilons_sorted:
        row = f"| {epsilon} |"
        for frac in fractions:
            # Average Jaccard for this (eps, frac) pair over all repeats
            matching = [r for r in agreement_records
                       if r["epsilon"] == epsilon and r["fraction"] == frac]
            if matching:
                avg_jaccard = np.mean([r["jaccard"] for r in matching])
                avg_random_jaccard = np.mean([r["random_jaccard"] for r in matching])
                row += f" {avg_jaccard:.3f} ({avg_random_jaccard:.3f}) |"
            else:
                row += " -- |"
        print(row)

    print("\n" + "="*80)
    print("UTILITY ACROSS MASK MODES")
    print("="*80)
    print("Note: utility should be essentially identical across mask modes.")
    print("The mask choice is a privacy/leakage decision, not a utility one.\n")

    print("| Mask Mode | Accuracy | Recall | F1 | AUC |")
    print("|---|---:|---:|---:|---:|")

    for run in utility_runs:
        mode = run.get("mask_mode", "unknown")
        metrics = run.get("metrics", {})
        accuracy = metrics.get("accuracy", 0)
        recall = metrics.get("recall", 0)
        f1 = metrics.get("f1", 0)
        auc = metrics.get("auc", 0)

        print(f"| {mode} | {accuracy:.4f} | {recall:.4f} | {f1:.4f} | {auc:.4f} |")

    print("\n" + "="*80)
    print("KEY INSIGHT")
    print("="*80)
    print("Utility is independent of mask_mode because SPE is lossless: "
          "which parameters get encrypted does not affect the final aggregate's")
    print("magnitude or distribution -- only the privacy leakage through the mask changes.")
    print("Thus, if clean vs DP Jaccard >> random Jaccard, the DP mask still captures")
    print("the sensitive parameters, and SPE's privacy targeting survives DP noise.")


if __name__ == "__main__":
    main()
