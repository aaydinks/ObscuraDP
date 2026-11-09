"""Experiment A: full-stack arm matrix sweep.

Isolates each privacy leg via a single controlled change: plain -> dp adds DP,
plain -> he adds HE, dp_he adds both, dp_spe makes HE selective. Accuracy cost
is attributed to exactly one mechanism. Sweeps config.ARM_DATASETS x
config.ARM_ALGOS x config.ARM_STACKS at config.EPSILON_DEFAULT and
config.SPE_DEFAULT. Extends Experiment 2's methodology from 3 arms to 5, and
runs on both the IID credit task and the non-IID imbalanced clinical task.

Reuses one prepared bundle and one CKKS context per dataset to ensure arms are
exactly comparable.
"""
import argparse
import json
import os

import numpy as np

from obscura import config, he, orchestrator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Random seed (default: config.SEED)")
    parser.add_argument("--out", type=str, default="results/arms.json",
                        help="Output path (default: results/arms.json)")
    args = parser.parse_args()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    results = {
        "meta": {
            "trial": "07/27",
            "seed": args.seed,
            "rounds": config.ROUNDS,
            "local_epochs": config.LOCAL_EPOCHS,
            "num_clients": config.NUM_CLIENTS,
            "generated_by": "run_arms.py",
            "datasets": config.ARM_DATASETS,
            "algorithms": config.ARM_ALGOS,
            "stacks": config.ARM_STACKS,
            "epsilon": config.EPSILON_DEFAULT,
            "spe_fraction": config.SPE_DEFAULT,
            "mask_mode": config.MASK_MODE_DEFAULT,
        },
        "arms": [],
    }

    # Build and emit markdown tables per dataset
    tables = {}

    # Sweep datasets
    for dataset in config.ARM_DATASETS:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset}")
        print(f"{'='*70}")

        # Prepare once per dataset
        prepared = orchestrator.prepare(dataset, args.seed)
        ctx = he.make_context()

        # Collect arms for this dataset for markdown table
        dataset_arms = []

        # Sweep algorithms and stacks
        for algo in config.ARM_ALGOS:
            for stack in config.ARM_STACKS:
                print(f"  {algo:8s} + {stack:8s}...", end=" ", flush=True)

                arm = orchestrator.run_stack(
                    dataset=dataset,
                    stack=stack,
                    algo=algo,
                    epsilon=config.EPSILON_DEFAULT,
                    spe_fraction=config.SPE_DEFAULT,
                    mask_mode=config.MASK_MODE_DEFAULT,
                    seed=args.seed,
                    prepared=prepared,
                    ctx=ctx,
                )

                results["arms"].append(arm)
                dataset_arms.append(arm)
                m = arm["metrics"]
                print(f"acc={m['accuracy']:.4f} ", end="")
                if "recall" in m and m["recall"] is not None:
                    print(f"recall={m['recall']:.4f}", end="")
                print()

        # Generate markdown table for this dataset
        tables[dataset] = _render_table(dataset, dataset_arms)

    # Save results
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {args.out}")

    # Save and print markdown tables
    for dataset, table in tables.items():
        table_path = os.path.join(config.RESULTS_DIR,
                                   f"arms_table_{dataset}.md")
        with open(table_path, "w") as f:
            f.write(table)
        print(f"\nSaved {table_path}\n{table}")


def _render_table(dataset, arms):
    """Render markdown table with columns dependent on dataset."""
    # Determine primary metric
    primary = "accuracy" if dataset == "loan" else "recall"

    lines = []
    if dataset == "loan":
        lines.append(
            "| Stack | Algo | Accuracy | Recall | F1 | AUC | "
            "eps(round) | eps(cum) | p | Crypto s/round | Payload x | "
            "max|HE-plain| |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    else:  # stroke
        lines.append(
            "| Stack | Algo | Recall | F1 | AUC | Accuracy | "
            "eps(round) | eps(cum) | p | Crypto s/round | Payload x | "
            "max|HE-plain| |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for arm in arms:
        stack = arm["stack"]
        algo = arm["algo"]
        metrics = arm["metrics"]
        dp_info = arm.get("dp", {})
        crypto_info = arm.get("crypto", {})
        lossless = arm.get("lossless", {})

        # Extract metrics with NaN handling
        accuracy = metrics.get("accuracy")
        recall = metrics.get("recall")
        f1 = metrics.get("f1")
        auc = metrics.get("auc")

        acc_str = f"{accuracy:.4f}" if accuracy is not None else "--"
        recall_str = f"{recall:.4f}" if recall is not None else "--"
        f1_str = f"{f1:.4f}" if f1 is not None else "--"
        auc_str = f"{auc:.4f}" if auc is not None else "--"

        # DP info
        eps_round = dp_info.get("eps_round")
        eps_cum = dp_info.get("eps_cumulative")
        eps_round_str = f"{eps_round:.2f}" if eps_round is not None else "--"
        eps_cum_str = f"{eps_cum:.2f}" if eps_cum is not None else "--"

        # SPE fraction
        spe_frac = arm.get("spe_fraction", 0.0)
        p_str = f"{spe_frac:.2f}"

        # Crypto info (per-round means)
        crypto_sec = crypto_info.get("crypto_sec", 0.0)
        plaintext_bytes = crypto_info.get("plaintext_bytes", 1.0)
        ciphertext_bytes = crypto_info.get("ciphertext_bytes", 0.0)
        payload_x = (ciphertext_bytes / plaintext_bytes) if plaintext_bytes > 0 else 0.0
        crypto_sec_str = f"{crypto_sec:.4f}" if crypto_sec > 0 else "--"
        payload_x_str = f"{payload_x:.1f}x" if payload_x > 0 else "--"

        # Losslessness
        max_dev = lossless.get("max_abs_dev")
        max_dev_str = f"{max_dev:.2e}" if max_dev is not None else "--"

        if dataset == "loan":
            lines.append(
                f"| {stack} | {algo} | {acc_str} | {recall_str} | {f1_str} | "
                f"{auc_str} | {eps_round_str} | {eps_cum_str} | {p_str} | "
                f"{crypto_sec_str} | {payload_x_str} | {max_dev_str} |"
            )
        else:  # stroke
            lines.append(
                f"| {stack} | {algo} | {recall_str} | {f1_str} | {auc_str} | "
                f"{acc_str} | {eps_round_str} | {eps_cum_str} | {p_str} | "
                f"{crypto_sec_str} | {payload_x_str} | {max_dev_str} |"
            )

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
