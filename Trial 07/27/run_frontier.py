"""Experiment B: joint epsilon x p frontier -- the HEADLINE.

Tests SEPARABILITY: utility should depend only on epsilon and be flat in p,
while crypto cost should depend only on p and be flat in epsilon. If that
holds, the privacy-utility frontier and privacy-cost frontier are orthogonal
axes, and an operator can tune them independently -- the practical payoff of
composing DP with SPE-HE.

Maps (epsilon, p) to stacks: p == 0.0 -> "dp", p == 1.0 -> "dp_he",
else "dp_spe" with spe_fraction=p. All DP-on, so epsilon is always meaningful.

Reuses one prepared bundle and one CKKS context across the sweep. Accepts
optional --seed and --out arguments (for run_multiseed.py to invoke per seed).
"""
import argparse
import json
import os

import numpy as np

from obscura import config, he, orchestrator, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Random seed (default: config.SEED)")
    parser.add_argument("--out", type=str, default="results/frontier.json",
                        help="Output path (default: results/frontier.json)")
    args = parser.parse_args()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    dataset = config.FRONTIER_DATASET
    algo = config.FRONTIER_ALGO
    primary_metric_name = metrics.primary_metric(dataset)

    results = {
        "meta": {
            "trial": "07/27",
            "seed": args.seed,
            "rounds": config.ROUNDS,
            "local_epochs": config.LOCAL_EPOCHS,
            "num_clients": config.NUM_CLIENTS,
            "generated_by": "run_frontier.py",
            "dataset": dataset,
            "algorithm": algo,
            "epsilons": config.FRONTIER_EPSILONS,
            "fractions": config.FRONTIER_FRACTIONS,
            "mask_mode": config.MASK_MODE_DEFAULT,
        },
        "frontier": [],
    }

    print(f"\n{'='*70}")
    print(f"Frontier sweep: {dataset} + {algo}")
    print(f"Epsilons: {config.FRONTIER_EPSILONS}")
    print(f"p (encryption fractions): {config.FRONTIER_FRACTIONS}")
    print(f"{'='*70}\n")

    # Prepare once, reuse for all arms
    prepared = orchestrator.prepare(dataset, args.seed)
    ctx = he.make_context()

    # Collect results keyed by (epsilon, fraction) for pivot table construction
    results_grid = {}  # (eps, frac) -> arm dict

    # Sweep epsilon x fraction grid
    for epsilon in config.FRONTIER_EPSILONS:
        for fraction in config.FRONTIER_FRACTIONS:
            # Map fraction to stack
            if fraction == 0.0:
                stack = "dp"
            elif fraction == 1.0:
                stack = "dp_he"
            else:
                stack = "dp_spe"

            print(f"  eps={epsilon:5.1f} p={fraction:5.2f} ({stack:8s})...",
                  end=" ", flush=True)

            arm = orchestrator.run_stack(
                dataset=dataset,
                stack=stack,
                algo=algo,
                epsilon=epsilon,
                spe_fraction=fraction,
                mask_mode=config.MASK_MODE_DEFAULT,
                seed=args.seed,
                prepared=prepared,
                ctx=ctx,
            )

            results["frontier"].append(arm)
            results_grid[(epsilon, fraction)] = arm
            m = arm["metrics"]
            primary = m.get(primary_metric_name, 0.0)
            print(f"{primary_metric_name}={primary:.4f}")

    # Save results JSON
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {args.out}")

    # Print pivot tables
    _print_pivot_tables(dataset, config.FRONTIER_EPSILONS,
                       config.FRONTIER_FRACTIONS, results_grid,
                       primary_metric_name)


def _print_pivot_tables(dataset, epsilons, fractions, results_grid,
                        primary_metric_name):
    """Print two pivot tables: utility (primary metric) and crypto cost."""
    print(f"\n{'='*70}")
    print("Utility pivot table (primary metric = {})".format(
        primary_metric_name))
    print("Rows = epsilon, Columns = p (encryption fraction)")
    print("Claim: each row should be near-constant (utility flat in p)")
    print(f"{'='*70}")

    # Header
    header = "eps / p  |" + "".join(f"  p={f:4.2f}  |" for f in fractions)
    print(header)
    print("-" * len(header))

    # Rows for each epsilon
    for eps in epsilons:
        row = f" {eps:5.1f}  |"
        for frac in fractions:
            if (eps, frac) in results_grid:
                arm = results_grid[(eps, frac)]
                val = arm["metrics"].get(primary_metric_name, np.nan)
                if val is not None and not np.isnan(val):
                    row += f"  {val:6.4f}  |"
                else:
                    row += "    --    |"
            else:
                row += "    --    |"
        print(row)

    print(f"\n{'='*70}")
    print("Crypto cost pivot table (crypto_sec per round)")
    print("Rows = epsilon, Columns = p")
    print("Claim: each column should be near-constant (cost flat in eps)")
    print(f"{'='*70}")

    # Header
    header = "eps / p  |" + "".join(f"  p={f:4.2f}  |" for f in fractions)
    print(header)
    print("-" * len(header))

    # Rows for each epsilon
    for eps in epsilons:
        row = f" {eps:5.1f}  |"
        for frac in fractions:
            if (eps, frac) in results_grid:
                arm = results_grid[(eps, frac)]
                crypto = arm.get("crypto", {})
                val = crypto.get("crypto_sec", 0.0)
                if val is not None and val > 0:
                    row += f"  {val:6.4f}  |"
                else:
                    row += "    0.00  |"
            else:
                row += "    --    |"
        print(row)

    print()


if __name__ == "__main__":
    main()
