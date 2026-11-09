"""Multi-seed frontier campaign: mean +/- std over independent data splits.

Experiment 1 found the stroke test set has only ~42 positives, giving ~0.024 recall
granularity, so single-seed differences are not interpretable. Experiment 2 found DP
variance GROWS as epsilon shrinks. Both effects are present in this trial, so error
bars are mandatory: we run the epsilon x p frontier across SEEDS independent
splits/partitions/inits so the headline result is mean +/- std rather than one lucky
draw. Usage: python run_multiseed.py [--seeds 42 43 44 45 46] [--skip-existing]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
torch.set_num_threads(2)

from obscura import config, orchestrator, metrics


STACK_MAP = {
    0.0: "dp",
    1.0: "dp_he",
}


def epsilon_to_stack(p):
    if p in STACK_MAP:
        return STACK_MAP[p]
    return "dp_spe"


def run_seed(seed, skip_existing=False):
    out_path = os.path.join(config.RESULTS_DIR, f"seed_{seed}.json")

    if skip_existing and os.path.exists(out_path):
        print(f"[seed {seed}] skipping (file exists)")
        return

    print(f"[seed {seed}] preparing data")
    t0 = time.time()
    prepared = orchestrator.prepare(config.FRONTIER_DATASET, seed)
    ctx = None
    if any(p > 0.0 for p in config.FRONTIER_FRACTIONS):
        from obscura import he
        ctx = he.make_context()

    frontier = []

    for eps_idx, epsilon in enumerate(config.FRONTIER_EPSILONS):
        for frac_idx, p in enumerate(config.FRONTIER_FRACTIONS):
            stack = epsilon_to_stack(p)
            spe_fraction = p if stack == "dp_spe" else (1.0 if stack == "dp_he" else 0.0)

            t_round = time.time()
            rec = orchestrator.run_stack(
                dataset=config.FRONTIER_DATASET,
                stack=stack,
                algo=config.FRONTIER_ALGO,
                epsilon=epsilon,
                spe_fraction=spe_fraction,
                mask_mode=config.MASK_MODE_DEFAULT,
                seed=seed,
                ctx=ctx,
                prepared=prepared,
                log=None,
            )
            frontier.append(rec)

            elapsed_round = time.time() - t_round
            metric_val = rec["metrics"].get(
                metrics.primary_metric(config.FRONTIER_DATASET), 0.0
            )
            print(
                f"  eps={epsilon:6.1f} p={p:4.2f} {stack:8s} "
                f"| {metric_val:.3f} | {elapsed_round:.1f}s",
                flush=True
            )

            if (eps_idx + 1) * (frac_idx + 1) % 7 == 0:
                with open(out_path, "w") as f:
                    json.dump(
                        {
                            "meta": {
                                "trial": "07/27",
                                "seed": seed,
                                "rounds": config.ROUNDS,
                                "local_epochs": config.LOCAL_EPOCHS,
                                "num_clients": config.NUM_CLIENTS,
                                "generated_by": "run_multiseed.py",
                                "epsilons": config.FRONTIER_EPSILONS,
                                "fractions": config.FRONTIER_FRACTIONS,
                            },
                            "frontier": frontier,
                        },
                        f,
                        indent=2,
                    )

    elapsed = time.time() - t0
    with open(out_path, "w") as f:
        json.dump(
            {
                "meta": {
                    "trial": "07/27",
                    "seed": seed,
                    "rounds": config.ROUNDS,
                    "local_epochs": config.LOCAL_EPOCHS,
                    "num_clients": config.NUM_CLIENTS,
                    "generated_by": "run_multiseed.py",
                    "epsilons": config.FRONTIER_EPSILONS,
                    "fractions": config.FRONTIER_FRACTIONS,
                },
                "frontier": frontier,
            },
            f,
            indent=2,
        )

    print(f"[seed {seed}] DONE -> {out_path} ({elapsed:.1f}s)")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed frontier campaign"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=config.SEEDS,
        help=f"Seeds to run (default: {config.SEEDS})",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip seeds whose results/seed_<n>.json already exists",
    )
    args = parser.parse_args()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    for seed in args.seeds:
        run_seed(seed, skip_existing=args.skip_existing)

    print("\nAggregating seeds...")
    import aggregate_seeds
    aggregate_seeds.main()


if __name__ == "__main__":
    main()
