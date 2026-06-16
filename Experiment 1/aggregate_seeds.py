"""Aggregate per-seed runs into mean +/- std -> results/results_multiseed.json."""
import glob
import json
import os
import statistics as stats

from stroke_fl import config

METRICS = ["recall", "precision", "f1", "auc", "ap", "pred_pos_rate"]


def msd(values):
    return {
        "mean": stats.mean(values),
        "std": stats.pstdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
        "values": [round(v, 4) for v in values],
    }


def main():
    files = sorted(glob.glob(os.path.join(config.RESULTS_DIR, "seed_*.json")))
    runs = [json.load(open(f)) for f in files]
    seeds = [r["seed"] for r in runs]
    print(f"aggregating {len(runs)} seeds: {seeds}")

    out = {"seeds": seeds, "n_seeds": len(seeds),
           "config": {
               "num_clients": config.NUM_CLIENTS, "rounds": config.ROUNDS,
               "local_epochs": config.LOCAL_EPOCHS, "mu": config.MU,
               "epsilon_grid": config.EPSILON_GRID, "lr": config.LR,
               "batch_size": config.BATCH_SIZE,
               "max_grad_norm": config.MAX_GRAD_NORM,
               "target_delta": config.TARGET_DELTA,
               "dirichlet_alpha": config.DIRICHLET_ALPHA,
           },
           "baseline_no_smote": {}, "sweep": {"FedProx": {}, "FedAvg": {}}}

    # baseline
    for met in METRICS:
        out["baseline_no_smote"][met] = msd(
            [r["baseline_no_smote"][met] for r in runs])

    # sweep
    for strat in ["FedProx", "FedAvg"]:
        for eps in config.EPSILON_GRID:
            k = str(eps)
            entry = {}
            for met in METRICS:
                entry[met] = msd([r["sweep"][strat][k][met] for r in runs])
            out["sweep"][strat][k] = entry

    path = os.path.join(config.RESULTS_DIR, "results_multiseed.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", path)

    # console summary
    b = out["baseline_no_smote"]["recall"]
    print(f"\nbaseline (no SMOTE) recall = {b['mean']:.3f} +/- {b['std']:.3f}")
    print("\n eps  |    FedProx recall    |     FedAvg recall    | FP-FA gap")
    print("-" * 64)
    for eps in config.EPSILON_GRID:
        fp = out["sweep"]["FedProx"][str(eps)]["recall"]
        fa = out["sweep"]["FedAvg"][str(eps)]["recall"]
        gap = fp["mean"] - fa["mean"]
        print(f"{eps:5.1f} |  {fp['mean']:.3f} +/- {fp['std']:.3f}     |  "
              f"{fa['mean']:.3f} +/- {fa['std']:.3f}    | {gap:+.3f}")


if __name__ == "__main__":
    main()
