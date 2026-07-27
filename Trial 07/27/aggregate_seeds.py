"""Aggregate per-seed frontier runs into mean +/- std results.

Loads all seed_<n>.json files, groups records by (epsilon, spe_fraction, stack,
algo, dataset), and computes mean/std of all numeric metrics. Handles NaN values
with np.nanmean/np.nanstd so a NaN AUC in one seed does not poison the group.
"""
import glob
import json
import os

import numpy as np

from obscura import config, metrics as metrics_module


def load_seed_files(results_dir=None, seeds=None):
    """Load seed_*.json files and return list of (seed, records)."""
    if results_dir is None:
        results_dir = config.RESULTS_DIR

    if seeds is None:
        pattern = os.path.join(results_dir, "seed_*.json")
        files = sorted(glob.glob(pattern))
    else:
        files = [
            os.path.join(results_dir, f"seed_{s}.json")
            for s in sorted(seeds)
            if os.path.exists(os.path.join(results_dir, f"seed_{s}.json"))
        ]

    if not files:
        return []

    result = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
            seed = data["meta"]["seed"]
            frontier = data.get("frontier", [])
            result.append((seed, frontier))

    return result


def aggregate(records_by_seed):
    """Group frontier records by key and compute mean +/- std.

    Input: list of (seed, records) tuples.
    Output: list of dicts with aggregated statistics.
    """
    if not records_by_seed:
        return []

    groups = {}
    for seed, records in records_by_seed:
        for rec in records:
            key = (
                rec["epsilon"],
                rec["spe_fraction"],
                rec["stack"],
                rec["algo"],
                rec["dataset"],
            )
            if key not in groups:
                groups[key] = []
            groups[key].append(rec)

    aggregated = []
    for key, recs in sorted(groups.items()):
        epsilon, spe_fraction, stack, algo, dataset = key
        agg_rec = {
            "epsilon": epsilon,
            "spe_fraction": spe_fraction,
            "stack": stack,
            "algo": algo,
            "dataset": dataset,
            "n_seeds": len(recs),
        }

        # Aggregate numeric metrics from metrics dict
        if recs and "metrics" in recs[0]:
            for metric_key in recs[0]["metrics"].keys():
                if isinstance(recs[0]["metrics"][metric_key], (int, float)):
                    vals = np.array(
                        [
                            r["metrics"].get(metric_key, np.nan)
                            for r in recs
                        ],
                        dtype=float
                    )
                    agg_rec[f"{metric_key}_mean"] = float(np.nanmean(vals))
                    agg_rec[f"{metric_key}_std"] = float(np.nanstd(vals))

        # Aggregate crypto stats (already per-round means in the record)
        if recs and "crypto" in recs[0]:
            if "crypto_sec" in recs[0]["crypto"]:
                vals = np.array(
                    [
                        r["crypto"].get("crypto_sec", np.nan)
                        for r in recs
                    ],
                    dtype=float
                )
                agg_rec["crypto_sec_mean"] = float(np.nanmean(vals))
                agg_rec["crypto_sec_std"] = float(np.nanstd(vals))

            if "payload_ratio" in recs[0]["crypto"]:
                vals = np.array(
                    [
                        r["crypto"].get("payload_ratio", np.nan)
                        for r in recs
                    ],
                    dtype=float
                )
                agg_rec["payload_ratio_mean"] = float(np.nanmean(vals))

        # max_abs_dev_max: the maximum over all seeds
        if recs and "lossless" in recs[0]:
            max_devs = [
                r["lossless"].get("max_abs_dev", np.nan)
                for r in recs
            ]
            max_devs = [v for v in max_devs if not np.isnan(v)]
            if max_devs:
                agg_rec["max_abs_dev_max"] = float(max(max_devs))

        aggregated.append(agg_rec)

    return aggregated


def render_epsilon_by_fraction_table(aggregated, dataset):
    """Render markdown table: rows=epsilon, cols=spe_fraction, cells=primary metric."""
    if not aggregated:
        return ""

    # Filter for this dataset
    filtered = [r for r in aggregated if r["dataset"] == dataset]
    if not filtered:
        return ""

    # Extract unique epsilons and fractions
    epsilons = sorted(set(r["epsilon"] for r in filtered))
    fractions = sorted(set(r["spe_fraction"] for r in filtered))

    primary = metrics_module.primary_metric(dataset)

    # Build table header
    lines = [f"### Primary metric ({primary}) by epsilon and fraction"]
    lines.append("")
    lines.append(
        "| epsilon | " + " | ".join(f"p={f}" for f in fractions) + " |"
    )
    lines.append("|---|" + "|".join(["---"] * len(fractions)) + "|")

    # Build table rows
    for eps in epsilons:
        row = [f"{eps}"]
        for frac in fractions:
            recs = [
                r for r in filtered
                if r["epsilon"] == eps and r["spe_fraction"] == frac
            ]
            if recs:
                r = recs[0]
                mean_key = f"{primary}_mean"
                std_key = f"{primary}_std"
                if mean_key in r and std_key in r:
                    mean_val = r[mean_key]
                    std_val = r[std_key]
                    row.append(f"{mean_val:.3f} ± {std_val:.3f}")
                else:
                    row.append("—")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def render_crypto_table(aggregated, dataset):
    """Render markdown table: rows=epsilon, cols=spe_fraction, cells=crypto_sec."""
    if not aggregated:
        return ""

    filtered = [r for r in aggregated if r["dataset"] == dataset]
    if not filtered:
        return ""

    epsilons = sorted(set(r["epsilon"] for r in filtered))
    fractions = sorted(set(r["spe_fraction"] for r in filtered))

    lines = [f"### Crypto seconds per round by epsilon and fraction"]
    lines.append("")
    lines.append(
        "| epsilon | " + " | ".join(f"p={f}" for f in fractions) + " |"
    )
    lines.append("|---|" + "|".join(["---"] * len(fractions)) + "|")

    for eps in epsilons:
        row = [f"{eps}"]
        for frac in fractions:
            recs = [
                r for r in filtered
                if r["epsilon"] == eps and r["spe_fraction"] == frac
            ]
            if recs:
                r = recs[0]
                if "crypto_sec_mean" in r:
                    mean_val = r["crypto_sec_mean"]
                    std_val = r.get("crypto_sec_std", 0.0)
                    row.append(f"{mean_val:.4f} ± {std_val:.4f}")
                else:
                    row.append("—")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    runs_by_seed = load_seed_files()
    if not runs_by_seed:
        print(
            f"ERROR: No seed_*.json files found in {config.RESULTS_DIR}. "
            "Run run_multiseed.py first."
        )
        return

    seeds = [s for s, _ in runs_by_seed]
    print(f"Loaded {len(runs_by_seed)} seeds: {seeds}")

    aggregated = aggregate(runs_by_seed)
    print(f"Aggregated into {len(aggregated)} groups")

    # Collect all per-seed records for the "runs" key
    all_runs = []
    for seed, recs in runs_by_seed:
        for rec in recs:
            all_runs.append(rec)

    # Write multiseed.json
    out_path = os.path.join(config.RESULTS_DIR, "multiseed.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "meta": {
                    "trial": "07/27",
                    "seeds": seeds,
                    "n_seeds": len(seeds),
                    "rounds": config.ROUNDS,
                    "local_epochs": config.LOCAL_EPOCHS,
                    "num_clients": config.NUM_CLIENTS,
                    "generated_by": "aggregate_seeds.py",
                    "frontier_epsilons": config.FRONTIER_EPSILONS,
                    "frontier_fractions": config.FRONTIER_FRACTIONS,
                },
                "runs": all_runs,
                "aggregated": aggregated,
            },
            f,
            indent=2,
        )
    print(f"Saved {out_path}")

    # Write multiseed_table.md
    table_path = os.path.join(config.RESULTS_DIR, "multiseed_table.md")
    tables = []
    for dataset in [config.FRONTIER_DATASET]:
        t1 = render_epsilon_by_fraction_table(aggregated, dataset)
        t2 = render_crypto_table(aggregated, dataset)
        if t1:
            tables.append(t1)
        if t2:
            tables.append(t2)

    if tables:
        table_content = "\n\n".join(tables) + "\n"
        with open(table_path, "w") as f:
            f.write(table_content)
        print(f"Saved {table_path}")
        print("\n" + table_content)


if __name__ == "__main__":
    main()
