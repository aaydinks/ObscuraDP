"""Multi-seed campaign: mean +/- std of the five-way comparison over N seeds.

A single 854-sample test set resolves accuracy only to ~0.12% per sample, so the
14.13 vs 8.65 DP gap sits inside single-seed noise. Averaging over independent
seeds (each with its own split, IID partition and DP noise draw) smooths that out
and yields an honest error bar on the epsilon-ordering -- the same methodology as
Experiment 1.  Usage:  python run_multiseed.py [seed1 seed2 ...]
"""
import json
import os
import sys

import numpy as np

from credit_fl import config
import run_experiments as R

SEEDS = [42, 43, 44, 45, 46]


def main(seeds):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    per_seed = {}
    for s in seeds:
        res = R.run_all(seed=s)
        per_seed[s] = {name: {k: v for k, v in e.items() if k != "curve"}
                       for name, e in res["experiments"].items()}
        accs = {n: per_seed[s][n]["accuracy"] for n in R.ROW_ORDER}
        print(f"seed {s}: " + "  ".join(f"{n.split(' ')[0][:4]}"
              f"{'' if 'eps' not in n else n.split('=')[1].rstrip(')')}"
              f"={a*100:.2f}" for n, a in accs.items()), flush=True)

    # aggregate mean +/- std per configuration / metric
    agg = {}
    for name in R.ROW_ORDER:
        rows = [per_seed[s][name] for s in seeds]
        agg[name] = {}
        for metric in ["accuracy", "loss", "sec_per_round", "agg_ms_per_round"]:
            vals = np.array([r[metric] for r in rows], float)
            agg[name][metric] = {"mean": float(vals.mean()), "std": float(vals.std())}
        if rows[0].get("ciphertext_mb"):
            cts = np.array([r["ciphertext_mb"] for r in rows], float)
            agg[name]["ciphertext_mb"] = {"mean": float(cts.mean()),
                                          "std": float(cts.std())}
        if "epsilon_actual" in rows[0]:
            eps = np.array([r["epsilon_actual"] for r in rows], float)
            agg[name]["epsilon_actual"] = {"mean": float(eps.mean()),
                                           "std": float(eps.std())}

    out = {"seeds": seeds, "per_seed": per_seed, "aggregate": agg}
    with open(os.path.join(config.RESULTS_DIR, "results_multiseed.json"), "w") as f:
        json.dump(out, f, indent=2)

    table = render_multiseed(agg, seeds)
    print("\n" + table)
    with open(os.path.join(config.RESULTS_DIR, "results_multiseed_table.md"), "w") as f:
        f.write(table)
    print(f"saved -> {config.RESULTS_DIR}/results_multiseed.json + table")


def render_multiseed(agg, seeds):
    lines = [f"### Mean ± std over {len(seeds)} seeds ({seeds})", "",
             "| Configuration | Accuracy | Loss | Agg latency/round (ms) "
             "| Ciphertext size (MB) |", "|---|---:|---:|---:|---:|"]
    for name in R.ROW_ORDER:
        a = agg[name]
        ct = (f"{a['ciphertext_mb']['mean']:.3f}" if "ciphertext_mb" in a else "—")
        lines.append(
            f"| {name} | {a['accuracy']['mean']*100:.2f}% ± {a['accuracy']['std']*100:.2f} "
            f"| {a['loss']['mean']:.4f} | {a['agg_ms_per_round']['mean']:.3f} | {ct} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    seeds = [int(x) for x in sys.argv[1:]] or SEEDS
    main(seeds)
