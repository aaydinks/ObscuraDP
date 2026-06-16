"""Run the full sweep for ONE seed -> results/seed_<seed>.json.

Used by the multi-seed driver to estimate mean +/- std of the frontier.
Each seed gets its own data split, non-IID partition, SMOTE draw, init and
DP noise, so averaging across seeds gives an honest variance estimate.
"""
import json
import os
import sys

import torch
torch.set_num_threads(2)          # keep parallel seed-processes from oversubscribing

from stroke_fl import config, data, train


def main(seed):
    config.SEED = seed
    out = os.path.join(config.RESULTS_DIR, f"seed_{seed}.json")

    X_tr, y_tr, X_te, y_te, info = data.split_and_scale()
    nf = info["n_features"]
    parts, pinfo = data.dirichlet_partition(
        y_tr, config.NUM_CLIENTS, config.DIRICHLET_ALPHA,
        config.MIN_POS_PER_CLIENT, seed,
    )

    res = {"seed": seed,
           "data_info": {k: v for k, v in info.items()
                         if not k.startswith(("X_", "y_"))},
           "partition": pinfo,
           "baseline_no_smote": None,
           "sweep": {"FedProx": {}, "FedAvg": {}}}

    def save():
        with open(out, "w") as f:
            json.dump(res, f, indent=2)

    # Baseline: FedAvg + DP-SGD, no SMOTE (expect recall ~0).
    m = train.run_experiment(
        epsilon=9.0, mu=0.0, use_smote=False, partition_idx=parts,
        X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
        in_features=nf, log=None,
    )
    m["epsilon"] = 9.0
    res["baseline_no_smote"] = m
    print(f"[seed {seed}] baseline recall={m['recall']:.3f}", flush=True)
    save()

    for strat, mu in [("FedProx", config.MU), ("FedAvg", 0.0)]:
        for eps in config.EPSILON_GRID:
            m = train.run_experiment(
                epsilon=eps, mu=mu, use_smote=True, partition_idx=parts,
                X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
                in_features=nf, log=None,
            )
            m["epsilon"] = eps
            res["sweep"][strat][str(eps)] = m
            print(f"[seed {seed}] {strat:7s} eps={eps:<4} "
                  f"recall={m['recall']:.3f} auc={m['auc']:.3f}", flush=True)
            save()

    print(f"[seed {seed}] DONE -> {out}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
