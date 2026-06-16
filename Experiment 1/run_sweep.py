"""Main experiment runner: baseline failure case + the epsilon-vs-recall sweep.

Outputs results/results.json with:
  - baseline: FedAvg + DP-SGD WITHOUT SMOTETomek (expected near-zero recall)
  - sweep:    FedProx and FedAvg (both WITH SMOTETomek) across the epsilon grid
"""
import json
import os
import time

from stroke_fl import config, data, train


def main():
    t_start = time.time()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, "results.json")

    # --- Data ---
    X_tr, y_tr, X_te, y_te, info = data.split_and_scale()
    nf = info["n_features"]
    parts, pinfo = data.dirichlet_partition(
        y_tr, config.NUM_CLIENTS, config.DIRICHLET_ALPHA,
        config.MIN_POS_PER_CLIENT, config.SEED,
    )
    print("=" * 70)
    print("DATASET:", {k: v for k, v in info.items() if not k.startswith(("X_", "y_"))})
    print("PARTITION:", pinfo)
    print("EPSILON GRID:", config.EPSILON_GRID)
    print("=" * 70)

    results = {
        "meta": {
            "data_info": {k: v for k, v in info.items()
                          if not k.startswith(("X_", "y_"))},
            "partition": pinfo,
            "config": {
                "num_clients": config.NUM_CLIENTS, "rounds": config.ROUNDS,
                "local_epochs": config.LOCAL_EPOCHS, "lr": config.LR,
                "batch_size": config.BATCH_SIZE, "mu": config.MU,
                "max_grad_norm": config.MAX_GRAD_NORM,
                "target_delta": config.TARGET_DELTA,
                "dirichlet_alpha": config.DIRICHLET_ALPHA,
                "epsilon_grid": config.EPSILON_GRID,
            },
        },
        "baseline_no_smote": None,
        "no_smote_sweep": {},          # FedAvg, NO SMOTE, across a few eps
        "sweep": {"FedProx": {}, "FedAvg": {}},
    }
    NO_SMOTE_EPS = [0.5, 9.0, 50.0]

    def save():
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    # --- Baseline failure case: FedAvg + DP-SGD, NO SMOTE (raw imbalance) ---
    # Use a mid epsilon so privacy isn't the cause of the failure.
    print("\n[BASELINE] FedAvg + DP-SGD, NO SMOTETomek (eps=9.0) -- expect ~0 recall")
    t = time.time()
    m = train.run_experiment(
        epsilon=9.0, mu=0.0, use_smote=False, partition_idx=parts,
        X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te, in_features=nf,
    )
    m["epsilon"] = 9.0
    results["baseline_no_smote"] = m
    print(f"  -> recall={m['recall']:.3f} precision={m['precision']:.3f} "
          f"f1={m['f1']:.3f} pred_pos_rate={m['pred_pos_rate']:.3f} "
          f"[{time.time()-t:.0f}s]")
    save()

    # --- No-SMOTE across epsilon: recall should stay ~0 regardless of budget,
    #     proving imbalance (not privacy) is what breaks recall. ---
    for eps in NO_SMOTE_EPS:
        print(f"\n[NO-SMOTE FedAvg] eps={eps} -- expect recall ~0")
        t = time.time()
        m = train.run_experiment(
            epsilon=eps, mu=0.0, use_smote=False, partition_idx=parts,
            X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te, in_features=nf,
        )
        m["epsilon"] = eps
        results["no_smote_sweep"][str(eps)] = m
        print(f"  -> recall={m['recall']:.3f} auc={m['auc']:.3f} "
              f"f1={m['f1']:.3f} [{time.time()-t:.0f}s]")
        save()

    # --- epsilon sweep: FedProx vs FedAvg, both WITH SMOTETomek ---
    strategies = [("FedProx", config.MU), ("FedAvg", 0.0)]
    for strat, mu in strategies:
        for eps in config.EPSILON_GRID:
            print(f"\n[{strat}+SMOTE] eps={eps}")
            t = time.time()
            m = train.run_experiment(
                epsilon=eps, mu=mu, use_smote=True, partition_idx=parts,
                X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
                in_features=nf,
            )
            m["epsilon"] = eps
            results["sweep"][strat][str(eps)] = m
            print(f"  -> recall={m['recall']:.3f} precision={m['precision']:.3f} "
                  f"f1={m['f1']:.3f} auc={m['auc']:.3f} [{time.time()-t:.0f}s]")
            save()

    print("\n" + "=" * 70)
    print(f"DONE in {(time.time()-t_start)/60:.1f} min -> {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
