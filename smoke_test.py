"""Quick end-to-end validation on a tiny config (not the real sweep)."""
import time
from stroke_fl import config, data, train

t0 = time.time()
X_tr, y_tr, X_te, y_te, info = data.split_and_scale()
print("DATA:", {k: v for k, v in info.items() if not k.startswith(("X_", "y_"))})

parts, pinfo = data.dirichlet_partition(
    y_tr, config.NUM_CLIENTS, config.DIRICHLET_ALPHA,
    config.MIN_POS_PER_CLIENT, config.SEED,
)
print("PARTITION:", pinfo)
print(f"data+partition: {time.time()-t0:.1f}s")

# Tiny config: 2 rounds, 1 local epoch, 2 epsilons, both strategies.
for label, mu, smote in [("FedAvg+SMOTE", 0.0, True), ("FedProx+SMOTE", config.MU, True)]:
    for eps in [1.0, 9.0]:
        t1 = time.time()
        m = train.run_experiment(
            epsilon=eps, mu=mu, use_smote=smote, partition_idx=parts,
            X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
            in_features=info["n_features"], rounds=2, local_epochs=1, log=None,
        )
        print(f"{label:16s} eps={eps:4.1f}  recall={m['recall']:.3f} "
              f"prec={m['precision']:.3f} f1={m['f1']:.3f} "
              f"ppr={m['pred_pos_rate']:.2f}  [{time.time()-t1:.1f}s]")

print(f"TOTAL smoke: {time.time()-t0:.1f}s")
