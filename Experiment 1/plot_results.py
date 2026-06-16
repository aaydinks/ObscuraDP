"""Plot the epsilon-vs-recall frontier (FedProx vs FedAvg) and a metrics table."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stroke_fl import config

RES = os.path.join(config.RESULTS_DIR, "results.json")


def main():
    with open(RES) as f:
        R = json.load(f)

    # robust key lookup (json keys are the python str() of the float)
    def get(strat, eps, metric):
        d = R["sweep"][strat]
        for k in d:
            if abs(float(k) - eps) < 1e-9:
                return d[k][metric]
        return None

    eps_grid = sorted({float(k) for k in R["sweep"]["FedProx"]})

    fp_recall = [get("FedProx", e, "recall") for e in eps_grid]
    fa_recall = [get("FedAvg", e, "recall") for e in eps_grid]

    # ---- Figure 1: the recall frontier ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(eps_grid, fp_recall, "o-", color="#1f77b4", lw=2.2, ms=7,
            label="FedProx + SMOTETomek (μ=0.01)")
    ax.plot(eps_grid, fa_recall, "s--", color="#d62728", lw=2.2, ms=7,
            label="FedAvg + SMOTETomek")

    # No-SMOTE across epsilon: recall stays ~0 regardless of budget.
    ns = R.get("no_smote_sweep") or {}
    if ns:
        ns_eps = sorted(float(k) for k in ns)
        ns_rec = [ns[k if k in ns else f"{e:.1f}"]["recall"]
                  for k, e in zip([str(x) for x in ns_eps], ns_eps)]
        ax.plot(ns_eps, ns_rec, "x:", color="gray", lw=1.8, ms=9,
                label="FedAvg, NO SMOTETomek (raw imbalance)")

    base = R.get("baseline_no_smote")
    if base and not ns:
        ax.axhline(base["recall"], color="gray", ls=":", lw=1.8,
                   label=f"FedAvg, NO SMOTE (baseline, recall={base['recall']:.2f})")

    ax.axvline(9.0, color="green", ls="-.", alpha=0.5, lw=1.3)
    ax.annotate("ε≈9 knee:\nFedAvg recall\ndegrades here", xy=(9.0, 0.62),
                color="green", fontsize=9, ha="center")

    ax.set_xscale("log")
    ax.set_xticks([0.5, 1.0, 2.0, 5.0, 9.0, 15.0, 50.0])  # subset to avoid crowding
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()
    ax.set_xlabel("Privacy budget  ε  (log scale)", fontsize=12)
    ax.set_ylabel("Recall (stroke = positive class)", fontsize=12)
    ax.set_title("ε-vs-Recall Frontier on Non-IID Stroke Data\n"
                 "FedProx vs FedAvg under DP-SGD", fontsize=12.5)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9.5)
    fig.tight_layout()
    out1 = os.path.join(config.RESULTS_DIR, "recall_vs_epsilon.png")
    fig.savefig(out1, dpi=150)
    print("saved", out1)

    # ---- Figure 2: recall / precision / f1 / auc panels ----
    fig2, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    for ax, metric in zip(axes, ["recall", "precision", "f1", "auc"]):
        ax.plot(eps_grid, [get("FedProx", e, metric) for e in eps_grid],
                "o-", color="#1f77b4", label="FedProx")
        ax.plot(eps_grid, [get("FedAvg", e, metric) for e in eps_grid],
                "s--", color="#d62728", label="FedAvg")
        ax.set_xscale("log")
        ax.set_xticks([0.5, 2.0, 9.0, 50.0])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.minorticks_off()
        ax.set_xlabel("ε")
        ax.set_title(metric.capitalize())
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)
    fig2.suptitle("Metrics vs Privacy Budget (with SMOTETomek)", fontsize=13)
    fig2.tight_layout()
    out2 = os.path.join(config.RESULTS_DIR, "metrics_vs_epsilon.png")
    fig2.savefig(out2, dpi=150)
    print("saved", out2)

    # ---- console table ----
    print("\n  eps  | FP recall  FA recall | FP auc   FA auc | FP prec  FA prec")
    print("-" * 70)
    for e in eps_grid:
        print(f"{e:5.1f}  |  {get('FedProx', e, 'recall'):.3f}      "
              f"{get('FedAvg', e, 'recall'):.3f}    |  "
              f"{get('FedProx', e, 'auc'):.3f}   {get('FedAvg', e, 'auc'):.3f}  |  "
              f"{get('FedProx', e, 'precision'):.3f}    "
              f"{get('FedAvg', e, 'precision'):.3f}")
    if base:
        print("-" * 70)
        print(f"baseline (no SMOTE, eps=9): recall={base['recall']:.3f} "
              f"precision={base['precision']:.3f} auc={base.get('auc', float('nan')):.3f} "
              f"pred_pos_rate={base['pred_pos_rate']:.3f}")
    if ns:
        print("no-SMOTE across eps:",
              {k: round(ns[k]["recall"], 3) for k in sorted(ns, key=float)})
    # FedProx advantage summary
    gaps = [get("FedProx", e, "auc") - get("FedAvg", e, "auc") for e in eps_grid]
    print(f"\nMean FedProx-FedAvg AUC gap: {sum(gaps)/len(gaps):+.4f}  "
          f"(positive => FedProx better)")


if __name__ == "__main__":
    import matplotlib.ticker  # noqa
    main()
