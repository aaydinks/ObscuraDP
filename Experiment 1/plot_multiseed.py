"""Plot the multi-seed mean +/- std frontier (recall and AUC)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from stroke_fl import config

RES = os.path.join(config.RESULTS_DIR, "results_multiseed.json")


def main():
    R = json.load(open(RES))
    eps = config.EPSILON_GRID
    n = R["n_seeds"]

    def band(ax, strat, metric, color, marker, label):
        mean = [R["sweep"][strat][str(e)][metric]["mean"] for e in eps]
        std = [R["sweep"][strat][str(e)][metric]["std"] for e in eps]
        lo = [m - s for m, s in zip(mean, std)]
        hi = [m + s for m, s in zip(mean, std)]
        ax.plot(eps, mean, marker + "-", color=color, lw=2.2, ms=7, label=label)
        ax.fill_between(eps, lo, hi, color=color, alpha=0.15)
        return mean

    # ---- Figure: recall frontier with std bands ----
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    band(ax, "FedProx", "recall", "#1f77b4", "o",
         f"FedProx + SMOTETomek (μ=0.01)")
    band(ax, "FedAvg", "recall", "#d62728", "s", "FedAvg + SMOTETomek")

    br = R["baseline_no_smote"]["recall"]
    ax.axhline(br["mean"], color="gray", ls=":", lw=1.8,
               label=f"FedAvg, NO SMOTETomek (recall={br['mean']:.2f})")

    ax.axvline(9.0, color="green", ls="-.", alpha=0.5, lw=1.3)
    ax.annotate("ε≈9 knee\n(FedProx–FedAvg\ngap widens beyond)", xy=(9.0, 0.30),
                color="green", fontsize=9, ha="center")

    ax.set_xscale("log")
    ax.set_xticks([0.5, 1.0, 2.0, 5.0, 9.0, 15.0, 50.0])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()
    ax.set_xlabel("Privacy budget  ε  (log scale)", fontsize=12)
    ax.set_ylabel("Recall (stroke = positive class)", fontsize=12)
    ax.set_title(f"ε-vs-Recall Frontier on Non-IID Stroke Data — "
                 f"mean ± std over {n} seeds\nFedProx vs FedAvg under DP-SGD",
                 fontsize=12)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9.5)
    fig.tight_layout()
    p1 = os.path.join(config.RESULTS_DIR, "recall_vs_epsilon_multiseed.png")
    fig.savefig(p1, dpi=150)
    print("saved", p1)

    # ---- Figure: 4 metric panels with bands ----
    fig2, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    for ax, met in zip(axes, ["recall", "precision", "f1", "auc"]):
        band(ax, "FedProx", met, "#1f77b4", "o", "FedProx")
        band(ax, "FedAvg", met, "#d62728", "s", "FedAvg")
        ax.set_xscale("log")
        ax.set_xticks([0.5, 2.0, 9.0, 50.0])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.minorticks_off()
        ax.set_xlabel("ε")
        ax.set_title(met.capitalize())
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)
    fig2.suptitle(f"Metrics vs Privacy Budget (mean ± std over {n} seeds, "
                  f"with SMOTETomek)", fontsize=13)
    fig2.tight_layout()
    p2 = os.path.join(config.RESULTS_DIR, "metrics_vs_epsilon_multiseed.png")
    fig2.savefig(p2, dpi=150)
    print("saved", p2)


if __name__ == "__main__":
    main()
