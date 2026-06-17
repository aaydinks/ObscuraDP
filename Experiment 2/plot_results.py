"""Figures for the three-way FL comparison.

  1. accuracy_comparison.png -- accuracy per configuration with multi-seed error
     bars; the DP arms ordered by epsilon, HE overlaid on Standard.
  2. he_overhead.png -- the HE cost story: ciphertext blow-up and aggregation
     latency vs the plaintext baseline (log scale).

Reads results_multiseed.json if present (preferred, with error bars), else the
single-seed results.json.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from credit_fl import config

ROW_ORDER = ["Standard FL", "DP-FL (eps=14.13)", "DP-FL (eps=8.65)",
             "DP-FL (eps=5.74)", "HE-FL"]
COLORS = ["#2c7fb8", "#7fcdbb", "#41b6c4", "#1d91c0", "#f03b20"]
RES = config.RESULTS_DIR


def load():
    ms = os.path.join(RES, "results_multiseed.json")
    if os.path.exists(ms):
        d = json.load(open(ms))
        agg = d["aggregate"]
        acc = {n: (agg[n]["accuracy"]["mean"] * 100, agg[n]["accuracy"]["std"] * 100)
               for n in ROW_ORDER}
        ct = agg["HE-FL"]["ciphertext_mb"]["mean"]
        aggl = {n: agg[n]["agg_ms_per_round"]["mean"] for n in ROW_ORDER}
        sub = f"mean ± std over {len(d['seeds'])} seeds"
    else:
        d = json.load(open(os.path.join(RES, "results.json")))
        e = d["experiments"]
        acc = {n: (e[n]["accuracy"] * 100, 0.0) for n in ROW_ORDER}
        ct = e["HE-FL"]["ciphertext_mb"]
        aggl = {n: e[n]["agg_ms_per_round"] for n in ROW_ORDER}
        sub = f"single seed ({d['seed']})"
    return acc, ct, aggl, sub


def plot_accuracy(acc, sub):
    fig, ax = plt.subplots(figsize=(8, 5))
    names = ROW_ORDER
    means = [acc[n][0] for n in names]
    stds = [acc[n][1] for n in names]
    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=5,
                  color=COLORS, edgecolor="black", linewidth=0.6)
    std_acc = acc["Standard FL"][0]
    ax.axhline(std_acc, ls="--", c="#2c7fb8", lw=1, alpha=0.7,
               label=f"Standard FL = {std_acc:.2f}%")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.15, f"{m:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["Standard\nFL", "DP-FL\nε=14.13", "DP-FL\nε=8.65",
                        "DP-FL\nε=5.74", "HE-FL"])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(min(means) - 4, max(means) + 3)
    ax.set_title(f"Credit-risk FL: accuracy vs privacy mechanism\n({sub})")
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(RES, "accuracy_comparison.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)


def plot_overhead(ct, aggl, sub):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    pt_kb = 3.72  # plaintext float32 update per client
    a1.bar(["Plaintext\nupdate", "CKKS\nciphertext"], [pt_kb / 1024, ct],
           color=["#2c7fb8", "#f03b20"], edgecolor="black")
    a1.set_yscale("log")
    a1.set_ylabel("Per-client payload (MB, log)")
    a1.set_title(f"Communication payload ({ct*1024/pt_kb:.0f}× blow-up)")
    for i, v in enumerate([pt_kb / 1024, ct]):
        a1.text(i, v, f"{v*1024:.1f} KB" if v < 1 else f"{v:.2f} MB",
                ha="center", va="bottom", fontsize=9)

    pt_agg = aggl["Standard FL"]
    he_agg = aggl["HE-FL"]
    a2.bar(["Plaintext\nmean", "CKKS encrypt+\nsum+decrypt"], [pt_agg, he_agg],
           color=["#2c7fb8", "#f03b20"], edgecolor="black")
    a2.set_yscale("log")
    a2.set_ylabel("Aggregation latency / round (ms, log)")
    a2.set_title(f"Aggregation latency ({he_agg/max(pt_agg,1e-6):.0f}× slower)")
    for i, v in enumerate([pt_agg, he_agg]):
        a2.text(i, v, f"{v:.3f} ms" if v < 1 else f"{v:.1f} ms",
                ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"HE-FL overhead vs plaintext baseline ({sub})")
    fig.tight_layout()
    p = os.path.join(RES, "he_overhead.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    acc, ct, aggl, sub = load()
    plot_accuracy(acc, sub)
    plot_overhead(ct, aggl, sub)
