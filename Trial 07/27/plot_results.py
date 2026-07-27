"""Plot Trial 07/27 frontier, overhead, losslessness, and sensitivity results.

Figures demonstrating the three claims:
  1. Separability: utility depends only on ε, cost only on p (frontier.png).
  2. Losslessness under DP: CKKS/SPE aggregation of DP-noised updates match
     plaintext to ~1e-7, orders below the DP noise floor (lossless.png).
  3. Sensitivity fidelity: top-p mask ranking survives DP clipping+noise
     (sensitivity.png).

Also plots arms (performance matrix across stacks and datasets) and overhead
composition (crypto scaling with model size and encryption ratio).

Each figure is skipped gracefully with a printed note if its input JSON is
missing (drivers finish at different times).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from obscura import config

RES = config.RESULTS_DIR


def _load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def plot_frontier():
    """Two-panel frontier: utility vs ε (left), crypto cost vs p (right).

    Left panel plots primary metric vs epsilon for each encryption ratio p.
    All lines should overlap, proving utility is independent of p.
    Right panel plots crypto seconds vs p for each epsilon.
    All lines should overlap, proving cost is independent of epsilon.
    """
    d = _load("frontier.json")
    if not d:
        print("SKIPPED plot_frontier (no frontier.json)")
        return

    rows = d["frontier"]
    primary = config.LR.keys().__iter__().__next__()  # placeholder dataset
    if rows:
        primary = config.primary_metric(rows[0].get("dataset", "stroke"))

    epsilons = sorted(set(r["epsilon"] for r in rows))
    fractions = sorted(set(r["spe_fraction"] for r in rows))

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(fractions)))

    for frac, color in zip(fractions, colors):
        frac_rows = [r for r in rows if r["spe_fraction"] == frac]
        frac_rows_sorted = sorted(frac_rows, key=lambda r: r["epsilon"])
        eps_sorted = sorted(set(r["epsilon"] for r in frac_rows))
        metrics = [r["metrics"].get(primary, 0.0) for r in frac_rows_sorted]
        ax_left.plot(eps_sorted, metrics, "o-", color=color, label=f"p={frac:.2f}",
                     linewidth=2, markersize=6)

    ax_left.set_xscale("log")
    ax_left.set_xlabel("Privacy budget  ε  (log scale)", fontsize=11)
    ax_left.set_ylabel(f"Test {primary}", fontsize=11)
    ax_left.set_title("Utility: independent of p", fontsize=12)
    ax_left.grid(True, which="both", alpha=0.3)
    ax_left.legend(fontsize=9)

    for eps, color in zip(epsilons, plt.cm.tab10(np.linspace(0, 1, len(epsilons)))):
        eps_rows = [r for r in rows if r["epsilon"] == eps]
        eps_rows_sorted = sorted(eps_rows, key=lambda r: r["spe_fraction"])
        fracs_sorted = sorted(set(r["spe_fraction"] for r in eps_rows))
        crypto_secs = [r["crypto"].get("crypto_sec", 0.0) for r in eps_rows_sorted]
        ax_right.plot(fracs_sorted, crypto_secs, "s-", color=color,
                      label=f"ε={eps:.1f}", linewidth=2, markersize=6)

    ax_right.set_xlabel("Encryption ratio  p", fontsize=11)
    ax_right.set_ylabel("Crypto time / round (s)", fontsize=11)
    ax_right.set_title("Cost: independent of ε", fontsize=12)
    ax_right.grid(True, alpha=0.3)
    ax_right.legend(fontsize=9)

    fig.suptitle("Trial 07/27 Separability: utility ⊥ cost", fontsize=13, y=1.00)
    fig.tight_layout()
    p = os.path.join(RES, "frontier.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"wrote {p}")


def plot_arms():
    """Grouped bar chart of primary metric per stack and algorithm."""
    d = _load("arms.json")
    if not d:
        print("SKIPPED plot_arms (no arms.json)")
        return

    rows = d["arms"]
    if not rows:
        return

    datasets = sorted(set(r["dataset"] for r in rows))
    stacks = sorted(set(r["stack"] for r in rows))
    algos = sorted(set(r["algo"] for r in rows))

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 + 4 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        primary = config.primary_metric(dataset)
        dataset_rows = [r for r in rows if r["dataset"] == dataset]

        x = 0
        bar_width = 0.35
        x_labels = []
        x_ticks = []

        for stack in stacks:
            stack_rows = [r for r in dataset_rows if r["stack"] == stack]
            if not stack_rows:
                continue

            for algo in algos:
                algo_rows = [r for r in stack_rows if r["algo"] == algo]
                if not algo_rows:
                    continue

                metric_val = algo_rows[0]["metrics"].get(primary, 0.0)
                color = "#1f77b4" if algo == "fedprox" else "#d62728"
                bar = ax.bar(x, metric_val, bar_width, color=color, edgecolor="black",
                             linewidth=0.6)
                ax.text(x, metric_val + 0.01, f"{metric_val:.3f}", ha="center",
                        va="bottom", fontsize=8)
                x_ticks.append(x)
                x_labels.append(f"{stack}\n{algo[:6]}")
                x += bar_width

            x += 0.1

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=8, rotation=0)
        ax.set_ylabel(f"Test {primary}", fontsize=10)
        ax.set_title(f"{dataset}", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Test Metric by Stack and Algorithm", fontsize=13)
    fig.tight_layout()
    p = os.path.join(RES, "arms.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"wrote {p}")


def plot_lossless():
    """Log-scale scatter of max_abs_dev per epsilon for HE and SPE."""
    d = _load("lossless.json")
    if not d:
        print("SKIPPED plot_lossless (no lossless.json)")
        return

    rows = d["lossless"]
    if not rows:
        return

    epsilons = sorted(set(r["epsilon"] for r in rows))
    modes = sorted(set(r["mode"] for r in rows))

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"he": "#d7301f", "spe": "#31a354"}
    for mode in modes:
        mode_rows = [r for r in rows if r["mode"] == mode]
        mode_rows_sorted = sorted(mode_rows, key=lambda r: r["epsilon"])
        eps_sorted = sorted(set(r["epsilon"] for r in mode_rows))
        devs = [r["max_abs_dev"] for r in mode_rows_sorted]
        ax.loglog(eps_sorted, devs, "o-", color=colors.get(mode, "gray"),
                  label=mode.upper(), linewidth=2, markersize=7)

    for r in rows:
        if r.get("noise_std", 0) > 0:
            ax.axhline(r["noise_std"], color="gray", ls="--", alpha=0.3, linewidth=1)
            break

    ax.set_xlabel("Privacy budget  ε  (log scale)", fontsize=11)
    ax.set_ylabel("Max abs deviation from plaintext (log scale)", fontsize=11)
    ax.set_title("Losslessness: HE/SPE aggregation error << DP noise floor",
                 fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    p = os.path.join(RES, "lossless.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"wrote {p}")


def plot_sensitivity():
    """Jaccard and Spearman rank correlation vs epsilon, one line per p."""
    d = _load("sensitivity.json")
    if not d:
        print("SKIPPED plot_sensitivity (no sensitivity.json)")
        return

    rows = d["sensitivity"]
    if not rows:
        return

    fractions = sorted(set(r["fraction"] for r in rows if r.get("fraction")))
    epsilons = sorted(set(r["epsilon"] for r in rows))

    fig, (ax_j, ax_s) = plt.subplots(1, 2, figsize=(13, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(fractions)))

    for frac, color in zip(fractions, colors):
        frac_rows = [r for r in rows if r.get("fraction") == frac]
        frac_rows_sorted = sorted(frac_rows, key=lambda r: r["epsilon"])
        eps_sorted = sorted(set(r["epsilon"] for r in frac_rows))

        jaccard = [r.get("jaccard", 0.0) for r in frac_rows_sorted]
        spearman = [r.get("spearman", 0.0) for r in frac_rows_sorted]

        ax_j.plot(eps_sorted, jaccard, "o-", color=color, label=f"p={frac:.2f}",
                  linewidth=2, markersize=6)
        ax_s.plot(eps_sorted, spearman, "o-", color=color, label=f"p={frac:.2f}",
                  linewidth=2, markersize=6)

    random_rows = [r for r in rows if "random_jaccard" in r]
    if random_rows:
        random_j = random_rows[0].get("random_jaccard", 0.0)
        ax_j.axhline(random_j, color="gray", ls="--", linewidth=1.5,
                     label=f"random baseline={random_j:.3f}")

    ax_j.set_xscale("log")
    ax_j.set_xlabel("Privacy budget  ε  (log scale)", fontsize=11)
    ax_j.set_ylabel("Mask Jaccard agreement", fontsize=11)
    ax_j.set_title("Clean vs DP mask agreement", fontsize=12)
    ax_j.grid(True, which="both", alpha=0.3)
    ax_j.legend(fontsize=9)

    ax_s.set_xscale("log")
    ax_s.set_xlabel("Privacy budget  ε  (log scale)", fontsize=11)
    ax_s.set_ylabel("Spearman rank correlation", fontsize=11)
    ax_s.set_title("Score rank correlation (clean vs DP)", fontsize=12)
    ax_s.grid(True, which="both", alpha=0.3)
    ax_s.legend(fontsize=9)

    fig.suptitle("Sensitivity fidelity under DP noise", fontsize=13)
    fig.tight_layout()
    p = os.path.join(RES, "sensitivity.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"wrote {p}")


def plot_overhead():
    """Three panels: (a) crypto_sec and payload_ratio vs p, (b) log-log
    crypto_sec vs n_params, (c) dp_multiplier vs n_params."""
    d = _load("overhead.json")
    if not d:
        print("SKIPPED plot_overhead (no overhead.json)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    crypto_vs_p = d.get("crypto_vs_p", [])
    if crypto_vs_p:
        fracs = sorted(set(r.get("fraction", 0.0) for r in crypto_vs_p))
        for frac in fracs:
            frac_rows = sorted([r for r in crypto_vs_p if r.get("fraction") == frac],
                               key=lambda r: r.get("fraction"))
            if frac_rows:
                fracs_sorted = sorted(set(r.get("fraction") for r in frac_rows))
                crypto_secs = [r.get("crypto_sec", 0.0) for r in frac_rows]
                payload = [r.get("payload_ratio", 0.0) for r in frac_rows]
                ax = axes[0]
                ax.plot(fracs_sorted, crypto_secs, "o-", label=f"crypto_sec", linewidth=2)
                ax2 = ax.twinx()
                ax2.plot(fracs_sorted, payload, "s--", label=f"payload_ratio", linewidth=2)
                ax.set_xlabel("Encryption ratio  p", fontsize=10)
                ax.set_ylabel("Crypto time (s)", fontsize=10)
                ax2.set_ylabel("Payload ratio", fontsize=10)
                ax.set_title("(a) Overhead vs encryption ratio", fontsize=11)
                ax.grid(alpha=0.3)
                break

    crypto_vs_size = d.get("crypto_vs_size", [])
    if crypto_vs_size:
        rows_10 = [r for r in crypto_vs_size if abs(r.get("fraction", 1.0) - 1.0) < 0.01]
        rows_spe = [r for r in crypto_vs_size
                    if abs(r.get("fraction", 0.0) - config.SPE_DEFAULT) < 0.01]
        ax = axes[1]
        if rows_10:
            rows_10_sorted = sorted(rows_10, key=lambda r: r["n_params"])
            n_p = [r["n_params"] for r in rows_10_sorted]
            crypto = [r.get("crypto_sec", 1e-6) for r in rows_10_sorted]
            ax.loglog(n_p, crypto, "o-", label="p=1.0 (full HE)", linewidth=2, markersize=6)

        if rows_spe:
            rows_spe_sorted = sorted(rows_spe, key=lambda r: r["n_params"])
            n_p_spe = [r["n_params"] for r in rows_spe_sorted]
            crypto_spe = [r.get("crypto_sec", 1e-6) for r in rows_spe_sorted]
            ax.loglog(n_p_spe, crypto_spe, "s--", label=f"p={config.SPE_DEFAULT}",
                      linewidth=2, markersize=6)

        if rows_10:
            n_min, n_max = min(n_p), max(n_p)
            c_min = crypto[0] * (n_max / n_min) ** 1.0
            ax.loglog([n_min, n_max], [crypto[0], c_min], "x:", color="gray", alpha=0.5,
                      linewidth=1.5, label="O(n) reference")

        ax.set_xlabel("Model size (parameters, log scale)", fontsize=10)
        ax.set_ylabel("Crypto time (s, log scale)", fontsize=10)
        ax.set_title("(b) Overhead scales O(n)", fontsize=11)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)

    dp_tax = d.get("dp_tax", [])
    if dp_tax:
        rows_sorted = sorted(dp_tax, key=lambda r: r.get("n_params", 0))
        n_p = [r["n_params"] for r in rows_sorted]
        multipliers = [r.get("dp_multiplier", 1.0) for r in rows_sorted]
        ax = axes[2]
        ax.plot(n_p, multipliers, "D-", color="#31a354", linewidth=2, markersize=6)
        ax.set_xlabel("Model size (parameters)", fontsize=10)
        ax.set_ylabel("DP compute tax (multiplier)", fontsize=10)
        ax.set_title("(c) DP-SGD overhead", fontsize=11)
        ax.grid(alpha=0.3)

    fig.suptitle("Overhead composition: crypto vs p and model size, DP tax", fontsize=13)
    fig.tight_layout()
    p = os.path.join(RES, "overhead.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"wrote {p}")


def main():
    """Plot all available results."""
    print("\n--- Plotting Trial 07/27 results ---")
    plot_frontier()
    plot_arms()
    plot_lossless()
    plot_sensitivity()
    plot_overhead()
    print("--- Done ---\n")


if __name__ == "__main__":
    main()
