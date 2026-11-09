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
from obscura import metrics as metrics_mod

RES = config.RESULTS_DIR


def _load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def plot_frontier():
    """The separability figure: utility vs epsilon (left), crypto cost vs p (right).

    Prefers the aggregated multi-seed file so the headline result carries error
    bars, and falls back to the single-seed frontier when only that exists. The
    left panel's curves should coincide (utility independent of p) and the
    right panel's should too (cost independent of epsilon).
    """
    agg = _load("multiseed.json")
    rows = None
    if agg and agg.get("aggregated"):
        rows = agg["aggregated"]
        n_seeds = max(r.get("n_seeds", 1) for r in rows)
        source = f"{n_seeds} seeds, mean +/- s.d."
        getm = lambda r, k: r.get(f"{k}_mean", np.nan)
        gete = lambda r, k: r.get(f"{k}_std", 0.0)
        get_crypto = lambda r: r.get("crypto_sec_mean", 0.0)
        get_crypto_err = lambda r: r.get("crypto_sec_std", 0.0)
    else:
        d = _load("frontier.json")
        if not d:
            print("SKIPPED plot_frontier (no multiseed.json or frontier.json)")
            return
        rows = d["frontier"]
        source = f"seed {d.get('meta', {}).get('seed', '?')}, single run"
        getm = lambda r, k: r["metrics"].get(k, np.nan)
        gete = lambda r, k: 0.0
        get_crypto = lambda r: r["crypto"].get("crypto_sec", 0.0)
        get_crypto_err = lambda r: 0.0

    if not rows:
        return
    primary = metrics_mod.primary_metric(rows[0].get("dataset", "stroke"))
    epsilons = sorted(set(r["epsilon"] for r in rows))
    fractions = sorted(set(r["spe_fraction"] for r in rows))

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))

    # Offset the overlapping series slightly so a reader can see there really
    # are three of them; without this they hide behind whichever draws last,
    # and "the lines coincide" is the entire point of the panel.
    offs = np.linspace(-0.006, 0.006, max(1, len(fractions)))
    for frac, color, off in zip(fractions, plt.cm.tab10(np.linspace(0, 0.6, len(fractions))), offs):
        sel = sorted([r for r in rows if r["spe_fraction"] == frac],
                     key=lambda r: r["epsilon"])
        ax_left.errorbar([r["epsilon"] for r in sel],
                         [getm(r, primary) + off for r in sel],
                         yerr=[gete(r, primary) for r in sel],
                         fmt="o-", color=color, label=f"p={frac:.2f}",
                         linewidth=2, markersize=6, capsize=3, alpha=0.85)

    ax_left.set_xscale("log")
    ax_left.set_xlabel("Privacy budget  eps  (log scale)", fontsize=11)
    ax_left.set_ylabel(f"Test {primary}  (series offset for visibility)", fontsize=11)
    ax_left.set_title("Utility: depends on eps, not on p", fontsize=12)
    ax_left.grid(True, which="both", alpha=0.3)
    ax_left.legend(fontsize=9)

    for eps, color in zip(epsilons, plt.cm.viridis(np.linspace(0, 0.9, len(epsilons)))):
        sel = sorted([r for r in rows if r["epsilon"] == eps],
                     key=lambda r: r["spe_fraction"])
        ax_right.errorbar([r["spe_fraction"] for r in sel],
                          [get_crypto(r) for r in sel],
                          yerr=[get_crypto_err(r) for r in sel],
                          fmt="o-", color=color, label=f"eps={eps:g}",
                          linewidth=1.8, markersize=5, capsize=3, alpha=0.85)

    ax_right.set_xlabel("Encryption ratio  p", fontsize=11)
    ax_right.set_ylabel("Crypto time / round (s)", fontsize=11)
    ax_right.set_title("Cost: depends on p, not on eps", fontsize=12)
    ax_right.grid(alpha=0.3)
    ax_right.legend(fontsize=8, ncol=2)

    fig.suptitle(f"Separability: the utility and cost frontiers are orthogonal "
                 f"axes  ({source})", fontsize=12)
    fig.tight_layout()
    pth = os.path.join(RES, "frontier.png")
    fig.savefig(pth, dpi=150)
    plt.close(fig)
    print(f"wrote {pth}")


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
        primary = metrics_mod.primary_metric(dataset)
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

    fig, ax = plt.subplots(figsize=(10, 6))

    # One series per (mode, p): after de-duplication there are 4 "he" records
    # but 8 "spe" ones (two encryption ratios per epsilon), so grouping by mode
    # alone mismatches the x and y lengths.
    series = sorted(set((r["mode"], r.get("spe_fraction", 1.0)) for r in rows))
    colors = {"he": "#d7301f", "spe": "#31a354"}
    marks = ["o-", "s--", "^:", "d-."]
    for i, (mode, frac) in enumerate(series):
        sel = sorted([r for r in rows
                      if r["mode"] == mode and r.get("spe_fraction", 1.0) == frac],
                     key=lambda r: r["epsilon"])
        ax.loglog([r["epsilon"] for r in sel], [r["max_abs_dev"] for r in sel],
                  marks[i % len(marks)], color=colors.get(mode, "gray"),
                  label=f"{mode.upper()} p={frac:g}", linewidth=2, markersize=7,
                  alpha=0.9)

    # The DP noise standard deviation at each epsilon -- the whole point of the
    # figure is the many-orders-of-magnitude gap between it and the curves above.
    noise = sorted({(r["epsilon"], r["noise_std"]) for r in rows
                    if r.get("noise_std", 0) > 0})
    if noise:
        ax.loglog([e for e, _ in noise], [n for _, n in noise], "v-",
                  color="#404040", linewidth=2, markersize=7,
                  label="DP noise std (floor)")

    # The float32 quantisation floor a plaintext federation already pays.
    ax.axhline(1.49e-08, color="#888888", ls=":", linewidth=1.5,
               label="float32 storage floor")

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
    """Does the SPE mask survive DP noise? Mean +/- std over the repeat draws.

    Left panel is the tie-break-free primary metric (probability the DP-chosen
    mask lands on the same TENSORS as the clean one); right panel is the
    scalar-level Jaccard, which is retained only as a secondary view because
    every scalar inherits its tensor's single score, so most of the mask is
    settled by argpartition tie-breaking rather than by sensitivity.
    """
    d = _load("sensitivity.json")
    if not d:
        print("SKIPPED plot_sensitivity (no sensitivity.json)")
        return

    rows = d["sensitivity"]
    if not rows:
        return

    fractions = sorted(set(r["fraction"] for r in rows if r.get("fraction")))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, max(1, len(fractions))))

    def agg(sel, key):
        """Mean, and standard error, of `key` over the repeat draws."""
        v = np.array([r.get(key, np.nan) for r in sel], dtype=float)
        v = v[~np.isnan(v)]
        if v.size == 0:
            return np.nan, np.nan
        return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0

    fig, (ax_t, ax_j) = plt.subplots(1, 2, figsize=(13, 5))

    for frac, color in zip(fractions, colors):
        eps_sorted = sorted(set(r["epsilon"] for r in rows if r.get("fraction") == frac))
        t_mean, t_err, j_mean, j_err = [], [], [], []
        for e in eps_sorted:
            sel = [r for r in rows
                   if r.get("fraction") == frac and r["epsilon"] == e]
            m, se = agg(sel, "same_tensor_selection")
            t_mean.append(m); t_err.append(se)
            m, se = agg(sel, "jaccard")
            j_mean.append(m); j_err.append(se)
        ax_t.errorbar(eps_sorted, t_mean, yerr=t_err, fmt="o-", color=color,
                      label=f"p={frac:.2f}", linewidth=2, markersize=6, capsize=3)
        ax_j.errorbar(eps_sorted, j_mean, yerr=j_err, fmt="o-", color=color,
                      label=f"p={frac:.2f}", linewidth=2, markersize=6, capsize=3)
        # Analytic p/(2-p) baseline for a random top-p mask, per fraction.
        base = [r.get("random_jaccard_analytic") for r in rows
                if r.get("fraction") == frac and r.get("random_jaccard_analytic")]
        if base:
            ax_j.axhline(base[0], color=color, ls="--", linewidth=1.2, alpha=0.7)

    ax_t.set_xscale("log")
    ax_t.set_xlabel("Privacy budget  eps  (log scale)", fontsize=11)
    ax_t.set_ylabel("P(DP mask selects the same tensors)", fontsize=11)
    ax_t.set_title("Primary: tie-break-free tensor agreement", fontsize=12)
    ax_t.set_ylim(-0.05, 1.05)
    ax_t.grid(True, which="both", alpha=0.3)
    ax_t.legend(fontsize=9)

    ax_j.set_xscale("log")
    ax_j.set_xlabel("Privacy budget  eps  (log scale)", fontsize=11)
    ax_j.set_ylabel("Scalar-level mask Jaccard", fontsize=11)
    ax_j.set_title("Secondary: Jaccard (dashed = p/(2-p) random baseline)",
                   fontsize=12)
    ax_j.set_ylim(-0.05, 1.05)
    ax_j.grid(True, which="both", alpha=0.3)
    ax_j.legend(fontsize=9)

    fig.suptitle("SPE mask fidelity degrades as the privacy budget tightens "
                 "(mean +/- s.e. over repeat draws)", fontsize=12)
    fig.tight_layout()
    pth = os.path.join(RES, "sensitivity.png")
    fig.savefig(pth, dpi=150)
    plt.close(fig)
    print(f"wrote {pth}")


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
        ax = axes[0]
        ax2 = ax.twinx()
        # Records may span several model sizes: a sweep at n_params < CKKS_SLOTS
        # is degenerate (every p costs exactly one ciphertext), so each size is
        # drawn as its own series rather than pooled.
        widths = sorted(set(r.get("width_scale", 1) for r in crypto_vs_p))
        for w in widths:
            sel = sorted([r for r in crypto_vs_p if r.get("width_scale", 1) == w],
                         key=lambda r: r.get("fraction", 0.0))
            if not sel:
                continue
            n_par = sel[0].get("n_params", 0)
            degenerate = n_par <= config.CKKS_SLOTS
            lbl = f"{n_par:,} params" + (" (degenerate: 1 chunk)" if degenerate else "")
            style = ":" if degenerate else "-"
            ax.plot([r["fraction"] for r in sel], [r["crypto_sec"] for r in sel],
                    "o" + style, linewidth=2, label=lbl)
            ax2.plot([r["fraction"] for r in sel],
                     [r.get("payload_ratio", 0.0) for r in sel], "s--",
                     linewidth=1.2, alpha=0.6)
        ax.set_xlabel("Encryption ratio  p", fontsize=10)
        ax.set_ylabel("Crypto time (s)  [solid]", fontsize=10)
        ax2.set_ylabel("Payload ratio  [dashed]", fontsize=10)
        ax.set_title("(a) Overhead vs encryption ratio", fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

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
