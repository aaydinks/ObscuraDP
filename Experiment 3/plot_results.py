"""Figures for the FedML-HE reproduction.

  1. overhead_scaling.png -- encryption time & payload vs model size (O(n), log-log),
     naive HE vs plaintext, annotated with the ~15x payload gap.
  2. spe_overhead.png     -- crypto time & payload vs encryption ratio p, per model
     (SPE's ~linear reduction).
  3. accuracy_curves.png  -- test accuracy vs round for plaintext / full-HE / SPE
     (the three curves overlap -> HE is lossless).

Each figure is skipped gracefully if its results JSON is missing.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fedml_he import config

RES = config.RESULTS_DIR


def _load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def plot_overhead():
    d = _load("overhead.json")
    if not d:
        return
    rows = sorted(d["rows"], key=lambda r: r["n_params"])
    n = [r["n_params"] for r in rows]
    names = [r["model"] for r in rows]
    he_s = [r["he_ms"] / 1e3 for r in rows]
    pl_s = [r["plain_ms"] / 1e3 for r in rows]
    he_mb = [r["he_ciphertext_bytes"] / 1e6 for r in rows]
    pl_mb = [r["plain_payload_bytes"] / 1e6 for r in rows]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.loglog(n, he_s, "o-", color="#d7301f", label="naive HE (encrypt+aggregate)")
    a1.loglog(n, pl_s, "s--", color="#2c7fb8", label="plaintext FedAvg")
    a1.set_xlabel("model size (parameters)")
    a1.set_ylabel("time per round (s, log)")
    a1.set_title("Compute overhead is O(n)")
    for x, y, nm in zip(n, he_s, names):
        a1.annotate(nm, (x, y), fontsize=7, xytext=(0, 6),
                    textcoords="offset points", ha="center")
    a1.legend(); a1.grid(True, which="both", alpha=0.3)

    a2.loglog(n, he_mb, "o-", color="#d7301f", label="CKKS ciphertext")
    a2.loglog(n, pl_mb, "s--", color="#2c7fb8", label="plaintext float32")
    ratio = rows[len(rows) // 2]["payload_ratio"]
    a2.set_xlabel("model size (parameters)")
    a2.set_ylabel("per-client payload (MB, log)")
    a2.set_title(f"Communication overhead ~{ratio:.0f}× (constant)")
    a2.legend(); a2.grid(True, which="both", alpha=0.3)

    fig.suptitle("FedML-HE — naive HE vs plaintext FedAvg overhead scaling")
    fig.tight_layout()
    p = os.path.join(RES, "overhead_scaling.png")
    fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)


def plot_spe():
    d = _load("spe.json")
    if not d:
        return
    fracs = d["fractions"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.viridis([0.15, 0.5, 0.82])
    for (name, m), c in zip(d["models"].items(), colors):
        sweep = m["sweep"]
        a1.plot(fracs, [s["he_ms"] / 1e3 for s in sweep], "o-", color=c, label=name)
        a2.plot(fracs, [s["payload_bytes"] / 1e6 for s in sweep], "o-", color=c, label=name)
    a1.set_xlabel("encryption ratio p"); a1.set_ylabel("crypto time / round (s)")
    a1.set_title("SPE crypto time falls ~linearly with p")
    a1.grid(alpha=0.3); a1.legend()
    a2.set_xlabel("encryption ratio p"); a2.set_ylabel("per-client payload (MB)")
    a2.set_title("SPE payload falls ~linearly with p")
    a2.grid(alpha=0.3); a2.legend()
    fig.suptitle("FedML-HE — Selective Parameter Encryption overhead vs p")
    fig.tight_layout()
    p = os.path.join(RES, "spe_overhead.png")
    fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)


def plot_accuracy():
    d = _load("accuracy.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {"plaintext": ("#2c7fb8", "s-"),
              "full HE": ("#d7301f", "o--"),
              f"SPE (p={d['config']['spe_fraction']})": ("#31a354", "^:")}
    for label, curve in d["curves"].items():
        c, st = styles.get(label, ("gray", "-"))
        ax.plot(range(1, len(curve) + 1), [a * 100 for a in curve], st,
                color=c, label=f"{label} → {curve[-1]*100:.2f}%", linewidth=2,
                markersize=6, alpha=0.85)
    ax.set_xlabel("communication round"); ax.set_ylabel("test accuracy (%)")
    ax.set_title(f"ResNet-18 / CIFAR-10 — aggregation is lossless\n"
                 f"(final spread {d['max_spread_pct']:.2f}%; "
                 f"HE vs plaintext weights agree to {d['agreement']['he_vs_plain_max_abs']:.1e})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(RES, "accuracy_curves.png")
    fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)


if __name__ == "__main__":
    plot_overhead()
    plot_spe()
    plot_accuracy()
