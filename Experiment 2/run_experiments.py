"""Three-way credit-risk FL comparison: Standard FL vs. DP-FL vs. HE-FL.

Runs all five configurations on the same data split / partition / global init and
emits results/results.json plus a markdown summary table:

  rows    {Standard FL, DP-FL (eps=14.13), DP-FL (eps=8.65), DP-FL (eps=5.74), HE-FL}
  columns {Accuracy, Loss, Wall-clock/round, Aggregation latency/round, Ciphertext MB}

Key findings to verify:
  * HE-FL reproduces Standard FL accuracy exactly (CKKS is exact-up-to-rounding),
    but its aggregation step is orders of magnitude slower than a plaintext mean
    and its per-client payload is ~90x larger.
  * DP-FL trades accuracy for privacy; accuracy falls as epsilon shrinks, and the
    eps=8.65 point is the paper's best-balance operating point.

Timing note: local training is identical across all arms, so total wall-clock per
round barely moves -- at this model scale (929 params) the CKKS crypto is ~tens of
ms, dwarfed by training. The HE cost shows up sharply in the *aggregation* step
(plaintext mean vs. encrypt+homomorphic-sum+decrypt) and in payload size, which is
why both are reported as dedicated columns.
"""
import json
import os
import time

import numpy as np
import torch

from credit_fl import config, data, he
from credit_fl import train as T
from credit_fl.model import (new_model, get_weights, flatten_state,
                             unflatten_state)


def _initial_global(in_features):
    """Deterministic shared global init so every regime starts identically."""
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    return get_weights(new_model(in_features))


def _payload_mb(state):
    return flatten_state(state).astype(np.float32).nbytes / 1e6


def run_standard(parts, X_tr, y_tr, X_te, y_te, nf):
    """Vanilla FedAvg, no privacy. Times training and plaintext aggregation."""
    g = _initial_global(nf)
    round_t, train_t, agg_t, curve = [], [], [], []
    for _ in range(config.ROUNDS):
        t0 = time.perf_counter()
        states = [T.local_train_plain(g, X_tr[i], y_tr[i], nf)[0] for i in parts]
        t1 = time.perf_counter()
        g = T.aggregate(states)
        t2 = time.perf_counter()
        round_t.append(t2 - t0); train_t.append(t1 - t0); agg_t.append(t2 - t1)
        curve.append(T.evaluate(g, X_te, y_te, nf))
    m = T.evaluate(g, X_te, y_te, nf)
    return {**m, "sec_per_round": float(np.mean(round_t)),
            "train_sec_per_round": float(np.mean(train_t)),
            "agg_ms_per_round": float(np.mean(agg_t) * 1e3),
            "ciphertext_mb": None, "plaintext_payload_mb": _payload_mb(g),
            "curve": curve}


def run_dp(epsilon, parts, X_tr, y_tr, X_te, y_te, nf):
    """DP-FL: Opacus DP-SGD locally at a fixed sigma reused across rounds."""
    sigmas = [T.get_sigma(epsilon, len(i)) for i in parts]
    eps_actual = float(np.mean([T.actual_epsilon(s, len(i))
                                for s, i in zip(sigmas, parts)]))
    g = _initial_global(nf)
    round_t, train_t, agg_t, curve = [], [], [], []
    for _ in range(config.ROUNDS):
        t0 = time.perf_counter()
        states = [T.local_train_dp(g, X_tr[i], y_tr[i], s, nf)[0]
                  for s, i in zip(sigmas, parts)]
        t1 = time.perf_counter()
        g = T.aggregate(states)
        t2 = time.perf_counter()
        round_t.append(t2 - t0); train_t.append(t1 - t0); agg_t.append(t2 - t1)
        curve.append(T.evaluate(g, X_te, y_te, nf))
    m = T.evaluate(g, X_te, y_te, nf)
    return {**m, "sec_per_round": float(np.mean(round_t)),
            "train_sec_per_round": float(np.mean(train_t)),
            "agg_ms_per_round": float(np.mean(agg_t) * 1e3),
            "ciphertext_mb": None, "plaintext_payload_mb": _payload_mb(g),
            "epsilon_target": epsilon, "epsilon_actual": eps_actual,
            "sigma": float(np.mean(sigmas)), "curve": curve}


def run_he(parts, X_tr, y_tr, X_te, y_te, nf):
    """HE-FL: identical plaintext training, CKKS-encrypted aggregation.

    The aggregation timer brackets the whole encrypted pipeline -- per-client
    encryption, the homomorphic sum + 1/N scale on the server, and decryption --
    so it is directly comparable to the plaintext arms' aggregation latency."""
    context = he.make_context()
    g = _initial_global(nf)
    template = g
    n_params = flatten_state(g).size

    round_t, train_t, agg_t, curve = [], [], [], []
    enc_times, ct_sizes = [], []
    for _ in range(config.ROUNDS):
        t0 = time.perf_counter()
        states = [T.local_train_plain(g, X_tr[i], y_tr[i], nf)[0] for i in parts]
        t1 = time.perf_counter()
        # client side: flatten + encrypt each update
        enc_list = []
        for st in states:
            enc, dt = he.encrypt_vector(context, flatten_state(st))
            enc_list.append(enc); enc_times.append(dt)
            ct_sizes.append(he.ciphertext_bytes(enc))
        # server side: homomorphic mean over ciphertexts, then decrypt
        agg_ct, _ = he.aggregate_encrypted(enc_list)
        flat_mean = he.decrypt_vector(agg_ct, n_params)
        g = unflatten_state(flat_mean, template)
        t2 = time.perf_counter()
        round_t.append(t2 - t0); train_t.append(t1 - t0); agg_t.append(t2 - t1)
        curve.append(T.evaluate(g, X_te, y_te, nf))

    m = T.evaluate(g, X_te, y_te, nf)
    return {**m, "sec_per_round": float(np.mean(round_t)),
            "train_sec_per_round": float(np.mean(train_t)),
            "agg_ms_per_round": float(np.mean(agg_t) * 1e3),
            "ciphertext_mb": float(np.mean(ct_sizes) / 1e6),
            "plaintext_payload_mb": _payload_mb(g),
            "enc_ms_per_client": float(np.mean(enc_times) * 1e3),
            "n_params": int(n_params), "curve": curve}


def run_all(seed=None):
    """Run all five configurations once; returns the results dict."""
    if seed is not None:
        config.SEED = seed
    X_tr, y_tr, X_te, y_te, info = data.split_and_scale()
    nf = info["n_features"]
    parts, pinfo = data.iid_partition(len(y_tr), config.NUM_CLIENTS, config.SEED)

    exps = {}
    exps["Standard FL"] = run_standard(parts, X_tr, y_tr, X_te, y_te, nf)
    for eps in config.DP_EPSILONS:
        exps[f"DP-FL (eps={eps})"] = run_dp(eps, parts, X_tr, y_tr, X_te, y_te, nf)
    exps["HE-FL"] = run_he(parts, X_tr, y_tr, X_te, y_te, nf)
    return {"info": info, "partition": pinfo, "seed": config.SEED,
            "experiments": exps}


ROW_ORDER = ["Standard FL", "DP-FL (eps=14.13)", "DP-FL (eps=8.65)",
             "DP-FL (eps=5.74)", "HE-FL"]


def render_table(results):
    exps = results["experiments"]
    lines = ["| Configuration | Accuracy | Loss | Wall-clock/round (s) "
             "| Agg latency/round (ms) | Ciphertext size (MB) |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in ROW_ORDER:
        e = exps[name]
        ct = f"{e['ciphertext_mb']:.3f}" if e["ciphertext_mb"] else "—"
        lines.append(f"| {name} | {e['accuracy']*100:.2f}% | {e['loss']:.4f} "
                     f"| {e['sec_per_round']:.3f} | {e['agg_ms_per_round']:.3f} | {ct} |")
    pt = exps["Standard FL"]["plaintext_payload_mb"]
    he_e = exps["HE-FL"]
    lines += ["",
              f"_Plaintext update = {pt*1e3:.2f} KB/client; CKKS ciphertext = "
              f"{he_e['ciphertext_mb']:.3f} MB/client "
              f"({he_e['ciphertext_mb']*1e6/(pt*1e6):.0f}x blow-up). "
              f"HE encryption = {he_e['enc_ms_per_client']:.2f} ms/client; "
              f"HE aggregation is "
              f"{he_e['agg_ms_per_round']/max(exps['Standard FL']['agg_ms_per_round'],1e-6):.0f}x "
              f"slower than the plaintext mean._"]
    return "\n".join(lines) + "\n"


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    results = run_all()
    info, pinfo, exps = results["info"], results["partition"], results["experiments"]
    print(f"data: {info['n_train']} train / {info['n_test']} test, "
          f"{info['n_features']} features, client sizes {pinfo['sizes']}\n")
    for name in ROW_ORDER:
        e = exps[name]
        extra = (f"  eps_actual={e['epsilon_actual']:.2f} sigma={e['sigma']:.3f}"
                 if "epsilon_actual" in e else
                 f"  ciphertext={e['ciphertext_mb']:.3f} MB" if e["ciphertext_mb"]
                 else "")
        print(f"  {name:18s} acc={e['accuracy']:.4f} loss={e['loss']:.4f}{extra}")

    with open(os.path.join(config.RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    table = render_table(results)
    print("\n" + table)
    with open(os.path.join(config.RESULTS_DIR, "results_table.md"), "w") as f:
        f.write(table)
    print(f"saved -> {config.RESULTS_DIR}/results.json + results_table.md")


if __name__ == "__main__":
    main()
