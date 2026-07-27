"""Federated aggregation: the seam where plaintext, HE, and SPE meet.

This module is where the three prior experiments converge: Experiment 1's
sample-weighted FedAvg for non-IID data, Experiment 2's plaintext-vs-CKKS
arm switch, and Experiment 3's SPE mask. It presents ONE call that every
arm goes through, so the only thing that differs between arms is a mode
string -- that is what makes the comparison controlled. The mode parameter
gates whether the aggregation happens in plaintext, under full CKKS, or
under selective parameter encryption.
"""
import time

import numpy as np
import torch

from . import he, models


def fedavg_weights(counts, weighted=True):
    """Compute per-client weights for FedAvg aggregation.

    Returns a float64 array summing to 1. When weighted=True, returns
    counts / sum(counts); when weighted=False, returns uniform 1/N.

    The non-IID stroke partition (Dirichlet alpha=0.5) makes weighting
    matter: some clients have far fewer samples than others. This weight
    must be applied identically in the plaintext and encrypted paths, or
    the arms would not be comparable.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if weighted:
        return counts / np.sum(counts)
    else:
        return np.ones(len(counts), dtype=np.float64) / len(counts)


def aggregate_states(states, counts, mode="plaintext", mask=None, ctx=None, weighted=True):
    """Aggregate client states under plaintext, HE, or SPE.

    Routes the aggregation through three paths based on mode:
      - "plaintext": direct weighted mean in plaintext
      - "he": full CKKS aggregation (all-True mask internally)
      - "spe": selective parameter encryption (the supplied mask;
               degrades to plaintext if mask is None or all-False)

    Args:
        states: List of state dicts (output of models.get_weights).
        counts: List of sample counts per client.
        mode: "plaintext", "he", or "spe".
        mask: Boolean array (for "spe" mode); ignored for "plaintext" / "he".
        ctx: CKKS context (required for "he" / "spe", unused for "plaintext").
        weighted: If True, apply per-client sample weights; else unweighted 1/N.

    Returns:
        (agg_state dict, stats dict) where stats includes all he.fedavg_he_timed
        keys ("encrypt_sec", "aggregate_sec", "decrypt_sec", "crypto_sec",
        "ciphertext_bytes", "plaintext_bytes", "payload_ratio", "n_chunks",
        "n_encrypted", "encrypted_fraction", "n_params") plus "agg_sec"
        (wall-clock duration of the entire aggregate_states call).
    """
    t_start = time.perf_counter()

    if mode not in ("plaintext", "he", "spe"):
        raise ValueError(f"Unknown mode: {mode}")
    if mode in ("he", "spe") and ctx is None:
        raise ValueError(f"mode='{mode}' requires ctx (CKKS context)")

    template = states[0]
    flattened_states = [models.flatten_state(s) for s in states]
    weights = fedavg_weights(counts, weighted=weighted)

    if mode == "plaintext":
        agg_flat = he.fedavg_plaintext(flattened_states, weights=weights)
        stats = {
            "encrypt_sec": 0.0,
            "aggregate_sec": 0.0,
            "decrypt_sec": 0.0,
            "crypto_sec": 0.0,
            "ciphertext_bytes": 0,
            "plaintext_bytes": len(agg_flat) * 4,
            "payload_ratio": 0.0,
            "n_chunks": 0,
            "n_encrypted": 0,
            "encrypted_fraction": 0.0,
            "n_params": len(agg_flat),
        }

    elif mode == "he":
        all_true_mask = np.ones(len(flattened_states[0]), dtype=bool)
        agg_flat, he_stats = he.fedavg_he_timed(
            flattened_states, all_true_mask, ctx, weights=weights
        )
        stats = he_stats

    elif mode == "spe":
        if mask is None or np.sum(mask) == 0:
            agg_flat = he.fedavg_plaintext(flattened_states, weights=weights)
            n_params = len(flattened_states[0])
            stats = {
                "encrypt_sec": 0.0,
                "aggregate_sec": 0.0,
                "decrypt_sec": 0.0,
                "crypto_sec": 0.0,
                "ciphertext_bytes": 0,
                "plaintext_bytes": n_params * 4,
                "payload_ratio": 0.0,
                "n_chunks": 0,
                "n_encrypted": 0,
                "encrypted_fraction": 0.0,
                "n_params": n_params,
            }
        else:
            agg_flat, he_stats = he.fedavg_he_timed(
                flattened_states, mask, ctx, weights=weights
            )
            stats = he_stats

    agg_state = models.unflatten_state(agg_flat, template)

    agg_sec = time.perf_counter() - t_start
    stats["agg_sec"] = agg_sec

    return agg_state, stats


def plaintext_reference(states, counts, weighted=True):
    """Compute the plaintext weighted mean as a reference.

    Returns the flat float64 vector that HE aggregation should approximate.
    Used by the orchestrator to measure max|HE - plaintext| per round,
    the losslessness evidence. Kept independent of the HE path to serve
    as a genuine reference.
    """
    flattened_states = [models.flatten_state(s) for s in states]
    weights = fedavg_weights(counts, weighted=weighted)
    return he.fedavg_plaintext(flattened_states, weights=weights)
