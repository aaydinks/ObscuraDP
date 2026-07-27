"""Unified DP + FL + HE orchestration: five stacks from plain FL to full privacy.

The five stacks form a lattice where each adds a single orthogonal concern:
  plain      : FL only (baseline)
  dp         : FL + differential privacy (adds DP, keeps aggregation plaintext)
  he         : FL + homomorphic encryption (adds HE with all params, keeps DP off)
  dp_he      : FL + DP + HE (adds both, naive full-parameter encryption)
  dp_spe     : FL + DP + HE with selective encryption (top-p sensitive params only)

This structure isolates each leg by a single controlled change, enabling the
separability claim: utility depends on epsilon only, crypto cost depends on p only,
and the two compose without unexpected interaction.

Two critical design decisions enforced throughout:
1. Per-client sigma: under non-IID Dirichlet partitioning clients differ in size,
   so each gets its own sigma when DP is on. We record the mean for reporting.
2. Losslessness check every round: CKKS aggregation of DP-noised updates is checked
   against plaintext aggregation via aggregate.plaintext_reference(), producing the
   per-round max_abs_dev. This is the single most important number in the trial.
"""

import time
import numpy as np
import torch

from . import config, data, models, dp, he, sensitivity, fedprox, metrics, aggregate


STACK_FLAGS = {
    "plain":  {"dp": False, "agg": "plaintext"},
    "dp":     {"dp": True,  "agg": "plaintext"},
    "he":     {"dp": False, "agg": "he"},
    "dp_he":  {"dp": True,  "agg": "he"},
    "dp_spe": {"dp": True,  "agg": "spe"},
}


def prepare(dataset, seed=config.SEED, use_smote=None):
    """Build a shared data split + partition + client data bundle for sweeps.

    Called ONCE by a sweep driver and passed to every run_stack via prepared=,
    so all arms see identical train/test splits, partitions, and per-client data.
    This enforces the single-controlled-variable discipline established in Exp 2.

    Returns dict with keys: X_tr, y_tr, X_te, y_te, info, parts, part_info,
    client_data, in_features.
    """
    X_tr, y_tr, X_te, y_te, info = data.split_and_scale(dataset, seed=seed)
    in_features = info["n_features"]

    if dataset == "stroke":
        parts, part_info = data.partition(
            dataset, y_tr, num_clients=config.NUM_CLIENTS, seed=seed
        )
    else:
        parts, part_info = data.partition(
            dataset, y_tr, num_clients=config.NUM_CLIENTS, seed=seed
        )

    client_data = data.build_client_data(
        dataset, X_tr, y_tr, parts, use_smote=use_smote, seed=seed
    )

    return {
        "X_tr": X_tr,
        "y_tr": y_tr,
        "X_te": X_te,
        "y_te": y_te,
        "info": info,
        "parts": parts,
        "part_info": part_info,
        "client_data": client_data,
        "in_features": in_features,
    }


def _jsonable(obj):
    """Convert numpy scalars and arrays to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def run_stack(dataset="stroke", stack="dp_spe", algo="fedprox", epsilon=None,
              spe_fraction=None, mask_mode=None, seed=config.SEED, rounds=None,
              local_epochs=None, width_scale=1, use_smote=None, ctx=None,
              prepared=None, log=print):
    """Run one complete federated training loop under a specific stack.

    Args:
        dataset: "stroke" or "loan".
        stack: "plain", "dp", "he", "dp_he", or "dp_spe".
        algo: "fedprox" (mu=config.MU) or "fedavg" (mu=0.0).
        epsilon: privacy budget. Defaults to config.EPSILON_DEFAULT for DP stacks,
                 else None.
        spe_fraction: fraction of params to encrypt. Defaults depend on stack.
        mask_mode: "clean", "dp", or "random" (default config.MASK_MODE_DEFAULT).
        seed: RNG seed for torch/numpy.
        rounds: Number of federated rounds (default config.ROUNDS).
        local_epochs: Client local epochs (default config.LOCAL_EPOCHS).
        width_scale: Model width multiplier.
        use_smote: Whether to SMOTE-resample per-client data. None -> dataset default.
        ctx: CKKS context (built on demand if needed, else passed in).
        prepared: Pre-built dict from prepare() with X_tr, y_tr, X_te, y_te,
                  info, parts, part_info, client_data, in_features.
        log: Logging function (default print).

    Returns:
        Dict with keys: dataset, stack, algo, seed, epsilon, spe_fraction,
        mask_mode, rounds, local_epochs, width_scale, n_params, use_smote,
        metrics, curve, dp, crypto, timing, lossless, mask.
    """
    # Defaults
    if rounds is None:
        rounds = config.ROUNDS
    if local_epochs is None:
        local_epochs = config.LOCAL_EPOCHS
    if mask_mode is None:
        mask_mode = config.MASK_MODE_DEFAULT

    # Resolve algorithm
    if algo == "fedprox":
        mu = config.MU
    elif algo == "fedavg":
        mu = 0.0
    else:
        raise ValueError(f"Unknown algo: {algo}")

    # Resolve flags from stack
    flags = STACK_FLAGS[stack]
    use_dp = flags["dp"]
    agg_mode = flags["agg"]

    # Resolve epsilon
    if epsilon is None:
        if use_dp:
            epsilon = config.EPSILON_DEFAULT
        else:
            epsilon = None

    # Resolve spe_fraction
    if spe_fraction is None:
        if stack == "dp_spe":
            spe_fraction = config.SPE_DEFAULT
        elif stack in ("he", "dp_he"):
            spe_fraction = 1.0
        else:
            spe_fraction = 0.0

    # Seed everything
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load or build data
    if prepared is not None:
        X_tr = prepared["X_tr"]
        y_tr = prepared["y_tr"]
        X_te = prepared["X_te"]
        y_te = prepared["y_te"]
        info = prepared["info"]
        parts = prepared["parts"]
        part_info = prepared["part_info"]
        client_data = prepared["client_data"]
        in_features = prepared["in_features"]
    else:
        prep = prepare(dataset, seed=seed, use_smote=use_smote)
        X_tr = prep["X_tr"]
        y_tr = prep["y_tr"]
        X_te = prep["X_te"]
        y_te = prep["y_te"]
        info = prep["info"]
        parts = prep["parts"]
        part_info = prep["part_info"]
        client_data = prep["client_data"]
        in_features = prep["in_features"]

    # Build ONE shared global init
    global_state = models.get_weights(
        models.new_model(in_features, dataset=dataset, width_scale=width_scale)
    )
    n_params = models.count_params(
        models.new_model(in_features, dataset=dataset, width_scale=width_scale)
    )

    # Compute per-client sigma if DP is on
    sigmas = {}
    if use_dp:
        for client_idx, (Xc, yc) in enumerate(client_data):
            sigmas[client_idx] = dp.get_sigma(
                epsilon, len(yc), epochs=local_epochs,
                batch_size=config.BATCH_SIZE
            )
        mean_sigma = float(np.mean(list(sigmas.values())))
    else:
        mean_sigma = None

    # Build CKKS context if needed
    if ctx is None and agg_mode in ("he", "spe"):
        ctx = he.make_context()

    # Build SPE mask ONCE before round 1, from client 0's data
    mask = None
    if agg_mode == "spe":
        # Get a batch from client 0
        Xc_sample, yc_sample = client_data[0]
        batch_data = sensitivity.tabular_batch(
            Xc_sample, yc_sample, n=256, seed=seed
        )

        # Determine sigma for mask building (use client 0's sigma if DP is on)
        mask_sigma = sigmas.get(0) if use_dp else None

        # Build mask
        mask_model = models.new_model(
            in_features, dataset=dataset, width_scale=width_scale
        )
        models.set_weights(mask_model, global_state)
        mask_obj, _ = sensitivity.make_mask(
            mask_model, global_state, batch_data,
            fraction=spe_fraction, mode=mask_mode, sigma=mask_sigma, seed=seed
        )
        mask = mask_obj
    else:
        # For non-SPE stacks, still build a mask (all-False or all-True)
        # so the aggregation path stays consistent
        if agg_mode == "he":
            mask = np.ones(n_params, dtype=bool)
        else:
            mask = np.zeros(n_params, dtype=bool)

    # Counters for metrics
    curve = []
    train_times = []
    agg_times = []
    all_crypto_stats = []
    max_abs_devs = []
    f32_floors = []

    # ========================================================================
    # Main federated loop
    # ========================================================================
    for round_idx in range(rounds):
        t_round_start = time.perf_counter()

        # Each client trains locally
        t_train_start = time.perf_counter()
        states = []
        counts = []
        for client_idx, (Xc, yc) in enumerate(client_data):
            sigma = sigmas.get(client_idx) if use_dp else None
            state, n_samples, _ = fedprox.local_train(
                global_state, Xc, yc, in_features,
                dataset=dataset,
                dp=use_dp,
                sigma=sigma,
                mu=mu,
                width_scale=width_scale,
                lr=config.LR.get(dataset, config.LR["stroke"]),
                local_epochs=local_epochs,
                batch_size=config.BATCH_SIZE,
                seed=seed * 1000 + round_idx * 10 + client_idx
            )
            states.append(state)
            counts.append(n_samples)

        t_train_end = time.perf_counter()
        train_time_this_round = t_train_end - t_train_start

        # Aggregate states
        t_agg_start = time.perf_counter()
        agg_state, agg_stats = aggregate.aggregate_states(
            states, counts,
            mode=agg_mode,
            mask=mask,
            ctx=ctx,
            weighted=True
        )
        t_agg_end = time.perf_counter()
        agg_time_this_round = t_agg_end - t_agg_start

        # Update global state
        global_state = agg_state

        # Compute plaintext reference and losslessness deviation
        ref_agg_flat = aggregate.plaintext_reference(states, counts, weighted=True)
        used_agg_flat = models.flatten_state(agg_state)
        max_abs_dev = float(np.max(np.abs(used_agg_flat - ref_agg_flat)))
        # The aggregate is stored back into a float32 state_dict, so even the
        # plaintext arm deviates from the float64 reference by the float32
        # quantisation step. Recording that floor separately is what makes the
        # HE number interpretable: the honest claim is not "HE error is small"
        # but "HE error is within a small factor of the rounding the plaintext
        # path already pays just to hold the weights".
        ref_f32_flat = models.flatten_state(
            models.unflatten_state(ref_agg_flat, states[0]))
        f32_floors.append(float(np.max(np.abs(ref_f32_flat - ref_agg_flat))))
        max_abs_devs.append(max_abs_dev)

        # Evaluate on test set
        metrics_dict = metrics.evaluate(
            agg_state, X_te, y_te, in_features,
            dataset=dataset, width_scale=width_scale
        )
        # `curve` carries the round index; `metrics` (the final row) must not,
        # so the two never share a mutated dict.
        curve.append(dict(metrics_dict, round=round_idx + 1))

        # Accumulate timing and crypto stats
        train_times.append(train_time_this_round)
        agg_times.append(agg_time_this_round)
        if agg_stats is not None:
            all_crypto_stats.append(agg_stats)

        # Log every round or every few rounds
        if log and (round_idx % 2 == 0 or round_idx == rounds - 1):
            primary_metric = metrics_dict.get(
                metrics.primary_metric(dataset), metrics_dict.get("accuracy", 0)
            )
            log(f"[{stack:7s}] round {round_idx+1:2d}/{rounds}  "
                f"{metrics.primary_metric(dataset)}={primary_metric:.4f}  "
                f"max_abs_dev={max_abs_dev:.2e}")

    # ========================================================================
    # Compute aggregated results
    # ========================================================================

    # Final metrics
    # Strip the round index -- `metrics` is the headline row, `curve` is the trace.
    final_metrics = {k: v for k, v in curve[-1].items() if k != "round"} if curve else {}

    # Average timing
    avg_train_sec_per_round = float(np.mean(train_times))
    avg_agg_sec_per_round = float(np.mean(agg_times))
    total_sec = float(np.sum(train_times) + np.sum(agg_times))

    # Average crypto stats (if HE/SPE was used)
    if all_crypto_stats:
        crypto_stats_keys = [
            "encrypt_sec", "aggregate_sec", "decrypt_sec", "crypto_sec",
            "ciphertext_bytes", "plaintext_bytes", "payload_ratio",
            "n_chunks", "n_encrypted", "encrypted_fraction"
        ]
        crypto_block = {}
        for key in crypto_stats_keys:
            values = [s.get(key, 0) for s in all_crypto_stats]
            crypto_block[key] = float(np.mean(values))
    else:
        crypto_block = {
            "encrypt_sec": 0.0, "aggregate_sec": 0.0, "decrypt_sec": 0.0,
            "crypto_sec": 0.0, "ciphertext_bytes": 0.0, "plaintext_bytes": 0.0,
            "payload_ratio": 0.0, "n_chunks": 0, "n_encrypted": 0,
            "encrypted_fraction": 0.0
        }

    # DP block. Each client's sigma was solved for the target epsilon at THAT
    # client's own sampling rate, so epsilon must be re-derived per client and
    # then reduced -- evaluating the mean sigma at the global training-set size
    # would silently report a different (smaller) budget than was purchased.
    # The federation's guarantee is the worst client's, hence max(); the noise
    # floor the CKKS error is compared against is the SMALLEST one, so the
    # losslessness claim is stated against the most demanding case.
    eps_per_round = None
    eps_cumulative = None
    noise_std = None
    per_client_dp = None
    if use_dp and sigmas:
        eps_round_c, eps_cum_c, noise_c = [], [], []
        for client_idx, (Xc, yc) in enumerate(client_data):
            s = sigmas[client_idx]
            eps_round_c.append(float(dp.per_round_epsilon(
                s, len(yc), epochs=local_epochs, batch_size=config.BATCH_SIZE)))
            eps_cum_c.append(float(dp.cumulative_epsilon(
                s, len(yc), rounds=rounds, epochs=local_epochs,
                batch_size=config.BATCH_SIZE)))
            noise_c.append(float(dp.dp_noise_std(s, batch_size=config.BATCH_SIZE)))
        eps_per_round = float(np.max(eps_round_c))
        eps_cumulative = float(np.max(eps_cum_c))
        noise_std = float(np.min(noise_c))
        per_client_dp = {
            "sigma": [float(sigmas[i]) for i in range(len(client_data))],
            "eps_round": eps_round_c,
            "eps_cumulative": eps_cum_c,
            "noise_std": noise_c,
        }

    dp_block = {
        "sigma": mean_sigma,
        "eps_round": eps_per_round,
        "eps_cumulative": eps_cumulative,
        "noise_std": noise_std,
        "per_client": per_client_dp,
    }

    # Losslessness block
    lossless_block = {
        "max_abs_dev_per_round": max_abs_devs,
        "max_abs_dev": float(np.max(max_abs_devs)) if max_abs_devs else 0.0,
        # The float32 state_dict quantisation floor every arm pays, HE or not.
        "float32_floor": float(np.max(f32_floors)) if f32_floors else 0.0,
        "dev_over_float32_floor": (
            float(np.max(max_abs_devs) / np.max(f32_floors))
            if f32_floors and np.max(f32_floors) > 0 else None),
    }

    # Mask block
    if mask is not None:
        n_encrypted = int(np.sum(mask))
    else:
        n_encrypted = 0

    mask_block = {
        "mode": mask_mode,
        "fraction": float(spe_fraction),
        "n_encrypted": n_encrypted,
    }

    # Build result dict
    result = {
        "dataset": dataset,
        "stack": stack,
        "algo": algo,
        "seed": int(seed),
        "epsilon": epsilon,
        "spe_fraction": float(spe_fraction),
        "mask_mode": mask_mode,
        "rounds": int(rounds),
        "local_epochs": int(local_epochs),
        "width_scale": int(width_scale) if width_scale == int(width_scale) else float(width_scale),
        "n_params": int(n_params),
        "use_smote": use_smote,
        "metrics": _jsonable(final_metrics),
        "curve": _jsonable(curve),
        "dp": _jsonable(dp_block),
        "crypto": _jsonable(crypto_block),
        "timing": {
            "train_sec_per_round": avg_train_sec_per_round,
            "agg_sec_per_round": avg_agg_sec_per_round,
            "total_sec": total_sec,
        },
        "lossless": _jsonable(lossless_block),
        "mask": _jsonable(mask_block),
    }

    return result
