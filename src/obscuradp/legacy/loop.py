"""The Trial 07/27 training loop, line for line (comparison only, see the package docstring).

Five stacks: ``plain`` (federated learning only), ``dp`` (private training), ``he`` (encrypted
aggregation), ``dp_he`` (both) and ``dp_spe`` (both, with only part of the update encrypted).

Known defects reproduced on purpose, with their finding numbers from ``docs/AUDIT.md``:
the budget is per round (G2); stroke data is resampled before private training (G3); the
mask is built from participant 0's raw rows (G4); one object holds the secret key (G5);
per-participant seeds collide from 10 participants on (G17); an unknown dataset silently
gets stroke's learning rate, ``prepared`` silently overrides ``use_smote``, and
``use_smote`` is recorded as given (G18); the model is evaluated on the test set every
round (G13).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from obscuradp.legacy import aggregate, dp, he, metrics, sensitivity, training
from obscuradp.legacy import models as legacy_models
from obscuradp.legacy.config import LegacyTrialConfig
from obscuradp.legacy.data import Prepared, prepare
from obscuradp.state import count_parameters, flatten_state, get_state, set_state, unflatten_state

__all__ = ["STACK_FLAGS", "prepare", "run_stack"]

STACK_FLAGS: dict[str, dict[str, Any]] = {
    "plain": {"dp": False, "agg": "plaintext"},
    "dp": {"dp": True, "agg": "plaintext"},
    "he": {"dp": False, "agg": "he"},
    "dp_he": {"dp": True, "agg": "he"},
    "dp_spe": {"dp": True, "agg": "spe"},
}


def _jsonable(obj: Any) -> Any:
    """Convert numpy scalars and arrays to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def run_stack(
    cfg: LegacyTrialConfig,
    dataset: str = "stroke",
    stack: str = "dp_spe",
    algo: str = "fedprox",
    epsilon: float | None = None,
    spe_fraction: float | None = None,
    mask_mode: str | None = None,
    seed: int | None = None,
    rounds: int | None = None,
    local_epochs: int | None = None,
    width_scale: int = 1,
    use_smote: bool | None = None,
    ctx: Any = None,
    prepared: Prepared | None = None,
    log: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """Run one complete federated training loop under one stack and return its result record."""
    if seed is None:
        seed = cfg.seed
    if rounds is None:
        rounds = cfg.rounds
    if local_epochs is None:
        local_epochs = cfg.local_epochs
    if mask_mode is None:
        mask_mode = cfg.mask_mode_default

    # Resolve algorithm
    if algo == "fedprox":
        mu = cfg.mu
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
        epsilon = cfg.epsilon_default if use_dp else None

    # Resolve spe_fraction
    if spe_fraction is None:
        if stack == "dp_spe":
            spe_fraction = cfg.spe_default
        elif stack in ("he", "dp_he"):
            spe_fraction = 1.0
        else:
            spe_fraction = 0.0

    # Seed everything
    torch.manual_seed(seed)
    np.random.seed(seed)  # the research code seeds the global generator too

    # One weighting for the whole run.
    weighted = cfg.agg_weighted.get(dataset, True)

    # Load or build data
    prep = (
        prepared if prepared is not None else prepare(cfg, dataset, seed=seed, use_smote=use_smote)
    )
    X_te = prep["X_te"]
    y_te = prep["y_te"]
    client_data = prep["client_data"]
    in_features = prep["in_features"]

    # Build ONE shared global init (two models are built, as in the research code, so the
    # random generator is consumed identically).
    global_state = get_state(
        legacy_models.new_model(cfg, in_features, dataset=dataset, width_scale=width_scale)
    )
    n_params = count_parameters(
        legacy_models.new_model(cfg, in_features, dataset=dataset, width_scale=width_scale)
    )

    # Compute per-client sigma if DP is on
    sigmas: dict[int, float] = {}
    mean_sigma: float | None
    if use_dp:
        assert epsilon is not None
        for client_idx, (_, yc) in enumerate(client_data):
            sigmas[client_idx] = dp.get_sigma(
                cfg, epsilon, len(yc), epochs=local_epochs, batch_size=cfg.batch_size
            )
        mean_sigma = float(np.mean(list(sigmas.values())))
    else:
        mean_sigma = None

    # Build CKKS context if needed
    if ctx is None and agg_mode in ("he", "spe"):
        ctx = he.make_context(cfg)

    # Build SPE mask ONCE before round 1, from client 0's data
    if agg_mode == "spe":
        Xc_sample, yc_sample = client_data[0]
        batch_data = sensitivity.tabular_batch(Xc_sample, yc_sample, n=256, seed=seed)

        # Determine sigma for mask building (use client 0's sigma if DP is on)
        mask_sigma = sigmas.get(0) if use_dp else None

        mask_model = legacy_models.new_model(
            cfg, in_features, dataset=dataset, width_scale=width_scale
        )
        set_state(mask_model, global_state)
        mask, _ = sensitivity.make_mask(
            cfg,
            mask_model,
            global_state,
            batch_data,
            fraction=spe_fraction,
            mode=mask_mode,
            sigma=mask_sigma,
            seed=seed,
        )
    elif agg_mode == "he":
        mask = np.ones(n_params, dtype=bool)
    else:
        mask = np.zeros(n_params, dtype=bool)

    # Counters for metrics
    curve: list[dict[str, Any]] = []
    train_times = []
    agg_times = []
    all_crypto_stats = []
    max_abs_devs = []
    f32_floors = []

    for round_idx in range(rounds):
        # Each client trains locally
        t_train_start = time.perf_counter()
        states = []
        counts = []
        for client_idx, (Xc, yc) in enumerate(client_data):
            sigma = sigmas.get(client_idx) if use_dp else None
            state, n_samples, _ = training.local_train(
                cfg,
                global_state,
                Xc,
                yc,
                in_features,
                dataset=dataset,
                dp=use_dp,
                sigma=sigma,
                mu=mu,
                width_scale=width_scale,
                lr=cfg.lr.get(dataset, cfg.lr["stroke"]),
                local_epochs=local_epochs,
                batch_size=cfg.batch_size,
                seed=seed * 1000 + round_idx * 10 + client_idx,
            )
            states.append(state)
            counts.append(n_samples)
        train_time_this_round = time.perf_counter() - t_train_start

        # Aggregate states
        t_agg_start = time.perf_counter()
        agg_state, agg_stats = aggregate.aggregate_states(
            cfg, states, counts, mode=agg_mode, mask=mask, ctx=ctx, weighted=weighted
        )
        agg_time_this_round = time.perf_counter() - t_agg_start

        # Update global state
        global_state = agg_state

        # Compute plaintext reference and losslessness deviation
        ref_agg_flat = aggregate.plaintext_reference(states, counts, weighted=weighted)
        used_agg_flat = flatten_state(agg_state)
        max_abs_dev = float(np.max(np.abs(used_agg_flat - ref_agg_flat)))
        # Even the plaintext arm deviates from the float64 reference by the float32 rounding
        # step, so that floor is recorded separately.
        ref_f32_flat = flatten_state(unflatten_state(ref_agg_flat, states[0]))
        f32_floors.append(float(np.max(np.abs(ref_f32_flat - ref_agg_flat))))
        max_abs_devs.append(max_abs_dev)

        # Evaluate on test set
        metrics_dict = metrics.evaluate(
            cfg, agg_state, X_te, y_te, in_features, dataset=dataset, width_scale=width_scale
        )
        curve.append(dict(metrics_dict, round=round_idx + 1))

        # Accumulate timing and crypto stats
        train_times.append(train_time_this_round)
        agg_times.append(agg_time_this_round)
        if agg_stats is not None:
            all_crypto_stats.append(agg_stats)

        if log and (round_idx % 2 == 0 or round_idx == rounds - 1):
            primary_name = metrics.primary_metric(dataset)
            primary_value = metrics_dict.get(primary_name, metrics_dict.get("accuracy", 0))
            log(
                f"[{stack:7s}] round {round_idx + 1:2d}/{rounds}  "
                f"{primary_name}={primary_value:.4f}  "
                f"max_abs_dev={max_abs_dev:.2e}"
            )

    # Final metrics: the last row of the curve without its round index.
    final_metrics = {k: v for k, v in curve[-1].items() if k != "round"} if curve else {}

    # Average timing
    avg_train_sec_per_round = float(np.mean(train_times))
    avg_agg_sec_per_round = float(np.mean(agg_times))
    total_sec = float(np.sum(train_times) + np.sum(agg_times))

    # Average crypto stats
    crypto_block: dict[str, Any]
    if all_crypto_stats:
        crypto_stats_keys = [
            "encrypt_sec",
            "aggregate_sec",
            "decrypt_sec",
            "crypto_sec",
            "ciphertext_bytes",
            "plaintext_bytes",
            "payload_ratio",
            "wire_payload_ratio",
            "n_chunks",
            "n_encrypted",
            "encrypted_fraction",
        ]
        crypto_block = {}
        for key in crypto_stats_keys:
            values = [s.get(key, 0) for s in all_crypto_stats]
            crypto_block[key] = float(np.mean(values))
    else:
        crypto_block = {
            "encrypt_sec": 0.0,
            "aggregate_sec": 0.0,
            "decrypt_sec": 0.0,
            "crypto_sec": 0.0,
            "ciphertext_bytes": 0.0,
            "plaintext_bytes": 0.0,
            "payload_ratio": 0.0,
            "wire_payload_ratio": 1.0,
            "n_chunks": 0,
            "n_encrypted": 0,
            "encrypted_fraction": 0.0,
        }

    # DP block: the federation's budget is the worst client's; the noise floor is the smallest.
    eps_per_round = None
    eps_cumulative = None
    noise_std = None
    per_client_dp = None
    if use_dp and sigmas:
        eps_round_c, eps_cum_c, noise_c = [], [], []
        for client_idx, (_, yc) in enumerate(client_data):
            s = sigmas[client_idx]
            eps_round_c.append(
                float(
                    dp.per_round_epsilon(
                        cfg, s, len(yc), epochs=local_epochs, batch_size=cfg.batch_size
                    )
                )
            )
            eps_cum_c.append(
                float(
                    dp.cumulative_epsilon(
                        cfg,
                        s,
                        len(yc),
                        rounds=rounds,
                        epochs=local_epochs,
                        batch_size=cfg.batch_size,
                    )
                )
            )
            noise_c.append(float(dp.dp_noise_std(cfg, s, batch_size=cfg.batch_size)))
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

    lossless_block = {
        "max_abs_dev_per_round": max_abs_devs,
        "max_abs_dev": float(np.max(max_abs_devs)) if max_abs_devs else 0.0,
        "float32_floor": float(np.max(f32_floors)) if f32_floors else 0.0,
        "dev_over_float32_floor": (
            float(np.max(max_abs_devs) / np.max(f32_floors))
            if f32_floors and np.max(f32_floors) > 0
            else None
        ),
    }

    n_encrypted = int(np.sum(mask)) if mask is not None else 0
    mask_block = {
        "mode": mask_mode,
        "fraction": float(spe_fraction),
        "n_encrypted": n_encrypted,
    }

    return {
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
