"""The aggregation switch of Trial 07/27: plain, fully encrypted, or partly encrypted.

Comparison only, see the package docstring. Every arm of the research pipeline goes through
``aggregate_states``; the only thing that differs between arms is the ``mode`` string.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from obscuradp.aggregation.fedavg import fedavg_weights
from obscuradp.legacy import he
from obscuradp.legacy.config import LegacyTrialConfig
from obscuradp.state import State, flatten_state, unflatten_state


def aggregate_states(
    cfg: LegacyTrialConfig,
    states: Sequence[Mapping[str, torch.Tensor]],
    counts: Sequence[float],
    mode: str = "plaintext",
    mask: npt.NDArray[np.bool_] | None = None,
    ctx: Any = None,
    weighted: bool = True,
) -> tuple[State, dict[str, Any]]:
    """Aggregate client states under ``"plaintext"``, ``"he"`` or ``"spe"``.

    Returns ``(aggregated state, stats)``; ``stats`` carries the timing and size fields of
    ``he.fedavg_he_timed`` plus ``agg_sec``.
    """
    t_start = time.perf_counter()

    if mode not in ("plaintext", "he", "spe"):
        raise ValueError(f"Unknown mode: {mode}")
    if mode in ("he", "spe") and ctx is None:
        raise ValueError(f"mode='{mode}' requires ctx (CKKS context)")

    template = states[0]
    flattened_states = [flatten_state(s) for s in states]
    weights = fedavg_weights(counts, weighted=weighted)

    stats: dict[str, Any]
    if mode == "plaintext":
        agg_flat = he.fedavg_plaintext(flattened_states, weights=weights)
        stats = _plaintext_stats(len(agg_flat))
    elif mode == "he":
        all_true_mask = np.ones(len(flattened_states[0]), dtype=bool)
        agg_flat, stats = he.fedavg_he_timed(
            cfg, flattened_states, all_true_mask, ctx, weights=weights
        )
    elif mask is None or np.sum(mask) == 0:
        agg_flat = he.fedavg_plaintext(flattened_states, weights=weights)
        stats = _plaintext_stats(len(flattened_states[0]))
    else:
        agg_flat, stats = he.fedavg_he_timed(cfg, flattened_states, mask, ctx, weights=weights)

    agg_state = unflatten_state(agg_flat, template)
    stats["agg_sec"] = time.perf_counter() - t_start
    return agg_state, stats


def plaintext_reference(
    states: Sequence[Mapping[str, torch.Tensor]], counts: Sequence[float], weighted: bool = True
) -> npt.NDArray[np.float64]:
    """The weighted mean in the clear, as a flat float64 vector."""
    flattened_states = [flatten_state(s) for s in states]
    weights = fedavg_weights(counts, weighted=weighted)
    return he.fedavg_plaintext(flattened_states, weights=weights)


def _plaintext_stats(n_params: int) -> dict[str, Any]:
    return {
        "encrypt_sec": 0.0,
        "aggregate_sec": 0.0,
        "decrypt_sec": 0.0,
        "crypto_sec": 0.0,
        "ciphertext_bytes": 0,
        "plaintext_bytes": n_params * 4,
        "payload_ratio": 0.0,
        # Nothing is encrypted, so the client sends the whole model in the clear.
        "wire_payload_ratio": 1.0,
        "n_chunks": 0,
        "n_encrypted": 0,
        "encrypted_fraction": 0.0,
        "n_params": n_params,
    }
