"""Choose the learning rate and clipping norm on made-up tables, never on the real ones.

For each dataset shape, model and budget level, every combination on a small grid is run on
two stand-in tables and the one with the best validation AUC is kept. The result is written
to ``settings.json``, which ``run.py`` reads.

    python -m benchmarks.phase2.tune    (from the repository root)
"""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import numpy as np

from benchmarks.phase2.common import MODELS, Setting, build_simulation
from benchmarks.phase2.standin import make_standin

HERE = Path(__file__).resolve().parent
LEARNING_RATES = (0.02, 0.05, 0.1, 0.2, 0.5)
CLIP_NORMS = (0.5, 1.0)
# Budgets are tuned at three levels; each real budget uses the nearest level.
LEVELS: tuple[float | None, ...] = (None, 1.0, 8.0)
STANDIN_SEEDS = (0, 1)


def _score(task: tuple[str, str, float | None, float, float, int]) -> tuple[tuple, float]:
    import torch

    torch.set_num_threads(1)
    dataset, model, epsilon, learning_rate, clip_norm, seed = task
    setting = Setting(dataset, model, epsilon, False, learning_rate, clip_norm, seed)
    result = build_simulation(setting, make_standin(dataset, seed)).run(seed=seed)
    return task, float(result.validation["auc"])


def main() -> None:
    tasks = [
        (dataset, model, epsilon, learning_rate, clip_norm, seed)
        for dataset in ("stroke", "loan")
        for model in MODELS
        for epsilon in LEVELS
        for learning_rate in LEARNING_RATES
        for clip_norm in (CLIP_NORMS if epsilon is not None else (1.0,))
        for seed in STANDIN_SEEDS
    ]
    # Group by seed and dataset so that one process reuses its noise calibrations.
    tasks.sort(key=lambda task: (task[0], task[5], str(task[2])))
    with multiprocessing.Pool(4) as pool:
        scores = dict(pool.imap_unordered(_score, tasks, chunksize=1))

    chosen: dict[str, dict[str, float]] = {}
    grid: dict[str, float] = {}
    for dataset in ("stroke", "loan"):
        for model in MODELS:
            for epsilon in LEVELS:
                candidates = {}
                for learning_rate in LEARNING_RATES:
                    for clip_norm in CLIP_NORMS if epsilon is not None else (1.0,):
                        values = [
                            scores[(dataset, model, epsilon, learning_rate, clip_norm, seed)]
                            for seed in STANDIN_SEEDS
                        ]
                        candidates[(learning_rate, clip_norm)] = float(np.mean(values))
                        level = "none" if epsilon is None else f"{epsilon:g}"
                        grid[
                            f"{dataset}/{model}/{level}/lr={learning_rate:g}/clip={clip_norm:g}"
                        ] = candidates[(learning_rate, clip_norm)]
                (learning_rate, clip_norm), best = max(candidates.items(), key=lambda item: item[1])
                level = "none" if epsilon is None else f"{epsilon:g}"
                chosen[f"{dataset}/{model}/{level}"] = {
                    "learning_rate": learning_rate,
                    "clip_norm": clip_norm,
                    "standin_validation_auc": best,
                }
    output = {
        "note": "Chosen on made-up stand-in tables (standin.py), by mean validation AUC of two "
        "tables. No real row was used.",
        "learning_rates_tried": list(LEARNING_RATES),
        "clip_norms_tried": list(CLIP_NORMS),
        "chosen": chosen,
        "grid": grid,
    }
    (HERE / "settings.json").write_text(json.dumps(output, indent=1, sort_keys=True) + "\n")
    for key, value in chosen.items():
        print(key, value)


if __name__ == "__main__":
    main()
