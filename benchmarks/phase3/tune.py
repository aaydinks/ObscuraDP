"""Choose every arm's settings on made-up tables, never on the real ones.

For each table shape, model, arm and budget level, every combination on a small grid is run
on two stand-in tables (``benchmarks/phase2/standin.py``) and the one with the best mean
validation AUC is kept. The result is written to ``settings.json``, which ``run.py`` reads.

    python -m benchmarks.phase3.tune    (from the repository root)

What is tuned:

- ``shared`` (and ``server``, which uses the same settings so the two differ in the noise
  alone): learning rate, clipping norm, number of rounds, expected batch size.
- ``shared_whole``: learning rate, clipping norm, number of rounds (every row takes part).
- ``local_tuned``: learning rate, clipping norm, and the pair (rounds, local passes).
- ``participant``: learning rate and clipping norm.

The arms ``local``, ``local_masked`` and ``local_masked8`` and the unprotected baseline keep
the settings of Phase 2 (``benchmarks/phase2/settings.json``).
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.phase2.standin import make_standin
from benchmarks.phase3.common import FULL_BATCH, MODELS, Setting, build_simulation

HERE = Path(__file__).resolve().parent
LEVELS = (1.0, 8.0)
STANDIN_SEEDS = (0, 1)
DATASETS = ("stroke", "loan")

# arm: the values tried for (learning_rate, clip_norm, rounds, batch_size, local_epochs)
GRIDS: dict[str, dict[str, tuple[Any, ...]]] = {
    "shared": {
        "learning_rate": (0.1, 0.3, 1.0, 3.0),
        "clip_norm": (0.5, 1.0),
        "rounds": (60, 200, 600),
        "batch_size": (160, 640),
        "local_epochs": (1.0,),
    },
    "shared_whole": {
        "learning_rate": (0.1, 0.3, 1.0, 3.0),
        "clip_norm": (0.5, 1.0),
        "rounds": (20, 60, 200),
        "batch_size": (FULL_BATCH,),
        "local_epochs": (1.0,),
    },
    "local_tuned": {
        "learning_rate": (0.02, 0.05, 0.1, 0.2, 0.5),
        "clip_norm": (0.5, 1.0),
        "rounds": (20, 60),
        "batch_size": (32,),
        "local_epochs": (1.0, 3.0),
    },
    "participant": {
        "learning_rate": (0.02, 0.1, 0.5),
        "clip_norm": (0.5, 2.0),
        "rounds": (20,),
        "batch_size": (32,),
        "local_epochs": (3.0,),
    },
}
FIELDS = ("learning_rate", "clip_norm", "rounds", "batch_size", "local_epochs")


def combinations(arm: str) -> list[tuple[Any, ...]]:
    grid = GRIDS[arm]
    every = itertools.product(*(grid[name] for name in FIELDS))
    # Sixty rounds of three local passes each are left out: nine times the work of the
    # research protocol's local steps, and more local steps do not buy accuracy at a fixed
    # total budget (docs/ROADMAP.md, corrections).
    return [c for c in every if not (arm == "local_tuned" and c[2] == 60 and c[4] == 3.0)]


def _key(task: tuple[Any, ...]) -> str:
    return json.dumps(task)


def _score(task: tuple[Any, ...]) -> tuple[str, float]:
    import torch

    torch.set_num_threads(1)
    dataset, model, arm, epsilon, seed, learning_rate, clip_norm, rounds, batch, epochs = task
    setting = Setting(
        dataset, model, arm, epsilon, seed, learning_rate, clip_norm, rounds, batch, epochs
    )
    result = build_simulation(setting, make_standin(dataset, seed)).run(seed=seed)
    return _key(task), float(result.validation["auc"])


def all_tasks() -> list[tuple[Any, ...]]:
    tasks = [
        (dataset, model, arm, epsilon, seed, *combination)
        for dataset in DATASETS
        for model in MODELS
        for arm in GRIDS
        for epsilon in LEVELS
        for combination in combinations(arm)
        for seed in STANDIN_SEEDS
    ]
    # Runs that share a noise calibration next to each other, so a process can reuse it.
    tasks.sort(key=lambda t: (t[2], t[0], t[3], t[7], t[8], t[4]))
    return tasks


def choose(scores: dict[str, float]) -> tuple[dict[str, Any], dict[str, float]]:
    """The best combination per table, model, arm and level, and the whole grid."""
    chosen: dict[str, Any] = {}
    grid: dict[str, float] = {}
    for dataset in DATASETS:
        for model in MODELS:
            for arm in GRIDS:
                for epsilon in LEVELS:
                    best: tuple[float, tuple[Any, ...]] | None = None
                    for combination in combinations(arm):
                        values = [
                            scores[_key((dataset, model, arm, epsilon, seed, *combination))]
                            for seed in STANDIN_SEEDS
                        ]
                        mean = float(np.mean(values))
                        name = "/".join(
                            f"{f}={v:g}" for f, v in zip(FIELDS, combination, strict=True)
                        )
                        grid[f"{dataset}/{model}/{arm}/{epsilon:g}/{name}"] = mean
                        if best is None or mean > best[0]:
                            best = (mean, combination)
                    assert best is not None
                    chosen[f"{dataset}/{model}/{arm}/{epsilon:g}"] = {
                        **dict(zip(FIELDS, best[1], strict=True)),
                        "standin_validation_auc": best[0],
                    }
    return chosen, grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--cache", type=Path, help="keep finished scores here and resume")
    args = parser.parse_args()
    scores: dict[str, float] = {}
    if args.cache is not None and args.cache.exists():
        for line in args.cache.read_text().splitlines():
            key, value = json.loads(line)
            scores[key] = value
    todo = [task for task in all_tasks() if _key(task) not in scores]
    print(f"{len(scores)} scores kept, {len(todo)} runs to do", flush=True)
    with multiprocessing.Pool(args.processes) as pool:
        for count, (key, value) in enumerate(pool.imap_unordered(_score, todo, chunksize=1), 1):
            scores[key] = value
            if args.cache is not None:
                with args.cache.open("a") as handle:
                    handle.write(json.dumps([key, value]) + "\n")
            if count % 50 == 0:
                print(f"{count} of {len(todo)} done", flush=True)

    chosen, grid = choose(scores)
    output = {
        "note": "Chosen on made-up stand-in tables (benchmarks/phase2/standin.py), by mean "
        "validation AUC of two tables. No real row was used.",
        "levels": list(LEVELS),
        "tried": {
            arm: {name: list(values) for name, values in g.items()} for arm, g in GRIDS.items()
        },
        "chosen": chosen,
        "grid": grid,
    }
    (HERE / "settings.json").write_text(json.dumps(output, indent=1, sort_keys=True) + "\n")
    for key, value in chosen.items():
        print(key, value)


if __name__ == "__main__":
    main()
