"""Run the Phase 4 attack tests and store one file per table and seed.

    python -m benchmarks.phase4.run            # the reported runs (resumes where it stopped)
    python -m benchmarks.phase4.run --pilot    # the pilot: other seeds, fewer cells
    python -m benchmarks.phase4.run --quick    # two rounds, one seed, to check the pipeline

Grid: both tables; the arms of ``common.ARMS`` at their budgets and with their models; 30
seeds. Settings are those of Phases 2 and 3 (chosen on made-up tables), so that what is
attacked is what was measured there.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import warnings
from pathlib import Path
from typing import Any

from benchmarks.phase2.tables import make_clients
from benchmarks.phase3.common import Setting
from benchmarks.phase3.run import PHASE2_SETTINGS, level
from benchmarks.phase4.common import (
    ARMS,
    BASELINE,
    BATCH_SIZE,
    DATASETS,
    LOCAL_EPOCHS,
    PILOT_SEEDS,
    ROUNDS,
    SEEDS,
    SETTINGS_OF,
    run_setting,
)
from obscuradp.results import make_result

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "benchmarks" / "results" / "phase4"
PHASE3_SETTINGS = HERE.parent / "phase3" / "settings.json"
# The pilot looks at the default plan at two budgets and at the other arms once.
PILOT_BUDGETS = (None, 1.0, 8.0)


def settings_for(
    dataset: str, seed: int, *, quick: bool = False, pilot: bool = False
) -> list[Setting]:
    """Every audited run of one table and seed."""
    phase2 = json.loads(PHASE2_SETTINGS.read_text())["chosen"]
    phase3 = json.loads(PHASE3_SETTINGS.read_text())["chosen"]
    out = []
    for arm, (_, budgets, models) in ARMS.items():
        source = SETTINGS_OF.get(arm, arm)
        for model in models:
            for epsilon in budgets:
                if pilot and epsilon not in PILOT_BUDGETS:
                    continue
                if source == BASELINE:
                    tuned = phase2[f"{dataset}/{model}/none"]
                    rounds, batch, epochs = ROUNDS, BATCH_SIZE, LOCAL_EPOCHS
                elif source == "local":
                    assert epsilon is not None
                    tuned = phase2[f"{dataset}/{model}/{level(epsilon)}"]
                    rounds, batch, epochs = ROUNDS, BATCH_SIZE, LOCAL_EPOCHS
                else:
                    assert epsilon is not None
                    tuned = phase3[f"{dataset}/{model}/{source}/{level(epsilon)}"]
                    rounds, batch, epochs = (
                        int(tuned["rounds"]),
                        int(tuned["batch_size"]),
                        float(tuned["local_epochs"]),
                    )
                out.append(
                    Setting(
                        dataset,
                        model,
                        arm,
                        epsilon,
                        seed,
                        tuned["learning_rate"],
                        tuned["clip_norm"],
                        2 if quick else rounds,
                        batch,
                        epochs,
                    )
                )
    return out


def cell(setting: Setting) -> str:
    """The name of a run's cell: everything but the seed."""
    return setting.label.rsplit("_seed-", 1)[0]


def _run_group(group: tuple[str, int, bool, bool, str, int]) -> tuple[str, int, int]:
    """All runs of one table and seed in one process, so noise calibrations are reused."""
    dataset, seed, quick, pilot, directory, first_seed = group
    warnings.simplefilter("ignore")
    path = Path(directory) / f"{dataset}_seed-{seed}.json"
    cells: dict[str, Any] = json.loads(path.read_text())["data"]["cells"] if path.exists() else {}
    settings = settings_for(dataset, seed, quick=quick, pilot=pilot)
    clients = make_clients(dataset, seed)
    for setting in settings:
        if cell(setting) in cells:
            continue
        stored = run_setting(setting, clients)
        if seed != first_seed:
            stored["privacy_report"] = None
        cells[cell(setting)] = stored
        record = make_result(
            kind="phase4_seed",
            config={"dataset": dataset, "quick": quick, "pilot": pilot},
            seeds=[seed],
            data={"cells": cells},
        )
        # Written whole and then moved into place, so a stopped run leaves no torn file.
        scratch = path.with_suffix(".part")
        scratch.write_text(
            json.dumps(record, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
        )
        scratch.replace(path)
    return dataset, seed, len(settings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="two rounds and one seed only")
    parser.add_argument("--pilot", action="store_true", help="the pilot seeds and cells")
    parser.add_argument("--processes", type=int, default=4)
    args = parser.parse_args()
    seeds = PILOT_SEEDS if args.pilot else SEEDS
    if args.quick:
        seeds = seeds[:1]
    directory = RESULTS / ("quick" if args.quick else "pilot" if args.pilot else "runs")
    directory.mkdir(parents=True, exist_ok=True)
    groups = [
        (dataset, seed, args.quick, args.pilot, str(directory), seeds[0])
        for seed in seeds
        for dataset in DATASETS
    ]
    with multiprocessing.Pool(args.processes) as pool:
        for dataset, seed, count in pool.imap_unordered(_run_group, groups, chunksize=1):
            print(f"finished {count} runs: {dataset}, seed {seed}", flush=True)


if __name__ == "__main__":
    main()
