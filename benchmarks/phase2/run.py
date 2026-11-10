"""Run the corrected experiments of Phase 2 and store one record per run.

    python -m benchmarks.phase2.run            # everything (resumes: finished runs are kept)
    python -m benchmarks.phase2.run --quick    # two rounds, one seed, to check the pipeline

Grid: both tables; the research network and logistic regression; an unprotected baseline
and total budgets 0.5, 1, 2, 4, 8; five seeds; updates in the clear and through the secure
sum (the secure sum for the network at every budget, for logistic regression at budget 1
and the baseline). Settings come from ``settings.json`` (see ``tune.py``).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path

from benchmarks.phase2.common import BUDGETS, MODELS, SEEDS, Setting, run_setting
from benchmarks.phase2.tables import make_clients

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "benchmarks" / "results" / "phase2"


def _level(epsilon: float | None) -> str:
    """The tuning level a budget uses: the baseline, the low level (1) or the high one (8)."""
    if epsilon is None:
        return "none"
    return "1" if epsilon <= 2.0 else "8"


def settings_for(dataset: str, seed: int, rounds: int) -> list[Setting]:
    chosen = json.loads((HERE / "settings.json").read_text())["chosen"]
    out = []
    for model in MODELS:
        for epsilon in (None, *BUDGETS):
            tuned = chosen[f"{dataset}/{model}/{_level(epsilon)}"]
            for masked in (False, True):
                if masked and model == "logistic" and epsilon not in (None, 1.0):
                    continue
                out.append(
                    Setting(
                        dataset,
                        model,
                        epsilon,
                        masked,
                        tuned["learning_rate"],
                        tuned["clip_norm"],
                        seed,
                        rounds,
                    )
                )
    return out


def _run_group(group: tuple[str, int, int, str]) -> list[str]:
    """All runs of one dataset and seed in one process, so noise calibrations are reused."""
    dataset, seed, rounds, directory = group
    clients = make_clients(dataset, seed)
    done = []
    for setting in settings_for(dataset, seed, rounds):
        path = Path(directory) / f"{setting.label}.json"
        if not path.exists():
            record = run_setting(setting, clients)
            path.write_text(json.dumps(record, indent=1, sort_keys=True, allow_nan=False) + "\n")
        done.append(setting.label)
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="two rounds and one seed only")
    parser.add_argument("--processes", type=int, default=4)
    args = parser.parse_args()
    directory = RESULTS / ("quick" if args.quick else "runs")
    directory.mkdir(parents=True, exist_ok=True)
    seeds = SEEDS[:1] if args.quick else SEEDS
    rounds = 2 if args.quick else 20
    groups = [
        (dataset, seed, rounds, str(directory)) for seed in seeds for dataset in ("stroke", "loan")
    ]
    with multiprocessing.Pool(args.processes) as pool:
        for labels in pool.imap_unordered(_run_group, groups, chunksize=1):
            first = labels[0]
            print(f"finished {len(labels)} runs: {first.split('_')[0]}, seed {first[-2:]}")


if __name__ == "__main__":
    main()
