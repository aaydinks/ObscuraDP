"""Run the Phase 3 comparison and store one record per run.

    python -m benchmarks.phase3.run            # everything (resumes: finished runs are kept)
    python -m benchmarks.phase3.run --quick    # two rounds, one seed, to check the pipeline

Grid: both tables; the research network and logistic regression; an unprotected baseline and
the arms of ``common.ARMS`` at their budgets; ten seeds for the stroke table and five for the
loan table. Settings come from ``settings.json`` (see ``tune.py``) and, for the arms that keep
the Phase 2 protocol, from ``benchmarks/phase2/settings.json``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path
from typing import Any

from benchmarks.phase2.common import BATCH_SIZE, LOCAL_EPOCHS, ROUNDS
from benchmarks.phase2.tables import make_clients
from benchmarks.phase3.common import ARMS, BASELINE, MODELS, SEEDS, Setting, run_setting

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "benchmarks" / "results" / "phase3"
PHASE2_SETTINGS = HERE.parent / "phase2" / "settings.json"
KEEPS_PHASE2 = ("local", "local_masked", "local_masked8")


def level(epsilon: float) -> str:
    """The tuning level of a budget: the low one (1) up to a budget of 2, else the high one (8)."""
    return "1" if epsilon <= 2.0 else "8"


def settings_for(
    dataset: str, seed: int, *, quick: bool = False, chosen: dict[str, Any] | None = None
) -> list[Setting]:
    """Every run of one table and seed."""
    phase2 = json.loads(PHASE2_SETTINGS.read_text())["chosen"]
    phase3 = (
        chosen if chosen is not None else json.loads((HERE / "settings.json").read_text())["chosen"]
    )
    out = []
    for model in MODELS:
        plain = phase2[f"{dataset}/{model}/none"]
        out.append(
            Setting(
                dataset, model, BASELINE, None, seed, plain["learning_rate"], plain["clip_norm"]
            )
        )
        for arm, (_, budgets) in ARMS.items():
            for epsilon in budgets:
                if arm in KEEPS_PHASE2:
                    tuned = phase2[f"{dataset}/{model}/{level(epsilon)}"]
                    rounds, batch, epochs = ROUNDS, BATCH_SIZE, LOCAL_EPOCHS
                else:
                    # The trusted server uses the shared arm's settings: the two then differ
                    # in the noise alone.
                    source = "shared" if arm == "server" else arm
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
    if quick:
        out[0] = Setting(dataset, out[0].model, BASELINE, None, seed, out[0].learning_rate, 1.0, 2)
    return out


def _run_group(group: tuple[str, int, bool, str]) -> list[str]:
    """All runs of one table and seed in one process, so noise calibrations are reused."""
    dataset, seed, quick, directory = group
    clients = make_clients(dataset, seed)
    done = []
    for setting in settings_for(dataset, seed, quick=quick):
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
    groups = [
        (dataset, seed, args.quick, str(directory))
        for dataset, seeds in SEEDS.items()
        for seed in (seeds[:1] if args.quick else seeds)
    ]
    # The stroke groups take longest; start them first.
    groups.sort(key=lambda group: group[0], reverse=True)
    with multiprocessing.Pool(args.processes) as pool:
        for labels in pool.imap_unordered(_run_group, groups, chunksize=1):
            first = labels[0]
            print(
                f"finished {len(labels)} runs: {first.split('_')[0]}, seed {first[-2:]}", flush=True
            )


if __name__ == "__main__":
    main()
