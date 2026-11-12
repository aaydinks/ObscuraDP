"""The timing pilot: how long does a run take, before any run that counts?

    python -m benchmarks.phase5.pilot

One seed (900) that no reported run uses. For every dataset and model: the unprotected
run in full, and every protected arm for a few rounds only, at a total budget of 1 or 8.
What is stored is how long things took and, for the full unprotected runs, the score on
**validation** rows, as a check that the pipeline learns at all. Scores on test rows are
not stored and not looked at. The result, ``benchmarks/results/phase5/pilot.json``, feeds
the time estimates of the written plan.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.phase3.run import PHASE2_SETTINGS
from benchmarks.phase5.common import (
    BASELINE,
    DATASETS,
    PROTOCOL,
    STARTING_POINT,
    Setting,
    arms_of,
)
from benchmarks.phase5.run import RESULTS, run_all

HERE = Path(__file__).resolve().parent
PHASE3_SETTINGS = HERE.parent / "phase3" / "settings.json"
PILOT_SEED = 900
SHORT_ROUNDS = 3  # protected arms are only timed, over this many rounds


def planned(dataset: str, model: str, arm: str, epsilon: float | None) -> Setting:
    """A full run at the pilot's seed, with the starting values where a search is planned."""
    if dataset in PROTOCOL:
        rounds, batch, epochs = PROTOCOL[dataset][arm]
        rate, clip = STARTING_POINT[dataset]
        return Setting(dataset, model, arm, epsilon, PILOT_SEED, rate, clip, rounds, batch, epochs)
    # Stroke and loan: the settings of Phases 2 and 3.
    level = "none" if epsilon is None else ("1" if epsilon <= 2.0 else "8")
    if arm in (BASELINE, "local"):
        tuned = json.loads(PHASE2_SETTINGS.read_text())["chosen"][f"{dataset}/{model}/{level}"]
        rounds, batch, epochs = 20, 32, 3.0
    else:
        tuned = json.loads(PHASE3_SETTINGS.read_text())["chosen"][
            f"{dataset}/{model}/{arm}/{level}"
        ]
        rounds, batch, epochs = (
            int(tuned["rounds"]),
            int(tuned["batch_size"]),
            float(tuned["local_epochs"]),
        )
    return Setting(
        dataset,
        model,
        arm,
        epsilon,
        PILOT_SEED,
        tuned["learning_rate"],
        tuned["clip_norm"],
        rounds,
        batch,
        epochs,
    )


def pilot_settings() -> tuple[list[Setting], dict[str, int]]:
    """The pilot's runs, and for each the number of rounds a full run would have."""
    settings, full_rounds = [], {}
    for dataset, data in DATASETS.items():
        for model in data.models:
            whole = planned(dataset, model, BASELINE, None)
            settings.append(whole)
            full_rounds[whole.label] = whole.rounds
            for arm in arms_of(dataset):
                epsilon = 8.0 if data.unit == "client" else 1.0
                full = planned(dataset, model, arm, epsilon)
                short = Setting(**{**full.__dict__, "rounds": min(SHORT_ROUNDS, full.rounds)})
                settings.append(short)
                full_rounds[short.label] = full.rounds
    return settings, full_rounds


def main() -> None:
    settings, full_rounds = pilot_settings()
    directory = RESULTS / "pilot"
    run_all(settings, directory, keep_test=False)
    rows: list[dict[str, Any]] = []
    for setting in settings:
        record = json.loads((directory / f"{setting.label}.json").read_text())
        data = record["data"]
        seconds = float(data["seconds"])
        per_round = seconds / setting.rounds
        measure = DATASETS[setting.dataset].measure
        rows.append(
            {
                "dataset": setting.dataset,
                "model": setting.model,
                "arm": setting.arm,
                "epsilon": setting.epsilon,
                "rounds_run": setting.rounds,
                "rounds_planned": full_rounds[setting.label],
                "seconds": round(seconds, 1),
                # A short run pays its fixed costs (working out the noise, the final
                # scoring) once, so this overestimates long runs a little. That is the safe side.
                "estimated_seconds_full_run": round(per_round * full_rounds[setting.label], 1),
                "validation": data["validation"].get(measure) if setting.arm == BASELINE else None,
            }
        )
    out = {
        "seed": PILOT_SEED,
        "note": "Timing only. Scores on test rows were neither stored nor looked at.",
        "rows": rows,
    }
    (RESULTS / "pilot.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
