"""The reported runs of the benchmark, as the written plan fixes them (sections 3, 4 and 10).

    python -m benchmarks.phase5.main tables
    python -m benchmarks.phase5.main confirm
    python -m benchmarks.phase5.main handwriting
    python -m benchmarks.phase5.main text

The stages can be run one after another or again after a stop: a finished run is kept and
never run again. The tables come first because they are never cut. Settings are those of
``benchmarks/results/phase5/settings.json`` (health table and handwriting, from the search),
of the earlier phases (stroke and loan), or fixed by the plan (text).
"""

from __future__ import annotations

import json
import multiprocessing
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from benchmarks.phase5.common import (
    BASELINE,
    BUDGETS,
    DATASETS,
    PARTICIPANT_ARM,
    ROW_ARMS,
    Setting,
    load,
    run_setting,
)
from benchmarks.phase5.pilot import planned
from benchmarks.phase5.run import RESULTS, run_all, write_record
from benchmarks.phase5.search import SETTINGS, key_of

RUNS = RESULTS / "runs"
CONFIRM = RESULTS / "confirm"
SEEDS = (1000, 1001, 1002, 1003, 1004)
TEXT_SEEDS = (1000, 1001, 1002)
TABLES = ("health", "stroke", "loan")
HANDWRITING_BUDGETS = (2.0, 4.0, 8.0)
SINGLE_RUN = (1.0, 1000)  # one handwriting run at a budget of 1, with this seed
TEXT_BUDGET = 8.0
# The small real case: the first writers by name, a short run, real against simulated secure sum.
CONFIRM_WRITERS = 60
CONFIRM_ROUNDS = 20
CONFIRM_BUDGET = 8.0
CONFIRM_SEEDS = (1000, 1001, 1002)
SECURE_SUMS = ("real", "simulated")
STAGES = ("tables", "confirm", "handwriting", "text")


def searched_budget(dataset: str, epsilon: float | None) -> float | None:
    """The searched budget whose choice a run at ``epsilon`` uses."""
    if epsilon is None:
        return None
    if dataset == "femnist":
        return 8.0  # searched at 8 only; that choice is used at every budget
    return 1.0 if epsilon <= 2.0 else 8.0


def reported(
    dataset: str,
    model: str,
    arm: str,
    epsilon: float | None,
    seed: int,
    chosen: dict[str, dict[str, Any]],
) -> Setting:
    """The setting of one reported run."""
    fixed = replace(planned(dataset, model, arm, epsilon), seed=seed)
    if dataset not in ("health", "femnist"):
        return fixed  # stroke and loan: the earlier phases; text: fixed by the plan
    pick = chosen[key_of(dataset, model, arm, searched_budget(dataset, epsilon))]
    assert (pick["rounds"], pick["batch_size"]) == (fixed.rounds, fixed.batch_size)
    return replace(fixed, learning_rate=pick["learning_rate"], clip_norm=pick["clip_norm"])


def chosen_settings(path: Path = SETTINGS) -> dict[str, dict[str, Any]]:
    """What the search chose. The reported runs cannot start without it."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: run the search first")
    chosen: dict[str, dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))["chosen"]
    return chosen


def table_settings(chosen: dict[str, dict[str, Any]]) -> list[Setting]:
    """Three tables, two models, no protection and three arms at five budgets, five seeds."""
    settings = []
    for dataset in TABLES:
        for model in DATASETS[dataset].models:
            for seed in SEEDS:
                settings.append(reported(dataset, model, BASELINE, None, seed, chosen))
                for arm in ROW_ARMS:
                    for epsilon in BUDGETS:
                        settings.append(reported(dataset, model, arm, epsilon, seed, chosen))
    return settings


def handwriting_settings(chosen: dict[str, dict[str, Any]]) -> list[Setting]:
    """Five seeds without protection and at three budgets, and one run at a budget of 1."""
    settings = [
        reported("femnist", "cnn", PARTICIPANT_ARM, epsilon, seed, chosen)
        for epsilon in HANDWRITING_BUDGETS
        for seed in SEEDS
    ]
    budget, seed = SINGLE_RUN
    settings.append(reported("femnist", "cnn", PARTICIPANT_ARM, budget, seed, chosen))
    settings += [reported("femnist", "cnn", BASELINE, None, seed, chosen) for seed in SEEDS]
    return settings


def text_settings() -> list[Setting]:
    """Protected runs at a budget of 8 and unprotected runs, three seeds."""
    settings = [
        reported("shakespeare", "gru", PARTICIPANT_ARM, TEXT_BUDGET, seed, {})
        for seed in TEXT_SEEDS
    ]
    settings += [reported("shakespeare", "gru", BASELINE, None, seed, {}) for seed in TEXT_SEEDS]
    return settings


def reported_settings(chosen: dict[str, dict[str, Any]]) -> list[Setting]:
    """Every reported run of the benchmark."""
    return table_settings(chosen) + handwriting_settings(chosen) + text_settings()


def by_time(settings: list[Setting], pilot_path: Path = RESULTS / "pilot.json") -> list[Setting]:
    """The same runs, the longest first by the pilot's estimate, so that no core idles late."""
    rows = json.loads(pilot_path.read_text(encoding="utf-8"))["rows"]
    seconds = {
        (row["dataset"], row["model"], row["arm"]): float(row["estimated_seconds_full_run"])
        for row in rows
    }
    return sorted(settings, key=lambda s: -seconds[(s.dataset, s.model, s.arm)])


def confirm_setting(seed: int, chosen: dict[str, dict[str, Any]]) -> Setting:
    """The small real case: a short protected run on the first writers."""
    full = reported("femnist", "cnn", PARTICIPANT_ARM, CONFIRM_BUDGET, seed, chosen)
    return replace(full, rounds=CONFIRM_ROUNDS)


def _confirm_one(task: tuple[Setting, str, str]) -> str:
    setting, secure_sum, directory = task
    stem = f"{setting.label}_{secure_sum}"
    path = Path(directory) / f"{stem}.json"
    if not path.exists():
        clients = load("femnist", setting.seed)[:CONFIRM_WRITERS]
        record = run_setting(setting, clients, real_secure_sum=secure_sum == "real")
        record["data"]["secure_sum"] = secure_sum
        write_record(path, record)
    return stem


def confirm(
    chosen: dict[str, dict[str, Any]], directory: Path = CONFIRM, processes: int = 4
) -> int:
    """Run the small real case with the real and with the simulated secure sum."""
    directory.mkdir(parents=True, exist_ok=True)
    tasks = [
        (confirm_setting(seed, chosen), secure_sum, str(directory))
        for seed in CONFIRM_SEEDS
        for secure_sum in SECURE_SUMS
    ]
    with multiprocessing.Pool(processes) as pool:
        for stem in pool.imap_unordered(_confirm_one, tasks, chunksize=1):
            print(f"finished {stem}", flush=True)
    return len(tasks)


def main(arguments: list[str] | None = None) -> None:
    stages = list(sys.argv[1:] if arguments is None else arguments) or list(STAGES)
    unknown = sorted(set(stages) - set(STAGES))
    if unknown:
        raise SystemExit(f"unknown stage {unknown[0]!r}; the stages are {', '.join(STAGES)}")
    chosen = chosen_settings()
    for stage in stages:
        if stage == "tables":
            run_all(by_time(table_settings(chosen)), RUNS, per_task=1)
        elif stage == "confirm":
            confirm(chosen)
        elif stage == "handwriting":
            run_all(handwriting_settings(chosen), RUNS, per_task=1)
        else:
            run_all(text_settings(), RUNS, per_task=1)
        print(f"stage {stage} is complete", flush=True)


if __name__ == "__main__":
    main()
