"""Run benchmark settings and store one record per run. Finished runs are kept.

This module is the machinery: it takes a list of settings and runs those whose record is
not there yet, a few processes side by side, grouped so that runs on the same participants
share the loaded data. Which settings make up the benchmark is decided by the written plan
(``benchmarks/PREREGISTRATION.md``) and added in the step that runs it.
"""

from __future__ import annotations

import json
import multiprocessing
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmarks.phase5.common import Setting, load, run_setting

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "benchmarks" / "results" / "phase5"


def write_record(path: Path, record: dict[str, Any]) -> None:
    """Write a record whole and then move it into place, so a stopped run leaves no torn file."""
    scratch = path.with_suffix(".part")
    scratch.write_text(json.dumps(record, indent=1, sort_keys=True, allow_nan=False) + "\n")
    scratch.replace(path)


def slim(record: dict[str, Any]) -> dict[str, Any]:
    """What a search needs of a record: validation scores, the time and the budget spent."""
    data = record["data"]
    report = data.get("privacy_report") or {}
    kept = {
        "label": data["label"],
        "validation": data["validation"],
        "seconds": data["seconds"],
        "epsilon_spent": report.get("epsilon"),
        "within_budget": report.get("within_budget"),
    }
    return {**record, "data": kept}


def label_of(setting: Setting) -> str:
    """The usual file name of a run's record."""
    return setting.label


Group = tuple[str, int, list[Setting], str, bool, bool, Callable[[Setting], str]]


def _run_group(group: Group) -> list[str]:
    """Some runs on one dataset and seed in one process."""
    dataset, seed, settings, directory, keep_test, slimmed, name = group
    names = [name(setting) for setting in settings]
    pending = [
        (setting, stem)
        for setting, stem in zip(settings, names, strict=True)
        if not (Path(directory) / f"{stem}.json").exists()
    ]
    if not pending:
        return names
    clients = load(dataset, seed)
    for setting, stem in pending:
        record = run_setting(setting, clients, keep_test=keep_test)
        if slimmed:
            record = slim(record)
        write_record(Path(directory) / f"{stem}.json", record)
    return names


def run_all(
    settings: list[Setting],
    directory: Path,
    *,
    processes: int = 4,
    keep_test: bool = True,
    slimmed: bool = False,
    per_task: int | None = None,
    name: Callable[[Setting], str] = label_of,
) -> int:
    """Run every setting whose record is missing. Returns the number of settings.

    Runs on the same dataset and seed share one process and the loaded data, at most
    ``per_task`` of them (all, if it is not given). Tasks start in the order in which their
    first setting is listed, so the caller can put the long runs first. ``slimmed`` stores
    only what a search needs (see :func:`slim`). ``name`` gives a record's file name (a
    function of a module, so that it can be handed to the worker processes).
    """
    directory.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, int], list[Setting]] = defaultdict(list)
    for setting in settings:
        groups[(setting.dataset, setting.seed)].append(setting)
    tasks = []
    for (dataset, seed), members in groups.items():
        size = per_task or len(members)
        for begin in range(0, len(members), size):
            piece = members[begin : begin + size]
            tasks.append((dataset, seed, piece, str(directory), keep_test, slimmed, name))
    position = {id(setting): index for index, setting in enumerate(settings)}
    tasks.sort(key=lambda task: position[id(task[2][0])])
    with multiprocessing.Pool(processes) as pool:
        for labels in pool.imap_unordered(_run_group, tasks, chunksize=1):
            print(f"finished {len(labels)} runs: {labels[0]} ...", flush=True)
    return len(settings)
