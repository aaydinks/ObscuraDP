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


def _run_group(group: tuple[str, int, list[Setting], str, bool]) -> list[str]:
    """All runs on one dataset and seed in one process."""
    dataset, seed, settings, directory, keep_test = group
    pending = [s for s in settings if not (Path(directory) / f"{s.label}.json").exists()]
    if not pending:
        return [setting.label for setting in settings]
    clients = load(dataset, seed)
    for setting in pending:
        record = run_setting(setting, clients, keep_test=keep_test)
        write_record(Path(directory) / f"{setting.label}.json", record)
    return [setting.label for setting in settings]


def run_all(
    settings: list[Setting], directory: Path, *, processes: int = 4, keep_test: bool = True
) -> int:
    """Run every setting whose record is missing. Returns the number of settings."""
    directory.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, int], list[Setting]] = defaultdict(list)
    for setting in settings:
        groups[(setting.dataset, setting.seed)].append(setting)
    tasks = [
        (dataset, seed, members, str(directory), keep_test)
        for (dataset, seed), members in groups.items()
    ]
    with multiprocessing.Pool(processes) as pool:
        for labels in pool.imap_unordered(_run_group, tasks, chunksize=1):
            print(
                f"finished {len(labels)} runs: {labels[0].rsplit('_seed-', 1)[0]} ...", flush=True
            )
    return len(settings)
