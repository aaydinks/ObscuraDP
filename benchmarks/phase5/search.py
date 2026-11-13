"""The search for a learning rate and a clipping norm, as the written plan fixes it (section 5).

    python -m benchmarks.phase5.search

Health table and handwriting only. Stroke and loan keep the settings of the earlier phases,
and the text dataset is run without a search. The search uses seeds that no reported run
uses, and it reads **validation** rows only: scores on test rows are neither stored nor
looked at.

What is taken: the setting with the highest mean of the headline measure on validation
rows; on a tie the smaller learning rate, then the smaller clipping norm. A run whose score
is not a number counts as the worst.

Every run leaves a small record in ``benchmarks/results/phase5/search``; everything that
was tried and what was chosen is written to ``benchmarks/results/phase5/settings.json``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.phase5.common import BASELINE, DATASETS, PROTOCOL, Setting
from benchmarks.phase5.run import RESULTS, run_all

SEARCH = RESULTS / "search"
SETTINGS = RESULTS / "settings.json"
SEEDS = (901, 902, 903)
ONE_SEED = (901,)  # the larger models
UNUSED_CLIP = 1.0  # a run without protection shortens nothing

# (dataset, model, arm): learning rates, clipping norms, seeds. Section 5 of the written plan.
GRID: dict[tuple[str, str, str], tuple[tuple[float, ...], tuple[float, ...], tuple[int, ...]]] = {
    ("health", "logistic", BASELINE): ((0.05, 0.1, 0.2, 0.5), (UNUSED_CLIP,), SEEDS),
    ("health", "logistic", "local"): ((0.05, 0.1, 0.2, 0.5), (0.5, 1.0, 2.0), SEEDS),
    ("health", "logistic", "shared"): ((0.3, 1.0, 3.0), (0.5, 1.0, 2.0), SEEDS),
    ("health", "logistic", "shared_whole"): ((0.3, 1.0, 3.0), (0.5, 1.0, 2.0), SEEDS),
    ("health", "network", BASELINE): ((0.05, 0.1, 0.2, 0.5), (UNUSED_CLIP,), ONE_SEED),
    ("health", "network", "local"): ((0.05, 0.1, 0.2), (0.5, 1.0), ONE_SEED),
    ("health", "network", "shared"): ((0.3, 1.0, 3.0), (0.5, 1.0), ONE_SEED),
    ("health", "network", "shared_whole"): ((0.3, 1.0, 3.0), (0.5, 1.0), ONE_SEED),
    ("femnist", "cnn", BASELINE): ((0.05, 0.1, 0.2), (UNUSED_CLIP,), ONE_SEED),
    ("femnist", "cnn", "participant"): ((0.05, 0.1, 0.2), (0.5, 1.0), ONE_SEED),
}
# The budgets at which protected arms are searched.
BUDGETS = {"health": (1.0, 8.0), "femnist": (8.0,)}
# Long runs first, so that the four processes stay busy to the end.
ORDER = (
    ("femnist", "cnn", "participant"),
    ("health", "network", "shared"),
    ("health", "network", "local"),
    ("femnist", "cnn", BASELINE),
    ("health", "network", "shared_whole"),
    ("health", "logistic", "local"),
    ("health", "logistic", "shared"),
    ("health", "logistic", "shared_whole"),
    ("health", "network", BASELINE),
    ("health", "logistic", BASELINE),
)
PER_TASK = 1  # loading the participants is cheap; one run per task keeps the cores even


def level_name(epsilon: float | None) -> str:
    """The name under which a searched budget is stored."""
    return "none" if epsilon is None else f"{epsilon:g}"


def key_of(dataset: str, model: str, arm: str, epsilon: float | None) -> str:
    return f"{dataset}/{model}/{arm}/{level_name(epsilon)}"


def budgets_of(dataset: str, arm: str) -> tuple[float | None, ...]:
    return (None,) if arm == BASELINE else BUDGETS[dataset]


def search_settings() -> list[Setting]:
    """Every run of the search, long runs first."""
    assert set(ORDER) == set(GRID)
    settings = []
    for dataset, model, arm in ORDER:
        rates, clips, seeds = GRID[(dataset, model, arm)]
        rounds, batch, epochs = PROTOCOL[dataset][arm]
        for epsilon in budgets_of(dataset, arm):
            for rate in rates:
                for clip in clips:
                    for seed in seeds:
                        settings.append(
                            Setting(
                                dataset,
                                model,
                                arm,
                                epsilon,
                                seed,
                                rate,
                                clip,
                                rounds,
                                batch,
                                epochs,
                            )
                        )
    return settings


def file_name(setting: Setting) -> str:
    """A search record's file name. It carries the two searched values, which a label does not."""
    return f"{setting.label}_lr-{setting.learning_rate:g}_clip-{setting.clip_norm:g}"


def mean_or_worst(values: Sequence[float | None]) -> float:
    """The mean of the scores; minus infinity if one of them is missing or not a number."""
    if not values or any(value is None or not math.isfinite(value) for value in values):
        return -math.inf
    return float(sum(value for value in values if value is not None) / len(values))


def choose(tried: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """The setting that is taken: highest mean, then smaller learning rate, then smaller norm.

    Each entry of ``tried`` has ``learning_rate``, ``clip_norm`` and ``values`` (the scores
    on validation rows, one per seed).
    """
    if not tried:
        raise ValueError("nothing was tried")
    return min(
        tried,
        key=lambda row: (
            -mean_or_worst(row["values"]),
            row["learning_rate"],
            row["clip_norm"],
        ),
    )


def collect(directory: Path = SEARCH) -> dict[str, Any]:
    """Read the search's records and say what was tried and what is chosen."""
    tried: dict[str, list[dict[str, Any]]] = {}
    chosen: dict[str, dict[str, Any]] = {}
    seconds = 0.0
    commits: set[str] = set()
    for (dataset, model, arm), (rates, clips, seeds) in GRID.items():
        measure = DATASETS[dataset].measure
        rounds, batch, epochs = PROTOCOL[dataset][arm]
        for epsilon in budgets_of(dataset, arm):
            rows = []
            for rate in rates:
                for clip in clips:
                    values: list[float | None] = []
                    for seed in seeds:
                        setting = Setting(
                            dataset, model, arm, epsilon, seed, rate, clip, rounds, batch, epochs
                        )
                        record = json.loads(
                            (directory / f"{file_name(setting)}.json").read_text(encoding="utf-8")
                        )
                        if record["data"].get("test") is not None:
                            raise ValueError(f"{file_name(setting)} holds scores on test rows")
                        values.append(record["data"]["validation"].get(measure))
                        seconds += float(record["data"]["seconds"])
                        commits.add(str(record["manifest"]["git_commit"]))
                    mean = mean_or_worst(values)
                    rows.append(
                        {
                            "learning_rate": rate,
                            "clip_norm": clip,
                            "values": values,
                            "mean": mean if math.isfinite(mean) else None,
                        }
                    )
            key = key_of(dataset, model, arm, epsilon)
            tried[key] = rows
            best = choose(rows)
            chosen[key] = {
                "learning_rate": best["learning_rate"],
                "clip_norm": best["clip_norm"],
                "rounds": rounds,
                "batch_size": batch,
                "local_epochs": epochs,
                "validation_mean": best["mean"],
                "measure": measure,
                "seeds": list(seeds),
            }
    return {
        "note": (
            "Chosen on validation rows with seeds 901 to 903, which no reported run uses. "
            "Scores on test rows were neither stored nor looked at."
        ),
        "rule": (
            "highest mean on validation rows; on a tie the smaller learning rate, then the "
            "smaller clipping norm; a score that is not a number counts as the worst"
        ),
        "runs": sum(len(rows) * len(GRID[_grid_key(key)][2]) for key, rows in tried.items()),
        "computing_seconds": round(seconds),
        "commits": sorted(commits),
        "chosen": chosen,
        "tried": tried,
    }


def _grid_key(key: str) -> tuple[str, str, str]:
    dataset, model, arm, _ = key.split("/")
    return dataset, model, arm


def main() -> None:
    settings = search_settings()
    run_all(
        settings,
        SEARCH,
        keep_test=False,
        slimmed=True,
        per_task=PER_TASK,
        name=file_name,
    )
    result = collect()
    SETTINGS.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for key, value in result["chosen"].items():
        print(key, value["learning_rate"], value["clip_norm"], value["validation_mean"])


if __name__ == "__main__":
    main()
