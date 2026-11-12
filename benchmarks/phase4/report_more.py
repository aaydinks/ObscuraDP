# ruff: noqa: E501 - this script builds wide Markdown tables from long string literals
"""Check the stored results of the further attacks (step S16) and build their tables.

    python -m benchmarks.phase4.report_more

Reads ``benchmarks/results/phase4/more/*.json`` (written by ``benchmarks.phase4.more``):
one file per shadow-model cell, one per seed for the rare rows, one for reading records
back. Every file is validated and every cell must be complete. Writes ``summary.json`` and
``summary.md`` next to them. Exits with status 1 if any check fails.

Figures over runs are means with a 95% interval from resampling whole runs. The 32 shadow
runs of a cell share one split of the table and learn from each other, so they are not
independent; their intervals are a guide, not a guarantee. The pooled lower limit on the
budget uses only runs 0 to 15, attacked with runs 16 to 31 as shadows, which keeps it valid.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.phase4.common import DATASETS, MODELS
from benchmarks.phase4.more import (
    GUESS_SHARE,
    HIDDEN_COLUMNS,
    INVERSION_BATCH,
    POOLED_TARGETS,
    PROTECTION,
    RARE_SEEDS,
    SHADOW_SEEDS,
    cell_name,
)
from obscuradp.accounting.accountants import general_advantage_ceiling
from obscuradp.audit.membership import (
    ATTACK_FINAL_MODEL,
    ATTACK_UPDATES,
    RunSummary,
    summarize,
)
from obscuradp.audit.shadow import ATTACK_SHADOW
from obscuradp.results import validate_result

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "benchmarks" / "results" / "phase4" / "more"
DELTA = 1e-5
PROTECTION_NAMES = {
    None: "no protection",
    1.0: "default plan, budget 1",
    8.0: "default plan, budget 8",
}
RARE_BUDGETS = (1.0, 8.0)


def mean_and_interval(values: list[float], seed: int = 0) -> dict[str, float]:
    """The mean of ``values`` and a 95% interval from resampling them."""
    numbers = np.asarray(values, dtype=np.float64)
    picks = np.random.default_rng(seed).integers(0, numbers.size, size=(2000, numbers.size))
    low, high = np.quantile(numbers[picks].mean(axis=1), [0.025, 0.975])
    return {
        "mean": float(numbers.mean()),
        "low": float(low),
        "high": float(high),
        "n": int(numbers.size),
    }


def _read(path: Path, problems: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        problems.append(f"{path.name}: missing")
        return None
    record = json.loads(path.read_text())
    try:
        validate_result(record)
    except ValueError as error:
        problems.append(f"{path.name}: {error}")
        return None
    return dict(record)


def shadow_rows(directory: Path, problems: list[str]) -> list[dict[str, Any]]:
    """One row per shadow-model cell."""
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            for arm, epsilon in PROTECTION:
                name = cell_name(dataset, model, arm, epsilon)
                record = _read(directory / f"shadow_{name}.json", problems)
                if record is None:
                    continue
                if record["manifest"]["seeds"] != list(SHADOW_SEEDS):
                    problems.append(f"{name}: not all {len(SHADOW_SEEDS)} shadow runs are there")
                data = record["data"]
                runs = data["runs"]
                for run in runs:
                    if epsilon is not None and not run["epsilon"] <= epsilon + 1e-9:
                        problems.append(
                            f"{name}, seed {run['seed']}: spent {run['epsilon']} of {epsilon}"
                        )
                targets = [runs[index] for index in POOLED_TARGETS if "pooled" in runs[index]]
                if len(targets) != len(POOLED_TARGETS):
                    problems.append(f"{name}: not all pooled target runs are there")
                pooled = summarize(
                    [
                        RunSummary(
                            canaries=run["pooled"]["canaries"],
                            guesses={
                                ATTACK_SHADOW: {f"{GUESS_SHARE:g}": tuple(run["pooled"]["guesses"])}
                            },
                            advantage={ATTACK_SHADOW: run["pooled"]["advantage"]},
                            epsilon=run["epsilon"],
                            delta=run["delta"],
                            ceiling_mechanism=run["ceiling_mechanism"],
                        )
                        for run in targets
                    ],
                    ATTACK_SHADOW,
                    GUESS_SHARE,
                    claimed_epsilon=1.0 if epsilon is None else None,
                    delta=DELTA if epsilon is None else None,
                )
                hidden = {}
                for column in HIDDEN_COLUMNS[dataset]:
                    used = [run["hidden_columns"][column]["used"] for run in runs]
                    unused = [run["hidden_columns"][column]["unused"] for run in runs]
                    hidden[column] = {
                        "used": mean_and_interval(used),
                        "unused": mean_and_interval(unused),
                        "used_minus_unused": mean_and_interval(
                            [a - b for a, b in zip(used, unused, strict=True)]
                        ),
                        "model_alone_used": mean_and_interval(
                            [run["hidden_columns"][column]["model_alone_used"] for run in runs]
                        ),
                        **data["without_model"][column],
                    }
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "arm": arm,
                        "budget": epsilon,
                        "runs": len(runs),
                        "rows": data["rows"],
                        "rows_label_1": data["rows_label_1"],
                        "ceiling_mechanism": None
                        if epsilon is None
                        else max(run["ceiling_mechanism"] for run in runs),
                        "ceiling_any": None
                        if epsilon is None
                        else general_advantage_ceiling(epsilon, DELTA),
                        "shadow": {
                            group: mean_and_interval(
                                [run["shadow_advantage"][group] for run in runs]
                            )
                            for group in ("all", "label_1", "label_0")
                        },
                        "plain": {
                            group: mean_and_interval(
                                [run["plain_advantage"][group] for run in runs]
                            )
                            for group in ("all", "label_1", "label_0")
                        },
                        "pooled_runs": pooled.runs,
                        "pooled_guesses": pooled.guesses,
                        "pooled_correct": pooled.correct,
                        "epsilon_lower": pooled.epsilon_lower,
                        "best_case": pooled.best_case,
                        "flagged": None if epsilon is None else pooled.flagged,
                        "hidden_columns": hidden,
                        "seconds": sum(run["seconds"] for run in runs),
                    }
                )
    return rows


def rare_rows(directory: Path, problems: list[str]) -> list[dict[str, Any]]:
    """The stroke audit read by class: one row per model and budget."""
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for seed in RARE_SEEDS:
        record = _read(directory / f"rare_stroke_seed-{seed}.json", problems)
        if record is None:
            continue
        for cell, stored in record["data"]["cells"].items():
            by_cell.setdefault(cell, []).append(stored)
    rows = []
    for model in MODELS:
        for epsilon in RARE_BUDGETS:
            runs = by_cell.get(cell_name("stroke", model, "local", epsilon), [])
            if len(runs) != len(RARE_SEEDS):
                problems.append(
                    f"rare rows, {model}, budget {epsilon:g}: {len(runs)} of {len(RARE_SEEDS)} seeds"
                )
            if not runs:
                continue
            rows.append(
                {
                    "model": model,
                    "budget": epsilon,
                    "runs": len(runs),
                    "canaries": sum(run["canaries"] for run in runs),
                    "canaries_label_1": sum(run["canaries_label_1"] for run in runs),
                    "ceiling_mechanism": max(run["ceiling_mechanism"] for run in runs),
                    "advantage": {
                        attack: {
                            group: mean_and_interval(
                                [run["advantage"][attack][group] for run in runs]
                            )
                            for group in ("all", "label_1", "label_0")
                        }
                        for attack in (ATTACK_UPDATES, ATTACK_FINAL_MODEL)
                    },
                }
            )
    return rows


# ---------------------------------------------------------------------- tables


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _interval(stat: dict[str, float]) -> str:
    return f"{_percent(stat['mean'])} ({_percent(stat['low'])} to {_percent(stat['high'])})"


def _points(stat: dict[str, float]) -> str:
    return f"{100 * stat['mean']:+.1f} ({100 * stat['low']:+.1f} to {100 * stat['high']:+.1f})"


def shadow_table(rows: list[dict[str, Any]], dataset: str) -> str:
    lines = [
        "| Model | Protection | Limit for the noise used | One cut-off for all records | Shadow models | Lower limit on the budget shown | Most the test could show |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row["dataset"] != dataset:
            continue
        best = "n/a" if row["best_case"] is None else f"{row['best_case']:.2f}"
        lines.append(
            f"| {row['model']} | {PROTECTION_NAMES[row['budget']]} | {_percent(row['ceiling_mechanism'])} "
            f"| {_interval(row['plain']['all'])} | {_interval(row['shadow']['all'])} | {row['epsilon_lower']:.2f} | {best} |"
        )
    return "\n".join(lines)


def class_table(rows: list[dict[str, Any]], rare: list[dict[str, Any]]) -> str:
    lines = [
        "| Model | Protection | Attack | Limit for the noise used | Rows of the rare class | Rows of the common class |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row["dataset"] != "stroke":
            continue
        lines.append(
            f"| {row['model']} | {PROTECTION_NAMES[row['budget']]} | shadow models (outsider) | {_percent(row['ceiling_mechanism'])} "
            f"| {_interval(row['shadow']['label_1'])} | {_interval(row['shadow']['label_0'])} |"
        )
    for row in rare:
        stats = row["advantage"][ATTACK_UPDATES]
        lines.append(
            f"| {row['model']} | {PROTECTION_NAMES[row['budget']]} | the plan's attacker, every round | {_percent(row['ceiling_mechanism'])} "
            f"| {_interval(stats['label_1'])} | {_interval(stats['label_0'])} |"
        )
    return "\n".join(lines)


def hidden_table(rows: list[dict[str, Any]], dataset: str) -> str:
    lines = [
        "| Hidden column | Model | Protection | Without any model | With the model, used rows | With the model, unused rows | Used minus unused (points) |",
        "|---|---|---|---|---|---|---|",
    ]
    for column in HIDDEN_COLUMNS[dataset]:
        for row in rows:
            if row["dataset"] != dataset:
                continue
            stats = row["hidden_columns"][column]
            lines.append(
                f"| {column} | {row['model']} | {PROTECTION_NAMES[row['budget']]} | {_percent(stats['with_outcome'])} "
                f"| {_percent(stats['used']['mean'])} | {_percent(stats['unused']['mean'])} | {_points(stats['used_minus_unused'])} |"
            )
    return "\n".join(lines)


INVERSION_VIEWS = (
    "alone, no protection",
    f"one of {INVERSION_BATCH}, no protection",
    "alone, default plan, budget 8",
    "alone, default plan, budget 1",
    "no update at all (the average record)",
)


def inversion_table(cells: dict[str, Any]) -> str:
    names = [f"{dataset}_{model}" for dataset in DATASETS for model in MODELS]
    names = [name for name in names if name in cells]
    lines = [
        "| What the attacker holds | "
        + " | ".join(name.replace("_", ", ") for name in names)
        + " |",
        "|---|" + "---|" * len(names),
    ]
    for view in INVERSION_VIEWS:
        lines.append(
            f"| {view} | "
            + " | ".join(f"{cells[name]['median_error'][view]:.2f}" for name in names)
            + " |"
        )
    return "\n".join(lines)


def headline_markdown(summary: dict[str, Any]) -> str:
    """The tables the results memo quotes.

    ``docs/PHASE4_MORE_ATTACKS.md`` contains this text exactly; a test compares the two.
    """
    rows, rare = summary["shadow"], summary["rare"]
    parts = []
    for dataset in DATASETS:
        parts += [
            f"**{dataset.capitalize()} table: an outsider with the final model, with and without shadow models** (advantage = true-positive rate minus false-positive rate; mean of {len(SHADOW_SEEDS)} runs with a 95% interval)",
            "",
            shadow_table(rows, dataset),
            "",
        ]
    parts += [
        f"**Stroke table: the rare class apart** (advantage by class; shadow models: mean of {len(SHADOW_SEEDS)} runs; the plan's attacker: mean of {len(RARE_SEEDS)} runs in which every row of the rare class is a test record)",
        "",
        class_table(rows, rare),
        "",
    ]
    for dataset in DATASETS:
        parts += [
            f"**{dataset.capitalize()} table: guessing a hidden yes/no column** (balanced accuracy, 50% is guessing; mean of {len(SHADOW_SEEDS)} runs)",
            "",
            hidden_table(rows, dataset),
            "",
        ]
    parts += [
        "**Reading a record back out of an update** (median distance between the record and what is read back, as a share of the record's length: 0 is a perfect copy, 1 is as far off as answering nothing)",
        "",
        inversion_table(summary["inversion"]),
        "",
    ]
    return "\n".join(parts).rstrip("\n")


def build_summary(directory: Path | None = None) -> dict[str, Any]:
    """The summary of the stored results."""
    directory = directory or RESULTS
    problems: list[str] = []
    shadow = shadow_rows(directory, problems)
    rare = rare_rows(directory, problems)
    record = _read(directory / "inversion.json", problems)
    return {
        "failed_checks": problems,
        "guess_share": GUESS_SHARE,
        "shadow_runs": sum(row["runs"] for row in shadow),
        "rare_runs": sum(row["runs"] for row in rare),
        "shadow": shadow,
        "rare": rare,
        "inversion": {} if record is None else record["data"]["cells"],
    }


def main() -> int:
    summary = build_summary()
    text = (
        "# Phase 4, further attacks: shadow models, a hidden column, reading records back\n\n"
        f"Built by `python -m benchmarks.phase4.report_more` from {summary['shadow_runs']} shadow runs, "
        f"{summary['rare_runs']} audited runs on the rare rows and one experiment on single updates.\n\n"
        + (headline_markdown(summary) if not summary["failed_checks"] else "")
        + "\n"
    )
    if summary["failed_checks"]:
        text += (
            "\n## Failed checks\n\n"
            + "\n".join(f"- {item}" for item in summary["failed_checks"])
            + "\n"
        )
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    (RESULTS / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"{len(summary['failed_checks'])} failed checks")
    return 1 if summary["failed_checks"] else 0


if __name__ == "__main__":
    sys.exit(main())
