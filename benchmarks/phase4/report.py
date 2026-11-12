# ruff: noqa: E501 - this script builds wide Markdown tables from long string literals
"""Check every stored run of the Phase 4 attack tests and build their tables.

    python -m benchmarks.phase4.report            # summary.json and summary.md from the runs
    python -m benchmarks.phase4.report --pilot    # settings.json from the pilot runs

Reads ``benchmarks/results/phase4/runs/*.json`` (one file per table and seed). Every file
is validated, every stored privacy report is recomputed from its recorded releases, and no
run may be over its budget. Then the runs of every cell (table, model, arm, budget) are put
together (``obscuradp.audit.membership.summarize``): the pooled lower limit on the budget,
the measured advantage with its interval, the two proven limits and the most the test
could have shown. Exits with status 1 if any check fails.

For how many of its test records an attack guesses is fixed **before** the reported runs:
``--pilot`` reads the pilot runs (other seeds), picks for each attack the share of records
with the highest average lower limit over the protected cells, and writes it to
``settings.json``. The reported runs are then read with that share and no other.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks.phase4.common import ARMS, BASELINE, BUDGETS, DATASETS, MODELS, SEEDS, SILENT
from obscuradp.audit.membership import (
    ATTACK_FINAL_MODEL,
    ATTACK_UPDATES,
    ATTACKS,
    GUESS_SHARES,
    RunSummary,
    summarize,
)
from obscuradp.report import verify_report
from obscuradp.results import validate_result

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "benchmarks" / "results" / "phase4"
SETTINGS = HERE / "settings.json"
DELTA = 1e-5
TARGET = 0.05  # the website's target for the measured advantage at a budget of 1

NAMES = {
    BASELINE: "no protection",
    "local": "default plan (local noise)",
    SILENT: "planted defect: default plan, noise switched off",
    "shared": "shared noise, ordinary",
    "shared_whole": "shared noise, whole numbers",
}
ATTACK_NAMES = {
    ATTACK_UPDATES: "the plan's attacker, every round",
    ATTACK_FINAL_MODEL: "an outsider with the final model",
}
# What the planted defect and the unprotected runs are held against: the budget the
# defective pipeline states, and the same figure as a yardstick for no protection.
CLAIMED_WITHOUT_REPORT = 1.0


def load_runs(
    directory: Path | None = None,
) -> tuple[dict[str, dict[int, dict[str, Any]]], list[str]]:
    """Every stored run by cell and seed, plus a list of whatever failed its checks."""
    cells: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    problems = []
    for path in sorted((directory or RESULTS / "runs").glob("*.json")):
        record = json.loads(path.read_text())
        try:
            validate_result(record)
        except ValueError as error:
            problems.append(f"{path.name}: {error}")
            continue
        (seed,) = record["manifest"]["seeds"]
        for cell, run in record["data"]["cells"].items():
            report = run["privacy_report"]
            if report is not None:
                problems += [f"{path.name}, {cell}: {text}" for text in verify_report(report)]
                if not report["within_budget"]:
                    problems.append(f"{path.name}, {cell}: over budget")
            budget = cell.rsplit("_eps-", 1)[1]
            if budget != "none" and not run["epsilon"] <= float(budget) + 1e-9:
                problems.append(f"{path.name}, {cell}: spent {run['epsilon']} of {budget}")
            cells[cell][seed] = run
    return dict(cells), problems


def cell_name(dataset: str, model: str, arm: str, epsilon: float | None) -> str:
    budget = "none" if epsilon is None else f"{epsilon:g}"
    return f"{dataset}_{model}_{arm}_eps-{budget}"


def grid() -> list[tuple[str, str, str, float | None]]:
    """Every cell of the reported runs."""
    return [
        (dataset, model, arm, epsilon)
        for dataset in DATASETS
        for arm, (_, budgets, models) in ARMS.items()
        for model in models
        for epsilon in budgets
    ]


def _summaries(runs: dict[int, dict[str, Any]]) -> list[RunSummary]:
    return [RunSummary.from_dict(runs[seed]) for seed in sorted(runs)]


def summarize_cells(
    cells: dict[str, dict[int, dict[str, Any]]], shares: dict[str, float]
) -> list[dict[str, Any]]:
    """One row per cell and attack."""
    rows = []
    for dataset, model, arm, epsilon in grid():
        runs = cells.get(cell_name(dataset, model, arm, epsilon))
        if not runs:
            continue
        stated = arm != BASELINE
        for attack in ATTACKS:
            found = summarize(
                _summaries(runs),
                attack,
                shares[attack],
                claimed_epsilon=None if arm not in (BASELINE, SILENT) else CLAIMED_WITHOUT_REPORT,
                delta=None if arm != BASELINE else DELTA,
            )
            row = found.to_dict()
            if not stated:
                # No budget is stated without protection: nothing to be flagged against.
                row.update(
                    epsilon_claimed=None, ceiling_any=None, flagged=None, above_noise_limit=None
                )
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "arm": arm,
                    "budget": epsilon,
                    "seeds": sorted(runs),
                    "attacker": runs[min(runs)]["attacker"],
                    "colluders": runs[min(runs)]["colluders"],
                    "test_auc": sum(run["test_auc"] for run in runs.values()) / len(runs),
                    "seconds": sum(run["seconds"] for run in runs.values()),
                    **row,
                }
            )
    return rows


# ---------------------------------------------------------------------- the pilot


def choose_shares(cells: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    """For each attack, the share of test records to guess for: the one that proves most.

    Judged on the pilot's protected cells only (a stated budget and its noise in place),
    by the average pooled lower limit. On a tie the larger share wins.
    """
    protected = {
        cell: runs
        for cell, runs in cells.items()
        if not any(f"_{arm}_eps" in cell for arm in (BASELINE, SILENT))
    }
    table: dict[str, dict[str, float]] = {}
    chosen: dict[str, float] = {}
    for attack in ATTACKS:
        table[attack] = {}
        for share in GUESS_SHARES:
            limits = [
                summarize(_summaries(runs), attack, share).epsilon_lower
                for runs in protected.values()
            ]
            table[attack][f"{share:g}"] = sum(limits) / len(limits)
        chosen[attack] = max(GUESS_SHARES, key=lambda share: (table[attack][f"{share:g}"], share))
    seconds = sum(run["seconds"] for runs in cells.values() for run in runs.values())
    return {
        "note": "Chosen on the pilot runs (seeds 90 to 94), which are not part of any reported number.",
        "guess_share": chosen,
        "mean_lower_limit_by_share": table,
        "pilot_cells": len(protected),
        "pilot_runs": sum(len(runs) for runs in cells.values()),
        "pilot_seconds": round(seconds),
    }


# ---------------------------------------------------------------------- tables


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _budget(epsilon: float | None) -> str:
    return "none" if epsilon is None else f"{epsilon:g}"


def _index(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {(r["dataset"], r["model"], r["arm"], r["budget"], r["attack"]): r for r in rows}


def _measured(row: dict[str, Any]) -> str:
    return f"{_percent(row['advantage'])} ({_percent(row['advantage_low'])} to {_percent(row['advantage_high'])})"


def _verdict(row: dict[str, Any]) -> str:
    """Whether the audit raised a flag, and by which of its two signs."""
    if row["flagged"] is None:
        return "n/a"
    signs = []
    if row["flagged"]:
        signs.append("lower limit above the stated budget")
    if row["above_noise_limit"]:
        signs.append("advantage above the limit for the stated noise")
    return f"**yes** ({'; '.join(signs)})" if signs else "no"


def three_numbers_table(rows: list[dict[str, Any]], dataset: str, model: str) -> str:
    """The default plan at every budget: both proven limits beside both measured values."""
    index = _index(rows)
    lines = [
        "| Total budget | Limit for any method | Limit for the noise used | Measured: the plan's attacker | Measured: outsider with the final model | Lower limit on the budget shown | Most the test could show |",
        "|---|---|---|---|---|---|---|",
    ]
    for epsilon in BUDGETS:
        strong = index.get((dataset, model, "local", epsilon, ATTACK_UPDATES))
        weak = index.get((dataset, model, "local", epsilon, ATTACK_FINAL_MODEL))
        if strong is None or weak is None:
            continue
        lines.append(
            f"| {epsilon:g} | {_percent(strong['ceiling_any'])} | {_percent(strong['ceiling_mechanism'])} "
            f"| {_measured(strong)} | {_measured(weak)} | {strong['epsilon_lower']:.2f} | {strong['best_case']:.2f} |"
        )
    return "\n".join(lines)


def other_arms_table(rows: list[dict[str, Any]], dataset: str) -> str:
    """Everything that is not the default plan, with the update attack."""
    index = _index(rows)
    lines = [
        "| Model | Arm | Stated budget | Limit for the noise used | Measured: the plan's attacker | Measured: outsider | Lower limit shown | Most the test could show | Flagged |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for model in MODELS:
        for arm, (_, budgets, models) in ARMS.items():
            if arm == "local" or model not in models:
                continue
            for epsilon in budgets:
                strong = index.get((dataset, model, arm, epsilon, ATTACK_UPDATES))
                weak = index.get((dataset, model, arm, epsilon, ATTACK_FINAL_MODEL))
                if strong is None or weak is None:
                    continue
                stated = "none" if arm == BASELINE else f"{strong['epsilon_claimed']:.2g}"
                best = "n/a" if strong["best_case"] is None else f"{strong['best_case']:.2f}"
                flagged = _verdict(strong)
                lines.append(
                    f"| {model} | {NAMES[arm]} | {stated} | {_percent(strong['ceiling_mechanism'])} "
                    f"| {_measured(strong)} | {_measured(weak)} | {strong['epsilon_lower']:.2f} | {best} | {flagged} |"
                )
    return "\n".join(lines)


def target_table(rows: list[dict[str, Any]]) -> str:
    """The website's target: at most 5% measured advantage at a total budget of 1."""
    index = _index(rows)
    lines = [
        "| Table | Model | Measured at a budget of 1 (the plan's attacker) | Upper end of the interval | Within 5%? | Largest budget whose upper end is within 5% |",
        "|---|---|---|---|---|---|",
    ]
    for dataset in DATASETS:
        for model in MODELS:
            row = index.get((dataset, model, "local", 1.0, ATTACK_UPDATES))
            if row is None:
                continue
            within = [
                epsilon
                for epsilon in BUDGETS
                if (dataset, model, "local", epsilon, ATTACK_UPDATES) in index
                and index[(dataset, model, "local", epsilon, ATTACK_UPDATES)]["advantage_high"]
                <= TARGET
            ]
            lines.append(
                f"| {dataset} | {model} | {_percent(row['advantage'])} | {_percent(row['advantage_high'])} "
                f"| {'yes' if row['advantage_high'] <= TARGET else '**no**'} | {max(within):g} |"
                if within
                else f"| {dataset} | {model} | {_percent(row['advantage'])} | {_percent(row['advantage_high'])} | **no** | none tried |"
            )
    return "\n".join(lines)


def headline_markdown(rows: list[dict[str, Any]]) -> str:
    """The tables the results memo quotes.

    ``docs/PHASE4_ATTACK_TESTS.md`` contains this text exactly; a test compares the two.
    """
    seeds = len(SEEDS)
    parts = [
        f"**The website's target, read off the default plan** (measured advantage of the plan's attacker, mean of {seeds} runs; the interval is 95%, from resampling whole runs)",
        "",
        target_table(rows),
        "",
    ]
    for dataset in DATASETS:
        for model in MODELS:
            parts += [
                f"**{dataset.capitalize()} table, {model}: the default plan, three numbers side by side** (advantage = true-positive rate minus false-positive rate; measured values are means of {seeds} runs with a 95% interval)",
                "",
                three_numbers_table(rows, dataset, model),
                "",
            ]
    for dataset in DATASETS:
        parts += [
            f"**{dataset.capitalize()} table: the other arms**",
            "",
            other_arms_table(rows, dataset),
            "",
        ]
    return "\n".join(parts).rstrip("\n")


def markdown(
    rows: list[dict[str, Any]], problems: list[str], count: int, settings: dict[str, Any]
) -> str:
    shares = settings["guess_share"]
    lines = [
        "# Phase 4 attack tests: what an attacker can tell about single records",
        "",
        f"Built by `python -m benchmarks.phase4.report` from {count} stored runs. Every file was validated and every stored privacy report was recomputed from its recorded releases.",
        "",
        "Each run trains once on a real table in which about a quarter of every participant's training rows (at most 250) are test records, each kept or left out by a fair coin. Advantage is the attacker's true-positive rate minus its false-positive rate at a cut-off chosen on other records. The lower limit on the budget pools the guesses of all runs of a cell (one-run method of Steinke, Nasr and Jagielski, 2023), at 95% confidence.",
        "",
        f"Share of the test records an attack guesses for, fixed on the pilot runs before these runs: {shares[ATTACK_UPDATES]:g} for the plan's attacker, {shares[ATTACK_FINAL_MODEL]:g} for the outsider.",
        "",
        headline_markdown(rows),
        "",
        "## Every cell",
        "",
        "| Table | Model | Arm | Budget | Attack | Runs | Test records | Guesses | Right | Lower limit (pooled) | Median of single runs | Most it could show | Advantage | Mean test AUC |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        best = "n/a" if row["best_case"] is None else f"{row['best_case']:.2f}"
        lines.append(
            f"| {row['dataset']} | {row['model']} | {NAMES[row['arm']]} | {_budget(row['budget'])} | {ATTACK_NAMES[row['attack']]} "
            f"| {row['runs']} | {row['canaries']} | {row['guesses']} | {row['correct']} | {row['epsilon_lower']:.3f} "
            f"| {row['epsilon_lower_median_run']:.3f} | {best} | {_measured(row)} | {row['test_auc']:.3f} |"
        )
    if problems:
        lines += ["", "## Failed checks", ""] + [f"- {text}" for text in problems]
    return "\n".join(lines).rstrip("\n") + "\n"


def build_summary(directory: Path | None = None) -> tuple[dict[str, Any], str]:
    """The summary of the stored runs and its Markdown form."""
    cells, problems = load_runs(directory)
    settings = json.loads(SETTINGS.read_text())
    shares = {attack: float(settings["guess_share"][attack]) for attack in ATTACKS}
    rows = summarize_cells(cells, shares)
    count = sum(len(runs) for runs in cells.values())
    missing = [
        cell_name(*cell)
        for cell in grid()
        if sorted(cells.get(cell_name(*cell), {})) != list(SEEDS)
    ]
    problems += [f"{cell}: not all {len(SEEDS)} seeds are there" for cell in missing]
    summary = {
        "runs": count,
        "failed_checks": problems,
        "budgets": list(BUDGETS),
        "seeds": list(SEEDS),
        "guess_share": shares,
        "rows": rows,
    }
    return summary, markdown(rows, problems, count, settings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--pilot", action="store_true", help="choose the guess shares from the pilot"
    )
    args = parser.parse_args()
    if args.pilot:
        cells, problems = load_runs(RESULTS / "pilot")
        settings = choose_shares(cells)
        SETTINGS.write_text(json.dumps(settings, indent=1, sort_keys=True) + "\n")
        print(json.dumps(settings, indent=1, sort_keys=True))
        for text in problems[:20]:
            print("  -", text)
        return 1 if problems else 0
    summary, text = build_summary()
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    (RESULTS / "summary.md").write_text(text, encoding="utf-8")
    print(headline_markdown(summary["rows"]))
    print(f"{summary['runs']} runs, {len(summary['failed_checks'])} failed checks")
    for problem in summary["failed_checks"][:20]:
        print("  -", problem)
    return 1 if summary["failed_checks"] else 0


if __name__ == "__main__":
    sys.exit(main())
