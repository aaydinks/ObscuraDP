# ruff: noqa: E501 - this script builds wide Markdown tables from long string literals
"""Check every stored run of the Phase 3 comparison and build its tables.

    python -m benchmarks.phase3.report

Reads ``benchmarks/results/phase3/runs/*.json``. Every record is validated, every stored
privacy report is recomputed from its recorded releases (the same check as ``obscuradp
report verify``), and the runs of the ``local`` arm are compared with the stored Phase 2
runs of the same table, model, budget and seed, which they must equal exactly. Writes
``summary.json`` and ``summary.md`` next to the runs. Exits with status 1 if any check fails.

A difference between two arms is computed seed by seed (the same seed means the same split
of the table), and counts as shown only if the 95% interval of those differences does not
contain zero.

The yardstick for the new arms is "the better local arm": of the two arms in which every
participant adds all of its own noise (the Phase 2 protocol, and the same with rounds chosen
on made-up tables), the one with the higher mean AUC at that table, model and budget. It is
picked after looking at the held-out rows, which can only favour local noise, so a gain
shown against it is on the safe side.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks.phase2.common import mean_and_interval
from benchmarks.phase3.common import ARMS, BASELINE, BUDGETS, FEW_BUDGETS, MODELS, SEEDS
from obscuradp.report import verify_report
from obscuradp.results import validate_result

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "benchmarks" / "results" / "phase3"
PHASE2_RUNS = ROOT / "benchmarks" / "results" / "phase2" / "runs"

DATASETS = ("stroke", "loan")
REFERENCES = ("local", "local_tuned")
COMPARED = ("local_tuned", "shared", "shared_whole", "server")
SMALL_NUMBERS = ("local_masked", "local_masked8")
NAMES = {
    BASELINE: "no protection",
    "local": "local noise (Phase 2 protocol)",
    "local_tuned": "local noise, rounds tuned",
    "shared": "shared noise, ordinary",
    "shared_whole": "shared noise, whole numbers",
    "server": "trusted server (upper limit)",
    "local_masked": "local noise, secure sum, 12-bit values",
    "local_masked8": "local noise, secure sum, 8-bit values",
    "participant": "whole participants protected",
}


def load_runs(directory: Path | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    """Every stored run, plus a list of whatever failed its checks."""
    runs, problems = [], []
    for path in sorted((directory or RESULTS / "runs").glob("*.json")):
        record = json.loads(path.read_text())
        try:
            validate_result(record)
        except ValueError as error:
            problems.append(f"{path.name}: {error}")
            continue
        report = record["data"]["privacy_report"]
        if report is not None:
            problems += [f"{path.name}: {text}" for text in verify_report(report)]
            if not report["within_budget"]:
                problems.append(f"{path.name}: over budget")
        runs.append(record)
    return runs, problems


def check_against_phase2(runs: list[dict[str, Any]]) -> tuple[int, list[str]]:
    """The ``local`` arm repeats Phase 2: same table, model, budget and seed, same numbers."""
    compared, problems = 0, []
    for record in runs:
        setting = record["config"]["benchmark"]
        if setting["arm"] not in ("local", BASELINE):
            continue
        budget = "none" if setting["epsilon"] is None else f"{setting['epsilon']:g}"
        name = f"{setting['dataset']}_{setting['model']}_eps-{budget}_clear_seed-{setting['seed']}.json"
        path = PHASE2_RUNS / name
        if not path.exists():
            continue  # Phase 2 had five seeds; the others have nothing to be compared with
        earlier = json.loads(path.read_text())["data"]
        compared += 1
        for part in ("test", "validation", "test_at_chosen_threshold"):
            if earlier[part] != record["data"][part]:
                problems.append(f"{record['data']['label']}: {part} differs from the Phase 2 run")
    return compared, problems


def _differences(
    members: dict[int, dict[str, Any]], reference: dict[int, dict[str, Any]], measure: str
) -> dict[str, Any]:
    """Seed-by-seed difference of a test measure against a reference arm."""
    seeds = sorted(set(members) & set(reference))
    values = [members[s]["test"][measure] - reference[s]["test"][measure] for s in seeds]
    stat = mean_and_interval(values)
    shown = stat["n"] >= 2 and (stat["low"] > 0.0 or stat["high"] < 0.0)
    verdict = "no difference shown"
    if shown:
        verdict = "better" if stat["mean"] > 0 else "worse"
    return {**stat, "verdict": verdict}


def summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per table, model, arm and budget, over the seeds."""
    groups: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = defaultdict(dict)
    settings: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in runs:
        setting = record["config"]["benchmark"]
        key = (setting["dataset"], setting["model"], setting["arm"], setting["epsilon"])
        groups[key][setting["seed"]] = record["data"]
        settings[key] = setting
    rows = []
    for key, members in groups.items():
        dataset, model, arm, epsilon = key
        data = [members[seed] for seed in sorted(members)]
        setting = settings[key]
        row: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "arm": arm,
            "epsilon": epsilon,
            "seeds": sorted(members),
            "learning_rate": setting["learning_rate"],
            "clip_norm": setting["clip_norm"],
            "rounds": setting["rounds"],
            "batch_size": setting["batch_size"],
            "local_epochs": setting["local_epochs"],
            "test_positive_rows": [d["test_positive_rows"] for d in data],
            "bits_per_value": data[0]["communication"]["bits_per_value"],
            "upload_ratio_per_round": data[0]["communication"]["ratio"],
            "upload_bytes_per_round": data[0]["communication"]["upload_bytes_per_round"],
            "upload_bytes_per_run": data[0]["upload_bytes_per_run"],
            "saturated_share_max": max(d["communication"]["saturated_share"] for d in data),
            "seconds_mean": sum(d["seconds"] for d in data) / len(data),
        }
        for name in ("auc", "ap"):
            row[f"test_{name}"] = mean_and_interval([d["test"][name] for d in data])
        for name in ("recall", "precision", "f1"):
            row[f"test_chosen_{name}"] = mean_and_interval(
                [d["test_at_chosen_threshold"][name] for d in data]
            )
        if epsilon is not None:
            reports = [d["privacy_report"] for d in data]
            sigmas = [value for d in data for value in d["noise_multipliers"].values()]
            row["epsilon_spent_max"] = max(r["epsilon"] for r in reports)
            row["noise_multiplier_min"] = min(sigmas)
            row["noise_multiplier_max"] = max(sigmas)
            row["attacker_advantage_ceiling_max"] = max(
                r["attacker_advantage_ceiling"] for r in reports
            )
            row["all_within_budget"] = all(r["within_budget"] for r in reports)
            row["unsafe_flags"] = sorted({flag for r in reports for flag in r["unsafe_flags"]})
            for reference in REFERENCES:
                other = groups.get((dataset, model, reference, epsilon))
                if other is not None and arm != reference:
                    for name in ("auc", "ap"):
                        row[f"{name}_minus_{reference}"] = _differences(members, other, name)
            both = [groups.get((dataset, model, reference, epsilon)) for reference in REFERENCES]
            if arm not in REFERENCES and all(other is not None for other in both):
                means = [
                    sum(d["test"]["auc"] for d in other.values()) / len(other)  # type: ignore[union-attr]
                    for other in both
                ]
                best = 0 if means[0] >= means[1] else 1
                row["best_local_arm"] = REFERENCES[best]
                for name in ("auc", "ap"):
                    row[f"{name}_minus_best_local"] = _differences(members, both[best], name)  # type: ignore[arg-type]
        rows.append(row)
    order = [BASELINE, *ARMS]
    rows.sort(
        key=lambda r: (
            DATASETS.index(r["dataset"]),
            MODELS.index(r["model"]),
            order.index(r["arm"]),
            -1.0 if r["epsilon"] is None else r["epsilon"],
        )
    )
    return rows


def _cell(stat: dict[str, float]) -> str:
    if stat["n"] == 0:
        return "n/a"
    if stat["n"] == 1:
        return f"{stat['mean']:.3f}"
    return f"{stat['mean']:.3f} ± {stat['std']:.3f}"


def _difference_cell(stat: dict[str, Any]) -> str:
    mark = {"better": " **better**", "worse": " **worse**"}.get(stat["verdict"], "")
    return f"{stat['mean']:+.3f} ({stat['low']:+.3f} to {stat['high']:+.3f}){mark}"


def _index(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {(r["dataset"], r["model"], r["arm"], r["epsilon"]): r for r in rows}


def measure_table(rows: list[dict[str, Any]], dataset: str, measure: str) -> str:
    """One measure for every arm at every budget, mean of the seeds plus or minus their spread."""
    index = _index(rows)
    lines = [
        "| Model | Arm | " + " | ".join(f"{b:g}" for b in BUDGETS) + " |",
        "|---|---|" + "---|" * len(BUDGETS),
    ]
    for model in MODELS:
        plain = index.get((dataset, model, BASELINE, None))
        if plain is not None:
            cells = [_cell(plain[measure])] + ["" for _ in BUDGETS[1:]]
            lines.append(f"| {model} | {NAMES[BASELINE]} (no budget) | " + " | ".join(cells) + " |")
        for arm in ARMS:
            cells = []
            for budget in BUDGETS:
                row = index.get((dataset, model, arm, budget))
                cells.append("" if row is None else _cell(row[measure]))
            lines.append(f"| {model} | {NAMES[arm]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def difference_table(rows: list[dict[str, Any]], dataset: str, measure: str, reference: str) -> str:
    """Seed-by-seed differences against a reference arm, with 95% intervals."""
    index = _index(rows)
    lines = [
        "| Model | Arm | " + " | ".join(f"{b:g}" for b in BUDGETS) + " |",
        "|---|---|" + "---|" * len(BUDGETS),
    ]
    for model in MODELS:
        for arm in COMPARED:
            if arm == reference or (reference == "best_local" and arm in REFERENCES):
                continue
            cells = []
            for budget in BUDGETS:
                row = index.get((dataset, model, arm, budget))
                stat = None if row is None else row.get(f"{measure}_minus_{reference}")
                cells.append("" if stat is None else _difference_cell(stat))
            lines.append(f"| {model} | {NAMES[arm]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def small_numbers_table(rows: list[dict[str, Any]], dataset: str) -> str:
    """Local noise through the secure sum with 12-bit and 8-bit values, against the clear."""
    index = _index(rows)
    lines = ["| Model | Arm | 1 | 8 |", "|---|---|---|---|"]
    for model in MODELS:
        for arm in SMALL_NUMBERS:
            cells = []
            for budget in FEW_BUDGETS:
                row = index.get((dataset, model, arm, budget))
                stat = None if row is None else row.get("auc_minus_local")
                cells.append("" if stat is None else _difference_cell(stat))
            lines.append(f"| {model} | {NAMES[arm]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def message_table(rows: list[dict[str, Any]], dataset: str) -> str:
    """What one participant uploads, per round and over the whole run, at a budget of 1."""
    index = _index(rows)
    lines = [
        "| Model | Arm | Rounds | Bits per number | Per round, against 32-bit numbers | Whole run (kB) | Whole run, against the local arm |",
        "|---|---|---|---|---|---|---|",
    ]
    for model in MODELS:
        local = index.get((dataset, model, "local", 1.0))
        for arm in ARMS:
            row = index.get((dataset, model, arm, 1.0))
            if row is None:
                continue
            against = (
                "n/a"
                if local is None
                else f"{row['upload_bytes_per_run'] / local['upload_bytes_per_run']:.2f}"
            )
            lines.append(
                f"| {model} | {NAMES[arm]} | {row['rounds']} | {row['bits_per_value']:g} | "
                f"{row['upload_ratio_per_round']:.2f} | {row['upload_bytes_per_run'] / 1000:.1f} | {against} |"
            )
    return "\n".join(lines)


def cutoff_table(rows: list[dict[str, Any]], dataset: str) -> str:
    """Recall, precision and F1 at the cut-off chosen on validation rows."""
    index = _index(rows)
    lines = ["| Model | Arm | Budget | Recall | Precision | F1 |", "|---|---|---|---|---|---|"]
    for model in MODELS:
        for arm in (BASELINE, *ARMS):
            for budget in (None,) if arm == BASELINE else FEW_BUDGETS:
                row = index.get((dataset, model, arm, budget))
                if row is None:
                    continue
                label = "none" if budget is None else f"{budget:g}"
                lines.append(
                    f"| {model} | {NAMES[arm]} | {label} | {_cell(row['test_chosen_recall'])} | "
                    f"{_cell(row['test_chosen_precision'])} | {_cell(row['test_chosen_f1'])} |"
                )
    return "\n".join(lines)


def settings_table(rows: list[dict[str, Any]], dataset: str) -> str:
    """The settings every arm ran with (chosen on made-up tables) and its noise."""
    index = _index(rows)
    lines = [
        "| Model | Arm | Budget | Rounds | Batch | Local passes | Learning rate | Clipping | Noise multiplier |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for model in MODELS:
        for arm in ARMS:
            for budget in FEW_BUDGETS:
                row = index.get((dataset, model, arm, budget))
                if row is None:
                    continue
                batch = "all rows" if row["batch_size"] > 10**6 else f"{row['batch_size']}"
                low, high = row["noise_multiplier_min"], row["noise_multiplier_max"]
                noise = f"{low:.2f}" if high - low < 0.005 else f"{low:.2f} to {high:.2f}"
                lines.append(
                    f"| {model} | {NAMES[arm]} | {budget:g} | {row['rounds']} | {batch} | "
                    f"{row['local_epochs']:g} | {row['learning_rate']:g} | {row['clip_norm']:g} | {noise} |"
                )
    return "\n".join(lines)


def headline_markdown(rows: list[dict[str, Any]]) -> str:
    """The tables the results memo quotes: AUC of every arm, and the differences that decide.

    ``docs/PHASE3_RESULTS.md`` contains this text exactly; a test compares the two.
    """
    parts = []
    for dataset in DATASETS:
        seeds = len(SEEDS[dataset])
        parts += [
            f"**{dataset.capitalize()} table, AUC on held-out rows at their natural mix** (mean of {seeds} seeds ± sample standard deviation; columns are total budgets)",
            "",
            measure_table(rows, dataset, "test_auc"),
            "",
            f"**{dataset.capitalize()} table, difference in AUC against the better local arm** (mean of the seed-by-seed differences, 95% interval in brackets; marked only if the interval excludes zero)",
            "",
            difference_table(rows, dataset, "auc", "best_local"),
            "",
        ]
    return "\n".join(parts).rstrip("\n")


def markdown(rows: list[dict[str, Any]], problems: list[str], count: int, compared: int) -> str:
    lines = [
        "# Phase 3 results: every way of adding the noise at equal total budgets",
        "",
        f"Built by `python -m benchmarks.phase3.report` from {count} stored runs. Every record was validated, every stored privacy report was recomputed from its recorded releases, and {compared} runs of the local arm were compared with the stored Phase 2 runs.",
        "",
        "All measures are on held-out rows at their natural mix. Recall, precision and F1 are given only at the cut-off chosen on validation rows. Settings were chosen on made-up tables (`tune.py`), never on these rows.",
        "",
        headline_markdown(rows),
        "",
    ]
    for dataset in DATASETS:
        lines += [
            f"## {dataset.capitalize()} table: further tables",
            "",
            "### Average precision",
            "",
            measure_table(rows, dataset, "test_ap"),
            "",
            "### Difference in average precision against the better local arm",
            "",
            difference_table(rows, dataset, "ap", "best_local"),
            "",
            "### Difference in AUC against local noise with the Phase 2 protocol",
            "",
            difference_table(rows, dataset, "auc", "local"),
            "",
            "### Difference in AUC against local noise with tuned rounds",
            "",
            difference_table(rows, dataset, "auc", "local_tuned"),
            "",
            "### Does rounding to fewer bits cost accuracy? (difference in AUC against local noise in the clear)",
            "",
            small_numbers_table(rows, dataset),
            "",
            "### At the cut-off chosen on validation rows",
            "",
            cutoff_table(rows, dataset),
            "",
            "### Messages at a total budget of 1",
            "",
            message_table(rows, dataset),
            "",
            "### Settings and noise",
            "",
            settings_table(rows, dataset),
            "",
        ]
    if problems:
        lines += ["## Failed checks", ""] + [f"- {text}" for text in problems]
    return "\n".join(lines).rstrip("\n") + "\n"


def main() -> int:
    runs, problems = load_runs()
    compared, differing = check_against_phase2(runs)
    problems += differing
    rows = summarize(runs)
    summary = {
        "runs": len(runs),
        "failed_checks": problems,
        "compared_with_phase2": compared,
        "budgets": list(BUDGETS),
        "seeds": {name: list(seeds) for name, seeds in SEEDS.items()},
        "rows": rows,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    (RESULTS / "summary.md").write_text(
        markdown(rows, problems, len(runs), compared), encoding="utf-8"
    )
    print(headline_markdown(rows))
    print(f"{len(runs)} runs, {compared} compared with Phase 2, {len(problems)} failed checks")
    for text in problems[:20]:
        print("  -", text)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
