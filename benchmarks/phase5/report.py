# ruff: noqa: E501 - this script builds wide Markdown tables from long string literals
"""Check the benchmark's stored results and build its tables.

    python -m benchmarks.phase5.report

Reads what ``benchmarks.phase5.search``, ``main`` and ``audit`` stored under
``benchmarks/results/phase5`` and checks it against the written plan
(``benchmarks/PREREGISTRATION.md``): every planned run is there with the planned settings,
every protected run stayed within its budget and its stored privacy report recomputes,
every run was made by code that came after the plan, and the plan's text above its
amendments is unchanged. Writes ``summary.json`` and ``summary.md``. Exits with status 1
if any check fails.

How results are judged is fixed by the plan (section 8): a cell is the mean of its seeds
with the sample standard deviation; two arms are compared seed by seed, and a difference
counts only if its 95% interval (Student's t) does not contain zero.
"""

from __future__ import annotations

import functools
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

import obscuradp as odp
from benchmarks.phase5 import audit, plan
from benchmarks.phase5.common import BASELINE, BUDGETS, DATASETS, DELTA, PARTICIPANT_ARM, ROW_ARMS
from benchmarks.phase5.main import (
    CONFIRM_SEEDS,
    HANDWRITING_BUDGETS,
    SECURE_SUMS,
    SINGLE_RUN,
    TABLES,
    TEXT_BUDGET,
    confirm_setting,
    reported_settings,
)
from benchmarks.phase5.run import RESULTS
from obscuradp.audit.bounds import one_run_epsilon_lower_bound
from obscuradp.results import validate_result

SHARED_ARMS = ("shared", "shared_whole")
ARM_NAMES = {
    BASELINE: "no protection",
    "local": "local noise (default plan)",
    "shared": "shared noise, ordinary",
    "shared_whole": "shared noise, whole numbers",
    PARTICIPANT_ARM: "whole participants protected",
}
DATASET_NAMES = {
    "health": "Made-up health table",
    "stroke": "Stroke table",
    "loan": "Loan table",
    "femnist": "Handwriting",
    "shakespeare": "Text",
}
MODEL_NAMES = {
    "network": "network",
    "logistic": "logistic regression",
    "cnn": "image network",
    "gru": "text network",
}
EXPECTED_FROM = (1.0, 2.0, 4.0, 8.0)  # expectation 1: "at every budget from 1 upwards"
LOAN_TOLERANCE = 0.01  # expectation 2
NEAR_UNPROTECTED = 0.03  # expectation 3
Cell = tuple[str, str, str, float | None]


# ---------------------------------------------------------------------- the judging rule


def mean_and_spread(values: Sequence[float]) -> dict[str, float]:
    """Mean and sample standard deviation (dividing by one less than the number of values)."""
    numbers = np.asarray(values, dtype=np.float64)
    spread = float(numbers.std(ddof=1)) if numbers.size > 1 else 0.0
    return {"mean": float(numbers.mean()), "sd": spread, "n": int(numbers.size)}


def paired_difference(first: Mapping[int, float], second: Mapping[int, float]) -> dict[str, Any]:
    """``first`` minus ``second`` seed by seed, with a 95% interval (Student's t).

    ``counts`` says whether the interval leaves zero out, which is the plan's rule for
    calling two arms different.
    """
    seeds = sorted(set(first) & set(second))
    if len(seeds) < 2:
        raise ValueError("a comparison needs at least two seeds that both arms ran")
    differences = np.asarray([first[seed] - second[seed] for seed in seeds], dtype=np.float64)
    mean = float(differences.mean())
    spread = float(differences.std(ddof=1))
    half = float(stats.t.ppf(0.975, len(seeds) - 1)) * spread / math.sqrt(len(seeds))
    low, high = mean - half, mean + half
    return {
        "mean": mean,
        "low": low,
        "high": high,
        "n": len(seeds),
        "counts": bool(low > 0 or high < 0),
    }


# ---------------------------------------------------------------------- reading the records


def _read(path: Path, problems: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        problems.append(f"{path.name}: missing")
        return None
    record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    try:
        validate_result(record)
    except ValueError as error:
        problems.append(f"{path.name}: {error}")
        return None
    return record


def _check_protection(
    name: str, record: dict[str, Any], budget: float | None, problems: list[str]
) -> None:
    stored = record["data"]["privacy_report"]
    if budget is None:
        if stored is not None:
            problems.append(f"{name}: an unprotected run with a privacy report")
        return
    if stored is None:
        problems.append(f"{name}: no privacy report beside the scores")
        return
    if not stored["within_budget"] or stored["epsilon"] > budget * (1 + 1e-9):
        problems.append(f"{name}: over its budget")
    found = _recompute(json.dumps(stored, sort_keys=True))
    if found:
        problems.append(f"{name}: the stored privacy report does not recompute ({found[0]})")


@functools.cache
def _recompute(stored: str) -> tuple[str, ...]:
    """What ``verify_report`` finds wrong. Runs that store the same report share the work."""
    return tuple(odp.verify_report(json.loads(stored)))


def read_runs(
    directory: Path, chosen: dict[str, dict[str, Any]], problems: list[str], commits: set[str]
) -> dict[Cell, dict[int, dict[str, Any]]]:
    """Every reported run, checked, by cell and seed."""
    cells: dict[Cell, dict[int, dict[str, Any]]] = {}
    for setting in reported_settings(chosen):
        record = _read(directory / f"{setting.label}.json", problems)
        if record is None:
            continue
        if record["config"]["benchmark"] != asdict(setting):
            problems.append(f"{setting.label}: not run with the planned settings")
            continue
        if record["data"]["test"] is None:
            problems.append(f"{setting.label}: no scores on test rows")
            continue
        _check_protection(setting.label, record, setting.epsilon, problems)
        commits.add(str(record["manifest"]["git_commit"]))
        cell = (setting.dataset, setting.model, setting.arm, setting.epsilon)
        cells.setdefault(cell, {})[setting.seed] = record["data"]
    return cells


def check_commits(commits: set[str], problems: list[str]) -> dict[str, Any]:
    """Every run must have been made by committed code that came after the plan."""
    verdicts = {commit: plan.comes_after_plan(commit) for commit in sorted(commits)}
    for commit, verdict in verdicts.items():
        if verdict is False:
            problems.append(f"runs were made at {commit}, which does not come after the plan")
    if not plan.plan_is_unchanged():
        problems.append("the plan's text above its amendments has changed")
    judged = [verdict for verdict in verdicts.values() if verdict is not None]
    return {
        "plan_commit": plan.PLAN_COMMIT,
        "plan_unchanged": plan.plan_is_unchanged(),
        "run_commits": sorted(commits),
        # None: this copy of the repository has no history to check against.
        "all_after_plan": (all(judged) if len(judged) == len(verdicts) else None)
        if verdicts
        else None,
    }


# ---------------------------------------------------------------------- cells and comparisons


def headline(dataset: str, data: Mapping[str, Any]) -> float:
    return float(data["test"][DATASETS[dataset].measure])


def cell_rows(cells: Mapping[Cell, Mapping[int, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """One row per cell: the headline measure, further measures, the budget spent, messages, time."""
    rows = []
    for (dataset, model, arm, epsilon), runs in cells.items():
        seeds = sorted(runs)
        row: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "arm": arm,
            "epsilon": epsilon,
            "seeds": seeds,
            "measure": DATASETS[dataset].measure,
            "headline": mean_and_spread([headline(dataset, runs[seed]) for seed in seeds]),
            "seconds": float(np.mean([runs[seed]["seconds"] for seed in seeds])),
        }
        first = runs[seeds[0]]
        communication = first["communication"]
        rounds = int(first["history"][-1]["round"]) if first["history"] else None
        row["upload_bytes_per_round"] = communication["upload_bytes_per_round"]
        row["plain_bytes_per_round"] = communication["plain_bytes_per_round"]
        row["rounds"] = rounds
        if DATASETS[dataset].unit == "record":
            row["ap"] = mean_and_spread([runs[seed]["test"]["ap"] for seed in seeds])
            for name in ("recall", "precision", "f1"):
                row[name] = mean_and_spread(
                    [runs[seed]["test_at_chosen_threshold"][name] for seed in seeds]
                )
        if epsilon is not None:
            reports = [runs[seed]["privacy_report"] for seed in seeds]
            row["epsilon_spent"] = float(np.mean([report["epsilon"] for report in reports]))
            row["ceiling_mechanism"] = float(
                np.mean([report["attacker_advantage_ceiling"] for report in reports])
            )
            row["ceiling_general"] = float(
                np.mean([report["general_attacker_advantage_ceiling"] for report in reports])
            )
            row["simulated_secure_sum"] = any("simulated" in text for text in reports[0]["caveats"])
        rows.append(row)
    return rows


def _by_seed(cells: Mapping[Cell, Mapping[int, Mapping[str, Any]]], cell: Cell) -> dict[int, float]:
    return {seed: headline(cell[0], data) for seed, data in cells.get(cell, {}).items()}


def comparisons(cells: Mapping[Cell, Mapping[int, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Every pair of arms of a table at the same budget, seed by seed; and each arm against no protection."""
    rows = []
    arms = tuple(ROW_ARMS)
    for dataset in TABLES:
        for model in DATASETS[dataset].models:
            plain = _by_seed(cells, (dataset, model, BASELINE, None))
            for epsilon in BUDGETS:
                scores = {arm: _by_seed(cells, (dataset, model, arm, epsilon)) for arm in arms}
                pairs = [
                    (arms[j], arms[i]) for i in range(len(arms)) for j in range(i + 1, len(arms))
                ]
                pairs += [(BASELINE, arm) for arm in arms]
                for first, second in pairs:
                    left = plain if first == BASELINE else scores[first]
                    if len(set(left) & set(scores[second])) < 2:
                        continue
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "epsilon": epsilon,
                            "first": first,
                            "second": second,
                            **paired_difference(left, scores[second]),
                        }
                    )
    return rows


def _find(rows: Sequence[Mapping[str, Any]], **wanted: Any) -> Mapping[str, Any] | None:
    for row in rows:
        if all(row[key] == value for key, value in wanted.items()):
            return row
    return None


def expectations(
    rows: Sequence[Mapping[str, Any]], compared: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Expectations 1 to 3 of the plan, read as its amendment of 2026-11-12 fixes."""
    first = {"holds": 0, "of": 0, "against": 0}
    for dataset in ("health", "stroke"):
        for model in DATASETS[dataset].models:
            for arm in SHARED_ARMS:
                for epsilon in EXPECTED_FROM:
                    found = _find(
                        compared,
                        dataset=dataset,
                        model=model,
                        epsilon=epsilon,
                        first=arm,
                        second="local",
                    )
                    if found is None:
                        continue
                    first["of"] += 1
                    first["holds"] += int(found["counts"] and found["mean"] > 0)
                    first["against"] += int(found["counts"] and found["mean"] < 0)
    second = {"holds": 0, "of": 0, "largest_gap": 0.0}
    for model in DATASETS["loan"].models:
        for epsilon in BUDGETS:
            means = [
                row["headline"]["mean"]
                for arm in ROW_ARMS
                if (row := _find(rows, dataset="loan", model=model, arm=arm, epsilon=epsilon))
                is not None
            ]
            for i in range(len(means)):
                for j in range(i + 1, len(means)):
                    gap = abs(means[i] - means[j])
                    second["of"] += 1
                    second["holds"] += int(gap <= LOAN_TOLERANCE)
                    second["largest_gap"] = max(second["largest_gap"], gap)
    third = {"holds": 0, "of": 0, "largest_gap": 0.0}
    for dataset in TABLES:
        for model in DATASETS[dataset].models:
            plain = _find(rows, dataset=dataset, model=model, arm=BASELINE, epsilon=None)
            for arm in SHARED_ARMS:
                shared = _find(rows, dataset=dataset, model=model, arm=arm, epsilon=8.0)
                if plain is None or shared is None:
                    continue
                gap = plain["headline"]["mean"] - shared["headline"]["mean"]
                third["of"] += 1
                third["holds"] += int(gap <= NEAR_UNPROTECTED)
                third["largest_gap"] = max(third["largest_gap"], gap)
    planned = {"1": 32, "2": 30, "3": 12}
    result = {"1": first, "2": second, "3": third}
    for key, value in result.items():
        value["planned"] = planned[key]
        value["confirmed"] = bool(value["of"] == planned[key] and value["holds"] == value["of"])
    return result


# ---------------------------------------------------------------------- the small real case


def confirm_rows(
    directory: Path, chosen: dict[str, dict[str, Any]], problems: list[str], commits: set[str]
) -> dict[str, Any]:
    """Real against simulated secure sum on the first writers, seed by seed."""
    scores: dict[str, dict[int, float]] = {name: {} for name in SECURE_SUMS}
    for seed in CONFIRM_SEEDS:
        setting = confirm_setting(seed, chosen)
        for name in SECURE_SUMS:
            record = _read(directory / f"{setting.label}_{name}.json", problems)
            if record is None:
                continue
            simulated = record["config"].get("crypto_in_loop") is False
            if record["config"]["benchmark"] != asdict(setting) or simulated != (
                name == "simulated"
            ):
                problems.append(f"{setting.label}_{name}: not the planned run")
                continue
            _check_protection(f"{setting.label}_{name}", record, setting.epsilon, problems)
            commits.add(str(record["manifest"]["git_commit"]))
            scores[name][seed] = float(record["data"]["test"]["accuracy"])
    if any(len(values) < len(CONFIRM_SEEDS) for values in scores.values()):
        return {}
    difference = paired_difference(scores["real"], scores["simulated"])
    return {
        "real": mean_and_spread(list(scores["real"].values())),
        "simulated": mean_and_spread(list(scores["simulated"].values())),
        "difference": difference,
        "agree": not difference["counts"],
    }


# ---------------------------------------------------------------------- the attack tests


def _resampled(values: Sequence[float], seed: int = 0) -> dict[str, float]:
    """The mean of ``values`` over runs and a 95% interval from resampling whole runs."""
    numbers = np.asarray(values, dtype=np.float64)
    picks = np.random.default_rng(seed).integers(0, numbers.size, size=(2000, numbers.size))
    low, high = np.quantile(numbers[picks].mean(axis=1), [0.025, 0.975])
    return {
        "mean": float(numbers.mean()),
        "low": float(low),
        "high": float(high),
        "n": int(numbers.size),
    }


def health_audit_rows(
    directory: Path, problems: list[str], commits: set[str]
) -> list[dict[str, Any]]:
    """The audited runs on the health table, pooled over seeds, by class."""
    rows = []
    for model in audit.HEALTH_MODELS:
        records = []
        for seed in audit.HEALTH_SEEDS:
            record = _read(directory / f"health_{model}_seed-{seed}.json", problems)
            if record is not None:
                commits.add(str(record["manifest"]["git_commit"]))
                records.append(record["data"])
        if len(records) != len(audit.HEALTH_SEEDS):
            continue
        for epsilon in audit.HEALTH_BUDGETS:
            runs = [record[f"{epsilon:g}"] for record in records]
            if any(run["epsilon"] > epsilon * (1 + 1e-9) for run in runs):
                problems.append(
                    f"audit, health, {model}, budget {epsilon:g}: a run is over its budget"
                )
            for attack in audit.ATTACKS:
                row: dict[str, Any] = {
                    "model": model,
                    "epsilon": epsilon,
                    "attack": attack,
                    "runs": len(runs),
                    "ceiling_mechanism": float(np.mean([run["ceiling_mechanism"] for run in runs])),
                    "ceiling_general": float(np.mean([run["ceiling_general"] for run in runs])),
                }
                for group in audit.GROUPS:
                    figures = [run["attacks"][attack][group] for run in runs]
                    records_seen = sum(int(f["records"]) for f in figures)
                    guesses = sum(int(f["guesses"]) for f in figures)
                    right = sum(int(f["right"]) for f in figures)
                    row[group] = {
                        "advantage": _resampled([float(f["advantage"]) for f in figures]),
                        "records": records_seen,
                        "guesses": guesses,
                        "right": right,
                        "epsilon_lower_limit": one_run_epsilon_lower_bound(
                            records_seen, guesses, right, DELTA
                        ),
                    }
                rows.append(row)
    return rows


def writer_audit_row(
    directory: Path, problems: list[str], commits: set[str]
) -> dict[str, Any] | None:
    """The one audited handwriting run, or ``None`` if it was not done."""
    path = directory / f"femnist_writers_seed-{audit.WRITER_SEED}.json"
    if not path.is_file():
        return None
    record = _read(path, problems)
    if record is None:
        return None
    commits.add(str(record["manifest"]["git_commit"]))
    data: dict[str, Any] = record["data"]
    if data["epsilon"] > audit.WRITER_BUDGET * (1 + 1e-9):
        problems.append("audit, whole writers: the run is over its budget")
    return data


# ---------------------------------------------------------------------- tables in Markdown


def _pm(stat: Mapping[str, float], digits: int = 3) -> str:
    return f"{stat['mean']:.{digits}f} ± {stat['sd']:.{digits}f}"


def _signed(value: float) -> str:
    return f"{value:+.3f}"


def _budget(epsilon: float | None) -> str:
    return "none" if epsilon is None else f"{epsilon:g}"


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def quality_table(rows: Sequence[Mapping[str, Any]], dataset: str, model: str) -> str:
    plain = _find(rows, dataset=dataset, model=model, arm=BASELINE, epsilon=None)
    lines = [
        "| Arm | " + " | ".join(f"budget {budget:g}" for budget in BUDGETS) + " |",
        "|---|" + "---|" * len(BUDGETS),
    ]
    for arm in ROW_ARMS:
        cells = [
            _pm(row["headline"])
            if (row := _find(rows, dataset=dataset, model=model, arm=arm, epsilon=budget))
            else "missing"
            for budget in BUDGETS
        ]
        lines.append(f"| {ARM_NAMES[arm]} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"No protection: {_pm(plain['headline']) if plain else 'missing'}.")
    return "\n".join(lines)


def difference_table(compared: Sequence[Mapping[str, Any]], dataset: str, model: str) -> str:
    lines = [
        "| Compared | " + " | ".join(f"budget {budget:g}" for budget in BUDGETS) + " |",
        "|---|" + "---|" * len(BUDGETS),
    ]
    pairs = [("shared", "local"), ("shared_whole", "local"), ("shared_whole", "shared")]
    for first, second in pairs:
        cells = []
        for budget in BUDGETS:
            found = _find(
                compared, dataset=dataset, model=model, epsilon=budget, first=first, second=second
            )
            if found is None:
                cells.append("missing")
                continue
            mark = "**counts**" if found["counts"] else "does not count"
            cells.append(
                f"{_signed(found['mean'])} [{_signed(found['low'])}, {_signed(found['high'])}], {mark}"
            )
        lines.append(
            f"| {ARM_NAMES[first]} minus {ARM_NAMES[second]} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines)


def crowd_table(
    rows: Sequence[Mapping[str, Any]], dataset: str, model: str, budgets: Sequence[float]
) -> str:
    chance = DATASETS[dataset].chance
    lines = [
        "| Protection | Share right on test rows | Seeds | Proven limit on the attacker's advantage for the noise used |",
        "|---|---|---|---|",
    ]
    plain = _find(rows, dataset=dataset, model=model, arm=BASELINE, epsilon=None)
    if plain:
        lines.append(f"| none | {_pm(plain['headline'])} | {plain['headline']['n']} | no limit |")
    for budget in sorted(budgets, reverse=True):
        row = _find(rows, dataset=dataset, model=model, arm=PARTICIPANT_ARM, epsilon=budget)
        if row is None:
            continue
        lines.append(
            f"| whole participants, budget {budget:g} | {_pm(row['headline'])} | {row['headline']['n']} | {_percent(row['ceiling_mechanism'])} |"
        )
    lines.append("")
    lines.append(f"A model that learned nothing: about {chance:g}.")
    return "\n".join(lines)


def message_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Dataset | Model | Arm | Budget | Bytes one participant uploads per round | Against plain 32-bit numbers |",
        "|---|---|---|---|---|---|",
    ]
    for dataset, data in DATASETS.items():
        budget = 8.0 if data.unit == "client" else 1.0
        arms = (PARTICIPANT_ARM,) if data.unit == "client" else tuple(ROW_ARMS)
        for model in data.models:
            for arm in arms:
                row = _find(rows, dataset=dataset, model=model, arm=arm, epsilon=budget)
                if row is None:
                    continue
                ratio = row["upload_bytes_per_round"] / row["plain_bytes_per_round"]
                lines.append(
                    f"| {DATASET_NAMES[dataset]} | {MODEL_NAMES[model]} | {ARM_NAMES[arm]} | {budget:g} | {row['upload_bytes_per_round']:,.0f} | {ratio:.2f} times |"
                )
    return "\n".join(lines)


def expectation_table(found: Mapping[str, Mapping[str, Any]]) -> str:
    def verdict(key: str) -> str:
        return "**confirmed**" if found[key]["confirmed"] else "**not confirmed**"

    one, two, three = found["1"], found["2"], found["3"]
    return "\n".join(
        [
            "| Expectation | Verdict | What was found |",
            "|---|---|---|",
            f"| 1. Health and stroke: shared noise beats local noise at every budget from 1 upwards | {verdict('1')} | holds in {one['holds']} of {one['of']} comparisons; in {one['against']} local noise is ahead by the same rule |",
            f"| 2. Loan: no two protected arms differ by more than {LOAN_TOLERANCE:g} AUC | {verdict('2')} | holds in {two['holds']} of {two['of']} comparisons; largest gap {two['largest_gap']:.3f} |",
            f"| 3. Budget 8: shared noise within {NEAR_UNPROTECTED:g} AUC of no protection | {verdict('3')} | holds in {three['holds']} of {three['of']} comparisons; largest gap {three['largest_gap']:.3f} |",
        ]
    )


def health_audit_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Model | Budget | Attack | Proven limit for the noise used | Rows of the rare class | Rows of the common class | Lower limit on the budget, rare class |",
        "|---|---|---|---|---|---|---|",
    ]

    def interval(stat: Mapping[str, float]) -> str:
        return f"{_percent(stat['mean'])} [{_percent(stat['low'])}, {_percent(stat['high'])}]"

    for row in rows:
        lines.append(
            f"| {MODEL_NAMES[row['model']]} | {row['epsilon']:g} | {row['attack']} | {_percent(row['ceiling_mechanism'])} | {interval(row['label_1']['advantage'])} | {interval(row['label_0']['advantage'])} | {row['label_1']['epsilon_lower_limit']:.2f} |"
        )
    return "\n".join(lines)


def writer_audit_text(found: Mapping[str, Any] | None) -> str:
    if found is None:
        return "Not done."
    return "\n".join(
        [
            "| Figure | Value |",
            "|---|---|",
            f"| Test writers (kept in the run) | {found['test_units']} ({found['kept']}) |",
            f"| Guesses made, of them right | {found['guesses']}, {found['right']} |",
            f"| Measured advantage of the attack | {_percent(found['advantage'])} |",
            f"| Proven limit for the noise used | {_percent(found['ceiling_mechanism'])} |",
            f"| Proven limit for any method with this budget | {_percent(found['ceiling_general'])} |",
            f"| Budget spent | {found['epsilon']:.2f} |",
            f"| Lower limit on the budget shown by the test | {found['epsilon_lower_limit']:.2f} |",
            f"| The most this test could have shown | {found['most_the_test_could_show']:.2f} |",
        ]
    )


def headline_markdown(summary: Mapping[str, Any]) -> str:
    """The tables of the results memo. A test checks that the memo quotes them exactly."""
    rows, compared = summary["cells"], summary["comparisons"]
    parts: list[str] = [
        "**What was expected, and what was found**",
        "",
        expectation_table(summary["expectations"]),
        "",
    ]
    for dataset in TABLES:
        for model in DATASETS[dataset].models:
            title = f"{DATASET_NAMES[dataset]}, {MODEL_NAMES[model]}"
            parts += [
                f"**{title}: AUC on test rows** (mean of five seeds ± sample standard deviation)",
                "",
                quality_table(rows, dataset, model),
                "",
                f"**{title}: differences seed by seed** (mean difference in AUC with its 95% interval)",
                "",
                difference_table(compared, dataset, model),
                "",
            ]
    parts += [
        "**Handwriting: 3,400 writers, whole writers protected**",
        "",
        crowd_table(rows, "femnist", "cnn", (*HANDWRITING_BUDGETS, SINGLE_RUN[0])),
        "",
        "**Text: 644 speaking roles, whole roles protected**",
        "",
        crowd_table(rows, "shakespeare", "gru", (TEXT_BUDGET,)),
        "",
        "**What one participant uploads per round**",
        "",
        message_table(rows),
        "",
    ]
    confirm = summary["confirm"]
    if confirm:
        difference = confirm["difference"]
        parts += [
            "**The small real case: real against simulated secure sum** (first 60 writers, 20 rounds, budget 8, three seeds)",
            "",
            "| Secure sum | Share right on test rows |",
            "|---|---|",
            f"| real | {_pm(confirm['real'])} |",
            f"| simulated | {_pm(confirm['simulated'])} |",
            "",
            f"Difference seed by seed: {_signed(difference['mean'])} [{_signed(difference['low'])}, {_signed(difference['high'])}]. The two {'agree' if confirm['agree'] else '**do not agree**'} by the plan's rule.",
            "",
        ]
    parts += [
        f"**Attack tests on the health table, by class** (default plan; mean advantage over {len(audit.HEALTH_SEEDS)} runs with a 95% interval from resampling runs)",
        "",
        health_audit_table(summary["health_audit"]),
        "",
        "**Attack test on whole writers** (handwriting, budget 8, one run)",
        "",
        writer_audit_text(summary["writer_audit"]),
    ]
    return "\n".join(parts).rstrip("\n")


# ---------------------------------------------------------------------- putting it together


def build_summary(directory: Path | None = None) -> dict[str, Any]:
    """The summary of the stored results, with every failed check listed."""
    directory = directory or RESULTS
    problems: list[str] = []
    commits: set[str] = set()
    settings_path = directory / "settings.json"
    if not settings_path.is_file():
        return {"failed_checks": ["settings.json: missing (the search has not been run)"]}
    search = json.loads(settings_path.read_text(encoding="utf-8"))
    chosen = search["chosen"]
    commits.update(search.get("commits", []))
    cells = read_runs(directory / "runs", chosen, problems, commits)
    rows = cell_rows(cells)
    compared = comparisons(cells)
    planned = reported_settings(chosen)
    confirm = confirm_rows(directory / "confirm", chosen, problems, commits)
    health = health_audit_rows(directory / "audit", problems, commits)
    writers = writer_audit_row(directory / "audit", problems, commits)
    return {
        "failed_checks": problems,
        "commits": check_commits(commits, problems),
        "runs_planned": len(planned),
        "runs_found": sum(len(runs) for runs in cells.values()),
        "search_runs": search["runs"],
        "computing_hours": round(
            (
                sum(data["seconds"] for runs in cells.values() for data in runs.values())
                + search["computing_seconds"]
            )
            / 3600,
            1,
        ),
        "chosen": chosen,
        "cells": rows,
        "comparisons": compared,
        "expectations": expectations(rows, compared),
        "confirm": confirm,
        "health_audit": health,
        "writer_audit": writers,
    }


def main() -> int:
    summary = build_summary()
    failed = summary["failed_checks"]
    text = "# Phase 5, the benchmark: results\n\n"
    if not failed:
        text += (
            f"Built by `python -m benchmarks.phase5.report` from {summary['runs_found']} reported runs, "
            f"{summary['search_runs']} runs of the settings search and the attack tests "
            f"(about {summary['computing_hours']:g} hours of computing for the search and the reported runs).\n\n"
            + headline_markdown(summary)
            + "\n"
        )
    else:
        text += "## Failed checks\n\n" + "\n".join(f"- {item}" for item in failed) + "\n"
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    (RESULTS / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"{len(failed)} failed checks")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
