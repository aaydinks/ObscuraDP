# ruff: noqa: E501 - this script builds wide Markdown tables from long string literals
"""Check every stored run and build the Phase 2 tables, old numbers beside new ones.

    python -m benchmarks.phase2.report

Reads ``benchmarks/results/phase2/runs/*.json``. Every record is validated, and every stored
privacy report is recomputed from its recorded releases (the same check as ``obscuradp
report verify``). Writes ``summary.json`` and ``summary.md`` next to the runs. Exits with
status 1 if any record or report fails its check.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks.phase2.common import BUDGETS, MODELS, mean_and_interval
from obscuradp.report import verify_report
from obscuradp.results import validate_result

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "benchmarks" / "results" / "phase2"
OLD = ROOT / "Trial 07" / "27" / "results"

NATURAL = ("auc", "ap", "recall", "precision", "f1", "accuracy")
BALANCED = ("recall", "accuracy", "f1", "auc")


def load_runs() -> tuple[list[dict[str, Any]], list[str]]:
    """Every stored run, plus a list of whatever failed its checks."""
    runs, problems = [], []
    for path in sorted((RESULTS / "runs").glob("*.json")):
        record = json.loads(path.read_text())
        try:
            validate_result(record)
        except ValueError as error:
            problems.append(f"{path.name}: {error}")
            continue
        report = record["data"]["privacy_report"]
        if report is not None:
            problems += [f"{path.name}: {text}" for text in verify_report(report)]
        runs.append(record)
    return runs, problems


def summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per dataset, model, budget and transport, over the seeds."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in runs:
        setting = record["config"]["benchmark"]
        key = (setting["dataset"], setting["model"], setting["epsilon"], setting["secure_sum"])
        groups[key].append(record)
    rows = []
    for (dataset, model, epsilon, secure_sum), members in groups.items():
        data = [record["data"] for record in members]
        setting = members[0]["config"]["benchmark"]
        row: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "epsilon": epsilon,
            "secure_sum": secure_sum,
            "seeds": sorted(record["config"]["benchmark"]["seed"] for record in members),
            "learning_rate": setting["learning_rate"],
            "clip_norm": setting["clip_norm"],
            "test_rows": [d["test_rows"] for d in data],
            "test_positive_rows": [d["test_positive_rows"] for d in data],
            "balanced_rows": [d["balanced_rows"] for d in data],
            "upload_ratio": data[0]["communication"]["ratio"],
            "saturated_share_max": max(d["communication"]["saturated_share"] for d in data),
            "seconds_mean": sum(d["seconds"] for d in data) / len(data),
        }
        for name in NATURAL:
            row[f"test_{name}"] = mean_and_interval([d["test"][name] for d in data])
        for name in ("recall", "precision", "f1"):
            row[f"test_chosen_{name}"] = mean_and_interval(
                [d["test_at_chosen_threshold"][name] for d in data]
            )
        for name in BALANCED:
            row[f"balanced_{name}"] = mean_and_interval([d["test_balanced"][name] for d in data])
        for name in ("recall", "accuracy"):
            row[f"balanced_chosen_{name}"] = mean_and_interval(
                [d["test_balanced_at_chosen_threshold"][name] for d in data]
            )
        row["pred_pos_rate_at_half"] = mean_and_interval([d["test"]["pred_pos_rate"] for d in data])
        if epsilon is not None:
            spent = [d["privacy_report"]["epsilon"] for d in data]
            sigmas = [value for d in data for value in d["noise_multipliers"].values()]
            ceilings = [d["privacy_report"]["attacker_advantage_ceiling"] for d in data]
            row["epsilon_spent_max"] = max(spent)
            row["noise_multiplier_min"] = min(sigmas)
            row["noise_multiplier_max"] = max(sigmas)
            row["attacker_advantage_ceiling_max"] = max(ceilings)
            row["all_within_budget"] = all(d["privacy_report"]["within_budget"] for d in data)
            row["unsafe_flags"] = sorted(
                {f for d in data for f in d["privacy_report"]["unsafe_flags"]}
            )
        rows.append(row)
    rows.sort(
        key=lambda r: (
            r["dataset"],
            r["model"],
            r["secure_sum"],
            -1.0 if r["epsilon"] is None else r["epsilon"],
        )
    )
    return rows


def old_results() -> dict[str, Any]:
    """The committed research numbers the new ones are set beside."""
    multiseed = json.loads((OLD / "multiseed.json").read_text())
    frontier = []
    for entry in multiseed["aggregated"]:
        if entry["spe_fraction"] != 0.0:
            continue
        members = [
            run
            for run in multiseed["runs"]
            if run["epsilon"] == entry["epsilon"] and run["spe_fraction"] == 0.0
        ]
        whole_run = [max(run["dp"]["per_client"]["eps_cumulative"]) for run in members]
        frontier.append(
            {
                "label_epsilon_per_round": entry["epsilon"],
                "whole_run_epsilon_by_old_code_max": max(whole_run),
                "noise_multiplier_min": min(min(r["dp"]["per_client"]["sigma"]) for r in members),
                "noise_multiplier_max": max(max(r["dp"]["per_client"]["sigma"]) for r in members),
                # The research tables used the population standard deviation; recomputed here
                # with the sample standard deviation so old and new spreads are comparable.
                **{
                    name: mean_and_interval([run["metrics"][name] for run in members])
                    for name in ("recall", "accuracy", "f1", "auc")
                },
            }
        )
    arms = json.loads((OLD / "arms.json").read_text())["arms"]
    single = [
        {
            "dataset": arm["dataset"],
            "stack": arm["stack"],
            "algo": arm["algo"],
            **{name: arm["metrics"][name] for name in ("recall", "accuracy", "f1", "auc")},
        }
        for arm in arms
        if arm["stack"] in ("plain", "dp")
    ]
    return {"stroke_frontier_fedprox_dp": frontier, "arms_seed_42": single}


def _cell(stat: dict[str, float]) -> str:
    if stat["n"] == 0:
        return "n/a"
    if stat["n"] == 1:
        return f"{stat['mean']:.3f}"
    return f"{stat['mean']:.3f} ± {stat['std']:.3f}"


def _budget(epsilon: float | None) -> str:
    return "none" if epsilon is None else f"{epsilon:g}"


def markdown(
    rows: list[dict[str, Any]], old: dict[str, Any], problems: list[str], count: int
) -> str:
    lines = [
        "# Phase 2 results: corrected runs beside the earlier ones",
        "",
        "Generated by `benchmarks/phase2/report.py` from the records in `runs/`. Do not edit by hand.",
        "",
        f"Runs: {count}. Records or privacy reports that failed their check: {len(problems)}.",
        "",
        "Every entry is the mean over seeds, plus or minus the sample standard deviation.",
        "A budget is the total $\\varepsilon$ for the whole training run at $\\delta = 10^{-5}$, per record.",
        "",
    ]
    for dataset in ("stroke", "loan"):
        for model in MODELS:
            subset = [
                r
                for r in rows
                if r["dataset"] == dataset and r["model"] == model and not r["secure_sum"]
            ]
            if not subset:
                continue
            first = subset[0]
            lines += [
                f"## {dataset}, {model}: updates in the clear",
                "",
                f"Natural test rows per seed: {first['test_rows']} with {first['test_positive_rows']} positive. "
                f"Balanced comparison rows per seed: {first['balanced_rows']}.",
                "",
                "| Budget | Noise multiplier | AUC | Average precision | Recall at 0.5 | Share predicted positive at 0.5 "
                "| F1 at chosen threshold | Balanced-set recall at 0.5 | Balanced-set accuracy at 0.5 "
                "| Balanced-set accuracy at chosen threshold |",
                "|---|---|---|---|---|---|---|---|---|---|",
            ]
            for row in subset:
                sigma = (
                    "none"
                    if row["epsilon"] is None
                    else f"{row['noise_multiplier_min']:.2f} to {row['noise_multiplier_max']:.2f}"
                )
                lines.append(
                    f"| {_budget(row['epsilon'])} | {sigma} | {_cell(row['test_auc'])} | {_cell(row['test_ap'])} "
                    f"| {_cell(row['test_recall'])} | {_cell(row['pred_pos_rate_at_half'])} "
                    f"| {_cell(row['test_chosen_f1'])} "
                    f"| {_cell(row['balanced_recall'])} | {_cell(row['balanced_accuracy'])} "
                    f"| {_cell(row['balanced_chosen_accuracy'])} |"
                )
            lines.append("")
    lines += [
        "## Secure sum against updates in the clear",
        "",
        "| Dataset | Model | Budget | AUC in the clear | AUC through the secure sum | Upload size against 32-bit numbers |",
        "|---|---|---|---|---|---|",
    ]
    index = {(r["dataset"], r["model"], r["epsilon"], r["secure_sum"]): r for r in rows}
    for row in rows:
        if not row["secure_sum"]:
            continue
        clear = index.get((row["dataset"], row["model"], row["epsilon"], False))
        if clear is None:
            continue
        lines.append(
            f"| {row['dataset']} | {row['model']} | {_budget(row['epsilon'])} | {_cell(clear['test_auc'])} "
            f"| {_cell(row['test_auc'])} | {row['upload_ratio']:.3f} |"
        )
    lines += [
        "",
        "## Earlier stroke results (Trial 07/27), for comparison",
        "",
        "These used synthetic rows made from real ones before private training, a noise level sized for one",
        "round, and a balanced 84-row test set. The label is the budget of one round; the next column is the",
        "whole-run figure the research code itself computed. Neither describes real records, because of the",
        "synthetic rows.",
        "",
        "| Label (per round) | Whole run, by the old code | Noise multiplier | Balanced-set recall | Balanced-set accuracy | AUC |",
        "|---|---|---|---|---|---|",
    ]
    for entry in old["stroke_frontier_fedprox_dp"]:
        lines.append(
            f"| {entry['label_epsilon_per_round']:g} | {entry['whole_run_epsilon_by_old_code_max']:.1f} "
            f"| {entry['noise_multiplier_min']:.2f} to {entry['noise_multiplier_max']:.2f} "
            f"| {_cell(entry['recall'])} | {_cell(entry['accuracy'])} | {_cell(entry['auc'])} |"
        )
    lines += [
        "",
        "Earlier single-seed arms (seed 42, label 9 per round):",
        "",
        "| Dataset | Arm | Algorithm | Recall | Accuracy | AUC |",
        "|---|---|---|---|---|---|",
    ]
    for arm in old["arms_seed_42"]:
        lines.append(
            f"| {arm['dataset']} | {arm['stack']} | {arm['algo']} | {arm['recall']:.3f} "
            f"| {arm['accuracy']:.3f} | {arm['auc']:.3f} |"
        )
    if problems:
        lines += ["", "## Failed checks", ""] + [f"- {text}" for text in problems]
    return "\n".join(lines) + "\n"


def headline_markdown(rows: list[dict[str, Any]]) -> str:
    """The short table the gate memo quotes: AUC on natural test rows at every budget.

    ``docs/GATE_PHASE2.md`` contains this text exactly; a test compares the two.
    """
    budgets = (None, *BUDGETS)
    lines = [
        "| Table, model | " + " | ".join(_budget(b) for b in budgets) + " |",
        "|---|" + "---|" * len(budgets),
    ]
    index = {(r["dataset"], r["model"], r["epsilon"]): r for r in rows if not r["secure_sum"]}
    for dataset in ("loan", "stroke"):
        for model in MODELS:
            cells = [_cell(index[(dataset, model, budget)]["test_auc"]) for budget in budgets]
            lines.append(f"| {dataset}, {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    runs, problems = load_runs()
    rows = summarize(runs)
    old = old_results()
    summary = {
        "runs": len(runs),
        "failed_checks": problems,
        "budgets": list(BUDGETS),
        "rows": rows,
        "earlier_results": old,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    (RESULTS / "summary.md").write_text(markdown(rows, old, problems, len(runs)), encoding="utf-8")
    print(headline_markdown(rows))
    print(f"{len(runs)} runs, {len(problems)} failed checks")
    for text in problems[:20]:
        print("  -", text)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
