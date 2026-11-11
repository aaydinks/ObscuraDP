"""The Phase 3 comparison scripts: every arm on made-up rows, and the stored results.

The scripts live in ``benchmarks/`` and are not part of the installed package, so these
tests only run from a full checkout of the repository.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

import obscuradp as odp
from obscuradp.results import validate_result

ROOT = Path(__file__).resolve().parents[1]

if not (ROOT / "benchmarks" / "phase3" / "common.py").is_file():
    pytest.skip("the benchmark scripts are not part of this installation", allow_module_level=True)

sys.path.insert(0, str(ROOT))

from benchmarks.phase2 import standin  # noqa: E402
from benchmarks.phase3 import common, report, run, tune  # noqa: E402

RUNS_PER_TABLE_AND_SEED = 2 * (1 + 5 * 5 + 3 * 2)


@pytest.fixture(scope="module")
def tiny_records() -> list[dict[str, Any]]:
    """Two rounds of every arm on a made-up loan-shaped table, for two seeds."""
    records = []
    for seed in (0, 1):
        clients = standin.make_standin("loan", seed)
        for arm in (common.BASELINE, *common.ARMS):
            epsilon = None if arm == common.BASELINE else 8.0
            batch = common.FULL_BATCH if arm == "shared_whole" else 160
            setting = common.Setting("loan", "logistic", arm, epsilon, seed, 0.2, 1.0, 2, batch)
            records.append(common.run_setting(setting, clients))
    return records


def test_every_arm_gives_a_checked_record(tiny_records: list[dict[str, Any]]) -> None:
    assert len(tiny_records) == 2 * (1 + len(common.ARMS))
    for record in tiny_records:
        validate_result(record)
        assert record["kind"] == "phase3_run"
        data, setting = record["data"], record["config"]["benchmark"]
        assert data["label"] == (
            f"loan_logistic_{setting['arm']}_eps-"
            f"{'none' if setting['epsilon'] is None else '8'}_seed-{setting['seed']}"
        )
        communication = data["communication"]
        assert data["upload_bytes_per_run"] == (
            communication["upload_bytes_per_round"] * 2 + communication["setup_bytes"]
        )
        stored = data["privacy_report"]
        if setting["arm"] == common.BASELINE:
            assert stored is None
            continue
        assert odp.verify_report(stored) == []
        assert stored["within_budget"]
        # Only the trusted server is flagged; every other arm is a plan the library allows.
        assert bool(stored["unsafe_flags"]) == (setting["arm"] == "server")
        assert stored["policy"]["unit"] == (
            "client" if setting["arm"] == "participant" else "record"
        )
        if setting["arm"] in ("shared", "shared_whole", "participant"):
            assert len(stored["mode"]["collusion_table"]) == 5
            assert stored["mode"]["max_colluding"] == common.COLLUDING
    by_arm = {r["config"]["benchmark"]["arm"]: r["data"] for r in tiny_records}
    assert by_arm["shared_whole"]["privacy_report"]["mode"]["noise"] == "skellam"
    assert by_arm["local_masked8"]["communication"]["bits_per_value"] == 11.0
    assert by_arm["local_masked"]["communication"]["bits_per_value"] == 15.0
    assert by_arm["local"]["communication"]["ratio"] == 1.0


def test_the_report_compares_arms_seed_by_seed(tiny_records: list[dict[str, Any]]) -> None:
    rows = report.summarize(tiny_records)
    assert [row["arm"] for row in rows] == [common.BASELINE, *common.ARMS]
    index = {row["arm"]: row for row in rows}
    assert index["shared"]["seeds"] == [0, 1]
    assert index["shared"]["test_auc"]["n"] == 2
    for arm in ("shared", "shared_whole", "server", "participant", "local_tuned"):
        difference = index[arm]["auc_minus_local"]
        assert difference["n"] == 2
        assert difference["verdict"] in ("better", "worse", "no difference shown")
        assert difference["mean"] == pytest.approx(
            index[arm]["test_auc"]["mean"] - index["local"]["test_auc"]["mean"]
        )
    assert "auc_minus_local" not in index["local"]
    assert "auc_minus_local_tuned" not in index["local_tuned"]
    assert "epsilon_spent_max" not in index[common.BASELINE]
    assert index["server"]["unsafe_flags"] != []
    text = report.markdown(rows, [], len(tiny_records), 0)
    assert chr(0x2014) not in text
    assert "## Loan table: further tables" in text
    assert "| logistic | shared noise, whole numbers |" in text
    assert report.headline_markdown(rows) in text


def test_a_difference_counts_only_if_its_interval_excludes_zero() -> None:
    def members(values: list[float]) -> dict[int, dict[str, Any]]:
        return {seed: {"test": {"auc": value}} for seed, value in enumerate(values)}

    reference = members([0.70, 0.72, 0.71, 0.69, 0.73])
    clear = report._differences(members([0.80, 0.83, 0.80, 0.79, 0.84]), reference, "auc")
    assert clear["verdict"] == "better"
    assert clear["low"] > 0
    lower = report._differences(members([0.60, 0.61, 0.62, 0.60, 0.63]), reference, "auc")
    assert lower["verdict"] == "worse"
    noisy = report._differences(members([0.80, 0.60, 0.75, 0.66, 0.74]), reference, "auc")
    assert noisy["verdict"] == "no difference shown"
    assert noisy["low"] < 0 < noisy["high"]
    # Only seeds that both arms ran are compared.
    partial = report._differences(members([0.80, 0.83]), reference, "auc")
    assert partial["n"] == 2
    single = report._differences(members([0.9]), reference, "auc")
    assert single["verdict"] == "no difference shown"


def test_budgets_use_the_nearest_tuning_level() -> None:
    assert [run.level(budget) for budget in common.BUDGETS] == ["1", "1", "1", "8", "8"]


def test_the_tuning_grid_is_what_the_settings_file_records() -> None:
    assert len(tune.combinations("shared")) == 48
    assert len(tune.combinations("shared_whole")) == 24
    assert len(tune.combinations("participant")) == 6
    tuned = tune.combinations("local_tuned")
    assert len(tuned) == 30
    assert all(not (c[2] == 60 and c[4] == 3.0) for c in tuned)
    assert len(tune.all_tasks()) == 2 * 2 * 2 * 2 * (48 + 24 + 30 + 6)
    path = ROOT / "benchmarks" / "phase3" / "settings.json"
    if not path.is_file():
        pytest.skip("the settings have not been chosen yet")
    settings = json.loads(path.read_text(encoding="utf-8"))
    assert "No real row was used" in settings["note"]
    assert len(settings["chosen"]) == 2 * 2 * 4 * 2
    assert len(settings["grid"]) == 2 * 2 * 2 * (48 + 24 + 30 + 6)
    for key, entry in settings["chosen"].items():
        arm = key.split("/")[2]
        combination = tuple(entry[name] for name in tune.FIELDS)
        assert combination in tune.combinations(arm), key
        assert 0.0 <= entry["standin_validation_auc"] <= 1.0
    # Choosing again from the recorded grid gives the recorded choice.
    for dataset in ("stroke", "loan"):
        runs = run.settings_for(dataset, 42)
        assert len(runs) == RUNS_PER_TABLE_AND_SEED
        by = {(s.model, s.arm, s.epsilon): s for s in runs}
        for model in common.MODELS:
            for budget in common.BUDGETS:
                shared, server = by[(model, "shared", budget)], by[(model, "server", budget)]
                # The trusted server differs from the shared arm in the noise alone.
                assert (shared.rounds, shared.batch_size, shared.learning_rate) == (
                    server.rounds,
                    server.batch_size,
                    server.learning_rate,
                )
                assert by[(model, "shared_whole", budget)].batch_size == common.FULL_BATCH
                local = by[(model, "local", budget)]
                assert (local.rounds, local.batch_size, local.local_epochs) == (20, 32, 3.0)
    quick = run.settings_for("loan", 42, quick=True)
    assert all(setting.rounds == 2 for setting in quick)


# --------------------------------------------------------------------------- the stored results

RESULTS = ROOT / "benchmarks" / "results" / "phase3"
MEMO = ROOT / "docs" / "PHASE3_RESULTS.md"
EXPECTED_RUNS = RUNS_PER_TABLE_AND_SEED * (10 + 5)


@pytest.fixture(scope="module")
def summary() -> dict[str, Any]:
    if not (RESULTS / "summary.json").is_file():
        pytest.skip("the Phase 3 results are not present")
    data: dict[str, Any] = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    return data


def test_the_stored_summary_covers_the_whole_grid_and_passed_its_checks(
    summary: dict[str, Any],
) -> None:
    assert summary["runs"] == EXPECTED_RUNS
    assert summary["failed_checks"] == []
    assert len(list((RESULTS / "runs").glob("*.json"))) == EXPECTED_RUNS
    # The local arm and the baseline at the five Phase 2 seeds: (1 + 5) per table and model.
    assert summary["compared_with_phase2"] == 2 * 2 * 6 * 5
    rows = summary["rows"]
    assert len(rows) == 2 * RUNS_PER_TABLE_AND_SEED
    for row in rows:
        assert row["seeds"] == summary["seeds"][row["dataset"]]
        if row["epsilon"] is None:
            continue
        assert row["all_within_budget"]
        assert row["epsilon_spent_max"] <= row["epsilon"] * (1 + 1e-6)
        assert row["epsilon_spent_max"] > 0.98 * row["epsilon"]
        assert bool(row["unsafe_flags"]) == (row["arm"] == "server")
    assert len(summary["seeds"]["stroke"]) == 10
    assert len(summary["seeds"]["loan"]) == 5


def test_one_stored_run_of_each_new_mode_verifies(summary: dict[str, Any]) -> None:
    for arm in ("shared", "shared_whole", "participant"):
        path = RESULTS / "runs" / f"stroke_logistic_{arm}_eps-1_seed-42.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        validate_result(record)
        stored = record["data"]["privacy_report"]
        assert odp.verify_report(stored) == []
        assert stored["policy"]["epsilon"] == 1.0
        stored["mode"]["collusion_table"][2]["epsilon"] = 0.1
        assert odp.verify_report(stored) != []


def test_the_results_memo_quotes_the_stored_numbers(summary: dict[str, Any]) -> None:
    if not MEMO.is_file():
        pytest.skip("the results memo is not present")
    memo = MEMO.read_text(encoding="utf-8")
    assert report.headline_markdown(summary["rows"]) in memo
    assert chr(0x2014) not in memo
    stored = (RESULTS / "summary.md").read_text(encoding="utf-8")
    assert report.headline_markdown(summary["rows"]) in stored
    assert chr(0x2014) not in stored


@pytest.mark.slow
def test_the_stored_summary_is_what_the_report_script_produces(summary: dict[str, Any]) -> None:
    """Rebuild everything from the stored runs: every report recomputed, every row equal."""
    runs, problems = report.load_runs()
    compared, differing = report.check_against_phase2(runs)
    assert problems == []
    assert differing == []
    assert compared == summary["compared_with_phase2"]
    rebuilt = {
        (r["dataset"], r["model"], r["arm"], r["epsilon"]): r for r in report.summarize(runs)
    }
    for row in summary["rows"]:
        fresh = rebuilt[(row["dataset"], row["model"], row["arm"], row["epsilon"])]
        assert fresh["test_auc"]["mean"] == pytest.approx(row["test_auc"]["mean"], abs=1e-12)
        assert fresh["upload_bytes_per_run"] == row["upload_bytes_per_run"]
