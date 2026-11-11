"""The Phase 4 attack-test scripts: every arm on made-up rows, and the stored results.

The scripts live in ``benchmarks/`` and are not part of the installed package, so these
tests only run from a full checkout of the repository.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

import obscuradp as odp
from obscuradp.audit.membership import ATTACK_FINAL_MODEL, ATTACK_UPDATES, ATTACKS, GUESS_SHARES
from obscuradp.mechanisms import dpsgd as dpsgd_module
from obscuradp.results import validate_result

ROOT = Path(__file__).resolve().parents[1]

if not (ROOT / "benchmarks" / "phase4" / "common.py").is_file():
    pytest.skip("the benchmark scripts are not part of this installation", allow_module_level=True)

sys.path.insert(0, str(ROOT))

from benchmarks.phase2 import standin  # noqa: E402
from benchmarks.phase3.common import FULL_BATCH, Setting  # noqa: E402
from benchmarks.phase4 import common, report, run  # noqa: E402

CELLS_PER_TABLE = 2 * (1 + 5 + 1) + 2 * 2  # both models; the shared arms, logistic only
PILOT_CELLS_PER_TABLE = 2 * (1 + 2 + 1) + 2 * 2


def test_the_grid_is_the_one_the_memo_describes() -> None:
    settings = run.settings_for("stroke", 100)
    assert len(settings) == CELLS_PER_TABLE
    assert len({run.cell(setting) for setting in settings}) == CELLS_PER_TABLE
    assert len(report.grid()) == 2 * CELLS_PER_TABLE
    assert len(run.settings_for("loan", 90, pilot=True)) == PILOT_CELLS_PER_TABLE
    # The reported seeds are thirty, and none of them was used by the pilot or earlier phases.
    assert len(common.SEEDS) == 30
    assert not set(common.SEEDS) & set(common.PILOT_SEEDS)
    assert not (set(common.SEEDS) | set(common.PILOT_SEEDS)) & set(range(42, 52))
    by_arm = {setting.arm for setting in settings}
    assert by_arm == set(common.ARMS)
    # The default plan and the planted defect keep the Phase 2 protocol.
    for setting in settings:
        if setting.arm in ("local", common.SILENT, common.BASELINE):
            assert (setting.rounds, setting.batch_size, setting.local_epochs) == (20, 32, 3.0)
    quick = run.settings_for("stroke", 100, quick=True)
    assert {setting.rounds for setting in quick} == {2}


@pytest.fixture(scope="module")
def tiny_cells() -> dict[str, dict[int, dict[str, Any]]]:
    """Two rounds of every arm on a made-up loan-shaped table, for two seeds."""
    cells: dict[str, dict[int, dict[str, Any]]] = {}
    for seed in (0, 1):
        clients = standin.make_standin("loan", seed)
        for arm in common.ARMS:
            epsilon = None if arm == common.BASELINE else 1.0
            batch = FULL_BATCH if arm == "shared_whole" else 160
            setting = Setting("loan", "logistic", arm, epsilon, seed, 0.2, 1.0, 2, batch)
            cells.setdefault(run.cell(setting), {})[seed] = common.run_setting(setting, clients)
    return cells


def test_every_arm_gives_counts_and_no_single_records_score(
    tiny_cells: dict[str, dict[int, dict[str, Any]]],
) -> None:
    assert len(tiny_cells) == len(common.ARMS)
    for cell, runs in tiny_cells.items():
        for stored in runs.values():
            assert set(stored["guesses"]) == set(ATTACKS)
            for attack in ATTACKS:
                assert set(stored["guesses"][attack]) == {f"{share:g}" for share in GUESS_SHARES}
                made, right = stored["guesses"][attack]["1"]
                assert 0 <= right <= made <= stored["canaries"]
                assert -1.0 <= stored["advantage"][attack] <= 1.0
            # Nothing as long as the list of test records is stored.
            longest = max(len(value) for value in stored.values() if isinstance(value, list))
            assert longest < 10
            if "_none_" in cell:
                assert stored["privacy_report"] is None
                assert stored["epsilon"] is None
            else:
                assert odp.verify_report(stored["privacy_report"]) == []
                assert stored["epsilon"] == pytest.approx(1.0, abs=0.01)
            shared = "_shared" in cell
            assert stored["colluders"] == (["site-4"] if shared else [])
            assert stored["canaries"] == (596 if shared else 745)
    json.dumps(tiny_cells, allow_nan=False)


def test_the_planted_defect_is_planted_in_the_benchmark_only(
    tiny_cells: dict[str, dict[int, dict[str, Any]]],
) -> None:
    honest = dpsgd_module.gaussian_noise
    with common.noise_switched_off():
        assert dpsgd_module.gaussian_noise is not honest
    assert dpsgd_module.gaussian_noise is honest
    # The defective runs still state a budget of 1: their ledger believes the noise.
    silent = tiny_cells[f"loan_logistic_{common.SILENT}_eps-1"]
    assert all(stored["epsilon"] == pytest.approx(1.0, abs=0.01) for stored in silent.values())


def test_cells_are_put_together_with_the_shares_fixed_beforehand(
    tiny_cells: dict[str, dict[int, dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(report, "DATASETS", ("loan",))
    monkeypatch.setattr(report, "MODELS", ("logistic",))
    monkeypatch.setattr(report, "BUDGETS", (1.0,))
    tiny_arms = {
        arm: (text, (None,) if arm == common.BASELINE else (1.0,), ("logistic",))
        for arm, (text, _, _) in common.ARMS.items()
    }
    monkeypatch.setattr(report, "ARMS", tiny_arms)
    shares = {ATTACK_UPDATES: 0.5, ATTACK_FINAL_MODEL: 1.0}
    rows = report.summarize_cells(tiny_cells, shares)
    assert len(rows) == 2 * len(common.ARMS)
    for row in rows:
        assert row["runs"] == 2
        assert row["seeds"] == [0, 1]
        assert row["share"] == shares[row["attack"]]
        assert row["advantage_low"] <= row["advantage"] <= row["advantage_high"]
        if row["arm"] == common.BASELINE:
            assert row["epsilon_claimed"] is None
            assert row["flagged"] is None
            assert row["ceiling_mechanism"] is None
        else:
            assert row["epsilon_claimed"] == pytest.approx(1.0, abs=0.01)
            assert row["ceiling_any"] == pytest.approx(0.462, abs=1e-3)
            assert row["flagged"] in (True, False)
    text = report.headline_markdown(rows)
    for part in ("three numbers side by side", "Limit for any method", "46.2%", "the other arms"):
        assert part in text
    assert "planted defect" in text
    assert chr(0x2014) not in text
    chosen = report.choose_shares(tiny_cells)
    assert set(chosen["guess_share"]) == set(ATTACKS)
    assert all(share in GUESS_SHARES for share in chosen["guess_share"].values())
    # Only the three protected arms count for the choice.
    assert chosen["pilot_cells"] == 3
    assert chosen["pilot_runs"] == 2 * len(common.ARMS)


# --------------------------------------------------------------------------- stored results

RESULTS = ROOT / "benchmarks" / "results" / "phase4"
MEMO = ROOT / "docs" / "PHASE4_ATTACK_TESTS.md"
EXPECTED_RUNS = 2 * CELLS_PER_TABLE * 30


@pytest.fixture(scope="module")
def summary() -> dict[str, Any]:
    if not (RESULTS / "summary.json").is_file():
        pytest.skip("the Phase 4 results are not present")
    data: dict[str, Any] = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    return data


def test_the_stored_summary_covers_the_whole_grid_and_passed_its_checks(
    summary: dict[str, Any],
) -> None:
    assert summary["runs"] == EXPECTED_RUNS
    assert summary["failed_checks"] == []
    assert summary["seeds"] == list(common.SEEDS)
    assert len(list((RESULTS / "runs").glob("*.json"))) == 2 * 30
    rows = summary["rows"]
    assert len(rows) == 2 * 2 * CELLS_PER_TABLE
    settings = json.loads(report.SETTINGS.read_text(encoding="utf-8"))
    assert summary["guess_share"] == settings["guess_share"]
    for row in rows:
        assert row["runs"] == 30
        assert row["seeds"] == list(common.SEEDS)
        assert row["share"] == settings["guess_share"][row["attack"]]
        if row["arm"] == common.BASELINE:
            continue
        # A protected run never leaks more than its report says. The planted defect may.
        if row["arm"] != common.SILENT:
            assert row["flagged"] is False, row
            assert row["epsilon_lower"] <= row["epsilon_claimed"]
        assert row["epsilon_claimed"] <= row["budget"] * (1 + 1e-6)
        assert row["epsilon_claimed"] > 0.98 * row["budget"]


def test_the_pilot_did_not_use_a_reported_seed() -> None:
    if not report.SETTINGS.is_file():
        pytest.skip("the pilot's settings are not present")
    settings = json.loads(report.SETTINGS.read_text(encoding="utf-8"))
    assert set(settings["guess_share"]) == set(ATTACKS)
    assert "seeds 90 to 94" in settings["note"]
    for path in (RESULTS / "pilot").glob("*.json"):
        (seed,) = json.loads(path.read_text(encoding="utf-8"))["manifest"]["seeds"]
        assert seed in common.PILOT_SEEDS


def test_one_stored_privacy_report_of_each_arm_verifies(summary: dict[str, Any]) -> None:
    first = RESULTS / "runs" / f"stroke_seed-{common.SEEDS[0]}.json"
    record = json.loads(first.read_text(encoding="utf-8"))
    validate_result(record)
    assert record["kind"] == "phase4_seed"
    for arm in ("local", "shared", "shared_whole"):
        stored = record["data"]["cells"][f"stroke_logistic_{arm}_eps-1"]["privacy_report"]
        assert odp.verify_report(stored) == []
        assert stored["policy"]["epsilon"] == 1.0
        stored["scopes"][0]["epsilon"] = 0.1
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
    rebuilt, _ = report.build_summary()
    assert rebuilt["failed_checks"] == []
    assert rebuilt["runs"] == summary["runs"]
    assert len(rebuilt["rows"]) == len(summary["rows"])
    for fresh, row in zip(rebuilt["rows"], summary["rows"], strict=True):
        for key in ("dataset", "model", "arm", "budget", "attack", "guesses", "correct", "flagged"):
            assert fresh[key] == row[key]
        for key in ("epsilon_lower", "advantage", "advantage_low", "advantage_high", "best_case"):
            assert fresh[key] == pytest.approx(row[key], abs=1e-12)
