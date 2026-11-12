"""The Phase 4 attack-test scripts: every arm on made-up rows, and the stored results.

The scripts live in ``benchmarks/`` and are not part of the installed package, so these
tests only run from a full checkout of the repository.
"""

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import obscuradp as odp
from obscuradp.audit.attribute import attribute_from_schema
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
            assert row["above_noise_limit"] is None
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
            assert row["above_noise_limit"] is False, row
            assert row["epsilon_lower"] <= row["epsilon_claimed"]
            assert row["advantage"] <= row["ceiling_mechanism"]
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
        for key in (
            "dataset",
            "model",
            "arm",
            "budget",
            "attack",
            "guesses",
            "correct",
            "flagged",
            "above_noise_limit",
        ):
            assert fresh[key] == row[key]
        for key in ("epsilon_lower", "advantage", "advantage_low", "advantage_high", "best_case"):
            assert fresh[key] == pytest.approx(row[key], abs=1e-12)


# --------------------------------------------------------------------------- further attacks (S16)

from benchmarks.phase4 import more, report_more  # noqa: E402

MORE = RESULTS / "more"
MORE_MEMO = ROOT / "docs" / "PHASE4_MORE_ATTACKS.md"


def _made_up_with_proper_columns(name: str, seed: int) -> list[Any]:
    """A made-up table whose yes/no columns are encoded the way the real ones are."""
    clients = []
    for client in standin.make_standin(name, seed):
        parts = {}
        for part in ("train_x", "validation_x", "test_x"):
            table = getattr(client, part).copy()
            for column in more.HIDDEN_COLUMNS[name]:
                hidden = attribute_from_schema(more.SCHEMAS[name], column)
                first = table[:, hidden.columns[0]] > 0
                if len(hidden.columns) == 1:  # a flag stored as one number
                    table[:, hidden.columns[0]] = np.where(first, 1.0, -1.0)
                else:  # one 0/1 column per category
                    table[:, hidden.columns[0]] = first
                    table[:, hidden.columns[1]] = ~first
            parts[part] = table
        clients.append(dataclasses.replace(client, **parts))
    return clients


@pytest.fixture(scope="module")
def tiny_more(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The quick mode of every part on made-up tables shaped like the real ones."""
    directory = tmp_path_factory.mktemp("more")
    cache = tmp_path_factory.mktemp("cache")
    patch = pytest.MonkeyPatch()
    patch.setattr(more, "make_clients", _made_up_with_proper_columns)
    patch.setattr(more, "DATASETS", ("stroke",))
    # One model, and the rare rows on the split the other parts use: the noise levels, which
    # take longest to work out, are then shared by all parts.
    patch.setattr(more, "MODELS", ("logistic",))
    try:
        for arm, epsilon in more.PROTECTION[:2]:
            task = ("stroke", "logistic", arm, epsilon, True, str(directory), str(cache))
            assert more.shadow_cell(task) == more.cell_name("stroke", "logistic", arm, epsilon)
        assert more.rare_seed((42, True, str(directory))) == "rare_stroke_seed-42"
        assert more.inversion_part(str(directory), True) == "inversion"
        # A finished part is not computed again.
        before = (directory / "inversion.json").stat().st_mtime_ns
        more.inversion_part(str(directory), True)
        assert (directory / "inversion.json").stat().st_mtime_ns == before
    finally:
        patch.undo()
    return directory


def test_shadow_cells_store_figures_per_run_and_no_single_records_score(tiny_more: Path) -> None:
    for name in ("none_eps-none", "local_eps-1"):
        record = json.loads((tiny_more / f"shadow_stroke_logistic_{name}.json").read_text())
        validate_result(record)
        data = record["data"]
        assert len(data["runs"]) == 8  # the quick mode's number of runs
        assert data["rows_label_1"] < data["rows"] / 5
        assert set(data["without_model"]) == set(more.HIDDEN_COLUMNS["stroke"])
        for index, stored in enumerate(data["runs"]):
            run = stored
            assert set(run["shadow_advantage"]) == {"all", "label_1", "label_0"}
            assert set(run["hidden_columns"]) == set(more.HIDDEN_COLUMNS["stroke"])
            for figures in run["hidden_columns"].values():
                assert all(0.0 <= value <= 1.0 for value in figures.values())
            # Only the runs set apart as targets carry guesses that may be pooled.
            assert ("pooled" in run) == (index < 4)
            if "pooled" in run:
                made, right = run["pooled"]["guesses"]
                assert 0 <= right <= made <= run["pooled"]["canaries"] == data["rows"]
            assert (run["epsilon"] is None) == (name == "none_eps-none")
        # Nothing as long as the list of rows is stored.
        assert len(json.dumps(record["data"])) < 12_000


def test_the_rare_rows_are_all_test_records_beside_as_many_common_ones(tiny_more: Path) -> None:
    record = json.loads((tiny_more / "rare_stroke_seed-42.json").read_text())
    validate_result(record)
    cells = record["data"]["cells"]
    assert set(cells) == {f"stroke_logistic_local_eps-{budget}" for budget in (1, 8)}
    clients = _made_up_with_proper_columns("stroke", 42)
    rare = sum(int((client.train_y > 0.5).sum()) for client in clients)
    # A participant with fewer common rows than rare ones adds all it has.
    common = sum(
        min(int((client.train_y > 0.5).sum()), int((client.train_y <= 0.5).sum()))
        for client in clients
    )
    for stored in cells.values():
        assert stored["canaries_label_1"] == rare
        assert stored["canaries"] == rare + common
        assert set(stored["advantage"]) == {ATTACK_UPDATES, ATTACK_FINAL_MODEL}


def test_a_record_alone_is_read_back_and_a_protected_one_is_not(tiny_more: Path) -> None:
    record = json.loads((tiny_more / "inversion.json").read_text())
    validate_result(record)
    for cell in record["data"]["cells"].values():
        errors = cell["median_error"]
        assert errors["alone, no protection"] < 1e-3
        assert cell["share_within_10_percent"]["alone, no protection"] == 1.0
        average = errors["no update at all (the average record)"]
        # Behind the plan's noise, and among other records, nothing better than the average.
        for view in ("alone, default plan, budget 1", "alone, default plan, budget 8"):
            assert errors[view] > average
            assert cell["share_within_10_percent"][view] == 0.0
        assert errors["one of 32, no protection"] > 0.5 * average


def test_the_tables_of_the_further_attacks_are_built_from_what_is_stored(
    tiny_more: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(report_more, "DATASETS", ("stroke",))
    monkeypatch.setattr(report_more, "MODELS", ("logistic",))
    monkeypatch.setattr(report_more, "PROTECTION", more.PROTECTION[:2])
    monkeypatch.setattr(report_more, "SHADOW_SEEDS", more.SHADOW_SEEDS[:8])
    monkeypatch.setattr(report_more, "POOLED_TARGETS", more.POOLED_TARGETS[:4])
    monkeypatch.setattr(report_more, "RARE_SEEDS", (42,))
    monkeypatch.setattr(report_more, "RARE_BUDGETS", (1.0,))
    summary = report_more.build_summary(tiny_more)
    assert summary["failed_checks"] == []
    assert (summary["shadow_runs"], summary["rare_runs"]) == (16, 1)
    protected = summary["shadow"][1]
    assert protected["budget"] == 1.0
    assert protected["flagged"] is False
    assert protected["pooled_runs"] == 4
    assert protected["ceiling_any"] == pytest.approx(0.462, abs=1e-3)
    assert summary["shadow"][0]["flagged"] is None
    text = report_more.headline_markdown(summary)
    for part in (
        "Shadow models",
        "the rare class apart",
        "hidden yes/no column",
        "Reading a record",
    ):
        assert part in text
    assert chr(0x2014) not in text
    # A cell that is missing is reported, not passed over.
    monkeypatch.setattr(report_more, "PROTECTION", more.PROTECTION)
    assert any("missing" in item for item in report_more.build_summary(tiny_more)["failed_checks"])


@pytest.fixture(scope="module")
def more_summary() -> dict[str, Any]:
    if not (MORE / "summary.json").is_file():
        pytest.skip("the results of the further attacks are not present")
    data: dict[str, Any] = json.loads((MORE / "summary.json").read_text(encoding="utf-8"))
    return data


def test_the_stored_further_attacks_are_complete_and_passed_their_checks(
    more_summary: dict[str, Any],
) -> None:
    assert more_summary["failed_checks"] == []
    assert more_summary["shadow_runs"] == 2 * 2 * 3 * 32
    assert more_summary["rare_runs"] == 2 * 2 * 30
    assert more_summary["guess_share"] == 0.05
    for row in more_summary["shadow"]:
        if row["budget"] is None:
            continue
        # No protected cell leaks more than its report states, by either sign.
        assert row["flagged"] is False
        assert row["shadow"]["all"]["mean"] <= row["ceiling_mechanism"]
        assert row["epsilon_lower"] <= row["budget"]
    for cell in more_summary["inversion"].values():
        assert cell["median_error"]["alone, no protection"] < 1e-3
        assert cell["share_within_10_percent"]["alone, default plan, budget 1"] == 0.0


def test_the_memo_of_the_further_attacks_quotes_the_stored_numbers(
    more_summary: dict[str, Any],
) -> None:
    if not MORE_MEMO.is_file():
        pytest.skip("the memo of the further attacks is not present")
    memo = MORE_MEMO.read_text(encoding="utf-8")
    assert report_more.headline_markdown(more_summary) in memo
    assert chr(0x2014) not in memo
    stored = (MORE / "summary.md").read_text(encoding="utf-8")
    assert report_more.headline_markdown(more_summary) in stored


@pytest.mark.slow
def test_the_stored_further_attacks_are_what_the_report_script_produces(
    more_summary: dict[str, Any],
) -> None:
    rebuilt = report_more.build_summary()
    assert rebuilt["failed_checks"] == []
    assert json.loads(json.dumps(rebuilt)) == more_summary
