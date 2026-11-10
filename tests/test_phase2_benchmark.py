"""The Phase 2 benchmark scripts: data preparation and one tiny run on made-up rows.

The scripts live in ``benchmarks/`` and are not part of the installed package, so these
tests only run from a full checkout of the repository.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import obscuradp as odp
from obscuradp.results import validate_result

ROOT = Path(__file__).resolve().parents[1]

if not (ROOT / "benchmarks" / "phase2" / "tables.py").is_file():
    pytest.skip("the benchmark scripts are not part of this installation", allow_module_level=True)

sys.path.insert(0, str(ROOT))

from benchmarks.phase2 import common, report, run, standin, tables  # noqa: E402


def test_declared_schemas_encode_the_real_tables_without_statistics() -> None:
    if not tables.FILES["stroke"].is_file():
        pytest.skip("the research data files are not present")
    columns, labels = tables.load_table("stroke")
    encoded = tables.STROKE_SCHEMA.encode(columns)
    assert encoded.shape == (4908, 20)  # 5,110 rows minus 201 without BMI and 1 "Other"
    assert int(labels.sum()) == 209
    assert float(np.abs(encoded).max()) <= 1.0
    columns, labels = tables.load_table("loan")
    encoded = tables.LOAN_SCHEMA.encode(columns)
    assert encoded.shape == (4269, 13)
    assert int(labels.sum()) == 2656
    with pytest.raises(ValueError, match="unknown dataset"):
        tables.load_table("iris")


def test_participants_get_every_row_once_and_keep_three_parts() -> None:
    if not tables.FILES["stroke"].is_file():
        pytest.skip("the research data files are not present")
    for name, rows in (("stroke", 4908), ("loan", 4269)):
        clients = tables.make_clients(name, seed=42)
        assert len(clients) == tables.NUM_CLIENTS
        total = sum(c.train_y.size + c.validation_y.size + c.test_y.size for c in clients)
        assert total == rows
        assert all(c.validation_y.size > 0 and c.test_y.size > 0 for c in clients)
        again = tables.make_clients(name, seed=42)
        assert all(
            np.array_equal(a.train_x, b.train_x) for a, b in zip(clients, again, strict=True)
        )
        features, labels = tables.balanced_rows(clients, seed=42)
        assert labels.mean() == 0.5
        assert features.shape[0] == labels.size
    # Stroke participants each hold at least 12 positive rows, as in the research split.
    stroke = tables.make_clients("stroke", seed=42)
    for client in stroke:
        positives = client.train_y.sum() + client.validation_y.sum() + client.test_y.sum()
        assert positives >= 12


def test_the_stand_in_copies_only_public_shape() -> None:
    for name in ("stroke", "loan"):
        clients = standin.make_standin(name, seed=0)
        rows, share, _ = standin.SHAPES[name]
        labels = np.concatenate(
            [np.concatenate([c.train_y, c.validation_y, c.test_y]) for c in clients]
        )
        assert labels.size == rows
        assert float(labels.mean()) == pytest.approx(share, abs=0.002)
        assert clients[0].train_x.shape[1] == tables.SCHEMAS[name].width
        assert not np.array_equal(
            clients[0].train_x, standin.make_standin(name, seed=1)[0].train_x[: clients[0].n_train]
        )


def test_budgets_use_the_nearest_tuning_level() -> None:
    assert run._level(None) == "none"
    assert [run._level(budget) for budget in common.BUDGETS] == ["1", "1", "1", "8", "8"]


def test_one_tiny_run_gives_a_checked_record() -> None:
    clients = standin.make_standin("loan", seed=0)
    setting = common.Setting("loan", "logistic", 8.0, True, 0.2, 1.0, seed=0, rounds=1)
    assert setting.label == "loan_logistic_eps-8_masked_seed-0"
    record = common.run_setting(setting, clients)
    validate_result(record)
    data = record["data"]
    assert odp.verify_report(data["privacy_report"]) == []
    assert data["privacy_report"]["within_budget"]
    assert data["communication"]["ratio"] < 0.5
    assert set(data["test_balanced"]) == set(data["test"])
    assert data["balanced_rows"] % 2 == 0
    assert record["config"]["benchmark"]["epsilon"] == 8.0

    baseline = common.Setting("loan", "network", None, False, 0.2, 1.0, seed=0, rounds=1)
    assert baseline.label == "loan_network_eps-none_clear_seed-0"
    plain = common.run_setting(baseline, clients)
    assert plain["data"]["privacy_report"] is None

    rows = report.summarize([record, plain])
    assert [row["epsilon"] for row in rows] == [8.0, None]
    private = rows[0]
    assert private["all_within_budget"]
    assert private["unsafe_flags"] == []
    assert private["test_auc"]["n"] == 1
    text = report.markdown(rows, {"stroke_frontier_fedprox_dp": [], "arms_seed_42": []}, [], 2)
    assert chr(0x2014) not in text
    assert "## loan, network: updates in the clear" in text
    assert "| none | none |" in text


def test_interval_helper() -> None:
    stat = common.mean_and_interval([0.5, 0.7, 0.6])
    assert stat["mean"] == pytest.approx(0.6)
    assert stat["std"] == pytest.approx(0.1)
    # Student's t with 2 degrees of freedom: 4.303 * 0.1 / sqrt(3).
    assert stat["high"] - stat["mean"] == pytest.approx(0.2484, abs=1e-3)
    assert common.mean_and_interval([0.5])["n"] == 1
    assert common.mean_and_interval([float("nan")])["n"] == 0


# --------------------------------------------------------------------------- the stored results

RESULTS = ROOT / "benchmarks" / "results" / "phase2"
MEMO = ROOT / "docs" / "GATE_PHASE2.md"


@pytest.fixture(scope="module")
def summary() -> dict:  # type: ignore[type-arg]
    import json

    if not (RESULTS / "summary.json").is_file():
        pytest.skip("the Phase 2 results are not present")
    return json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_the_stored_summary_covers_the_whole_grid_and_passed_its_checks(summary: dict) -> None:  # type: ignore[type-arg]
    assert summary["runs"] == 200
    assert summary["failed_checks"] == []
    assert len(list((RESULTS / "runs").glob("*.json"))) == 200
    rows = summary["rows"]
    assert len(rows) == 40
    assert all(row["seeds"] == [42, 43, 44, 45, 46] for row in rows)
    protected = [row for row in rows if row["epsilon"] is not None]
    assert len(protected) == 32
    for row in protected:
        assert row["all_within_budget"]
        assert row["unsafe_flags"] == []
        assert row["epsilon_spent_max"] <= row["epsilon"]
        assert row["epsilon_spent_max"] > 0.99 * row["epsilon"]
    masked = [row for row in rows if row["secure_sum"]]
    assert all(0.46 < row["upload_ratio"] < 0.5 for row in masked)  # the bytes target
    assert all(row["saturated_share_max"] == 0.0 for row in masked)


def test_one_stored_run_verifies_from_its_own_recorded_releases() -> None:
    import json

    path = RESULTS / "runs" / "stroke_logistic_eps-1_masked_seed-42.json"
    if not path.is_file():
        pytest.skip("the Phase 2 results are not present")
    record = json.loads(path.read_text(encoding="utf-8"))
    validate_result(record)
    stored = record["data"]["privacy_report"]
    assert odp.verify_report(stored) == []
    assert stored["policy"]["epsilon"] == 1.0
    assert record["data"]["communication"]["setup_bytes"] == 160.0
    # Tampering with a stored result is noticed.
    stored["scopes"][0]["events"][0]["noise_multiplier"] /= 2
    assert odp.verify_report(stored) != []


def test_the_gate_memo_quotes_the_stored_numbers(summary: dict) -> None:  # type: ignore[type-arg]
    if not MEMO.is_file():
        pytest.skip("the gate memo is not present")
    memo = MEMO.read_text(encoding="utf-8")
    assert report.headline_markdown(summary["rows"]) in memo
    assert chr(0x2014) not in memo
    earlier = {
        e["label_epsilon_per_round"]: e
        for e in summary["earlier_results"]["stroke_frontier_fedprox_dp"]
    }
    for label in (0.5, 1.0, 9.0, 50.0):
        entry = earlier[label]
        recall = f"{entry['recall']['mean']:.3f} ± {entry['recall']['std']:.3f}"
        whole_run = f"{entry['whole_run_epsilon_by_old_code_max']:.1f}"
        assert f"| {label:g} | {whole_run} | {recall} |" in memo


@pytest.mark.slow
def test_the_stored_summary_is_what_the_report_script_produces(summary: dict) -> None:  # type: ignore[type-arg]
    """Rebuild the rows from the stored runs and compare a few of them with the stored file."""
    runs, problems = report.load_runs()
    assert problems == []
    rebuilt = {
        (r["dataset"], r["model"], r["epsilon"], r["secure_sum"]): r for r in report.summarize(runs)
    }
    for row in summary["rows"]:
        fresh = rebuilt[(row["dataset"], row["model"], row["epsilon"], row["secure_sum"])]
        assert fresh["test_auc"]["mean"] == pytest.approx(row["test_auc"]["mean"], abs=1e-12)
        assert fresh["upload_ratio"] == row["upload_ratio"]
