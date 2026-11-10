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
