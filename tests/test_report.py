"""The privacy report, recomputing it, and the command line."""

import copy
import json
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

import obscuradp
from obscuradp.accounting import (
    GaussianEvent,
    SubsampledGaussianEvent,
    attacker_advantage_ceiling,
    calibrate_noise_multiplier,
    epsilon_pld,
    epsilon_rdp,
    general_advantage_ceiling,
)
from obscuradp.cli import main
from obscuradp.ledger import PrivacyLedger
from obscuradp.policy import PrivacyPolicy
from obscuradp.report import NOT_COVERED, SCHEMA, PrivacyReport, load_report, verify_report

DELTA = 1e-5
RATE = 0.1
STEPS = 200


@pytest.fixture(scope="module")
def ledger() -> PrivacyLedger:
    """Two participants: one uses the whole budget of 1, the other 60% of the steps."""
    sigma = calibrate_noise_multiplier(1.0, DELTA, RATE, STEPS).noise_multiplier
    ledger = PrivacyLedger(PrivacyPolicy(1.0, DELTA))
    ledger.charge("client-0", SubsampledGaussianEvent(sigma, RATE, STEPS))
    ledger.charge("client-1", SubsampledGaussianEvent(sigma, RATE, 120))
    return ledger


@pytest.fixture(scope="module")
def report(ledger: PrivacyLedger) -> PrivacyReport:
    return PrivacyReport.from_ledger(
        ledger,
        protection=["Each participant's records are protected from the server and from others."],
        caveats=["Evaluation used a held-out table that is not protected."],
    )


@pytest.fixture
def stored(report: PrivacyReport) -> dict[str, Any]:
    return copy.deepcopy(report.to_dict())


# --------------------------------------------------------------------------- the report itself


def test_report_numbers_come_from_the_recorded_releases(
    ledger: PrivacyLedger, report: PrivacyReport
) -> None:
    assert obscuradp.PrivacyReport is PrivacyReport
    events = ledger.events("client-0")
    first, second = report.scopes
    assert (first.scope, second.scope) == ("client-0", "client-1")
    assert first.events == events
    assert first.budget == 1.0
    assert first.epsilon == epsilon_pld(events, DELTA)
    assert first.check.pld_lower == epsilon_pld(events, DELTA, pessimistic=False)
    assert first.check.rdp == epsilon_rdp(events, DELTA)
    assert first.advantage_ceiling == attacker_advantage_ceiling(events)
    assert second.epsilon < first.epsilon

    # The run as a whole is as private as its least private participant.
    assert report.epsilon == first.epsilon
    assert 0.99 < report.epsilon <= 1.0
    assert report.within_budget
    assert report.is_safe
    assert report.advantage_ceiling == first.advantage_ceiling
    assert report.general_advantage_ceiling == general_advantage_ceiling(report.epsilon, DELTA)
    assert report.advantage_ceiling < report.general_advantage_ceiling
    assert report.library_version == metadata.version("obscuradp")


def test_summary_is_plain_and_complete(report: PrivacyReport) -> None:
    text = report.summary()
    assert text.startswith("PRIVACY REPORT")
    assert "one record" in text
    assert "(within budget; the largest spend of any scope)" in text
    assert "true-positive rate minus false-positive rate" in text
    assert "client-0: epsilon = " in text
    assert "(200 noisy releases in 1 group(s))" in text
    assert "Who is protected from whom:" in text
    assert "UNSAFE" not in text
    for line in NOT_COVERED:
        assert line in text
    assert "Evaluation used a held-out table" in text
    assert "obscuradp report verify" in text
    assert chr(0x2014) not in text  # house style: no long dashes


def test_an_empty_run_spent_nothing() -> None:
    report = PrivacyReport.from_ledger(PrivacyLedger(PrivacyPolicy(1.0, DELTA)))
    assert report.epsilon == 0.0
    assert report.advantage_ceiling == 0.0
    assert report.within_budget
    assert "No release was recorded" in report.summary()
    assert verify_report(report.to_dict()) == []


def test_unsafe_flags_make_a_report_unsafe_and_say_so(ledger: PrivacyLedger) -> None:
    report = PrivacyReport.from_ledger(
        ledger, unsafe_flags=["Synthetic rows were made before the private step."]
    )
    assert report.within_budget
    assert not report.is_safe
    assert "UNSAFE: the stated guarantee is weakened or void because:" in report.summary()
    assert "Synthetic rows were made" in report.summary()
    assert report.to_dict()["unsafe_flags"] == ["Synthetic rows were made before the private step."]


def test_a_run_over_budget_is_reported_as_over() -> None:
    """A ledger rebuilt from stored data is trusted as given, so it can be over budget."""
    policy = PrivacyPolicy(0.5, DELTA)
    data = {
        "policy": policy.to_dict(),
        "scopes": {"a": [SubsampledGaussianEvent(2.0, 0.1, 200).to_dict()]},
    }
    report = PrivacyReport.from_ledger(PrivacyLedger.from_dict(data))
    assert report.epsilon > 0.5
    assert not report.within_budget
    assert not report.is_safe
    assert "(OVER budget;" in report.summary()
    problems = verify_report(report.to_dict())
    assert problems == ["at least one scope spent more than its budget"]


def test_a_short_run_notes_that_the_older_method_is_much_looser() -> None:
    ledger = PrivacyLedger(PrivacyPolicy(1.0, DELTA))
    ledger.charge("a", SubsampledGaussianEvent(1.0, 0.01, 60))
    text = PrivacyReport.from_ledger(ledger).summary()
    assert "Note: the RDP limit" in text
    assert "normal for short runs" in text


# --------------------------------------------------------------------------- files and verifying


def test_stored_report_is_strict_json_and_verifies(report: PrivacyReport, tmp_path: Path) -> None:
    path = tmp_path / "privacy_report.json"
    report.to_json(path)
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "NaN" not in text
    assert "Infinity" not in text
    data = load_report(path)
    assert data == json.loads(json.dumps(report.to_dict()))
    assert data["schema"] == SCHEMA
    assert data["not_covered"] == list(NOT_COVERED)
    assert data["scopes"][0]["events"] == [event.to_dict() for event in report.scopes[0].events]
    assert verify_report(data) == []


def test_verify_notices_an_edited_headline(stored: dict[str, Any]) -> None:
    stored["epsilon"] = 0.3
    problems = verify_report(stored)
    assert len(problems) == 1
    assert "states a total of epsilon = 0.3" in problems[0]


def test_verify_notices_an_edited_scope_figure(stored: dict[str, Any]) -> None:
    stored["scopes"][1]["epsilon"] = 0.01
    stored["scopes"][1]["attacker_advantage_ceiling"] = 0.0
    problems = verify_report(stored)
    assert len(problems) == 2
    assert all(text.startswith("scope 'client-1': the report states") for text in problems)


def test_verify_notices_edited_releases(stored: dict[str, Any]) -> None:
    """Halving the stored noise changes every recomputed figure and breaks the budget."""
    stored["scopes"][0]["events"][0]["noise_multiplier"] /= 2
    problems = verify_report(stored)
    assert any("scope 'client-0': the report states epsilon" in text for text in problems)
    assert any("within_budget = True, but recomputing gives False" in text for text in problems)
    assert "at least one scope spent more than its budget" in problems


def test_verify_notices_an_edited_budget_or_flag(stored: dict[str, Any]) -> None:
    stored["scopes"][0]["budget"] = 8.0
    stored["within_budget"] = False
    problems = verify_report(stored)
    assert any("stated budget does not match the policy (1)" in text for text in problems)
    assert any("within_budget = False, but recomputing gives True" in text for text in problems)


def test_verify_refuses_what_it_cannot_read(stored: dict[str, Any]) -> None:
    assert verify_report({"schema": "something/2"}) == [
        f"not a {SCHEMA} file (schema is 'something/2')"
    ]
    assert verify_report({"schema": SCHEMA})[0].startswith("the report cannot be read")
    broken_policy = {**stored, "policy": {"epsilon": -1, "delta": DELTA}}
    assert verify_report(broken_policy)[0].startswith("the report cannot be read")
    stored["scopes"][0]["events"][0]["mechanism"] = "laplace"
    stored["scopes"][1]["epsilon"] = "a lot"
    problems = verify_report(stored)
    assert any(text.startswith("a scope cannot be read") for text in problems)
    assert any("states epsilon = 'a lot'" in text for text in problems)


def test_verify_counts_mixed_releases(tmp_path: Path) -> None:
    ledger = PrivacyLedger(PrivacyPolicy(2.0, DELTA))
    ledger.charge("a", SubsampledGaussianEvent(3.0, 0.05, 100))
    ledger.charge("a", GaussianEvent(6.0))
    path = tmp_path / "r.json"
    PrivacyReport.from_ledger(ledger).to_json(path)
    assert verify_report(load_report(path)) == []


# --------------------------------------------------------------------------- command line


def test_command_line_verify_and_show(
    report: PrivacyReport, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = tmp_path / "good.json"
    report.to_json(good)
    assert main(["report", "verify", str(good)]) == 0
    out = capsys.readouterr().out
    assert "consistent. epsilon = " in out
    assert "2 scope(s)" in out
    assert "unsafe" not in out

    assert main(["report", "show", str(good)]) == 0
    out = capsys.readouterr().out
    assert f"schema: {SCHEMA}" in out
    assert "policy: epsilon = 1.0, delta = 1e-05, unit = record" in out
    assert "within budget: True" in out
    assert "client-1: epsilon = " in out

    data = load_report(good)
    data["epsilon"] = 0.25
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    assert main(["report", "verify", str(bad)]) == 1
    out = capsys.readouterr().out
    assert "NOT CONSISTENT" in out
    assert "states a total of epsilon = 0.25" in out


def test_command_line_reports_unsafe_flags_and_unreadable_files(
    ledger: PrivacyLedger, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    flagged = tmp_path / "flagged.json"
    PrivacyReport.from_ledger(ledger, unsafe_flags=["SMOTE was forced on."]).to_json(flagged)
    assert main(["report", "verify", str(flagged)]) == 0
    out = capsys.readouterr().out
    assert "carries unsafe flags" in out
    assert "SMOTE was forced on." in out

    assert main(["report", "verify", str(tmp_path / "missing.json")]) == 2
    assert "cannot read" in capsys.readouterr().err
    garbage = tmp_path / "garbage.json"
    garbage.write_text("{not json", encoding="utf-8")
    assert main(["report", "show", str(garbage)]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_command_line_version_and_usage(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        main(["--version"])
    assert stop.value.code == 0
    assert capsys.readouterr().out.strip() == f"obscuradp {metadata.version('obscuradp')}"
    with pytest.raises(SystemExit) as stop:
        main([])
    assert stop.value.code == 2
    capsys.readouterr()


def test_the_command_is_installed_and_runs_as_a_module(
    report: PrivacyReport, tmp_path: Path
) -> None:
    scripts = metadata.entry_points(group="console_scripts")
    assert scripts["obscuradp"].value == "obscuradp.cli:main"
    path = tmp_path / "r.json"
    report.to_json(path)
    done = subprocess.run(
        [sys.executable, "-m", "obscuradp", "report", "verify", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "consistent" in done.stdout
