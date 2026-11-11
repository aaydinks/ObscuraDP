"""The privacy report, recomputing it, and the command line."""

import copy
import json
import math
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
from obscuradp.report import (
    NOT_COVERED,
    SCHEMA,
    ModeReport,
    PrivacyReport,
    collusion_table,
    load_report,
    verify_report,
)

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


# --------------------------------------------------------------------------- shared noise

PARTICIPANTS = 4
COLLUDING = 1


def _shared_mode(ledger: PrivacyLedger, dropouts: int = 0) -> ModeReport:
    return ModeReport(
        mode="distributed",
        participants=PARTICIPANTS,
        attempts=STEPS,
        completed_rounds=STEPS - 3,
        max_colluding=COLLUDING,
        max_dropouts=dropouts,
        steps_per_share=4.0,
        table=collusion_table(
            {scope: ledger.events(scope) for scope in ledger.scopes},
            DELTA,
            PARTICIPANTS,
            COLLUDING,
            dropouts,
        ),
    )


@pytest.fixture(scope="module")
def shared_report(ledger: PrivacyLedger) -> PrivacyReport:
    return PrivacyReport.from_ledger(ledger, mode=_shared_mode(ledger))


@pytest.fixture
def shared_stored(shared_report: PrivacyReport) -> dict[str, Any]:
    return copy.deepcopy(shared_report.to_dict())


def test_the_table_recomputes_the_budget_for_every_number_of_colluders(
    ledger: PrivacyLedger, shared_report: PrivacyReport
) -> None:
    assert shared_report.mode is not None
    rows = shared_report.mode.table
    assert [row.colluding for row in rows] == [0, 1, 2, 3]
    assert [row.assumed for row in rows] == [True, True, False, False]
    sigma = ledger.events("client-0")[0].noise_multiplier
    for row in rows:
        factor = math.sqrt((PARTICIPANTS - row.colluding) / (PARTICIPANTS - COLLUDING))
        assert row.noise_multiplier == pytest.approx(sigma * factor)
        # The scope with the most releases decides, as for the report's total.
        expected = epsilon_pld([SubsampledGaussianEvent(sigma * factor, RATE, STEPS)], DELTA)
        assert row.epsilon == pytest.approx(expected)
    # The row for the assumed number is the stated budget; fewer colluders is better.
    assert rows[COLLUDING].epsilon == pytest.approx(shared_report.epsilon)
    budgets = [row.epsilon for row in rows]
    assert budgets == sorted(budgets)
    assert budgets[0] < shared_report.epsilon < budgets[2] < budgets[3]


def test_an_allowance_for_missing_participants_makes_every_row_better(
    ledger: PrivacyLedger, shared_report: PrivacyReport
) -> None:
    assert shared_report.mode is not None
    with_allowance = _shared_mode(ledger, dropouts=1)
    for plain, better in zip(shared_report.mode.table, with_allowance.table, strict=True):
        assert better.epsilon < plain.epsilon
    text = "\n".join(with_allowance.lines())
    assert "at most 1 are missing from a sum" in text
    assert "only opens when nobody is missing" in text
    report = PrivacyReport.from_ledger(ledger, mode=with_allowance)
    assert verify_report(report.to_dict()) == []


def test_the_table_is_part_of_the_summary_and_the_file(
    shared_report: PrivacyReport, tmp_path: Path
) -> None:
    text = shared_report.summary()
    assert "How the noise was added: shared. Each of the 4 participants adds a slice" in text
    assert "197 round(s) completed; 3 further round(s) failed" in text
    assert "Budget by the real number of participants who pool with the server:" in text
    assert f"  1: epsilon = {shared_report.epsilon:.4g}\n" in text
    assert "(more than assumed)" in text
    assert "on a grid 4 times finer than its noise slice" in text
    assert chr(0x2014) not in text
    path = tmp_path / "shared.json"
    shared_report.to_json(path)
    stored = load_report(path)
    assert stored["schema"] == SCHEMA
    assert stored["mode"]["max_colluding"] == COLLUDING
    assert len(stored["mode"]["collusion_table"]) == PARTICIPANTS
    assert verify_report(stored) == []


def test_reports_of_the_local_mode_keep_exactly_their_keys(report: PrivacyReport) -> None:
    assert report.mode is None
    assert set(report.to_dict()) == {
        "schema",
        "library_version",
        "policy",
        "epsilon",
        "within_budget",
        "attacker_advantage_ceiling",
        "general_attacker_advantage_ceiling",
        "scopes",
        "protection",
        "unsafe_flags",
        "caveats",
        "not_covered",
    }
    assert "How the noise was added" not in report.summary()


def test_rows_without_protection_are_stored_as_null() -> None:
    ledger = PrivacyLedger(PrivacyPolicy(150.0, DELTA))
    ledger.charge("a", SubsampledGaussianEvent(1.0, 1.0, 100))
    table = collusion_table({"a": ledger.events("a")}, DELTA, 3, 0, 0)
    assert [row.unbounded for row in table] == [False, False, True]
    mode = ModeReport(
        mode="distributed",
        participants=3,
        attempts=100,
        completed_rounds=100,
        max_colluding=0,
        max_dropouts=0,
        steps_per_share=4.0,
        table=table,
    )
    report = PrivacyReport.from_ledger(ledger, mode=mode)
    stored = report.to_dict()
    assert stored["mode"]["collusion_table"][2]["epsilon"] is None
    assert stored["mode"]["collusion_table"][2]["unbounded"] is True
    json.dumps(stored, allow_nan=False)
    assert "  2: no protection   (more than assumed)" in report.summary()
    assert verify_report(stored) == []
    # A row that claims protection where there is none is caught.
    stored["mode"]["collusion_table"][2].update(epsilon=5.0, unbounded=False)
    assert any("row for 2 colluding" in problem for problem in verify_report(stored))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("collusion_table", 2, "epsilon"), 1.0, "row for 2 colluding"),
        (("collusion_table", 0, "effective_noise_multiplier"), 9.0, "row for 0 colluding"),
        (("collusion_table", 3, "assumed"), True, "row for 3 colluding"),
        (("collusion_table", 1, "colluding"), 2, "row for 1 colluding"),
        (("collusion_table", 1, "unbounded"), True, "row for 1 colluding"),
        (("max_colluding",), 2, "does not match the recorded releases"),
        (("max_dropouts",), 1, "does not match the recorded releases"),
        (("participants",), 5, "has 4 rows, expected 5"),
        (("participants",), 1, "fewer participants than there are scopes"),
        (("attempts",), 150, "more than the 150 attempts"),
        (("completed_rounds",), 500, "round counts are impossible"),
        (("mode",), "magic", "cannot be read"),
        (("max_colluding",), 7, "cannot be read"),
        (("attempts",), "many", "cannot be read"),
    ],
)
def test_verify_notices_an_edited_mode_section(
    shared_stored: dict[str, Any], path: tuple[Any, ...], value: Any, message: str
) -> None:
    target = shared_stored["mode"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    problems = verify_report(shared_stored)
    assert any(message in problem for problem in problems), problems


def test_verify_notices_a_broken_mode_section(shared_stored: dict[str, Any]) -> None:
    removed = copy.deepcopy(shared_stored)
    del removed["mode"]["collusion_table"][0]
    assert any("has 3 rows, expected 4" in problem for problem in verify_report(removed))
    incomplete = copy.deepcopy(shared_stored)
    del incomplete["mode"]["collusion_table"][1]["epsilon"]
    assert any("row for 1 colluding" in problem for problem in verify_report(incomplete))
    missing = copy.deepcopy(shared_stored)
    del missing["mode"]["max_colluding"]
    assert any("cannot be read" in problem for problem in verify_report(missing))
    nonsense = copy.deepcopy(shared_stored)
    nonsense["mode"] = "distributed"
    assert verify_report(nonsense) == ["the mode section cannot be read"]
    # Shared noise means one noise level; two levels cannot come from an honest run.
    mixed = copy.deepcopy(shared_stored)
    mixed["scopes"][1]["events"][0]["noise_multiplier"] *= 1.5
    assert any("the scopes record several" in problem for problem in verify_report(mixed))
    # The row for the assumed number is recomputed from the same releases as the total.
    assert shared_stored["mode"]["collusion_table"][COLLUDING]["epsilon"] == pytest.approx(
        shared_stored["epsilon"]
    )


def test_a_mode_section_must_be_complete(ledger: PrivacyLedger) -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        ModeReport(mode="local", participants=3, attempts=1, completed_rounds=1)
    with pytest.raises(ValueError, match="needs max_colluding and max_dropouts"):
        ModeReport(mode="distributed", participants=3, attempts=1, completed_rounds=1)
    events = {"client-0": ledger.events("client-0")}
    with pytest.raises(ValueError, match="nobody is left"):
        collusion_table(events, DELTA, 3, 2, 1)
    with pytest.raises(ValueError, match="nothing to tabulate"):
        collusion_table({}, DELTA, 3, 1, 0)
    central = ModeReport(mode="central", participants=3, attempts=5, completed_rounds=5)
    assert central.to_dict() == {
        "mode": "central",
        "participants": 3,
        "attempts": 5,
        "completed_rounds": 5,
    }
    assert "by the server" in central.lines()[0]


def test_command_line_shows_and_verifies_a_shared_noise_report(
    shared_report: PrivacyReport, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "shared.json"
    shared_report.to_json(path)
    assert main(["report", "verify", str(path)]) == 0
    assert "consistent. epsilon = " in capsys.readouterr().out
    assert main(["report", "show", str(path)]) == 0
    out = capsys.readouterr().out
    assert "mode: distributed, 4 participants, 197 of 200 rounds completed" in out
    assert "assumed: at most 1 colluding, at most 0 missing" in out
    assert "  3 colluding: epsilon = " in out

    data = load_report(path)
    data["mode"]["collusion_table"][3]["epsilon"] = 0.5
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    assert main(["report", "verify", str(bad)]) == 1
    assert "row for 3 colluding" in capsys.readouterr().out

    data = load_report(path)
    data["mode"]["collusion_table"][3].update(epsilon=None, unbounded=True)
    odd = tmp_path / "odd.json"
    odd.write_text(json.dumps(data), encoding="utf-8")
    assert main(["report", "show", str(odd)]) == 0
    assert "  3 colluding: epsilon = no protection" in capsys.readouterr().out
