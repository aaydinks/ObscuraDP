"""The ``obscuradp`` command.

    obscuradp report verify REPORT.json    recompute a privacy report from its recorded releases
    obscuradp report show REPORT.json      print the stored report's main figures
    obscuradp audit self-test              check that the attack tests catch a broken pipeline
    obscuradp --version

``verify`` exits with status 0 if the report is consistent and 1 if anything disagrees, so it
can be used in scripts and pipelines. ``audit self-test`` exits with status 0 if the auditor
flags the broken pipelines it is shown and passes the correct ones, and 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from importlib import metadata

from obscuradp.report import load_report, verify_report


def _version() -> str:
    try:
        return metadata.version("obscuradp")
    except metadata.PackageNotFoundError:
        return "unknown"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="obscuradp", description="Tools for the obscuradp privacy library."
    )
    parser.add_argument("--version", action="version", version=f"obscuradp {_version()}")
    commands = parser.add_subparsers(dest="command", required=True)

    report = commands.add_parser("report", help="work with privacy report files")
    actions = report.add_subparsers(dest="action", required=True)
    verify = actions.add_parser(
        "verify", help="recompute a report from its recorded releases and compare"
    )
    verify.add_argument("path", help="a privacy report written by this library (JSON)")
    show = actions.add_parser("show", help="print the main figures stored in a report")
    show.add_argument("path", help="a privacy report written by this library (JSON)")

    audit = commands.add_parser("audit", help="attack tests of the privacy the library reports")
    audit_actions = audit.add_subparsers(dest="action", required=True)
    self_test = audit_actions.add_parser(
        "self-test",
        help="show the auditor a correct and a broken pipeline; it must tell them apart",
    )
    self_test.add_argument(
        "--trials",
        type=int,
        default=2000,
        help="runs per world for the single-step check (default: 2000)",
    )
    return parser


def _read(path: str) -> dict[str, object] | None:
    try:
        return load_report(path)
    except (OSError, json.JSONDecodeError) as error:
        print(f"cannot read {path}: {error}", file=sys.stderr)
        return None


def _verify(path: str) -> int:
    data = _read(path)
    if data is None:
        return 2
    problems = verify_report(data)
    if problems:
        print(f"{path}: NOT CONSISTENT")
        for text in problems:
            print(f"  - {text}")
        return 1
    scopes = data.get("scopes")
    count = len(scopes) if isinstance(scopes, list) else 0
    print(
        f"{path}: consistent. epsilon = {data.get('epsilon'):.6g} recomputed from the recorded "
        f"releases of {count} scope(s)."
    )
    flags = data.get("unsafe_flags")
    if isinstance(flags, list) and flags:
        print("Note: the report carries unsafe flags, so its guarantee is weakened or void:")
        for text in flags:
            print(f"  - {text}")
    return 0


def _show(path: str) -> int:
    data = _read(path)
    if data is None:
        return 2
    policy = data.get("policy")
    print(f"schema: {data.get('schema')}")
    if isinstance(policy, dict):
        print(
            f"policy: epsilon = {policy.get('epsilon')}, delta = {policy.get('delta')}, "
            f"unit = {policy.get('unit')}"
        )
    print(f"epsilon spent: {data.get('epsilon')}")
    print(f"within budget: {data.get('within_budget')}")
    scopes = data.get("scopes")
    if isinstance(scopes, list):
        for scope in scopes:
            print(
                f"  {scope.get('scope')}: epsilon = {scope.get('epsilon')} of {scope.get('budget')}"
            )
    mode = data.get("mode")
    if isinstance(mode, dict):
        print(
            f"mode: {mode.get('mode')}, {mode.get('participants')} participants, "
            f"{mode.get('completed_rounds')} of {mode.get('attempts')} rounds completed"
        )
        rows = mode.get("collusion_table")
        if isinstance(rows, list):
            print(
                f"  assumed: at most {mode.get('max_colluding')} colluding, "
                f"at most {mode.get('max_dropouts')} missing"
            )
            for row in rows:
                budget = "no protection" if row.get("unbounded") else row.get("epsilon")
                print(f"  {row.get('colluding')} colluding: epsilon = {budget}")
    print("These are the stored figures. Run 'obscuradp report verify' to recompute them.")
    return 0


def _self_test(trials: int) -> int:
    # Imported here: the report commands should not have to load the training code.
    from obscuradp.audit.selftest import passed, self_test

    checks = self_test(trials=trials)
    for check in checks:
        wanted = "must be flagged" if check.must_flag else "must not be flagged"
        print(f"[{'ok' if check.passed else 'FAILED'}] {check.name} ({wanted})")
        print(f"     {check.detail}")
    if passed(checks):
        print("Self-test passed: the auditor catches the broken pipelines and passes the others.")
        return 0
    print("Self-test FAILED: the auditor cannot be relied on in this installation.")
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line. Returns the exit status."""
    args = _build_parser().parse_args(argv)
    if args.command == "report" and args.action == "verify":
        return _verify(args.path)
    if args.command == "report" and args.action == "show":
        return _show(args.path)
    if args.command == "audit" and args.action == "self-test":
        return _self_test(args.trials)
    return 2  # pragma: no cover - argparse refuses unknown commands before this


if __name__ == "__main__":
    sys.exit(main())
