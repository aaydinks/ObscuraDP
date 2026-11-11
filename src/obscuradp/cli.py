"""The ``obscuradp`` command.

    obscuradp report verify REPORT.json    recompute a privacy report from its recorded releases
    obscuradp report show REPORT.json      print the stored report's main figures
    obscuradp --version

``verify`` exits with status 0 if the report is consistent and 1 if anything disagrees, so it
can be used in scripts and pipelines.
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line. Returns the exit status."""
    args = _build_parser().parse_args(argv)
    if args.command == "report" and args.action == "verify":
        return _verify(args.path)
    if args.command == "report" and args.action == "show":
        return _show(args.path)
    return 2  # pragma: no cover - argparse refuses unknown commands before this


if __name__ == "__main__":
    sys.exit(main())
