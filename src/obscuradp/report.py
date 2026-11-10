"""The privacy report: what a run promised, what it spent, and how to check it.

A report is the library's answer to "how private was this run?". It states the promise in
plain words, lists every recorded release per scope, and gives the budget those releases
spent, computed three ways. Because the releases themselves are stored, anyone can
recompute the numbers: ``verify_report`` does exactly that, and so does the command
``obscuradp report verify``.

A report describes the releases the library was told about. It cannot know about releases
made outside it, which is why the list of things not covered is part of every report.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from obscuradp.accounting.accountants import attacker_advantage_ceiling, general_advantage_ceiling
from obscuradp.accounting.crosscheck import CrossCheck, cross_check
from obscuradp.accounting.events import Event, event_from_dict, merge_events
from obscuradp.ledger import PrivacyLedger
from obscuradp.policy import PrivacyPolicy

SCHEMA = "obscuradp.privacy_report/1"

# What no report can vouch for, stated in every report.
NOT_COVERED = (
    "Anything computed from private data outside this library, such as statistics used for "
    "preprocessing or settings chosen by looking at private results.",
    "The evaluation data and the numbers computed on it.",
    "Side channels: timing, message sizes, and which participants were online.",
    "The facts the protocol treats as public: declared dataset sizes, the feature layout and "
    "the model architecture.",
    "A participant or server that does not follow the protocol.",
)

_RELATIVE_TOLERANCE = 1e-6
_ABSOLUTE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ScopeReport:
    """One scope's releases and the budget they spent."""

    scope: str
    budget: float
    events: tuple[Event, ...]
    check: CrossCheck
    advantage_ceiling: float

    @property
    def epsilon(self) -> float:
        return self.check.epsilon

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "budget": self.budget,
            "events": [event.to_dict() for event in self.events],
            "epsilon": self.check.pld_upper,
            "epsilon_lower_estimate": self.check.pld_lower,
            "epsilon_rdp": self.check.rdp,
            "attacker_advantage_ceiling": self.advantage_ceiling,
        }


@dataclass(frozen=True)
class PrivacyReport:
    """The privacy outcome of one run."""

    policy: PrivacyPolicy
    scopes: tuple[ScopeReport, ...]
    protection: tuple[str, ...] = ()
    unsafe_flags: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    library_version: str = ""

    # ------------------------------------------------------------------ building

    @classmethod
    def from_ledger(
        cls,
        ledger: PrivacyLedger,
        *,
        protection: Sequence[str] = (),
        unsafe_flags: Sequence[str] = (),
        caveats: Sequence[str] = (),
    ) -> PrivacyReport:
        """Build the report of everything ``ledger`` recorded.

        Every scope's budget is recomputed three ways and the three must be consistent
        (see ``obscuradp.accounting.crosscheck``); otherwise ``AccountingMismatch`` is raised
        and no report is produced.

        Args:
            ledger: The ledger of the finished run.
            protection: Statements of who is protected from whom in this configuration.
            unsafe_flags: Anything about the run that weakens or voids the stated guarantee.
            caveats: Further limits that apply to this run.
        """
        delta = ledger.policy.delta
        scopes = []
        for scope in ledger.scopes:
            events = ledger.events(scope)
            scopes.append(
                ScopeReport(
                    scope=scope,
                    budget=ledger.policy.epsilon_for(scope),
                    events=events,
                    check=cross_check(events, delta),
                    advantage_ceiling=attacker_advantage_ceiling(events),
                )
            )
        return cls(
            policy=ledger.policy,
            scopes=tuple(scopes),
            protection=tuple(protection),
            unsafe_flags=tuple(unsafe_flags),
            caveats=tuple(caveats),
            library_version=_library_version(),
        )

    # ------------------------------------------------------------------ reading

    @property
    def epsilon(self) -> float:
        """The budget spent by the run as a whole: the largest spend of any scope."""
        return max((scope.epsilon for scope in self.scopes), default=0.0)

    @property
    def within_budget(self) -> bool:
        """True if no scope spent more than its budget."""
        return all(
            scope.epsilon <= scope.budget * (1 + _RELATIVE_TOLERANCE) + _ABSOLUTE_TOLERANCE
            for scope in self.scopes
        )

    @property
    def is_safe(self) -> bool:
        """True if the run stayed within budget and raised no unsafe flag."""
        return self.within_budget and not self.unsafe_flags

    @property
    def general_advantage_ceiling(self) -> float:
        """Largest attacker advantage any mechanism with this run's budget could allow."""
        return general_advantage_ceiling(self.epsilon, self.policy.delta)

    @property
    def advantage_ceiling(self) -> float:
        """Largest attacker advantage against the releases actually made (worst scope)."""
        return max((scope.advantage_ceiling for scope in self.scopes), default=0.0)

    def summary(self) -> str:
        """The report in plain words."""
        lines = ["PRIVACY REPORT", "", self.policy.describe(), ""]
        if not self.scopes:
            lines.append("No release was recorded, so no budget was spent.")
        else:
            verdict = "within" if self.within_budget else "OVER"
            lines.append(
                f"Budget spent by the run as a whole: epsilon = {self.epsilon:.4g} "
                f"({verdict} budget; the largest spend of any scope)."
            )
            lines.append(
                f"Cross-checks of that number: lower estimate {self._worst().check.pld_lower:.4g}, "
                f"older RDP method {self._worst().check.rdp:.4g} (an upper limit, always looser)."
            )
            lines += [f"Note: {text}." for text in self._worst().check.warnings()]
            lines.append(
                "Worst case for an attacker guessing whether a record was used "
                "(true-positive rate minus false-positive rate): "
                f"at most {self.advantage_ceiling:.1%} against these releases; "
                f"{self.general_advantage_ceiling:.1%} for any method with this budget."
            )
            lines.append("")
            lines.append("Per scope:")
            for scope in self.scopes:
                steps = sum(event.repetitions for event in scope.events)
                lines.append(
                    f"  {scope.scope}: epsilon = {scope.epsilon:.4g} of {scope.budget:g} "
                    f"({steps} noisy releases in {len(scope.events)} group(s))"
                )
        if self.protection:
            lines += ["", "Who is protected from whom:"]
            lines += [f"  - {text}" for text in self.protection]
        if self.unsafe_flags:
            lines += ["", "UNSAFE: the stated guarantee is weakened or void because:"]
            lines += [f"  - {text}" for text in self.unsafe_flags]
        lines += ["", "Not covered by this report:"]
        lines += [f"  - {text}" for text in (*NOT_COVERED, *self.caveats)]
        lines += ["", "Check these numbers yourself: obscuradp report verify <this file>"]
        return "\n".join(lines)

    def _worst(self) -> ScopeReport:
        return max(self.scopes, key=lambda scope: scope.epsilon)

    # ------------------------------------------------------------------ storing

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "library_version": self.library_version,
            "policy": self.policy.to_dict(),
            "epsilon": self.epsilon,
            "within_budget": self.within_budget,
            "attacker_advantage_ceiling": self.advantage_ceiling,
            "general_attacker_advantage_ceiling": self.general_advantage_ceiling,
            "scopes": [scope.to_dict() for scope in self.scopes],
            "protection": list(self.protection),
            "unsafe_flags": list(self.unsafe_flags),
            "caveats": list(self.caveats),
            "not_covered": list(NOT_COVERED),
        }

    def to_json(self, path: str | os.PathLike[str]) -> None:
        """Write the report as strict, sorted JSON."""
        text = json.dumps(self.to_dict(), indent=1, sort_keys=True, allow_nan=False)
        Path(path).write_text(text + "\n", encoding="utf-8")


def load_report(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a report file as a plain dictionary (use ``verify_report`` to check it)."""
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return data


def verify_report(data: Mapping[str, Any]) -> list[str]:
    """Recompute a stored report from its recorded releases and list every disagreement.

    Returns an empty list if the report is consistent: every scope's stated budget figures
    are reproduced from its stored events, no scope is over budget unless the report says
    so, and the three accountants agree with each other.
    """
    problems: list[str] = []
    if data.get("schema") != SCHEMA:
        return [f"not a {SCHEMA} file (schema is {data.get('schema')!r})"]
    try:
        policy = PrivacyPolicy.from_dict(data["policy"])
        stored_scopes = list(data["scopes"])
    except (KeyError, TypeError, ValueError) as error:
        return [f"the report cannot be read: {error}"]

    worst = 0.0
    all_within = True
    for entry in stored_scopes:
        try:
            name = str(entry["scope"])
            events = merge_events(event_from_dict(e) for e in entry["events"])
        except (KeyError, TypeError, ValueError) as error:
            problems.append(f"a scope cannot be read: {error}")
            continue
        check = cross_check(events, policy.delta, strict=False)
        problems += [f"scope {name!r}: {text}" for text in check.problems()]
        for key, recomputed in (
            ("epsilon", check.pld_upper),
            ("epsilon_lower_estimate", check.pld_lower),
            ("epsilon_rdp", check.rdp),
            ("attacker_advantage_ceiling", attacker_advantage_ceiling(events)),
        ):
            if not _close(entry.get(key), recomputed):
                problems.append(
                    f"scope {name!r}: the report states {key} = {entry.get(key)!r}, "
                    f"but its recorded releases give {recomputed:.9g}"
                )
        budget = policy.epsilon_for(name)
        if not _close(entry.get("budget"), budget):
            problems.append(
                f"scope {name!r}: the stated budget does not match the policy ({budget:g})"
            )
        if check.pld_upper > budget * (1 + _RELATIVE_TOLERANCE) + _ABSOLUTE_TOLERANCE:
            all_within = False
        worst = max(worst, check.pld_upper)

    if not _close(data.get("epsilon"), worst):
        problems.append(
            f"the report states a total of epsilon = {data.get('epsilon')!r}, "
            f"but its recorded releases give {worst:.9g}"
        )
    if data.get("within_budget") is not all_within:
        problems.append(
            f"the report says within_budget = {data.get('within_budget')!r}, "
            f"but recomputing gives {all_within}"
        )
    if not all_within:
        problems.append("at least one scope spent more than its budget")
    return problems


def _close(stated: Any, recomputed: float) -> bool:
    if isinstance(stated, bool) or not isinstance(stated, (int, float)):
        return False
    return abs(float(stated) - recomputed) <= _ABSOLUTE_TOLERANCE + _RELATIVE_TOLERANCE * abs(
        recomputed
    )


def _library_version() -> str:
    try:
        return metadata.version("obscuradp")
    except metadata.PackageNotFoundError:
        return "unknown"
