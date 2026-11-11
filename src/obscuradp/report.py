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
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from obscuradp.accounting.accountants import (
    attacker_advantage_ceiling,
    epsilon_pld,
    general_advantage_ceiling,
)
from obscuradp.accounting.crosscheck import CrossCheck, cross_check
from obscuradp.accounting.events import Event, event_from_dict, merge_events
from obscuradp.ledger import PrivacyLedger
from obscuradp.mechanisms.shared_noise import effective_noise_multiplier, honest_shares
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


MODE_DISTRIBUTED = "distributed"
MODE_CENTRAL = "central"
MODES = (MODE_DISTRIBUTED, MODE_CENTRAL)


@dataclass(frozen=True)
class CollusionRow:
    """The budget if ``colluding`` participants pool what they know with the server."""

    colluding: int
    noise_multiplier: float
    epsilon: float
    assumed: bool

    @property
    def unbounded(self) -> bool:
        """True if the budget is too large to mean anything."""
        return math.isinf(self.epsilon)

    def to_dict(self) -> dict[str, Any]:
        return {
            "colluding": self.colluding,
            "effective_noise_multiplier": self.noise_multiplier,
            "epsilon": None if self.unbounded else self.epsilon,
            "unbounded": self.unbounded,
            "assumed": self.assumed,
        }


def collusion_table(
    events_by_scope: Mapping[str, Sequence[Event]],
    delta: float,
    participants: int,
    max_colluding: int,
    max_dropouts: int,
) -> tuple[CollusionRow, ...]:
    """How the budget of a shared-noise run weakens with the real number of colluders.

    One row for every number $t'$ of colluding participants from 0 to $N - 1$. The stored
    releases carry the multiplier $\\sigma$ the run was calibrated for; a row recomputes the
    budget with $\\sigma \\sqrt{(N - t') / (N - t - d)}$ in its place and takes the largest
    result over all scopes. Rows up to $t' = t$ are within what the plan assumed.
    """
    honest = honest_shares(participants, max_colluding, max_dropouts)
    sigmas = [e.noise_multiplier for events in events_by_scope.values() for e in events]
    if not sigmas:
        raise ValueError("no release was recorded, so there is nothing to tabulate")
    rows = []
    for colluding in range(participants):
        factor = math.sqrt((participants - colluding) / honest)
        worst = 0.0
        for events in events_by_scope.values():
            scaled = [e.with_noise_multiplier(e.noise_multiplier * factor) for e in events]
            if scaled:
                worst = max(worst, epsilon_pld(scaled, delta))
        rows.append(
            CollusionRow(
                colluding=colluding,
                noise_multiplier=effective_noise_multiplier(
                    max(sigmas), participants, max_colluding, max_dropouts, colluding
                ),
                epsilon=worst,
                assumed=colluding <= max_colluding,
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class ModeReport:
    """How the noise was added, for the modes that take one step per round.

    Reports of the local mode carry no such section.

    Attributes:
        mode: ``"distributed"`` (the participants share the noise) or ``"central"`` (a
            trusted server adds it).
        participants: Number of participants $N$.
        attempts: Rounds that were started and charged to the budget.
        completed_rounds: Rounds whose sum was opened and moved the model.
        max_colluding: $t$ (distributed mode only).
        max_dropouts: $d$ (distributed mode only).
        steps_per_share: How many grid steps one noise slice spans (its standard deviation
            divided by the rounding step of the secure sum; distributed mode with Gaussian
            noise only).
        noise: ``"gaussian"`` or ``"skellam"`` (distributed mode only).
        table: The budget for every number of colluding participants (distributed mode).
    """

    mode: str
    participants: int
    attempts: int
    completed_rounds: int
    max_colluding: int | None = None
    max_dropouts: int | None = None
    steps_per_share: float | None = None
    table: tuple[CollusionRow, ...] = ()
    noise: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.mode == MODE_DISTRIBUTED and (
            self.max_colluding is None or self.max_dropouts is None
        ):
            raise ValueError("the distributed mode needs max_colluding and max_dropouts")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "mode": self.mode,
            "participants": self.participants,
            "attempts": self.attempts,
            "completed_rounds": self.completed_rounds,
        }
        if self.mode == MODE_DISTRIBUTED:
            data.update(
                max_colluding=self.max_colluding,
                max_dropouts=self.max_dropouts,
                steps_per_share=self.steps_per_share,
                collusion_table=[row.to_dict() for row in self.table],
            )
            if self.noise is not None:
                data["noise"] = self.noise
        return data

    def lines(self) -> list[str]:
        """The section of the plain-words summary."""
        failed = self.attempts - self.completed_rounds
        rounds = f"{self.completed_rounds} round(s) completed"
        if failed:
            rounds += (
                f"; {failed} further round(s) failed because a participant was missing and "
                "still count against the budget"
            )
        if self.mode == MODE_CENTRAL:
            return [
                "How the noise was added: by the server, which sees every participant's clean "
                f"sums ({self.participants} participants, one step per round, {rounds}).",
            ]
        lines = [
            f"How the noise was added: shared. Each of the {self.participants} participants "
            f"adds a slice (one step per round, {rounds}).",
            f"Assumed: at most {self.max_colluding} participant(s) pool what they know with "
            f"the server, at most {self.max_dropouts} are missing from a sum.",
            "Budget by the real number of participants who pool with the server:",
        ]
        for row in self.table:
            budget = "no protection" if row.unbounded else f"epsilon = {row.epsilon:.4g}"
            mark = "" if row.assumed else "   (more than assumed)"
            lines.append(f"  {row.colluding}: {budget}{mark}")
        if self.max_dropouts:
            lines.append(
                "  This secure sum only opens when nobody is missing, so the allowance for "
                "missing participants made every slice larger and the rows above are better "
                "than the stated budget."
            )
        if self.noise == "skellam":
            lines.append(
                "The noise is made of whole numbers. Every row took part in every round. "
                "Each row's contribution was rounded before the noise was added and its "
                "length was checked exactly, so the budget rests on no argument about "
                "rounding and no floating-point number carries the noise."
            )
        elif self.steps_per_share is not None:
            lines.append(
                f"Each participant rounds its noisy sum to whole numbers before the secure "
                f"sum, on a grid {self.steps_per_share:g} times finer than its noise slice. "
                "The budget is computed as if nothing were rounded. That is supported by an "
                "argument and a numerical check, not yet by a machine-checked proof."
            )
        return lines


@dataclass(frozen=True)
class PrivacyReport:
    """The privacy outcome of one run."""

    policy: PrivacyPolicy
    scopes: tuple[ScopeReport, ...]
    protection: tuple[str, ...] = ()
    unsafe_flags: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    library_version: str = ""
    mode: ModeReport | None = None

    # ------------------------------------------------------------------ building

    @classmethod
    def from_ledger(
        cls,
        ledger: PrivacyLedger,
        *,
        protection: Sequence[str] = (),
        unsafe_flags: Sequence[str] = (),
        caveats: Sequence[str] = (),
        mode: ModeReport | None = None,
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
            mode: How the noise was added, for the shared-noise and central modes.
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
            mode=mode,
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
            if self._worst().check.single_method:
                lines.append(
                    "Counted with the RDP method, the one method proven for whole-number "
                    "noise. For comparison: ordinary noise of the same size would cost "
                    f"{self._worst().check.pld_lower:.4g} by the tighter method."
                )
            else:
                lines.append(
                    "Cross-checks of that number: lower estimate "
                    f"{self._worst().check.pld_lower:.4g}, older RDP method "
                    f"{self._worst().check.rdp:.4g} (an upper limit, always looser)."
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
        if self.mode is not None:
            lines += ["", *self.mode.lines()]
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
        data = {
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
        # Only the one-step modes carry this section, so reports of the local mode keep
        # exactly the keys they always had.
        if self.mode is not None:
            data["mode"] = self.mode.to_dict()
        return data

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
    if "mode" in data:
        problems += _verify_mode(data["mode"], stored_scopes, policy.delta)
    return problems


def _verify_mode(mode: Any, stored_scopes: Sequence[Any], delta: float) -> list[str]:
    """Check the section of a one-step mode against the report's recorded releases.

    The table is recomputed from the same stored releases as the report's total, so the row
    for the assumed number of colluders agrees with that total by construction.
    """
    if not isinstance(mode, Mapping) or mode.get("mode") not in MODES:
        return ["the mode section cannot be read"]
    problems: list[str] = []
    try:
        participants = int(mode["participants"])
        attempts = int(mode["attempts"])
        completed = int(mode["completed_rounds"])
        events_by_scope = {
            str(entry["scope"]): merge_events(event_from_dict(e) for e in entry["events"])
            for entry in stored_scopes
        }
    except (KeyError, TypeError, ValueError) as error:
        return [f"the mode section cannot be read: {error}"]
    if participants < len(events_by_scope):
        problems.append("the mode section names fewer participants than there are scopes")
    if not 0 <= completed <= attempts:
        problems.append("the mode section's round counts are impossible")
    most = max(
        (sum(e.repetitions for e in events) for events in events_by_scope.values()), default=0
    )
    if most > attempts:
        problems.append(
            f"a scope recorded {most} releases, more than the {attempts} attempts the mode "
            "section states"
        )
    if mode["mode"] != MODE_DISTRIBUTED:
        return problems
    try:
        max_colluding = int(mode["max_colluding"])
        max_dropouts = int(mode["max_dropouts"])
        stored_rows = list(mode["collusion_table"])
        rows = collusion_table(events_by_scope, delta, participants, max_colluding, max_dropouts)
    except (KeyError, TypeError, ValueError) as error:
        return [*problems, f"the table of colluding participants cannot be read: {error}"]
    sigmas = {e.noise_multiplier for events in events_by_scope.values() for e in events}
    if len(sigmas) > 1:
        problems.append("shared noise means one noise level, but the scopes record several")
    if len(stored_rows) != len(rows):
        problems.append(
            f"the table of colluding participants has {len(stored_rows)} rows, expected {len(rows)}"
        )
        return problems
    for stored, row in zip(stored_rows, rows, strict=True):
        expected = row.to_dict()
        try:
            same = (
                stored["colluding"] == expected["colluding"]
                and stored["unbounded"] is expected["unbounded"]
                and stored["assumed"] is expected["assumed"]
                and _close(stored["effective_noise_multiplier"], row.noise_multiplier)
                and (
                    stored["epsilon"] is None
                    if row.unbounded
                    else _close(stored["epsilon"], row.epsilon)
                )
            )
        except (KeyError, TypeError):
            same = False
        if not same:
            problems.append(
                f"the table row for {row.colluding} colluding participant(s) does not match "
                f"the recorded releases (recomputed: epsilon = {row.epsilon:.9g})"
            )
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
