"""The privacy ledger: every release is written down before it happens.

The ledger is where a run's privacy budget is enforced. Code that is about to let something
derived from private data leave a participant must call ``charge`` first. If the release
would push that participant's total over its budget, ``charge`` raises ``BudgetExceeded``
and records nothing, so the release must not happen.

Budgets are kept per scope. For record-level privacy a scope is one participant: each
participant's records are protected by that participant's own noise, and the guarantee for
the whole federation is the largest spend of any participant.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from obscuradp.accounting.accountants import epsilon_pld, epsilon_rdp
from obscuradp.accounting.events import Event, covers, event_from_dict, merge_events
from obscuradp.policy import PrivacyPolicy

# Slack for floating-point effects when comparing a spend with its budget.
_BUDGET_SLACK = 1e-9


class BudgetExceeded(RuntimeError):
    """A release was refused because it would exceed the privacy budget."""


class PrivacyLedger:
    """Records releases per scope and refuses any that would exceed the policy."""

    def __init__(self, policy: PrivacyPolicy) -> None:
        self.policy = policy
        self._charged: dict[str, list[Event]] = {}
        self._planned: dict[str, tuple[Event, ...]] = {}
        self._planned_epsilon: dict[str, float] = {}

    # ------------------------------------------------------------------ computing spends

    def _epsilon(self, events: Sequence[Event]) -> float:
        if not events:
            return 0.0
        if self.policy.accountant == "rdp":
            return epsilon_rdp(events, self.policy.delta)
        return epsilon_pld(events, self.policy.delta)

    def epsilon_of(self, events: Iterable[Event]) -> float:
        """Budget a set of events would spend, by this ledger's accountant."""
        return self._epsilon(merge_events(events))

    # ------------------------------------------------------------------ planning and charging

    def plan(self, scope: str, events: Iterable[Event]) -> float:
        """Reserve the releases of a whole run for ``scope`` and check that they fit.

        Planning a run in advance has two benefits: an impossible run is refused before any
        data is touched, and later ``charge`` calls that stay inside the plan are accepted
        without recomputing the budget. Returns the budget the plan will spend.

        Raises:
            BudgetExceeded: If the planned releases, on top of what was already charged,
                would exceed the scope's budget. Nothing is reserved in that case.
        """
        planned = merge_events([*self._charged.get(scope, []), *events])
        epsilon = self._epsilon(planned)
        budget = self.policy.epsilon_for(scope)
        if epsilon > budget + _BUDGET_SLACK:
            raise BudgetExceeded(
                f"the planned releases for {scope!r} would spend epsilon = {epsilon:.6g}, "
                f"more than its budget of {budget:g}"
            )
        self._planned[scope] = planned
        self._planned_epsilon[scope] = epsilon
        return epsilon

    def charge(self, scope: str, event: Event) -> None:
        """Record one release for ``scope``. Call this BEFORE the release happens.

        Raises:
            BudgetExceeded: If the release would exceed the scope's budget. Nothing is
                recorded in that case, and the release must not happen.
        """
        if event.repetitions == 0:
            return
        after = [*self._charged.get(scope, []), event]
        planned = self._planned.get(scope)
        if planned is None or not covers(planned, after):
            epsilon = self._epsilon(merge_events(after))
            budget = self.policy.epsilon_for(scope)
            if epsilon > budget + _BUDGET_SLACK:
                raise BudgetExceeded(
                    f"this release would bring {scope!r} to epsilon = {epsilon:.6g}, "
                    f"more than its budget of {budget:g}"
                )
        self._charged[scope] = list(merge_events(after))

    def can_afford(self, scope: str, event: Event) -> bool:
        """True if ``charge(scope, event)`` would be accepted. Records nothing."""
        after = merge_events([*self._charged.get(scope, []), event])
        planned = self._planned.get(scope)
        if planned is not None and covers(planned, after):
            return True
        return self._epsilon(after) <= self.policy.epsilon_for(scope) + _BUDGET_SLACK

    # ------------------------------------------------------------------ reading

    @property
    def scopes(self) -> tuple[str, ...]:
        """Scopes with at least one recorded release, in order of first release."""
        return tuple(self._charged)

    def events(self, scope: str) -> tuple[Event, ...]:
        """The recorded releases of ``scope``, merged by settings."""
        return merge_events(self._charged.get(scope, []))

    def planned(self, scope: str) -> tuple[Event, ...]:
        """The releases reserved for ``scope`` with ``plan``, or an empty tuple."""
        return self._planned.get(scope, ())

    def spent(self, scope: str) -> float:
        """Budget ``scope`` has spent so far."""
        return self._epsilon(self.events(scope))

    def remaining(self, scope: str) -> float:
        """Budget ``scope`` has left (never negative)."""
        return max(0.0, self.policy.epsilon_for(scope) - self.spent(scope))

    def spent_max(self) -> float:
        """The largest spend of any scope: the guarantee of the federation as a whole."""
        return max((self.spent(scope) for scope in self._charged), default=0.0)

    # ------------------------------------------------------------------ storing

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "scopes": {
                scope: [event.to_dict() for event in self.events(scope)] for scope in self._charged
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PrivacyLedger:
        """Rebuild a ledger from ``to_dict`` output. The stored releases are trusted as given."""
        ledger = cls(PrivacyPolicy.from_dict(data["policy"]))
        for scope, events in data.get("scopes", {}).items():
            ledger._charged[str(scope)] = list(merge_events(event_from_dict(e) for e in events))
        return ledger
